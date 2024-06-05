#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)

#include "buffer_allocator.hpp"

#include <linux/mman.h>
#include <sys/mman.h>

#include "osal.hpp"
#include "infra/settable.h"
#include "event_triggered_logger.hpp"

#include "synapse_runtime_logging.h"
#include "types.h"
#include "utils.h"
#include "memory_allocator_utils.hpp"
#include "syn_singleton.hpp"
#include "habana_global_conf_runtime.h"

#define HUGE_PAGE_SIZE    (2 * 1024 * 1024ULL)
#define REGULAR_PAGE_SIZE (4 * 1024)

std::atomic<uint64_t> HostAllocator::m_numOfHugePagesAllocated {0};
std::atomic<uint64_t> HostAllocator::m_numOfRegularPagesAllocated {0};

const uint64_t ManagedBufferAllocator::m_defaultAlignment = 128;

uint64_t DeviceAllocator::m_numOfHugePagesAllocated {0};

synStatus IoctlDeviceAllocator::MemoryIoctl(uint64_t  deviceVirtAddr,
                                            uint64_t  hostVirtAddr,
                                            uint32_t  op,
                                            uint64_t  memSize,
                                            uint32_t  flags,
                                            uint64_t* pRetAddress,
                                            uint64_t* pHandle,
                                            bool      useHandle)
{
    if (op == HL_MEM_OP_MAP)
    {
        if (pRetAddress == nullptr)
        {
            LOG_ERR(SYN_MEM_ALLOC, "null retAddress pointer");
            return synInvalidArgument;
        }
    }

    if ((useHandle) && (pHandle == nullptr))
    {
        LOG_ERR(SYN_MEM_ALLOC, "null handle pointer");
        return synInvalidArgument;
    }

    int ret = 0;
    int fd  = OSAL::getInstance().getFd();

    switch (op)
    {
        case HL_MEM_OP_UNMAP:
            ret = hlthunk_memory_unmap(fd, deviceVirtAddr);
            break;

        case HL_MEM_OP_MAP:
            // this flag is only valid for the host-allocator
            if (flags & HL_MEM_USERPTR)
            {
                *pRetAddress = hlthunk_host_memory_map(fd, (void*)hostVirtAddr, 0, memSize);
                if (!*pRetAddress) ret = -1;
            }
            else if (useHandle)
            {
                *pRetAddress = hlthunk_device_memory_map(fd, *pHandle, 0);
                if (!*pRetAddress) ret = -1;
            }
            break;

        case HL_MEM_OP_ALLOC:
            *pHandle = hlthunk_device_memory_alloc(fd, memSize, 0, HL_MEM_CONTIGUOUS, 0x0);
            if (!*pHandle) ret = -1;
            break;

        case HL_MEM_OP_FREE:
            ret = hlthunk_device_memory_free(fd, *pHandle);
            break;
    }

    if (ret < 0)
    {
        _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::memcopyIoctlFailed);
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);

        LOG_ERR(SYN_MEM_ALLOC, "{}: Operation {} failed", HLLOG_FUNC, op);
        return synFail;
    }

    return synSuccess;
}

synStatus DeviceAllocator::AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest)
{
    synStatus status(synSuccess);

    uint64_t handle = 0;

    // 1. Validation
    if (size == 0)
    {
        LOG_ERR(SYN_MEM_ALLOC, "invalid size - requested to allocate buffer with 0 bytes");
        return synInvalidArgument;
    }

    if ((uint64_t)getDeviceVa())
    {
        LOG_WARN(SYN_MEM_ALLOC, "Memory already allocated");
        return synObjectAlreadyInitialized;
    }

    if (!OSAL::getInstance().isAcquired())
    {
        LOG_ERR(SYN_MEM_ALLOC, "Device not acquired");
        return synInvalidArgument;
    }

    // 2. Allocation
    status = MemoryIoctl(reqVAAddress, 0, HL_MEM_OP_ALLOC, size, 0, nullptr, &handle, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL,
                "{}: Device allocation failed for virtual address {}, size {}",
                HLLOG_FUNC,
                reqVAAddress,
                size);
        LOG_ERR(SYN_OSAL, "{}: Currently allocated {}", HLLOG_FUNC, m_numOfHugePagesAllocated * HUGE_PAGE_SIZE);

        return status;
    }

    m_numOfHugePagesAllocated++;

    // 3. Mapping
    uint64_t deviceAddress = 0;
    status                 = MemoryIoctl(reqVAAddress, 0, HL_MEM_OP_MAP, size, 0, &deviceAddress, &handle, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Device mapping failed for virtual address {}, size {}", reqVAAddress, size);

        return status;
    }

    LOG_DEBUG(SYN_MEM_ALLOC, "{}: deviceAddress 0x{:x}", HLLOG_FUNC, deviceAddress);

    // 4. Storing information
    setSize(size);
    setFlags(0);
    setDeviceVa(deviceAddress);
    setHandle(handle);
    setShouldFreeMemory(true);

    return synSuccess;
}

synStatus DeviceAllocator::FreeMemory()
{
    synStatus status(synSuccess);

    uint64_t deviceVa = (uint64_t)getDeviceVa();
    uint64_t handle   = getHandle();

    // 1. Validation
    if (!OSAL::getInstance().isAcquired())
    {
        LOG_ERR(SYN_MEM_ALLOC, "Device not acquired");
        return synInvalidArgument;
    }

    // 2. Un-map
    status = MemoryIoctl(deviceVa, 0, HL_MEM_OP_UNMAP, 0, 0, nullptr, nullptr);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Device unmapping failed for (device) virtual address 0x{:x}", deviceVa);

        return status;
    }

    // 3. Free
    if (shouldFreeMemory())
    {
        status = MemoryIoctl(0, 0, HL_MEM_OP_FREE, 0, 0, nullptr, &handle, true);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_OSAL, "Device memory-free failed for virtual address {} handle {}", deviceVa, handle);

            return status;
        }

        m_numOfHugePagesAllocated--;
    }

    LOG_DEBUG(SYN_MEM_ALLOC, "{}: deviceAddress 0x{:x}", HLLOG_FUNC, deviceVa);

    // 4. Clear information
    setSize(0);
    setFlags(0);
    setDeviceVa(0);
    setHandle(0);
    setShouldFreeMemory(false);

    return synSuccess;
}

synStatus HostAllocator::AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest)
{
    void* hostAddress;

    synStatus status = _allocateMemory(size, &hostAddress, isUserRequest);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Host allocation failed with size {}", size);
        return status;
    }

    uint64_t allocatedSize = getSize();
    uint64_t deviceAddress = 0;

    status = _mapMemory(allocatedSize, (uint64_t)hostAddress, deviceAddress);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Host mapping failed for virtual address {}, size {}", deviceAddress, size);

        if (munmap((void*)hostAddress, allocatedSize) == -1)
        {
            LOG_ERR(SYN_MEM_ALLOC, "Failed to unmap host memory on error {}", errno);
            return synFail;
        }

        return status;
    }

    setFlags(0);
    setDeviceVa(deviceAddress);
    setHostVa(hostAddress);
    setShouldFreeMemory(true);

    LOG_DEBUG(SYN_MEM_ALLOC,
              "Allocated {} bytes on host at address {} and mapped to address {}",
              size,
              hostAddress,
              (void*)deviceAddress);

    return synSuccess;
}

synStatus HostAllocator::FreeMemory()
{
    synStatus status(synSuccess);

    uint64_t size        = getSize();
    uint64_t deviceVA    = getDeviceVa();
    void*    hostAddress = getHostVa();
    uint64_t retAddr     = 0;

    LOG_DEBUG(SYN_MEM_ALLOC,
              "{} try to unmap host address {} from its virtual-address {}",
              HLLOG_FUNC,
              hostAddress,
              (void*)deviceVA);

    status = _unmapMemory(size, (uint64_t)deviceVA, retAddr);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_MEM_ALLOC, "Failed to unmap host memory {} with device-VA {}", hostAddress, deviceVA);
        return synFailedToFreeDeviceMemory;
    }

    if (shouldFreeMemory())
    {
        LOG_DEBUG(SYN_MEM_ALLOC, "trying to de-allocate {} bytes on host at address {}", size, hostAddress);
        if (munmap(hostAddress, size) == -1)
        {
            LOG_ERR(SYN_MEM_ALLOC, "Failed to free host memory {} on error {}", hostAddress, errno);
            return synFailedToFreeDeviceMemory;
        }

        if (m_isHugePageAllocated)
        {
            m_numOfHugePagesAllocated -= div_round_up(size, HUGE_PAGE_SIZE);
        }
        else
        {
            m_numOfRegularPagesAllocated -= div_round_up(size, REGULAR_PAGE_SIZE);
        }
        m_isHugePageAllocated = false;
    }

    setSize(0);
    setFlags(0);
    setDeviceVa(0);
    setShouldFreeMemory(false);

    return synSuccess;
}

synStatus HostAllocator::MapHostMemory(void* hostAddress, uint64_t size, uint64_t reqVAAddress)
{
    synStatus status(synSuccess);

    uint64_t deviceAddress = 0;

    status = _mapMemory(size, (uint64_t)hostAddress, deviceAddress);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Host mapping failed for virtual address {}, size {} on device", deviceAddress, size);

        return status;
    }

    setSize(size);
    setFlags(0);
    setHostVa(hostAddress);
    setShouldFreeMemory(false);

    setDeviceVa(deviceAddress);

    LOG_DEBUG(SYN_MEM_ALLOC, "Mapped host buffer {} to device virtual-address {}", hostAddress, (void*)deviceAddress);
    return status;
}

synStatus HostAllocator::UnmapHostMemory()
{
    synStatus status(synSuccess);

    uint64_t size     = getSize();
    uint64_t deviceVA = getDeviceVa();
    uint64_t retAddr  = 0;

    status = _unmapMemory(size, deviceVA, retAddr);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_MEM_ALLOC, "Failed to unmap host memory with device-VA {}", deviceVA);
        return synFailedToFreeDeviceMemory;
    }

    LOG_DEBUG(SYN_MEM_ALLOC, "Un-mapped host buffer {} from device virtual-address {}", getHostVa(), deviceVA);
    return status;
}

synStatus HostAllocator::_allocateMemory(const uint64_t& size, void** pHostAddress, bool isUserRequest)
{
    static bool s_hpLoggedAsErr = false;
    unsigned    logLevel        = SPDLOG_LEVEL_WARN;
    if (pHostAddress == nullptr)
    {
        LOG_ERR(SYN_MEM_ALLOC, "null pHostAddress pointer");
        return synInvalidArgument;
    }

    int  mmapFlags      = MAP_SHARED | MAP_ANONYMOUS;
    int  prot           = PROT_READ | PROT_WRITE;
    bool allocHugePages = false;

    if ((size > (HUGE_PAGE_SIZE / 2)) &&
        ((GCFG_DISABLE_SYNAPSE_HUGE_PAGES.value() == false) || (isUserRequest)))
    {
        allocHugePages = true;
        mmapFlags |= MAP_HUGE_2MB | MAP_HUGETLB;
    }

    uint64_t allocatedPageSize = allocHugePages ? HUGE_PAGE_SIZE : REGULAR_PAGE_SIZE;
    uint64_t allocatedSize     = allocatedPageSize * div_round_up(size, allocatedPageSize);

    *pHostAddress = mmap(0, size, prot, mmapFlags, -1, 0);

    if (*pHostAddress == MAP_FAILED)
    {
        if (allocHugePages)
        {
            if (unlikely(s_hpLoggedAsErr == false))
            {
                logLevel        = SPDLOG_LEVEL_ERROR;
                s_hpLoggedAsErr = true;
            }
            LOG_PERIODIC_BY_LEVEL(SYN_MEM_ALLOC,
                                  logLevel,
                                  std::chrono::milliseconds(1000),
                                  10,
                                  "Cannot allocate HPs in host memory due to {},"
                                  " requested size for alloc {} total HPs allocated by Synapse{},"
                                  " Total HPs on this machine: {}, free HPs after synapse init {},"
                                  " will allocate regular-pages (performance degradation)",
                                  strerror(errno),
                                  size,
                                  m_numOfHugePagesAllocated,
                                  OSAL::getInstance().getTotalHugePages(),
                                  OSAL::getInstance().getFreeHugePagesAtAppStart());
            mmapFlags     = MAP_SHARED | MAP_ANONYMOUS;
            *pHostAddress = mmap(0, size, prot, mmapFlags, -1, 0);
            allocatedSize = REGULAR_PAGE_SIZE * div_round_up(size, REGULAR_PAGE_SIZE);
        }

        if (*pHostAddress == MAP_FAILED)
        {
            LOG_ERR(SYN_MEM_ALLOC, "Failed to allocate host memory due to {}", strerror(errno));
            LOG_ERR(SYN_MEM_ALLOC,
                    "Host-allocations status: huge-page-num {} regular-pages-num {}",
                    m_numOfHugePagesAllocated,
                    m_numOfRegularPagesAllocated);
            return synOutOfHostMemory;
        }

        m_numOfRegularPagesAllocated += div_round_up(size, REGULAR_PAGE_SIZE);
    }
    else
    {
        if (allocHugePages)
        {
            m_isHugePageAllocated = true;
            m_numOfHugePagesAllocated += div_round_up(size, HUGE_PAGE_SIZE);
        }
        else
        {
            m_numOfRegularPagesAllocated += div_round_up(size, REGULAR_PAGE_SIZE);
        }
    }

    /* in order to prevent physical pages mapped to device from being swapped due to fork copyOnWrite
       mechanism, thus causing MMU mapping problems, need to add madvise flag to those pages.*/
    int ret = madvise((void*)*pHostAddress, allocatedSize, MADV_DONTFORK);
    if (ret)
    {
        LOG_ERR(SYN_API,
                "madvise failed with errno={}, hostAddr=0x{:x}, allocatedSize={}",
                std::strerror(errno),
                (uint64_t)*pHostAddress,
                allocatedSize);
        if (munmap((void*)*pHostAddress, allocatedSize) == -1)
        {
            LOG_ERR(SYN_API, "Failed to unmap host memory on error {}", errno);
        }
        return synFail;
    }

    setSize(allocatedSize);
    return synSuccess;
}

synStatus HostAllocator::_mapMemory(const uint64_t& size, const uint64_t& hostAddress, uint64_t& deviceAddress)
{
    return MemoryIoctl(0, hostAddress, HL_MEM_OP_MAP, size, HL_MEM_USERPTR, &deviceAddress);
}

synStatus HostAllocator::_unmapMemory(const uint64_t& size, const uint64_t& deviceAddress, uint64_t retAddr)
{
    return MemoryIoctl(deviceAddress, 0, HL_MEM_OP_UNMAP, size, HL_MEM_USERPTR, &retAddr);
}

void DeviceHeapAllocatorManager::Free(deviceAddrOffset ptr)
{
    uint64_t released = HeapAllocatorBestFit::FreeReturnInfo(ptr);
    if (m_isUpdateDriver)
    {
        updateDriver(released);
    }
};

Settable<deviceAddrOffset> DeviceHeapAllocatorManager::Allocate(uint64_t size,
                                                                uint64_t alignment,
                                                                uint64_t offset,
                                                                bool     allowFailure,
                                                                uint64_t requestedAddress)
{
    std::pair<Settable<deviceAddrOffset>, uint64_t> addrAndSize =
        HeapAllocatorBestFit::AllocateReturnInfo(size, alignment, offset, allowFailure, requestedAddress);

    if (addrAndSize.first.is_set())
    {
        if (m_isUpdateDriver)
        {
            updateDriver(addrAndSize.second);
        }
    }
    return addrAndSize.first;
}

bool DeviceHeapAllocatorManager::allocateReqRange(const Range& reqRange, unsigned pad)
{
    bool res = HeapAllocatorBestFit::allocateReqRange(reqRange, pad);
    if (res)
    {
        if (m_isUpdateDriver)
        {
            updateDriver(reqRange.size);
        }
    }
    return res;
};

void DeviceHeapAllocatorManager::updateDriver(uint64_t allocSize)
{
    if (0 != allocSize)
    {
        hlthunk_device_memory_alloc(m_fd, allocSize, 0, HL_MEM_CONTIGUOUS, 0x0);  // function always returns 0
    }
    else
    {
        LOG_ERR(SYN_MEM_ALLOC, "{}: Trying to update driver about 0 sized mem allocation", HLLOG_FUNC);
    }
}

ManagedBufferAllocator::ManagedBufferAllocator(std::shared_ptr<DeviceHeapAllocatorManager> heapAllocator)
{
    if (heapAllocator == nullptr)
    {
        LOG_ERR(SYN_MEM_ALLOC, "{}: allocator is nullptr", HLLOG_FUNC);
        throw ConstructObjectFailure();
    }

    m_spAllocatorManager = heapAllocator;
}

synStatus ManagedBufferAllocator::AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest)
{
    // 1. Validation
    if (size == 0)
    {
        LOG_ERR(SYN_MEM_ALLOC, "invalid size - requested to allocate buffer with 0 bytes");
        return synInvalidArgument;
    }

    return _allocateMemory(reqVAAddress, size, m_defaultAlignment, true);
}

synStatus ManagedBufferAllocator::FreeMemory()
{
    if (shouldFreeMemory())
    {
        // Until refactoring...
        deviceAddrOffset deviceAddress = (deviceAddrOffset)getHostVa();

        m_spAllocatorManager->Free(deviceAddress);
    }

    return synSuccess;
}

synStatus
ManagedBufferAllocator::_allocateMemory(uint64_t reqVAAddress, uint64_t size, unsigned alignment, bool shouldFreeMemory)
{
    Settable<deviceAddrOffset> allocatedAddress =
        m_spAllocatorManager->Allocate(size, alignment, 0, false, reqVAAddress);

    if (!allocatedAddress.is_set())
    {
        LOG_ERR(SYN_MEM_ALLOC,
                "{}: Failed to allocate {} bytes with alignment of {} bytes",
                HLLOG_FUNC,
                size,
                alignment);
        return synFail;
    }

    setSize(size);
    setShouldFreeMemory(shouldFreeMemory);
    // Until refactoring...
    uint64_t deviceAddress = allocatedAddress.value();
    setHostVa((uint64_t*)deviceAddress);
    setDeviceVa(deviceAddress);

    return synSuccess;
}
