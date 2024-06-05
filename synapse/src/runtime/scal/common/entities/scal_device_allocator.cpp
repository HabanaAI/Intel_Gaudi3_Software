#include "scal_device_allocator.hpp"

#include "log_manager.h"
#include "runtime/common/osal/osal.hpp"

#include "runtime/scal/common/entities/scal_memory_pool.hpp"

ScalDeviceAllocator::ScalDeviceAllocator(ScalMemoryPool& mpGlobalHbm) : m_mpGlobalHbm(mpGlobalHbm) {}

synStatus ScalDeviceAllocator::AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest)
{
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
        LOG_ERR(SYN_MEM_ALLOC, "device not acquired");
        return synInvalidArgument;
    }

    // 2. Allocation
    scal_buffer_handle_t ctrlBuffHndl = nullptr;
    synStatus            status       = m_mpGlobalHbm.allocateDeviceMemory(size, ctrlBuffHndl);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL,
                "{}: Device allocation failed for virtual address {}, size {} on device",
                HLLOG_FUNC,
                reqVAAddress,
                size);
        return status;
    }

    // 3. Mapping
    uint32_t coreAddr = 0;
    uint64_t devAddr  = 0;
    status            = m_mpGlobalHbm.getDeviceMemoryAddress(ctrlBuffHndl, coreAddr, devAddr);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL,
                "Device mapping failed for virtual address {}, size {} on device",
                reqVAAddress,
                size);

        return status;
    }

    // 4. Storing information
    setSize(size);
    setFlags(0);
    setDeviceVa(devAddr);
    setHandle(reinterpret_cast<uint64_t>(ctrlBuffHndl));
    setShouldFreeMemory(true);

    LOG_DEBUG(SYN_MEM_ALLOC, "{}: devAddr 0x{:x}", HLLOG_FUNC, devAddr);
    return synSuccess;
}

synStatus ScalDeviceAllocator::FreeMemory()
{
    synStatus status(synSuccess);

    uint64_t             deviceVa     = getDeviceVa();
    uint64_t             handle       = getHandle();
    scal_buffer_handle_t ctrlBuffHndl = reinterpret_cast<scal_buffer_handle_t>(handle);

    // 1. Validation
    if (!OSAL::getInstance().isAcquired())
    {
        LOG_ERR(SYN_MEM_ALLOC, "device not acquired");
        return synInvalidArgument;
    }

    // 2. Free
    if (shouldFreeMemory())
    {
        status = m_mpGlobalHbm.releaseDeviceMemory(ctrlBuffHndl);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_OSAL,
                    "Device memory-free failed for virtual address {} handle {}",
                    deviceVa,
                    handle);

            return status;
        }
    }

    // 3. Clear information
    setSize(0);
    setFlags(0);
    setDeviceVa(0);
    setHandle(0);
    setShouldFreeMemory(false);

    LOG_DEBUG(SYN_MEM_ALLOC, "{}: deviceAddress 0x{:x}", HLLOG_FUNC, deviceVa);
    return synSuccess;
}
