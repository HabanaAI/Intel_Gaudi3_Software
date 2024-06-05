#include "device_mem_alloc.hpp"
#include "defenders.h"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "runtime/common/osal/osal.hpp"
#include "runtime/common/osal/buffer_allocator.hpp"
#include "runtime/scal/common/entities/scal_device_allocator.hpp"
#include "runtime/scal/common/entities/scal_dev.hpp"

#include <unistd.h>

const synMemFlags bringup_ddr_bug_workaround_flags = synMemFlags::synMemHost;

DevMemoryAlloc::DevMemoryAlloc(synDeviceType devType, uint64_t dramSize, uint64_t dramBaseAddress)
: m_devType(devType),
  m_dramSize(dramSize),
  m_dramBaseAddress(dramBaseAddress),
  m_dramUsed(0),
  m_spDeviceMemoryAllocatorManager(new DeviceHeapAllocatorManager(OSAL::getInstance().getFd(), true)),
  m_userAddressMapper("User"),
  m_topologiesAddressMapper("Synapse")
{
}

synStatus DevMemoryAlloc::deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest)
{
    LOG_TRACE(SYN_MEM_ALLOC,
              "{}: pBuffer 0x{:x} flags {} isUserRequest {}",
              HLLOG_FUNC,
              (uint64_t)pBuffer,
              flags,
              isUserRequest);

    synStatus        status;
    uint64_t         buffSize  = 0;
    BufferAllocator* pBufAlloc = nullptr;
    uint64_t         vaAddress = 0;

    if (pBuffer == nullptr)
    {
        return synSuccess;
    }

    if (flags == synMemFlags::synMemHost)
    {
        // When deallocating, we don't want to pass the buffer-size, but we do want to validate that we delete the exact
        // buffer (key)
        bool isExactKeyFound = false;

        eMappingStatus mappingStatus = getDeviceVirtualAddress(isUserRequest, pBuffer, 1, &vaAddress, &isExactKeyFound);
        if ((mappingStatus != HATVA_MAPPING_STATUS_FOUND) || (!isExactKeyFound))
        {
            LOG_ERR(SYN_MEM_ALLOC, "{}: Can not deallocate buffer 0x{:x} (not found)", HLLOG_FUNC, (uint64_t)pBuffer);
            return synInvalidArgument;
        }
    }
    else
    {
        vaAddress = (uint64_t)pBuffer;
    }

    LOG_TRACE(SYN_MEM_ALLOC,
              "{}: deallocating VA 0x{:x} of buffer 0x{:x}",
              HLLOG_FUNC,
              (uint64_t)vaAddress,
              (uint64_t)pBuffer);

    if (pBuffer == nullptr)
    {
        LOG_ERR(SYN_MEM_ALLOC, "{}: buffer is nullptr", HLLOG_FUNC);
        return synInvalidArgument;
    }

    status = OSAL::getInstance().getBufferAllocator(vaAddress, &pBufAlloc);

    if ((status != synSuccess) || (pBufAlloc == nullptr))
    {
        LOG_ERR(SYN_MEM_ALLOC,
                "{}: Can not delete an unrecognized pointer 0x{:x} status {} pBufAlloc 0x{:x}",
                HLLOG_FUNC,
                TO64(pBuffer),
                status,
                TO64(pBufAlloc));
        return synFail;
    }

    buffSize = pBufAlloc->getSize();

    status = OSAL::getInstance().clearBufferAllocator(vaAddress);
    if (status == synSuccess)
    {
        if (flags == synMemFlags::synMemHost)
        {
            getAddressMapper(isUserRequest).clearMapping(pBuffer);
        }
        else
        {
            m_dramUsed -= buffSize;
        }
    }

    status = pBufAlloc->FreeMemory();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_MEM_ALLOC, "{}: Can not FreeMemory of pBufAlloc ", HLLOG_FUNC);
    }
    else
    {
        LOG_TRACE(SYN_MEM_ALLOC, "Destroyed device buffer successfully");
    }
    delete pBufAlloc;

    return status;
}

eMappingStatus DevMemoryAlloc::getDeviceVirtualAddress(bool         isUserRequest,
                                                       void*        hostAddress,
                                                       uint64_t     bufferSize,
                                                       uint64_t*    pDeviceVA,
                                                       bool*        pIsExactKeyFound)
{
    return getAddressMapper(isUserRequest)
        .getDeviceVirtualAddress(hostAddress, bufferSize, pDeviceVA, pIsExactKeyFound);
}

bool DevMemoryAlloc::setHostToDeviceAddressMapping(void*              hostAddress,
                                                   uint64_t           bufferSize,
                                                   uint64_t           vaAddress,
                                                   bool               isUserRequest,
                                                   const std::string& mappingDesc)
{
    HostAddrToVirtualAddrMapper& mapper = isUserRequest ? m_userAddressMapper : m_topologiesAddressMapper;
    return mapper.setMapping(hostAddress, vaAddress, bufferSize, isUserRequest, mappingDesc);
}

synStatus DevMemoryAlloc::mapBufferToDevice(uint64_t           size,
                                            void*              buffer,
                                            bool               isUserRequest,
                                            uint64_t           reqVAAddress,
                                            const std::string& mappingDesc)
{
    synStatus status    = synSuccess;
    uint64_t  vaAddress = 0;
    eMappingStatus mappingStatus;

    const unsigned sleepTimeMsSec  = 1;
    unsigned       MaxNumerOfTries = GCFG_MAX_WAIT_TIME_FOR_MAPPING_IN_STREAM_COPY.value() / sleepTimeMsSec;
    unsigned       iter            = 0;
    do
    {
        mappingStatus = getDeviceVirtualAddress(isUserRequest, buffer, size, &vaAddress);
        if (mappingStatus == HATVA_MAPPING_STATUS_FOUND)
        {
            if (isUserRequest)
            {
                LOG_TRACE(SYN_MEM_ALLOC,
                          "{}: Buffer 0x{:x} (isUserRequest {}) is already mapped, mappingDesc {}",
                          HLLOG_FUNC,
                          (uint64_t)buffer,
                          isUserRequest,
                          mappingDesc);
                return synSuccess;
            }
            if (iter > MaxNumerOfTries)
            {
                LOG_ERR(SYN_MEM_ALLOC,
                        "{}: Can not map between Host address {} (isUserRequest {}), buffer is "
                        "already mapped,  mappingDesc {}",
                        HLLOG_FUNC,
                        buffer,
                        isUserRequest,
                        mappingDesc);
                return synFail;
            }

            LOG_DEBUG(SYN_MEM_ALLOC,
                      "{}: Buffer 0x{:x} (isUserRequest {}) is already mapped iter {},  mappingDesc {}",
                      HLLOG_FUNC,
                      (uint64_t)buffer,
                      isUserRequest,
                      iter,
                      mappingDesc);
            usleep(sleepTimeMsSec * 1000);
            iter++;
        }
        else
        {
            break;
        }
    } while (true);

    if (mappingStatus != HATVA_MAPPING_STATUS_NOT_FOUND)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                "{}: Invalid request for mapping Buffer 0x{:x} (isUserRequest {}) mappingStatus = {} "
                "mappingDesc {}",
                HLLOG_FUNC,
                (uint64_t)buffer,
                isUserRequest,
                mappingStatus,
                mappingDesc);
        return synInvalidArgument;
    }

    BufferAllocator* pBufAlloc = new HostAllocator;

    status = pBufAlloc->MapHostMemory(buffer, size, reqVAAddress);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_MEM_ALLOC, "Can not map memory");

        delete pBufAlloc;

        return status;
    }

    vaAddress = pBufAlloc->getDeviceVa();
    LOG_TRACE(SYN_MEM_ALLOC,
              "Mapping: host va 0x{:x} device va 0x{:x} size 0x{:x} mappingDesc {}",
              (uint64_t)buffer,
              (uint64_t)vaAddress,
              size,
              mappingDesc);

    bool ret = setHostToDeviceAddressMapping(buffer, size, vaAddress, isUserRequest, mappingDesc);

    if (ret)
    {
        LOG_TRACE(SYN_MEM_ALLOC,
                  "{}: Setting mapping between Host address {} and device VA {} mappingDesc {}",
                  HLLOG_FUNC,
                  buffer,
                  vaAddress,
                  mappingDesc);
    }
    else
    {
        LOG_WARN(SYN_MEM_ALLOC,
                 "{}: Can not map between Host address {} and device VA {} mappingDesc {}",
                 HLLOG_FUNC,
                 buffer,
                 vaAddress,
                 mappingDesc);

        pBufAlloc->UnmapHostMemory();
        delete pBufAlloc;

        return synFail;
    }

    // Register the VA globally
    OSAL::getInstance().setBufferAllocator(vaAddress, pBufAlloc);

    return status;
}

synStatus DevMemoryAlloc::unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize)
{
    LOG_TRACE(SYN_API, "{}: buffer = 0x{:x}", HLLOG_FUNC, (uint64_t)buffer);

    synStatus status;

    BufferAllocator* pBufAlloc = nullptr;
    // When un-mapping, we don't want to pass the buffer-size, but we do want to validate that we delete the exact
    // buffer (key)
    uint64_t vaAddress = 0;
    bool  isExactKeyFound = false;

    eMappingStatus mappingStatus = getDeviceVirtualAddress(isUserRequest, buffer, 1, &vaAddress, &isExactKeyFound);
    if ((mappingStatus == HATVA_MAPPING_STATUS_NOT_FOUND) || (!isExactKeyFound))
    {
        if (isUserRequest)
        {
            LOG_TRACE(SYN_API,
                      "{}: Buffer 0x{:x} (isUserRequest {}) is already un-mapped",
                      HLLOG_FUNC,
                      (uint64_t)buffer,
                      isUserRequest);
        }
        else
        {
            LOG_ERR(SYN_API,
                    "{}: Buffer 0x{:x} (isUserRequest {}) is already un-mapped",
                    HLLOG_FUNC,
                    (uint64_t)buffer,
                    isUserRequest);
        }
        return isUserRequest ? synSuccess : synFail;
    }

    status = OSAL::getInstance().getBufferAllocator(vaAddress, &pBufAlloc);

    if ((status != synSuccess) || (pBufAlloc == nullptr))
    {
        LOG_ERR(SYN_API, "{}: Can not un-map an unrecognized buffer {}", HLLOG_FUNC, buffer);
        return synFail;
    }
    if (bufferSize != nullptr)
    {
        *bufferSize = pBufAlloc->getSize();
    }
    status = OSAL::getInstance().clearBufferAllocator(vaAddress);

    getAddressMapper(isUserRequest).clearMapping(buffer);

    status = pBufAlloc->FreeMemory();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_MEM_ALLOC, "{}: Can not FreeMemory of pBufAlloc ", HLLOG_FUNC);
    }
    else
    {
        LOG_TRACE(SYN_MEM_ALLOC, "Destroyed device buffer successfully");
    }
    delete pBufAlloc;

    return status;
}

synStatus DevMemoryAlloc::getDramMemInfo(uint64_t& free, uint64_t& total) const
{
    synStatus status;
    switch (m_devType)
    {
        case synDeviceGaudi:
        {
            status = getNonDeviceMMUMemInfo(free, total);
            break;
        }
        // We are not suppose to get here for Gaudi2/3
        case synDeviceGaudi2:
        case synDeviceGaudi3:
        {
            status = getDeviceMMUMemInfo(free, total);
            break;
        }
        default:
        {
            status = synFail;
        }
    }

    return status;
}

void DevMemoryAlloc::getValidAddressesRange(uint64_t& lowestValidAddress, uint64_t& highestValidAddress) const
{
    lowestValidAddress  = m_dramBaseAddress;
    highestValidAddress = m_dramBaseAddress + m_dramSize;
}

/* This function is used to return the device memory info when there is no MMU
 * towards the DRAM/HBM. As of now Gaudi1 is not using the HBM MMU, so memory allocations are
 * managed by RT
 */
synStatus DevMemoryAlloc::getNonDeviceMMUMemInfo(uint64_t& free, uint64_t& total) const
{
    uint64_t dramUsed = 0;

    // todo anat make sure recipeCacheSize is calculated and reported here- currentely done in test itself
    total    = m_spDeviceMemoryAllocatorManager->GetMemorySize();
    dramUsed = m_spDeviceMemoryAllocatorManager->GetCurrentlyUsed();

    free = total - dramUsed;

    LOG_TRACE(SYN_DEVICE, "{} total memory {} m_dramUsed {} free memory {}", HLLOG_FUNC, total, dramUsed, free);

    return synSuccess;
}

/* This function is used to return the device memory info when there is MMU is active
 * towards the DRAM/HBM. Memory allocations are managed by the LKD via IOCTL
 */
synStatus DevMemoryAlloc::getDeviceMMUMemInfo(uint64_t& free, uint64_t& total) const
{
    uint64_t dramUsed = 0;

    total    = m_dramSize;
    dramUsed = m_dramUsed;

    free = total - dramUsed;

    LOG_TRACE(SYN_DEVICE, "{} total memory {} m_dramUsed {} free memory {}", HLLOG_FUNC, total, dramUsed, free);

    return synSuccess;
}

void DevMemoryAlloc::dfaLogMappedMem() const
{
    m_userAddressMapper.dfaLogMapper();
    m_topologiesAddressMapper.dfaLogMapper();
}

synStatus DevMemoryAlloc::destroyHostAllocations(bool isUserAllocations)
{
    synStatus status = synSuccess;

    HostAddrToVirtualAddrMapper&  addressMapper   = getAddressMapper(isUserAllocations);
    HostAddrToVirtualAddrIterator hostAddressIter = addressMapper.begin();

    if (addressMapper.size() == 0)
    {
        return synSuccess;
    }

    if (!isUserAllocations)
    {
        LOG_WARN(SYN_DEVICE, "{}: isUserAllocations {} Size {}", HLLOG_FUNC, isUserAllocations, addressMapper.size());
    }

    while (hostAddressIter != addressMapper.end())
    {
        void* hostAddress = hostAddressIter->first;

        hostAddressIter++;
        if (deallocateMemory(hostAddress, synMemFlags::synMemHost, isUserAllocations) != synSuccess)
        {
            status = synFail;
        }
    }
    return status;
}

/**********************************************************************************/

DevMemoryAllocCommon::DevMemoryAllocCommon(synDeviceType devType,
                                           uint64_t      dramSize,
                                           uint64_t      dramBaseAddress)
: DevMemoryAlloc(devType, dramSize, dramBaseAddress), m_mngMemoryAllocated(false)
{
}

DevMemoryAllocCommon::~DevMemoryAllocCommon() {}

synStatus DevMemoryAllocCommon::allocate()
{
    m_spDeviceMemoryAllocatorManager->Init(m_dramSize, m_dramBaseAddress);

    return synSuccess;
}

synStatus DevMemoryAllocCommon::release()
{
    return synSuccess;
}

synStatus DevMemoryAllocCommon::allocateMemory(uint64_t           size,
                                               uint32_t           flags,
                                               void**             buffer,
                                               bool               isUserRequest,
                                               uint64_t           reqVAAddress,
                                               const std::string& mappingDesc,
                                               uint64_t*          deviceVA)
{
    LOG_DEBUG(SYN_MEM_ALLOC, "{}: isUserRequest {}", HLLOG_FUNC, isUserRequest);

    VERIFY_IS_NULL_POINTER(SYN_MEM_ALLOC, buffer, "buffer");

    if (size == 0)
    {
        LOG_DEBUG(SYN_MEM_ALLOC, "{}: zero size allocation requested", HLLOG_FUNC);
        *buffer = 0;
        return synSuccess;
    }

    // Allocate on Host memory or on device
    BufferAllocator* pBufAlloc = nullptr;
    // Allocate on Host memory
    if (flags & synMemFlags::synMemHost)
    {
        pBufAlloc = new HostAllocator;
    }
    // Allocate on device
    else
    {
        if (m_devType == synDeviceGaudi)
        {
            pBufAlloc = allocateManagedBufferAllocator();

            if (pBufAlloc == nullptr)
            {
                return synFail;
            }
        }
        else
        {
            pBufAlloc = new DeviceAllocator;
        }
    }

    if (pBufAlloc == nullptr)
    {
        LOG_ERR(SYN_MEM_ALLOC, "{}: Can not allocate buffer-allocator, no memory on host", HLLOG_FUNC);
        *buffer = 0;
        return synOutOfHostMemory;
    }

    synStatus status = pBufAlloc->AllocateMemory(reqVAAddress, size, isUserRequest);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_MEM_ALLOC, "Can not allocate memory");
        *buffer = 0;
        delete pBufAlloc;
        return status;
    }

    uint64_t vaAddress = pBufAlloc->getDeviceVa();
    LOG_TRACE(SYN_MEM_ALLOC,
              "host va: 0x{:x} device va: 0x{:x} size: 0x{:x}",
              (uint64_t)pBufAlloc->getHostVa(),
              vaAddress,
              (uint64_t)size);

    if (flags & synMemFlags::synMemHost)
    {
        LOG_DEBUG(SYN_MEM_ALLOC, "{}: Allocated {} bytes of host-memory for device", HLLOG_FUNC, size);

        *buffer = (uint64_t*)pBufAlloc->getHostVa();

        setHostToDeviceAddressMapping(*buffer, size, vaAddress, isUserRequest, mappingDesc);
        LOG_DEBUG(SYN_MEM_ALLOC,
                  "{}: Setting mapping between Host address {} and device VA {}",
                  HLLOG_FUNC,
                  *buffer,
                  vaAddress);
    }
    else
    {
        LOG_DEBUG(SYN_MEM_ALLOC, "Allocated {} bytes of device-memory", size);

        *buffer = (uint64_t*)vaAddress;
        m_dramUsed += size;
    }

    if (deviceVA)
    {
        *deviceVA = vaAddress;
    }

    // Register the VA globally
    OSAL::getInstance().setBufferAllocator(vaAddress, pBufAlloc);

    return status;
}

BufferAllocator* DevMemoryAllocCommon::allocateManagedBufferAllocator()
{
    ManagedBufferAllocator* pManagedBufferAllocator;

    try
    {
        pManagedBufferAllocator = new ManagedBufferAllocator(m_spDeviceMemoryAllocatorManager);
    }
    catch (ConstructObjectFailure&)
    {
        LOG_ERR(SYN_MEM_ALLOC, "Can not initialize a managed buffer allocator");
        return nullptr;
    }

    return pManagedBufferAllocator;
}


/**********************************************************************************/
DevMemoryAllocScal::DevMemoryAllocScal(ScalDev*      scalDev,
                                       synDeviceType devType,
                                       uint64_t      dramSize,
                                       uint64_t      dramBaseAddress)
: DevMemoryAlloc(devType, dramSize, dramBaseAddress), m_scalDev(scalDev)
{
    s_debugLastConstructedAllocator = this;
}

synStatus DevMemoryAllocScal::allocateMemory(uint64_t           size,
                                             uint32_t           flags,
                                             void**             buffer,
                                             bool               isUserRequest,
                                             uint64_t           reqVAAddress,
                                             const std::string& mappingDesc,
                                             uint64_t*          deviceVA)
{
    LOG_DEBUG(SYN_MEM_ALLOC, "{}: isUserRequest {}", HLLOG_FUNC, isUserRequest);

    VERIFY_IS_NULL_POINTER(SYN_MEM_ALLOC, buffer, "buffer");

    if (size == 0)
    {
        LOG_DEBUG(SYN_MEM_ALLOC, "{}: zero size allocation requested", HLLOG_FUNC);
        *buffer = 0;
        return synSuccess;
    }

    // Allocate on Host memory or on device
    BufferAllocator* pBufAlloc = nullptr;
    // Allocate on Host memory
    if (flags & synMemFlags::synMemHost)
    {
        pBufAlloc = new HostAllocator;
    }
    // Allocate on device
    else
    {
        pBufAlloc = new ScalDeviceAllocator((*m_scalDev->getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)));
    }

    if (pBufAlloc == nullptr)
    {
        LOG_ERR(SYN_MEM_ALLOC, "{}: Can not allocate buffer-allocator, no memory on host", HLLOG_FUNC);
        *buffer = 0;
        return synOutOfHostMemory;
    }

    synStatus status = pBufAlloc->AllocateMemory(reqVAAddress, size, isUserRequest);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_MEM_ALLOC, "Can not allocate memory");
        *buffer = 0;
        delete pBufAlloc;
        return status;
    }

    uint64_t vaAddress = pBufAlloc->getDeviceVa();
    LOG_TRACE(SYN_MEM_ALLOC,
              "host va: 0x{:x} device va: 0x{:x} size: 0x{:x}",
              (uint64_t)pBufAlloc->getHostVa(),
              (uint64_t)vaAddress,
              (uint64_t)size);

    if (flags & synMemFlags::synMemHost)
    {
        LOG_DEBUG(SYN_MEM_ALLOC, "{}: Allocated {} bytes of host-memory", HLLOG_FUNC, size);

        *buffer = pBufAlloc->getHostVa();

        setHostToDeviceAddressMapping(*buffer, size, vaAddress, isUserRequest, mappingDesc);
        LOG_DEBUG(SYN_MEM_ALLOC,
                  "{}: Setting mapping between Host address {} and device VA {}",
                  HLLOG_FUNC,
                  *buffer,
                  vaAddress);
    }
    else
    {
        LOG_DEBUG(SYN_MEM_ALLOC, "Allocated {} bytes of device-memory", size);

        *buffer = (uint64_t*)vaAddress;
        m_dramUsed += size;
    }

    if (deviceVA)
    {
        *deviceVA = vaAddress;
    }

    // Register the VA globally
    OSAL::getInstance().setBufferAllocator(vaAddress, pBufAlloc);

    return status;
}

DevMemoryAlloc* DevMemoryAllocScal::s_debugLastConstructedAllocator = nullptr;
