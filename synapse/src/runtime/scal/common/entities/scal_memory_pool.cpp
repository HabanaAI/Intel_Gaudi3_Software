#include "scal_memory_pool.hpp"

#include "defs.h"

#include "log_manager.h"
#include "runtime/scal/common/infra/scal_types.hpp"

ScalMemoryPool::ScalMemoryPool(scal_handle_t devHndl, const std::string& name)
: m_devHndl(devHndl), m_name(name), m_mpHndl(nullptr), m_mpInfo {}, m_memReserved(0)
{
    HB_ASSERT((m_devHndl != nullptr), "m_devHndl is nullptr");
}

synStatus ScalMemoryPool::init()
{
    ScalRtn rc = scal_get_pool_handle_by_name(m_devHndl, m_name.c_str(), &m_mpHndl);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                     "devHndl 0x{:x} scal_get_pool_handle_by_name failed with rc {}",
                     TO64(m_devHndl),
                     rc);
        return synFail;
    }

    rc = scal_pool_get_info(m_mpHndl, &m_mpInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                     "devHndl 0x{:x} m_mpHndl 0x{:x} scal_get_pool_handle_by_name failed with rc {}",
                     TO64(m_devHndl),
                     TO64(m_mpHndl),
                     rc);
        return synFail;
    }

    m_memReserved = m_mpInfo.totalSize - m_mpInfo.freeSize;

    LOG_DEBUG(
        SYN_MEM_ALLOC,
        "devHndl 0x{:x} name {} mpHndl 0x{:x} dAddr 0x{:x} hAddr 0x{:x} total {} free {} reserved {}",
        TO64(m_devHndl),
        m_name,
        TO64(m_mpHndl),
        m_mpInfo.device_base_address,
        TO64(m_mpInfo.host_base_address),
        m_mpInfo.totalSize,
        m_mpInfo.freeSize,
        m_memReserved);

    return synSuccess;
}

synStatus ScalMemoryPool::allocateDeviceMemory(uint64_t memSize, scal_buffer_handle_t& bufHndl)
{
    const ScalRtn rc = scal_allocate_buffer(m_mpHndl, memSize, &bufHndl);

    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                     "devHndl 0x{:x} m_mpHndl 0x{:x} scal_allocate_buffer size {} failed with rc {}",
                     TO64(m_devHndl),
                     TO64(m_mpHndl),
                     memSize,
                     rc);
        return synFail;
    }

    LOG_DEBUG(SYN_MEM_ALLOC,
                   "devHndl 0x{:x} m_mpHndl 0x{:x} bufHndl 0x{:x} memSize {} allocated",
                   TO64(m_devHndl),
                   TO64(m_mpHndl),
                   TO64(bufHndl),
                   memSize);

    return synSuccess;
}

synStatus
ScalMemoryPool::getDeviceMemoryAddress(const scal_buffer_handle_t& bufHndl, uint32_t& coreAddr, uint64_t& devAddr)
{
    scal_buffer_info_t bufferInfo;
    const ScalRtn      rc = scal_buffer_get_info(bufHndl, &bufferInfo);

    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                     "devHndl 0x{:x} m_mpHndl 0x{:x} bufHndl 0x{:x} scal_buffer_get_info failed with rc {}",
                     TO64(m_devHndl),
                     TO64(m_mpHndl),
                     TO64(bufHndl),
                     rc);
        return synFail;
    }

    coreAddr = bufferInfo.core_address;
    devAddr  = bufferInfo.device_address;
    LOG_DEBUG(SYN_MEM_ALLOC,
                   "devHndl 0x{:x} mpHndl 0x{:x} bufHndl 0x{:x} coreAddr 0x{:x} devAddr 0x{:x}",
                   TO64(m_devHndl),
                   TO64(m_mpHndl),
                   TO64(bufHndl),
                   coreAddr,
                   devAddr);

    return synSuccess;
}

synStatus ScalMemoryPool::releaseDeviceMemory(const scal_buffer_handle_t& bufHndl)
{
    const ScalRtn rc = scal_free_buffer(bufHndl);

    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                     "devHndl 0x{:x} m_mpHndl 0x{:x} bufHndl 0x{:x} scal_free_buffer failed with rc {}",
                     TO64(m_devHndl),
                     TO64(m_mpHndl),
                     TO64(bufHndl),
                     rc);
        return synFail;
    }

    LOG_DEBUG(SYN_MEM_ALLOC,
                   "devHndl 0x{:x} m_mpHndl 0x{:x} bufHndl 0x{:x} released",
                   TO64(m_devHndl),
                   TO64(m_mpHndl),
                   TO64(bufHndl));

    return synSuccess;
}

synStatus ScalMemoryPool::getMemoryStatus(PoolMemoryStatus& poolMemoryStatus) const
{
    scal_memory_pool_info info;
    const ScalRtn         rc = scal_pool_get_info(m_mpHndl, &info);

    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                     "devHndl 0x{:x} m_mpHndl 0x{:x} scal_pool_get_info failed with rc {}",
                     TO64(m_devHndl),
                     TO64(m_mpHndl),
                     rc);
        return synFail;
    }
/*
    Both API functions: synDeviceGetInfo and synDeviceGetMemoryInfo should return the same values for 'dramSize' and 'total' (respectively).
    synDeviceGetMemoryInfo function:
        total - is the total device memory (HBM) that is available for the User (framework), which means it equals to 'free' after Synapse init.
        When the SCAL layer (or the Runtime) allocates device memory (for internal use), this will be reduced from the total memory size.
        free - will always indicates how much device memory the User may use for his workloads
        Summary:
            total = (HBM phy value size) - (Synapse internal use)
            free = total - (user's device memory allocations).
    synDeviceGetInfo:
        dramSize = (HBM phy value size) - (Synapse internal use)    [same as total].
*/
    poolMemoryStatus.free        = info.freeSize;
    poolMemoryStatus.total       = info.totalSize - m_memReserved;
    poolMemoryStatus.devBaseAddr = info.device_base_address;
    poolMemoryStatus.totalSize   = info.totalSize;

    return synSuccess;
}
