#include "memory_manager.hpp"

#include "habana_global_conf_runtime.h"
#include "memory_allocator_utils.hpp"
#include "runtime/common/device/device_mem_alloc.hpp"
#include "synapse_runtime_logging.h"
#include "types_exception.h"

MemoryManager::MemoryManager(DevMemoryAllocInterface& rDevMemAlloc)
: m_rDevMemAlloc(rDevMemAlloc), m_mappedMemorySize(0)
{
}

MemoryManager::~MemoryManager() {}

synStatus MemoryManager::mapBuffer(uint64_t&          hostVirtualAddress,
                                   void*              buffer,
                                   uint64_t           size,
                                   bool               isUserRequest,
                                   const std::string& mappingDesc,
                                   uint64_t           requestedVirtualAddress /* = INVALID_ADDRESS */) const
{
    synStatus status =
        m_rDevMemAlloc.mapBufferToDevice(size, buffer, isUserRequest, requestedVirtualAddress, mappingDesc);

    if (status == synSuccess)
    {
        eMappingStatus mappingStatus =
            m_rDevMemAlloc.getDeviceVirtualAddress(isUserRequest, buffer, size, &hostVirtualAddress, nullptr);
        if (mappingStatus == HATVA_MAPPING_STATUS_FOUND)
        {
            LOG_TRACE(SYN_MEM_ALLOC,
                      "{}: pBuffer 0x{:x} is mapped to device VA 0x{:x} with size {}",
                      HLLOG_FUNC,
                      (uint64_t)buffer,
                      (uint64_t)hostVirtualAddress,
                      size);
            m_mappedMemorySize += size;
            return synSuccess;
        }
    }

    LOG_ERR(SYN_MEM_ALLOC,
            "{}: Failed mapping pBuffer 0x{:x} size 0x{:x} mappingDesc {}, isUserRequest {}",
            HLLOG_FUNC,
            (uint64_t)buffer,
            size,
            mappingDesc,
            isUserRequest);
    return status;
}

synStatus MemoryManager::unmapBuffer(void* buffer, bool isUserRequest) const
{
    uint64_t  bufferSize = 0;
    synStatus status     = m_rDevMemAlloc.unmapBufferFromDevice(buffer, isUserRequest, &bufferSize);
    if (status != synSuccess)
    {
        LOG_CRITICAL(SYN_MEM_ALLOC,
                     "{}: Failed to unmap commands-buffer 0x{:x} from device",
                     HLLOG_FUNC,
                     (uint64_t)buffer);
        return status;
    }
    m_mappedMemorySize -= bufferSize;
    return synSuccess;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PoolMemoryMapper::PoolMemoryMapper(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize)
: MemoryManager(rDevMemAlloc)
{
    // pool is enabled. size is poolSize * MB
    m_poolSize = poolSize * 1024 * 1024;
    // if an error occurs in allocating, an exception is raised
    m_poolBaseAddr = (uint64_t)MemoryAllocatorUtils::alloc_memory_to_be_mapped_to_device(m_poolSize);

    std::string mappingDesc("PoolMemoryMapper");

    synStatus status =
        MemoryManager::mapBuffer(m_poolBaseMapped, (void*)m_poolBaseAddr, m_poolSize, false, mappingDesc);
    if (status != synSuccess)
    {
        LOG_CRITICAL(SYN_MEM_ALLOC,
                     "{}: Failed to map pool memory. buf= 0x{:x} size = 0x{:x} to device",
                     HLLOG_FUNC,
                     m_poolBaseAddr,
                     m_poolSize);
        throw SynapseException("PoolMemoryMapper:PoolMemoryMapper");
    }
    m_mappedMemorySize = 0;  // count usage
};

PoolMemoryMapper::~PoolMemoryMapper()
{
    synStatus status = MemoryManager::unmapBuffer((void*)m_poolBaseAddr, false);
    if (status != synSuccess)
    {
        LOG_CRITICAL(SYN_MEM_ALLOC, "{}: Failed to unmap pool memory. buf= 0x{:x}", HLLOG_FUNC, m_poolBaseAddr);
    }
    MemoryAllocatorUtils::free_memory((void*)m_poolBaseAddr, m_poolSize);
}

synStatus
PoolMemoryMapper::mapBufferEx(uint64_t& hostVirtualAddress, uint64_t* newHostAddress, void* buffer, uint64_t size)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    // possible cases
    //    1 - simple from end  (pi >= ci)
    //    2 - wrap  (pi >= ci but not enough room, size < ci so wrap to start)
    //    3 - from start (pi < ci,  pi+size <= ci)
    //    4 - no room - (pi >= ci but size also > ci)
    //        or        (pi < ci pi+size > ci)
    //
    //    we keep max_pi  to detect when ci advances to the last used pi (e.g this is the last allocation on this side
    //    of the pool) and can be reset to 0
    if (m_poolIsFull)
    {
        // no room (case 4)
        LOG_DEBUG(SYN_MEM_ALLOC,
                  "{}: no room in pool memory. size= 0x{:x} pi={} ci={}",
                  HLLOG_FUNC,
                  size,
                  m_pi,
                  m_ci);
        return synFail;
    }
    if (m_pi >= m_ci)
    {
        if (m_pi + size <= m_poolSize)
        {  // case 1 - more room ahead
            hostVirtualAddress = m_poolBaseMapped + m_pi;
            *newHostAddress    = m_poolBaseAddr + m_pi;
            m_pi += size;
            m_max_pi = m_pi;
        }
        else if (size <= m_ci)
        {
            // case 2
            // overflow but we have room at the start
            // wrap! start from 0
            hostVirtualAddress = m_poolBaseMapped;
            *newHostAddress    = m_poolBaseAddr;
            m_max_pi           = m_pi;
            m_pi               = size;  // last allocation was from 0 to size
            if (m_pi == m_ci)
            {
                // pool is full, mark it until we release something
                m_poolIsFull = true;
            }
        }
        else
        {
            // no room (case 4)
            LOG_DEBUG(SYN_MEM_ALLOC,
                      "{}: no room in pool memory. size= 0x{:x} pi={} ci={}",
                      HLLOG_FUNC,
                      size,
                      m_pi,
                      m_ci);
            return synFail;
        }
    }
    else
    {  //  pi < ci
        if (m_pi + size <= m_ci)
        {
            // case 3 - from start (pi < ci,  pi+size <= ci)
            hostVirtualAddress = m_poolBaseMapped + m_pi;
            *newHostAddress    = m_poolBaseAddr + m_pi;
            m_pi += size;
            // in this case max_last_pi should stay since it denotes the last place ci could go
            if (m_pi == m_ci)
            {
                // pool is full, mark it until we release something
                m_poolIsFull = true;
            }
        }
        else
        {
            // no room - case 4
            LOG_DEBUG(SYN_MEM_ALLOC,
                      "{}: no room in pool memory. size= 0x{:x} pi={} ci={}",
                      HLLOG_FUNC,
                      size,
                      m_pi,
                      m_ci);
            return synFail;
        }
    }
    m_mappedMemorySize += size;
    return synSuccess;
}
bool PoolMemoryMapper::unmapBufferExHelper(uint64_t offset, uint64_t size)
{
    // possible cases
    //  1 - rollback - where a buffer is unmapped immediately after mapping, so pi only need to be set backwards to
    //  last_pi 2 - normal - where a buffer is unmapped from ci pos, just inc ci 3 - reset all - when after unmapping pi
    //  == ci,  set both to 0 4 - reset ci - when after unmapping  ci == max_pi,  set ci to 0 5 - out of order
    //      when unmapping addr is not from ci or last_pi
    //           just save it in a vector, and check if after the next unmap to see if it fits the previous cases now
    bool handled = true;
    if (offset == m_ci)
    {
        // case 2
        // release the memory from m_ci to m_ci + size
        m_ci = m_ci + size;
        if (m_ci == m_pi)
        {
            // reset all to starting state
            m_ci     = 0;
            m_pi     = 0;
            m_max_pi = (uint64_t)-1;
        }
        else if (m_ci == m_max_pi)
        {
            // pi didn't go higher than that, so ci needs to reset to 0
            m_ci     = 0;
            m_max_pi = m_pi;
        }
    }
    else if (offset + size == m_pi)
    {  // case 1
        m_pi = offset;
        if (m_ci == m_pi)
        {
            // reset all to starting phase
            m_ci     = 0;
            m_pi     = 0;
            m_max_pi = (uint64_t)-1;
        }
        else
        {
            //  on unmap(L)
            // |     CI    L   PI,MAX |  set max to L
            // | L   PI    CI   MAX   |  don't touch max
            if ((m_pi > m_ci) && (m_max_pi > m_ci)) m_max_pi = m_pi;
        }
    }
    else
    {
        handled = false;
    }
    return handled;
}

void PoolMemoryMapper::handleOOOunmapRequest(uint64_t bufferOffset, uint64_t size, bool handled)
{
    if (!handled)
    {
        // unmap request out of order
        // save it in a vector
        struct UnmapRequest rq = {bufferOffset, size};
        m_unmapRequests.push_back(rq);
    }
    else
    {
        if (m_unmapRequests.size() > 0)
        {
            // try to handle previous unhandled requests
            while (true)
            {
                handled = false;
                // cannot use for (auto rq: m_unmapRequests) since we use erase inside the loop
                for (auto rq = m_unmapRequests.begin(); rq != m_unmapRequests.end();)
                {
                    if (unmapBufferExHelper(
                            (*rq).offset,
                            (*rq).size))  // if the request was handled, delete it and go over the vector again
                    {
                        m_unmapRequests.erase(rq);
                        handled = true;
                        break;
                    }
                    ++rq;
                }
                if (!handled) break;
            }
        }
    }
}

synStatus PoolMemoryMapper::unmapBufferEx(uint64_t mappedBuffer, uint64_t size)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    // assuming mappedBuffer is the device mapped addr taken from this pool
    uint64_t bufferOffset = mappedBuffer - m_poolBaseMapped;
    bool     handled      = unmapBufferExHelper(bufferOffset, size);
    // handle unmap request out of order
    handleOOOunmapRequest(bufferOffset, size, handled);
    if (handled)
    {
        m_poolIsFull = false;  // we released something ..
    }
    m_mappedMemorySize -= size;
    return synSuccess;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// a variant of the pool that when the pool is full, calls the LKD to map and unmap
//
PoolMemoryMapperNoWait::PoolMemoryMapperNoWait(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize)
: PoolMemoryMapper(rDevMemAlloc, poolSize)
{
}

synStatus PoolMemoryMapperNoWait::mapBufferEx(uint64_t&          hostVirtualAddress,
                                              uint64_t*          newHostAddress,
                                              void*              buffer,
                                              uint64_t           size,
                                              const std::string& mappingDesc)
{
    // try pool mode
    synStatus status = synFail;
    if (newHostAddress)
    {
        status = PoolMemoryMapper::mapBufferEx(hostVirtualAddress, newHostAddress, buffer, size);
    }
    if (status != synSuccess)
    {
        // try non pool mode
        status = MemoryManager::mapBuffer(hostVirtualAddress, buffer, size, false, mappingDesc);
    }
    return status;
}

synStatus PoolMemoryMapperNoWait::unmapBufferEx(uint64_t mappedBuffer, const void* buffer, uint64_t size)
{
    if (mappedBuffer)  // pool mode
    {
        return PoolMemoryMapper::unmapBufferEx(mappedBuffer, size);
    }
    else  // non pool mode
    {
        return MemoryManager::unmapBuffer((void*)buffer, false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// a variant of the pool that when the pool is full, waits for room to become available
//
PoolMemoryMapperWait::PoolMemoryMapperWait(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize)
: PoolMemoryMapper(rDevMemAlloc, poolSize)
{
}

synStatus PoolMemoryMapperWait::mapBufferEx(uint64_t&          hostVirtualAddress,
                                            uint64_t*          newHostAddress,
                                            void*              buffer,
                                            uint64_t           size,
                                            const std::string& mappingDesc)
{
    // try pool mode
    synStatus status = synFail;
    if (newHostAddress)
    {
        while (status != synSuccess)
        {
            status = PoolMemoryMapper::mapBufferEx(hostVirtualAddress, newHostAddress, buffer, size);
            if (status != synSuccess)
            {
                std::unique_lock<std::mutex> mutex_cv(m_condMutex);
                m_cvFlag = false;
                m_numWaiting++;
                m_condVar.wait(mutex_cv, [&] { return m_cvFlag; });
                m_numWaiting--;
            }
        }
    }
    else
    {
        // non pool mode
        status = MemoryManager::mapBuffer(hostVirtualAddress, buffer, size, false, mappingDesc);
    }
    return status;
}

synStatus PoolMemoryMapperWait::unmapBufferEx(uint64_t mappedBuffer, const void* buffer, uint64_t size)
{
    if (mappedBuffer)  // pool mode
    {
        synStatus status = PoolMemoryMapper::unmapBufferEx(mappedBuffer, size);
        // mutex is outside the block to handle the case where mapBuffer just increased m_numWaiting but still didn't
        // enter wait so if the check of m_numWaiting was outside the mutex, it could miss that someone is about to wait
        std::unique_lock<std::mutex> mutex_cv(m_condMutex);
        if (m_numWaiting)
        {
            m_cvFlag = true;
            m_condVar.notify_all();
        }
        return status;
    }
    else  // non pool mode
    {
        return MemoryManager::unmapBuffer((void*)buffer, false);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<PoolMemoryMapper>
createPoolMemoryManager(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize, bool waitOnPoolFull)
{
    // uint64_t poolSize = GCFG_POOL_MAPPING_SIZE_IN_STREAM_COPY.value();
    if (poolSize > 0)
    {
        // pool is enabled. size is poolSize * MB
        if (!waitOnPoolFull)
        {
            return std::unique_ptr<PoolMemoryMapper>(new PoolMemoryMapperNoWait(rDevMemAlloc, poolSize));
        }
        else
        {
            return std::unique_ptr<PoolMemoryMapper>(new PoolMemoryMapperWait(rDevMemAlloc, poolSize));
        }
    }
    LOG_CRITICAL(SYN_MEM_ALLOC, "{}: should not be called when pool size is 0", HLLOG_FUNC);
    return nullptr;  // error
}

std::unique_ptr<MemoryManager> createMemoryManager(DevMemoryAllocInterface& rDevMemAlloc)
{
    return std::unique_ptr<MemoryManager>(new MemoryManager(rDevMemAlloc));
}
