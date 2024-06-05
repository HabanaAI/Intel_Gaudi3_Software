#pragma once

#include "host_buffers_mapper.hpp"
#include <cstdint>
#include <vector>
#include <mutex>               // std::mutex, std::unique_lock
#include <condition_variable>  // std::condition_variable, std::cv_status

class DevMemoryAllocInterface;

class MemoryManager : public HostBuffersMapper
{
public:
    MemoryManager(DevMemoryAllocInterface& rDevMemAlloc);

    virtual ~MemoryManager();

    virtual synStatus mapBuffer(uint64_t&          hostVirtualAddress,
                                void*              buffer,
                                uint64_t           size,
                                bool               isUserRequest,
                                const std::string& mappingDesc,
                                uint64_t           requestedVirtualAddress = INVALID_ADDRESS) const override;

    virtual synStatus unmapBuffer(void* buffer, bool isUserRequest) const override;

    virtual uint64_t getMappedSize() const override { return m_mappedMemorySize; }

protected:
    DevMemoryAllocInterface& m_rDevMemAlloc;
    mutable uint64_t         m_mappedMemorySize;
};

class PoolMemoryMapper : protected MemoryManager
{
public:
    PoolMemoryMapper(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize);
    virtual ~PoolMemoryMapper();
    virtual synStatus mapBufferEx(uint64_t&          hostVirtualAddress,
                                  uint64_t*          newHostAddress,
                                  void*              buffer,
                                  uint64_t           size,
                                  const std::string& mappingDesc) = 0;

    virtual synStatus unmapBufferEx(uint64_t mappedBuffer, const void* buffer, uint64_t size) = 0;

    virtual uint64_t getMappedSize() const override { return m_mappedMemorySize; }

protected:
    struct UnmapRequest
    {
        uint64_t offset;
        uint64_t size;
    };
    synStatus mapBufferEx(uint64_t& hostVirtualAddress, uint64_t* newHostAddress, void* buffer, uint64_t size);
    synStatus unmapBufferEx(uint64_t mappedBuffer, uint64_t size);
    bool      unmapBufferExHelper(uint64_t offset, uint64_t size);
    void      handleOOOunmapRequest(uint64_t bufferOffset, uint64_t size, bool handled);
    //
    uint64_t                         m_pi             = 0;
    uint64_t                         m_max_pi         = (uint64_t)-1;
    uint64_t                         m_ci             = 0;
    uint64_t                         m_poolSize       = 0;
    uint64_t                         m_poolBaseAddr   = 0;
    uint64_t                         m_poolBaseMapped = 0;
    std::vector<struct UnmapRequest> m_unmapRequests;
    std::mutex                       m_mutex;
    bool                             m_poolIsFull = false;
};

class PoolMemoryMapperNoWait : public PoolMemoryMapper
{
public:
    PoolMemoryMapperNoWait(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize);
    virtual ~PoolMemoryMapperNoWait() {}

    synStatus mapBufferEx(uint64_t&          hostVirtualAddress,
                          uint64_t*          newHostAddress,
                          void*              buffer,
                          uint64_t           size,
                          const std::string& mappingDesc);

    synStatus unmapBufferEx(uint64_t mappedBuffer, const void* buffer, uint64_t size);

private:
};

class PoolMemoryMapperWait : public PoolMemoryMapper
{
public:
    PoolMemoryMapperWait(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize);
    ~PoolMemoryMapperWait() {}

    synStatus mapBufferEx(uint64_t&          hostVirtualAddress,
                          uint64_t*          newHostAddress,
                          void*              buffer,
                          uint64_t           size,
                          const std::string& mappingDesc);

    synStatus unmapBufferEx(uint64_t mappedBuffer, const void* buffer, uint64_t size);

private:
    bool                            m_cvFlag = false;
    mutable std::mutex              m_condMutex;
    mutable std::condition_variable m_condVar;
    uint32_t                        m_numWaiting = 0;
};

std::unique_ptr<PoolMemoryMapper>
createPoolMemoryManager(DevMemoryAllocInterface& rDevMemAlloc, uint64_t poolSize, bool waitOnPoolFull);
std::unique_ptr<MemoryManager> createMemoryManager(DevMemoryAllocInterface& rDevMemAlloc);
