#pragma once

#include <memory>
#include <vector>
#include <atomic>

#include "synapse_types.h"

#include "infra/generic_exceptions.hpp"
#include "memory_management/heap_allocator.h"

/*
 * *************************************************************************************************
 *  @brief BufferAllocator class implements the following logic:
 *  1.) Holds VAs of an allocation
 *  2.)
 * *************************************************************************************************
 */
class BufferAllocator
{
public:
    BufferAllocator() : m_flags(0), m_size(0), m_deviceVa(0), m_hostVa((void*)0), m_handle(0), m_shouldFreeMemory(false)
    {
    }
    virtual ~BufferAllocator() {}

    virtual synStatus AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest) = 0;

    virtual synStatus FreeMemory() = 0;

    virtual synStatus MapHostMemory(void* hostAddress, uint64_t size, uint64_t reqVAAddress) { return synUnsupported; }

    virtual synStatus UnmapHostMemory() { return synUnsupported; }

    // Setter/Getters
    virtual uint32_t getFlags() const { return m_flags; }
    virtual void     setFlags(uint32_t flags) { m_flags = flags; }

    virtual uint64_t getSize() const { return m_size; }
    virtual void     setSize(uint64_t size) { m_size = size; }

    virtual uint64_t getDeviceVa() const { return m_deviceVa; }
    virtual void     setDeviceVa(uint64_t va) { m_deviceVa = va; }

    virtual void* getHostVa() const { return m_hostVa; }
    virtual void  setHostVa(void* va) { m_hostVa = va; }

    virtual uint64_t getHandle() const { return m_handle; }
    virtual void     setHandle(uint64_t handle) { m_handle = handle; }

    virtual bool shouldFreeMemory() const { return m_shouldFreeMemory; }
    virtual void setShouldFreeMemory(bool shouldFreeMemory) { m_shouldFreeMemory = shouldFreeMemory; }

private:
    BufferAllocator(const BufferAllocator& orig);
    BufferAllocator& operator=(const BufferAllocator& other);

private:
    uint32_t m_flags;
    uint64_t m_size;
    uint64_t m_deviceVa;
    void*    m_hostVa;
    uint64_t m_handle;

    bool m_shouldFreeMemory;
};

class IoctlDeviceAllocator : public BufferAllocator
{
public:
    virtual ~IoctlDeviceAllocator() = default;

protected:
    static synStatus MemoryIoctl(uint64_t  deviceVirtAddr,
                                 uint64_t  hostVirtAddr,
                                 uint32_t  op,
                                 uint64_t  memSize,
                                 uint32_t  flags,
                                 uint64_t* pRetAddress = nullptr,
                                 // Handle is only valid for DRAM allocation
                                 uint64_t* pHandle   = nullptr,
                                 bool      useHandle = false);
};

class DeviceAllocator : public IoctlDeviceAllocator
{
public:
    DeviceAllocator() {};
    virtual ~DeviceAllocator() {};

    virtual synStatus AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest) override;
    virtual synStatus FreeMemory() override;

private:
    static uint64_t m_numOfHugePagesAllocated;
};

class HostAllocator : public IoctlDeviceAllocator
{
public:
    HostAllocator() : m_isHugePageAllocated(false) {};
    virtual ~HostAllocator() {};

    virtual synStatus AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest) override;

    virtual synStatus FreeMemory() override;

    virtual synStatus MapHostMemory(void* hostAddress, uint64_t size, uint64_t reqVAAddress) override;

    virtual synStatus UnmapHostMemory() override;

protected:
    synStatus _allocateMemory(const uint64_t& size, void** pHostAddress, bool isUserRequest);

    synStatus _mapMemory(const uint64_t& size, const uint64_t& hostAddress, uint64_t& deviceAddress);

    synStatus _unmapMemory(const uint64_t& size, const uint64_t& deviceAddress, uint64_t retAddr);

    bool m_isHugePageAllocated;

    static std::atomic<uint64_t> m_numOfHugePagesAllocated;
    static std::atomic<uint64_t> m_numOfRegularPagesAllocated;
};

class DeviceHeapAllocatorManager : public HeapAllocatorBestFit
{
public:
    DeviceHeapAllocatorManager(int fd, bool isUpdateDriver)
    : HeapAllocatorBestFit("DRAM"), m_fd(fd), m_isUpdateDriver(isUpdateDriver) {};
    virtual ~DeviceHeapAllocatorManager() {};

    void                               Free(deviceAddrOffset ptr) override;
    virtual Settable<deviceAddrOffset> Allocate(uint64_t size,
                                                uint64_t alignment,
                                                uint64_t offset           = 0,
                                                bool     allowFailure     = false,
                                                uint64_t requestedAddress = 0) override;
    bool                               allocateReqRange(const Range& reqRange, unsigned pad);

private:
    void updateDriver(uint64_t allocSize);

    const int  m_fd;
    const bool m_isUpdateDriver;
};

class ManagedBufferAllocator : public IoctlDeviceAllocator
{
public:
    ManagedBufferAllocator(std::shared_ptr<DeviceHeapAllocatorManager> heapAllocator);
    virtual ~ManagedBufferAllocator() {};

    virtual synStatus AllocateMemory(uint64_t reqVAAddress, uint64_t size, bool isUserRequest) override;

    virtual synStatus FreeMemory() override;

    static const uint64_t m_defaultAlignment;

protected:
    virtual synStatus _allocateMemory(uint64_t reqVAAddress, uint64_t size, unsigned alignment, bool shouldFreeMemory);

    std::shared_ptr<DeviceHeapAllocatorManager> m_spAllocatorManager;
};
