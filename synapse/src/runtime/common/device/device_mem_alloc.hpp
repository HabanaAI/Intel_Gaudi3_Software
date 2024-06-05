#pragma once

#include "synapse_common_types.h"
#include "host_to_virtual_address_mapper.hpp"
#include <atomic>
#include "runtime/common/osal/buffer_allocator.hpp"

class ScalDev;

class DevMemoryAllocInterface
{
public:
    virtual ~DevMemoryAllocInterface() = default;

    virtual synStatus allocate() = 0;

    virtual synStatus release() = 0;

    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) = 0;

    virtual synStatus deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest) = 0;

    virtual eMappingStatus getDeviceVirtualAddress(bool         isUserRequest,
                                                   void*        hostAddress,
                                                   uint64_t     bufferSize,
                                                   uint64_t*    pDeviceVA,
                                                   bool*        pIsExactKeyFound = nullptr) = 0;

    virtual synStatus mapBufferToDevice(uint64_t           size,
                                        void*              buffer,
                                        bool               isUserRequest,
                                        uint64_t           reqVAAddress,
                                        const std::string& mappingDesc) = 0;

    virtual synStatus unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize) = 0;

    virtual synStatus getDramMemInfo(uint64_t& free, uint64_t& total) const = 0;

    virtual void getValidAddressesRange(uint64_t& lowestValidAddress, uint64_t& highestValidAddress) const = 0;

    virtual void dfaLogMappedMem() const = 0;

    virtual synStatus destroyHostAllocations(bool isUserAllocations) = 0;
};

class DevMemoryAlloc : public DevMemoryAllocInterface
{
public:
    DevMemoryAlloc(synDeviceType devType, uint64_t dramSize, uint64_t dramBaseAddress);

    virtual ~DevMemoryAlloc() = default;

    virtual synStatus deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest) override;

    eMappingStatus getDeviceVirtualAddress(bool         isUserRequest,
                                           void*        hostAddress,
                                           uint64_t     bufferSize,
                                           uint64_t*    pDeviceVA,
                                           bool*        pIsExactKeyFound = nullptr) override;

    virtual synStatus mapBufferToDevice(uint64_t           size,
                                        void*              buffer,
                                        bool               isUserRequest,
                                        uint64_t           reqVAAddress,
                                        const std::string& mappingDesc) override;

    virtual synStatus unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize) override;

    virtual synStatus getDramMemInfo(uint64_t& free, uint64_t& total) const override;

    virtual void getValidAddressesRange(uint64_t& lowestValidAddress, uint64_t& highestValidAddress) const override;

    virtual void dfaLogMappedMem() const override;

    synStatus destroyHostAllocations(bool isUserAllocations) override;

protected:
    HostAddrToVirtualAddrMapper& getAddressMapper(bool isUserRequest)
    {
        return isUserRequest ? m_userAddressMapper : m_topologiesAddressMapper;
    }

    synStatus getNonDeviceMMUMemInfo(uint64_t& free, uint64_t& total) const;

    synStatus getDeviceMMUMemInfo(uint64_t& free, uint64_t& total) const;

    bool setHostToDeviceAddressMapping(void*              hostAddress,
                                       uint64_t           bufferSize,
                                       uint64_t           vaAddress,
                                       bool               isUserRequest,
                                       const std::string& mappingDesc);

    const synDeviceType    m_devType;
    const uint64_t         m_dramSize;
    const uint64_t         m_dramBaseAddress;
    std::atomic<uint64_t>  m_dramUsed;

    std::shared_ptr<DeviceHeapAllocatorManager> m_spDeviceMemoryAllocatorManager;

private:
    HostAddrToVirtualAddrMapper m_userAddressMapper;
    // Will be used when needs to map a buffer that the Synapse allocated for the topology
    HostAddrToVirtualAddrMapper m_topologiesAddressMapper;
};

class  DevMemoryAllocCommon : public DevMemoryAlloc
{
public:
    DevMemoryAllocCommon(synDeviceType devType, uint64_t dramSize, uint64_t dramBaseAddress);

    virtual ~DevMemoryAllocCommon();

    virtual synStatus allocate() override;

    virtual synStatus release() override;

    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) override;

private:
    BufferAllocator* allocateManagedBufferAllocator();

    DeviceAllocator m_mngMemoryAllocator;
    bool            m_mngMemoryAllocated;
};

class DevMemoryAllocScal : public DevMemoryAlloc
{
public:
    DevMemoryAllocScal(ScalDev*      scalDev,
                       synDeviceType devType,
                       uint64_t      dramSize,
                       uint64_t      dramBaseAddress);

    virtual ~DevMemoryAllocScal() = default;

    virtual synStatus allocate() override { return synSuccess; }

    virtual synStatus release() override { return synSuccess; }

    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) override;

    static DevMemoryAlloc* debugGetLastConstructedAllocator() { return s_debugLastConstructedAllocator; }

private:
    static DevMemoryAlloc* s_debugLastConstructedAllocator;
    ScalDev*               m_scalDev;
};
