#pragma once

#include "runtime/common/device/device_mem_alloc.hpp"

class DevMemoryAllocMock : public DevMemoryAllocInterface
{
public:
    virtual ~DevMemoryAllocMock() = default;

    virtual synStatus allocate() override { return synSuccess; }

    virtual synStatus release() override { return synSuccess; }

    virtual synStatus allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA = nullptr) override;

    virtual synStatus deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest) override;

    virtual eMappingStatus getDeviceVirtualAddress(bool         isUserRequest,
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

    virtual void dfaLogMappedMem() const override {};

    virtual synStatus destroyHostAllocations(bool isUserAllocations) override;
};
