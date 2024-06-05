#include "dev_memory_alloc_mock.hpp"
#include <gtest/gtest.h>

uint64_t DummyAddr;

synStatus DevMemoryAllocMock::allocateMemory(uint64_t           size,
                                             uint32_t           flags,
                                             void**             buffer,
                                             bool               isUserRequest,
                                             uint64_t           reqVAAddress,
                                             const std::string& mappingDesc,
                                             uint64_t*          deviceVA)
{
    *buffer = &DummyAddr;
    return synSuccess;
}

synStatus DevMemoryAllocMock::deallocateMemory(void* pBuffer, uint32_t flags, bool isUserRequest)
{
    EXPECT_EQ(pBuffer, &DummyAddr);
    return synSuccess;
}

uint64_t MapppedDeviceAddr;

eMappingStatus DevMemoryAllocMock::getDeviceVirtualAddress(bool         isUserRequest,
                                                           void*        hostAddress,
                                                           uint64_t     bufferSize,
                                                           uint64_t*    pDeviceVA,
                                                           bool*        pIsExactKeyFound)
{
    *pDeviceVA = MapppedDeviceAddr;
    return HATVA_MAPPING_STATUS_FOUND;
}

synStatus DevMemoryAllocMock::mapBufferToDevice(uint64_t           size,
                                                void*              buffer,
                                                bool               isUserRequest,
                                                uint64_t           reqVAAddress,
                                                const std::string& mappingDesc)
{
    return synSuccess;
}

synStatus DevMemoryAllocMock::unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize)
{
    return synSuccess;
}

synStatus DevMemoryAllocMock::getDramMemInfo(uint64_t& free, uint64_t& total) const
{
    return synSuccess;
}

void DevMemoryAllocMock::getValidAddressesRange(uint64_t& lowestValidAddress, uint64_t& highestValidAddress) const
{
    lowestValidAddress  = 0;
    highestValidAddress = 0x1000;
}

synStatus DevMemoryAllocMock::destroyHostAllocations(bool isUserAllocations)
{
    return synSuccess;
}
