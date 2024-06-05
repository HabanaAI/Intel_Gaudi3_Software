#pragma once

#include "syn_base_test.hpp"
class TestStream;
class TestDeviceBufferAlloc;
class TestHostBufferMalloc;



class SynMemcopyTests : public SynBaseTest
{
public:
    SynMemcopyTests() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }

    bool MemcopyHostBufferThroughDevice(uint64_t numOfElements, uint64_t copySizeInBytes, uint32_t threadId,
                                        TestHostBufferMalloc*    pInputBuffer,
                                        TestHostBufferMalloc*    pOutputBuffer,
                                        TestDeviceBufferAlloc*   pDevbuffer,
                                        TestStream*              pStream);

    static const uint32_t numberOfThreads = 1;

private:
    std::mutex singleAsicUserMutex;
};