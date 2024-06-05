#include "memory_test.hpp"
#include "synapse_api.h"
#include <mutex>
#include <thread>
#include "habana_global_conf_runtime.h"
#include "runtime/scal/common/recipe_launcher/mapped_mem_mgr.hpp"
#include "test_device.hpp"

REGISTER_SUITE(SynMemcopyTests, ALL_TEST_PACKAGES);

bool SynMemcopyTests::MemcopyHostBufferThroughDevice(uint64_t                 numOfElements,
                                                     uint64_t                 copySizeInBytes,
                                                     uint32_t                 threadId,
                                                     TestHostBufferMalloc*    pInputBuffer,
                                                     TestHostBufferMalloc*    pOutputBuffer,
                                                     TestDeviceBufferAlloc*   pDevBuffer,
                                                     TestStream*              pStream)
{
    {
        pInputBuffer->fill(threadId);
    }

    {
        std::unique_lock<std::mutex> lock(singleAsicUserMutex);

        pStream->memcopyAsync(*pInputBuffer, *pDevBuffer);

        pStream->memcopyAsync(*pDevBuffer, *pOutputBuffer);

        pStream->synchronize();
    }

    {
        float* currInputBuffer  = (float*)pInputBuffer->getBuffer();
        float* currOutputBuffer = (float*)pOutputBuffer->getBuffer();

        bool recheck = false;
        for (uint64_t i = 0; i < numOfElements; i++)
        {
            if (currInputBuffer[i] != currOutputBuffer[i])
            {
                LOG_ERR(SYN_TEST, "0x{:x}: Input {} Output {}", i, currInputBuffer[i], currOutputBuffer[i]);
                recheck = true;
                break;
            }
        }

        if (recheck)
        {
            for (uint64_t i = 0; i < numOfElements; i++)
            {
                ASSERT_EQ(currInputBuffer[i], currOutputBuffer[i]) << "Mismatch on output";
            }
        }
    }
    return true;
}

void memcopyValidationTest(SynMemcopyTests*         pTest,
                           uint32_t                 iterIndex,
                           uint64_t                 numOfElements,
                           uint64_t                 copySizeInBytes,
                           uint32_t                 threadId,
                           TestHostBufferMalloc*    pInputBuffer,
                           TestHostBufferMalloc*    pOutputBuffer,
                           TestDeviceBufferAlloc*   pDevBuffer,
                           TestStream*              pStream)
{
    bool status = pTest->MemcopyHostBufferThroughDevice(numOfElements, copySizeInBytes, threadId, pInputBuffer, pOutputBuffer, pDevBuffer, pStream);

    ASSERT_EQ(true, status) << "Test failed iterIndex " << iterIndex << " threadId " << threadId << " numOfElements "
                            << numOfElements << " copy-size " << copySizeInBytes;
}

// REMARK:
//    For allowing the other test(s) to run on the CI, the amount of devices uses had been set to one
//    When enabling this test, take care of that variable, used during setup;
//         * numberOfThreads  = 1
TEST_F_SYN(SynMemcopyTests, memcopy_size_variations_mt, {synTestPackage::ASIC})
{
    const uint64_t maxExponentParam = 29;
    const uint64_t maxNumOfElements = exp2(maxExponentParam);
    const uint64_t bufferSize       = maxNumOfElements * sizeof(float);

    std::vector<TestHostBufferMalloc> inputBuffer, outputBuffer;

    TestDevice device(m_deviceType);
    TestStream stream = device.createStream();

    for (uint32_t i = 0; i < numberOfThreads; i++)
    {
        inputBuffer.emplace_back(device.allocateHostBuffer(bufferSize, 0));
        outputBuffer.emplace_back(device.allocateHostBuffer(bufferSize, 0));
    }

    TestDeviceBufferAlloc  devBuffer = device.allocateDeviceBuffer(bufferSize, 0);

    const uint32_t numberOfTestIterations = 3;
    const uint32_t firstCopyIndex         = 1;
    const uint32_t lastCopyIndex          = maxExponentParam;

    std::vector<std::thread> threads;

    for (uint32_t iterIndex = 0; iterIndex < numberOfTestIterations; iterIndex++)
    {
        for (uint32_t copyIndex = firstCopyIndex; copyIndex <= lastCopyIndex; copyIndex++)
        {
            uint64_t numOfElements   = exp2(copyIndex);
            uint64_t copySizeInBytes = numOfElements * sizeof(float);
            LOG_ERR(SYN_RT_TEST,
                    "Copying 0x{:x} floats (0x{:x} Bytes with {} copyIndex)",
                    numOfElements,
                    copySizeInBytes,
                    copyIndex);

            threads.clear();

            for (int threadId = 0; threadId < numberOfThreads; threadId++)
            {
                threads.push_back(
                    std::thread(memcopyValidationTest, this, iterIndex, numOfElements, copySizeInBytes,
                                 threadId, &inputBuffer[threadId], &outputBuffer[threadId], &devBuffer, &stream));
            }

            for (auto& thread : threads)
            {
                thread.join();
            }
        }
    }

    stream.synchronize();
}
