#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "test_device.hpp"
#include "syn_singleton.hpp"

class SynGaudiMemcopySmallBufferTests : public SynBaseTest
{
public:
    SynGaudiMemcopySmallBufferTests() : SynBaseTest() { setSupportedDevices({synDeviceGaudi}); }
};

REGISTER_SUITE(SynGaudiMemcopySmallBufferTests, ALL_TEST_PACKAGES);

TEST_F_SYN(SynGaudiMemcopySmallBufferTests, check_recipe_cache_overrun_validation)
{
    TestDevice device(m_deviceType);

    const uint64_t maxNumOfElements = exp2(4);
    const uint64_t bufferSize       = maxNumOfElements * sizeof(float);

    float *         inputBuffer[1], *outputBuffer[1];
    synEventHandle  streamSyncEventHandle, completionEventHandle;
    synStreamHandle streamHandleDownload, streamHandleUpload, streamHandleDevToDev;

    auto status = synHostMalloc(device.getDeviceId(), bufferSize, 0, (void**)&inputBuffer[0]);
    ASSERT_EQ(status, synSuccess) << "Failed malloc input buffer";

    status = synHostMalloc(device.getDeviceId(), bufferSize, 0, (void**)&outputBuffer[0]);
    ASSERT_EQ(status, synSuccess) << "Failed malloc output buffer";

    uint64_t deviceBuffer;
    status = synDeviceMalloc(device.getDeviceId(), bufferSize, 0, 0, &deviceBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate HBM-memory for device-buffer";

    status = synEventCreate(&streamSyncEventHandle, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create event (streamSyncEventHandle)";

    status = synEventCreate(&completionEventHandle, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create event (completionEventHandle)";

    status = synStreamCreateGeneric(&streamHandleDownload, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create Download stream";

    status = synStreamCreateGeneric(&streamHandleUpload, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create Upload stream";

    status = synStreamCreateGeneric(&streamHandleDevToDev, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create Device-To-Device stream";

    uint64_t recipeCacheBaseAddress = 0;
    uint64_t recipeCacheEndAddress  = 0;
    status = _SYN_SINGLETON_INTERNAL->getCacheDeviceAddressRange(recipeCacheBaseAddress, recipeCacheEndAddress);
    ASSERT_EQ(status, synSuccess);

    const uint64_t size = 1;

    const std::string dmaDirectionDescription[DIRECTION_ENUM_MAX] = {"HOST_TO_DRAM", "DRAM_TO_HOST", "DRAM_TO_DRAM"};

    enum eOperationValidationType
    {
        OPT_RECIPE_CACHE_BASE_OVERRUN,
        OPT_RECIPE_CACHE_END_OVERRUN,
        OPT_VALID_COPY,
        OPT_ENUM_MAX
    };
    eOperationValidationType operationValidationType = OPT_RECIPE_CACHE_BASE_OVERRUN;

    const std::string operationValidationDescription[OPT_ENUM_MAX] = {
        "Succesfull of an invalid copy operation (Recipe-Cache base overrun)",
        "Succesfull of an invalid copy operation (Recipe-Cache end overrun)",
        "Failure on a valid operation"};

    uint64_t srcAddress = 0;
    uint64_t dstAddress = 0;

    const synStreamHandle streamHandles[DIRECTION_ENUM_MAX] = {streamHandleDownload,
                                                               streamHandleUpload,
                                                               streamHandleDevToDev};

    synDmaDir       dmaDirection = HOST_TO_DRAM;
    synStreamHandle streamHandle = streamHandles[dmaDirection];

    // Testing source
    for (uint32_t i = HOST_TO_DRAM; i < DIRECTION_ENUM_MAX; i++)
    {
        dmaDirection = (synDmaDir)i;
        streamHandle = streamHandles[dmaDirection];

        if ((dmaDirection != DRAM_TO_HOST) && (dmaDirection != DRAM_TO_DRAM))
        {
            continue;
        }

        if (dmaDirection == DRAM_TO_HOST)
        {
            dstAddress = (uint64_t)inputBuffer[0];
        }
        else
        {
            dstAddress = deviceBuffer;
        }

        srcAddress              = recipeCacheBaseAddress;
        operationValidationType = OPT_RECIPE_CACHE_BASE_OVERRUN;
        status                  = synMemCopyAsync(streamHandle, srcAddress, size, dstAddress, dmaDirection);
        ASSERT_EQ(status, synInvalidArgument) << operationValidationDescription[operationValidationType] << " "
                                              << dmaDirectionDescription[dmaDirection] << " (source)";

        srcAddress              = recipeCacheEndAddress;
        operationValidationType = OPT_RECIPE_CACHE_END_OVERRUN;
        status                  = synMemCopyAsync(streamHandle, srcAddress, size, dstAddress, dmaDirection);
        ASSERT_EQ(status, synInvalidArgument) << operationValidationDescription[operationValidationType] << " "
                                              << dmaDirectionDescription[dmaDirection] << " (source)";

        srcAddress              = deviceBuffer + 1;
        operationValidationType = OPT_VALID_COPY;
        status                  = synMemCopyAsync(streamHandle, srcAddress, size, dstAddress, dmaDirection);
        ASSERT_EQ(status, synSuccess) << operationValidationDescription[operationValidationType] << " "
                                      << dmaDirectionDescription[dmaDirection] << " (source)";
    }

    // Testing destination
    for (uint32_t i = HOST_TO_DRAM; i < DIRECTION_ENUM_MAX; i++)
    {
        dmaDirection = (synDmaDir)i;
        streamHandle = streamHandles[dmaDirection];

        if ((dmaDirection != HOST_TO_DRAM) && (dmaDirection != DRAM_TO_DRAM))
        {
            continue;
        }

        if (dmaDirection == HOST_TO_DRAM)
        {
            srcAddress = (uint64_t)inputBuffer[0];
        }
        else
        {
            srcAddress = deviceBuffer;
        }

        dstAddress              = recipeCacheBaseAddress;
        operationValidationType = OPT_RECIPE_CACHE_BASE_OVERRUN;
        status                  = synMemCopyAsync(streamHandle, srcAddress, size, dstAddress, dmaDirection);
        ASSERT_EQ(status, synInvalidArgument) << operationValidationDescription[operationValidationType] << " "
                                              << dmaDirectionDescription[dmaDirection] << " (destination)";

        dstAddress              = recipeCacheEndAddress;
        operationValidationType = OPT_RECIPE_CACHE_END_OVERRUN;
        status                  = synMemCopyAsync(streamHandle, srcAddress, size, dstAddress, dmaDirection);
        ASSERT_EQ(status, synInvalidArgument) << operationValidationDescription[operationValidationType] << " "
                                              << dmaDirectionDescription[dmaDirection] << " (destination)";

        dstAddress              = deviceBuffer + 1;
        operationValidationType = OPT_VALID_COPY;
        status                  = synMemCopyAsync(streamHandle, srcAddress, size, dstAddress, dmaDirection);
        ASSERT_EQ(status, synSuccess) << operationValidationDescription[operationValidationType] << " "
                                      << dmaDirectionDescription[dmaDirection] << " (destination)";
    }

    status = synStreamSynchronize(streamHandleDownload);
    ASSERT_EQ(status, synSuccess) << "Failed to sync stream-Download";

    status = synStreamSynchronize(streamHandleUpload);
    ASSERT_EQ(status, synSuccess) << "Failed to sync stream-Upload";

    status = synStreamSynchronize(streamHandleDevToDev);
    ASSERT_EQ(status, synSuccess) << "Failed to sync stream-DevToDev";

    status = synHostFree(device.getDeviceId(), inputBuffer[0], 0);
    ASSERT_EQ(status, synSuccess) << "Failed to free host bufer (Input)";

    status = synHostFree(device.getDeviceId(), outputBuffer[0], 0);
    ASSERT_EQ(status, synSuccess) << "Failed to free host bufer (Output)";

    status = synDeviceFree(device.getDeviceId(), deviceBuffer, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory (deviceBuffer)";

    status = synEventDestroy(streamSyncEventHandle);
    ASSERT_EQ(status, synSuccess) << "Failed destroy event (streamSyncEventHandle)";

    status = synEventDestroy(completionEventHandle);
    ASSERT_EQ(status, synSuccess) << "Failed destroy event (completionEventHandle)";

    status = synStreamDestroy(streamHandleDownload);
    ASSERT_EQ(status, synSuccess) << "Failed destroy stream (Download)";

    status = synStreamDestroy(streamHandleUpload);
    ASSERT_EQ(status, synSuccess) << "Failed destroy stream (Upload)";
}