#include "syn_base_test.hpp"
#include "test_recipe_nop_x_nodes.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "synapse_api.h"
#include "syn_singleton.hpp"
#include "runtime/common/device/device_common.hpp"
#include "runtime/common/streams/stream.hpp"

// TODO - Support fixture which gets supported devices (not part of this commit)

class MultiStreamsTest : public SynBaseTest
{
public:
    MultiStreamsTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }

    ~MultiStreamsTest() override = default;

    // Allocate events, streams and memory for copy operations
    // In case bufferSize is zero, do not allocate copy-buffers
    void preExecution(const TestDevice& rDevice, uint64_t bufferSize, unsigned numOfBuffers);
    void postExecution(TestDevice& rDevice);

    Stream* getStream(synStreamHandle stream, DeviceCommon* device)
    {
        auto streamSptr = device->loadAndValidateStream(stream, __FUNCTION__);
        return streamSptr.get();
    }

    std::vector<void*>    m_hostBuffers   = {};
    std::vector<uint64_t> m_deviceBuffers = {};
};

REGISTER_SUITE(MultiStreamsTest, ALL_TEST_PACKAGES);

void MultiStreamsTest::preExecution(const TestDevice& rDevice, uint64_t bufferSize, unsigned numOfBuffers)
{
    if (bufferSize != 0)
    {
        m_hostBuffers.resize(numOfBuffers);
        m_deviceBuffers.resize(numOfBuffers);

        for (unsigned i = 0; i < numOfBuffers; i++)
        {
            // Host buffer allocation
            synStatus status = synHostMalloc(rDevice.getDeviceId(), bufferSize, 0, &m_hostBuffers[i]);
            ASSERT_EQ(status, synSuccess) << "Failed alocate host buffer";
            std::memset(m_hostBuffers[i], 0xCA, bufferSize);

            // Device buffer allocation
            status = synDeviceMalloc(rDevice.getDeviceId(), bufferSize, 0, 0, &m_deviceBuffers[i]);
            ASSERT_EQ(status, synSuccess) << "Failed to alocate HBM memory";
        }
    }
}

void MultiStreamsTest::postExecution(TestDevice& rDevice)
{
    synStatus status(synSuccess);

    for (auto singleDeviceBuffer : m_deviceBuffers)
    {
        status = synDeviceFree(rDevice.getDeviceId(), singleDeviceBuffer, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to free HBM memory";
    }
    m_deviceBuffers.clear();

    for (auto singleHostBuffer : m_hostBuffers)
    {
        status = synHostFree(rDevice.getDeviceId(), singleHostBuffer, 0);
        ASSERT_EQ(status, synSuccess) << "Failed free host buffer";
    }
    m_hostBuffers.clear();
}

TEST_F_SYN(MultiStreamsTest, stream_create_without_explicit_stream_destroy_test)
{
    const unsigned  buffersAmount = 1;
    const uint64_t  copySize      = 10;
    synStreamHandle streamHandle;

    {
        TestDevice device(m_deviceType);

        preExecution(device, copySize, buffersAmount);
        ASSERT_EQ(m_hostBuffers.size(), buffersAmount) << "Host buffer not created";
        ASSERT_EQ(m_deviceBuffers.size(), buffersAmount) << "Device buffer not created";

        auto status = synStreamCreateGeneric(&streamHandle, device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create stream";

        status = synMemCopyAsync(streamHandle, (uint64_t)m_hostBuffers[0], copySize, m_deviceBuffers[0], HOST_TO_DRAM);
        ASSERT_EQ(status, synSuccess) << "Failed to copy while owning the device";

        // Stream will be implicitly destroyed by releasing of the device
    }

    synStatus status =
        synMemCopyAsync(streamHandle, (uint64_t)m_hostBuffers[0], copySize, m_deviceBuffers[0], HOST_TO_DRAM);
    ASSERT_EQ(status, synFail) << "Unexpectedly succeeded to copy after releasing the device";
}

TEST_F_SYN(MultiStreamsTest, affinity_api_test)
{
    TestDevice device(m_deviceType);

    synStreamHandle streamHandle;
    auto            status = synStreamCreateGeneric(&streamHandle, device.getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create multi stream";

    const uint64_t affinityMaskDefault = 0x1;

    uint64_t affinityMaskQueryDefault(0);
    status = synStreamGetAffinity(device.getDeviceId(), streamHandle, &affinityMaskQueryDefault);
    ASSERT_EQ(status, synSuccess) << "Failed to get stream affinity (1)";
    ASSERT_EQ(affinityMaskDefault, affinityMaskQueryDefault) << "Unexpected affinity retrieved (1)";

    uint64_t deviceAffinityMask(0);
    status = synDeviceGetAffinityMaskRange(device.getDeviceId(), &deviceAffinityMask);
    ASSERT_EQ(status, synSuccess) << "Failed to get affinity mask-range";

    if ((deviceAffinityMask & 0x2) == 0x2)
    {
        uint64_t affinityMaskSetFirst = 0x2;
        status                        = synStreamSetAffinity(device.getDeviceId(), streamHandle, affinityMaskSetFirst);
        ASSERT_EQ(status, synSuccess) << "Failed to set stream affinity";

        uint64_t affinityMaskQueryFirst(0);
        status = synStreamGetAffinity(device.getDeviceId(), streamHandle, &affinityMaskQueryFirst);
        ASSERT_EQ(status, synSuccess) << "Failed to get stream affinity (2)";
        ASSERT_EQ(affinityMaskSetFirst, affinityMaskQueryFirst) << "Unexpected affinity retrieved (2)";

        uint64_t affinityMaskSetSecond = 0x1;
        status = synStreamSetAffinity(device.getDeviceId(), streamHandle, affinityMaskSetSecond);
        ASSERT_EQ(status, synFail) << "Unexpectedly succeeded to re-set affinity";
    }

    status = synStreamDestroy(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to destroy stream";
}

TEST_F_SYN(MultiStreamsTest, affinity_mak_range_validation)
{
    TestDevice device(m_deviceType);

    uint64_t deviceAffinityMask(0);
    synDeviceGetAffinityMaskRange(device.getDeviceId(), &deviceAffinityMask);

    switch (m_deviceType)
    {
        case synDeviceGaudi:
        {
            ASSERT_EQ(deviceAffinityMask, 0x3);
            break;
        }

        case synDeviceGaudi2:
        case synDeviceGaudi3:
        {
            ASSERT_EQ(deviceAffinityMask, 0x7);
            break;
        }

        case synDeviceEmulator:
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
        {
            break;
        }
    }
}

TEST_F_SYN(MultiStreamsTest, implicit_affinity_test)
{
    const unsigned hostBuffersAmount = 0;
    const uint64_t copySize          = 0;

    TestDevice device(m_deviceType);

    TestStream      stream       = device.createStream();
    synStreamHandle streamHandle = stream;

    preExecution(device, copySize, hostBuffersAmount);

    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    // implicit set affinity
    synStatus status = synLaunchExt(streamHandle,
                                    recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                                    recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                                    recipeLaunchParams.getWorkspace(),
                                    recipe.getRecipe(),
                                    0 /* flags */);
    ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

    const uint64_t expectedAffinityMask(0x1);
    uint64_t       affinityMaskQuery(0x0);
    status = synStreamGetAffinity(device.getDeviceId(), streamHandle, &affinityMaskQuery);
    ASSERT_EQ(status, synSuccess) << "Failed to get stream affinity";
    ASSERT_EQ(expectedAffinityMask, affinityMaskQuery) << "invalid affinity received";

    uint64_t affinityMaskSet = 0x2;
    status                   = synStreamSetAffinity(device.getDeviceId(), streamHandle, affinityMaskSet);
    ASSERT_EQ(status, synFail) << "Unexpectedly succeeded to update (set) stream affinity after first job";

    postExecution(device);
}

TEST_F_SYN(MultiStreamsTest, implicit_affinity_qman_architecture_test)
{
    // Specific for this test
    std::vector<synDeviceType> supporttedDeviceTypes = {synDeviceGaudi};
    if (std::find(supporttedDeviceTypes.begin(), supporttedDeviceTypes.end(), m_deviceType) ==
        supporttedDeviceTypes.end())
    {
        GTEST_SKIP() << "Test is not supported for deviceType " << m_deviceType;
    }

    const unsigned streamsAmount     = 3;
    const unsigned hostBuffersAmount = 0;
    const uint64_t copySize          = 0;

    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream0 = device.createStream();
    TestStream stream1 = device.createStream();
    TestStream stream2 = device.createStream();

    std::array<synStreamHandle, streamsAmount> streamHandles {stream0, stream1, stream2};

    preExecution(device, copySize, hostBuffersAmount);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    for (synStreamHandle streamHandle : streamHandles)
    {
        synStatus status = synLaunchExt(streamHandle,
                                        recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                                        recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    const uint64_t affinityMaskExpected = 0x1;
    for (uint64_t iter = 0; iter < streamHandles.size(); iter++)
    {
        uint64_t  affinityMaskQuery(0x0);
        synStatus status = synStreamGetAffinity(device.getDeviceId(), streamHandles[iter], &affinityMaskQuery);
        ASSERT_EQ(status, synSuccess) << "Failed to get stream affinity";
        ASSERT_EQ(affinityMaskExpected, affinityMaskQuery) << "invalid affinity received";
    }

    postExecution(device);
}

// This test requires the device to support at least two affinities
TEST_F_SYN(MultiStreamsTest, implicit_affinity_scal_round_robin_test)
{
    // Specific for this test
    std::vector<synDeviceType> supporttedDeviceTypes = {synDeviceGaudi2, synDeviceGaudi3};
    if (std::find(supporttedDeviceTypes.begin(), supporttedDeviceTypes.end(), m_deviceType) ==
        supporttedDeviceTypes.end())
    {
        GTEST_SKIP() << "Test is not supported for deviceType " << m_deviceType;
    }

    synStatus status(synSuccess);

    const unsigned streamsAmount     = 3;
    const unsigned hostBuffersAmount = 0;
    const uint64_t copySize          = 0;

    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream                                 stream0 = device.createStream();
    TestStream                                 stream1 = device.createStream();
    TestStream                                 stream2 = device.createStream();
    std::array<synStreamHandle, streamsAmount> streamHandles {stream0, stream1, stream2};

    preExecution(device, copySize, hostBuffersAmount);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    for (const synStreamHandle& streamHandle : streamHandles)
    {
        status = synLaunchExt(streamHandle,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    for (uint64_t iter = 0; iter < streamHandles.size(); iter++)
    {
        const uint64_t affinityMaskExpected = 1 << iter;

        uint64_t affinityMaskQuery(0x0);
        status = synStreamGetAffinity(device.getDeviceId(), streamHandles[iter], &affinityMaskQuery);
        ASSERT_EQ(status, synSuccess) << "Failed to get stream affinity";
        ASSERT_EQ(affinityMaskExpected, affinityMaskQuery) << "invalid affinity received";
    }

    postExecution(device);
}

// This test requires the device to support at least two affinities
TEST_F_SYN(MultiStreamsTest, affinity_hw_queues_validation_test)
{
    synStatus status(synSuccess);

    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
    DeviceCommon*                    pDevice         = (DeviceCommon*)(deviceInterface.get());

    // First implicit affinity to affinity 0
    {
        synStreamHandle streamHandleCompute;

        status = synStreamCreateGeneric(&streamHandleCompute, device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create Compute stream";

        Stream* pStream = getStream(streamHandleCompute, pDevice);
        ASSERT_NE(pStream, nullptr) << "Failed to load Stream";

        // implicit set affinity with single stream
        status = synLaunchExt(streamHandleCompute,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

        QueueInterface* computeQueue;
        pStream->testGetQueueInterface(QUEUE_TYPE_COMPUTE, computeQueue);
        ASSERT_NE(computeQueue, nullptr) << "testGetQueueInterface allocated null compute queue";

        status = synStreamDestroy(streamHandleCompute);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy stream";
    }

    // implicit set affinity with single stream with load balancing by the stream type
    {
        synStreamHandle streamHandleCompute0, streamHandleCompute1;

        status = synStreamCreateGeneric(&streamHandleCompute0, device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create Compute stream";

        status = synStreamCreateGeneric(&streamHandleCompute1, device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create Compute stream";

        Stream* streamCompute0 = getStream(streamHandleCompute0, pDevice);
        ASSERT_NE(streamCompute0, nullptr) << "Failed to load Stream";
        Stream* streamCompute1 = getStream(streamHandleCompute1, pDevice);
        ASSERT_NE(streamCompute1, nullptr) << "Failed to load Stream";

        // after first launch affinity should be set to diffrent queues
        status = synLaunchExt(streamHandleCompute0,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

        status = synLaunchExt(streamHandleCompute1,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

        uint64_t getAffinity0, getAffinity1;
        status = synStreamGetAffinity(device.getDeviceId(), streamHandleCompute0, &getAffinity0);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";
        //
        status = synStreamGetAffinity(device.getDeviceId(), streamHandleCompute1, &getAffinity1);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";

        // Note the implicit behaviour is different between Gaudi1 (same 0x1) and Gaudi2 (different 0x1 vs. 0x2)
        if (m_deviceType == synDeviceGaudi)
        {
            ASSERT_EQ(getAffinity0, getAffinity1) << "synLaunchExt has to keep the affinity";
        }
        else
        {
            ASSERT_NE(getAffinity0, getAffinity1) << "synLaunchExt has to change the affinity";
        }

        QueueInterface* compute0Queue;
        streamCompute0->testGetQueueInterface(QUEUE_TYPE_COMPUTE, compute0Queue);
        //
        QueueInterface* compute1Queue;
        streamCompute1->testGetQueueInterface(QUEUE_TYPE_COMPUTE, compute1Queue);

        // Note the implicit behaviour is different between Gaudi1 (same stream) and Gaudi2 (different stream)
        if (m_deviceType == synDeviceGaudi)
        {
            ASSERT_EQ(compute0Queue, compute1Queue) << "synLaunchExt has to keep the stream";
        }
        else
        {
            ASSERT_NE(compute0Queue, compute1Queue) << "synLaunchExt has to change the stream";
        }

        status = synStreamDestroy(streamHandleCompute0);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy stream";
        //
        status = synStreamDestroy(streamHandleCompute1);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy stream";
    }

    // Explicit set affinity with implicit affinity - load balancing by stream set usage
    {
        synStreamHandle streamHandle0, streamHandle1, streamHandle2;

        status = synStreamCreateGeneric(&streamHandle0, device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create stream";
        //
        status = synStreamCreateGeneric(&streamHandle1, device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create stream";
        //
        status = synStreamCreateGeneric(&streamHandle2, device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create stream";

        Stream* stream0 = getStream(streamHandle0, pDevice);
        ASSERT_NE(stream0, nullptr) << "Failed to load Stream";
        Stream* stream1 = getStream(streamHandle1, pDevice);
        ASSERT_NE(stream1, nullptr) << "Failed to load Stream";
        Stream* stream2 = getStream(streamHandle2, pDevice);
        ASSERT_NE(stream2, nullptr) << "Failed to load Stream";

        uint64_t getAffinity0, getAffinity1, getAffinity2;

        // Validate set affinity changing HW queue and get affinity return correct queue
        uint64_t affinityMask = 0x1;
        status                = synStreamSetAffinity(device.getDeviceId(), streamHandle0, affinityMask);
        ASSERT_EQ(status, synSuccess) << "Failed to set stream affinity";

        affinityMask = 0x2;
        status       = synStreamSetAffinity(device.getDeviceId(), streamHandle1, affinityMask);
        ASSERT_EQ(status, synSuccess) << "Failed to set stream affinity";

        affinityMask = 0x3;
        status       = synStreamSetAffinity(device.getDeviceId(), streamHandle2, affinityMask);
        ASSERT_EQ(status, synSuccess) << "Failed to set stream affinity";

        status = synStreamGetAffinity(device.getDeviceId(), streamHandle0, &getAffinity0);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";

        status = synStreamGetAffinity(device.getDeviceId(), streamHandle1, &getAffinity1);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";

        status = synStreamGetAffinity(device.getDeviceId(), streamHandle2, &getAffinity2);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";

        ASSERT_EQ(getAffinity0, 0x1);
        ASSERT_EQ(getAffinity1, 0x2);
        ASSERT_EQ(getAffinity2, 0x1);

        QueueInterface* queue0;
        QueueInterface* queue1;
        QueueInterface* queue2;
        if (m_deviceType != synDeviceGaudi)
        {
            for (unsigned type = QUEUE_TYPE_COPY_DEVICE_TO_HOST; type < QUEUE_TYPE_NETWORK_COLLECTIVE; type++)
            {
                QueueType queueType = (QueueType)type;

                stream0->testGetQueueInterface(queueType, queue0);
                stream1->testGetQueueInterface(queueType, queue1);
                stream2->testGetQueueInterface(queueType, queue2);

                ASSERT_NE(queue0, queue1) << "synStreamSetAffinity did not change the stream queue for type: " << type;
                ASSERT_EQ(queue0, queue2) << "synStreamSetAffinity changed the stream queue for type: " << type;
            }
        }
        else
        {
            stream0->testGetQueueInterface(QUEUE_TYPE_COMPUTE, queue0);
            stream1->testGetQueueInterface(QUEUE_TYPE_COMPUTE, queue1);
            stream2->testGetQueueInterface(QUEUE_TYPE_COMPUTE, queue2);

            ASSERT_NE(queue0, queue1) << "synStreamSetAffinity did not change the compute queue";
            ASSERT_EQ(queue0, queue2) << "synStreamSetAffinity changed the compute queue";
        }

        ASSERT_NE(queue0, queue1) << "synStreamSetAffinity did not change the compute queue";
        ASSERT_EQ(queue0, queue2) << "synStreamSetAffinity changed the compute queue";

        status = synLaunchExt(streamHandle0,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

        status = synLaunchExt(streamHandle1,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

        status = synStreamGetAffinity(device.getDeviceId(), streamHandle0, &getAffinity0);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";

        status = synStreamGetAffinity(device.getDeviceId(), streamHandle1, &getAffinity1);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";

        status = synStreamGetAffinity(device.getDeviceId(), streamHandle2, &getAffinity2);
        ASSERT_EQ(status, synSuccess) << "Get stream affinity failed";

        ASSERT_EQ(getAffinity0, 0x1);
        ASSERT_EQ(getAffinity1, 0x2);
        ASSERT_EQ(getAffinity2, 0x1);

        status = synStreamDestroy(streamHandle0);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy stream";

        status = synStreamDestroy(streamHandle1);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy stream";

        status = synStreamDestroy(streamHandle2);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy stream";
    }
}

TEST_F_SYN(MultiStreamsTest, use_stream_after_deletion_test)
{
    // Calculate tensor size and allocate device memory
    synStatus status(synSuccess);

    const unsigned hostBuffersAmount = 0;
    const uint64_t copySize          = 0;

    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    synStreamHandle streamHandle;
    {
        TestStream stream = device.createStream();
        streamHandle      = stream;

        preExecution(device, copySize, hostBuffersAmount);

        status = synLaunchExt(streamHandle,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    status = synLaunchExt(streamHandle,
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0 /* flags */);
    ASSERT_EQ(status, synInvalidArgument) << "synLaunch returned different status than synInvalidArgument";

    postExecution(device);
}

TEST_F_SYN(MultiStreamsTest, basic_multistream_memcopy_validation_test)
{
    synStatus status(synSuccess);

    enum class HostBuffer
    {
        WRITE = 0,
        READ  = 1,
        NUM   = 2
    };

    const unsigned buffersAmount = (unsigned)HostBuffer::NUM;
    const uint64_t copySize      = 16 * 1024;

    TestDevice device(m_deviceType);

    TestStream      stream       = device.createStream();
    synStreamHandle streamHandle = stream;

    preExecution(device, copySize, buffersAmount);
    ASSERT_EQ(m_hostBuffers.size(), buffersAmount) << "Not all Host buffers created";
    ASSERT_EQ(m_deviceBuffers.size(), buffersAmount) << "Not all Device buffer created";

    // Test normal mem-copies
    // Host to Device
    status = synMemCopyAsync(streamHandle,
                             (uint64_t)m_hostBuffers[(unsigned)HostBuffer::WRITE],
                             copySize,
                             m_deviceBuffers[0],
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device";
    //
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (download)";
    //
    // Device to Host
    status = synMemCopyAsync(streamHandle,
                             m_deviceBuffers[0],
                             copySize,
                             (uint64_t)m_hostBuffers[(unsigned)HostBuffer::READ],
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device";
    //
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (upload)";

    // Test empty mem-copies
    // Host to Device
    status = synMemCopyAsync(streamHandle,
                             (uint64_t)m_hostBuffers[(unsigned)HostBuffer::WRITE],
                             0 /* copySize */,
                             m_deviceBuffers[0],
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device";
    //
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (download)";
    //
    // Device to Host
    status = synMemCopyAsync(streamHandle,
                             m_deviceBuffers[0],
                             0 /* copySize */,
                             (uint64_t)m_hostBuffers[(unsigned)HostBuffer::READ],
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device";
    //
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (upload)";
    //
    // Device to Device
    status = synMemCopyAsync(streamHandle, m_deviceBuffers[0], 0 /* copySize */, m_deviceBuffers[0], DRAM_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy device to device";
    //
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (upload)";

    postExecution(device);
}

TEST_F_SYN(MultiStreamsTest, basic_multistream_launch_validation_test)
{
    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice      device(m_deviceType);
    TestStream      stream       = device.createStream();
    synStreamHandle streamHandle = stream;

    synStatus status(synSuccess);

    const unsigned hostBuffersAmount = 0;
    const uint64_t copySize          = 0;
    preExecution(device, copySize, hostBuffersAmount);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    status = synLaunchExt(streamHandle,
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0 /* flags */);
    ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream";

    postExecution(device);
}

TEST_F_SYN(MultiStreamsTest, all_in_one_stream_test)
{
    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    // Calculate tensor size and allocate device memory
    synStatus status(synSuccess);

    enum class HostBuffer
    {
        WRITE = 0,
        READ  = 1,
        NUM   = 2
    };

    TestDevice      device(m_deviceType);
    TestStream      stream       = device.createStream();
    synStreamHandle streamHandle = stream;
    auto            event        = device.createEvent(0);
    synEventHandle  eventHandle  = event;

    const unsigned hostBuffersAmount = (unsigned)HostBuffer::NUM;
    const uint64_t copySize          = 16 * 1024;
    preExecution(device, copySize, hostBuffersAmount);
    ASSERT_EQ(m_hostBuffers.size(), hostBuffersAmount) << "Not all Host buffers created";
    ASSERT_EQ(m_deviceBuffers.size(), hostBuffersAmount) << "Not all Device buffer created";

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    status = synLaunchExt(streamHandle,
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0 /* flags */);
    ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

    status = synMemCopyAsync(streamHandle,
                             (uint64_t)m_hostBuffers[(unsigned)HostBuffer::WRITE],
                             copySize,
                             m_deviceBuffers[0],
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device";

    status = synMemCopyAsync(streamHandle,
                             m_deviceBuffers[0],
                             copySize,
                             (uint64_t)m_hostBuffers[(unsigned)HostBuffer::READ],
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device";

    status = synMemCopyAsync(streamHandle, m_deviceBuffers[0], copySize, m_deviceBuffers[1], DRAM_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device";

    status = synEventRecord(eventHandle, streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed record-event (copy to the device)";

    status = synStreamWaitEvent(streamHandle, eventHandle, 0);
    ASSERT_EQ(status, synSuccess) << "Failed stream-wait-event (completion of copy to the device)";

    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream";

    postExecution(device);
}

TEST_F_SYN(MultiStreamsTest, api_test_get_device_streams_affinity)
{
    TestDevice device(m_deviceType);

    uint64_t streamsAffinity;
    ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
        << "Failed to synDeviceGetNextStreamAffinity";
    ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

    synStreamHandle streamHandle1;
    ASSERT_EQ(synStreamCreateGeneric(&streamHandle1, device.getDeviceId(), 0), synSuccess)
        << "Failed to synStreamCreateGeneric";

    synStreamHandle streamHandle2;
    ASSERT_EQ(synStreamCreateGeneric(&streamHandle2, device.getDeviceId(), 0), synSuccess)
        << "Failed to synStreamCreateGeneric";

    synStreamHandle streamHandle3;
    ASSERT_EQ(synStreamCreateGeneric(&streamHandle3, device.getDeviceId(), 0), synSuccess)
        << "Failed to synStreamCreateGeneric";

    synStreamHandle streamHandle4;
    ASSERT_EQ(synStreamCreateGeneric(&streamHandle4, device.getDeviceId(), 0), synSuccess)
        << "Failed to synStreamCreateGeneric";

    if (m_deviceType == synDeviceGaudi)
    {
        // [4,0]
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        // set the stream affinity
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle1, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        // should fail affinity already set
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle1, 0x1), synFail)
            << "succeed to set stream affinity after first job";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        // force the stream affinity with 0x2  [3,1]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle2, 0x2), synSuccess)
            << "succeed to set stream affinity after first job";

        // force the stream affinity with 0x1
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle3, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        // next avail affiinty should be 0x2
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x2, streamsAffinity) << "invalid affinity received";

        // set the affiinty  [2,2]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle4, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";
    }
    else
    {
        // [4,0,0]
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x2, streamsAffinity) << "invalid affinity received";

        // [3,1,0]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle1, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        // should fail affinity already set
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle1, 0x1), synFail)
            << "succeed to set stream affinity after first job";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x4, streamsAffinity) << "invalid affinity received";

        // force the stream affinity with 0x4  [2,1,1]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle2, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x2, streamsAffinity) << "invalid affinity received";

        // force the stream affinity with 0x2  [1,2,1]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle3, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        // next avail affiinty should be 0x1
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        // set the affiinty 0x1
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle4, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";
    }

    ASSERT_EQ(synStreamDestroy(streamHandle1), synSuccess);
    ASSERT_EQ(synStreamDestroy(streamHandle2), synSuccess);
    ASSERT_EQ(synStreamDestroy(streamHandle3), synSuccess);
    ASSERT_EQ(synStreamDestroy(streamHandle4), synSuccess);
}

TEST_F_SYN(MultiStreamsTest, api_test_get_device_streams_affinity_with_destroy)
{
    TestDevice device(m_deviceType);

    uint64_t streamsAffinity;
    ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
        << "Failed to synDeviceGetNextStreamAffinity";
    ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

    // both streams default affinities are 0x1
    synStreamHandle streamHandle1;
    ASSERT_EQ(synStreamCreateGeneric(&streamHandle1, device.getDeviceId(), 0), synSuccess)
        << "Failed to synStreamCreateGeneric";

    synStreamHandle streamHandle2;
    ASSERT_EQ(synStreamCreateGeneric(&streamHandle2, device.getDeviceId(), 0), synSuccess)
        << "Failed to synStreamCreateGeneric";

    if (m_deviceType == synDeviceGaudi)
    {
        // set the stream affinity 0x1  [2,0]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle1, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        // force the stream affinity with 0x2  [1,1]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle2, 0x2), synSuccess)
            << "succeed to set stream affinity after first job";

        // next avail affiinty should be 0x1
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        // destroy stream2   [1,0]
        ASSERT_EQ(synStreamDestroy(streamHandle2), synSuccess);

        // next avail affiinty should be 0x2
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x2, streamsAffinity) << "invalid affinity received";

        synStreamHandle streamHandle3;  //[2,0]
        ASSERT_EQ(synStreamCreateGeneric(&streamHandle3, device.getDeviceId(), 0), synSuccess)
            << "Failed to synStreamCreateGeneric";

        // set the affiinty with 0x2   //[1,1]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle3, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        // next avail affiinty should be 0x1
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        ASSERT_EQ(synStreamDestroy(streamHandle1), synSuccess);
        ASSERT_EQ(synStreamDestroy(streamHandle3), synSuccess);
    }
    else
    {
        // //[2,0,0]
        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x2, streamsAffinity) << "invalid affinity received";

        // set the stream affinity 0x2   [1,1,0]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle2, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x4, streamsAffinity) << "invalid affinity received";

        // destoy stream1   [0,1,0]
        ASSERT_EQ(synStreamDestroy(streamHandle1), synSuccess);

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        synStreamHandle streamHandle3;  //[1,1,0]
        ASSERT_EQ(synStreamCreateGeneric(&streamHandle3, device.getDeviceId(), 0), synSuccess)
            << "Failed to synStreamCreateGeneric";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x4, streamsAffinity) << "invalid affinity received";

        // set the stream affinity with 0x4  [0,1,1]
        ASSERT_EQ(synStreamSetAffinity(device.getDeviceId(), streamHandle3, streamsAffinity), synSuccess)
            << "succeed to set stream affinity after first job";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        synStreamHandle streamHandle4;  // [1,1,1]
        ASSERT_EQ(synStreamCreateGeneric(&streamHandle4, device.getDeviceId(), 0), synSuccess)
            << "Failed to synStreamCreateGeneric";

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";

        ASSERT_EQ(synStreamDestroy(streamHandle2), synSuccess);  // [1,0,1]

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x2, streamsAffinity) << "invalid affinity received";

        ASSERT_EQ(synStreamDestroy(streamHandle3), synSuccess);  // [1,0,0]

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x2, streamsAffinity) << "invalid affinity received";

        ASSERT_EQ(synStreamDestroy(streamHandle4), synSuccess);  // [0,0,0]

        ASSERT_EQ(synDeviceGetNextStreamAffinity(device.getDeviceId(), &streamsAffinity), synSuccess)
            << "Failed to synDeviceGetNextStreamAffinity";
        ASSERT_EQ(0x1, streamsAffinity) << "invalid affinity received";
    }
}