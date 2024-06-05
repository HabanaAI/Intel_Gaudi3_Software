#include "syn_base_test.hpp"
#include "log_manager.h"
#include "synapse_common_types.h"
#include "test_recipe_nop_x_nodes.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "package_support_macros.h"

// After the introduction of multistream, the behavior of synEventRecord
// has changed - the event is now recorded on the last used physical
// queue, as opposed to the queue defined by the user at the creation of
// the old stream.

// As a result, synEventElapsedTime behaves slightly differently as well -
// while still providing the time between two events, the events themselves
// can be on different queues and therefore not wrap an indiviual job or
// job type.

class SynEventElapsedTimeTest : public SynBaseTest
{
public:
    SynEventElapsedTimeTest() : SynBaseTest()
    {
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
    }

    ~SynEventElapsedTimeTest() override = default;

    // Allocate handles & compile recipe
    void preExecution(const TestDevice& rDevice);
    void postExecution(const TestDevice& rDevice);

    void simpleTest();
    void twoStreamsTest();
    void twoStreamsLongDmaTest();

private:
    struct CopyOnlyBuffers
    {
        void*    m_hostBuffer;
        uint64_t m_deviceBuffer;
        uint64_t m_size;
    };

    std::vector<CopyOnlyBuffers> m_copyOnlyBuffers = {};
};

REGISTER_SUITE(SynEventElapsedTimeTest, ALL_TEST_PACKAGES);

void SynEventElapsedTimeTest::preExecution(const TestDevice& rDevice)
{
    for (auto& singleBuffer : m_copyOnlyBuffers)
    {
        // Host buffer allocation
        synStatus status = synHostMalloc(rDevice.getDeviceId(), singleBuffer.m_size, 0, &singleBuffer.m_hostBuffer);
        ASSERT_EQ(status, synSuccess) << "Failed alocate host buffer";
        std::memset(singleBuffer.m_hostBuffer, 0xCA, singleBuffer.m_size);

        // Device buffer allocation
        status = synDeviceMalloc(rDevice.getDeviceId(), singleBuffer.m_size, 0, 0, &singleBuffer.m_deviceBuffer);
        ASSERT_EQ(status, synSuccess) << "Failed to alocate HBM memory";
    }
}

void SynEventElapsedTimeTest::postExecution(const TestDevice& rDevice)
{
    synStatus status(synSuccess);

    for (auto singleBuffer : m_copyOnlyBuffers)
    {
        status = synDeviceFree(rDevice.getDeviceId(), singleBuffer.m_deviceBuffer, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to free HBM memory";

        status = synHostFree(rDevice.getDeviceId(), singleBuffer.m_hostBuffer, 0);
        ASSERT_EQ(status, synSuccess) << "Failed alocate host buffer";
    }
}

void SynEventElapsedTimeTest::simpleTest()
{
    enum EventId
    {
        START = 0,
        END   = 1,
        NUM   = 2
    };

    synStatus status(synSuccess);

    const uint64_t copySize = 16 * 1024;
    m_copyOnlyBuffers.resize(1);
    m_copyOnlyBuffers[0].m_size = copySize;

    TestRecipeNopXNodes recipe(m_deviceType);

    recipe.generateRecipe();

    TestDevice                                     device(m_deviceType);
    auto                                           stream       = device.createStream();
    synStreamHandle                                streamHandle = stream;
    auto                                           eventStart   = device.createEvent(EVENT_COLLECT_TIME);
    auto                                           eventEnd     = device.createEvent(EVENT_COLLECT_TIME);
    const std::array<synEventHandle, EventId::NUM> eventHandles {eventStart, eventEnd};

    preExecution(device);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    status = synEventRecord(eventHandles[EventId::START], streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed eventRecord";

    status = synLaunchExt(streamHandle,
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          SYN_FLAGS_TENSOR_NAME);
    ASSERT_EQ(status, synSuccess) << "Failed to launch";

    status = synMemCopyAsync(streamHandle,
                             (uint64_t)m_copyOnlyBuffers[0].m_hostBuffer,
                             copySize,
                             m_copyOnlyBuffers[0].m_deviceBuffer,
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device";

    status = synEventRecord(eventHandles[EventId::END], streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed eventRecord";

    status = synEventSynchronize(eventHandles[EventId::END]);
    ASSERT_EQ(status, synSuccess) << "Failed synEventSynchronize";

    uint64_t nanoSeconds = 0;
    //
    status = synEventElapsedTime(&nanoSeconds, eventHandles[EventId::START], eventHandles[EventId::END]);
    ASSERT_EQ(status, synSuccess) << "Failed synEventElapsedTime";
    LOG_DEBUG(SYN_RT_TEST, "Simple Multistream Time: {}ns", nanoSeconds);

    postExecution(device);
}

void SynEventElapsedTimeTest::twoStreamsTest()
{
    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    const unsigned                                   streamsAmount = 2;
    auto                                             stream0       = device.createStream();
    auto                                             stream1       = device.createStream();
    const std::array<synStreamHandle, streamsAmount> streamHandles {stream0, stream1};

    enum EventId
    {
        START = 0,
        END   = 1,
        NUM   = 2
    };
    auto                                           eventStart = device.createEvent(EVENT_COLLECT_TIME);
    auto                                           eventEnd   = device.createEvent(EVENT_COLLECT_TIME);
    const std::array<synEventHandle, EventId::NUM> eventHandles {eventStart, eventEnd};

    preExecution(device);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    // Record start and Launch are over the first Stream
    // Record end depends on the index
    for (unsigned streamIndex = 0; streamIndex < streamsAmount; streamIndex++)
    {
        synStatus status = synEventRecord(eventHandles[EventId::START], streamHandles[0]);
        ASSERT_EQ(status, synSuccess) << "Failed eventRecord start (streamIndex = " << streamIndex << ")";

        status = synLaunchExt(streamHandles[0],
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              recipe.getRecipe(),
                              SYN_FLAGS_TENSOR_NAME);
        ASSERT_EQ(status, synSuccess) << "Failed to launch (streamIndex = " << streamIndex << ")";

        status = synEventRecord(eventHandles[EventId::END], streamHandles[streamIndex]);
        ASSERT_EQ(status, synSuccess) << "Failed eventRecord end (streamIndex = " << streamIndex << ")";

        status = synEventSynchronize(eventHandles[EventId::END]);
        ASSERT_EQ(status, synSuccess) << "Failed synEventSynchronize (streamIndex = " << streamIndex << ")";

        uint64_t nanoSeconds = 0;
        //
        status = synEventElapsedTime(&nanoSeconds, eventHandles[EventId::START], eventHandles[EventId::END]);
        ASSERT_EQ(status, synSuccess) << "Failed synEventElapsedTime";
        LOG_DEBUG(SYN_RT_TEST, "Multistream {} Time: {}ns", streamIndex, nanoSeconds);
    }

    postExecution(device);
}

void SynEventElapsedTimeTest::twoStreamsLongDmaTest()
{
    enum class CopyId
    {
        REGULAR = 0,
        BIG     = 1,
        NUM     = 2
    };

    const uint64_t copySize    = 16 * 1024;
    const uint64_t bigCopySize = 100 * copySize;
    m_copyOnlyBuffers.resize((unsigned)CopyId::NUM);
    m_copyOnlyBuffers[(unsigned)CopyId::REGULAR].m_size = copySize;
    m_copyOnlyBuffers[(unsigned)CopyId::BIG].m_size     = bigCopySize;

    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    const unsigned                                   streamsAmount = 2;
    auto                                             stream0       = device.createStream();
    auto                                             stream1       = device.createStream();
    const std::array<synStreamHandle, streamsAmount> streamHandles {stream0, stream1};

    enum EventId
    {
        START = 0,
        MID   = 1,
        END   = 2,
        NUM   = 3
    };
    auto                                           eventStart = device.createEvent(EVENT_COLLECT_TIME);
    auto                                           eventMid   = device.createEvent(EVENT_COLLECT_TIME);
    auto                                           eventEnd   = device.createEvent(EVENT_COLLECT_TIME);
    const std::array<synEventHandle, EventId::NUM> eventHandles {eventStart, eventMid, eventEnd};

    preExecution(device);

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    // Record start over first-stream
    synStatus status = synEventRecord(eventHandles[EventId::START], streamHandles[0]);
    ASSERT_EQ(status, synSuccess) << "Failed eventRecord (start)";

    // Launch over first-stream
    status = synLaunchExt(streamHandles[0],
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          SYN_FLAGS_TENSOR_NAME);
    ASSERT_EQ(status, synSuccess) << "Failed to launch";

    // MemCopy over second-stream
    status = synMemCopyAsync(streamHandles[1],
                             (uint64_t)m_copyOnlyBuffers[(unsigned)CopyId::BIG].m_hostBuffer,
                             bigCopySize,
                             m_copyOnlyBuffers[(unsigned)CopyId::BIG].m_deviceBuffer,
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device on stream 2";

    // Record middle over first-stream
    status = synEventRecord(eventHandles[EventId::MID], streamHandles[0]);
    ASSERT_EQ(status, synSuccess) << "Failed eventRecord (middle)";

    // MemCopy over first-stream
    status = synMemCopyAsync(streamHandles[0],
                             (uint64_t)m_copyOnlyBuffers[(unsigned)CopyId::REGULAR].m_hostBuffer,
                             copySize,
                             m_copyOnlyBuffers[(unsigned)CopyId::REGULAR].m_deviceBuffer,
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device on stream 1";

    // Record end over first-stream
    status = synEventRecord(eventHandles[EventId::END], streamHandles[0]);
    ASSERT_EQ(status, synSuccess) << "Failed eventRecord";

    // Synchronize bith streams
    status = synStreamSynchronize(streamHandles[0]);
    ASSERT_EQ(status, synSuccess) << "Failed to sync stream 1";
    status = synStreamSynchronize(streamHandles[1]);
    ASSERT_EQ(status, synSuccess) << "Failed to sync stream 2";

    // Check elapsed-time
    uint64_t midTime, endTime, totalTime = 0;
    //
    status = synEventElapsedTime(&totalTime, eventHandles[EventId::START], eventHandles[EventId::END]);
    ASSERT_EQ(status, synSuccess) << "Failed synEventElapsedTime - Compute+DMA";
    LOG_DEBUG(SYN_RT_TEST, "Compute + DMA Time: {}ns", totalTime);

    status = synEventElapsedTime(&midTime, eventHandles[EventId::START], eventHandles[EventId::MID]);
    ASSERT_EQ(status, synSuccess) << "Failed synEventElapsedTime - Compute Only";
    LOG_DEBUG(SYN_RT_TEST, "Compute Only Time: {}ns", midTime);

    status = synEventElapsedTime(&endTime, eventHandles[EventId::MID], eventHandles[EventId::END]);
    ASSERT_EQ(status, synSuccess) << "Failed synEventElapsedTime - DMA Only";
    LOG_DEBUG(SYN_RT_TEST, "DMA Only Time (DMA Only): {}ns", endTime);

    ASSERT_EQ(totalTime, midTime + endTime) << "Times don't line up: sum is " << midTime + endTime;

    postExecution(device);
}

// Single multistream, time elapsed of launch and memcopy
TEST_F_SYN(SynEventElapsedTimeTest, simple_test)
{
    simpleTest();
}

// Two multistreams, comparing launch + event record on a single multistream,
// versus launch on one stream and eventRecord on the other.
TEST_F_SYN(SynEventElapsedTimeTest, two_streams_compare)
{
    twoStreamsTest();
}

// Two multistreams, with one running a long dma job,
// and the other trying to synEventElapsedTime of a launch and a dma job.
//
// The second stream's time is going to be longer, since it accounts
// for a part of the long dma that the 1st stream is running.
TEST_F_SYN(SynEventElapsedTimeTest, two_streams_long_dma)
{
    twoStreamsLongDmaTest();
}