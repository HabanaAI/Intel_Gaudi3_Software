#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_recipe_nop_x_nodes.hpp"
#include "synapse_api.h"
#include "test_launcher.hpp"

class synGaudiRtPerf : public SynBaseTest
{
public:
    synGaudiRtPerf() { setSupportedDevices({synDeviceGaudi}); }

    static const unsigned numOfSynLaunch = 20;

    void runTpcNop(int numNodes, uint64_t results[numOfSynLaunch]);
};

REGISTER_SUITE(synGaudiRtPerf, ALL_TEST_PACKAGES);

void synGaudiRtPerf::runTpcNop(int numNodes, uint64_t results[numOfSynLaunch])
{
    TestRecipeNopXNodes nopRecipe(m_deviceType, 1000);
    nopRecipe.generateRecipe();

    TestDevice device(m_deviceType);

    enum EventId
    {
        START,
        END,
        COMPUTE_TO_UPLOAD,
        NUM
    };

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(nopRecipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    auto computeStreamHandle  = device.createStream();
    auto uploadStreamHandle   = device.createStream();
    auto downloadStreamHandle = device.createStream();

    auto                                           eventStart   = device.createEvent(EVENT_COLLECT_TIME);
    auto                                           eventEnd     = device.createEvent(EVENT_COLLECT_TIME);
    auto                                           eventCompute = device.createEvent(EVENT_COLLECT_TIME);
    const std::array<synEventHandle, EventId::NUM> eventHandles {eventStart, eventEnd, eventCompute};

    const TensorInfo* inputTensorInfo  = nopRecipe.getTensorInfo("input");
    const TensorInfo* outputTensorInfo = nopRecipe.getTensorInfo("output");

    // Copy data from device to host
    const auto& hostInput   = recipeLaunchParams.getHostInput(0);
    const auto& deviceInput = recipeLaunchParams.getDeviceInput(0);

    synStatus status = synMemCopyAsync(downloadStreamHandle,
                                       (uint64_t)hostInput.getBuffer(),
                                       inputTensorInfo->m_tensorSize,
                                       deviceInput.getBuffer(),
                                       HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device to output";

    status = synEventRecord(eventHandles[EventId::START], computeStreamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed eventRecord";

    for (unsigned launchNumber = 0; launchNumber < numOfSynLaunch; launchNumber++)
    {
        status = synLaunchExt(computeStreamHandle,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                              recipeLaunchParams.getWorkspace(),
                              nopRecipe.getRecipe(),
                              SYN_FLAGS_TENSOR_NAME);
        ASSERT_EQ(status, synSuccess) << "Failed to launch";

        status = synEventRecord(eventHandles[EventId::COMPUTE_TO_UPLOAD], computeStreamHandle);
        ASSERT_EQ(status, synSuccess) << "Failed eventRecord";

        status = synStreamWaitEvent(uploadStreamHandle, eventHandles[EventId::COMPUTE_TO_UPLOAD], 0);
        ASSERT_EQ(status, synSuccess) << "Failed eventRecord";

        const auto& hostOutput   = recipeLaunchParams.getHostOutput(0);
        const auto& deviceOutput = recipeLaunchParams.getDeviceOutput(0);

        status = synMemCopyAsync(uploadStreamHandle,
                                 deviceOutput.getBuffer(),
                                 outputTensorInfo->m_tensorSize,
                                 (uint64_t)hostOutput.getBuffer(),
                                 DRAM_TO_HOST);
        ASSERT_EQ(status, synSuccess) << "Failed copy from the device to output";

        status = synStreamSynchronize(uploadStreamHandle);  // wait for the MemCopy
        ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (copy from the device)";
    }

    status = synEventRecord(eventHandles[EventId::END], computeStreamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed eventRecord";

    status = synEventSynchronize(eventHandles[EventId::END]);
    ASSERT_EQ(status, synSuccess) << "Failed synEventSynchronize";

    uint64_t nanoSeconds = 0;
    //
    status = synEventElapsedTime(&nanoSeconds, eventHandles[EventId::START], eventHandles[EventId::END]);
    ASSERT_EQ(status, synSuccess) << "Failed synEventElapsedTime";
    LOG_DEBUG(SYN_RT_TEST, "Simple Multistream Time: {}ns", nanoSeconds);
}

TEST_F_SYN(synGaudiRtPerf, DISABLED_NOP_perf)
{
    unsigned    numNodes = 1;
    const char* envValue = getenv("TPC_NOP_BENCHMARK_NUM_NODES");
    if (envValue)
    {
        numNodes = std::stoi(envValue);
    }

    uint64_t results[numOfSynLaunch];

    runTpcNop(numNodes, results);
}
