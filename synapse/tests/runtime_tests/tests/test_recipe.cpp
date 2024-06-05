#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "test_device.hpp"
#include "test_recipe_multiple_memcpy.hpp"
#include "runtime/common/recipe/recipe_verification.hpp"
#include "runtime/qman/common/qman_event.hpp"
#include "syn_singleton.hpp"
#include "test_launcher.hpp"

class SynGaudiTestRecipe : public SynBaseTest
{
public:
    SynGaudiTestRecipe() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }

    void analyzeTensorsPerformance(std::string prefix);
};

REGISTER_SUITE(SynGaudiTestRecipe,
               synTestPackage::CI,
               synTestPackage::SIM,
               synTestPackage::ASIC,
               synTestPackage::ASIC_CI);

TEST_F_SYN(SynGaudiTestRecipe, testAnalyzeTensors)
{
    synConfigurationSet("CHECK_SECTION_OVERLAP_CHECK", "true");

    const int numNodes = 5;

    TestRecipeMultipleMemcpy recipe(m_deviceType, numNodes);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);
    TestStream streamHandle = device.createStream();

    TestLauncher launcher(device);
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});
    // All good, should pass
    auto launchTensors = recipeLaunchParams.getSynLaunchTensorInfoVec();
    {
        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors.data(),
                                        launchTensors.size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    // duplicate section addr
    {
        uint64_t temp                   = launchTensors[3].pTensorAddress;
        launchTensors[3].pTensorAddress = launchTensors[2].pTensorAddress;

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors.data(),
                                        launchTensors.size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
        launchTensors[3].pTensorAddress = temp;
    }

    // overlap section addr
    {
        uint64_t temp                   = launchTensors[3].pTensorAddress;
        launchTensors[3].pTensorAddress = launchTensors[2].pTensorAddress + 10;

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors.data(),
                                        launchTensors.size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
        launchTensors[3].pTensorAddress = temp;
    }

    // Another overlap section addr
    {
        uint64_t temp                   = launchTensors[3].pTensorAddress;
        launchTensors[3].pTensorAddress = launchTensors[2].pTensorAddress - 10;

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors.data(),
                                        launchTensors.size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
        launchTensors[3].pTensorAddress = temp;
    }

    // Overlap section addr first
    {
        uint64_t temp                   = launchTensors[2].pTensorAddress;
        launchTensors[2].pTensorAddress = launchTensors[2].pTensorAddress - 10;

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors.data(),
                                        launchTensors.size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
        launchTensors[2].pTensorAddress = temp;
    }

    // Overlap section addr last
    {
        uint64_t temp                          = launchTensors[numNodes].pTensorAddress;
        launchTensors[numNodes].pTensorAddress = launchTensors[numNodes].pTensorAddress - 10;

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors.data(),
                                        launchTensors.size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
        launchTensors[numNodes].pTensorAddress = temp;
    }

    // All good, should pass
    {
        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors.data(),
                                        launchTensors.size(),
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    };
}

TEST_F_SYN(SynGaudiTestRecipe, testTensorChecking)
{
    const int numNodes = 1;

    TestDevice device(m_deviceType);
    TestStream streamHandle = device.createStream();

    TestRecipeMultipleMemcpy recipe(m_deviceType, numNodes);
    recipe.generateRecipe();

    TestLauncher       launcher(device);
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    uint64_t devMemBuff = recipeLaunchParams.getSynLaunchTensorInfoVec()[0].pTensorAddress;
    uint64_t vecSize    = recipe.getTensorInfoInput(0)->m_tensorSize;

    const char* inName  = recipeLaunchParams.getSynLaunchTensorInfoVec()[0].tensorName;
    const char* outName = recipeLaunchParams.getSynLaunchTensorInfoVec()[1].tensorName;
    uint64_t    inId    = recipeLaunchParams.getSynLaunchTensorInfoVec()[0].tensorId;
    uint64_t    outId   = recipeLaunchParams.getSynLaunchTensorInfoVec()[1].tensorId;

    {
        LOG_INFO(SYN_RT_TEST, "---Testing good case");
        synLaunchTensorInfoExt launchTensors[] = {{inName, devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}, inId},
                                                  {outName, devMemBuff + vecSize, DATA_TENSOR, {0, 0, 0, 0, 0}, outId}};

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors,
                                        2,
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    // Duplicate tensor, should pass
    {
        LOG_INFO(SYN_RT_TEST, "---Testing Duplicate Tensor");
        synLaunchTensorInfoExt launchTensors[] = {{inName, devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}, inId},
                                                  {outName, devMemBuff + vecSize, DATA_TENSOR, {0, 0, 0, 0, 0}, outId},
                                                  {inName, devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}, inId}};

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors,
                                        3,
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    // Extra tensor (unused), should pass
    {
        LOG_INFO(SYN_RT_TEST, "---Testing Extra Tensor");
        synLaunchTensorInfoExt launchTensors[] = {{inName, devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}, inId},
                                                  {outName, devMemBuff + vecSize, DATA_TENSOR, {0, 0, 0, 0, 0}, outId},
                                                  {"in2", devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}, inId}};

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors,
                                        3,
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    // Bad tensor type, should pass, ignored for non-DSD recipe
    {
        LOG_INFO(SYN_RT_TEST, "---Testing bad tensor type");
        synLaunchTensorInfoExt launchTensors[] = {
            {inName, devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}, inId},
            {outName, devMemBuff + vecSize, SHAPE_TENSOR, {0, 0, 0, 0, 0}, outId}};

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors,
                                        2,
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    }

    // Wrong name
    {
        LOG_INFO(SYN_RT_TEST, "---Testing Wrnog name");
        synLaunchTensorInfoExt launchTensors[] = {{inName, devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}},
                                                  {"out1", devMemBuff + vecSize, DATA_TENSOR, {0, 0, 0, 0, 0}}};

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors,
                                        2,
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
    }

    // missing tensor
    {
        LOG_INFO(SYN_RT_TEST, "---Testing Missing Tensor1");
        synLaunchTensorInfoExt launchTensors[] = {{inName, devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}, inId}};

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors,
                                        1,
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
    }

    // missing tensor, second try
    {
        LOG_INFO(SYN_RT_TEST, "---Testing Missing Tensor2");
        synLaunchTensorInfoExt launchTensors[] = {{outName, devMemBuff + vecSize, DATA_TENSOR, {0, 0, 0, 0, 0}, outId}};

        synStatus status = synLaunchExt(streamHandle,
                                        launchTensors,
                                        1,
                                        recipeLaunchParams.getWorkspace(),
                                        recipe.getRecipe(),
                                        0 /* flags */);
        ASSERT_EQ(status, synFailedSectionValidation) << "Failed to synLaunch";
    }
}

void SynGaudiTestRecipe::analyzeTensorsPerformance(std::string prefixParam)
{
    const int numNodes = 1000;

    TestRecipeMultipleMemcpy recipe(m_deviceType, numNodes);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);
    TestStream streamHandle = device.createStream();

    TestLauncher launcher(device);
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    synStatus status = synLaunchExt(streamHandle,
                                    recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                                    recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                                    recipeLaunchParams.getWorkspace(),
                                    recipe.getRecipe(),
                                    0 /* flags */);
    ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

    status = synStreamSynchronize(streamHandle);  // wait for the MemCopy
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (copy from the device)";
}

// Note, run with env ENABLE_STATS=true

TEST_F_SYN(SynGaudiTestRecipe, DISABLED_testAnalyzeTensorsPerformance)
{
    // All names have 8 chars, add 3/42 to get to 11/50
    auto consoleHolder = hl_logger::addConsole(synapse::LogManager::LogType::PERF);
    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 0);
    synConfigurationSet("CHECK_SECTION_OVERLAP_CHECK", "true");
    LOG_INFO(PERF, "---- CHECK_SECTION_OVERLAP_CHECK + short ----");
    analyzeTensorsPerformance("123");
    LOG_INFO(PERF, "---- CHECK_SECTION_OVERLAP_CHECK + long ----");
    analyzeTensorsPerformance("123456789012345678901234567890123456789012");
    //                                  10        20        30        40
    synConfigurationSet("CHECK_SECTION_OVERLAP_CHECK", "false");
    LOG_INFO(PERF, "---- NO CHECK_SECTION_OVERLAP_CHECK + short ----");
    analyzeTensorsPerformance("123");
    LOG_INFO(PERF, "---- CHECK_SECTION_OVERLAP_CHECK + long ----");
    analyzeTensorsPerformance("123456789012345678901234567890123456789012");
    //                                   10        20        30        40
}

TEST_F_SYN(SynGaudiTestRecipe, testElapseTime_ASIC_CI, {synTestPackage::ASIC_CI, synTestPackage::ASIC})
{
    const int numNodes = 1;

    TestRecipeMultipleMemcpy recipe(m_deviceType, numNodes);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);
    TestStream streamHandle   = device.createStream();
    TestStream streamHandleUp = device.createStream();

    const uint32_t deviceId = device.getDeviceId();

    TestLauncher launcher(device);
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    uint64_t devMemBuff = recipeLaunchParams.getSynLaunchTensorInfoVec()[0].pTensorAddress;
    uint64_t vecSize    = recipe.getTensorInfoInput(0)->m_tensorSize;

    // synLaunchTensorInfo launchTensors[] = {{"in", devMemBuff, DATA_TENSOR, {0, 0, 0, 0, 0}},
    //                                       {"out", devMemBuff + vecSize, DATA_TENSOR, {0, 0, 0, 0, 0}}};

    /*--------------------------*/
    /*       Test begins here   */
    /*--------------------------*/
    // Create two event
    synEventHandle eventStart;
    synStatus      status = synEventCreate(&eventStart, deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(status, synSuccess) << "Failed to create start event";
    synEventHandle eventEnd;
    status = synEventCreate(&eventEnd, deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(status, synSuccess) << "Failed to create end event";

//#define CHK_BUSY // This is risky to add to master as timing might change and fail the test. Keeping it here of
// development only.
#ifdef CHK_BUSY
    // try to fill compute stream (synLaunch), Record event, synlaunch, record a second event
    for (int i = 0; i < 20; i++)
    {
        status = graph.launch(computeStream, launchTensors, 2);
        ASSERT_EQ(status, synSuccess) << "Failed to launch1";
    }
#endif
    status = synEventRecord(eventStart, streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to record event start1";

    status = synLaunchExt(streamHandle,
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipeLaunchParams.getSynLaunchTensorInfoVec().size(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0 /* flags */);
    ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";

    status = synEventRecord(eventEnd, streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to record event end1";

    uint64_t elapseTime;
#ifdef CHK_BUSY
    status = synEventElapsedTime(&elapseTime, eventStart, eventEnd);
    printf("Elapse time 1 %ld\n", elapseTime);
    // This is expected to fail with busy, but I can't be sure, keeping it as a comments
    ASSERT_EQ(status, synBusy) << "Failed to get elapsed time";  // TODO!!!
#endif
    for (int cnt = 0; cnt < 50; cnt++)
    {
        status = synEventElapsedTime(&elapseTime, eventStart, eventEnd);
        if (status == synSuccess) break;
        usleep(1000);
    }
    ASSERT_EQ(status, synSuccess);

    // Check one second
    status = synEventRecord(eventStart, streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to record event start2";
    sleep(1);
    status = synEventRecord(eventEnd, streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to record event end2";

    for (int cnt = 0; cnt < 50; cnt++)
    {
        status = synEventElapsedTime(&elapseTime, eventStart, eventEnd);
        if (status == synSuccess) break;
        usleep(1000);
    }

    ASSERT_EQ(status, synSuccess) << "Failed to get elapsed time";

    // ASSERT_FALSE((elapseTime != 0) && ((elapseTime > 1100000000) || (elapseTime < 900000000))) << "Time expected to
    // be around 1 second:" << elapseTime; // TBD - add this check once timestamps interupts are more robust

    synEventHandle eventCompute;
    status = synEventCreate(&eventCompute, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "create event";

    status = synEventElapsedTime(&elapseTime, eventStart, eventCompute);
    ASSERT_EQ(status, synInvalidArgument) << "Should fail, compute doesn't collect time, second argument";

    status = synEventElapsedTime(&elapseTime, eventCompute, eventStart);
    ASSERT_EQ(status, synInvalidArgument) << "Should fail, compute doesn't collect time, first argument";

    synEventHandle eventUnused;
    status = synEventCreate(&eventUnused, deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(status, synSuccess) << "Failed to create start event";

    status = synEventElapsedTime(&elapseTime, eventStart, eventUnused);
    ASSERT_EQ(status, synObjectNotInitialized) << "Should fail, unused event gone in LKD";

    status = synEventElapsedTime(&elapseTime, eventUnused, eventStart);
    ASSERT_EQ(status, synObjectNotInitialized) << "Should fail, unused event gone in LKD";

    QmanEvent* internalEventHandle = dynamic_cast<QmanEvent*>(_SYN_SINGLETON_INTERNAL->getEventInterface(eventUnused));
    if (internalEventHandle)
    {
        internalEventHandle->testingOnlySetSeqId(10000);
        status = synEventElapsedTime(&elapseTime, eventStart, eventUnused);
        ASSERT_NE(status, synSuccess) << "Should fail, unused event gone in LKD";

        status = synEventElapsedTime(&elapseTime, eventUnused, eventStart);
        ASSERT_NE(status, synSuccess) << "Should fail, unused event gone in LKD";
    }

    /*--------------------------*/
    /*   Tesing done, cleanup   */
    /*--------------------------*/

    // Host buffer allocation
    void* dataOut;
    status = synHostMalloc(deviceId, vecSize, 0, &dataOut);
    ASSERT_EQ(status, synSuccess) << "Failed alocate host buffer";
    std::memset(dataOut, 0xCA, vecSize);

    status = synStreamWaitEvent(streamHandleUp, eventCompute, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to wait on event";

    status = synMemCopyAsync(streamHandleUp, devMemBuff + vecSize, vecSize, (uint64_t)dataOut, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device to output";

    status = synStreamSynchronize(streamHandleUp);  // wait for the MemCopy
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (copy from the device)";

    status = synHostFree(deviceId, dataOut, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to unmap out vec to device";
}

TEST_F_SYN(SynGaudiTestRecipe, testElapseTimeGaudi2_ASIC_CI, {synDeviceGaudi2})
{
    TestDevice device(m_deviceType);

    const uint32_t deviceId = device.getDeviceId();

    // Create streams
    synStreamHandle streamH2D, streamD2H;
    synStatus       status = synStreamCreateGeneric(&streamH2D, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";
    status = synStreamCreateGeneric(&streamD2H, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data from the device";

    synEventHandle eventStart, eventEnd, eventDefault;
    status = synEventCreate(&eventStart, deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(status, synSuccess) << "Failed to create start event";
    status = synEventCreate(&eventEnd, deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(status, synSuccess) << "Failed to create end event";
    status = synEventCreate(&eventDefault, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create end event";

    // Device-side (HBM) buffers for input and output
    const uint64_t size = 16;
    uint8_t *      input, *output;
    status = synHostMalloc(deviceId, size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    status = synHostMalloc(deviceId, size, 0, (void**)&output);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for output";
    memset(input, 0xEE, size);
    memset(output, 0, size);

    uint64_t deviceAddress;
    status = synDeviceMalloc(deviceId, size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    status = synEventRecord(eventStart, streamH2D);
    ASSERT_EQ(status, synSuccess) << "Failed to record event start1";

    // Copy data from host to device
    status = synMemCopyAsync(streamH2D, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    status = synStreamWaitEvent(streamD2H, eventStart, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to record event start1";

    status = synEventRecord(eventEnd, streamD2H);
    ASSERT_EQ(status, synSuccess) << "Failed to record event start1";

    // wait for timestamp
    uint64_t elapseTime = 0;
    for (int cnt = 0; cnt < 50; cnt++)
    {
        status = synEventElapsedTime(&elapseTime, eventStart, eventEnd);
        if (status == synSuccess) break;
        usleep(10000);
    }
    // ASSERT_EQ(status, synSuccess);
    status = synEventSynchronize(eventEnd);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";
    status = synEventElapsedTime(&elapseTime, eventStart, eventEnd);
    ASSERT_EQ(status, synSuccess);

    // test single event timestamp
    uint64_t eventTimeStart, eventTimeEnd;
    status = synEventElapsedTime(&eventTimeStart, eventStart, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to get a single event timestamp";

    status = synEventElapsedTime(&eventTimeEnd, eventEnd, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to get a single event timestamp";
    ASSERT_EQ(eventTimeEnd - eventTimeStart, elapseTime) << "single event timestamp inconsistent";

    // Copy data from device to host
    status = synMemCopyAsync(streamD2H, deviceAddress, size, (uint64_t)output, DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed to copy outputs from device memory";

    status = synEventRecord(eventEnd, streamD2H);
    ASSERT_EQ(status, synSuccess) << "Failed to record event start1";

    // Wait for everything to finish by blocking on the copy from device to host
    status = synEventSynchronize(eventEnd);
    ASSERT_EQ(status, synSuccess) << "Failed to wait for copy out stream";

    status = synEventElapsedTime(&elapseTime, eventStart, eventEnd);
    ASSERT_EQ(status, synSuccess);

    // Check one second
    status = synEventRecord(eventStart, streamD2H);
    ASSERT_EQ(status, synSuccess) << "Failed to record event start2";
    sleep(1);
    status = synEventRecord(eventEnd, streamD2H);
    ASSERT_EQ(status, synSuccess) << "Failed to record event end2";
    for (int cnt = 0; cnt < 50; cnt++)
    {
        status = synEventElapsedTime(&elapseTime, eventStart, eventEnd);
        if (status == synSuccess) break;
        usleep(1000);
    }
    ASSERT_EQ(status, synSuccess) << "Failed to get elapsed time";
    // ASSERT_FALSE((elapseTime > 1100000000) || (elapseTime < 900000000)) << "Time expected to be around 1 second"; //
    // TBD - add this check once timestamps interupts are more robust

    status = synEventElapsedTime(&elapseTime, eventStart, eventDefault);
    ASSERT_EQ(status, synInvalidArgument) << "Should fail, eventDefault doesn't collect time, second argument";

    status = synEventElapsedTime(&elapseTime, eventDefault, eventStart);
    ASSERT_EQ(status, synInvalidArgument) << "Should fail, eventDefault doesn't collect time, first argument";

    synEventHandle eventUnused;
    status = synEventCreate(&eventUnused, deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(status, synSuccess) << "Failed to create start event";

    status = synEventElapsedTime(&elapseTime, eventStart, eventUnused);
    ASSERT_EQ(status, synObjectNotInitialized) << "Should fail, event is collecting time but was not recorded";

    status = synEventElapsedTime(&elapseTime, eventUnused, eventStart);
    ASSERT_EQ(status, synObjectNotInitialized) << "Should fail, event is collecting time but was not recorded";

    // Cleanup
    status = synStreamDestroy(streamH2D);
    ASSERT_EQ(status, synSuccess) << "Failed synStreamDestroy (streamH2D)";

    status = synStreamDestroy(streamD2H);
    ASSERT_EQ(status, synSuccess) << "Failed synStreamDestroy (streamD2H)";

    status = synEventDestroy(eventStart);
    ASSERT_EQ(status, synSuccess) << "Failed to synEventDestroy eventStart";

    status = synEventDestroy(eventEnd);
    ASSERT_EQ(status, synSuccess) << "Failed to eventEnd eventStart";

    status = synEventDestroy(eventDefault);
    ASSERT_EQ(status, synSuccess) << "Failed to eventDefault eventStart";

    status = synDeviceFree(deviceId, deviceAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    status = synHostFree(deviceId, input, 0);
    ASSERT_EQ(status, synSuccess) << "Failed synHostFree input";

    status = synHostFree(deviceId, output, 0);
    ASSERT_EQ(status, synSuccess) << "Failed synHostFree output";
}

TEST_F_SYN(SynGaudiTestRecipe, testElapseTimeGaudi2Stress_ASIC_CI, {synDeviceGaudi2})
{
    TestDevice device(m_deviceType);

    const uint32_t deviceId = device.getDeviceId();

    // Create streams
    synStreamHandle streamH2D;
    synStatus       status = synStreamCreateGeneric(&streamH2D, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";

    const unsigned NUM_EVENTS = 512;
    synEventHandle event[NUM_EVENTS];
    for (int i = 0; i < NUM_EVENTS; i++)
    {
        status = synEventCreate(&event[i], deviceId, EVENT_COLLECT_TIME);
        ASSERT_EQ(status, synSuccess) << "Failed to create event " << i;
    }

    // Device-side (HBM) buffers for input and output
    const uint64_t size = 10 * 1024 * 1024;  // 10MB
    uint8_t*       input;
    status = synHostMalloc(deviceId, size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";
    memset(input, 0xEE, size);

    uint64_t deviceAddress;
    status = synDeviceMalloc(deviceId, size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    // Copy data from host to device
    status = synMemCopyAsync(streamH2D, (uint64_t)input, size, deviceAddress, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    for (int i = 0; i < NUM_EVENTS; i++)
    {
        status = synEventRecord(event[i], streamH2D);
        ASSERT_EQ(status, synSuccess) << "Failed to record event " << i;
    }

    status = synStreamSynchronize(streamH2D);
    ASSERT_EQ(status, synSuccess) << "Failed to synStreamSynchronize";

    // wait for timestamp
    uint64_t elapseTime = 0;
    for (int i = 1; i < NUM_EVENTS; i++)
    {
        status = synEventElapsedTime(&elapseTime, event[i - 1], event[i]);
        ASSERT_EQ(status, synSuccess) << "Failed synEventElapsedTime i=" << i;
    }

    // Cleanup
    status = synStreamDestroy(streamH2D);
    ASSERT_EQ(status, synSuccess) << "Failed synStreamDestroy (streamH2D)";

    for (int i = 0; i < NUM_EVENTS; i++)
    {
        status = synEventDestroy(event[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to synEventDestroy eventStart";
    }

    status = synDeviceFree(deviceId, deviceAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    status = synHostFree(deviceId, input, 0);
    ASSERT_EQ(status, synSuccess) << "Failed synHostFree input";
}
