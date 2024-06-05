#include "syn_base_test.hpp"

#include "habana_global_conf_runtime.h"
#include "hcl_api.hpp"
#include "synapse_profiler_api.hpp"

#include "runtime/common/host_to_virtual_address_mapper.hpp"

#include "runtime/common/device/device_mem_alloc.hpp"
#include "runtime/common/device/device_common.hpp"

#include "runtime/common/osal/buffer_allocator.hpp"

#include "runtime/qman/common/command_submission.hpp"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include "runtime/qman/common/data_chunk/data_chunk.hpp"
#include "runtime/qman/common/data_chunk/data_chunks_allocator.hpp"
#include "runtime/qman/common/memory_manager.hpp"
#include "runtime/qman/common/queue_compute_qman.hpp"

#include "sanity_test_c.h"
#include "synapse_api.h"
#include "syn_singleton.hpp"
#include "../infra/test_types.hpp"

#include "test_device.hpp"
#include "test_launcher.hpp"

#include "test_recipe_addf32.hpp"
#include "test_recipe_dsd_gemm.hpp"
#include "test_recipe_dynamic_split.hpp"
#include "test_recipe_gemm.hpp"
#include "test_recipe_hcl.hpp"
#include "test_recipe_nop_x_nodes.hpp"
#include "test_recipe_tpc_const_section.hpp"

#include "utils.h"

#include <gaudi/gaudi_packets.h>

#include <hlthunk.h>

#include <list>
#include <queue>

using StreamAffinityArr = std::array<uint32_t, synDeviceTypeSize>;

static void execShellCmd(const char* cmd)
{
// Enable to run hl-smi
#if 0
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        printf("Error\n");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    printf("%s\n", result.c_str());
#else
    UNUSED(cmd);
#endif
}

static void testDcMprotectSignalHandler(int sig, siginfo_t* si, void* unused)
{
    if ((uint64_t)si->si_addr != 0xf0000000)
    {
        LOG_ERR(SYN_TEST, "Got SIGSEGV at wrong address: {}, test failed!", si->si_addr);
        exit(EXIT_FAILURE);
    }

    LOG_INFO(SYN_TEST, "Got SIGSEGV at address: {}, test succeeded!", si->si_addr);
    QueueComputeQman::setDcMprotectSignalHandler();
    exit(EXIT_SUCCESS);
}

static void setDcMprotectTestSignalHandler()
{
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = testDcMprotectSignalHandler;
    if (sigaction(SIGSEGV, &sa, NULL) == -1)
    {
        LOG_ERR(SYN_TEST, "unable to call sigaction");
    }
}

static void execute(TestDevice&     rDevice,
                    TestStream&     rStream,
                    TestLauncher&   rLauncher,
                    TestRecipeBase& rRecipe,
                    TensorInitInfo  tensorInitInfo)
{
    RecipeLaunchParams recipeLaunchParam = rLauncher.createRecipeLaunchParams(rRecipe, tensorInitInfo);

    TestLauncher::execute(rStream, rRecipe, recipeLaunchParam);

    rStream.synchronize();

    rRecipe.validateResults(recipeLaunchParam.getLaunchTensorMemory());
}

class SynAPITestNoInit : public SynBaseTest
{
public:
    SynAPITestNoInit()  = default;
    ~SynAPITestNoInit() = default;

    void SetUp() override { return; };
    void TearDown() override { return; };
};

REGISTER_SUITE(SynAPITestNoInit, ALL_TEST_PACKAGES);

class SynAPITest : public SynBaseTest
, public ::testing::WithParamInterface<bool>
{
public:
    SynAPITest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
    ~SynAPITest() = default;

protected:
    // useDeserialize             - Otherwise, compile
    // useSerialize               - Serialize is executed only at the end of the call
    // clearConstSectionHostBuffer - Free const-section's host-buffer prior of serialize
    //
    // status:
    //      synSuccess         - All is good
    //      synInvalidArgument - Const-section program-data is not set (had been cleared)
    synStatus constSectionTest(bool useDeserialize, bool useSerialize, bool clearConstSectionHostBuffer);

    // When forceCompilation is false, force deserialize
    void sfgGraphTest(bool forceCompilation);

    void getDeviceStreamsRestriction(uint32_t& rStreamAffinity) const;

    typedef bool (*pfnCompareParams)(const void* newParams, const void* origParams);
    void TestSetAndGetNodeParams(TestDevice&      rDevice,
                                 TestRecipeBase&  rRecipe,
                                 synNodeId        uniqueNodeId,
                                 const void*      origParams,
                                 unsigned         paramsSize,
                                 pfnCompareParams compareParams);

    void testGenerateApiId();

    static void testGetQueueName();

    static void allocateConstSectionsHostBuffers(AllocHostBuffersVec&  constSectionsHostBuffers,
                                                 const SectionInfoVec& sectionsInfo,
                                                 SectionMemoryVec&     sectionsMemory);

    static const StreamAffinityArr s_streamAffinityArr;
};

REGISTER_SUITE(SynAPITest, ALL_TEST_PACKAGES);

const StreamAffinityArr SynAPITest::s_streamAffinityArr {0,   // synDeviceGoyaDeprecated
                                                         0,   // synDeviceGreco
                                                         2,   // synDeviceGaudi
                                                         0,   // synDeviceGaudiM
                                                         3,   // synDeviceGaudi2
                                                         3,   // synDeviceGaudi3
                                                         0,   // synDeviceEmulator
                                                         0};  // synDeviceTypeInvalid

class SynCanaryProtectionTest : public SynBaseTest
{
public:
    SynCanaryProtectionTest() { setSupportedDevices({synDeviceGaudi}); }
    ~SynCanaryProtectionTest() = default;
};

REGISTER_SUITE(SynCanaryProtectionTest, synTestPackage::DEATH);

synStatus SynAPITest::constSectionTest(bool useDeserialize, bool useSerialize, bool clearConstSectionHostBuffer)
{
    synStatus status(synSuccess);

    TestRecipeTpcConstSection recipe(m_deviceType);

    const SectionInfoVec& sectionsInfo   = recipe.getUniqueSectionsInfo();
    SectionMemoryVec&     sectionsMemory = recipe.getUniqueSectionsMemory();
    AllocHostBuffersVec   usersConstSectionsHostBuffers;
    allocateConstSectionsHostBuffers(usersConstSectionsHostBuffers, sectionsInfo, sectionsMemory);

    if (useDeserialize)
    {
        LOG_DEBUG(SYN_RT_TEST, "Deserialize");
        recipe.recipeDeserialize();
    }
    else
    {
        LOG_DEBUG(SYN_RT_TEST, "Compile only");
        // We don't want to serialize yet
        recipe.compileGraph();
    }

    std::vector<synSectionId> constSectionIdDB;
    recipe.getConstSectionsIds(constSectionIdDB);
    ASSERT_EQ(constSectionIdDB.size(), 1 /* numOfConstSections */) << "Invalid amount of const-sections";
    synSectionId constSectionId = constSectionIdDB[0];

    TestDevice device(m_deviceType);

    TestStream   stream = device.createStream();
    TestLauncher launcher(device);

    AllocDeviceBuffersVec allocDeviceBuffers;
    // The buffer itself is managed by the Recipe (created as part of recipe-creation),
    // up until the user calls to the synRecipeSectionHostBuffersClear API, and takes control over the buffer itself
    std::vector<uint64_t> gcConstSectionsHostBufferAddress;
    status = launcher.downloadConstSections(stream,
                                            recipe,
                                            constSectionIdDB,
                                            allocDeviceBuffers,
                                            gcConstSectionsHostBufferAddress);
    if (status != synSuccess)
    {
        return synInvalidArgument;
    }

    RecipeLaunchParams launchParams =
        launcher.createRecipeLaunchParams(recipe, allocDeviceBuffers, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});
    // 1. Regular execution (with the const-section)
    launcher.execute(stream, recipe, launchParams);

    stream.synchronize();

    recipe.validateResults(launchParams.getLaunchTensorMemory(), gcConstSectionsHostBufferAddress);

    // 2. Retrieve const-section info (pre clearing host-address)
    uint64_t sectionSize     = 0;
    void*    hostSectionData = nullptr;
    recipe.getConstSectionInfo(sectionSize, hostSectionData, constSectionId);

    if (clearConstSectionHostBuffer)
    {
        recipe.clearConstSectionHostBuffers(constSectionIdDB);

        delete[](char*) hostSectionData;
    }

    if (useSerialize)
    {
        LOG_DEBUG(SYN_RT_TEST, "Serialize");
        recipe.recipeSerialize();
    }

    return synSuccess;
}

void SynAPITest::sfgGraphTest(bool forceCompilation)
{
    synStatus status(synSuccess);

    // 1. Generate recipe
    TestRecipeHcl recipe(m_deviceType, true /* isSfgGraph */);
    if (forceCompilation)
    {
        recipe.compileGraph();
        recipe.recipeSerialize();
    }
    else
    {
        recipe.recipeDeserialize();
    }

    // 2. Validate amount of external tensors
    uint64_t amountOfExternalTensors(0);
    recipe.retrieveAmountOfExternalTensors(amountOfExternalTensors);
    ASSERT_EQ(amountOfExternalTensors, 1);

    // 3. Validate execution-order tensors-id
    const std::vector<uint64_t>& orderedTensorsId = recipe.retrieveOrderedTensorsId();
    ASSERT_EQ(orderedTensorsId.size(), 1) << "Mismatch in amount of external-tensors";
    ASSERT_EQ(orderedTensorsId[0], 1) << "Unexpected tensor-id for the sole external-tensor ";
    uint64_t externalTensorId = orderedTensorsId[0];

    // 4. Create device, launcher, streams and events
    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);
    //
    TestStream computeStream     = device.createStream();
    TestStream sfgObserverStream = device.createStream();
    //
    std::vector<TestEvent> testEvents;
    testEvents.emplace_back(std::move(device.createEvent(EVENT_COLLECT_TIME /* flags */)));
    TestEvent& sfgTestEvent = testEvents[0];
    //
    TestEvent dummyEventHandle(std::move(device.createEvent(EVENT_COLLECT_TIME /* flags */)));
    computeStream.eventRecord(dummyEventHandle);

    // 5. Create launch params *per launch*
    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    // 6. Map event to external-tensor
    const SynLaunchTensorInfoVec& synLaunchTensorsInfoDB = recipeLaunchParams.getSynLaunchTensorInfoVec();
    recipe.mapEventToExternalTensor(sfgTestEvent, &synLaunchTensorsInfoDB[externalTensorId]);

    // 7. Check that elapsed time fails, as the sfg-event was not yet recorded
    uint64_t dummyTimelapse(0);
    status = synEventElapsedTime(&dummyTimelapse, sfgTestEvent, dummyEventHandle);
    ASSERT_NE(status, synSuccess);

    // 8. Launch
    TestLauncher::launchWithExternalEvents(computeStream, recipe, recipeLaunchParams, testEvents);

    // 9. Wait over SFG event and synchronize
    sfgObserverStream.eventWait(sfgTestEvent, 0 /* flags */);
    sfgObserverStream.synchronize();

    // 10. Check that elapsed time fails, as the sfg-event does not trigger interrupt
    // Hence, it cannot result be collecting the event's completion-time
    status = synEventElapsedTime(&dummyTimelapse, dummyEventHandle, sfgTestEvent);
    ASSERT_NE(status, synSuccess);
}

void SynAPITest::getDeviceStreamsRestriction(uint32_t& rStreamAffinity) const
{
    ASSERT_LT(m_deviceType, synDeviceTypeSize) << "Unknown device type " << m_deviceType;
    rStreamAffinity = s_streamAffinityArr[m_deviceType];
}

void SynAPITest::TestSetAndGetNodeParams(TestDevice&      rDevice,
                                         TestRecipeBase&  rRecipe,
                                         synNodeId        uniqueNodeId,
                                         const void*      origParams,
                                         unsigned         paramsSize,
                                         pfnCompareParams compareParams)
{
    synStatus status(synSuccess);

    synGraphHandle graphHandle = rRecipe.getGraphHandle();

    // Set node params
    status = synNodeSetUserParams(graphHandle, uniqueNodeId, origParams, paramsSize);
    ASSERT_EQ(status, synSuccess) << "Failed on set user params for Node " << uniqueNodeId;

    // Get node params' buffer-size
    unsigned bufferSize;
    status = synNodeGetUserParams(graphHandle, uniqueNodeId, nullptr, &bufferSize);
    ASSERT_EQ(status, synSuccess) << "Failed on get params for Node id " << uniqueNodeId;
    ASSERT_EQ(bufferSize, paramsSize) << "Incorrect size of params for Node id " << uniqueNodeId;

    // Allocate buffer for rertieving params
    TestHostBufferMalloc allocBufferParams = rDevice.allocateHostBuffer(paramsSize, 0 /* flags */);
    void*                bufferParams      = allocBufferParams.getBuffer();

    // Retrieve node's params
    status = synNodeGetUserParams(graphHandle, uniqueNodeId, bufferParams, &bufferSize);
    ASSERT_EQ(status, synSuccess) << "Failed on get params for Node id " << uniqueNodeId;
    ASSERT_TRUE(compareParams(bufferParams, origParams)) << "Incorrect value of params for Node id " << uniqueNodeId;

    // Test faulty usage of synNodeSetUserParams/synNodeGetUserParams
    //
    // 1. nullptr graph handle (Set method)
    status = synNodeSetUserParams(nullptr, uniqueNodeId, origParams, paramsSize);
    ASSERT_NE(status, synSuccess) << "set params succeeded for null graph handle";
    //
    // 2. nullptr graph handle (Get method)
    status = synNodeGetUserParams(nullptr, uniqueNodeId, bufferParams, &bufferSize);
    ASSERT_NE(status, synSuccess) << "get params succeeded for null graph handle";
    //
    // 3. nullptr buffer-size (Get method)
    status = synNodeGetUserParams(graphHandle, uniqueNodeId, bufferParams, nullptr);
    ASSERT_NE(status, synSuccess) << "get params succeeded for null buffer size";
    //
    // 4. Invalid buffer-size (Get method)
    unsigned wrongBufferSize = paramsSize + 1;
    status                   = synNodeGetUserParams(graphHandle, uniqueNodeId, bufferParams, &wrongBufferSize);
    ASSERT_NE(status, synSuccess) << "get params succeeded for incompatible buffer size and buffer params";
    //
    // 5. nullptr params (Set method)
    if (paramsSize != 0)
    {
        status = synNodeSetUserParams(graphHandle, uniqueNodeId, nullptr, paramsSize);
        ASSERT_NE(status, synSuccess) << "set params succeeded for null buffer params and non-zero param size";
    }
    //
    // 6. Invalid (zero) buffer-size (Set method)
    if (origParams != nullptr)
    {
        status = synNodeSetUserParams(graphHandle, uniqueNodeId, origParams, 0);
        ASSERT_NE(status, synSuccess) << "set params succeeded for param size 0 and non-null buffer params";
    }

    rRecipe.graphCompile();
}

void SynAPITest::testGenerateApiId()
{
    TestDevice device(m_deviceType);

    uint8_t   apiIdLast;
    synStatus status = synGenerateApiId(apiIdLast);
    ASSERT_EQ(status, synSuccess);

    for (unsigned index = 0; index < 64; index++)
    {
        uint8_t apiIdCurrent;
        status = synGenerateApiId(apiIdCurrent);
        ASSERT_EQ(status, synSuccess);
        ASSERT_EQ(((apiIdLast + 1) & DeviceCommon::s_apiIdMask), (apiIdCurrent & DeviceCommon::s_apiIdMask));
        apiIdLast = apiIdCurrent;
    }
}

void SynAPITest::testGetQueueName()
{
    const uint8_t queueDir   = (uint8_t)PdmaDirCtx::UP;
    bool          isValidDir = IS_VALID_QUEUE_DIR(queueDir);
    ASSERT_TRUE(isValidDir);
    const uint8_t queueType   = (uint8_t)INTERNAL_STREAM_TYPE_DMA_UP;
    bool          isValidType = IS_VALID_QUEUE_TYPE(queueType);
    ASSERT_TRUE(isValidType);
    const uint8_t queueIndex   = 0;
    bool          isValidIndex = IS_VALID_QUEUE_INDEX(queueIndex);
    ASSERT_TRUE(isValidIndex);
    std::string nameActual(GET_QUEUE_NAME(queueDir, queueType, queueIndex));
    std::string nameExpected("D2H pdma_rx 0");
    ASSERT_EQ(nameActual, nameExpected);
}

void SynAPITest::allocateConstSectionsHostBuffers(AllocHostBuffersVec&  constSectionsHostBuffers,
                                                  const SectionInfoVec& sectionsInfo,
                                                  SectionMemoryVec&     sectionsMemory)
{
    sectionsMemory.clear();

    for (auto singleSectionInfo : sectionsInfo)
    {
        if (!singleSectionInfo.m_isConstSection)
        {
            continue;
        }

        TestHostBufferAlloc hostBuffer(singleSectionInfo.m_sectionSize, 0 /* flags */);

        sectionsMemory.push_back(hostBuffer.getBuffer());
        constSectionsHostBuffers.push_back(std::move(hostBuffer));
    }
}

// ## 1) Pre-initialize tests

TEST_F_SYN(SynAPITestNoInit, get_version)
{
    const int maxLen = 256;
    char      synVersion[maxLen];
    ASSERT_EQ(synDriverGetVersion(synVersion, maxLen), synSuccess) << "Failed to get synapse's driver version";
}

// ## 2) Pre-acquire tests

TEST_F_SYN(SynAPITest, global_config)
{
    // access to synapse and hcl config values
    const unsigned bufferSize = 128;
    char           buffer[bufferSize];
    ASSERT_EQ(synConfigurationGet("SYNAPSE_DATA_TYPE_SELECTION", buffer, bufferSize), synSuccess);
    ASSERT_NE(synConfigurationGet("SYNAPSE_DATA_TYPE_SELECTION_987987", buffer, bufferSize), synSuccess);

    ASSERT_EQ(synConfigurationSet("SYNAPSE_DATA_TYPE_SELECTION", "true"), synSuccess);
    ASSERT_NE(synConfigurationSet("SYNAPSE_DATA_TYPE_SELECTION_987987", "true"), synSuccess);

    // access to hcl config
    ASSERT_EQ(synConfigurationGet("HCL_IMB_SIZE", buffer, bufferSize), synSuccess);
    ASSERT_NE(synConfigurationGet("HCL_IMB_SIZE_09uoijh", buffer, bufferSize), synSuccess);

    ASSERT_EQ(synConfigurationSet("HCL_IMB_SIZE", "512KB"), synSuccess);
    ASSERT_NE(synConfigurationSet("HCL_IMB_SIZE_09uoijh", "512KB"), synSuccess);
}

// ## 3) Acquire tests

TEST_F_SYN(SynAPITest, device_acquire)
{
    synStatus     status(synSuccess);
    unsigned      deviceId;
    synDeviceInfo deviceInfo[2];

    // Acquire
    status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "synDeviceAcquireByDeviceType failed";

    // Validation: fd & device-type
    deviceInfo[0].fd = UINT32_MAX;
    status           = synDeviceGetInfo(deviceId, &deviceInfo[0]);
    ASSERT_EQ(status, synSuccess) << "Failed to Get device info (" << status << ")";
    ASSERT_EQ(deviceInfo[0].deviceType, m_deviceType);
    ASSERT_LT(deviceInfo[0].fd, UINT32_MAX) << "Failed to set fd";

    // Release
    status = synDeviceRelease(deviceId);
    EXPECT_EQ(synSuccess, status) << "Failed to release device";

    // Invalid acquire call (invalid device-type)
    status =
        synDeviceAcquireByDeviceType(&deviceId, (m_deviceType == synDeviceGaudi2) ? synDeviceGaudi : synDeviceGaudi2);
    EXPECT_EQ(synDeviceTypeMismatch, status) << "Test support only (" << m_deviceType << ")";

    // Re-acquire
    status = synDeviceAcquire(&deviceId, nullptr);
    EXPECT_EQ(synSuccess, status) << "Failed to acquire Any device";

    // Validation: fd & device-type
    deviceInfo[1].fd = UINT32_MAX;
    status           = synDeviceGetInfo(deviceId, &deviceInfo[1]);
    ASSERT_EQ(status, synSuccess) << "Failed to Get device info (" << status << ")";
    ASSERT_EQ(deviceInfo[1].deviceType, m_deviceType);
    ASSERT_LT(deviceInfo[1].fd, UINT32_MAX) << "Failed to set fd";
}

TEST_F_SYN(SynAPITest, device_info)
{
    synStatus       status(synSuccess);
    unsigned        deviceId;
    synDeviceInfo   deviceInfo;
    synDeviceInfoV2 deviceInfoV2;

    // Acquire
    status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "synDeviceAcquireByDeviceType failed (1)";

    // DeviceGetInfo
    status = synDeviceGetInfo(deviceId, &deviceInfo);
    ASSERT_EQ(status, synSuccess) << "Failed to Get device info (V1): " << status;
    status = synDeviceGetInfoV2(deviceId, &deviceInfoV2);
    ASSERT_EQ(status, synSuccess) << "Failed to Get device info (V2): " << status;

    // Compare between the old and new synDeviceInfo
    ASSERT_EQ(deviceInfo.sramBaseAddress, deviceInfoV2.sramBaseAddress) << "Invalid sramBaseAddress";
    ASSERT_EQ(deviceInfo.dramBaseAddress, deviceInfoV2.dramBaseAddress) << "Invalid dramBaseAddress";
    ASSERT_EQ(deviceInfo.sramSize,        deviceInfoV2.sramSize)        << "Invalid sramSize";
    ASSERT_EQ(deviceInfo.dramSize,        deviceInfoV2.dramSize)        << "Invalid dramSize";
    ASSERT_EQ(deviceInfo.tpcEnabledMask,  deviceInfoV2.tpcEnabledMask)  << "Invalid tpcEnabledMask";
    ASSERT_EQ(deviceInfo.dramEnabled,     deviceInfoV2.dramEnabled)     << "Invalid dramEnabled";
    ASSERT_EQ(deviceInfo.deviceId,        deviceInfoV2.deviceId)        << "Invalid deviceId";
    ASSERT_EQ(deviceInfo.fd,              deviceInfoV2.fd)              << "Invalid fd";
    ASSERT_EQ(deviceInfo.deviceType,      deviceInfoV2.deviceType)      << "Invalid deviceType";

    // Device-Index (check index is valid)
    ASSERT_LT(deviceInfoV2.deviceIndex, 8) << "Invalid device-index";
}

TEST_F_SYN(SynAPITest, device_acquire_c)
{
    TestDevice device(m_deviceType);

    synStatus status = run_sanity_test_c(device.getDeviceId(), m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed";
}

// ## 4) Retrieve graph related information tests

TEST_F_SYN(SynAPITest, set_graph_attributes)
{
    synGraphHandle graphHandle;
    synStatus      status = synGraphCreate(&graphHandle, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create graph";

    uint32_t             size          = 2;
    double               backoffFactor = 1.3;
    synGraphAttributeVal backoffFactorVal {.dAttrVal = backoffFactor};
    synGraphAttributeVal values[]     = {true, backoffFactorVal};
    synGraphAttributeVal new_values[] = {true, 0.0};
    synGraphAttribute    att[]        = {GRAPH_ATTRIBUTE_INFERENCE, GRAPH_ATTRIBUTE_BACKOFF_FACTOR};

    // 0. Get default values
    status = synGraphGetAttributes(graphHandle, att, new_values, size);
    ASSERT_EQ(status, synSuccess) << "Failed to get graph default attributes";
    ASSERT_EQ(new_values[0].iAttrVal, false) << "Got wrong graph default attribute - Inference Mode";
    ASSERT_EQ(new_values[1].dAttrVal, 1.0) << "Got wrong graph default attribute - Backoff Factor";

    // 1. Set followed by get
    status = synGraphSetAttributes(graphHandle, att, values, size);
    ASSERT_EQ(status, synSuccess) << "Failed to set graph attributes";
    //
    status = synGraphGetAttributes(graphHandle, att, new_values, size);
    ASSERT_EQ(status, synSuccess) << "Failed to set graph attributes";
    ASSERT_EQ(new_values[0].iAttrVal, true) << "Got wrong graph attributes - Inference Mode";
    ASSERT_EQ(new_values[1].dAttrVal, backoffFactor) << "Got wrong graph attributes - Backoff Factor";

    // 2. Invalid size (zero) set & get
    size   = 0;
    status = synGraphSetAttributes(graphHandle, att, values, size);
    ASSERT_EQ(status, synFail) << "Success on a negative case (size 0) to set graph attributes";
    //
    status = synGraphGetAttributes(graphHandle, att, new_values, size);
    ASSERT_EQ(status, synFail) << "Success on a negative case (size 0) to get graph attributes";

    // 3. Invalid size (max+1) set & get
    size   = GRAPH_ATTRIBUTE_MAX + 1;
    status = synGraphSetAttributes(graphHandle, att, values, size);
    ASSERT_EQ(status, synFail) << "Success on a negative case (size too big) to set graph attributes";
    //
    status = synGraphGetAttributes(graphHandle, att, new_values, size);
    ASSERT_EQ(status, synFail) << "Success on a negative case (size too big) to get graph attributes";

    // 4. Verify synGraphSetAttributes fail to change graph mode if nodes already added to the graph
    //
    //      A. Get attributes - Pre nodes' creation
    TestRecipeNopXNodes testRecipe(m_deviceType);
    testRecipe.createGraphHandle();
    size               = 1;
    values[0].iAttrVal = false;
    status             = synGraphSetAttributes(testRecipe.getGraphHandle(), att, values, size);
    ASSERT_EQ(status, synSuccess) << "Failed to set graph attributes before node add";
    //
    //      B. Graph creation
    testRecipe.graphCreation();
    //
    //      C. Get attributes - Post nodes' creation
    values[0].iAttrVal = true;
    status             = synGraphSetAttributes(testRecipe.getGraphHandle(), att, values, size);
    ASSERT_EQ(status, synFail) << "Success to set graph mode after adding a node to the graph";

    // 5. Invalid backoff factor value set & get
    size               = 2;
    values[1].dAttrVal = 0.0;
    status             = synGraphSetAttributes(graphHandle, att, values, size);
    ASSERT_EQ(status, synFail) << "Success on an invalid case to set graph backoff factor = 0";
    values[1].dAttrVal = -1.2;
    status             = synGraphSetAttributes(graphHandle, att, values, size);
    ASSERT_EQ(status, synFail) << "Success on an invalid case to set graph backoff factor < 0";
    //
    new_values[1].dAttrVal = 0.0;
    status                 = synGraphGetAttributes(graphHandle, att, new_values, size);
    ASSERT_EQ(status, synSuccess) << "Failed to get graph attributes after invalid attempt to set backoff factor";
    // value should stay the same from the last successful set.
    ASSERT_EQ(new_values[1].dAttrVal, 1.3) << "Got wrong graph attributes - Backoff Factor";
}

// ## 5) Retrieve device related information tests

TEST_F_SYN(SynAPITest, hl_smi_device_mem, {synDeviceGaudi})
{
    TestDevice  device(m_deviceType);
    synDeviceId deviceId         = device.getDeviceId();
    uint64_t    size             = 16 * 1024 * 1024;
    uint32_t    flags            = 0;
    uint64_t    requestedAddress = 0;

    uint64_t  buffer = 0;
    synStatus status = synSuccess;

    std::vector<std::pair<uint64_t, uint64_t>> bufferAndSize;

    std::string cmd = R"(hl-smi -L|grep 'Memory Usage' -A 3|grep ':'|cut -d":" -f2|cut -d" " -f2)";

    synDeviceInfo deviceInfo;
    device.getDeviceInfo(deviceInfo);

    execShellCmd(cmd.c_str());
    uint64_t freeMemAtStart = 0;
    uint64_t total          = 0;
    device.getDeviceMemoryInfo(freeMemAtStart, total);

    uint64_t preAllocatedMem = total - freeMemAtStart;
    unsigned padding         = calcPaddingSize(preAllocatedMem, ManagedBufferAllocator::m_defaultAlignment);
    uint64_t dramBaseAddress = deviceInfo.dramBaseAddress + preAllocatedMem + padding;

    unsigned operationIndex = 0;

    hlthunk_dram_usage_info usage;
    hlthunk_get_dram_usage(deviceInfo.fd, &usage);

    uint64_t allocated = usage.ctx_dram_mem;  // This is the alloctaed size we start the test with

    // Allocate the dram-base
    requestedAddress = dramBaseAddress;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_EQ(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    bufferAndSize.push_back({buffer, size});

    allocated += size;
    hlthunk_get_dram_usage(deviceInfo.fd, &usage);
    ASSERT_EQ(allocated, usage.ctx_dram_mem);

    execShellCmd(cmd.c_str());

    synDeviceMalloc(deviceId, size * 2, 0, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    operationIndex++;
    bufferAndSize.push_back({buffer, size * 2});

    allocated += size * 2;
    hlthunk_get_dram_usage(deviceInfo.fd, &usage);
    ASSERT_EQ(allocated, usage.ctx_dram_mem);

    execShellCmd(cmd.c_str());

    for (auto singleBuffer : bufferAndSize)
    {
        status = synDeviceFree(deviceId, singleBuffer.first, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to free (operationIndex " << operationIndex << ")";
        operationIndex++;
        allocated -= singleBuffer.second;
        hlthunk_get_dram_usage(deviceInfo.fd, &usage);
        ASSERT_EQ(allocated, usage.ctx_dram_mem);
        execShellCmd(cmd.c_str());
    }
}

TEST_F_SYN(SynAPITest, get_attribute)
{
    TestDevice device(m_deviceType);

    const synDeviceAttribute attributes[] = {DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS,
                                             DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS,
                                             DEVICE_ATTRIBUTE_SRAM_SIZE,
                                             DEVICE_ATTRIBUTE_DRAM_SIZE,
                                             // DEVICE_ATTRIBUTE_DRAM_FREE_SIZE attribute require device
                                             DEVICE_ATTRIBUTE_TPC_ENABLED_MASK,
                                             DEVICE_ATTRIBUTE_DRAM_ENABLED,
                                             DEVICE_ATTRIBUTE_DEVICE_TYPE,
                                             DEVICE_ATTRIBUTE_CLK_RATE,
                                             DEVICE_ATTRIBUTE_MAX_RMW_SIZE,
                                             // DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE attribute require device
                                             DEVICE_ATTRIBUTE_ADDRESS_ALIGNMENT_SIZE,
                                             DEVICE_ATTRIBUTE_MAX_DIMS};

    const size_t nofAttributes         = sizeof(attributes) / sizeof(attributes[0]);
    uint64_t     values[nofAttributes] = {0};
    synStatus    res                   = synDeviceGetAttribute(values, attributes, nofAttributes, device.getDeviceId());
    ASSERT_EQ(synSuccess, res);

    uint64_t  constValues[nofAttributes] = {0};
    synStatus deviceTypeRes = synDeviceTypeGetAttribute(constValues, attributes, nofAttributes, m_deviceType);
    ASSERT_EQ(synSuccess, deviceTypeRes);

    for (size_t attrIdx = 0; attrIdx < nofAttributes; attrIdx++)
    {
        LOG_INFO(SYN_TEST,
                 "attribute index {} device value {} const deviceType value {}",
                 attrIdx,
                 values[attrIdx],
                 constValues[attrIdx]);
        ASSERT_EQ(values[attrIdx], constValues[attrIdx])
            << "attribute index " << attrIdx << " device value " << values[attrIdx] << " const deviceType value "
            << constValues[attrIdx];
    }
}

TEST_F_SYN(SynAPITest, get_attribute_by_module_ids)
{
    // The following attributes requires a device allocation:
    //      DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS
    //      DEVICE_ATTRIBUTE_SRAM_SIZE
    //      DEVICE_ATTRIBUTE_DRAM_SIZE
    TestDevice device(m_deviceType);

    const synDeviceAttribute attributes[] = {DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS,
                                             DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS,
                                             DEVICE_ATTRIBUTE_SRAM_SIZE,
                                             DEVICE_ATTRIBUTE_DRAM_SIZE,
                                             // DEVICE_ATTRIBUTE_DRAM_FREE_SIZE attribute requires device
                                             DEVICE_ATTRIBUTE_TPC_ENABLED_MASK,
                                             DEVICE_ATTRIBUTE_DRAM_ENABLED,
                                             DEVICE_ATTRIBUTE_DEVICE_TYPE,
                                             //  DEVICE_ATTRIBUTE_CLK_RATE, attribute requires device
                                             DEVICE_ATTRIBUTE_MAX_RMW_SIZE,
                                             // DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE attribute requires device
                                             DEVICE_ATTRIBUTE_ADDRESS_ALIGNMENT_SIZE,
                                             DEVICE_ATTRIBUTE_MAX_DIMS};

    const size_t nofAttributes = sizeof(attributes) / sizeof(attributes[0]);

    uint64_t  constValues[nofAttributes] = {0};
    synStatus status = synDeviceTypeGetAttribute(constValues, attributes, nofAttributes, m_deviceType);
    ASSERT_EQ(synSuccess, status);

    uint32_t numOfDevices = 0;
    status                = synDeviceGetCount(&numOfDevices);
    ASSERT_EQ(synSuccess, status);

    LOG_INFO(SYN_TEST, "Num of devices: {}", numOfDevices);
    uint32_t deviceModuleIds[MAX_NUM_OF_DEVICES_PER_HOST];
    uint32_t numOfModuleIds = numOfDevices;
    status                  = synDeviceGetModuleIDs(deviceModuleIds, &numOfModuleIds);
    ASSERT_EQ(synSuccess, status);

    LOG_INFO(SYN_TEST, "Num of module IDs: {}\n", numOfModuleIds);
    for (uint32_t moduleIndex = 0; moduleIndex < numOfModuleIds; moduleIndex++)
    {
        uint64_t values[nofAttributes] = {0};

        status = synDeviceGetAttributeByModuleId(values, attributes, nofAttributes, deviceModuleIds[moduleIndex]);
        ASSERT_EQ(synSuccess, status);

        LOG_INFO(SYN_TEST, "Attributes of module IDs: {}", deviceModuleIds[moduleIndex]);
        for (size_t attrIdx = 0; attrIdx < nofAttributes; attrIdx++)
        {
            LOG_INFO(SYN_TEST,
                     "attribute index {} device value {} const deviceType value {}{}",
                     attrIdx,
                     values[attrIdx],
                     constValues[attrIdx],
                     (attrIdx == (nofAttributes - 1)) ? "\n" : "");

            ASSERT_EQ(values[attrIdx], constValues[attrIdx])
                << "attribute index " << attrIdx << " device value " << values[attrIdx] << " const deviceType value "
                << constValues[attrIdx];
        }
    }
}

TEST_F_SYN(SynAPITest, get_attribute_tpc)
{
    TestDevice               device(m_deviceType);
    const synDeviceAttribute attributes[]          = {DEVICE_ATTRIBUTE_TPC_ENABLED_MASK};
    const size_t             nofAttributes         = sizeof(attributes) / sizeof(attributes[0]);
    uint64_t                 values[nofAttributes] = {0};
    synStatus                res = synDeviceGetAttribute(values, attributes, nofAttributes, device.getDeviceId());
    ASSERT_EQ(synSuccess, res);

    if (m_deviceType == synDeviceGaudi3)
    {
        if (GCFG_GAUDI3_SINGLE_DIE_CHIP.value())
        {
            ASSERT_EQ(values[0], 0xffffffff);
        }
        else
        {
            ASSERT_EQ(values[0], 0xffffffffffffffff);
        }
    }
    else if (m_deviceType == synDeviceGaudi2)
    {
        ASSERT_EQ(values[0], 0xffffff);
    }
    else
    {
        ASSERT_EQ(values[0], 0xff);
    }
}

TEST_F_SYN(SynAPITest,
           max_rmw_size_should_retrieved_via_attribute_query,
           {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    TestDevice               device(m_deviceType);
    const synDeviceAttribute attributes[]  = {DEVICE_ATTRIBUTE_MAX_RMW_SIZE};
    const size_t             nofAttributes = sizeof(attributes) / sizeof(attributes[0]);

    uint64_t values[nofAttributes] = {0};

    synStatus res = synDeviceGetAttribute(values, attributes, nofAttributes, device.getDeviceId());
    ASSERT_EQ(synSuccess, res);
    auto expMaxRMW = 16ull * 1024 * 1024;
    ASSERT_EQ(expMaxRMW, values[0]);

    // TODO need to add API to shim layer
    uint64_t constValues[nofAttributes] = {0};
    res = synDeviceTypeGetAttribute(constValues, attributes, nofAttributes, m_deviceType);
    ASSERT_EQ(synSuccess, res);
    ASSERT_EQ(expMaxRMW, constValues[0]);
}

// ## 6) Memory allocation tests

TEST_F_SYN(SynAPITest, host_malloc)
{
    synStatus  status(synSuccess);
    TestDevice device(m_deviceType);

    void* buffer = nullptr;
    status       = synHostMalloc(device.getDeviceId(), 128 /* size */, 0 /* flags */, &buffer);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_TEST, "Could be that no huge pages defined in the system.");
        LOG_ERR(SYN_TEST, "Run the following - echo 2 | sudo tee /proc/sys/vm/nr_hugepages");
        ASSERT_EQ(true, false) << "Failed to allocate buffer";
    }

    status = synHostFree(device.getDeviceId(), buffer, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to free buffer";
}

TEST_F_SYN(SynAPITest, device_malloc)
{
    synStatus  status(synSuccess);
    TestDevice device(m_deviceType);

    uint64_t inputDeviceBuffer = 0;
    status =
        synDeviceMalloc(device.getDeviceId(), 100 /* size */, 0 /* reqAddress */, 0 /* flags */, &inputDeviceBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate buffer";

    status = synDeviceFree(device.getDeviceId(), inputDeviceBuffer, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to free buffer";
}

TEST_F_SYN(SynAPITest, map_buffer_twice)
{
    unsigned elemSize = 1024;
    char*    buffer   = new char[elemSize];

    TestDevice device(m_deviceType);

    // mapping isn't exists
    auto status = synHostUnmap(device.getDeviceId(), buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to unmap out vector to device";

    status = synHostMap(device.getDeviceId(), elemSize * sizeof(char), buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to map out vector to device";

    status = synHostUnmap(device.getDeviceId(), buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to unmap out vector to device";

    // second mapping already exists
    status = synHostMap(device.getDeviceId(), elemSize * sizeof(char), buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to map out vector to device";

    status = synHostMap(device.getDeviceId(), elemSize * sizeof(char), buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to map out vector to device";

    status = synHostUnmap(device.getDeviceId(), buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to unmap out vec to device";

    status = synHostUnmap(device.getDeviceId(), buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to unmap out vec to device";

    delete[] buffer;
}

TEST_F_SYN(SynAPITest, allocation_with_requested_address, {synDeviceGaudi})
{
    uint64_t alignmentSize    = 128;
    uint64_t size             = alignmentSize;
    uint64_t alignmentDiff    = alignmentSize / 2;
    uint32_t flags            = 0;
    uint64_t requestedAddress = 0;

    uint64_t  buffer = 0;
    synStatus status = synSuccess;

    std::list<uint64_t> buffers;

    TestDevice    device(m_deviceType);
    synDeviceId   deviceId = device.getDeviceId();
    synDeviceInfo deviceInfo;
    device.getDeviceInfo(deviceInfo);

    uint64_t freeMemAtStart = 0;
    uint64_t total          = 0;
    device.getDeviceMemoryInfo(freeMemAtStart, total);

    uint64_t preAllocatedMem = total - freeMemAtStart;
    unsigned padding         = calcPaddingSize(preAllocatedMem, alignmentSize);

    uint64_t dramBaseAddress = deviceInfo.dramBaseAddress + preAllocatedMem + padding;
    uint64_t dramSize        = deviceInfo.dramSize - preAllocatedMem - padding;

    // For allocating a range that does not reach any of the DRAM boundaries
    uint64_t specificRangeBaseAddress = dramBaseAddress + (alignmentSize * 1000);

    unsigned operationIndex = 0;

    // Allocate the dram-base
    requestedAddress = dramBaseAddress;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_EQ(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    buffers.push_back(buffer);
    // Buffers: [dramBaseAddress, dramBaseAddress + size]

    // Retry allocating dram-base (and free whatever had been allocated)
    requestedAddress = dramBaseAddress;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_NE(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    // Buffers: [dramBaseAddress, dramBaseAddress + size], [other, other + size]
    //
    status = synDeviceFree(deviceId, buffer, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to free (operationIndex " << operationIndex << ")";
    operationIndex++;
    // Buffers: [dramBaseAddress, dramBaseAddress + size]

    // Free dram-base
    requestedAddress = dramBaseAddress;
    status           = synDeviceFree(deviceId, requestedAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to free (operationIndex " << operationIndex << ")";
    operationIndex++;
    buffers.remove(requestedAddress);
    // Buffers: []

    // Re-allocate dram-base
    requestedAddress = dramBaseAddress;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_EQ(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    buffers.push_back(buffer);
    // Buffers: [dramBaseAddress, dramBaseAddress + size]

    // Try to allocate non-valid address (and free whatever had been allocated)
    requestedAddress = dramBaseAddress + dramSize - alignmentDiff;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_NE(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    // Buffers: [dramBaseAddress, dramBaseAddress + size], [other, other + size]
    status = synDeviceFree(deviceId, buffer, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to free (operationIndex " << operationIndex << ")";
    operationIndex++;
    // Buffers: [dramBaseAddress, dramBaseAddress + size]

    // Allocate a chunk of memory at the end of DRAM area
    requestedAddress = dramBaseAddress + dramSize - size;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_EQ(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    buffers.push_back(buffer);
    // Buffers: [dramBaseAddress, dramBaseAddress + size], [dramEndAddress - size, dramEndAddress]

    // Allocate a chunk of memory at the middle of the DRAM area
    requestedAddress = specificRangeBaseAddress;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_EQ(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    buffers.push_back(buffer);
    // Buffers: [dramBaseAddress, dramBaseAddress + size], [dramEndAddress - size, dramEndAddress]
    //          [specificRangeBaseAddress, specificRangeBaseAddress + size]

    // Allocatong a buffer just before the "specificRangeBaseAddress" base address
    requestedAddress = specificRangeBaseAddress - size;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_EQ(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    buffers.push_back(buffer);
    // Buffers: [dramBaseAddress, dramBaseAddress + size], [dramEndAddress - size, dramEndAddress]
    //          [specificRangeBaseAddress - size, specificRangeBaseAddress]
    //          [specificRangeBaseAddress, specificRangeBaseAddress + size]

    // Allocatong a buffer just after the "specificRangeBaseAddress" base address
    requestedAddress = specificRangeBaseAddress + size;
    status           = synDeviceMalloc(deviceId, size, requestedAddress, flags, &buffer);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate (operationIndex " << operationIndex << ")";
    ASSERT_EQ(buffer, requestedAddress) << "Wrong allocation address (operationIndex " << operationIndex << ")";
    operationIndex++;
    buffers.push_back(buffer);
    // Buffers: [dramBaseAddress, dramBaseAddress + size], [dramEndAddress - size, dramEndAddress]
    //          [specificRangeBaseAddress - size, specificRangeBaseAddress]
    //          [specificRangeBaseAddress, specificRangeBaseAddress + size]
    //          [specificRangeBaseAddress + size, specificRangeBaseAddress + size]

    // Free all buffers
    for (auto singleBuffer : buffers)
    {
        status = synDeviceFree(deviceId, singleBuffer, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to free (operationIndex " << operationIndex << ")";
        operationIndex++;
    }
}

TEST_F_SYN(SynAPITest, check_mem_info_Api, {synDeviceGaudi2, synDeviceGaudi3})
{
    TestDevice device(m_deviceType);

    synDeviceInfo deviceInfo;
    device.getDeviceInfo(deviceInfo);

    uint64_t freeAfterInit;
    uint64_t totalAfterInit;
    device.getDeviceMemoryInfo(freeAfterInit, totalAfterInit);

    uint64_t preAllocatedMem = totalAfterInit - freeAfterInit;
    unsigned padding         = calcPaddingSize(preAllocatedMem, ManagedBufferAllocator::m_defaultAlignment);
    ASSERT_EQ(totalAfterInit, deviceInfo.dramSize)
        << "total memory is not equal " << totalAfterInit << " " << deviceInfo.dramSize;
    ASSERT_EQ(freeAfterInit, totalAfterInit - preAllocatedMem - padding)
        << "free memory is not equal " << freeAfterInit << " " << totalAfterInit - preAllocatedMem - padding;

    // Allocate buffer and check memory status
    {
        const uint64_t numOfBytesToAllocate = 0x1000;
        const uint64_t freeExpected         = freeAfterInit - numOfBytesToAllocate;

        TestDeviceBufferAlloc deviceBuffer = device.allocateDeviceBuffer(numOfBytesToAllocate, 0 /* flags */);

        uint64_t freeAfterAllocation;
        uint64_t totalAfterAllocation;
        device.getDeviceMemoryInfo(freeAfterAllocation, totalAfterAllocation);
        ASSERT_EQ(totalAfterAllocation, totalAfterInit)
            << "total memory is not equal " << totalAfterAllocation << " " << totalAfterInit;
        ASSERT_EQ(freeAfterAllocation, freeExpected)
            << "free memory is not equal " << freeAfterAllocation << " " << freeExpected;
    }

    // Buffer is now free => check memory status
    {
        uint64_t freeAfterRelease;
        uint64_t totalAfterRelease;
        device.getDeviceMemoryInfo(freeAfterRelease, totalAfterRelease);
        ASSERT_EQ(totalAfterRelease, totalAfterInit)
            << "total memory is not equal " << totalAfterRelease << " " << totalAfterInit;
        ASSERT_EQ(freeAfterRelease, freeAfterInit)
            << "free memory is not equal " << freeAfterRelease << " " << freeAfterInit;
    }
}

TEST_F_SYN(SynAPITest, check_many_allocations, {synDeviceGaudi, synDeviceGaudi3})
{
    TestDevice device(m_deviceType);

    uint64_t freeAfterInit(0);
    uint64_t totalAfterInit(0);
    device.getDeviceMemoryInfo(freeAfterInit, totalAfterInit);

    const uint64_t allocationSize = 1024 * 1024;
    const uint64_t iterMax        = freeAfterInit / allocationSize;

    AllocDeviceBuffersVec allocatedBuffers;

    for (uint64_t iter = 0; iter < iterMax; iter++)
    {
        allocatedBuffers.push_back(std::move(device.allocateDeviceBuffer(allocationSize, 0 /* flags */)));
    }
}

// ## 7) Streams creation tests

TEST_F_SYN(SynAPITest, concurrently_create_all_streams, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    uint32_t streamAffinity(0);
    getDeviceStreamsRestriction(streamAffinity);

    TestDevice device(m_deviceType);

    std::queue<synStreamHandle> streamHandles;
    const uint32_t              flags = 0x0;

    for (uint32_t iter = 0; iter < streamAffinity; iter++)
    {
        synStreamHandle streamHandle;
        synStatus       status = synStreamCreateGeneric(&streamHandle, device.getDeviceId(), flags);
        ASSERT_EQ(status, synSuccess) << "Failed to create stream iter " << iter;
        streamHandles.push(streamHandle);
    }

    while (!streamHandles.empty())
    {
        const synStreamHandle streamHandle = streamHandles.front();
        synStatus             status       = synStreamDestroy(streamHandle);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy stream " << streamHandle;
        streamHandles.pop();
    }
}

TEST_F_SYN(SynAPITest, section_host_buffer_free)
{
    synStatus status(synSuccess);

    // A. Execute recipe "as is"
    // Serialize is executed only at the end of the call
    LOG_DEBUG(SYN_RT_TEST, "Test A");
    status = constSectionTest(false /* allowDeserialize */,
                              true /* useSerialize */,
                              false /* clearConstSectionHostBuffer */);
    ASSERT_EQ(status, synSuccess) << "Sub-test A have failed";

    // B. Execute recipe "as is", after deserialize (will not re-compile)
    LOG_DEBUG(SYN_RT_TEST, "Test B");
    status =
        constSectionTest(true /* allowDeserialize */, true /* useSerialize */, true /* clearConstSectionHostBuffer */);
    ASSERT_EQ(status, synSuccess) << "Sub-test B have failed";

    // C. Execute recipe from a deserialized recipe, which had its const-section data had been free
    LOG_DEBUG(SYN_RT_TEST, "Test C");
    status =
        constSectionTest(true /* allowDeserialize */, true /* useSerialize */, false /* clearConstSectionHostBuffer */);
    ASSERT_EQ(status, synInvalidArgument) << "Sub-test C have failed";
}

// ## 8) Graph manipulation tests

TEST_F_SYN(SynAPITest, set_node_params_for_gemm)
{
    TestRecipeGemm recipe(m_deviceType, {16, 16});
    // Only create the graph, but do not compile
    recipe.createGraphHandle();
    recipe.graphCreation();

    TestDevice device(m_deviceType);

    synGEMMParams gemmParams(true /* transpose_a */, false /* transpose_b */);

    pfnCompareParams compareParams = [](const void* newParams, const void* origParams) {
        synGEMMParams newGemmParams  = *((synGEMMParams*)newParams);
        synGEMMParams origGemmParams = *((synGEMMParams*)origParams);
        if ((newGemmParams.transpose_a != origGemmParams.transpose_a) ||
            (newGemmParams.transpose_b != origGemmParams.transpose_b))
        {
            return false;
        }
        return true;
    };

    TestSetAndGetNodeParams(device,
                            recipe,
                            recipe.getUniqueNodeId(),
                            (void*)&gemmParams,
                            sizeof(synGEMMParams),
                            compareParams);
}

TEST_F_SYN(SynAPITest, set_node_params_for_dynamic_split_node)
{
    TestRecipeDynamicSplit recipe(m_deviceType);
    // Only create the graph, but do not compile
    recipe.createGraphHandle();
    recipe.graphCreation();

    TestDevice device(m_deviceType);

    pfnCompareParams compareParams = [](const void* newParams, const void* origParams) {
        synSplitParams newSplitParams  = *((synSplitParams*)newParams);
        synSplitParams origSplitParams = *((synSplitParams*)origParams);
        bool           result          = (newSplitParams.axis != origSplitParams.axis) ? false : true;
        return result;
    };

    synSplitParams splitParams = {1};

    TestSetAndGetNodeParams(device,
                            recipe,
                            recipe.getUniqueNodeId(),
                            (void*)&splitParams,
                            sizeof(synSplitParams),
                            compareParams);
}

TEST_F_SYN(SynAPITest, hcl_api_device_generate_api_id)
{
    testGenerateApiId();
}

TEST_F_SYN(SynAPITest, synapse_profiler_api_get_queue_name)
{
    testGetQueueName();
}

TEST_F_SYN(SynAPITest, check_undefined_opcode_in_cs_dcs, {synDeviceGaudi})
{
    TestDevice device(m_deviceType);

    // 1. Allocate buffer
    uint64_t             chunkSize        = 1000;
    TestHostBufferMalloc mallocHostBuffer = device.allocateHostBuffer(chunkSize, 0 /* flags */);
    uint8_t*             buffer           = (uint8_t*)mallocHostBuffer.getBuffer();

    // 2. Retrieve deviceVA
    bool                                isExactKeyFound = false;
    uint64_t                            deviceVA        = 0;
    eHostAddrToVirtualAddrMappingStatus mappingStatus =
        synSingleton::getInstanceInternal()->_getDeviceVirtualAddress(true,
                                                                      buffer,
                                                                      chunkSize,
                                                                      &deviceVA,
                                                                      &isExactKeyFound);
    ASSERT_EQ(true, (mappingStatus == HATVA_MAPPING_STATUS_FOUND) && (isExactKeyFound))
        << "Failed to find mapped buffer";

    // 3. Create DC
    DataChunkMmuBuffer  dataChunk(chunkSize, buffer, deviceVA);
    DataChunkMmuBuffer* pDataChunk = &dataChunk;
    DataChunksDB        dataChunks = {pDataChunk};

    // 4. Create and init CS-DC
    size_t csDcMappingDbSize = (size_t)gaudi_queue_id::GAUDI_QUEUE_ID_SIZE;
    //
    CommandSubmissionDataChunks  cmdSubmissionDataChunks(CS_DC_TYPE_MEMCOPY, synDeviceGaudi, csDcMappingDbSize);
    CommandSubmissionDataChunks* pCsDc = &cmdSubmissionDataChunks;
    //
    pCsDc->addProgramBlobsDataChunks(dataChunks);
    //
    CommandSubmission* pCommandSubmission = new CommandSubmission(true);
    pCsDc->setCommandSubmissionInstance(pCommandSubmission);

    // 5. Update and test DCs (valid CS-DC)
    // 5.1. Update DCs
    uint8_t* pTmpBuffer     = buffer;  // packet generation will increment this PTR
    uint8_t* pCurrentBuffer = buffer;  // will be used for editing the packet
    uint64_t usedSize       = 0;
    unsigned enginesAmount  = 1;
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = false;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    generatePacketOpCode<packet_wait, PACKET_WAIT>(pTmpBuffer);
    pCurrentBuffer += sizeof(packet_wait);
    usedSize += sizeof(packet_wait);
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = true;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    pDataChunk->updateUsedSize(usedSize);
    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, usedSize, deviceVA);
    //
    // 5.2, Run validation over the CS-DC (valid use-case)
    bool operationStatus = checkForCsUndefinedOpcode(pCsDc, pCommandSubmission, enginesAmount, true);
    ASSERT_EQ(true, operationStatus) << "Valid CS verification failed";

    // 6. Update and test DCs (Invalid CS-DC - Amount of engines in ARB-group mismatch)
    // 6.1. Init
    enginesAmount = 2;
    //
    // 6.2. Update DCs
    pCommandSubmission->clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, usedSize, deviceVA);
    //
    // 6.3, Run validation over the CS-DC
    operationStatus = checkForCsUndefinedOpcode(pCsDc, pCommandSubmission, enginesAmount, true);
    ASSERT_EQ(false, operationStatus) << "Invalid CS (Amount of engines in ARB-group mismatch) verification passed";

    // 7. Update and test DCs (Invalid CS-DC - Missing ARB-Clear)
    // 7.1. Init
    pTmpBuffer     = buffer;
    pCurrentBuffer = buffer;
    usedSize       = 0;
    enginesAmount  = 1;
    //
    // 7.2. Update DCs
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = false;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    generatePacketOpCode<packet_wait, PACKET_WAIT>(pTmpBuffer);
    pCurrentBuffer += sizeof(packet_wait);
    usedSize += sizeof(packet_wait);
    //
    pDataChunk->updateUsedSize(usedSize);
    pCommandSubmission->clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, usedSize, deviceVA);
    //
    // 7.3, Run validation over the CS-DC
    operationStatus = checkForCsUndefinedOpcode(pCsDc, pCommandSubmission, enginesAmount, true);
    ASSERT_EQ(false, operationStatus) << "Invalid CS (Missing ARB-Release) verification passed";

    // 8. Update and test DCs (Invalid CS-DC - Consecutive two ARB-Requests)
    // 8.1. Init
    pTmpBuffer     = buffer;
    pCurrentBuffer = buffer;
    usedSize       = 0;
    //
    // 8.2. Update DCs
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = false;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    generatePacketOpCode<packet_wait, PACKET_WAIT>(pTmpBuffer);
    pCurrentBuffer += sizeof(packet_wait);
    usedSize += sizeof(packet_wait);
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = false;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    pDataChunk->updateUsedSize(usedSize);
    pCommandSubmission->clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, usedSize, deviceVA);
    //
    // 8.3, Run validation over the CS-DC
    operationStatus = checkForCsUndefinedOpcode(pCsDc, pCommandSubmission, enginesAmount, true);
    ASSERT_EQ(false, operationStatus) << "Invalid CS (Consecutive two ARB-Requests) verification passed";

    // 9. Update and test DCs (Invalid CS-DC - Consecutive two ARB-Release)
    // 9.1. Init
    pTmpBuffer     = buffer;
    pCurrentBuffer = buffer;
    usedSize       = 0;
    //
    // 9.2. Update DCs
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = false;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    generatePacketOpCode<packet_wait, PACKET_WAIT>(pTmpBuffer);
    pCurrentBuffer += sizeof(packet_wait);
    usedSize += sizeof(packet_wait);
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = true;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = true;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    pDataChunk->updateUsedSize(usedSize);
    pCommandSubmission->clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, usedSize, deviceVA);
    //
    // 9.3, Run validation over the CS-DC
    operationStatus = checkForCsUndefinedOpcode(pCsDc, pCommandSubmission, enginesAmount, true);
    ASSERT_EQ(false, operationStatus) << "Invalid CS (Consecutive two ARB-Release) verification passed";

    // 10. Update and test DCs (Invalid CS-DC - Invalid opcode)
    // 10.1. Init
    pTmpBuffer     = buffer;
    pCurrentBuffer = buffer;
    usedSize       = 0;
    //
    const unsigned invalidOpcode = 0;
    //
    // 10.2. Update DCs
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = false;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    generatePacketOpCode<packet_wait, PACKET_WAIT>(pTmpBuffer);
    ((packet_wait*)pCurrentBuffer)->opcode = invalidOpcode;
    pCurrentBuffer += sizeof(packet_wait);
    usedSize += sizeof(packet_wait);
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = true;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    pDataChunk->updateUsedSize(usedSize);
    pCommandSubmission->clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, usedSize, deviceVA);
    //
    // 10.3, Run validation over the CS-DC
    operationStatus = checkForCsUndefinedOpcode(pCsDc, pCommandSubmission, enginesAmount, true);
    ASSERT_EQ(false, operationStatus) << "Invalid CS (Invalid opcode) verification passed";

    // 11. Update and test DCs (Invalid CS-DC - Invalid PQ-size)
    // 11.1. Init
    pTmpBuffer     = buffer;
    pCurrentBuffer = buffer;
    usedSize       = 0;
    //
    // 11.2. Update DCs
    std::memset(buffer, 0, chunkSize);
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = false;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    generatePacketOpCode<packet_wait, PACKET_WAIT>(pTmpBuffer);
    pCurrentBuffer += sizeof(packet_wait);
    usedSize += sizeof(packet_wait);
    //
    generatePacketOpCode<packet_arb_point, PACKET_ARB_POINT>(pTmpBuffer);
    ((packet_arb_point*)pCurrentBuffer)->rls = true;
    pCurrentBuffer += sizeof(packet_arb_point);
    usedSize += sizeof(packet_arb_point);
    //
    pDataChunk->updateUsedSize(usedSize);
    pCommandSubmission->clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, usedSize - 1, deviceVA);
    //
    // 11.3, Run validation over the CS-DC
    operationStatus = checkForCsUndefinedOpcode(pCsDc, pCommandSubmission, enginesAmount, true);
    ASSERT_EQ(false, operationStatus) << "Invalid CS (Invalid buffer-size) verification passed";
}

// ## 9) DC-Allocator test

TEST_F_SYN(SynAPITest, data_chunks_allocator_test)
{
    bool status(true);

    TestDevice device(m_deviceType);

    // 1. Define an allocator
    uint16_t                         maxAmountOfDcs = 10;
    DataChunksAllocatorCommandBuffer allocator("debug", maxAmountOfDcs);

    // 2. Initialize the allocator with three different DC-Cache elements
    uint64_t multiplicandForMinCacheAmount     = 5;
    uint64_t multiplicandForMaxCacheFreeAmount = 50;
    uint64_t multiplicandForMaxCacheAmount     = 100;
    //
    uint64_t chunkSize1 = 8;
    uint64_t chunkSize2 = chunkSize1 * (maxAmountOfDcs + 1);
    //
    uint64_t currentCacheSize = chunkSize1;
    status                    = allocator.addDataChunksCache(currentCacheSize,
                                          currentCacheSize * multiplicandForMinCacheAmount,
                                          currentCacheSize * multiplicandForMaxCacheFreeAmount,
                                          currentCacheSize * multiplicandForMaxCacheAmount);
    ASSERT_EQ(status, true) << "Failed add DC-Cache 1";
    //
    currentCacheSize = chunkSize2;
    status           = allocator.addDataChunksCache(currentCacheSize,
                                          currentCacheSize * multiplicandForMinCacheAmount,
                                          currentCacheSize * multiplicandForMaxCacheFreeAmount,
                                          currentCacheSize * multiplicandForMaxCacheAmount);
    ASSERT_EQ(status, true) << "Failed add DC-Cache 2";
    //
    currentCacheSize = chunkSize2;
    status           = allocator.addDataChunksCache(currentCacheSize,
                                          currentCacheSize * multiplicandForMinCacheAmount,
                                          currentCacheSize * multiplicandForMaxCacheFreeAmount,
                                          currentCacheSize * multiplicandForMaxCacheAmount);
    ASSERT_EQ(status, false) << "Succeeded to re-add DC-Cache 2";

    DataChunksDB dataChunks;

    // Test-1: chunkSize2 - single allocation (of a chunkSize2 chunk-size)
    uint64_t allocSize      = chunkSize2;
    uint16_t amountOfChunks = 1;
    status                  = allocator.acquireDataChunks(dataChunks, allocSize);
    ASSERT_EQ(status, true) << "Failed acquire DC of " << allocSize << " chunk-size";
    ASSERT_EQ(dataChunks.size(), amountOfChunks)
        << "Invalid amount of DCs upon acquire request of " << allocSize << " chunk-size";
    auto dataChunksIter = dataChunks.begin();
    for (unsigned i = 0; i < amountOfChunks; i++, dataChunksIter++)
    {
        ASSERT_EQ((*dataChunksIter)->getChunkSize(), allocSize) << "Invalid Chunk-Size of DC allocated";
    }
    //
    status = allocator.releaseDataChunks(dataChunks);
    ASSERT_EQ(status, true) << "Failed to release DC of " << allocSize << " chunk-size";
    //
    allocator.updateCache();
    dataChunks.clear();

    // Test-2: chunkSize1 * maxAmountOfDcs - maxAmountOfDcs allocations
    allocSize      = chunkSize1 * maxAmountOfDcs;
    amountOfChunks = maxAmountOfDcs;
    status         = allocator.acquireDataChunks(dataChunks, allocSize);
    ASSERT_EQ(status, true) << "Failed acquire DC of " << allocSize << " chunk-size";
    ASSERT_EQ(dataChunks.size(), amountOfChunks)
        << "Invalid amount of DCs upon acquire request of " << allocSize << " chunk-size";
    dataChunksIter = dataChunks.begin();
    for (unsigned i = 0; i < amountOfChunks; i++, dataChunksIter++)
    {
        ASSERT_EQ((*dataChunksIter)->getChunkSize(), chunkSize1) << "Invalid Chunk-Size of DC allocated";
    }
    //
    status = allocator.releaseDataChunks(dataChunks);
    ASSERT_EQ(status, true) << "Failed to release DC of " << allocSize << " chunk-size";
    //
    allocator.updateCache();
    dataChunks.clear();

    // Test-3: chunkSize2 - single allocation
    allocSize      = chunkSize2;
    amountOfChunks = 1;
    status         = allocator.acquireDataChunks(dataChunks, allocSize);
    ASSERT_EQ(status, true) << "Failed acquire DC of " << allocSize << " chunk-size";
    ASSERT_EQ(dataChunks.size(), amountOfChunks)
        << "Invalid amount of DCs upon acquire request of " << allocSize << " chunk-size";
    dataChunksIter = dataChunks.begin();
    for (unsigned i = 0; i < amountOfChunks; i++, dataChunksIter++)
    {
        ASSERT_EQ((*dataChunksIter)->getChunkSize(), chunkSize2) << "Invalid Chunk-Size of DC allocated";
    }
    //
    status = allocator.releaseDataChunks(dataChunks);
    ASSERT_EQ(status, true) << "Failed to release DC of " << allocSize << " chunk-size";
    //
    allocator.updateCache();
    dataChunks.clear();

    // Test-4: chunkSize1 + chunkSize2 - multiple DC allocations
    allocSize      = chunkSize1 + chunkSize2;
    amountOfChunks = 2;
    status         = allocator.acquireDataChunks(dataChunks, allocSize);
    ASSERT_EQ(status, true) << "Failed acquire DC of " << allocSize << " chunk-size";
    ASSERT_EQ(dataChunks.size(), amountOfChunks)
        << "Invalid amount of DCs upon acquire request of " << allocSize << " chunk-size";
    dataChunksIter = dataChunks.begin();
    ASSERT_EQ((*dataChunksIter)->getChunkSize(), chunkSize2) << "Invalid Chunk-Size of DC allocated";
    dataChunksIter++;
    ASSERT_EQ((*dataChunksIter)->getChunkSize(), chunkSize2) << "Invalid Chunk-Size of DC allocated";
    //
    status = allocator.releaseDataChunks(dataChunks);
    ASSERT_EQ(status, true) << "Failed to release DC of " << allocSize << " chunk-size";
    //
    allocator.updateCache();
    dataChunks.clear();
}

// ## 9) SynLaunch Execution tests

TEST_F_SYN(SynAPITest, same_recipe_different_inputs)
{
    std::vector<TSize> sizes = {64, 64};
    TestRecipeAddf32   recipe(m_deviceType, sizes, false);
    recipe.generateRecipe();

    TestDevice   device(m_deviceType);
    TestLauncher launcher(device);
    TestStream   stream = device.createStream();

    // Test with first set of inputs
    execute(device, stream, launcher, recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0, false});
    // Test with second set of inputs
    execute(device, stream, launcher, recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0, false});
}

INSTANTIATE_TEST_SUITE_P(, SynAPITest, ::testing::Values(true, false));

TEST_P_SYN(SynAPITest, multiple_user_streams, {synTestPackage::ASIC_CI})
{
    bool isDsd = GetParam();
    if (isDsd && m_deviceType == synDeviceGaudi3)
    {
        GTEST_SKIP() << "DSD not supported in Gaudi3";
    }

    // Test's enums
    enum TestRecipeId
    {
        RECIPE_1,
        RECIPE_2,
        RECIPE_MAX
    };
    //
    enum TestStreamId
    {
        STREAM_1,
        STREAM_2,
        STREAM_3,
        STREAM_MAX
    };
    //
    enum TestWorkspaceId
    {
        WORKSPACE_1,
        WORKSPACE_2,
        WORKSPACE_MAX
    };

    // Test's parameters
    struct TestParameters
    {
        TestRecipeId    m_recipeId;
        TestStreamId    m_streamId;
        TestWorkspaceId m_workspaceId;
    };

    const TestParameters testsParams[] = {{RECIPE_1, STREAM_1, WORKSPACE_1},
                                          {RECIPE_1, STREAM_1, WORKSPACE_2},
                                          {RECIPE_1, STREAM_2, WORKSPACE_1},
                                          {RECIPE_1, STREAM_2, WORKSPACE_2},
                                          {RECIPE_2, STREAM_3, WORKSPACE_1},
                                          {RECIPE_1, STREAM_2, WORKSPACE_2}};

    const uint32_t testsAmount = sizeof(testsParams) / sizeof(TestParameters);

    std::vector<std::unique_ptr<TestRecipeBase>> testRecipeDB;
    std::vector<TestTensorsDimensions>           recipeTensorsDimensionsDB(RECIPE_MAX);
    // 1. Create recipes
    //
    if (isDsd)
    {
        const unsigned dimsAmount = 4;
        for (uint64_t recipeIndex = 0; recipeIndex < RECIPE_MAX; recipeIndex++)
        {
            // 1.1 Define graph's parameters
            std::vector<unsigned> recipeSizes(dimsAmount);
            unsigned              singleDimSize = 10 * (recipeIndex + 1);
            // Use different dimensions in order to achieve recipe differentiation
            for (unsigned i = 0; i < dimsAmount; i++)
            {
                recipeSizes.push_back(singleDimSize);
            }

            // 1.2 Create recipe
            const unsigned dynamicityRangeMul  = 4;
            const unsigned dynamicityActualMul = 2;
            const unsigned recipeMultiplier    = recipeIndex + 1;  // Diffrentiate between the recipes
            //
            unsigned op1StaticDim        = 256 * recipeMultiplier;
            unsigned op1DynamicDimMin    = 64 * recipeMultiplier;
            unsigned op1DynamicDimMax    = op1DynamicDimMin * dynamicityRangeMul;
            unsigned op1DynamicDimActual = op1DynamicDimMin * dynamicityActualMul;
            //
            unsigned op2StaticDim        = 256 * recipeMultiplier;
            unsigned op2DynamicDimMin    = 64 * recipeMultiplier;
            unsigned op2DynamicDimMax    = op2DynamicDimMin * dynamicityRangeMul;
            unsigned op2DynamicDimActual = op2DynamicDimMin * dynamicityActualMul;
            //
            TestRecipeDsdGemm* recipe = new TestRecipeDsdGemm(m_deviceType,
                                                              true /* isDynamic */,
                                                              false /* isSharedInputSection */,
                                                              {op1StaticDim, op1DynamicDimMax},
                                                              {op1StaticDim, op1DynamicDimMin},
                                                              {op2DynamicDimMax, op2StaticDim},
                                                              {op2DynamicDimMin, op2StaticDim},
                                                              {op2DynamicDimMax, op1DynamicDimMax},
                                                              {op2DynamicDimMin, op1DynamicDimMin});
            recipe->generateRecipe();

            // 1.3 Select actual dynamicity
            recipe->setExecutionDynamicSize(op1DynamicDimActual, op2DynamicDimActual);
            TestTensorsDimensions& rCurrRecipeTensorsDimensionsDB = recipeTensorsDimensionsDB[recipeIndex];
            enum class TensorsDiff
            {
                INPUT_0,
                INPUT_1,
                OUTPUT,
                MAX
            };
            //
            TensorDimensions tensorDimensions = {op1StaticDim, op1DynamicDimActual, 0, 0, 0};
            rCurrRecipeTensorsDimensionsDB.setDimensions(true, 0, tensorDimensions);
            //
            tensorDimensions = {op2DynamicDimActual, op2StaticDim, 0, 0, 0};
            rCurrRecipeTensorsDimensionsDB.setDimensions(true, 1, tensorDimensions);
            //
            tensorDimensions = {op2DynamicDimActual, op1DynamicDimActual, 0, 0, 0};
            rCurrRecipeTensorsDimensionsDB.setDimensions(false, 0, tensorDimensions);

            // 1.4 Store recipe on DB
            testRecipeDB.emplace_back(recipe);
        }
    }
    else
    {
        for (uint64_t recipeIndex = 0; recipeIndex < RECIPE_MAX; recipeIndex++)
        {
            TestRecipeAddf32* recipe = new TestRecipeAddf32(m_deviceType, {64, 64}, false);
            recipe->generateRecipe();
            testRecipeDB.emplace_back(recipe);
        }
    }

    // 2. Create a device, a launcher, streams and WS-events
    //
    TestDevice              device(m_deviceType);
    TestLauncher            launcher(device);
    std::vector<TestStream> testStreams;
    std::vector<TestEvent>  testEvents;
    //
    for (unsigned i = 0; i < STREAM_MAX; i++)
    {
        testStreams.emplace_back(std::move(device.createStream()));
    }
    //
    for (unsigned i = 0; i < WORKSPACE_MAX; i++)
    {
        testEvents.emplace_back(std::move(device.createEvent(0 /* flags */)));
    }

    // 3. Create launch params *per launch*, and store workspace-addresses
    std::vector<uint64_t> workspaceAddressDB(WORKSPACE_MAX);
    RecipeLaunchParamsVec launchParamsDB;
    for (unsigned i = 0; i < testsAmount; i++)
    {
        const TestParameters&    currTestParams = testsParams[i];
        const TestRecipeId&      recipeId       = currTestParams.m_recipeId;
        if (isDsd)
        {
            const TestRecipeDsdGemm& recipe = *(reinterpret_cast<TestRecipeDsdGemm*>(testRecipeDB[recipeId].get()));
            TestTensorsDimensions&   rCurrRecipeTensorsDimensionsDB = recipeTensorsDimensionsDB[recipeId];
            RecipeLaunchParams       currLaunchParams =
                launcher.createRecipeLaunchParams(recipe,
                                                  {TensorInitOp::RANDOM_WITH_NEGATIVE, 0, false},
                                                  rCurrRecipeTensorsDimensionsDB);

            // We will overun some, as a given WS appear on several tests' LaunchParams (but we don't care...)
            uint64_t workspaceAddress = currLaunchParams.getWorkspace();
            ASSERT_NE(workspaceAddress, 0) << "There is no workspace for this recipe";
            workspaceAddressDB[currTestParams.m_workspaceId] = workspaceAddress;
            launchParamsDB.emplace_back(std::move(currLaunchParams));
        }
        else
        {
            const TestRecipeAddf32& recipe = *(reinterpret_cast<TestRecipeAddf32*>(testRecipeDB[recipeId].get()));
            RecipeLaunchParams      currLaunchParams =
                launcher.createRecipeLaunchParams(recipe,
                                                  {TensorInitOp::RANDOM_WITH_NEGATIVE, 0, false});

            // We will overun some, as a given WS appear on several tests' LaunchParams (but we don't care...)
            uint64_t workspaceAddress = currLaunchParams.getWorkspace();
            workspaceAddressDB[currTestParams.m_workspaceId] = workspaceAddress;
            launchParamsDB.emplace_back(std::move(currLaunchParams));
        }
    }
    // Ensure we have enough WS addresses
    ASSERT_GE(workspaceAddressDB.size(), WORKSPACE_MAX);

    // 4. DWL all
    unsigned testIndex = 0;
    for (auto currTestParam : testsParams)
    {
        launcher.download(testStreams[currTestParam.m_streamId],
                          *testRecipeDB[currTestParam.m_recipeId],
                          launchParamsDB[testIndex]);
        testIndex++;
    }

    // 5. Launch all
    testIndex = 0;
    for (auto currTestParam : testsParams)
    {
        TestRecipeId        recipeId         = currTestParam.m_recipeId;
        TestWorkspaceId     workspaceId      = currTestParam.m_workspaceId;
        TestStream&         stream           = testStreams[currTestParam.m_streamId];
        RecipeLaunchParams& currLaunchParams = launchParamsDB[testIndex];

        // 5.1 Wait for previous usage of current WS
        stream.eventWait(testEvents[workspaceId], 0 /* flags */);

        // 5.2 Launch
        SynLaunchTensorInfoVec launchTensorsInfo = currLaunchParams.getSynLaunchTensorInfoVec();
        TestRecipeBase&        rTestRecipe       = *testRecipeDB[recipeId];

        stream.launch(launchTensorsInfo.data(),
                      launchTensorsInfo.size(),
                      workspaceAddressDB[workspaceId],
                      rTestRecipe.getRecipe(),
                      0 /* flags */);

        // 5.3 Record WS-event
        stream.eventRecord(testEvents[workspaceId]);

        testIndex++;
    }

    // 6. UPL all
    testIndex = 0;
    for (auto currTestParam : testsParams)
    {
        launcher.upload(testStreams[currTestParam.m_streamId],
                        *testRecipeDB[currTestParam.m_recipeId],
                        launchParamsDB[testIndex]);
        testIndex++;
    }

    // 7. Synchronize all streams
    for (auto& currStream : testStreams)
    {
        currStream.synchronize();
    }

    // 8. Validate
    testIndex = 0;
    for (auto currTestParam : testsParams)
    {
        testRecipeDB[currTestParam.m_recipeId]->validateResults(launchParamsDB[testIndex].getLaunchTensorMemory());
        testIndex++;
    }
}

TEST_F_SYN(SynAPITest, user_signaling, {synTestPackage::ASIC_CI, synTestPackage::ASIC})
{
    synStatus status(synSuccess);

    // Each type defines how Synapse will wait to the Compute part to complete prior of executing the Upload operation
    enum class SyncType
    {
        WAIT_FOR_COMPUTE_EVENT,   // Event-Synchronize
        WAIT_FOR_COMPUTE_STREAM,  // Stream-Synchronize
        SYNC_COMPUTE_EVENT,       // Stream-Wait-For-Event
        MAX
    };

    enum class StreamType
    {
        DOWNLOAD,
        COMPUTE,
        UPLOAD,
        MAX
    };

    // 1. Generate recipe
    TestRecipeHcl recipe(m_deviceType, false /* isSfgGraph */);
    recipe.generateRecipe();

    // 2. Create a device, a launcher, streams and events
    //
    TestDevice              device(m_deviceType);
    TestLauncher            launcher(device);
    std::vector<TestStream> testStreams;
    TestEvent               testEvent(std::move(device.createEvent(0 /* flags */)));
    //
    for (unsigned i = 0; i < (unsigned)StreamType::MAX; i++)
    {
        testStreams.emplace_back(std::move(device.createStream()));
    }
    TestStream& downloadStream = testStreams[(unsigned)StreamType::DOWNLOAD];
    TestStream& uploadStream   = testStreams[(unsigned)StreamType::UPLOAD];
    TestStream& computeStream  = testStreams[(unsigned)StreamType::COMPUTE];

    // 3. Create launch params *per launch* and execute
    for (unsigned i = 0; i < (unsigned)SyncType::MAX; i++)
    {
        RecipeLaunchParams recipeLaunchParams =
            launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0, false});

        launcher.download(downloadStream, recipe, recipeLaunchParams);
        downloadStream.eventRecord(testEvent);

        computeStream.eventWait(testEvent, 0 /* flags */);
        launcher.launch(computeStream, recipe, recipeLaunchParams);
        computeStream.eventRecord(testEvent);

        // This could only be tested on simulator
        // For a chip, we should push more work on it prior of expecting it to be busy
        /*
                testEvent.query(status);
                ASSERT_EQ(status, synBusy) << "Event had been expected to be busy";

                computeStream.query(status);
                ASSERT_EQ(status, synBusy) << "Compute-Stream had been expected to be busy";
         */

        switch ((SyncType)i)
        {
            case SyncType::SYNC_COMPUTE_EVENT:
            {
                uploadStream.eventWait(testEvent, 0 /* flags */);
            }
            break;

            case SyncType::WAIT_FOR_COMPUTE_EVENT:
            {
                testEvent.synchronize();
            }
            break;

            case SyncType::WAIT_FOR_COMPUTE_STREAM:
            {
                computeStream.synchronize();
            }
            break;

            case SyncType::MAX:
            {
                ASSERT_TRUE(false) << "Not a valid use case";
            }
            break;
        }
        launcher.upload(uploadStream, recipe, recipeLaunchParams);
        uploadStream.synchronize();

        // make sure compute job finished
        int maxIter = 20;
        int iter    = 0;
        while (iter < maxIter)
        {
            testEvent.query(status);
            if (status == synSuccess)
            {
                break;
            }

            usleep(1000);
            iter++;
        }
        ASSERT_NE(iter, maxIter) << "Test didn't finished in time 20 milliseconds";

        uploadStream.query(status);
        ASSERT_EQ(status, synSuccess) << "Upload-stream Query fails while event already concluded";

        recipe.validateResults(recipeLaunchParams.getLaunchTensorMemory());
    }
}

TEST_F_SYN(SynAPITest, unsupported_signal_from_graph, {synDeviceGaudi})
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensor tensor;
    synStatus status = synTensorHandleCreate(&tensor, graphHandle, DATA_TENSOR, "tensor");
    ASSERT_EQ(status, synSuccess) << "Failed to create tensor handle";

    status = synTensorSetExternal(tensor, true);
    ASSERT_EQ(status, synSuccess);

    synGraphDestroy(graphHandle);
}

TEST_F_SYN(SynAPITest, signal_from_graph, {synDeviceGaudi, synDeviceGaudi2})
{
    // A. Compile graph and test
    LOG_DEBUG(SYN_TEST, "Test A");
    sfgGraphTest(true /* forceCompilation */);

    // B. Deserialize recipe and test
    LOG_DEBUG(SYN_TEST, "Test B");
    sfgGraphTest(false /* forceCompilation */);
}

// ## 10) Canary-protection test

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
TEST_F_SYN(SynCanaryProtectionTest, DEATH_TEST_check_dc_canary_protection, {synDeviceGaudi})
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER", "1");
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    uint64_t singleChunkSize        = GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP.value() * 1024;
    uint64_t minimalCacheAmount     = GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP.value();
    uint64_t maximalFreeCacheAmount = minimalCacheAmount;
    uint64_t maximalCacheAmount     = minimalCacheAmount;

    // As we allocate streams upon acquire, we will set the following for reducing the memory usage size
    // For mprotecgt, a single DC's size must be 4KB aligned. Hence, we set its corresponding GCFG flag to 4.
    synConfigurationSet("STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP", "4");
    synConfigurationSet("STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP", "1024");

    TestDevice device(m_deviceType);

    synConfigurationSet("STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP", std::to_string(singleChunkSize).c_str());
    synConfigurationSet("STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP", std::to_string(minimalCacheAmount).c_str());

    DevMemoryAllocCommon devMemoryAlloc(m_deviceType, 0x10000000, 0xf0000000);
    MemoryManager        hostBufferMapper(devMemoryAlloc);

    DataChunksCacheMmuBuffer DcBuffer(&hostBufferMapper,
                                      singleChunkSize,
                                      minimalCacheAmount,
                                      maximalFreeCacheAmount,
                                      maximalCacheAmount,
                                      false,
                                      true);

    setDcMprotectTestSignalHandler();

    ASSERT_EXIT((memset((void*)0xf0000000, 0, 0x1000), exit(1)), ::testing::ExitedWithCode(EXIT_SUCCESS), ".*");
    GCFG_NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER.setValue(0);
}
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop