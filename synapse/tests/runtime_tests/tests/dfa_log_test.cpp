#include "syn_base_dfa_test.hpp"
#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_stream.hpp"
#include "test_launcher.hpp"
#include "runtime/common/device/device_common.hpp"
#include "test_recipe_tpc.hpp"
#include "test_recipe_addf32.hpp"
#include <chrono>
#include <vector>
#include <functional>
#include "scoped_configuration_change.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "syn_singleton.hpp"
#include "habana_global_conf_runtime.h"

static const char* API_CALL_FAIL_MSG = "got synSynapseTerminated from api call";

class SynCommonDfaLog : public SynBaseDfaTest
{
public:
    struct BusyStatus
    {
        bool computeBusy = true;
        bool copyBusy    = true;
    };

    void multipleLaunchStuck(BusyStatus& busyStatus);
    void singleLaunch();
    void launchDumpCheckdevFailFile(BusyStatus status);
    void launchDumpCheckSuspectedFile(BusyStatus status);
    void fakeDfaCheckLog(bool chkHlSmi);

private:
    std::string m_recipeName;
};
REGISTER_SUITE(SynCommonDfaLog, ALL_TEST_PACKAGES);

/*
 ***************************************************************************************************
 *   @brief fakeDfaError() fakes a dfa error
 *
 *   Function gets te devId and calls notifyHlthunkFailure() to simulate a dfa error
 *   It is expected that at the end of the dfa the thread will call kill, so it has to run
 *   with EXPECT_DEATH or something similar
 *
 *   @return DFA flow doesn't return but kills itself
 *
 ***************************************************************************************************
 */
static void fakeDfaError()
{
    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
    DeviceCommon*                    devCommon       = dynamic_cast<DeviceCommon*>(deviceInterface.get());

    // fake DFA
    DfaErrorCode dfaErrorCode = DfaErrorCode::tdrFailed;
    devCommon->notifyHlthunkFailure(dfaErrorCode);
}

/*
 ***************************************************************************************************
 *   @brief fakeDfaCheckReturnValueMain()
 *
 *   The function starts synapse and calls on a new thread a function to simulate DFA.
 *   It then waits until an API returns with "synSynapseTerminated" and terminates.
 *   Before terminating, it logs to the stderr a message so the caller (using EXPECT_DEATH or something
 *   similar) can know that the termination was from here and not from the DFA flow.
 *
 *   @param  None
 *   @return If all is good, kills itself after logging a message to the caller (in stderr)
 *
 ***************************************************************************************************
 */
static void fakeDfaCheckReturnValueMain(synDeviceType deviceType)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    TestDevice device(deviceType);
    auto       devId  = device.getDeviceId();
    TestStream stream = device.createStream();

    std::thread t1(fakeDfaError);

    // While thread is running verify that we start returning synSynapseTerminated;
    synDeviceInfo deviceInfo;

    synStatus status;
    int       cnt = 0;

    // We trigger here by simulating a hlthunk error. There is a delay in the code for that, wait the same before
    // starting the test
    std::this_thread::sleep_for(dfaHlthunkTriggerDelay);

    while ((status = synDeviceGetInfo(devId, &deviceInfo)) != synSynapseTerminated)
    {
        ASSERT_EQ(status, synSuccess) << "if DFA hasn't started, it should return synSuccess";
        usleep(100000);

        // user thread is expected to be held, so only few (maybe 3) returns of synSuccess are expected (because
        // of a race condition with the new thread).
        cnt++;
        ASSERT_LE(cnt, 3) << "User thread should be held until DFA is done";
    }

    // if we are here, we got terminated. Kill the process. The caller will check that DFA was done.
    // we should have 5 seconds to kill from here before DFA terminates the process
    std::cerr << API_CALL_FAIL_MSG << "\n";
    kill(getpid(), SIGKILL);

    t1.join();
}

static void fakeDfa(synDeviceType deviceType)
{
    TestDevice device(deviceType);
    TestStream stream = device.createStream();

    fakeDfaError();  // this should kill process and return to the test (the one who called ASSERT_EXIT)
}

static bool isDirectModeUserDownloadStream()
{
    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
    DeviceCommon*                    devCommon       = dynamic_cast<DeviceCommon*>(deviceInterface.get());
    return devCommon->isDirectModeUserDownloadStream();
}

using SynCommonDfaLogDeathTest = SynCommonDfaLog;
REGISTER_SUITE(SynCommonDfaLogDeathTest, ALL_TEST_PACKAGES);

// This test checks the return value during DFA (synSynapseTerminated). The test sets a flag not to read registers so
// it can run faster
// Note: the name *DeathTest cause the tests to run first
TEST_F(SynCommonDfaLogDeathTest, fakeDfaCheckReturnValue)
{
    // stop periodic flush - it starts a thread and google death test doesn't like more than one thread
    synapse::LogManager::instance().enablePeriodicFlush(false);

    ASSERT_EXIT(fakeDfaCheckReturnValueMain(m_deviceType), ::testing::KilledBySignal(SIGKILL), API_CALL_FAIL_MSG);

    std::vector<ExpectedWords> expectedSynFail = {
        {"Logging failure info done", 1, std::equal_to<uint32_t>()},
    };

    expectedInFile(loggerSynDevFail, expectedSynFail);

    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}

static void causeSigsegvWithSynInit(synDeviceType deviceType)
{
    synInitialize();  // synInit - initialize on crash
    TestDevice device(deviceType);

    int* x = nullptr;
    *x     = 5;
}

static void causeSigsegv()
{
    int* x = nullptr;
    *x     = 5;
}

class SynCommonDfaLogExceptionDeathTest : public SynCommonDfaLogDeathTest
{
public:
    // override SetUp and TearDown in order to skip synInit
    virtual void SetUp() override
    {
        synDeviceType deviceType = getDeviceType();
        ASSERT_NE(deviceType, INVALID_DEVICE_TYPE) << "getDeviceType failed";
        if (!isSupportedDeviceTypeForTest(deviceType))
        {
            GTEST_SKIP() << "isSupportedDeviceTypeForTest skipped with deviceType " << deviceType;
        }
        m_deviceType = deviceType;
        dfaFilesCheck::init();
        setTestLoggers();
    }
    virtual void TearDown() override {}

    void checkExceptionLog(bool logDfaOnException)
    {
        std::unique_ptr<ScopedEnvChange> dfaOnTerminate;
        auto                             expectedSignal = SIGSEGV;

        ScopedEnvChange expFlags("EXP_FLAGS", "true");
        if (!logDfaOnException)
        {
            dfaOnTerminate = std::unique_ptr<ScopedEnvChange>(new ScopedEnvChange("DFA_ON_SIGNAL", "false"));
        }

        // google test needs only one thread running (close spdlog thread)
        synapse::LogManager::instance().enablePeriodicFlush(false);

        ASSERT_EXIT(causeSigsegvWithSynInit(m_deviceType), ::testing::KilledBySignal(expectedSignal), "");

        std::vector<ExpectedWords> expectedSynFail = {
            {"Version:", 1, std::equal_to<uint32_t>()},
            {"exception 11 Segmentation fault", 1, std::equal_to<uint32_t>()},
            {"backtrace (up to", 1, std::equal_to<uint32_t>()},
        };

        if (logDfaOnException)
        {
            expectedSynFail.push_back(
                {"Logging failure info done", 1, std::equal_to<uint32_t>()});  // make sure we did DFA
        }

        expectedInFile(loggerSynDevFail, expectedSynFail);

        if (!::testing::Test::HasFailure())
        {
            removeTestDfaFiles();
        }
    }
};
REGISTER_SUITE(SynCommonDfaLogExceptionDeathTest, ALL_TEST_PACKAGES);

TEST_F(SynCommonDfaLogExceptionDeathTest, checkDfaExceptionNoSynInit)
{
    // google test needs only one thread running (close spdlog thread)
    synapse::LogManager::instance().enablePeriodicFlush(false);

    // synIinit and then destroy to print info at the beggining of the synapse logs
    synInitialize();
    synDestroy();

    ASSERT_EXIT(causeSigsegv(), ::testing::KilledBySignal(SIGSEGV), "");

    std::vector<ExpectedWords> expectedSynFail = {
        {"Version:", 1, std::equal_to<uint32_t>()},
        {"exception 11 Segmentation fault", 0, std::equal_to<uint32_t>()},
        {"backtrace (up to", 0, std::equal_to<uint32_t>()},
    };

    expectedInFile(loggerSynDevFail, expectedSynFail);

    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}

TEST_F(SynCommonDfaLogExceptionDeathTest, checkNoDfaExceptionLog)
{
    checkExceptionLog(false);
}

TEST_F(SynCommonDfaLogExceptionDeathTest, checkExceptionLog)
{
    checkExceptionLog(true);
}

// This test checks the DFA log files
// 1) Deletes existing files
// 2) Fakes DFA
// 3) Checks for expected words in the DFA files
TEST_F(SynCommonDfaLogDeathTest, fakeDfaCheckLog)
{
    fakeDfaCheckLog(false);
}

class SynCommon_ASIC_CI_DfaLogDeathTest : public SynCommonDfaLogDeathTest
{
};
REGISTER_SUITE(SynCommon_ASIC_CI_DfaLogDeathTest, synTestPackage::ASIC_CI, synTestPackage::ASIC);

TEST_F(SynCommon_ASIC_CI_DfaLogDeathTest, fakeDfaCheckLog)
{
    fakeDfaCheckLog(true);
}

using SynCommon_ASIC_CI_DfaLogExceptionDeathTest = SynCommonDfaLogExceptionDeathTest;
REGISTER_SUITE(SynCommon_ASIC_CI_DfaLogExceptionDeathTest, synTestPackage::ASIC_CI, synTestPackage::ASIC);

TEST_F(SynCommon_ASIC_CI_DfaLogExceptionDeathTest, checkNoDfaExceptionLog)
{
    checkExceptionLog(false);
}

TEST_F(SynCommon_ASIC_CI_DfaLogExceptionDeathTest, checkExceptionLog)
{
    checkExceptionLog(true);
}

void SynCommonDfaLog::fakeDfaCheckLog(bool chkHlSmi)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("DFA_COLLECT_CCB", "true");

    // simulate DFA
    // Check 'user unexpected interrupt' is printed to dmesg on DFA
    const uint32_t numOfLines = 100;
    uint32_t       dmesgMsgCountPre;
    uint32_t       dmesgMsgCountPost;
    std::string refTimestamp {"null"};
    wordCountInDmesg(dmesgMsgCountPre, "unexpected", refTimestamp, numOfLines);
    LOG_INFO(SYN_RT_TEST, "refTimestamp value is: {}", refTimestamp);

    // google test needs only one thread running (close spdlog thread)
    synapse::LogManager::instance().enablePeriodicFlush(false);
    ASSERT_EXIT(fakeDfa(m_deviceType), ::testing::KilledBySignal(SIGKILL), DFA_KILL_MSG);

    checkDfaBegin();  // check all files have '#DFA begin' with the same timestamp

    std::vector<ExpectedWords> expected;
    getExpectedDevFail(expected, chkHlSmi);

    if (m_deviceType != synDeviceGaudi)
    {
        expected.push_back({"cyclic buffers", 1, std::equal_to<uint32_t>()});
        expected.push_back({"#ccb", 13, std::greater_equal<uint32_t>()});
    }

    expectedInFile(DfaLoggerEnum::loggerSynDevFail, expected);

    expected = getCommonExpectedDmesg();
    expectedInFile(DfaLoggerEnum::loggerDmesgCpy, expected);

    wordCountInDmesg(dmesgMsgCountPost, "unexpected", refTimestamp, numOfLines * 10);
    if (m_deviceType == synDeviceGaudi2 || m_deviceType == synDeviceGaudi3)
    {
        ASSERT_EQ(dmesgMsgCountPre + 1, dmesgMsgCountPost);
    }

    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}

// This test does few launches, fakes dfa and verifies the recipes are logged in the dfa log
// It uses only one set of tensors for all launches.
// Reading registers is not checked here
void SynCommonDfaLog::multipleLaunchStuck(BusyStatus& busyStatus)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("TERMINATE_SYNAPSE_UPON_DFA",
                        std::to_string((uint64_t)DfaSynapseTerminationState::disabled).c_str());
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    synStatus  status = synSuccess;
    TestDevice device(m_deviceType);

    TestStream streamDown     = device.createStream();
    TestStream streamCompute0 = device.createStream();
    TestStream streamCompute1 = device.createStream();
    TestStream streamUp       = device.createStream();

    TestEvent event0 = device.createEvent();
    TestEvent event1 = device.createEvent();

    // prepare the recipe
    const std::vector<TSize>          size = {32 * 1024 * 1024};
    std::unique_ptr<TestRecipeAddf32> recipe[CHK_LAUNCH_DUMP_NUM];
    for (int i = 0; i < CHK_LAUNCH_DUMP_NUM; i++)
    {
        // for some reason, scal runs much slower. We need it to run long enough for dfa to log the "stuck" recipe
        recipe[i] = std::make_unique<TestRecipeAddf32>(m_deviceType, size);
        recipe[i]->generateRecipe();
    }
    m_recipeName = recipe[0]->getRecipeName();

    std::unique_ptr<TestLauncher> launcher[CHK_LAUNCH_DUMP_NUM];
    for (int i = 0; i < CHK_LAUNCH_DUMP_NUM; i++)
    {
        launcher[i] = std::make_unique<TestLauncher>(device);
    }

    // prepare tensors
    // only one set of tensors for all recipes
    auto recipeLaunchParams =
        launcher[0]->createRecipeLaunchParams(*recipe[0].get(), {TensorInitOp::RANDOM_POSITIVE, 0});
    TestLauncher::download(streamDown, *recipe[0].get(), recipeLaunchParams);
    device.synchronize();

    // do the launch
    for (int i = 0; i < CHK_LAUNCH_DUMP_NUM; i++)
    {
        auto recipeLaunchParams = launcher[i]->createRecipeLaunchParams(*recipe[i].get(), TensorInitInfo());

        status = synLaunchExt(streamCompute0,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipe[0]->getTensorInfoVecSize(),
                              recipeLaunchParams.getWorkspace(),
                              recipe[i]->getRecipe(),
                              SYN_FLAGS_TENSOR_NAME);
        ASSERT_EQ(status, synSuccess) << "Failed to launch";
        status = synLaunchExt(streamCompute1,
                              recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                              recipe[0]->getTensorInfoVecSize(),
                              recipeLaunchParams.getWorkspace(),
                              recipe[i]->getRecipe(),
                              SYN_FLAGS_TENSOR_NAME);
        ASSERT_EQ(status, synSuccess) << "Failed to launch";
        if (i == 0)
        {
            streamCompute0.eventRecord(event0);
            streamCompute1.eventRecord(event1);
        }
    }

    event0.synchronize();
    event1.synchronize();

    // redo the copy, just that we have something to check in the log
    TestLauncher::download(streamDown, *recipe[0].get(), recipeLaunchParams);

    // fake DFA
    fakeDfaError();

    // check if both dma and compute are still running. If so, indicate it to the calling test, so it can check
    // all the information. This is the usual case, there are some race conditions where the work is done
    // before the dfa is completely logged. In this case, don't check.
    busyStatus = {true, true};

    std::string testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    streamCompute0.eventRecord(event0);
    if (event0.query() != synBusy)
    {
        LOG_ERR(SYN_RT_TEST, "test {} can't check output, stream compute0 already done", testName);
        busyStatus.computeBusy = false;
    }

    streamCompute1.eventRecord(event0);
    if (event0.query() != synBusy)
    {
        LOG_ERR(SYN_RT_TEST, "test {} can't check output, stream compute1 already done", testName);
        busyStatus.computeBusy = false;
    }

    bool isDirectModeDmaDown = isDirectModeUserDownloadStream();
    streamDown.eventRecord(event0);
    if ((!isDirectModeDmaDown) && (event0.query() != synBusy))
    {
        LOG_ERR(SYN_RT_TEST, "test {} can't check output, stream streamDown already done", testName);
        busyStatus.copyBusy = false;
    }

    device.synchronize();  // we have to wait for all work to finish avoiding a simulator crash
}

void SynCommonDfaLog::singleLaunch()
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("TERMINATE_SYNAPSE_UPON_DFA",
                        std::to_string((uint64_t)DfaSynapseTerminationState::disabled).c_str());
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    TestDevice device(m_deviceType);

    auto streamDown     = device.createStream();
    auto streamCompute0 = device.createStream();
    auto streamUp       = device.createStream();

    // prepare the recipe
    const std::vector<TSize> size = {32 * 1024 * 1024};
    TestRecipeAddf32         recipe(m_deviceType, size);
    m_recipeName = recipe.getRecipeName();
    recipe.generateRecipe();

    TestLauncher launcher(device);

    // prepare tensors
    auto recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_POSITIVE, 0});
    TestLauncher::download(streamDown, recipe, recipeLaunchParams);
    device.synchronize();

    launcher.launch(streamCompute0, recipe, recipeLaunchParams);

    streamCompute0.synchronize();

    // fake DFA
    fakeDfaError();
}

void SynCommonDfaLog::launchDumpCheckdevFailFile(BusyStatus busy)
{
    std::vector<ExpectedWords> expected;

    bool isScal = isScalDevice();

    if (isScal)
    {
        expected = {
            // Each should be log twice, once for oldest, one when showing all work
            {"stream pdma_tx0", 4, std::equal_to<uint32_t>()},
            {"stream compute0", 4, std::equal_to<uint32_t>()},
            {"stream compute1", 4, std::equal_to<uint32_t>()},
            {"stream pdma_rx0", 4, std::equal_to<uint32_t>()},
        };

        if (busy.computeBusy)
        {
            expected.push_back({"Recipe stats", 2, std::greater_equal<uint32_t>()});
        }
    }
    else
    {
        expected = {
            {"stream has 0 csdc-s", 1, std::greater<uint32_t>()},
        };
        if (busy.copyBusy)
        {
            expected.push_back({"copy direction", 1, std::greater<uint32_t>()});  // work on dma is expected
        }
    }

    if (busy.computeBusy)
    {
        for (int i = 1; i < CHK_LAUNCH_DUMP_NUM; i++)
        {
            expected.push_back({m_recipeName.c_str(), 2, std::greater_equal<uint32_t>()});
        }
        expected.push_back({"Num of successful launches", 1, std::greater<uint32_t>()});
        expected.push_back({"Shape plan recipe", 1, std::greater<uint32_t>()});
        expected.push_back({std::string("Dumping recipe to ") + SUSPECTED_RECIPES, 1, std::greater_equal<uint32_t>()});
        expected.push_back({"#TPC_MASK", 1, std::equal_to<uint32_t>()});
        expected.push_back({"#MME_NUM", 1, std::equal_to<uint32_t>()});
        expected.push_back({"#DMA_MASK", 1, std::equal_to<uint32_t>()});
        expected.push_back({"#ROT_NUM", 1, std::equal_to<uint32_t>()});
    }
    expected.push_back({"Function: synHostMalloc", 1, std::greater_equal<uint32_t>()});

    uint32_t numCcb = (GCFG_DFA_COLLECT_CCB.value() && (m_deviceType != synDeviceGaudi)) ? 13 : 0; // if not set, maker sure we don't collect ccb (13: at least 13 streams)
    expected.push_back({"#ccb", numCcb, std::greater_equal<uint32_t>()});

    // Check file against expected
    expectedInFile(DfaLoggerEnum::loggerSynDevFail, expected);

    std::vector<ExpectedWords> expectedInDfaApi = {{"SYN_PROGRESS", 10, std::greater_equal<uint32_t>()}};

    expectedInFile(DfaLoggerEnum::loggerDfaApiInfo, expectedInDfaApi);
}

void SynCommonDfaLog::launchDumpCheckSuspectedFile(BusyStatus status)
{
    if (!status.computeBusy)
    {
        return;
    }

    std::vector<ExpectedWords> expected = {
        {"Dump recipe", 1, std::greater_equal<uint32_t>()},
        {"Version", 1, std::greater_equal<uint32_t>()},
        {"Number nodes", 1, std::greater_equal<uint32_t>()},
        {"Blobs", 1, std::greater_equal<uint32_t>()},
        {"Not DSD", 1, std::greater_equal<uint32_t>()},
        {"#Dump sync-scheme", 1, std::greater_equal<uint32_t>()},
        {"config params", 1, std::greater_equal<uint32_t>()},
    };
    expectedInFile(DfaLoggerEnum::loggerFailedRecipe, expected);
}

// This test does few launches, fakes dfa and verifies the recipes are logged in the dfa log
// It uses only one set of tensors for all launches
// The test doesn't collect device registers (multipleLaunchStuck() is setting the skip option)
TEST_F(SynCommonDfaLog, checkLaunchDump)
{
    synapse::LogManager::instance().enablePeriodicFlush(false);

    BusyStatus busyStatus;
    multipleLaunchStuck(busyStatus);

    if (!busyStatus.computeBusy || !busyStatus.copyBusy)
    {
        LOG_ERR(SYN_RT_TEST, "Some checks are skipped as some work was done before dfa could log it");
        std::cout << "Couldn't check some of the dfa logs (can happen due to race conditions, ignore)\n";
    }

    launchDumpCheckdevFailFile(busyStatus);
    launchDumpCheckSuspectedFile(busyStatus);

    // Set expected output
    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}

TEST_F(SynCommonDfaLog, checkTpcKernelDump)
{
    synapse::LogManager::instance().enablePeriodicFlush(false);

    singleLaunch();

    bool tpcKernelHexDumpSupported = true;

    switch (m_deviceType)
    {
        case synDeviceGaudi:
        case synDeviceGaudi2:
        case synDeviceGaudi3:
            break;

        default:
            tpcKernelHexDumpSupported = false;
    }

    // verify here if file contains the dump
    std::vector<ExpectedWords> expectedTpcDump;
    if (tpcKernelHexDumpSupported)
    {
        expectedTpcDump.push_back({"tpc_kernel 0 does not exist", 0, std::equal_to<uint32_t>()});
        expectedTpcDump.push_back({"tpc_kernel 1 is the same as tpc_kernel 0", 1, std::equal_to<uint32_t>()});
    }
    expectedInFile(loggerSynDevFail, expectedTpcDump);

    // Set expected output
    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}

TEST_F(SynCommonDfaLog, DISABLED_checkUsrMsg)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("TERMINATE_SYNAPSE_UPON_DFA",
                        std::to_string((uint64_t)DfaSynapseTerminationState::disabledRepeat).c_str());
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    TestDevice device(m_deviceType);

    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
    DeviceCommon*                    devCommon       = dynamic_cast<DeviceCommon*>(deviceInterface.get());

    // fake DFA
    devCommon->notifyDeviceFailure(DfaErrorCode::tdrFailed);

    devCommon->notifyHlthunkFailure(DfaErrorCode::streamSyncFailed);

    devCommon->notifyEventFd(HL_NOTIFIER_EVENT_DEVICE_RESET | HL_NOTIFIER_EVENT_UNDEFINED_OPCODE);

    devCommon->notifyDeviceFailure(DfaErrorCode::scalTdrFailed);

    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}
