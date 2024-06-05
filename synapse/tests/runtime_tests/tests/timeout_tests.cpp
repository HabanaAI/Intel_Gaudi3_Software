#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "habana_global_conf_runtime.h"
#include "syn_singleton.hpp"
#include "runtime/scal/common/device_scal.hpp"
#include "test_device.hpp"

class TimeoutTests : public SynBaseTest
{
public:
    TimeoutTests()
    : SynBaseTest(),
      m_streamDown(nullptr),
      m_streamDown2(nullptr),
      m_hostBuff(nullptr),
      m_devBuff(0),
      m_devScal(nullptr)
    {
        setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});
    };

    const uint64_t SIZE = 32 * 1024ULL * 1024ULL;  // any size will do, should take at least 1msec

    void runDma(TestDevice&                    rDevice,
                std::array<synStreamHandle, 2> rStreamHandles,
                const TestEvent&               rEventDown,
                std::string&                   skipReason,
                bool                           shouldDisableTermination,
                scal_timeouts_t const&         timeouts = scal_timeouts_t {SCAL_TIMEOUT_NOT_SET, SCAL_TIMEOUT_NOT_SET},
                bool                           disableTimeouts = false);
    void runDma(TestDevice&                    rDevice,
                std::array<synStreamHandle, 2> rStreamHandles,
                const TestEvent&               rEventDown,
                std::string&                   skipReason,
                scal_timeouts_t const&         timeouts,
                bool                           disbaleTimeouts);

    void awaitSynapseTermination(const TestDevice& rDevice, const TestEvent& rEventDown);
    void awaitTestDeath();

    synStreamHandle     m_streamDown;
    synStreamHandle     m_streamDown2;
    void*               m_hostBuff;
    uint64_t            m_devBuff;
    common::DeviceScal* m_devScal;
};

REGISTER_SUITE(TimeoutTests, ALL_TEST_PACKAGES);

void TimeoutTests::awaitSynapseTermination(const TestDevice& rDevice, const TestEvent& rEventDown)
{
    LOG_DEBUG(SYN_RT_TEST, "Test awaits termination!");
    synStatus status = synSuccess;
    while (status != synSynapseTerminated)
    {
        usleep(DfaBase::SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US * 0.5);
        status = synEventQuery(rEventDown);
    }
    ASSERT_EQ(m_devScal->getDfaStatus().hasError(DfaErrorCode::tdrFailed), true) << "Timeout is expected";

    status = synDeviceSynchronize(rDevice.getDeviceId());
    ASSERT_EQ(status, synSynapseTerminated) << "API is not blocked upon termination";

    LOG_DEBUG(SYN_RT_TEST, "Test had been Completed!");

    if (GCFG_TERMINATE_SYNAPSE_UPON_DFA.value() == (uint64_t)DfaSynapseTerminationState::busyWait)
    {
        LOG_DEBUG(SYN_RT_TEST, "Terminating test due to busy-wait!");

        if (0 == kill(getpid(), SIGKILL))
        {
            LOG_DEBUG(SYN_RT_TEST, "Sent SIGKILL to process");
            synapse::LogManager::instance().flush();
        }
        else
        {
            LOG_CRITICAL(SYN_RT_TEST, "The sigkill execution failed");
        }
    }
}

void TimeoutTests::awaitTestDeath()
{
    usleep(DfaBase::SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US * 1.5);
}

void TimeoutTests::runDma(TestDevice&                    rDevice,
                          std::array<synStreamHandle, 2> rStreamHandles,
                          const TestEvent&               rEventDown,
                          std::string&                   skipReason,
                          bool                           shouldDisableTermination,
                          scal_timeouts_t const&         timeouts,
                          bool                           disableTimeouts)
{
    // Do not collect registers to save CI time
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    if (shouldDisableTermination)
    {
        synConfigurationSet("TERMINATE_SYNAPSE_UPON_DFA",
                            std::to_string((uint64_t)DfaSynapseTerminationState::disabled).c_str());

        runDma(rDevice, rStreamHandles, rEventDown, skipReason, timeouts, disableTimeouts);
    }
    else
    {
        runDma(rDevice, rStreamHandles, rEventDown, skipReason, timeouts, disableTimeouts);

        if (!skipReason.empty())
        {
            return;
        }

        awaitSynapseTermination(rDevice, rEventDown);
        awaitTestDeath();
    }
}

void TimeoutTests::runDma(TestDevice&                    rDevice,
                          std::array<synStreamHandle, 2> rStreamHandles,
                          const TestEvent&               rEventDown,
                          std::string&                   skipReason,
                          scal_timeouts_t const&         timeouts,
                          bool                           disableTimeouts)
{
    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
    m_devScal                                        = dynamic_cast<common::DeviceScal*>(deviceInterface.get());
    ASSERT_NE(m_devScal, nullptr) << "Failed to find device, this is a test error";

    LOG_INFO(SYN_RT_TEST,
             "Setting timeouts and bgFreq: eng timeout: {} noProgress {} disable {}",
             timeouts.timeoutUs,
             timeouts.timeoutNoProgressUs,
             disableTimeouts);
    m_devScal->setTimeouts(timeouts, disableTimeouts);
    m_devScal->testingOnlySetBgFreq(std::chrono::milliseconds(0));  // Immediate - 0 msec

    m_streamDown  = rStreamHandles[0];
    m_streamDown2 = rStreamHandles[1];

    synStatus status = synHostMalloc(rDevice.getDeviceId(), SIZE, 0, &m_hostBuff);
    ASSERT_EQ(status, synSuccess) << "Failed alocate host buffer";

    status = synDeviceMalloc(rDevice.getDeviceId(), SIZE, 0, 0, &m_devBuff);
    ASSERT_EQ(status, synSuccess) << "Failed to alocate HBM memory";

    status = synMemCopyAsync(m_streamDown, (uint64_t)m_hostBuff, SIZE, m_devBuff, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    status = synEventRecord(rEventDown, m_streamDown);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";
}

// This test runs dma, them waits for the work to finish. It doesn't expect dfa to be set
TEST_F_SYN(TimeoutTests, noTimeout)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    std::string skipReason;
    runDma(device, streamHandles, eventDown, skipReason, true);

    while (true)
    {
        auto status = synEventQuery(eventDown);
        if (status != synBusy)
        {
            LOG_INFO(SYN_RT_TEST, "Status is {}", status);
            break;
        }

        sleep(1);
    }

    sleep(3);  // Give the background thread some time to set the DFA flag, in case DFA is triggered by mistake (it
               // shouldn't set the flag in this test)

    ASSERT_EQ((m_devScal->getDfaStatus()).isSuccess(), true) << "No timeout is expected";
}

// This test sets a very small timeout value and then runs dma. It expects dfa to set the tdrFailed bit.
// The termination is disabled so we can be sure the dma is done before the end of the test
TEST_F_SYN(TimeoutTests, timeout)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    LOG_ERR_T(SYN_API, "-----THIS TEST IS EXPECTED TO LOG ERRORS-----");
    std::string skipReason;
    // set a very small timeout
    runDma(device, streamHandles, eventDown, skipReason, true, scal_timeouts_t {1, SCAL_TIMEOUT_NOT_SET});
    if (!skipReason.empty())
    {
        GTEST_SKIP() << skipReason;
    }

    unsigned  cnt = 0;
    synStatus status {synSuccess};
    while (true)
    {
        status = synEventQuery(eventDown);
        if (status != synBusy)
        {
            LOG_INFO(SYN_RT_TEST, "Status is {}", status);
            break;
        }

        sleep(1);
        cnt++;
    }

    sleep(3);  // see jira 111515 for more details. I suspect the tdr thread detected the error and went
               // away before setting the dfaStatus. Then the test thread noticed the test was done but the error
               // wasn't set yet.

    bool tdrFailed = m_devScal->getDfaStatus().hasError(DfaErrorCode::tdrFailed);

    // in case the test fails, logged some information to help with the debug
    if (!tdrFailed)  // collect more information so we can debug the test
    {
        LOG_ERR(SYN_DEV_FAIL,
                "Test failed, Dfa status {:x} status {} cnt {}",
                m_devScal->getDfaStatus().getRawData(),
                status,
                cnt);
    }

    ASSERT_EQ(tdrFailed, true) << "Timeout is expected";
}

// This test disables the timeout checks, sets a small timeout value and verifies no timeout is detected
TEST_F_SYN(TimeoutTests, timeoutDisabled)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    std::string skipReason;
    // set a very small timeout
    runDma(device, streamHandles, eventDown, skipReason, true, scal_timeouts_t {1, SCAL_TIMEOUT_NOT_SET}, true);

    while (true)
    {
        synStatus status = synEventQuery(eventDown);
        if (status != synBusy)
        {
            LOG_INFO(SYN_RT_TEST, "Status is {}", status);
            break;
        }

        sleep(1);
    }

    sleep(3);  // Give the background thread some time to set the DFA flag, in case DFA is triggered by mistake (it
               // shouldn't set the flag in this test)

    ASSERT_EQ(m_devScal->getDfaStatus().isSuccess(), true) << "Timeout not expected, it is disabled";
}

// This test sets a small no-progress timeout, runs a dma and verifies timeout is triggered in the dfa status
// The test disables termination, so we can check the dma is done before we end the test
TEST_F_SYN(TimeoutTests, noProgressTimeout1)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    std::string skipReason;
    // set a very small timeout no progress
    runDma(device, streamHandles, eventDown, skipReason, true, scal_timeouts_t {SCAL_TIMEOUT_NOT_SET, 1});

    while (true)
    {
        synStatus status = synEventQuery(eventDown);
        if (status != synBusy)
        {
            ASSERT_EQ(status,
                      synSuccess);  // termination is disabled, so DFA holds the API durig DFA and then returns success
            break;
        }

        sleep(1);
    }

    sleep(
        3);  // give the background thread (event-fd thread) some time to set the DFA flag (just to be on the safe side)
    ASSERT_EQ(m_devScal->getDfaStatus().hasError(DfaErrorCode::scalTdrFailed), true)
        << "Timeout expected " << m_devScal->getDfaStatus().getRawData();
}

// This test fakes a no-progress tdr, runs dma and just after it checks the tdr didn't trigger yet (no-progress tdr is
// 5 minutes long, so we don't expect it to trigger)
TEST_F_SYN(TimeoutTests, noProgressNoTimeout)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    std::string skipReason;
    runDma(device, streamHandles, eventDown, skipReason, true);
    scal_comp_group_handle_t compGrp;
    scal_get_completion_group_handle_by_name(m_devScal->testOnlyGetScalHandle(), "pdma_tx_completion_queue0", &compGrp);
    scal_completion_group_set_expected_ctr(compGrp, 1);

    while (true)
    {
        synStatus status = synEventQuery(eventDown);
        if (status != synBusy)
        {
            LOG_INFO(SYN_RT_TEST, "Status is {}", status);
            break;
        }

        sleep(1);
    }

    sleep(3);  // Give the background thread some time to set the DFA flag, in case DFA is triggered by mistake (it
               // shouldn't set the flag in this test)

    ASSERT_EQ(m_devScal->getDfaStatus().hasError(DfaErrorCode::scalTdrFailed), false) << "Timeout not expected";
}

// This test sets a very small value to no-progress tdr, then fakes an expected value to 1 and runs a dma. We expect
// no-progress tdr.
// We disable the dfa so we can check the dma is done before the test is done
TEST_F_SYN(TimeoutTests, noProgressTimeout2)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    std::string skipReason;
    // set a very small timeout no progress
    runDma(device, streamHandles, eventDown, skipReason, true, scal_timeouts_t {SCAL_TIMEOUT_NOT_SET, 1});

    scal_comp_group_handle_t compGrp;
    scal_get_completion_group_handle_by_name(m_devScal->testOnlyGetScalHandle(), "pdma_tx_completion_queue0", &compGrp);

    while (true)
    {
        synStatus status = synEventQuery(eventDown);
        if (status != synBusy)
        {
            LOG_INFO(SYN_RT_TEST, "Status is {}", status);
            break;
        }
        sleep(1);
    }
    sleep(3);  // just to be on the safe side, give the background thread some time so to set the flag
    ASSERT_EQ(m_devScal->getDfaStatus().hasError(DfaErrorCode::scalTdrFailed), true) << "scal timeout expected";
}

// The test sets a very small no-progress time, and disables the timeout. Then it runs a dma copy. It expects no timeout
TEST_F_SYN(TimeoutTests, noProgressTimeoutDisabled)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    std::string skipReason;
    // set a very small timeout no progress
    runDma(device, streamHandles, eventDown, skipReason, true, scal_timeouts_t {SCAL_TIMEOUT_NOT_SET, 1}, true);

    scal_comp_group_handle_t compGrp;
    scal_get_completion_group_handle_by_name(m_devScal->testOnlyGetScalHandle(), "pdma_tx_completion_queue0", &compGrp);

    scal_completion_group_set_expected_ctr(compGrp, 1);

    while (true)
    {
        synStatus status = synEventQuery(eventDown);
        if (status != synBusy)
        {
            LOG_INFO(SYN_RT_TEST, "Status is {}", status);
            break;
        }
        sleep(1);
    }
    sleep(3);  // Give the background thread some time to set the DFA flag, in case DFA is triggered by mistake (it
               // shouldn't set the flag in this test)
    ASSERT_EQ(m_devScal->getDfaStatus().isSuccess(), true) << "scal timeout not expected";
}

// Test description
// - Disable tdr timeout (by setting to high value).
// - Set no-progress timeout to 5 seconds
// - Set expected on stream#1 (timeout should happen in 5 second)
// - Start measuring time
// - Wait for 3 seconds
// - Send work on stream#2. We expect stream #1 to timeout after 5 seconds from now (because the progress
//   on stream#2 restarts the time)
// - Wait for the timeout on stream 1. Measure the time. It should happen around 5 seconds after the work
//    we sent on stream#2 (8 seconds after we started measuring the time)
// - Make sure the error is at least 7 seconds (should close to 8) after we set the expected ctr for stream#1
//   sure we timeout only after timeout on the second stream
// Because this test runs with runDma(true) -> the dfa is disabled, and the code to check if the work was done is OK, so
// we don't expect the simulator to crash
TEST_F_SYN(TimeoutTests, noProgressTimeout2streams)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    std::string skipReason;
    // 100 seconds timeout == disabled
    runDma(device, streamHandles, eventDown, skipReason, true, scal_timeouts_t {100'000'000ul, 5'000'000});

    scal_comp_group_handle_t compGrp;
    scal_get_completion_group_handle_by_name(m_devScal->testOnlyGetScalHandle(), "pdma_tx_completion_queue0", &compGrp);
    scal_completion_group_set_expected_ctr(compGrp, 3);  // timeout should happen in 5 seconds

    auto start = TimeTools::timeNow();

    sleep(3);

    scal_comp_group_handle_t compGrp2;
    scal_get_completion_group_handle_by_name(m_devScal->testOnlyGetScalHandle(),
                                             "pdma_tx_completion_queue1",
                                             &compGrp2);
    scal_completion_group_set_expected_ctr(compGrp2, 2);  // timer should restart, 5 seconds from now

    m_streamDown  = streamHandles[0];
    m_streamDown2 = streamHandles[1];

    const auto& eventDown2 = device.createEvent(0);

    auto status = synMemCopyAsync(m_streamDown, (uint64_t)m_hostBuff, SIZE, m_devBuff, HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    status = synEventRecord(eventDown2, m_streamDown2);
    ASSERT_EQ(status, synSuccess) << "Failed to copy inputs to device memory";

    while (true)
    {
        status = synEventQuery(eventDown2);
        if (status != synBusy)
        {
            LOG_INFO(SYN_RT_TEST, "Status is {}", status);
            break;
        }

        usleep(100000);
    }

    while (m_devScal->getDfaStatus().isSuccess())
    {
        usleep(100000);
    }

    auto     timePassed = TimeTools::timeFromUs(start);
    uint64_t expectedTimeAtLeast =
        3 * 1000 * 1000 + 5 * 1000 * 1000 - 1000 * 1000;  // 3 seconds sleep + SCAL_TIMEOUT_NO_PROGRESS - 1

    ASSERT_GT(timePassed, expectedTimeAtLeast);
}

// This test sets a very small value for timeout, then calls runDma and expect it to die. Because we can't check if
// the dma is done before the death, we might crash the simulator
TEST_F_SYN(TimeoutTests, DISABLED_timeoutProcessDeath)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    LOG_ERR_T(SYN_API, "-----THIS TEST IS EXPECTED TO LOG ERRORS-----");

    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    std::string skipReason;
    // set a very small timeout
    ASSERT_DEATH(runDma(device, streamHandles, eventDown, skipReason, false, scal_timeouts_t {1, SCAL_TIMEOUT_NOT_SET}),
                 "");
}

// This test sets a very small value for timeout, sets busyWait mode and then calls runDma and expect it to loop
// forever. Because we can't check if the dma is done before the death, we might crash the simulator
TEST_F_SYN(TimeoutTests, DISABLED_timeoutProcessBusyWaitDeath)
{
    TestDevice                     device(m_deviceType);
    auto                           stream0 = device.createStream();
    auto                           stream1 = device.createStream();
    std::array<synStreamHandle, 2> streamHandles {stream0, stream1};
    auto                           eventDown = device.createEvent(0);

    LOG_ERR_T(SYN_API, "-----THIS TEST IS EXPECTED TO LOG ERRORS-----");

    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GCFG_TERMINATE_SYNAPSE_UPON_DFA.setValue((uint64_t)DfaSynapseTerminationState::busyWait);

    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    std::string skipReason;
    // set a very small timeout
    ASSERT_DEATH(runDma(device, streamHandles, eventDown, skipReason, false, scal_timeouts_t {1, SCAL_TIMEOUT_NOT_SET}),
                 "");
}
