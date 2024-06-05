#pragma once
#undef TEST_CONFIGURATION_SUPPORT_MEMBERS
#include "syn_base_test.hpp"
#include "habana_global_conf_runtime.h"
#include "synapse_api.h"
#include "synapse_common_types.h"

#include "runtime/common/device/dfa_observer.hpp"
#include "runtime/common/device/dfa_base.hpp"

#include "test_device.hpp"

#include <condition_variable>

enum class csTimeoutTestType
{
    NORMAL_TEST,
    DEATH_TEST,
    ACQUIRE_AFTER_RESET_TEST
};

class DeviceInterface;

class DeviceFailureTests
: public SynBaseTest
, public DfaObserver
{
public:
    DeviceFailureTests() { setSupportedDevices({synDeviceGaudi}); };

    ~DeviceFailureTests() {};

    void TearDown() override;


    bool notifyFailureObserved();
    void undefinedOpcodeTest();
    void mmuPagefaultTest();
    void assertAsyncDuringLaunch();
    void awaitSynapseTermination(DeviceInterface* device, synDeviceId deviceId, DfaErrorCode dfaError);
    void awaitTestDeath();

    mutable std::condition_variable m_cv;
    mutable std::mutex              m_mutex;
    bool                            m_notified;

private:
    void launchCorruptedRecipe(TestDevice& myDevice, uint32_t opCode);
    void csTimeoutTest(csTimeoutTestType type);
    void csDmaTimeoutTest(bool isDeathTest);

    FRIEND_TEST(DeviceFailureTests, DEATH_TEST_cs_dma_timeout);
    FRIEND_TEST(DeviceFailureTests, DEATH_TEST_cs_timeout);
    FRIEND_TEST(DeviceFailureTests, DEATH_TEST_cs_timeout_busy_wait_death);
    FRIEND_TEST(DeviceFailureTests, DEATH_TEST_cs_timeout_death);
    FRIEND_TEST(DeviceFailureTests, DEATH_TEST_cs_timeout_acquire_after_reset);
};