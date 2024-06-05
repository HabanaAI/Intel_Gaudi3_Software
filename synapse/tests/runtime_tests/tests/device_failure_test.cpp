#include "device_failure_test.hpp"

#include "habana_global_conf_runtime.h"
#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "syn_singleton.hpp"

#include "runtime/common/device/dfa_base.hpp"

#include "runtime/common/device/device_interface.hpp"
#include "runtime/common/streams/stream.hpp"
#include "gaudi/gaudi_packets.h"

#include "hlthunk.h"

#include "test_launcher.hpp"
#include "test_config.hpp"
#include "test_recipe_assert_async.hpp"
#include "test_recipe_nop_x_nodes.hpp"
#include <chrono>
#include <vector>

static const uint64_t TENSOR_TO_ADDR_FACTOR = 0x10000;
static const uint64_t WAIT_FOR_DFA_TIMEOUT  = 100;  // Timeout to wait for notify from DFA in seconds
static const int      NUM_TRY               = 20;
static const int      SLEEP_TIME            = 3;

REGISTER_SUITE(DeviceFailureTests, synTestPackage::DEATH);

void DeviceFailureTests::TearDown()
{
    // Do not call synDestroy !!
    //  we expect a DFA crash and synDestroy will just fail.
    // Do not call SynBaseTest::TearDown();
}

bool DeviceFailureTests::notifyFailureObserved()
{
    std::unique_lock<std::mutex> lck(m_mutex);
    m_notified = true;
    m_cv.notify_one();
    return true;
}

void DeviceFailureTests::launchCorruptedRecipe(TestDevice& myDevice, uint32_t opCode)
{
    synDeviceId deviceId = myDevice.getDeviceId();
    // we will create a stream but will not release it
    // as this test crashes by design (and the release fails after the crash)
    synStreamHandle streamHandle = 0;
    synStatus       status       = synStreamCreateGeneric(&streamHandle, deviceId, 0);
    ASSERT_EQ(status, synSuccess);

    TestRecipeNopXNodes recipe(m_deviceType);
    recipe.generateRecipe();
    TestLauncher launcher(myDevice);

    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});
    // corrupt the recipe to cause DFA
    recipe_t* r                                     = recipe.getRecipe()->basicRecipeHandle.recipe;
    ((packet_msg_short*)(r->blobs[0].data))->opcode = opCode;

    status = synLaunchExt(streamHandle,
                          recipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                          recipe.getTensorInfoVecSize(),
                          recipeLaunchParams.getWorkspace(),
                          recipe.getRecipe(),
                          0 /* flags */);
    ASSERT_EQ(status, synSuccess) << "Failed to synLaunch";
    sleep(10);
}

void DeviceFailureTests::awaitSynapseTermination(DeviceInterface* device, synDeviceId deviceId, DfaErrorCode dfaError)
{
    LOG_INFO(SYN_TEST, "Test awaits termination!");
    bool hasDfaError = false;

    std::unique_lock<std::mutex> lck(m_mutex);
    if (m_cv.wait_for(lck, std::chrono::seconds(WAIT_FOR_DFA_TIMEOUT), [&]() { return m_notified; }))
    {
        if (device->getDfaStatus().hasError(dfaError))
        {
            hasDfaError = true;
        }
    }
    ASSERT_EQ(hasDfaError, true) << "DFA had not returned expected error record-type, waiting time "
                                 << WAIT_FOR_DFA_TIMEOUT;

    // Allow DFA to collect info before killing the process
    sleep(3);

    LOG_INFO(SYN_TEST, "Test had been Completed!");

    if (GCFG_TERMINATE_SYNAPSE_UPON_DFA.value() == (uint64_t)DfaSynapseTerminationState::busyWait)
    {
        LOG_INFO(SYN_TEST, "Terminating test due to busy-wait!");

        if (0 == kill(getpid(), SIGKILL))
        {
            LOG_INFO(SYN_TEST, "Sent SIGKILL to process");
            synapse::LogManager::instance().flush();
        }
        else
        {
            LOG_CRITICAL(SYN_TEST, "The sigkill execution failed");
        }
    }
}

void DeviceFailureTests::awaitTestDeath()
{
    usleep(DfaBase::SLEEP_BETWEEN_SYNAPSE_TERMINATION_AND_KILL_IN_US * 1.5);
}

void DeviceFailureTests::csDmaTimeoutTest(bool isDeathTest)
{
    constexpr uint64_t size = 1024 * 1024 * 1024;

    synStatus   status;
    TestDevice  myDevice(m_deviceType);
    synDeviceId deviceId = myDevice.getDeviceId();

    std::shared_ptr<DeviceInterface> device = _SYN_SINGLETON_INTERNAL->getDevice();
    ASSERT_NE(device, nullptr);

    m_notified = false;
    device->addDfaObserver(this);

    synStreamHandle downStream;

    status = synStreamCreateGeneric(&downStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess);

    void*    inBuff;
    uint64_t hbmBuff;

    status = synHostMalloc(deviceId, size, 0, &inBuff);
    ASSERT_EQ(status, synSuccess);

    status = synDeviceMalloc(deviceId, size, 0, 0, &hbmBuff);
    ASSERT_EQ(status, synSuccess);

    status = synMemCopyAsync(downStream, (uint64_t)inBuff, size, hbmBuff, synDmaDir::HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess);

    awaitSynapseTermination(device.get(), deviceId, DfaErrorCode::csTimeout);

    if (isDeathTest)
    {
        awaitTestDeath();
    }
}

void DeviceFailureTests::csTimeoutTest(csTimeoutTestType type)
{
    TestDevice  myDevice(m_deviceType);
    synDeviceId deviceId = myDevice.getDeviceId();

    std::shared_ptr<DeviceInterface> device = _SYN_SINGLETON_INTERNAL->getDevice();

    m_notified = false;
    device->addDfaObserver(this);

    // create a recipe and launch it
    launchCorruptedRecipe(myDevice, PACKET_STOP);
    // This mode get the device to reset state since the CS is still active
    if (type == csTimeoutTestType::ACQUIRE_AFTER_RESET_TEST)
    {
        LOG_INFO(SYN_TEST, "Terminating test for device reset state");
        if (0 == kill(getpid(), SIGKILL))
        {
            LOG_INFO(SYN_TEST, "Sent SIGKILL to process");
            synapse::LogManager::instance().flush();
        }
        else
        {
            LOG_CRITICAL(SYN_TEST, "The sigkill execution failed");
        }
    }

    awaitSynapseTermination(device.get(), deviceId, DfaErrorCode::csTimeout);

    if (type == csTimeoutTestType::DEATH_TEST)
    {
        awaitTestDeath();
    }
}

void DeviceFailureTests::mmuPagefaultTest()
{
    TestDevice  myDevice(m_deviceType);
    synDeviceId deviceId = myDevice.getDeviceId();

    std::shared_ptr<DeviceInterface> device = _SYN_SINGLETON_INTERNAL->getDevice();

    m_notified = false;
    device->addDfaObserver(this);

    // Host buffers for input & output
    uint8_t*       input;
    const uint64_t size = 1024 * 1024 * 1024;

    // Input and output need to be mapped to the device as they are copied from / to
    synStatus status = synHostMalloc(deviceId, size, 0, (void**)&input);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for input";

    // Create streams
    synStreamHandle copyInStream;
    status = synStreamCreateGeneric(&copyInStream, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Could not create stream to copy data to the device";

    // Device-side (HBM) buffers for input and output
    uint64_t deviceAddress;
    status = synDeviceMalloc(deviceId, size, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate input buffer in device memory";

    uint64_t    deviceId64 = deviceId;
    std::thread executionThread([deviceId64, copyInStream, input, deviceAddress, size]() {
        // Awaits that main thread will start waiting for the termination-notification
        sleep(1);

        uint16_t numOfCopyOperations               = 1000;
        uint64_t srcAddresses[numOfCopyOperations] = {(uint64_t)input};
        uint64_t dstAddresses[numOfCopyOperations] = {deviceAddress};
        uint64_t copySize[numOfCopyOperations]     = {size};

        synMemCopyAsyncMultiple(copyInStream, srcAddresses, copySize, dstAddresses, HOST_TO_DRAM, numOfCopyOperations);

        synHostUnmap(deviceId64, input);
    });

    awaitSynapseTermination(device.get(), deviceId, DfaErrorCode::mmuPageFault);
    executionThread.join();

    // By definition, in Gaudi, this test will result by SIGKILL (whether by the LKD or the Synapse)
    awaitTestDeath();
}

void DeviceFailureTests::undefinedOpcodeTest()
{
    TestDevice  myDevice(m_deviceType);
    synDeviceId deviceId = myDevice.getDeviceId();

    std::shared_ptr<DeviceInterface> device = _SYN_SINGLETON_INTERNAL->getDevice();

    m_notified = false;
    device->addDfaObserver(this);

    uint32_t opCode;
    switch (m_deviceType)
    {
        case synDeviceGaudi:
            opCode = 0x10;
            break;
        case synDeviceGaudi2:
            opCode = 0x1F;
            break;
        default:
            GTEST_SKIP() << "Device not supported for test";
            break;
    }

    // create a recipe and launch it
    launchCorruptedRecipe(myDevice, opCode);

    awaitSynapseTermination(device.get(), deviceId, DfaErrorCode::undefinedOpCode);
    awaitTestDeath();
}

void DeviceFailureTests::assertAsyncDuringLaunch()
{
    TestRecipeAssertAsync recipe(m_deviceType);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    RecipeLaunchParams launchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::ALL_ZERO, 0});

    TestLauncher::download(stream, recipe, launchParams);
    TestLauncher::launch(stream, recipe, launchParams);

    // wait for assert async event to arrive
    usleep(300 * 1000);

    stream.synchronize();

    awaitTestDeath();
}

TEST_F_SYN(DeviceFailureTests, DEATH_TEST_cs_timeout)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GCFG_TERMINATE_SYNAPSE_UPON_DFA.setValue((uint64_t)DfaSynapseTerminationState::disabled);

    csTimeoutTest(csTimeoutTestType::NORMAL_TEST);
}

// Note: this tests is doing a big dma and assumes it will timeout. It will happen only if the driver starts with a
//       small timeout value and on simulator (-p "timeout_locked=2"). This is the reason it is disabled by default
TEST_F_SYN(DeviceFailureTests, DEATH_TEST_cs_dma_timeout)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GCFG_TERMINATE_SYNAPSE_UPON_DFA.setValue((uint64_t)DfaSynapseTerminationState::disabled);

    csDmaTimeoutTest(false);
}

TEST_F_SYN(DeviceFailureTests, DEATH_TEST_cs_timeout_busy_wait_death)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GCFG_TERMINATE_SYNAPSE_UPON_DFA.setValue((uint64_t)DfaSynapseTerminationState::busyWait);

    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    ASSERT_DEATH(csTimeoutTest(csTimeoutTestType::DEATH_TEST), "");
}

TEST_F_SYN(DeviceFailureTests, DEATH_TEST_cs_timeout_death)
{
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    ASSERT_DEATH(csTimeoutTest(csTimeoutTestType::DEATH_TEST), "");
}

TEST_F_SYN(DeviceFailureTests, DEATH_TEST_cs_timeout_acquire_after_reset)
{
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    ASSERT_DEATH(csTimeoutTest(csTimeoutTestType::ACQUIRE_AFTER_RESET_TEST), "");
}

TEST_F_SYN(DeviceFailureTests, DEATH_TEST_mmu_pagefault_death)
{
    synapse::LogManager::instance().enablePeriodicFlush(false);
    // By definition, in Gaudi, this test will result by SIGKILL (whether by the LKD or the Synapse)
    ASSERT_DEATH(mmuPagefaultTest(), "");
}

TEST_F_SYN(DeviceFailureTests, DEATH_TEST_undefined_op_code, {synDeviceGaudi, synDeviceGaudi2})
{
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    ASSERT_DEATH(undefinedOpcodeTest(), "");
}

TEST_F_SYN(DeviceFailureTests, DEATH_TEST_assert_async_during_launch, {synDeviceGaudi})
{
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    ASSERT_DEATH(assertAsyncDuringLaunch(), "");
}