#include "syn_base_dfa_test.hpp"

#include "synapse_api.h"
#include "habana_global_conf_runtime.h"
#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"  // ?
#include "syn_singleton.hpp"

#include "stream_base_scal.hpp"  //??
#include "device_scal.hpp"

#include "runtime/common/device/device_interface.hpp"
#include "runtime/common/streams/stream.hpp"
#include "runtime/scal/common/entities/scal_stream_copy_interface.hpp"
#include "gaudi2/gaudi2.h"

#include "test_device.hpp"

#include "hlthunk.h"

#include <chrono>
#include <vector>

using namespace std::literals::chrono_literals;

static const uint64_t TENSOR_TO_ADDR_FACTOR = 0x10000;
static const uint64_t WAIT_FOR_DFA_TIMEOUT  = 100;  // Timeout to wait for notify from DFA in seconds
static const int      NUM_TRY               = 20;

class DfaDevCrashTests : public SynBaseDfaTest
{
public:
    enum AccessMethod
    {
        LBWWRITE,
        MEMCOPY
    };

    enum FailureType
    {
        PAGE_FAULT,
        RAZWI
    };

    void razwiAndMmuPageFault(bool razwiOnly);
    void fakeMmuPageFault(synDeviceType deviceType, bool razwiOnly);

    void runAndExpectFailure(AccessMethod method, FailureType expectedFailure, uint64_t address);
    void triggerFailure(AccessMethod method, uint64_t address);

    void MemCopyToAddress(TestDevice& dev, uint64_t address);
    void LBWWriteToAddress(TestDevice& dev, uint64_t address);
};

REGISTER_SUITE(DfaDevCrashTests, synTestPackage::DEATH);

void DfaDevCrashTests::fakeMmuPageFault(synDeviceType deviceType, bool razwiOnly)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    // After razwi/page-fault the device is operational so the script starts the test but the device still fails
    // to acquire as LKD didn't reset the device yet. Give LKD some time to reset the device
    TestDevice dev(m_deviceType, NUM_TRY, 3ms);

    const uint64_t size = 1024 * 1024 * 1024;

    auto streamDown  = dev.createStream();
    auto hostBuff    = dev.allocateHostBuffer(size, 0);
    auto hostBuffMap = dev.mapHostBuffer(hostBuff.getBuffer(), size);
    auto devBuff     = dev.allocateDeviceBuffer(size, 0);

    if (razwiOnly)
    {
        synDeviceInfo devInfo;
        ASSERT_EQ(synDeviceGetInfo(dev.getDeviceId(), &devInfo), synSuccess);

        uint64_t badAddr = 0x7FFC000000ull + PCIE_FW_SRAM_ADDR; // Taken from LKD test

        streamDown.memcopyAsync((uint64_t)hostBuff.getBuffer(),
                                size,
                                badAddr,
                                HOST_TO_DRAM);
    }
    else  // this will cause razwi+mmu page fault
    {
        for (int i = 0; i < 10; i++)
        {
            streamDown.memcopyAsync((uint64_t)hostBuff.getBuffer(),
                                    size,
                                    devBuff.getBuffer(),  // make the address outside the dram
                                    HOST_TO_DRAM);
        }
        hostBuffMap.unmap();
    }

    sleep(10);  // sleep for some time as it might take time until we get the event-fd
}

void DfaDevCrashTests::razwiAndMmuPageFault(bool razwiOnly)
{
    // google test needs only one thread running (close spdlog thread)
    synapse::LogManager::instance().enablePeriodicFlush(false);

    ASSERT_EXIT(fakeMmuPageFault(m_deviceType, razwiOnly), ::testing::KilledBySignal(SIGKILL), DFA_KILL_MSG);

    synapse::LogManager::instance().flush();

    std::vector<ExpectedWords> expected = {
        // Each should be log twice, once for oldest, one when showing all work
        {"usrEngineErr", 1, std::equal_to<uint32_t>()},
        {"deviceReset", 1, std::equal_to<uint32_t>()},
        {"event Id RAZWI", 1, std::equal_to<uint32_t>()},
        {"Device Mapping", 1, std::equal_to<uint32_t>()},
        {"SW Mapped memory", 1, std::equal_to<uint32_t>()},
        {"Mapper name", 1, std::greater_equal<uint32_t>()},
    };

    if (!razwiOnly)
    {
        expected.push_back({"event Id PAGE_FAULT", 1, std::equal_to<uint32_t>()});
        expected.push_back({"dev-addr/size", 1, std::greater_equal<uint32_t>()});
    }

    expectedInFile(DfaLoggerEnum::loggerSynDevFail, expected);

    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}

void DfaDevCrashTests::runAndExpectFailure(AccessMethod method, FailureType expectedFailure, uint64_t address)
{
    ASSERT_EXIT(triggerFailure(method, address), ::testing::KilledBySignal(SIGKILL), DFA_KILL_MSG);

    // verify expected info in DFA
    synapse::LogManager::instance().flush();
    std::vector<ExpectedWords> expected = {{"deviceReset", 1, std::equal_to<uint32_t>()},
                                           {"Device Mapping", 1, std::equal_to<uint32_t>()},
                                           {"SW Mapped memory", 1, std::equal_to<uint32_t>()},
                                           {"Mapper name", 1, std::greater_equal<uint32_t>()}};

    switch (expectedFailure)
    {
        case FailureType::RAZWI:
            expected.push_back({"event Id RAZWI", 1, std::equal_to<uint32_t>()});
            break;
        case FailureType::PAGE_FAULT:
            expected.push_back({"event Id PAGE_FAULT", 1, std::equal_to<uint32_t>()});
            expected.push_back({"dev-addr/size", 1, std::greater_equal<uint32_t>()});
            break;
    }
    expectedInFile(DfaLoggerEnum::loggerSynDevFail, expected);

    if (!::testing::Test::HasFailure())
    {
        removeTestDfaFiles();
    }
}

void DfaDevCrashTests::triggerFailure(AccessMethod method, uint64_t address)
{
    // google test needs only one thread running (close spdlog thread)
    synapse::LogManager::instance().enablePeriodicFlush(false);

    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("DFA_READ_REG_MODE", std::to_string((uint64_t)ReadRegMode::skip).c_str());

    // After razwi/page-fault the device is operational so the script starts the test but the device still fails
    // to acquire as LKD didn't reset the device yet. Give LKD some time to reset the device
    TestDevice dev(m_deviceType, 10, 6s);

    // access cases
    switch (method)
    {
        case AccessMethod::LBWWRITE:
            LBWWriteToAddress(dev, address);
            break;
        case AccessMethod::MEMCOPY:
            MemCopyToAddress(dev, address);
            break;
    }
}

void DfaDevCrashTests::LBWWriteToAddress(TestDevice& dev, uint64_t address)
{
    auto streamDown = dev.createStream();

    QueueInterface* pQueueInterface;
    // get streamHandle - the way of getting it depends on whether multi operation stream is used or nut.
    // we use a multi-operation stream, so to get the "actual" stream handle, we must get it using pStream
    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();
    DeviceCommon*                    device          = (DeviceCommon*)(deviceInterface.get());
    Stream*                          streamHandleDown;
    {
        auto streamSptr = device->loadAndValidateStream(streamDown, __FUNCTION__);
        ASSERT_NE(streamSptr, nullptr) << "Failed to load Stream";
        streamHandleDown = streamSptr.get();
    }
    streamHandleDown->testGetQueueInterface(QUEUE_TYPE_COPY_HOST_TO_DEVICE, pQueueInterface);
    HB_ASSERT_PTR(pQueueInterface);

    QueueBaseScalCommon* streamBaseScal = dynamic_cast<QueueBaseScalCommon*>(pQueueInterface);

    ScalStreamCopyInterface* scalStream = streamBaseScal->getScalStream();

    synStatus status = scalStream->addLbwWrite(address, (uint32_t)0, true, true, false);
    ASSERT_EQ(status, synSuccess);

    sleep(10);  // sleep for some time as it might take time until we get the event-fd
}

void DfaDevCrashTests::MemCopyToAddress(TestDevice& dev, uint64_t address)
{
    const uint64_t size        = 1024 * 1024 * 1024;
    auto           streamDown  = dev.createStream();
    auto           hostBuff    = dev.allocateHostBuffer(size, 0);
    auto           hostBuffMap = dev.mapHostBuffer(hostBuff.getBuffer(), size);

    streamDown.memcopyAsync((uint64_t)hostBuff.getBuffer(),
                            size,
                            address,  // make the address outside the dram
                            HOST_TO_DRAM);
    streamDown.synchronize();

    sleep(10);  // sleep for some time as it might take time until we get the event-fd
}

TEST_F_SYN(DfaDevCrashTests, DEATH_TEST_razwiAndMmuPageFault, {synDeviceGaudi})
{
    razwiAndMmuPageFault(false);
}

TEST_F_SYN(DfaDevCrashTests,
           DEATH_TEST_razwiOnly,
           {synDeviceGaudi})
{
    razwiAndMmuPageFault(true);
}

TEST_F_SYN(DfaDevCrashTests, DEATH_TEST_gaudi2_LBW_Razwi, {synDeviceGaudi2})
{
    uint64_t address = 0xFD456000;  // mmNIC0_TX_AXUSER_BASE;
    runAndExpectFailure(AccessMethod::LBWWRITE, FailureType::RAZWI, address);
}

TEST_F_SYN(DfaDevCrashTests, DEATH_TEST_gaudi2_LBW_PageFault, {synDeviceGaudi2})
{
    uint64_t address = 0x0;  // address which doesn't exist
    runAndExpectFailure(AccessMethod::LBWWRITE, FailureType::PAGE_FAULT, address);
}

// NOTICE: this test fails.
// LKD should fix it (ticket: SW-119146 - device won't reset)
TEST_F_SYN(DfaDevCrashTests, DEATH_TEST_gaudi2_MemCopy_Razwi, {synDeviceGaudi2})
{
    uint64_t address = CFG_BASE;  // not an address we can write to using memcopy
    runAndExpectFailure(AccessMethod::MEMCOPY, FailureType::RAZWI, address);
}

TEST_F_SYN(DfaDevCrashTests, DEATH_TEST_gaudi2_MemCopy_PageFault, {synDeviceGaudi2})
{
    uint64_t address = 0x1;  // address which doesn't exist (and isn't 0)
    runAndExpectFailure(AccessMethod::MEMCOPY, FailureType::PAGE_FAULT, address);
}
