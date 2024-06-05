#include "syn_base_test.hpp"
#include "synapse_api.h"
#include <future>

class SynGaudiDeviceInfoTest : public SynBaseTest
{
public:
    SynGaudiDeviceInfoTest() : SynBaseTest()
    {
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
    }
    ~SynGaudiDeviceInfoTest() = default;

    void acquireAndReleaseDevice(uint32_t threadID);
};
REGISTER_SUITE(SynGaudiDeviceInfoTest, ALL_TEST_PACKAGES);

void SynGaudiDeviceInfoTest::acquireAndReleaseDevice(uint32_t threadID)
{
    uint32_t deviceId;
    usleep(1);  // let all threads a chance to wake up

    unsigned loop    = 0;
    unsigned maxIter = 1000;

    // every thread should be able to acquire a device once
    while (loop < maxIter)
    {
        synStatus ret = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
        if (ret == synSuccess)
        {
            ret = synDeviceRelease(deviceId);
            ASSERT_EQ(ret, synSuccess) << "Failed to release deviceType " << m_deviceType;
            break;
        }
        usleep(20);
        loop++;
    }

    ASSERT_TRUE(loop != maxIter && "reached max loop and not all devices released");
}

TEST_F_SYN(SynGaudiDeviceInfoTest, devAcquire)
{
    // Normal acquire
    unsigned deviceId;
    ASSERT_EQ(synDeviceAcquire(&deviceId, nullptr), synSuccess) << "Failed to acquire device";
    unsigned dummy;
    ASSERT_EQ(synDeviceAcquire(&dummy, nullptr), synDeviceAlreadyAcquired) << "Should fail to acquire device";
    ASSERT_EQ(synDeviceRelease(deviceId), synSuccess) << "Failed to release device";

    // By device type
    ASSERT_EQ(synDeviceAcquireByDeviceType(&deviceId, m_deviceType), synSuccess)
        << "Acquire by device type: Failed to acquire type " << m_deviceType;

    ASSERT_EQ(synDeviceAcquireByDeviceType(&dummy, m_deviceType), synDeviceAlreadyAcquired)
        << "Acquire by device type: should fail";

    ASSERT_EQ(synDeviceRelease(deviceId), synSuccess) << "Failed to release device";
}

TEST_F_SYN(SynGaudiDeviceInfoTest, parallel_acquire_and_release)
{
    const unsigned                 numberOfThreads = 20;
    std::vector<std::future<void>> basicTestThreads;

    for (unsigned threadIndex = 0; threadIndex < numberOfThreads; threadIndex++)
    {
        std::future<void> thread = std::async(&SynGaudiDeviceInfoTest::acquireAndReleaseDevice, this, threadIndex);
        basicTestThreads.push_back(std::move(thread));
    }

    for (unsigned threadIndex = 0; threadIndex < basicTestThreads.size(); threadIndex++)
    {
        basicTestThreads[threadIndex].wait();
    }
}