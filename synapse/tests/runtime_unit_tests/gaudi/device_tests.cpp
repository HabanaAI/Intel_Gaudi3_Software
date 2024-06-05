#include <cstdint>
#include <gtest/gtest.h>
#include "synapse_common_types.h"
#include "runtime/qman/gaudi/device_gaudi.hpp"

class UTGaudiDeviceTest : public ::testing::Test
{
};

// Todo Enable this test (at the moment used for debug arb settings)
TEST_F(UTGaudiDeviceTest, DISABLED_acquire_release)
{
    const synDeviceInfo deviceInfo {};
    const int           fdControl = 1;
    std::atomic<bool>   deviceBeingReleased = false;

    DeviceConstructInfo deviceConstructInfo {.deviceInfo = deviceInfo, .fdControl = fdControl, .hlIdx = 0};

    DeviceGaudi dev(deviceConstructInfo);

    synStatus status = dev.acquire(100);
    ASSERT_EQ(status, synSuccess);

    status = dev.release(deviceBeingReleased);
    ASSERT_EQ(status, synSuccess);
}
