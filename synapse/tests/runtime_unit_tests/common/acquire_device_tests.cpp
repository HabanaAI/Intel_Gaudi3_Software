#include <gtest/gtest.h>

#define protected public
#define private public
#include "runtime/common/osal/osal.hpp"

#include <unordered_map>
#include <string>

#include "runtime/common/device/device_manager.hpp"
#include "synapse_common_types.h"

class UTDeviceAcquireTest : public ::testing::Test
{
public:
private:
};

TEST_F(UTDeviceAcquireTest, device_type_identification)
{
    const std::unordered_map<std::string, synDeviceType> deviceNameToDeviceTypeDB = DeviceManager::_stringToDeviceType();

    std::unordered_map<std::string, synDeviceType> deviceNamesToExpectedDeviceType = {
        {"GAUDI", synDeviceGaudi},
        {"GAUDI Simulator", synDeviceGaudi},
        {"GAUDI2", synDeviceGaudi2},
        {"GAUDI 2", synDeviceGaudi},
        {"GAUDI SEC", synDeviceGaudi},
        {"GAUDI SEC Simulator", synDeviceGaudi},
        {"GAUDI Simulator SEC", synDeviceGaudi}};

    for (auto singleDeviceNameToExpectedDeviceType : deviceNamesToExpectedDeviceType)
    {
        uint32_t outCount[synDeviceTypeSize];
        unsigned numberOfDevices      = 0;
        bool     shouldStripSimulator = true;

        memset(outCount, 0, synDeviceTypeSize * sizeof(uint32_t));

        OSAL::getInstance()._updateDeviceCount(outCount,
                                               numberOfDevices,
                                               deviceNameToDeviceTypeDB,
                                               singleDeviceNameToExpectedDeviceType.first,
                                               shouldStripSimulator);

        ASSERT_EQ(numberOfDevices, 1);

        for (unsigned deviceType = 0; deviceType < synDeviceTypeSize; deviceType++)
        {
            if (deviceType == singleDeviceNameToExpectedDeviceType.second)
            {
                ASSERT_EQ(outCount[deviceType], 1);
            }
            else
            {
                ASSERT_EQ(outCount[deviceType], 0);
            }
        }
    }
}
