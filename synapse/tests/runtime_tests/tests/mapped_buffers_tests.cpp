#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "test_device.hpp"

class MappedBufferTest : public SynBaseTest
{
public:
    MappedBufferTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
};

REGISTER_SUITE(MappedBufferTest, ALL_TEST_PACKAGES);

TEST_F_SYN(MappedBufferTest, mapBufferTwrice)
{
    const unsigned elemSize         = 1024;
    char           buffer[elemSize] = {};

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
}
