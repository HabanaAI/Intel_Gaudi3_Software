#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "habana_global_conf_runtime.h"
#include "../infra/test_types.hpp"
#include "synapse_api.h"

class EventLifecycle : public SynBaseTest
{
public:
    EventLifecycle() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
};

REGISTER_SUITE(EventLifecycle, ALL_TEST_PACKAGES);

TEST_F_SYN(EventLifecycle, eventCreateDestroy)
{
    TestDevice device(m_deviceType);

    const uint32_t  numEvents      = GCFG_NUM_OF_USER_STREAM_EVENTS.value();
    synEventHandle* eventHandleArr = new synEventHandle[numEvents];
    synEventHandle  extraEvent     = (synEventHandle)8;  // put a bad value to verify that when synEventCreate
                                                         // fails it puts null in it
    // Create max number of events
    for (int i = 0; i < numEvents; i++)
    {
        auto status = synEventCreate(&eventHandleArr[i], device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create event";
    }

    // try to create another one, should fail
    auto status = synEventCreate(&extraEvent, device.getDeviceId(), 0);
    ASSERT_EQ(status, synAllResourcesTaken) << "Should fail to create event";
    ASSERT_EQ(extraEvent, nullptr) << "extraEvent should be nullptr";

    // Destroy all events
    for (int i = 0; i < numEvents; i++)
    {
        status = synEventDestroy(eventHandleArr[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy event";
    }

    // Create again max number of events
    for (int i = 0; i < numEvents; i++)
    {
        status = synEventCreate(&eventHandleArr[i], device.getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create event";
    }

    // Destroy all events
    for (int i = 0; i < numEvents; i++)
    {
        status = synEventDestroy(eventHandleArr[i]);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy event";
    }

    delete[] eventHandleArr;
}
