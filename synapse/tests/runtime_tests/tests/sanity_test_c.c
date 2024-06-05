#include "sanity_test_c.h"
#include <stdio.h>
/*
    C sanity test
    1. check that synapse_api.h is C compatible
    2. run simple api test in C
 */
#define ASSERT_EQ(actual, expected, message)                                                                           \
    if (actual != expected)                                                                                            \
    {                                                                                                                  \
        fprintf(stderr, "%d %s actual %u expected %u\n", __LINE__, message, (uint32_t)actual, (uint32_t)expected);     \
        return synFail;                                                                                                \
    }

#define ASSERT_LT(actual, expected, message)                                                                           \
    if ((uint32_t)actual >= (uint32_t)expected)                                                                        \
    {                                                                                                                  \
        fprintf(stderr, "%d %s actual %u expected %u\n", __LINE__, message, (uint32_t)actual, (uint32_t)expected);     \
        return synFail;                                                                                                \
    }

synStatus run_sanity_test_c(synDeviceId deviceId, synDeviceType deviceType)
{
    synStatus     status;
    synDeviceInfo deviceInfo[2];
    deviceInfo[0].fd = UINT32_MAX;
    status           = synDeviceGetInfo(deviceId, deviceInfo);
    ASSERT_EQ(status, synSuccess, "Failed to acquire any device");
    ASSERT_EQ(deviceInfo[0].deviceType, deviceType, "device type mismatch");
    ASSERT_LT(deviceInfo[0].fd, UINT32_MAX, "Failed to set fd");
    status = synDeviceRelease(deviceId);
    ASSERT_EQ(status, synSuccess, "Failed to release device");

    status = synDeviceAcquire(&deviceId, NULL);
    ASSERT_EQ(status, synSuccess, "Failed to acquire device");
    deviceInfo[1].fd = UINT32_MAX;
    status           = synDeviceGetInfo(deviceId, deviceInfo + 1);
    ASSERT_EQ(status, synSuccess, "Failed to acquire any device");
    ASSERT_EQ(deviceInfo[1].deviceType, deviceType, "device type mismatch");
    ASSERT_LT(deviceInfo[1].fd, UINT32_MAX, "Failed to set fd");
    return status;
}
