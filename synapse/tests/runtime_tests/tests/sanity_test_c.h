#pragma once

#include "synapse_api.h"
/* sanity C test to make sure we do not break C compatibility */
#ifdef __cplusplus
extern "C" {
#endif
synStatus run_sanity_test_c(synDeviceId deviceId, synDeviceType deviceType);

#ifdef __cplusplus
}
#endif