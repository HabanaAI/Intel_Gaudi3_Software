#ifndef SHARED_LAYER_AGENT_API_H
#define SHARED_LAYER_AGENT_API_H

#include "shared_layer_api.hpp"

/* This function initializes the Shared Layer.
 *
 *@return SHARED_LAYER_SUCCESS for success, SHARED_LAYER_FAILED for failure
 */
extern "C" SharedLayer::Return_t HLSL_init();

/* HLSL_GetKernelNames returns the list of all the Ops for supported for a specific
 * device.
 * @return 0 for success, negative value for failure
 */
extern "C"
SharedLayer::Return_t
HLSL_GetKernelNames(char**                      guidNameArray,
                    int*                        guidCount,
                    const SharedLayer::DeviceId device);


/* HLSL_validate_guid checks if a specific guid exists in the Ops Database.
 * The function also performs parameters and attributes validation.
 * @param in guidName the guid which needs to be validated
 * @param in ctx structure containing the instantiated guid parameter values
 *
 * @return 0 for success, integer value other than 0 depicting the respective
 * err condition
 */
extern "C"
SharedLayer::Return_t
HLSL_validate_guid(const SharedLayer::Params_t* const params);

#endif // SHARED_LAYER_AGENT_API_H