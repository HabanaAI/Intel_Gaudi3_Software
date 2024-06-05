#pragma once
#include "tpc_kernel_lib_interface.h"
#include "tpc_kernel_lib_interface_private.h"
#include <gc_protocol.hpp>

#ifdef __cplusplus
extern "C" {
#endif

tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* params,
                                                    tpc_lib_api::NodeDataLayouts*          supportedLayouts,
                                                    uint32_t*                              layoutCount);

tpc_lib_api::GlueCodeReturn GetFunctionalComplexGuids(tpc_lib_api::DeviceId deviceId,
                                                      unsigned* guidCount,
                                                      tpc_lib_api::GuidInfo* guids);

tpc_lib_api::GlueCodeReturn GetPerformanceComplexGuids(tpc_lib_api::DeviceId deviceId,
                                                       unsigned* guidCount,
                                                       tpc_lib_api::GuidInfo* guids);

tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams*     params,
                                                     tpc_lib_api::TensorManipulationSuggestion* suggestion);

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams*  params,
                                                 tpc_lib_api::HabanaKernelInstantiation* instance);

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId              deviceId,
                                              tpc_lib_api::ShapeInferenceParams* inputParams,
                                              tpc_lib_api::ShapeInferenceOutput* outputData);

uint64_t GetLibVersion();

tpc_lib_api::GlueCodeReturn ExtractFunctionalComplexGUID(const gc_protocol::ProtocolGraph* inputGraph,
                                                         gc_protocol::ProtocolGraph**      outputGraph);

tpc_lib_api::GlueCodeReturn ExtractPerformanceComplexGUID(const gc_protocol::ProtocolGraph* inputGraph,
                                                          gc_protocol::ProtocolGraph**      outputGraph);

#ifdef __cplusplus
}

#endif
