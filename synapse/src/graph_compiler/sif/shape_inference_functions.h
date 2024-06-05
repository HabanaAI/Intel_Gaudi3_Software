#pragma once

#include "types.h"

SifReturn dmaMemcpyShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dmaMemsetShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn concatenateShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn flattenShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn splitShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn expandDimsShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn mergeShapesInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn sliceShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn sliceAxisShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn sliceBackwardShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn sliceInsertShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn reshapeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn staticReshapeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn broadcastShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn identityShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn reductionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn stridedViewShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn stridedInsertShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn tensorViewShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn transposeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn convolutionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn convDeDwShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn convDeDxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn gemmShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn gemmDeDwShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn gemmDeDxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn gemmFcShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn batchGemmShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn batchGemmDeDwShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn batchGemmDeDxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dmaPhysicalConcatSplitDMAShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dmaPhysicalConcatContainerShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn squeezeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn frobeniusNormShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn momentsShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn nopShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn rotateShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn tfBatchNormShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn noSupportShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn cudBnFwdExShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dynamicSplitShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn physicalSplitShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn cudBnBwdShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn einsumShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dynamicReshapeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn splitFusedKernelsShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn einsumExpandShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn reinterpretCastShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn inferMaxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn tileShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn bnBatchSizeToH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dynamicStridedDmaExpandH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dynamicStridedDmaReinterpretH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn dynamicSliceDmaExpandH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn stridedOpsConversionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn sliceConversionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);

SifReturn transposeSliceH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output);
