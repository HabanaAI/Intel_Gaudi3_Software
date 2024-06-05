#pragma once
#include "smf/shape_func_registry.h"

// The SMF for gaudi1 and gaudi2 looks almost identical,
// except for gaudi2 we patch the index space with zero size when
// the ROI is completely outside of the tensor. In gaudi1
// we do not need to do this because fully-ouside ROIs
// are handled by the Qman. Hence patchZeroProjection parameter.

void tpcIndexSpaceSmfHelper(const ShapeManipulationParams* params,
                            ShapeManipulationOutputs*      outputs,
                            bool                           patchZeroProjection);
