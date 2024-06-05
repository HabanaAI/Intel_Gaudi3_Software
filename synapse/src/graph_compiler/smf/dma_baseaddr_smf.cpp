#include <defs.h>
#include <algorithm>
#include "math_utils.h"
#include "vtune_stat.h"
#include "recipe_metadata.h"
#include "smf/shape_func_registry.h"

void dmaBaseAddressManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();

    const auto& concatSplitParams = *static_cast<physical_concat_split_sm_params_t*>(params->metadata);
    auto        concatDim         = concatSplitParams.concat_split_dim;

    uint64_t delta = 0;
    for (unsigned tensorNum = 0; tensorNum < concatSplitParams.number_in_concat_split; tensorNum++)
    {
        delta += (params->inputTensors[1+tensorNum]->max_dims[concatDim] -
                  params->inputTensors[1+tensorNum]->infer_info.geometry.maxSizes[concatDim]);
    }

    delta *= concatSplitParams.output_strides[concatDim];
    delta *= concatSplitParams.element_size;

    auto newAddress = concatSplitParams.roi_base_address - delta;

    outputs->outPatchValuesNr = 2;
    *reinterpret_cast<uint64_t*>(outputs->outputPatchValues) = newAddress;
    outputs->outputShouldBypass = 0;
}
