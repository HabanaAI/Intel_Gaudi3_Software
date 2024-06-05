#include "smf.h"
#include "defs.h"
#include "graph_compiler/types.h"
#include "gaudi2/mme.h"
#include "vtune_stat.h"

namespace gaudi3
{

inline tensor_info_t* getTensorByMetadata(mme_multi_dcore_sm_params_t* metadata, tensor_info_t** input_tensors, tensor_info_t** output_tensors)
{
    if (metadata->tensor_input_index != INDEX_NOT_APPLICABLE)
    {
        return input_tensors[metadata->tensor_input_index];
    }
    return output_tensors[metadata->tensor_output_index];
}

void mmeValidElementsGaudi3SMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    outputs->outPatchValuesNr   = 0;
    outputs->outputShouldBypass = 0;

    auto& metadata = *static_cast<mme_multi_dcore_sm_params_t*>(params->metadata);

    tensor_info_t* currTensor = getTensorByMetadata(&metadata, params->inputTensors, params->outputTensors);

    uint64_t newSize = 0;
    if (currTensor->infer_info.geometry.maxSizes[metadata.dim] < metadata.dcore_roi_offset)
    {
        // new ROI size is zero
        // TODO make it a null descriptor?
        newSize = 0;
    }
    else if (currTensor->infer_info.geometry.maxSizes[metadata.dim] >= metadata.dcore_roi_offset + metadata.dcore_roi_size)
    {
        // ROI size does not change
        newSize = metadata.dcore_roi_size;
        LOG_TRACE_DYNAMIC_PATCHING("MME_GAUDI3_SMF no change");
        return;
    }
    else
    {
        // new ROI size
        newSize = currTensor->infer_info.geometry.maxSizes[metadata.dim] - metadata.dcore_roi_offset;
    }
    auto oldValue = outputs->outputPatchValues[0];
    outputs->outputPatchValues[0] = newSize * metadata.multiply_factor;
    outputs->outPatchValuesNr     = 1;
    bool is_input = metadata.tensor_input_index != INDEX_NOT_APPLICABLE;
    LOG_TRACE_DYNAMIC_PATCHING("MME_GAUDI3_SMF is_input = {} tensor = {} dim = {} factor = {} max = {} actual = {} roi_offset = {} roi_size = {} old value = {} new value = {}",
            is_input,
            is_input ? metadata.tensor_input_index : metadata.tensor_output_index,
            metadata.dim,
            metadata.multiply_factor,
            currTensor->max_dims[metadata.dim],
            currTensor->infer_info.geometry.maxSizes[metadata.dim],
            metadata.dcore_roi_offset,
            metadata.dcore_roi_size,
            oldValue,
            outputs->outputPatchValues[0]);
}

} // namespace gaudi3
