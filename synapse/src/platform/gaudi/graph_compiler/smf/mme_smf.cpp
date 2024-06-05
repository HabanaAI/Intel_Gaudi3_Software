#include "smf.h"
#include "defs.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "vtune_stat.h"

namespace gaudi
{

inline tensor_info_t* getTensorByMetadata(mme_sm_params_t* metadata, tensor_info_t** input_tensors,
                                          tensor_info_t** output_tensors)
{
    if (metadata->tensor_input_index != INDEX_NOT_APPLICABLE)
    {
        return input_tensors[metadata->tensor_input_index];
    }
    return output_tensors[metadata->tensor_output_index];
}

void mmeValidElementsSMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    outputs->outPatchValuesNr = 0;
    outputs->outputShouldBypass = 0;

    auto& metadata = *static_cast<mme_sm_params_t*>(params->metadata);

    tensor_info_t* currTensor = getTensorByMetadata(&metadata, params->inputTensors, params->outputTensors);

    // In the mme we will always have a single roi per node (Because sb reuse is disabled)
    // We only have to test if the actual size of the tensor is different from the max size to know if we should patch.
    // We also only have to test the dimension in the metadata - this is a pp that patches this field.
    if (currTensor->infer_info.geometry.maxSizes[metadata.dim] !=
        currTensor->max_dims[metadata.dim])
    {
        outputs->outputPatchValues[0] = currTensor->infer_info.geometry.maxSizes[metadata.dim] *
                                        metadata.multiply_factor;
        outputs->outPatchValuesNr = 1;
        bool is_input = metadata.tensor_input_index != INDEX_NOT_APPLICABLE;
        LOG_TRACE_DYNAMIC_PATCHING("MME_SMF is_input = {} tensor = {} dim = {} new value = {}",
                                   is_input,
                                   is_input ? metadata.tensor_input_index : metadata.tensor_output_index,
                                   metadata.dim,
                                   outputs->outputPatchValues[0]);
    }
    else
    {
        LOG_TRACE_DYNAMIC_PATCHING("MME_SMF no change");
    }
}

}  // namespace gaudi
