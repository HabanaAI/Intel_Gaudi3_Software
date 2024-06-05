#include "smf.h"
#include "defs.h"
#include "graph_compiler/types.h"
#include "gaudi2/mme.h"
#include "vtune_stat.h"

namespace gaudi2
{
inline tensor_info_t*
getTensorByMetadata(mme_sm_params_t* metadata, tensor_info_t** input_tensors, tensor_info_t** output_tensors)
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
    outputs->outPatchValuesNr   = 0;
    outputs->outputShouldBypass = 0;

    auto& metadata = *static_cast<mme_sm_params_t*>(params->metadata);

    tensor_info_t* currTensor = getTensorByMetadata(&metadata, params->inputTensors, params->outputTensors);

    // In the mme we will always have a single roi per node (Because sb reuse is disabled)
    // We only have to test if the actual size of the tensor is different from the max size to know if we should patch.
    // We also only have to test the dimension in the metadata - this is a pp that patches this field.
    if (currTensor->infer_info.geometry.maxSizes[metadata.dim] != currTensor->max_dims[metadata.dim])
    {
        auto oldValue = outputs->outputPatchValues[0];
        outputs->outputPatchValues[0] = currTensor->infer_info.geometry.maxSizes[metadata.dim] * metadata.multiply_factor;
        outputs->outPatchValuesNr     = 1;
        bool is_input = metadata.tensor_input_index != INDEX_NOT_APPLICABLE;
        LOG_TRACE_DYNAMIC_PATCHING("MME_SMF is_input = {} tensor = {} dim = {} factor = {} max = {} actual = {} old value = {} new value = {}",
                                   is_input,
                                   is_input ? metadata.tensor_input_index : metadata.tensor_output_index,
                                   metadata.dim,
                                   metadata.multiply_factor,
                                   currTensor->max_dims[metadata.dim],
                                   currTensor->infer_info.geometry.maxSizes[metadata.dim],
                                   oldValue,
                                   outputs->outputPatchValues[0]);
    }
    else
    {
        LOG_TRACE_DYNAMIC_PATCHING("MME_SMF no change");
    }
}

void mmeSyncObjectSMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    constexpr auto      valueBitSize = GET_BIT_FIELD_WIDTH(Gaudi2::Mme::MmeSyncObjectVal, soValue);
    auto&               metadata     = *static_cast<mme_sync_sm_params_t*>(params->metadata);
    RoiIntersectionType intersection = getIntersectionTypeFromAllTensors(params);

    if (intersection == COMPLETELY_OUTSIDE)
    {
        // validation - no overflow in patching value
        HB_ASSERT(metadata.num_signals < (1 << valueBitSize), "overflow detected in mmeSyncObjectSMF patching!");

        Gaudi2::Mme::MmeSyncObjectVal newValue;
        newValue.dw      = outputs->outputPatchValues[0];
        newValue.soValue = metadata.num_signals;
        LOG_TRACE_DYNAMIC_PATCHING("mmeSyncObjectSMF old value {}, new value {}",
                                   outputs->outputPatchValues[0],
                                   newValue.dw);
        outputs->outPatchValuesNr = 1;
        outputs->outputPatchValues[0] = newValue.dw;
    }
    else
    {
        // do nothing
        LOG_TRACE_DYNAMIC_PATCHING("mmeSyncObjectSMF no change");
        outputs->outPatchValuesNr = 0;
    }
}

void mmeDynamicExecutionSMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();

    RoiIntersectionType intersection = getIntersectionTypeFromAllTensors(params);

    if (intersection == COMPLETELY_OUTSIDE)
    {
        LOG_TRACE_DYNAMIC_PATCHING("mmeDynamicExecutionSMF new cmd word {}", outputs->outputPatchValues[0]);
        outputs->outPatchValuesNr     = 1;
        outputs->outputPatchValues[0] = *static_cast<uint32_t*>(params->metadata);
    }
    else
    {
        // do nothing
        LOG_TRACE_DYNAMIC_PATCHING("mmeDynamicExecutionSMF no change");
        outputs->outPatchValuesNr = 0;
    }
}

}  // namespace gaudi2
