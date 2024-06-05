#include "tensor.h"
#include "types.h"
#include "defs.h"
#include "smf/shape_func_registry.h"
#include "smf_utils.h"
#include "vtune_stat.h"

#include <include/gaudi/mme_descriptor_generator.h>

void patchOnZeroSizeShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto intersection = getIntersectionTypeFromAllTensors(params);
    auto metadata     = *static_cast<uint32_t*>(params->metadata);
    if (intersection == COMPLETELY_OUTSIDE)
    {
        // The node is disabled
        // Return our data to patch the registers
        // We assume the caller knows how many items to allocate
        HB_ASSERT(params->inPatchValuesNr == 1, "Invalid size of patch values buffer (expected 1)");

        std::copy(&metadata, &metadata + 1, outputs->outputPatchValues);

        outputs->outPatchValuesNr   = 1;
        outputs->outputShouldBypass = true;

        LOG_TRACE_DYNAMIC_PATCHING("SMF_PATCH_ON_ZERO_SIZE: completely outside");
    }
    else
    {
        // The node is enabled, we don't need to patch anything
        // Return zero as the length of patched data
        outputs->outPatchValuesNr = 0;
        // When getting not ENABLE_BYPASS we alway want to patch.
        outputs->outputShouldBypass = false;

        LOG_TRACE_DYNAMIC_PATCHING("SMF_PATCH_ON_ZERO_SIZE {}",
                                   intersection == COMPLETELY_INSIDE ? "completly inside" : "intersects");
    }
}

void patchOnZeroSizeFirstInputShapeManipulationFunction(const ShapeManipulationParams* params,
                                                        ShapeManipulationOutputs*      outputs)
{
    STAT_FUNCTION();
    auto intersection = getIntersectionType(params->activationRoi->roi_in_tensors[0], params->inputTensors[0]);
    if (intersection != COMPLETELY_OUTSIDE)
    {
        intersection = getIntersectionType(params->activationRoi->roi_out_tensors[0], params->outputTensors[0]);
    }

    auto metadata = *static_cast<uint32_t*>(params->metadata);
    if (intersection == COMPLETELY_OUTSIDE)
    {
        // The node is disabled
        // Return our data to patch the registers
        // We assume the caller knows how many items to allocate
        HB_ASSERT(params->inPatchValuesNr == 1, "Invalid size of patch values buffer (expected 1)");

        std::copy(&metadata, &metadata + 1, outputs->outputPatchValues);

        outputs->outPatchValuesNr   = 1;
        outputs->outputShouldBypass = true;

        LOG_TRACE_DYNAMIC_PATCHING("SMF_PATCH_ON_ZERO_SIZE_FIRST_INPUT: completely outside");
    }
    else
    {
        // The node is enabled, we don't need to patch anything
        // Return zero as the length of patched data
        outputs->outPatchValuesNr = 0;
        // When getting not ENABLE_BYPASS we alway want to patch.
        outputs->outputShouldBypass = false;

        LOG_TRACE_DYNAMIC_PATCHING("SMF_PATCH_ON_ZERO_SIZE_FIRST_INPUT {}",
                                   intersection == COMPLETELY_INSIDE ? "completly inside" : "intersects");
    }
}

void dynamicOffsetShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& addressParams = *static_cast<address_sm_params_t*>(params->metadata);

    HB_ASSERT(params->inputTensorsNr <= 1, "Number of input tensors is greater than 1");
    HB_ASSERT(params->outputTensorsNr == 1, "Number of output tensors is not 1");
    HB_ASSERT(params->activationRoi->roi_in_tensor_nr == params->inputTensorsNr &&
                  params->activationRoi->roi_out_tensor_nr == params->outputTensorsNr,
              "Number of ROI tensors does not match number of node tensors");

    const tensor_roi_t& activation =
        addressParams.is_src ? params->activationRoi->roi_in_tensors[0] : params->activationRoi->roi_out_tensors[0];
    const tensor_info_t* tensor = addressParams.is_src ? params->inputTensors[0] : params->outputTensors[0];
    auto                 dims   = tensor->infer_info.geometry.dims;

    int64_t delta  = 0;
    int64_t stride = addressParams.element_size;
    for (size_t i = 0; i < dims; i++)
    {
        int64_t oldStride = getStrideForDimension(tensor, i, addressParams);
        delta += activation.roi_offset_dims[i] * (stride - oldStride);
        stride *= tensor->infer_info.geometry.maxSizes[i];
    }

    uint64_t patchedAddressValue = (int64_t)addressParams.base_address + delta;
    outputs->outPatchValuesNr                                = 2;
    *reinterpret_cast<uint64_t*>(outputs->outputPatchValues) = patchedAddressValue;
    outputs->outputShouldBypass                              = 0;

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyOffsetUpdate)
    {
        callbacks->notifyOffsetUpdate(params->nodeIdx, tensor->tensor_info_name, patchedAddressValue);
    }
}

static int64_t computeSamePadding(int64_t kernel, int64_t stride, int64_t input)
{
    if (input % stride == 0)
    {
        return std::max(int64_t {}, kernel - stride);
    }
    else
    {
        return std::max(int64_t {}, kernel - (input % stride));
    }
}

void dynamicMmePaddingShapeManipulationFunction(const ShapeManipulationParams* params,
                                                ShapeManipulationOutputs*      outputs)
{
    /* calculate new pad values */

    unsigned newPadding[MAX_CONV_DIMS], oldPadding[MAX_CONV_DIMS];
    int      newOffsets[MAX_DIMENSIONS_NUM];

    auto& metadata = *static_cast<mme_padding_sm_params_t*>(params->metadata);
    auto  pTensor  = (metadata.opType == MmeCommon::e_mme_dedx || metadata.opType == MmeCommon::e_mme_transposed_dedx)
                         ? params->outputTensors[0]
                         : params->inputTensors[0];
    for (std::size_t i = 0; i < MAX_CONV_DIMS; ++i)
    {
        auto     newSize          = pTensor->infer_info.geometry.maxSizes[i + DIM_W];
        unsigned newPaddingForDim = computeSamePadding(metadata.conv_kernel[i], metadata.conv_stride[i], newSize);
        unsigned newPaddingBefore = newPaddingForDim / 2;
        // MME only needs padding before, metadata stores both padding before and padding after.
        newPadding[i] = newPaddingBefore;
        oldPadding[i] = metadata.old_padding[2 * i];
    }

    std::copy(std::begin(metadata.old_offsets), std::end(metadata.old_offsets), newOffsets);

    gaudi::patchPadding(newOffsets,
                        metadata.tensor_strides,
                        oldPadding,
                        metadata.conv_stride,
                        metadata.conv_dilation,
                        newPadding,
                        metadata.opType);

    outputs->outPatchValuesNr   = 1;
    *outputs->outputPatchValues = newOffsets[metadata.this_dim];

    outputs->outputShouldBypass = 0;

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyOffsetUpdate)
    {
        callbacks->notifyOffsetUpdate(params->nodeIdx, pTensor->tensor_info_name, newOffsets[metadata.this_dim]);
    }
}
