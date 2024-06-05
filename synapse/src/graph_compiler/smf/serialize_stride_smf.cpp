#include <defs.h>
#include "utils.h"
#include "smf_utils.h"
#include "vtune_stat.h"
#include "h2d_tensors.h"

unsigned getFirstDynamicDim(tensor_info_t* tensor)
{
    for (int d = 0; d < tensor->infer_info.geometry.dims; d++)
    {
        if (tensor->max_dims[d] != tensor->infer_info.geometry.maxSizes[d])
        {
            return d;
        }
    }
    return SYN_MAX_TENSOR_DIM;
}

uint32_t patchSizeAtDim(tensor_info_t* inTensor, roi_info_t* roi, unsigned dim)
{
    // Completely outside
    if (roi->roi_in_tensors[0].roi_offset_dims[dim] >= inTensor->infer_info.geometry.maxSizes[dim])
    {
        return roi->roi_in_tensors[0].roi_size_dims[dim];
    }
    // Intersect
    else if (roi->roi_in_tensors[0].roi_offset_dims[dim] + roi->roi_in_tensors[0].roi_size_dims[dim] >
             inTensor->infer_info.geometry.maxSizes[dim])
    {
        return inTensor->infer_info.geometry.maxSizes[dim] - roi->roi_in_tensors[0].roi_offset_dims[dim];
    }
    // Completely inside
    else
    {
        return roi->roi_in_tensors[0].roi_size_dims[dim];
    }
}

void lastStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    tensor_info_t* inTensor         = params->inputTensors[0];
    auto&          lastStrideParams = *static_cast<last_stride_sm_params_t*>(params->metadata);

    // Check what's the first dim that was patched. If none exist, we don't need to patch.
    unsigned firstPatchDim = getFirstDynamicDim(inTensor);
    if (firstPatchDim >= inTensor->infer_info.geometry.dims)
    {
        LOG_TRACE_DYNAMIC_PATCHING("SMF_LAST_STRIDE no change");
        outputs->outPatchValuesNr = 0;
        return;
    }

    uint32_t stride = lastStrideParams.element_size;
    for (int i = 0; i < SYN_MAX_TENSOR_DIM - 1; i++)
    {
        stride *= inTensor->infer_info.geometry.maxSizes[i];
    }

    outputs->outputPatchValues[0] = stride;
    outputs->outPatchValuesNr     = 1;
    outputs->outputShouldBypass   = 0;

    auto callbacks = SmfCallbacks::get();

    if (callbacks && callbacks->notifyStrideUpdate && inTensor->infer_info.geometry.dims == SYN_MAX_TENSOR_DIM)
    {
        callbacks->notifyStrideUpdate(params->nodeIdx, inTensor->tensor_info_name, SYN_MAX_TENSOR_DIM, stride);
    }

    LOG_TRACE_DYNAMIC_PATCHING("SMF_LAST_STRIDE is_input = {} new value = {}", lastStrideParams.is_src, stride);
}

void bulkSizeStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto&          bulkParams = *static_cast<bulk_size_stride_sm_params_t*>(params->metadata);
    unsigned       dim        = bulkParams.first_dynamic_dim;
    tensor_info_t* inTensor   = params->inputTensors[0];

    if (getFirstDynamicDim(inTensor) >= inTensor->infer_info.geometry.dims)
    {
        LOG_TRACE_DYNAMIC_PATCHING("SMF_MANY_STRIDES no change");
        outputs->outPatchValuesNr = 0;
        return;
    }

    // -1 because the first size and last stride are on different pp's
    unsigned dimsToPatch = SYN_MAX_TENSOR_DIM - dim - 1;
    if (params->inPatchValuesNr < dimsToPatch * 2)
    {
        LOG_DSD_ERR("Too many values to patch for buffer ({} values to patch, but input has only {} values)",
                    dimsToPatch * 2,
                    params->inPatchValuesNr);
        return;
    }
    outputs->outPatchValuesNr = 0;

    uint32_t currStride;
    if (dim == 1)
    {
        currStride = bulkParams.element_size * inTensor->infer_info.geometry.maxSizes[0];
    }
    else
    {
        currStride = inTensor->strides[dim - 1];
    }

    auto callbacks = SmfCallbacks::get();
    for (unsigned i = 0; i < dimsToPatch; i++)
    {
        unsigned currDim = dim + i;

        outputs->outputPatchValues[2 * i] = currStride;
        if (callbacks && callbacks->notifyStrideUpdate)
        {
            callbacks->notifyStrideUpdate(params->nodeIdx, inTensor->tensor_info_name, currDim, currStride);
        }

        if (currDim >= inTensor->infer_info.geometry.dims)
        {
            outputs->outputPatchValues[2 * i + 1] = 1;
        }
        else
        {
            outputs->outputPatchValues[2 * i + 1] = patchSizeAtDim(inTensor, params->activationRoi, currDim + 1);
            currStride *= inTensor->infer_info.geometry.maxSizes[currDim];
        }

        outputs->outPatchValuesNr += 2;
    }
    LOG_TRACE_DYNAMIC_PATCHING("SMF_MANY_STRIDES is_input = {} new values {}",
                               bulkParams.is_src,
                               getDimStr(outputs->outputPatchValues, outputs->outPatchValuesNr));
    outputs->outputShouldBypass = 0;
}

void viewSizeStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& viewParams = *static_cast<view_stride_sm_params_t*>(params->metadata);
    HB_ASSERT(params->inputTensorsNr == 2 || params->inputTensorsNr == 3, "SMF_DMA_VIEW_STRIDE - wrong number of inputs");
    HB_ASSERT(viewParams.this_dim > 0, "SMF_DMA_VIEW_STRIDE - patching fcd stride");
    tensor_info_t* dataTensor = viewParams.is_src ? params->inputTensors[0] : params->outputTensors[0];
    tensor_info_t* h2dTensor  = params->inputTensors[1];
    HB_ASSERT(h2dTensor->infer_info.hostAddress, "SMF_DMA_VIEW_STRIDE - host address is null");

    synDynamicStridedDmaH2dTensor* h2dData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(h2dTensor->infer_info.hostAddress);

    // validations
    HB_ASSERT(h2dData, "SMF_DMA_VIEW_STRIDE - no h2d data");
    HB_ASSERT(dataTensor->infer_info.geometry.dims == h2dData->num_strides, "SMF_DMA_VIEW_STRIDE - illegal dimensions");
    HB_ASSERT(h2dData->strides[0] == 1, "SMF_DMA_VIEW_STRIDE - strided fcd");

    uint64_t lastElement = h2dData->offset;
    for (unsigned i = 0; i < dataTensor->infer_info.geometry.dims; i++)
    {
        lastElement += (uint64_t)(dataTensor->infer_info.geometry.maxSizes[i] - 1) * h2dData->strides[i];
    }
    HB_ASSERT(lastElement < viewParams.num_real_elements,
              "illegal memory access by dynamic strided {}",
              viewParams.is_src ? "view" : "insert");

    // calculation - patch dynamic stride
    outputs->outPatchValuesNr     = 1;
    outputs->outputPatchValues[0] = h2dData->strides[viewParams.this_dim] * viewParams.element_size;

    LOG_TRACE_DYNAMIC_PATCHING("SMF_DMA_VIEW_STRIDE is_input = {}, this_dim = {}, new value {}",
                               viewParams.is_src,
                               viewParams.this_dim,
                               outputs->outputPatchValues[0]);
    outputs->outputShouldBypass = 0;

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyStrideUpdate)
    {
        callbacks->notifyStrideUpdate(params->nodeIdx,
                                      dataTensor->tensor_info_name,
                                      viewParams.this_dim + 1,
                                      outputs->outputPatchValues[0]);
    }
}

void viewBaseAddressShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& addressParams = *static_cast<view_address_sm_params_t*>(params->metadata);

    HB_ASSERT(params->inputTensorsNr == 2, "SMF_DMA_VIEW_OFFSET - wrong number of inputs");
    const tensor_roi_t& activation =
        addressParams.is_src ? params->activationRoi->roi_in_tensors[0] : params->activationRoi->roi_out_tensors[0];

    int64_t delta = 0;

    tensor_info_t* h2dTensor = params->inputTensors[1];
    HB_ASSERT(h2dTensor->infer_info.hostAddress, "SMF_DMA_VIEW_OFFSET - host address is null");

    synDynamicStridedDmaH2dTensor* h2dData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(h2dTensor->infer_info.hostAddress);

    // validations
    HB_ASSERT(h2dData, "SMF_DMA_VIEW_OFFSET - no h2d data");
    HB_ASSERT(h2dData->strides[0] == 1, "SMF_DMA_VIEW_OFFSET - strided fcd");

    delta = addressParams.element_size * ((int64_t)h2dData->offset - (int64_t)addressParams.max_offset);
    for (unsigned i = 0; i < h2dData->num_strides; i++)
    {
        int64_t strideDiff = (int64_t)h2dData->strides[i] - (int64_t)addressParams.max_strides[i];
        delta += (int64_t)activation.roi_offset_dims[i] * addressParams.element_size * strideDiff;
    }

    uint64_t patchedAddressValue                             = (int64_t)addressParams.base_address + delta;
    outputs->outPatchValuesNr                                = 2;
    *reinterpret_cast<uint64_t*>(outputs->outputPatchValues) = patchedAddressValue;
    outputs->outputShouldBypass                              = 0;

    LOG_TRACE_DYNAMIC_PATCHING("SMF_DMA_VIEW_OFFSET is_input = {}, delta {}", addressParams.is_src, delta);

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyOffsetUpdate)
    {
        callbacks->notifyOffsetUpdate(params->nodeIdx, params->inputTensors[0]->tensor_info_name, patchedAddressValue);
    }
}

void sliceStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& sliceParams = *static_cast<slice_stride_sm_params_t*>(params->metadata);
    HB_ASSERT(params->inputTensorsNr == 3, "SMF_DMA_SLICE_STRIDE - wrong number of inputs");
    tensor_info_t* dataTensor  = sliceParams.is_src ? params->inputTensors[0] : params->outputTensors[0];
    tensor_info_t* stepsTensor = params->inputTensors[1];

    // validations
    HB_ASSERT(dataTensor->infer_info.geometry.dims == stepsTensor->infer_info.geometry.dims, "SMF_DMA_SLICE_STRIDE - illegal dimensions");
    HB_ASSERT(stepsTensor->infer_info.geometry.maxSizes[0] == 1, "SMF_DMA_SLICE_STRIDE - strided fcd");
    HB_ASSERT(sliceParams.dim > 0, "SMF_DMA_SLICE_STRIDE - cannot patch fcd stride");

    // calculation - patch dynamic stride
    outputs->outPatchValuesNr = 1;
    uint64_t originalStride =
        getStrideForDimension(dataTensor, sliceParams.dim, sliceParams) / stepsTensor->max_dims[sliceParams.dim];
    outputs->outputPatchValues[0] = stepsTensor->infer_info.geometry.maxSizes[sliceParams.dim] * originalStride;

    LOG_TRACE_DYNAMIC_PATCHING("SMF_DMA_SLICE_STRIDE dim = {}, new value {}",
                               sliceParams.dim,
                               outputs->outputPatchValues[0]);
    outputs->outputShouldBypass = 0;

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyStrideUpdate)
    {
        callbacks->notifyStrideUpdate(params->nodeIdx,
                                      dataTensor->tensor_info_name,
                                      sliceParams.dim + 1,
                                      outputs->outputPatchValues[0]);
    }
}

void sliceBaseAddressShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& addressParams = *static_cast<slice_address_sm_params_t*>(params->metadata);

    HB_ASSERT(params->inputTensorsNr == 3, "SMF_DMA_SLICE_OFFSET - wrong number of inputs");
    tensor_info_t*      dataTensor   = addressParams.is_src ? params->inputTensors[0] : params->outputTensors[0];
    tensor_info_t*      stepsTensor  = params->inputTensors[1];
    tensor_info_t*      startsTensor = params->inputTensors[2];
    const tensor_roi_t& activation   = params->activationRoi->roi_in_tensors[0];

    // validations
    HB_ASSERT(dataTensor->infer_info.geometry.dims == stepsTensor->infer_info.geometry.dims, "SMF_DMA_SLICE_OFFSET - illegal dimensions");
    HB_ASSERT(dataTensor->infer_info.geometry.dims == startsTensor->infer_info.geometry.dims, "SMF_DMA_SLICE_OFFSET - illegal dimensions");
    HB_ASSERT(stepsTensor->infer_info.geometry.maxSizes[0] == 1, "SMF_DMA_SLICE_OFFSET - strided fcd");

    // VALIDATION - make sure we don't get out of bound of the original tensor
    uint64_t lastElement = 0;
    uint64_t denseStride = 1;
    for (unsigned i = 0; i < dataTensor->infer_info.geometry.dims; i++)
    {
        uint64_t start     = startsTensor->infer_info.geometry.maxSizes[i];
        uint64_t step      = stepsTensor->infer_info.geometry.maxSizes[i];
        uint64_t size      = dataTensor->infer_info.geometry.maxSizes[i];
        uint64_t newStride = denseStride * step;                          // new step strid
        lastElement += (start * denseStride) + ((size - 1) * newStride);  // for validation

        denseStride *= dataTensor->max_dims[i];
    }
    HB_ASSERT(lastElement < addressParams.num_real_elements,
              "illegal memory access by dynamic slice {}",
              addressParams.is_src ? "fwd" : "bwd");

    // patch base address
    int64_t delta = 0;
    for (unsigned i = 0; i < dataTensor->infer_info.geometry.dims; i++)
    {
        int64_t startDiff = (int64_t)startsTensor->infer_info.geometry.maxSizes[i] - (int64_t)startsTensor->max_dims[i];

        int64_t oldStep = stepsTensor->max_dims[i];
        int64_t newStep = stepsTensor->infer_info.geometry.maxSizes[i];

        int64_t oldStepStride = getStrideForDimension(dataTensor, i, addressParams);
        int64_t stride        = oldStepStride / oldStep;
        int64_t newStepStride = stride * newStep;

        delta += startDiff * stride;  // "original" size offset - multiply by original stride
        delta += activation.roi_offset_dims[i] *
                 (newStepStride - oldStepStride);  // current dimension ROI offset - multiply by new stride
    }

    uint64_t patchedAddressValue                             = (int64_t)addressParams.base_address + delta;
    outputs->outPatchValuesNr                                = 2;
    *reinterpret_cast<uint64_t*>(outputs->outputPatchValues) = patchedAddressValue;
    outputs->outputShouldBypass                              = 0;
    LOG_TRACE_DYNAMIC_PATCHING("SMF_DMA_SLICE_OFFSET old value {}. delta {}", addressParams.base_address, delta);

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyOffsetUpdate)
    {
        callbacks->notifyOffsetUpdate(params->nodeIdx, dataTensor->tensor_info_name, patchedAddressValue);
    }
}
