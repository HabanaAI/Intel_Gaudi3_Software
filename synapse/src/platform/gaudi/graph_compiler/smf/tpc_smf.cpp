#include "smf.h"

#include "defs.h"
#include "h2d_tensors.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "math_utils.h"
#include "vtune_stat.h"

#include <algorithm>

namespace gaudi
{
void tpcSizeShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& tpcParams = *static_cast<tpc_size_sm_params_t*>(params->metadata);

    auto* workTensor = tpcParams.is_output ? params->outputTensors[tpcParams.tensor_index] :
                                             params->inputTensors[tpcParams.tensor_index];

    outputs->outputShouldBypass = 0;
    outputs->outPatchValuesNr = 0;

    // We need to update the tpc sizes anyway and let the other PPs decide if this is actually going to run.
    if (workTensor->infer_info.geometry.maxSizes[tpcParams.this_dim] != workTensor->max_dims[tpcParams.this_dim])
    {
        HB_ASSERT(params->inPatchValuesNr >= 1, "Output patch values buffer is zero length!");
        outputs->outPatchValuesNr = 1;

        // TPC kernels are instantiated once (with big tensors sizes), TPC slice descriptors should be
        // patched based on the offset from the big tensor.
        auto old_value = outputs->outputPatchValues[0];
        outputs->outputPatchValues[0] = workTensor->infer_info.geometry.maxSizes[tpcParams.this_dim] + tpcParams.offset;
        if (tpcParams.this_dim == 0 && (workTensor->data_type & (syn_type_int64 | syn_type_uint64)))
        {
            // single 64b elements will be treated as 2 32b elements (32b low, 32b high)
            // hence FCD of size is doubled
            outputs->outputPatchValues[0] *= 2;
        }

        LOG_TRACE_DYNAMIC_PATCHING(
            "SMF_TPC_SIZE {} tensor {} dim = {} offset in big tensor = {} old_value = {} new value = {}",
            tpcParams.is_output ? "output" : "input",
            tpcParams.tensor_index,
            tpcParams.this_dim,
            tpcParams.offset,
            old_value,
            outputs->outputPatchValues[0]);
    }
    else
    {
        LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_SIZE no change");
    }
}

void tpcStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& tpcParams = *static_cast<tpc_stride_sm_params_t*>(params->metadata);

    auto* workTensor = tpcParams.is_output ? params->outputTensors[tpcParams.tensor_index] :
                                             params->inputTensors[tpcParams.tensor_index];
    auto firstDynDim = tpcParams.first_dynamic_dim;

    outputs->outputShouldBypass = 0;
    outputs->outPatchValuesNr = 0;

    if (tpcParams.this_dim >= firstDynDim)
    {
        HB_ASSERT(params->inPatchValuesNr >= 1, "Output patch values buffer is zero length!");
        outputs->outPatchValuesNr = 1;

        // calculate new stride (in elements)
        uint32_t currStride = 1;
        for (int i = 0; i <= tpcParams.this_dim; i++) // including this_dim
        {
            currStride *= workTensor->infer_info.geometry.maxSizes[i];
        }
        auto oldValue = outputs->outputPatchValues[0];
        outputs->outputPatchValues[0] = currStride;
        if (workTensor->data_type & (syn_type_int64 | syn_type_uint64))
        {
            // single 64b elements will be treated as 2 32b elements (32b low, 32b high)
            // hence FCD of size is doubled, and so are *all* strides
            outputs->outputPatchValues[0] *= 2;
        }

        auto callbacks = SmfCallbacks::get();
        if (callbacks && callbacks->notifyStrideUpdate)
        {
            // Send new strides to TD (in bytes)
            callbacks->notifyStrideUpdate(params->nodeIdx,
                                          workTensor->tensor_info_name,
                                          tpcParams.this_dim + 1,
                                          currStride * tpcParams.element_size);
        }

        LOG_TRACE_DYNAMIC_PATCHING(
            "SMF_TPC_STRIDE {} tensor {} dim = {} dynamic dim = {} old value = {} new value = {}",
            tpcParams.is_output ? "output" : "input",
            tpcParams.tensor_index,
            tpcParams.this_dim,
            tpcParams.first_dynamic_dim,
            oldValue,
            outputs->outputPatchValues[0]);
    }
    else
    {
        LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_STRIDE no change");
    }
}

void tpcSliceStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto sliceParams = static_cast<slice_stride_sm_params_t*>(params->metadata);
    HB_ASSERT(params->inputTensorsNr == 3, "SMF_TPC_SLICE_STRIDE - wrong number of inputs");
    tensor_info_t* dataTensor  = sliceParams->is_src ? params->inputTensors[0] : params->outputTensors[0];
    tensor_info_t* stepsTensor = params->inputTensors[1];

    // validations
    HB_ASSERT(dataTensor->infer_info.geometry.dims == stepsTensor->infer_info.geometry.dims, "SMF_TPC_SLICE_STRIDE - illegal dimensions");
    HB_ASSERT(stepsTensor->infer_info.geometry.maxSizes[0] == 1, "SMF_TPC_SLICE_STRIDE - strided fcd");
    HB_ASSERT(sliceParams->dim > 0, "SMF_TPC_SLICE_STRIDE - cannot patch fcd stride");

    uint32_t prepatchValue = outputs->outputPatchValues[0];

    // calculation - patch dynamic stride
    uint64_t originalStride = sliceParams->dim == 0 ? 1 : dataTensor->strides[sliceParams->dim - 1] / sliceParams->element_size;
    outputs->outputPatchValues[0] = originalStride / stepsTensor->max_dims[sliceParams->dim] * stepsTensor->infer_info.geometry.maxSizes[sliceParams->dim];
    outputs->outPatchValuesNr = 1;

    LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_SLICE_STRIDE dim = {}, prepatch = {}. old value = {}, new value = {}",
                               sliceParams->dim,
                               prepatchValue,
                               originalStride,
                               outputs->outputPatchValues[0]);
    outputs->outputShouldBypass = 0;

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyStrideUpdate)
    {
        callbacks->notifyStrideUpdate(params->nodeIdx,
                                      dataTensor->tensor_info_name,
                                      sliceParams->dim + 1,
                                      outputs->outputPatchValues[0]);
    }
}

void tpcSliceBaseAddressShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& addressParams = *static_cast<slice_address_sm_params_t*>(params->metadata);

    HB_ASSERT(params->inputTensorsNr == 3, "SMF_TPC_SLICE_OFFSET - wrong number of inputs");
    tensor_info_t*      dataTensor   = addressParams.is_src ? params->inputTensors[0] : params->outputTensors[0];
    tensor_info_t*      stepsTensor  = params->inputTensors[1];
    tensor_info_t*      startsTensor = params->inputTensors[2];
    const tensor_roi_t& activation   = params->activationRoi->roi_in_tensors[0];

    // validations
    HB_ASSERT(dataTensor->infer_info.geometry.dims == stepsTensor->infer_info.geometry.dims, "SMF_TPC_SLICE_OFFSET - illegal dimensions");
    HB_ASSERT(dataTensor->infer_info.geometry.dims == startsTensor->infer_info.geometry.dims, "SMF_TPC_SLICE_OFFSET - illegal dimensions");
    HB_ASSERT(stepsTensor->infer_info.geometry.maxSizes[0] == 1, "SMF_TPC_SLICE_OFFSET - strided fcd");

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
              "SMF_TPC_SLICE_OFFSET - illegal memory access by dynamic slice {}",
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
    LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_SLICE_OFFSET old value {}. delta {}", addressParams.base_address, delta);

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyOffsetUpdate)
    {
        callbacks->notifyOffsetUpdate(params->nodeIdx, dataTensor->tensor_info_name, patchedAddressValue);
    }
}

void tcpViewStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& viewParams = *static_cast<view_stride_sm_params_t*>(params->metadata);
    HB_ASSERT(params->inputTensorsNr == 2 || params->inputTensorsNr == 3, "SMF_TPC_VIEW_STRIDE - wrong number of inputs");
    HB_ASSERT(viewParams.this_dim > 0, "SMF_TPC_VIEW_STRIDE - patching fcd stride");
    tensor_info_t* dataTensor = viewParams.is_src ? params->inputTensors[0] : params->outputTensors[0];
    tensor_info_t* h2dTensor  = params->inputTensors[1];
    HB_ASSERT(h2dTensor->infer_info.hostAddress, "SMF_TPC_VIEW_STRIDE - host address is null");

    synDynamicStridedDmaH2dTensor* h2dData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(h2dTensor->infer_info.hostAddress);

    // validations
    HB_ASSERT(h2dData, "SMF_TPC_VIEW_STRIDE - no h2d data");
    HB_ASSERT(dataTensor->infer_info.geometry.dims == h2dData->num_strides, "SMF_TPC_VIEW_STRIDE - illegal dimensions");
    HB_ASSERT(h2dData->strides[0] == 1, "SMF_TPC_VIEW_STRIDE - strided fcd");

    uint64_t lastElement = h2dData->offset;
    for (unsigned i = 0; i < dataTensor->infer_info.geometry.dims; i++)
    {
        lastElement += (uint64_t)(dataTensor->infer_info.geometry.maxSizes[i] - 1) * h2dData->strides[i];
    }
    HB_ASSERT(lastElement < viewParams.num_real_elements,
              "SMF_TPC_VIEW_STRIDE - illegal memory access by dynamic strided {}",
              viewParams.is_src ? "view" : "insert");

    // calculation - patch dynamic stride
    outputs->outPatchValuesNr     = 1;
    outputs->outputPatchValues[0] = h2dData->strides[viewParams.this_dim];

    LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_VIEW_STRIDE is_input = {}, this_dim = {}, new value {}",
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
                                      outputs->outputPatchValues[0] * viewParams.element_size);
    }
}

void tpcViewBaseAddressShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& addressParams = *static_cast<view_address_sm_params_t*>(params->metadata);

    HB_ASSERT(params->inputTensorsNr == 2, "SMF_TPC_VIEW_OFFSET - wrong number of inputs");
    const tensor_roi_t& activation =
        addressParams.is_src ? params->activationRoi->roi_in_tensors[0] : params->activationRoi->roi_out_tensors[0];

    int64_t delta = 0;

    tensor_info_t* h2dTensor = params->inputTensors[1];
    HB_ASSERT(h2dTensor->infer_info.hostAddress, "SMF_TPC_VIEW_OFFSET - host address is null");

    synDynamicStridedDmaH2dTensor* h2dData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(h2dTensor->infer_info.hostAddress);

    // validations
    HB_ASSERT(h2dData, "SMF_TPC_VIEW_OFFSET - no h2d data");
    HB_ASSERT(h2dData->strides[0] == 1, "SMF_TPC_VIEW_OFFSET - strided fcd");

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

    LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_VIEW_OFFSET is_input = {}, delta {}", addressParams.is_src, delta);

    auto callbacks = SmfCallbacks::get();
    if (callbacks && callbacks->notifyOffsetUpdate)
    {
        callbacks->notifyOffsetUpdate(params->nodeIdx, params->inputTensors[0]->tensor_info_name, patchedAddressValue);
    }
}


}  // namespace gaudi
