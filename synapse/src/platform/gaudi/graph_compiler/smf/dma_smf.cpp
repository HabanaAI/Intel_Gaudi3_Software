#include "smf.h"

#include "defs.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "math_utils.h"
#include "vtune_stat.h"

#include <algorithm>


namespace gaudi
{
namespace DmaDynamicShape
{

RoiIntersectionType getTensorRoiIntersection(
                                      roi_info_t*    roi,
                                      tensor_info_t* tensor,
                                      uint32_t*      patchedSizeValue,
                                      uint32_t       dimToPatch,
                                      bool           isDestination,
                                      bool           isMemset,
                                      bool           isTotal,
                                      uint32_t       elementSize)
{
    HB_ASSERT(roi->roi_out_tensor_nr == 1, "Number of output tensors for memcmp is not 1");
    auto& roiTensor = roi->roi_out_tensors[0];
    if (roiTensor.roi_offset_dims[dimToPatch] >= tensor->infer_info.geometry.maxSizes[dimToPatch])
    {
        // this dimension is completely outside of the current size
        // other dimensions are not interesting, we are skipping this roi
        return COMPLETELY_OUTSIDE;
    }

    if (roiTensor.roi_offset_dims[dimToPatch] + roiTensor.roi_size_dims[dimToPatch] > tensor->infer_info.geometry.maxSizes[dimToPatch])
    {
        // this dimension intersects
        *patchedSizeValue = tensor->infer_info.geometry.maxSizes[dimToPatch] - roiTensor.roi_offset_dims[dimToPatch];
        if (dimToPatch == 0)
        {
            // we are getting sizes in elements but need to write size in bytes
            *patchedSizeValue *= elementSize;
        }

        return INTERSECTS;
    }

    return COMPLETELY_INSIDE;
}

bool checkIfShouldBypass(roi_info_t*    roi,
                         tensor_info_t* tensor,
                         uint32_t*      patchedSizeValue,
                         uint32_t       dimToPatch,
                         bool           isDestination,
                         bool           isMemset,
                         bool           isTotal,
                         uint32_t       elementSize)
{
    for (int d = 0; d < tensor->infer_info.geometry.dims; d++)
    {
        auto intersectionType = getTensorRoiIntersection(roi, tensor, patchedSizeValue, d,
                                                         isDestination, isMemset, isTotal, elementSize);
        if (intersectionType != COMPLETELY_INSIDE)
        {
            return false;
        }
    }
    return true;
}
}
}


namespace gaudi
{
void dmaSizeShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    using namespace DmaDynamicShape;

    auto& dma_params = *static_cast<dma_sm_params_t*>(params->metadata);

    HB_ASSERT(params->activationRoi->roi_in_tensor_nr <= 1, "Number of input tensors is greater than 1");
    HB_ASSERT(params->activationRoi->roi_out_tensor_nr == 1, "Number of output tensors is not 1");

    // For internal DMA nodes which are pgysical concatenation subnodes,
    // params->inputTensorsNr can be greater than 1 but only the first one
    // is used for data, all this others do not have corresponding ROIs
    HB_ASSERT(params->activationRoi->roi_in_tensor_nr <= params->inputTensorsNr &&
          params->activationRoi->roi_out_tensor_nr == params->outputTensorsNr,
            "Number of ROI tensors does not match number of node tensors");

    uint32_t patchedSizeValue;

    tensor_info_t* workTensor = dma_params.is_memset ? params->outputTensors[0] : params->inputTensors[0];

    auto intersectionType = getTensorRoiIntersection(params->activationRoi,
                                                     workTensor,
                                                     &patchedSizeValue,
                                                     dma_params.this_dim,
                                                     dma_params.is_destination,
                                                     dma_params.is_memset,
                                                     dma_params.is_total,
                                                     dma_params.element_size);
    outputs->outputShouldBypass = 0;
    if (intersectionType == COMPLETELY_INSIDE ||
        intersectionType == COMPLETELY_OUTSIDE)
    {
        // The node is fully enabled or disabled, we don't need to patch anything
        // Return zero as the length of patched data
        outputs->outPatchValuesNr = 0;
        LOG_TRACE_DYNAMIC_PATCHING("SMF_DMA_SIZE no change");
    }
    else
    {
        // The node is partial, we need to patch
        // Return our data to patch the registers
        // We assume the caller knows how many items to allocate
        HB_ASSERT(params->inPatchValuesNr >= 1, "Output patch values buffer is zero lenght!");

        outputs->outPatchValuesNr   = 1;
        *outputs->outputPatchValues = patchedSizeValue;
        LOG_TRACE_DYNAMIC_PATCHING("SMF_DMA_SIZE is_input = {} dim = {} new value = {}",
                                   dma_params.is_destination,
                                   dma_params.this_dim,
                                   patchedSizeValue);
    }
}

}  // namespace gaudi
