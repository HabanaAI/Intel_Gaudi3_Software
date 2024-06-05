#include <defs.h>
#include <algorithm>
#include "math_utils.h"
#include "index_space_utils.h"
#include "recipe_metadata.h"
#include "vtune_stat.h"

void tpcIndexSpaceSmfHelper(const ShapeManipulationParams* params,
                            ShapeManipulationOutputs*      outputs,
                            bool                           patchZeroProjection)
{
    STAT_FUNCTION();
    outputs->outputShouldBypass = 0;

    // Cannot have less than 1 iteration of the TPC loop.
    uint32_t maxBackProjection = 1;
    auto&    roi               = *params->activationRoi;
    auto&    tpcParams         = *static_cast<tpc_sm_params_t*>(params->metadata);

    bool needPatch           = false;
    bool foundZeroProjection = false;

    LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_INDEX_SPACE Metadata: numProjections {}, this dim {}", tpcParams.num_projections, tpcParams.this_dim);

    for (uint32_t i = 0; i < tpcParams.num_projections; ++i)
    {
        LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_INDEX_SPACE Metadata: projection {} a {} is_output {} size {} tensor_dim {} tensor_idx {}", i,
                tpcParams.projections[i].a,
                tpcParams.projections[i].is_output,
                tpcParams.projections[i].size,
                tpcParams.projections[i].tensor_dim,
                tpcParams.projections[i].tensor_idx);

        const auto& projectionData = tpcParams.projections[i];
        auto        tensorIndex    = projectionData.tensor_idx;
        auto        tensorDim      = projectionData.tensor_dim;
        auto& roiTensor = projectionData.is_output ? roi.roi_out_tensors[tensorIndex] : roi.roi_in_tensors[tensorIndex];
        auto* workTensor =
            projectionData.is_output ? params->outputTensors[tensorIndex] : params->inputTensors[tensorIndex];
        if (roiTensor.roi_offset_dims[tensorDim] >= workTensor->infer_info.geometry.maxSizes[tensorDim])
        {
            foundZeroProjection = true;
            maxBackProjection   = 0;
            needPatch           = patchZeroProjection;

            // we are completely outside
            LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_INDEX_SPACE skipped {}{} dim {}: ROI offset: {}, inferred size: {}",
                                       projectionData.is_output ? "output" : "input",
                                       tensorIndex,
                                       tensorDim,
                                       roiTensor.roi_offset_dims[tensorDim],
                                       workTensor->infer_info.geometry.maxSizes[tensorDim]);
            continue;
        }

        // We calculate the actual index space size from the actual tensor sizes.

        // We have these relations:
        //
        //     tensor_index >= (index_space_index * a + start_b)
        //     tensor_index <= (index_space_index * a + end_b)
        //
        // This is what TPC is allowed to touch when given index_space_index.
        // So if the index space size (max index in given dimension) is P, then the tensor size
        // in that dimension is
        //
        //     Q = P * a + end_b
        //
        // We know this relation always holds w.r.t. max sizes, this is computed by the graph compiler.
        // Now if we shrink the tensor and the actual size is Q' = Q - delta, then we must shrink the
        // index space size by delta/a:
        //
        //      P' = P - delta/a
        //
        // so that
        //
        //      Q' = P' * a + end_b
        //
        // still holds, at least up to the nearest multiple of a which is the granularity of the kernel.
        //

        // See comment in GaudiTPCPatchPointGenerator::AddDynamicShapePatchPointIndexSpace for more info.
        // We patch the index space size register (tid_size_dim_N) with the calculated value.

        // 'a' is the number of elements in index space vector. Simply find out how many vectors is needed to fit Q'.

        LOG_TRACE_DYNAMIC_PATCHING(
            "SMF_TPC_INDEX_SPACE patching {}{} dim {}: ROI offset: {}, ROI size: {}, inferred size: {}, "
            "a: {}, size: {}",
            projectionData.is_output ? "output" : "input",
            tensorIndex,
            tensorDim,
            roiTensor.roi_offset_dims[tensorDim],
            roiTensor.roi_size_dims[tensorDim],
            workTensor->infer_info.geometry.maxSizes[tensorDim],
            projectionData.a,
            projectionData.size);

        if (projectionData.a == 0.0F)
        {
            LOG_WARN(SYN_API, "SMF_TPC_INDEX_SPACE dim: {} a is 0", tensorDim);
        }
        // projectionData.a should never be zero, it stands for the TPC loop increment

        // Actual size is the lesser between ROI max size and the portion of ROI that fits into inferred tensor size.
        // Find out how many a's needed to cover the actual ROI size.
        // If we get 0 or less, it means some portion of the index space was covered
        // by the end_b parameter, and now at the reduced tensor size
        // all of it is covered by end_b. We still need 1 iteration to run the kernel.
        // (The first min argument is strictly positive because we already checked for zero projection)
        HB_ASSERT(workTensor->infer_info.geometry.maxSizes[tensorDim] - roiTensor.roi_offset_dims[tensorDim] > 0,
                  "SMF_TPC_INDEX_SPACE found zero projection, but we already checked for it");
        uint32_t roiActualSize =
            std::min(workTensor->infer_info.geometry.maxSizes[tensorDim] - roiTensor.roi_offset_dims[tensorDim],
                     roiTensor.roi_size_dims[tensorDim]);
        uint32_t currentBackProjection =
            (projectionData.a == 0.0f || roiActualSize == roiTensor.roi_size_dims[tensorDim])
                ? projectionData.size
                : div_round_up((TSize)projectionData.size * roiActualSize, roiTensor.roi_size_dims[tensorDim]);
        maxBackProjection = std::max(maxBackProjection, currentBackProjection);

        needPatch = true;
    }

    if (foundZeroProjection && !patchZeroProjection)
    {
        needPatch = false;
    }

    if (roi.index_space_size[tpcParams.this_dim] == maxBackProjection)
    {
        needPatch = false;
    }

    if (needPatch)
    {
        HB_DEBUG_VALIDATE(params->inPatchValuesNr >= 1 && "Output patch values buffer is zero length!");
        outputs->outPatchValuesNr     = 1;
        outputs->outputPatchValues[0] = maxBackProjection;
        LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_INDEX_SPACE dim = {} old value = {}, new value = {}",
                                   tpcParams.this_dim,
                                   roi.index_space_size[tpcParams.this_dim],
                                   maxBackProjection);
    }
    else
    {
        outputs->outPatchValuesNr = 0;
        LOG_TRACE_DYNAMIC_PATCHING("SMF_TPC_INDEX_SPACE no change");
    }
}
