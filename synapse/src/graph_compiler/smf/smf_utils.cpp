#include <defs.h>
#include <algorithm>
#include "vtune_stat.h"
#include "graph_compiler/types.h"

#include "smf_utils.h"

RoiIntersectionType getDimIntersectionType(tensor_roi_t& roiTensors, tensor_info_t* tensor, uint64_t dim)
{
    if (roiTensors.roi_offset_dims[dim] >= tensor->infer_info.geometry.maxSizes[dim])
    {
        return COMPLETELY_OUTSIDE;
    }

    if (roiTensors.roi_offset_dims[dim] + roiTensors.roi_size_dims[dim] > tensor->infer_info.geometry.maxSizes[dim])
    {
        return INTERSECTS;
    }

    return COMPLETELY_INSIDE;
}

RoiIntersectionType max(RoiIntersectionType first, RoiIntersectionType second)
{
    // The highest priority is completely outside - If one of the tensors is outside, the activation should be disabled.
    if ((first == COMPLETELY_OUTSIDE) || (second == COMPLETELY_OUTSIDE))
    {
        return COMPLETELY_OUTSIDE;
    }
    // The next priority is intersection
    if ((first == INTERSECTS) || (second == INTERSECTS))
    {
        return INTERSECTS;
    }
    // Only if all the tensors are inside - treat as fully-in
    return COMPLETELY_INSIDE;
}

RoiIntersectionType getIntersectionType(tensor_roi_t& roiTensors, tensor_info_t* tensor)
{
    RoiIntersectionType intersection = COMPLETELY_INSIDE;
    for (uint64_t dim = 0; dim < tensor->infer_info.geometry.dims; ++dim)
    {
        intersection = max(intersection, getDimIntersectionType(roiTensors, tensor, dim));
        if (intersection == COMPLETELY_OUTSIDE)
        {
            return COMPLETELY_OUTSIDE;
        }
    }

    return intersection;
}

RoiIntersectionType getIntersectionType(tensor_roi_t* rois, tensor_info_t** tensors, uint64_t TensorsNr)
{
    RoiIntersectionType intersection = COMPLETELY_INSIDE;

    for (int i = 0; i < TensorsNr; i++)
    {
        if (tensors[i]->tensor_type == tensor_info_t::ETensorType::SHAPE_TENSOR) continue;
        if (tensors[i]->tensor_flags & tensor_info_t::ETensorFlags::HAS_HOST_ADDRESS) continue;

        intersection = max(intersection, getIntersectionType(rois[i], tensors[i]));
        if (intersection == COMPLETELY_OUTSIDE)
        {
            return COMPLETELY_OUTSIDE;
        }
    }

    return intersection;
}

RoiIntersectionType getIntersectionTypeFromAllTensors(const ShapeManipulationParams* params)
{
    RoiIntersectionType intersection =
        getIntersectionType(params->activationRoi->roi_in_tensors, params->inputTensors, params->inputTensorsNr);

    if (intersection != COMPLETELY_OUTSIDE)
    {
        intersection = max(intersection,
                           getIntersectionType(params->activationRoi->roi_out_tensors,
                                               params->outputTensors,
                                               params->outputTensorsNr));
    }

    return intersection;
}

RoiIntersectionType getIntersectionTypeFromProjection(const ShapeManipulationParams* params)
{
    auto& metadata = *static_cast<dynamic_execution_sm_params_t*>(params->metadata);

    uint64_t allTensorsMask       = 0;
    uint64_t outsideTensorsMask   = 0;
    uint64_t intersectTensorsMask = 0;
    bool     projectionFound      = false;

    if (metadata.num_projections == 0)
    {
        LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION Have no projections, thus return COMPLETELY_INSIDE");
        // have no data to determine intersection, err on the side of caution
        return COMPLETELY_INSIDE;
    }

    LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                               "Start computing intersections");

    for (int i = 0; i < metadata.num_projections; i++)
    {
        projectionFound = true;

        uint16_t tensorIdx = metadata.projections[i].tensor_idx;
        uint16_t tensorDim = metadata.projections[i].tensor_dim;
        uint16_t isOutput  = metadata.projections[i].is_output;

        uint64_t tensorBit = 1 << tensorIdx;
        if (isOutput) tensorBit <<= 32;

        allTensorsMask |= tensorBit;

        tensor_info_t** tensors = isOutput ? params->outputTensors : params->inputTensors;
        tensor_roi_t* rois = isOutput ? params->activationRoi->roi_out_tensors : params->activationRoi->roi_in_tensors;

        LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                                   "Computing intersection : tidx {} dim {} isOutput {} "
                                   "roi_offset {} roi_size {} actualSize {}",
                                   tensorIdx,
                                   tensorDim,
                                   isOutput,
                                   rois[tensorIdx].roi_offset_dims[tensorDim],
                                   rois[tensorIdx].roi_size_dims[tensorDim],
                                   tensors[tensorIdx]->infer_info.geometry.maxSizes[tensorDim]);

        RoiIntersectionType intersection = getDimIntersectionType(rois[tensorIdx], tensors[tensorIdx], tensorDim);

        LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                                   "Computed intersection {}",
                                   toString(intersection));

        if (intersection == COMPLETELY_OUTSIDE) outsideTensorsMask |= tensorBit;
        if (intersection == INTERSECTS) intersectTensorsMask |= tensorBit;
    }

    LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                               "End computing intersections");

    if (!projectionFound)
    {
        // Everything is static -> everything is inside
        LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                                   "No dynamic projections, thus returning COMPLETELY_INSIDE");
        return COMPLETELY_INSIDE;
    }

    // ALL tensors have at least one dim that is completely outside
    if (outsideTensorsMask == allTensorsMask)
    {
        LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                                   "All tensors are outside, thus returning COMPLETELY_OUTSIDE");
        return COMPLETELY_OUTSIDE;
    }

    // At least ONE tensor has at least one dim that intersects
    if (intersectTensorsMask)
    {
        LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                                   "Have an intersecting tensor, thus returning INTERSECTS");
        return INTERSECTS;
    }

    // Not tensors outside and no tensors that intersect -> everything is inside

    LOG_TRACE_DYNAMIC_PATCHING("INTERSECTION_FROM_PROJECTION "
                               "All tensors are inside, thus returning COMPLETELY_INSIDE");
    return COMPLETELY_INSIDE;
}

const char* toString(RoiIntersectionType type)
{
    switch (type)
    {
        case COMPLETELY_OUTSIDE:
            return "COMPLETELY_OUTSIDE";
        case COMPLETELY_INSIDE:
            return "COMPLETELY_INSIDE";
        case INTERSECTS:
            return "INTERSECTS";
    }
    return "UNKNOWN";
}
