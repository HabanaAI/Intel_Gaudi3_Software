#pragma once

#include <cstdint>

#include "recipe.h"
#include "smf/shape_func_registry.h"

#include "recipe_metadata.h"

#define GET_BIT_FIELD_WIDTH(T, f)                                                                                      \
    []() constexpr->unsigned int                                                                                       \
    {                                                                                                                  \
        T            t {};                                                                                             \
        unsigned int bitCount = 0;                                                                                     \
        for (t.f = 1; t.f != 0; t.f <<= 1, bitCount++)                                                                 \
            ;                                                                                                          \
        return bitCount;                                                                                               \
    }                                                                                                                  \
    ()

static constexpr int INDEX_NOT_APPLICABLE = -1;

enum RoiIntersectionType
{
    COMPLETELY_INSIDE,
    COMPLETELY_OUTSIDE,
    INTERSECTS
};

RoiIntersectionType max(RoiIntersectionType first, RoiIntersectionType second);

RoiIntersectionType getDimIntersectionType(tensor_roi_t& roiTensors, tensor_info_t* tensor, uint64_t dim);
RoiIntersectionType getIntersectionType(tensor_roi_t& roiTensors, tensor_info_t* tensor);
RoiIntersectionType getIntersectionType(tensor_roi_t* rois, tensor_info_t** tensors, uint64_t TensorsNr);
RoiIntersectionType getIntersectionTypeFromAllTensors(const ShapeManipulationParams* params);
RoiIntersectionType getIntersectionTypeFromProjection(const ShapeManipulationParams* params);

const char* toString(RoiIntersectionType type);

template<class ParamsT>
inline uint64_t getStrideForDimension(const tensor_info_t* tensor, unsigned dim, ParamsT& params)
{
    return dim == 0 ? params.element_size : tensor->strides[dim - 1];
}
