#pragma once

#include <cstdint>

enum RoiShapeType
{
    UNSPECIFIED,
    FIXED_ROI,
    DYNAMIC_ROI
};

RoiShapeType getRoiShapeType(uint32_t dimCount, const TSize* minSizes, const TOffset* roiOffset, const TSize* roiSize);