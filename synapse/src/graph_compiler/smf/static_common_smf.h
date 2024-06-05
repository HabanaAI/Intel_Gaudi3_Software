#pragma once
#include "smf/shape_func_registry.h"
#include "smf/shape_manipulation_functions.h"

// Functions listed below will be auto-registered by ShapeFuncRegistry

#define ENTRY(id, func)                                                                                                \
    {                                                                                                                  \
        id, func, #id                                                                                                  \
    }

// This file should be only included in shape_func_registry.cpp
// Including it elsewhere should result in a linker error
namespace _do_not_use
{
char static_common_smf_h_included_twice = 0;
};

inline constexpr StaticSmfEntry commonStaticSMFs[] = {
    // TODO replace this as we have the first common SMF
    ENTRY(SMF_PATCH_ON_ZERO_SIZE, patchOnZeroSizeShapeManipulationFunction),
    ENTRY(SMF_PATCH_ON_ZERO_SIZE_FIRST_INPUT, patchOnZeroSizeFirstInputShapeManipulationFunction),
    ENTRY(SMF_DYNAMIC_OFFSET, dynamicOffsetShapeManipulationFunction),
    ENTRY(SMF_MME_PADDING, dynamicMmePaddingShapeManipulationFunction),
    ENTRY(SMF_DMA_VIEW_STRIDE, viewSizeStrideShapeManipulationFunction),
    ENTRY(SMF_DMA_VIEW_OFFSET, viewBaseAddressShapeManipulationFunction),
    ENTRY(SMF_DMA_SLICE_STRIDE, sliceStrideShapeManipulationFunction),
    ENTRY(SMF_DMA_SLICE_OFFSET, sliceBaseAddressShapeManipulationFunction),
    ENTRY(SMF_MANY_STRIDES, bulkSizeStrideShapeManipulationFunction),
    ENTRY(SMF_LAST_STRIDE, lastStrideShapeManipulationFunction),
    ENTRY(SMF_DMA_BASEADDR, dmaBaseAddressManipulationFunction)};
;

#undef ENTRY
