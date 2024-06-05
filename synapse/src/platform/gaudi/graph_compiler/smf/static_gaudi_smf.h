#pragma once
#include "shape_func_registry.h"
#include "sif/shape_inference_functions.h"

// Functions listed below will be auto-registered by ShapeFuncRegistry

#define ENTRY(id, func)                                                                                                \
    {                                                                                                                  \
        id, func, #id                                                                                                  \
    }

namespace gaudi
{
// This file should be only included in shape_func_registry.cpp
// Including it elsewhere should result in a linker error
namespace _do_not_use
{
char static_gaudi_smf_h_included_twice = 0;
}

// smf
extern non_ptr_smf_t dynamicExecutionShapeManipulationFunction;
extern non_ptr_smf_t dmaSizeShapeManipulationFunction;
extern non_ptr_smf_t mmeValidElementsSMF;
extern non_ptr_smf_t tpcSizeShapeManipulationFunction;
extern non_ptr_smf_t tpcStrideShapeManipulationFunction;
extern non_ptr_smf_t tpcIndexSpaceShapeManipulationFunction;
extern non_ptr_smf_t tpcSliceStrideShapeManipulationFunction;
extern non_ptr_smf_t tpcSliceBaseAddressShapeManipulationFunction;
extern non_ptr_smf_t tcpViewStrideShapeManipulationFunction;
extern non_ptr_smf_t tpcViewBaseAddressShapeManipulationFunction;

inline constexpr StaticSmfEntry staticSMFs[] = {
    ENTRY(SMF_DYNAMIC_EXE, dynamicExecutionShapeManipulationFunction),
    ENTRY(SMF_DMA_SIZE, dmaSizeShapeManipulationFunction),
    ENTRY(SMF_MME, mmeValidElementsSMF),
    ENTRY(SMF_TPC_SIZE, tpcSizeShapeManipulationFunction),
    ENTRY(SMF_TPC_STRIDE, tpcStrideShapeManipulationFunction),
    ENTRY(SMF_TPC_INDEX_SPACE, tpcIndexSpaceShapeManipulationFunction),
    ENTRY(SMF_TPC_SLICE_STRIDE, tpcSliceStrideShapeManipulationFunction),
    ENTRY(SMF_TPC_SLICE_OFFSET, tpcSliceBaseAddressShapeManipulationFunction),
    ENTRY(SMF_TPC_VIEW_STRIDE, tcpViewStrideShapeManipulationFunction),
    ENTRY(SMF_TPC_VIEW_OFFSET, tpcViewBaseAddressShapeManipulationFunction)
};

}  // namespace gaudi

#undef ENTRY
