#pragma once
#include "shape_func_registry.h"
#include "sif/shape_inference_functions.h"

// Functions listed below will be auto-registered by ShapeFuncRegistry

#define ENTRY(id, func)                                                                                                \
    {                                                                                                                  \
        id, func, #id                                                                                                  \
    }

namespace gaudi3
{
// This file should be only included in shape_func_registry.cpp
// Including it elsewhere should result in a linker error
namespace _do_not_use
{
char static_gaudi3_smf_h_included_twice = 0;
}

extern non_ptr_smf_t tpcIndexSpaceGaudi3SMF;
extern non_ptr_smf_t mmeValidElementsGaudi3SMF;

inline constexpr StaticSmfEntry staticSMFs[] = {
    ENTRY(SMF_TPC_INDEX_SPACE_GAUDI3, tpcIndexSpaceGaudi3SMF),
    ENTRY(SMF_GAUDI3_MME_SIZE, mmeValidElementsGaudi3SMF)

};

}  // namespace gaudi2

#undef ENTRY
