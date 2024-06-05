#pragma once
#include "shape_func_registry.h"
#include "sif/shape_inference_functions.h"

// Functions listed below will be auto-registered by ShapeFuncRegistry

#define ENTRY(id, func)                                                                                                \
    {                                                                                                                  \
        id, func, #id                                                                                                  \
    }

namespace gaudi2
{
// This file should be only included in shape_func_registry.cpp
// Including it elsewhere should result in a linker error
namespace _do_not_use
{
char static_gaudi2_smf_h_included_twice = 0;
}

extern non_ptr_smf_t mmeValidElementsSMF;
extern non_ptr_smf_t mmeDynamicExecutionSMF;
extern non_ptr_smf_t mmeSyncObjectSMF;
extern non_ptr_smf_t tpcIndexSpaceGaudi2SMF;

inline constexpr StaticSmfEntry staticSMFs[] = {ENTRY(SMF_GAUDI2_MME_SIZE, mmeValidElementsSMF),
                                                ENTRY(SMF_GAUDI2_MME_NULL_DESC, mmeDynamicExecutionSMF),
                                                ENTRY(SMF_GAUDI2_MME_SYNC, mmeSyncObjectSMF),
                                                ENTRY(SMF_TPC_INDEX_SPACE_GAUDI2, tpcIndexSpaceGaudi2SMF),
};

}  // namespace gaudi2

#undef ENTRY
