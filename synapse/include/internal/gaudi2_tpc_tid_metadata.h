#pragma once

// This header cannot be included
// together with gaudi3_tpc_tid_metadata.h
// therefore both must be kept sepatare from
// recipe_metadata

#include <stdint.h>

#include "gaudi2_arc_eng_packets.h"

#include "common_tpc_tid_metadata.h"

// One giant metadata for one giant patch point per node

struct tpc_sm_params_gaudi2_t
{
    // tpc_wd_ctxt_t below is a Gaudi2 struct
    // defined in gaudi2_arc_eng_packets.h above.
    // There is a Gaudi3 struct by the same name
    // (which is an ODR violation, but it exists in
    // Synapse for a long time).
    // Namespaces cannot be used for reasons beyond our control.
    index_space_tensor_t      m_indexSpaceCopy;
    uint32_t                  m_dimensions_mask;
    tpc_sm_params_t           m_dimensions[DYNAMIC_EXECUTE_MAX_DIMENSIONS];
};
