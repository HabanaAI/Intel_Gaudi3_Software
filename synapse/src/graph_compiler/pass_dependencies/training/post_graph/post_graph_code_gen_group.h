#pragma once

#include "post_graph_code_gen_logical_roi_group.h"
#include "post_graph_code_gen_linear_ranges_group.h"
#include "post_graph_code_gen_main_group.h"

// clang-format off
// ================================================================================================================== //
/*
    post graph code generation code gen group
*/
// group members
#define POST_GRAPH_CODE_GEN_GROUP_MEMBERS                           GROUP_ID_POST_GRAPH_CODE_GEN_LINEAR_RANGES,\
                                                                    GROUP_ID_POST_GRAPH_CODE_GEN_LOGICAL_ROI,\
                                                                    GROUP_ID_POST_GRAPH_CODE_GEN_MAIN_GROUP

// group member dependencies
#define POST_GRAPH_CODE_GEN_LOGICAL_ROI_DEPENDENCY_SET

#define POST_GRAPH_CODE_GEN_LINEAR_RANGES_DEPENDENCY_SET            GROUP_ID_POST_GRAPH_CODE_GEN_LOGICAL_ROI

#define POST_GRAPH_CODE_GEN_MAIN_GROUP_DEPENDENCY_SET               GROUP_ID_POST_GRAPH_CODE_GEN_LOGICAL_ROI,\
                                                                    GROUP_ID_POST_GRAPH_CODE_GEN_LINEAR_RANGES

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on


