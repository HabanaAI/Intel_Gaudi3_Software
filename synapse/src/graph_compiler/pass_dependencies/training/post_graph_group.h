#pragma once

// include group members internal dependencies
#include "post_graph/post_graph_code_gen_group.h"
#include "post_graph/post_graph_memory_group.h"
#include "post_graph/post_graph_validation_group.h"

// clang-format off
// ================================================================================================================== //
/*
    post graph compilation group.
    all passes here run after the graph has finished mutating
*/
// group members
#define POST_GRAPH_GROUP_MEMBERS                                               GROUP_ID_POST_GRAPH_VALIDATIONS,\
                                                                               GROUP_ID_POST_GRAPH_MEMORY,\
                                                                               GROUP_ID_POST_GRAPH_CODE_GEN,\
                                                                               PASS_ID_GRAPH_VISUALIZATION_POST,\
                                                                               PASS_ID_GENERATE_PROFILER_DEBUG_INFO

// group member dependencies
#define POST_GRAPH_VALIDATION_GROUP_DEPENDENCY_SET

#define POST_GRAPH_MEMORY_GROUP_DEPENDENCY_SET                                 GROUP_ID_POST_GRAPH_VALIDATIONS

#define GENERATE_PROFILER_DEBUG_INFO_DEPENDENCY_SET                            GROUP_ID_POST_GRAPH_VALIDATIONS

#define POST_GRAPH_CODE_GEN_GROUP_DEPENDENCY_SET                               GROUP_ID_POST_GRAPH_VALIDATIONS,\
                                                                               GROUP_ID_POST_GRAPH_MEMORY,\
                                                                               PASS_ID_GENERATE_PROFILER_DEBUG_INFO

#define GRAPH_VISUALIZATION_POST_DEPENDENCY_SET                                GROUP_ID_POST_GRAPH_VALIDATIONS,\
                                                                               GROUP_ID_POST_GRAPH_MEMORY,\
                                                                               GROUP_ID_POST_GRAPH_CODE_GEN,\
                                                                               PASS_ID_GENERATE_PROFILER_DEBUG_INFO
// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on