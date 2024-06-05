#pragma once

// include group members internal dependencies
#include "correct_user_level_data_layout.h"
#include "data_type_selection.h"
#include "habana_pass.h"

// clang-format off
// ================================================================================================================== //
// ################################################################################################################## //
// =========================== PRE_COMPILE_CORRECT_USER_LEVEL_GROUP =================================================== //
// group members
#define PRE_COMPILE_CORRECT_USER_LEVEL_GROUP_MEMBERS            PASS_ID_FUSE_WAITS,\
                                                                PASS_ID_LINK_REDUCTION_MEMSET_SHAPES,\
                                                                PASS_ID_CHECK_MAX_DIMS_PRE,\
                                                                PASS_ID_INTERNAL_TENSORS_DYNAMIC_SHAPE,\
                                                                GROUP_ID_CORRECT_USER_GRAPH_DATA_LAYOUT,\
                                                                GROUP_ID_PRE_COMPILE_DATA_TYPE_SELECTION,\
                                                                PASS_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE

// group member dependencies
#define CORRECT_USER_GRAPH_DATA_LAYOUT_DEPENDENCY_SET

#define CHECK_MAX_DIMS_PRE_DEPENDENCY_SET                       GROUP_ID_CORRECT_USER_GRAPH_DATA_LAYOUT

#define FUSE_WAITS_DEPENDENCY_SET

#define LINK_REDUCTION_MEMSET_SHAPES_DEPENDENCY_SET

#define INTERNAL_TENSORS_DYNAMIC_SHAPE_DEPENDENCY_SET           PASS_ID_CHECK_MAX_DIMS_PRE,\
                                                                GROUP_ID_CORRECT_USER_GRAPH_DATA_LAYOUT,\
                                                                PASS_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE

#define PRE_COMPILE_DATA_TYPE_SELECTION_GROUP_DEPENDENCY_SET    PASS_ID_FUSE_WAITS,\
                                                                PASS_ID_LINK_REDUCTION_MEMSET_SHAPES,\
                                                                PASS_ID_INTERNAL_TENSORS_DYNAMIC_SHAPE,\
                                                                GROUP_ID_CORRECT_USER_GRAPH_DATA_LAYOUT

// clang-format on