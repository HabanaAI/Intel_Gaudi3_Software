#pragma once

// clang-format off
// ================================================================================================================== //
/*
    backend decision passes that run after graph finished mutating entirely.
    the graph connections must not change beyond this point.
*/
// group members
#define BE_POST_GRAPH_GROUP_MEMBERS                                                PASS_ID_SET_REDUCTION_MEMSET,\
                                                                                   PASS_ID_IN_PLACE_INPUT_REUSE_SUGGESTION,\
                                                                                   PASS_ID_SET_TENSOR_IN_HBM,\
                                                                                   PASS_ID_ALIGN_TRANSPOSE_VIA_GEMM_OUTPUT,\
                                                                                   PASS_ID_SET_FLASH_ATTN_SCHEDULE

// group member dependencies
#define SET_REDUCTION_MEMSET_DEPENDENCY_SET

#define IN_PLACE_INPUT_REUSE_SUGGESTION_DEPENDENCY_SET

#define SET_TENSOR_IN_HBM_DEPENDENCY_SET                                           PASS_ID_SET_REDUCTION_MEMSET,\
                                                                                   PASS_ID_IN_PLACE_INPUT_REUSE_SUGGESTION

#define ALIGN_TRANSPOSE_VIA_GEMM_OUTPUT_DEPENDENCY_SET                             PASS_ID_SET_TENSOR_IN_HBM

#define SET_FLASH_ATTN_SCHEDULE_DEPENDENCY_SET

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on