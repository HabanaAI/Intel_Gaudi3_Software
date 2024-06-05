#pragma once

// clang-format off
// ================================================================================================================== //
/*
    post graph linear ranges calculation sub-group.
*/
// group members
#define POST_GRAPH_CODE_GEN_LINEAR_RANGES_GROUP_MEMBERS         PASS_ID_ASSIGN_ADDRESSES_TO_TENSOR_ROIS,\
                                                                PASS_ID_PROJECT_NODE_ROIS,\
                                                                PASS_ID_PATCH_MME_DESCRIPTORS,\
                                                                PASS_ID_CALCULATE_TENSOR_ROIS_LINEAR_RANGES

// group member dependencies
#define PROJECT_NODE_ROIS_DEPENDENCY_SET

#define ASSIGN_ADDRESSES_TO_TENSOR_ROIS_DEPENDENCY_SET          PASS_ID_PROJECT_NODE_ROIS

#define PATCH_MME_DESCRIPTORS_DEPENDENCY_SET                    PASS_ID_ASSIGN_ADDRESSES_TO_TENSOR_ROIS

#define CALCULATE_TENSOR_ROIS_LINEAR_RANGES_DEPENDENCY_SET      PASS_ID_ASSIGN_ADDRESSES_TO_TENSOR_ROIS,\
                                                                PASS_ID_PATCH_MME_DESCRIPTORS

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on