#pragma once

// clang-format off
// ================================================================================================================== //
/*
    post graph code generation split to logical roi sub-group
*/
// group members
#define POST_GRAPH_CODE_GEN_LOGICAL_ROI_GROUP_MEMBERS           PASS_ID_SET_DMA_PARALLEL_LEVEL,\
                                                                PASS_ID_GENERATE_ROIS,\
                                                                PASS_ID_SPLIT_TPC_DIMS,\
                                                                PASS_ID_DISABLE_BUNDLE_ROIS,\
                                                                PASS_ID_SPLIT_TO_LOGICAL_ROIS,\
                                                                PASS_ID_GENERATE_MME_DESCRIPTORS,\
                                                                PASS_ID_SPLIT_TO_DCORE_ROIS

// group member dependencies
#define SET_DMA_PARALLEL_LEVEL_DEPENDENCY_SET

#define GENERATE_ROIS_DEPENDENCY_SET

#define SPLIT_TPC_DIMS_DEPENDENCY_SET                           PASS_ID_GENERATE_ROIS

#define SPLIT_TO_DCORE_ROIS_DEPENDENCY_SET                      PASS_ID_GENERATE_ROIS

#define DISABLE_BUNDLE_ROIS_DEPENDENCY_SET

#define GENERATE_MME_DESCRIPTORS_DEPENDENCY_SET                 PASS_ID_DISABLE_BUNDLE_ROIS,\
                                                                PASS_ID_GENERATE_ROIS,\
                                                                PASS_ID_SPLIT_TO_DCORE_ROIS

#define SPLIT_TO_LOGICAL_ROIS_DEPENDENCY_SET                    PASS_ID_GENERATE_ROIS,\
                                                                PASS_ID_DISABLE_BUNDLE_ROIS,\
                                                                PASS_ID_SPLIT_TPC_DIMS,\
                                                                PASS_ID_SET_DMA_PARALLEL_LEVEL,\
                                                                PASS_ID_SPLIT_TO_DCORE_ROIS

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on