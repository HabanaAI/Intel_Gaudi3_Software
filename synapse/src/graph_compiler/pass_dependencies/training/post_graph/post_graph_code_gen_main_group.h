#pragma once

// clang-format off
// ================================================================================================================== //
/*
    post graph code generation actual code gen sub-group
*/
// group members
#define POST_GRAPH_CODE_GEN_MAIN_GROUP_MEMBERS                  PASS_ID_CREATE_DMA_DISPATCHERS,\
                                                                PASS_ID_ALLOCATE_SYNCS,\
                                                                PASS_ID_SIGNAL_OUT_FROM_GRAPH,\
                                                                PASS_ID_SPLIT_TO_PHYSICAL_ROIS,\
                                                                PASS_ID_GENERATE_WORK_DISTRIBUTION,\
                                                                PASS_ID_SET_ROI_SHAPE_TYPE,\
                                                                PASS_ID_GENERATE_CACHE_MAINTENANCE_TASKS,\
                                                                PASS_ID_PATCH_MME_MCIDS

// group member dependencies
#define CREATE_DMA_DISPATCHERS_DEPENDENCY_SET

#define ALLOCATE_SYNCS_DEPENDENCY_SET                           PASS_ID_CREATE_DMA_DISPATCHERS

#define GENERATE_CACHE_MAINTENANCE_TASKS_DEPENDENCY_SET         PASS_ID_ALLOCATE_SYNCS

#define PATCH_MME_MCIDS_DEPENDENCY_SET                          PASS_ID_GENERATE_CACHE_MAINTENANCE_TASKS

#define SIGNAL_OUT_FROM_GRAPH_DEPENDENCY_SET                    PASS_ID_ALLOCATE_SYNCS

#define SPLIT_TO_PHYSICAL_ROIS_DEPENDENCY_SET                   PASS_ID_ALLOCATE_SYNCS

#define GENERATE_WORK_DISTRIBUTION_DEPENDENCY_SET               PASS_ID_SPLIT_TO_PHYSICAL_ROIS,\
                                                                PASS_ID_ALLOCATE_SYNCS

#define SET_ROI_SHAPE_TYPE_DEPENDENCY_SET                       PASS_ID_SPLIT_TO_PHYSICAL_ROIS,\
                                                                PASS_ID_ALLOCATE_SYNCS

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on