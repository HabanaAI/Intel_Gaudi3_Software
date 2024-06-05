#pragma once

// clang-format off
// ================================================================================================================== //
/*
    post graph tensor and memory allocation sub-group.
*/
// group members
#define POST_GRAPH_MEMORY_GROUP_MEMBERS                         PASS_ID_ALLOCATE_TPC_KERNELS,\
                                                                PASS_ID_SET_NON_PERSISTENT_SECTION_INFO,\
                                                                PASS_ID_ALLOCATE_TENSORS,\
                                                                PASS_ID_VALIDATE_MEMORY_ALLOCATION,\
                                                                PASS_ID_VALIDATE_DMA_NODES,\
                                                                PASS_ID_MANAGE_BASE_REGS_CACHE

// group member dependencies
#define ALLOCATE_TPC_KERNELS_DEPENDENCY_SET

#define SET_NON_PERSISTENT_SECTION_INFO_DEPENDENCY_SET

#define ALLOCATE_TENSORS_DEPENDENCY_SET                         PASS_ID_SET_NON_PERSISTENT_SECTION_INFO

#define MANAGE_BASE_REGS_CACHE_DEPENDENCY_SET                   PASS_ID_SET_NON_PERSISTENT_SECTION_INFO,\
                                                                PASS_ID_ALLOCATE_TENSORS

// validations
#define VALIDATE_MEMORY_ALLOCATION_DEPENDENCY_SET               PASS_ID_ALLOCATE_TPC_KERNELS,\
                                                                PASS_ID_ALLOCATE_TENSORS,\
                                                                PASS_ID_SET_NON_PERSISTENT_SECTION_INFO

#define VALIDATE_DMA_NODES_DEPENDENCY_SET                       PASS_ID_ALLOCATE_TPC_KERNELS,\
                                                                PASS_ID_ALLOCATE_TENSORS,\
                                                                PASS_ID_SET_NON_PERSISTENT_SECTION_INFO

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on