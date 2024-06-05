#pragma once

// clang-format off
// ================================================================================================================== //
/*
    post graph validations sub-group
*/
// group members
#define POST_GRAPH_VALIDATION_GROUP_MEMBERS                                     PASS_ID_VALIDATE_MEMORY_SECTION_TENSORS,\
                                                                                PASS_ID_VALIDATE_DYNAMIC_SHAPES,\
                                                                                PASS_ID_VERIFY_MEMSET_BROADCAST_OUTPUT_SHAPE,\
                                                                                PASS_ID_VALIDATE_MME_NODES,\
                                                                                PASS_ID_VALIDATE_ATOMIC_NODES,\
                                                                                PASS_ID_VALIDATE_EXECUTION_SCHEDULE_BUNDLES,\
                                                                                PASS_ID_GC_PERF_CHECKS,\
                                                                                PASS_ID_CHECK_MAX_DIMS_POST

// group member dependencies
#define VALIDATE_MEMORY_SECTION_TENSORS_DEPENDENCY_SET

#define VALIDATE_DYNAMIC_SHAPES_DEPENDENCY_SET                                  PASS_ID_CHECK_MAX_DIMS_POST

#define VERIFY_MEMSET_BROADCAST_OUTPUT_SHAPE_DEPENDENCY_SET

#define VALIDATE_MME_NODES_DEPENDENCY_SET

#define VALIDATE_ATOMIC_NODES_DEPENDENCY_SET

#define VALIDATE_EXECUTION_SCHEDULE_BUNDLES_DEPENDENCY_SET

#define GC_PERF_CHECKS_DEPENDENCY_SET

#define CHECK_MAX_DIMS_POST_DEPENDENCY_SET

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on