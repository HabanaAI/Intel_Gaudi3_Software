#pragma once

#include "be_tpc_fuser_group.h"

// clang-format off
// ================================================================================================================== //
/*
    backend graph optimizations sub-group.

*/
// group members
#define BE_GRAPH_OPT_GROUP_MEMBERS                                             PASS_ID_TRANSPOSE_REDUCE_DIMENSIONS,\
                                                                               PASS_ID_TRANSPOSE_FCD_BROADCAST,\
                                                                               PASS_ID_FUSE_BATCH_NORM,\
                                                                               PASS_ID_REMOVE_REDUNDANT_MEMCPY_NODES,\
                                                                               PASS_ID_REMOVE_REDUNDANT_LOGICAL_NODES,\
                                                                               PASS_ID_REMOVE_CONTIGUOUS_CAST_NODES,\
                                                                               GROUP_ID_BE_TPC_FUSER_GROUP,\
                                                                               PASS_ID_EXTRACT_PERFORMANCE_COMPLEX_GUID_NODES,\
                                                                               PASS_ID_FUSE_CAST_MME

// group member dependencies


// ================================================================================================================== //
// ################################################################################################################## //

#define TRANSPOSE_REDUCE_DIMENSIONS_DEPENDENCY_SET

#define TRANSPOSE_FCD_BROADCAST_DEPENDENCY_SET

#define FUSE_BATCH_NORM_DEPENDENCY_SET

#define REMOVE_REDUNDANT_MEMCPY_NODES_DEPENDENCY_SET

#define REMOVE_REDUNDANT_LOGICAL_NODES_DEPENDENCY_SET                          PASS_ID_REMOVE_REDUNDANT_MEMCPY_NODES

#define REMOVE_CONTIGUOUS_CAST_NODES_DEPENDENCY_SET                            PASS_ID_FUSE_BATCH_NORM

#define BE_TPC_FUSER_GROUP_DEPENDENCY_SET                                      PASS_ID_FUSE_BATCH_NORM,\
                                                                               PASS_ID_REMOVE_CONTIGUOUS_CAST_NODES

#define EXTRACT_PERFORMANCE_COMPLEX_GUID_NODES_DEPENDENCY_SET                  PASS_ID_FUSE_BATCH_NORM,\
                                                                               PASS_ID_REMOVE_CONTIGUOUS_CAST_NODES,\
                                                                               GROUP_ID_BE_TPC_FUSER_GROUP

#define FUSE_CAST_MME_DEPENDENCY_SET                                           PASS_ID_FUSE_BATCH_NORM,\
                                                                               PASS_ID_REMOVE_CONTIGUOUS_CAST_NODES,\
                                                                               PASS_ID_REMOVE_REDUNDANT_MEMCPY_NODES,\
                                                                               PASS_ID_REMOVE_REDUNDANT_LOGICAL_NODES,\
                                                                               GROUP_ID_BE_TPC_FUSER_GROUP

// clang-format on
