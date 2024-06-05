#pragma once

// include group members internal dependencies
#include "optimize_control_edges.h"

// clang-format off
// =========================== OPTIMIZE_USER_LEVEL_GRAPH_GROUP =================================================== //
// group members
#define PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH_GROUP_MEMBERS         GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION,\
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES,\
                                                                    PASS_ID_ELIMINATE_NODES_WITH_STATIC_INPUTS,\
                                                                    PASS_ID_OPTIMIZE_STRIDED_INSERT,\
                                                                    PASS_ID_FUSE_BROADCAST_TPC,\
                                                                    PASS_ID_REMOVE_CONTIGUOUS_TRANSPOSES,\
                                                                    PASS_ID_FUSE_TRANSPOSE_MME,\
                                                                    PASS_ID_FUSE_CONST_TRANSPOSE,\
                                                                    PASS_ID_REMOVE_CONTIGUOUS_RESHAPES,\
                                                                    PASS_ID_REMOVE_ZERO_SIZED_PAD,\
                                                                    PASS_ID_FUSE_BN_CONV

// group member dependencies
#define PRE_COMPILE_OPTIMIZE_CONTROL_EDGES_GROUP_DEPENDENCY_SET

#define CSE_OPTIMIZATION_DEPENDENCY_SET                             GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES

#define REMOVE_ZERO_SIZED_PAD_DEPENDENCY_SET                        GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES

#define ELIMINATE_REDUNDANT_NODES_DEPENDENCY_SET                    GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION

#define OPTIMIZE_STRIDED_INSERT_DEPENDENCY_SET                      GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES

#define FUSE_BROADCAST_TPC_DEPENDENCY_SET                           GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION,\
                                                                    PASS_ID_OPTIMIZE_STRIDED_INSERT, \
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES

#define REMOVE_CONTIGUOUS_TRANSPOSES_DEPENDENCY_SET                 GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION,\
                                                                    PASS_ID_OPTIMIZE_STRIDED_INSERT,\
                                                                    PASS_ID_FUSE_BROADCAST_TPC, \
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES,\
                                                                    PASS_ID_REMOVE_ZERO_SIZED_PAD

#define FUSE_TRANSPOSE_MME_DEPENDENCY_SET                           GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION,\
                                                                    PASS_ID_OPTIMIZE_STRIDED_INSERT,\
                                                                    PASS_ID_FUSE_BROADCAST_TPC,\
                                                                    PASS_ID_REMOVE_CONTIGUOUS_TRANSPOSES, \
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES,\
                                                                    PASS_ID_REMOVE_ZERO_SIZED_PAD

#define FUSE_CONST_TRANSPOSE_DEPENDENCY_SET                         GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION,\
                                                                    PASS_ID_OPTIMIZE_STRIDED_INSERT,\
                                                                    PASS_ID_FUSE_BROADCAST_TPC,\
                                                                    PASS_ID_REMOVE_CONTIGUOUS_TRANSPOSES, \
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES,\
                                                                    PASS_ID_REMOVE_ZERO_SIZED_PAD

#define REMOVE_CONTIGUOUS_RESHAPES_DEPENDENCY_SET                   GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION,\
                                                                    PASS_ID_OPTIMIZE_STRIDED_INSERT,\
                                                                    PASS_ID_FUSE_BROADCAST_TPC,\
                                                                    PASS_ID_REMOVE_CONTIGUOUS_TRANSPOSES,\
                                                                    PASS_ID_FUSE_TRANSPOSE_MME,\
                                                                    PASS_ID_FUSE_CONST_TRANSPOSE,\
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES,\
                                                                    PASS_ID_REMOVE_ZERO_SIZED_PAD

#define ELIMINATE_NODES_WITH_STATIC_INPUTS_DEPENDENCY_SET           GROUP_ID_PRE_COMPILE_OPTIMIZE_CONTROL_EDGES,\
                                                                    PASS_ID_CSE_OPTIMIZATION,\
                                                                    PASS_ID_ELIMINATE_REDUNDANT_NODES,\
                                                                    REMOVE_CONTIGUOUS_TRANSPOSES_DEPENDENCY_SET,\
                                                                    REMOVE_CONTIGUOUS_RESHAPES_DEPENDENCY_SET

#define FUSE_BN_CONV_DEPENDENCY_SET                                 PASS_ID_ELIMINATE_NODES_WITH_STATIC_INPUTS
// clang-format on