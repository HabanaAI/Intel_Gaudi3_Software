#pragma once

#include "be_gc_brain_pre_slicing.h"
#include "be_gc_brain_post_slicing.h"
#include "be_gc_brain_logical_ops.h"

// clang-format off
// ================================================================================================================== //
/*
    GC brain passes sub-group.

*/
// group members
#define BE_GC_BRAIN_GROUP_MEMBERS                           GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_RUN_LAYERED_BRAIN,\
                                                            PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,\
                                                            GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,\
                                                            GROUP_ID_BE_GC_BRAIN_LOGICAL_OPS_GROUP,\
                                                            PASS_ID_BUNDLE_NODES_SCHEDULE,\
                                                            GROUP_ID_BE_GC_BRAIN_MEMORY_GROUP,\
                                                            PASS_ID_HANDLE_PARTIALS_WRITES,\
                                                            PASS_ID_BUNDLE_MEMORY_MANAGEMENT

// group member dependencies
#define BE_GC_BRAIN_PRE_SLICING_DEPENDENCY_SET

#define RUN_LAYERED_BRAIN_DEPENDENCY_SET                    GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP

#define SLICE_GRAPH_TO_SRAM_CAPACITY_DEPENDENCY_SET         GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_RUN_LAYERED_BRAIN

#define BE_GC_BRAIN_POST_SLICING_DEPENDENCY_SET             GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,\
                                                            PASS_ID_RUN_LAYERED_BRAIN

#define BE_GC_BRAIN_LOGICAL_OPS_DEPENDENCY_SET              GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,\
                                                            GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,\
                                                            PASS_ID_RUN_LAYERED_BRAIN

#define BUNDLE_NODES_SCHEDULE_DEPENDENCY_SET                GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,\
                                                            GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,\
                                                            GROUP_ID_BE_GC_BRAIN_LOGICAL_OPS_GROUP,\
                                                            PASS_ID_HANDLE_PARTIALS_WRITES,\
                                                            PASS_ID_RUN_LAYERED_BRAIN

#define BE_GC_BRAIN_MEMORY_GROUP_DEPENDENCY_SET             GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,\
                                                            GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,\
                                                            GROUP_ID_BE_GC_BRAIN_LOGICAL_OPS_GROUP,\
                                                            PASS_ID_BUNDLE_NODES_SCHEDULE,\
                                                            PASS_ID_RUN_LAYERED_BRAIN

#define BUNDLE_MEMORY_MANAGEMENT_DEPENDENCY_SET             GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,\
                                                            GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,\
                                                            GROUP_ID_BE_GC_BRAIN_LOGICAL_OPS_GROUP,\
                                                            PASS_ID_BUNDLE_NODES_SCHEDULE,\
                                                            PASS_ID_HANDLE_PARTIALS_WRITES,\
                                                            GROUP_ID_BE_GC_BRAIN_MEMORY_GROUP,\
                                                            PASS_ID_RUN_LAYERED_BRAIN

#define HANDLE_PARTIALS_WRITES_DEPENDENCY_SET               GROUP_ID_BE_GC_BRAIN_PRE_SLICING_GROUP,\
                                                            PASS_ID_SLICE_GRAPH_TO_SRAM_CAPACITY,\
                                                            GROUP_ID_BE_GC_BRAIN_POST_SLICING_GROUP,\
                                                            GROUP_ID_BE_GC_BRAIN_LOGICAL_OPS_GROUP,\
                                                            PASS_ID_RUN_LAYERED_BRAIN

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on