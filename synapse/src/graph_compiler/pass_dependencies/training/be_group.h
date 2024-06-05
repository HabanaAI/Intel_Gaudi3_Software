#pragma once

// include group members internal dependencies
#include "be/be_graph_optimizations.h"
#include "be/be_main_group.h"
#include "be/be_gc_brain.h"
#include "be/be_gc_brain_memory.h"
#include "be/be_post_graph_operations.h"

// clang-format off
// ================================================================================================================== //
/*
    GC backend sub-group.
*/
// group members
#define BE_GROUP_MEMBERS                                                       GROUP_ID_BE_GRAPH_OPT_GROUP,\
                                                                               GROUP_ID_BE_MAIN_GROUP,\
                                                                               GROUP_ID_BE_GC_BRAIN_GROUP,\
                                                                               GROUP_ID_BE_POST_GRAPH_GROUP

// group member dependencies
#define BE_GRAPH_OPT_GROUP_DEPENDENCY_SET

#define BE_MAIN_GROUP_DEPENDENCY_SET                                           GROUP_ID_BE_GRAPH_OPT_GROUP

#define BE_GC_BRAIN_GROUP_DEPENDENCY_SET                                       GROUP_ID_BE_GRAPH_OPT_GROUP,\
                                                                               GROUP_ID_BE_MAIN_GROUP

#define BE_POST_GRAPH_GROUP_DEPENDENCY_SET                                     GROUP_ID_BE_GRAPH_OPT_GROUP,\
                                                                               GROUP_ID_BE_MAIN_GROUP,\
                                                                               GROUP_ID_BE_GC_BRAIN_GROUP
// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on