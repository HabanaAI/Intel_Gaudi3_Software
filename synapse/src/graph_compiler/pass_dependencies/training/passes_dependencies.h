#pragma once
// include group members internal dependencies
#include "pre_compile_group.h"
#include "lowering_group.h"
#include "be_group.h"
#include "post_graph_group.h"

// clang-format off
// ################################################################################################################## //
/*
    pre compilation group.
*/
#define PRE_COMPILE_GROUP_DEPENDENCY_SET
#define LOWERING_GROUP_DEPENDENCY_SET                      GROUP_ID_PRE_COMPILE_GROUP
#define BE_GROUP_DEPENDENCY_SET                            GROUP_ID_LOWERING_GROUP
#define POST_GRAPH_GROUP_DEPENDENCY_SET                    GROUP_ID_BE_GROUP
// clang-format on