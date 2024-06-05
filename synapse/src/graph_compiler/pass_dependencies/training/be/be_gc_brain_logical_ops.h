#pragma once

#include <array>
#include "habana_pass.h"

// clang-format off
// ================================================================================================================== //
/*
    handle logical ops backend passes sub-group.
    this group handles all the logical ops related work that if performed after the graph stops changing.
    any pass running beyond this point should be aware of tensor aliases and GC control edges
*/
// group members
#define BE_GC_BRAIN_LOGICAL_OPS_GROUP_MEMBERS                   PASS_ID_HANDLE_LOGICAL_OPERATIONS,\
                                                                PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS,\
                                                                PASS_ID_HANDLE_LOGICAL_OPERATIONS_POST_PROCESS,\
                                                                PASS_ID_HANDLE_MEMORY_REUSE,\
                                                                PASS_ID_HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES,\
                                                                PASS_ID_IN_PLACE_INPUT_REUSE_BINDING

// group member dependencies
#define HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS_DEPENDENCY_SET

#define HANDLE_LOGICAL_OPERATIONS_POST_PROCESS_DEPENDENCY_SET   PASS_ID_HANDLE_LOGICAL_OPERATIONS,\
                                                                PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS,\
                                                                PASS_ID_HANDLE_MEMORY_REUSE,\
                                                                PASS_ID_HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES

#define HANDLE_LOGICAL_OPERATIONS_DEPENDENCY_SET                PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS

#define HANDLE_MEMORY_REUSE_DEPENDENCY_SET                      PASS_ID_HANDLE_LOGICAL_OPERATIONS,\
                                                                PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS

#define HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES_DEPENDENCY_SET      PASS_ID_HANDLE_LOGICAL_OPERATIONS,\
                                                                PASS_ID_HANDLE_LOGICAL_OPERATIONS_PRE_PROCESS

#define IN_PLACE_INPUT_REUSE_BINDING_DEPENDENCY_SET             PASS_ID_HANDLE_LOGICAL_OPERATIONS,\
                                                                PASS_ID_HANDLE_CTRL_EDGES_FOR_LOGICAL_NODES,\
                                                                PASS_ID_HANDLE_LOGICAL_OPERATIONS_POST_PROCESS


// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on