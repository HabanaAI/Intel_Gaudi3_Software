#pragma once

#include "be_tpc_group.h"
#include "be_mme_optimization_group.h"

// clang-format off
// ================================================================================================================== //
/*
    backend passes sub-group.

*/
// group members
#define BE_MAIN_GROUP_MEMBERS                                                   GROUP_ID_BE_TPC_GROUP,\
                                                                                GROUP_ID_BE_MME_OPTIMIZATION_GROUP,\
                                                                                PASS_ID_INSERT_SERIALIZE_DESERIALIZE,\
                                                                                PASS_ID_ADD_H2D_OP,\
                                                                                PASS_ID_EXTRACT_DATA_MOVEMENT_MULTI_NODES

// group member dependencies

#define INSERT_SERIALIZE_DESERIALIZE_DEPENDENCY_SET

#define ADD_H2D_OP_DEPENDENCY_SET

#define BE_TPC_GROUP_DEPENDENCY_SET                                            PASS_ID_INSERT_SERIALIZE_DESERIALIZE,\
                                                                               PASS_ID_ADD_H2D_OP

#define BE_MME_OPTIMIZATION_GROUP_DEPENDENCY_SET                               GROUP_ID_BE_TPC_GROUP

#define EXTRACT_DATA_MOVEMENT_MULTI_NODES_DEPENDENCY_SET                       PASS_ID_INSERT_SERIALIZE_DESERIALIZE,\
                                                                               PASS_ID_ADD_H2D_OP,\
                                                                               GROUP_ID_BE_TPC_GROUP,\
                                                                               GROUP_ID_BE_MME_OPTIMIZATION_GROUP

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on