#pragma once

// clang-format off
// ================================================================================================================== //
/*
    backend tpc fuser subgroup

*/
// group members
#define BE_TPC_FUSER_GROUP_MEMBERS                                            PASS_ID_TPC_FUSER, \
                                                                              PASS_ID_ADD_STATIC_SHAPE_TENSORS

// group member dependencies

#define ADD_STATIC_SHAPE_TENSORS_DEPENDENCY_SET

#define TPC_FUSER_DEPENDENCY_SET                                              PASS_ID_ADD_STATIC_SHAPE_TENSORS


// ================================================================================================================== //
// ################################################################################################################## //

// clang-format on
