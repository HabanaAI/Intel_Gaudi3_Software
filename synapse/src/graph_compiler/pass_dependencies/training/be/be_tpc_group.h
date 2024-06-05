#pragma once

// clang-format off
// ================================================================================================================== //
/*
    backend tpc-loading passes sub-group.

*/
// group members
#define BE_TPC_GROUP_MEMBERS                                                   PASS_ID_HANDLE_RMW_TPC_KERNELS,\
                                                                               PASS_ID_MARK_REDUCTION_INPUTS,\
                                                                               PASS_ID_LOAD_TPC_KERNELS,\
                                                                               PASS_ID_OPTIMIZE_TPC_KERNELS

// group member dependencies

#define HANDLE_RMW_TPC_KERNELS_DEPENDENCY_SET

#define MARK_REDUCTION_INPUTS_DEPENDENCY_SET                                  PASS_ID_HANDLE_RMW_TPC_KERNELS

#define LOAD_TPC_KERNELS_DEPENDENCY_SET                                       PASS_ID_MARK_REDUCTION_INPUTS,\
                                                                              PASS_ID_HANDLE_RMW_TPC_KERNELS

#define OPTIMIZE_TPC_KERNELS_DEPENDENCY_SET                                   PASS_ID_LOAD_TPC_KERNELS

// ================================================================================================================== //
// ################################################################################################################## //

// clang-format on