#pragma once

// clang-format off
// ================================================================================================================== //
// ################################################################################################################## //
// ===================================== HUGE_TENSORS_HANDLING_GROUP ================================================ //
// group members
#define HUGE_TENSORS_HANDLING_GROUP_MEMBERS                      PASS_ID_HANDLE_HUGE_TENSORS, \
                                                                 PASS_ID_REMOVE_OPPOSITE_SPLIT_CONCAT

// group member dependencies
#define HANDLE_HUGE_TENSORS_DEPENDENCY_SET

#define REMOVE_OPPOSITE_SPLIT_CONCAT_DEPENDENCY_SET              PASS_ID_HANDLE_HUGE_TENSORS

// clang-format on
