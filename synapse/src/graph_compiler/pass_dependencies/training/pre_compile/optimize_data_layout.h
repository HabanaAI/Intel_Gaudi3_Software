#pragma once

// clang-format off
// =========================== OPTIMIZE_DATA_LAYOUT_GROUP =================================================== //
// group members
#define PRE_COMPILE_OPTIMIZE_DATA_LAYOUT_GROUP_MEMBERS              PASS_ID_TRANSPOSE_DONT_CARE_NODES,\
                                                                    PASS_ID_HANDLE_PERMUTED_TENSORS



// group member dependencies
#define TRANSPOSE_DONT_CARE_NODES_DEPENDENCY_SET

#define HANDLE_PERMUTED_TENSORS_DEPENDENCY_SET                      PASS_ID_TRANSPOSE_DONT_CARE_NODES

// clang-format on