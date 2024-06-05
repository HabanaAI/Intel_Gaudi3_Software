#pragma once

// clang-format off
// =========================== OPTIMIZE_CONTROL_EDGES_GROUP =================================================== //
// group members
#define PRE_COMPILE_OPTIMIZE_CONTROL_EDGES_GROUP_MEMBERS              PASS_ID_CONTROL_DEP_RELAXATION,\
                                                                      PASS_ID_SPILL_PERSISTENT_TENSORS

// group member dependencies
#define HANDLE_CONTROL_DEP_RELAXATION_DEPENDENCY_SET

#define SPILL_PERSISTENT_TENSORS_DEPENDENCY_SET                       PASS_ID_CONTROL_DEP_RELAXATION
// clang-format on