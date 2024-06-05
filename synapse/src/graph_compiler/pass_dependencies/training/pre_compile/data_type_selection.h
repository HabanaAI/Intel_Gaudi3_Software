#pragma once

// clang-format off
// =========================== DATA_TYPE_SELECTION_GROUP =================================================== //
// group members
#define PRE_COMPILE_DATA_TYPE_SELECTION_GROUP_MEMBERS               PASS_ID_CALC_DYNAMIC_RANGE,\
                                                                    PASS_ID_UPDATE_MME_NODES_PRECISION

// group member dependencies
#define CALC_DYNAMIC_RANGE_DEPENDENCY_SET

#define UPDATE_MME_NODES_PRECISION_DEPENDENCY_SET                   PASS_ID_CALC_DYNAMIC_RANGE

// clang-format on