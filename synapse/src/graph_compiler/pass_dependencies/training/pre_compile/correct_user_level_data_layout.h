#pragma once

// clang-format off
// ################################################################################################################## //
// =========================== PRE_COMPILE_CORRECT_USER_LEVEL_GROUP =================================================== //
// data layout sub-group - correcting the user level graph from data layout perspective

#define CORRECT_USER_GRAPH_DATA_LAYOUT_GROUP_MEMBERS            PASS_ID_SET_SUPPORTED_LAYOUTS,\
                                                                PASS_ID_SET_HABANA_LAYOUTS,\
                                                                PASS_ID_ADJUST_DATA_LAYOUT,\
                                                                PASS_ID_VALIDATE_NODES_LAYOUT

#define SET_SUPPORTED_LAYOUTS_DEPENDENCY_SET

#define SET_HABANA_LAYOUTS_DEPENDENCY_SET                       PASS_ID_SET_SUPPORTED_LAYOUTS

#define ADJUST_DATA_LAYOUT_DEPENDENCY_SET                       PASS_ID_SET_HABANA_LAYOUTS,\
                                                                PASS_ID_SET_SUPPORTED_LAYOUTS

#define VALIDATE_NODES_LAYOUT_DEPENDENCY_SET                    PASS_ID_SET_HABANA_LAYOUTS,\
                                                                PASS_ID_SET_SUPPORTED_LAYOUTS,\
                                                                PASS_ID_ADJUST_DATA_LAYOUT

// clang-format on
