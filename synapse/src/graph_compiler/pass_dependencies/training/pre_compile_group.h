#pragma once

// include group members internal dependencies
#include "pre_compile/correct_user_level.h"
#include "pre_compile/pre_proccess.h"
#include "pre_compile/optimize_user_level_graph.h"
#include "pre_compile/optimize_data_layout.h"
#include "pre_compile/calc_quantization_info.h"
#include "pre_compile/casts_injection.h"


// clang-format off
// ================================================================================================================== //
/*
    pre compilation sub-group.
    no graph mutations in this sub group
*/
// group members
#define PRE_COMPILE_GROUP_MEMBERS                                               GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP,\
                                                                                GROUP_ID_PRE_COMPILE_CORRECT_USER_LEVEL,\
                                                                                GROUP_ID_PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH,\
                                                                                GROUP_ID_PRE_COMPILE_PRE_COMPILE_OPTIMIZE_DATA_LAYOUT,\
                                                                                GROUP_ID_PRE_COMPILE_CALC_QUANT_INFO,\
                                                                                GROUP_ID_PRE_COMPILE_CASTS_INJECTION

// group members dependencies
#define PRE_COMPILE_PRE_PROCCESS_GROUP_DEPENDENCY_SET

#define PRE_COMPILE_CORRECT_USER_LEVEL_GROUP_DEPENDENCY_SET                     GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP

#define PRE_COMPILE_CALC_QUANT_INFO_GROUP_DEPENDENCY_SET                        GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP,\
                                                                                GROUP_ID_PRE_COMPILE_CORRECT_USER_LEVEL

#define PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH_GROUP_DEPENDENCY_SET              GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP,\
                                                                                GROUP_ID_PRE_COMPILE_CORRECT_USER_LEVEL,\
                                                                                GROUP_ID_PRE_COMPILE_CALC_QUANT_INFO

#define PRE_COMPILE_OPTIMIZE_DATA_LAYOUT_GROUP_DEPENDENCY_SET                   GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP,\
                                                                                GROUP_ID_PRE_COMPILE_CORRECT_USER_LEVEL,\
                                                                                GROUP_ID_PRE_COMPILE_CALC_QUANT_INFO,\
                                                                                GROUP_ID_PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH

#define PRE_COMPILE_CASTS_INJECTION_GROUP_DEPENDENCY_SET                        GROUP_ID_PRE_COMPILE_PRE_PROCCESS_GROUP,\
                                                                                GROUP_ID_PRE_COMPILE_CORRECT_USER_LEVEL,\
                                                                                GROUP_ID_PRE_COMPILE_CALC_QUANT_INFO,\
                                                                                GROUP_ID_PRE_COMPILE_OPTIMIZE_USER_LEVEL_GRAPH,\
                                                                                GROUP_ID_PRE_COMPILE_PRE_COMPILE_OPTIMIZE_DATA_LAYOUT,\
                                                                                GROUP_ID_PRE_COMPILE_CALC_QUANT_INFO


// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on
