#pragma once

// clang-format off
// ================================================================================================================== //
// ################################################################################################################## //
// ===================================== GCONV_HANDLING_GROUP =================================================== //
// group members
#define GCONV_HANDLING_GROUP_MEMBERS                             PASS_ID_REPLACE_GROUP_CONV_FILTER2D,               \
                                                                 PASS_ID_HANDLE_GROUPED_CONVOLUTIONS,               \

// group member dependencies
#define REPLACE_GROUP_CONV_FILTER2D_DEPENDENCY_SET

#define HANDLE_GROUPED_CONVOLUTIONS_DEPENDENCY_SET               PASS_ID_REPLACE_GROUP_CONV_FILTER2D

// clang-format on
