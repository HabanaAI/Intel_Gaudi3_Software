#pragma once

// include group members internal dependencies
#include "lowering/norm_ops.h"
#include "lowering/gconv_handling.h"
#include "lowering/huge_tensors_handling.h"

// clang-format off
// ================================================================================================================== //
/*
    lowering to synapse sub-group.

*/
// group members
#define LOWERING_GROUP_MEMBERS                                                  GROUP_ID_LOWERING_NORM_OPS,            \
                                                                                PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS, \
                                                                                GROUP_ID_GCONV_HANDLING_GROUP, \
                                                                                PASS_ID_REPLACE_OPS_WITH_LOGICAL_OPS, \
                                                                                PASS_ID_ADD_MME_BIAS, \
                                                                                PASS_ID_EXTRACT_MULTI_NODES, \
                                                                                PASS_ID_HANDLE_PARTIAL_BROADCAST_BGEMM, \
                                                                                PASS_ID_FORCE_EXPLICIT_PADDING, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP

// group member dependencies
#define EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES_DEPENDENCY_SET

#define HUGE_TENSORS_HANDLING_GROUP_DEPENDENCY_SET                              PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES

#define REMOVE_ZERO_SIZED_TENSORS_DEPENDENCY_SET                                PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES

#define LOWERING_NORM_OPS_GROUP_DEPENDENCY_SET                                  PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS

#define REPLACE_OPS_WITH_LOGICAL_OPS_DEPENDENCY_SET                             PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS, \
                                                                                PASS_ID_EXTRACT_MULTI_NODES, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP, \
                                                                                GROUP_ID_LOWERING_NORM_OPS

#define ADD_MME_BIAS_DEPENDENCY_SET                                             PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP, \
                                                                                GROUP_ID_LOWERING_NORM_OPS

#define EXTRACT_MULTI_NODES_DEPENDENCY_SET                                      PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP, \
                                                                                GROUP_ID_LOWERING_NORM_OPS

#define HANDLE_PARTIAL_BROADCAST_BGEMM_DEPENDENCY_SET                           PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP, \
                                                                                GROUP_ID_LOWERING_NORM_OPS

#define FORCE_EXPLICIT_PADDING_DEPENDENCY_SET                                   PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP, \
                                                                                GROUP_ID_LOWERING_NORM_OPS

#define GCONV_HANDLING_GROUP_DEPENDENCY_SET                                     PASS_ID_EXTRACT_FUNCTIONAL_COMPLEX_GUID_NODES, \
                                                                                PASS_ID_REMOVE_ZERO_SIZED_TENSORS, \
                                                                                GROUP_ID_HUGE_TENSORS_HANDLING_GROUP, \
                                                                                GROUP_ID_LOWERING_NORM_OPS, \
                                                                                PASS_ID_FORCE_EXPLICIT_PADDING

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on