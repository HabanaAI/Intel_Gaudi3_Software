#pragma once

// clang-format off
// ================================================================================================================== //
// ################################################################################################################## //
// ===================================== LOWERING_NORM_OPS_GROUP =================================================== //
// group members
#define LOWERING_NORM_OPS_GROUP_MEMBERS                          PASS_ID_UPDATE_NODES_WITH_ALIAS_TENSORS,               \
                                                                 PASS_ID_SPLIT_TF_BATCH_NORM,                           \
                                                                 PASS_ID_SPLIT_BATCH_NORM,                              \
                                                                 PASS_ID_SPLIT_FROBENIUS_LAYER_NORM,                    \
                                                                 PASS_ID_SPLIT_MOMENTS,                                 \
                                                                 PASS_ID_SPLIT_LAYER_NORM_BWD
// group member dependencies
#define UPDATE_NODES_WITH_ALIAS_TENSORS_DEPENDENCY_SET

#define SPLIT_TF_BATCH_NORM_DEPENDENCY_SET                       PASS_ID_UPDATE_NODES_WITH_ALIAS_TENSORS

#define SPLIT_BATCH_NORM_DEPENDENCY_SET                          PASS_ID_UPDATE_NODES_WITH_ALIAS_TENSORS

#define SPLIT_FROBENIUS_LAYER_NORM_DEPENDENCY_SET

#define SPLIT_MOMENTS_DEPENDENCY_SET                             PASS_ID_SPLIT_TF_BATCH_NORM

#define SPLIT_LAYER_NORM_BWD_DEPENDENCY_SET

// clang-format on
