#pragma once

// clang-format off
// ========================================= CALC_QUANT_INFO_GROUP ========================================= //
// group members
#define PRE_COMPILE_CALC_QUANT_INFO_GROUP_MEMBERS           PASS_ID_CALC_QUANTIZATION_INFO,\
                                                            PASS_ID_PROPAGATE_CAST_NODES,\
                                                            PASS_ID_ADJUST_RESTRICTIONS,\
                                                            PASS_ID_UPDATE_PAD_QUANTIZER,\
                                                            PASS_ID_ADJUST_SCALES,\
                                                            PASS_ID_ENFORCE_NODE_PRECISION,\
                                                            PASS_ID_VALIDATE_QUANTIZATION,\
                                                            PASS_ID_INJECT_SCALE_FOR_MME_NODES,\
                                                            PASS_ID_FUSE_PAD_INTO_CONV_POOL, \
                                                            PASS_ID_FUSE_CONVERT_MME, \
                                                            PASS_ID_REMOVE_CONTIGUOUS_CONVERTS
// group member dependencies
#define CALC_QUANTIZATION_INFO_DEPENDENCY_SET

#define PROPAGATE_CAST_NODES_DEPENDENCY_SET                 PASS_ID_CALC_QUANTIZATION_INFO

#define ADJUST_RESTRICTIONS_DEPENDENCY_SET                  PASS_ID_CALC_QUANTIZATION_INFO,\
                                                            PASS_ID_PROPAGATE_CAST_NODES

#define UPDATE_PAD_QUANTIZER_DEPENDENCY_SET                 PASS_ID_CALC_QUANTIZATION_INFO,\
                                                            PASS_ID_PROPAGATE_CAST_NODES,\
                                                            PASS_ID_ADJUST_RESTRICTIONS

#define ADJUST_SCALES_DEPENDENCY_SET                        PASS_ID_CALC_QUANTIZATION_INFO,\
                                                            PASS_ID_PROPAGATE_CAST_NODES,\
                                                            PASS_ID_ADJUST_RESTRICTIONS,\
                                                            PASS_ID_UPDATE_PAD_QUANTIZER

#define ENFORCE_NODE_PRECISION_DEPENDENCY_SET               PASS_ID_CALC_QUANTIZATION_INFO,\
                                                            PASS_ID_PROPAGATE_CAST_NODES,\
                                                            PASS_ID_ADJUST_RESTRICTIONS,\
                                                            PASS_ID_UPDATE_PAD_QUANTIZER,\
                                                            PASS_ID_ADJUST_SCALES

#define VALIDATE_QUANTIZATION_DEPENDENCY_SET                PASS_ID_CALC_QUANTIZATION_INFO,\
                                                            PASS_ID_PROPAGATE_CAST_NODES,\
                                                            PASS_ID_ADJUST_RESTRICTIONS,\
                                                            PASS_ID_UPDATE_PAD_QUANTIZER,\
                                                            PASS_ID_ADJUST_SCALES,\
                                                            PASS_ID_ENFORCE_NODE_PRECISION

#define INJECT_SCALE_FOR_MME_NODES_DEPENDENCY_SET           PASS_ID_CALC_QUANTIZATION_INFO,\
                                                            PASS_ID_PROPAGATE_CAST_NODES,\
                                                            PASS_ID_ADJUST_RESTRICTIONS,\
                                                            PASS_ID_UPDATE_PAD_QUANTIZER,\
                                                            PASS_ID_ADJUST_SCALES,\
                                                            PASS_ID_ENFORCE_NODE_PRECISION,\
                                                            PASS_ID_VALIDATE_QUANTIZATION

#define FUSE_PAD_INTO_CONV_POOL_DEPENDENCY_SET              PASS_ID_VALIDATE_QUANTIZATION

#define FUSE_CONVERT_MME_DEPENDENCY_SET                     PASS_ID_PROPAGATE_CAST_NODES

#define REMOVE_CONTIGUOUS_CONVERTS_DEPENDENCY_SET           PASS_ID_PROPAGATE_CAST_NODES


// clang-format on