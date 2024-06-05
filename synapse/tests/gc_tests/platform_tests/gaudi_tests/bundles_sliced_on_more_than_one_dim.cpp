#include "syn_gaudi_two_run_compare_test.h"
#include "test_types.hpp"

TEST_F_GC(SynTrainingTwoRunCompareTest, bundle_with_bgemm_sliced_on_two_batch_dims_and_relu_producer)
{
    unsigned  a_before_relu_max_sizes[] = {2048, 2048, 16, 10};
    unsigned  a_before_relu_min_sizes[] = {2048, 2048, 16, 10};
    unsigned  a_before_relu             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "a_before_relu",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           a_before_relu_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           a_before_relu_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    unsigned  a_after_relu              = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "a_after_relu",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          a_before_relu_max_sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          a_before_relu_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId relu_id;
    addNodeToGraph("relu_fwd_bf16", {a_before_relu}, {a_after_relu}, nullptr, 0, "relu", 0 /*graphIndex*/, &relu_id);

    unsigned b_max_sizes[] = {64, 2048, 16, 10};
    unsigned b_min_sizes[] = {64, 2048, 16, 10};
    unsigned b             = createTensors(1,
                               INPUT_TENSOR,
                               true,
                               "b",
                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                               nullptr,
                               b_max_sizes,
                               4,
                               syn_type_bf16,
                               nullptr,
                               0,
                               0,
                               nullptr,
                               false,
                               b_min_sizes,
                               synTensorType::DATA_TENSOR)[0];

    unsigned      bgemm_output_max_sizes[] = {64, 2048, 16, 10};
    unsigned      bgemm_output_min_sizes[] = {64, 2048, 16, 10};
    unsigned      bgemm_output             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "bgemm_output",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          bgemm_output_max_sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          bgemm_output_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId     batch_gemm_id;
    unsigned char batch_gemm_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {a_after_relu, b},
                   {bgemm_output},
                   (void*)batch_gemm_params,
                   2,
                   "batch_gemm_0",
                   0 /*graphIndex*/,
                   &batch_gemm_id);

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    compareRunsResults({bgemm_output});
}

TEST_F_GC(SynTrainingTwoRunCompareTest, bundle_with_two_bgemm_sliced_on_two_batch_dims_and_fused_tpc_producer)
{
    // Graph #0

    /*************
     * g_0_memcpy_1634_0 node
     * inputs:
     *     g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy[2048, 2048, 16,
     *10] (dtype=bf16) outputs: g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward[2048,
     *2048, 16, 10] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy tensor
    unsigned g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy_max_sizes[] =
        {2048, 2048, 16, 10};
    unsigned g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy_min_sizes[] =
        {2048, 2048, 16, 10};
    unsigned g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward tensor
    unsigned g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_max_sizes[] = {2048,
                                                                                                            2048,
                                                                                                            16,
                                                                                                            10};
    unsigned g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_min_sizes[] = {2048,
                                                                                                            2048,
                                                                                                            16,
                                                                                                            10};
    unsigned g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1634_0_id;
    addNodeToGraph("memcpy",
                   {g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward_before_memcpy},
                   {g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward},
                   nullptr,
                   0,
                   "g_0_memcpy_1634_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1634_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1710_0 node
     * inputs:
     *     g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward[2048, 2048, 16, 10]
     *(dtype=bf16) g_0_tensor_1053__placeholder_1[2048, 2048, 16, 10] (dtype=int8) outputs: g_0_tensor_1055[2048, 2048,
     *16, 10] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1053__placeholder_1 tensor
    unsigned g_0_tensor_1053__placeholder_1_max_sizes[] = {2048, 2048, 16, 10};
    unsigned g_0_tensor_1053__placeholder_1_min_sizes[] = {2048, 2048, 16, 10};
    unsigned g_0_tensor_1053__placeholder_1             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1053__placeholder_1",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1053__placeholder_1_max_sizes,
                                                            4,
                                                            syn_type_int8,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1053__placeholder_1_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1055 tensor
    unsigned  g_0_tensor_1055_max_sizes[] = {2048, 2048, 16, 10};
    unsigned  g_0_tensor_1055_min_sizes[] = {2048, 2048, 16, 10};
    unsigned  g_0_tensor_1055             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_1055",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1055_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1055_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1710_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_1051_id_5746_gradient_transformer_7_attn_c_proj_hpu__matmul_backward,
                    g_0_tensor_1053__placeholder_1},
                   {g_0_tensor_1055},
                   nullptr,
                   0,
                   "g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1710_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1710_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1711_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_1056[2048, 2048, 16, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1056 tensor
    unsigned      g_0_tensor_1056_max_sizes[] = {2048, 2048, 16, 10};
    unsigned      g_0_tensor_1056_min_sizes[] = {2048, 2048, 16, 10};
    unsigned      g_0_tensor_1056             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_1056",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1056_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1056_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1711_0_id;
    unsigned char g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1711_0_params[] = {228, 56, 142, 63};
    addNodeToGraph("constant_bf16",
                   {},
                   {g_0_tensor_1056},
                   (void*)g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1711_0_params,
                   4,
                   "g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1711_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1711_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1712_0 node
     * inputs:
     *     g_0_tensor_1055[2048, 2048, 16, 10] (dtype=bf16)
     *     g_0_tensor_1056[2048, 2048, 16, 10] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1057[2048, 2048, 16, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1057 tensor
    unsigned  g_0_tensor_1057_max_sizes[] = {2048, 2048, 16, 10};
    unsigned  g_0_tensor_1057_min_sizes[] = {2048, 2048, 16, 10};
    unsigned  g_0_tensor_1057             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_1057",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1057_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1057_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1712_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_1055, g_0_tensor_1056},
                   {g_0_tensor_1057},
                   nullptr,
                   0,
                   "g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1712_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_mult_fwd_bf16_1712_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1693_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_1032[1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1032 tensor
    unsigned      g_0_tensor_1032_max_sizes[] = {1};
    unsigned      g_0_tensor_1032_min_sizes[] = {1};
    unsigned      g_0_tensor_1032             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1032",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1032_max_sizes,
                                             1,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1032_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1693_0_id;
    unsigned char g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1693_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("constant_bf16",
                   {},
                   {g_0_tensor_1032},
                   (void*)g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1693_0_params,
                   4,
                   "g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1693_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_constant_bf16_1693_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_softmax_bwd_bf16_1713_0 node
     * inputs:
     *     g_0_tensor_1058__placeholder_1[2048, 2048, 16, 10] (dtype=bf16)
     *     g_0_tensor_1057[2048, 2048, 16, 10] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1060[2048, 2048, 16, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1058__placeholder_1 tensor
    unsigned g_0_tensor_1058__placeholder_1_max_sizes[] = {2048, 2048, 16, 10};
    unsigned g_0_tensor_1058__placeholder_1_min_sizes[] = {2048, 2048, 16, 10};
    unsigned g_0_tensor_1058__placeholder_1             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1058__placeholder_1",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1058__placeholder_1_max_sizes,
                                                            4,
                                                            syn_type_bf16,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1058__placeholder_1_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1060 tensor
    unsigned      g_0_tensor_1060_max_sizes[] = {2048, 2048, 16, 10};
    unsigned      g_0_tensor_1060_min_sizes[] = {2048, 2048, 16, 10};
    unsigned      g_0_tensor_1060             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_1060",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1060_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1060_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_transformer_7_attn_attn_dropout_softmax_bwd_bf16_1713_0_id;
    unsigned char g_0_gradient_transformer_7_attn_attn_dropout_softmax_bwd_bf16_1713_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("softmax_bwd_bf16",
                   {g_0_tensor_1058__placeholder_1, g_0_tensor_1057},
                   {g_0_tensor_1060},
                   (void*)g_0_gradient_transformer_7_attn_attn_dropout_softmax_bwd_bf16_1713_0_params,
                   4,
                   "g_0_gradient_transformer_7_attn_attn_dropout_softmax_bwd_bf16_1713_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_softmax_bwd_bf16_1713_0_id);

    /*************
     * g_0_memcpy_1632_0 node
     * inputs:
     *     g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy[64, 16, 2048, 10]
     *(dtype=bf16) outputs: g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view[64, 16, 2048,
     *10] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy tensor
    unsigned g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy_max_sizes[] =
        {64, 16, 2048, 10};
    unsigned g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy_min_sizes[] =
        {64, 16, 2048, 10};
    unsigned g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy = createTensors(
        1,
        INPUT_TENSOR,
        true,
        "g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view tensor
    unsigned g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_max_sizes[] = {64,
                                                                                                        16,
                                                                                                        2048,
                                                                                                        10};
    unsigned g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_min_sizes[] = {64,
                                                                                                        16,
                                                                                                        2048,
                                                                                                        10};
    unsigned g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1632_0_id;
    addNodeToGraph("memcpy",
                   {g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view_before_memcpy},
                   {g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view},
                   nullptr,
                   0,
                   "g_0_memcpy_1632_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1632_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_transpose_1696_0 node
     * inputs:
     *     g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view[64, 16, 2048, 10] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute[64, 2048, 16, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute tensor
    unsigned g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute_max_sizes[] = {64,
                                                                                                           2048,
                                                                                                           16,
                                                                                                           10};
    unsigned g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute_min_sizes[] = {64,
                                                                                                           2048,
                                                                                                           16,
                                                                                                           10};
    unsigned g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_transformer_7_attn_attn_dropout_transpose_1696_0_id;
    unsigned char g_0_gradient_transformer_7_attn_attn_dropout_transpose_1696_0_params[] = {
        0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_1035_id_4404_gradient_transformer_7_attn_attn_dropout_aten__view},
                   {g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute},
                   (void*)g_0_gradient_transformer_7_attn_attn_dropout_transpose_1696_0_params,
                   24,
                   "g_0_gradient_transformer_7_attn_attn_dropout_transpose_1696_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_transpose_1696_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_where_fwd_bf16_1714_0 node
     * inputs:
     *     g_0_tensor_1061__placeholder_0[2048, 2048, 1, 1] (dtype=int8)
     *     g_0_tensor_1060[2048, 2048, 16, 10] (dtype=bf16)
     *     g_0_tensor_1032[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where[2048, 2048, 16, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1061__placeholder_0 tensor
    unsigned g_0_tensor_1061__placeholder_0_max_sizes[] = {2048, 2048, 1, 1};
    unsigned g_0_tensor_1061__placeholder_0_min_sizes[] = {2048, 2048, 1, 1};
    unsigned g_0_tensor_1061__placeholder_0             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1061__placeholder_0",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1061__placeholder_0_max_sizes,
                                                            4,
                                                            syn_type_int8,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1061__placeholder_0_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where tensor
    unsigned g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where_max_sizes[] = {2048,
                                                                                                         2048,
                                                                                                         16,
                                                                                                         10};
    unsigned g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where_min_sizes[] = {2048,
                                                                                                         2048,
                                                                                                         16,
                                                                                                         10};
    unsigned g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_transformer_7_attn_attn_dropout_where_fwd_bf16_1714_0_id;
    addNodeToGraph("where_fwd_bf16",
                   {g_0_tensor_1061__placeholder_0, g_0_tensor_1060, g_0_tensor_1032},
                   {g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where},
                   nullptr,
                   0,
                   "g_0_gradient_transformer_7_attn_attn_dropout_where_fwd_bf16_1714_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_where_fwd_bf16_1714_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_transpose_1697_0 node
     * inputs:
     *     g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute[64, 2048, 16, 10] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose[2048, 64, 16, 10]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose tensor
    unsigned g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose_max_sizes[] = {2048,
                                                                                                             64,
                                                                                                             16,
                                                                                                             10};
    unsigned g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose_min_sizes[] = {2048,
                                                                                                             64,
                                                                                                             16,
                                                                                                             10};
    unsigned g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_transformer_7_attn_attn_dropout_transpose_1697_0_id;
    unsigned char g_0_gradient_transformer_7_attn_attn_dropout_transpose_1697_0_params[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_1036_id_4405_gradient_transformer_7_attn_attn_dropout_aten__permute},
                   {g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose},
                   (void*)g_0_gradient_transformer_7_attn_attn_dropout_transpose_1697_0_params,
                   24,
                   "g_0_gradient_transformer_7_attn_attn_dropout_transpose_1697_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_transpose_1697_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_div_bf16_1715_0 node
     * inputs:
     *     g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where[2048, 2048, 16, 10] (dtype=bf16)
     *     g_0_tensor_1063__placeholder_1[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div[2048, 2048, 16, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1063__placeholder_1 tensor
    unsigned g_0_tensor_1063__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_1063__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_1063__placeholder_1             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1063__placeholder_1",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1063__placeholder_1_max_sizes,
                                                            1,
                                                            syn_type_bf16,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1063__placeholder_1_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div tensor
    unsigned g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div_max_sizes[] = {2048,
                                                                                                       2048,
                                                                                                       16,
                                                                                                       10};
    unsigned g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div_min_sizes[] = {2048,
                                                                                                       2048,
                                                                                                       16,
                                                                                                       10};
    unsigned g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_transformer_7_attn_attn_dropout_div_bf16_1715_0_id;
    addNodeToGraph(
        "div_bf16",
        {g_0_tensor_1062_id_5762_gradient_transformer_7_attn_attn_dropout_aten__where, g_0_tensor_1063__placeholder_1},
        {g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div},
        nullptr,
        0,
        "g_0_gradient_transformer_7_attn_attn_dropout_div_bf16_1715_0",
        0 /*graphIndex*/,
        &g_0_gradient_transformer_7_attn_attn_dropout_div_bf16_1715_0_id);

    /*************
     * g_0_memcpy_1633_0 node
     * inputs:
     *     g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy[64, 2048, 16,
     *10] (dtype=bf16) outputs: g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute[64, 2048,
     *16, 10] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy tensor
    unsigned g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy_max_sizes[] =
        {64, 2048, 16, 10};
    unsigned g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy_min_sizes[] =
        {64, 2048, 16, 10};
    unsigned g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute tensor
    unsigned g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_max_sizes[] = {64,
                                                                                                           2048,
                                                                                                           16,
                                                                                                           10};
    unsigned g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_min_sizes[] = {64,
                                                                                                           2048,
                                                                                                           16,
                                                                                                           10};
    unsigned g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1633_0_id;
    addNodeToGraph("memcpy",
                   {g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute_before_memcpy},
                   {g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute},
                   nullptr,
                   0,
                   "g_0_memcpy_1633_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1633_0_id);

    /*************
     * g_0_gradient_transformer_7_attn_attn_dropout_matmul_bwd_bf16_1716_0 node
     * inputs:
     *     g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div[2048, 2048, 16, 10] (dtype=bf16)
     *     g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute[64, 2048, 16, 10] (dtype=bf16)
     *     g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose[2048, 64, 16, 10]
     *(dtype=bf16) outputs: g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward[64,
     *2048, 16, 10] (dtype=bf16)
     *     g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward[2048, 64, 16, 10]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward tensor
    unsigned g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_max_sizes[] = {64,
                                                                                                                  2048,
                                                                                                                  16,
                                                                                                                  10};
    unsigned g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_min_sizes[] = {64,
                                                                                                                  2048,
                                                                                                                  16,
                                                                                                                  10};
    unsigned g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward tensor
    unsigned g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_max_sizes[] = {2048,
                                                                                                                  64,
                                                                                                                  16,
                                                                                                                  10};
    unsigned g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_min_sizes[] = {2048,
                                                                                                                  64,
                                                                                                                  16,
                                                                                                                  10};
    unsigned g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_transformer_7_attn_attn_dropout_matmul_bwd_bf16_1716_0_id;
    unsigned char g_0_gradient_transformer_7_attn_attn_dropout_matmul_bwd_bf16_1716_0_params[] = {0};
    addNodeToGraph("matmul_bwd_bf16",
                   {g_0_tensor_1064_id_5765_gradient_transformer_7_attn_attn_dropout_aten__div,
                    g_0_tensor_1040_id_4403_gradient_transformer_7_attn_attn_dropout_aten__permute,
                    g_0_tensor_1037_id_4408_gradient_transformer_7_attn_attn_dropout_aten__transpose},
                   {g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward,
                    g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward},
                   (void*)g_0_gradient_transformer_7_attn_attn_dropout_matmul_bwd_bf16_1716_0_params,
                   1,
                   "g_0_gradient_transformer_7_attn_attn_dropout_matmul_bwd_bf16_1716_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_transformer_7_attn_attn_dropout_matmul_bwd_bf16_1716_0_id);

    /*************
     * g_0_memcpy_1631_0 node
     * inputs:
     *     g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward[64, 2048, 16, 10]
     *(dtype=bf16) outputs:
     *     g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy[64, 2048, 16,
     *10] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy tensor
    unsigned g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy_max_sizes[] =
        {64, 2048, 16, 10};
    unsigned g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy_min_sizes[] =
        {64, 2048, 16, 10};
    unsigned g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1631_0_id;
    addNodeToGraph("memcpy",
                   {g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward},
                   {g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1631_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1631_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_REMOVE_REDUNDANT_MEMCPY", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    compareRunsResults({g_0_tensor_1065_id_5767_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward_memcpy,
                        g_0_tensor_1066_id_5769_gradient_transformer_7_attn_attn_dropout_hpu__matmul_backward});
}
