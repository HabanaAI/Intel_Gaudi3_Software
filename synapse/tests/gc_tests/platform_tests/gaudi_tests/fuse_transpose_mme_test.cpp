#include "syn_gaudi_two_run_compare_test.h"

TEST_F_GC(SynTrainingTwoRunCompareTest, fuse_transpose_identity)
{
    // Graph #0

    /*************
     * g_0_gradient_pre_head_ln_layer_norm_bwd_bf16_576_0 node
     * inputs:
     *     g_0_tensor_699[48, 288, 1, 1] (dtype=bf16)
     *     g_0_tensor_700[48, 288, 1, 1] (dtype=bf16)
     *     g_0_tensor_702[1, 288, 1, 1] (dtype=float32)
     *     g_0_tensor_703[1, 288, 1, 1] (dtype=float32)
     *     g_0_tensor_701[48] (dtype=float32)
     * outputs:
     *     g_0_tensor_704[48, 288, 1, 1] (dtype=bf16)
     *     g_0_tensor_705[48] (dtype=float32)
     *     g_0_tensor_706[48] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_699 tensor
    unsigned g_0_tensor_699_max_sizes[] = {48, 288, 1, 1};
    unsigned g_0_tensor_699_min_sizes[] = {48, 288, 1, 1};
    unsigned g_0_tensor_699             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_699",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_699_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_699_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_700 tensor
    unsigned g_0_tensor_700_max_sizes[] = {48, 288, 1, 1};
    unsigned g_0_tensor_700_min_sizes[] = {48, 288, 1, 1};
    unsigned g_0_tensor_700             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_700",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_700_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_700_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_702 tensor
    unsigned g_0_tensor_702_max_sizes[] = {1, 288, 1, 1};
    unsigned g_0_tensor_702_min_sizes[] = {1, 288, 1, 1};
    unsigned g_0_tensor_702             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_702",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_702_max_sizes,
                                            4,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_702_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_703 tensor
    unsigned g_0_tensor_703_max_sizes[] = {1, 288, 1, 1};
    unsigned g_0_tensor_703_min_sizes[] = {1, 288, 1, 1};
    unsigned g_0_tensor_703             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_703",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_703_max_sizes,
                                            4,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_703_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_701 tensor
    unsigned g_0_tensor_701_max_sizes[] = {48};
    unsigned g_0_tensor_701_min_sizes[] = {48};
    unsigned g_0_tensor_701             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_701",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_701_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_701_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_704 tensor
    unsigned g_0_tensor_704_max_sizes[] = {48, 288, 1, 1};
    unsigned g_0_tensor_704_min_sizes[] = {48, 288, 1, 1};
    unsigned g_0_tensor_704             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_704",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_704_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_704_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_705 tensor
    unsigned g_0_tensor_705_max_sizes[] = {48};
    unsigned g_0_tensor_705_min_sizes[] = {48};
    unsigned g_0_tensor_705             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_705",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_705_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_705_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_706 tensor
    unsigned      g_0_tensor_706_max_sizes[] = {48};
    unsigned      g_0_tensor_706_min_sizes[] = {48};
    unsigned      g_0_tensor_706             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_706",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_706_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_706_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_pre_head_ln_layer_norm_bwd_bf16_576_0_id;
    unsigned char g_0_gradient_pre_head_ln_layer_norm_bwd_bf16_576_0_params[] = {0, 52, 196, 172, 255, 127, 0, 0};
    addNodeToGraph("layer_norm_bwd_bf16",
                   {g_0_tensor_699, g_0_tensor_700, g_0_tensor_702, g_0_tensor_703, g_0_tensor_701},
                   {g_0_tensor_704, g_0_tensor_705, g_0_tensor_706},
                   (void*)g_0_gradient_pre_head_ln_layer_norm_bwd_bf16_576_0_params,
                   8,
                   "g_0_gradient_pre_head_ln_layer_norm_bwd_bf16_576_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_pre_head_ln_layer_norm_bwd_bf16_576_0_id);

    /*************
     * g_0_gradient_pre_head_ln_reshape_577_0 node
     * inputs:
     *     g_0_tensor_704[48, 288, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward[48, 12, 24] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward tensor
    unsigned g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward_max_sizes[] = {48, 12, 24};
    unsigned g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward_min_sizes[] = {48, 12, 24};
    unsigned g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_pre_head_ln_reshape_577_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_704},
                   {g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward},
                   nullptr,
                   0,
                   "g_0_gradient_pre_head_ln_reshape_577_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_pre_head_ln_reshape_577_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_transpose_588_0 node
     * inputs:
     *     g_0_tensor_717[12, 128, 24] (dtype=bf16)
     * outputs:
     *     g_0_tensor_718[128, 12, 24] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_717 tensor
    unsigned g_0_tensor_717_max_sizes[] = {12, 128, 24};
    unsigned g_0_tensor_717_min_sizes[] = {12, 128, 24};
    unsigned g_0_tensor_717             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_717",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_717_max_sizes,
                                            3,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_717_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_718 tensor
    unsigned      g_0_tensor_718_max_sizes[] = {128, 12, 24};
    unsigned      g_0_tensor_718_min_sizes[] = {128, 12, 24};
    unsigned      g_0_tensor_718             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_718",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_718_max_sizes,
                                            3,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_718_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_11_channel_mlp_block_transpose_588_0_id;
    unsigned char g_0_gradient_11_channel_mlp_block_transpose_588_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
                                                                                0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_717},
                   {g_0_tensor_718},
                   (void*)g_0_gradient_11_channel_mlp_block_transpose_588_0_params,
                   24,
                   "g_0_gradient_11_channel_mlp_block_transpose_588_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_transpose_588_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_reshape_589_0 node
     * inputs:
     *     g_0_tensor_718[128, 12, 24] (dtype=bf16)
     * outputs:
     *     g_0_tensor_719[128, 288] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_719 tensor
    unsigned  g_0_tensor_719_max_sizes[] = {128, 288};
    unsigned  g_0_tensor_719_min_sizes[] = {128, 288};
    unsigned  g_0_tensor_719             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_719",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_719_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_719_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_11_channel_mlp_block_reshape_589_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_718},
                   {g_0_tensor_719},
                   nullptr,
                   0,
                   "g_0_gradient_11_channel_mlp_block_reshape_589_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_reshape_589_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_reshape_590_0 node
     * inputs:
     *     g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward[48, 12, 24] (dtype=bf16)
     * outputs:
     *     g_0_tensor_720[48, 288] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_720 tensor
    unsigned  g_0_tensor_720_max_sizes[] = {48, 288};
    unsigned  g_0_tensor_720_min_sizes[] = {48, 288};
    unsigned  g_0_tensor_720             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_720",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_720_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_720_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_11_channel_mlp_block_reshape_590_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_707_2575_gradient_pre_head_ln_aten_native_layer_norm_backward},
                   {g_0_tensor_720},
                   nullptr,
                   0,
                   "g_0_gradient_11_channel_mlp_block_reshape_590_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_reshape_590_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_transpose_591_0 node
     * inputs:
     *     g_0_tensor_719[128, 288] (dtype=bf16)
     * outputs:
     *     g_0_tensor_721[288, 128] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_721 tensor
    unsigned      g_0_tensor_721_max_sizes[] = {288, 128};
    unsigned      g_0_tensor_721_min_sizes[] = {288, 128};
    unsigned      g_0_tensor_721             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_721",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_721_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_721_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_11_channel_mlp_block_transpose_591_0_id;
    unsigned char g_0_gradient_11_channel_mlp_block_transpose_591_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_719},
                   {g_0_tensor_721},
                   (void*)g_0_gradient_11_channel_mlp_block_transpose_591_0_params,
                   24,
                   "g_0_gradient_11_channel_mlp_block_transpose_591_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_transpose_591_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_gemm_592_0 node
     * inputs:
     *     g_0_tensor_721[288, 128] (dtype=bf16)
     *     g_0_tensor_720[48, 288] (dtype=bf16)
     * outputs:
     *     g_0_tensor_722[48, 128] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_722 tensor
    unsigned      g_0_tensor_722_max_sizes[] = {48, 128};
    unsigned      g_0_tensor_722_min_sizes[] = {48, 128};
    unsigned      g_0_tensor_722             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_722",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_722_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_722_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_11_channel_mlp_block_gemm_592_0_id;
    unsigned char g_0_gradient_11_channel_mlp_block_gemm_592_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_tensor_721, g_0_tensor_720},
                   {g_0_tensor_722},
                   (void*)g_0_gradient_11_channel_mlp_block_gemm_592_0_params,
                   2,
                   "g_0_gradient_11_channel_mlp_block_gemm_592_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_gemm_592_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_identity_593_0 node
     * inputs:
     *     g_0_tensor_722[48, 128] (dtype=bf16)
     * outputs:
     *     g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward[48, 128] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward tensor
    unsigned g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward_max_sizes[] = {48, 128};
    unsigned g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward_min_sizes[] = {48, 128};
    unsigned g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_11_channel_mlp_block_identity_593_0_id;
    addNodeToGraph("identity",
                   {g_0_tensor_722},
                   {g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward},
                   nullptr,
                   0,
                   "g_0_gradient_11_channel_mlp_block_identity_593_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_identity_593_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_transpose_594_0 node
     * inputs:
     *     g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward[48, 128] (dtype=bf16)
     * outputs:
     *     g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t[128, 48] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t tensor
    unsigned g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t_max_sizes[] = {128, 48};
    unsigned g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t_min_sizes[] = {128, 48};
    unsigned g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_11_channel_mlp_block_transpose_594_0_id;
    unsigned char g_0_gradient_11_channel_mlp_block_transpose_594_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_723_2590_gradient_11_channel_mlp_block_hpu_matmul_backward},
                   {g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t},
                   (void*)g_0_gradient_11_channel_mlp_block_transpose_594_0_params,
                   24,
                   "g_0_gradient_11_channel_mlp_block_transpose_594_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_transpose_594_0_id);

    /*************
     * g_0_gradient_11_channel_mlp_block_cast_bf16_to_f32_595_0 node
     * inputs:
     *     g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t[128, 48] (dtype=bf16)
     * outputs:
     *     g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast[128, 48] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast tensor
    unsigned g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast_max_sizes[] = {128, 48};
    unsigned g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast_min_sizes[] = {128, 48};
    unsigned g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_11_channel_mlp_block_cast_bf16_to_f32_595_0_id;
    unsigned char g_0_gradient_11_channel_mlp_block_cast_bf16_to_f32_595_0_params[] = {4, 0, 0, 0};
    addNodeToGraph("cast_bf16_to_f32",
                   {g_0_tensor_724_2593_gradient_11_channel_mlp_block_aten_t},
                   {g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast},
                   (void*)g_0_gradient_11_channel_mlp_block_cast_bf16_to_f32_595_0_params,
                   4,
                   "g_0_gradient_11_channel_mlp_block_cast_bf16_to_f32_595_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_11_channel_mlp_block_cast_bf16_to_f32_595_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME", "true");
    compareRunsResults({g_0_tensor_725_2599_gradient_11_channel_mlp_block_hpu_cast});
}

TEST_F_GC(SynTrainingTwoRunCompareTest, fuse_identity_transpose_cast_to_gemm_output)
{
    // The test includes this sequence: Transpose->GEMM->Identity->Transpose->Cast
    // The transposes and cast should be fused to the GEMM.

    unsigned g_0_tensor_178_id_30436_model_mapping_aten__gelu_max_sizes[] = {1280, 8};
    unsigned g_0_tensor_178_id_30436_model_mapping_aten__gelu_min_sizes[] = {1280, 8};
    unsigned g_0_tensor_178_id_30436_model_mapping_aten__gelu =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_178_id_30436_model_mapping_aten__gelu",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_178_id_30436_model_mapping_aten__gelu_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_178_id_30436_model_mapping_aten__gelu_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_tensor_19137_max_sizes[] = {8, 1280};
    unsigned      g_0_tensor_19137_min_sizes[] = {8, 1280};
    unsigned      g_0_tensor_19137             = createTensors(1,
                                              OUTPUT_TENSOR,
                                              false,
                                              "g_0_tensor_19137",
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              g_0_tensor_19137_max_sizes,
                                              2,
                                              syn_type_bf16,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              false,
                                              g_0_tensor_19137_min_sizes,
                                              synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20329_0_id;
    unsigned char g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20329_0_params[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_178_id_30436_model_mapping_aten__gelu},
                   {g_0_tensor_19137},
                   (void*)g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20329_0_params,
                   24,
                   "g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20329_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20329_0_id);

    unsigned g_0_tensor_19132_id_61144_gradient_model_model_up_0_0_pre_ff_0_norm_xg_aten__cat_max_sizes[] = {4096, 8};
    unsigned g_0_tensor_19132_id_61144_gradient_model_model_up_0_0_pre_ff_0_norm_xg_aten__cat_min_sizes[] = {4096, 8};
    unsigned g_0_tensor_19132_id_61144_gradient_model_model_up_0_0_pre_ff_0_norm_xg_aten__cat =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_19132_id_61144_gradient_model_model_up_0_0_pre_ff_0_norm_xg_aten__cat",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_19132_id_61144_gradient_model_model_up_0_0_pre_ff_0_norm_xg_aten__cat_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_19132_id_61144_gradient_model_model_up_0_0_pre_ff_0_norm_xg_aten__cat_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_tensor_19138_max_sizes[] = {4096, 1280};
    unsigned      g_0_tensor_19138_min_sizes[] = {4096, 1280};
    unsigned      g_0_tensor_19138             = createTensors(1,
                                              OUTPUT_TENSOR,
                                              false,
                                              "g_0_tensor_19138",
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              g_0_tensor_19138_max_sizes,
                                              2,
                                              syn_type_bf16,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              false,
                                              g_0_tensor_19138_min_sizes,
                                              synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_gemm_20330_0_id;
    unsigned char g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_gemm_20330_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_tensor_19137, g_0_tensor_19132_id_61144_gradient_model_model_up_0_0_pre_ff_0_norm_xg_aten__cat},
                   {g_0_tensor_19138},
                   (void*)g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_gemm_20330_0_params,
                   2,
                   "g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_gemm_20330_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_gemm_20330_0_id);

    unsigned  g_0_tensor_19139_max_sizes[] = {4096, 1280};
    unsigned  g_0_tensor_19139_min_sizes[] = {4096, 1280};
    unsigned  g_0_tensor_19139             = createTensors(1,
                                              OUTPUT_TENSOR,
                                              false,
                                              "g_0_tensor_19139",
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              g_0_tensor_19139_max_sizes,
                                              2,
                                              syn_type_bf16,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              false,
                                              g_0_tensor_19139_min_sizes,
                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_identity_20331_0_id;
    addNodeToGraph("identity",
                   {g_0_tensor_19138},
                   {g_0_tensor_19139},
                   nullptr,
                   0,
                   "g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_identity_20331_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_identity_20331_0_id);

    unsigned
        g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd_max_sizes
            [] = {1280, 4096};
    unsigned
        g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd_min_sizes
            [] = {1280, 4096};
    unsigned g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20332_0_id;
    unsigned char g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20332_0_params[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph(
        "transpose",
        {g_0_tensor_19139},
        {g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd},
        (void*)g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20332_0_params,
        24,
        "g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20332_0",
        0 /*graphIndex*/,
        &g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_transpose_20332_0_id);

    unsigned g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast_max_sizes[] = {
        1280,
        4096};
    unsigned g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast_min_sizes[] = {
        1280,
        4096};
    unsigned g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast_max_sizes,
        2,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_cast_bf16_to_f32_20336_0_id;
    unsigned char g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_cast_bf16_to_f32_20336_0_params[] = {4,
                                                                                                                  0,
                                                                                                                  0,
                                                                                                                  0};
    addNodeToGraph(
        "cast_bf16_to_f32",
        {g_0_tensor_19140_id_61149_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__linear_non2d_bwd},
        {g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast},
        (void*)g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_cast_bf16_to_f32_20336_0_params,
        4,
        "g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_cast_bf16_to_f32_20336_0",
        0 /*graphIndex*/,
        &g_0_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_cast_bf16_to_f32_20336_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_FUSE_TRANSPOSE_TO_GEMM_OUTPUT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FUSE_TRANSPOSE_TO_GEMM_OUTPUT", "true");
    compareRunsResults({g_0_tensor_19144_id_61165_gradient_model_model_up_0_0_pre_ff_0_norm_xg_proj_cond_hpu__cast});
}