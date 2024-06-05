#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "../gc_tests/unit_tests/batch_norm_eviction_fuser_test_common.h"
#include "syn_gaudi_two_run_compare_test.h"

TEST_F_GC(SynGaudiTwoRunCompareTest, BN_eviction_Fusion)
{
    // Graph #0

    /*************
     * g_0_layer1_1_bn2_0 node
     * inputs:
     *     g_0_layer1_1_conv2_output[64, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_bn2_bias[64] (dtype=float32)
     *     g_0_layer1_1_bn2_weight[64] (dtype=float32)
     *     g_0_layer1_1_bn2_running_mean[64] (dtype=float32)
     *     g_0_layer1_1_bn2_running_var[64] (dtype=float32)
     * outputs:
     *     g_0_layer1_1_bn2_output[64, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_bn2_saved_mean[64, 1, 1, 1] (dtype=float32)
     *     g_0_layer1_1_bn2_saved_var[64, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer1_1_conv2_output tensor
    unsigned g_0_layer1_1_conv2_output_max_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_1_conv2_output_min_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_1_conv2_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer1_1_conv2_output",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_1_conv2_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_1_conv2_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn2_bias tensor
    unsigned g_0_layer1_1_bn2_bias_max_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_bias_min_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_bias             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_layer1_1_bn2_bias",
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   g_0_layer1_1_bn2_bias_max_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_layer1_1_bn2_bias_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn2_weight tensor
    unsigned g_0_layer1_1_bn2_weight_max_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_weight_min_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer1_1_bn2_weight",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_layer1_1_bn2_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_1_bn2_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn2_running_mean tensor
    unsigned g_0_layer1_1_bn2_running_mean_max_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_running_mean_min_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_running_mean             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_1_bn2_running_mean",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_layer1_1_bn2_running_mean_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_1_bn2_running_mean_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn2_running_var tensor
    unsigned g_0_layer1_1_bn2_running_var_max_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_running_var_min_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_running_var             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_layer1_1_bn2_running_var",
                                                          MEM_INIT_ALL_ZERO,
                                                          nullptr,
                                                          g_0_layer1_1_bn2_running_var_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer1_1_bn2_running_var_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn2_output tensor
    unsigned g_0_layer1_1_bn2_output_max_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_1_bn2_output_min_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_1_bn2_output             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     false,
                                                     "g_0_layer1_1_bn2_output",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_layer1_1_bn2_output_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_1_bn2_output_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn2_saved_mean tensor
    unsigned g_0_layer1_1_bn2_saved_mean_max_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_saved_mean_min_sizes[] = {64};
    unsigned g_0_layer1_1_bn2_saved_mean             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer1_1_bn2_saved_mean",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_layer1_1_bn2_saved_mean_max_sizes,
                                                         1,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer1_1_bn2_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn2_saved_var tensor
    unsigned      g_0_layer1_1_bn2_saved_var_max_sizes[] = {64};
    unsigned      g_0_layer1_1_bn2_saved_var_min_sizes[] = {64};
    unsigned      g_0_layer1_1_bn2_saved_var             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer1_1_bn2_saved_var",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_layer1_1_bn2_saved_var_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer1_1_bn2_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_1_bn2_0_id;
    unsigned char g_0_layer1_1_bn2_0_params[] = {172, 197, 39, 55, 205, 204, 204, 61, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_layer1_1_conv2_output,
                    g_0_layer1_1_bn2_bias,
                    g_0_layer1_1_bn2_weight,
                    g_0_layer1_1_bn2_running_mean,
                    g_0_layer1_1_bn2_running_var},
                   {g_0_layer1_1_bn2_output, g_0_layer1_1_bn2_saved_mean, g_0_layer1_1_bn2_saved_var},
                   (void*)g_0_layer1_1_bn2_0_params,
                   12,
                   "g_0_layer1_1_bn2_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_1_bn2_0_id);

    /*************
     * g_0_layer1_1_relu2_0 node
     * inputs:
     *     g_0_layer1_1_bn2_output[64, 56, 56, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer1_1_relu2_output[64, 56, 56, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer1_1_relu2_output tensor
    unsigned  g_0_layer1_1_relu2_output_max_sizes[] = {64, 56, 56, 64};
    unsigned  g_0_layer1_1_relu2_output_min_sizes[] = {64, 56, 56, 64};
    unsigned  g_0_layer1_1_relu2_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_layer1_1_relu2_output",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_1_relu2_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_1_relu2_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer1_1_relu2_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_layer1_1_bn2_output},
                   {g_0_layer1_1_relu2_output},
                   nullptr,
                   0,
                   "g_0_layer1_1_relu2_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_1_relu2_0_id);

    /*************
     * g_0_layer1_1_conv3_0 node
     * inputs:
     *     g_0_layer1_1_relu2_output[64, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_conv3_weight[256, 64, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_layer1_1_conv3_output[256, 56, 56, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer1_1_conv3_weight tensor
    unsigned g_0_layer1_1_conv3_weight_max_sizes[] = {256, 64, 1, 1};
    unsigned g_0_layer1_1_conv3_weight_min_sizes[] = {256, 64, 1, 1};
    unsigned g_0_layer1_1_conv3_weight             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer1_1_conv3_weight",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_1_conv3_weight_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_1_conv3_weight_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_conv3_output tensor
    unsigned      g_0_layer1_1_conv3_output_max_sizes[] = {256, 56, 56, 64};
    unsigned      g_0_layer1_1_conv3_output_min_sizes[] = {256, 56, 56, 64};
    unsigned      g_0_layer1_1_conv3_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_layer1_1_conv3_output",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_1_conv3_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_1_conv3_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_1_conv3_0_id;
    unsigned char g_0_layer1_1_conv3_0_params[] = {1, 0, 0, 0, 1, 0, 0,  0,   1, 0, 0, 0, 1, 0, 0,   0,   0, 0,
                                                   0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0, 0, 0, 0, 1,   0,   0, 0,
                                                   1, 0, 0, 0, 0, 0, 96, 236, 1, 0, 0, 0, 0, 0, 0,   0,   0, 0,
                                                   0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 1, 0, 0, 0, 253, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_layer1_1_relu2_output, g_0_layer1_1_conv3_weight},
                   {g_0_layer1_1_conv3_output},
                   (void*)g_0_layer1_1_conv3_0_params,
                   72,
                   "g_0_layer1_1_conv3_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_1_conv3_0_id);

    /*************
     * g_0_layer1_1_bn3_0 node
     * inputs:
     *     g_0_layer1_1_conv3_output[256, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_bn3_bias[256] (dtype=float32)
     *     g_0_layer1_1_bn3_weight[256] (dtype=float32)
     *     g_0_layer1_1_bn3_running_mean[256] (dtype=float32)
     *     g_0_layer1_1_bn3_running_var[256] (dtype=float32)
     * outputs:
     *     g_0_layer1_1_bn3_output[256, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_bn3_saved_mean[256, 1, 1, 1] (dtype=float32)
     *     g_0_layer1_1_bn3_saved_var[256, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer1_1_bn3_bias tensor
    unsigned g_0_layer1_1_bn3_bias_max_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_bias_min_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_bias             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_layer1_1_bn3_bias",
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   g_0_layer1_1_bn3_bias_max_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_layer1_1_bn3_bias_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_weight tensor
    unsigned g_0_layer1_1_bn3_weight_max_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_weight_min_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer1_1_bn3_weight",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_layer1_1_bn3_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_1_bn3_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_running_mean tensor
    unsigned g_0_layer1_1_bn3_running_mean_max_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_running_mean_min_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_running_mean             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_1_bn3_running_mean",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_layer1_1_bn3_running_mean_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_1_bn3_running_mean_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_running_var tensor
    unsigned g_0_layer1_1_bn3_running_var_max_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_running_var_min_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_running_var             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_layer1_1_bn3_running_var",
                                                          MEM_INIT_ALL_ZERO,
                                                          nullptr,
                                                          g_0_layer1_1_bn3_running_var_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer1_1_bn3_running_var_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_output tensor
    unsigned g_0_layer1_1_bn3_output_max_sizes[] = {256, 56, 56, 64};
    unsigned g_0_layer1_1_bn3_output_min_sizes[] = {256, 56, 56, 64};
    unsigned g_0_layer1_1_bn3_output             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     true,
                                                     "g_0_layer1_1_bn3_output",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_layer1_1_bn3_output_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_1_bn3_output_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_saved_mean tensor
    unsigned g_0_layer1_1_bn3_saved_mean_max_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_saved_mean_min_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_saved_mean             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_layer1_1_bn3_saved_mean",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_layer1_1_bn3_saved_mean_max_sizes,
                                                         1,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer1_1_bn3_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_saved_var tensor
    unsigned      g_0_layer1_1_bn3_saved_var_max_sizes[] = {256};
    unsigned      g_0_layer1_1_bn3_saved_var_min_sizes[] = {256};
    unsigned      g_0_layer1_1_bn3_saved_var             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        false,
                                                        "g_0_layer1_1_bn3_saved_var",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_layer1_1_bn3_saved_var_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer1_1_bn3_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_1_bn3_0_id;
    unsigned char g_0_layer1_1_bn3_0_params[] = {172, 197, 39, 55, 205, 204, 204, 61, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_layer1_1_conv3_output,
                    g_0_layer1_1_bn3_bias,
                    g_0_layer1_1_bn3_weight,
                    g_0_layer1_1_bn3_running_mean,
                    g_0_layer1_1_bn3_running_var},
                   {g_0_layer1_1_bn3_output, g_0_layer1_1_bn3_saved_mean, g_0_layer1_1_bn3_saved_var},
                   (void*)g_0_layer1_1_bn3_0_params,
                   12,
                   "g_0_layer1_1_bn3_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_1_bn3_0_id);

    /*************
     * g_0_layer1_1_relu2_bwd_0 node
     * inputs:
     *     g_0_layer1_1_conv3_grad_input[64, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_relu2_output[64, 56, 56, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer1_1_relu2_grad_input[64, 56, 56, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer1_1_conv3_grad_input tensor
    unsigned g_0_layer1_1_conv3_grad_input_max_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_1_conv3_grad_input_min_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_1_conv3_grad_input             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_1_conv3_grad_input",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_layer1_1_conv3_grad_input_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_1_conv3_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_relu2_grad_input tensor
    unsigned  g_0_layer1_1_relu2_grad_input_max_sizes[] = {64, 56, 56, 64};
    unsigned  g_0_layer1_1_relu2_grad_input_min_sizes[] = {64, 56, 56, 64};
    unsigned  g_0_layer1_1_relu2_grad_input             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_1_relu2_grad_input",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_layer1_1_relu2_grad_input_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_1_relu2_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer1_1_relu2_bwd_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_layer1_1_conv3_grad_input, g_0_layer1_1_relu2_output},
                   {g_0_layer1_1_relu2_grad_input},
                   nullptr,
                   0,
                   "g_0_layer1_1_relu2_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_1_relu2_bwd_0_id);

    /*************
     * g_0_layer1_1_bn3_bwd_0 node
     * inputs:
     *     g_0_layer1_1_conv3_output[256, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_add_residual_grad_input0[256, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_bn3_saved_mean[256, 1, 1, 1] (dtype=float32)
     *     g_0_layer1_1_bn3_saved_var[256, 1, 1, 1] (dtype=float32)
     *     g_0_layer1_1_bn3_weight[256] (dtype=float32)
     * outputs:
     *     g_0_layer1_1_bn3_grad_input[256, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_1_bn3_bias_grad[256] (dtype=float32)
     *     g_0_layer1_1_bn3_weight_grad[256] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer1_1_add_residual_grad_input0 tensor
    unsigned g_0_layer1_1_add_residual_grad_input0_max_sizes[] = {256, 56, 56, 64};
    unsigned g_0_layer1_1_add_residual_grad_input0_min_sizes[] = {256, 56, 56, 64};
    unsigned g_0_layer1_1_add_residual_grad_input0             = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_layer1_1_add_residual_grad_input0",
                                                                   MEM_INIT_ALL_ZERO,
                                                                   nullptr,
                                                                   g_0_layer1_1_add_residual_grad_input0_max_sizes,
                                                                   4,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_layer1_1_add_residual_grad_input0_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_grad_input tensor
    unsigned g_0_layer1_1_bn3_grad_input_max_sizes[] = {256, 56, 56, 64};
    unsigned g_0_layer1_1_bn3_grad_input_min_sizes[] = {256, 56, 56, 64};
    unsigned g_0_layer1_1_bn3_grad_input             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer1_1_bn3_grad_input",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_layer1_1_bn3_grad_input_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer1_1_bn3_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_bias_grad tensor
    unsigned g_0_layer1_1_bn3_bias_grad_max_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_bias_grad_min_sizes[] = {256};
    unsigned g_0_layer1_1_bn3_bias_grad             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer1_1_bn3_bias_grad",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_layer1_1_bn3_bias_grad_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer1_1_bn3_bias_grad_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_1_bn3_weight_grad tensor
    unsigned  g_0_layer1_1_bn3_weight_grad_max_sizes[] = {256};
    unsigned  g_0_layer1_1_bn3_weight_grad_min_sizes[] = {256};
    unsigned  g_0_layer1_1_bn3_weight_grad             = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          true,
                                                          "g_0_layer1_1_bn3_weight_grad",
                                                          MEM_INIT_ALL_ZERO,
                                                          nullptr,
                                                          g_0_layer1_1_bn3_weight_grad_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer1_1_bn3_weight_grad_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer1_1_bn3_bwd_0_id;
    addNodeToGraph("batch_norm_bwd_bf16",
                   {g_0_layer1_1_conv3_output,
                    g_0_layer1_1_add_residual_grad_input0,
                    g_0_layer1_1_bn3_saved_mean,
                    g_0_layer1_1_bn3_saved_var,
                    g_0_layer1_1_bn3_weight},
                   {g_0_layer1_1_bn3_grad_input, g_0_layer1_1_bn3_bias_grad, g_0_layer1_1_bn3_weight_grad},
                   nullptr,
                   0,
                   "g_0_layer1_1_bn3_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_1_bn3_bwd_0_id);
    addConfigurationToRun(FIRST_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_BATCH_NORM_MEMCPY_FUSION", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_BATCH_NORM_MEMCPY_FUSION", "false");
    compareRunsResults({g_0_layer1_1_conv2_output,
                        g_0_layer1_1_bn2_bias,
                        g_0_layer1_1_bn2_weight,
                        g_0_layer1_1_bn2_running_mean,
                        g_0_layer1_1_bn2_running_var,
                        g_0_layer1_1_bn2_saved_mean,
                        g_0_layer1_1_bn2_saved_var});
}