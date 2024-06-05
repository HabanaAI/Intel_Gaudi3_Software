#include "gc_gaudi_test_infra.h"

TEST_F_GC(SynTrainingTestInfra, batch_one_pytorch)
{
    // Graph #0

    /*************
     * g_0_Convolution823_0 node
     * inputs:
     *     g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0[256, 16, 16, 16, 1] (dtype=float32)
     *     g_0_tensor_65[320, 256, 3, 3, 3] (dtype=float32)
     * outputs:
     *     g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0[320, 8, 8, 8, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0 tensor
    unsigned g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0_max_sizes[] = {256,16,16,16,1};
    unsigned g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0_min_sizes[] = {256,16,16,16,1};
    unsigned g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0 = createTensors(1,
                                                                                           INPUT_TENSOR,
                                                                                           true,
                                                                                           "g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0",
                                                                                           MEM_INIT_ALL_ZERO,
                                                                                           nullptr,
                                                                                           g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0_max_sizes,
                                                                                           5,
                                                                                           syn_type_single,
                                                                                           nullptr,
                                                                                           0,
                                                                                           0,
                                                                                           nullptr,
                                                                                           false,
                                                                                           g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0_min_sizes,
                                                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_65 tensor
    unsigned g_0_tensor_65_max_sizes[] = {320,256,3,3,3};
    unsigned g_0_tensor_65_min_sizes[] = {320,256,3,3,3};
    unsigned g_0_tensor_65 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "g_0_tensor_65",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       g_0_tensor_65_max_sizes,
                                       5,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_65_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0 tensor
    unsigned g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0_max_sizes[] = {320,8,8,8,1};
    unsigned g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0_min_sizes[] = {320,8,8,8,1};
    unsigned g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0 = createTensors(1,
                                                                                                       OUTPUT_TENSOR,
                                                                                                       true,
                                                                                                       "g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0",
                                                                                                       MEM_INIT_ALL_ZERO,
                                                                                                       nullptr,
                                                                                                       g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0_max_sizes,
                                                                                                       5,
                                                                                                       syn_type_single,
                                                                                                       nullptr,
                                                                                                       0,
                                                                                                       0,
                                                                                                       nullptr,
                                                                                                       false,
                                                                                                       g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0_min_sizes,
                                                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_Convolution823_0_id;
    unsigned char g_0_Convolution823_0_params[] = {3,0,0,0,3,0,0,0,3,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,206,85,0,0};
    addNodeToGraph("spatial_convolution3d", {g_0_tensor_64_t1229_model_model_2_conv2_lrelu_aten__leaky_relu__0, g_0_tensor_65}, {g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0}, (void*)g_0_Convolution823_0_params, 96, "g_0_Convolution823_0", 0 /*graphIndex*/, &g_0_Convolution823_0_id);

    /*************
     * g_0__complex_reshape_in0_instance_norm_node_42_0 node
     * inputs:
     *     g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0[320, 8, 8, 8, 1] (dtype=float32)
     * outputs:
     *     g_0_complex_reshaped_in_instance_norm_tensor_105[320, 8, 64, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_complex_reshaped_in_instance_norm_tensor_105 tensor
    unsigned g_0_complex_reshaped_in_instance_norm_tensor_105_max_sizes[] = {320,8,64,1};
    unsigned g_0_complex_reshaped_in_instance_norm_tensor_105_min_sizes[] = {320,8,64,1};
    unsigned g_0_complex_reshaped_in_instance_norm_tensor_105 = createTensors(1,
                                                                          OUTPUT_TENSOR,
                                                                          false,
                                                                          "g_0_complex_reshaped_in_instance_norm_tensor_105",
                                                                          MEM_INIT_ALL_ZERO,
                                                                          nullptr,
                                                                          g_0_complex_reshaped_in_instance_norm_tensor_105_max_sizes,
                                                                          4,
                                                                          syn_type_single,
                                                                          nullptr,
                                                                          0,
                                                                          0,
                                                                          nullptr,
                                                                          false,
                                                                          g_0_complex_reshaped_in_instance_norm_tensor_105_min_sizes,
                                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__complex_reshape_in0_instance_norm_node_42_0_id;
    addNodeToGraph("reshape", {g_0_tensor_66_t1237_model_model_3_conv1_conv_aten__convolution_overrideable_0}, {g_0_complex_reshaped_in_instance_norm_tensor_105}, nullptr, 0, "g_0__complex_reshape_in0_instance_norm_node_42_0", 0 /*graphIndex*/, &g_0__complex_reshape_in0_instance_norm_node_42_0_id);

    /*************
     * g_0__complex_batch_norm_fwd_batch_size1_43_0 node
     * inputs:
     *     g_0_complex_reshaped_in_instance_norm_tensor_105[320, 8, 64, 1] (dtype=float32)
     *     g_0_tensor_68[320] (dtype=float32)
     *     g_0_tensor_67[320] (dtype=float32)
     *     g_0_tensor_68[320] (dtype=float32)
     *     g_0_tensor_67[320] (dtype=float32)
     * outputs:
     *     g_0_complex_reshaped_out_instance_norm_tensor_106[320, 8, 64, 1] (dtype=float32)
     *     g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1[320, 1] (dtype=float32)
     *     g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2[320, 1] (dtype=float32)
     *     g_0_complex_bn_fwd_running_mean_out_tensor_103[320] (dtype=float32)
     *     g_0_complex_bn_fwd_running_var_out_tensor_104[320] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_68 tensor
    unsigned g_0_tensor_68_max_sizes[] = {320};
    unsigned g_0_tensor_68_min_sizes[] = {320};
    unsigned g_0_tensor_68 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "g_0_tensor_68",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       g_0_tensor_68_max_sizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_68_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_67 tensor
    unsigned g_0_tensor_67_max_sizes[] = {320};
    unsigned g_0_tensor_67_min_sizes[] = {320};
    unsigned g_0_tensor_67 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "g_0_tensor_67",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       g_0_tensor_67_max_sizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_67_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_complex_reshaped_out_instance_norm_tensor_106 tensor
    unsigned g_0_complex_reshaped_out_instance_norm_tensor_106_max_sizes[] = {320,8,64,1};
    unsigned g_0_complex_reshaped_out_instance_norm_tensor_106_min_sizes[] = {320,8,64,1};
    unsigned g_0_complex_reshaped_out_instance_norm_tensor_106 = createTensors(1,
                                                                           OUTPUT_TENSOR,
                                                                           true,
                                                                           "g_0_complex_reshaped_out_instance_norm_tensor_106",
                                                                           MEM_INIT_ALL_ZERO,
                                                                           nullptr,
                                                                           g_0_complex_reshaped_out_instance_norm_tensor_106_max_sizes,
                                                                           4,
                                                                           syn_type_single,
                                                                           nullptr,
                                                                           0,
                                                                           0,
                                                                           nullptr,
                                                                           false,
                                                                           g_0_complex_reshaped_out_instance_norm_tensor_106_min_sizes,
                                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1 tensor
    unsigned g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1_max_sizes[] = {320,1};
    unsigned g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1_min_sizes[] = {320,1};
    unsigned g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1 = createTensors(1,
                                                                                           OUTPUT_TENSOR,
                                                                                           true,
                                                                                           "g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1",
                                                                                           MEM_INIT_ALL_ZERO,
                                                                                           nullptr,
                                                                                           g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1_max_sizes,
                                                                                           2,
                                                                                           syn_type_single,
                                                                                           nullptr,
                                                                                           0,
                                                                                           0,
                                                                                           nullptr,
                                                                                           false,
                                                                                           g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1_min_sizes,
                                                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2 tensor
    unsigned g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2_max_sizes[] = {320,1};
    unsigned g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2_min_sizes[] = {320,1};
    unsigned g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2 = createTensors(1,
                                                                                           OUTPUT_TENSOR,
                                                                                           true,
                                                                                           "g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2",
                                                                                           MEM_INIT_ALL_ZERO,
                                                                                           nullptr,
                                                                                           g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2_max_sizes,
                                                                                           2,
                                                                                           syn_type_single,
                                                                                           nullptr,
                                                                                           0,
                                                                                           0,
                                                                                           nullptr,
                                                                                           false,
                                                                                           g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2_min_sizes,
                                                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_complex_bn_fwd_running_mean_out_tensor_103 tensor
    unsigned g_0_complex_bn_fwd_running_mean_out_tensor_103_max_sizes[] = {320};
    unsigned g_0_complex_bn_fwd_running_mean_out_tensor_103_min_sizes[] = {320};
    unsigned g_0_complex_bn_fwd_running_mean_out_tensor_103 = createTensors(1,
                                                                        OUTPUT_TENSOR,
                                                                        true,
                                                                        "g_0_complex_bn_fwd_running_mean_out_tensor_103",
                                                                        MEM_INIT_ALL_ZERO,
                                                                        nullptr,
                                                                        g_0_complex_bn_fwd_running_mean_out_tensor_103_max_sizes,
                                                                        1,
                                                                        syn_type_single,
                                                                        nullptr,
                                                                        0,
                                                                        0,
                                                                        nullptr,
                                                                        false,
                                                                        g_0_complex_bn_fwd_running_mean_out_tensor_103_min_sizes,
                                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_complex_bn_fwd_running_var_out_tensor_104 tensor
    unsigned g_0_complex_bn_fwd_running_var_out_tensor_104_max_sizes[] = {320};
    unsigned g_0_complex_bn_fwd_running_var_out_tensor_104_min_sizes[] = {320};
    unsigned g_0_complex_bn_fwd_running_var_out_tensor_104 = createTensors(1,
                                                                       OUTPUT_TENSOR,
                                                                       true,
                                                                       "g_0_complex_bn_fwd_running_var_out_tensor_104",
                                                                       MEM_INIT_ALL_ZERO,
                                                                       nullptr,
                                                                       g_0_complex_bn_fwd_running_var_out_tensor_104_max_sizes,
                                                                       1,
                                                                       syn_type_single,
                                                                       nullptr,
                                                                       0,
                                                                       0,
                                                                       nullptr,
                                                                       false,
                                                                       g_0_complex_bn_fwd_running_var_out_tensor_104_min_sizes,
                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__complex_batch_norm_fwd_batch_size1_43_0_id;
    unsigned char g_0__complex_batch_norm_fwd_batch_size1_43_0_params[] = {0,0,0,0,102,102,102,63,172,197,39,55,1,11,32,141};
    addNodeToGraph("batch_norm_fwd_f32", {g_0_complex_reshaped_in_instance_norm_tensor_105, g_0_tensor_68, g_0_tensor_67, g_0_tensor_68, g_0_tensor_67}, {g_0_complex_reshaped_out_instance_norm_tensor_106, g_0_tensor_70_t1241_model_model_3_conv1_norm_hpu__instance_norm_1, g_0_tensor_71_t1243_model_model_3_conv1_norm_hpu__instance_norm_2, g_0_complex_bn_fwd_running_mean_out_tensor_103, g_0_complex_bn_fwd_running_var_out_tensor_104}, (void*)g_0__complex_batch_norm_fwd_batch_size1_43_0_params, 16, "g_0__complex_batch_norm_fwd_batch_size1_43_0", 0 /*graphIndex*/, &g_0__complex_batch_norm_fwd_batch_size1_43_0_id);


    compileTopology("batch_one_pytorch", 0);
}
