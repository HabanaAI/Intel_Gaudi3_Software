

/***************************************************************************************
 ***************************************************************************************
 ***************************************************************************************
 ***                                                                                 ***
 ***        This file is auto-generated. DO NOT EDIT!!!                              ***
 ***                                                                                 ***
 ***        To update this file use json_to_synapse.py script from the gc_tools      ***
 ***                                                                                 ***
 ***************************************************************************************
 ***************************************************************************************
 ***************************************************************************************/

#include "synapse_api.h"
#include <vector>
#include <stdint.h>
#include "gaudi2_resnet_float8_demo_test.h"
TEST_F_GC(SynGaudi2ResNetFloat8Test, resnet50_fwd_downsample_float32_float8_ms0, {synDeviceGaudi2})
{
    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_downsample node
     * inputs: [layer3_5_relu3_output[64, 14, 14, 1024](dtype=fp8_152_t), layer4_downsample_weight[1, 1, 1024, 2048](dtype=fp8_152_t)]
     * output: [layer4_downsample_output(64, 7, 7, 2048)(dtype=fp8_152_t)]
     *************/
    synConvolutionParams layer4_downsample_kernel_params;
    layer4_downsample_kernel_params.dH = 2;
    layer4_downsample_kernel_params.dW = 2;
    layer4_downsample_kernel_params.kH = 1;
    layer4_downsample_kernel_params.kW = 1;
    layer4_downsample_kernel_params.padT = 0;
    layer4_downsample_kernel_params.padB = 0;
    layer4_downsample_kernel_params.padL = 0;
    layer4_downsample_kernel_params.padR = 0;
    layer4_downsample_kernel_params.dilH = 1;
    layer4_downsample_kernel_params.dilW = 1;

    // create layer3_5_relu3_output tensor
    const unsigned layer3_5_relu3_output_sizes[] = {64, 14, 14, 1024};
    uint64_t layer3_5_relu3_output_dram;
    unsigned layer3_5_relu3_output_size = 64*14*14*1024;
    unsigned layer3_5_relu3_output_size_in_bytes = layer3_5_relu3_output_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer3_5_relu3_output_size_in_bytes, &layer3_5_relu3_output_dram, "layer3_5_relu3_output");
    ASSERT_TRUE(status == synSuccess && "layer3_5_relu3_output dram malloc failed!");
    synLaunchTensorInfo layer3_5_relu3_output_tr_info = {"layer3_5_relu3_output", layer3_5_relu3_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer3_5_relu3_output = createTensor(4U, syn_type_fp8_152, layer3_5_relu3_output_sizes, true, "layer3_5_relu3_output");

    // create layer4_downsample_weight tensor
    const unsigned layer4_downsample_weight_sizes[] = {1, 1, 1024, 2048};
    uint64_t layer4_downsample_weight_dram;
    unsigned layer4_downsample_weight_size = 1*1*1024*2048;
    unsigned layer4_downsample_weight_size_in_bytes = layer4_downsample_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_downsample_weight_size_in_bytes, &layer4_downsample_weight_dram, "layer4_downsample_weight");
    ASSERT_TRUE(status == synSuccess && "layer4_downsample_weight dram malloc failed!");
    synLaunchTensorInfo layer4_downsample_weight_tr_info = {"layer4_downsample_weight", layer4_downsample_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_downsample_weight = createTensor(4U, syn_type_fp8_152, layer4_downsample_weight_sizes, true, "layer4_downsample_weight");

    synTensor layer4_downsample_in_vec[4] = {layer3_5_relu3_output, layer4_downsample_weight, nullptr, nullptr};


    // create layer4_downsample_output tensor
    const unsigned layer4_downsample_output_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_downsample_output = createTensor(4U, syn_type_fp8_152, layer4_downsample_output_sizes, false, "layer4_downsample_output");

    synTensor layer4_downsample_out_vec[1] = {layer4_downsample_output};


    status = synNodeCreate(graphHandle, layer4_downsample_in_vec, layer4_downsample_out_vec, 4, 1, (void *)&layer4_downsample_kernel_params, sizeof(layer4_downsample_kernel_params), "spatial_convolution", "layer4_downsample", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample failed!");

    /*************
     * layer4_downsample_output_cast node
     * inputs: [layer4_downsample_output(64, 7, 7, 2048)(dtype=fp8_152_t)]
     * output: [layer4_downsample_output_cast[64, 7, 7, 2048](dtype=float32)]
     *************/

    synTensor layer4_downsample_output_cast_in_vec[1] = {layer4_downsample_output};


    // create layer4_downsample_output_cast tensor
    const unsigned layer4_downsample_output_cast_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_downsample_output_cast = createTensor(4U, syn_type_single, layer4_downsample_output_cast_sizes, false, "layer4_downsample_output_cast");

    synTensor layer4_downsample_output_cast_out_vec[1] = {layer4_downsample_output_cast};


    status = synNodeCreate(graphHandle, layer4_downsample_output_cast_in_vec, layer4_downsample_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "layer4_downsample_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample_output_cast failed!");

    /*************
     * layer4_bn node
     * inputs: [layer4_downsample_output_cast[64, 7, 7, 2048](dtype=float32), layer4_bn_bias[2048](dtype=float32), layer4_bn_weight[2048](dtype=float32), layer4_bn_running_mean[2048](dtype=float32), layer4_bn_running_var[2048](dtype=float32)]
     * output: [layer4_bn_output(64, 7, 7, 2048)(dtype=float32), layer4_bn_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_bn_saved_var(1, 1, 1, 2048)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_bn_kernel_params;
    layer4_bn_kernel_params.momentum = 0.1;
    layer4_bn_kernel_params.threshold.f = 1e-05;
    layer4_bn_kernel_params.epsilon = 1e-05;

    // create layer4_bn_bias tensor
    const unsigned layer4_bn_bias_sizes[] = {2048,};
    uint64_t layer4_bn_bias_dram;
    unsigned layer4_bn_bias_size = 2048;
    unsigned layer4_bn_bias_size_in_bytes = layer4_bn_bias_size * sizeof(float) ;
    status = hbmAlloc(layer4_bn_bias_size_in_bytes, &layer4_bn_bias_dram, "layer4_bn_bias");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_bias dram malloc failed!");
    synLaunchTensorInfo layer4_bn_bias_tr_info = {"layer4_bn_bias", layer4_bn_bias_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_bias = createTensor(1U, syn_type_single, layer4_bn_bias_sizes, true, "layer4_bn_bias");

    // create layer4_bn_weight tensor
    const unsigned layer4_bn_weight_sizes[] = {2048,};
    uint64_t layer4_bn_weight_dram;
    unsigned layer4_bn_weight_size = 2048;
    unsigned layer4_bn_weight_size_in_bytes = layer4_bn_weight_size * sizeof(float) ;
    status = hbmAlloc(layer4_bn_weight_size_in_bytes, &layer4_bn_weight_dram, "layer4_bn_weight");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_weight dram malloc failed!");
    synLaunchTensorInfo layer4_bn_weight_tr_info = {"layer4_bn_weight", layer4_bn_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_weight = createTensor(1U, syn_type_single, layer4_bn_weight_sizes, true, "layer4_bn_weight");

    // create layer4_bn_running_mean tensor
    const unsigned layer4_bn_running_mean_sizes[] = {2048,};
    uint64_t layer4_bn_running_mean_dram;
    unsigned layer4_bn_running_mean_size = 2048;
    unsigned layer4_bn_running_mean_size_in_bytes = layer4_bn_running_mean_size * sizeof(float) ;
    status = hbmAlloc(layer4_bn_running_mean_size_in_bytes, &layer4_bn_running_mean_dram, "layer4_bn_running_mean");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_running_mean dram malloc failed!");
    synLaunchTensorInfo layer4_bn_running_mean_tr_info = {"layer4_bn_running_mean", layer4_bn_running_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_running_mean = createTensor(1U, syn_type_single, layer4_bn_running_mean_sizes, true, "layer4_bn_running_mean");

    // create layer4_bn_running_var tensor
    const unsigned layer4_bn_running_var_sizes[] = {2048,};
    uint64_t layer4_bn_running_var_dram;
    unsigned layer4_bn_running_var_size = 2048;
    unsigned layer4_bn_running_var_size_in_bytes = layer4_bn_running_var_size * sizeof(float) ;
    status = hbmAlloc(layer4_bn_running_var_size_in_bytes, &layer4_bn_running_var_dram, "layer4_bn_running_var");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_running_var dram malloc failed!");
    synLaunchTensorInfo layer4_bn_running_var_tr_info = {"layer4_bn_running_var", layer4_bn_running_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_running_var = createTensor(1U, syn_type_single, layer4_bn_running_var_sizes, true, "layer4_bn_running_var");

    synTensor layer4_bn_in_vec[5] = {layer4_downsample_output_cast, layer4_bn_bias, layer4_bn_weight, layer4_bn_running_mean, layer4_bn_running_var};


    // create layer4_bn_output tensor
    const unsigned layer4_bn_output_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_bn_output = createTensor(4U, syn_type_single, layer4_bn_output_sizes, false, "layer4_bn_output");

    // create layer4_bn_saved_mean tensor
    const unsigned layer4_bn_saved_mean_sizes[] = {2048};
    uint64_t layer4_bn_saved_mean_dram;
    unsigned layer4_bn_saved_mean_size = 1*1*1*2048;
    unsigned layer4_bn_saved_mean_size_in_bytes = layer4_bn_saved_mean_size * sizeof(float) ;
    status = hbmAlloc(layer4_bn_saved_mean_size_in_bytes, &layer4_bn_saved_mean_dram, "layer4_bn_saved_mean");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_saved_mean dram malloc failed!");
    synLaunchTensorInfo layer4_bn_saved_mean_tr_info = {"layer4_bn_saved_mean", layer4_bn_saved_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_saved_mean = createTensor(1U, syn_type_single, layer4_bn_saved_mean_sizes, true, "layer4_bn_saved_mean");

    // create layer4_bn_saved_var tensor
    const unsigned layer4_bn_saved_var_sizes[] = {2048};
    uint64_t layer4_bn_saved_var_dram;
    unsigned layer4_bn_saved_var_size = 1*1*1*2048;
    unsigned layer4_bn_saved_var_size_in_bytes = layer4_bn_saved_var_size * sizeof(float) ;
    status = hbmAlloc(layer4_bn_saved_var_size_in_bytes, &layer4_bn_saved_var_dram, "layer4_bn_saved_var");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_saved_var dram malloc failed!");
    synLaunchTensorInfo layer4_bn_saved_var_tr_info = {"layer4_bn_saved_var", layer4_bn_saved_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_saved_var = createTensor(1U, syn_type_single, layer4_bn_saved_var_sizes, true, "layer4_bn_saved_var");

    synTensor layer4_bn_out_vec[3] = {layer4_bn_output, layer4_bn_saved_mean, layer4_bn_saved_var};


    status = synNodeCreate(graphHandle, layer4_bn_in_vec, layer4_bn_out_vec, 5, 3, (void *)&layer4_bn_kernel_params, sizeof(layer4_bn_kernel_params), "batch_norm_fwd_f32", "layer4_bn", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_bn failed!");

    /*************
     * layer4_bn_output_cast node
     * inputs: [layer4_bn_output(64, 7, 7, 2048)(dtype=float32)]
     * output: [layer4_bn_output_cast(64, 7, 7, 2048)(dtype=fp8_152_t)]
     *************/

    synTensor layer4_bn_output_cast_in_vec[1] = {layer4_bn_output};


    // create layer4_bn_output_cast tensor
    const unsigned layer4_bn_output_cast_sizes[] = {64, 7, 7, 2048};
    uint64_t layer4_bn_output_cast_dram;
    unsigned layer4_bn_output_cast_size = 64*7*7*2048;
    unsigned layer4_bn_output_cast_size_in_bytes = layer4_bn_output_cast_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_bn_output_cast_size_in_bytes, &layer4_bn_output_cast_dram, "layer4_bn_output_cast");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_output_cast dram malloc failed!");
    synLaunchTensorInfo layer4_bn_output_cast_tr_info = {"layer4_bn_output_cast", layer4_bn_output_cast_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_output_cast = createTensor(4U, syn_type_fp8_152, layer4_bn_output_cast_sizes, true, "layer4_bn_output_cast");

    synTensor layer4_bn_output_cast_out_vec[1] = {layer4_bn_output_cast};


    status = synNodeCreate(graphHandle, layer4_bn_output_cast_in_vec, layer4_bn_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f32_to_f8", "layer4_bn_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_bn_output_cast failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer3_5_relu3_output_tr_info);
    graph_inputs.push_back(layer4_downsample_weight_tr_info);
    graph_inputs.push_back(layer4_bn_weight_tr_info);
    graph_inputs.push_back(layer4_bn_bias_tr_info);
    graph_inputs.push_back(layer4_bn_running_mean_tr_info);
    graph_inputs.push_back(layer4_bn_running_var_tr_info);

    graph_outputs.push_back(layer4_bn_saved_mean_tr_info);
    graph_outputs.push_back(layer4_bn_saved_var_tr_info);
    graph_outputs.push_back(layer4_bn_output_cast_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs, true /* skip validation */);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
