

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
TEST_F_GC(SynGaudi2ResNetFloat8Test, resnet50_brc_float32_float8_ms0, {synDeviceGaudi2})
{
    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer1_0_conv1_output_cast node
     * inputs: [layer1_0_conv1_output[64, 56, 56, 64](dtype=fp8_152_t)]
     * output: [layer1_0_conv1_output_cast[64, 56, 56, 64](dtype=float32)]
     *************/

    // create layer1_0_conv1_output tensor
    const unsigned layer1_0_conv1_output_sizes[] = {64, 56, 56, 64};
    uint64_t layer1_0_conv1_output_dram;
    unsigned layer1_0_conv1_output_size = 64*56*56*64;
    unsigned layer1_0_conv1_output_size_in_bytes = layer1_0_conv1_output_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer1_0_conv1_output_size_in_bytes, &layer1_0_conv1_output_dram, "layer1_0_conv1_output");
    ASSERT_TRUE(status == synSuccess && "layer1_0_conv1_output dram malloc failed!");
    synLaunchTensorInfo layer1_0_conv1_output_tr_info = {"layer1_0_conv1_output", layer1_0_conv1_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_conv1_output = createTensor(4U, syn_type_fp8_152, layer1_0_conv1_output_sizes, true, "layer1_0_conv1_output");

    synTensor layer1_0_conv1_output_cast_in_vec[1] = {layer1_0_conv1_output};


    // create layer1_0_conv1_output_cast tensor
    const unsigned layer1_0_conv1_output_cast_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_conv1_output_cast = createTensor(4U, syn_type_single, layer1_0_conv1_output_cast_sizes, false, "layer1_0_conv1_output_cast");

    synTensor layer1_0_conv1_output_cast_out_vec[1] = {layer1_0_conv1_output_cast};


    status = synNodeCreate(graphHandle, layer1_0_conv1_output_cast_in_vec, layer1_0_conv1_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "layer1_0_conv1_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv1_output_cast failed!");

    /*************
     * layer1_0_bn1 node
     * inputs: [layer1_0_conv1_output_cast[64, 56, 56, 64](dtype=float32), layer1_0_bn1_bias[64](dtype=float32), layer1_0_bn1_weight[64](dtype=float32), layer1_0_bn1_running_mean[64](dtype=float32), layer1_0_bn1_running_var[64](dtype=float32)]
     * output: [layer1_0_bn1_output(64, 56, 56, 64)(dtype=float32), layer1_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn1_kernel_params;
    layer1_0_bn1_kernel_params.momentum = 0.1;
    layer1_0_bn1_kernel_params.threshold.f = 1e-05;
    layer1_0_bn1_kernel_params.epsilon = 1e-05;

    // create layer1_0_bn1_bias tensor
    const unsigned layer1_0_bn1_bias_sizes[] = {64,};
    uint64_t layer1_0_bn1_bias_dram;
    unsigned layer1_0_bn1_bias_size = 64;
    unsigned layer1_0_bn1_bias_size_in_bytes = layer1_0_bn1_bias_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn1_bias_size_in_bytes, &layer1_0_bn1_bias_dram, "layer1_0_bn1_bias");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_bias dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn1_bias_tr_info = {"layer1_0_bn1_bias", layer1_0_bn1_bias_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn1_bias = createTensor(1U, syn_type_single, layer1_0_bn1_bias_sizes, true, "layer1_0_bn1_bias");

    // create layer1_0_bn1_weight tensor
    const unsigned layer1_0_bn1_weight_sizes[] = {64,};
    uint64_t layer1_0_bn1_weight_dram;
    unsigned layer1_0_bn1_weight_size = 64;
    unsigned layer1_0_bn1_weight_size_in_bytes = layer1_0_bn1_weight_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn1_weight_size_in_bytes, &layer1_0_bn1_weight_dram, "layer1_0_bn1_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_weight dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn1_weight_tr_info = {"layer1_0_bn1_weight", layer1_0_bn1_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn1_weight = createTensor(1U, syn_type_single, layer1_0_bn1_weight_sizes, true, "layer1_0_bn1_weight");

    // create layer1_0_bn1_running_mean tensor
    const unsigned layer1_0_bn1_running_mean_sizes[] = {64,};
    uint64_t layer1_0_bn1_running_mean_dram;
    unsigned layer1_0_bn1_running_mean_size = 64;
    unsigned layer1_0_bn1_running_mean_size_in_bytes = layer1_0_bn1_running_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn1_running_mean_size_in_bytes, &layer1_0_bn1_running_mean_dram, "layer1_0_bn1_running_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_running_mean dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn1_running_mean_tr_info = {"layer1_0_bn1_running_mean", layer1_0_bn1_running_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn1_running_mean = createTensor(1U, syn_type_single, layer1_0_bn1_running_mean_sizes, true, "layer1_0_bn1_running_mean");

    // create layer1_0_bn1_running_var tensor
    const unsigned layer1_0_bn1_running_var_sizes[] = {64,};
    uint64_t layer1_0_bn1_running_var_dram;
    unsigned layer1_0_bn1_running_var_size = 64;
    unsigned layer1_0_bn1_running_var_size_in_bytes = layer1_0_bn1_running_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn1_running_var_size_in_bytes, &layer1_0_bn1_running_var_dram, "layer1_0_bn1_running_var");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_running_var dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn1_running_var_tr_info = {"layer1_0_bn1_running_var", layer1_0_bn1_running_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn1_running_var = createTensor(1U, syn_type_single, layer1_0_bn1_running_var_sizes, true, "layer1_0_bn1_running_var");

    synTensor layer1_0_bn1_in_vec[5] = {layer1_0_conv1_output_cast, layer1_0_bn1_bias, layer1_0_bn1_weight, layer1_0_bn1_running_mean, layer1_0_bn1_running_var};


    // create layer1_0_bn1_output tensor
    const unsigned layer1_0_bn1_output_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_bn1_output = createTensor(4U, syn_type_single, layer1_0_bn1_output_sizes, false, "layer1_0_bn1_output");

    // create layer1_0_bn1_saved_mean tensor
    const unsigned layer1_0_bn1_saved_mean_sizes[] = {64};
    uint64_t layer1_0_bn1_saved_mean_dram;
    unsigned layer1_0_bn1_saved_mean_size = 1*1*1*64;
    unsigned layer1_0_bn1_saved_mean_size_in_bytes = layer1_0_bn1_saved_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn1_saved_mean_size_in_bytes, &layer1_0_bn1_saved_mean_dram, "layer1_0_bn1_saved_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_saved_mean dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn1_saved_mean_tr_info = {"layer1_0_bn1_saved_mean", layer1_0_bn1_saved_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn1_saved_mean = createTensor(1U, syn_type_single, layer1_0_bn1_saved_mean_sizes, true, "layer1_0_bn1_saved_mean");

    // create layer1_0_bn1_saved_var tensor
    const unsigned layer1_0_bn1_saved_var_sizes[] = {64};
    uint64_t layer1_0_bn1_saved_var_dram;
    unsigned layer1_0_bn1_saved_var_size = 1*1*1*64;
    unsigned layer1_0_bn1_saved_var_size_in_bytes = layer1_0_bn1_saved_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn1_saved_var_size_in_bytes, &layer1_0_bn1_saved_var_dram, "layer1_0_bn1_saved_var");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_saved_var dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn1_saved_var_tr_info = {"layer1_0_bn1_saved_var", layer1_0_bn1_saved_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn1_saved_var = createTensor(1U, syn_type_single, layer1_0_bn1_saved_var_sizes, true, "layer1_0_bn1_saved_var");

    synTensor layer1_0_bn1_out_vec[3] = {layer1_0_bn1_output, layer1_0_bn1_saved_mean, layer1_0_bn1_saved_var};


    status = synNodeCreate(graphHandle, layer1_0_bn1_in_vec, layer1_0_bn1_out_vec, 5, 3, (void *)&layer1_0_bn1_kernel_params, sizeof(layer1_0_bn1_kernel_params), "batch_norm_fwd_f32", "layer1_0_bn1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn1 failed!");

    /*************
     * layer1_0_relu1 node
     * inputs: [layer1_0_bn1_output(64, 56, 56, 64)(dtype=float32)]
     * output: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=float32)]
     *************/

    synTensor layer1_0_relu1_in_vec[1] = {layer1_0_bn1_output};


    // create layer1_0_relu1_output tensor
    const unsigned layer1_0_relu1_output_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_relu1_output = createTensor(4U, syn_type_single, layer1_0_relu1_output_sizes, false, "layer1_0_relu1_output");

    synTensor layer1_0_relu1_out_vec[1] = {layer1_0_relu1_output};


    status = synNodeCreate(graphHandle, layer1_0_relu1_in_vec, layer1_0_relu1_out_vec, 1, 1, nullptr, 0, "relu_fwd_f32", "layer1_0_relu1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu1 failed!");

    /*************
     * layer1_0_relu1_output_cast node
     * inputs: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=float32)]
     * output: [layer1_0_relu1_output_cast[64, 56, 56, 64](dtype=fp8_152_t)]
     *************/

    synTensor layer1_0_relu1_output_cast_in_vec[1] = {layer1_0_relu1_output};


    // create layer1_0_relu1_output_cast tensor
    const unsigned layer1_0_relu1_output_cast_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_relu1_output_cast = createTensor(4U, syn_type_fp8_152, layer1_0_relu1_output_cast_sizes, false, "layer1_0_relu1_output_cast");

    synTensor layer1_0_relu1_output_cast_out_vec[1] = {layer1_0_relu1_output_cast};


    status = synNodeCreate(graphHandle, layer1_0_relu1_output_cast_in_vec, layer1_0_relu1_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f32_to_f8", "layer1_0_relu1_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu1_output_cast failed!");

    /*************
     * layer1_0_conv2 node
     * inputs: [layer1_0_relu1_output_cast[64, 56, 56, 64](dtype=fp8_152_t), layer1_0_conv2_weight[3, 3, 64, 64](dtype=fp8_152_t)]
     * output: [layer1_0_conv2_output(64, 56, 56, 64)(dtype=fp8_152_t)]
     *************/
    synConvolutionParams layer1_0_conv2_kernel_params;
    layer1_0_conv2_kernel_params.dH = 1;
    layer1_0_conv2_kernel_params.dW = 1;
    layer1_0_conv2_kernel_params.kH = 3;
    layer1_0_conv2_kernel_params.kW = 3;
    layer1_0_conv2_kernel_params.padT = 1;
    layer1_0_conv2_kernel_params.padB = 1;
    layer1_0_conv2_kernel_params.padL = 1;
    layer1_0_conv2_kernel_params.padR = 1;
    layer1_0_conv2_kernel_params.dilH = 1;
    layer1_0_conv2_kernel_params.dilW = 1;

    // create layer1_0_conv2_weight tensor
    const unsigned layer1_0_conv2_weight_sizes[] = {3, 3, 64, 64};
    uint64_t layer1_0_conv2_weight_dram;
    unsigned layer1_0_conv2_weight_size = 3*3*64*64;
    unsigned layer1_0_conv2_weight_size_in_bytes = layer1_0_conv2_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer1_0_conv2_weight_size_in_bytes, &layer1_0_conv2_weight_dram, "layer1_0_conv2_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_0_conv2_weight dram malloc failed!");
    synLaunchTensorInfo layer1_0_conv2_weight_tr_info = {"layer1_0_conv2_weight", layer1_0_conv2_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_conv2_weight = createTensor(4U, syn_type_fp8_152, layer1_0_conv2_weight_sizes, true, "layer1_0_conv2_weight");

    synTensor layer1_0_conv2_in_vec[4] = {layer1_0_relu1_output_cast, layer1_0_conv2_weight, nullptr, nullptr};


    // create layer1_0_conv2_output tensor
    const unsigned layer1_0_conv2_output_sizes[] = {64, 56, 56, 64};
    uint64_t layer1_0_conv2_output_dram;
    unsigned layer1_0_conv2_output_size = 64*56*56*64;
    unsigned layer1_0_conv2_output_size_in_bytes = layer1_0_conv2_output_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer1_0_conv2_output_size_in_bytes, &layer1_0_conv2_output_dram, "layer1_0_conv2_output");
    ASSERT_TRUE(status == synSuccess && "layer1_0_conv2_output dram malloc failed!");
    synLaunchTensorInfo layer1_0_conv2_output_tr_info = {"layer1_0_conv2_output", layer1_0_conv2_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_conv2_output = createTensor(4U, syn_type_fp8_152, layer1_0_conv2_output_sizes, true, "layer1_0_conv2_output");

    synTensor layer1_0_conv2_out_vec[1] = {layer1_0_conv2_output};


    status = synNodeCreate(graphHandle, layer1_0_conv2_in_vec, layer1_0_conv2_out_vec, 4, 1, (void *)&layer1_0_conv2_kernel_params, sizeof(layer1_0_conv2_kernel_params), "spatial_convolution", "layer1_0_conv2", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv2 failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer1_0_conv1_output_tr_info);
    graph_inputs.push_back(layer1_0_bn1_weight_tr_info);
    graph_inputs.push_back(layer1_0_bn1_bias_tr_info);
    graph_inputs.push_back(layer1_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer1_0_bn1_running_var_tr_info);
    graph_inputs.push_back(layer1_0_conv2_weight_tr_info);

    graph_outputs.push_back(layer1_0_bn1_saved_mean_tr_info);
    graph_outputs.push_back(layer1_0_bn1_saved_var_tr_info);
    graph_outputs.push_back(layer1_0_conv2_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs, true /* skip validation */);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
