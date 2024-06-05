

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
TEST_F_GC(SynGaudi2ResNetFloat8Test, resnet50_single_residual_float32_float8_ms0, {synDeviceGaudi2})
{
    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer1_0_conv1 node
     * inputs: [worker_0_maxpool_output[64, 56, 56, 64](dtype=fp8_152_t), layer1_0_conv1_weight[1, 1, 64, 64](dtype=fp8_152_t)]
     * output: [layer1_0_conv1_output(64, 56, 56, 64)(dtype=fp8_152_t)]
     *************/
    synConvolutionParams layer1_0_conv1_kernel_params;
    layer1_0_conv1_kernel_params.dH = 1;
    layer1_0_conv1_kernel_params.dW = 1;
    layer1_0_conv1_kernel_params.kH = 1;
    layer1_0_conv1_kernel_params.kW = 1;
    layer1_0_conv1_kernel_params.padT = 0;
    layer1_0_conv1_kernel_params.padB = 0;
    layer1_0_conv1_kernel_params.padL = 0;
    layer1_0_conv1_kernel_params.padR = 0;
    layer1_0_conv1_kernel_params.dilH = 1;
    layer1_0_conv1_kernel_params.dilW = 1;

    // create worker_0_maxpool_output tensor
    const unsigned worker_0_maxpool_output_sizes[] = {64, 56, 56, 64};
    uint64_t worker_0_maxpool_output_dram;
    unsigned worker_0_maxpool_output_size = 64*56*56*64;
    unsigned worker_0_maxpool_output_size_in_bytes = worker_0_maxpool_output_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(worker_0_maxpool_output_size_in_bytes, &worker_0_maxpool_output_dram, "worker_0_maxpool_output");
    ASSERT_TRUE(status == synSuccess && "worker_0_maxpool_output dram malloc failed!");
    synLaunchTensorInfo worker_0_maxpool_output_tr_info = {"worker_0_maxpool_output", worker_0_maxpool_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_maxpool_output = createTensor(4U, syn_type_fp8_152, worker_0_maxpool_output_sizes, true, "worker_0_maxpool_output");

    // create layer1_0_conv1_weight tensor
    const unsigned layer1_0_conv1_weight_sizes[] = {1, 1, 64, 64};
    uint64_t layer1_0_conv1_weight_dram;
    unsigned layer1_0_conv1_weight_size = 1*1*64*64;
    unsigned layer1_0_conv1_weight_size_in_bytes = layer1_0_conv1_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer1_0_conv1_weight_size_in_bytes, &layer1_0_conv1_weight_dram, "layer1_0_conv1_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_0_conv1_weight dram malloc failed!");
    synLaunchTensorInfo layer1_0_conv1_weight_tr_info = {"layer1_0_conv1_weight", layer1_0_conv1_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_conv1_weight = createTensor(4U, syn_type_fp8_152, layer1_0_conv1_weight_sizes, true, "layer1_0_conv1_weight");

    synTensor layer1_0_conv1_in_vec[4] = {worker_0_maxpool_output, layer1_0_conv1_weight, nullptr, nullptr};


    // create layer1_0_conv1_output tensor
    const unsigned layer1_0_conv1_output_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_conv1_output = createTensor(4U, syn_type_fp8_152, layer1_0_conv1_output_sizes, false, "layer1_0_conv1_output");

    synTensor layer1_0_conv1_out_vec[1] = {layer1_0_conv1_output};


    status = synNodeCreate(graphHandle, layer1_0_conv1_in_vec, layer1_0_conv1_out_vec, 4, 1, (void *)&layer1_0_conv1_kernel_params, sizeof(layer1_0_conv1_kernel_params), "spatial_convolution", "layer1_0_conv1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv1 failed!");

    /*************
     * layer1_0_conv1_output_cast node
     * inputs: [layer1_0_conv1_output(64, 56, 56, 64)(dtype=fp8_152_t)]
     * output: [layer1_0_conv1_output_cast[64, 56, 56, 64](dtype=float32)]
     *************/

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
    synTensor layer1_0_conv2_output = createTensor(4U, syn_type_fp8_152, layer1_0_conv2_output_sizes, false, "layer1_0_conv2_output");

    synTensor layer1_0_conv2_out_vec[1] = {layer1_0_conv2_output};


    status = synNodeCreate(graphHandle, layer1_0_conv2_in_vec, layer1_0_conv2_out_vec, 4, 1, (void *)&layer1_0_conv2_kernel_params, sizeof(layer1_0_conv2_kernel_params), "spatial_convolution", "layer1_0_conv2", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv2 failed!");

    /*************
     * layer1_0_conv2_output_cast node
     * inputs: [layer1_0_conv2_output(64, 56, 56, 64)(dtype=fp8_152_t)]
     * output: [layer1_0_conv2_output_cast[64, 56, 56, 64](dtype=float32)]
     *************/

    synTensor layer1_0_conv2_output_cast_in_vec[1] = {layer1_0_conv2_output};


    // create layer1_0_conv2_output_cast tensor
    const unsigned layer1_0_conv2_output_cast_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_conv2_output_cast = createTensor(4U, syn_type_single, layer1_0_conv2_output_cast_sizes, false, "layer1_0_conv2_output_cast");

    synTensor layer1_0_conv2_output_cast_out_vec[1] = {layer1_0_conv2_output_cast};


    status = synNodeCreate(graphHandle, layer1_0_conv2_output_cast_in_vec, layer1_0_conv2_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "layer1_0_conv2_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv2_output_cast failed!");

    /*************
     * layer1_0_bn2 node
     * inputs: [layer1_0_conv2_output_cast[64, 56, 56, 64](dtype=float32), layer1_0_bn2_bias[64](dtype=float32), layer1_0_bn2_weight[64](dtype=float32), layer1_0_bn2_running_mean[64](dtype=float32), layer1_0_bn2_running_var[64](dtype=float32)]
     * output: [layer1_0_bn2_output(64, 56, 56, 64)(dtype=float32), layer1_0_bn2_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn2_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn2_kernel_params;
    layer1_0_bn2_kernel_params.momentum = 0.1;
    layer1_0_bn2_kernel_params.threshold.f = 1e-05;
    layer1_0_bn2_kernel_params.epsilon = 1e-05;

    // create layer1_0_bn2_bias tensor
    const unsigned layer1_0_bn2_bias_sizes[] = {64,};
    uint64_t layer1_0_bn2_bias_dram;
    unsigned layer1_0_bn2_bias_size = 64;
    unsigned layer1_0_bn2_bias_size_in_bytes = layer1_0_bn2_bias_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn2_bias_size_in_bytes, &layer1_0_bn2_bias_dram, "layer1_0_bn2_bias");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn2_bias dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn2_bias_tr_info = {"layer1_0_bn2_bias", layer1_0_bn2_bias_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn2_bias = createTensor(1U, syn_type_single, layer1_0_bn2_bias_sizes, true, "layer1_0_bn2_bias");

    // create layer1_0_bn2_weight tensor
    const unsigned layer1_0_bn2_weight_sizes[] = {64,};
    uint64_t layer1_0_bn2_weight_dram;
    unsigned layer1_0_bn2_weight_size = 64;
    unsigned layer1_0_bn2_weight_size_in_bytes = layer1_0_bn2_weight_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn2_weight_size_in_bytes, &layer1_0_bn2_weight_dram, "layer1_0_bn2_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn2_weight dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn2_weight_tr_info = {"layer1_0_bn2_weight", layer1_0_bn2_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn2_weight = createTensor(1U, syn_type_single, layer1_0_bn2_weight_sizes, true, "layer1_0_bn2_weight");

    // create layer1_0_bn2_running_mean tensor
    const unsigned layer1_0_bn2_running_mean_sizes[] = {64,};
    uint64_t layer1_0_bn2_running_mean_dram;
    unsigned layer1_0_bn2_running_mean_size = 64;
    unsigned layer1_0_bn2_running_mean_size_in_bytes = layer1_0_bn2_running_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn2_running_mean_size_in_bytes, &layer1_0_bn2_running_mean_dram, "layer1_0_bn2_running_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn2_running_mean dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn2_running_mean_tr_info = {"layer1_0_bn2_running_mean", layer1_0_bn2_running_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn2_running_mean = createTensor(1U, syn_type_single, layer1_0_bn2_running_mean_sizes, true, "layer1_0_bn2_running_mean");

    // create layer1_0_bn2_running_var tensor
    const unsigned layer1_0_bn2_running_var_sizes[] = {64,};
    uint64_t layer1_0_bn2_running_var_dram;
    unsigned layer1_0_bn2_running_var_size = 64;
    unsigned layer1_0_bn2_running_var_size_in_bytes = layer1_0_bn2_running_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn2_running_var_size_in_bytes, &layer1_0_bn2_running_var_dram, "layer1_0_bn2_running_var");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn2_running_var dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn2_running_var_tr_info = {"layer1_0_bn2_running_var", layer1_0_bn2_running_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn2_running_var = createTensor(1U, syn_type_single, layer1_0_bn2_running_var_sizes, true, "layer1_0_bn2_running_var");

    synTensor layer1_0_bn2_in_vec[5] = {layer1_0_conv2_output_cast, layer1_0_bn2_bias, layer1_0_bn2_weight, layer1_0_bn2_running_mean, layer1_0_bn2_running_var};


    // create layer1_0_bn2_output tensor
    const unsigned layer1_0_bn2_output_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_bn2_output = createTensor(4U, syn_type_single, layer1_0_bn2_output_sizes, false, "layer1_0_bn2_output");

    // create layer1_0_bn2_saved_mean tensor
    const unsigned layer1_0_bn2_saved_mean_sizes[] = {64};
    uint64_t layer1_0_bn2_saved_mean_dram;
    unsigned layer1_0_bn2_saved_mean_size = 1*1*1*64;
    unsigned layer1_0_bn2_saved_mean_size_in_bytes = layer1_0_bn2_saved_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn2_saved_mean_size_in_bytes, &layer1_0_bn2_saved_mean_dram, "layer1_0_bn2_saved_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn2_saved_mean dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn2_saved_mean_tr_info = {"layer1_0_bn2_saved_mean", layer1_0_bn2_saved_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn2_saved_mean = createTensor(1U, syn_type_single, layer1_0_bn2_saved_mean_sizes, true, "layer1_0_bn2_saved_mean");

    // create layer1_0_bn2_saved_var tensor
    const unsigned layer1_0_bn2_saved_var_sizes[] = {64};
    uint64_t layer1_0_bn2_saved_var_dram;
    unsigned layer1_0_bn2_saved_var_size = 1*1*1*64;
    unsigned layer1_0_bn2_saved_var_size_in_bytes = layer1_0_bn2_saved_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn2_saved_var_size_in_bytes, &layer1_0_bn2_saved_var_dram, "layer1_0_bn2_saved_var");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn2_saved_var dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn2_saved_var_tr_info = {"layer1_0_bn2_saved_var", layer1_0_bn2_saved_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn2_saved_var = createTensor(1U, syn_type_single, layer1_0_bn2_saved_var_sizes, true, "layer1_0_bn2_saved_var");

    synTensor layer1_0_bn2_out_vec[3] = {layer1_0_bn2_output, layer1_0_bn2_saved_mean, layer1_0_bn2_saved_var};


    status = synNodeCreate(graphHandle, layer1_0_bn2_in_vec, layer1_0_bn2_out_vec, 5, 3, (void *)&layer1_0_bn2_kernel_params, sizeof(layer1_0_bn2_kernel_params), "batch_norm_fwd_f32", "layer1_0_bn2", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn2 failed!");

    /*************
     * layer1_0_relu2 node
     * inputs: [layer1_0_bn2_output(64, 56, 56, 64)(dtype=float32)]
     * output: [layer1_0_relu2_output(64, 56, 56, 64)(dtype=float32)]
     *************/

    synTensor layer1_0_relu2_in_vec[1] = {layer1_0_bn2_output};


    // create layer1_0_relu2_output tensor
    const unsigned layer1_0_relu2_output_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_relu2_output = createTensor(4U, syn_type_single, layer1_0_relu2_output_sizes, false, "layer1_0_relu2_output");

    synTensor layer1_0_relu2_out_vec[1] = {layer1_0_relu2_output};


    status = synNodeCreate(graphHandle, layer1_0_relu2_in_vec, layer1_0_relu2_out_vec, 1, 1, nullptr, 0, "relu_fwd_f32", "layer1_0_relu2", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu2 failed!");

    /*************
     * layer1_0_relu2_output_cast node
     * inputs: [layer1_0_relu2_output(64, 56, 56, 64)(dtype=float32)]
     * output: [layer1_0_relu2_output_cast[64, 56, 56, 64](dtype=fp8_152_t)]
     *************/

    synTensor layer1_0_relu2_output_cast_in_vec[1] = {layer1_0_relu2_output};


    // create layer1_0_relu2_output_cast tensor
    const unsigned layer1_0_relu2_output_cast_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_relu2_output_cast = createTensor(4U, syn_type_fp8_152, layer1_0_relu2_output_cast_sizes, false, "layer1_0_relu2_output_cast");

    synTensor layer1_0_relu2_output_cast_out_vec[1] = {layer1_0_relu2_output_cast};


    status = synNodeCreate(graphHandle, layer1_0_relu2_output_cast_in_vec, layer1_0_relu2_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f32_to_f8", "layer1_0_relu2_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu2_output_cast failed!");

    /*************
     * layer1_0_conv3 node
     * inputs: [layer1_0_relu2_output_cast[64, 56, 56, 64](dtype=fp8_152_t), layer1_0_conv3_weight[1, 1, 64, 256](dtype=fp8_152_t)]
     * output: [layer1_0_conv3_output(64, 56, 56, 256)(dtype=fp8_152_t)]
     *************/
    synConvolutionParams layer1_0_conv3_kernel_params;
    layer1_0_conv3_kernel_params.dH = 1;
    layer1_0_conv3_kernel_params.dW = 1;
    layer1_0_conv3_kernel_params.kH = 1;
    layer1_0_conv3_kernel_params.kW = 1;
    layer1_0_conv3_kernel_params.padT = 0;
    layer1_0_conv3_kernel_params.padB = 0;
    layer1_0_conv3_kernel_params.padL = 0;
    layer1_0_conv3_kernel_params.padR = 0;
    layer1_0_conv3_kernel_params.dilH = 1;
    layer1_0_conv3_kernel_params.dilW = 1;

    // create layer1_0_conv3_weight tensor
    const unsigned layer1_0_conv3_weight_sizes[] = {1, 1, 64, 256};
    uint64_t layer1_0_conv3_weight_dram;
    unsigned layer1_0_conv3_weight_size = 1*1*64*256;
    unsigned layer1_0_conv3_weight_size_in_bytes = layer1_0_conv3_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer1_0_conv3_weight_size_in_bytes, &layer1_0_conv3_weight_dram, "layer1_0_conv3_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_0_conv3_weight dram malloc failed!");
    synLaunchTensorInfo layer1_0_conv3_weight_tr_info = {"layer1_0_conv3_weight", layer1_0_conv3_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_conv3_weight = createTensor(4U, syn_type_fp8_152, layer1_0_conv3_weight_sizes, true, "layer1_0_conv3_weight");

    synTensor layer1_0_conv3_in_vec[4] = {layer1_0_relu2_output_cast, layer1_0_conv3_weight, nullptr, nullptr};


    // create layer1_0_conv3_output tensor
    const unsigned layer1_0_conv3_output_sizes[] = {64, 56, 56, 256};
    synTensor layer1_0_conv3_output = createTensor(4U, syn_type_fp8_152, layer1_0_conv3_output_sizes, false, "layer1_0_conv3_output");

    synTensor layer1_0_conv3_out_vec[1] = {layer1_0_conv3_output};


    status = synNodeCreate(graphHandle, layer1_0_conv3_in_vec, layer1_0_conv3_out_vec, 4, 1, (void *)&layer1_0_conv3_kernel_params, sizeof(layer1_0_conv3_kernel_params), "spatial_convolution", "layer1_0_conv3", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv3 failed!");

    /*************
     * layer1_0_conv3_output_cast node
     * inputs: [layer1_0_conv3_output(64, 56, 56, 256)(dtype=fp8_152_t)]
     * output: [layer1_0_conv3_output_cast[64, 56, 56, 256](dtype=float32)]
     *************/

    synTensor layer1_0_conv3_output_cast_in_vec[1] = {layer1_0_conv3_output};


    // create layer1_0_conv3_output_cast tensor
    const unsigned layer1_0_conv3_output_cast_sizes[] = {64, 56, 56, 256};
    synTensor layer1_0_conv3_output_cast = createTensor(4U, syn_type_single, layer1_0_conv3_output_cast_sizes, false, "layer1_0_conv3_output_cast");

    synTensor layer1_0_conv3_output_cast_out_vec[1] = {layer1_0_conv3_output_cast};


    status = synNodeCreate(graphHandle, layer1_0_conv3_output_cast_in_vec, layer1_0_conv3_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "layer1_0_conv3_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv3_output_cast failed!");

    /*************
     * layer1_0_bn3 node
     * inputs: [layer1_0_conv3_output_cast[64, 56, 56, 256](dtype=float32), layer1_0_bn3_bias[256](dtype=float32), layer1_0_bn3_weight[256](dtype=float32), layer1_0_bn3_running_mean[256](dtype=float32), layer1_0_bn3_running_var[256](dtype=float32)]
     * output: [layer1_0_bn3_output(64, 56, 56, 256)(dtype=float32), layer1_0_bn3_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_0_bn3_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn3_kernel_params;
    layer1_0_bn3_kernel_params.momentum = 0.1;
    layer1_0_bn3_kernel_params.threshold.f = 1e-05;
    layer1_0_bn3_kernel_params.epsilon = 1e-05;

    // create layer1_0_bn3_bias tensor
    const unsigned layer1_0_bn3_bias_sizes[] = {256,};
    uint64_t layer1_0_bn3_bias_dram;
    unsigned layer1_0_bn3_bias_size = 256;
    unsigned layer1_0_bn3_bias_size_in_bytes = layer1_0_bn3_bias_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn3_bias_size_in_bytes, &layer1_0_bn3_bias_dram, "layer1_0_bn3_bias");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn3_bias dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn3_bias_tr_info = {"layer1_0_bn3_bias", layer1_0_bn3_bias_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn3_bias = createTensor(1U, syn_type_single, layer1_0_bn3_bias_sizes, true, "layer1_0_bn3_bias");

    // create layer1_0_bn3_weight tensor
    const unsigned layer1_0_bn3_weight_sizes[] = {256,};
    uint64_t layer1_0_bn3_weight_dram;
    unsigned layer1_0_bn3_weight_size = 256;
    unsigned layer1_0_bn3_weight_size_in_bytes = layer1_0_bn3_weight_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn3_weight_size_in_bytes, &layer1_0_bn3_weight_dram, "layer1_0_bn3_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn3_weight dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn3_weight_tr_info = {"layer1_0_bn3_weight", layer1_0_bn3_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn3_weight = createTensor(1U, syn_type_single, layer1_0_bn3_weight_sizes, true, "layer1_0_bn3_weight");

    // create layer1_0_bn3_running_mean tensor
    const unsigned layer1_0_bn3_running_mean_sizes[] = {256,};
    uint64_t layer1_0_bn3_running_mean_dram;
    unsigned layer1_0_bn3_running_mean_size = 256;
    unsigned layer1_0_bn3_running_mean_size_in_bytes = layer1_0_bn3_running_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn3_running_mean_size_in_bytes, &layer1_0_bn3_running_mean_dram, "layer1_0_bn3_running_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn3_running_mean dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn3_running_mean_tr_info = {"layer1_0_bn3_running_mean", layer1_0_bn3_running_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn3_running_mean = createTensor(1U, syn_type_single, layer1_0_bn3_running_mean_sizes, true, "layer1_0_bn3_running_mean");

    // create layer1_0_bn3_running_var tensor
    const unsigned layer1_0_bn3_running_var_sizes[] = {256,};
    uint64_t layer1_0_bn3_running_var_dram;
    unsigned layer1_0_bn3_running_var_size = 256;
    unsigned layer1_0_bn3_running_var_size_in_bytes = layer1_0_bn3_running_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn3_running_var_size_in_bytes, &layer1_0_bn3_running_var_dram, "layer1_0_bn3_running_var");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn3_running_var dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn3_running_var_tr_info = {"layer1_0_bn3_running_var", layer1_0_bn3_running_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn3_running_var = createTensor(1U, syn_type_single, layer1_0_bn3_running_var_sizes, true, "layer1_0_bn3_running_var");

    synTensor layer1_0_bn3_in_vec[5] = {layer1_0_conv3_output_cast, layer1_0_bn3_bias, layer1_0_bn3_weight, layer1_0_bn3_running_mean, layer1_0_bn3_running_var};


    // create layer1_0_bn3_output tensor
    const unsigned layer1_0_bn3_output_sizes[] = {64, 56, 56, 256};
    synTensor layer1_0_bn3_output = createTensor(4U, syn_type_single, layer1_0_bn3_output_sizes, false, "layer1_0_bn3_output");

    // create layer1_0_bn3_saved_mean tensor
    const unsigned layer1_0_bn3_saved_mean_sizes[] = {256};
    uint64_t layer1_0_bn3_saved_mean_dram;
    unsigned layer1_0_bn3_saved_mean_size = 1*1*1*256;
    unsigned layer1_0_bn3_saved_mean_size_in_bytes = layer1_0_bn3_saved_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn3_saved_mean_size_in_bytes, &layer1_0_bn3_saved_mean_dram, "layer1_0_bn3_saved_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn3_saved_mean dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn3_saved_mean_tr_info = {"layer1_0_bn3_saved_mean", layer1_0_bn3_saved_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn3_saved_mean = createTensor(1U, syn_type_single, layer1_0_bn3_saved_mean_sizes, true, "layer1_0_bn3_saved_mean");

    // create layer1_0_bn3_saved_var tensor
    const unsigned layer1_0_bn3_saved_var_sizes[] = {256};
    uint64_t layer1_0_bn3_saved_var_dram;
    unsigned layer1_0_bn3_saved_var_size = 1*1*1*256;
    unsigned layer1_0_bn3_saved_var_size_in_bytes = layer1_0_bn3_saved_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_0_bn3_saved_var_size_in_bytes, &layer1_0_bn3_saved_var_dram, "layer1_0_bn3_saved_var");
    ASSERT_TRUE(status == synSuccess && "layer1_0_bn3_saved_var dram malloc failed!");
    synLaunchTensorInfo layer1_0_bn3_saved_var_tr_info = {"layer1_0_bn3_saved_var", layer1_0_bn3_saved_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_bn3_saved_var = createTensor(1U, syn_type_single, layer1_0_bn3_saved_var_sizes, true, "layer1_0_bn3_saved_var");

    synTensor layer1_0_bn3_out_vec[3] = {layer1_0_bn3_output, layer1_0_bn3_saved_mean, layer1_0_bn3_saved_var};


    status = synNodeCreate(graphHandle, layer1_0_bn3_in_vec, layer1_0_bn3_out_vec, 5, 3, (void *)&layer1_0_bn3_kernel_params, sizeof(layer1_0_bn3_kernel_params), "batch_norm_fwd_f32", "layer1_0_bn3", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn3 failed!");

    /*************
     * layer1_downsample node
     * inputs: [worker_0_maxpool_output[64, 56, 56, 64](dtype=fp8_152_t), layer1_downsample_weight[1, 1, 64, 256](dtype=fp8_152_t)]
     * output: [layer1_downsample_output(64, 56, 56, 256)(dtype=fp8_152_t)]
     *************/
    synConvolutionParams layer1_downsample_kernel_params;
    layer1_downsample_kernel_params.dH = 1;
    layer1_downsample_kernel_params.dW = 1;
    layer1_downsample_kernel_params.kH = 1;
    layer1_downsample_kernel_params.kW = 1;
    layer1_downsample_kernel_params.padT = 0;
    layer1_downsample_kernel_params.padB = 0;
    layer1_downsample_kernel_params.padL = 0;
    layer1_downsample_kernel_params.padR = 0;
    layer1_downsample_kernel_params.dilH = 1;
    layer1_downsample_kernel_params.dilW = 1;

    // create layer1_downsample_weight tensor
    const unsigned layer1_downsample_weight_sizes[] = {1, 1, 64, 256};
    uint64_t layer1_downsample_weight_dram;
    unsigned layer1_downsample_weight_size = 1*1*64*256;
    unsigned layer1_downsample_weight_size_in_bytes = layer1_downsample_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer1_downsample_weight_size_in_bytes, &layer1_downsample_weight_dram, "layer1_downsample_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_downsample_weight dram malloc failed!");
    synLaunchTensorInfo layer1_downsample_weight_tr_info = {"layer1_downsample_weight", layer1_downsample_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_downsample_weight = createTensor(4U, syn_type_fp8_152, layer1_downsample_weight_sizes, true, "layer1_downsample_weight");

    synTensor layer1_downsample_in_vec[4] = {worker_0_maxpool_output, layer1_downsample_weight, nullptr, nullptr};


    // create layer1_downsample_output tensor
    const unsigned layer1_downsample_output_sizes[] = {64, 56, 56, 256};
    synTensor layer1_downsample_output = createTensor(4U, syn_type_fp8_152, layer1_downsample_output_sizes, false, "layer1_downsample_output");

    synTensor layer1_downsample_out_vec[1] = {layer1_downsample_output};


    status = synNodeCreate(graphHandle, layer1_downsample_in_vec, layer1_downsample_out_vec, 4, 1, (void *)&layer1_downsample_kernel_params, sizeof(layer1_downsample_kernel_params), "spatial_convolution", "layer1_downsample", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_downsample failed!");

    /*************
     * layer1_downsample_output_cast node
     * inputs: [layer1_downsample_output(64, 56, 56, 256)(dtype=fp8_152_t)]
     * output: [layer1_downsample_output_cast[64, 56, 56, 256](dtype=float32)]
     *************/

    synTensor layer1_downsample_output_cast_in_vec[1] = {layer1_downsample_output};


    // create layer1_downsample_output_cast tensor
    const unsigned layer1_downsample_output_cast_sizes[] = {64, 56, 56, 256};
    synTensor layer1_downsample_output_cast = createTensor(4U, syn_type_single, layer1_downsample_output_cast_sizes, false, "layer1_downsample_output_cast");

    synTensor layer1_downsample_output_cast_out_vec[1] = {layer1_downsample_output_cast};


    status = synNodeCreate(graphHandle, layer1_downsample_output_cast_in_vec, layer1_downsample_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "layer1_downsample_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_downsample_output_cast failed!");

    /*************
     * layer1_bn node
     * inputs: [layer1_downsample_output_cast[64, 56, 56, 256](dtype=float32), layer1_bn_bias[256](dtype=float32), layer1_bn_weight[256](dtype=float32), layer1_bn_running_mean[256](dtype=float32), layer1_bn_running_var[256](dtype=float32)]
     * output: [layer1_bn_output(64, 56, 56, 256)(dtype=float32), layer1_bn_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_bn_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_bn_kernel_params;
    layer1_bn_kernel_params.momentum = 0.1;
    layer1_bn_kernel_params.threshold.f = 1e-05;
    layer1_bn_kernel_params.epsilon = 1e-05;

    // create layer1_bn_bias tensor
    const unsigned layer1_bn_bias_sizes[] = {256,};
    uint64_t layer1_bn_bias_dram;
    unsigned layer1_bn_bias_size = 256;
    unsigned layer1_bn_bias_size_in_bytes = layer1_bn_bias_size * sizeof(float) ;
    status = hbmAlloc(layer1_bn_bias_size_in_bytes, &layer1_bn_bias_dram, "layer1_bn_bias");
    ASSERT_TRUE(status == synSuccess && "layer1_bn_bias dram malloc failed!");
    synLaunchTensorInfo layer1_bn_bias_tr_info = {"layer1_bn_bias", layer1_bn_bias_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_bn_bias = createTensor(1U, syn_type_single, layer1_bn_bias_sizes, true, "layer1_bn_bias");

    // create layer1_bn_weight tensor
    const unsigned layer1_bn_weight_sizes[] = {256,};
    uint64_t layer1_bn_weight_dram;
    unsigned layer1_bn_weight_size = 256;
    unsigned layer1_bn_weight_size_in_bytes = layer1_bn_weight_size * sizeof(float) ;
    status = hbmAlloc(layer1_bn_weight_size_in_bytes, &layer1_bn_weight_dram, "layer1_bn_weight");
    ASSERT_TRUE(status == synSuccess && "layer1_bn_weight dram malloc failed!");
    synLaunchTensorInfo layer1_bn_weight_tr_info = {"layer1_bn_weight", layer1_bn_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_bn_weight = createTensor(1U, syn_type_single, layer1_bn_weight_sizes, true, "layer1_bn_weight");

    // create layer1_bn_running_mean tensor
    const unsigned layer1_bn_running_mean_sizes[] = {256,};
    uint64_t layer1_bn_running_mean_dram;
    unsigned layer1_bn_running_mean_size = 256;
    unsigned layer1_bn_running_mean_size_in_bytes = layer1_bn_running_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_bn_running_mean_size_in_bytes, &layer1_bn_running_mean_dram, "layer1_bn_running_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_bn_running_mean dram malloc failed!");
    synLaunchTensorInfo layer1_bn_running_mean_tr_info = {"layer1_bn_running_mean", layer1_bn_running_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_bn_running_mean = createTensor(1U, syn_type_single, layer1_bn_running_mean_sizes, true, "layer1_bn_running_mean");

    // create layer1_bn_running_var tensor
    const unsigned layer1_bn_running_var_sizes[] = {256,};
    uint64_t layer1_bn_running_var_dram;
    unsigned layer1_bn_running_var_size = 256;
    unsigned layer1_bn_running_var_size_in_bytes = layer1_bn_running_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_bn_running_var_size_in_bytes, &layer1_bn_running_var_dram, "layer1_bn_running_var");
    ASSERT_TRUE(status == synSuccess && "layer1_bn_running_var dram malloc failed!");
    synLaunchTensorInfo layer1_bn_running_var_tr_info = {"layer1_bn_running_var", layer1_bn_running_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_bn_running_var = createTensor(1U, syn_type_single, layer1_bn_running_var_sizes, true, "layer1_bn_running_var");

    synTensor layer1_bn_in_vec[5] = {layer1_downsample_output_cast, layer1_bn_bias, layer1_bn_weight, layer1_bn_running_mean, layer1_bn_running_var};


    // create layer1_bn_output tensor
    const unsigned layer1_bn_output_sizes[] = {64, 56, 56, 256};
    synTensor layer1_bn_output = createTensor(4U, syn_type_single, layer1_bn_output_sizes, false, "layer1_bn_output");

    // create layer1_bn_saved_mean tensor
    const unsigned layer1_bn_saved_mean_sizes[] = {256};
    uint64_t layer1_bn_saved_mean_dram;
    unsigned layer1_bn_saved_mean_size = 1*1*1*256;
    unsigned layer1_bn_saved_mean_size_in_bytes = layer1_bn_saved_mean_size * sizeof(float) ;
    status = hbmAlloc(layer1_bn_saved_mean_size_in_bytes, &layer1_bn_saved_mean_dram, "layer1_bn_saved_mean");
    ASSERT_TRUE(status == synSuccess && "layer1_bn_saved_mean dram malloc failed!");
    synLaunchTensorInfo layer1_bn_saved_mean_tr_info = {"layer1_bn_saved_mean", layer1_bn_saved_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_bn_saved_mean = createTensor(1U, syn_type_single, layer1_bn_saved_mean_sizes, true, "layer1_bn_saved_mean");

    // create layer1_bn_saved_var tensor
    const unsigned layer1_bn_saved_var_sizes[] = {256};
    uint64_t layer1_bn_saved_var_dram;
    unsigned layer1_bn_saved_var_size = 1*1*1*256;
    unsigned layer1_bn_saved_var_size_in_bytes = layer1_bn_saved_var_size * sizeof(float) ;
    status = hbmAlloc(layer1_bn_saved_var_size_in_bytes, &layer1_bn_saved_var_dram, "layer1_bn_saved_var");
    ASSERT_TRUE(status == synSuccess && "layer1_bn_saved_var dram malloc failed!");
    synLaunchTensorInfo layer1_bn_saved_var_tr_info = {"layer1_bn_saved_var", layer1_bn_saved_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_bn_saved_var = createTensor(1U, syn_type_single, layer1_bn_saved_var_sizes, true, "layer1_bn_saved_var");

    synTensor layer1_bn_out_vec[3] = {layer1_bn_output, layer1_bn_saved_mean, layer1_bn_saved_var};


    status = synNodeCreate(graphHandle, layer1_bn_in_vec, layer1_bn_out_vec, 5, 3, (void *)&layer1_bn_kernel_params, sizeof(layer1_bn_kernel_params), "batch_norm_fwd_f32", "layer1_bn", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_bn failed!");

    /*************
     * layer1_0_add_residual_fwd0 node
     * inputs: [layer1_0_bn3_output(64, 56, 56, 256)(dtype=float32), layer1_bn_output(64, 56, 56, 256)(dtype=float32)]
     * output: [layer1_0_add_residual_fwd(64, 56, 56, 256)(dtype=float32)]
     *************/

    synTensor layer1_0_add_residual_fwd0_in_vec[2] = {layer1_0_bn3_output, layer1_bn_output};


    // create layer1_0_add_residual_fwd tensor
    const unsigned layer1_0_add_residual_fwd_sizes[] = {64, 56, 56, 256};
    synTensor layer1_0_add_residual_fwd = createTensor(4U, syn_type_single, layer1_0_add_residual_fwd_sizes, false, "layer1_0_add_residual_fwd");

    synTensor layer1_0_add_residual_fwd0_out_vec[1] = {layer1_0_add_residual_fwd};


    status = synNodeCreate(graphHandle, layer1_0_add_residual_fwd0_in_vec, layer1_0_add_residual_fwd0_out_vec, 2, 1, nullptr, 0, "add_fwd_f32", "layer1_0_add_residual_fwd0", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_add_residual_fwd0 failed!");

    /*************
     * layer1_0_relu3 node
     * inputs: [layer1_0_add_residual_fwd(64, 56, 56, 256)(dtype=float32)]
     * output: [layer1_0_relu3_output(64, 56, 56, 256)(dtype=float32)]
     *************/

    synTensor layer1_0_relu3_in_vec[1] = {layer1_0_add_residual_fwd};


    // create layer1_0_relu3_output tensor
    const unsigned layer1_0_relu3_output_sizes[] = {64, 56, 56, 256};
    synTensor layer1_0_relu3_output = createTensor(4U, syn_type_single, layer1_0_relu3_output_sizes, false, "layer1_0_relu3_output");

    synTensor layer1_0_relu3_out_vec[1] = {layer1_0_relu3_output};


    status = synNodeCreate(graphHandle, layer1_0_relu3_in_vec, layer1_0_relu3_out_vec, 1, 1, nullptr, 0, "relu_fwd_f32", "layer1_0_relu3", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu3 failed!");

    /*************
     * layer1_0_relu3_output_cast node
     * inputs: [layer1_0_relu3_output(64, 56, 56, 256)(dtype=float32)]
     * output: [layer1_0_relu3_output_cast(64, 56, 56, 256)(dtype=fp8_152_t)]
     *************/

    synTensor layer1_0_relu3_output_cast_in_vec[1] = {layer1_0_relu3_output};


    // create layer1_0_relu3_output_cast tensor
    const unsigned layer1_0_relu3_output_cast_sizes[] = {64, 56, 56, 256};
    uint64_t layer1_0_relu3_output_cast_dram;
    unsigned layer1_0_relu3_output_cast_size = 64*56*56*256;
    unsigned layer1_0_relu3_output_cast_size_in_bytes = layer1_0_relu3_output_cast_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer1_0_relu3_output_cast_size_in_bytes, &layer1_0_relu3_output_cast_dram, "layer1_0_relu3_output_cast");
    ASSERT_TRUE(status == synSuccess && "layer1_0_relu3_output_cast dram malloc failed!");
    synLaunchTensorInfo layer1_0_relu3_output_cast_tr_info = {"layer1_0_relu3_output_cast", layer1_0_relu3_output_cast_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer1_0_relu3_output_cast = createTensor(4U, syn_type_fp8_152, layer1_0_relu3_output_cast_sizes, true, "layer1_0_relu3_output_cast");

    synTensor layer1_0_relu3_output_cast_out_vec[1] = {layer1_0_relu3_output_cast};


    status = synNodeCreate(graphHandle, layer1_0_relu3_output_cast_in_vec, layer1_0_relu3_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f32_to_f8", "layer1_0_relu3_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu3_output_cast failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_maxpool_output_tr_info);
    graph_inputs.push_back(layer1_0_conv1_weight_tr_info);
    graph_inputs.push_back(layer1_0_bn1_weight_tr_info);
    graph_inputs.push_back(layer1_0_bn1_bias_tr_info);
    graph_inputs.push_back(layer1_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer1_0_bn1_running_var_tr_info);
    graph_inputs.push_back(layer1_0_conv2_weight_tr_info);
    graph_inputs.push_back(layer1_0_bn2_weight_tr_info);
    graph_inputs.push_back(layer1_0_bn2_bias_tr_info);
    graph_inputs.push_back(layer1_0_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer1_0_bn2_running_var_tr_info);
    graph_inputs.push_back(layer1_0_conv3_weight_tr_info);
    graph_inputs.push_back(layer1_0_bn3_weight_tr_info);
    graph_inputs.push_back(layer1_0_bn3_bias_tr_info);
    graph_inputs.push_back(layer1_0_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer1_0_bn3_running_var_tr_info);
    graph_inputs.push_back(layer1_downsample_weight_tr_info);
    graph_inputs.push_back(layer1_bn_weight_tr_info);
    graph_inputs.push_back(layer1_bn_bias_tr_info);
    graph_inputs.push_back(layer1_bn_running_mean_tr_info);
    graph_inputs.push_back(layer1_bn_running_var_tr_info);

    graph_outputs.push_back(layer1_0_bn1_saved_mean_tr_info);
    graph_outputs.push_back(layer1_0_bn1_saved_var_tr_info);
    graph_outputs.push_back(layer1_0_bn2_saved_mean_tr_info);
    graph_outputs.push_back(layer1_0_bn2_saved_var_tr_info);
    graph_outputs.push_back(layer1_0_bn3_saved_mean_tr_info);
    graph_outputs.push_back(layer1_0_bn3_saved_var_tr_info);
    graph_outputs.push_back(layer1_bn_saved_mean_tr_info);
    graph_outputs.push_back(layer1_bn_saved_var_tr_info);
    graph_outputs.push_back(layer1_0_relu3_output_cast_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs, true /* skip validation */);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
