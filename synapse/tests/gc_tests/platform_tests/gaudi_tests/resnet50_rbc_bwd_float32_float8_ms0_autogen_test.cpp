

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
TEST_F_GC(SynGaudi2ResNetFloat8Test, resnet50_rbc_bwd_float32_float8_ms0, {synDeviceGaudi2})
{
    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_avgpool_grad_input_cast node
     * inputs: [worker_0_avgpool_grad_input[64, 7, 7, 2048](dtype=fp8_152_t)]
     * output: [worker_0_avgpool_grad_input_cast[64, 7, 7, 2048](dtype=float32)]
     *************/

    // create worker_0_avgpool_grad_input tensor
    const unsigned worker_0_avgpool_grad_input_sizes[] = {64, 7, 7, 2048};
    uint64_t worker_0_avgpool_grad_input_dram;
    unsigned worker_0_avgpool_grad_input_size = 64*7*7*2048;
    unsigned worker_0_avgpool_grad_input_size_in_bytes = worker_0_avgpool_grad_input_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(worker_0_avgpool_grad_input_size_in_bytes, &worker_0_avgpool_grad_input_dram, "worker_0_avgpool_grad_input");
    ASSERT_TRUE(status == synSuccess && "worker_0_avgpool_grad_input dram malloc failed!");
    synLaunchTensorInfo worker_0_avgpool_grad_input_tr_info = {"worker_0_avgpool_grad_input", worker_0_avgpool_grad_input_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_avgpool_grad_input = createTensor(4U, syn_type_fp8_152, worker_0_avgpool_grad_input_sizes, true, "worker_0_avgpool_grad_input");

    synTensor worker_0_avgpool_grad_input_cast_in_vec[1] = {worker_0_avgpool_grad_input};


    // create worker_0_avgpool_grad_input_cast tensor
    const unsigned worker_0_avgpool_grad_input_cast_sizes[] = {64, 7, 7, 2048};
    synTensor worker_0_avgpool_grad_input_cast = createTensor(4U, syn_type_single, worker_0_avgpool_grad_input_cast_sizes, false, "worker_0_avgpool_grad_input_cast");

    synTensor worker_0_avgpool_grad_input_cast_out_vec[1] = {worker_0_avgpool_grad_input_cast};


    status = synNodeCreate(graphHandle, worker_0_avgpool_grad_input_cast_in_vec, worker_0_avgpool_grad_input_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "worker_0_avgpool_grad_input_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_avgpool_grad_input_cast failed!");

    /*************
     * layer4_2_relu3_bwd node
     * inputs: [worker_0_avgpool_grad_input_cast[64, 7, 7, 2048](dtype=float32), layer4_2_relu3_output[64, 7, 7, 2048](dtype=float32)]
     * output: [layer4_2_relu3_grad_input(64, 7, 7, 2048)(dtype=float32)]
     *************/

    // create layer4_2_relu3_output tensor
    const unsigned layer4_2_relu3_output_sizes[] = {64, 7, 7, 2048};
    uint64_t layer4_2_relu3_output_dram;
    unsigned layer4_2_relu3_output_size = 64*7*7*2048;
    unsigned layer4_2_relu3_output_size_in_bytes = layer4_2_relu3_output_size * sizeof(float32) ;
    status = hbmAlloc(layer4_2_relu3_output_size_in_bytes, &layer4_2_relu3_output_dram, "layer4_2_relu3_output");
    ASSERT_TRUE(status == synSuccess && "layer4_2_relu3_output dram malloc failed!");
    synLaunchTensorInfo layer4_2_relu3_output_tr_info = {"layer4_2_relu3_output", layer4_2_relu3_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_relu3_output = createTensor(4U, syn_type_single, layer4_2_relu3_output_sizes, true, "layer4_2_relu3_output");

    synTensor layer4_2_relu3_bwd_in_vec[2] = {worker_0_avgpool_grad_input_cast, layer4_2_relu3_output};


    // create layer4_2_relu3_grad_input tensor
    const unsigned layer4_2_relu3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_2_relu3_grad_input = createTensor(4U, syn_type_single, layer4_2_relu3_grad_input_sizes, false, "layer4_2_relu3_grad_input");

    synTensor layer4_2_relu3_bwd_out_vec[1] = {layer4_2_relu3_grad_input};


    status = synNodeCreate(graphHandle, layer4_2_relu3_bwd_in_vec, layer4_2_relu3_bwd_out_vec, 2, 1, nullptr, 0, "relu_bwd_f32", "layer4_2_relu3_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu3_bwd failed!");

    /*************
     * layer4_2_add_residual_bwd node
     * inputs: [layer4_2_relu3_grad_input(64, 7, 7, 2048)(dtype=float32)]
     * output: [layer4_2_add_residual_grad_input0(64, 7, 7, 2048)(dtype=float32), layer4_2_add_residual_grad_input1(64, 7, 7, 2048)(dtype=float32)]
     *************/

    synTensor layer4_2_add_residual_bwd_in_vec[1] = {layer4_2_relu3_grad_input};


    // create layer4_2_add_residual_grad_input0 tensor
    const unsigned layer4_2_add_residual_grad_input0_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_2_add_residual_grad_input0 = createTensor(4U, syn_type_single, layer4_2_add_residual_grad_input0_sizes, false, "layer4_2_add_residual_grad_input0");

    // create layer4_2_add_residual_grad_input1 tensor
    const unsigned layer4_2_add_residual_grad_input1_sizes[] = {64, 7, 7, 2048};
    uint64_t layer4_2_add_residual_grad_input1_dram;
    unsigned layer4_2_add_residual_grad_input1_size = 64*7*7*2048;
    unsigned layer4_2_add_residual_grad_input1_size_in_bytes = layer4_2_add_residual_grad_input1_size * sizeof(float) ;
    status = hbmAlloc(layer4_2_add_residual_grad_input1_size_in_bytes, &layer4_2_add_residual_grad_input1_dram, "layer4_2_add_residual_grad_input1");
    ASSERT_TRUE(status == synSuccess && "layer4_2_add_residual_grad_input1 dram malloc failed!");
    synLaunchTensorInfo layer4_2_add_residual_grad_input1_tr_info = {"layer4_2_add_residual_grad_input1", layer4_2_add_residual_grad_input1_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_add_residual_grad_input1 = createTensor(4U, syn_type_single, layer4_2_add_residual_grad_input1_sizes, true, "layer4_2_add_residual_grad_input1");

    synTensor layer4_2_add_residual_bwd_out_vec[2] = {layer4_2_add_residual_grad_input0, layer4_2_add_residual_grad_input1};


    status = synNodeCreate(graphHandle, layer4_2_add_residual_bwd_in_vec, layer4_2_add_residual_bwd_out_vec, 1, 2, nullptr, 0, "add_bwd_f32", "layer4_2_add_residual_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_add_residual_bwd failed!");

    /*************
     * layer4_2_conv3_output_cast node
     * inputs: [layer4_2_conv3_output[64, 7, 7, 2048](dtype=fp8_152_t)]
     * output: [layer4_2_conv3_output_cast[64, 7, 7, 2048](dtype=float32)]
     *************/

    // create layer4_2_conv3_output tensor
    const unsigned layer4_2_conv3_output_sizes[] = {64, 7, 7, 2048};
    uint64_t layer4_2_conv3_output_dram;
    unsigned layer4_2_conv3_output_size = 64*7*7*2048;
    unsigned layer4_2_conv3_output_size_in_bytes = layer4_2_conv3_output_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_2_conv3_output_size_in_bytes, &layer4_2_conv3_output_dram, "layer4_2_conv3_output");
    ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_output dram malloc failed!");
    synLaunchTensorInfo layer4_2_conv3_output_tr_info = {"layer4_2_conv3_output", layer4_2_conv3_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_conv3_output = createTensor(4U, syn_type_fp8_152, layer4_2_conv3_output_sizes, true, "layer4_2_conv3_output");

    synTensor layer4_2_conv3_output_cast_in_vec[1] = {layer4_2_conv3_output};


    // create layer4_2_conv3_output_cast tensor
    const unsigned layer4_2_conv3_output_cast_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_2_conv3_output_cast = createTensor(4U, syn_type_single, layer4_2_conv3_output_cast_sizes, false, "layer4_2_conv3_output_cast");

    synTensor layer4_2_conv3_output_cast_out_vec[1] = {layer4_2_conv3_output_cast};


    status = synNodeCreate(graphHandle, layer4_2_conv3_output_cast_in_vec, layer4_2_conv3_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "layer4_2_conv3_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv3_output_cast failed!");

    /*************
     * layer4_2_bn3_bwd node
     * inputs: [layer4_2_conv3_output_cast[64, 7, 7, 2048](dtype=float32), layer4_2_add_residual_grad_input0(64, 7, 7, 2048)(dtype=float32), layer4_2_bn3_saved_mean[1, 1, 1, 2048](dtype=float32), layer4_2_bn3_saved_var[1, 1, 1, 2048](dtype=float32), layer4_2_bn3_weight[2048](dtype=float32)]
     * output: [layer4_2_bn3_grad_input(64, 7, 7, 2048)(dtype=float32), layer4_2_bn3_bias_grad(2048,)(dtype=float32), layer4_2_bn3_weight_grad(2048,)(dtype=float32)]
     *************/

    // create layer4_2_bn3_saved_mean tensor
    const unsigned layer4_2_bn3_saved_mean_sizes[] = {2048};
    uint64_t layer4_2_bn3_saved_mean_dram;
    unsigned layer4_2_bn3_saved_mean_size = 1*1*1*2048;
    unsigned layer4_2_bn3_saved_mean_size_in_bytes = layer4_2_bn3_saved_mean_size * sizeof(float) ;
    status = hbmAlloc(layer4_2_bn3_saved_mean_size_in_bytes, &layer4_2_bn3_saved_mean_dram, "layer4_2_bn3_saved_mean");
    ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_saved_mean dram malloc failed!");
    synLaunchTensorInfo layer4_2_bn3_saved_mean_tr_info = {"layer4_2_bn3_saved_mean", layer4_2_bn3_saved_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_bn3_saved_mean = createTensor(1U, syn_type_single, layer4_2_bn3_saved_mean_sizes, true, "layer4_2_bn3_saved_mean");

    // create layer4_2_bn3_saved_var tensor
    const unsigned layer4_2_bn3_saved_var_sizes[] = {2048};
    uint64_t layer4_2_bn3_saved_var_dram;
    unsigned layer4_2_bn3_saved_var_size = 1*1*1*2048;
    unsigned layer4_2_bn3_saved_var_size_in_bytes = layer4_2_bn3_saved_var_size * sizeof(float) ;
    status = hbmAlloc(layer4_2_bn3_saved_var_size_in_bytes, &layer4_2_bn3_saved_var_dram, "layer4_2_bn3_saved_var");
    ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_saved_var dram malloc failed!");
    synLaunchTensorInfo layer4_2_bn3_saved_var_tr_info = {"layer4_2_bn3_saved_var", layer4_2_bn3_saved_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_bn3_saved_var = createTensor(1U, syn_type_single, layer4_2_bn3_saved_var_sizes, true, "layer4_2_bn3_saved_var");

    // create layer4_2_bn3_weight tensor
    const unsigned layer4_2_bn3_weight_sizes[] = {2048,};
    uint64_t layer4_2_bn3_weight_dram;
    unsigned layer4_2_bn3_weight_size = 2048;
    unsigned layer4_2_bn3_weight_size_in_bytes = layer4_2_bn3_weight_size * sizeof(float) ;
    status = hbmAlloc(layer4_2_bn3_weight_size_in_bytes, &layer4_2_bn3_weight_dram, "layer4_2_bn3_weight");
    ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_weight dram malloc failed!");
    synLaunchTensorInfo layer4_2_bn3_weight_tr_info = {"layer4_2_bn3_weight", layer4_2_bn3_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_bn3_weight = createTensor(1U, syn_type_single, layer4_2_bn3_weight_sizes, true, "layer4_2_bn3_weight");

    synTensor layer4_2_bn3_bwd_in_vec[5] = {layer4_2_conv3_output_cast, layer4_2_add_residual_grad_input0, layer4_2_bn3_saved_mean, layer4_2_bn3_saved_var, layer4_2_bn3_weight};


    // create layer4_2_bn3_grad_input tensor
    const unsigned layer4_2_bn3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_2_bn3_grad_input = createTensor(4U, syn_type_single, layer4_2_bn3_grad_input_sizes, false, "layer4_2_bn3_grad_input");

    // create layer4_2_bn3_bias_grad tensor
    const unsigned layer4_2_bn3_bias_grad_sizes[] = {2048,};
    uint64_t layer4_2_bn3_bias_grad_dram;
    unsigned layer4_2_bn3_bias_grad_size = 2048;
    unsigned layer4_2_bn3_bias_grad_size_in_bytes = layer4_2_bn3_bias_grad_size * sizeof(float) ;
    status = hbmAlloc(layer4_2_bn3_bias_grad_size_in_bytes, &layer4_2_bn3_bias_grad_dram, "layer4_2_bn3_bias_grad");
    ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_bias_grad dram malloc failed!");
    synLaunchTensorInfo layer4_2_bn3_bias_grad_tr_info = {"layer4_2_bn3_bias_grad", layer4_2_bn3_bias_grad_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_bn3_bias_grad = createTensor(1U, syn_type_single, layer4_2_bn3_bias_grad_sizes, true, "layer4_2_bn3_bias_grad");

    // create layer4_2_bn3_weight_grad tensor
    const unsigned layer4_2_bn3_weight_grad_sizes[] = {2048,};
    uint64_t layer4_2_bn3_weight_grad_dram;
    unsigned layer4_2_bn3_weight_grad_size = 2048;
    unsigned layer4_2_bn3_weight_grad_size_in_bytes = layer4_2_bn3_weight_grad_size * sizeof(float) ;
    status = hbmAlloc(layer4_2_bn3_weight_grad_size_in_bytes, &layer4_2_bn3_weight_grad_dram, "layer4_2_bn3_weight_grad");
    ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_weight_grad dram malloc failed!");
    synLaunchTensorInfo layer4_2_bn3_weight_grad_tr_info = {"layer4_2_bn3_weight_grad", layer4_2_bn3_weight_grad_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_bn3_weight_grad = createTensor(1U, syn_type_single, layer4_2_bn3_weight_grad_sizes, true, "layer4_2_bn3_weight_grad");

    synTensor layer4_2_bn3_bwd_out_vec[3] = {layer4_2_bn3_grad_input, layer4_2_bn3_bias_grad, layer4_2_bn3_weight_grad};


    status = synNodeCreate(graphHandle, layer4_2_bn3_bwd_in_vec, layer4_2_bn3_bwd_out_vec, 5, 3, nullptr, 0, "batch_norm_bwd_f32", "layer4_2_bn3_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn3_bwd failed!");

    /*************
     * layer4_2_bn3_grad_input_cast node
     * inputs: [layer4_2_bn3_grad_input(64, 7, 7, 2048)(dtype=float32)]
     * output: [layer4_2_bn3_grad_input_cast[64, 7, 7, 2048](dtype=fp8_152_t)]
     *************/

    synTensor layer4_2_bn3_grad_input_cast_in_vec[1] = {layer4_2_bn3_grad_input};


    // create layer4_2_bn3_grad_input_cast tensor
    const unsigned layer4_2_bn3_grad_input_cast_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_2_bn3_grad_input_cast = createTensor(4U, syn_type_fp8_152, layer4_2_bn3_grad_input_cast_sizes, false, "layer4_2_bn3_grad_input_cast");

    synTensor layer4_2_bn3_grad_input_cast_out_vec[1] = {layer4_2_bn3_grad_input_cast};


    status = synNodeCreate(graphHandle, layer4_2_bn3_grad_input_cast_in_vec, layer4_2_bn3_grad_input_cast_out_vec, 1, 1, nullptr, 0, "cast_f32_to_f8", "layer4_2_bn3_grad_input_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn3_grad_input_cast failed!");

    /*************
     * layer4_2_conv3_dedw node
     * inputs: [layer4_2_bn3_grad_input_cast[64, 7, 7, 2048](dtype=fp8_152_t), layer4_2_relu2_output[64, 7, 7, 512](dtype=fp8_152_t)]
     * output: [layer4_2_conv3_weight_grad(1, 1, 512, 2048)(dtype=float32)]
     *************/
    synConvolutionParams layer4_2_conv3_dedw_kernel_params;
    layer4_2_conv3_dedw_kernel_params.dH = 1;
    layer4_2_conv3_dedw_kernel_params.dW = 1;
    layer4_2_conv3_dedw_kernel_params.kH = 1;
    layer4_2_conv3_dedw_kernel_params.kW = 1;
    layer4_2_conv3_dedw_kernel_params.padT = 0;
    layer4_2_conv3_dedw_kernel_params.padB = 0;
    layer4_2_conv3_dedw_kernel_params.padL = 0;
    layer4_2_conv3_dedw_kernel_params.padR = 0;
    layer4_2_conv3_dedw_kernel_params.dilH = 1;
    layer4_2_conv3_dedw_kernel_params.dilW = 1;

    // create layer4_2_relu2_output tensor
    const unsigned layer4_2_relu2_output_sizes[] = {64, 7, 7, 512};
    uint64_t layer4_2_relu2_output_dram;
    unsigned layer4_2_relu2_output_size = 64*7*7*512;
    unsigned layer4_2_relu2_output_size_in_bytes = layer4_2_relu2_output_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_2_relu2_output_size_in_bytes, &layer4_2_relu2_output_dram, "layer4_2_relu2_output");
    ASSERT_TRUE(status == synSuccess && "layer4_2_relu2_output dram malloc failed!");
    synLaunchTensorInfo layer4_2_relu2_output_tr_info = {"layer4_2_relu2_output", layer4_2_relu2_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_relu2_output = createTensor(4U, syn_type_fp8_152, layer4_2_relu2_output_sizes, true, "layer4_2_relu2_output");

    synTensor layer4_2_conv3_dedw_in_vec[2] = {layer4_2_bn3_grad_input_cast, layer4_2_relu2_output};


    // create layer4_2_conv3_weight_grad tensor
    const unsigned layer4_2_conv3_weight_grad_sizes[] = {1, 1, 512, 2048};
    uint64_t layer4_2_conv3_weight_grad_dram;
    unsigned layer4_2_conv3_weight_grad_size = 1*1*512*2048;
    unsigned layer4_2_conv3_weight_grad_size_in_bytes = layer4_2_conv3_weight_grad_size * sizeof(float) ;
    status = hbmAlloc(layer4_2_conv3_weight_grad_size_in_bytes, &layer4_2_conv3_weight_grad_dram, "layer4_2_conv3_weight_grad");
    ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_weight_grad dram malloc failed!");
    synLaunchTensorInfo layer4_2_conv3_weight_grad_tr_info = {"layer4_2_conv3_weight_grad", layer4_2_conv3_weight_grad_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_conv3_weight_grad = createTensor(4U, syn_type_single, layer4_2_conv3_weight_grad_sizes, true, "layer4_2_conv3_weight_grad");

    synTensor layer4_2_conv3_dedw_out_vec[1] = {layer4_2_conv3_weight_grad};


    status = synNodeCreate(graphHandle, layer4_2_conv3_dedw_in_vec, layer4_2_conv3_dedw_out_vec, 2, 1, (void *)&layer4_2_conv3_dedw_kernel_params, sizeof(layer4_2_conv3_dedw_kernel_params), "dedw", "layer4_2_conv3_dedw", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv3_dedw failed!");

    /*************
     * layer4_2_conv3_dedx node
     * inputs: [layer4_2_bn3_grad_input_cast[64, 7, 7, 2048](dtype=fp8_152_t), layer4_2_conv3_weight[1, 1, 512, 2048](dtype=fp8_152_t)]
     * output: [layer4_2_conv3_grad_input(64, 7, 7, 512)(dtype=fp8_152_t)]
     *************/
    synConvolutionParams layer4_2_conv3_dedx_kernel_params;
    layer4_2_conv3_dedx_kernel_params.dH = 1;
    layer4_2_conv3_dedx_kernel_params.dW = 1;
    layer4_2_conv3_dedx_kernel_params.kH = 1;
    layer4_2_conv3_dedx_kernel_params.kW = 1;
    layer4_2_conv3_dedx_kernel_params.padT = 0;
    layer4_2_conv3_dedx_kernel_params.padB = 0;
    layer4_2_conv3_dedx_kernel_params.padL = 0;
    layer4_2_conv3_dedx_kernel_params.padR = 0;
    layer4_2_conv3_dedx_kernel_params.dilH = 1;
    layer4_2_conv3_dedx_kernel_params.dilW = 1;

    // create layer4_2_conv3_weight tensor
    const unsigned layer4_2_conv3_weight_sizes[] = {1, 1, 512, 2048};
    uint64_t layer4_2_conv3_weight_dram;
    unsigned layer4_2_conv3_weight_size = 1*1*512*2048;
    unsigned layer4_2_conv3_weight_size_in_bytes = layer4_2_conv3_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_2_conv3_weight_size_in_bytes, &layer4_2_conv3_weight_dram, "layer4_2_conv3_weight");
    ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_weight dram malloc failed!");
    synLaunchTensorInfo layer4_2_conv3_weight_tr_info = {"layer4_2_conv3_weight", layer4_2_conv3_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_conv3_weight = createTensor(4U, syn_type_fp8_152, layer4_2_conv3_weight_sizes, true, "layer4_2_conv3_weight");

    synTensor layer4_2_conv3_dedx_in_vec[2] = {layer4_2_bn3_grad_input_cast, layer4_2_conv3_weight};


    // create layer4_2_conv3_grad_input tensor
    const unsigned layer4_2_conv3_grad_input_sizes[] = {64, 7, 7, 512};
    uint64_t layer4_2_conv3_grad_input_dram;
    unsigned layer4_2_conv3_grad_input_size = 64*7*7*512;
    unsigned layer4_2_conv3_grad_input_size_in_bytes = layer4_2_conv3_grad_input_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_2_conv3_grad_input_size_in_bytes, &layer4_2_conv3_grad_input_dram, "layer4_2_conv3_grad_input");
    ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_grad_input dram malloc failed!");
    synLaunchTensorInfo layer4_2_conv3_grad_input_tr_info = {"layer4_2_conv3_grad_input", layer4_2_conv3_grad_input_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_2_conv3_grad_input = createTensor(4U, syn_type_fp8_152, layer4_2_conv3_grad_input_sizes, true, "layer4_2_conv3_grad_input");

    synTensor layer4_2_conv3_dedx_out_vec[1] = {layer4_2_conv3_grad_input};


    status = synNodeCreate(graphHandle, layer4_2_conv3_dedx_in_vec, layer4_2_conv3_dedx_out_vec, 2, 1, (void *)&layer4_2_conv3_dedx_kernel_params, sizeof(layer4_2_conv3_dedx_kernel_params), "dedx", "layer4_2_conv3_dedx", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv3_dedx failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_avgpool_grad_input_tr_info);
    graph_inputs.push_back(layer4_2_relu3_output_tr_info);
    graph_inputs.push_back(layer4_2_conv3_output_tr_info);
    graph_inputs.push_back(layer4_2_bn3_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn3_saved_mean_tr_info);
    graph_inputs.push_back(layer4_2_bn3_saved_var_tr_info);
    graph_inputs.push_back(layer4_2_relu2_output_tr_info);
    graph_inputs.push_back(layer4_2_conv3_weight_tr_info);

    graph_outputs.push_back(layer4_2_add_residual_grad_input1_tr_info);
    graph_outputs.push_back(layer4_2_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer4_2_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_conv3_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs, true /* skip validation */);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
