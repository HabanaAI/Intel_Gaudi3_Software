

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
TEST_F_GC(SynGaudi2ResNetFloat8Test, resnet50_bwd_downsample_float32_float8_ms0, {synDeviceGaudi2})
{
    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_downsample_dedx node
     * inputs: [layer4_bn_grad_input[64, 7, 7, 2048](dtype=fp8_152_t), layer4_downsample_weight[1, 1, 1024, 2048](dtype=fp8_152_t)]
     * output: [layer4_downsample_grad_input(64, 14, 14, 1024)(dtype=fp8_152_t)]
     *************/
    synConvolutionParams layer4_downsample_dedx_kernel_params;
    layer4_downsample_dedx_kernel_params.dH = 2;
    layer4_downsample_dedx_kernel_params.dW = 2;
    layer4_downsample_dedx_kernel_params.kH = 1;
    layer4_downsample_dedx_kernel_params.kW = 1;
    layer4_downsample_dedx_kernel_params.padT = 0;
    layer4_downsample_dedx_kernel_params.padB = 0;
    layer4_downsample_dedx_kernel_params.padL = 0;
    layer4_downsample_dedx_kernel_params.padR = 0;
    layer4_downsample_dedx_kernel_params.dilH = 1;
    layer4_downsample_dedx_kernel_params.dilW = 1;

    // create layer4_bn_grad_input tensor
    const unsigned layer4_bn_grad_input_sizes[] = {64, 7, 7, 2048};
    uint64_t layer4_bn_grad_input_dram;
    unsigned layer4_bn_grad_input_size = 64*7*7*2048;
    unsigned layer4_bn_grad_input_size_in_bytes = layer4_bn_grad_input_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_bn_grad_input_size_in_bytes, &layer4_bn_grad_input_dram, "layer4_bn_grad_input");
    ASSERT_TRUE(status == synSuccess && "layer4_bn_grad_input dram malloc failed!");
    synLaunchTensorInfo layer4_bn_grad_input_tr_info = {"layer4_bn_grad_input", layer4_bn_grad_input_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_bn_grad_input = createTensor(4U, syn_type_fp8_152, layer4_bn_grad_input_sizes, true, "layer4_bn_grad_input");

    // create layer4_downsample_weight tensor
    const unsigned layer4_downsample_weight_sizes[] = {1, 1, 1024, 2048};
    uint64_t layer4_downsample_weight_dram;
    unsigned layer4_downsample_weight_size = 1*1*1024*2048;
    unsigned layer4_downsample_weight_size_in_bytes = layer4_downsample_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_downsample_weight_size_in_bytes, &layer4_downsample_weight_dram, "layer4_downsample_weight");
    ASSERT_TRUE(status == synSuccess && "layer4_downsample_weight dram malloc failed!");
    synLaunchTensorInfo layer4_downsample_weight_tr_info = {"layer4_downsample_weight", layer4_downsample_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_downsample_weight = createTensor(4U, syn_type_fp8_152, layer4_downsample_weight_sizes, true, "layer4_downsample_weight");

    synTensor layer4_downsample_dedx_in_vec[2] = {layer4_bn_grad_input, layer4_downsample_weight};


    // create layer4_downsample_grad_input tensor
    const unsigned layer4_downsample_grad_input_sizes[] = {64, 14, 14, 1024};
    uint64_t layer4_downsample_grad_input_dram;
    unsigned layer4_downsample_grad_input_size = 64*14*14*1024;
    unsigned layer4_downsample_grad_input_size_in_bytes = layer4_downsample_grad_input_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer4_downsample_grad_input_size_in_bytes, &layer4_downsample_grad_input_dram, "layer4_downsample_grad_input");
    ASSERT_TRUE(status == synSuccess && "layer4_downsample_grad_input dram malloc failed!");
    synLaunchTensorInfo layer4_downsample_grad_input_tr_info = {"layer4_downsample_grad_input", layer4_downsample_grad_input_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_downsample_grad_input = createTensor(4U, syn_type_fp8_152, layer4_downsample_grad_input_sizes, true, "layer4_downsample_grad_input");

    synTensor layer4_downsample_dedx_out_vec[1] = {layer4_downsample_grad_input};


    status = synNodeCreate(graphHandle, layer4_downsample_dedx_in_vec, layer4_downsample_dedx_out_vec, 2, 1, (void *)&layer4_downsample_dedx_kernel_params, sizeof(layer4_downsample_dedx_kernel_params), "dedx", "layer4_downsample_dedx", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample_dedx failed!");

    /*************
     * layer4_downsample_dedw node
     * inputs: [layer4_bn_grad_input[64, 7, 7, 2048](dtype=fp8_152_t), layer3_5_relu3_output[64, 14, 14, 1024](dtype=fp8_152_t)]
     * output: [layer4_downsample_weight_grad(1, 1, 1024, 2048)(dtype=float32)]
     *************/
    synConvolutionParams layer4_downsample_dedw_kernel_params;
    layer4_downsample_dedw_kernel_params.dH = 2;
    layer4_downsample_dedw_kernel_params.dW = 2;
    layer4_downsample_dedw_kernel_params.kH = 1;
    layer4_downsample_dedw_kernel_params.kW = 1;
    layer4_downsample_dedw_kernel_params.padT = 0;
    layer4_downsample_dedw_kernel_params.padB = 0;
    layer4_downsample_dedw_kernel_params.padL = 0;
    layer4_downsample_dedw_kernel_params.padR = 0;
    layer4_downsample_dedw_kernel_params.dilH = 1;
    layer4_downsample_dedw_kernel_params.dilW = 1;

    // create layer3_5_relu3_output tensor
    const unsigned layer3_5_relu3_output_sizes[] = {64, 14, 14, 1024};
    uint64_t layer3_5_relu3_output_dram;
    unsigned layer3_5_relu3_output_size = 64*14*14*1024;
    unsigned layer3_5_relu3_output_size_in_bytes = layer3_5_relu3_output_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(layer3_5_relu3_output_size_in_bytes, &layer3_5_relu3_output_dram, "layer3_5_relu3_output");
    ASSERT_TRUE(status == synSuccess && "layer3_5_relu3_output dram malloc failed!");
    synLaunchTensorInfo layer3_5_relu3_output_tr_info = {"layer3_5_relu3_output", layer3_5_relu3_output_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer3_5_relu3_output = createTensor(4U, syn_type_fp8_152, layer3_5_relu3_output_sizes, true, "layer3_5_relu3_output");

    synTensor layer4_downsample_dedw_in_vec[2] = {layer4_bn_grad_input, layer3_5_relu3_output};


    // create layer4_downsample_weight_grad tensor
    const unsigned layer4_downsample_weight_grad_sizes[] = {1, 1, 1024, 2048};
    uint64_t layer4_downsample_weight_grad_dram;
    unsigned layer4_downsample_weight_grad_size = 1*1*1024*2048;
    unsigned layer4_downsample_weight_grad_size_in_bytes = layer4_downsample_weight_grad_size * sizeof(float) ;
    status = hbmAlloc(layer4_downsample_weight_grad_size_in_bytes, &layer4_downsample_weight_grad_dram, "layer4_downsample_weight_grad");
    ASSERT_TRUE(status == synSuccess && "layer4_downsample_weight_grad dram malloc failed!");
    synLaunchTensorInfo layer4_downsample_weight_grad_tr_info = {"layer4_downsample_weight_grad", layer4_downsample_weight_grad_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor layer4_downsample_weight_grad = createTensor(4U, syn_type_single, layer4_downsample_weight_grad_sizes, true, "layer4_downsample_weight_grad");

    synTensor layer4_downsample_dedw_out_vec[1] = {layer4_downsample_weight_grad};


    status = synNodeCreate(graphHandle, layer4_downsample_dedw_in_vec, layer4_downsample_dedw_out_vec, 2, 1, (void *)&layer4_downsample_dedw_kernel_params, sizeof(layer4_downsample_dedw_kernel_params), "dedw", "layer4_downsample_dedw", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample_dedw failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer4_bn_grad_input_tr_info);
    graph_inputs.push_back(layer4_downsample_weight_tr_info);
    graph_inputs.push_back(layer3_5_relu3_output_tr_info);

    graph_outputs.push_back(layer4_downsample_grad_input_tr_info);
    graph_outputs.push_back(layer4_downsample_weight_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs, true /* skip validation */);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

