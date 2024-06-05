

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
#include "../gc_resnet_demo_test.h"
#include <cstdint>

TEST_F_GC(SynGaudi2ResNetTestEager, DISABLED_resnet50_bwd_downsample_bf16_eager_ASIC_CI)
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_downsample_dedx node
     * inputs: [layer4_bn_grad_input[64, 7, 7, 2048](dtype=bf16), layer4_downsample_weight[1, 1, 1024, 2048](dtype=bf16)]
     * output: [layer4_downsample_grad_input(64, 14, 14, 1024)(dtype=bf16)]
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
    synTensor              layer4_bn_grad_input {};
    synLaunchTensorInfo layer4_bn_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_grad_input dram malloc failed!");

        layer4_bn_grad_input_tr_info = synLaunchTensorInfo {"layer4_bn_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_grad_input = createTensor(dims.size(),
                                            syn_type_bf16,
                                            dims.data(),
                                            /*is_presist*/ true,
                                            "layer4_bn_grad_input",
                                            /*graphHandle*/ nullptr,
                                            /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_downsample_weight tensor
    synTensor              layer4_downsample_weight {};
    synLaunchTensorInfo layer4_downsample_weight_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 1024, 2048};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_downsample_weight");
        ASSERT_TRUE(status == synSuccess && "layer4_downsample_weight dram malloc failed!");

        layer4_downsample_weight_tr_info = synLaunchTensorInfo {"layer4_downsample_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_downsample_weight = createTensor(dims.size(),
                                                syn_type_bf16,
                                                dims.data(),
                                                /*is_presist*/ true,
                                                "layer4_downsample_weight",
                                                /*graphHandle*/ nullptr,
                                                /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_downsample_dedx_in_vec[2] = {layer4_bn_grad_input, layer4_downsample_weight};


    // create layer4_downsample_grad_input tensor
    synTensor              layer4_downsample_grad_input {};
    synLaunchTensorInfo layer4_downsample_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 14, 14, 1024};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_downsample_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_downsample_grad_input dram malloc failed!");

        layer4_downsample_grad_input_tr_info = synLaunchTensorInfo {"layer4_downsample_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_downsample_grad_input = createTensor(dims.size(),
                                                    syn_type_bf16,
                                                    dims.data(),
                                                    /*is_presist*/ true,
                                                    "layer4_downsample_grad_input",
                                                    /*graphHandle*/ nullptr,
                                                    /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_downsample_dedx_out_vec[1] = {layer4_downsample_grad_input};


    status = synNodeCreate(graphHandle, layer4_downsample_dedx_in_vec, layer4_downsample_dedx_out_vec, 2, 1, (void *)&layer4_downsample_dedx_kernel_params, sizeof(layer4_downsample_dedx_kernel_params), "dedx", "layer4_downsample_dedx", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample_dedx failed!");

    /*************
     * layer4_downsample_dedw node
     * inputs: [layer4_bn_grad_input[64, 7, 7, 2048](dtype=bf16), layer3_5_relu3_output[64, 14, 14, 1024](dtype=bf16)]
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
    synTensor              layer3_5_relu3_output {};
    synLaunchTensorInfo layer3_5_relu3_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 14, 14, 1024};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer3_5_relu3_output");
        ASSERT_TRUE(status == synSuccess && "layer3_5_relu3_output dram malloc failed!");

        layer3_5_relu3_output_tr_info = synLaunchTensorInfo {"layer3_5_relu3_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer3_5_relu3_output = createTensor(dims.size(),
                                             syn_type_bf16,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer3_5_relu3_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_downsample_dedw_in_vec[2] = {layer4_bn_grad_input, layer3_5_relu3_output};


    // create layer4_downsample_weight_grad tensor
    synTensor              layer4_downsample_weight_grad {};
    synLaunchTensorInfo layer4_downsample_weight_grad_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 1024, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_downsample_weight_grad");
        ASSERT_TRUE(status == synSuccess && "layer4_downsample_weight_grad dram malloc failed!");

        layer4_downsample_weight_grad_tr_info = synLaunchTensorInfo {"layer4_downsample_weight_grad", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_downsample_weight_grad = createTensor(dims.size(),
                                                     syn_type_single,
                                                     dims.data(),
                                                     /*is_presist*/ true,
                                                     "layer4_downsample_weight_grad",
                                                     /*graphHandle*/ nullptr,
                                                     /*deviceAddr*/ eagerMode ? addr : -1);
    }

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
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
