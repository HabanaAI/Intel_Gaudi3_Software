

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

TEST_F_GC(SynGaudi2ResNetTestEager, DISABLED_resnet50_fwd_downsample_bf16_eager_ASIC_CI)
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_downsample node
     * inputs: [layer3_5_relu3_output[64, 14, 14, 1024](dtype=bf16), layer4_downsample_weight[1, 1, 1024, 2048](dtype=bf16)]
     * output: [layer4_downsample_output(64, 7, 7, 2048)(dtype=bf16)]
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

    synTensor layer4_downsample_in_vec[4] = {layer3_5_relu3_output, layer4_downsample_weight, nullptr, nullptr};


    // create layer4_downsample_output tensor
    const unsigned layer4_downsample_output_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_downsample_output = createTensor(4U, syn_type_bf16, layer4_downsample_output_sizes, false, "layer4_downsample_output");

    synTensor layer4_downsample_out_vec[1] = {layer4_downsample_output};


    status = synNodeCreate(graphHandle, layer4_downsample_in_vec, layer4_downsample_out_vec, 4, 1, (void *)&layer4_downsample_kernel_params, sizeof(layer4_downsample_kernel_params), "spatial_convolution", "layer4_downsample", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample failed!");

    /*************
     * layer4_bn node
     * inputs: [layer4_downsample_output(64, 7, 7, 2048)(dtype=bf16), layer4_bn_bias[2048](dtype=float32), layer4_bn_weight[2048](dtype=float32), layer4_bn_running_mean[2048](dtype=float32), layer4_bn_running_var[2048](dtype=float32)]
     * output: [layer4_bn_output(64, 7, 7, 2048)(dtype=bf16), layer4_bn_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_bn_saved_var(1, 1, 1, 2048)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_bn_kernel_params;
    layer4_bn_kernel_params.momentum = 0.1;
    layer4_bn_kernel_params.threshold.f = 1e-05;
    layer4_bn_kernel_params.epsilon = 1e-05;

    // create layer4_bn_bias tensor
    synTensor              layer4_bn_bias {};
    synLaunchTensorInfo layer4_bn_bias_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_bias");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_bias dram malloc failed!");

        layer4_bn_bias_tr_info = synLaunchTensorInfo {"layer4_bn_bias", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_bias = createTensor(dims.size(),
                                      syn_type_single,
                                      dims.data(),
                                      /*is_presist*/ true,
                                      "layer4_bn_bias",
                                      /*graphHandle*/ nullptr,
                                      /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_bn_weight tensor
    synTensor              layer4_bn_weight {};
    synLaunchTensorInfo layer4_bn_weight_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_weight");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_weight dram malloc failed!");

        layer4_bn_weight_tr_info = synLaunchTensorInfo {"layer4_bn_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_weight = createTensor(dims.size(),
                                        syn_type_single,
                                        dims.data(),
                                        /*is_presist*/ true,
                                        "layer4_bn_weight",
                                        /*graphHandle*/ nullptr,
                                        /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_bn_running_mean tensor
    synTensor              layer4_bn_running_mean {};
    synLaunchTensorInfo layer4_bn_running_mean_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_running_mean");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_running_mean dram malloc failed!");

        layer4_bn_running_mean_tr_info = synLaunchTensorInfo {"layer4_bn_running_mean", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_running_mean = createTensor(dims.size(),
                                              syn_type_single,
                                              dims.data(),
                                              /*is_presist*/ true,
                                              "layer4_bn_running_mean",
                                              /*graphHandle*/ nullptr,
                                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_bn_running_var tensor
    synTensor              layer4_bn_running_var {};
    synLaunchTensorInfo layer4_bn_running_var_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_running_var");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_running_var dram malloc failed!");

        layer4_bn_running_var_tr_info = synLaunchTensorInfo {"layer4_bn_running_var", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_running_var = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer4_bn_running_var",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_bn_in_vec[5] = {layer4_downsample_output, layer4_bn_bias, layer4_bn_weight, layer4_bn_running_mean, layer4_bn_running_var};


    // create layer4_bn_output tensor
    synTensor              layer4_bn_output {};
    synLaunchTensorInfo layer4_bn_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_output");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_output dram malloc failed!");

        layer4_bn_output_tr_info = synLaunchTensorInfo {"layer4_bn_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_output = createTensor(dims.size(),
                                        syn_type_bf16,
                                        dims.data(),
                                        /*is_presist*/ true,
                                        "layer4_bn_output",
                                        /*graphHandle*/ nullptr,
                                        /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_bn_saved_mean tensor
    synTensor              layer4_bn_saved_mean {};
    synLaunchTensorInfo layer4_bn_saved_mean_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_saved_mean");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_saved_mean dram malloc failed!");

        layer4_bn_saved_mean_tr_info = synLaunchTensorInfo {"layer4_bn_saved_mean", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_saved_mean = createTensor(dims.size(),
                                            syn_type_single,
                                            dims.data(),
                                            /*is_presist*/ true,
                                            "layer4_bn_saved_mean",
                                            /*graphHandle*/ nullptr,
                                            /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_bn_saved_var tensor
    synTensor              layer4_bn_saved_var {};
    synLaunchTensorInfo layer4_bn_saved_var_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_bn_saved_var");
        ASSERT_TRUE(status == synSuccess && "layer4_bn_saved_var dram malloc failed!");

        layer4_bn_saved_var_tr_info = synLaunchTensorInfo {"layer4_bn_saved_var", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_bn_saved_var = createTensor(dims.size(),
                                           syn_type_single,
                                           dims.data(),
                                           /*is_presist*/ true,
                                           "layer4_bn_saved_var",
                                           /*graphHandle*/ nullptr,
                                           /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_bn_out_vec[3] = {layer4_bn_output, layer4_bn_saved_mean, layer4_bn_saved_var};


    status = synNodeCreate(graphHandle, layer4_bn_in_vec, layer4_bn_out_vec, 5, 3, (void *)&layer4_bn_kernel_params, sizeof(layer4_bn_kernel_params), "batch_norm_fwd_bf16", "layer4_bn", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_bn failed!");


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

    graph_outputs.push_back(layer4_bn_output_tr_info);
    graph_outputs.push_back(layer4_bn_saved_mean_tr_info);
    graph_outputs.push_back(layer4_bn_saved_var_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
