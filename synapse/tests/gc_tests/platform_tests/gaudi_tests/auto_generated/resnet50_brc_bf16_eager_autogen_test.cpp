

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

TEST_F_GC(SynGaudi2ResNetTestEager, DISABLED_resnet50_brc_bf16_eager_ASIC_CI)
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer1_0_bn1 node
     * inputs: [layer1_0_conv1_output[64, 56, 56, 64](dtype=bf16), layer1_0_bn1_bias[64](dtype=float32), layer1_0_bn1_weight[64](dtype=float32), layer1_0_bn1_running_mean[64](dtype=float32), layer1_0_bn1_running_var[64](dtype=float32)]
     * output: [layer1_0_bn1_output(64, 56, 56, 64)(dtype=bf16), layer1_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn1_kernel_params;
    layer1_0_bn1_kernel_params.momentum = 0.1;
    layer1_0_bn1_kernel_params.threshold.f = 1e-05;
    layer1_0_bn1_kernel_params.epsilon = 1e-05;

    // create layer1_0_conv1_output tensor
    synTensor              layer1_0_conv1_output {};
    synLaunchTensorInfo layer1_0_conv1_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_conv1_output");
        ASSERT_TRUE(status == synSuccess && "layer1_0_conv1_output dram malloc failed!");

        layer1_0_conv1_output_tr_info = synLaunchTensorInfo {"layer1_0_conv1_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_conv1_output = createTensor(dims.size(),
                                             syn_type_bf16,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer1_0_conv1_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer1_0_bn1_bias tensor
    synTensor              layer1_0_bn1_bias {};
    synLaunchTensorInfo layer1_0_bn1_bias_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_bn1_bias");
        ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_bias dram malloc failed!");

        layer1_0_bn1_bias_tr_info = synLaunchTensorInfo {"layer1_0_bn1_bias", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_bn1_bias = createTensor(dims.size(),
                                         syn_type_single,
                                         dims.data(),
                                         /*is_presist*/ true,
                                         "layer1_0_bn1_bias",
                                         /*graphHandle*/ nullptr,
                                         /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer1_0_bn1_weight tensor
    synTensor              layer1_0_bn1_weight {};
    synLaunchTensorInfo layer1_0_bn1_weight_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_bn1_weight");
        ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_weight dram malloc failed!");

        layer1_0_bn1_weight_tr_info = synLaunchTensorInfo {"layer1_0_bn1_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_bn1_weight = createTensor(dims.size(),
                                           syn_type_single,
                                           dims.data(),
                                           /*is_presist*/ true,
                                           "layer1_0_bn1_weight",
                                           /*graphHandle*/ nullptr,
                                           /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer1_0_bn1_running_mean tensor
    synTensor              layer1_0_bn1_running_mean {};
    synLaunchTensorInfo layer1_0_bn1_running_mean_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_bn1_running_mean");
        ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_running_mean dram malloc failed!");

        layer1_0_bn1_running_mean_tr_info = synLaunchTensorInfo {"layer1_0_bn1_running_mean", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_bn1_running_mean = createTensor(dims.size(),
                                                 syn_type_single,
                                                 dims.data(),
                                                 /*is_presist*/ true,
                                                 "layer1_0_bn1_running_mean",
                                                 /*graphHandle*/ nullptr,
                                                 /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer1_0_bn1_running_var tensor
    synTensor              layer1_0_bn1_running_var {};
    synLaunchTensorInfo layer1_0_bn1_running_var_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_bn1_running_var");
        ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_running_var dram malloc failed!");

        layer1_0_bn1_running_var_tr_info = synLaunchTensorInfo {"layer1_0_bn1_running_var", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_bn1_running_var = createTensor(dims.size(),
                                                syn_type_single,
                                                dims.data(),
                                                /*is_presist*/ true,
                                                "layer1_0_bn1_running_var",
                                                /*graphHandle*/ nullptr,
                                                /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer1_0_bn1_in_vec[5] = {layer1_0_conv1_output, layer1_0_bn1_bias, layer1_0_bn1_weight, layer1_0_bn1_running_mean, layer1_0_bn1_running_var};


    // create layer1_0_bn1_output tensor
    const unsigned layer1_0_bn1_output_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_bn1_output = createTensor(4U, syn_type_bf16, layer1_0_bn1_output_sizes, false, "layer1_0_bn1_output");

    // create layer1_0_bn1_saved_mean tensor
    synTensor              layer1_0_bn1_saved_mean {};
    synLaunchTensorInfo layer1_0_bn1_saved_mean_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_bn1_saved_mean");
        ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_saved_mean dram malloc failed!");

        layer1_0_bn1_saved_mean_tr_info = synLaunchTensorInfo {"layer1_0_bn1_saved_mean", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_bn1_saved_mean = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "layer1_0_bn1_saved_mean",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer1_0_bn1_saved_var tensor
    synTensor              layer1_0_bn1_saved_var {};
    synLaunchTensorInfo layer1_0_bn1_saved_var_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_bn1_saved_var");
        ASSERT_TRUE(status == synSuccess && "layer1_0_bn1_saved_var dram malloc failed!");

        layer1_0_bn1_saved_var_tr_info = synLaunchTensorInfo {"layer1_0_bn1_saved_var", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_bn1_saved_var = createTensor(dims.size(),
                                              syn_type_single,
                                              dims.data(),
                                              /*is_presist*/ true,
                                              "layer1_0_bn1_saved_var",
                                              /*graphHandle*/ nullptr,
                                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer1_0_bn1_out_vec[3] = {layer1_0_bn1_output, layer1_0_bn1_saved_mean, layer1_0_bn1_saved_var};


    status = synNodeCreate(graphHandle, layer1_0_bn1_in_vec, layer1_0_bn1_out_vec, 5, 3, (void *)&layer1_0_bn1_kernel_params, sizeof(layer1_0_bn1_kernel_params), "batch_norm_fwd_bf16", "layer1_0_bn1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn1 failed!");

    /*************
     * layer1_0_relu1 node
     * inputs: [layer1_0_bn1_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_0_relu1_in_vec[1] = {layer1_0_bn1_output};


    // create layer1_0_relu1_output tensor
    const unsigned layer1_0_relu1_output_sizes[] = {64, 56, 56, 64};
    synTensor layer1_0_relu1_output = createTensor(4U, syn_type_bf16, layer1_0_relu1_output_sizes, false, "layer1_0_relu1_output");

    synTensor layer1_0_relu1_out_vec[1] = {layer1_0_relu1_output};


    status = synNodeCreate(graphHandle, layer1_0_relu1_in_vec, layer1_0_relu1_out_vec, 1, 1, nullptr, 0, "relu_fwd_bf16", "layer1_0_relu1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu1 failed!");

    /*************
     * layer1_0_conv2 node
     * inputs: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=bf16), layer1_0_conv2_weight[3, 3, 64, 64](dtype=bf16)]
     * output: [layer1_0_conv2_output(64, 56, 56, 64)(dtype=bf16)]
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
    synTensor              layer1_0_conv2_weight {};
    synLaunchTensorInfo layer1_0_conv2_weight_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {3, 3, 64, 64};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_conv2_weight");
        ASSERT_TRUE(status == synSuccess && "layer1_0_conv2_weight dram malloc failed!");

        layer1_0_conv2_weight_tr_info = synLaunchTensorInfo {"layer1_0_conv2_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_conv2_weight = createTensor(dims.size(),
                                             syn_type_bf16,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer1_0_conv2_weight",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer1_0_conv2_in_vec[4] = {layer1_0_relu1_output, layer1_0_conv2_weight, nullptr, nullptr};


    // create layer1_0_conv2_output tensor
    synTensor              layer1_0_conv2_output {};
    synLaunchTensorInfo layer1_0_conv2_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(bfloat16);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_conv2_output");
        ASSERT_TRUE(status == synSuccess && "layer1_0_conv2_output dram malloc failed!");

        layer1_0_conv2_output_tr_info = synLaunchTensorInfo {"layer1_0_conv2_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_conv2_output = createTensor(dims.size(),
                                             syn_type_bf16,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer1_0_conv2_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

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
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
