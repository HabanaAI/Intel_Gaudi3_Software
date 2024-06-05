

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

TEST_F_GC(SynGaudi2ResNetTestEager, DISABLED_resnet50_cbr_float32_eager_ASIC_CI)
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_conv1 node
     * inputs: [input[64, 224, 224, 3](dtype=float32), worker_0_conv1_weight[7, 7, 3, 64](dtype=float32)]
     * output: [worker_0_conv1_output(64, 112, 112, 64)(dtype=float32)]
     *************/
    synConvolutionParams worker_0_conv1_kernel_params;
    worker_0_conv1_kernel_params.dH = 2;
    worker_0_conv1_kernel_params.dW = 2;
    worker_0_conv1_kernel_params.kH = 7;
    worker_0_conv1_kernel_params.kW = 7;
    worker_0_conv1_kernel_params.padT = 3;
    worker_0_conv1_kernel_params.padB = 3;
    worker_0_conv1_kernel_params.padL = 3;
    worker_0_conv1_kernel_params.padR = 3;
    worker_0_conv1_kernel_params.dilH = 1;
    worker_0_conv1_kernel_params.dilW = 1;

    // create input tensor
    synTensor              input {};
    synLaunchTensorInfo input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 224, 224, 3};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "input");
        ASSERT_TRUE(status == synSuccess && "input dram malloc failed!");

        input_tr_info = synLaunchTensorInfo {"input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        input = createTensor(dims.size(),
                             syn_type_single,
                             dims.data(),
                             /*is_presist*/ true,
                             "input",
                             /*graphHandle*/ nullptr,
                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_conv1_weight tensor
    synTensor              worker_0_conv1_weight {};
    synLaunchTensorInfo worker_0_conv1_weight_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {7, 7, 3, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_conv1_weight");
        ASSERT_TRUE(status == synSuccess && "worker_0_conv1_weight dram malloc failed!");

        worker_0_conv1_weight_tr_info = synLaunchTensorInfo {"worker_0_conv1_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_conv1_weight = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "worker_0_conv1_weight",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_conv1_in_vec[4] = {input, worker_0_conv1_weight, nullptr, nullptr};


    // create worker_0_conv1_output tensor
    const unsigned worker_0_conv1_output_sizes[] = {64, 112, 112, 64};
    synTensor worker_0_conv1_output = createTensor(4U, syn_type_single, worker_0_conv1_output_sizes, false, "worker_0_conv1_output");

    synTensor worker_0_conv1_out_vec[1] = {worker_0_conv1_output};


    status = synNodeCreate(graphHandle, worker_0_conv1_in_vec, worker_0_conv1_out_vec, 4, 1, (void *)&worker_0_conv1_kernel_params, sizeof(worker_0_conv1_kernel_params), "spatial_convolution", "worker_0_conv1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_conv1 failed!");

    /*************
     * worker_0_bn1 node
     * inputs: [worker_0_conv1_output(64, 112, 112, 64)(dtype=float32), worker_0_bn1_bias[64](dtype=float32), worker_0_bn1_weight[64](dtype=float32), worker_0_bn1_running_mean[64](dtype=float32), worker_0_bn1_running_var[64](dtype=float32)]
     * output: [worker_0_bn1_output(64, 112, 112, 64)(dtype=float32), worker_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), worker_0_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params worker_0_bn1_kernel_params;
    worker_0_bn1_kernel_params.momentum = 0.1;
    worker_0_bn1_kernel_params.threshold.f = 1e-05;
    worker_0_bn1_kernel_params.epsilon = 1e-05;

    // create worker_0_bn1_bias tensor
    synTensor              worker_0_bn1_bias {};
    synLaunchTensorInfo worker_0_bn1_bias_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_bias");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_bias dram malloc failed!");

        worker_0_bn1_bias_tr_info = synLaunchTensorInfo {"worker_0_bn1_bias", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_bias = createTensor(dims.size(),
                                         syn_type_single,
                                         dims.data(),
                                         /*is_presist*/ true,
                                         "worker_0_bn1_bias",
                                         /*graphHandle*/ nullptr,
                                         /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_bn1_weight tensor
    synTensor              worker_0_bn1_weight {};
    synLaunchTensorInfo worker_0_bn1_weight_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_weight");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_weight dram malloc failed!");

        worker_0_bn1_weight_tr_info = synLaunchTensorInfo {"worker_0_bn1_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_weight = createTensor(dims.size(),
                                           syn_type_single,
                                           dims.data(),
                                           /*is_presist*/ true,
                                           "worker_0_bn1_weight",
                                           /*graphHandle*/ nullptr,
                                           /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_bn1_running_mean tensor
    synTensor              worker_0_bn1_running_mean {};
    synLaunchTensorInfo worker_0_bn1_running_mean_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_running_mean");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_running_mean dram malloc failed!");

        worker_0_bn1_running_mean_tr_info = synLaunchTensorInfo {"worker_0_bn1_running_mean", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_running_mean = createTensor(dims.size(),
                                                 syn_type_single,
                                                 dims.data(),
                                                 /*is_presist*/ true,
                                                 "worker_0_bn1_running_mean",
                                                 /*graphHandle*/ nullptr,
                                                 /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_bn1_running_var tensor
    synTensor              worker_0_bn1_running_var {};
    synLaunchTensorInfo worker_0_bn1_running_var_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_running_var");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_running_var dram malloc failed!");

        worker_0_bn1_running_var_tr_info = synLaunchTensorInfo {"worker_0_bn1_running_var", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_running_var = createTensor(dims.size(),
                                                syn_type_single,
                                                dims.data(),
                                                /*is_presist*/ true,
                                                "worker_0_bn1_running_var",
                                                /*graphHandle*/ nullptr,
                                                /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_bn1_in_vec[5] = {worker_0_conv1_output, worker_0_bn1_bias, worker_0_bn1_weight, worker_0_bn1_running_mean, worker_0_bn1_running_var};


    // create worker_0_bn1_output tensor
    const unsigned worker_0_bn1_output_sizes[] = {64, 112, 112, 64};
    synTensor worker_0_bn1_output = createTensor(4U, syn_type_single, worker_0_bn1_output_sizes, false, "worker_0_bn1_output");

    // create worker_0_bn1_saved_mean tensor
    synTensor              worker_0_bn1_saved_mean {};
    synLaunchTensorInfo worker_0_bn1_saved_mean_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_saved_mean");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_saved_mean dram malloc failed!");

        worker_0_bn1_saved_mean_tr_info = synLaunchTensorInfo {"worker_0_bn1_saved_mean", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_saved_mean = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "worker_0_bn1_saved_mean",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_bn1_saved_var tensor
    synTensor              worker_0_bn1_saved_var {};
    synLaunchTensorInfo worker_0_bn1_saved_var_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_saved_var");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_saved_var dram malloc failed!");

        worker_0_bn1_saved_var_tr_info = synLaunchTensorInfo {"worker_0_bn1_saved_var", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_saved_var = createTensor(dims.size(),
                                              syn_type_single,
                                              dims.data(),
                                              /*is_presist*/ true,
                                              "worker_0_bn1_saved_var",
                                              /*graphHandle*/ nullptr,
                                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_bn1_out_vec[3] = {worker_0_bn1_output, worker_0_bn1_saved_mean, worker_0_bn1_saved_var};


    status = synNodeCreate(graphHandle, worker_0_bn1_in_vec, worker_0_bn1_out_vec, 5, 3, (void *)&worker_0_bn1_kernel_params, sizeof(worker_0_bn1_kernel_params), "batch_norm_fwd_f32", "worker_0_bn1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_bn1 failed!");

    /*************
     * worker_0_relu node
     * inputs: [worker_0_bn1_output(64, 112, 112, 64)(dtype=float32)]
     * output: [worker_0_relu_output(64, 112, 112, 64)(dtype=float32)]
     *************/

    synTensor worker_0_relu_in_vec[1] = {worker_0_bn1_output};


    // create worker_0_relu_output tensor
    synTensor              worker_0_relu_output {};
    synLaunchTensorInfo worker_0_relu_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 112, 112, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_relu_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_relu_output dram malloc failed!");

        worker_0_relu_output_tr_info = synLaunchTensorInfo {"worker_0_relu_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_relu_output = createTensor(dims.size(),
                                            syn_type_single,
                                            dims.data(),
                                            /*is_presist*/ true,
                                            "worker_0_relu_output",
                                            /*graphHandle*/ nullptr,
                                            /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_relu_out_vec[1] = {worker_0_relu_output};


    status = synNodeCreate(graphHandle, worker_0_relu_in_vec, worker_0_relu_out_vec, 1, 1, nullptr, 0, "relu_fwd_f32", "worker_0_relu", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_relu failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(input_tr_info);
    graph_inputs.push_back(worker_0_conv1_weight_tr_info);
    graph_inputs.push_back(worker_0_bn1_weight_tr_info);
    graph_inputs.push_back(worker_0_bn1_bias_tr_info);
    graph_inputs.push_back(worker_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(worker_0_bn1_running_var_tr_info);

    graph_outputs.push_back(worker_0_bn1_saved_mean_tr_info);
    graph_outputs.push_back(worker_0_bn1_saved_var_tr_info);
    graph_outputs.push_back(worker_0_relu_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
