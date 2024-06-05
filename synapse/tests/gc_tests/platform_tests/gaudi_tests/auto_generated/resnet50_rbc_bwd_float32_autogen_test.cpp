

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
#include "scoped_configuration_change.h"

TEST_F_GC(SynTrainingResNetTest, resnet50_rbc_bwd_float32_ASIC_CI)
{
    const bool eagerMode = false;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_2_relu3_bwd node
     * inputs: [worker_0_avgpool_grad_input[64, 7, 7, 2048](dtype=float32), layer4_2_relu3_output[64, 7, 7, 2048](dtype=float32)]
     * output: [layer4_2_relu3_grad_input(64, 7, 7, 2048)(dtype=float32)]
     *************/

    // create worker_0_avgpool_grad_input tensor
    synTensor              worker_0_avgpool_grad_input {};
    synLaunchTensorInfo worker_0_avgpool_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_avgpool_grad_input");
        ASSERT_TRUE(status == synSuccess && "worker_0_avgpool_grad_input dram malloc failed!");

        worker_0_avgpool_grad_input_tr_info = synLaunchTensorInfo {"worker_0_avgpool_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_avgpool_grad_input = createTensor(dims.size(),
                                                   syn_type_single,
                                                   dims.data(),
                                                   /*is_presist*/ true,
                                                   "worker_0_avgpool_grad_input",
                                                   /*graphHandle*/ nullptr,
                                                   /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_2_relu3_output tensor
    synTensor              layer4_2_relu3_output {};
    synLaunchTensorInfo layer4_2_relu3_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_relu3_output");
        ASSERT_TRUE(status == synSuccess && "layer4_2_relu3_output dram malloc failed!");

        layer4_2_relu3_output_tr_info = synLaunchTensorInfo {"layer4_2_relu3_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_relu3_output = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer4_2_relu3_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_relu3_bwd_in_vec[2] = {worker_0_avgpool_grad_input, layer4_2_relu3_output};


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
    synTensor              layer4_2_add_residual_grad_input1 {};
    synLaunchTensorInfo layer4_2_add_residual_grad_input1_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_add_residual_grad_input1");
        ASSERT_TRUE(status == synSuccess && "layer4_2_add_residual_grad_input1 dram malloc failed!");

        layer4_2_add_residual_grad_input1_tr_info = synLaunchTensorInfo {"layer4_2_add_residual_grad_input1", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_add_residual_grad_input1 = createTensor(dims.size(),
                                                         syn_type_single,
                                                         dims.data(),
                                                         /*is_presist*/ true,
                                                         "layer4_2_add_residual_grad_input1",
                                                         /*graphHandle*/ nullptr,
                                                         /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_add_residual_bwd_out_vec[2] = {layer4_2_add_residual_grad_input0, layer4_2_add_residual_grad_input1};


    status = synNodeCreate(graphHandle, layer4_2_add_residual_bwd_in_vec, layer4_2_add_residual_bwd_out_vec, 1, 2, nullptr, 0, "add_bwd_f32", "layer4_2_add_residual_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_add_residual_bwd failed!");

    /*************
     * layer4_2_bn3_bwd node
     * inputs: [layer4_2_conv3_output[64, 7, 7, 2048](dtype=float32), layer4_2_add_residual_grad_input0(64, 7, 7, 2048)(dtype=float32), layer4_2_bn3_saved_mean[1, 1, 1, 2048](dtype=float32), layer4_2_bn3_saved_var[1, 1, 1, 2048](dtype=float32), layer4_2_bn3_weight[2048](dtype=float32)]
     * output: [layer4_2_bn3_grad_input(64, 7, 7, 2048)(dtype=float32), layer4_2_bn3_bias_grad(2048,)(dtype=float32), layer4_2_bn3_weight_grad(2048,)(dtype=float32)]
     *************/

    // create layer4_2_conv3_output tensor
    synTensor              layer4_2_conv3_output {};
    synLaunchTensorInfo layer4_2_conv3_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_conv3_output");
        ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_output dram malloc failed!");

        layer4_2_conv3_output_tr_info = synLaunchTensorInfo {"layer4_2_conv3_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_conv3_output = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer4_2_conv3_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_2_bn3_saved_mean tensor
    synTensor              layer4_2_bn3_saved_mean {};
    synLaunchTensorInfo layer4_2_bn3_saved_mean_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_saved_mean");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_saved_mean dram malloc failed!");

        layer4_2_bn3_saved_mean_tr_info = synLaunchTensorInfo {"layer4_2_bn3_saved_mean", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_saved_mean = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "layer4_2_bn3_saved_mean",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_2_bn3_saved_var tensor
    synTensor              layer4_2_bn3_saved_var {};
    synLaunchTensorInfo layer4_2_bn3_saved_var_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_saved_var");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_saved_var dram malloc failed!");

        layer4_2_bn3_saved_var_tr_info = synLaunchTensorInfo {"layer4_2_bn3_saved_var", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_saved_var = createTensor(dims.size(),
                                              syn_type_single,
                                              dims.data(),
                                              /*is_presist*/ true,
                                              "layer4_2_bn3_saved_var",
                                              /*graphHandle*/ nullptr,
                                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_2_bn3_weight tensor
    synTensor              layer4_2_bn3_weight {};
    synLaunchTensorInfo layer4_2_bn3_weight_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_weight");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_weight dram malloc failed!");

        layer4_2_bn3_weight_tr_info = synLaunchTensorInfo {"layer4_2_bn3_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_weight = createTensor(dims.size(),
                                           syn_type_single,
                                           dims.data(),
                                           /*is_presist*/ true,
                                           "layer4_2_bn3_weight",
                                           /*graphHandle*/ nullptr,
                                           /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_bn3_bwd_in_vec[5] = {layer4_2_conv3_output, layer4_2_add_residual_grad_input0, layer4_2_bn3_saved_mean, layer4_2_bn3_saved_var, layer4_2_bn3_weight};


    // create layer4_2_bn3_grad_input tensor
    const unsigned layer4_2_bn3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_2_bn3_grad_input = createTensor(4U, syn_type_single, layer4_2_bn3_grad_input_sizes, false, "layer4_2_bn3_grad_input");

    // create layer4_2_bn3_bias_grad tensor
    synTensor              layer4_2_bn3_bias_grad {};
    synLaunchTensorInfo layer4_2_bn3_bias_grad_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_bias_grad");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_bias_grad dram malloc failed!");

        layer4_2_bn3_bias_grad_tr_info = synLaunchTensorInfo {"layer4_2_bn3_bias_grad", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_bias_grad = createTensor(dims.size(),
                                              syn_type_single,
                                              dims.data(),
                                              /*is_presist*/ true,
                                              "layer4_2_bn3_bias_grad",
                                              /*graphHandle*/ nullptr,
                                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer4_2_bn3_weight_grad tensor
    synTensor              layer4_2_bn3_weight_grad {};
    synLaunchTensorInfo layer4_2_bn3_weight_grad_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_weight_grad");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_weight_grad dram malloc failed!");

        layer4_2_bn3_weight_grad_tr_info = synLaunchTensorInfo {"layer4_2_bn3_weight_grad", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_weight_grad = createTensor(dims.size(),
                                                syn_type_single,
                                                dims.data(),
                                                /*is_presist*/ true,
                                                "layer4_2_bn3_weight_grad",
                                                /*graphHandle*/ nullptr,
                                                /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_bn3_bwd_out_vec[3] = {layer4_2_bn3_grad_input, layer4_2_bn3_bias_grad, layer4_2_bn3_weight_grad};


    status = synNodeCreate(graphHandle, layer4_2_bn3_bwd_in_vec, layer4_2_bn3_bwd_out_vec, 5, 3, nullptr, 0, "batch_norm_bwd_f32", "layer4_2_bn3_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn3_bwd failed!");

    /*************
     * layer4_2_conv3_dedw node
     * inputs: [layer4_2_bn3_grad_input(64, 7, 7, 2048)(dtype=float32), layer4_2_relu2_output[64, 7, 7, 512](dtype=float32)]
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
    synTensor              layer4_2_relu2_output {};
    synLaunchTensorInfo layer4_2_relu2_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 512};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_relu2_output");
        ASSERT_TRUE(status == synSuccess && "layer4_2_relu2_output dram malloc failed!");

        layer4_2_relu2_output_tr_info = synLaunchTensorInfo {"layer4_2_relu2_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_relu2_output = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer4_2_relu2_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_conv3_dedw_in_vec[2] = {layer4_2_bn3_grad_input, layer4_2_relu2_output};


    // create layer4_2_conv3_weight_grad tensor
    synTensor              layer4_2_conv3_weight_grad {};
    synLaunchTensorInfo layer4_2_conv3_weight_grad_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 512, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_conv3_weight_grad");
        ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_weight_grad dram malloc failed!");

        layer4_2_conv3_weight_grad_tr_info = synLaunchTensorInfo {"layer4_2_conv3_weight_grad", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_conv3_weight_grad = createTensor(dims.size(),
                                                  syn_type_single,
                                                  dims.data(),
                                                  /*is_presist*/ true,
                                                  "layer4_2_conv3_weight_grad",
                                                  /*graphHandle*/ nullptr,
                                                  /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_conv3_dedw_out_vec[1] = {layer4_2_conv3_weight_grad};


    status = synNodeCreate(graphHandle, layer4_2_conv3_dedw_in_vec, layer4_2_conv3_dedw_out_vec, 2, 1, (void *)&layer4_2_conv3_dedw_kernel_params, sizeof(layer4_2_conv3_dedw_kernel_params), "dedw", "layer4_2_conv3_dedw", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv3_dedw failed!");

    /*************
     * layer4_2_conv3_dedx node
     * inputs: [layer4_2_bn3_grad_input(64, 7, 7, 2048)(dtype=float32), layer4_2_conv3_weight[1, 1, 512, 2048](dtype=float32)]
     * output: [layer4_2_conv3_grad_input(64, 7, 7, 512)(dtype=float32)]
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
    synTensor              layer4_2_conv3_weight {};
    synLaunchTensorInfo layer4_2_conv3_weight_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 512, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_conv3_weight");
        ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_weight dram malloc failed!");

        layer4_2_conv3_weight_tr_info = synLaunchTensorInfo {"layer4_2_conv3_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_conv3_weight = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "layer4_2_conv3_weight",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_conv3_dedx_in_vec[2] = {layer4_2_bn3_grad_input, layer4_2_conv3_weight};


    // create layer4_2_conv3_grad_input tensor
    synTensor              layer4_2_conv3_grad_input {};
    synLaunchTensorInfo layer4_2_conv3_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 512};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_conv3_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_2_conv3_grad_input dram malloc failed!");

        layer4_2_conv3_grad_input_tr_info = synLaunchTensorInfo {"layer4_2_conv3_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_conv3_grad_input = createTensor(dims.size(),
                                                 syn_type_single,
                                                 dims.data(),
                                                 /*is_presist*/ true,
                                                 "layer4_2_conv3_grad_input",
                                                 /*graphHandle*/ nullptr,
                                                 /*deviceAddr*/ eagerMode ? addr : -1);
    }

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
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
