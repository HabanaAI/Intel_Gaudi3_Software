

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

namespace
{
struct SynGaudi2ResNetTestFloat32 : SynGaudi2ResNetTestEager
{
    void conv2dFwd();
    void batchNorm2dFwd();
    void reluFwd();
    void maxPool2dFwd();
    void avgPool2dFwd();
    void linearFwd();
    void crossEntropyLossLogSoftmax();
    void crossEntropyLossLogSoftmaxBwd();
    void linearDedx();
    void linearDedw();
    void linearDedb();
    void avgPool2dBwd();
    void reluBwd();
    void addResidualBwd();
    void batchNorm2dBwd();
    void conv2DDedx();
    void conv2DDedw();
    void addResidualFwd();
    void maxPool2dBwd();
};
} // anonymous namespace

void SynGaudi2ResNetTestFloat32::conv2dFwd()
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
    synTensor              worker_0_conv1_output {};
    synLaunchTensorInfo worker_0_conv1_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 112, 112, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_conv1_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_conv1_output dram malloc failed!");

        worker_0_conv1_output_tr_info = synLaunchTensorInfo {"worker_0_conv1_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_conv1_output = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "worker_0_conv1_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_conv1_out_vec[1] = {worker_0_conv1_output};


    status = synNodeCreate(graphHandle, worker_0_conv1_in_vec, worker_0_conv1_out_vec, 4, 1, (void *)&worker_0_conv1_kernel_params, sizeof(worker_0_conv1_kernel_params), "spatial_convolution", "worker_0_conv1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_conv1 failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(input_tr_info);
    graph_inputs.push_back(worker_0_conv1_weight_tr_info);

    graph_outputs.push_back(worker_0_conv1_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::batchNorm2dFwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_bn1 node
     * inputs: [worker_0_conv1_output[64, 112, 112, 64](dtype=float32), worker_0_bn1_bias[64](dtype=float32), worker_0_bn1_weight[64](dtype=float32), worker_0_bn1_running_mean[64](dtype=float32), worker_0_bn1_running_var[64](dtype=float32)]
     * output: [worker_0_bn1_output(64, 112, 112, 64)(dtype=float32), worker_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), worker_0_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params worker_0_bn1_kernel_params;
    worker_0_bn1_kernel_params.momentum = 0.1;
    worker_0_bn1_kernel_params.threshold.f = 1e-05;
    worker_0_bn1_kernel_params.epsilon = 1e-05;

    // create worker_0_conv1_output tensor
    synTensor              worker_0_conv1_output {};
    synLaunchTensorInfo worker_0_conv1_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 112, 112, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_conv1_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_conv1_output dram malloc failed!");

        worker_0_conv1_output_tr_info = synLaunchTensorInfo {"worker_0_conv1_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_conv1_output = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "worker_0_conv1_output",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

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
    synTensor              worker_0_bn1_output {};
    synLaunchTensorInfo worker_0_bn1_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 112, 112, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_output dram malloc failed!");

        worker_0_bn1_output_tr_info = synLaunchTensorInfo {"worker_0_bn1_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_output = createTensor(dims.size(),
                                           syn_type_single,
                                           dims.data(),
                                           /*is_presist*/ true,
                                           "worker_0_bn1_output",
                                           /*graphHandle*/ nullptr,
                                           /*deviceAddr*/ eagerMode ? addr : -1);
    }

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


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_conv1_output_tr_info);
    graph_inputs.push_back(worker_0_bn1_weight_tr_info);
    graph_inputs.push_back(worker_0_bn1_bias_tr_info);
    graph_inputs.push_back(worker_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(worker_0_bn1_running_var_tr_info);

    graph_outputs.push_back(worker_0_bn1_output_tr_info);
    graph_outputs.push_back(worker_0_bn1_saved_mean_tr_info);
    graph_outputs.push_back(worker_0_bn1_saved_var_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::reluFwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_relu node
     * inputs: [worker_0_bn1_output[64, 112, 112, 64](dtype=float32)]
     * output: [worker_0_relu_output(64, 112, 112, 64)(dtype=float32)]
     *************/

    // create worker_0_bn1_output tensor
    synTensor              worker_0_bn1_output {};
    synLaunchTensorInfo worker_0_bn1_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 112, 112, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_bn1_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_bn1_output dram malloc failed!");

        worker_0_bn1_output_tr_info = synLaunchTensorInfo {"worker_0_bn1_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_bn1_output = createTensor(dims.size(),
                                           syn_type_single,
                                           dims.data(),
                                           /*is_presist*/ true,
                                           "worker_0_bn1_output",
                                           /*graphHandle*/ nullptr,
                                           /*deviceAddr*/ eagerMode ? addr : -1);
    }

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

    graph_inputs.push_back(worker_0_bn1_output_tr_info);

    graph_outputs.push_back(worker_0_relu_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::maxPool2dFwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_maxpool node
     * inputs: [worker_0_relu_output[64, 112, 112, 64](dtype=float32)]
     * output: [worker_0_maxpoolmax_indices(64, 56, 56, 64)(dtype=uint8), worker_0_maxpool_output(64, 56, 56, 64)(dtype=float32)]
     *************/
    ns_SpatialReduction::Params worker_0_maxpool_kernel_params;
    worker_0_maxpool_kernel_params.kernel_w = 3;
    worker_0_maxpool_kernel_params.kernel_h = 3;
    worker_0_maxpool_kernel_params.stride_w = 2;
    worker_0_maxpool_kernel_params.stride_h = 2;
    worker_0_maxpool_kernel_params.pad_w_begin = 1;
    worker_0_maxpool_kernel_params.pad_w_end = 1;
    worker_0_maxpool_kernel_params.pad_h_begin = 1;
    worker_0_maxpool_kernel_params.pad_h_end = 1;
    worker_0_maxpool_kernel_params.dilation_w = 1;
    worker_0_maxpool_kernel_params.dilation_h = 1;
    worker_0_maxpool_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

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

    synTensor worker_0_maxpool_in_vec[1] = {worker_0_relu_output};


    // create worker_0_maxpoolmax_indices tensor
    synTensor              worker_0_maxpoolmax_indices {};
    synLaunchTensorInfo worker_0_maxpoolmax_indices_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(uint8_t);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_maxpoolmax_indices");
        ASSERT_TRUE(status == synSuccess && "worker_0_maxpoolmax_indices dram malloc failed!");

        worker_0_maxpoolmax_indices_tr_info = synLaunchTensorInfo {"worker_0_maxpoolmax_indices", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_maxpoolmax_indices = createTensor(dims.size(),
                                                   syn_type_uint8,
                                                   dims.data(),
                                                   /*is_presist*/ true,
                                                   "worker_0_maxpoolmax_indices",
                                                   /*graphHandle*/ nullptr,
                                                   /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_maxpool_output tensor
    synTensor              worker_0_maxpool_output {};
    synLaunchTensorInfo worker_0_maxpool_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_maxpool_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_maxpool_output dram malloc failed!");

        worker_0_maxpool_output_tr_info = synLaunchTensorInfo {"worker_0_maxpool_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_maxpool_output = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "worker_0_maxpool_output",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_maxpool_out_vec[2] = {worker_0_maxpoolmax_indices, worker_0_maxpool_output};


    status = synNodeCreate(graphHandle, worker_0_maxpool_in_vec, worker_0_maxpool_out_vec, 1, 2, (void *)&worker_0_maxpool_kernel_params, sizeof(worker_0_maxpool_kernel_params), "maxpool_2d_fwd_f32", "worker_0_maxpool", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_maxpool failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_relu_output_tr_info);

    graph_outputs.push_back(worker_0_maxpool_output_tr_info);
    graph_outputs.push_back(worker_0_maxpoolmax_indices_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::avgPool2dFwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_avgpool node
     * inputs: [layer4_2_relu3_output[64, 7, 7, 2048](dtype=float32)]
     * output: [worker_0_avgpool_output(64, 1, 1, 2048)(dtype=float32)]
     *************/
    ns_AveragePooling::Params worker_0_avgpool_kernel_params;
    worker_0_avgpool_kernel_params.kernel_w = 7;
    worker_0_avgpool_kernel_params.kernel_h = 7;
    worker_0_avgpool_kernel_params.stride_w = 1;
    worker_0_avgpool_kernel_params.stride_h = 1;
    worker_0_avgpool_kernel_params.pad_w_begin = 0;
    worker_0_avgpool_kernel_params.pad_w_end = 0;
    worker_0_avgpool_kernel_params.pad_h_begin = 0;
    worker_0_avgpool_kernel_params.pad_h_end = 0;
    worker_0_avgpool_kernel_params.dilation_w = 1;
    worker_0_avgpool_kernel_params.dilation_h = 1;
    worker_0_avgpool_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

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

    synTensor worker_0_avgpool_in_vec[1] = {layer4_2_relu3_output};


    // create worker_0_avgpool_output tensor
    synTensor              worker_0_avgpool_output {};
    synLaunchTensorInfo worker_0_avgpool_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_avgpool_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_avgpool_output dram malloc failed!");

        worker_0_avgpool_output_tr_info = synLaunchTensorInfo {"worker_0_avgpool_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_avgpool_output = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "worker_0_avgpool_output",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_avgpool_out_vec[1] = {worker_0_avgpool_output};


    status = synNodeCreate(graphHandle, worker_0_avgpool_in_vec, worker_0_avgpool_out_vec, 1, 1, (void *)&worker_0_avgpool_kernel_params, sizeof(worker_0_avgpool_kernel_params), "avg_pool_2d_fwd_f32", "worker_0_avgpool", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_avgpool failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer4_2_relu3_output_tr_info);

    graph_outputs.push_back(worker_0_avgpool_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::linearFwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_fc node
     * inputs: [worker_0_avgpool_output(64, 1, 1, 2048)(dtype=float32), worker_0_fc_weight(1, 1, 2048, 1000)(dtype=float32), worker_0_fc_bias[1000](dtype=float32)]
     * output: [worker_0_fc_output(64, 1, 1, 1000)(dtype=float32)]
     *************/
    synConvolutionParams worker_0_fc_kernel_params;
    worker_0_fc_kernel_params.dH = 1;
    worker_0_fc_kernel_params.dW = 1;
    worker_0_fc_kernel_params.kH = 1;
    worker_0_fc_kernel_params.kW = 1;
    worker_0_fc_kernel_params.padT = 0;
    worker_0_fc_kernel_params.padB = 0;
    worker_0_fc_kernel_params.padL = 0;
    worker_0_fc_kernel_params.padR = 0;
    worker_0_fc_kernel_params.dilH = 1;
    worker_0_fc_kernel_params.dilW = 1;

    // create worker_0_avgpool_output tensor
    synTensor              worker_0_avgpool_output {};
    synLaunchTensorInfo worker_0_avgpool_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_avgpool_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_avgpool_output dram malloc failed!");

        worker_0_avgpool_output_tr_info = synLaunchTensorInfo {"worker_0_avgpool_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_avgpool_output = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "worker_0_avgpool_output",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_fc_weight tensor
    synTensor              worker_0_fc_weight {};
    synLaunchTensorInfo worker_0_fc_weight_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 2048, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_weight");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_weight dram malloc failed!");

        worker_0_fc_weight_tr_info = synLaunchTensorInfo {"worker_0_fc_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_weight = createTensor(dims.size(),
                                          syn_type_single,
                                          dims.data(),
                                          /*is_presist*/ true,
                                          "worker_0_fc_weight",
                                          /*graphHandle*/ nullptr,
                                          /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_fc_bias tensor
    synTensor              worker_0_fc_bias {};
    synLaunchTensorInfo worker_0_fc_bias_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_bias");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_bias dram malloc failed!");

        worker_0_fc_bias_tr_info = synLaunchTensorInfo {"worker_0_fc_bias", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_bias = createTensor(dims.size(),
                                        syn_type_single,
                                        dims.data(),
                                        /*is_presist*/ true,
                                        "worker_0_fc_bias",
                                        /*graphHandle*/ nullptr,
                                        /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_in_vec[4] = {worker_0_avgpool_output, worker_0_fc_weight, worker_0_fc_bias, nullptr};


    // create worker_0_fc_output tensor
    synTensor              worker_0_fc_output {};
    synLaunchTensorInfo worker_0_fc_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_output dram malloc failed!");

        worker_0_fc_output_tr_info = synLaunchTensorInfo {"worker_0_fc_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_output = createTensor(dims.size(),
                                          syn_type_single,
                                          dims.data(),
                                          /*is_presist*/ true,
                                          "worker_0_fc_output",
                                          /*graphHandle*/ nullptr,
                                          /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_out_vec[1] = {worker_0_fc_output};


    status = synNodeCreate(graphHandle, worker_0_fc_in_vec, worker_0_fc_out_vec, 4, 1, (void *)&worker_0_fc_kernel_params, sizeof(worker_0_fc_kernel_params), "spatial_convolution", "worker_0_fc", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_avgpool_output_tr_info);
    graph_inputs.push_back(worker_0_fc_weight_tr_info);
    graph_inputs.push_back(worker_0_fc_bias_tr_info);

    graph_outputs.push_back(worker_0_fc_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::crossEntropyLossLogSoftmax()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * cross_entropy_loss0_log_softmax node
     * inputs: [worker_0_fc_output[64, 1000](dtype=float32), target[64](dtype=int32)]
     * output: [cross_entropy_loss0_output(1,)(dtype=float32), cross_entropy_loss0_logs_output(64, 1000)(dtype=float32)]
     *************/
    ns_SoftmaxCrossEntropy::Params cross_entropy_loss0_log_softmax_kernel_params;
    cross_entropy_loss0_log_softmax_kernel_params.mode = CROSS_ENTROPY_MODE_MEAN;
    cross_entropy_loss0_log_softmax_kernel_params.batchSize = batchSize;

    // create worker_0_fc_output tensor
    synTensor              worker_0_fc_output {};
    synLaunchTensorInfo worker_0_fc_output_tr_info {};
    {
        const std::array<unsigned, 2> dims  = {64, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_output dram malloc failed!");

        worker_0_fc_output_tr_info = synLaunchTensorInfo {"worker_0_fc_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_output = createTensor(dims.size(),
                                          syn_type_single,
                                          dims.data(),
                                          /*is_presist*/ true,
                                          "worker_0_fc_output",
                                          /*graphHandle*/ nullptr,
                                          /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create target tensor
    synTensor              target {};
    synLaunchTensorInfo target_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(int32_t);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "target");
        ASSERT_TRUE(status == synSuccess && "target dram malloc failed!");

        target_tr_info = synLaunchTensorInfo {"target", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        target = createTensor(dims.size(),
                              syn_type_int32,
                              dims.data(),
                              /*is_presist*/ true,
                              "target",
                              /*graphHandle*/ nullptr,
                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor cross_entropy_loss0_log_softmax_in_vec[2] = {worker_0_fc_output, target};


    // create cross_entropy_loss0_output tensor
    synTensor              cross_entropy_loss0_output {};
    synLaunchTensorInfo cross_entropy_loss0_output_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {1};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "cross_entropy_loss0_output");
        ASSERT_TRUE(status == synSuccess && "cross_entropy_loss0_output dram malloc failed!");

        cross_entropy_loss0_output_tr_info = synLaunchTensorInfo {"cross_entropy_loss0_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        cross_entropy_loss0_output = createTensor(dims.size(),
                                                  syn_type_single,
                                                  dims.data(),
                                                  /*is_presist*/ true,
                                                  "cross_entropy_loss0_output",
                                                  /*graphHandle*/ nullptr,
                                                  /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create cross_entropy_loss0_logs_output tensor
    synTensor              cross_entropy_loss0_logs_output {};
    synLaunchTensorInfo cross_entropy_loss0_logs_output_tr_info {};
    {
        const std::array<unsigned, 2> dims  = {64, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "cross_entropy_loss0_logs_output");
        ASSERT_TRUE(status == synSuccess && "cross_entropy_loss0_logs_output dram malloc failed!");

        cross_entropy_loss0_logs_output_tr_info = synLaunchTensorInfo {"cross_entropy_loss0_logs_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        cross_entropy_loss0_logs_output = createTensor(dims.size(),
                                                       syn_type_single,
                                                       dims.data(),
                                                       /*is_presist*/ true,
                                                       "cross_entropy_loss0_logs_output",
                                                       /*graphHandle*/ nullptr,
                                                       /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor cross_entropy_loss0_log_softmax_out_vec[2] = {cross_entropy_loss0_output, cross_entropy_loss0_logs_output};


    status = synNodeCreate(graphHandle, cross_entropy_loss0_log_softmax_in_vec, cross_entropy_loss0_log_softmax_out_vec, 2, 2, (void *)&cross_entropy_loss0_log_softmax_kernel_params, sizeof(cross_entropy_loss0_log_softmax_kernel_params), "softmax_cross_entropy_fwd_f32", "cross_entropy_loss0_log_softmax", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for cross_entropy_loss0_log_softmax failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_fc_output_tr_info);
    graph_inputs.push_back(target_tr_info);

    graph_outputs.push_back(cross_entropy_loss0_output_tr_info);
    graph_outputs.push_back(cross_entropy_loss0_logs_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::crossEntropyLossLogSoftmaxBwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * cross_entropy_loss0_log_softmax_bwd node
     * inputs: [cross_entropy_loss0_logs_output[64, 1000](dtype=float32), target[64](dtype=int32)]
     * output: [cross_entropy_loss0_grad_input(64, 1000)(dtype=float32)]
     *************/
    ns_SoftmaxCrossEntropy::Params cross_entropy_loss0_log_softmax_bwd_kernel_params;
    cross_entropy_loss0_log_softmax_bwd_kernel_params.mode = CROSS_ENTROPY_MODE_MEAN;
    cross_entropy_loss0_log_softmax_bwd_kernel_params.batchSize = batchSize;

    // create cross_entropy_loss0_logs_output tensor
    synTensor              cross_entropy_loss0_logs_output {};
    synLaunchTensorInfo cross_entropy_loss0_logs_output_tr_info {};
    {
        const std::array<unsigned, 2> dims  = {64, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "cross_entropy_loss0_logs_output");
        ASSERT_TRUE(status == synSuccess && "cross_entropy_loss0_logs_output dram malloc failed!");

        cross_entropy_loss0_logs_output_tr_info = synLaunchTensorInfo {"cross_entropy_loss0_logs_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        cross_entropy_loss0_logs_output = createTensor(dims.size(),
                                                       syn_type_single,
                                                       dims.data(),
                                                       /*is_presist*/ true,
                                                       "cross_entropy_loss0_logs_output",
                                                       /*graphHandle*/ nullptr,
                                                       /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create target tensor
    synTensor              target {};
    synLaunchTensorInfo target_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {64};
        const unsigned                bytes = prod(dims) * sizeof(int32_t);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "target");
        ASSERT_TRUE(status == synSuccess && "target dram malloc failed!");

        target_tr_info = synLaunchTensorInfo {"target", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        target = createTensor(dims.size(),
                              syn_type_int32,
                              dims.data(),
                              /*is_presist*/ true,
                              "target",
                              /*graphHandle*/ nullptr,
                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor cross_entropy_loss0_log_softmax_bwd_in_vec[2] = {cross_entropy_loss0_logs_output, target};


    // create cross_entropy_loss0_grad_input tensor
    synTensor              cross_entropy_loss0_grad_input {};
    synLaunchTensorInfo cross_entropy_loss0_grad_input_tr_info {};
    {
        const std::array<unsigned, 2> dims  = {64, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "cross_entropy_loss0_grad_input");
        ASSERT_TRUE(status == synSuccess && "cross_entropy_loss0_grad_input dram malloc failed!");

        cross_entropy_loss0_grad_input_tr_info = synLaunchTensorInfo {"cross_entropy_loss0_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        cross_entropy_loss0_grad_input = createTensor(dims.size(),
                                                      syn_type_single,
                                                      dims.data(),
                                                      /*is_presist*/ true,
                                                      "cross_entropy_loss0_grad_input",
                                                      /*graphHandle*/ nullptr,
                                                      /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor cross_entropy_loss0_log_softmax_bwd_out_vec[1] = {cross_entropy_loss0_grad_input};


    status = synNodeCreate(graphHandle, cross_entropy_loss0_log_softmax_bwd_in_vec, cross_entropy_loss0_log_softmax_bwd_out_vec, 2, 1, (void *)&cross_entropy_loss0_log_softmax_bwd_kernel_params, sizeof(cross_entropy_loss0_log_softmax_bwd_kernel_params), "softmax_cross_entropy_bwd_f32", "cross_entropy_loss0_log_softmax_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for cross_entropy_loss0_log_softmax_bwd failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(cross_entropy_loss0_logs_output_tr_info);
    graph_inputs.push_back(target_tr_info);

    graph_outputs.push_back(cross_entropy_loss0_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::linearDedx()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_fc_dedx node
     * inputs: [cross_entropy_loss0_grad_input(64, 1, 1, 1000)(dtype=float32), worker_0_fc_weight(1, 1, 2048, 1000)(dtype=float32)]
     * output: [worker_0_fc_grad_input(64, 1, 1, 2048)(dtype=float32)]
     *************/
    synConvolutionParams worker_0_fc_dedx_kernel_params;
    worker_0_fc_dedx_kernel_params.dH = 1;
    worker_0_fc_dedx_kernel_params.dW = 1;
    worker_0_fc_dedx_kernel_params.kH = 1;
    worker_0_fc_dedx_kernel_params.kW = 1;
    worker_0_fc_dedx_kernel_params.padT = 0;
    worker_0_fc_dedx_kernel_params.padB = 0;
    worker_0_fc_dedx_kernel_params.padL = 0;
    worker_0_fc_dedx_kernel_params.padR = 0;
    worker_0_fc_dedx_kernel_params.dilH = 1;
    worker_0_fc_dedx_kernel_params.dilW = 1;

    // create cross_entropy_loss0_grad_input tensor
    synTensor              cross_entropy_loss0_grad_input {};
    synLaunchTensorInfo cross_entropy_loss0_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "cross_entropy_loss0_grad_input");
        ASSERT_TRUE(status == synSuccess && "cross_entropy_loss0_grad_input dram malloc failed!");

        cross_entropy_loss0_grad_input_tr_info = synLaunchTensorInfo {"cross_entropy_loss0_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        cross_entropy_loss0_grad_input = createTensor(dims.size(),
                                                      syn_type_single,
                                                      dims.data(),
                                                      /*is_presist*/ true,
                                                      "cross_entropy_loss0_grad_input",
                                                      /*graphHandle*/ nullptr,
                                                      /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_fc_weight tensor
    synTensor              worker_0_fc_weight {};
    synLaunchTensorInfo worker_0_fc_weight_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 2048, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_weight");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_weight dram malloc failed!");

        worker_0_fc_weight_tr_info = synLaunchTensorInfo {"worker_0_fc_weight", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_weight = createTensor(dims.size(),
                                          syn_type_single,
                                          dims.data(),
                                          /*is_presist*/ true,
                                          "worker_0_fc_weight",
                                          /*graphHandle*/ nullptr,
                                          /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_dedx_in_vec[2] = {cross_entropy_loss0_grad_input, worker_0_fc_weight};


    // create worker_0_fc_grad_input tensor
    synTensor              worker_0_fc_grad_input {};
    synLaunchTensorInfo worker_0_fc_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_grad_input");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_grad_input dram malloc failed!");

        worker_0_fc_grad_input_tr_info = synLaunchTensorInfo {"worker_0_fc_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_grad_input = createTensor(dims.size(),
                                              syn_type_single,
                                              dims.data(),
                                              /*is_presist*/ true,
                                              "worker_0_fc_grad_input",
                                              /*graphHandle*/ nullptr,
                                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_dedx_out_vec[1] = {worker_0_fc_grad_input};


    status = synNodeCreate(graphHandle, worker_0_fc_dedx_in_vec, worker_0_fc_dedx_out_vec, 2, 1, (void *)&worker_0_fc_dedx_kernel_params, sizeof(worker_0_fc_dedx_kernel_params), "dedx", "worker_0_fc_dedx", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedx failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(cross_entropy_loss0_grad_input_tr_info);
    graph_inputs.push_back(worker_0_fc_weight_tr_info);

    graph_outputs.push_back(worker_0_fc_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::linearDedw()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_fc_dedw node
     * inputs: [cross_entropy_loss0_grad_input(64, 1, 1, 1000)(dtype=float32), worker_0_avgpool_output(64, 1, 1, 2048)(dtype=float32)]
     * output: [worker_0_fc_weight_grad(1, 1, 2048, 1000)(dtype=float32)]
     *************/
    synConvolutionParams worker_0_fc_dedw_kernel_params;
    worker_0_fc_dedw_kernel_params.dH = 1;
    worker_0_fc_dedw_kernel_params.dW = 1;
    worker_0_fc_dedw_kernel_params.kH = 1;
    worker_0_fc_dedw_kernel_params.kW = 1;
    worker_0_fc_dedw_kernel_params.padT = 0;
    worker_0_fc_dedw_kernel_params.padB = 0;
    worker_0_fc_dedw_kernel_params.padL = 0;
    worker_0_fc_dedw_kernel_params.padR = 0;
    worker_0_fc_dedw_kernel_params.dilH = 1;
    worker_0_fc_dedw_kernel_params.dilW = 1;

    // create cross_entropy_loss0_grad_input tensor
    synTensor              cross_entropy_loss0_grad_input {};
    synLaunchTensorInfo cross_entropy_loss0_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "cross_entropy_loss0_grad_input");
        ASSERT_TRUE(status == synSuccess && "cross_entropy_loss0_grad_input dram malloc failed!");

        cross_entropy_loss0_grad_input_tr_info = synLaunchTensorInfo {"cross_entropy_loss0_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        cross_entropy_loss0_grad_input = createTensor(dims.size(),
                                                      syn_type_single,
                                                      dims.data(),
                                                      /*is_presist*/ true,
                                                      "cross_entropy_loss0_grad_input",
                                                      /*graphHandle*/ nullptr,
                                                      /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_avgpool_output tensor
    synTensor              worker_0_avgpool_output {};
    synLaunchTensorInfo worker_0_avgpool_output_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_avgpool_output");
        ASSERT_TRUE(status == synSuccess && "worker_0_avgpool_output dram malloc failed!");

        worker_0_avgpool_output_tr_info = synLaunchTensorInfo {"worker_0_avgpool_output", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_avgpool_output = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "worker_0_avgpool_output",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_dedw_in_vec[2] = {cross_entropy_loss0_grad_input, worker_0_avgpool_output};


    // create worker_0_fc_weight_grad tensor
    synTensor              worker_0_fc_weight_grad {};
    synLaunchTensorInfo worker_0_fc_weight_grad_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {1, 1, 2048, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_weight_grad");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_weight_grad dram malloc failed!");

        worker_0_fc_weight_grad_tr_info = synLaunchTensorInfo {"worker_0_fc_weight_grad", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_weight_grad = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "worker_0_fc_weight_grad",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_dedw_out_vec[1] = {worker_0_fc_weight_grad};


    status = synNodeCreate(graphHandle, worker_0_fc_dedw_in_vec, worker_0_fc_dedw_out_vec, 2, 1, (void *)&worker_0_fc_dedw_kernel_params, sizeof(worker_0_fc_dedw_kernel_params), "dedw", "worker_0_fc_dedw", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedw failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(cross_entropy_loss0_grad_input_tr_info);
    graph_inputs.push_back(worker_0_avgpool_output_tr_info);

    graph_outputs.push_back(worker_0_fc_weight_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::linearDedb()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_fc_dedb node
     * inputs: [cross_entropy_loss0_grad_input[64, 1000](dtype=float32)]
     * output: [worker_0_fc_bias_grad(1000,)(dtype=float32)]
     *************/
    ns_Reduction::Params worker_0_fc_dedb_kernel_params;
    worker_0_fc_dedb_kernel_params.reductionDimension = 1;

    // create cross_entropy_loss0_grad_input tensor
    synTensor              cross_entropy_loss0_grad_input {};
    synLaunchTensorInfo cross_entropy_loss0_grad_input_tr_info {};
    {
        const std::array<unsigned, 2> dims  = {64, 1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "cross_entropy_loss0_grad_input");
        ASSERT_TRUE(status == synSuccess && "cross_entropy_loss0_grad_input dram malloc failed!");

        cross_entropy_loss0_grad_input_tr_info = synLaunchTensorInfo {"cross_entropy_loss0_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        cross_entropy_loss0_grad_input = createTensor(dims.size(),
                                                      syn_type_single,
                                                      dims.data(),
                                                      /*is_presist*/ true,
                                                      "cross_entropy_loss0_grad_input",
                                                      /*graphHandle*/ nullptr,
                                                      /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_dedb_in_vec[1] = {cross_entropy_loss0_grad_input};


    // create worker_0_fc_bias_grad tensor
    synTensor              worker_0_fc_bias_grad {};
    synLaunchTensorInfo worker_0_fc_bias_grad_tr_info {};
    {
        const std::array<unsigned, 1> dims  = {1000};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_bias_grad");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_bias_grad dram malloc failed!");

        worker_0_fc_bias_grad_tr_info = synLaunchTensorInfo {"worker_0_fc_bias_grad", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_bias_grad = createTensor(dims.size(),
                                             syn_type_single,
                                             dims.data(),
                                             /*is_presist*/ true,
                                             "worker_0_fc_bias_grad",
                                             /*graphHandle*/ nullptr,
                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_fc_dedb_out_vec[1] = {worker_0_fc_bias_grad};


    status = synNodeCreate(graphHandle, worker_0_fc_dedb_in_vec, worker_0_fc_dedb_out_vec, 1, 1, (void *)&worker_0_fc_dedb_kernel_params, sizeof(worker_0_fc_dedb_kernel_params), "reduce_sum_fwd_f32", "worker_0_fc_dedb", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedb failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(cross_entropy_loss0_grad_input_tr_info);

    graph_outputs.push_back(worker_0_fc_bias_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::avgPool2dBwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_avgpool_bwd node
     * inputs: [worker_0_fc_grad_input[64, 1, 1, 2048](dtype=float32)]
     * output: [worker_0_avgpool_grad_input(64, 7, 7, 2048)(dtype=float32)]
     *************/
    ns_AveragePooling::Params worker_0_avgpool_bwd_kernel_params;
    worker_0_avgpool_bwd_kernel_params.kernel_w = 7;
    worker_0_avgpool_bwd_kernel_params.kernel_h = 7;
    worker_0_avgpool_bwd_kernel_params.stride_w = 1;
    worker_0_avgpool_bwd_kernel_params.stride_h = 1;
    worker_0_avgpool_bwd_kernel_params.pad_w_begin = 0;
    worker_0_avgpool_bwd_kernel_params.pad_w_end = 0;
    worker_0_avgpool_bwd_kernel_params.pad_h_begin = 0;
    worker_0_avgpool_bwd_kernel_params.pad_h_end = 0;
    worker_0_avgpool_bwd_kernel_params.dilation_w = 1;
    worker_0_avgpool_bwd_kernel_params.dilation_h = 1;
    worker_0_avgpool_bwd_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    // create worker_0_fc_grad_input tensor
    synTensor              worker_0_fc_grad_input {};
    synLaunchTensorInfo worker_0_fc_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 1, 1, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_fc_grad_input");
        ASSERT_TRUE(status == synSuccess && "worker_0_fc_grad_input dram malloc failed!");

        worker_0_fc_grad_input_tr_info = synLaunchTensorInfo {"worker_0_fc_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_fc_grad_input = createTensor(dims.size(),
                                              syn_type_single,
                                              dims.data(),
                                              /*is_presist*/ true,
                                              "worker_0_fc_grad_input",
                                              /*graphHandle*/ nullptr,
                                              /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_avgpool_bwd_in_vec[1] = {worker_0_fc_grad_input};


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

    synTensor worker_0_avgpool_bwd_out_vec[1] = {worker_0_avgpool_grad_input};


    status = synNodeCreate(graphHandle, worker_0_avgpool_bwd_in_vec, worker_0_avgpool_bwd_out_vec, 1, 1, (void *)&worker_0_avgpool_bwd_kernel_params, sizeof(worker_0_avgpool_bwd_kernel_params), "avg_pool_2d_bwd_f32", "worker_0_avgpool_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_avgpool_bwd failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_fc_grad_input_tr_info);

    graph_outputs.push_back(worker_0_avgpool_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::reluBwd()
{
    const bool eagerMode = true;

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
    synTensor              layer4_2_relu3_grad_input {};
    synLaunchTensorInfo layer4_2_relu3_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_relu3_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_2_relu3_grad_input dram malloc failed!");

        layer4_2_relu3_grad_input_tr_info = synLaunchTensorInfo {"layer4_2_relu3_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_relu3_grad_input = createTensor(dims.size(),
                                                 syn_type_single,
                                                 dims.data(),
                                                 /*is_presist*/ true,
                                                 "layer4_2_relu3_grad_input",
                                                 /*graphHandle*/ nullptr,
                                                 /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_relu3_bwd_out_vec[1] = {layer4_2_relu3_grad_input};


    status = synNodeCreate(graphHandle, layer4_2_relu3_bwd_in_vec, layer4_2_relu3_bwd_out_vec, 2, 1, nullptr, 0, "relu_bwd_f32", "layer4_2_relu3_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu3_bwd failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(worker_0_avgpool_grad_input_tr_info);
    graph_inputs.push_back(layer4_2_relu3_output_tr_info);

    graph_outputs.push_back(layer4_2_relu3_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::addResidualBwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_2_add_residual_bwd node
     * inputs: [layer4_2_relu3_grad_input[64, 7, 7, 2048](dtype=float32)]
     * output: [layer4_2_add_residual_grad_input0(64, 7, 7, 2048)(dtype=float32), layer4_2_add_residual_grad_input1(64, 7, 7, 2048)(dtype=float32)]
     *************/

    // create layer4_2_relu3_grad_input tensor
    synTensor              layer4_2_relu3_grad_input {};
    synLaunchTensorInfo layer4_2_relu3_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_relu3_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_2_relu3_grad_input dram malloc failed!");

        layer4_2_relu3_grad_input_tr_info = synLaunchTensorInfo {"layer4_2_relu3_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_relu3_grad_input = createTensor(dims.size(),
                                                 syn_type_single,
                                                 dims.data(),
                                                 /*is_presist*/ true,
                                                 "layer4_2_relu3_grad_input",
                                                 /*graphHandle*/ nullptr,
                                                 /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer4_2_add_residual_bwd_in_vec[1] = {layer4_2_relu3_grad_input};


    // create layer4_2_add_residual_grad_input0 tensor
    synTensor              layer4_2_add_residual_grad_input0 {};
    synLaunchTensorInfo layer4_2_add_residual_grad_input0_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_add_residual_grad_input0");
        ASSERT_TRUE(status == synSuccess && "layer4_2_add_residual_grad_input0 dram malloc failed!");

        layer4_2_add_residual_grad_input0_tr_info = synLaunchTensorInfo {"layer4_2_add_residual_grad_input0", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_add_residual_grad_input0 = createTensor(dims.size(),
                                                         syn_type_single,
                                                         dims.data(),
                                                         /*is_presist*/ true,
                                                         "layer4_2_add_residual_grad_input0",
                                                         /*graphHandle*/ nullptr,
                                                         /*deviceAddr*/ eagerMode ? addr : -1);
    }

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


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer4_2_relu3_grad_input_tr_info);

    graph_outputs.push_back(layer4_2_add_residual_grad_input0_tr_info);
    graph_outputs.push_back(layer4_2_add_residual_grad_input1_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::batchNorm2dBwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_2_bn3_bwd node
     * inputs: [layer4_2_conv3_output[64, 7, 7, 2048](dtype=float32), layer4_2_add_residual_grad_input0[64, 7, 7, 2048](dtype=float32), layer4_2_bn3_saved_mean[1, 1, 1, 2048](dtype=float32), layer4_2_bn3_saved_var[1, 1, 1, 2048](dtype=float32), layer4_2_bn3_weight[2048](dtype=float32)]
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

    // create layer4_2_add_residual_grad_input0 tensor
    synTensor              layer4_2_add_residual_grad_input0 {};
    synLaunchTensorInfo layer4_2_add_residual_grad_input0_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_add_residual_grad_input0");
        ASSERT_TRUE(status == synSuccess && "layer4_2_add_residual_grad_input0 dram malloc failed!");

        layer4_2_add_residual_grad_input0_tr_info = synLaunchTensorInfo {"layer4_2_add_residual_grad_input0", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_add_residual_grad_input0 = createTensor(dims.size(),
                                                         syn_type_single,
                                                         dims.data(),
                                                         /*is_presist*/ true,
                                                         "layer4_2_add_residual_grad_input0",
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
    synTensor              layer4_2_bn3_grad_input {};
    synLaunchTensorInfo layer4_2_bn3_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_grad_input dram malloc failed!");

        layer4_2_bn3_grad_input_tr_info = synLaunchTensorInfo {"layer4_2_bn3_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_grad_input = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "layer4_2_bn3_grad_input",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

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


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer4_2_add_residual_grad_input0_tr_info);
    graph_inputs.push_back(layer4_2_conv3_output_tr_info);
    graph_inputs.push_back(layer4_2_bn3_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn3_saved_mean_tr_info);
    graph_inputs.push_back(layer4_2_bn3_saved_var_tr_info);

    graph_outputs.push_back(layer4_2_bn3_grad_input_tr_info);
    graph_outputs.push_back(layer4_2_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn3_bias_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::conv2DDedx()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_2_conv3_dedx node
     * inputs: [layer4_2_bn3_grad_input[64, 7, 7, 2048](dtype=float32), layer4_2_conv3_weight[1, 1, 512, 2048](dtype=float32)]
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

    // create layer4_2_bn3_grad_input tensor
    synTensor              layer4_2_bn3_grad_input {};
    synLaunchTensorInfo layer4_2_bn3_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_grad_input dram malloc failed!");

        layer4_2_bn3_grad_input_tr_info = synLaunchTensorInfo {"layer4_2_bn3_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_grad_input = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "layer4_2_bn3_grad_input",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

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

    graph_inputs.push_back(layer4_2_bn3_grad_input_tr_info);
    graph_inputs.push_back(layer4_2_conv3_weight_tr_info);

    graph_outputs.push_back(layer4_2_conv3_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::conv2DDedw()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer4_2_conv3_dedw node
     * inputs: [layer4_2_bn3_grad_input[64, 7, 7, 2048](dtype=float32), layer4_2_relu2_output[64, 7, 7, 512](dtype=float32)]
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

    // create layer4_2_bn3_grad_input tensor
    synTensor              layer4_2_bn3_grad_input {};
    synLaunchTensorInfo layer4_2_bn3_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 7, 7, 2048};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer4_2_bn3_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer4_2_bn3_grad_input dram malloc failed!");

        layer4_2_bn3_grad_input_tr_info = synLaunchTensorInfo {"layer4_2_bn3_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer4_2_bn3_grad_input = createTensor(dims.size(),
                                               syn_type_single,
                                               dims.data(),
                                               /*is_presist*/ true,
                                               "layer4_2_bn3_grad_input",
                                               /*graphHandle*/ nullptr,
                                               /*deviceAddr*/ eagerMode ? addr : -1);
    }

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


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer4_2_bn3_grad_input_tr_info);
    graph_inputs.push_back(layer4_2_relu2_output_tr_info);

    graph_outputs.push_back(layer4_2_conv3_weight_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::addResidualFwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * layer1_0_add_residual_fwd1 node
     * inputs: [layer1_0_conv1_grad_input[64, 56, 56, 64](dtype=float32), layer1_downsample_grad_input[64, 56, 56, 64](dtype=float32)]
     * output: [layer1_0_residual_upstream_grad_input(64, 56, 56, 64)(dtype=float32)]
     *************/

    // create layer1_0_conv1_grad_input tensor
    synTensor              layer1_0_conv1_grad_input {};
    synLaunchTensorInfo layer1_0_conv1_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_conv1_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer1_0_conv1_grad_input dram malloc failed!");

        layer1_0_conv1_grad_input_tr_info = synLaunchTensorInfo {"layer1_0_conv1_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_conv1_grad_input = createTensor(dims.size(),
                                                 syn_type_single,
                                                 dims.data(),
                                                 /*is_presist*/ true,
                                                 "layer1_0_conv1_grad_input",
                                                 /*graphHandle*/ nullptr,
                                                 /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create layer1_downsample_grad_input tensor
    synTensor              layer1_downsample_grad_input {};
    synLaunchTensorInfo layer1_downsample_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_downsample_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer1_downsample_grad_input dram malloc failed!");

        layer1_downsample_grad_input_tr_info = synLaunchTensorInfo {"layer1_downsample_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_downsample_grad_input = createTensor(dims.size(),
                                                    syn_type_single,
                                                    dims.data(),
                                                    /*is_presist*/ true,
                                                    "layer1_downsample_grad_input",
                                                    /*graphHandle*/ nullptr,
                                                    /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer1_0_add_residual_fwd1_in_vec[2] = {layer1_0_conv1_grad_input, layer1_downsample_grad_input};


    // create layer1_0_residual_upstream_grad_input tensor
    synTensor              layer1_0_residual_upstream_grad_input {};
    synLaunchTensorInfo layer1_0_residual_upstream_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_residual_upstream_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer1_0_residual_upstream_grad_input dram malloc failed!");

        layer1_0_residual_upstream_grad_input_tr_info = synLaunchTensorInfo {"layer1_0_residual_upstream_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_residual_upstream_grad_input = createTensor(dims.size(),
                                                             syn_type_single,
                                                             dims.data(),
                                                             /*is_presist*/ true,
                                                             "layer1_0_residual_upstream_grad_input",
                                                             /*graphHandle*/ nullptr,
                                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor layer1_0_add_residual_fwd1_out_vec[1] = {layer1_0_residual_upstream_grad_input};


    status = synNodeCreate(graphHandle, layer1_0_add_residual_fwd1_in_vec, layer1_0_add_residual_fwd1_out_vec, 2, 1, nullptr, 0, "add_fwd_f32", "layer1_0_add_residual_fwd1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_add_residual_fwd1 failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer1_0_conv1_grad_input_tr_info);
    graph_inputs.push_back(layer1_downsample_grad_input_tr_info);

    graph_outputs.push_back(layer1_0_residual_upstream_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

void SynGaudi2ResNetTestFloat32::maxPool2dBwd()
{
    const bool eagerMode = true;

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_maxpool_bwd node
     * inputs: [layer1_0_residual_upstream_grad_input[64, 56, 56, 64](dtype=float32), worker_0_maxpoolmax_indices[64, 56, 56, 64](dtype=uint8)]
     * output: [worker_0_maxpool_grad_input(64, 112, 112, 64)(dtype=float32)]
     *************/
    ns_SpatialReduction::Params worker_0_maxpool_bwd_kernel_params;
    worker_0_maxpool_bwd_kernel_params.kernel_w = 3;
    worker_0_maxpool_bwd_kernel_params.kernel_h = 3;
    worker_0_maxpool_bwd_kernel_params.stride_w = 2;
    worker_0_maxpool_bwd_kernel_params.stride_h = 2;
    worker_0_maxpool_bwd_kernel_params.pad_w_begin = 1;
    worker_0_maxpool_bwd_kernel_params.pad_w_end = 1;
    worker_0_maxpool_bwd_kernel_params.pad_h_begin = 1;
    worker_0_maxpool_bwd_kernel_params.pad_h_end = 1;
    worker_0_maxpool_bwd_kernel_params.dilation_w = 1;
    worker_0_maxpool_bwd_kernel_params.dilation_h = 1;
    worker_0_maxpool_bwd_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    // create layer1_0_residual_upstream_grad_input tensor
    synTensor              layer1_0_residual_upstream_grad_input {};
    synLaunchTensorInfo layer1_0_residual_upstream_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "layer1_0_residual_upstream_grad_input");
        ASSERT_TRUE(status == synSuccess && "layer1_0_residual_upstream_grad_input dram malloc failed!");

        layer1_0_residual_upstream_grad_input_tr_info = synLaunchTensorInfo {"layer1_0_residual_upstream_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        layer1_0_residual_upstream_grad_input = createTensor(dims.size(),
                                                             syn_type_single,
                                                             dims.data(),
                                                             /*is_presist*/ true,
                                                             "layer1_0_residual_upstream_grad_input",
                                                             /*graphHandle*/ nullptr,
                                                             /*deviceAddr*/ eagerMode ? addr : -1);
    }

    // create worker_0_maxpoolmax_indices tensor
    synTensor              worker_0_maxpoolmax_indices {};
    synLaunchTensorInfo worker_0_maxpoolmax_indices_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 56, 56, 64};
        const unsigned                bytes = prod(dims) * sizeof(uint8_t);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_maxpoolmax_indices");
        ASSERT_TRUE(status == synSuccess && "worker_0_maxpoolmax_indices dram malloc failed!");

        worker_0_maxpoolmax_indices_tr_info = synLaunchTensorInfo {"worker_0_maxpoolmax_indices", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_maxpoolmax_indices = createTensor(dims.size(),
                                                   syn_type_uint8,
                                                   dims.data(),
                                                   /*is_presist*/ true,
                                                   "worker_0_maxpoolmax_indices",
                                                   /*graphHandle*/ nullptr,
                                                   /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_maxpool_bwd_in_vec[2] = {layer1_0_residual_upstream_grad_input, worker_0_maxpoolmax_indices};


    // create worker_0_maxpool_grad_input tensor
    synTensor              worker_0_maxpool_grad_input {};
    synLaunchTensorInfo worker_0_maxpool_grad_input_tr_info {};
    {
        const std::array<unsigned, 4> dims  = {64, 112, 112, 64};
        const unsigned                bytes = prod(dims) * sizeof(float);

        uint64_t addr;
        status = hbmAlloc(bytes, &addr, "worker_0_maxpool_grad_input");
        ASSERT_TRUE(status == synSuccess && "worker_0_maxpool_grad_input dram malloc failed!");

        worker_0_maxpool_grad_input_tr_info = synLaunchTensorInfo {"worker_0_maxpool_grad_input", addr, DATA_TENSOR};
        // TODO: Should be perfectly fine to use addr instead of -1 for the noneager case??;
        worker_0_maxpool_grad_input = createTensor(dims.size(),
                                                   syn_type_single,
                                                   dims.data(),
                                                   /*is_presist*/ true,
                                                   "worker_0_maxpool_grad_input",
                                                   /*graphHandle*/ nullptr,
                                                   /*deviceAddr*/ eagerMode ? addr : -1);
    }

    synTensor worker_0_maxpool_bwd_out_vec[1] = {worker_0_maxpool_grad_input};


    status = synNodeCreate(graphHandle, worker_0_maxpool_bwd_in_vec, worker_0_maxpool_bwd_out_vec, 2, 1, (void *)&worker_0_maxpool_bwd_kernel_params, sizeof(worker_0_maxpool_bwd_kernel_params), "maxpool_2d_bwd_f32", "worker_0_maxpool_bwd", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_maxpool_bwd failed!");


    // generate graph
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(layer1_0_residual_upstream_grad_input_tr_info);
    graph_inputs.push_back(worker_0_maxpoolmax_indices_tr_info);

    graph_outputs.push_back(worker_0_maxpool_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Conv2D_fwd_eager)
{
    conv2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Conv2D_fwd_eager_ASIC_CI)
{
    conv2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_BatchNorm2D_fwd_eager_ASIC_CI)
{
    batchNorm2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_BatchNorm2D_fwd_eager)
{
    batchNorm2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_ReLU_fwd_eager)
{
    reluFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_ReLU_fwd_eager_ASIC_CI)
{
    reluFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_MaxPool2D_fwd_eager)
{
    maxPool2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_MaxPool2D_fwd_eager_ASIC_CI)
{
    maxPool2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_AvgPool2D_fwd_eager)
{
    avgPool2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_AvgPool2D_fwd_eager_ASIC_CI)
{
    avgPool2dFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_fwd_eager)
{
    linearFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_fwd_eager_ASIC_CI)
{
    linearFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_CrossEntropyLoss_log_softmax_eager)
{
    crossEntropyLossLogSoftmax();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_CrossEntropyLoss_log_softmax_eager_ASIC_CI)
{
    crossEntropyLossLogSoftmax();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_CrossEntropyLoss_log_softmax_bwd_eager)
{
    crossEntropyLossLogSoftmaxBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_CrossEntropyLoss_log_softmax_bwd_eager_ASIC_CI)
{
    crossEntropyLossLogSoftmaxBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_dedx_eager)
{
    linearDedx();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_dedx_eager_ASIC_CI)
{
    linearDedx();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_dedw_eager)
{
    linearDedw();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_dedw_eager_ASIC_CI)
{
    linearDedw();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_dedb_eager)
{
    linearDedb();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Linear_dedb_eager_ASIC_CI)
{
    linearDedb();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_AvgPool2D_bwd_eager)
{
    avgPool2dBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_AvgPool2D_bwd_eager_ASIC_CI)
{
    avgPool2dBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_ReLU_bwd_eager)
{
    reluBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_ReLU_bwd_eager_ASIC_CI)
{
    reluBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_add_residual_bwd_eager)
{
    addResidualBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_add_residual_bwd_eager_ASIC_CI)
{
    addResidualBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_BatchNorm2D_bwd_eager)
{
    batchNorm2dBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_BatchNorm2D_bwd_eager_ASIC_CI)
{
    batchNorm2dBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Conv2D_dedx_eager)
{
    conv2DDedx();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Conv2D_dedx_eager_ASIC_CI)
{
    conv2DDedx();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Conv2D_dedw_eager)
{
    conv2DDedw();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_Conv2D_dedw_eager_ASIC_CI)
{
    conv2DDedw();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_add_residual_fwd_eager)
{
    addResidualFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_add_residual_fwd_eager_ASIC_CI)
{
    addResidualFwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_MaxPool2D_bwd_eager)
{
    maxPool2dBwd();
}

TEST_F_GC(SynGaudi2ResNetTestFloat32, resnet50_MaxPool2D_bwd_eager_ASIC_CI)
{
    maxPool2dBwd();
}
