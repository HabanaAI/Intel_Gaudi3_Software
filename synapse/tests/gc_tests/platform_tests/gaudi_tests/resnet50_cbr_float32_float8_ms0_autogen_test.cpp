

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
TEST_F_GC(SynGaudi2ResNetFloat8Test, resnet50_cbr_float32_float8_ms0, {synDeviceGaudi2})
{
    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    synStatus status = synSuccess;
    clearDramMap();
    /*************
     * worker_0_conv1 node
     * inputs: [input[64, 224, 224, 3](dtype=fp8_152_t), worker_0_conv1_weight[7, 7, 3, 64](dtype=fp8_152_t)]
     * output: [worker_0_conv1_output(64, 112, 112, 64)(dtype=fp8_152_t)]
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
    const unsigned input_sizes[] = {64, 224, 224, 3};
    uint64_t input_dram;
    unsigned input_size = 64*224*224*3;
    unsigned input_size_in_bytes = input_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(input_size_in_bytes, &input_dram, "input");
    ASSERT_TRUE(status == synSuccess && "input dram malloc failed!");
    synLaunchTensorInfo input_tr_info = {"input", input_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor input = createTensor(4U, syn_type_fp8_152, input_sizes, true, "input");

    // create worker_0_conv1_weight tensor
    const unsigned worker_0_conv1_weight_sizes[] = {7, 7, 3, 64};
    uint64_t worker_0_conv1_weight_dram;
    unsigned worker_0_conv1_weight_size = 7*7*3*64;
    unsigned worker_0_conv1_weight_size_in_bytes = worker_0_conv1_weight_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(worker_0_conv1_weight_size_in_bytes, &worker_0_conv1_weight_dram, "worker_0_conv1_weight");
    ASSERT_TRUE(status == synSuccess && "worker_0_conv1_weight dram malloc failed!");
    synLaunchTensorInfo worker_0_conv1_weight_tr_info = {"worker_0_conv1_weight", worker_0_conv1_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_conv1_weight = createTensor(4U, syn_type_fp8_152, worker_0_conv1_weight_sizes, true, "worker_0_conv1_weight");

    synTensor worker_0_conv1_in_vec[4] = {input, worker_0_conv1_weight, nullptr, nullptr};


    // create worker_0_conv1_output tensor
    const unsigned worker_0_conv1_output_sizes[] = {64, 112, 112, 64};
    synTensor worker_0_conv1_output = createTensor(4U, syn_type_fp8_152, worker_0_conv1_output_sizes, false, "worker_0_conv1_output");

    synTensor worker_0_conv1_out_vec[1] = {worker_0_conv1_output};


    status = synNodeCreate(graphHandle, worker_0_conv1_in_vec, worker_0_conv1_out_vec, 4, 1, (void *)&worker_0_conv1_kernel_params, sizeof(worker_0_conv1_kernel_params), "spatial_convolution", "worker_0_conv1", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_conv1 failed!");

    /*************
     * worker_0_conv1_output_cast node
     * inputs: [worker_0_conv1_output(64, 112, 112, 64)(dtype=fp8_152_t)]
     * output: [worker_0_conv1_output_cast[64, 112, 112, 64](dtype=float32)]
     *************/

    synTensor worker_0_conv1_output_cast_in_vec[1] = {worker_0_conv1_output};


    // create worker_0_conv1_output_cast tensor
    const unsigned worker_0_conv1_output_cast_sizes[] = {64, 112, 112, 64};
    synTensor worker_0_conv1_output_cast = createTensor(4U, syn_type_single, worker_0_conv1_output_cast_sizes, false, "worker_0_conv1_output_cast");

    synTensor worker_0_conv1_output_cast_out_vec[1] = {worker_0_conv1_output_cast};


    status = synNodeCreate(graphHandle, worker_0_conv1_output_cast_in_vec, worker_0_conv1_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f8_to_f32", "worker_0_conv1_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_conv1_output_cast failed!");

    /*************
     * worker_0_bn1 node
     * inputs: [worker_0_conv1_output_cast[64, 112, 112, 64](dtype=float32), worker_0_bn1_bias[64](dtype=float32), worker_0_bn1_weight[64](dtype=float32), worker_0_bn1_running_mean[64](dtype=float32), worker_0_bn1_running_var[64](dtype=float32)]
     * output: [worker_0_bn1_output(64, 112, 112, 64)(dtype=float32), worker_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), worker_0_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params worker_0_bn1_kernel_params;
    worker_0_bn1_kernel_params.momentum = 0.1;
    worker_0_bn1_kernel_params.threshold.f = 1e-05;
    worker_0_bn1_kernel_params.epsilon = 1e-05;

    // create worker_0_bn1_bias tensor
    const unsigned worker_0_bn1_bias_sizes[] = {64,};
    uint64_t worker_0_bn1_bias_dram;
    unsigned worker_0_bn1_bias_size = 64;
    unsigned worker_0_bn1_bias_size_in_bytes = worker_0_bn1_bias_size * sizeof(float) ;
    status = hbmAlloc(worker_0_bn1_bias_size_in_bytes, &worker_0_bn1_bias_dram, "worker_0_bn1_bias");
    ASSERT_TRUE(status == synSuccess && "worker_0_bn1_bias dram malloc failed!");
    synLaunchTensorInfo worker_0_bn1_bias_tr_info = {"worker_0_bn1_bias", worker_0_bn1_bias_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_bn1_bias = createTensor(1U, syn_type_single, worker_0_bn1_bias_sizes, true, "worker_0_bn1_bias");

    // create worker_0_bn1_weight tensor
    const unsigned worker_0_bn1_weight_sizes[] = {64,};
    uint64_t worker_0_bn1_weight_dram;
    unsigned worker_0_bn1_weight_size = 64;
    unsigned worker_0_bn1_weight_size_in_bytes = worker_0_bn1_weight_size * sizeof(float) ;
    status = hbmAlloc(worker_0_bn1_weight_size_in_bytes, &worker_0_bn1_weight_dram, "worker_0_bn1_weight");
    ASSERT_TRUE(status == synSuccess && "worker_0_bn1_weight dram malloc failed!");
    synLaunchTensorInfo worker_0_bn1_weight_tr_info = {"worker_0_bn1_weight", worker_0_bn1_weight_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_bn1_weight = createTensor(1U, syn_type_single, worker_0_bn1_weight_sizes, true, "worker_0_bn1_weight");

    // create worker_0_bn1_running_mean tensor
    const unsigned worker_0_bn1_running_mean_sizes[] = {64,};
    uint64_t worker_0_bn1_running_mean_dram;
    unsigned worker_0_bn1_running_mean_size = 64;
    unsigned worker_0_bn1_running_mean_size_in_bytes = worker_0_bn1_running_mean_size * sizeof(float) ;
    status = hbmAlloc(worker_0_bn1_running_mean_size_in_bytes, &worker_0_bn1_running_mean_dram, "worker_0_bn1_running_mean");
    ASSERT_TRUE(status == synSuccess && "worker_0_bn1_running_mean dram malloc failed!");
    synLaunchTensorInfo worker_0_bn1_running_mean_tr_info = {"worker_0_bn1_running_mean", worker_0_bn1_running_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_bn1_running_mean = createTensor(1U, syn_type_single, worker_0_bn1_running_mean_sizes, true, "worker_0_bn1_running_mean");

    // create worker_0_bn1_running_var tensor
    const unsigned worker_0_bn1_running_var_sizes[] = {64,};
    uint64_t worker_0_bn1_running_var_dram;
    unsigned worker_0_bn1_running_var_size = 64;
    unsigned worker_0_bn1_running_var_size_in_bytes = worker_0_bn1_running_var_size * sizeof(float) ;
    status = hbmAlloc(worker_0_bn1_running_var_size_in_bytes, &worker_0_bn1_running_var_dram, "worker_0_bn1_running_var");
    ASSERT_TRUE(status == synSuccess && "worker_0_bn1_running_var dram malloc failed!");
    synLaunchTensorInfo worker_0_bn1_running_var_tr_info = {"worker_0_bn1_running_var", worker_0_bn1_running_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_bn1_running_var = createTensor(1U, syn_type_single, worker_0_bn1_running_var_sizes, true, "worker_0_bn1_running_var");

    synTensor worker_0_bn1_in_vec[5] = {worker_0_conv1_output_cast, worker_0_bn1_bias, worker_0_bn1_weight, worker_0_bn1_running_mean, worker_0_bn1_running_var};


    // create worker_0_bn1_output tensor
    const unsigned worker_0_bn1_output_sizes[] = {64, 112, 112, 64};
    synTensor worker_0_bn1_output = createTensor(4U, syn_type_single, worker_0_bn1_output_sizes, false, "worker_0_bn1_output");

    // create worker_0_bn1_saved_mean tensor
    const unsigned worker_0_bn1_saved_mean_sizes[] = {64};
    uint64_t worker_0_bn1_saved_mean_dram;
    unsigned worker_0_bn1_saved_mean_size = 1*1*1*64;
    unsigned worker_0_bn1_saved_mean_size_in_bytes = worker_0_bn1_saved_mean_size * sizeof(float) ;
    status = hbmAlloc(worker_0_bn1_saved_mean_size_in_bytes, &worker_0_bn1_saved_mean_dram, "worker_0_bn1_saved_mean");
    ASSERT_TRUE(status == synSuccess && "worker_0_bn1_saved_mean dram malloc failed!");
    synLaunchTensorInfo worker_0_bn1_saved_mean_tr_info = {"worker_0_bn1_saved_mean", worker_0_bn1_saved_mean_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_bn1_saved_mean = createTensor(1U, syn_type_single, worker_0_bn1_saved_mean_sizes, true, "worker_0_bn1_saved_mean");

    // create worker_0_bn1_saved_var tensor
    const unsigned worker_0_bn1_saved_var_sizes[] = {64};
    uint64_t worker_0_bn1_saved_var_dram;
    unsigned worker_0_bn1_saved_var_size = 1*1*1*64;
    unsigned worker_0_bn1_saved_var_size_in_bytes = worker_0_bn1_saved_var_size * sizeof(float) ;
    status = hbmAlloc(worker_0_bn1_saved_var_size_in_bytes, &worker_0_bn1_saved_var_dram, "worker_0_bn1_saved_var");
    ASSERT_TRUE(status == synSuccess && "worker_0_bn1_saved_var dram malloc failed!");
    synLaunchTensorInfo worker_0_bn1_saved_var_tr_info = {"worker_0_bn1_saved_var", worker_0_bn1_saved_var_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_bn1_saved_var = createTensor(1U, syn_type_single, worker_0_bn1_saved_var_sizes, true, "worker_0_bn1_saved_var");

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
    const unsigned worker_0_relu_output_sizes[] = {64, 112, 112, 64};
    synTensor worker_0_relu_output = createTensor(4U, syn_type_single, worker_0_relu_output_sizes, false, "worker_0_relu_output");

    synTensor worker_0_relu_out_vec[1] = {worker_0_relu_output};


    status = synNodeCreate(graphHandle, worker_0_relu_in_vec, worker_0_relu_out_vec, 1, 1, nullptr, 0, "relu_fwd_f32", "worker_0_relu", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_relu failed!");

    /*************
     * worker_0_relu_output_cast node
     * inputs: [worker_0_relu_output(64, 112, 112, 64)(dtype=float32)]
     * output: [worker_0_relu_output_cast(64, 112, 112, 64)(dtype=fp8_152_t)]
     *************/

    synTensor worker_0_relu_output_cast_in_vec[1] = {worker_0_relu_output};


    // create worker_0_relu_output_cast tensor
    const unsigned worker_0_relu_output_cast_sizes[] = {64, 112, 112, 64};
    uint64_t worker_0_relu_output_cast_dram;
    unsigned worker_0_relu_output_cast_size = 64*112*112*64;
    unsigned worker_0_relu_output_cast_size_in_bytes = worker_0_relu_output_cast_size * sizeof(fp8_152_t) ;
    status = hbmAlloc(worker_0_relu_output_cast_size_in_bytes, &worker_0_relu_output_cast_dram, "worker_0_relu_output_cast");
    ASSERT_TRUE(status == synSuccess && "worker_0_relu_output_cast dram malloc failed!");
    synLaunchTensorInfo worker_0_relu_output_cast_tr_info = {"worker_0_relu_output_cast", worker_0_relu_output_cast_dram, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synTensor worker_0_relu_output_cast = createTensor(4U, syn_type_fp8_152, worker_0_relu_output_cast_sizes, true, "worker_0_relu_output_cast");

    synTensor worker_0_relu_output_cast_out_vec[1] = {worker_0_relu_output_cast};


    status = synNodeCreate(graphHandle, worker_0_relu_output_cast_in_vec, worker_0_relu_output_cast_out_vec, 1, 1, nullptr, 0, "cast_f32_to_f8", "worker_0_relu_output_cast", nullptr, nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_relu_output_cast failed!");


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
    graph_outputs.push_back(worker_0_relu_output_cast_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs, true /* skip validation */);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}
