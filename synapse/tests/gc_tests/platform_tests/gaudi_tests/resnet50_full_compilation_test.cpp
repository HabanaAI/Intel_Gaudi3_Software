#include "synapse_api.h"
#include <vector>
#include "../gaudi_tests/gc_resnet_demo_test.h"
#include "synapse_common_types.h"

class SynTrainingResNetCompilationTest : public SynTrainingResNetTest
{
protected:
    SynTrainingResNetCompilationTest()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi2");
            m_deviceType = synDeviceGaudi2;
        }
    }
    void SetUpTest() override;
    void compileGraph(synGraphHandle graphHandle);
};

void SynTrainingResNetCompilationTest::compileGraph(synGraphHandle graphHandle)
{
    LaunchInfo launchInfo;
    synStatus  status;
    UNUSED(status);

    LOG_DEBUG(SYN_TEST, "Compiling {}...", GetTestFileName().c_str());

    status = synGraphCompile(&launchInfo.m_recipeHandle, graphHandle, GetTestFileName().c_str(), nullptr);
    ASSERT_EQ(status, synSuccess) << "Synapse graph compilation failed!";

    if (GCFG_PRESERVE_TESTS_RECIPE.value())
    {
        status = synRecipeSerialize(launchInfo.m_recipeHandle, GetTestFileName().c_str());
        ASSERT_EQ(status, synSuccess) << "Failed to serialize recipe";
    }

    status = synGraphDestroy(graphHandle);
    ASSERT_EQ(status, synSuccess) << "Synapse graph destruction failed!";
}

void SynTrainingResNetCompilationTest::SetUpTest()
{
    SynTrainingResNetTest::SetUpTest();
}

/* This is a temporary test for gaudi2 resnet50 compilation */
/* This test is taken from gaudi1 automated tests           */
TEST_F_GC(SynTrainingResNetCompilationTest, resnet50_training_compilation_test)
{
    synGraphHandle graphHandle;
    synStatus      status = synGraphCreate(&graphHandle, m_deviceType);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");

    synConfigurationSet("SB_REUSE", "0");

    clearDramMap();
    /*************
     * worker_0_conv1 node
     * inputs: [input[64, 224, 224, 3](dtype=bf16), worker_0_conv1_weight[7, 7, 3, 64](dtype=bf16)]
     * output: [worker_0_conv1_output(64, 112, 112, 64)(dtype=bf16)]
     *************/
    synConvolutionParams worker_0_conv1_kernel_params;
    worker_0_conv1_kernel_params.dH   = 2;
    worker_0_conv1_kernel_params.dW   = 2;
    worker_0_conv1_kernel_params.kH   = 7;
    worker_0_conv1_kernel_params.kW   = 7;
    worker_0_conv1_kernel_params.padT = 3;
    worker_0_conv1_kernel_params.padB = 3;
    worker_0_conv1_kernel_params.padL = 3;
    worker_0_conv1_kernel_params.padR = 3;
    worker_0_conv1_kernel_params.dilH = 1;
    worker_0_conv1_kernel_params.dilW = 1;

    // create input tensor
    const unsigned input_sizes[] = {64, 224, 224, 3};
    uint64_t       input_dram    = 0;

    synLaunchTensorInfo input_tr_info = {"input", input_dram};
    synTensor           input         = createTensor(4U, syn_type_bf16, input_sizes, true, "input");

    // create worker_0_conv1_weight tensor
    const unsigned      worker_0_conv1_weight_sizes[] = {7, 7, 3, 64};
    uint64_t            worker_0_conv1_weight_dram    = 0;
    synLaunchTensorInfo worker_0_conv1_weight_tr_info = {"worker_0_conv1_weight", worker_0_conv1_weight_dram};
    synTensor           worker_0_conv1_weight =
        createTensor(4U, syn_type_bf16, worker_0_conv1_weight_sizes, true, "worker_0_conv1_weight");

    synTensor worker_0_conv1_in_vec[4] = {input, worker_0_conv1_weight, nullptr, nullptr};

    // create worker_0_conv1_output tensor
    const unsigned worker_0_conv1_output_sizes[] = {64, 112, 112, 64};
    synTensor      worker_0_conv1_output =
        createTensor(4U, syn_type_bf16, worker_0_conv1_output_sizes, false, "worker_0_conv1_output");

    synTensor worker_0_conv1_out_vec[1] = {worker_0_conv1_output};

    status = synNodeCreate(graphHandle,
                           worker_0_conv1_in_vec,
                           worker_0_conv1_out_vec,
                           4,
                           1,
                           (void*)&worker_0_conv1_kernel_params,
                           sizeof(worker_0_conv1_kernel_params),
                           "spatial_convolution",
                           "worker_0_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_conv1 failed!");

    /*************
     * worker_0_bn1 node
     * inputs: [worker_0_conv1_output(64, 112, 112, 64)(dtype=bf16), worker_0_bn1_bias[64](dtype=float32),
     *worker_0_bn1_weight[64](dtype=float32), worker_0_bn1_running_mean[64](dtype=float32),
     *worker_0_bn1_running_var[64](dtype=float32)] output: [worker_0_bn1_output(64, 112, 112, 64)(dtype=bf16),
     *worker_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), worker_0_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params worker_0_bn1_kernel_params;
    worker_0_bn1_kernel_params.momentum    = 0.1;
    worker_0_bn1_kernel_params.threshold.f = 1e-05;
    worker_0_bn1_kernel_params.epsilon     = 1e-05;

    // create worker_0_bn1_bias tensor
    const unsigned worker_0_bn1_bias_sizes[] = {
        64,
    };
    uint64_t            worker_0_bn1_bias_dram    = 0;
    synLaunchTensorInfo worker_0_bn1_bias_tr_info = {"worker_0_bn1_bias", worker_0_bn1_bias_dram};
    synTensor worker_0_bn1_bias = createTensor(1U, syn_type_single, worker_0_bn1_bias_sizes, true, "worker_0_bn1_bias");

    // create worker_0_bn1_weight tensor
    const unsigned worker_0_bn1_weight_sizes[] = {
        64,
    };
    uint64_t            worker_0_bn1_weight_dram    = 0;
    synLaunchTensorInfo worker_0_bn1_weight_tr_info = {"worker_0_bn1_weight", worker_0_bn1_weight_dram};
    synTensor           worker_0_bn1_weight =
        createTensor(1U, syn_type_single, worker_0_bn1_weight_sizes, true, "worker_0_bn1_weight");

    // create worker_0_bn1_running_mean tensor
    const unsigned worker_0_bn1_running_mean_sizes[] = {
        64,
    };
    uint64_t            worker_0_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo worker_0_bn1_running_mean_tr_info = {"worker_0_bn1_running_mean",
                                                             worker_0_bn1_running_mean_dram};
    synTensor           worker_0_bn1_running_mean =
        createTensor(1U, syn_type_single, worker_0_bn1_running_mean_sizes, true, "worker_0_bn1_running_mean");

    // create worker_0_bn1_running_var tensor
    const unsigned worker_0_bn1_running_var_sizes[] = {
        64,
    };
    uint64_t            worker_0_bn1_running_var_dram    = 0;
    synLaunchTensorInfo worker_0_bn1_running_var_tr_info = {"worker_0_bn1_running_var", worker_0_bn1_running_var_dram};
    synTensor           worker_0_bn1_running_var =
        createTensor(1U, syn_type_single, worker_0_bn1_running_var_sizes, true, "worker_0_bn1_running_var");

    synTensor worker_0_bn1_in_vec[5] = {worker_0_conv1_output,
                                        worker_0_bn1_bias,
                                        worker_0_bn1_weight,
                                        worker_0_bn1_running_mean,
                                        worker_0_bn1_running_var};

    // create worker_0_bn1_output tensor
    const unsigned worker_0_bn1_output_sizes[] = {64, 112, 112, 64};
    synTensor      worker_0_bn1_output =
        createTensor(4U, syn_type_bf16, worker_0_bn1_output_sizes, false, "worker_0_bn1_output");

    // create worker_0_bn1_saved_mean tensor
    const unsigned worker_0_bn1_saved_mean_sizes[] = {64};
    synTensor      worker_0_bn1_saved_mean =
        createTensor(1U, syn_type_single, worker_0_bn1_saved_mean_sizes, false, "worker_0_bn1_saved_mean");

    // create worker_0_bn1_saved_var tensor
    const unsigned worker_0_bn1_saved_var_sizes[] = {64};
    synTensor      worker_0_bn1_saved_var =
        createTensor(1U, syn_type_single, worker_0_bn1_saved_var_sizes, false, "worker_0_bn1_saved_var");

    synTensor worker_0_bn1_out_vec[3] = {worker_0_bn1_output, worker_0_bn1_saved_mean, worker_0_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           worker_0_bn1_in_vec,
                           worker_0_bn1_out_vec,
                           5,
                           3,
                           (void*)&worker_0_bn1_kernel_params,
                           sizeof(worker_0_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "worker_0_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_bn1 failed!");

    /*************
     * worker_0_relu node
     * inputs: [worker_0_bn1_output(64, 112, 112, 64)(dtype=bf16)]
     * output: [worker_0_relu_output(64, 112, 112, 64)(dtype=bf16)]
     *************/

    synTensor worker_0_relu_in_vec[1] = {worker_0_bn1_output};

    // create worker_0_relu_output tensor
    const unsigned worker_0_relu_output_sizes[] = {64, 112, 112, 64};
    synTensor      worker_0_relu_output =
        createTensor(4U, syn_type_bf16, worker_0_relu_output_sizes, false, "worker_0_relu_output");

    synTensor worker_0_relu_out_vec[1] = {worker_0_relu_output};

    status = synNodeCreate(graphHandle,
                           worker_0_relu_in_vec,
                           worker_0_relu_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "worker_0_relu",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_relu failed!");

    /*************
     * worker_0_maxpool node
     * inputs: [worker_0_relu_output(64, 112, 112, 64)(dtype=bf16)]
     * output: [worker_0_maxpoolmax_indices(64, 56, 56, 64)(dtype=uint16), worker_0_maxpool_output(64, 56, 56,
     *64)(dtype=bf16)]
     *************/
    ns_SpatialReduction::Params worker_0_maxpool_kernel_params;
    worker_0_maxpool_kernel_params.kernel_w           = 3;
    worker_0_maxpool_kernel_params.kernel_h           = 3;
    worker_0_maxpool_kernel_params.stride_w           = 2;
    worker_0_maxpool_kernel_params.stride_h           = 2;
    worker_0_maxpool_kernel_params.pad_w_begin        = 1;
    worker_0_maxpool_kernel_params.pad_w_end          = 1;
    worker_0_maxpool_kernel_params.pad_h_begin        = 1;
    worker_0_maxpool_kernel_params.pad_h_end          = 1;
    worker_0_maxpool_kernel_params.dilation_w         = 1;
    worker_0_maxpool_kernel_params.dilation_h         = 1;
    worker_0_maxpool_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    synTensor worker_0_maxpool_in_vec[1] = {worker_0_relu_output};

    // create worker_0_maxpoolmax_indices tensor
    const unsigned worker_0_maxpoolmax_indices_sizes[] = {64, 56, 56, 64};
    synTensor      worker_0_maxpoolmax_indices =
        createTensor(4U, syn_type_int16, worker_0_maxpoolmax_indices_sizes, false, "worker_0_maxpoolmax_indices");

    // create worker_0_maxpool_output tensor
    const unsigned worker_0_maxpool_output_sizes[] = {64, 56, 56, 64};
    synTensor      worker_0_maxpool_output =
        createTensor(4U, syn_type_bf16, worker_0_maxpool_output_sizes, false, "worker_0_maxpool_output");

    synTensor worker_0_maxpool_out_vec[2] = {worker_0_maxpoolmax_indices, worker_0_maxpool_output};

    status = synNodeCreate(graphHandle,
                           worker_0_maxpool_in_vec,
                           worker_0_maxpool_out_vec,
                           1,
                           2,
                           (void*)&worker_0_maxpool_kernel_params,
                           sizeof(worker_0_maxpool_kernel_params),
                           "maxpool_2d_fwd_bf16",
                           "worker_0_maxpool",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_maxpool failed!");

    /*************
     * layer1_0_conv1 node
     * inputs: [worker_0_maxpool_output(64, 56, 56, 64)(dtype=bf16), layer1_0_conv1_weight[1, 1, 64, 64](dtype=bf16)]
     * output: [layer1_0_conv1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_0_conv1_kernel_params;
    layer1_0_conv1_kernel_params.dH   = 1;
    layer1_0_conv1_kernel_params.dW   = 1;
    layer1_0_conv1_kernel_params.kH   = 1;
    layer1_0_conv1_kernel_params.kW   = 1;
    layer1_0_conv1_kernel_params.padT = 0;
    layer1_0_conv1_kernel_params.padB = 0;
    layer1_0_conv1_kernel_params.padL = 0;
    layer1_0_conv1_kernel_params.padR = 0;
    layer1_0_conv1_kernel_params.dilH = 1;
    layer1_0_conv1_kernel_params.dilW = 1;

    // create layer1_0_conv1_weight tensor
    const unsigned      layer1_0_conv1_weight_sizes[] = {1, 1, 64, 64};
    uint64_t            layer1_0_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer1_0_conv1_weight_tr_info = {"layer1_0_conv1_weight", layer1_0_conv1_weight_dram};
    synTensor           layer1_0_conv1_weight =
        createTensor(4U, syn_type_bf16, layer1_0_conv1_weight_sizes, true, "layer1_0_conv1_weight");

    synTensor layer1_0_conv1_in_vec[4] = {worker_0_maxpool_output, layer1_0_conv1_weight, nullptr, nullptr};

    // create layer1_0_conv1_output tensor
    const unsigned layer1_0_conv1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_conv1_output =
        createTensor(4U, syn_type_bf16, layer1_0_conv1_output_sizes, false, "layer1_0_conv1_output");

    synTensor layer1_0_conv1_out_vec[1] = {layer1_0_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv1_in_vec,
                           layer1_0_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer1_0_conv1_kernel_params,
                           sizeof(layer1_0_conv1_kernel_params),
                           "spatial_convolution",
                           "layer1_0_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv1 failed!");

    /*************
     * layer1_0_bn1 node
     * inputs: [layer1_0_conv1_output(64, 56, 56, 64)(dtype=bf16), layer1_0_bn1_bias[64](dtype=float32),
     *layer1_0_bn1_weight[64](dtype=float32), layer1_0_bn1_running_mean[64](dtype=float32),
     *layer1_0_bn1_running_var[64](dtype=float32)] output: [layer1_0_bn1_output(64, 56, 56, 64)(dtype=bf16),
     *layer1_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn1_kernel_params;
    layer1_0_bn1_kernel_params.momentum    = 0.1;
    layer1_0_bn1_kernel_params.threshold.f = 1e-05;
    layer1_0_bn1_kernel_params.epsilon     = 1e-05;

    // create layer1_0_bn1_bias tensor
    const unsigned layer1_0_bn1_bias_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer1_0_bn1_bias_tr_info = {"layer1_0_bn1_bias", layer1_0_bn1_bias_dram};
    synTensor layer1_0_bn1_bias = createTensor(1U, syn_type_single, layer1_0_bn1_bias_sizes, true, "layer1_0_bn1_bias");

    // create layer1_0_bn1_weight tensor
    const unsigned layer1_0_bn1_weight_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer1_0_bn1_weight_tr_info = {"layer1_0_bn1_weight", layer1_0_bn1_weight_dram};
    synTensor           layer1_0_bn1_weight =
        createTensor(1U, syn_type_single, layer1_0_bn1_weight_sizes, true, "layer1_0_bn1_weight");

    // create layer1_0_bn1_running_mean tensor
    const unsigned layer1_0_bn1_running_mean_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_0_bn1_running_mean_tr_info = {"layer1_0_bn1_running_mean",
                                                             layer1_0_bn1_running_mean_dram};
    synTensor           layer1_0_bn1_running_mean =
        createTensor(1U, syn_type_single, layer1_0_bn1_running_mean_sizes, true, "layer1_0_bn1_running_mean");

    // create layer1_0_bn1_running_var tensor
    const unsigned layer1_0_bn1_running_var_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer1_0_bn1_running_var_tr_info = {"layer1_0_bn1_running_var", layer1_0_bn1_running_var_dram};
    synTensor           layer1_0_bn1_running_var =
        createTensor(1U, syn_type_single, layer1_0_bn1_running_var_sizes, true, "layer1_0_bn1_running_var");

    synTensor layer1_0_bn1_in_vec[5] = {layer1_0_conv1_output,
                                        layer1_0_bn1_bias,
                                        layer1_0_bn1_weight,
                                        layer1_0_bn1_running_mean,
                                        layer1_0_bn1_running_var};

    // create layer1_0_bn1_output tensor
    const unsigned layer1_0_bn1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_bn1_output =
        createTensor(4U, syn_type_bf16, layer1_0_bn1_output_sizes, false, "layer1_0_bn1_output");

    // create layer1_0_bn1_saved_mean tensor
    const unsigned layer1_0_bn1_saved_mean_sizes[] = {64};
    synTensor      layer1_0_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer1_0_bn1_saved_mean_sizes, false, "layer1_0_bn1_saved_mean");

    // create layer1_0_bn1_saved_var tensor
    const unsigned layer1_0_bn1_saved_var_sizes[] = {64};
    synTensor      layer1_0_bn1_saved_var =
        createTensor(1U, syn_type_single, layer1_0_bn1_saved_var_sizes, false, "layer1_0_bn1_saved_var");

    synTensor layer1_0_bn1_out_vec[3] = {layer1_0_bn1_output, layer1_0_bn1_saved_mean, layer1_0_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_0_bn1_in_vec,
                           layer1_0_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer1_0_bn1_kernel_params,
                           sizeof(layer1_0_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_0_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn1 failed!");

    /*************
     * layer1_0_relu1 node
     * inputs: [layer1_0_bn1_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_0_relu1_in_vec[1] = {layer1_0_bn1_output};

    // create layer1_0_relu1_output tensor
    const unsigned layer1_0_relu1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_relu1_output =
        createTensor(4U, syn_type_bf16, layer1_0_relu1_output_sizes, false, "layer1_0_relu1_output");

    synTensor layer1_0_relu1_out_vec[1] = {layer1_0_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer1_0_relu1_in_vec,
                           layer1_0_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_0_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu1 failed!");

    /*************
     * layer1_0_conv2 node
     * inputs: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=bf16), layer1_0_conv2_weight[3, 3, 64, 64](dtype=bf16)]
     * output: [layer1_0_conv2_output(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_0_conv2_kernel_params;
    layer1_0_conv2_kernel_params.dH   = 1;
    layer1_0_conv2_kernel_params.dW   = 1;
    layer1_0_conv2_kernel_params.kH   = 3;
    layer1_0_conv2_kernel_params.kW   = 3;
    layer1_0_conv2_kernel_params.padT = 1;
    layer1_0_conv2_kernel_params.padB = 1;
    layer1_0_conv2_kernel_params.padL = 1;
    layer1_0_conv2_kernel_params.padR = 1;
    layer1_0_conv2_kernel_params.dilH = 1;
    layer1_0_conv2_kernel_params.dilW = 1;

    // create layer1_0_conv2_weight tensor
    const unsigned      layer1_0_conv2_weight_sizes[] = {3, 3, 64, 64};
    uint64_t            layer1_0_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer1_0_conv2_weight_tr_info = {"layer1_0_conv2_weight", layer1_0_conv2_weight_dram};
    synTensor           layer1_0_conv2_weight =
        createTensor(4U, syn_type_bf16, layer1_0_conv2_weight_sizes, true, "layer1_0_conv2_weight");

    synTensor layer1_0_conv2_in_vec[4] = {layer1_0_relu1_output, layer1_0_conv2_weight, nullptr, nullptr};

    // create layer1_0_conv2_output tensor
    const unsigned layer1_0_conv2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_conv2_output =
        createTensor(4U, syn_type_bf16, layer1_0_conv2_output_sizes, false, "layer1_0_conv2_output");

    synTensor layer1_0_conv2_out_vec[1] = {layer1_0_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv2_in_vec,
                           layer1_0_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer1_0_conv2_kernel_params,
                           sizeof(layer1_0_conv2_kernel_params),
                           "spatial_convolution",
                           "layer1_0_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv2 failed!");

    /*************
     * layer1_0_bn2 node
     * inputs: [layer1_0_conv2_output(64, 56, 56, 64)(dtype=bf16), layer1_0_bn2_bias[64](dtype=float32),
     *layer1_0_bn2_weight[64](dtype=float32), layer1_0_bn2_running_mean[64](dtype=float32),
     *layer1_0_bn2_running_var[64](dtype=float32)] output: [layer1_0_bn2_output(64, 56, 56, 64)(dtype=bf16),
     *layer1_0_bn2_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn2_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn2_kernel_params;
    layer1_0_bn2_kernel_params.momentum    = 0.1;
    layer1_0_bn2_kernel_params.threshold.f = 1e-05;
    layer1_0_bn2_kernel_params.epsilon     = 1e-05;

    // create layer1_0_bn2_bias tensor
    const unsigned layer1_0_bn2_bias_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer1_0_bn2_bias_tr_info = {"layer1_0_bn2_bias", layer1_0_bn2_bias_dram};
    synTensor layer1_0_bn2_bias = createTensor(1U, syn_type_single, layer1_0_bn2_bias_sizes, true, "layer1_0_bn2_bias");

    // create layer1_0_bn2_weight tensor
    const unsigned layer1_0_bn2_weight_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer1_0_bn2_weight_tr_info = {"layer1_0_bn2_weight", layer1_0_bn2_weight_dram};
    synTensor           layer1_0_bn2_weight =
        createTensor(1U, syn_type_single, layer1_0_bn2_weight_sizes, true, "layer1_0_bn2_weight");

    // create layer1_0_bn2_running_mean tensor
    const unsigned layer1_0_bn2_running_mean_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_0_bn2_running_mean_tr_info = {"layer1_0_bn2_running_mean",
                                                             layer1_0_bn2_running_mean_dram};
    synTensor           layer1_0_bn2_running_mean =
        createTensor(1U, syn_type_single, layer1_0_bn2_running_mean_sizes, true, "layer1_0_bn2_running_mean");

    // create layer1_0_bn2_running_var tensor
    const unsigned layer1_0_bn2_running_var_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer1_0_bn2_running_var_tr_info = {"layer1_0_bn2_running_var", layer1_0_bn2_running_var_dram};
    synTensor           layer1_0_bn2_running_var =
        createTensor(1U, syn_type_single, layer1_0_bn2_running_var_sizes, true, "layer1_0_bn2_running_var");

    synTensor layer1_0_bn2_in_vec[5] = {layer1_0_conv2_output,
                                        layer1_0_bn2_bias,
                                        layer1_0_bn2_weight,
                                        layer1_0_bn2_running_mean,
                                        layer1_0_bn2_running_var};

    // create layer1_0_bn2_output tensor
    const unsigned layer1_0_bn2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_bn2_output =
        createTensor(4U, syn_type_bf16, layer1_0_bn2_output_sizes, false, "layer1_0_bn2_output");

    // create layer1_0_bn2_saved_mean tensor
    const unsigned layer1_0_bn2_saved_mean_sizes[] = {64};
    synTensor      layer1_0_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer1_0_bn2_saved_mean_sizes, false, "layer1_0_bn2_saved_mean");

    // create layer1_0_bn2_saved_var tensor
    const unsigned layer1_0_bn2_saved_var_sizes[] = {64};
    synTensor      layer1_0_bn2_saved_var =
        createTensor(1U, syn_type_single, layer1_0_bn2_saved_var_sizes, false, "layer1_0_bn2_saved_var");

    synTensor layer1_0_bn2_out_vec[3] = {layer1_0_bn2_output, layer1_0_bn2_saved_mean, layer1_0_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_0_bn2_in_vec,
                           layer1_0_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer1_0_bn2_kernel_params,
                           sizeof(layer1_0_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_0_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn2 failed!");

    /*************
     * layer1_0_relu2 node
     * inputs: [layer1_0_bn2_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_0_relu2_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_0_relu2_in_vec[1] = {layer1_0_bn2_output};

    // create layer1_0_relu2_output tensor
    const unsigned layer1_0_relu2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_relu2_output =
        createTensor(4U, syn_type_bf16, layer1_0_relu2_output_sizes, false, "layer1_0_relu2_output");

    synTensor layer1_0_relu2_out_vec[1] = {layer1_0_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer1_0_relu2_in_vec,
                           layer1_0_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_0_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu2 failed!");

    /*************
     * layer1_0_conv3 node
     * inputs: [layer1_0_relu2_output(64, 56, 56, 64)(dtype=bf16), layer1_0_conv3_weight[1, 1, 64, 256](dtype=bf16)]
     * output: [layer1_0_conv3_output(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_0_conv3_kernel_params;
    layer1_0_conv3_kernel_params.dH   = 1;
    layer1_0_conv3_kernel_params.dW   = 1;
    layer1_0_conv3_kernel_params.kH   = 1;
    layer1_0_conv3_kernel_params.kW   = 1;
    layer1_0_conv3_kernel_params.padT = 0;
    layer1_0_conv3_kernel_params.padB = 0;
    layer1_0_conv3_kernel_params.padL = 0;
    layer1_0_conv3_kernel_params.padR = 0;
    layer1_0_conv3_kernel_params.dilH = 1;
    layer1_0_conv3_kernel_params.dilW = 1;

    // create layer1_0_conv3_weight tensor
    const unsigned      layer1_0_conv3_weight_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_0_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer1_0_conv3_weight_tr_info = {"layer1_0_conv3_weight", layer1_0_conv3_weight_dram};
    synTensor           layer1_0_conv3_weight =
        createTensor(4U, syn_type_bf16, layer1_0_conv3_weight_sizes, true, "layer1_0_conv3_weight");

    synTensor layer1_0_conv3_in_vec[4] = {layer1_0_relu2_output, layer1_0_conv3_weight, nullptr, nullptr};

    // create layer1_0_conv3_output tensor
    const unsigned layer1_0_conv3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_conv3_output =
        createTensor(4U, syn_type_bf16, layer1_0_conv3_output_sizes, false, "layer1_0_conv3_output");

    synTensor layer1_0_conv3_out_vec[1] = {layer1_0_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv3_in_vec,
                           layer1_0_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer1_0_conv3_kernel_params,
                           sizeof(layer1_0_conv3_kernel_params),
                           "spatial_convolution",
                           "layer1_0_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv3 failed!");

    /*************
     * layer1_0_bn3 node
     * inputs: [layer1_0_conv3_output(64, 56, 56, 256)(dtype=bf16), layer1_0_bn3_bias[256](dtype=float32),
     *layer1_0_bn3_weight[256](dtype=float32), layer1_0_bn3_running_mean[256](dtype=float32),
     *layer1_0_bn3_running_var[256](dtype=float32)] output: [layer1_0_bn3_output(64, 56, 56, 256)(dtype=bf16),
     *layer1_0_bn3_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_0_bn3_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn3_kernel_params;
    layer1_0_bn3_kernel_params.momentum    = 0.1;
    layer1_0_bn3_kernel_params.threshold.f = 1e-05;
    layer1_0_bn3_kernel_params.epsilon     = 1e-05;

    // create layer1_0_bn3_bias tensor
    const unsigned layer1_0_bn3_bias_sizes[] = {
        256,
    };
    uint64_t            layer1_0_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer1_0_bn3_bias_tr_info = {"layer1_0_bn3_bias", layer1_0_bn3_bias_dram};
    synTensor layer1_0_bn3_bias = createTensor(1U, syn_type_single, layer1_0_bn3_bias_sizes, true, "layer1_0_bn3_bias");

    // create layer1_0_bn3_weight tensor
    const unsigned layer1_0_bn3_weight_sizes[] = {
        256,
    };
    uint64_t            layer1_0_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer1_0_bn3_weight_tr_info = {"layer1_0_bn3_weight", layer1_0_bn3_weight_dram};
    synTensor           layer1_0_bn3_weight =
        createTensor(1U, syn_type_single, layer1_0_bn3_weight_sizes, true, "layer1_0_bn3_weight");

    // create layer1_0_bn3_running_mean tensor
    const unsigned layer1_0_bn3_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer1_0_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_0_bn3_running_mean_tr_info = {"layer1_0_bn3_running_mean",
                                                             layer1_0_bn3_running_mean_dram};
    synTensor           layer1_0_bn3_running_mean =
        createTensor(1U, syn_type_single, layer1_0_bn3_running_mean_sizes, true, "layer1_0_bn3_running_mean");

    // create layer1_0_bn3_running_var tensor
    const unsigned layer1_0_bn3_running_var_sizes[] = {
        256,
    };
    uint64_t            layer1_0_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer1_0_bn3_running_var_tr_info = {"layer1_0_bn3_running_var", layer1_0_bn3_running_var_dram};
    synTensor           layer1_0_bn3_running_var =
        createTensor(1U, syn_type_single, layer1_0_bn3_running_var_sizes, true, "layer1_0_bn3_running_var");

    synTensor layer1_0_bn3_in_vec[5] = {layer1_0_conv3_output,
                                        layer1_0_bn3_bias,
                                        layer1_0_bn3_weight,
                                        layer1_0_bn3_running_mean,
                                        layer1_0_bn3_running_var};

    // create layer1_0_bn3_output tensor
    const unsigned layer1_0_bn3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_bn3_output =
        createTensor(4U, syn_type_bf16, layer1_0_bn3_output_sizes, false, "layer1_0_bn3_output");

    // create layer1_0_bn3_saved_mean tensor
    const unsigned layer1_0_bn3_saved_mean_sizes[] = {256};
    synTensor      layer1_0_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer1_0_bn3_saved_mean_sizes, false, "layer1_0_bn3_saved_mean");

    // create layer1_0_bn3_saved_var tensor
    const unsigned layer1_0_bn3_saved_var_sizes[] = {256};
    synTensor      layer1_0_bn3_saved_var =
        createTensor(1U, syn_type_single, layer1_0_bn3_saved_var_sizes, false, "layer1_0_bn3_saved_var");

    synTensor layer1_0_bn3_out_vec[3] = {layer1_0_bn3_output, layer1_0_bn3_saved_mean, layer1_0_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_0_bn3_in_vec,
                           layer1_0_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer1_0_bn3_kernel_params,
                           sizeof(layer1_0_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_0_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn3 failed!");

    /*************
     * layer1_downsample node
     * inputs: [worker_0_maxpool_output(64, 56, 56, 64)(dtype=bf16), layer1_downsample_weight[1, 1, 64,
     *256](dtype=bf16)] output: [layer1_downsample_output(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_downsample_kernel_params;
    layer1_downsample_kernel_params.dH   = 1;
    layer1_downsample_kernel_params.dW   = 1;
    layer1_downsample_kernel_params.kH   = 1;
    layer1_downsample_kernel_params.kW   = 1;
    layer1_downsample_kernel_params.padT = 0;
    layer1_downsample_kernel_params.padB = 0;
    layer1_downsample_kernel_params.padL = 0;
    layer1_downsample_kernel_params.padR = 0;
    layer1_downsample_kernel_params.dilH = 1;
    layer1_downsample_kernel_params.dilW = 1;

    // create layer1_downsample_weight tensor
    const unsigned      layer1_downsample_weight_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_downsample_weight_dram    = 0;
    synLaunchTensorInfo layer1_downsample_weight_tr_info = {"layer1_downsample_weight", layer1_downsample_weight_dram};
    synTensor           layer1_downsample_weight =
        createTensor(4U, syn_type_bf16, layer1_downsample_weight_sizes, true, "layer1_downsample_weight");

    synTensor layer1_downsample_in_vec[4] = {worker_0_maxpool_output, layer1_downsample_weight, nullptr, nullptr};

    // create layer1_downsample_output tensor
    const unsigned layer1_downsample_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_downsample_output =
        createTensor(4U, syn_type_bf16, layer1_downsample_output_sizes, false, "layer1_downsample_output");

    synTensor layer1_downsample_out_vec[1] = {layer1_downsample_output};

    status = synNodeCreate(graphHandle,
                           layer1_downsample_in_vec,
                           layer1_downsample_out_vec,
                           4,
                           1,
                           (void*)&layer1_downsample_kernel_params,
                           sizeof(layer1_downsample_kernel_params),
                           "spatial_convolution",
                           "layer1_downsample",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_downsample failed!");

    /*************
     * layer1_bn node
     * inputs: [layer1_downsample_output(64, 56, 56, 256)(dtype=bf16), layer1_bn_bias[256](dtype=float32),
     *layer1_bn_weight[256](dtype=float32), layer1_bn_running_mean[256](dtype=float32),
     *layer1_bn_running_var[256](dtype=float32)] output: [layer1_bn_output(64, 56, 56, 256)(dtype=bf16),
     *layer1_bn_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_bn_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_bn_kernel_params;
    layer1_bn_kernel_params.momentum    = 0.1;
    layer1_bn_kernel_params.threshold.f = 1e-05;
    layer1_bn_kernel_params.epsilon     = 1e-05;

    // create layer1_bn_bias tensor
    const unsigned layer1_bn_bias_sizes[] = {
        256,
    };
    uint64_t            layer1_bn_bias_dram    = 0;
    synLaunchTensorInfo layer1_bn_bias_tr_info = {"layer1_bn_bias", layer1_bn_bias_dram};
    synTensor layer1_bn_bias = createTensor(1U, syn_type_single, layer1_bn_bias_sizes, true, "layer1_bn_bias");

    // create layer1_bn_weight tensor
    const unsigned layer1_bn_weight_sizes[] = {
        256,
    };
    uint64_t            layer1_bn_weight_dram    = 0;
    synLaunchTensorInfo layer1_bn_weight_tr_info = {"layer1_bn_weight", layer1_bn_weight_dram};
    synTensor layer1_bn_weight = createTensor(1U, syn_type_single, layer1_bn_weight_sizes, true, "layer1_bn_weight");

    // create layer1_bn_running_mean tensor
    const unsigned layer1_bn_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer1_bn_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_bn_running_mean_tr_info = {"layer1_bn_running_mean", layer1_bn_running_mean_dram};
    synTensor           layer1_bn_running_mean =
        createTensor(1U, syn_type_single, layer1_bn_running_mean_sizes, true, "layer1_bn_running_mean");

    // create layer1_bn_running_var tensor
    const unsigned layer1_bn_running_var_sizes[] = {
        256,
    };
    uint64_t            layer1_bn_running_var_dram    = 0;
    synLaunchTensorInfo layer1_bn_running_var_tr_info = {"layer1_bn_running_var", layer1_bn_running_var_dram};
    synTensor           layer1_bn_running_var =
        createTensor(1U, syn_type_single, layer1_bn_running_var_sizes, true, "layer1_bn_running_var");

    synTensor layer1_bn_in_vec[5] = {layer1_downsample_output,
                                     layer1_bn_bias,
                                     layer1_bn_weight,
                                     layer1_bn_running_mean,
                                     layer1_bn_running_var};

    // create layer1_bn_output tensor
    const unsigned layer1_bn_output_sizes[] = {64, 56, 56, 256};
    synTensor layer1_bn_output = createTensor(4U, syn_type_bf16, layer1_bn_output_sizes, false, "layer1_bn_output");

    // create layer1_bn_saved_mean tensor
    const unsigned layer1_bn_saved_mean_sizes[] = {256};
    synTensor      layer1_bn_saved_mean =
        createTensor(1U, syn_type_single, layer1_bn_saved_mean_sizes, false, "layer1_bn_saved_mean");

    // create layer1_bn_saved_var tensor
    const unsigned layer1_bn_saved_var_sizes[] = {256};
    synTensor      layer1_bn_saved_var =
        createTensor(1U, syn_type_single, layer1_bn_saved_var_sizes, false, "layer1_bn_saved_var");

    synTensor layer1_bn_out_vec[3] = {layer1_bn_output, layer1_bn_saved_mean, layer1_bn_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_bn_in_vec,
                           layer1_bn_out_vec,
                           5,
                           3,
                           (void*)&layer1_bn_kernel_params,
                           sizeof(layer1_bn_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_bn",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_bn failed!");

    /*************
     * layer1_0_add_residual_fwd0 node
     * inputs: [layer1_0_bn3_output(64, 56, 56, 256)(dtype=bf16), layer1_bn_output(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_0_add_residual_fwd(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_0_add_residual_fwd0_in_vec[2] = {layer1_0_bn3_output, layer1_bn_output};

    // create layer1_0_add_residual_fwd tensor
    const unsigned layer1_0_add_residual_fwd_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer1_0_add_residual_fwd_sizes, false, "layer1_0_add_residual_fwd");

    synTensor layer1_0_add_residual_fwd0_out_vec[1] = {layer1_0_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer1_0_add_residual_fwd0_in_vec,
                           layer1_0_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer1_0_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_add_residual_fwd0 failed!");

    /*************
     * layer1_0_relu3 node
     * inputs: [layer1_0_add_residual_fwd(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_0_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_0_relu3_in_vec[1] = {layer1_0_add_residual_fwd};

    // create layer1_0_relu3_output tensor
    const unsigned layer1_0_relu3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_relu3_output =
        createTensor(4U, syn_type_bf16, layer1_0_relu3_output_sizes, false, "layer1_0_relu3_output");

    synTensor layer1_0_relu3_out_vec[1] = {layer1_0_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer1_0_relu3_in_vec,
                           layer1_0_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_0_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu3 failed!");

    /*************
     * layer1_1_conv1 node
     * inputs: [layer1_0_relu3_output(64, 56, 56, 256)(dtype=bf16), layer1_1_conv1_weight[1, 1, 256, 64](dtype=bf16)]
     * output: [layer1_1_conv1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_1_conv1_kernel_params;
    layer1_1_conv1_kernel_params.dH   = 1;
    layer1_1_conv1_kernel_params.dW   = 1;
    layer1_1_conv1_kernel_params.kH   = 1;
    layer1_1_conv1_kernel_params.kW   = 1;
    layer1_1_conv1_kernel_params.padT = 0;
    layer1_1_conv1_kernel_params.padB = 0;
    layer1_1_conv1_kernel_params.padL = 0;
    layer1_1_conv1_kernel_params.padR = 0;
    layer1_1_conv1_kernel_params.dilH = 1;
    layer1_1_conv1_kernel_params.dilW = 1;

    // create layer1_1_conv1_weight tensor
    const unsigned      layer1_1_conv1_weight_sizes[] = {1, 1, 256, 64};
    uint64_t            layer1_1_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer1_1_conv1_weight_tr_info = {"layer1_1_conv1_weight", layer1_1_conv1_weight_dram};
    synTensor           layer1_1_conv1_weight =
        createTensor(4U, syn_type_bf16, layer1_1_conv1_weight_sizes, true, "layer1_1_conv1_weight");

    synTensor layer1_1_conv1_in_vec[4] = {layer1_0_relu3_output, layer1_1_conv1_weight, nullptr, nullptr};

    // create layer1_1_conv1_output tensor
    const unsigned layer1_1_conv1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_conv1_output =
        createTensor(4U, syn_type_bf16, layer1_1_conv1_output_sizes, false, "layer1_1_conv1_output");

    synTensor layer1_1_conv1_out_vec[1] = {layer1_1_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv1_in_vec,
                           layer1_1_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer1_1_conv1_kernel_params,
                           sizeof(layer1_1_conv1_kernel_params),
                           "spatial_convolution",
                           "layer1_1_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv1 failed!");

    /*************
     * layer1_1_bn1 node
     * inputs: [layer1_1_conv1_output(64, 56, 56, 64)(dtype=bf16), layer1_1_bn1_bias[64](dtype=float32),
     *layer1_1_bn1_weight[64](dtype=float32), layer1_1_bn1_running_mean[64](dtype=float32),
     *layer1_1_bn1_running_var[64](dtype=float32)] output: [layer1_1_bn1_output(64, 56, 56, 64)(dtype=bf16),
     *layer1_1_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_1_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_1_bn1_kernel_params;
    layer1_1_bn1_kernel_params.momentum    = 0.1;
    layer1_1_bn1_kernel_params.threshold.f = 1e-05;
    layer1_1_bn1_kernel_params.epsilon     = 1e-05;

    // create layer1_1_bn1_bias tensor
    const unsigned layer1_1_bn1_bias_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer1_1_bn1_bias_tr_info = {"layer1_1_bn1_bias", layer1_1_bn1_bias_dram};
    synTensor layer1_1_bn1_bias = createTensor(1U, syn_type_single, layer1_1_bn1_bias_sizes, true, "layer1_1_bn1_bias");

    // create layer1_1_bn1_weight tensor
    const unsigned layer1_1_bn1_weight_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer1_1_bn1_weight_tr_info = {"layer1_1_bn1_weight", layer1_1_bn1_weight_dram};
    synTensor           layer1_1_bn1_weight =
        createTensor(1U, syn_type_single, layer1_1_bn1_weight_sizes, true, "layer1_1_bn1_weight");

    // create layer1_1_bn1_running_mean tensor
    const unsigned layer1_1_bn1_running_mean_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_1_bn1_running_mean_tr_info = {"layer1_1_bn1_running_mean",
                                                             layer1_1_bn1_running_mean_dram};
    synTensor           layer1_1_bn1_running_mean =
        createTensor(1U, syn_type_single, layer1_1_bn1_running_mean_sizes, true, "layer1_1_bn1_running_mean");

    // create layer1_1_bn1_running_var tensor
    const unsigned layer1_1_bn1_running_var_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer1_1_bn1_running_var_tr_info = {"layer1_1_bn1_running_var", layer1_1_bn1_running_var_dram};
    synTensor           layer1_1_bn1_running_var =
        createTensor(1U, syn_type_single, layer1_1_bn1_running_var_sizes, true, "layer1_1_bn1_running_var");

    synTensor layer1_1_bn1_in_vec[5] = {layer1_1_conv1_output,
                                        layer1_1_bn1_bias,
                                        layer1_1_bn1_weight,
                                        layer1_1_bn1_running_mean,
                                        layer1_1_bn1_running_var};

    // create layer1_1_bn1_output tensor
    const unsigned layer1_1_bn1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_bn1_output =
        createTensor(4U, syn_type_bf16, layer1_1_bn1_output_sizes, false, "layer1_1_bn1_output");

    // create layer1_1_bn1_saved_mean tensor
    const unsigned layer1_1_bn1_saved_mean_sizes[] = {64};
    synTensor      layer1_1_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer1_1_bn1_saved_mean_sizes, false, "layer1_1_bn1_saved_mean");

    // create layer1_1_bn1_saved_var tensor
    const unsigned layer1_1_bn1_saved_var_sizes[] = {64};
    synTensor      layer1_1_bn1_saved_var =
        createTensor(1U, syn_type_single, layer1_1_bn1_saved_var_sizes, false, "layer1_1_bn1_saved_var");

    synTensor layer1_1_bn1_out_vec[3] = {layer1_1_bn1_output, layer1_1_bn1_saved_mean, layer1_1_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_1_bn1_in_vec,
                           layer1_1_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer1_1_bn1_kernel_params,
                           sizeof(layer1_1_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_1_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_bn1 failed!");

    /*************
     * layer1_1_relu1 node
     * inputs: [layer1_1_bn1_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_1_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_1_relu1_in_vec[1] = {layer1_1_bn1_output};

    // create layer1_1_relu1_output tensor
    const unsigned layer1_1_relu1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_relu1_output =
        createTensor(4U, syn_type_bf16, layer1_1_relu1_output_sizes, false, "layer1_1_relu1_output");

    synTensor layer1_1_relu1_out_vec[1] = {layer1_1_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer1_1_relu1_in_vec,
                           layer1_1_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_1_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_relu1 failed!");

    /*************
     * layer1_1_conv2 node
     * inputs: [layer1_1_relu1_output(64, 56, 56, 64)(dtype=bf16), layer1_1_conv2_weight[3, 3, 64, 64](dtype=bf16)]
     * output: [layer1_1_conv2_output(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_1_conv2_kernel_params;
    layer1_1_conv2_kernel_params.dH   = 1;
    layer1_1_conv2_kernel_params.dW   = 1;
    layer1_1_conv2_kernel_params.kH   = 3;
    layer1_1_conv2_kernel_params.kW   = 3;
    layer1_1_conv2_kernel_params.padT = 1;
    layer1_1_conv2_kernel_params.padB = 1;
    layer1_1_conv2_kernel_params.padL = 1;
    layer1_1_conv2_kernel_params.padR = 1;
    layer1_1_conv2_kernel_params.dilH = 1;
    layer1_1_conv2_kernel_params.dilW = 1;

    // create layer1_1_conv2_weight tensor
    const unsigned      layer1_1_conv2_weight_sizes[] = {3, 3, 64, 64};
    uint64_t            layer1_1_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer1_1_conv2_weight_tr_info = {"layer1_1_conv2_weight", layer1_1_conv2_weight_dram};
    synTensor           layer1_1_conv2_weight =
        createTensor(4U, syn_type_bf16, layer1_1_conv2_weight_sizes, true, "layer1_1_conv2_weight");

    synTensor layer1_1_conv2_in_vec[4] = {layer1_1_relu1_output, layer1_1_conv2_weight, nullptr, nullptr};

    // create layer1_1_conv2_output tensor
    const unsigned layer1_1_conv2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_conv2_output =
        createTensor(4U, syn_type_bf16, layer1_1_conv2_output_sizes, false, "layer1_1_conv2_output");

    synTensor layer1_1_conv2_out_vec[1] = {layer1_1_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv2_in_vec,
                           layer1_1_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer1_1_conv2_kernel_params,
                           sizeof(layer1_1_conv2_kernel_params),
                           "spatial_convolution",
                           "layer1_1_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv2 failed!");

    /*************
     * layer1_1_bn2 node
     * inputs: [layer1_1_conv2_output(64, 56, 56, 64)(dtype=bf16), layer1_1_bn2_bias[64](dtype=float32),
     *layer1_1_bn2_weight[64](dtype=float32), layer1_1_bn2_running_mean[64](dtype=float32),
     *layer1_1_bn2_running_var[64](dtype=float32)] output: [layer1_1_bn2_output(64, 56, 56, 64)(dtype=bf16),
     *layer1_1_bn2_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_1_bn2_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_1_bn2_kernel_params;
    layer1_1_bn2_kernel_params.momentum    = 0.1;
    layer1_1_bn2_kernel_params.threshold.f = 1e-05;
    layer1_1_bn2_kernel_params.epsilon     = 1e-05;

    // create layer1_1_bn2_bias tensor
    const unsigned layer1_1_bn2_bias_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer1_1_bn2_bias_tr_info = {"layer1_1_bn2_bias", layer1_1_bn2_bias_dram};
    synTensor layer1_1_bn2_bias = createTensor(1U, syn_type_single, layer1_1_bn2_bias_sizes, true, "layer1_1_bn2_bias");

    // create layer1_1_bn2_weight tensor
    const unsigned layer1_1_bn2_weight_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer1_1_bn2_weight_tr_info = {"layer1_1_bn2_weight", layer1_1_bn2_weight_dram};
    synTensor           layer1_1_bn2_weight =
        createTensor(1U, syn_type_single, layer1_1_bn2_weight_sizes, true, "layer1_1_bn2_weight");

    // create layer1_1_bn2_running_mean tensor
    const unsigned layer1_1_bn2_running_mean_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_1_bn2_running_mean_tr_info = {"layer1_1_bn2_running_mean",
                                                             layer1_1_bn2_running_mean_dram};
    synTensor           layer1_1_bn2_running_mean =
        createTensor(1U, syn_type_single, layer1_1_bn2_running_mean_sizes, true, "layer1_1_bn2_running_mean");

    // create layer1_1_bn2_running_var tensor
    const unsigned layer1_1_bn2_running_var_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer1_1_bn2_running_var_tr_info = {"layer1_1_bn2_running_var", layer1_1_bn2_running_var_dram};
    synTensor           layer1_1_bn2_running_var =
        createTensor(1U, syn_type_single, layer1_1_bn2_running_var_sizes, true, "layer1_1_bn2_running_var");

    synTensor layer1_1_bn2_in_vec[5] = {layer1_1_conv2_output,
                                        layer1_1_bn2_bias,
                                        layer1_1_bn2_weight,
                                        layer1_1_bn2_running_mean,
                                        layer1_1_bn2_running_var};

    // create layer1_1_bn2_output tensor
    const unsigned layer1_1_bn2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_bn2_output =
        createTensor(4U, syn_type_bf16, layer1_1_bn2_output_sizes, false, "layer1_1_bn2_output");

    // create layer1_1_bn2_saved_mean tensor
    const unsigned layer1_1_bn2_saved_mean_sizes[] = {64};
    synTensor      layer1_1_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer1_1_bn2_saved_mean_sizes, false, "layer1_1_bn2_saved_mean");

    // create layer1_1_bn2_saved_var tensor
    const unsigned layer1_1_bn2_saved_var_sizes[] = {64};
    synTensor      layer1_1_bn2_saved_var =
        createTensor(1U, syn_type_single, layer1_1_bn2_saved_var_sizes, false, "layer1_1_bn2_saved_var");

    synTensor layer1_1_bn2_out_vec[3] = {layer1_1_bn2_output, layer1_1_bn2_saved_mean, layer1_1_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_1_bn2_in_vec,
                           layer1_1_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer1_1_bn2_kernel_params,
                           sizeof(layer1_1_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_1_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_bn2 failed!");

    /*************
     * layer1_1_relu2 node
     * inputs: [layer1_1_bn2_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_1_relu2_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_1_relu2_in_vec[1] = {layer1_1_bn2_output};

    // create layer1_1_relu2_output tensor
    const unsigned layer1_1_relu2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_relu2_output =
        createTensor(4U, syn_type_bf16, layer1_1_relu2_output_sizes, false, "layer1_1_relu2_output");

    synTensor layer1_1_relu2_out_vec[1] = {layer1_1_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer1_1_relu2_in_vec,
                           layer1_1_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_1_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_relu2 failed!");

    /*************
     * layer1_1_conv3 node
     * inputs: [layer1_1_relu2_output(64, 56, 56, 64)(dtype=bf16), layer1_1_conv3_weight[1, 1, 64, 256](dtype=bf16)]
     * output: [layer1_1_conv3_output(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_1_conv3_kernel_params;
    layer1_1_conv3_kernel_params.dH   = 1;
    layer1_1_conv3_kernel_params.dW   = 1;
    layer1_1_conv3_kernel_params.kH   = 1;
    layer1_1_conv3_kernel_params.kW   = 1;
    layer1_1_conv3_kernel_params.padT = 0;
    layer1_1_conv3_kernel_params.padB = 0;
    layer1_1_conv3_kernel_params.padL = 0;
    layer1_1_conv3_kernel_params.padR = 0;
    layer1_1_conv3_kernel_params.dilH = 1;
    layer1_1_conv3_kernel_params.dilW = 1;

    // create layer1_1_conv3_weight tensor
    const unsigned      layer1_1_conv3_weight_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_1_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer1_1_conv3_weight_tr_info = {"layer1_1_conv3_weight", layer1_1_conv3_weight_dram};
    synTensor           layer1_1_conv3_weight =
        createTensor(4U, syn_type_bf16, layer1_1_conv3_weight_sizes, true, "layer1_1_conv3_weight");

    synTensor layer1_1_conv3_in_vec[4] = {layer1_1_relu2_output, layer1_1_conv3_weight, nullptr, nullptr};

    // create layer1_1_conv3_output tensor
    const unsigned layer1_1_conv3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_conv3_output =
        createTensor(4U, syn_type_bf16, layer1_1_conv3_output_sizes, false, "layer1_1_conv3_output");

    synTensor layer1_1_conv3_out_vec[1] = {layer1_1_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv3_in_vec,
                           layer1_1_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer1_1_conv3_kernel_params,
                           sizeof(layer1_1_conv3_kernel_params),
                           "spatial_convolution",
                           "layer1_1_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv3 failed!");

    /*************
     * layer1_1_bn3 node
     * inputs: [layer1_1_conv3_output(64, 56, 56, 256)(dtype=bf16), layer1_1_bn3_bias[256](dtype=float32),
     *layer1_1_bn3_weight[256](dtype=float32), layer1_1_bn3_running_mean[256](dtype=float32),
     *layer1_1_bn3_running_var[256](dtype=float32)] output: [layer1_1_bn3_output(64, 56, 56, 256)(dtype=bf16),
     *layer1_1_bn3_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_1_bn3_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_1_bn3_kernel_params;
    layer1_1_bn3_kernel_params.momentum    = 0.1;
    layer1_1_bn3_kernel_params.threshold.f = 1e-05;
    layer1_1_bn3_kernel_params.epsilon     = 1e-05;

    // create layer1_1_bn3_bias tensor
    const unsigned layer1_1_bn3_bias_sizes[] = {
        256,
    };
    uint64_t            layer1_1_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer1_1_bn3_bias_tr_info = {"layer1_1_bn3_bias", layer1_1_bn3_bias_dram};
    synTensor layer1_1_bn3_bias = createTensor(1U, syn_type_single, layer1_1_bn3_bias_sizes, true, "layer1_1_bn3_bias");

    // create layer1_1_bn3_weight tensor
    const unsigned layer1_1_bn3_weight_sizes[] = {
        256,
    };
    uint64_t            layer1_1_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer1_1_bn3_weight_tr_info = {"layer1_1_bn3_weight", layer1_1_bn3_weight_dram};
    synTensor           layer1_1_bn3_weight =
        createTensor(1U, syn_type_single, layer1_1_bn3_weight_sizes, true, "layer1_1_bn3_weight");

    // create layer1_1_bn3_running_mean tensor
    const unsigned layer1_1_bn3_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer1_1_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_1_bn3_running_mean_tr_info = {"layer1_1_bn3_running_mean",
                                                             layer1_1_bn3_running_mean_dram};
    synTensor           layer1_1_bn3_running_mean =
        createTensor(1U, syn_type_single, layer1_1_bn3_running_mean_sizes, true, "layer1_1_bn3_running_mean");

    // create layer1_1_bn3_running_var tensor
    const unsigned layer1_1_bn3_running_var_sizes[] = {
        256,
    };
    uint64_t            layer1_1_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer1_1_bn3_running_var_tr_info = {"layer1_1_bn3_running_var", layer1_1_bn3_running_var_dram};
    synTensor           layer1_1_bn3_running_var =
        createTensor(1U, syn_type_single, layer1_1_bn3_running_var_sizes, true, "layer1_1_bn3_running_var");

    synTensor layer1_1_bn3_in_vec[5] = {layer1_1_conv3_output,
                                        layer1_1_bn3_bias,
                                        layer1_1_bn3_weight,
                                        layer1_1_bn3_running_mean,
                                        layer1_1_bn3_running_var};

    // create layer1_1_bn3_output tensor
    const unsigned layer1_1_bn3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_bn3_output =
        createTensor(4U, syn_type_bf16, layer1_1_bn3_output_sizes, false, "layer1_1_bn3_output");

    // create layer1_1_bn3_saved_mean tensor
    const unsigned layer1_1_bn3_saved_mean_sizes[] = {256};
    synTensor      layer1_1_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer1_1_bn3_saved_mean_sizes, false, "layer1_1_bn3_saved_mean");

    // create layer1_1_bn3_saved_var tensor
    const unsigned layer1_1_bn3_saved_var_sizes[] = {256};
    synTensor      layer1_1_bn3_saved_var =
        createTensor(1U, syn_type_single, layer1_1_bn3_saved_var_sizes, false, "layer1_1_bn3_saved_var");

    synTensor layer1_1_bn3_out_vec[3] = {layer1_1_bn3_output, layer1_1_bn3_saved_mean, layer1_1_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_1_bn3_in_vec,
                           layer1_1_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer1_1_bn3_kernel_params,
                           sizeof(layer1_1_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_1_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_bn3 failed!");

    /*************
     * layer1_1_add_residual_fwd0 node
     * inputs: [layer1_1_bn3_output(64, 56, 56, 256)(dtype=bf16), layer1_0_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_1_add_residual_fwd(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_1_add_residual_fwd0_in_vec[2] = {layer1_1_bn3_output, layer1_0_relu3_output};

    // create layer1_1_add_residual_fwd tensor
    const unsigned layer1_1_add_residual_fwd_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer1_1_add_residual_fwd_sizes, false, "layer1_1_add_residual_fwd");

    synTensor layer1_1_add_residual_fwd0_out_vec[1] = {layer1_1_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer1_1_add_residual_fwd0_in_vec,
                           layer1_1_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer1_1_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_add_residual_fwd0 failed!");

    /*************
     * layer1_1_relu3 node
     * inputs: [layer1_1_add_residual_fwd(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_1_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_1_relu3_in_vec[1] = {layer1_1_add_residual_fwd};

    // create layer1_1_relu3_output tensor
    const unsigned layer1_1_relu3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_relu3_output =
        createTensor(4U, syn_type_bf16, layer1_1_relu3_output_sizes, false, "layer1_1_relu3_output");

    synTensor layer1_1_relu3_out_vec[1] = {layer1_1_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer1_1_relu3_in_vec,
                           layer1_1_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_1_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_relu3 failed!");

    /*************
     * layer1_2_conv1 node
     * inputs: [layer1_1_relu3_output(64, 56, 56, 256)(dtype=bf16), layer1_2_conv1_weight[1, 1, 256, 64](dtype=bf16)]
     * output: [layer1_2_conv1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_2_conv1_kernel_params;
    layer1_2_conv1_kernel_params.dH   = 1;
    layer1_2_conv1_kernel_params.dW   = 1;
    layer1_2_conv1_kernel_params.kH   = 1;
    layer1_2_conv1_kernel_params.kW   = 1;
    layer1_2_conv1_kernel_params.padT = 0;
    layer1_2_conv1_kernel_params.padB = 0;
    layer1_2_conv1_kernel_params.padL = 0;
    layer1_2_conv1_kernel_params.padR = 0;
    layer1_2_conv1_kernel_params.dilH = 1;
    layer1_2_conv1_kernel_params.dilW = 1;

    // create layer1_2_conv1_weight tensor
    const unsigned      layer1_2_conv1_weight_sizes[] = {1, 1, 256, 64};
    uint64_t            layer1_2_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer1_2_conv1_weight_tr_info = {"layer1_2_conv1_weight", layer1_2_conv1_weight_dram};
    synTensor           layer1_2_conv1_weight =
        createTensor(4U, syn_type_bf16, layer1_2_conv1_weight_sizes, true, "layer1_2_conv1_weight");

    synTensor layer1_2_conv1_in_vec[4] = {layer1_1_relu3_output, layer1_2_conv1_weight, nullptr, nullptr};

    // create layer1_2_conv1_output tensor
    const unsigned layer1_2_conv1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_conv1_output =
        createTensor(4U, syn_type_bf16, layer1_2_conv1_output_sizes, false, "layer1_2_conv1_output");

    synTensor layer1_2_conv1_out_vec[1] = {layer1_2_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv1_in_vec,
                           layer1_2_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer1_2_conv1_kernel_params,
                           sizeof(layer1_2_conv1_kernel_params),
                           "spatial_convolution",
                           "layer1_2_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv1 failed!");

    /*************
     * layer1_2_bn1 node
     * inputs: [layer1_2_conv1_output(64, 56, 56, 64)(dtype=bf16), layer1_2_bn1_bias[64](dtype=float32),
     *layer1_2_bn1_weight[64](dtype=float32), layer1_2_bn1_running_mean[64](dtype=float32),
     *layer1_2_bn1_running_var[64](dtype=float32)] output: [layer1_2_bn1_output(64, 56, 56, 64)(dtype=bf16),
     *layer1_2_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_2_bn1_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_2_bn1_kernel_params;
    layer1_2_bn1_kernel_params.momentum    = 0.1;
    layer1_2_bn1_kernel_params.threshold.f = 1e-05;
    layer1_2_bn1_kernel_params.epsilon     = 1e-05;

    // create layer1_2_bn1_bias tensor
    const unsigned layer1_2_bn1_bias_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer1_2_bn1_bias_tr_info = {"layer1_2_bn1_bias", layer1_2_bn1_bias_dram};
    synTensor layer1_2_bn1_bias = createTensor(1U, syn_type_single, layer1_2_bn1_bias_sizes, true, "layer1_2_bn1_bias");

    // create layer1_2_bn1_weight tensor
    const unsigned layer1_2_bn1_weight_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer1_2_bn1_weight_tr_info = {"layer1_2_bn1_weight", layer1_2_bn1_weight_dram};
    synTensor           layer1_2_bn1_weight =
        createTensor(1U, syn_type_single, layer1_2_bn1_weight_sizes, true, "layer1_2_bn1_weight");

    // create layer1_2_bn1_running_mean tensor
    const unsigned layer1_2_bn1_running_mean_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_2_bn1_running_mean_tr_info = {"layer1_2_bn1_running_mean",
                                                             layer1_2_bn1_running_mean_dram};
    synTensor           layer1_2_bn1_running_mean =
        createTensor(1U, syn_type_single, layer1_2_bn1_running_mean_sizes, true, "layer1_2_bn1_running_mean");

    // create layer1_2_bn1_running_var tensor
    const unsigned layer1_2_bn1_running_var_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer1_2_bn1_running_var_tr_info = {"layer1_2_bn1_running_var", layer1_2_bn1_running_var_dram};
    synTensor           layer1_2_bn1_running_var =
        createTensor(1U, syn_type_single, layer1_2_bn1_running_var_sizes, true, "layer1_2_bn1_running_var");

    synTensor layer1_2_bn1_in_vec[5] = {layer1_2_conv1_output,
                                        layer1_2_bn1_bias,
                                        layer1_2_bn1_weight,
                                        layer1_2_bn1_running_mean,
                                        layer1_2_bn1_running_var};

    // create layer1_2_bn1_output tensor
    const unsigned layer1_2_bn1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_bn1_output =
        createTensor(4U, syn_type_bf16, layer1_2_bn1_output_sizes, false, "layer1_2_bn1_output");

    // create layer1_2_bn1_saved_mean tensor
    const unsigned layer1_2_bn1_saved_mean_sizes[] = {64};
    synTensor      layer1_2_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer1_2_bn1_saved_mean_sizes, false, "layer1_2_bn1_saved_mean");

    // create layer1_2_bn1_saved_var tensor
    const unsigned layer1_2_bn1_saved_var_sizes[] = {64};
    synTensor      layer1_2_bn1_saved_var =
        createTensor(1U, syn_type_single, layer1_2_bn1_saved_var_sizes, false, "layer1_2_bn1_saved_var");

    synTensor layer1_2_bn1_out_vec[3] = {layer1_2_bn1_output, layer1_2_bn1_saved_mean, layer1_2_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_2_bn1_in_vec,
                           layer1_2_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer1_2_bn1_kernel_params,
                           sizeof(layer1_2_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_2_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_bn1 failed!");

    /*************
     * layer1_2_relu1 node
     * inputs: [layer1_2_bn1_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_2_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_2_relu1_in_vec[1] = {layer1_2_bn1_output};

    // create layer1_2_relu1_output tensor
    const unsigned layer1_2_relu1_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_relu1_output =
        createTensor(4U, syn_type_bf16, layer1_2_relu1_output_sizes, false, "layer1_2_relu1_output");

    synTensor layer1_2_relu1_out_vec[1] = {layer1_2_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer1_2_relu1_in_vec,
                           layer1_2_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_2_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_relu1 failed!");

    /*************
     * layer1_2_conv2 node
     * inputs: [layer1_2_relu1_output(64, 56, 56, 64)(dtype=bf16), layer1_2_conv2_weight[3, 3, 64, 64](dtype=bf16)]
     * output: [layer1_2_conv2_output(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_2_conv2_kernel_params;
    layer1_2_conv2_kernel_params.dH   = 1;
    layer1_2_conv2_kernel_params.dW   = 1;
    layer1_2_conv2_kernel_params.kH   = 3;
    layer1_2_conv2_kernel_params.kW   = 3;
    layer1_2_conv2_kernel_params.padT = 1;
    layer1_2_conv2_kernel_params.padB = 1;
    layer1_2_conv2_kernel_params.padL = 1;
    layer1_2_conv2_kernel_params.padR = 1;
    layer1_2_conv2_kernel_params.dilH = 1;
    layer1_2_conv2_kernel_params.dilW = 1;

    // create layer1_2_conv2_weight tensor
    const unsigned      layer1_2_conv2_weight_sizes[] = {3, 3, 64, 64};
    uint64_t            layer1_2_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer1_2_conv2_weight_tr_info = {"layer1_2_conv2_weight", layer1_2_conv2_weight_dram};
    synTensor           layer1_2_conv2_weight =
        createTensor(4U, syn_type_bf16, layer1_2_conv2_weight_sizes, true, "layer1_2_conv2_weight");

    synTensor layer1_2_conv2_in_vec[4] = {layer1_2_relu1_output, layer1_2_conv2_weight, nullptr, nullptr};

    // create layer1_2_conv2_output tensor
    const unsigned layer1_2_conv2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_conv2_output =
        createTensor(4U, syn_type_bf16, layer1_2_conv2_output_sizes, false, "layer1_2_conv2_output");

    synTensor layer1_2_conv2_out_vec[1] = {layer1_2_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv2_in_vec,
                           layer1_2_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer1_2_conv2_kernel_params,
                           sizeof(layer1_2_conv2_kernel_params),
                           "spatial_convolution",
                           "layer1_2_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv2 failed!");

    /*************
     * layer1_2_bn2 node
     * inputs: [layer1_2_conv2_output(64, 56, 56, 64)(dtype=bf16), layer1_2_bn2_bias[64](dtype=float32),
     *layer1_2_bn2_weight[64](dtype=float32), layer1_2_bn2_running_mean[64](dtype=float32),
     *layer1_2_bn2_running_var[64](dtype=float32)] output: [layer1_2_bn2_output(64, 56, 56, 64)(dtype=bf16),
     *layer1_2_bn2_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_2_bn2_saved_var(1, 1, 1, 64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_2_bn2_kernel_params;
    layer1_2_bn2_kernel_params.momentum    = 0.1;
    layer1_2_bn2_kernel_params.threshold.f = 1e-05;
    layer1_2_bn2_kernel_params.epsilon     = 1e-05;

    // create layer1_2_bn2_bias tensor
    const unsigned layer1_2_bn2_bias_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer1_2_bn2_bias_tr_info = {"layer1_2_bn2_bias", layer1_2_bn2_bias_dram};
    synTensor layer1_2_bn2_bias = createTensor(1U, syn_type_single, layer1_2_bn2_bias_sizes, true, "layer1_2_bn2_bias");

    // create layer1_2_bn2_weight tensor
    const unsigned layer1_2_bn2_weight_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer1_2_bn2_weight_tr_info = {"layer1_2_bn2_weight", layer1_2_bn2_weight_dram};
    synTensor           layer1_2_bn2_weight =
        createTensor(1U, syn_type_single, layer1_2_bn2_weight_sizes, true, "layer1_2_bn2_weight");

    // create layer1_2_bn2_running_mean tensor
    const unsigned layer1_2_bn2_running_mean_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_2_bn2_running_mean_tr_info = {"layer1_2_bn2_running_mean",
                                                             layer1_2_bn2_running_mean_dram};
    synTensor           layer1_2_bn2_running_mean =
        createTensor(1U, syn_type_single, layer1_2_bn2_running_mean_sizes, true, "layer1_2_bn2_running_mean");

    // create layer1_2_bn2_running_var tensor
    const unsigned layer1_2_bn2_running_var_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer1_2_bn2_running_var_tr_info = {"layer1_2_bn2_running_var", layer1_2_bn2_running_var_dram};
    synTensor           layer1_2_bn2_running_var =
        createTensor(1U, syn_type_single, layer1_2_bn2_running_var_sizes, true, "layer1_2_bn2_running_var");

    synTensor layer1_2_bn2_in_vec[5] = {layer1_2_conv2_output,
                                        layer1_2_bn2_bias,
                                        layer1_2_bn2_weight,
                                        layer1_2_bn2_running_mean,
                                        layer1_2_bn2_running_var};

    // create layer1_2_bn2_output tensor
    const unsigned layer1_2_bn2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_bn2_output =
        createTensor(4U, syn_type_bf16, layer1_2_bn2_output_sizes, false, "layer1_2_bn2_output");

    // create layer1_2_bn2_saved_mean tensor
    const unsigned layer1_2_bn2_saved_mean_sizes[] = {64};
    synTensor      layer1_2_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer1_2_bn2_saved_mean_sizes, false, "layer1_2_bn2_saved_mean");

    // create layer1_2_bn2_saved_var tensor
    const unsigned layer1_2_bn2_saved_var_sizes[] = {64};
    synTensor      layer1_2_bn2_saved_var =
        createTensor(1U, syn_type_single, layer1_2_bn2_saved_var_sizes, false, "layer1_2_bn2_saved_var");

    synTensor layer1_2_bn2_out_vec[3] = {layer1_2_bn2_output, layer1_2_bn2_saved_mean, layer1_2_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_2_bn2_in_vec,
                           layer1_2_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer1_2_bn2_kernel_params,
                           sizeof(layer1_2_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_2_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_bn2 failed!");

    /*************
     * layer1_2_relu2 node
     * inputs: [layer1_2_bn2_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_2_relu2_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_2_relu2_in_vec[1] = {layer1_2_bn2_output};

    // create layer1_2_relu2_output tensor
    const unsigned layer1_2_relu2_output_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_relu2_output =
        createTensor(4U, syn_type_bf16, layer1_2_relu2_output_sizes, false, "layer1_2_relu2_output");

    synTensor layer1_2_relu2_out_vec[1] = {layer1_2_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer1_2_relu2_in_vec,
                           layer1_2_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_2_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_relu2 failed!");

    /*************
     * layer1_2_conv3 node
     * inputs: [layer1_2_relu2_output(64, 56, 56, 64)(dtype=bf16), layer1_2_conv3_weight[1, 1, 64, 256](dtype=bf16)]
     * output: [layer1_2_conv3_output(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_2_conv3_kernel_params;
    layer1_2_conv3_kernel_params.dH   = 1;
    layer1_2_conv3_kernel_params.dW   = 1;
    layer1_2_conv3_kernel_params.kH   = 1;
    layer1_2_conv3_kernel_params.kW   = 1;
    layer1_2_conv3_kernel_params.padT = 0;
    layer1_2_conv3_kernel_params.padB = 0;
    layer1_2_conv3_kernel_params.padL = 0;
    layer1_2_conv3_kernel_params.padR = 0;
    layer1_2_conv3_kernel_params.dilH = 1;
    layer1_2_conv3_kernel_params.dilW = 1;

    // create layer1_2_conv3_weight tensor
    const unsigned      layer1_2_conv3_weight_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_2_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer1_2_conv3_weight_tr_info = {"layer1_2_conv3_weight", layer1_2_conv3_weight_dram};
    synTensor           layer1_2_conv3_weight =
        createTensor(4U, syn_type_bf16, layer1_2_conv3_weight_sizes, true, "layer1_2_conv3_weight");

    synTensor layer1_2_conv3_in_vec[4] = {layer1_2_relu2_output, layer1_2_conv3_weight, nullptr, nullptr};

    // create layer1_2_conv3_output tensor
    const unsigned layer1_2_conv3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_conv3_output =
        createTensor(4U, syn_type_bf16, layer1_2_conv3_output_sizes, false, "layer1_2_conv3_output");

    synTensor layer1_2_conv3_out_vec[1] = {layer1_2_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv3_in_vec,
                           layer1_2_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer1_2_conv3_kernel_params,
                           sizeof(layer1_2_conv3_kernel_params),
                           "spatial_convolution",
                           "layer1_2_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv3 failed!");

    /*************
     * layer1_2_bn3 node
     * inputs: [layer1_2_conv3_output(64, 56, 56, 256)(dtype=bf16), layer1_2_bn3_bias[256](dtype=float32),
     *layer1_2_bn3_weight[256](dtype=float32), layer1_2_bn3_running_mean[256](dtype=float32),
     *layer1_2_bn3_running_var[256](dtype=float32)] output: [layer1_2_bn3_output(64, 56, 56, 256)(dtype=bf16),
     *layer1_2_bn3_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_2_bn3_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_2_bn3_kernel_params;
    layer1_2_bn3_kernel_params.momentum    = 0.1;
    layer1_2_bn3_kernel_params.threshold.f = 1e-05;
    layer1_2_bn3_kernel_params.epsilon     = 1e-05;

    // create layer1_2_bn3_bias tensor
    const unsigned layer1_2_bn3_bias_sizes[] = {
        256,
    };
    uint64_t            layer1_2_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer1_2_bn3_bias_tr_info = {"layer1_2_bn3_bias", layer1_2_bn3_bias_dram};
    synTensor layer1_2_bn3_bias = createTensor(1U, syn_type_single, layer1_2_bn3_bias_sizes, true, "layer1_2_bn3_bias");

    // create layer1_2_bn3_weight tensor
    const unsigned layer1_2_bn3_weight_sizes[] = {
        256,
    };
    uint64_t            layer1_2_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer1_2_bn3_weight_tr_info = {"layer1_2_bn3_weight", layer1_2_bn3_weight_dram};
    synTensor           layer1_2_bn3_weight =
        createTensor(1U, syn_type_single, layer1_2_bn3_weight_sizes, true, "layer1_2_bn3_weight");

    // create layer1_2_bn3_running_mean tensor
    const unsigned layer1_2_bn3_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer1_2_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer1_2_bn3_running_mean_tr_info = {"layer1_2_bn3_running_mean",
                                                             layer1_2_bn3_running_mean_dram};
    synTensor           layer1_2_bn3_running_mean =
        createTensor(1U, syn_type_single, layer1_2_bn3_running_mean_sizes, true, "layer1_2_bn3_running_mean");

    // create layer1_2_bn3_running_var tensor
    const unsigned layer1_2_bn3_running_var_sizes[] = {
        256,
    };
    uint64_t            layer1_2_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer1_2_bn3_running_var_tr_info = {"layer1_2_bn3_running_var", layer1_2_bn3_running_var_dram};
    synTensor           layer1_2_bn3_running_var =
        createTensor(1U, syn_type_single, layer1_2_bn3_running_var_sizes, true, "layer1_2_bn3_running_var");

    synTensor layer1_2_bn3_in_vec[5] = {layer1_2_conv3_output,
                                        layer1_2_bn3_bias,
                                        layer1_2_bn3_weight,
                                        layer1_2_bn3_running_mean,
                                        layer1_2_bn3_running_var};

    // create layer1_2_bn3_output tensor
    const unsigned layer1_2_bn3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_bn3_output =
        createTensor(4U, syn_type_bf16, layer1_2_bn3_output_sizes, false, "layer1_2_bn3_output");

    // create layer1_2_bn3_saved_mean tensor
    const unsigned layer1_2_bn3_saved_mean_sizes[] = {256};
    synTensor      layer1_2_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer1_2_bn3_saved_mean_sizes, false, "layer1_2_bn3_saved_mean");

    // create layer1_2_bn3_saved_var tensor
    const unsigned layer1_2_bn3_saved_var_sizes[] = {256};
    synTensor      layer1_2_bn3_saved_var =
        createTensor(1U, syn_type_single, layer1_2_bn3_saved_var_sizes, false, "layer1_2_bn3_saved_var");

    synTensor layer1_2_bn3_out_vec[3] = {layer1_2_bn3_output, layer1_2_bn3_saved_mean, layer1_2_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer1_2_bn3_in_vec,
                           layer1_2_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer1_2_bn3_kernel_params,
                           sizeof(layer1_2_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer1_2_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_bn3 failed!");

    /*************
     * layer1_2_add_residual_fwd0 node
     * inputs: [layer1_2_bn3_output(64, 56, 56, 256)(dtype=bf16), layer1_1_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_2_add_residual_fwd(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_2_add_residual_fwd0_in_vec[2] = {layer1_2_bn3_output, layer1_1_relu3_output};

    // create layer1_2_add_residual_fwd tensor
    const unsigned layer1_2_add_residual_fwd_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer1_2_add_residual_fwd_sizes, false, "layer1_2_add_residual_fwd");

    synTensor layer1_2_add_residual_fwd0_out_vec[1] = {layer1_2_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer1_2_add_residual_fwd0_in_vec,
                           layer1_2_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer1_2_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_add_residual_fwd0 failed!");

    /*************
     * layer1_2_relu3 node
     * inputs: [layer1_2_add_residual_fwd(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_2_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_2_relu3_in_vec[1] = {layer1_2_add_residual_fwd};

    // create layer1_2_relu3_output tensor
    const unsigned layer1_2_relu3_output_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_relu3_output =
        createTensor(4U, syn_type_bf16, layer1_2_relu3_output_sizes, false, "layer1_2_relu3_output");

    synTensor layer1_2_relu3_out_vec[1] = {layer1_2_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer1_2_relu3_in_vec,
                           layer1_2_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer1_2_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_relu3 failed!");

    /*************
     * layer2_0_conv1 node
     * inputs: [layer1_2_relu3_output(64, 56, 56, 256)(dtype=bf16), layer2_0_conv1_weight[1, 1, 256, 128](dtype=bf16)]
     * output: [layer2_0_conv1_output(64, 56, 56, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_0_conv1_kernel_params;
    layer2_0_conv1_kernel_params.dH   = 1;
    layer2_0_conv1_kernel_params.dW   = 1;
    layer2_0_conv1_kernel_params.kH   = 1;
    layer2_0_conv1_kernel_params.kW   = 1;
    layer2_0_conv1_kernel_params.padT = 0;
    layer2_0_conv1_kernel_params.padB = 0;
    layer2_0_conv1_kernel_params.padL = 0;
    layer2_0_conv1_kernel_params.padR = 0;
    layer2_0_conv1_kernel_params.dilH = 1;
    layer2_0_conv1_kernel_params.dilW = 1;

    // create layer2_0_conv1_weight tensor
    const unsigned      layer2_0_conv1_weight_sizes[] = {1, 1, 256, 128};
    uint64_t            layer2_0_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer2_0_conv1_weight_tr_info = {"layer2_0_conv1_weight", layer2_0_conv1_weight_dram};
    synTensor           layer2_0_conv1_weight =
        createTensor(4U, syn_type_bf16, layer2_0_conv1_weight_sizes, true, "layer2_0_conv1_weight");

    synTensor layer2_0_conv1_in_vec[4] = {layer1_2_relu3_output, layer2_0_conv1_weight, nullptr, nullptr};

    // create layer2_0_conv1_output tensor
    const unsigned layer2_0_conv1_output_sizes[] = {64, 56, 56, 128};
    synTensor      layer2_0_conv1_output =
        createTensor(4U, syn_type_bf16, layer2_0_conv1_output_sizes, false, "layer2_0_conv1_output");

    synTensor layer2_0_conv1_out_vec[1] = {layer2_0_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv1_in_vec,
                           layer2_0_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer2_0_conv1_kernel_params,
                           sizeof(layer2_0_conv1_kernel_params),
                           "spatial_convolution",
                           "layer2_0_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv1 failed!");

    /*************
     * layer2_0_bn1 node
     * inputs: [layer2_0_conv1_output(64, 56, 56, 128)(dtype=bf16), layer2_0_bn1_bias[128](dtype=float32),
     *layer2_0_bn1_weight[128](dtype=float32), layer2_0_bn1_running_mean[128](dtype=float32),
     *layer2_0_bn1_running_var[128](dtype=float32)] output: [layer2_0_bn1_output(64, 56, 56, 128)(dtype=bf16),
     *layer2_0_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_0_bn1_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_0_bn1_kernel_params;
    layer2_0_bn1_kernel_params.momentum    = 0.1;
    layer2_0_bn1_kernel_params.threshold.f = 1e-05;
    layer2_0_bn1_kernel_params.epsilon     = 1e-05;

    // create layer2_0_bn1_bias tensor
    const unsigned layer2_0_bn1_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer2_0_bn1_bias_tr_info = {"layer2_0_bn1_bias", layer2_0_bn1_bias_dram};
    synTensor layer2_0_bn1_bias = createTensor(1U, syn_type_single, layer2_0_bn1_bias_sizes, true, "layer2_0_bn1_bias");

    // create layer2_0_bn1_weight tensor
    const unsigned layer2_0_bn1_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer2_0_bn1_weight_tr_info = {"layer2_0_bn1_weight", layer2_0_bn1_weight_dram};
    synTensor           layer2_0_bn1_weight =
        createTensor(1U, syn_type_single, layer2_0_bn1_weight_sizes, true, "layer2_0_bn1_weight");

    // create layer2_0_bn1_running_mean tensor
    const unsigned layer2_0_bn1_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_0_bn1_running_mean_tr_info = {"layer2_0_bn1_running_mean",
                                                             layer2_0_bn1_running_mean_dram};
    synTensor           layer2_0_bn1_running_mean =
        createTensor(1U, syn_type_single, layer2_0_bn1_running_mean_sizes, true, "layer2_0_bn1_running_mean");

    // create layer2_0_bn1_running_var tensor
    const unsigned layer2_0_bn1_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer2_0_bn1_running_var_tr_info = {"layer2_0_bn1_running_var", layer2_0_bn1_running_var_dram};
    synTensor           layer2_0_bn1_running_var =
        createTensor(1U, syn_type_single, layer2_0_bn1_running_var_sizes, true, "layer2_0_bn1_running_var");

    synTensor layer2_0_bn1_in_vec[5] = {layer2_0_conv1_output,
                                        layer2_0_bn1_bias,
                                        layer2_0_bn1_weight,
                                        layer2_0_bn1_running_mean,
                                        layer2_0_bn1_running_var};

    // create layer2_0_bn1_output tensor
    const unsigned layer2_0_bn1_output_sizes[] = {64, 56, 56, 128};
    synTensor      layer2_0_bn1_output =
        createTensor(4U, syn_type_bf16, layer2_0_bn1_output_sizes, false, "layer2_0_bn1_output");

    // create layer2_0_bn1_saved_mean tensor
    const unsigned layer2_0_bn1_saved_mean_sizes[] = {128};
    synTensor      layer2_0_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer2_0_bn1_saved_mean_sizes, false, "layer2_0_bn1_saved_mean");

    // create layer2_0_bn1_saved_var tensor
    const unsigned layer2_0_bn1_saved_var_sizes[] = {128};
    synTensor      layer2_0_bn1_saved_var =
        createTensor(1U, syn_type_single, layer2_0_bn1_saved_var_sizes, false, "layer2_0_bn1_saved_var");

    synTensor layer2_0_bn1_out_vec[3] = {layer2_0_bn1_output, layer2_0_bn1_saved_mean, layer2_0_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_0_bn1_in_vec,
                           layer2_0_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer2_0_bn1_kernel_params,
                           sizeof(layer2_0_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_0_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_bn1 failed!");

    /*************
     * layer2_0_relu1 node
     * inputs: [layer2_0_bn1_output(64, 56, 56, 128)(dtype=bf16)]
     * output: [layer2_0_relu1_output(64, 56, 56, 128)(dtype=bf16)]
     *************/

    synTensor layer2_0_relu1_in_vec[1] = {layer2_0_bn1_output};

    // create layer2_0_relu1_output tensor
    const unsigned layer2_0_relu1_output_sizes[] = {64, 56, 56, 128};
    synTensor      layer2_0_relu1_output =
        createTensor(4U, syn_type_bf16, layer2_0_relu1_output_sizes, false, "layer2_0_relu1_output");

    synTensor layer2_0_relu1_out_vec[1] = {layer2_0_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer2_0_relu1_in_vec,
                           layer2_0_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_0_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_relu1 failed!");

    /*************
     * layer2_0_conv2 node
     * inputs: [layer2_0_relu1_output(64, 56, 56, 128)(dtype=bf16), layer2_0_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_0_conv2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_0_conv2_kernel_params;
    layer2_0_conv2_kernel_params.dH   = 2;
    layer2_0_conv2_kernel_params.dW   = 2;
    layer2_0_conv2_kernel_params.kH   = 3;
    layer2_0_conv2_kernel_params.kW   = 3;
    layer2_0_conv2_kernel_params.padT = 1;
    layer2_0_conv2_kernel_params.padB = 1;
    layer2_0_conv2_kernel_params.padL = 1;
    layer2_0_conv2_kernel_params.padR = 1;
    layer2_0_conv2_kernel_params.dilH = 1;
    layer2_0_conv2_kernel_params.dilW = 1;

    // create layer2_0_conv2_weight tensor
    const unsigned      layer2_0_conv2_weight_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_0_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer2_0_conv2_weight_tr_info = {"layer2_0_conv2_weight", layer2_0_conv2_weight_dram};
    synTensor           layer2_0_conv2_weight =
        createTensor(4U, syn_type_bf16, layer2_0_conv2_weight_sizes, true, "layer2_0_conv2_weight");

    synTensor layer2_0_conv2_in_vec[4] = {layer2_0_relu1_output, layer2_0_conv2_weight, nullptr, nullptr};

    // create layer2_0_conv2_output tensor
    const unsigned layer2_0_conv2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_0_conv2_output =
        createTensor(4U, syn_type_bf16, layer2_0_conv2_output_sizes, false, "layer2_0_conv2_output");

    synTensor layer2_0_conv2_out_vec[1] = {layer2_0_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv2_in_vec,
                           layer2_0_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer2_0_conv2_kernel_params,
                           sizeof(layer2_0_conv2_kernel_params),
                           "spatial_convolution",
                           "layer2_0_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv2 failed!");

    /*************
     * layer2_0_bn2 node
     * inputs: [layer2_0_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_0_bn2_bias[128](dtype=float32),
     *layer2_0_bn2_weight[128](dtype=float32), layer2_0_bn2_running_mean[128](dtype=float32),
     *layer2_0_bn2_running_var[128](dtype=float32)] output: [layer2_0_bn2_output(64, 28, 28, 128)(dtype=bf16),
     *layer2_0_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_0_bn2_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_0_bn2_kernel_params;
    layer2_0_bn2_kernel_params.momentum    = 0.1;
    layer2_0_bn2_kernel_params.threshold.f = 1e-05;
    layer2_0_bn2_kernel_params.epsilon     = 1e-05;

    // create layer2_0_bn2_bias tensor
    const unsigned layer2_0_bn2_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer2_0_bn2_bias_tr_info = {"layer2_0_bn2_bias", layer2_0_bn2_bias_dram};
    synTensor layer2_0_bn2_bias = createTensor(1U, syn_type_single, layer2_0_bn2_bias_sizes, true, "layer2_0_bn2_bias");

    // create layer2_0_bn2_weight tensor
    const unsigned layer2_0_bn2_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer2_0_bn2_weight_tr_info = {"layer2_0_bn2_weight", layer2_0_bn2_weight_dram};
    synTensor           layer2_0_bn2_weight =
        createTensor(1U, syn_type_single, layer2_0_bn2_weight_sizes, true, "layer2_0_bn2_weight");

    // create layer2_0_bn2_running_mean tensor
    const unsigned layer2_0_bn2_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_0_bn2_running_mean_tr_info = {"layer2_0_bn2_running_mean",
                                                             layer2_0_bn2_running_mean_dram};
    synTensor           layer2_0_bn2_running_mean =
        createTensor(1U, syn_type_single, layer2_0_bn2_running_mean_sizes, true, "layer2_0_bn2_running_mean");

    // create layer2_0_bn2_running_var tensor
    const unsigned layer2_0_bn2_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer2_0_bn2_running_var_tr_info = {"layer2_0_bn2_running_var", layer2_0_bn2_running_var_dram};
    synTensor           layer2_0_bn2_running_var =
        createTensor(1U, syn_type_single, layer2_0_bn2_running_var_sizes, true, "layer2_0_bn2_running_var");

    synTensor layer2_0_bn2_in_vec[5] = {layer2_0_conv2_output,
                                        layer2_0_bn2_bias,
                                        layer2_0_bn2_weight,
                                        layer2_0_bn2_running_mean,
                                        layer2_0_bn2_running_var};

    // create layer2_0_bn2_output tensor
    const unsigned layer2_0_bn2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_0_bn2_output =
        createTensor(4U, syn_type_bf16, layer2_0_bn2_output_sizes, false, "layer2_0_bn2_output");

    // create layer2_0_bn2_saved_mean tensor
    const unsigned layer2_0_bn2_saved_mean_sizes[] = {128};
    synTensor      layer2_0_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer2_0_bn2_saved_mean_sizes, false, "layer2_0_bn2_saved_mean");

    // create layer2_0_bn2_saved_var tensor
    const unsigned layer2_0_bn2_saved_var_sizes[] = {128};
    synTensor      layer2_0_bn2_saved_var =
        createTensor(1U, syn_type_single, layer2_0_bn2_saved_var_sizes, false, "layer2_0_bn2_saved_var");

    synTensor layer2_0_bn2_out_vec[3] = {layer2_0_bn2_output, layer2_0_bn2_saved_mean, layer2_0_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_0_bn2_in_vec,
                           layer2_0_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer2_0_bn2_kernel_params,
                           sizeof(layer2_0_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_0_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_bn2 failed!");

    /*************
     * layer2_0_relu2 node
     * inputs: [layer2_0_bn2_output(64, 28, 28, 128)(dtype=bf16)]
     * output: [layer2_0_relu2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_0_relu2_in_vec[1] = {layer2_0_bn2_output};

    // create layer2_0_relu2_output tensor
    const unsigned layer2_0_relu2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_0_relu2_output =
        createTensor(4U, syn_type_bf16, layer2_0_relu2_output_sizes, false, "layer2_0_relu2_output");

    synTensor layer2_0_relu2_out_vec[1] = {layer2_0_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer2_0_relu2_in_vec,
                           layer2_0_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_0_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_relu2 failed!");

    /*************
     * layer2_0_conv3 node
     * inputs: [layer2_0_relu2_output(64, 28, 28, 128)(dtype=bf16), layer2_0_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_0_conv3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_0_conv3_kernel_params;
    layer2_0_conv3_kernel_params.dH   = 1;
    layer2_0_conv3_kernel_params.dW   = 1;
    layer2_0_conv3_kernel_params.kH   = 1;
    layer2_0_conv3_kernel_params.kW   = 1;
    layer2_0_conv3_kernel_params.padT = 0;
    layer2_0_conv3_kernel_params.padB = 0;
    layer2_0_conv3_kernel_params.padL = 0;
    layer2_0_conv3_kernel_params.padR = 0;
    layer2_0_conv3_kernel_params.dilH = 1;
    layer2_0_conv3_kernel_params.dilW = 1;

    // create layer2_0_conv3_weight tensor
    const unsigned      layer2_0_conv3_weight_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_0_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer2_0_conv3_weight_tr_info = {"layer2_0_conv3_weight", layer2_0_conv3_weight_dram};
    synTensor           layer2_0_conv3_weight =
        createTensor(4U, syn_type_bf16, layer2_0_conv3_weight_sizes, true, "layer2_0_conv3_weight");

    synTensor layer2_0_conv3_in_vec[4] = {layer2_0_relu2_output, layer2_0_conv3_weight, nullptr, nullptr};

    // create layer2_0_conv3_output tensor
    const unsigned layer2_0_conv3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_conv3_output =
        createTensor(4U, syn_type_bf16, layer2_0_conv3_output_sizes, false, "layer2_0_conv3_output");

    synTensor layer2_0_conv3_out_vec[1] = {layer2_0_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv3_in_vec,
                           layer2_0_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer2_0_conv3_kernel_params,
                           sizeof(layer2_0_conv3_kernel_params),
                           "spatial_convolution",
                           "layer2_0_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv3 failed!");

    /*************
     * layer2_0_bn3 node
     * inputs: [layer2_0_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_0_bn3_bias[512](dtype=float32),
     *layer2_0_bn3_weight[512](dtype=float32), layer2_0_bn3_running_mean[512](dtype=float32),
     *layer2_0_bn3_running_var[512](dtype=float32)] output: [layer2_0_bn3_output(64, 28, 28, 512)(dtype=bf16),
     *layer2_0_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_0_bn3_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_0_bn3_kernel_params;
    layer2_0_bn3_kernel_params.momentum    = 0.1;
    layer2_0_bn3_kernel_params.threshold.f = 1e-05;
    layer2_0_bn3_kernel_params.epsilon     = 1e-05;

    // create layer2_0_bn3_bias tensor
    const unsigned layer2_0_bn3_bias_sizes[] = {
        512,
    };
    uint64_t            layer2_0_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer2_0_bn3_bias_tr_info = {"layer2_0_bn3_bias", layer2_0_bn3_bias_dram};
    synTensor layer2_0_bn3_bias = createTensor(1U, syn_type_single, layer2_0_bn3_bias_sizes, true, "layer2_0_bn3_bias");

    // create layer2_0_bn3_weight tensor
    const unsigned layer2_0_bn3_weight_sizes[] = {
        512,
    };
    uint64_t            layer2_0_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer2_0_bn3_weight_tr_info = {"layer2_0_bn3_weight", layer2_0_bn3_weight_dram};
    synTensor           layer2_0_bn3_weight =
        createTensor(1U, syn_type_single, layer2_0_bn3_weight_sizes, true, "layer2_0_bn3_weight");

    // create layer2_0_bn3_running_mean tensor
    const unsigned layer2_0_bn3_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer2_0_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_0_bn3_running_mean_tr_info = {"layer2_0_bn3_running_mean",
                                                             layer2_0_bn3_running_mean_dram};
    synTensor           layer2_0_bn3_running_mean =
        createTensor(1U, syn_type_single, layer2_0_bn3_running_mean_sizes, true, "layer2_0_bn3_running_mean");

    // create layer2_0_bn3_running_var tensor
    const unsigned layer2_0_bn3_running_var_sizes[] = {
        512,
    };
    uint64_t            layer2_0_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer2_0_bn3_running_var_tr_info = {"layer2_0_bn3_running_var", layer2_0_bn3_running_var_dram};
    synTensor           layer2_0_bn3_running_var =
        createTensor(1U, syn_type_single, layer2_0_bn3_running_var_sizes, true, "layer2_0_bn3_running_var");

    synTensor layer2_0_bn3_in_vec[5] = {layer2_0_conv3_output,
                                        layer2_0_bn3_bias,
                                        layer2_0_bn3_weight,
                                        layer2_0_bn3_running_mean,
                                        layer2_0_bn3_running_var};

    // create layer2_0_bn3_output tensor
    const unsigned layer2_0_bn3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_bn3_output =
        createTensor(4U, syn_type_bf16, layer2_0_bn3_output_sizes, false, "layer2_0_bn3_output");

    // create layer2_0_bn3_saved_mean tensor
    const unsigned layer2_0_bn3_saved_mean_sizes[] = {512};
    synTensor      layer2_0_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer2_0_bn3_saved_mean_sizes, false, "layer2_0_bn3_saved_mean");

    // create layer2_0_bn3_saved_var tensor
    const unsigned layer2_0_bn3_saved_var_sizes[] = {512};
    synTensor      layer2_0_bn3_saved_var =
        createTensor(1U, syn_type_single, layer2_0_bn3_saved_var_sizes, false, "layer2_0_bn3_saved_var");

    synTensor layer2_0_bn3_out_vec[3] = {layer2_0_bn3_output, layer2_0_bn3_saved_mean, layer2_0_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_0_bn3_in_vec,
                           layer2_0_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer2_0_bn3_kernel_params,
                           sizeof(layer2_0_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_0_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_bn3 failed!");

    /*************
     * layer2_downsample node
     * inputs: [layer1_2_relu3_output(64, 56, 56, 256)(dtype=bf16), layer2_downsample_weight[1, 1, 256,
     *512](dtype=bf16)] output: [layer2_downsample_output(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_downsample_kernel_params;
    layer2_downsample_kernel_params.dH   = 2;
    layer2_downsample_kernel_params.dW   = 2;
    layer2_downsample_kernel_params.kH   = 1;
    layer2_downsample_kernel_params.kW   = 1;
    layer2_downsample_kernel_params.padT = 0;
    layer2_downsample_kernel_params.padB = 0;
    layer2_downsample_kernel_params.padL = 0;
    layer2_downsample_kernel_params.padR = 0;
    layer2_downsample_kernel_params.dilH = 1;
    layer2_downsample_kernel_params.dilW = 1;

    // create layer2_downsample_weight tensor
    const unsigned      layer2_downsample_weight_sizes[] = {1, 1, 256, 512};
    uint64_t            layer2_downsample_weight_dram    = 0;
    synLaunchTensorInfo layer2_downsample_weight_tr_info = {"layer2_downsample_weight", layer2_downsample_weight_dram};
    synTensor           layer2_downsample_weight =
        createTensor(4U, syn_type_bf16, layer2_downsample_weight_sizes, true, "layer2_downsample_weight");

    synTensor layer2_downsample_in_vec[4] = {layer1_2_relu3_output, layer2_downsample_weight, nullptr, nullptr};

    // create layer2_downsample_output tensor
    const unsigned layer2_downsample_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_downsample_output =
        createTensor(4U, syn_type_bf16, layer2_downsample_output_sizes, false, "layer2_downsample_output");

    synTensor layer2_downsample_out_vec[1] = {layer2_downsample_output};

    status = synNodeCreate(graphHandle,
                           layer2_downsample_in_vec,
                           layer2_downsample_out_vec,
                           4,
                           1,
                           (void*)&layer2_downsample_kernel_params,
                           sizeof(layer2_downsample_kernel_params),
                           "spatial_convolution",
                           "layer2_downsample",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_downsample failed!");

    /*************
     * layer2_bn node
     * inputs: [layer2_downsample_output(64, 28, 28, 512)(dtype=bf16), layer2_bn_bias[512](dtype=float32),
     *layer2_bn_weight[512](dtype=float32), layer2_bn_running_mean[512](dtype=float32),
     *layer2_bn_running_var[512](dtype=float32)] output: [layer2_bn_output(64, 28, 28, 512)(dtype=bf16),
     *layer2_bn_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_bn_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_bn_kernel_params;
    layer2_bn_kernel_params.momentum    = 0.1;
    layer2_bn_kernel_params.threshold.f = 1e-05;
    layer2_bn_kernel_params.epsilon     = 1e-05;

    // create layer2_bn_bias tensor
    const unsigned layer2_bn_bias_sizes[] = {
        512,
    };
    uint64_t            layer2_bn_bias_dram    = 0;
    synLaunchTensorInfo layer2_bn_bias_tr_info = {"layer2_bn_bias", layer2_bn_bias_dram};
    synTensor layer2_bn_bias = createTensor(1U, syn_type_single, layer2_bn_bias_sizes, true, "layer2_bn_bias");

    // create layer2_bn_weight tensor
    const unsigned layer2_bn_weight_sizes[] = {
        512,
    };
    uint64_t            layer2_bn_weight_dram    = 0;
    synLaunchTensorInfo layer2_bn_weight_tr_info = {"layer2_bn_weight", layer2_bn_weight_dram};
    synTensor layer2_bn_weight = createTensor(1U, syn_type_single, layer2_bn_weight_sizes, true, "layer2_bn_weight");

    // create layer2_bn_running_mean tensor
    const unsigned layer2_bn_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer2_bn_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_bn_running_mean_tr_info = {"layer2_bn_running_mean", layer2_bn_running_mean_dram};
    synTensor           layer2_bn_running_mean =
        createTensor(1U, syn_type_single, layer2_bn_running_mean_sizes, true, "layer2_bn_running_mean");

    // create layer2_bn_running_var tensor
    const unsigned layer2_bn_running_var_sizes[] = {
        512,
    };
    uint64_t            layer2_bn_running_var_dram    = 0;
    synLaunchTensorInfo layer2_bn_running_var_tr_info = {"layer2_bn_running_var", layer2_bn_running_var_dram};
    synTensor           layer2_bn_running_var =
        createTensor(1U, syn_type_single, layer2_bn_running_var_sizes, true, "layer2_bn_running_var");

    synTensor layer2_bn_in_vec[5] = {layer2_downsample_output,
                                     layer2_bn_bias,
                                     layer2_bn_weight,
                                     layer2_bn_running_mean,
                                     layer2_bn_running_var};

    // create layer2_bn_output tensor
    const unsigned layer2_bn_output_sizes[] = {64, 28, 28, 512};
    synTensor layer2_bn_output = createTensor(4U, syn_type_bf16, layer2_bn_output_sizes, false, "layer2_bn_output");

    // create layer2_bn_saved_mean tensor
    const unsigned layer2_bn_saved_mean_sizes[] = {512};
    synTensor      layer2_bn_saved_mean =
        createTensor(1U, syn_type_single, layer2_bn_saved_mean_sizes, false, "layer2_bn_saved_mean");

    // create layer2_bn_saved_var tensor
    const unsigned layer2_bn_saved_var_sizes[] = {512};
    synTensor      layer2_bn_saved_var =
        createTensor(1U, syn_type_single, layer2_bn_saved_var_sizes, false, "layer2_bn_saved_var");

    synTensor layer2_bn_out_vec[3] = {layer2_bn_output, layer2_bn_saved_mean, layer2_bn_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_bn_in_vec,
                           layer2_bn_out_vec,
                           5,
                           3,
                           (void*)&layer2_bn_kernel_params,
                           sizeof(layer2_bn_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_bn",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_bn failed!");

    /*************
     * layer2_0_add_residual_fwd0 node
     * inputs: [layer2_0_bn3_output(64, 28, 28, 512)(dtype=bf16), layer2_bn_output(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_0_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_0_add_residual_fwd0_in_vec[2] = {layer2_0_bn3_output, layer2_bn_output};

    // create layer2_0_add_residual_fwd tensor
    const unsigned layer2_0_add_residual_fwd_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer2_0_add_residual_fwd_sizes, false, "layer2_0_add_residual_fwd");

    synTensor layer2_0_add_residual_fwd0_out_vec[1] = {layer2_0_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer2_0_add_residual_fwd0_in_vec,
                           layer2_0_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_0_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_add_residual_fwd0 failed!");

    /*************
     * layer2_0_relu3 node
     * inputs: [layer2_0_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_0_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_0_relu3_in_vec[1] = {layer2_0_add_residual_fwd};

    // create layer2_0_relu3_output tensor
    const unsigned layer2_0_relu3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_relu3_output =
        createTensor(4U, syn_type_bf16, layer2_0_relu3_output_sizes, false, "layer2_0_relu3_output");

    synTensor layer2_0_relu3_out_vec[1] = {layer2_0_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer2_0_relu3_in_vec,
                           layer2_0_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_0_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_relu3 failed!");

    /*************
     * layer2_1_conv1 node
     * inputs: [layer2_0_relu3_output(64, 28, 28, 512)(dtype=bf16), layer2_1_conv1_weight[1, 1, 512, 128](dtype=bf16)]
     * output: [layer2_1_conv1_output(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_1_conv1_kernel_params;
    layer2_1_conv1_kernel_params.dH   = 1;
    layer2_1_conv1_kernel_params.dW   = 1;
    layer2_1_conv1_kernel_params.kH   = 1;
    layer2_1_conv1_kernel_params.kW   = 1;
    layer2_1_conv1_kernel_params.padT = 0;
    layer2_1_conv1_kernel_params.padB = 0;
    layer2_1_conv1_kernel_params.padL = 0;
    layer2_1_conv1_kernel_params.padR = 0;
    layer2_1_conv1_kernel_params.dilH = 1;
    layer2_1_conv1_kernel_params.dilW = 1;

    // create layer2_1_conv1_weight tensor
    const unsigned      layer2_1_conv1_weight_sizes[] = {1, 1, 512, 128};
    uint64_t            layer2_1_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer2_1_conv1_weight_tr_info = {"layer2_1_conv1_weight", layer2_1_conv1_weight_dram};
    synTensor           layer2_1_conv1_weight =
        createTensor(4U, syn_type_bf16, layer2_1_conv1_weight_sizes, true, "layer2_1_conv1_weight");

    synTensor layer2_1_conv1_in_vec[4] = {layer2_0_relu3_output, layer2_1_conv1_weight, nullptr, nullptr};

    // create layer2_1_conv1_output tensor
    const unsigned layer2_1_conv1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_conv1_output =
        createTensor(4U, syn_type_bf16, layer2_1_conv1_output_sizes, false, "layer2_1_conv1_output");

    synTensor layer2_1_conv1_out_vec[1] = {layer2_1_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv1_in_vec,
                           layer2_1_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer2_1_conv1_kernel_params,
                           sizeof(layer2_1_conv1_kernel_params),
                           "spatial_convolution",
                           "layer2_1_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv1 failed!");

    /*************
     * layer2_1_bn1 node
     * inputs: [layer2_1_conv1_output(64, 28, 28, 128)(dtype=bf16), layer2_1_bn1_bias[128](dtype=float32),
     *layer2_1_bn1_weight[128](dtype=float32), layer2_1_bn1_running_mean[128](dtype=float32),
     *layer2_1_bn1_running_var[128](dtype=float32)] output: [layer2_1_bn1_output(64, 28, 28, 128)(dtype=bf16),
     *layer2_1_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_1_bn1_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_1_bn1_kernel_params;
    layer2_1_bn1_kernel_params.momentum    = 0.1;
    layer2_1_bn1_kernel_params.threshold.f = 1e-05;
    layer2_1_bn1_kernel_params.epsilon     = 1e-05;

    // create layer2_1_bn1_bias tensor
    const unsigned layer2_1_bn1_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer2_1_bn1_bias_tr_info = {"layer2_1_bn1_bias", layer2_1_bn1_bias_dram};
    synTensor layer2_1_bn1_bias = createTensor(1U, syn_type_single, layer2_1_bn1_bias_sizes, true, "layer2_1_bn1_bias");

    // create layer2_1_bn1_weight tensor
    const unsigned layer2_1_bn1_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer2_1_bn1_weight_tr_info = {"layer2_1_bn1_weight", layer2_1_bn1_weight_dram};
    synTensor           layer2_1_bn1_weight =
        createTensor(1U, syn_type_single, layer2_1_bn1_weight_sizes, true, "layer2_1_bn1_weight");

    // create layer2_1_bn1_running_mean tensor
    const unsigned layer2_1_bn1_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_1_bn1_running_mean_tr_info = {"layer2_1_bn1_running_mean",
                                                             layer2_1_bn1_running_mean_dram};
    synTensor           layer2_1_bn1_running_mean =
        createTensor(1U, syn_type_single, layer2_1_bn1_running_mean_sizes, true, "layer2_1_bn1_running_mean");

    // create layer2_1_bn1_running_var tensor
    const unsigned layer2_1_bn1_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer2_1_bn1_running_var_tr_info = {"layer2_1_bn1_running_var", layer2_1_bn1_running_var_dram};
    synTensor           layer2_1_bn1_running_var =
        createTensor(1U, syn_type_single, layer2_1_bn1_running_var_sizes, true, "layer2_1_bn1_running_var");

    synTensor layer2_1_bn1_in_vec[5] = {layer2_1_conv1_output,
                                        layer2_1_bn1_bias,
                                        layer2_1_bn1_weight,
                                        layer2_1_bn1_running_mean,
                                        layer2_1_bn1_running_var};

    // create layer2_1_bn1_output tensor
    const unsigned layer2_1_bn1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_bn1_output =
        createTensor(4U, syn_type_bf16, layer2_1_bn1_output_sizes, false, "layer2_1_bn1_output");

    // create layer2_1_bn1_saved_mean tensor
    const unsigned layer2_1_bn1_saved_mean_sizes[] = {128};
    synTensor      layer2_1_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer2_1_bn1_saved_mean_sizes, false, "layer2_1_bn1_saved_mean");

    // create layer2_1_bn1_saved_var tensor
    const unsigned layer2_1_bn1_saved_var_sizes[] = {128};
    synTensor      layer2_1_bn1_saved_var =
        createTensor(1U, syn_type_single, layer2_1_bn1_saved_var_sizes, false, "layer2_1_bn1_saved_var");

    synTensor layer2_1_bn1_out_vec[3] = {layer2_1_bn1_output, layer2_1_bn1_saved_mean, layer2_1_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_1_bn1_in_vec,
                           layer2_1_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer2_1_bn1_kernel_params,
                           sizeof(layer2_1_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_1_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_bn1 failed!");

    /*************
     * layer2_1_relu1 node
     * inputs: [layer2_1_bn1_output(64, 28, 28, 128)(dtype=bf16)]
     * output: [layer2_1_relu1_output(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_1_relu1_in_vec[1] = {layer2_1_bn1_output};

    // create layer2_1_relu1_output tensor
    const unsigned layer2_1_relu1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_relu1_output =
        createTensor(4U, syn_type_bf16, layer2_1_relu1_output_sizes, false, "layer2_1_relu1_output");

    synTensor layer2_1_relu1_out_vec[1] = {layer2_1_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer2_1_relu1_in_vec,
                           layer2_1_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_1_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_relu1 failed!");

    /*************
     * layer2_1_conv2 node
     * inputs: [layer2_1_relu1_output(64, 28, 28, 128)(dtype=bf16), layer2_1_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_1_conv2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_1_conv2_kernel_params;
    layer2_1_conv2_kernel_params.dH   = 1;
    layer2_1_conv2_kernel_params.dW   = 1;
    layer2_1_conv2_kernel_params.kH   = 3;
    layer2_1_conv2_kernel_params.kW   = 3;
    layer2_1_conv2_kernel_params.padT = 1;
    layer2_1_conv2_kernel_params.padB = 1;
    layer2_1_conv2_kernel_params.padL = 1;
    layer2_1_conv2_kernel_params.padR = 1;
    layer2_1_conv2_kernel_params.dilH = 1;
    layer2_1_conv2_kernel_params.dilW = 1;

    // create layer2_1_conv2_weight tensor
    const unsigned      layer2_1_conv2_weight_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_1_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer2_1_conv2_weight_tr_info = {"layer2_1_conv2_weight", layer2_1_conv2_weight_dram};
    synTensor           layer2_1_conv2_weight =
        createTensor(4U, syn_type_bf16, layer2_1_conv2_weight_sizes, true, "layer2_1_conv2_weight");

    synTensor layer2_1_conv2_in_vec[4] = {layer2_1_relu1_output, layer2_1_conv2_weight, nullptr, nullptr};

    // create layer2_1_conv2_output tensor
    const unsigned layer2_1_conv2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_conv2_output =
        createTensor(4U, syn_type_bf16, layer2_1_conv2_output_sizes, false, "layer2_1_conv2_output");

    synTensor layer2_1_conv2_out_vec[1] = {layer2_1_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv2_in_vec,
                           layer2_1_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer2_1_conv2_kernel_params,
                           sizeof(layer2_1_conv2_kernel_params),
                           "spatial_convolution",
                           "layer2_1_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv2 failed!");

    /*************
     * layer2_1_bn2 node
     * inputs: [layer2_1_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_1_bn2_bias[128](dtype=float32),
     *layer2_1_bn2_weight[128](dtype=float32), layer2_1_bn2_running_mean[128](dtype=float32),
     *layer2_1_bn2_running_var[128](dtype=float32)] output: [layer2_1_bn2_output(64, 28, 28, 128)(dtype=bf16),
     *layer2_1_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_1_bn2_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_1_bn2_kernel_params;
    layer2_1_bn2_kernel_params.momentum    = 0.1;
    layer2_1_bn2_kernel_params.threshold.f = 1e-05;
    layer2_1_bn2_kernel_params.epsilon     = 1e-05;

    // create layer2_1_bn2_bias tensor
    const unsigned layer2_1_bn2_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer2_1_bn2_bias_tr_info = {"layer2_1_bn2_bias", layer2_1_bn2_bias_dram};
    synTensor layer2_1_bn2_bias = createTensor(1U, syn_type_single, layer2_1_bn2_bias_sizes, true, "layer2_1_bn2_bias");

    // create layer2_1_bn2_weight tensor
    const unsigned layer2_1_bn2_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer2_1_bn2_weight_tr_info = {"layer2_1_bn2_weight", layer2_1_bn2_weight_dram};
    synTensor           layer2_1_bn2_weight =
        createTensor(1U, syn_type_single, layer2_1_bn2_weight_sizes, true, "layer2_1_bn2_weight");

    // create layer2_1_bn2_running_mean tensor
    const unsigned layer2_1_bn2_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_1_bn2_running_mean_tr_info = {"layer2_1_bn2_running_mean",
                                                             layer2_1_bn2_running_mean_dram};
    synTensor           layer2_1_bn2_running_mean =
        createTensor(1U, syn_type_single, layer2_1_bn2_running_mean_sizes, true, "layer2_1_bn2_running_mean");

    // create layer2_1_bn2_running_var tensor
    const unsigned layer2_1_bn2_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer2_1_bn2_running_var_tr_info = {"layer2_1_bn2_running_var", layer2_1_bn2_running_var_dram};
    synTensor           layer2_1_bn2_running_var =
        createTensor(1U, syn_type_single, layer2_1_bn2_running_var_sizes, true, "layer2_1_bn2_running_var");

    synTensor layer2_1_bn2_in_vec[5] = {layer2_1_conv2_output,
                                        layer2_1_bn2_bias,
                                        layer2_1_bn2_weight,
                                        layer2_1_bn2_running_mean,
                                        layer2_1_bn2_running_var};

    // create layer2_1_bn2_output tensor
    const unsigned layer2_1_bn2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_bn2_output =
        createTensor(4U, syn_type_bf16, layer2_1_bn2_output_sizes, false, "layer2_1_bn2_output");

    // create layer2_1_bn2_saved_mean tensor
    const unsigned layer2_1_bn2_saved_mean_sizes[] = {128};
    synTensor      layer2_1_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer2_1_bn2_saved_mean_sizes, false, "layer2_1_bn2_saved_mean");

    // create layer2_1_bn2_saved_var tensor
    const unsigned layer2_1_bn2_saved_var_sizes[] = {128};
    synTensor      layer2_1_bn2_saved_var =
        createTensor(1U, syn_type_single, layer2_1_bn2_saved_var_sizes, false, "layer2_1_bn2_saved_var");

    synTensor layer2_1_bn2_out_vec[3] = {layer2_1_bn2_output, layer2_1_bn2_saved_mean, layer2_1_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_1_bn2_in_vec,
                           layer2_1_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer2_1_bn2_kernel_params,
                           sizeof(layer2_1_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_1_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_bn2 failed!");

    /*************
     * layer2_1_relu2 node
     * inputs: [layer2_1_bn2_output(64, 28, 28, 128)(dtype=bf16)]
     * output: [layer2_1_relu2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_1_relu2_in_vec[1] = {layer2_1_bn2_output};

    // create layer2_1_relu2_output tensor
    const unsigned layer2_1_relu2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_relu2_output =
        createTensor(4U, syn_type_bf16, layer2_1_relu2_output_sizes, false, "layer2_1_relu2_output");

    synTensor layer2_1_relu2_out_vec[1] = {layer2_1_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer2_1_relu2_in_vec,
                           layer2_1_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_1_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_relu2 failed!");

    /*************
     * layer2_1_conv3 node
     * inputs: [layer2_1_relu2_output(64, 28, 28, 128)(dtype=bf16), layer2_1_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_1_conv3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_1_conv3_kernel_params;
    layer2_1_conv3_kernel_params.dH   = 1;
    layer2_1_conv3_kernel_params.dW   = 1;
    layer2_1_conv3_kernel_params.kH   = 1;
    layer2_1_conv3_kernel_params.kW   = 1;
    layer2_1_conv3_kernel_params.padT = 0;
    layer2_1_conv3_kernel_params.padB = 0;
    layer2_1_conv3_kernel_params.padL = 0;
    layer2_1_conv3_kernel_params.padR = 0;
    layer2_1_conv3_kernel_params.dilH = 1;
    layer2_1_conv3_kernel_params.dilW = 1;

    // create layer2_1_conv3_weight tensor
    const unsigned      layer2_1_conv3_weight_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_1_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer2_1_conv3_weight_tr_info = {"layer2_1_conv3_weight", layer2_1_conv3_weight_dram};
    synTensor           layer2_1_conv3_weight =
        createTensor(4U, syn_type_bf16, layer2_1_conv3_weight_sizes, true, "layer2_1_conv3_weight");

    synTensor layer2_1_conv3_in_vec[4] = {layer2_1_relu2_output, layer2_1_conv3_weight, nullptr, nullptr};

    // create layer2_1_conv3_output tensor
    const unsigned layer2_1_conv3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_conv3_output =
        createTensor(4U, syn_type_bf16, layer2_1_conv3_output_sizes, false, "layer2_1_conv3_output");

    synTensor layer2_1_conv3_out_vec[1] = {layer2_1_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv3_in_vec,
                           layer2_1_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer2_1_conv3_kernel_params,
                           sizeof(layer2_1_conv3_kernel_params),
                           "spatial_convolution",
                           "layer2_1_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv3 failed!");

    /*************
     * layer2_1_bn3 node
     * inputs: [layer2_1_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_1_bn3_bias[512](dtype=float32),
     *layer2_1_bn3_weight[512](dtype=float32), layer2_1_bn3_running_mean[512](dtype=float32),
     *layer2_1_bn3_running_var[512](dtype=float32)] output: [layer2_1_bn3_output(64, 28, 28, 512)(dtype=bf16),
     *layer2_1_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_1_bn3_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_1_bn3_kernel_params;
    layer2_1_bn3_kernel_params.momentum    = 0.1;
    layer2_1_bn3_kernel_params.threshold.f = 1e-05;
    layer2_1_bn3_kernel_params.epsilon     = 1e-05;

    // create layer2_1_bn3_bias tensor
    const unsigned layer2_1_bn3_bias_sizes[] = {
        512,
    };
    uint64_t            layer2_1_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer2_1_bn3_bias_tr_info = {"layer2_1_bn3_bias", layer2_1_bn3_bias_dram};
    synTensor layer2_1_bn3_bias = createTensor(1U, syn_type_single, layer2_1_bn3_bias_sizes, true, "layer2_1_bn3_bias");

    // create layer2_1_bn3_weight tensor
    const unsigned layer2_1_bn3_weight_sizes[] = {
        512,
    };
    uint64_t            layer2_1_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer2_1_bn3_weight_tr_info = {"layer2_1_bn3_weight", layer2_1_bn3_weight_dram};
    synTensor           layer2_1_bn3_weight =
        createTensor(1U, syn_type_single, layer2_1_bn3_weight_sizes, true, "layer2_1_bn3_weight");

    // create layer2_1_bn3_running_mean tensor
    const unsigned layer2_1_bn3_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer2_1_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_1_bn3_running_mean_tr_info = {"layer2_1_bn3_running_mean",
                                                             layer2_1_bn3_running_mean_dram};
    synTensor           layer2_1_bn3_running_mean =
        createTensor(1U, syn_type_single, layer2_1_bn3_running_mean_sizes, true, "layer2_1_bn3_running_mean");

    // create layer2_1_bn3_running_var tensor
    const unsigned layer2_1_bn3_running_var_sizes[] = {
        512,
    };
    uint64_t            layer2_1_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer2_1_bn3_running_var_tr_info = {"layer2_1_bn3_running_var", layer2_1_bn3_running_var_dram};
    synTensor           layer2_1_bn3_running_var =
        createTensor(1U, syn_type_single, layer2_1_bn3_running_var_sizes, true, "layer2_1_bn3_running_var");

    synTensor layer2_1_bn3_in_vec[5] = {layer2_1_conv3_output,
                                        layer2_1_bn3_bias,
                                        layer2_1_bn3_weight,
                                        layer2_1_bn3_running_mean,
                                        layer2_1_bn3_running_var};

    // create layer2_1_bn3_output tensor
    const unsigned layer2_1_bn3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_bn3_output =
        createTensor(4U, syn_type_bf16, layer2_1_bn3_output_sizes, false, "layer2_1_bn3_output");

    // create layer2_1_bn3_saved_mean tensor
    const unsigned layer2_1_bn3_saved_mean_sizes[] = {512};
    synTensor      layer2_1_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer2_1_bn3_saved_mean_sizes, false, "layer2_1_bn3_saved_mean");

    // create layer2_1_bn3_saved_var tensor
    const unsigned layer2_1_bn3_saved_var_sizes[] = {512};
    synTensor      layer2_1_bn3_saved_var =
        createTensor(1U, syn_type_single, layer2_1_bn3_saved_var_sizes, false, "layer2_1_bn3_saved_var");

    synTensor layer2_1_bn3_out_vec[3] = {layer2_1_bn3_output, layer2_1_bn3_saved_mean, layer2_1_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_1_bn3_in_vec,
                           layer2_1_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer2_1_bn3_kernel_params,
                           sizeof(layer2_1_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_1_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_bn3 failed!");

    /*************
     * layer2_1_add_residual_fwd0 node
     * inputs: [layer2_1_bn3_output(64, 28, 28, 512)(dtype=bf16), layer2_0_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_1_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_1_add_residual_fwd0_in_vec[2] = {layer2_1_bn3_output, layer2_0_relu3_output};

    // create layer2_1_add_residual_fwd tensor
    const unsigned layer2_1_add_residual_fwd_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer2_1_add_residual_fwd_sizes, false, "layer2_1_add_residual_fwd");

    synTensor layer2_1_add_residual_fwd0_out_vec[1] = {layer2_1_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer2_1_add_residual_fwd0_in_vec,
                           layer2_1_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_1_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_add_residual_fwd0 failed!");

    /*************
     * layer2_1_relu3 node
     * inputs: [layer2_1_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_1_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_1_relu3_in_vec[1] = {layer2_1_add_residual_fwd};

    // create layer2_1_relu3_output tensor
    const unsigned layer2_1_relu3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_relu3_output =
        createTensor(4U, syn_type_bf16, layer2_1_relu3_output_sizes, false, "layer2_1_relu3_output");

    synTensor layer2_1_relu3_out_vec[1] = {layer2_1_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer2_1_relu3_in_vec,
                           layer2_1_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_1_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_relu3 failed!");

    /*************
     * layer2_2_conv1 node
     * inputs: [layer2_1_relu3_output(64, 28, 28, 512)(dtype=bf16), layer2_2_conv1_weight[1, 1, 512, 128](dtype=bf16)]
     * output: [layer2_2_conv1_output(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_2_conv1_kernel_params;
    layer2_2_conv1_kernel_params.dH   = 1;
    layer2_2_conv1_kernel_params.dW   = 1;
    layer2_2_conv1_kernel_params.kH   = 1;
    layer2_2_conv1_kernel_params.kW   = 1;
    layer2_2_conv1_kernel_params.padT = 0;
    layer2_2_conv1_kernel_params.padB = 0;
    layer2_2_conv1_kernel_params.padL = 0;
    layer2_2_conv1_kernel_params.padR = 0;
    layer2_2_conv1_kernel_params.dilH = 1;
    layer2_2_conv1_kernel_params.dilW = 1;

    // create layer2_2_conv1_weight tensor
    const unsigned      layer2_2_conv1_weight_sizes[] = {1, 1, 512, 128};
    uint64_t            layer2_2_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer2_2_conv1_weight_tr_info = {"layer2_2_conv1_weight", layer2_2_conv1_weight_dram};
    synTensor           layer2_2_conv1_weight =
        createTensor(4U, syn_type_bf16, layer2_2_conv1_weight_sizes, true, "layer2_2_conv1_weight");

    synTensor layer2_2_conv1_in_vec[4] = {layer2_1_relu3_output, layer2_2_conv1_weight, nullptr, nullptr};

    // create layer2_2_conv1_output tensor
    const unsigned layer2_2_conv1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_conv1_output =
        createTensor(4U, syn_type_bf16, layer2_2_conv1_output_sizes, false, "layer2_2_conv1_output");

    synTensor layer2_2_conv1_out_vec[1] = {layer2_2_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv1_in_vec,
                           layer2_2_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer2_2_conv1_kernel_params,
                           sizeof(layer2_2_conv1_kernel_params),
                           "spatial_convolution",
                           "layer2_2_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv1 failed!");

    /*************
     * layer2_2_bn1 node
     * inputs: [layer2_2_conv1_output(64, 28, 28, 128)(dtype=bf16), layer2_2_bn1_bias[128](dtype=float32),
     *layer2_2_bn1_weight[128](dtype=float32), layer2_2_bn1_running_mean[128](dtype=float32),
     *layer2_2_bn1_running_var[128](dtype=float32)] output: [layer2_2_bn1_output(64, 28, 28, 128)(dtype=bf16),
     *layer2_2_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_2_bn1_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_2_bn1_kernel_params;
    layer2_2_bn1_kernel_params.momentum    = 0.1;
    layer2_2_bn1_kernel_params.threshold.f = 1e-05;
    layer2_2_bn1_kernel_params.epsilon     = 1e-05;

    // create layer2_2_bn1_bias tensor
    const unsigned layer2_2_bn1_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer2_2_bn1_bias_tr_info = {"layer2_2_bn1_bias", layer2_2_bn1_bias_dram};
    synTensor layer2_2_bn1_bias = createTensor(1U, syn_type_single, layer2_2_bn1_bias_sizes, true, "layer2_2_bn1_bias");

    // create layer2_2_bn1_weight tensor
    const unsigned layer2_2_bn1_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer2_2_bn1_weight_tr_info = {"layer2_2_bn1_weight", layer2_2_bn1_weight_dram};
    synTensor           layer2_2_bn1_weight =
        createTensor(1U, syn_type_single, layer2_2_bn1_weight_sizes, true, "layer2_2_bn1_weight");

    // create layer2_2_bn1_running_mean tensor
    const unsigned layer2_2_bn1_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_2_bn1_running_mean_tr_info = {"layer2_2_bn1_running_mean",
                                                             layer2_2_bn1_running_mean_dram};
    synTensor           layer2_2_bn1_running_mean =
        createTensor(1U, syn_type_single, layer2_2_bn1_running_mean_sizes, true, "layer2_2_bn1_running_mean");

    // create layer2_2_bn1_running_var tensor
    const unsigned layer2_2_bn1_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer2_2_bn1_running_var_tr_info = {"layer2_2_bn1_running_var", layer2_2_bn1_running_var_dram};
    synTensor           layer2_2_bn1_running_var =
        createTensor(1U, syn_type_single, layer2_2_bn1_running_var_sizes, true, "layer2_2_bn1_running_var");

    synTensor layer2_2_bn1_in_vec[5] = {layer2_2_conv1_output,
                                        layer2_2_bn1_bias,
                                        layer2_2_bn1_weight,
                                        layer2_2_bn1_running_mean,
                                        layer2_2_bn1_running_var};

    // create layer2_2_bn1_output tensor
    const unsigned layer2_2_bn1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_bn1_output =
        createTensor(4U, syn_type_bf16, layer2_2_bn1_output_sizes, false, "layer2_2_bn1_output");

    // create layer2_2_bn1_saved_mean tensor
    const unsigned layer2_2_bn1_saved_mean_sizes[] = {128};
    synTensor      layer2_2_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer2_2_bn1_saved_mean_sizes, false, "layer2_2_bn1_saved_mean");

    // create layer2_2_bn1_saved_var tensor
    const unsigned layer2_2_bn1_saved_var_sizes[] = {128};
    synTensor      layer2_2_bn1_saved_var =
        createTensor(1U, syn_type_single, layer2_2_bn1_saved_var_sizes, false, "layer2_2_bn1_saved_var");

    synTensor layer2_2_bn1_out_vec[3] = {layer2_2_bn1_output, layer2_2_bn1_saved_mean, layer2_2_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_2_bn1_in_vec,
                           layer2_2_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer2_2_bn1_kernel_params,
                           sizeof(layer2_2_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_2_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_bn1 failed!");

    /*************
     * layer2_2_relu1 node
     * inputs: [layer2_2_bn1_output(64, 28, 28, 128)(dtype=bf16)]
     * output: [layer2_2_relu1_output(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_2_relu1_in_vec[1] = {layer2_2_bn1_output};

    // create layer2_2_relu1_output tensor
    const unsigned layer2_2_relu1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_relu1_output =
        createTensor(4U, syn_type_bf16, layer2_2_relu1_output_sizes, false, "layer2_2_relu1_output");

    synTensor layer2_2_relu1_out_vec[1] = {layer2_2_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer2_2_relu1_in_vec,
                           layer2_2_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_2_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_relu1 failed!");

    /*************
     * layer2_2_conv2 node
     * inputs: [layer2_2_relu1_output(64, 28, 28, 128)(dtype=bf16), layer2_2_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_2_conv2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_2_conv2_kernel_params;
    layer2_2_conv2_kernel_params.dH   = 1;
    layer2_2_conv2_kernel_params.dW   = 1;
    layer2_2_conv2_kernel_params.kH   = 3;
    layer2_2_conv2_kernel_params.kW   = 3;
    layer2_2_conv2_kernel_params.padT = 1;
    layer2_2_conv2_kernel_params.padB = 1;
    layer2_2_conv2_kernel_params.padL = 1;
    layer2_2_conv2_kernel_params.padR = 1;
    layer2_2_conv2_kernel_params.dilH = 1;
    layer2_2_conv2_kernel_params.dilW = 1;

    // create layer2_2_conv2_weight tensor
    const unsigned      layer2_2_conv2_weight_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_2_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer2_2_conv2_weight_tr_info = {"layer2_2_conv2_weight", layer2_2_conv2_weight_dram};
    synTensor           layer2_2_conv2_weight =
        createTensor(4U, syn_type_bf16, layer2_2_conv2_weight_sizes, true, "layer2_2_conv2_weight");

    synTensor layer2_2_conv2_in_vec[4] = {layer2_2_relu1_output, layer2_2_conv2_weight, nullptr, nullptr};

    // create layer2_2_conv2_output tensor
    const unsigned layer2_2_conv2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_conv2_output =
        createTensor(4U, syn_type_bf16, layer2_2_conv2_output_sizes, false, "layer2_2_conv2_output");

    synTensor layer2_2_conv2_out_vec[1] = {layer2_2_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv2_in_vec,
                           layer2_2_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer2_2_conv2_kernel_params,
                           sizeof(layer2_2_conv2_kernel_params),
                           "spatial_convolution",
                           "layer2_2_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv2 failed!");

    /*************
     * layer2_2_bn2 node
     * inputs: [layer2_2_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_2_bn2_bias[128](dtype=float32),
     *layer2_2_bn2_weight[128](dtype=float32), layer2_2_bn2_running_mean[128](dtype=float32),
     *layer2_2_bn2_running_var[128](dtype=float32)] output: [layer2_2_bn2_output(64, 28, 28, 128)(dtype=bf16),
     *layer2_2_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_2_bn2_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_2_bn2_kernel_params;
    layer2_2_bn2_kernel_params.momentum    = 0.1;
    layer2_2_bn2_kernel_params.threshold.f = 1e-05;
    layer2_2_bn2_kernel_params.epsilon     = 1e-05;

    // create layer2_2_bn2_bias tensor
    const unsigned layer2_2_bn2_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer2_2_bn2_bias_tr_info = {"layer2_2_bn2_bias", layer2_2_bn2_bias_dram};
    synTensor layer2_2_bn2_bias = createTensor(1U, syn_type_single, layer2_2_bn2_bias_sizes, true, "layer2_2_bn2_bias");

    // create layer2_2_bn2_weight tensor
    const unsigned layer2_2_bn2_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer2_2_bn2_weight_tr_info = {"layer2_2_bn2_weight", layer2_2_bn2_weight_dram};
    synTensor           layer2_2_bn2_weight =
        createTensor(1U, syn_type_single, layer2_2_bn2_weight_sizes, true, "layer2_2_bn2_weight");

    // create layer2_2_bn2_running_mean tensor
    const unsigned layer2_2_bn2_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_2_bn2_running_mean_tr_info = {"layer2_2_bn2_running_mean",
                                                             layer2_2_bn2_running_mean_dram};
    synTensor           layer2_2_bn2_running_mean =
        createTensor(1U, syn_type_single, layer2_2_bn2_running_mean_sizes, true, "layer2_2_bn2_running_mean");

    // create layer2_2_bn2_running_var tensor
    const unsigned layer2_2_bn2_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer2_2_bn2_running_var_tr_info = {"layer2_2_bn2_running_var", layer2_2_bn2_running_var_dram};
    synTensor           layer2_2_bn2_running_var =
        createTensor(1U, syn_type_single, layer2_2_bn2_running_var_sizes, true, "layer2_2_bn2_running_var");

    synTensor layer2_2_bn2_in_vec[5] = {layer2_2_conv2_output,
                                        layer2_2_bn2_bias,
                                        layer2_2_bn2_weight,
                                        layer2_2_bn2_running_mean,
                                        layer2_2_bn2_running_var};

    // create layer2_2_bn2_output tensor
    const unsigned layer2_2_bn2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_bn2_output =
        createTensor(4U, syn_type_bf16, layer2_2_bn2_output_sizes, false, "layer2_2_bn2_output");

    // create layer2_2_bn2_saved_mean tensor
    const unsigned layer2_2_bn2_saved_mean_sizes[] = {128};
    synTensor      layer2_2_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer2_2_bn2_saved_mean_sizes, false, "layer2_2_bn2_saved_mean");

    // create layer2_2_bn2_saved_var tensor
    const unsigned layer2_2_bn2_saved_var_sizes[] = {128};
    synTensor      layer2_2_bn2_saved_var =
        createTensor(1U, syn_type_single, layer2_2_bn2_saved_var_sizes, false, "layer2_2_bn2_saved_var");

    synTensor layer2_2_bn2_out_vec[3] = {layer2_2_bn2_output, layer2_2_bn2_saved_mean, layer2_2_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_2_bn2_in_vec,
                           layer2_2_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer2_2_bn2_kernel_params,
                           sizeof(layer2_2_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_2_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_bn2 failed!");

    /*************
     * layer2_2_relu2 node
     * inputs: [layer2_2_bn2_output(64, 28, 28, 128)(dtype=bf16)]
     * output: [layer2_2_relu2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_2_relu2_in_vec[1] = {layer2_2_bn2_output};

    // create layer2_2_relu2_output tensor
    const unsigned layer2_2_relu2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_relu2_output =
        createTensor(4U, syn_type_bf16, layer2_2_relu2_output_sizes, false, "layer2_2_relu2_output");

    synTensor layer2_2_relu2_out_vec[1] = {layer2_2_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer2_2_relu2_in_vec,
                           layer2_2_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_2_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_relu2 failed!");

    /*************
     * layer2_2_conv3 node
     * inputs: [layer2_2_relu2_output(64, 28, 28, 128)(dtype=bf16), layer2_2_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_2_conv3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_2_conv3_kernel_params;
    layer2_2_conv3_kernel_params.dH   = 1;
    layer2_2_conv3_kernel_params.dW   = 1;
    layer2_2_conv3_kernel_params.kH   = 1;
    layer2_2_conv3_kernel_params.kW   = 1;
    layer2_2_conv3_kernel_params.padT = 0;
    layer2_2_conv3_kernel_params.padB = 0;
    layer2_2_conv3_kernel_params.padL = 0;
    layer2_2_conv3_kernel_params.padR = 0;
    layer2_2_conv3_kernel_params.dilH = 1;
    layer2_2_conv3_kernel_params.dilW = 1;

    // create layer2_2_conv3_weight tensor
    const unsigned      layer2_2_conv3_weight_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_2_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer2_2_conv3_weight_tr_info = {"layer2_2_conv3_weight", layer2_2_conv3_weight_dram};
    synTensor           layer2_2_conv3_weight =
        createTensor(4U, syn_type_bf16, layer2_2_conv3_weight_sizes, true, "layer2_2_conv3_weight");

    synTensor layer2_2_conv3_in_vec[4] = {layer2_2_relu2_output, layer2_2_conv3_weight, nullptr, nullptr};

    // create layer2_2_conv3_output tensor
    const unsigned layer2_2_conv3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_conv3_output =
        createTensor(4U, syn_type_bf16, layer2_2_conv3_output_sizes, false, "layer2_2_conv3_output");

    synTensor layer2_2_conv3_out_vec[1] = {layer2_2_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv3_in_vec,
                           layer2_2_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer2_2_conv3_kernel_params,
                           sizeof(layer2_2_conv3_kernel_params),
                           "spatial_convolution",
                           "layer2_2_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv3 failed!");

    /*************
     * layer2_2_bn3 node
     * inputs: [layer2_2_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_2_bn3_bias[512](dtype=float32),
     *layer2_2_bn3_weight[512](dtype=float32), layer2_2_bn3_running_mean[512](dtype=float32),
     *layer2_2_bn3_running_var[512](dtype=float32)] output: [layer2_2_bn3_output(64, 28, 28, 512)(dtype=bf16),
     *layer2_2_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_2_bn3_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_2_bn3_kernel_params;
    layer2_2_bn3_kernel_params.momentum    = 0.1;
    layer2_2_bn3_kernel_params.threshold.f = 1e-05;
    layer2_2_bn3_kernel_params.epsilon     = 1e-05;

    // create layer2_2_bn3_bias tensor
    const unsigned layer2_2_bn3_bias_sizes[] = {
        512,
    };
    uint64_t            layer2_2_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer2_2_bn3_bias_tr_info = {"layer2_2_bn3_bias", layer2_2_bn3_bias_dram};
    synTensor layer2_2_bn3_bias = createTensor(1U, syn_type_single, layer2_2_bn3_bias_sizes, true, "layer2_2_bn3_bias");

    // create layer2_2_bn3_weight tensor
    const unsigned layer2_2_bn3_weight_sizes[] = {
        512,
    };
    uint64_t            layer2_2_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer2_2_bn3_weight_tr_info = {"layer2_2_bn3_weight", layer2_2_bn3_weight_dram};
    synTensor           layer2_2_bn3_weight =
        createTensor(1U, syn_type_single, layer2_2_bn3_weight_sizes, true, "layer2_2_bn3_weight");

    // create layer2_2_bn3_running_mean tensor
    const unsigned layer2_2_bn3_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer2_2_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_2_bn3_running_mean_tr_info = {"layer2_2_bn3_running_mean",
                                                             layer2_2_bn3_running_mean_dram};
    synTensor           layer2_2_bn3_running_mean =
        createTensor(1U, syn_type_single, layer2_2_bn3_running_mean_sizes, true, "layer2_2_bn3_running_mean");

    // create layer2_2_bn3_running_var tensor
    const unsigned layer2_2_bn3_running_var_sizes[] = {
        512,
    };
    uint64_t            layer2_2_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer2_2_bn3_running_var_tr_info = {"layer2_2_bn3_running_var", layer2_2_bn3_running_var_dram};
    synTensor           layer2_2_bn3_running_var =
        createTensor(1U, syn_type_single, layer2_2_bn3_running_var_sizes, true, "layer2_2_bn3_running_var");

    synTensor layer2_2_bn3_in_vec[5] = {layer2_2_conv3_output,
                                        layer2_2_bn3_bias,
                                        layer2_2_bn3_weight,
                                        layer2_2_bn3_running_mean,
                                        layer2_2_bn3_running_var};

    // create layer2_2_bn3_output tensor
    const unsigned layer2_2_bn3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_bn3_output =
        createTensor(4U, syn_type_bf16, layer2_2_bn3_output_sizes, false, "layer2_2_bn3_output");

    // create layer2_2_bn3_saved_mean tensor
    const unsigned layer2_2_bn3_saved_mean_sizes[] = {512};
    synTensor      layer2_2_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer2_2_bn3_saved_mean_sizes, false, "layer2_2_bn3_saved_mean");

    // create layer2_2_bn3_saved_var tensor
    const unsigned layer2_2_bn3_saved_var_sizes[] = {512};
    synTensor      layer2_2_bn3_saved_var =
        createTensor(1U, syn_type_single, layer2_2_bn3_saved_var_sizes, false, "layer2_2_bn3_saved_var");

    synTensor layer2_2_bn3_out_vec[3] = {layer2_2_bn3_output, layer2_2_bn3_saved_mean, layer2_2_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_2_bn3_in_vec,
                           layer2_2_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer2_2_bn3_kernel_params,
                           sizeof(layer2_2_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_2_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_bn3 failed!");

    /*************
     * layer2_2_add_residual_fwd0 node
     * inputs: [layer2_2_bn3_output(64, 28, 28, 512)(dtype=bf16), layer2_1_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_2_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_2_add_residual_fwd0_in_vec[2] = {layer2_2_bn3_output, layer2_1_relu3_output};

    // create layer2_2_add_residual_fwd tensor
    const unsigned layer2_2_add_residual_fwd_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer2_2_add_residual_fwd_sizes, false, "layer2_2_add_residual_fwd");

    synTensor layer2_2_add_residual_fwd0_out_vec[1] = {layer2_2_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer2_2_add_residual_fwd0_in_vec,
                           layer2_2_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_2_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_add_residual_fwd0 failed!");

    /*************
     * layer2_2_relu3 node
     * inputs: [layer2_2_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_2_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_2_relu3_in_vec[1] = {layer2_2_add_residual_fwd};

    // create layer2_2_relu3_output tensor
    const unsigned layer2_2_relu3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_relu3_output =
        createTensor(4U, syn_type_bf16, layer2_2_relu3_output_sizes, false, "layer2_2_relu3_output");

    synTensor layer2_2_relu3_out_vec[1] = {layer2_2_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer2_2_relu3_in_vec,
                           layer2_2_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_2_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_relu3 failed!");

    /*************
     * layer2_3_conv1 node
     * inputs: [layer2_2_relu3_output(64, 28, 28, 512)(dtype=bf16), layer2_3_conv1_weight[1, 1, 512, 128](dtype=bf16)]
     * output: [layer2_3_conv1_output(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_3_conv1_kernel_params;
    layer2_3_conv1_kernel_params.dH   = 1;
    layer2_3_conv1_kernel_params.dW   = 1;
    layer2_3_conv1_kernel_params.kH   = 1;
    layer2_3_conv1_kernel_params.kW   = 1;
    layer2_3_conv1_kernel_params.padT = 0;
    layer2_3_conv1_kernel_params.padB = 0;
    layer2_3_conv1_kernel_params.padL = 0;
    layer2_3_conv1_kernel_params.padR = 0;
    layer2_3_conv1_kernel_params.dilH = 1;
    layer2_3_conv1_kernel_params.dilW = 1;

    // create layer2_3_conv1_weight tensor
    const unsigned      layer2_3_conv1_weight_sizes[] = {1, 1, 512, 128};
    uint64_t            layer2_3_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer2_3_conv1_weight_tr_info = {"layer2_3_conv1_weight", layer2_3_conv1_weight_dram};
    synTensor           layer2_3_conv1_weight =
        createTensor(4U, syn_type_bf16, layer2_3_conv1_weight_sizes, true, "layer2_3_conv1_weight");

    synTensor layer2_3_conv1_in_vec[4] = {layer2_2_relu3_output, layer2_3_conv1_weight, nullptr, nullptr};

    // create layer2_3_conv1_output tensor
    const unsigned layer2_3_conv1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_conv1_output =
        createTensor(4U, syn_type_bf16, layer2_3_conv1_output_sizes, false, "layer2_3_conv1_output");

    synTensor layer2_3_conv1_out_vec[1] = {layer2_3_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv1_in_vec,
                           layer2_3_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer2_3_conv1_kernel_params,
                           sizeof(layer2_3_conv1_kernel_params),
                           "spatial_convolution",
                           "layer2_3_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv1 failed!");

    /*************
     * layer2_3_bn1 node
     * inputs: [layer2_3_conv1_output(64, 28, 28, 128)(dtype=bf16), layer2_3_bn1_bias[128](dtype=float32),
     *layer2_3_bn1_weight[128](dtype=float32), layer2_3_bn1_running_mean[128](dtype=float32),
     *layer2_3_bn1_running_var[128](dtype=float32)] output: [layer2_3_bn1_output(64, 28, 28, 128)(dtype=bf16),
     *layer2_3_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_3_bn1_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_3_bn1_kernel_params;
    layer2_3_bn1_kernel_params.momentum    = 0.1;
    layer2_3_bn1_kernel_params.threshold.f = 1e-05;
    layer2_3_bn1_kernel_params.epsilon     = 1e-05;

    // create layer2_3_bn1_bias tensor
    const unsigned layer2_3_bn1_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer2_3_bn1_bias_tr_info = {"layer2_3_bn1_bias", layer2_3_bn1_bias_dram};
    synTensor layer2_3_bn1_bias = createTensor(1U, syn_type_single, layer2_3_bn1_bias_sizes, true, "layer2_3_bn1_bias");

    // create layer2_3_bn1_weight tensor
    const unsigned layer2_3_bn1_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer2_3_bn1_weight_tr_info = {"layer2_3_bn1_weight", layer2_3_bn1_weight_dram};
    synTensor           layer2_3_bn1_weight =
        createTensor(1U, syn_type_single, layer2_3_bn1_weight_sizes, true, "layer2_3_bn1_weight");

    // create layer2_3_bn1_running_mean tensor
    const unsigned layer2_3_bn1_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_3_bn1_running_mean_tr_info = {"layer2_3_bn1_running_mean",
                                                             layer2_3_bn1_running_mean_dram};
    synTensor           layer2_3_bn1_running_mean =
        createTensor(1U, syn_type_single, layer2_3_bn1_running_mean_sizes, true, "layer2_3_bn1_running_mean");

    // create layer2_3_bn1_running_var tensor
    const unsigned layer2_3_bn1_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer2_3_bn1_running_var_tr_info = {"layer2_3_bn1_running_var", layer2_3_bn1_running_var_dram};
    synTensor           layer2_3_bn1_running_var =
        createTensor(1U, syn_type_single, layer2_3_bn1_running_var_sizes, true, "layer2_3_bn1_running_var");

    synTensor layer2_3_bn1_in_vec[5] = {layer2_3_conv1_output,
                                        layer2_3_bn1_bias,
                                        layer2_3_bn1_weight,
                                        layer2_3_bn1_running_mean,
                                        layer2_3_bn1_running_var};

    // create layer2_3_bn1_output tensor
    const unsigned layer2_3_bn1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_bn1_output =
        createTensor(4U, syn_type_bf16, layer2_3_bn1_output_sizes, false, "layer2_3_bn1_output");

    // create layer2_3_bn1_saved_mean tensor
    const unsigned layer2_3_bn1_saved_mean_sizes[] = {128};
    synTensor      layer2_3_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer2_3_bn1_saved_mean_sizes, false, "layer2_3_bn1_saved_mean");

    // create layer2_3_bn1_saved_var tensor
    const unsigned layer2_3_bn1_saved_var_sizes[] = {128};
    synTensor      layer2_3_bn1_saved_var =
        createTensor(1U, syn_type_single, layer2_3_bn1_saved_var_sizes, false, "layer2_3_bn1_saved_var");

    synTensor layer2_3_bn1_out_vec[3] = {layer2_3_bn1_output, layer2_3_bn1_saved_mean, layer2_3_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_3_bn1_in_vec,
                           layer2_3_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer2_3_bn1_kernel_params,
                           sizeof(layer2_3_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_3_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_bn1 failed!");

    /*************
     * layer2_3_relu1 node
     * inputs: [layer2_3_bn1_output(64, 28, 28, 128)(dtype=bf16)]
     * output: [layer2_3_relu1_output(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_3_relu1_in_vec[1] = {layer2_3_bn1_output};

    // create layer2_3_relu1_output tensor
    const unsigned layer2_3_relu1_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_relu1_output =
        createTensor(4U, syn_type_bf16, layer2_3_relu1_output_sizes, false, "layer2_3_relu1_output");

    synTensor layer2_3_relu1_out_vec[1] = {layer2_3_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer2_3_relu1_in_vec,
                           layer2_3_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_3_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_relu1 failed!");

    /*************
     * layer2_3_conv2 node
     * inputs: [layer2_3_relu1_output(64, 28, 28, 128)(dtype=bf16), layer2_3_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_3_conv2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_3_conv2_kernel_params;
    layer2_3_conv2_kernel_params.dH   = 1;
    layer2_3_conv2_kernel_params.dW   = 1;
    layer2_3_conv2_kernel_params.kH   = 3;
    layer2_3_conv2_kernel_params.kW   = 3;
    layer2_3_conv2_kernel_params.padT = 1;
    layer2_3_conv2_kernel_params.padB = 1;
    layer2_3_conv2_kernel_params.padL = 1;
    layer2_3_conv2_kernel_params.padR = 1;
    layer2_3_conv2_kernel_params.dilH = 1;
    layer2_3_conv2_kernel_params.dilW = 1;

    // create layer2_3_conv2_weight tensor
    const unsigned      layer2_3_conv2_weight_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_3_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer2_3_conv2_weight_tr_info = {"layer2_3_conv2_weight", layer2_3_conv2_weight_dram};
    synTensor           layer2_3_conv2_weight =
        createTensor(4U, syn_type_bf16, layer2_3_conv2_weight_sizes, true, "layer2_3_conv2_weight");

    synTensor layer2_3_conv2_in_vec[4] = {layer2_3_relu1_output, layer2_3_conv2_weight, nullptr, nullptr};

    // create layer2_3_conv2_output tensor
    const unsigned layer2_3_conv2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_conv2_output =
        createTensor(4U, syn_type_bf16, layer2_3_conv2_output_sizes, false, "layer2_3_conv2_output");

    synTensor layer2_3_conv2_out_vec[1] = {layer2_3_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv2_in_vec,
                           layer2_3_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer2_3_conv2_kernel_params,
                           sizeof(layer2_3_conv2_kernel_params),
                           "spatial_convolution",
                           "layer2_3_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv2 failed!");

    /*************
     * layer2_3_bn2 node
     * inputs: [layer2_3_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_3_bn2_bias[128](dtype=float32),
     *layer2_3_bn2_weight[128](dtype=float32), layer2_3_bn2_running_mean[128](dtype=float32),
     *layer2_3_bn2_running_var[128](dtype=float32)] output: [layer2_3_bn2_output(64, 28, 28, 128)(dtype=bf16),
     *layer2_3_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_3_bn2_saved_var(1, 1, 1, 128)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_3_bn2_kernel_params;
    layer2_3_bn2_kernel_params.momentum    = 0.1;
    layer2_3_bn2_kernel_params.threshold.f = 1e-05;
    layer2_3_bn2_kernel_params.epsilon     = 1e-05;

    // create layer2_3_bn2_bias tensor
    const unsigned layer2_3_bn2_bias_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer2_3_bn2_bias_tr_info = {"layer2_3_bn2_bias", layer2_3_bn2_bias_dram};
    synTensor layer2_3_bn2_bias = createTensor(1U, syn_type_single, layer2_3_bn2_bias_sizes, true, "layer2_3_bn2_bias");

    // create layer2_3_bn2_weight tensor
    const unsigned layer2_3_bn2_weight_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer2_3_bn2_weight_tr_info = {"layer2_3_bn2_weight", layer2_3_bn2_weight_dram};
    synTensor           layer2_3_bn2_weight =
        createTensor(1U, syn_type_single, layer2_3_bn2_weight_sizes, true, "layer2_3_bn2_weight");

    // create layer2_3_bn2_running_mean tensor
    const unsigned layer2_3_bn2_running_mean_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_3_bn2_running_mean_tr_info = {"layer2_3_bn2_running_mean",
                                                             layer2_3_bn2_running_mean_dram};
    synTensor           layer2_3_bn2_running_mean =
        createTensor(1U, syn_type_single, layer2_3_bn2_running_mean_sizes, true, "layer2_3_bn2_running_mean");

    // create layer2_3_bn2_running_var tensor
    const unsigned layer2_3_bn2_running_var_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer2_3_bn2_running_var_tr_info = {"layer2_3_bn2_running_var", layer2_3_bn2_running_var_dram};
    synTensor           layer2_3_bn2_running_var =
        createTensor(1U, syn_type_single, layer2_3_bn2_running_var_sizes, true, "layer2_3_bn2_running_var");

    synTensor layer2_3_bn2_in_vec[5] = {layer2_3_conv2_output,
                                        layer2_3_bn2_bias,
                                        layer2_3_bn2_weight,
                                        layer2_3_bn2_running_mean,
                                        layer2_3_bn2_running_var};

    // create layer2_3_bn2_output tensor
    const unsigned layer2_3_bn2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_bn2_output =
        createTensor(4U, syn_type_bf16, layer2_3_bn2_output_sizes, false, "layer2_3_bn2_output");

    // create layer2_3_bn2_saved_mean tensor
    const unsigned layer2_3_bn2_saved_mean_sizes[] = {128};
    synTensor      layer2_3_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer2_3_bn2_saved_mean_sizes, false, "layer2_3_bn2_saved_mean");

    // create layer2_3_bn2_saved_var tensor
    const unsigned layer2_3_bn2_saved_var_sizes[] = {128};
    synTensor      layer2_3_bn2_saved_var =
        createTensor(1U, syn_type_single, layer2_3_bn2_saved_var_sizes, false, "layer2_3_bn2_saved_var");

    synTensor layer2_3_bn2_out_vec[3] = {layer2_3_bn2_output, layer2_3_bn2_saved_mean, layer2_3_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_3_bn2_in_vec,
                           layer2_3_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer2_3_bn2_kernel_params,
                           sizeof(layer2_3_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_3_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_bn2 failed!");

    /*************
     * layer2_3_relu2 node
     * inputs: [layer2_3_bn2_output(64, 28, 28, 128)(dtype=bf16)]
     * output: [layer2_3_relu2_output(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_3_relu2_in_vec[1] = {layer2_3_bn2_output};

    // create layer2_3_relu2_output tensor
    const unsigned layer2_3_relu2_output_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_relu2_output =
        createTensor(4U, syn_type_bf16, layer2_3_relu2_output_sizes, false, "layer2_3_relu2_output");

    synTensor layer2_3_relu2_out_vec[1] = {layer2_3_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer2_3_relu2_in_vec,
                           layer2_3_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_3_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_relu2 failed!");

    /*************
     * layer2_3_conv3 node
     * inputs: [layer2_3_relu2_output(64, 28, 28, 128)(dtype=bf16), layer2_3_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_3_conv3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_3_conv3_kernel_params;
    layer2_3_conv3_kernel_params.dH   = 1;
    layer2_3_conv3_kernel_params.dW   = 1;
    layer2_3_conv3_kernel_params.kH   = 1;
    layer2_3_conv3_kernel_params.kW   = 1;
    layer2_3_conv3_kernel_params.padT = 0;
    layer2_3_conv3_kernel_params.padB = 0;
    layer2_3_conv3_kernel_params.padL = 0;
    layer2_3_conv3_kernel_params.padR = 0;
    layer2_3_conv3_kernel_params.dilH = 1;
    layer2_3_conv3_kernel_params.dilW = 1;

    // create layer2_3_conv3_weight tensor
    const unsigned      layer2_3_conv3_weight_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_3_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer2_3_conv3_weight_tr_info = {"layer2_3_conv3_weight", layer2_3_conv3_weight_dram};
    synTensor           layer2_3_conv3_weight =
        createTensor(4U, syn_type_bf16, layer2_3_conv3_weight_sizes, true, "layer2_3_conv3_weight");

    synTensor layer2_3_conv3_in_vec[4] = {layer2_3_relu2_output, layer2_3_conv3_weight, nullptr, nullptr};

    // create layer2_3_conv3_output tensor
    const unsigned layer2_3_conv3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_conv3_output =
        createTensor(4U, syn_type_bf16, layer2_3_conv3_output_sizes, false, "layer2_3_conv3_output");

    synTensor layer2_3_conv3_out_vec[1] = {layer2_3_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv3_in_vec,
                           layer2_3_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer2_3_conv3_kernel_params,
                           sizeof(layer2_3_conv3_kernel_params),
                           "spatial_convolution",
                           "layer2_3_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv3 failed!");

    /*************
     * layer2_3_bn3 node
     * inputs: [layer2_3_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_3_bn3_bias[512](dtype=float32),
     *layer2_3_bn3_weight[512](dtype=float32), layer2_3_bn3_running_mean[512](dtype=float32),
     *layer2_3_bn3_running_var[512](dtype=float32)] output: [layer2_3_bn3_output(64, 28, 28, 512)(dtype=bf16),
     *layer2_3_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_3_bn3_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer2_3_bn3_kernel_params;
    layer2_3_bn3_kernel_params.momentum    = 0.1;
    layer2_3_bn3_kernel_params.threshold.f = 1e-05;
    layer2_3_bn3_kernel_params.epsilon     = 1e-05;

    // create layer2_3_bn3_bias tensor
    const unsigned layer2_3_bn3_bias_sizes[] = {
        512,
    };
    uint64_t            layer2_3_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer2_3_bn3_bias_tr_info = {"layer2_3_bn3_bias", layer2_3_bn3_bias_dram};
    synTensor layer2_3_bn3_bias = createTensor(1U, syn_type_single, layer2_3_bn3_bias_sizes, true, "layer2_3_bn3_bias");

    // create layer2_3_bn3_weight tensor
    const unsigned layer2_3_bn3_weight_sizes[] = {
        512,
    };
    uint64_t            layer2_3_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer2_3_bn3_weight_tr_info = {"layer2_3_bn3_weight", layer2_3_bn3_weight_dram};
    synTensor           layer2_3_bn3_weight =
        createTensor(1U, syn_type_single, layer2_3_bn3_weight_sizes, true, "layer2_3_bn3_weight");

    // create layer2_3_bn3_running_mean tensor
    const unsigned layer2_3_bn3_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer2_3_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer2_3_bn3_running_mean_tr_info = {"layer2_3_bn3_running_mean",
                                                             layer2_3_bn3_running_mean_dram};
    synTensor           layer2_3_bn3_running_mean =
        createTensor(1U, syn_type_single, layer2_3_bn3_running_mean_sizes, true, "layer2_3_bn3_running_mean");

    // create layer2_3_bn3_running_var tensor
    const unsigned layer2_3_bn3_running_var_sizes[] = {
        512,
    };
    uint64_t            layer2_3_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer2_3_bn3_running_var_tr_info = {"layer2_3_bn3_running_var", layer2_3_bn3_running_var_dram};
    synTensor           layer2_3_bn3_running_var =
        createTensor(1U, syn_type_single, layer2_3_bn3_running_var_sizes, true, "layer2_3_bn3_running_var");

    synTensor layer2_3_bn3_in_vec[5] = {layer2_3_conv3_output,
                                        layer2_3_bn3_bias,
                                        layer2_3_bn3_weight,
                                        layer2_3_bn3_running_mean,
                                        layer2_3_bn3_running_var};

    // create layer2_3_bn3_output tensor
    const unsigned layer2_3_bn3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_bn3_output =
        createTensor(4U, syn_type_bf16, layer2_3_bn3_output_sizes, false, "layer2_3_bn3_output");

    // create layer2_3_bn3_saved_mean tensor
    const unsigned layer2_3_bn3_saved_mean_sizes[] = {512};
    synTensor      layer2_3_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer2_3_bn3_saved_mean_sizes, false, "layer2_3_bn3_saved_mean");

    // create layer2_3_bn3_saved_var tensor
    const unsigned layer2_3_bn3_saved_var_sizes[] = {512};
    synTensor      layer2_3_bn3_saved_var =
        createTensor(1U, syn_type_single, layer2_3_bn3_saved_var_sizes, false, "layer2_3_bn3_saved_var");

    synTensor layer2_3_bn3_out_vec[3] = {layer2_3_bn3_output, layer2_3_bn3_saved_mean, layer2_3_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer2_3_bn3_in_vec,
                           layer2_3_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer2_3_bn3_kernel_params,
                           sizeof(layer2_3_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer2_3_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_bn3 failed!");

    /*************
     * layer2_3_add_residual_fwd0 node
     * inputs: [layer2_3_bn3_output(64, 28, 28, 512)(dtype=bf16), layer2_2_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_3_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_3_add_residual_fwd0_in_vec[2] = {layer2_3_bn3_output, layer2_2_relu3_output};

    // create layer2_3_add_residual_fwd tensor
    const unsigned layer2_3_add_residual_fwd_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer2_3_add_residual_fwd_sizes, false, "layer2_3_add_residual_fwd");

    synTensor layer2_3_add_residual_fwd0_out_vec[1] = {layer2_3_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer2_3_add_residual_fwd0_in_vec,
                           layer2_3_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_3_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_add_residual_fwd0 failed!");

    /*************
     * layer2_3_relu3 node
     * inputs: [layer2_3_add_residual_fwd(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_3_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_3_relu3_in_vec[1] = {layer2_3_add_residual_fwd};

    // create layer2_3_relu3_output tensor
    const unsigned layer2_3_relu3_output_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_relu3_output =
        createTensor(4U, syn_type_bf16, layer2_3_relu3_output_sizes, false, "layer2_3_relu3_output");

    synTensor layer2_3_relu3_out_vec[1] = {layer2_3_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer2_3_relu3_in_vec,
                           layer2_3_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer2_3_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_relu3 failed!");

    /*************
     * layer3_0_conv1 node
     * inputs: [layer2_3_relu3_output(64, 28, 28, 512)(dtype=bf16), layer3_0_conv1_weight[1, 1, 512, 256](dtype=bf16)]
     * output: [layer3_0_conv1_output(64, 28, 28, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_0_conv1_kernel_params;
    layer3_0_conv1_kernel_params.dH   = 1;
    layer3_0_conv1_kernel_params.dW   = 1;
    layer3_0_conv1_kernel_params.kH   = 1;
    layer3_0_conv1_kernel_params.kW   = 1;
    layer3_0_conv1_kernel_params.padT = 0;
    layer3_0_conv1_kernel_params.padB = 0;
    layer3_0_conv1_kernel_params.padL = 0;
    layer3_0_conv1_kernel_params.padR = 0;
    layer3_0_conv1_kernel_params.dilH = 1;
    layer3_0_conv1_kernel_params.dilW = 1;

    // create layer3_0_conv1_weight tensor
    const unsigned      layer3_0_conv1_weight_sizes[] = {1, 1, 512, 256};
    uint64_t            layer3_0_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer3_0_conv1_weight_tr_info = {"layer3_0_conv1_weight", layer3_0_conv1_weight_dram};
    synTensor           layer3_0_conv1_weight =
        createTensor(4U, syn_type_bf16, layer3_0_conv1_weight_sizes, true, "layer3_0_conv1_weight");

    synTensor layer3_0_conv1_in_vec[4] = {layer2_3_relu3_output, layer3_0_conv1_weight, nullptr, nullptr};

    // create layer3_0_conv1_output tensor
    const unsigned layer3_0_conv1_output_sizes[] = {64, 28, 28, 256};
    synTensor      layer3_0_conv1_output =
        createTensor(4U, syn_type_bf16, layer3_0_conv1_output_sizes, false, "layer3_0_conv1_output");

    synTensor layer3_0_conv1_out_vec[1] = {layer3_0_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv1_in_vec,
                           layer3_0_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer3_0_conv1_kernel_params,
                           sizeof(layer3_0_conv1_kernel_params),
                           "spatial_convolution",
                           "layer3_0_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv1 failed!");

    /*************
     * layer3_0_bn1 node
     * inputs: [layer3_0_conv1_output(64, 28, 28, 256)(dtype=bf16), layer3_0_bn1_bias[256](dtype=float32),
     *layer3_0_bn1_weight[256](dtype=float32), layer3_0_bn1_running_mean[256](dtype=float32),
     *layer3_0_bn1_running_var[256](dtype=float32)] output: [layer3_0_bn1_output(64, 28, 28, 256)(dtype=bf16),
     *layer3_0_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_0_bn1_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_0_bn1_kernel_params;
    layer3_0_bn1_kernel_params.momentum    = 0.1;
    layer3_0_bn1_kernel_params.threshold.f = 1e-05;
    layer3_0_bn1_kernel_params.epsilon     = 1e-05;

    // create layer3_0_bn1_bias tensor
    const unsigned layer3_0_bn1_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer3_0_bn1_bias_tr_info = {"layer3_0_bn1_bias", layer3_0_bn1_bias_dram};
    synTensor layer3_0_bn1_bias = createTensor(1U, syn_type_single, layer3_0_bn1_bias_sizes, true, "layer3_0_bn1_bias");

    // create layer3_0_bn1_weight tensor
    const unsigned layer3_0_bn1_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer3_0_bn1_weight_tr_info = {"layer3_0_bn1_weight", layer3_0_bn1_weight_dram};
    synTensor           layer3_0_bn1_weight =
        createTensor(1U, syn_type_single, layer3_0_bn1_weight_sizes, true, "layer3_0_bn1_weight");

    // create layer3_0_bn1_running_mean tensor
    const unsigned layer3_0_bn1_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_0_bn1_running_mean_tr_info = {"layer3_0_bn1_running_mean",
                                                             layer3_0_bn1_running_mean_dram};
    synTensor           layer3_0_bn1_running_mean =
        createTensor(1U, syn_type_single, layer3_0_bn1_running_mean_sizes, true, "layer3_0_bn1_running_mean");

    // create layer3_0_bn1_running_var tensor
    const unsigned layer3_0_bn1_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer3_0_bn1_running_var_tr_info = {"layer3_0_bn1_running_var", layer3_0_bn1_running_var_dram};
    synTensor           layer3_0_bn1_running_var =
        createTensor(1U, syn_type_single, layer3_0_bn1_running_var_sizes, true, "layer3_0_bn1_running_var");

    synTensor layer3_0_bn1_in_vec[5] = {layer3_0_conv1_output,
                                        layer3_0_bn1_bias,
                                        layer3_0_bn1_weight,
                                        layer3_0_bn1_running_mean,
                                        layer3_0_bn1_running_var};

    // create layer3_0_bn1_output tensor
    const unsigned layer3_0_bn1_output_sizes[] = {64, 28, 28, 256};
    synTensor      layer3_0_bn1_output =
        createTensor(4U, syn_type_bf16, layer3_0_bn1_output_sizes, false, "layer3_0_bn1_output");

    // create layer3_0_bn1_saved_mean tensor
    const unsigned layer3_0_bn1_saved_mean_sizes[] = {256};
    synTensor      layer3_0_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer3_0_bn1_saved_mean_sizes, false, "layer3_0_bn1_saved_mean");

    // create layer3_0_bn1_saved_var tensor
    const unsigned layer3_0_bn1_saved_var_sizes[] = {256};
    synTensor      layer3_0_bn1_saved_var =
        createTensor(1U, syn_type_single, layer3_0_bn1_saved_var_sizes, false, "layer3_0_bn1_saved_var");

    synTensor layer3_0_bn1_out_vec[3] = {layer3_0_bn1_output, layer3_0_bn1_saved_mean, layer3_0_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_0_bn1_in_vec,
                           layer3_0_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer3_0_bn1_kernel_params,
                           sizeof(layer3_0_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_0_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_bn1 failed!");

    /*************
     * layer3_0_relu1 node
     * inputs: [layer3_0_bn1_output(64, 28, 28, 256)(dtype=bf16)]
     * output: [layer3_0_relu1_output(64, 28, 28, 256)(dtype=bf16)]
     *************/

    synTensor layer3_0_relu1_in_vec[1] = {layer3_0_bn1_output};

    // create layer3_0_relu1_output tensor
    const unsigned layer3_0_relu1_output_sizes[] = {64, 28, 28, 256};
    synTensor      layer3_0_relu1_output =
        createTensor(4U, syn_type_bf16, layer3_0_relu1_output_sizes, false, "layer3_0_relu1_output");

    synTensor layer3_0_relu1_out_vec[1] = {layer3_0_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer3_0_relu1_in_vec,
                           layer3_0_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_0_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_relu1 failed!");

    /*************
     * layer3_0_conv2 node
     * inputs: [layer3_0_relu1_output(64, 28, 28, 256)(dtype=bf16), layer3_0_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_0_conv2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_0_conv2_kernel_params;
    layer3_0_conv2_kernel_params.dH   = 2;
    layer3_0_conv2_kernel_params.dW   = 2;
    layer3_0_conv2_kernel_params.kH   = 3;
    layer3_0_conv2_kernel_params.kW   = 3;
    layer3_0_conv2_kernel_params.padT = 1;
    layer3_0_conv2_kernel_params.padB = 1;
    layer3_0_conv2_kernel_params.padL = 1;
    layer3_0_conv2_kernel_params.padR = 1;
    layer3_0_conv2_kernel_params.dilH = 1;
    layer3_0_conv2_kernel_params.dilW = 1;

    // create layer3_0_conv2_weight tensor
    const unsigned      layer3_0_conv2_weight_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_0_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer3_0_conv2_weight_tr_info = {"layer3_0_conv2_weight", layer3_0_conv2_weight_dram};
    synTensor           layer3_0_conv2_weight =
        createTensor(4U, syn_type_bf16, layer3_0_conv2_weight_sizes, true, "layer3_0_conv2_weight");

    synTensor layer3_0_conv2_in_vec[4] = {layer3_0_relu1_output, layer3_0_conv2_weight, nullptr, nullptr};

    // create layer3_0_conv2_output tensor
    const unsigned layer3_0_conv2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_0_conv2_output =
        createTensor(4U, syn_type_bf16, layer3_0_conv2_output_sizes, false, "layer3_0_conv2_output");

    synTensor layer3_0_conv2_out_vec[1] = {layer3_0_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv2_in_vec,
                           layer3_0_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer3_0_conv2_kernel_params,
                           sizeof(layer3_0_conv2_kernel_params),
                           "spatial_convolution",
                           "layer3_0_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv2 failed!");

    /*************
     * layer3_0_bn2 node
     * inputs: [layer3_0_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_0_bn2_bias[256](dtype=float32),
     *layer3_0_bn2_weight[256](dtype=float32), layer3_0_bn2_running_mean[256](dtype=float32),
     *layer3_0_bn2_running_var[256](dtype=float32)] output: [layer3_0_bn2_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_0_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_0_bn2_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_0_bn2_kernel_params;
    layer3_0_bn2_kernel_params.momentum    = 0.1;
    layer3_0_bn2_kernel_params.threshold.f = 1e-05;
    layer3_0_bn2_kernel_params.epsilon     = 1e-05;

    // create layer3_0_bn2_bias tensor
    const unsigned layer3_0_bn2_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer3_0_bn2_bias_tr_info = {"layer3_0_bn2_bias", layer3_0_bn2_bias_dram};
    synTensor layer3_0_bn2_bias = createTensor(1U, syn_type_single, layer3_0_bn2_bias_sizes, true, "layer3_0_bn2_bias");

    // create layer3_0_bn2_weight tensor
    const unsigned layer3_0_bn2_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer3_0_bn2_weight_tr_info = {"layer3_0_bn2_weight", layer3_0_bn2_weight_dram};
    synTensor           layer3_0_bn2_weight =
        createTensor(1U, syn_type_single, layer3_0_bn2_weight_sizes, true, "layer3_0_bn2_weight");

    // create layer3_0_bn2_running_mean tensor
    const unsigned layer3_0_bn2_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_0_bn2_running_mean_tr_info = {"layer3_0_bn2_running_mean",
                                                             layer3_0_bn2_running_mean_dram};
    synTensor           layer3_0_bn2_running_mean =
        createTensor(1U, syn_type_single, layer3_0_bn2_running_mean_sizes, true, "layer3_0_bn2_running_mean");

    // create layer3_0_bn2_running_var tensor
    const unsigned layer3_0_bn2_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer3_0_bn2_running_var_tr_info = {"layer3_0_bn2_running_var", layer3_0_bn2_running_var_dram};
    synTensor           layer3_0_bn2_running_var =
        createTensor(1U, syn_type_single, layer3_0_bn2_running_var_sizes, true, "layer3_0_bn2_running_var");

    synTensor layer3_0_bn2_in_vec[5] = {layer3_0_conv2_output,
                                        layer3_0_bn2_bias,
                                        layer3_0_bn2_weight,
                                        layer3_0_bn2_running_mean,
                                        layer3_0_bn2_running_var};

    // create layer3_0_bn2_output tensor
    const unsigned layer3_0_bn2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_0_bn2_output =
        createTensor(4U, syn_type_bf16, layer3_0_bn2_output_sizes, false, "layer3_0_bn2_output");

    // create layer3_0_bn2_saved_mean tensor
    const unsigned layer3_0_bn2_saved_mean_sizes[] = {256};
    synTensor      layer3_0_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer3_0_bn2_saved_mean_sizes, false, "layer3_0_bn2_saved_mean");

    // create layer3_0_bn2_saved_var tensor
    const unsigned layer3_0_bn2_saved_var_sizes[] = {256};
    synTensor      layer3_0_bn2_saved_var =
        createTensor(1U, syn_type_single, layer3_0_bn2_saved_var_sizes, false, "layer3_0_bn2_saved_var");

    synTensor layer3_0_bn2_out_vec[3] = {layer3_0_bn2_output, layer3_0_bn2_saved_mean, layer3_0_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_0_bn2_in_vec,
                           layer3_0_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer3_0_bn2_kernel_params,
                           sizeof(layer3_0_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_0_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_bn2 failed!");

    /*************
     * layer3_0_relu2 node
     * inputs: [layer3_0_bn2_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_0_relu2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_0_relu2_in_vec[1] = {layer3_0_bn2_output};

    // create layer3_0_relu2_output tensor
    const unsigned layer3_0_relu2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_0_relu2_output =
        createTensor(4U, syn_type_bf16, layer3_0_relu2_output_sizes, false, "layer3_0_relu2_output");

    synTensor layer3_0_relu2_out_vec[1] = {layer3_0_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer3_0_relu2_in_vec,
                           layer3_0_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_0_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_relu2 failed!");

    /*************
     * layer3_0_conv3 node
     * inputs: [layer3_0_relu2_output(64, 14, 14, 256)(dtype=bf16), layer3_0_conv3_weight[1, 1, 256, 1024](dtype=bf16)]
     * output: [layer3_0_conv3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_0_conv3_kernel_params;
    layer3_0_conv3_kernel_params.dH   = 1;
    layer3_0_conv3_kernel_params.dW   = 1;
    layer3_0_conv3_kernel_params.kH   = 1;
    layer3_0_conv3_kernel_params.kW   = 1;
    layer3_0_conv3_kernel_params.padT = 0;
    layer3_0_conv3_kernel_params.padB = 0;
    layer3_0_conv3_kernel_params.padL = 0;
    layer3_0_conv3_kernel_params.padR = 0;
    layer3_0_conv3_kernel_params.dilH = 1;
    layer3_0_conv3_kernel_params.dilW = 1;

    // create layer3_0_conv3_weight tensor
    const unsigned      layer3_0_conv3_weight_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_0_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer3_0_conv3_weight_tr_info = {"layer3_0_conv3_weight", layer3_0_conv3_weight_dram};
    synTensor           layer3_0_conv3_weight =
        createTensor(4U, syn_type_bf16, layer3_0_conv3_weight_sizes, true, "layer3_0_conv3_weight");

    synTensor layer3_0_conv3_in_vec[4] = {layer3_0_relu2_output, layer3_0_conv3_weight, nullptr, nullptr};

    // create layer3_0_conv3_output tensor
    const unsigned layer3_0_conv3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_conv3_output =
        createTensor(4U, syn_type_bf16, layer3_0_conv3_output_sizes, false, "layer3_0_conv3_output");

    synTensor layer3_0_conv3_out_vec[1] = {layer3_0_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv3_in_vec,
                           layer3_0_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer3_0_conv3_kernel_params,
                           sizeof(layer3_0_conv3_kernel_params),
                           "spatial_convolution",
                           "layer3_0_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv3 failed!");

    /*************
     * layer3_0_bn3 node
     * inputs: [layer3_0_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_0_bn3_bias[1024](dtype=float32),
     *layer3_0_bn3_weight[1024](dtype=float32), layer3_0_bn3_running_mean[1024](dtype=float32),
     *layer3_0_bn3_running_var[1024](dtype=float32)] output: [layer3_0_bn3_output(64, 14, 14, 1024)(dtype=bf16),
     *layer3_0_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_0_bn3_saved_var(1, 1, 1, 1024)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_0_bn3_kernel_params;
    layer3_0_bn3_kernel_params.momentum    = 0.1;
    layer3_0_bn3_kernel_params.threshold.f = 1e-05;
    layer3_0_bn3_kernel_params.epsilon     = 1e-05;

    // create layer3_0_bn3_bias tensor
    const unsigned layer3_0_bn3_bias_sizes[] = {
        1024,
    };
    uint64_t            layer3_0_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer3_0_bn3_bias_tr_info = {"layer3_0_bn3_bias", layer3_0_bn3_bias_dram};
    synTensor layer3_0_bn3_bias = createTensor(1U, syn_type_single, layer3_0_bn3_bias_sizes, true, "layer3_0_bn3_bias");

    // create layer3_0_bn3_weight tensor
    const unsigned layer3_0_bn3_weight_sizes[] = {
        1024,
    };
    uint64_t            layer3_0_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer3_0_bn3_weight_tr_info = {"layer3_0_bn3_weight", layer3_0_bn3_weight_dram};
    synTensor           layer3_0_bn3_weight =
        createTensor(1U, syn_type_single, layer3_0_bn3_weight_sizes, true, "layer3_0_bn3_weight");

    // create layer3_0_bn3_running_mean tensor
    const unsigned layer3_0_bn3_running_mean_sizes[] = {
        1024,
    };
    uint64_t            layer3_0_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_0_bn3_running_mean_tr_info = {"layer3_0_bn3_running_mean",
                                                             layer3_0_bn3_running_mean_dram};
    synTensor           layer3_0_bn3_running_mean =
        createTensor(1U, syn_type_single, layer3_0_bn3_running_mean_sizes, true, "layer3_0_bn3_running_mean");

    // create layer3_0_bn3_running_var tensor
    const unsigned layer3_0_bn3_running_var_sizes[] = {
        1024,
    };
    uint64_t            layer3_0_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer3_0_bn3_running_var_tr_info = {"layer3_0_bn3_running_var", layer3_0_bn3_running_var_dram};
    synTensor           layer3_0_bn3_running_var =
        createTensor(1U, syn_type_single, layer3_0_bn3_running_var_sizes, true, "layer3_0_bn3_running_var");

    synTensor layer3_0_bn3_in_vec[5] = {layer3_0_conv3_output,
                                        layer3_0_bn3_bias,
                                        layer3_0_bn3_weight,
                                        layer3_0_bn3_running_mean,
                                        layer3_0_bn3_running_var};

    // create layer3_0_bn3_output tensor
    const unsigned layer3_0_bn3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_bn3_output =
        createTensor(4U, syn_type_bf16, layer3_0_bn3_output_sizes, false, "layer3_0_bn3_output");

    // create layer3_0_bn3_saved_mean tensor
    const unsigned layer3_0_bn3_saved_mean_sizes[] = {1024};
    synTensor      layer3_0_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer3_0_bn3_saved_mean_sizes, false, "layer3_0_bn3_saved_mean");

    // create layer3_0_bn3_saved_var tensor
    const unsigned layer3_0_bn3_saved_var_sizes[] = {1024};
    synTensor      layer3_0_bn3_saved_var =
        createTensor(1U, syn_type_single, layer3_0_bn3_saved_var_sizes, false, "layer3_0_bn3_saved_var");

    synTensor layer3_0_bn3_out_vec[3] = {layer3_0_bn3_output, layer3_0_bn3_saved_mean, layer3_0_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_0_bn3_in_vec,
                           layer3_0_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer3_0_bn3_kernel_params,
                           sizeof(layer3_0_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_0_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_bn3 failed!");

    /*************
     * layer3_downsample node
     * inputs: [layer2_3_relu3_output(64, 28, 28, 512)(dtype=bf16), layer3_downsample_weight[1, 1, 512,
     *1024](dtype=bf16)] output: [layer3_downsample_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_downsample_kernel_params;
    layer3_downsample_kernel_params.dH   = 2;
    layer3_downsample_kernel_params.dW   = 2;
    layer3_downsample_kernel_params.kH   = 1;
    layer3_downsample_kernel_params.kW   = 1;
    layer3_downsample_kernel_params.padT = 0;
    layer3_downsample_kernel_params.padB = 0;
    layer3_downsample_kernel_params.padL = 0;
    layer3_downsample_kernel_params.padR = 0;
    layer3_downsample_kernel_params.dilH = 1;
    layer3_downsample_kernel_params.dilW = 1;

    // create layer3_downsample_weight tensor
    const unsigned      layer3_downsample_weight_sizes[] = {1, 1, 512, 1024};
    uint64_t            layer3_downsample_weight_dram    = 0;
    synLaunchTensorInfo layer3_downsample_weight_tr_info = {"layer3_downsample_weight", layer3_downsample_weight_dram};
    synTensor           layer3_downsample_weight =
        createTensor(4U, syn_type_bf16, layer3_downsample_weight_sizes, true, "layer3_downsample_weight");

    synTensor layer3_downsample_in_vec[4] = {layer2_3_relu3_output, layer3_downsample_weight, nullptr, nullptr};

    // create layer3_downsample_output tensor
    const unsigned layer3_downsample_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_downsample_output =
        createTensor(4U, syn_type_bf16, layer3_downsample_output_sizes, false, "layer3_downsample_output");

    synTensor layer3_downsample_out_vec[1] = {layer3_downsample_output};

    status = synNodeCreate(graphHandle,
                           layer3_downsample_in_vec,
                           layer3_downsample_out_vec,
                           4,
                           1,
                           (void*)&layer3_downsample_kernel_params,
                           sizeof(layer3_downsample_kernel_params),
                           "spatial_convolution",
                           "layer3_downsample",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_downsample failed!");

    /*************
     * layer3_bn node
     * inputs: [layer3_downsample_output(64, 14, 14, 1024)(dtype=bf16), layer3_bn_bias[1024](dtype=float32),
     *layer3_bn_weight[1024](dtype=float32), layer3_bn_running_mean[1024](dtype=float32),
     *layer3_bn_running_var[1024](dtype=float32)] output: [layer3_bn_output(64, 14, 14, 1024)(dtype=bf16),
     *layer3_bn_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_bn_saved_var(1, 1, 1, 1024)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_bn_kernel_params;
    layer3_bn_kernel_params.momentum    = 0.1;
    layer3_bn_kernel_params.threshold.f = 1e-05;
    layer3_bn_kernel_params.epsilon     = 1e-05;

    // create layer3_bn_bias tensor
    const unsigned layer3_bn_bias_sizes[] = {
        1024,
    };
    uint64_t            layer3_bn_bias_dram    = 0;
    synLaunchTensorInfo layer3_bn_bias_tr_info = {"layer3_bn_bias", layer3_bn_bias_dram};
    synTensor layer3_bn_bias = createTensor(1U, syn_type_single, layer3_bn_bias_sizes, true, "layer3_bn_bias");

    // create layer3_bn_weight tensor
    const unsigned layer3_bn_weight_sizes[] = {
        1024,
    };
    uint64_t            layer3_bn_weight_dram    = 0;
    synLaunchTensorInfo layer3_bn_weight_tr_info = {"layer3_bn_weight", layer3_bn_weight_dram};
    synTensor layer3_bn_weight = createTensor(1U, syn_type_single, layer3_bn_weight_sizes, true, "layer3_bn_weight");

    // create layer3_bn_running_mean tensor
    const unsigned layer3_bn_running_mean_sizes[] = {
        1024,
    };
    uint64_t            layer3_bn_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_bn_running_mean_tr_info = {"layer3_bn_running_mean", layer3_bn_running_mean_dram};
    synTensor           layer3_bn_running_mean =
        createTensor(1U, syn_type_single, layer3_bn_running_mean_sizes, true, "layer3_bn_running_mean");

    // create layer3_bn_running_var tensor
    const unsigned layer3_bn_running_var_sizes[] = {
        1024,
    };
    uint64_t            layer3_bn_running_var_dram    = 0;
    synLaunchTensorInfo layer3_bn_running_var_tr_info = {"layer3_bn_running_var", layer3_bn_running_var_dram};
    synTensor           layer3_bn_running_var =
        createTensor(1U, syn_type_single, layer3_bn_running_var_sizes, true, "layer3_bn_running_var");

    synTensor layer3_bn_in_vec[5] = {layer3_downsample_output,
                                     layer3_bn_bias,
                                     layer3_bn_weight,
                                     layer3_bn_running_mean,
                                     layer3_bn_running_var};

    // create layer3_bn_output tensor
    const unsigned layer3_bn_output_sizes[] = {64, 14, 14, 1024};
    synTensor layer3_bn_output = createTensor(4U, syn_type_bf16, layer3_bn_output_sizes, false, "layer3_bn_output");

    // create layer3_bn_saved_mean tensor
    const unsigned layer3_bn_saved_mean_sizes[] = {1024};
    synTensor      layer3_bn_saved_mean =
        createTensor(1U, syn_type_single, layer3_bn_saved_mean_sizes, false, "layer3_bn_saved_mean");

    // create layer3_bn_saved_var tensor
    const unsigned layer3_bn_saved_var_sizes[] = {1024};
    synTensor      layer3_bn_saved_var =
        createTensor(1U, syn_type_single, layer3_bn_saved_var_sizes, false, "layer3_bn_saved_var");

    synTensor layer3_bn_out_vec[3] = {layer3_bn_output, layer3_bn_saved_mean, layer3_bn_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_bn_in_vec,
                           layer3_bn_out_vec,
                           5,
                           3,
                           (void*)&layer3_bn_kernel_params,
                           sizeof(layer3_bn_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_bn",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_bn failed!");

    /*************
     * layer3_0_add_residual_fwd0 node
     * inputs: [layer3_0_bn3_output(64, 14, 14, 1024)(dtype=bf16), layer3_bn_output(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_0_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_0_add_residual_fwd0_in_vec[2] = {layer3_0_bn3_output, layer3_bn_output};

    // create layer3_0_add_residual_fwd tensor
    const unsigned layer3_0_add_residual_fwd_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer3_0_add_residual_fwd_sizes, false, "layer3_0_add_residual_fwd");

    synTensor layer3_0_add_residual_fwd0_out_vec[1] = {layer3_0_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer3_0_add_residual_fwd0_in_vec,
                           layer3_0_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_0_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_add_residual_fwd0 failed!");

    /*************
     * layer3_0_relu3 node
     * inputs: [layer3_0_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_0_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_0_relu3_in_vec[1] = {layer3_0_add_residual_fwd};

    // create layer3_0_relu3_output tensor
    const unsigned layer3_0_relu3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_relu3_output =
        createTensor(4U, syn_type_bf16, layer3_0_relu3_output_sizes, false, "layer3_0_relu3_output");

    synTensor layer3_0_relu3_out_vec[1] = {layer3_0_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer3_0_relu3_in_vec,
                           layer3_0_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_0_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_relu3 failed!");

    /*************
     * layer3_1_conv1 node
     * inputs: [layer3_0_relu3_output(64, 14, 14, 1024)(dtype=bf16), layer3_1_conv1_weight[1, 1, 1024, 256](dtype=bf16)]
     * output: [layer3_1_conv1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_1_conv1_kernel_params;
    layer3_1_conv1_kernel_params.dH   = 1;
    layer3_1_conv1_kernel_params.dW   = 1;
    layer3_1_conv1_kernel_params.kH   = 1;
    layer3_1_conv1_kernel_params.kW   = 1;
    layer3_1_conv1_kernel_params.padT = 0;
    layer3_1_conv1_kernel_params.padB = 0;
    layer3_1_conv1_kernel_params.padL = 0;
    layer3_1_conv1_kernel_params.padR = 0;
    layer3_1_conv1_kernel_params.dilH = 1;
    layer3_1_conv1_kernel_params.dilW = 1;

    // create layer3_1_conv1_weight tensor
    const unsigned      layer3_1_conv1_weight_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_1_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer3_1_conv1_weight_tr_info = {"layer3_1_conv1_weight", layer3_1_conv1_weight_dram};
    synTensor           layer3_1_conv1_weight =
        createTensor(4U, syn_type_bf16, layer3_1_conv1_weight_sizes, true, "layer3_1_conv1_weight");

    synTensor layer3_1_conv1_in_vec[4] = {layer3_0_relu3_output, layer3_1_conv1_weight, nullptr, nullptr};

    // create layer3_1_conv1_output tensor
    const unsigned layer3_1_conv1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_conv1_output =
        createTensor(4U, syn_type_bf16, layer3_1_conv1_output_sizes, false, "layer3_1_conv1_output");

    synTensor layer3_1_conv1_out_vec[1] = {layer3_1_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv1_in_vec,
                           layer3_1_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer3_1_conv1_kernel_params,
                           sizeof(layer3_1_conv1_kernel_params),
                           "spatial_convolution",
                           "layer3_1_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv1 failed!");

    /*************
     * layer3_1_bn1 node
     * inputs: [layer3_1_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_1_bn1_bias[256](dtype=float32),
     *layer3_1_bn1_weight[256](dtype=float32), layer3_1_bn1_running_mean[256](dtype=float32),
     *layer3_1_bn1_running_var[256](dtype=float32)] output: [layer3_1_bn1_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_1_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_1_bn1_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_1_bn1_kernel_params;
    layer3_1_bn1_kernel_params.momentum    = 0.1;
    layer3_1_bn1_kernel_params.threshold.f = 1e-05;
    layer3_1_bn1_kernel_params.epsilon     = 1e-05;

    // create layer3_1_bn1_bias tensor
    const unsigned layer3_1_bn1_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer3_1_bn1_bias_tr_info = {"layer3_1_bn1_bias", layer3_1_bn1_bias_dram};
    synTensor layer3_1_bn1_bias = createTensor(1U, syn_type_single, layer3_1_bn1_bias_sizes, true, "layer3_1_bn1_bias");

    // create layer3_1_bn1_weight tensor
    const unsigned layer3_1_bn1_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer3_1_bn1_weight_tr_info = {"layer3_1_bn1_weight", layer3_1_bn1_weight_dram};
    synTensor           layer3_1_bn1_weight =
        createTensor(1U, syn_type_single, layer3_1_bn1_weight_sizes, true, "layer3_1_bn1_weight");

    // create layer3_1_bn1_running_mean tensor
    const unsigned layer3_1_bn1_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_1_bn1_running_mean_tr_info = {"layer3_1_bn1_running_mean",
                                                             layer3_1_bn1_running_mean_dram};
    synTensor           layer3_1_bn1_running_mean =
        createTensor(1U, syn_type_single, layer3_1_bn1_running_mean_sizes, true, "layer3_1_bn1_running_mean");

    // create layer3_1_bn1_running_var tensor
    const unsigned layer3_1_bn1_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer3_1_bn1_running_var_tr_info = {"layer3_1_bn1_running_var", layer3_1_bn1_running_var_dram};
    synTensor           layer3_1_bn1_running_var =
        createTensor(1U, syn_type_single, layer3_1_bn1_running_var_sizes, true, "layer3_1_bn1_running_var");

    synTensor layer3_1_bn1_in_vec[5] = {layer3_1_conv1_output,
                                        layer3_1_bn1_bias,
                                        layer3_1_bn1_weight,
                                        layer3_1_bn1_running_mean,
                                        layer3_1_bn1_running_var};

    // create layer3_1_bn1_output tensor
    const unsigned layer3_1_bn1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_bn1_output =
        createTensor(4U, syn_type_bf16, layer3_1_bn1_output_sizes, false, "layer3_1_bn1_output");

    // create layer3_1_bn1_saved_mean tensor
    const unsigned layer3_1_bn1_saved_mean_sizes[] = {256};
    synTensor      layer3_1_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer3_1_bn1_saved_mean_sizes, false, "layer3_1_bn1_saved_mean");

    // create layer3_1_bn1_saved_var tensor
    const unsigned layer3_1_bn1_saved_var_sizes[] = {256};
    synTensor      layer3_1_bn1_saved_var =
        createTensor(1U, syn_type_single, layer3_1_bn1_saved_var_sizes, false, "layer3_1_bn1_saved_var");

    synTensor layer3_1_bn1_out_vec[3] = {layer3_1_bn1_output, layer3_1_bn1_saved_mean, layer3_1_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_1_bn1_in_vec,
                           layer3_1_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer3_1_bn1_kernel_params,
                           sizeof(layer3_1_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_1_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_bn1 failed!");

    /*************
     * layer3_1_relu1 node
     * inputs: [layer3_1_bn1_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_1_relu1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_1_relu1_in_vec[1] = {layer3_1_bn1_output};

    // create layer3_1_relu1_output tensor
    const unsigned layer3_1_relu1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_relu1_output =
        createTensor(4U, syn_type_bf16, layer3_1_relu1_output_sizes, false, "layer3_1_relu1_output");

    synTensor layer3_1_relu1_out_vec[1] = {layer3_1_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer3_1_relu1_in_vec,
                           layer3_1_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_1_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_relu1 failed!");

    /*************
     * layer3_1_conv2 node
     * inputs: [layer3_1_relu1_output(64, 14, 14, 256)(dtype=bf16), layer3_1_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_1_conv2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_1_conv2_kernel_params;
    layer3_1_conv2_kernel_params.dH   = 1;
    layer3_1_conv2_kernel_params.dW   = 1;
    layer3_1_conv2_kernel_params.kH   = 3;
    layer3_1_conv2_kernel_params.kW   = 3;
    layer3_1_conv2_kernel_params.padT = 1;
    layer3_1_conv2_kernel_params.padB = 1;
    layer3_1_conv2_kernel_params.padL = 1;
    layer3_1_conv2_kernel_params.padR = 1;
    layer3_1_conv2_kernel_params.dilH = 1;
    layer3_1_conv2_kernel_params.dilW = 1;

    // create layer3_1_conv2_weight tensor
    const unsigned      layer3_1_conv2_weight_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_1_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer3_1_conv2_weight_tr_info = {"layer3_1_conv2_weight", layer3_1_conv2_weight_dram};
    synTensor           layer3_1_conv2_weight =
        createTensor(4U, syn_type_bf16, layer3_1_conv2_weight_sizes, true, "layer3_1_conv2_weight");

    synTensor layer3_1_conv2_in_vec[4] = {layer3_1_relu1_output, layer3_1_conv2_weight, nullptr, nullptr};

    // create layer3_1_conv2_output tensor
    const unsigned layer3_1_conv2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_conv2_output =
        createTensor(4U, syn_type_bf16, layer3_1_conv2_output_sizes, false, "layer3_1_conv2_output");

    synTensor layer3_1_conv2_out_vec[1] = {layer3_1_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv2_in_vec,
                           layer3_1_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer3_1_conv2_kernel_params,
                           sizeof(layer3_1_conv2_kernel_params),
                           "spatial_convolution",
                           "layer3_1_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv2 failed!");

    /*************
     * layer3_1_bn2 node
     * inputs: [layer3_1_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_1_bn2_bias[256](dtype=float32),
     *layer3_1_bn2_weight[256](dtype=float32), layer3_1_bn2_running_mean[256](dtype=float32),
     *layer3_1_bn2_running_var[256](dtype=float32)] output: [layer3_1_bn2_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_1_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_1_bn2_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_1_bn2_kernel_params;
    layer3_1_bn2_kernel_params.momentum    = 0.1;
    layer3_1_bn2_kernel_params.threshold.f = 1e-05;
    layer3_1_bn2_kernel_params.epsilon     = 1e-05;

    // create layer3_1_bn2_bias tensor
    const unsigned layer3_1_bn2_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer3_1_bn2_bias_tr_info = {"layer3_1_bn2_bias", layer3_1_bn2_bias_dram};
    synTensor layer3_1_bn2_bias = createTensor(1U, syn_type_single, layer3_1_bn2_bias_sizes, true, "layer3_1_bn2_bias");

    // create layer3_1_bn2_weight tensor
    const unsigned layer3_1_bn2_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer3_1_bn2_weight_tr_info = {"layer3_1_bn2_weight", layer3_1_bn2_weight_dram};
    synTensor           layer3_1_bn2_weight =
        createTensor(1U, syn_type_single, layer3_1_bn2_weight_sizes, true, "layer3_1_bn2_weight");

    // create layer3_1_bn2_running_mean tensor
    const unsigned layer3_1_bn2_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_1_bn2_running_mean_tr_info = {"layer3_1_bn2_running_mean",
                                                             layer3_1_bn2_running_mean_dram};
    synTensor           layer3_1_bn2_running_mean =
        createTensor(1U, syn_type_single, layer3_1_bn2_running_mean_sizes, true, "layer3_1_bn2_running_mean");

    // create layer3_1_bn2_running_var tensor
    const unsigned layer3_1_bn2_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer3_1_bn2_running_var_tr_info = {"layer3_1_bn2_running_var", layer3_1_bn2_running_var_dram};
    synTensor           layer3_1_bn2_running_var =
        createTensor(1U, syn_type_single, layer3_1_bn2_running_var_sizes, true, "layer3_1_bn2_running_var");

    synTensor layer3_1_bn2_in_vec[5] = {layer3_1_conv2_output,
                                        layer3_1_bn2_bias,
                                        layer3_1_bn2_weight,
                                        layer3_1_bn2_running_mean,
                                        layer3_1_bn2_running_var};

    // create layer3_1_bn2_output tensor
    const unsigned layer3_1_bn2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_bn2_output =
        createTensor(4U, syn_type_bf16, layer3_1_bn2_output_sizes, false, "layer3_1_bn2_output");

    // create layer3_1_bn2_saved_mean tensor
    const unsigned layer3_1_bn2_saved_mean_sizes[] = {256};
    synTensor      layer3_1_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer3_1_bn2_saved_mean_sizes, false, "layer3_1_bn2_saved_mean");

    // create layer3_1_bn2_saved_var tensor
    const unsigned layer3_1_bn2_saved_var_sizes[] = {256};
    synTensor      layer3_1_bn2_saved_var =
        createTensor(1U, syn_type_single, layer3_1_bn2_saved_var_sizes, false, "layer3_1_bn2_saved_var");

    synTensor layer3_1_bn2_out_vec[3] = {layer3_1_bn2_output, layer3_1_bn2_saved_mean, layer3_1_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_1_bn2_in_vec,
                           layer3_1_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer3_1_bn2_kernel_params,
                           sizeof(layer3_1_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_1_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_bn2 failed!");

    /*************
     * layer3_1_relu2 node
     * inputs: [layer3_1_bn2_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_1_relu2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_1_relu2_in_vec[1] = {layer3_1_bn2_output};

    // create layer3_1_relu2_output tensor
    const unsigned layer3_1_relu2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_relu2_output =
        createTensor(4U, syn_type_bf16, layer3_1_relu2_output_sizes, false, "layer3_1_relu2_output");

    synTensor layer3_1_relu2_out_vec[1] = {layer3_1_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer3_1_relu2_in_vec,
                           layer3_1_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_1_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_relu2 failed!");

    /*************
     * layer3_1_conv3 node
     * inputs: [layer3_1_relu2_output(64, 14, 14, 256)(dtype=bf16), layer3_1_conv3_weight[1, 1, 256, 1024](dtype=bf16)]
     * output: [layer3_1_conv3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_1_conv3_kernel_params;
    layer3_1_conv3_kernel_params.dH   = 1;
    layer3_1_conv3_kernel_params.dW   = 1;
    layer3_1_conv3_kernel_params.kH   = 1;
    layer3_1_conv3_kernel_params.kW   = 1;
    layer3_1_conv3_kernel_params.padT = 0;
    layer3_1_conv3_kernel_params.padB = 0;
    layer3_1_conv3_kernel_params.padL = 0;
    layer3_1_conv3_kernel_params.padR = 0;
    layer3_1_conv3_kernel_params.dilH = 1;
    layer3_1_conv3_kernel_params.dilW = 1;

    // create layer3_1_conv3_weight tensor
    const unsigned      layer3_1_conv3_weight_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_1_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer3_1_conv3_weight_tr_info = {"layer3_1_conv3_weight", layer3_1_conv3_weight_dram};
    synTensor           layer3_1_conv3_weight =
        createTensor(4U, syn_type_bf16, layer3_1_conv3_weight_sizes, true, "layer3_1_conv3_weight");

    synTensor layer3_1_conv3_in_vec[4] = {layer3_1_relu2_output, layer3_1_conv3_weight, nullptr, nullptr};

    // create layer3_1_conv3_output tensor
    const unsigned layer3_1_conv3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_conv3_output =
        createTensor(4U, syn_type_bf16, layer3_1_conv3_output_sizes, false, "layer3_1_conv3_output");

    synTensor layer3_1_conv3_out_vec[1] = {layer3_1_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv3_in_vec,
                           layer3_1_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer3_1_conv3_kernel_params,
                           sizeof(layer3_1_conv3_kernel_params),
                           "spatial_convolution",
                           "layer3_1_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv3 failed!");

    /*************
     * layer3_1_bn3 node
     * inputs: [layer3_1_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_1_bn3_bias[1024](dtype=float32),
     *layer3_1_bn3_weight[1024](dtype=float32), layer3_1_bn3_running_mean[1024](dtype=float32),
     *layer3_1_bn3_running_var[1024](dtype=float32)] output: [layer3_1_bn3_output(64, 14, 14, 1024)(dtype=bf16),
     *layer3_1_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_1_bn3_saved_var(1, 1, 1, 1024)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_1_bn3_kernel_params;
    layer3_1_bn3_kernel_params.momentum    = 0.1;
    layer3_1_bn3_kernel_params.threshold.f = 1e-05;
    layer3_1_bn3_kernel_params.epsilon     = 1e-05;

    // create layer3_1_bn3_bias tensor
    const unsigned layer3_1_bn3_bias_sizes[] = {
        1024,
    };
    uint64_t            layer3_1_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer3_1_bn3_bias_tr_info = {"layer3_1_bn3_bias", layer3_1_bn3_bias_dram};
    synTensor layer3_1_bn3_bias = createTensor(1U, syn_type_single, layer3_1_bn3_bias_sizes, true, "layer3_1_bn3_bias");

    // create layer3_1_bn3_weight tensor
    const unsigned layer3_1_bn3_weight_sizes[] = {
        1024,
    };
    uint64_t            layer3_1_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer3_1_bn3_weight_tr_info = {"layer3_1_bn3_weight", layer3_1_bn3_weight_dram};
    synTensor           layer3_1_bn3_weight =
        createTensor(1U, syn_type_single, layer3_1_bn3_weight_sizes, true, "layer3_1_bn3_weight");

    // create layer3_1_bn3_running_mean tensor
    const unsigned layer3_1_bn3_running_mean_sizes[] = {
        1024,
    };
    uint64_t            layer3_1_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_1_bn3_running_mean_tr_info = {"layer3_1_bn3_running_mean",
                                                             layer3_1_bn3_running_mean_dram};
    synTensor           layer3_1_bn3_running_mean =
        createTensor(1U, syn_type_single, layer3_1_bn3_running_mean_sizes, true, "layer3_1_bn3_running_mean");

    // create layer3_1_bn3_running_var tensor
    const unsigned layer3_1_bn3_running_var_sizes[] = {
        1024,
    };
    uint64_t            layer3_1_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer3_1_bn3_running_var_tr_info = {"layer3_1_bn3_running_var", layer3_1_bn3_running_var_dram};
    synTensor           layer3_1_bn3_running_var =
        createTensor(1U, syn_type_single, layer3_1_bn3_running_var_sizes, true, "layer3_1_bn3_running_var");

    synTensor layer3_1_bn3_in_vec[5] = {layer3_1_conv3_output,
                                        layer3_1_bn3_bias,
                                        layer3_1_bn3_weight,
                                        layer3_1_bn3_running_mean,
                                        layer3_1_bn3_running_var};

    // create layer3_1_bn3_output tensor
    const unsigned layer3_1_bn3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_bn3_output =
        createTensor(4U, syn_type_bf16, layer3_1_bn3_output_sizes, false, "layer3_1_bn3_output");

    // create layer3_1_bn3_saved_mean tensor
    const unsigned layer3_1_bn3_saved_mean_sizes[] = {1024};
    synTensor      layer3_1_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer3_1_bn3_saved_mean_sizes, false, "layer3_1_bn3_saved_mean");

    // create layer3_1_bn3_saved_var tensor
    const unsigned layer3_1_bn3_saved_var_sizes[] = {1024};
    synTensor      layer3_1_bn3_saved_var =
        createTensor(1U, syn_type_single, layer3_1_bn3_saved_var_sizes, false, "layer3_1_bn3_saved_var");

    synTensor layer3_1_bn3_out_vec[3] = {layer3_1_bn3_output, layer3_1_bn3_saved_mean, layer3_1_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_1_bn3_in_vec,
                           layer3_1_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer3_1_bn3_kernel_params,
                           sizeof(layer3_1_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_1_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_bn3 failed!");

    /*************
     * layer3_1_add_residual_fwd0 node
     * inputs: [layer3_1_bn3_output(64, 14, 14, 1024)(dtype=bf16), layer3_0_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_1_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_1_add_residual_fwd0_in_vec[2] = {layer3_1_bn3_output, layer3_0_relu3_output};

    // create layer3_1_add_residual_fwd tensor
    const unsigned layer3_1_add_residual_fwd_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer3_1_add_residual_fwd_sizes, false, "layer3_1_add_residual_fwd");

    synTensor layer3_1_add_residual_fwd0_out_vec[1] = {layer3_1_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer3_1_add_residual_fwd0_in_vec,
                           layer3_1_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_1_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_add_residual_fwd0 failed!");

    /*************
     * layer3_1_relu3 node
     * inputs: [layer3_1_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_1_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_1_relu3_in_vec[1] = {layer3_1_add_residual_fwd};

    // create layer3_1_relu3_output tensor
    const unsigned layer3_1_relu3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_relu3_output =
        createTensor(4U, syn_type_bf16, layer3_1_relu3_output_sizes, false, "layer3_1_relu3_output");

    synTensor layer3_1_relu3_out_vec[1] = {layer3_1_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer3_1_relu3_in_vec,
                           layer3_1_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_1_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_relu3 failed!");

    /*************
     * layer3_2_conv1 node
     * inputs: [layer3_1_relu3_output(64, 14, 14, 1024)(dtype=bf16), layer3_2_conv1_weight[1, 1, 1024, 256](dtype=bf16)]
     * output: [layer3_2_conv1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_2_conv1_kernel_params;
    layer3_2_conv1_kernel_params.dH   = 1;
    layer3_2_conv1_kernel_params.dW   = 1;
    layer3_2_conv1_kernel_params.kH   = 1;
    layer3_2_conv1_kernel_params.kW   = 1;
    layer3_2_conv1_kernel_params.padT = 0;
    layer3_2_conv1_kernel_params.padB = 0;
    layer3_2_conv1_kernel_params.padL = 0;
    layer3_2_conv1_kernel_params.padR = 0;
    layer3_2_conv1_kernel_params.dilH = 1;
    layer3_2_conv1_kernel_params.dilW = 1;

    // create layer3_2_conv1_weight tensor
    const unsigned      layer3_2_conv1_weight_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_2_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer3_2_conv1_weight_tr_info = {"layer3_2_conv1_weight", layer3_2_conv1_weight_dram};
    synTensor           layer3_2_conv1_weight =
        createTensor(4U, syn_type_bf16, layer3_2_conv1_weight_sizes, true, "layer3_2_conv1_weight");

    synTensor layer3_2_conv1_in_vec[4] = {layer3_1_relu3_output, layer3_2_conv1_weight, nullptr, nullptr};

    // create layer3_2_conv1_output tensor
    const unsigned layer3_2_conv1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_conv1_output =
        createTensor(4U, syn_type_bf16, layer3_2_conv1_output_sizes, false, "layer3_2_conv1_output");

    synTensor layer3_2_conv1_out_vec[1] = {layer3_2_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv1_in_vec,
                           layer3_2_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer3_2_conv1_kernel_params,
                           sizeof(layer3_2_conv1_kernel_params),
                           "spatial_convolution",
                           "layer3_2_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv1 failed!");

    /*************
     * layer3_2_bn1 node
     * inputs: [layer3_2_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_2_bn1_bias[256](dtype=float32),
     *layer3_2_bn1_weight[256](dtype=float32), layer3_2_bn1_running_mean[256](dtype=float32),
     *layer3_2_bn1_running_var[256](dtype=float32)] output: [layer3_2_bn1_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_2_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_2_bn1_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_2_bn1_kernel_params;
    layer3_2_bn1_kernel_params.momentum    = 0.1;
    layer3_2_bn1_kernel_params.threshold.f = 1e-05;
    layer3_2_bn1_kernel_params.epsilon     = 1e-05;

    // create layer3_2_bn1_bias tensor
    const unsigned layer3_2_bn1_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer3_2_bn1_bias_tr_info = {"layer3_2_bn1_bias", layer3_2_bn1_bias_dram};
    synTensor layer3_2_bn1_bias = createTensor(1U, syn_type_single, layer3_2_bn1_bias_sizes, true, "layer3_2_bn1_bias");

    // create layer3_2_bn1_weight tensor
    const unsigned layer3_2_bn1_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer3_2_bn1_weight_tr_info = {"layer3_2_bn1_weight", layer3_2_bn1_weight_dram};
    synTensor           layer3_2_bn1_weight =
        createTensor(1U, syn_type_single, layer3_2_bn1_weight_sizes, true, "layer3_2_bn1_weight");

    // create layer3_2_bn1_running_mean tensor
    const unsigned layer3_2_bn1_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_2_bn1_running_mean_tr_info = {"layer3_2_bn1_running_mean",
                                                             layer3_2_bn1_running_mean_dram};
    synTensor           layer3_2_bn1_running_mean =
        createTensor(1U, syn_type_single, layer3_2_bn1_running_mean_sizes, true, "layer3_2_bn1_running_mean");

    // create layer3_2_bn1_running_var tensor
    const unsigned layer3_2_bn1_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer3_2_bn1_running_var_tr_info = {"layer3_2_bn1_running_var", layer3_2_bn1_running_var_dram};
    synTensor           layer3_2_bn1_running_var =
        createTensor(1U, syn_type_single, layer3_2_bn1_running_var_sizes, true, "layer3_2_bn1_running_var");

    synTensor layer3_2_bn1_in_vec[5] = {layer3_2_conv1_output,
                                        layer3_2_bn1_bias,
                                        layer3_2_bn1_weight,
                                        layer3_2_bn1_running_mean,
                                        layer3_2_bn1_running_var};

    // create layer3_2_bn1_output tensor
    const unsigned layer3_2_bn1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_bn1_output =
        createTensor(4U, syn_type_bf16, layer3_2_bn1_output_sizes, false, "layer3_2_bn1_output");

    // create layer3_2_bn1_saved_mean tensor
    const unsigned layer3_2_bn1_saved_mean_sizes[] = {256};
    synTensor      layer3_2_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer3_2_bn1_saved_mean_sizes, false, "layer3_2_bn1_saved_mean");

    // create layer3_2_bn1_saved_var tensor
    const unsigned layer3_2_bn1_saved_var_sizes[] = {256};
    synTensor      layer3_2_bn1_saved_var =
        createTensor(1U, syn_type_single, layer3_2_bn1_saved_var_sizes, false, "layer3_2_bn1_saved_var");

    synTensor layer3_2_bn1_out_vec[3] = {layer3_2_bn1_output, layer3_2_bn1_saved_mean, layer3_2_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_2_bn1_in_vec,
                           layer3_2_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer3_2_bn1_kernel_params,
                           sizeof(layer3_2_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_2_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_bn1 failed!");

    /*************
     * layer3_2_relu1 node
     * inputs: [layer3_2_bn1_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_2_relu1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_2_relu1_in_vec[1] = {layer3_2_bn1_output};

    // create layer3_2_relu1_output tensor
    const unsigned layer3_2_relu1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_relu1_output =
        createTensor(4U, syn_type_bf16, layer3_2_relu1_output_sizes, false, "layer3_2_relu1_output");

    synTensor layer3_2_relu1_out_vec[1] = {layer3_2_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer3_2_relu1_in_vec,
                           layer3_2_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_2_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_relu1 failed!");

    /*************
     * layer3_2_conv2 node
     * inputs: [layer3_2_relu1_output(64, 14, 14, 256)(dtype=bf16), layer3_2_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_2_conv2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_2_conv2_kernel_params;
    layer3_2_conv2_kernel_params.dH   = 1;
    layer3_2_conv2_kernel_params.dW   = 1;
    layer3_2_conv2_kernel_params.kH   = 3;
    layer3_2_conv2_kernel_params.kW   = 3;
    layer3_2_conv2_kernel_params.padT = 1;
    layer3_2_conv2_kernel_params.padB = 1;
    layer3_2_conv2_kernel_params.padL = 1;
    layer3_2_conv2_kernel_params.padR = 1;
    layer3_2_conv2_kernel_params.dilH = 1;
    layer3_2_conv2_kernel_params.dilW = 1;

    // create layer3_2_conv2_weight tensor
    const unsigned      layer3_2_conv2_weight_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_2_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer3_2_conv2_weight_tr_info = {"layer3_2_conv2_weight", layer3_2_conv2_weight_dram};
    synTensor           layer3_2_conv2_weight =
        createTensor(4U, syn_type_bf16, layer3_2_conv2_weight_sizes, true, "layer3_2_conv2_weight");

    synTensor layer3_2_conv2_in_vec[4] = {layer3_2_relu1_output, layer3_2_conv2_weight, nullptr, nullptr};

    // create layer3_2_conv2_output tensor
    const unsigned layer3_2_conv2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_conv2_output =
        createTensor(4U, syn_type_bf16, layer3_2_conv2_output_sizes, false, "layer3_2_conv2_output");

    synTensor layer3_2_conv2_out_vec[1] = {layer3_2_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv2_in_vec,
                           layer3_2_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer3_2_conv2_kernel_params,
                           sizeof(layer3_2_conv2_kernel_params),
                           "spatial_convolution",
                           "layer3_2_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv2 failed!");

    /*************
     * layer3_2_bn2 node
     * inputs: [layer3_2_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_2_bn2_bias[256](dtype=float32),
     *layer3_2_bn2_weight[256](dtype=float32), layer3_2_bn2_running_mean[256](dtype=float32),
     *layer3_2_bn2_running_var[256](dtype=float32)] output: [layer3_2_bn2_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_2_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_2_bn2_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_2_bn2_kernel_params;
    layer3_2_bn2_kernel_params.momentum    = 0.1;
    layer3_2_bn2_kernel_params.threshold.f = 1e-05;
    layer3_2_bn2_kernel_params.epsilon     = 1e-05;

    // create layer3_2_bn2_bias tensor
    const unsigned layer3_2_bn2_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer3_2_bn2_bias_tr_info = {"layer3_2_bn2_bias", layer3_2_bn2_bias_dram};
    synTensor layer3_2_bn2_bias = createTensor(1U, syn_type_single, layer3_2_bn2_bias_sizes, true, "layer3_2_bn2_bias");

    // create layer3_2_bn2_weight tensor
    const unsigned layer3_2_bn2_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer3_2_bn2_weight_tr_info = {"layer3_2_bn2_weight", layer3_2_bn2_weight_dram};
    synTensor           layer3_2_bn2_weight =
        createTensor(1U, syn_type_single, layer3_2_bn2_weight_sizes, true, "layer3_2_bn2_weight");

    // create layer3_2_bn2_running_mean tensor
    const unsigned layer3_2_bn2_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_2_bn2_running_mean_tr_info = {"layer3_2_bn2_running_mean",
                                                             layer3_2_bn2_running_mean_dram};
    synTensor           layer3_2_bn2_running_mean =
        createTensor(1U, syn_type_single, layer3_2_bn2_running_mean_sizes, true, "layer3_2_bn2_running_mean");

    // create layer3_2_bn2_running_var tensor
    const unsigned layer3_2_bn2_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer3_2_bn2_running_var_tr_info = {"layer3_2_bn2_running_var", layer3_2_bn2_running_var_dram};
    synTensor           layer3_2_bn2_running_var =
        createTensor(1U, syn_type_single, layer3_2_bn2_running_var_sizes, true, "layer3_2_bn2_running_var");

    synTensor layer3_2_bn2_in_vec[5] = {layer3_2_conv2_output,
                                        layer3_2_bn2_bias,
                                        layer3_2_bn2_weight,
                                        layer3_2_bn2_running_mean,
                                        layer3_2_bn2_running_var};

    // create layer3_2_bn2_output tensor
    const unsigned layer3_2_bn2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_bn2_output =
        createTensor(4U, syn_type_bf16, layer3_2_bn2_output_sizes, false, "layer3_2_bn2_output");

    // create layer3_2_bn2_saved_mean tensor
    const unsigned layer3_2_bn2_saved_mean_sizes[] = {256};
    synTensor      layer3_2_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer3_2_bn2_saved_mean_sizes, false, "layer3_2_bn2_saved_mean");

    // create layer3_2_bn2_saved_var tensor
    const unsigned layer3_2_bn2_saved_var_sizes[] = {256};
    synTensor      layer3_2_bn2_saved_var =
        createTensor(1U, syn_type_single, layer3_2_bn2_saved_var_sizes, false, "layer3_2_bn2_saved_var");

    synTensor layer3_2_bn2_out_vec[3] = {layer3_2_bn2_output, layer3_2_bn2_saved_mean, layer3_2_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_2_bn2_in_vec,
                           layer3_2_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer3_2_bn2_kernel_params,
                           sizeof(layer3_2_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_2_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_bn2 failed!");

    /*************
     * layer3_2_relu2 node
     * inputs: [layer3_2_bn2_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_2_relu2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_2_relu2_in_vec[1] = {layer3_2_bn2_output};

    // create layer3_2_relu2_output tensor
    const unsigned layer3_2_relu2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_relu2_output =
        createTensor(4U, syn_type_bf16, layer3_2_relu2_output_sizes, false, "layer3_2_relu2_output");

    synTensor layer3_2_relu2_out_vec[1] = {layer3_2_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer3_2_relu2_in_vec,
                           layer3_2_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_2_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_relu2 failed!");

    /*************
     * layer3_2_conv3 node
     * inputs: [layer3_2_relu2_output(64, 14, 14, 256)(dtype=bf16), layer3_2_conv3_weight[1, 1, 256, 1024](dtype=bf16)]
     * output: [layer3_2_conv3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_2_conv3_kernel_params;
    layer3_2_conv3_kernel_params.dH   = 1;
    layer3_2_conv3_kernel_params.dW   = 1;
    layer3_2_conv3_kernel_params.kH   = 1;
    layer3_2_conv3_kernel_params.kW   = 1;
    layer3_2_conv3_kernel_params.padT = 0;
    layer3_2_conv3_kernel_params.padB = 0;
    layer3_2_conv3_kernel_params.padL = 0;
    layer3_2_conv3_kernel_params.padR = 0;
    layer3_2_conv3_kernel_params.dilH = 1;
    layer3_2_conv3_kernel_params.dilW = 1;

    // create layer3_2_conv3_weight tensor
    const unsigned      layer3_2_conv3_weight_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_2_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer3_2_conv3_weight_tr_info = {"layer3_2_conv3_weight", layer3_2_conv3_weight_dram};
    synTensor           layer3_2_conv3_weight =
        createTensor(4U, syn_type_bf16, layer3_2_conv3_weight_sizes, true, "layer3_2_conv3_weight");

    synTensor layer3_2_conv3_in_vec[4] = {layer3_2_relu2_output, layer3_2_conv3_weight, nullptr, nullptr};

    // create layer3_2_conv3_output tensor
    const unsigned layer3_2_conv3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_conv3_output =
        createTensor(4U, syn_type_bf16, layer3_2_conv3_output_sizes, false, "layer3_2_conv3_output");

    synTensor layer3_2_conv3_out_vec[1] = {layer3_2_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv3_in_vec,
                           layer3_2_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer3_2_conv3_kernel_params,
                           sizeof(layer3_2_conv3_kernel_params),
                           "spatial_convolution",
                           "layer3_2_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv3 failed!");

    /*************
     * layer3_2_bn3 node
     * inputs: [layer3_2_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_2_bn3_bias[1024](dtype=float32),
     *layer3_2_bn3_weight[1024](dtype=float32), layer3_2_bn3_running_mean[1024](dtype=float32),
     *layer3_2_bn3_running_var[1024](dtype=float32)] output: [layer3_2_bn3_output(64, 14, 14, 1024)(dtype=bf16),
     *layer3_2_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_2_bn3_saved_var(1, 1, 1, 1024)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_2_bn3_kernel_params;
    layer3_2_bn3_kernel_params.momentum    = 0.1;
    layer3_2_bn3_kernel_params.threshold.f = 1e-05;
    layer3_2_bn3_kernel_params.epsilon     = 1e-05;

    // create layer3_2_bn3_bias tensor
    const unsigned layer3_2_bn3_bias_sizes[] = {
        1024,
    };
    uint64_t            layer3_2_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer3_2_bn3_bias_tr_info = {"layer3_2_bn3_bias", layer3_2_bn3_bias_dram};
    synTensor layer3_2_bn3_bias = createTensor(1U, syn_type_single, layer3_2_bn3_bias_sizes, true, "layer3_2_bn3_bias");

    // create layer3_2_bn3_weight tensor
    const unsigned layer3_2_bn3_weight_sizes[] = {
        1024,
    };
    uint64_t            layer3_2_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer3_2_bn3_weight_tr_info = {"layer3_2_bn3_weight", layer3_2_bn3_weight_dram};
    synTensor           layer3_2_bn3_weight =
        createTensor(1U, syn_type_single, layer3_2_bn3_weight_sizes, true, "layer3_2_bn3_weight");

    // create layer3_2_bn3_running_mean tensor
    const unsigned layer3_2_bn3_running_mean_sizes[] = {
        1024,
    };
    uint64_t            layer3_2_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_2_bn3_running_mean_tr_info = {"layer3_2_bn3_running_mean",
                                                             layer3_2_bn3_running_mean_dram};
    synTensor           layer3_2_bn3_running_mean =
        createTensor(1U, syn_type_single, layer3_2_bn3_running_mean_sizes, true, "layer3_2_bn3_running_mean");

    // create layer3_2_bn3_running_var tensor
    const unsigned layer3_2_bn3_running_var_sizes[] = {
        1024,
    };
    uint64_t            layer3_2_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer3_2_bn3_running_var_tr_info = {"layer3_2_bn3_running_var", layer3_2_bn3_running_var_dram};
    synTensor           layer3_2_bn3_running_var =
        createTensor(1U, syn_type_single, layer3_2_bn3_running_var_sizes, true, "layer3_2_bn3_running_var");

    synTensor layer3_2_bn3_in_vec[5] = {layer3_2_conv3_output,
                                        layer3_2_bn3_bias,
                                        layer3_2_bn3_weight,
                                        layer3_2_bn3_running_mean,
                                        layer3_2_bn3_running_var};

    // create layer3_2_bn3_output tensor
    const unsigned layer3_2_bn3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_bn3_output =
        createTensor(4U, syn_type_bf16, layer3_2_bn3_output_sizes, false, "layer3_2_bn3_output");

    // create layer3_2_bn3_saved_mean tensor
    const unsigned layer3_2_bn3_saved_mean_sizes[] = {1024};
    synTensor      layer3_2_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer3_2_bn3_saved_mean_sizes, false, "layer3_2_bn3_saved_mean");

    // create layer3_2_bn3_saved_var tensor
    const unsigned layer3_2_bn3_saved_var_sizes[] = {1024};
    synTensor      layer3_2_bn3_saved_var =
        createTensor(1U, syn_type_single, layer3_2_bn3_saved_var_sizes, false, "layer3_2_bn3_saved_var");

    synTensor layer3_2_bn3_out_vec[3] = {layer3_2_bn3_output, layer3_2_bn3_saved_mean, layer3_2_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_2_bn3_in_vec,
                           layer3_2_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer3_2_bn3_kernel_params,
                           sizeof(layer3_2_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_2_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_bn3 failed!");

    /*************
     * layer3_2_add_residual_fwd0 node
     * inputs: [layer3_2_bn3_output(64, 14, 14, 1024)(dtype=bf16), layer3_1_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_2_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_2_add_residual_fwd0_in_vec[2] = {layer3_2_bn3_output, layer3_1_relu3_output};

    // create layer3_2_add_residual_fwd tensor
    const unsigned layer3_2_add_residual_fwd_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer3_2_add_residual_fwd_sizes, false, "layer3_2_add_residual_fwd");

    synTensor layer3_2_add_residual_fwd0_out_vec[1] = {layer3_2_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer3_2_add_residual_fwd0_in_vec,
                           layer3_2_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_2_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_add_residual_fwd0 failed!");

    /*************
     * layer3_2_relu3 node
     * inputs: [layer3_2_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_2_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_2_relu3_in_vec[1] = {layer3_2_add_residual_fwd};

    // create layer3_2_relu3_output tensor
    const unsigned layer3_2_relu3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_relu3_output =
        createTensor(4U, syn_type_bf16, layer3_2_relu3_output_sizes, false, "layer3_2_relu3_output");

    synTensor layer3_2_relu3_out_vec[1] = {layer3_2_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer3_2_relu3_in_vec,
                           layer3_2_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_2_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_relu3 failed!");

    /*************
     * layer3_3_conv1 node
     * inputs: [layer3_2_relu3_output(64, 14, 14, 1024)(dtype=bf16), layer3_3_conv1_weight[1, 1, 1024, 256](dtype=bf16)]
     * output: [layer3_3_conv1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_3_conv1_kernel_params;
    layer3_3_conv1_kernel_params.dH   = 1;
    layer3_3_conv1_kernel_params.dW   = 1;
    layer3_3_conv1_kernel_params.kH   = 1;
    layer3_3_conv1_kernel_params.kW   = 1;
    layer3_3_conv1_kernel_params.padT = 0;
    layer3_3_conv1_kernel_params.padB = 0;
    layer3_3_conv1_kernel_params.padL = 0;
    layer3_3_conv1_kernel_params.padR = 0;
    layer3_3_conv1_kernel_params.dilH = 1;
    layer3_3_conv1_kernel_params.dilW = 1;

    // create layer3_3_conv1_weight tensor
    const unsigned      layer3_3_conv1_weight_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_3_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer3_3_conv1_weight_tr_info = {"layer3_3_conv1_weight", layer3_3_conv1_weight_dram};
    synTensor           layer3_3_conv1_weight =
        createTensor(4U, syn_type_bf16, layer3_3_conv1_weight_sizes, true, "layer3_3_conv1_weight");

    synTensor layer3_3_conv1_in_vec[4] = {layer3_2_relu3_output, layer3_3_conv1_weight, nullptr, nullptr};

    // create layer3_3_conv1_output tensor
    const unsigned layer3_3_conv1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_conv1_output =
        createTensor(4U, syn_type_bf16, layer3_3_conv1_output_sizes, false, "layer3_3_conv1_output");

    synTensor layer3_3_conv1_out_vec[1] = {layer3_3_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv1_in_vec,
                           layer3_3_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer3_3_conv1_kernel_params,
                           sizeof(layer3_3_conv1_kernel_params),
                           "spatial_convolution",
                           "layer3_3_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv1 failed!");

    /*************
     * layer3_3_bn1 node
     * inputs: [layer3_3_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_3_bn1_bias[256](dtype=float32),
     *layer3_3_bn1_weight[256](dtype=float32), layer3_3_bn1_running_mean[256](dtype=float32),
     *layer3_3_bn1_running_var[256](dtype=float32)] output: [layer3_3_bn1_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_3_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_3_bn1_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_3_bn1_kernel_params;
    layer3_3_bn1_kernel_params.momentum    = 0.1;
    layer3_3_bn1_kernel_params.threshold.f = 1e-05;
    layer3_3_bn1_kernel_params.epsilon     = 1e-05;

    // create layer3_3_bn1_bias tensor
    const unsigned layer3_3_bn1_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer3_3_bn1_bias_tr_info = {"layer3_3_bn1_bias", layer3_3_bn1_bias_dram};
    synTensor layer3_3_bn1_bias = createTensor(1U, syn_type_single, layer3_3_bn1_bias_sizes, true, "layer3_3_bn1_bias");

    // create layer3_3_bn1_weight tensor
    const unsigned layer3_3_bn1_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer3_3_bn1_weight_tr_info = {"layer3_3_bn1_weight", layer3_3_bn1_weight_dram};
    synTensor           layer3_3_bn1_weight =
        createTensor(1U, syn_type_single, layer3_3_bn1_weight_sizes, true, "layer3_3_bn1_weight");

    // create layer3_3_bn1_running_mean tensor
    const unsigned layer3_3_bn1_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_3_bn1_running_mean_tr_info = {"layer3_3_bn1_running_mean",
                                                             layer3_3_bn1_running_mean_dram};
    synTensor           layer3_3_bn1_running_mean =
        createTensor(1U, syn_type_single, layer3_3_bn1_running_mean_sizes, true, "layer3_3_bn1_running_mean");

    // create layer3_3_bn1_running_var tensor
    const unsigned layer3_3_bn1_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer3_3_bn1_running_var_tr_info = {"layer3_3_bn1_running_var", layer3_3_bn1_running_var_dram};
    synTensor           layer3_3_bn1_running_var =
        createTensor(1U, syn_type_single, layer3_3_bn1_running_var_sizes, true, "layer3_3_bn1_running_var");

    synTensor layer3_3_bn1_in_vec[5] = {layer3_3_conv1_output,
                                        layer3_3_bn1_bias,
                                        layer3_3_bn1_weight,
                                        layer3_3_bn1_running_mean,
                                        layer3_3_bn1_running_var};

    // create layer3_3_bn1_output tensor
    const unsigned layer3_3_bn1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_bn1_output =
        createTensor(4U, syn_type_bf16, layer3_3_bn1_output_sizes, false, "layer3_3_bn1_output");

    // create layer3_3_bn1_saved_mean tensor
    const unsigned layer3_3_bn1_saved_mean_sizes[] = {256};
    synTensor      layer3_3_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer3_3_bn1_saved_mean_sizes, false, "layer3_3_bn1_saved_mean");

    // create layer3_3_bn1_saved_var tensor
    const unsigned layer3_3_bn1_saved_var_sizes[] = {256};
    synTensor      layer3_3_bn1_saved_var =
        createTensor(1U, syn_type_single, layer3_3_bn1_saved_var_sizes, false, "layer3_3_bn1_saved_var");

    synTensor layer3_3_bn1_out_vec[3] = {layer3_3_bn1_output, layer3_3_bn1_saved_mean, layer3_3_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_3_bn1_in_vec,
                           layer3_3_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer3_3_bn1_kernel_params,
                           sizeof(layer3_3_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_3_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_bn1 failed!");

    /*************
     * layer3_3_relu1 node
     * inputs: [layer3_3_bn1_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_3_relu1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_3_relu1_in_vec[1] = {layer3_3_bn1_output};

    // create layer3_3_relu1_output tensor
    const unsigned layer3_3_relu1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_relu1_output =
        createTensor(4U, syn_type_bf16, layer3_3_relu1_output_sizes, false, "layer3_3_relu1_output");

    synTensor layer3_3_relu1_out_vec[1] = {layer3_3_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer3_3_relu1_in_vec,
                           layer3_3_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_3_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_relu1 failed!");

    /*************
     * layer3_3_conv2 node
     * inputs: [layer3_3_relu1_output(64, 14, 14, 256)(dtype=bf16), layer3_3_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_3_conv2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_3_conv2_kernel_params;
    layer3_3_conv2_kernel_params.dH   = 1;
    layer3_3_conv2_kernel_params.dW   = 1;
    layer3_3_conv2_kernel_params.kH   = 3;
    layer3_3_conv2_kernel_params.kW   = 3;
    layer3_3_conv2_kernel_params.padT = 1;
    layer3_3_conv2_kernel_params.padB = 1;
    layer3_3_conv2_kernel_params.padL = 1;
    layer3_3_conv2_kernel_params.padR = 1;
    layer3_3_conv2_kernel_params.dilH = 1;
    layer3_3_conv2_kernel_params.dilW = 1;

    // create layer3_3_conv2_weight tensor
    const unsigned      layer3_3_conv2_weight_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_3_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer3_3_conv2_weight_tr_info = {"layer3_3_conv2_weight", layer3_3_conv2_weight_dram};
    synTensor           layer3_3_conv2_weight =
        createTensor(4U, syn_type_bf16, layer3_3_conv2_weight_sizes, true, "layer3_3_conv2_weight");

    synTensor layer3_3_conv2_in_vec[4] = {layer3_3_relu1_output, layer3_3_conv2_weight, nullptr, nullptr};

    // create layer3_3_conv2_output tensor
    const unsigned layer3_3_conv2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_conv2_output =
        createTensor(4U, syn_type_bf16, layer3_3_conv2_output_sizes, false, "layer3_3_conv2_output");

    synTensor layer3_3_conv2_out_vec[1] = {layer3_3_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv2_in_vec,
                           layer3_3_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer3_3_conv2_kernel_params,
                           sizeof(layer3_3_conv2_kernel_params),
                           "spatial_convolution",
                           "layer3_3_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv2 failed!");

    /*************
     * layer3_3_bn2 node
     * inputs: [layer3_3_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_3_bn2_bias[256](dtype=float32),
     *layer3_3_bn2_weight[256](dtype=float32), layer3_3_bn2_running_mean[256](dtype=float32),
     *layer3_3_bn2_running_var[256](dtype=float32)] output: [layer3_3_bn2_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_3_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_3_bn2_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_3_bn2_kernel_params;
    layer3_3_bn2_kernel_params.momentum    = 0.1;
    layer3_3_bn2_kernel_params.threshold.f = 1e-05;
    layer3_3_bn2_kernel_params.epsilon     = 1e-05;

    // create layer3_3_bn2_bias tensor
    const unsigned layer3_3_bn2_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer3_3_bn2_bias_tr_info = {"layer3_3_bn2_bias", layer3_3_bn2_bias_dram};
    synTensor layer3_3_bn2_bias = createTensor(1U, syn_type_single, layer3_3_bn2_bias_sizes, true, "layer3_3_bn2_bias");

    // create layer3_3_bn2_weight tensor
    const unsigned layer3_3_bn2_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer3_3_bn2_weight_tr_info = {"layer3_3_bn2_weight", layer3_3_bn2_weight_dram};
    synTensor           layer3_3_bn2_weight =
        createTensor(1U, syn_type_single, layer3_3_bn2_weight_sizes, true, "layer3_3_bn2_weight");

    // create layer3_3_bn2_running_mean tensor
    const unsigned layer3_3_bn2_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_3_bn2_running_mean_tr_info = {"layer3_3_bn2_running_mean",
                                                             layer3_3_bn2_running_mean_dram};
    synTensor           layer3_3_bn2_running_mean =
        createTensor(1U, syn_type_single, layer3_3_bn2_running_mean_sizes, true, "layer3_3_bn2_running_mean");

    // create layer3_3_bn2_running_var tensor
    const unsigned layer3_3_bn2_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer3_3_bn2_running_var_tr_info = {"layer3_3_bn2_running_var", layer3_3_bn2_running_var_dram};
    synTensor           layer3_3_bn2_running_var =
        createTensor(1U, syn_type_single, layer3_3_bn2_running_var_sizes, true, "layer3_3_bn2_running_var");

    synTensor layer3_3_bn2_in_vec[5] = {layer3_3_conv2_output,
                                        layer3_3_bn2_bias,
                                        layer3_3_bn2_weight,
                                        layer3_3_bn2_running_mean,
                                        layer3_3_bn2_running_var};

    // create layer3_3_bn2_output tensor
    const unsigned layer3_3_bn2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_bn2_output =
        createTensor(4U, syn_type_bf16, layer3_3_bn2_output_sizes, false, "layer3_3_bn2_output");

    // create layer3_3_bn2_saved_mean tensor
    const unsigned layer3_3_bn2_saved_mean_sizes[] = {256};
    synTensor      layer3_3_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer3_3_bn2_saved_mean_sizes, false, "layer3_3_bn2_saved_mean");

    // create layer3_3_bn2_saved_var tensor
    const unsigned layer3_3_bn2_saved_var_sizes[] = {256};
    synTensor      layer3_3_bn2_saved_var =
        createTensor(1U, syn_type_single, layer3_3_bn2_saved_var_sizes, false, "layer3_3_bn2_saved_var");

    synTensor layer3_3_bn2_out_vec[3] = {layer3_3_bn2_output, layer3_3_bn2_saved_mean, layer3_3_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_3_bn2_in_vec,
                           layer3_3_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer3_3_bn2_kernel_params,
                           sizeof(layer3_3_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_3_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_bn2 failed!");

    /*************
     * layer3_3_relu2 node
     * inputs: [layer3_3_bn2_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_3_relu2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_3_relu2_in_vec[1] = {layer3_3_bn2_output};

    // create layer3_3_relu2_output tensor
    const unsigned layer3_3_relu2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_relu2_output =
        createTensor(4U, syn_type_bf16, layer3_3_relu2_output_sizes, false, "layer3_3_relu2_output");

    synTensor layer3_3_relu2_out_vec[1] = {layer3_3_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer3_3_relu2_in_vec,
                           layer3_3_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_3_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_relu2 failed!");

    /*************
     * layer3_3_conv3 node
     * inputs: [layer3_3_relu2_output(64, 14, 14, 256)(dtype=bf16), layer3_3_conv3_weight[1, 1, 256, 1024](dtype=bf16)]
     * output: [layer3_3_conv3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_3_conv3_kernel_params;
    layer3_3_conv3_kernel_params.dH   = 1;
    layer3_3_conv3_kernel_params.dW   = 1;
    layer3_3_conv3_kernel_params.kH   = 1;
    layer3_3_conv3_kernel_params.kW   = 1;
    layer3_3_conv3_kernel_params.padT = 0;
    layer3_3_conv3_kernel_params.padB = 0;
    layer3_3_conv3_kernel_params.padL = 0;
    layer3_3_conv3_kernel_params.padR = 0;
    layer3_3_conv3_kernel_params.dilH = 1;
    layer3_3_conv3_kernel_params.dilW = 1;

    // create layer3_3_conv3_weight tensor
    const unsigned      layer3_3_conv3_weight_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_3_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer3_3_conv3_weight_tr_info = {"layer3_3_conv3_weight", layer3_3_conv3_weight_dram};
    synTensor           layer3_3_conv3_weight =
        createTensor(4U, syn_type_bf16, layer3_3_conv3_weight_sizes, true, "layer3_3_conv3_weight");

    synTensor layer3_3_conv3_in_vec[4] = {layer3_3_relu2_output, layer3_3_conv3_weight, nullptr, nullptr};

    // create layer3_3_conv3_output tensor
    const unsigned layer3_3_conv3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_conv3_output =
        createTensor(4U, syn_type_bf16, layer3_3_conv3_output_sizes, false, "layer3_3_conv3_output");

    synTensor layer3_3_conv3_out_vec[1] = {layer3_3_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv3_in_vec,
                           layer3_3_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer3_3_conv3_kernel_params,
                           sizeof(layer3_3_conv3_kernel_params),
                           "spatial_convolution",
                           "layer3_3_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv3 failed!");

    /*************
     * layer3_3_bn3 node
     * inputs: [layer3_3_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_3_bn3_bias[1024](dtype=float32),
     *layer3_3_bn3_weight[1024](dtype=float32), layer3_3_bn3_running_mean[1024](dtype=float32),
     *layer3_3_bn3_running_var[1024](dtype=float32)] output: [layer3_3_bn3_output(64, 14, 14, 1024)(dtype=bf16),
     *layer3_3_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_3_bn3_saved_var(1, 1, 1, 1024)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_3_bn3_kernel_params;
    layer3_3_bn3_kernel_params.momentum    = 0.1;
    layer3_3_bn3_kernel_params.threshold.f = 1e-05;
    layer3_3_bn3_kernel_params.epsilon     = 1e-05;

    // create layer3_3_bn3_bias tensor
    const unsigned layer3_3_bn3_bias_sizes[] = {
        1024,
    };
    uint64_t            layer3_3_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer3_3_bn3_bias_tr_info = {"layer3_3_bn3_bias", layer3_3_bn3_bias_dram};
    synTensor layer3_3_bn3_bias = createTensor(1U, syn_type_single, layer3_3_bn3_bias_sizes, true, "layer3_3_bn3_bias");

    // create layer3_3_bn3_weight tensor
    const unsigned layer3_3_bn3_weight_sizes[] = {
        1024,
    };
    uint64_t            layer3_3_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer3_3_bn3_weight_tr_info = {"layer3_3_bn3_weight", layer3_3_bn3_weight_dram};
    synTensor           layer3_3_bn3_weight =
        createTensor(1U, syn_type_single, layer3_3_bn3_weight_sizes, true, "layer3_3_bn3_weight");

    // create layer3_3_bn3_running_mean tensor
    const unsigned layer3_3_bn3_running_mean_sizes[] = {
        1024,
    };
    uint64_t            layer3_3_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_3_bn3_running_mean_tr_info = {"layer3_3_bn3_running_mean",
                                                             layer3_3_bn3_running_mean_dram};
    synTensor           layer3_3_bn3_running_mean =
        createTensor(1U, syn_type_single, layer3_3_bn3_running_mean_sizes, true, "layer3_3_bn3_running_mean");

    // create layer3_3_bn3_running_var tensor
    const unsigned layer3_3_bn3_running_var_sizes[] = {
        1024,
    };
    uint64_t            layer3_3_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer3_3_bn3_running_var_tr_info = {"layer3_3_bn3_running_var", layer3_3_bn3_running_var_dram};
    synTensor           layer3_3_bn3_running_var =
        createTensor(1U, syn_type_single, layer3_3_bn3_running_var_sizes, true, "layer3_3_bn3_running_var");

    synTensor layer3_3_bn3_in_vec[5] = {layer3_3_conv3_output,
                                        layer3_3_bn3_bias,
                                        layer3_3_bn3_weight,
                                        layer3_3_bn3_running_mean,
                                        layer3_3_bn3_running_var};

    // create layer3_3_bn3_output tensor
    const unsigned layer3_3_bn3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_bn3_output =
        createTensor(4U, syn_type_bf16, layer3_3_bn3_output_sizes, false, "layer3_3_bn3_output");

    // create layer3_3_bn3_saved_mean tensor
    const unsigned layer3_3_bn3_saved_mean_sizes[] = {1024};
    synTensor      layer3_3_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer3_3_bn3_saved_mean_sizes, false, "layer3_3_bn3_saved_mean");

    // create layer3_3_bn3_saved_var tensor
    const unsigned layer3_3_bn3_saved_var_sizes[] = {1024};
    synTensor      layer3_3_bn3_saved_var =
        createTensor(1U, syn_type_single, layer3_3_bn3_saved_var_sizes, false, "layer3_3_bn3_saved_var");

    synTensor layer3_3_bn3_out_vec[3] = {layer3_3_bn3_output, layer3_3_bn3_saved_mean, layer3_3_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_3_bn3_in_vec,
                           layer3_3_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer3_3_bn3_kernel_params,
                           sizeof(layer3_3_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_3_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_bn3 failed!");

    /*************
     * layer3_3_add_residual_fwd0 node
     * inputs: [layer3_3_bn3_output(64, 14, 14, 1024)(dtype=bf16), layer3_2_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_3_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_3_add_residual_fwd0_in_vec[2] = {layer3_3_bn3_output, layer3_2_relu3_output};

    // create layer3_3_add_residual_fwd tensor
    const unsigned layer3_3_add_residual_fwd_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer3_3_add_residual_fwd_sizes, false, "layer3_3_add_residual_fwd");

    synTensor layer3_3_add_residual_fwd0_out_vec[1] = {layer3_3_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer3_3_add_residual_fwd0_in_vec,
                           layer3_3_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_3_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_add_residual_fwd0 failed!");

    /*************
     * layer3_3_relu3 node
     * inputs: [layer3_3_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_3_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_3_relu3_in_vec[1] = {layer3_3_add_residual_fwd};

    // create layer3_3_relu3_output tensor
    const unsigned layer3_3_relu3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_relu3_output =
        createTensor(4U, syn_type_bf16, layer3_3_relu3_output_sizes, false, "layer3_3_relu3_output");

    synTensor layer3_3_relu3_out_vec[1] = {layer3_3_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer3_3_relu3_in_vec,
                           layer3_3_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_3_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_relu3 failed!");

    /*************
     * layer3_4_conv1 node
     * inputs: [layer3_3_relu3_output(64, 14, 14, 1024)(dtype=bf16), layer3_4_conv1_weight[1, 1, 1024, 256](dtype=bf16)]
     * output: [layer3_4_conv1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_4_conv1_kernel_params;
    layer3_4_conv1_kernel_params.dH   = 1;
    layer3_4_conv1_kernel_params.dW   = 1;
    layer3_4_conv1_kernel_params.kH   = 1;
    layer3_4_conv1_kernel_params.kW   = 1;
    layer3_4_conv1_kernel_params.padT = 0;
    layer3_4_conv1_kernel_params.padB = 0;
    layer3_4_conv1_kernel_params.padL = 0;
    layer3_4_conv1_kernel_params.padR = 0;
    layer3_4_conv1_kernel_params.dilH = 1;
    layer3_4_conv1_kernel_params.dilW = 1;

    // create layer3_4_conv1_weight tensor
    const unsigned      layer3_4_conv1_weight_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_4_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer3_4_conv1_weight_tr_info = {"layer3_4_conv1_weight", layer3_4_conv1_weight_dram};
    synTensor           layer3_4_conv1_weight =
        createTensor(4U, syn_type_bf16, layer3_4_conv1_weight_sizes, true, "layer3_4_conv1_weight");

    synTensor layer3_4_conv1_in_vec[4] = {layer3_3_relu3_output, layer3_4_conv1_weight, nullptr, nullptr};

    // create layer3_4_conv1_output tensor
    const unsigned layer3_4_conv1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_conv1_output =
        createTensor(4U, syn_type_bf16, layer3_4_conv1_output_sizes, false, "layer3_4_conv1_output");

    synTensor layer3_4_conv1_out_vec[1] = {layer3_4_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv1_in_vec,
                           layer3_4_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer3_4_conv1_kernel_params,
                           sizeof(layer3_4_conv1_kernel_params),
                           "spatial_convolution",
                           "layer3_4_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv1 failed!");

    /*************
     * layer3_4_bn1 node
     * inputs: [layer3_4_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_4_bn1_bias[256](dtype=float32),
     *layer3_4_bn1_weight[256](dtype=float32), layer3_4_bn1_running_mean[256](dtype=float32),
     *layer3_4_bn1_running_var[256](dtype=float32)] output: [layer3_4_bn1_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_4_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_4_bn1_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_4_bn1_kernel_params;
    layer3_4_bn1_kernel_params.momentum    = 0.1;
    layer3_4_bn1_kernel_params.threshold.f = 1e-05;
    layer3_4_bn1_kernel_params.epsilon     = 1e-05;

    // create layer3_4_bn1_bias tensor
    const unsigned layer3_4_bn1_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer3_4_bn1_bias_tr_info = {"layer3_4_bn1_bias", layer3_4_bn1_bias_dram};
    synTensor layer3_4_bn1_bias = createTensor(1U, syn_type_single, layer3_4_bn1_bias_sizes, true, "layer3_4_bn1_bias");

    // create layer3_4_bn1_weight tensor
    const unsigned layer3_4_bn1_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer3_4_bn1_weight_tr_info = {"layer3_4_bn1_weight", layer3_4_bn1_weight_dram};
    synTensor           layer3_4_bn1_weight =
        createTensor(1U, syn_type_single, layer3_4_bn1_weight_sizes, true, "layer3_4_bn1_weight");

    // create layer3_4_bn1_running_mean tensor
    const unsigned layer3_4_bn1_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_4_bn1_running_mean_tr_info = {"layer3_4_bn1_running_mean",
                                                             layer3_4_bn1_running_mean_dram};
    synTensor           layer3_4_bn1_running_mean =
        createTensor(1U, syn_type_single, layer3_4_bn1_running_mean_sizes, true, "layer3_4_bn1_running_mean");

    // create layer3_4_bn1_running_var tensor
    const unsigned layer3_4_bn1_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer3_4_bn1_running_var_tr_info = {"layer3_4_bn1_running_var", layer3_4_bn1_running_var_dram};
    synTensor           layer3_4_bn1_running_var =
        createTensor(1U, syn_type_single, layer3_4_bn1_running_var_sizes, true, "layer3_4_bn1_running_var");

    synTensor layer3_4_bn1_in_vec[5] = {layer3_4_conv1_output,
                                        layer3_4_bn1_bias,
                                        layer3_4_bn1_weight,
                                        layer3_4_bn1_running_mean,
                                        layer3_4_bn1_running_var};

    // create layer3_4_bn1_output tensor
    const unsigned layer3_4_bn1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_bn1_output =
        createTensor(4U, syn_type_bf16, layer3_4_bn1_output_sizes, false, "layer3_4_bn1_output");

    // create layer3_4_bn1_saved_mean tensor
    const unsigned layer3_4_bn1_saved_mean_sizes[] = {256};
    synTensor      layer3_4_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer3_4_bn1_saved_mean_sizes, false, "layer3_4_bn1_saved_mean");

    // create layer3_4_bn1_saved_var tensor
    const unsigned layer3_4_bn1_saved_var_sizes[] = {256};
    synTensor      layer3_4_bn1_saved_var =
        createTensor(1U, syn_type_single, layer3_4_bn1_saved_var_sizes, false, "layer3_4_bn1_saved_var");

    synTensor layer3_4_bn1_out_vec[3] = {layer3_4_bn1_output, layer3_4_bn1_saved_mean, layer3_4_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_4_bn1_in_vec,
                           layer3_4_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer3_4_bn1_kernel_params,
                           sizeof(layer3_4_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_4_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_bn1 failed!");

    /*************
     * layer3_4_relu1 node
     * inputs: [layer3_4_bn1_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_4_relu1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_4_relu1_in_vec[1] = {layer3_4_bn1_output};

    // create layer3_4_relu1_output tensor
    const unsigned layer3_4_relu1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_relu1_output =
        createTensor(4U, syn_type_bf16, layer3_4_relu1_output_sizes, false, "layer3_4_relu1_output");

    synTensor layer3_4_relu1_out_vec[1] = {layer3_4_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer3_4_relu1_in_vec,
                           layer3_4_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_4_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_relu1 failed!");

    /*************
     * layer3_4_conv2 node
     * inputs: [layer3_4_relu1_output(64, 14, 14, 256)(dtype=bf16), layer3_4_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_4_conv2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_4_conv2_kernel_params;
    layer3_4_conv2_kernel_params.dH   = 1;
    layer3_4_conv2_kernel_params.dW   = 1;
    layer3_4_conv2_kernel_params.kH   = 3;
    layer3_4_conv2_kernel_params.kW   = 3;
    layer3_4_conv2_kernel_params.padT = 1;
    layer3_4_conv2_kernel_params.padB = 1;
    layer3_4_conv2_kernel_params.padL = 1;
    layer3_4_conv2_kernel_params.padR = 1;
    layer3_4_conv2_kernel_params.dilH = 1;
    layer3_4_conv2_kernel_params.dilW = 1;

    // create layer3_4_conv2_weight tensor
    const unsigned      layer3_4_conv2_weight_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_4_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer3_4_conv2_weight_tr_info = {"layer3_4_conv2_weight", layer3_4_conv2_weight_dram};
    synTensor           layer3_4_conv2_weight =
        createTensor(4U, syn_type_bf16, layer3_4_conv2_weight_sizes, true, "layer3_4_conv2_weight");

    synTensor layer3_4_conv2_in_vec[4] = {layer3_4_relu1_output, layer3_4_conv2_weight, nullptr, nullptr};

    // create layer3_4_conv2_output tensor
    const unsigned layer3_4_conv2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_conv2_output =
        createTensor(4U, syn_type_bf16, layer3_4_conv2_output_sizes, false, "layer3_4_conv2_output");

    synTensor layer3_4_conv2_out_vec[1] = {layer3_4_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv2_in_vec,
                           layer3_4_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer3_4_conv2_kernel_params,
                           sizeof(layer3_4_conv2_kernel_params),
                           "spatial_convolution",
                           "layer3_4_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv2 failed!");

    /*************
     * layer3_4_bn2 node
     * inputs: [layer3_4_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_4_bn2_bias[256](dtype=float32),
     *layer3_4_bn2_weight[256](dtype=float32), layer3_4_bn2_running_mean[256](dtype=float32),
     *layer3_4_bn2_running_var[256](dtype=float32)] output: [layer3_4_bn2_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_4_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_4_bn2_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_4_bn2_kernel_params;
    layer3_4_bn2_kernel_params.momentum    = 0.1;
    layer3_4_bn2_kernel_params.threshold.f = 1e-05;
    layer3_4_bn2_kernel_params.epsilon     = 1e-05;

    // create layer3_4_bn2_bias tensor
    const unsigned layer3_4_bn2_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer3_4_bn2_bias_tr_info = {"layer3_4_bn2_bias", layer3_4_bn2_bias_dram};
    synTensor layer3_4_bn2_bias = createTensor(1U, syn_type_single, layer3_4_bn2_bias_sizes, true, "layer3_4_bn2_bias");

    // create layer3_4_bn2_weight tensor
    const unsigned layer3_4_bn2_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer3_4_bn2_weight_tr_info = {"layer3_4_bn2_weight", layer3_4_bn2_weight_dram};
    synTensor           layer3_4_bn2_weight =
        createTensor(1U, syn_type_single, layer3_4_bn2_weight_sizes, true, "layer3_4_bn2_weight");

    // create layer3_4_bn2_running_mean tensor
    const unsigned layer3_4_bn2_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_4_bn2_running_mean_tr_info = {"layer3_4_bn2_running_mean",
                                                             layer3_4_bn2_running_mean_dram};
    synTensor           layer3_4_bn2_running_mean =
        createTensor(1U, syn_type_single, layer3_4_bn2_running_mean_sizes, true, "layer3_4_bn2_running_mean");

    // create layer3_4_bn2_running_var tensor
    const unsigned layer3_4_bn2_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer3_4_bn2_running_var_tr_info = {"layer3_4_bn2_running_var", layer3_4_bn2_running_var_dram};
    synTensor           layer3_4_bn2_running_var =
        createTensor(1U, syn_type_single, layer3_4_bn2_running_var_sizes, true, "layer3_4_bn2_running_var");

    synTensor layer3_4_bn2_in_vec[5] = {layer3_4_conv2_output,
                                        layer3_4_bn2_bias,
                                        layer3_4_bn2_weight,
                                        layer3_4_bn2_running_mean,
                                        layer3_4_bn2_running_var};

    // create layer3_4_bn2_output tensor
    const unsigned layer3_4_bn2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_bn2_output =
        createTensor(4U, syn_type_bf16, layer3_4_bn2_output_sizes, false, "layer3_4_bn2_output");

    // create layer3_4_bn2_saved_mean tensor
    const unsigned layer3_4_bn2_saved_mean_sizes[] = {256};
    synTensor      layer3_4_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer3_4_bn2_saved_mean_sizes, false, "layer3_4_bn2_saved_mean");

    // create layer3_4_bn2_saved_var tensor
    const unsigned layer3_4_bn2_saved_var_sizes[] = {256};
    synTensor      layer3_4_bn2_saved_var =
        createTensor(1U, syn_type_single, layer3_4_bn2_saved_var_sizes, false, "layer3_4_bn2_saved_var");

    synTensor layer3_4_bn2_out_vec[3] = {layer3_4_bn2_output, layer3_4_bn2_saved_mean, layer3_4_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_4_bn2_in_vec,
                           layer3_4_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer3_4_bn2_kernel_params,
                           sizeof(layer3_4_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_4_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_bn2 failed!");

    /*************
     * layer3_4_relu2 node
     * inputs: [layer3_4_bn2_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_4_relu2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_4_relu2_in_vec[1] = {layer3_4_bn2_output};

    // create layer3_4_relu2_output tensor
    const unsigned layer3_4_relu2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_relu2_output =
        createTensor(4U, syn_type_bf16, layer3_4_relu2_output_sizes, false, "layer3_4_relu2_output");

    synTensor layer3_4_relu2_out_vec[1] = {layer3_4_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer3_4_relu2_in_vec,
                           layer3_4_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_4_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_relu2 failed!");

    /*************
     * layer3_4_conv3 node
     * inputs: [layer3_4_relu2_output(64, 14, 14, 256)(dtype=bf16), layer3_4_conv3_weight[1, 1, 256, 1024](dtype=bf16)]
     * output: [layer3_4_conv3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_4_conv3_kernel_params;
    layer3_4_conv3_kernel_params.dH   = 1;
    layer3_4_conv3_kernel_params.dW   = 1;
    layer3_4_conv3_kernel_params.kH   = 1;
    layer3_4_conv3_kernel_params.kW   = 1;
    layer3_4_conv3_kernel_params.padT = 0;
    layer3_4_conv3_kernel_params.padB = 0;
    layer3_4_conv3_kernel_params.padL = 0;
    layer3_4_conv3_kernel_params.padR = 0;
    layer3_4_conv3_kernel_params.dilH = 1;
    layer3_4_conv3_kernel_params.dilW = 1;

    // create layer3_4_conv3_weight tensor
    const unsigned      layer3_4_conv3_weight_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_4_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer3_4_conv3_weight_tr_info = {"layer3_4_conv3_weight", layer3_4_conv3_weight_dram};
    synTensor           layer3_4_conv3_weight =
        createTensor(4U, syn_type_bf16, layer3_4_conv3_weight_sizes, true, "layer3_4_conv3_weight");

    synTensor layer3_4_conv3_in_vec[4] = {layer3_4_relu2_output, layer3_4_conv3_weight, nullptr, nullptr};

    // create layer3_4_conv3_output tensor
    const unsigned layer3_4_conv3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_conv3_output =
        createTensor(4U, syn_type_bf16, layer3_4_conv3_output_sizes, false, "layer3_4_conv3_output");

    synTensor layer3_4_conv3_out_vec[1] = {layer3_4_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv3_in_vec,
                           layer3_4_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer3_4_conv3_kernel_params,
                           sizeof(layer3_4_conv3_kernel_params),
                           "spatial_convolution",
                           "layer3_4_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv3 failed!");

    /*************
     * layer3_4_bn3 node
     * inputs: [layer3_4_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_4_bn3_bias[1024](dtype=float32),
     *layer3_4_bn3_weight[1024](dtype=float32), layer3_4_bn3_running_mean[1024](dtype=float32),
     *layer3_4_bn3_running_var[1024](dtype=float32)] output: [layer3_4_bn3_output(64, 14, 14, 1024)(dtype=bf16),
     *layer3_4_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_4_bn3_saved_var(1, 1, 1, 1024)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_4_bn3_kernel_params;
    layer3_4_bn3_kernel_params.momentum    = 0.1;
    layer3_4_bn3_kernel_params.threshold.f = 1e-05;
    layer3_4_bn3_kernel_params.epsilon     = 1e-05;

    // create layer3_4_bn3_bias tensor
    const unsigned layer3_4_bn3_bias_sizes[] = {
        1024,
    };
    uint64_t            layer3_4_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer3_4_bn3_bias_tr_info = {"layer3_4_bn3_bias", layer3_4_bn3_bias_dram};
    synTensor layer3_4_bn3_bias = createTensor(1U, syn_type_single, layer3_4_bn3_bias_sizes, true, "layer3_4_bn3_bias");

    // create layer3_4_bn3_weight tensor
    const unsigned layer3_4_bn3_weight_sizes[] = {
        1024,
    };
    uint64_t            layer3_4_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer3_4_bn3_weight_tr_info = {"layer3_4_bn3_weight", layer3_4_bn3_weight_dram};
    synTensor           layer3_4_bn3_weight =
        createTensor(1U, syn_type_single, layer3_4_bn3_weight_sizes, true, "layer3_4_bn3_weight");

    // create layer3_4_bn3_running_mean tensor
    const unsigned layer3_4_bn3_running_mean_sizes[] = {
        1024,
    };
    uint64_t            layer3_4_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_4_bn3_running_mean_tr_info = {"layer3_4_bn3_running_mean",
                                                             layer3_4_bn3_running_mean_dram};
    synTensor           layer3_4_bn3_running_mean =
        createTensor(1U, syn_type_single, layer3_4_bn3_running_mean_sizes, true, "layer3_4_bn3_running_mean");

    // create layer3_4_bn3_running_var tensor
    const unsigned layer3_4_bn3_running_var_sizes[] = {
        1024,
    };
    uint64_t            layer3_4_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer3_4_bn3_running_var_tr_info = {"layer3_4_bn3_running_var", layer3_4_bn3_running_var_dram};
    synTensor           layer3_4_bn3_running_var =
        createTensor(1U, syn_type_single, layer3_4_bn3_running_var_sizes, true, "layer3_4_bn3_running_var");

    synTensor layer3_4_bn3_in_vec[5] = {layer3_4_conv3_output,
                                        layer3_4_bn3_bias,
                                        layer3_4_bn3_weight,
                                        layer3_4_bn3_running_mean,
                                        layer3_4_bn3_running_var};

    // create layer3_4_bn3_output tensor
    const unsigned layer3_4_bn3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_bn3_output =
        createTensor(4U, syn_type_bf16, layer3_4_bn3_output_sizes, false, "layer3_4_bn3_output");

    // create layer3_4_bn3_saved_mean tensor
    const unsigned layer3_4_bn3_saved_mean_sizes[] = {1024};
    synTensor      layer3_4_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer3_4_bn3_saved_mean_sizes, false, "layer3_4_bn3_saved_mean");

    // create layer3_4_bn3_saved_var tensor
    const unsigned layer3_4_bn3_saved_var_sizes[] = {1024};
    synTensor      layer3_4_bn3_saved_var =
        createTensor(1U, syn_type_single, layer3_4_bn3_saved_var_sizes, false, "layer3_4_bn3_saved_var");

    synTensor layer3_4_bn3_out_vec[3] = {layer3_4_bn3_output, layer3_4_bn3_saved_mean, layer3_4_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_4_bn3_in_vec,
                           layer3_4_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer3_4_bn3_kernel_params,
                           sizeof(layer3_4_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_4_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_bn3 failed!");

    /*************
     * layer3_4_add_residual_fwd0 node
     * inputs: [layer3_4_bn3_output(64, 14, 14, 1024)(dtype=bf16), layer3_3_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_4_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_4_add_residual_fwd0_in_vec[2] = {layer3_4_bn3_output, layer3_3_relu3_output};

    // create layer3_4_add_residual_fwd tensor
    const unsigned layer3_4_add_residual_fwd_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer3_4_add_residual_fwd_sizes, false, "layer3_4_add_residual_fwd");

    synTensor layer3_4_add_residual_fwd0_out_vec[1] = {layer3_4_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer3_4_add_residual_fwd0_in_vec,
                           layer3_4_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_4_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_add_residual_fwd0 failed!");

    /*************
     * layer3_4_relu3 node
     * inputs: [layer3_4_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_4_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_4_relu3_in_vec[1] = {layer3_4_add_residual_fwd};

    // create layer3_4_relu3_output tensor
    const unsigned layer3_4_relu3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_relu3_output =
        createTensor(4U, syn_type_bf16, layer3_4_relu3_output_sizes, false, "layer3_4_relu3_output");

    synTensor layer3_4_relu3_out_vec[1] = {layer3_4_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer3_4_relu3_in_vec,
                           layer3_4_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_4_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_relu3 failed!");

    /*************
     * layer3_5_conv1 node
     * inputs: [layer3_4_relu3_output(64, 14, 14, 1024)(dtype=bf16), layer3_5_conv1_weight[1, 1, 1024, 256](dtype=bf16)]
     * output: [layer3_5_conv1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_5_conv1_kernel_params;
    layer3_5_conv1_kernel_params.dH   = 1;
    layer3_5_conv1_kernel_params.dW   = 1;
    layer3_5_conv1_kernel_params.kH   = 1;
    layer3_5_conv1_kernel_params.kW   = 1;
    layer3_5_conv1_kernel_params.padT = 0;
    layer3_5_conv1_kernel_params.padB = 0;
    layer3_5_conv1_kernel_params.padL = 0;
    layer3_5_conv1_kernel_params.padR = 0;
    layer3_5_conv1_kernel_params.dilH = 1;
    layer3_5_conv1_kernel_params.dilW = 1;

    // create layer3_5_conv1_weight tensor
    const unsigned      layer3_5_conv1_weight_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_5_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer3_5_conv1_weight_tr_info = {"layer3_5_conv1_weight", layer3_5_conv1_weight_dram};
    synTensor           layer3_5_conv1_weight =
        createTensor(4U, syn_type_bf16, layer3_5_conv1_weight_sizes, true, "layer3_5_conv1_weight");

    synTensor layer3_5_conv1_in_vec[4] = {layer3_4_relu3_output, layer3_5_conv1_weight, nullptr, nullptr};

    // create layer3_5_conv1_output tensor
    const unsigned layer3_5_conv1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_conv1_output =
        createTensor(4U, syn_type_bf16, layer3_5_conv1_output_sizes, false, "layer3_5_conv1_output");

    synTensor layer3_5_conv1_out_vec[1] = {layer3_5_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv1_in_vec,
                           layer3_5_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer3_5_conv1_kernel_params,
                           sizeof(layer3_5_conv1_kernel_params),
                           "spatial_convolution",
                           "layer3_5_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv1 failed!");

    /*************
     * layer3_5_bn1 node
     * inputs: [layer3_5_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_5_bn1_bias[256](dtype=float32),
     *layer3_5_bn1_weight[256](dtype=float32), layer3_5_bn1_running_mean[256](dtype=float32),
     *layer3_5_bn1_running_var[256](dtype=float32)] output: [layer3_5_bn1_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_5_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_5_bn1_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_5_bn1_kernel_params;
    layer3_5_bn1_kernel_params.momentum    = 0.1;
    layer3_5_bn1_kernel_params.threshold.f = 1e-05;
    layer3_5_bn1_kernel_params.epsilon     = 1e-05;

    // create layer3_5_bn1_bias tensor
    const unsigned layer3_5_bn1_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer3_5_bn1_bias_tr_info = {"layer3_5_bn1_bias", layer3_5_bn1_bias_dram};
    synTensor layer3_5_bn1_bias = createTensor(1U, syn_type_single, layer3_5_bn1_bias_sizes, true, "layer3_5_bn1_bias");

    // create layer3_5_bn1_weight tensor
    const unsigned layer3_5_bn1_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer3_5_bn1_weight_tr_info = {"layer3_5_bn1_weight", layer3_5_bn1_weight_dram};
    synTensor           layer3_5_bn1_weight =
        createTensor(1U, syn_type_single, layer3_5_bn1_weight_sizes, true, "layer3_5_bn1_weight");

    // create layer3_5_bn1_running_mean tensor
    const unsigned layer3_5_bn1_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_5_bn1_running_mean_tr_info = {"layer3_5_bn1_running_mean",
                                                             layer3_5_bn1_running_mean_dram};
    synTensor           layer3_5_bn1_running_mean =
        createTensor(1U, syn_type_single, layer3_5_bn1_running_mean_sizes, true, "layer3_5_bn1_running_mean");

    // create layer3_5_bn1_running_var tensor
    const unsigned layer3_5_bn1_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer3_5_bn1_running_var_tr_info = {"layer3_5_bn1_running_var", layer3_5_bn1_running_var_dram};
    synTensor           layer3_5_bn1_running_var =
        createTensor(1U, syn_type_single, layer3_5_bn1_running_var_sizes, true, "layer3_5_bn1_running_var");

    synTensor layer3_5_bn1_in_vec[5] = {layer3_5_conv1_output,
                                        layer3_5_bn1_bias,
                                        layer3_5_bn1_weight,
                                        layer3_5_bn1_running_mean,
                                        layer3_5_bn1_running_var};

    // create layer3_5_bn1_output tensor
    const unsigned layer3_5_bn1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_bn1_output =
        createTensor(4U, syn_type_bf16, layer3_5_bn1_output_sizes, false, "layer3_5_bn1_output");

    // create layer3_5_bn1_saved_mean tensor
    const unsigned layer3_5_bn1_saved_mean_sizes[] = {256};
    synTensor      layer3_5_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer3_5_bn1_saved_mean_sizes, false, "layer3_5_bn1_saved_mean");

    // create layer3_5_bn1_saved_var tensor
    const unsigned layer3_5_bn1_saved_var_sizes[] = {256};
    synTensor      layer3_5_bn1_saved_var =
        createTensor(1U, syn_type_single, layer3_5_bn1_saved_var_sizes, false, "layer3_5_bn1_saved_var");

    synTensor layer3_5_bn1_out_vec[3] = {layer3_5_bn1_output, layer3_5_bn1_saved_mean, layer3_5_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_5_bn1_in_vec,
                           layer3_5_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer3_5_bn1_kernel_params,
                           sizeof(layer3_5_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_5_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_bn1 failed!");

    /*************
     * layer3_5_relu1 node
     * inputs: [layer3_5_bn1_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_5_relu1_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_5_relu1_in_vec[1] = {layer3_5_bn1_output};

    // create layer3_5_relu1_output tensor
    const unsigned layer3_5_relu1_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_relu1_output =
        createTensor(4U, syn_type_bf16, layer3_5_relu1_output_sizes, false, "layer3_5_relu1_output");

    synTensor layer3_5_relu1_out_vec[1] = {layer3_5_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer3_5_relu1_in_vec,
                           layer3_5_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_5_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_relu1 failed!");

    /*************
     * layer3_5_conv2 node
     * inputs: [layer3_5_relu1_output(64, 14, 14, 256)(dtype=bf16), layer3_5_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_5_conv2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_5_conv2_kernel_params;
    layer3_5_conv2_kernel_params.dH   = 1;
    layer3_5_conv2_kernel_params.dW   = 1;
    layer3_5_conv2_kernel_params.kH   = 3;
    layer3_5_conv2_kernel_params.kW   = 3;
    layer3_5_conv2_kernel_params.padT = 1;
    layer3_5_conv2_kernel_params.padB = 1;
    layer3_5_conv2_kernel_params.padL = 1;
    layer3_5_conv2_kernel_params.padR = 1;
    layer3_5_conv2_kernel_params.dilH = 1;
    layer3_5_conv2_kernel_params.dilW = 1;

    // create layer3_5_conv2_weight tensor
    const unsigned      layer3_5_conv2_weight_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_5_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer3_5_conv2_weight_tr_info = {"layer3_5_conv2_weight", layer3_5_conv2_weight_dram};
    synTensor           layer3_5_conv2_weight =
        createTensor(4U, syn_type_bf16, layer3_5_conv2_weight_sizes, true, "layer3_5_conv2_weight");

    synTensor layer3_5_conv2_in_vec[4] = {layer3_5_relu1_output, layer3_5_conv2_weight, nullptr, nullptr};

    // create layer3_5_conv2_output tensor
    const unsigned layer3_5_conv2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_conv2_output =
        createTensor(4U, syn_type_bf16, layer3_5_conv2_output_sizes, false, "layer3_5_conv2_output");

    synTensor layer3_5_conv2_out_vec[1] = {layer3_5_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv2_in_vec,
                           layer3_5_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer3_5_conv2_kernel_params,
                           sizeof(layer3_5_conv2_kernel_params),
                           "spatial_convolution",
                           "layer3_5_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv2 failed!");

    /*************
     * layer3_5_bn2 node
     * inputs: [layer3_5_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_5_bn2_bias[256](dtype=float32),
     *layer3_5_bn2_weight[256](dtype=float32), layer3_5_bn2_running_mean[256](dtype=float32),
     *layer3_5_bn2_running_var[256](dtype=float32)] output: [layer3_5_bn2_output(64, 14, 14, 256)(dtype=bf16),
     *layer3_5_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_5_bn2_saved_var(1, 1, 1, 256)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_5_bn2_kernel_params;
    layer3_5_bn2_kernel_params.momentum    = 0.1;
    layer3_5_bn2_kernel_params.threshold.f = 1e-05;
    layer3_5_bn2_kernel_params.epsilon     = 1e-05;

    // create layer3_5_bn2_bias tensor
    const unsigned layer3_5_bn2_bias_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer3_5_bn2_bias_tr_info = {"layer3_5_bn2_bias", layer3_5_bn2_bias_dram};
    synTensor layer3_5_bn2_bias = createTensor(1U, syn_type_single, layer3_5_bn2_bias_sizes, true, "layer3_5_bn2_bias");

    // create layer3_5_bn2_weight tensor
    const unsigned layer3_5_bn2_weight_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer3_5_bn2_weight_tr_info = {"layer3_5_bn2_weight", layer3_5_bn2_weight_dram};
    synTensor           layer3_5_bn2_weight =
        createTensor(1U, syn_type_single, layer3_5_bn2_weight_sizes, true, "layer3_5_bn2_weight");

    // create layer3_5_bn2_running_mean tensor
    const unsigned layer3_5_bn2_running_mean_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_5_bn2_running_mean_tr_info = {"layer3_5_bn2_running_mean",
                                                             layer3_5_bn2_running_mean_dram};
    synTensor           layer3_5_bn2_running_mean =
        createTensor(1U, syn_type_single, layer3_5_bn2_running_mean_sizes, true, "layer3_5_bn2_running_mean");

    // create layer3_5_bn2_running_var tensor
    const unsigned layer3_5_bn2_running_var_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer3_5_bn2_running_var_tr_info = {"layer3_5_bn2_running_var", layer3_5_bn2_running_var_dram};
    synTensor           layer3_5_bn2_running_var =
        createTensor(1U, syn_type_single, layer3_5_bn2_running_var_sizes, true, "layer3_5_bn2_running_var");

    synTensor layer3_5_bn2_in_vec[5] = {layer3_5_conv2_output,
                                        layer3_5_bn2_bias,
                                        layer3_5_bn2_weight,
                                        layer3_5_bn2_running_mean,
                                        layer3_5_bn2_running_var};

    // create layer3_5_bn2_output tensor
    const unsigned layer3_5_bn2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_bn2_output =
        createTensor(4U, syn_type_bf16, layer3_5_bn2_output_sizes, false, "layer3_5_bn2_output");

    // create layer3_5_bn2_saved_mean tensor
    const unsigned layer3_5_bn2_saved_mean_sizes[] = {256};
    synTensor      layer3_5_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer3_5_bn2_saved_mean_sizes, false, "layer3_5_bn2_saved_mean");

    // create layer3_5_bn2_saved_var tensor
    const unsigned layer3_5_bn2_saved_var_sizes[] = {256};
    synTensor      layer3_5_bn2_saved_var =
        createTensor(1U, syn_type_single, layer3_5_bn2_saved_var_sizes, false, "layer3_5_bn2_saved_var");

    synTensor layer3_5_bn2_out_vec[3] = {layer3_5_bn2_output, layer3_5_bn2_saved_mean, layer3_5_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_5_bn2_in_vec,
                           layer3_5_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer3_5_bn2_kernel_params,
                           sizeof(layer3_5_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_5_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_bn2 failed!");

    /*************
     * layer3_5_relu2 node
     * inputs: [layer3_5_bn2_output(64, 14, 14, 256)(dtype=bf16)]
     * output: [layer3_5_relu2_output(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_5_relu2_in_vec[1] = {layer3_5_bn2_output};

    // create layer3_5_relu2_output tensor
    const unsigned layer3_5_relu2_output_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_relu2_output =
        createTensor(4U, syn_type_bf16, layer3_5_relu2_output_sizes, false, "layer3_5_relu2_output");

    synTensor layer3_5_relu2_out_vec[1] = {layer3_5_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer3_5_relu2_in_vec,
                           layer3_5_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_5_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_relu2 failed!");

    /*************
     * layer3_5_conv3 node
     * inputs: [layer3_5_relu2_output(64, 14, 14, 256)(dtype=bf16), layer3_5_conv3_weight[1, 1, 256, 1024](dtype=bf16)]
     * output: [layer3_5_conv3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_5_conv3_kernel_params;
    layer3_5_conv3_kernel_params.dH   = 1;
    layer3_5_conv3_kernel_params.dW   = 1;
    layer3_5_conv3_kernel_params.kH   = 1;
    layer3_5_conv3_kernel_params.kW   = 1;
    layer3_5_conv3_kernel_params.padT = 0;
    layer3_5_conv3_kernel_params.padB = 0;
    layer3_5_conv3_kernel_params.padL = 0;
    layer3_5_conv3_kernel_params.padR = 0;
    layer3_5_conv3_kernel_params.dilH = 1;
    layer3_5_conv3_kernel_params.dilW = 1;

    // create layer3_5_conv3_weight tensor
    const unsigned      layer3_5_conv3_weight_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_5_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer3_5_conv3_weight_tr_info = {"layer3_5_conv3_weight", layer3_5_conv3_weight_dram};
    synTensor           layer3_5_conv3_weight =
        createTensor(4U, syn_type_bf16, layer3_5_conv3_weight_sizes, true, "layer3_5_conv3_weight");

    synTensor layer3_5_conv3_in_vec[4] = {layer3_5_relu2_output, layer3_5_conv3_weight, nullptr, nullptr};

    // create layer3_5_conv3_output tensor
    const unsigned layer3_5_conv3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_conv3_output =
        createTensor(4U, syn_type_bf16, layer3_5_conv3_output_sizes, false, "layer3_5_conv3_output");

    synTensor layer3_5_conv3_out_vec[1] = {layer3_5_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv3_in_vec,
                           layer3_5_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer3_5_conv3_kernel_params,
                           sizeof(layer3_5_conv3_kernel_params),
                           "spatial_convolution",
                           "layer3_5_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv3 failed!");

    /*************
     * layer3_5_bn3 node
     * inputs: [layer3_5_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_5_bn3_bias[1024](dtype=float32),
     *layer3_5_bn3_weight[1024](dtype=float32), layer3_5_bn3_running_mean[1024](dtype=float32),
     *layer3_5_bn3_running_var[1024](dtype=float32)] output: [layer3_5_bn3_output(64, 14, 14, 1024)(dtype=bf16),
     *layer3_5_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_5_bn3_saved_var(1, 1, 1, 1024)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer3_5_bn3_kernel_params;
    layer3_5_bn3_kernel_params.momentum    = 0.1;
    layer3_5_bn3_kernel_params.threshold.f = 1e-05;
    layer3_5_bn3_kernel_params.epsilon     = 1e-05;

    // create layer3_5_bn3_bias tensor
    const unsigned layer3_5_bn3_bias_sizes[] = {
        1024,
    };
    uint64_t            layer3_5_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer3_5_bn3_bias_tr_info = {"layer3_5_bn3_bias", layer3_5_bn3_bias_dram};
    synTensor layer3_5_bn3_bias = createTensor(1U, syn_type_single, layer3_5_bn3_bias_sizes, true, "layer3_5_bn3_bias");

    // create layer3_5_bn3_weight tensor
    const unsigned layer3_5_bn3_weight_sizes[] = {
        1024,
    };
    uint64_t            layer3_5_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer3_5_bn3_weight_tr_info = {"layer3_5_bn3_weight", layer3_5_bn3_weight_dram};
    synTensor           layer3_5_bn3_weight =
        createTensor(1U, syn_type_single, layer3_5_bn3_weight_sizes, true, "layer3_5_bn3_weight");

    // create layer3_5_bn3_running_mean tensor
    const unsigned layer3_5_bn3_running_mean_sizes[] = {
        1024,
    };
    uint64_t            layer3_5_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer3_5_bn3_running_mean_tr_info = {"layer3_5_bn3_running_mean",
                                                             layer3_5_bn3_running_mean_dram};
    synTensor           layer3_5_bn3_running_mean =
        createTensor(1U, syn_type_single, layer3_5_bn3_running_mean_sizes, true, "layer3_5_bn3_running_mean");

    // create layer3_5_bn3_running_var tensor
    const unsigned layer3_5_bn3_running_var_sizes[] = {
        1024,
    };
    uint64_t            layer3_5_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer3_5_bn3_running_var_tr_info = {"layer3_5_bn3_running_var", layer3_5_bn3_running_var_dram};
    synTensor           layer3_5_bn3_running_var =
        createTensor(1U, syn_type_single, layer3_5_bn3_running_var_sizes, true, "layer3_5_bn3_running_var");

    synTensor layer3_5_bn3_in_vec[5] = {layer3_5_conv3_output,
                                        layer3_5_bn3_bias,
                                        layer3_5_bn3_weight,
                                        layer3_5_bn3_running_mean,
                                        layer3_5_bn3_running_var};

    // create layer3_5_bn3_output tensor
    const unsigned layer3_5_bn3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_bn3_output =
        createTensor(4U, syn_type_bf16, layer3_5_bn3_output_sizes, false, "layer3_5_bn3_output");

    // create layer3_5_bn3_saved_mean tensor
    const unsigned layer3_5_bn3_saved_mean_sizes[] = {1024};
    synTensor      layer3_5_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer3_5_bn3_saved_mean_sizes, false, "layer3_5_bn3_saved_mean");

    // create layer3_5_bn3_saved_var tensor
    const unsigned layer3_5_bn3_saved_var_sizes[] = {1024};
    synTensor      layer3_5_bn3_saved_var =
        createTensor(1U, syn_type_single, layer3_5_bn3_saved_var_sizes, false, "layer3_5_bn3_saved_var");

    synTensor layer3_5_bn3_out_vec[3] = {layer3_5_bn3_output, layer3_5_bn3_saved_mean, layer3_5_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer3_5_bn3_in_vec,
                           layer3_5_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer3_5_bn3_kernel_params,
                           sizeof(layer3_5_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer3_5_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_bn3 failed!");

    /*************
     * layer3_5_add_residual_fwd0 node
     * inputs: [layer3_5_bn3_output(64, 14, 14, 1024)(dtype=bf16), layer3_4_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_5_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_5_add_residual_fwd0_in_vec[2] = {layer3_5_bn3_output, layer3_4_relu3_output};

    // create layer3_5_add_residual_fwd tensor
    const unsigned layer3_5_add_residual_fwd_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer3_5_add_residual_fwd_sizes, false, "layer3_5_add_residual_fwd");

    synTensor layer3_5_add_residual_fwd0_out_vec[1] = {layer3_5_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer3_5_add_residual_fwd0_in_vec,
                           layer3_5_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_5_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_add_residual_fwd0 failed!");

    /*************
     * layer3_5_relu3 node
     * inputs: [layer3_5_add_residual_fwd(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_5_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_5_relu3_in_vec[1] = {layer3_5_add_residual_fwd};

    // create layer3_5_relu3_output tensor
    const unsigned layer3_5_relu3_output_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_relu3_output =
        createTensor(4U, syn_type_bf16, layer3_5_relu3_output_sizes, false, "layer3_5_relu3_output");

    synTensor layer3_5_relu3_out_vec[1] = {layer3_5_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer3_5_relu3_in_vec,
                           layer3_5_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer3_5_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_relu3 failed!");

    /*************
     * layer4_0_conv1 node
     * inputs: [layer3_5_relu3_output(64, 14, 14, 1024)(dtype=bf16), layer4_0_conv1_weight[1, 1, 1024, 512](dtype=bf16)]
     * output: [layer4_0_conv1_output(64, 14, 14, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_0_conv1_kernel_params;
    layer4_0_conv1_kernel_params.dH   = 1;
    layer4_0_conv1_kernel_params.dW   = 1;
    layer4_0_conv1_kernel_params.kH   = 1;
    layer4_0_conv1_kernel_params.kW   = 1;
    layer4_0_conv1_kernel_params.padT = 0;
    layer4_0_conv1_kernel_params.padB = 0;
    layer4_0_conv1_kernel_params.padL = 0;
    layer4_0_conv1_kernel_params.padR = 0;
    layer4_0_conv1_kernel_params.dilH = 1;
    layer4_0_conv1_kernel_params.dilW = 1;

    // create layer4_0_conv1_weight tensor
    const unsigned      layer4_0_conv1_weight_sizes[] = {1, 1, 1024, 512};
    uint64_t            layer4_0_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer4_0_conv1_weight_tr_info = {"layer4_0_conv1_weight", layer4_0_conv1_weight_dram};
    synTensor           layer4_0_conv1_weight =
        createTensor(4U, syn_type_bf16, layer4_0_conv1_weight_sizes, true, "layer4_0_conv1_weight");

    synTensor layer4_0_conv1_in_vec[4] = {layer3_5_relu3_output, layer4_0_conv1_weight, nullptr, nullptr};

    // create layer4_0_conv1_output tensor
    const unsigned layer4_0_conv1_output_sizes[] = {64, 14, 14, 512};
    synTensor      layer4_0_conv1_output =
        createTensor(4U, syn_type_bf16, layer4_0_conv1_output_sizes, false, "layer4_0_conv1_output");

    synTensor layer4_0_conv1_out_vec[1] = {layer4_0_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv1_in_vec,
                           layer4_0_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer4_0_conv1_kernel_params,
                           sizeof(layer4_0_conv1_kernel_params),
                           "spatial_convolution",
                           "layer4_0_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv1 failed!");

    /*************
     * layer4_0_bn1 node
     * inputs: [layer4_0_conv1_output(64, 14, 14, 512)(dtype=bf16), layer4_0_bn1_bias[512](dtype=float32),
     *layer4_0_bn1_weight[512](dtype=float32), layer4_0_bn1_running_mean[512](dtype=float32),
     *layer4_0_bn1_running_var[512](dtype=float32)] output: [layer4_0_bn1_output(64, 14, 14, 512)(dtype=bf16),
     *layer4_0_bn1_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_0_bn1_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_0_bn1_kernel_params;
    layer4_0_bn1_kernel_params.momentum    = 0.1;
    layer4_0_bn1_kernel_params.threshold.f = 1e-05;
    layer4_0_bn1_kernel_params.epsilon     = 1e-05;

    // create layer4_0_bn1_bias tensor
    const unsigned layer4_0_bn1_bias_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer4_0_bn1_bias_tr_info = {"layer4_0_bn1_bias", layer4_0_bn1_bias_dram};
    synTensor layer4_0_bn1_bias = createTensor(1U, syn_type_single, layer4_0_bn1_bias_sizes, true, "layer4_0_bn1_bias");

    // create layer4_0_bn1_weight tensor
    const unsigned layer4_0_bn1_weight_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer4_0_bn1_weight_tr_info = {"layer4_0_bn1_weight", layer4_0_bn1_weight_dram};
    synTensor           layer4_0_bn1_weight =
        createTensor(1U, syn_type_single, layer4_0_bn1_weight_sizes, true, "layer4_0_bn1_weight");

    // create layer4_0_bn1_running_mean tensor
    const unsigned layer4_0_bn1_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_0_bn1_running_mean_tr_info = {"layer4_0_bn1_running_mean",
                                                             layer4_0_bn1_running_mean_dram};
    synTensor           layer4_0_bn1_running_mean =
        createTensor(1U, syn_type_single, layer4_0_bn1_running_mean_sizes, true, "layer4_0_bn1_running_mean");

    // create layer4_0_bn1_running_var tensor
    const unsigned layer4_0_bn1_running_var_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer4_0_bn1_running_var_tr_info = {"layer4_0_bn1_running_var", layer4_0_bn1_running_var_dram};
    synTensor           layer4_0_bn1_running_var =
        createTensor(1U, syn_type_single, layer4_0_bn1_running_var_sizes, true, "layer4_0_bn1_running_var");

    synTensor layer4_0_bn1_in_vec[5] = {layer4_0_conv1_output,
                                        layer4_0_bn1_bias,
                                        layer4_0_bn1_weight,
                                        layer4_0_bn1_running_mean,
                                        layer4_0_bn1_running_var};

    // create layer4_0_bn1_output tensor
    const unsigned layer4_0_bn1_output_sizes[] = {64, 14, 14, 512};
    synTensor      layer4_0_bn1_output =
        createTensor(4U, syn_type_bf16, layer4_0_bn1_output_sizes, false, "layer4_0_bn1_output");

    // create layer4_0_bn1_saved_mean tensor
    const unsigned layer4_0_bn1_saved_mean_sizes[] = {512};
    synTensor      layer4_0_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer4_0_bn1_saved_mean_sizes, false, "layer4_0_bn1_saved_mean");

    // create layer4_0_bn1_saved_var tensor
    const unsigned layer4_0_bn1_saved_var_sizes[] = {512};
    synTensor      layer4_0_bn1_saved_var =
        createTensor(1U, syn_type_single, layer4_0_bn1_saved_var_sizes, false, "layer4_0_bn1_saved_var");

    synTensor layer4_0_bn1_out_vec[3] = {layer4_0_bn1_output, layer4_0_bn1_saved_mean, layer4_0_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_0_bn1_in_vec,
                           layer4_0_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer4_0_bn1_kernel_params,
                           sizeof(layer4_0_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_0_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_bn1 failed!");

    /*************
     * layer4_0_relu1 node
     * inputs: [layer4_0_bn1_output(64, 14, 14, 512)(dtype=bf16)]
     * output: [layer4_0_relu1_output(64, 14, 14, 512)(dtype=bf16)]
     *************/

    synTensor layer4_0_relu1_in_vec[1] = {layer4_0_bn1_output};

    // create layer4_0_relu1_output tensor
    const unsigned layer4_0_relu1_output_sizes[] = {64, 14, 14, 512};
    synTensor      layer4_0_relu1_output =
        createTensor(4U, syn_type_bf16, layer4_0_relu1_output_sizes, false, "layer4_0_relu1_output");

    synTensor layer4_0_relu1_out_vec[1] = {layer4_0_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer4_0_relu1_in_vec,
                           layer4_0_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_0_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_relu1 failed!");

    /*************
     * layer4_0_conv2 node
     * inputs: [layer4_0_relu1_output(64, 14, 14, 512)(dtype=bf16), layer4_0_conv2_weight[3, 3, 512, 512](dtype=bf16)]
     * output: [layer4_0_conv2_output(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_0_conv2_kernel_params;
    layer4_0_conv2_kernel_params.dH   = 2;
    layer4_0_conv2_kernel_params.dW   = 2;
    layer4_0_conv2_kernel_params.kH   = 3;
    layer4_0_conv2_kernel_params.kW   = 3;
    layer4_0_conv2_kernel_params.padT = 1;
    layer4_0_conv2_kernel_params.padB = 1;
    layer4_0_conv2_kernel_params.padL = 1;
    layer4_0_conv2_kernel_params.padR = 1;
    layer4_0_conv2_kernel_params.dilH = 1;
    layer4_0_conv2_kernel_params.dilW = 1;

    // create layer4_0_conv2_weight tensor
    const unsigned      layer4_0_conv2_weight_sizes[] = {3, 3, 512, 512};
    uint64_t            layer4_0_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer4_0_conv2_weight_tr_info = {"layer4_0_conv2_weight", layer4_0_conv2_weight_dram};
    synTensor           layer4_0_conv2_weight =
        createTensor(4U, syn_type_bf16, layer4_0_conv2_weight_sizes, true, "layer4_0_conv2_weight");

    synTensor layer4_0_conv2_in_vec[4] = {layer4_0_relu1_output, layer4_0_conv2_weight, nullptr, nullptr};

    // create layer4_0_conv2_output tensor
    const unsigned layer4_0_conv2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_0_conv2_output =
        createTensor(4U, syn_type_bf16, layer4_0_conv2_output_sizes, false, "layer4_0_conv2_output");

    synTensor layer4_0_conv2_out_vec[1] = {layer4_0_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv2_in_vec,
                           layer4_0_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer4_0_conv2_kernel_params,
                           sizeof(layer4_0_conv2_kernel_params),
                           "spatial_convolution",
                           "layer4_0_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv2 failed!");

    /*************
     * layer4_0_bn2 node
     * inputs: [layer4_0_conv2_output(64, 7, 7, 512)(dtype=bf16), layer4_0_bn2_bias[512](dtype=float32),
     *layer4_0_bn2_weight[512](dtype=float32), layer4_0_bn2_running_mean[512](dtype=float32),
     *layer4_0_bn2_running_var[512](dtype=float32)] output: [layer4_0_bn2_output(64, 7, 7, 512)(dtype=bf16),
     *layer4_0_bn2_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_0_bn2_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_0_bn2_kernel_params;
    layer4_0_bn2_kernel_params.momentum    = 0.1;
    layer4_0_bn2_kernel_params.threshold.f = 1e-05;
    layer4_0_bn2_kernel_params.epsilon     = 1e-05;

    // create layer4_0_bn2_bias tensor
    const unsigned layer4_0_bn2_bias_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer4_0_bn2_bias_tr_info = {"layer4_0_bn2_bias", layer4_0_bn2_bias_dram};
    synTensor layer4_0_bn2_bias = createTensor(1U, syn_type_single, layer4_0_bn2_bias_sizes, true, "layer4_0_bn2_bias");

    // create layer4_0_bn2_weight tensor
    const unsigned layer4_0_bn2_weight_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer4_0_bn2_weight_tr_info = {"layer4_0_bn2_weight", layer4_0_bn2_weight_dram};
    synTensor           layer4_0_bn2_weight =
        createTensor(1U, syn_type_single, layer4_0_bn2_weight_sizes, true, "layer4_0_bn2_weight");

    // create layer4_0_bn2_running_mean tensor
    const unsigned layer4_0_bn2_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_0_bn2_running_mean_tr_info = {"layer4_0_bn2_running_mean",
                                                             layer4_0_bn2_running_mean_dram};
    synTensor           layer4_0_bn2_running_mean =
        createTensor(1U, syn_type_single, layer4_0_bn2_running_mean_sizes, true, "layer4_0_bn2_running_mean");

    // create layer4_0_bn2_running_var tensor
    const unsigned layer4_0_bn2_running_var_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer4_0_bn2_running_var_tr_info = {"layer4_0_bn2_running_var", layer4_0_bn2_running_var_dram};
    synTensor           layer4_0_bn2_running_var =
        createTensor(1U, syn_type_single, layer4_0_bn2_running_var_sizes, true, "layer4_0_bn2_running_var");

    synTensor layer4_0_bn2_in_vec[5] = {layer4_0_conv2_output,
                                        layer4_0_bn2_bias,
                                        layer4_0_bn2_weight,
                                        layer4_0_bn2_running_mean,
                                        layer4_0_bn2_running_var};

    // create layer4_0_bn2_output tensor
    const unsigned layer4_0_bn2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_0_bn2_output =
        createTensor(4U, syn_type_bf16, layer4_0_bn2_output_sizes, false, "layer4_0_bn2_output");

    // create layer4_0_bn2_saved_mean tensor
    const unsigned layer4_0_bn2_saved_mean_sizes[] = {512};
    synTensor      layer4_0_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer4_0_bn2_saved_mean_sizes, false, "layer4_0_bn2_saved_mean");

    // create layer4_0_bn2_saved_var tensor
    const unsigned layer4_0_bn2_saved_var_sizes[] = {512};
    synTensor      layer4_0_bn2_saved_var =
        createTensor(1U, syn_type_single, layer4_0_bn2_saved_var_sizes, false, "layer4_0_bn2_saved_var");

    synTensor layer4_0_bn2_out_vec[3] = {layer4_0_bn2_output, layer4_0_bn2_saved_mean, layer4_0_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_0_bn2_in_vec,
                           layer4_0_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer4_0_bn2_kernel_params,
                           sizeof(layer4_0_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_0_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_bn2 failed!");

    /*************
     * layer4_0_relu2 node
     * inputs: [layer4_0_bn2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_0_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_0_relu2_in_vec[1] = {layer4_0_bn2_output};

    // create layer4_0_relu2_output tensor
    const unsigned layer4_0_relu2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_0_relu2_output =
        createTensor(4U, syn_type_bf16, layer4_0_relu2_output_sizes, false, "layer4_0_relu2_output");

    synTensor layer4_0_relu2_out_vec[1] = {layer4_0_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer4_0_relu2_in_vec,
                           layer4_0_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_0_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_relu2 failed!");

    /*************
     * layer4_0_conv3 node
     * inputs: [layer4_0_relu2_output(64, 7, 7, 512)(dtype=bf16), layer4_0_conv3_weight[1, 1, 512, 2048](dtype=bf16)]
     * output: [layer4_0_conv3_output(64, 7, 7, 2048)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_0_conv3_kernel_params;
    layer4_0_conv3_kernel_params.dH   = 1;
    layer4_0_conv3_kernel_params.dW   = 1;
    layer4_0_conv3_kernel_params.kH   = 1;
    layer4_0_conv3_kernel_params.kW   = 1;
    layer4_0_conv3_kernel_params.padT = 0;
    layer4_0_conv3_kernel_params.padB = 0;
    layer4_0_conv3_kernel_params.padL = 0;
    layer4_0_conv3_kernel_params.padR = 0;
    layer4_0_conv3_kernel_params.dilH = 1;
    layer4_0_conv3_kernel_params.dilW = 1;

    // create layer4_0_conv3_weight tensor
    const unsigned      layer4_0_conv3_weight_sizes[] = {1, 1, 512, 2048};
    uint64_t            layer4_0_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer4_0_conv3_weight_tr_info = {"layer4_0_conv3_weight", layer4_0_conv3_weight_dram};
    synTensor           layer4_0_conv3_weight =
        createTensor(4U, syn_type_bf16, layer4_0_conv3_weight_sizes, true, "layer4_0_conv3_weight");

    synTensor layer4_0_conv3_in_vec[4] = {layer4_0_relu2_output, layer4_0_conv3_weight, nullptr, nullptr};

    // create layer4_0_conv3_output tensor
    const unsigned layer4_0_conv3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_conv3_output =
        createTensor(4U, syn_type_bf16, layer4_0_conv3_output_sizes, false, "layer4_0_conv3_output");

    synTensor layer4_0_conv3_out_vec[1] = {layer4_0_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv3_in_vec,
                           layer4_0_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer4_0_conv3_kernel_params,
                           sizeof(layer4_0_conv3_kernel_params),
                           "spatial_convolution",
                           "layer4_0_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv3 failed!");

    /*************
     * layer4_0_bn3 node
     * inputs: [layer4_0_conv3_output(64, 7, 7, 2048)(dtype=bf16), layer4_0_bn3_bias[2048](dtype=float32),
     *layer4_0_bn3_weight[2048](dtype=float32), layer4_0_bn3_running_mean[2048](dtype=float32),
     *layer4_0_bn3_running_var[2048](dtype=float32)] output: [layer4_0_bn3_output(64, 7, 7, 2048)(dtype=bf16),
     *layer4_0_bn3_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_0_bn3_saved_var(1, 1, 1, 2048)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_0_bn3_kernel_params;
    layer4_0_bn3_kernel_params.momentum    = 0.1;
    layer4_0_bn3_kernel_params.threshold.f = 1e-05;
    layer4_0_bn3_kernel_params.epsilon     = 1e-05;

    // create layer4_0_bn3_bias tensor
    const unsigned layer4_0_bn3_bias_sizes[] = {
        2048,
    };
    uint64_t            layer4_0_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer4_0_bn3_bias_tr_info = {"layer4_0_bn3_bias", layer4_0_bn3_bias_dram};
    synTensor layer4_0_bn3_bias = createTensor(1U, syn_type_single, layer4_0_bn3_bias_sizes, true, "layer4_0_bn3_bias");

    // create layer4_0_bn3_weight tensor
    const unsigned layer4_0_bn3_weight_sizes[] = {
        2048,
    };
    uint64_t            layer4_0_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer4_0_bn3_weight_tr_info = {"layer4_0_bn3_weight", layer4_0_bn3_weight_dram};
    synTensor           layer4_0_bn3_weight =
        createTensor(1U, syn_type_single, layer4_0_bn3_weight_sizes, true, "layer4_0_bn3_weight");

    // create layer4_0_bn3_running_mean tensor
    const unsigned layer4_0_bn3_running_mean_sizes[] = {
        2048,
    };
    uint64_t            layer4_0_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_0_bn3_running_mean_tr_info = {"layer4_0_bn3_running_mean",
                                                             layer4_0_bn3_running_mean_dram};
    synTensor           layer4_0_bn3_running_mean =
        createTensor(1U, syn_type_single, layer4_0_bn3_running_mean_sizes, true, "layer4_0_bn3_running_mean");

    // create layer4_0_bn3_running_var tensor
    const unsigned layer4_0_bn3_running_var_sizes[] = {
        2048,
    };
    uint64_t            layer4_0_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer4_0_bn3_running_var_tr_info = {"layer4_0_bn3_running_var", layer4_0_bn3_running_var_dram};
    synTensor           layer4_0_bn3_running_var =
        createTensor(1U, syn_type_single, layer4_0_bn3_running_var_sizes, true, "layer4_0_bn3_running_var");

    synTensor layer4_0_bn3_in_vec[5] = {layer4_0_conv3_output,
                                        layer4_0_bn3_bias,
                                        layer4_0_bn3_weight,
                                        layer4_0_bn3_running_mean,
                                        layer4_0_bn3_running_var};

    // create layer4_0_bn3_output tensor
    const unsigned layer4_0_bn3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_bn3_output =
        createTensor(4U, syn_type_bf16, layer4_0_bn3_output_sizes, false, "layer4_0_bn3_output");

    // create layer4_0_bn3_saved_mean tensor
    const unsigned layer4_0_bn3_saved_mean_sizes[] = {2048};
    synTensor      layer4_0_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer4_0_bn3_saved_mean_sizes, false, "layer4_0_bn3_saved_mean");

    // create layer4_0_bn3_saved_var tensor
    const unsigned layer4_0_bn3_saved_var_sizes[] = {2048};
    synTensor      layer4_0_bn3_saved_var =
        createTensor(1U, syn_type_single, layer4_0_bn3_saved_var_sizes, false, "layer4_0_bn3_saved_var");

    synTensor layer4_0_bn3_out_vec[3] = {layer4_0_bn3_output, layer4_0_bn3_saved_mean, layer4_0_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_0_bn3_in_vec,
                           layer4_0_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer4_0_bn3_kernel_params,
                           sizeof(layer4_0_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_0_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_bn3 failed!");

    /*************
     * layer4_downsample node
     * inputs: [layer3_5_relu3_output(64, 14, 14, 1024)(dtype=bf16), layer4_downsample_weight[1, 1, 1024,
     *2048](dtype=bf16)] output: [layer4_downsample_output(64, 7, 7, 2048)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_downsample_kernel_params;
    layer4_downsample_kernel_params.dH   = 2;
    layer4_downsample_kernel_params.dW   = 2;
    layer4_downsample_kernel_params.kH   = 1;
    layer4_downsample_kernel_params.kW   = 1;
    layer4_downsample_kernel_params.padT = 0;
    layer4_downsample_kernel_params.padB = 0;
    layer4_downsample_kernel_params.padL = 0;
    layer4_downsample_kernel_params.padR = 0;
    layer4_downsample_kernel_params.dilH = 1;
    layer4_downsample_kernel_params.dilW = 1;

    // create layer4_downsample_weight tensor
    const unsigned      layer4_downsample_weight_sizes[] = {1, 1, 1024, 2048};
    uint64_t            layer4_downsample_weight_dram    = 0;
    synLaunchTensorInfo layer4_downsample_weight_tr_info = {"layer4_downsample_weight", layer4_downsample_weight_dram};
    synTensor           layer4_downsample_weight =
        createTensor(4U, syn_type_bf16, layer4_downsample_weight_sizes, true, "layer4_downsample_weight");

    synTensor layer4_downsample_in_vec[4] = {layer3_5_relu3_output, layer4_downsample_weight, nullptr, nullptr};

    // create layer4_downsample_output tensor
    const unsigned layer4_downsample_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_downsample_output =
        createTensor(4U, syn_type_bf16, layer4_downsample_output_sizes, false, "layer4_downsample_output");

    synTensor layer4_downsample_out_vec[1] = {layer4_downsample_output};

    status = synNodeCreate(graphHandle,
                           layer4_downsample_in_vec,
                           layer4_downsample_out_vec,
                           4,
                           1,
                           (void*)&layer4_downsample_kernel_params,
                           sizeof(layer4_downsample_kernel_params),
                           "spatial_convolution",
                           "layer4_downsample",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample failed!");

    /*************
     * layer4_bn node
     * inputs: [layer4_downsample_output(64, 7, 7, 2048)(dtype=bf16), layer4_bn_bias[2048](dtype=float32),
     *layer4_bn_weight[2048](dtype=float32), layer4_bn_running_mean[2048](dtype=float32),
     *layer4_bn_running_var[2048](dtype=float32)] output: [layer4_bn_output(64, 7, 7, 2048)(dtype=bf16),
     *layer4_bn_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_bn_saved_var(1, 1, 1, 2048)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_bn_kernel_params;
    layer4_bn_kernel_params.momentum    = 0.1;
    layer4_bn_kernel_params.threshold.f = 1e-05;
    layer4_bn_kernel_params.epsilon     = 1e-05;

    // create layer4_bn_bias tensor
    const unsigned layer4_bn_bias_sizes[] = {
        2048,
    };
    uint64_t            layer4_bn_bias_dram    = 0;
    synLaunchTensorInfo layer4_bn_bias_tr_info = {"layer4_bn_bias", layer4_bn_bias_dram};
    synTensor layer4_bn_bias = createTensor(1U, syn_type_single, layer4_bn_bias_sizes, true, "layer4_bn_bias");

    // create layer4_bn_weight tensor
    const unsigned layer4_bn_weight_sizes[] = {
        2048,
    };
    uint64_t            layer4_bn_weight_dram    = 0;
    synLaunchTensorInfo layer4_bn_weight_tr_info = {"layer4_bn_weight", layer4_bn_weight_dram};
    synTensor layer4_bn_weight = createTensor(1U, syn_type_single, layer4_bn_weight_sizes, true, "layer4_bn_weight");

    // create layer4_bn_running_mean tensor
    const unsigned layer4_bn_running_mean_sizes[] = {
        2048,
    };
    uint64_t            layer4_bn_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_bn_running_mean_tr_info = {"layer4_bn_running_mean", layer4_bn_running_mean_dram};
    synTensor           layer4_bn_running_mean =
        createTensor(1U, syn_type_single, layer4_bn_running_mean_sizes, true, "layer4_bn_running_mean");

    // create layer4_bn_running_var tensor
    const unsigned layer4_bn_running_var_sizes[] = {
        2048,
    };
    uint64_t            layer4_bn_running_var_dram    = 0;
    synLaunchTensorInfo layer4_bn_running_var_tr_info = {"layer4_bn_running_var", layer4_bn_running_var_dram};
    synTensor           layer4_bn_running_var =
        createTensor(1U, syn_type_single, layer4_bn_running_var_sizes, true, "layer4_bn_running_var");

    synTensor layer4_bn_in_vec[5] = {layer4_downsample_output,
                                     layer4_bn_bias,
                                     layer4_bn_weight,
                                     layer4_bn_running_mean,
                                     layer4_bn_running_var};

    // create layer4_bn_output tensor
    const unsigned layer4_bn_output_sizes[] = {64, 7, 7, 2048};
    synTensor layer4_bn_output = createTensor(4U, syn_type_bf16, layer4_bn_output_sizes, false, "layer4_bn_output");

    // create layer4_bn_saved_mean tensor
    const unsigned layer4_bn_saved_mean_sizes[] = {2048};
    synTensor      layer4_bn_saved_mean =
        createTensor(1U, syn_type_single, layer4_bn_saved_mean_sizes, false, "layer4_bn_saved_mean");

    // create layer4_bn_saved_var tensor
    const unsigned layer4_bn_saved_var_sizes[] = {2048};
    synTensor      layer4_bn_saved_var =
        createTensor(1U, syn_type_single, layer4_bn_saved_var_sizes, false, "layer4_bn_saved_var");

    synTensor layer4_bn_out_vec[3] = {layer4_bn_output, layer4_bn_saved_mean, layer4_bn_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_bn_in_vec,
                           layer4_bn_out_vec,
                           5,
                           3,
                           (void*)&layer4_bn_kernel_params,
                           sizeof(layer4_bn_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_bn",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_bn failed!");

    /*************
     * layer4_0_add_residual_fwd0 node
     * inputs: [layer4_0_bn3_output(64, 7, 7, 2048)(dtype=bf16), layer4_bn_output(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_0_add_residual_fwd(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_0_add_residual_fwd0_in_vec[2] = {layer4_0_bn3_output, layer4_bn_output};

    // create layer4_0_add_residual_fwd tensor
    const unsigned layer4_0_add_residual_fwd_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer4_0_add_residual_fwd_sizes, false, "layer4_0_add_residual_fwd");

    synTensor layer4_0_add_residual_fwd0_out_vec[1] = {layer4_0_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer4_0_add_residual_fwd0_in_vec,
                           layer4_0_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer4_0_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_add_residual_fwd0 failed!");

    /*************
     * layer4_0_relu3 node
     * inputs: [layer4_0_add_residual_fwd(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_0_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_0_relu3_in_vec[1] = {layer4_0_add_residual_fwd};

    // create layer4_0_relu3_output tensor
    const unsigned layer4_0_relu3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_relu3_output =
        createTensor(4U, syn_type_bf16, layer4_0_relu3_output_sizes, false, "layer4_0_relu3_output");

    synTensor layer4_0_relu3_out_vec[1] = {layer4_0_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer4_0_relu3_in_vec,
                           layer4_0_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_0_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_relu3 failed!");

    /*************
     * layer4_1_conv1 node
     * inputs: [layer4_0_relu3_output(64, 7, 7, 2048)(dtype=bf16), layer4_1_conv1_weight[1, 1, 2048, 512](dtype=bf16)]
     * output: [layer4_1_conv1_output(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_1_conv1_kernel_params;
    layer4_1_conv1_kernel_params.dH   = 1;
    layer4_1_conv1_kernel_params.dW   = 1;
    layer4_1_conv1_kernel_params.kH   = 1;
    layer4_1_conv1_kernel_params.kW   = 1;
    layer4_1_conv1_kernel_params.padT = 0;
    layer4_1_conv1_kernel_params.padB = 0;
    layer4_1_conv1_kernel_params.padL = 0;
    layer4_1_conv1_kernel_params.padR = 0;
    layer4_1_conv1_kernel_params.dilH = 1;
    layer4_1_conv1_kernel_params.dilW = 1;

    // create layer4_1_conv1_weight tensor
    const unsigned      layer4_1_conv1_weight_sizes[] = {1, 1, 2048, 512};
    uint64_t            layer4_1_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer4_1_conv1_weight_tr_info = {"layer4_1_conv1_weight", layer4_1_conv1_weight_dram};
    synTensor           layer4_1_conv1_weight =
        createTensor(4U, syn_type_bf16, layer4_1_conv1_weight_sizes, true, "layer4_1_conv1_weight");

    synTensor layer4_1_conv1_in_vec[4] = {layer4_0_relu3_output, layer4_1_conv1_weight, nullptr, nullptr};

    // create layer4_1_conv1_output tensor
    const unsigned layer4_1_conv1_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_conv1_output =
        createTensor(4U, syn_type_bf16, layer4_1_conv1_output_sizes, false, "layer4_1_conv1_output");

    synTensor layer4_1_conv1_out_vec[1] = {layer4_1_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv1_in_vec,
                           layer4_1_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer4_1_conv1_kernel_params,
                           sizeof(layer4_1_conv1_kernel_params),
                           "spatial_convolution",
                           "layer4_1_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv1 failed!");

    /*************
     * layer4_1_bn1 node
     * inputs: [layer4_1_conv1_output(64, 7, 7, 512)(dtype=bf16), layer4_1_bn1_bias[512](dtype=float32),
     *layer4_1_bn1_weight[512](dtype=float32), layer4_1_bn1_running_mean[512](dtype=float32),
     *layer4_1_bn1_running_var[512](dtype=float32)] output: [layer4_1_bn1_output(64, 7, 7, 512)(dtype=bf16),
     *layer4_1_bn1_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_1_bn1_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_1_bn1_kernel_params;
    layer4_1_bn1_kernel_params.momentum    = 0.1;
    layer4_1_bn1_kernel_params.threshold.f = 1e-05;
    layer4_1_bn1_kernel_params.epsilon     = 1e-05;

    // create layer4_1_bn1_bias tensor
    const unsigned layer4_1_bn1_bias_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer4_1_bn1_bias_tr_info = {"layer4_1_bn1_bias", layer4_1_bn1_bias_dram};
    synTensor layer4_1_bn1_bias = createTensor(1U, syn_type_single, layer4_1_bn1_bias_sizes, true, "layer4_1_bn1_bias");

    // create layer4_1_bn1_weight tensor
    const unsigned layer4_1_bn1_weight_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer4_1_bn1_weight_tr_info = {"layer4_1_bn1_weight", layer4_1_bn1_weight_dram};
    synTensor           layer4_1_bn1_weight =
        createTensor(1U, syn_type_single, layer4_1_bn1_weight_sizes, true, "layer4_1_bn1_weight");

    // create layer4_1_bn1_running_mean tensor
    const unsigned layer4_1_bn1_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_1_bn1_running_mean_tr_info = {"layer4_1_bn1_running_mean",
                                                             layer4_1_bn1_running_mean_dram};
    synTensor           layer4_1_bn1_running_mean =
        createTensor(1U, syn_type_single, layer4_1_bn1_running_mean_sizes, true, "layer4_1_bn1_running_mean");

    // create layer4_1_bn1_running_var tensor
    const unsigned layer4_1_bn1_running_var_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer4_1_bn1_running_var_tr_info = {"layer4_1_bn1_running_var", layer4_1_bn1_running_var_dram};
    synTensor           layer4_1_bn1_running_var =
        createTensor(1U, syn_type_single, layer4_1_bn1_running_var_sizes, true, "layer4_1_bn1_running_var");

    synTensor layer4_1_bn1_in_vec[5] = {layer4_1_conv1_output,
                                        layer4_1_bn1_bias,
                                        layer4_1_bn1_weight,
                                        layer4_1_bn1_running_mean,
                                        layer4_1_bn1_running_var};

    // create layer4_1_bn1_output tensor
    const unsigned layer4_1_bn1_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_bn1_output =
        createTensor(4U, syn_type_bf16, layer4_1_bn1_output_sizes, false, "layer4_1_bn1_output");

    // create layer4_1_bn1_saved_mean tensor
    const unsigned layer4_1_bn1_saved_mean_sizes[] = {512};
    synTensor      layer4_1_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer4_1_bn1_saved_mean_sizes, false, "layer4_1_bn1_saved_mean");

    // create layer4_1_bn1_saved_var tensor
    const unsigned layer4_1_bn1_saved_var_sizes[] = {512};
    synTensor      layer4_1_bn1_saved_var =
        createTensor(1U, syn_type_single, layer4_1_bn1_saved_var_sizes, false, "layer4_1_bn1_saved_var");

    synTensor layer4_1_bn1_out_vec[3] = {layer4_1_bn1_output, layer4_1_bn1_saved_mean, layer4_1_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_1_bn1_in_vec,
                           layer4_1_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer4_1_bn1_kernel_params,
                           sizeof(layer4_1_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_1_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_bn1 failed!");

    /*************
     * layer4_1_relu1 node
     * inputs: [layer4_1_bn1_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_1_relu1_output(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_1_relu1_in_vec[1] = {layer4_1_bn1_output};

    // create layer4_1_relu1_output tensor
    const unsigned layer4_1_relu1_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_relu1_output =
        createTensor(4U, syn_type_bf16, layer4_1_relu1_output_sizes, false, "layer4_1_relu1_output");

    synTensor layer4_1_relu1_out_vec[1] = {layer4_1_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer4_1_relu1_in_vec,
                           layer4_1_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_1_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_relu1 failed!");

    /*************
     * layer4_1_conv2 node
     * inputs: [layer4_1_relu1_output(64, 7, 7, 512)(dtype=bf16), layer4_1_conv2_weight[3, 3, 512, 512](dtype=bf16)]
     * output: [layer4_1_conv2_output(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_1_conv2_kernel_params;
    layer4_1_conv2_kernel_params.dH   = 1;
    layer4_1_conv2_kernel_params.dW   = 1;
    layer4_1_conv2_kernel_params.kH   = 3;
    layer4_1_conv2_kernel_params.kW   = 3;
    layer4_1_conv2_kernel_params.padT = 1;
    layer4_1_conv2_kernel_params.padB = 1;
    layer4_1_conv2_kernel_params.padL = 1;
    layer4_1_conv2_kernel_params.padR = 1;
    layer4_1_conv2_kernel_params.dilH = 1;
    layer4_1_conv2_kernel_params.dilW = 1;

    // create layer4_1_conv2_weight tensor
    const unsigned      layer4_1_conv2_weight_sizes[] = {3, 3, 512, 512};
    uint64_t            layer4_1_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer4_1_conv2_weight_tr_info = {"layer4_1_conv2_weight", layer4_1_conv2_weight_dram};
    synTensor           layer4_1_conv2_weight =
        createTensor(4U, syn_type_bf16, layer4_1_conv2_weight_sizes, true, "layer4_1_conv2_weight");

    synTensor layer4_1_conv2_in_vec[4] = {layer4_1_relu1_output, layer4_1_conv2_weight, nullptr, nullptr};

    // create layer4_1_conv2_output tensor
    const unsigned layer4_1_conv2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_conv2_output =
        createTensor(4U, syn_type_bf16, layer4_1_conv2_output_sizes, false, "layer4_1_conv2_output");

    synTensor layer4_1_conv2_out_vec[1] = {layer4_1_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv2_in_vec,
                           layer4_1_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer4_1_conv2_kernel_params,
                           sizeof(layer4_1_conv2_kernel_params),
                           "spatial_convolution",
                           "layer4_1_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv2 failed!");

    /*************
     * layer4_1_bn2 node
     * inputs: [layer4_1_conv2_output(64, 7, 7, 512)(dtype=bf16), layer4_1_bn2_bias[512](dtype=float32),
     *layer4_1_bn2_weight[512](dtype=float32), layer4_1_bn2_running_mean[512](dtype=float32),
     *layer4_1_bn2_running_var[512](dtype=float32)] output: [layer4_1_bn2_output(64, 7, 7, 512)(dtype=bf16),
     *layer4_1_bn2_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_1_bn2_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_1_bn2_kernel_params;
    layer4_1_bn2_kernel_params.momentum    = 0.1;
    layer4_1_bn2_kernel_params.threshold.f = 1e-05;
    layer4_1_bn2_kernel_params.epsilon     = 1e-05;

    // create layer4_1_bn2_bias tensor
    const unsigned layer4_1_bn2_bias_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer4_1_bn2_bias_tr_info = {"layer4_1_bn2_bias", layer4_1_bn2_bias_dram};
    synTensor layer4_1_bn2_bias = createTensor(1U, syn_type_single, layer4_1_bn2_bias_sizes, true, "layer4_1_bn2_bias");

    // create layer4_1_bn2_weight tensor
    const unsigned layer4_1_bn2_weight_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer4_1_bn2_weight_tr_info = {"layer4_1_bn2_weight", layer4_1_bn2_weight_dram};
    synTensor           layer4_1_bn2_weight =
        createTensor(1U, syn_type_single, layer4_1_bn2_weight_sizes, true, "layer4_1_bn2_weight");

    // create layer4_1_bn2_running_mean tensor
    const unsigned layer4_1_bn2_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_1_bn2_running_mean_tr_info = {"layer4_1_bn2_running_mean",
                                                             layer4_1_bn2_running_mean_dram};
    synTensor           layer4_1_bn2_running_mean =
        createTensor(1U, syn_type_single, layer4_1_bn2_running_mean_sizes, true, "layer4_1_bn2_running_mean");

    // create layer4_1_bn2_running_var tensor
    const unsigned layer4_1_bn2_running_var_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer4_1_bn2_running_var_tr_info = {"layer4_1_bn2_running_var", layer4_1_bn2_running_var_dram};
    synTensor           layer4_1_bn2_running_var =
        createTensor(1U, syn_type_single, layer4_1_bn2_running_var_sizes, true, "layer4_1_bn2_running_var");

    synTensor layer4_1_bn2_in_vec[5] = {layer4_1_conv2_output,
                                        layer4_1_bn2_bias,
                                        layer4_1_bn2_weight,
                                        layer4_1_bn2_running_mean,
                                        layer4_1_bn2_running_var};

    // create layer4_1_bn2_output tensor
    const unsigned layer4_1_bn2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_bn2_output =
        createTensor(4U, syn_type_bf16, layer4_1_bn2_output_sizes, false, "layer4_1_bn2_output");

    // create layer4_1_bn2_saved_mean tensor
    const unsigned layer4_1_bn2_saved_mean_sizes[] = {512};
    synTensor      layer4_1_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer4_1_bn2_saved_mean_sizes, false, "layer4_1_bn2_saved_mean");

    // create layer4_1_bn2_saved_var tensor
    const unsigned layer4_1_bn2_saved_var_sizes[] = {512};
    synTensor      layer4_1_bn2_saved_var =
        createTensor(1U, syn_type_single, layer4_1_bn2_saved_var_sizes, false, "layer4_1_bn2_saved_var");

    synTensor layer4_1_bn2_out_vec[3] = {layer4_1_bn2_output, layer4_1_bn2_saved_mean, layer4_1_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_1_bn2_in_vec,
                           layer4_1_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer4_1_bn2_kernel_params,
                           sizeof(layer4_1_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_1_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_bn2 failed!");

    /*************
     * layer4_1_relu2 node
     * inputs: [layer4_1_bn2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_1_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_1_relu2_in_vec[1] = {layer4_1_bn2_output};

    // create layer4_1_relu2_output tensor
    const unsigned layer4_1_relu2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_relu2_output =
        createTensor(4U, syn_type_bf16, layer4_1_relu2_output_sizes, false, "layer4_1_relu2_output");

    synTensor layer4_1_relu2_out_vec[1] = {layer4_1_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer4_1_relu2_in_vec,
                           layer4_1_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_1_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_relu2 failed!");

    /*************
     * layer4_1_conv3 node
     * inputs: [layer4_1_relu2_output(64, 7, 7, 512)(dtype=bf16), layer4_1_conv3_weight[1, 1, 512, 2048](dtype=bf16)]
     * output: [layer4_1_conv3_output(64, 7, 7, 2048)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_1_conv3_kernel_params;
    layer4_1_conv3_kernel_params.dH   = 1;
    layer4_1_conv3_kernel_params.dW   = 1;
    layer4_1_conv3_kernel_params.kH   = 1;
    layer4_1_conv3_kernel_params.kW   = 1;
    layer4_1_conv3_kernel_params.padT = 0;
    layer4_1_conv3_kernel_params.padB = 0;
    layer4_1_conv3_kernel_params.padL = 0;
    layer4_1_conv3_kernel_params.padR = 0;
    layer4_1_conv3_kernel_params.dilH = 1;
    layer4_1_conv3_kernel_params.dilW = 1;

    // create layer4_1_conv3_weight tensor
    const unsigned      layer4_1_conv3_weight_sizes[] = {1, 1, 512, 2048};
    uint64_t            layer4_1_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer4_1_conv3_weight_tr_info = {"layer4_1_conv3_weight", layer4_1_conv3_weight_dram};
    synTensor           layer4_1_conv3_weight =
        createTensor(4U, syn_type_bf16, layer4_1_conv3_weight_sizes, true, "layer4_1_conv3_weight");

    synTensor layer4_1_conv3_in_vec[4] = {layer4_1_relu2_output, layer4_1_conv3_weight, nullptr, nullptr};

    // create layer4_1_conv3_output tensor
    const unsigned layer4_1_conv3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_conv3_output =
        createTensor(4U, syn_type_bf16, layer4_1_conv3_output_sizes, false, "layer4_1_conv3_output");

    synTensor layer4_1_conv3_out_vec[1] = {layer4_1_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv3_in_vec,
                           layer4_1_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer4_1_conv3_kernel_params,
                           sizeof(layer4_1_conv3_kernel_params),
                           "spatial_convolution",
                           "layer4_1_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv3 failed!");

    /*************
     * layer4_1_bn3 node
     * inputs: [layer4_1_conv3_output(64, 7, 7, 2048)(dtype=bf16), layer4_1_bn3_bias[2048](dtype=float32),
     *layer4_1_bn3_weight[2048](dtype=float32), layer4_1_bn3_running_mean[2048](dtype=float32),
     *layer4_1_bn3_running_var[2048](dtype=float32)] output: [layer4_1_bn3_output(64, 7, 7, 2048)(dtype=bf16),
     *layer4_1_bn3_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_1_bn3_saved_var(1, 1, 1, 2048)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_1_bn3_kernel_params;
    layer4_1_bn3_kernel_params.momentum    = 0.1;
    layer4_1_bn3_kernel_params.threshold.f = 1e-05;
    layer4_1_bn3_kernel_params.epsilon     = 1e-05;

    // create layer4_1_bn3_bias tensor
    const unsigned layer4_1_bn3_bias_sizes[] = {
        2048,
    };
    uint64_t            layer4_1_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer4_1_bn3_bias_tr_info = {"layer4_1_bn3_bias", layer4_1_bn3_bias_dram};
    synTensor layer4_1_bn3_bias = createTensor(1U, syn_type_single, layer4_1_bn3_bias_sizes, true, "layer4_1_bn3_bias");

    // create layer4_1_bn3_weight tensor
    const unsigned layer4_1_bn3_weight_sizes[] = {
        2048,
    };
    uint64_t            layer4_1_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer4_1_bn3_weight_tr_info = {"layer4_1_bn3_weight", layer4_1_bn3_weight_dram};
    synTensor           layer4_1_bn3_weight =
        createTensor(1U, syn_type_single, layer4_1_bn3_weight_sizes, true, "layer4_1_bn3_weight");

    // create layer4_1_bn3_running_mean tensor
    const unsigned layer4_1_bn3_running_mean_sizes[] = {
        2048,
    };
    uint64_t            layer4_1_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_1_bn3_running_mean_tr_info = {"layer4_1_bn3_running_mean",
                                                             layer4_1_bn3_running_mean_dram};
    synTensor           layer4_1_bn3_running_mean =
        createTensor(1U, syn_type_single, layer4_1_bn3_running_mean_sizes, true, "layer4_1_bn3_running_mean");

    // create layer4_1_bn3_running_var tensor
    const unsigned layer4_1_bn3_running_var_sizes[] = {
        2048,
    };
    uint64_t            layer4_1_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer4_1_bn3_running_var_tr_info = {"layer4_1_bn3_running_var", layer4_1_bn3_running_var_dram};
    synTensor           layer4_1_bn3_running_var =
        createTensor(1U, syn_type_single, layer4_1_bn3_running_var_sizes, true, "layer4_1_bn3_running_var");

    synTensor layer4_1_bn3_in_vec[5] = {layer4_1_conv3_output,
                                        layer4_1_bn3_bias,
                                        layer4_1_bn3_weight,
                                        layer4_1_bn3_running_mean,
                                        layer4_1_bn3_running_var};

    // create layer4_1_bn3_output tensor
    const unsigned layer4_1_bn3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_bn3_output =
        createTensor(4U, syn_type_bf16, layer4_1_bn3_output_sizes, false, "layer4_1_bn3_output");

    // create layer4_1_bn3_saved_mean tensor
    const unsigned layer4_1_bn3_saved_mean_sizes[] = {2048};
    synTensor      layer4_1_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer4_1_bn3_saved_mean_sizes, false, "layer4_1_bn3_saved_mean");

    // create layer4_1_bn3_saved_var tensor
    const unsigned layer4_1_bn3_saved_var_sizes[] = {2048};
    synTensor      layer4_1_bn3_saved_var =
        createTensor(1U, syn_type_single, layer4_1_bn3_saved_var_sizes, false, "layer4_1_bn3_saved_var");

    synTensor layer4_1_bn3_out_vec[3] = {layer4_1_bn3_output, layer4_1_bn3_saved_mean, layer4_1_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_1_bn3_in_vec,
                           layer4_1_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer4_1_bn3_kernel_params,
                           sizeof(layer4_1_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_1_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_bn3 failed!");

    /*************
     * layer4_1_add_residual_fwd0 node
     * inputs: [layer4_1_bn3_output(64, 7, 7, 2048)(dtype=bf16), layer4_0_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_1_add_residual_fwd(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_1_add_residual_fwd0_in_vec[2] = {layer4_1_bn3_output, layer4_0_relu3_output};

    // create layer4_1_add_residual_fwd tensor
    const unsigned layer4_1_add_residual_fwd_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer4_1_add_residual_fwd_sizes, false, "layer4_1_add_residual_fwd");

    synTensor layer4_1_add_residual_fwd0_out_vec[1] = {layer4_1_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer4_1_add_residual_fwd0_in_vec,
                           layer4_1_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer4_1_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_add_residual_fwd0 failed!");

    /*************
     * layer4_1_relu3 node
     * inputs: [layer4_1_add_residual_fwd(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_1_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_1_relu3_in_vec[1] = {layer4_1_add_residual_fwd};

    // create layer4_1_relu3_output tensor
    const unsigned layer4_1_relu3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_relu3_output =
        createTensor(4U, syn_type_bf16, layer4_1_relu3_output_sizes, false, "layer4_1_relu3_output");

    synTensor layer4_1_relu3_out_vec[1] = {layer4_1_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer4_1_relu3_in_vec,
                           layer4_1_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_1_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_relu3 failed!");

    /*************
     * layer4_2_conv1 node
     * inputs: [layer4_1_relu3_output(64, 7, 7, 2048)(dtype=bf16), layer4_2_conv1_weight[1, 1, 2048, 512](dtype=bf16)]
     * output: [layer4_2_conv1_output(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_2_conv1_kernel_params;
    layer4_2_conv1_kernel_params.dH   = 1;
    layer4_2_conv1_kernel_params.dW   = 1;
    layer4_2_conv1_kernel_params.kH   = 1;
    layer4_2_conv1_kernel_params.kW   = 1;
    layer4_2_conv1_kernel_params.padT = 0;
    layer4_2_conv1_kernel_params.padB = 0;
    layer4_2_conv1_kernel_params.padL = 0;
    layer4_2_conv1_kernel_params.padR = 0;
    layer4_2_conv1_kernel_params.dilH = 1;
    layer4_2_conv1_kernel_params.dilW = 1;

    // create layer4_2_conv1_weight tensor
    const unsigned      layer4_2_conv1_weight_sizes[] = {1, 1, 2048, 512};
    uint64_t            layer4_2_conv1_weight_dram    = 0;
    synLaunchTensorInfo layer4_2_conv1_weight_tr_info = {"layer4_2_conv1_weight", layer4_2_conv1_weight_dram};
    synTensor           layer4_2_conv1_weight =
        createTensor(4U, syn_type_bf16, layer4_2_conv1_weight_sizes, true, "layer4_2_conv1_weight");

    synTensor layer4_2_conv1_in_vec[4] = {layer4_1_relu3_output, layer4_2_conv1_weight, nullptr, nullptr};

    // create layer4_2_conv1_output tensor
    const unsigned layer4_2_conv1_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_conv1_output =
        createTensor(4U, syn_type_bf16, layer4_2_conv1_output_sizes, false, "layer4_2_conv1_output");

    synTensor layer4_2_conv1_out_vec[1] = {layer4_2_conv1_output};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv1_in_vec,
                           layer4_2_conv1_out_vec,
                           4,
                           1,
                           (void*)&layer4_2_conv1_kernel_params,
                           sizeof(layer4_2_conv1_kernel_params),
                           "spatial_convolution",
                           "layer4_2_conv1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv1 failed!");

    /*************
     * layer4_2_bn1 node
     * inputs: [layer4_2_conv1_output(64, 7, 7, 512)(dtype=bf16), layer4_2_bn1_bias[512](dtype=float32),
     *layer4_2_bn1_weight[512](dtype=float32), layer4_2_bn1_running_mean[512](dtype=float32),
     *layer4_2_bn1_running_var[512](dtype=float32)] output: [layer4_2_bn1_output(64, 7, 7, 512)(dtype=bf16),
     *layer4_2_bn1_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_2_bn1_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_2_bn1_kernel_params;
    layer4_2_bn1_kernel_params.momentum    = 0.1;
    layer4_2_bn1_kernel_params.threshold.f = 1e-05;
    layer4_2_bn1_kernel_params.epsilon     = 1e-05;

    // create layer4_2_bn1_bias tensor
    const unsigned layer4_2_bn1_bias_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn1_bias_dram    = 0;
    synLaunchTensorInfo layer4_2_bn1_bias_tr_info = {"layer4_2_bn1_bias", layer4_2_bn1_bias_dram};
    synTensor layer4_2_bn1_bias = createTensor(1U, syn_type_single, layer4_2_bn1_bias_sizes, true, "layer4_2_bn1_bias");

    // create layer4_2_bn1_weight tensor
    const unsigned layer4_2_bn1_weight_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn1_weight_dram    = 0;
    synLaunchTensorInfo layer4_2_bn1_weight_tr_info = {"layer4_2_bn1_weight", layer4_2_bn1_weight_dram};
    synTensor           layer4_2_bn1_weight =
        createTensor(1U, syn_type_single, layer4_2_bn1_weight_sizes, true, "layer4_2_bn1_weight");

    // create layer4_2_bn1_running_mean tensor
    const unsigned layer4_2_bn1_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn1_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_2_bn1_running_mean_tr_info = {"layer4_2_bn1_running_mean",
                                                             layer4_2_bn1_running_mean_dram};
    synTensor           layer4_2_bn1_running_mean =
        createTensor(1U, syn_type_single, layer4_2_bn1_running_mean_sizes, true, "layer4_2_bn1_running_mean");

    // create layer4_2_bn1_running_var tensor
    const unsigned layer4_2_bn1_running_var_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn1_running_var_dram    = 0;
    synLaunchTensorInfo layer4_2_bn1_running_var_tr_info = {"layer4_2_bn1_running_var", layer4_2_bn1_running_var_dram};
    synTensor           layer4_2_bn1_running_var =
        createTensor(1U, syn_type_single, layer4_2_bn1_running_var_sizes, true, "layer4_2_bn1_running_var");

    synTensor layer4_2_bn1_in_vec[5] = {layer4_2_conv1_output,
                                        layer4_2_bn1_bias,
                                        layer4_2_bn1_weight,
                                        layer4_2_bn1_running_mean,
                                        layer4_2_bn1_running_var};

    // create layer4_2_bn1_output tensor
    const unsigned layer4_2_bn1_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_bn1_output =
        createTensor(4U, syn_type_bf16, layer4_2_bn1_output_sizes, false, "layer4_2_bn1_output");

    // create layer4_2_bn1_saved_mean tensor
    const unsigned layer4_2_bn1_saved_mean_sizes[] = {512};
    synTensor      layer4_2_bn1_saved_mean =
        createTensor(1U, syn_type_single, layer4_2_bn1_saved_mean_sizes, false, "layer4_2_bn1_saved_mean");

    // create layer4_2_bn1_saved_var tensor
    const unsigned layer4_2_bn1_saved_var_sizes[] = {512};
    synTensor      layer4_2_bn1_saved_var =
        createTensor(1U, syn_type_single, layer4_2_bn1_saved_var_sizes, false, "layer4_2_bn1_saved_var");

    synTensor layer4_2_bn1_out_vec[3] = {layer4_2_bn1_output, layer4_2_bn1_saved_mean, layer4_2_bn1_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_2_bn1_in_vec,
                           layer4_2_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer4_2_bn1_kernel_params,
                           sizeof(layer4_2_bn1_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_2_bn1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn1 failed!");

    /*************
     * layer4_2_relu1 node
     * inputs: [layer4_2_bn1_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_2_relu1_output(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_2_relu1_in_vec[1] = {layer4_2_bn1_output};

    // create layer4_2_relu1_output tensor
    const unsigned layer4_2_relu1_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_relu1_output =
        createTensor(4U, syn_type_bf16, layer4_2_relu1_output_sizes, false, "layer4_2_relu1_output");

    synTensor layer4_2_relu1_out_vec[1] = {layer4_2_relu1_output};

    status = synNodeCreate(graphHandle,
                           layer4_2_relu1_in_vec,
                           layer4_2_relu1_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_2_relu1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu1 failed!");

    /*************
     * layer4_2_conv2 node
     * inputs: [layer4_2_relu1_output(64, 7, 7, 512)(dtype=bf16), layer4_2_conv2_weight[3, 3, 512, 512](dtype=bf16)]
     * output: [layer4_2_conv2_output(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_2_conv2_kernel_params;
    layer4_2_conv2_kernel_params.dH   = 1;
    layer4_2_conv2_kernel_params.dW   = 1;
    layer4_2_conv2_kernel_params.kH   = 3;
    layer4_2_conv2_kernel_params.kW   = 3;
    layer4_2_conv2_kernel_params.padT = 1;
    layer4_2_conv2_kernel_params.padB = 1;
    layer4_2_conv2_kernel_params.padL = 1;
    layer4_2_conv2_kernel_params.padR = 1;
    layer4_2_conv2_kernel_params.dilH = 1;
    layer4_2_conv2_kernel_params.dilW = 1;

    // create layer4_2_conv2_weight tensor
    const unsigned      layer4_2_conv2_weight_sizes[] = {3, 3, 512, 512};
    uint64_t            layer4_2_conv2_weight_dram    = 0;
    synLaunchTensorInfo layer4_2_conv2_weight_tr_info = {"layer4_2_conv2_weight", layer4_2_conv2_weight_dram};
    synTensor           layer4_2_conv2_weight =
        createTensor(4U, syn_type_bf16, layer4_2_conv2_weight_sizes, true, "layer4_2_conv2_weight");

    synTensor layer4_2_conv2_in_vec[4] = {layer4_2_relu1_output, layer4_2_conv2_weight, nullptr, nullptr};

    // create layer4_2_conv2_output tensor
    const unsigned layer4_2_conv2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_conv2_output =
        createTensor(4U, syn_type_bf16, layer4_2_conv2_output_sizes, false, "layer4_2_conv2_output");

    synTensor layer4_2_conv2_out_vec[1] = {layer4_2_conv2_output};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv2_in_vec,
                           layer4_2_conv2_out_vec,
                           4,
                           1,
                           (void*)&layer4_2_conv2_kernel_params,
                           sizeof(layer4_2_conv2_kernel_params),
                           "spatial_convolution",
                           "layer4_2_conv2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv2 failed!");

    /*************
     * layer4_2_bn2 node
     * inputs: [layer4_2_conv2_output(64, 7, 7, 512)(dtype=bf16), layer4_2_bn2_bias[512](dtype=float32),
     *layer4_2_bn2_weight[512](dtype=float32), layer4_2_bn2_running_mean[512](dtype=float32),
     *layer4_2_bn2_running_var[512](dtype=float32)] output: [layer4_2_bn2_output(64, 7, 7, 512)(dtype=bf16),
     *layer4_2_bn2_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_2_bn2_saved_var(1, 1, 1, 512)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_2_bn2_kernel_params;
    layer4_2_bn2_kernel_params.momentum    = 0.1;
    layer4_2_bn2_kernel_params.threshold.f = 1e-05;
    layer4_2_bn2_kernel_params.epsilon     = 1e-05;

    // create layer4_2_bn2_bias tensor
    const unsigned layer4_2_bn2_bias_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn2_bias_dram    = 0;
    synLaunchTensorInfo layer4_2_bn2_bias_tr_info = {"layer4_2_bn2_bias", layer4_2_bn2_bias_dram};
    synTensor layer4_2_bn2_bias = createTensor(1U, syn_type_single, layer4_2_bn2_bias_sizes, true, "layer4_2_bn2_bias");

    // create layer4_2_bn2_weight tensor
    const unsigned layer4_2_bn2_weight_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn2_weight_dram    = 0;
    synLaunchTensorInfo layer4_2_bn2_weight_tr_info = {"layer4_2_bn2_weight", layer4_2_bn2_weight_dram};
    synTensor           layer4_2_bn2_weight =
        createTensor(1U, syn_type_single, layer4_2_bn2_weight_sizes, true, "layer4_2_bn2_weight");

    // create layer4_2_bn2_running_mean tensor
    const unsigned layer4_2_bn2_running_mean_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn2_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_2_bn2_running_mean_tr_info = {"layer4_2_bn2_running_mean",
                                                             layer4_2_bn2_running_mean_dram};
    synTensor           layer4_2_bn2_running_mean =
        createTensor(1U, syn_type_single, layer4_2_bn2_running_mean_sizes, true, "layer4_2_bn2_running_mean");

    // create layer4_2_bn2_running_var tensor
    const unsigned layer4_2_bn2_running_var_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn2_running_var_dram    = 0;
    synLaunchTensorInfo layer4_2_bn2_running_var_tr_info = {"layer4_2_bn2_running_var", layer4_2_bn2_running_var_dram};
    synTensor           layer4_2_bn2_running_var =
        createTensor(1U, syn_type_single, layer4_2_bn2_running_var_sizes, true, "layer4_2_bn2_running_var");

    synTensor layer4_2_bn2_in_vec[5] = {layer4_2_conv2_output,
                                        layer4_2_bn2_bias,
                                        layer4_2_bn2_weight,
                                        layer4_2_bn2_running_mean,
                                        layer4_2_bn2_running_var};

    // create layer4_2_bn2_output tensor
    const unsigned layer4_2_bn2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_bn2_output =
        createTensor(4U, syn_type_bf16, layer4_2_bn2_output_sizes, false, "layer4_2_bn2_output");

    // create layer4_2_bn2_saved_mean tensor
    const unsigned layer4_2_bn2_saved_mean_sizes[] = {512};
    synTensor      layer4_2_bn2_saved_mean =
        createTensor(1U, syn_type_single, layer4_2_bn2_saved_mean_sizes, false, "layer4_2_bn2_saved_mean");

    // create layer4_2_bn2_saved_var tensor
    const unsigned layer4_2_bn2_saved_var_sizes[] = {512};
    synTensor      layer4_2_bn2_saved_var =
        createTensor(1U, syn_type_single, layer4_2_bn2_saved_var_sizes, false, "layer4_2_bn2_saved_var");

    synTensor layer4_2_bn2_out_vec[3] = {layer4_2_bn2_output, layer4_2_bn2_saved_mean, layer4_2_bn2_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_2_bn2_in_vec,
                           layer4_2_bn2_out_vec,
                           5,
                           3,
                           (void*)&layer4_2_bn2_kernel_params,
                           sizeof(layer4_2_bn2_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_2_bn2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn2 failed!");

    /*************
     * layer4_2_relu2 node
     * inputs: [layer4_2_bn2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_2_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_2_relu2_in_vec[1] = {layer4_2_bn2_output};

    // create layer4_2_relu2_output tensor
    const unsigned layer4_2_relu2_output_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_relu2_output =
        createTensor(4U, syn_type_bf16, layer4_2_relu2_output_sizes, false, "layer4_2_relu2_output");

    synTensor layer4_2_relu2_out_vec[1] = {layer4_2_relu2_output};

    status = synNodeCreate(graphHandle,
                           layer4_2_relu2_in_vec,
                           layer4_2_relu2_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_2_relu2",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu2 failed!");

    /*************
     * layer4_2_conv3 node
     * inputs: [layer4_2_relu2_output(64, 7, 7, 512)(dtype=bf16), layer4_2_conv3_weight[1, 1, 512, 2048](dtype=bf16)]
     * output: [layer4_2_conv3_output(64, 7, 7, 2048)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_2_conv3_kernel_params;
    layer4_2_conv3_kernel_params.dH   = 1;
    layer4_2_conv3_kernel_params.dW   = 1;
    layer4_2_conv3_kernel_params.kH   = 1;
    layer4_2_conv3_kernel_params.kW   = 1;
    layer4_2_conv3_kernel_params.padT = 0;
    layer4_2_conv3_kernel_params.padB = 0;
    layer4_2_conv3_kernel_params.padL = 0;
    layer4_2_conv3_kernel_params.padR = 0;
    layer4_2_conv3_kernel_params.dilH = 1;
    layer4_2_conv3_kernel_params.dilW = 1;

    // create layer4_2_conv3_weight tensor
    const unsigned      layer4_2_conv3_weight_sizes[] = {1, 1, 512, 2048};
    uint64_t            layer4_2_conv3_weight_dram    = 0;
    synLaunchTensorInfo layer4_2_conv3_weight_tr_info = {"layer4_2_conv3_weight", layer4_2_conv3_weight_dram};
    synTensor           layer4_2_conv3_weight =
        createTensor(4U, syn_type_bf16, layer4_2_conv3_weight_sizes, true, "layer4_2_conv3_weight");

    synTensor layer4_2_conv3_in_vec[4] = {layer4_2_relu2_output, layer4_2_conv3_weight, nullptr, nullptr};

    // create layer4_2_conv3_output tensor
    const unsigned layer4_2_conv3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_conv3_output =
        createTensor(4U, syn_type_bf16, layer4_2_conv3_output_sizes, false, "layer4_2_conv3_output");

    synTensor layer4_2_conv3_out_vec[1] = {layer4_2_conv3_output};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv3_in_vec,
                           layer4_2_conv3_out_vec,
                           4,
                           1,
                           (void*)&layer4_2_conv3_kernel_params,
                           sizeof(layer4_2_conv3_kernel_params),
                           "spatial_convolution",
                           "layer4_2_conv3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv3 failed!");

    /*************
     * layer4_2_bn3 node
     * inputs: [layer4_2_conv3_output(64, 7, 7, 2048)(dtype=bf16), layer4_2_bn3_bias[2048](dtype=float32),
     *layer4_2_bn3_weight[2048](dtype=float32), layer4_2_bn3_running_mean[2048](dtype=float32),
     *layer4_2_bn3_running_var[2048](dtype=float32)] output: [layer4_2_bn3_output(64, 7, 7, 2048)(dtype=bf16),
     *layer4_2_bn3_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_2_bn3_saved_var(1, 1, 1, 2048)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer4_2_bn3_kernel_params;
    layer4_2_bn3_kernel_params.momentum    = 0.1;
    layer4_2_bn3_kernel_params.threshold.f = 1e-05;
    layer4_2_bn3_kernel_params.epsilon     = 1e-05;

    // create layer4_2_bn3_bias tensor
    const unsigned layer4_2_bn3_bias_sizes[] = {
        2048,
    };
    uint64_t            layer4_2_bn3_bias_dram    = 0;
    synLaunchTensorInfo layer4_2_bn3_bias_tr_info = {"layer4_2_bn3_bias", layer4_2_bn3_bias_dram};
    synTensor layer4_2_bn3_bias = createTensor(1U, syn_type_single, layer4_2_bn3_bias_sizes, true, "layer4_2_bn3_bias");

    // create layer4_2_bn3_weight tensor
    const unsigned layer4_2_bn3_weight_sizes[] = {
        2048,
    };
    uint64_t            layer4_2_bn3_weight_dram    = 0;
    synLaunchTensorInfo layer4_2_bn3_weight_tr_info = {"layer4_2_bn3_weight", layer4_2_bn3_weight_dram};
    synTensor           layer4_2_bn3_weight =
        createTensor(1U, syn_type_single, layer4_2_bn3_weight_sizes, true, "layer4_2_bn3_weight");

    // create layer4_2_bn3_running_mean tensor
    const unsigned layer4_2_bn3_running_mean_sizes[] = {
        2048,
    };
    uint64_t            layer4_2_bn3_running_mean_dram    = 0;
    synLaunchTensorInfo layer4_2_bn3_running_mean_tr_info = {"layer4_2_bn3_running_mean",
                                                             layer4_2_bn3_running_mean_dram};
    synTensor           layer4_2_bn3_running_mean =
        createTensor(1U, syn_type_single, layer4_2_bn3_running_mean_sizes, true, "layer4_2_bn3_running_mean");

    // create layer4_2_bn3_running_var tensor
    const unsigned layer4_2_bn3_running_var_sizes[] = {
        2048,
    };
    uint64_t            layer4_2_bn3_running_var_dram    = 0;
    synLaunchTensorInfo layer4_2_bn3_running_var_tr_info = {"layer4_2_bn3_running_var", layer4_2_bn3_running_var_dram};
    synTensor           layer4_2_bn3_running_var =
        createTensor(1U, syn_type_single, layer4_2_bn3_running_var_sizes, true, "layer4_2_bn3_running_var");

    synTensor layer4_2_bn3_in_vec[5] = {layer4_2_conv3_output,
                                        layer4_2_bn3_bias,
                                        layer4_2_bn3_weight,
                                        layer4_2_bn3_running_mean,
                                        layer4_2_bn3_running_var};

    // create layer4_2_bn3_output tensor
    const unsigned layer4_2_bn3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_bn3_output =
        createTensor(4U, syn_type_bf16, layer4_2_bn3_output_sizes, false, "layer4_2_bn3_output");

    // create layer4_2_bn3_saved_mean tensor
    const unsigned layer4_2_bn3_saved_mean_sizes[] = {2048};
    synTensor      layer4_2_bn3_saved_mean =
        createTensor(1U, syn_type_single, layer4_2_bn3_saved_mean_sizes, false, "layer4_2_bn3_saved_mean");

    // create layer4_2_bn3_saved_var tensor
    const unsigned layer4_2_bn3_saved_var_sizes[] = {2048};
    synTensor      layer4_2_bn3_saved_var =
        createTensor(1U, syn_type_single, layer4_2_bn3_saved_var_sizes, false, "layer4_2_bn3_saved_var");

    synTensor layer4_2_bn3_out_vec[3] = {layer4_2_bn3_output, layer4_2_bn3_saved_mean, layer4_2_bn3_saved_var};

    status = synNodeCreate(graphHandle,
                           layer4_2_bn3_in_vec,
                           layer4_2_bn3_out_vec,
                           5,
                           3,
                           (void*)&layer4_2_bn3_kernel_params,
                           sizeof(layer4_2_bn3_kernel_params),
                           "batch_norm_fwd_bf16",
                           "layer4_2_bn3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn3 failed!");

    /*************
     * layer4_2_add_residual_fwd0 node
     * inputs: [layer4_2_bn3_output(64, 7, 7, 2048)(dtype=bf16), layer4_1_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_2_add_residual_fwd(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_2_add_residual_fwd0_in_vec[2] = {layer4_2_bn3_output, layer4_1_relu3_output};

    // create layer4_2_add_residual_fwd tensor
    const unsigned layer4_2_add_residual_fwd_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_add_residual_fwd =
        createTensor(4U, syn_type_bf16, layer4_2_add_residual_fwd_sizes, false, "layer4_2_add_residual_fwd");

    synTensor layer4_2_add_residual_fwd0_out_vec[1] = {layer4_2_add_residual_fwd};

    status = synNodeCreate(graphHandle,
                           layer4_2_add_residual_fwd0_in_vec,
                           layer4_2_add_residual_fwd0_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer4_2_add_residual_fwd0",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_add_residual_fwd0 failed!");

    /*************
     * layer4_2_relu3 node
     * inputs: [layer4_2_add_residual_fwd(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_2_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_2_relu3_in_vec[1] = {layer4_2_add_residual_fwd};

    // create layer4_2_relu3_output tensor
    const unsigned layer4_2_relu3_output_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_relu3_output =
        createTensor(4U, syn_type_bf16, layer4_2_relu3_output_sizes, false, "layer4_2_relu3_output");

    synTensor layer4_2_relu3_out_vec[1] = {layer4_2_relu3_output};

    status = synNodeCreate(graphHandle,
                           layer4_2_relu3_in_vec,
                           layer4_2_relu3_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_bf16",
                           "layer4_2_relu3",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu3 failed!");

    /*************
     * worker_0_avgpool node
     * inputs: [layer4_2_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     * output: [worker_0_avgpool_output(64, 1, 1, 2048)(dtype=bf16)]
     *************/
    ns_AveragePooling::Params worker_0_avgpool_kernel_params;
    worker_0_avgpool_kernel_params.kernel_w           = 7;
    worker_0_avgpool_kernel_params.kernel_h           = 7;
    worker_0_avgpool_kernel_params.stride_w           = 1;
    worker_0_avgpool_kernel_params.stride_h           = 1;
    worker_0_avgpool_kernel_params.pad_w_begin        = 0;
    worker_0_avgpool_kernel_params.pad_w_end          = 0;
    worker_0_avgpool_kernel_params.pad_h_begin        = 0;
    worker_0_avgpool_kernel_params.pad_h_end          = 0;
    worker_0_avgpool_kernel_params.dilation_w         = 1;
    worker_0_avgpool_kernel_params.dilation_h         = 1;
    worker_0_avgpool_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    synTensor worker_0_avgpool_in_vec[1] = {layer4_2_relu3_output};

    // create worker_0_avgpool_output tensor
    const unsigned worker_0_avgpool_output_sizes[] = {64, 1, 1, 2048};
    synTensor      worker_0_avgpool_output =
        createTensor(4U, syn_type_bf16, worker_0_avgpool_output_sizes, false, "worker_0_avgpool_output");

    synTensor worker_0_avgpool_out_vec[1] = {worker_0_avgpool_output};

    status = synNodeCreate(graphHandle,
                           worker_0_avgpool_in_vec,
                           worker_0_avgpool_out_vec,
                           1,
                           1,
                           (void*)&worker_0_avgpool_kernel_params,
                           sizeof(worker_0_avgpool_kernel_params),
                           "avg_pool_2d_fwd_bf16",
                           "worker_0_avgpool",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_avgpool failed!");

    /*************
     * worker_0_fc node
     * inputs: [worker_0_avgpool_output(64, 1, 1, 2048)(dtype=bf16), worker_0_fc_weight(1, 1, 2048, 1000)(dtype=bf16),
     *worker_0_fc_bias[1000](dtype=bf16)] output: [worker_0_fc_output(64, 1, 1, 1000)(dtype=bf16)]
     *************/
    synConvolutionParams worker_0_fc_kernel_params;
    worker_0_fc_kernel_params.dH   = 1;
    worker_0_fc_kernel_params.dW   = 1;
    worker_0_fc_kernel_params.kH   = 1;
    worker_0_fc_kernel_params.kW   = 1;
    worker_0_fc_kernel_params.padT = 0;
    worker_0_fc_kernel_params.padB = 0;
    worker_0_fc_kernel_params.padL = 0;
    worker_0_fc_kernel_params.padR = 0;
    worker_0_fc_kernel_params.dilH = 1;
    worker_0_fc_kernel_params.dilW = 1;

    // create worker_0_fc_weight tensor
    const unsigned      worker_0_fc_weight_sizes[] = {1, 1, 2048, 1000};
    uint64_t            worker_0_fc_weight_dram    = 0;
    synLaunchTensorInfo worker_0_fc_weight_tr_info = {"worker_0_fc_weight", worker_0_fc_weight_dram};
    synTensor           worker_0_fc_weight =
        createTensor(4U, syn_type_bf16, worker_0_fc_weight_sizes, true, "worker_0_fc_weight");

    // create worker_0_fc_bias tensor
    const unsigned worker_0_fc_bias_sizes[] = {
        1000,
    };
    uint64_t            worker_0_fc_bias_dram    = 0;
    synLaunchTensorInfo worker_0_fc_bias_tr_info = {"worker_0_fc_bias", worker_0_fc_bias_dram};
    synTensor worker_0_fc_bias = createTensor(1U, syn_type_bf16, worker_0_fc_bias_sizes, true, "worker_0_fc_bias");

    synTensor worker_0_fc_in_vec[4] = {worker_0_avgpool_output, worker_0_fc_weight, worker_0_fc_bias, nullptr};

    // create worker_0_fc_output tensor
    const unsigned worker_0_fc_output_sizes[] = {64, 1, 1, 1000};
    synTensor      worker_0_fc_output =
        createTensor(4U, syn_type_bf16, worker_0_fc_output_sizes, false, "worker_0_fc_output");

    synTensor worker_0_fc_out_vec[1] = {worker_0_fc_output};

    status = synNodeCreate(graphHandle,
                           worker_0_fc_in_vec,
                           worker_0_fc_out_vec,
                           4,
                           1,
                           (void*)&worker_0_fc_kernel_params,
                           sizeof(worker_0_fc_kernel_params),
                           "spatial_convolution",
                           "worker_0_fc",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc failed!");

    /*************
     * cross_entropy_loss0_log_softmax_reshape node
     * inputs: [worker_0_fc_output(64, 1, 1, 1000)(dtype=bf16)]
     * output: [cross_entropy_loss0_log_softmax_tensor_reshape(64, 1000)(dtype=bf16)]
     *************/

    synTensor cross_entropy_loss0_log_softmax_reshape_in_vec[1] = {worker_0_fc_output};

    // create cross_entropy_loss0_log_softmax_tensor_reshape tensor
    const unsigned cross_entropy_loss0_log_softmax_tensor_reshape_sizes[] = {64, 1000};
    synTensor      cross_entropy_loss0_log_softmax_tensor_reshape =
        createTensor(2U,
                     syn_type_bf16,
                     cross_entropy_loss0_log_softmax_tensor_reshape_sizes,
                     false,
                     "cross_entropy_loss0_log_softmax_tensor_reshape");

    synTensor cross_entropy_loss0_log_softmax_reshape_out_vec[1] = {cross_entropy_loss0_log_softmax_tensor_reshape};

    status = synNodeCreate(graphHandle,
                           cross_entropy_loss0_log_softmax_reshape_in_vec,
                           cross_entropy_loss0_log_softmax_reshape_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "reshape",
                           "cross_entropy_loss0_log_softmax_reshape",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for cross_entropy_loss0_log_softmax_reshape failed!");

    /*************
     * cross_entropy_loss0_log_softmax node
     * inputs: [cross_entropy_loss0_log_softmax_tensor_reshape(64, 1000)(dtype=bf16), target[64](dtype=int32)]
     * output: [cross_entropy_loss0_output(1,)(dtype=bf16), cross_entropy_loss0_logs_output(64, 1000)(dtype=bf16)]
     *************/
    ns_SoftmaxCrossEntropy::Params cross_entropy_loss0_log_softmax_kernel_params;
    cross_entropy_loss0_log_softmax_kernel_params.mode      = CROSS_ENTROPY_MODE_MEAN;
    cross_entropy_loss0_log_softmax_kernel_params.batchSize = batchSize;

    // create target tensor
    const unsigned target_sizes[] = {
        64,
    };
    uint64_t            target_dram    = 0;
    synLaunchTensorInfo target_tr_info = {"target", target_dram};
    synTensor           target         = createTensor(1U, syn_type_int32, target_sizes, true, "target");

    synTensor cross_entropy_loss0_log_softmax_in_vec[2] = {cross_entropy_loss0_log_softmax_tensor_reshape, target};

    // create cross_entropy_loss0_output tensor
    const unsigned cross_entropy_loss0_output_sizes[] = {
        1,
    };
    uint64_t            cross_entropy_loss0_output_dram    = 0;
    synLaunchTensorInfo cross_entropy_loss0_output_tr_info = {"cross_entropy_loss0_output",
                                                              cross_entropy_loss0_output_dram};
    synTensor           cross_entropy_loss0_output =
        createTensor(1U, syn_type_bf16, cross_entropy_loss0_output_sizes, true, "cross_entropy_loss0_output");

    // create cross_entropy_loss0_logs_output tensor
    const unsigned cross_entropy_loss0_logs_output_sizes[] = {64, 1000};
    synTensor      cross_entropy_loss0_logs_output         = createTensor(2U,
                                                             syn_type_bf16,
                                                             cross_entropy_loss0_logs_output_sizes,
                                                             false,
                                                             "cross_entropy_loss0_logs_output");

    synTensor cross_entropy_loss0_log_softmax_out_vec[2] = {cross_entropy_loss0_output,
                                                            cross_entropy_loss0_logs_output};

    status = synNodeCreate(graphHandle,
                           cross_entropy_loss0_log_softmax_in_vec,
                           cross_entropy_loss0_log_softmax_out_vec,
                           2,
                           2,
                           (void*)&cross_entropy_loss0_log_softmax_kernel_params,
                           sizeof(cross_entropy_loss0_log_softmax_kernel_params),
                           "softmax_cross_entropy_fwd_bf16",
                           "cross_entropy_loss0_log_softmax",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for cross_entropy_loss0_log_softmax failed!");

    /*************
     * cross_entropy_loss0_log_softmax_bwd node
     * inputs: [cross_entropy_loss0_logs_output(64, 1000)(dtype=bf16), target[64](dtype=int32)]
     * output: [cross_entropy_loss0_grad_input(64, 1000)(dtype=bf16)]
     *************/
    ns_SoftmaxCrossEntropy::Params cross_entropy_loss0_log_softmax_bwd_kernel_params;
    cross_entropy_loss0_log_softmax_bwd_kernel_params.mode      = CROSS_ENTROPY_MODE_MEAN;
    cross_entropy_loss0_log_softmax_bwd_kernel_params.batchSize = batchSize;

    synTensor cross_entropy_loss0_log_softmax_bwd_in_vec[2] = {cross_entropy_loss0_logs_output, target};

    // create cross_entropy_loss0_grad_input tensor
    const unsigned cross_entropy_loss0_grad_input_sizes[] = {64, 1000};
    synTensor      cross_entropy_loss0_grad_input =
        createTensor(2U, syn_type_bf16, cross_entropy_loss0_grad_input_sizes, false, "cross_entropy_loss0_grad_input");

    synTensor cross_entropy_loss0_log_softmax_bwd_out_vec[1] = {cross_entropy_loss0_grad_input};

    status = synNodeCreate(graphHandle,
                           cross_entropy_loss0_log_softmax_bwd_in_vec,
                           cross_entropy_loss0_log_softmax_bwd_out_vec,
                           2,
                           1,
                           (void*)&cross_entropy_loss0_log_softmax_bwd_kernel_params,
                           sizeof(cross_entropy_loss0_log_softmax_bwd_kernel_params),
                           "softmax_cross_entropy_bwd_bf16",
                           "cross_entropy_loss0_log_softmax_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for cross_entropy_loss0_log_softmax_bwd failed!");

    /*************
     * worker_0_fc_dedx_reshape node
     * inputs: [cross_entropy_loss0_grad_input(64, 1000)(dtype=bf16)]
     * output: [worker_0_fc_dedx_tensor_reshape(64, 1, 1, 1000)(dtype=bf16)]
     *************/

    synTensor worker_0_fc_dedx_reshape_in_vec[1] = {cross_entropy_loss0_grad_input};

    // create worker_0_fc_dedx_tensor_reshape tensor
    const unsigned worker_0_fc_dedx_tensor_reshape_sizes[] = {64, 1, 1, 1000};
    synTensor      worker_0_fc_dedx_tensor_reshape         = createTensor(4U,
                                                             syn_type_bf16,
                                                             worker_0_fc_dedx_tensor_reshape_sizes,
                                                             false,
                                                             "worker_0_fc_dedx_tensor_reshape");

    synTensor worker_0_fc_dedx_reshape_out_vec[1] = {worker_0_fc_dedx_tensor_reshape};

    status = synNodeCreate(graphHandle,
                           worker_0_fc_dedx_reshape_in_vec,
                           worker_0_fc_dedx_reshape_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "reshape",
                           "worker_0_fc_dedx_reshape",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedx_reshape failed!");

    /*************
     * worker_0_fc_dedx node
     * inputs: [worker_0_fc_dedx_tensor_reshape(64, 1, 1, 1000)(dtype=bf16), worker_0_fc_weight(1, 1, 2048,
     *1000)(dtype=bf16)] output: [worker_0_fc_grad_input(64, 1, 1, 2048)(dtype=bf16)]
     *************/
    synConvolutionParams worker_0_fc_dedx_kernel_params;
    worker_0_fc_dedx_kernel_params.dH   = 1;
    worker_0_fc_dedx_kernel_params.dW   = 1;
    worker_0_fc_dedx_kernel_params.kH   = 1;
    worker_0_fc_dedx_kernel_params.kW   = 1;
    worker_0_fc_dedx_kernel_params.padT = 0;
    worker_0_fc_dedx_kernel_params.padB = 0;
    worker_0_fc_dedx_kernel_params.padL = 0;
    worker_0_fc_dedx_kernel_params.padR = 0;
    worker_0_fc_dedx_kernel_params.dilH = 1;
    worker_0_fc_dedx_kernel_params.dilW = 1;

    synTensor worker_0_fc_dedx_in_vec[2] = {worker_0_fc_dedx_tensor_reshape, worker_0_fc_weight};

    // create worker_0_fc_grad_input tensor
    const unsigned worker_0_fc_grad_input_sizes[] = {64, 1, 1, 2048};
    synTensor      worker_0_fc_grad_input =
        createTensor(4U, syn_type_bf16, worker_0_fc_grad_input_sizes, false, "worker_0_fc_grad_input");

    synTensor worker_0_fc_dedx_out_vec[1] = {worker_0_fc_grad_input};

    status = synNodeCreate(graphHandle,
                           worker_0_fc_dedx_in_vec,
                           worker_0_fc_dedx_out_vec,
                           2,
                           1,
                           (void*)&worker_0_fc_dedx_kernel_params,
                           sizeof(worker_0_fc_dedx_kernel_params),
                           "dedx",
                           "worker_0_fc_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedx failed!");

    /*************
     * worker_0_fc_dedw_reshape node
     * inputs: [cross_entropy_loss0_grad_input(64, 1000)(dtype=bf16)]
     * output: [worker_0_fc_dedw_tensor_reshape(64, 1, 1, 1000)(dtype=bf16)]
     *************/

    synTensor worker_0_fc_dedw_reshape_in_vec[1] = {cross_entropy_loss0_grad_input};

    // create worker_0_fc_dedw_tensor_reshape tensor
    const unsigned worker_0_fc_dedw_tensor_reshape_sizes[] = {64, 1, 1, 1000};
    synTensor      worker_0_fc_dedw_tensor_reshape         = createTensor(4U,
                                                             syn_type_bf16,
                                                             worker_0_fc_dedw_tensor_reshape_sizes,
                                                             false,
                                                             "worker_0_fc_dedw_tensor_reshape");

    synTensor worker_0_fc_dedw_reshape_out_vec[1] = {worker_0_fc_dedw_tensor_reshape};

    status = synNodeCreate(graphHandle,
                           worker_0_fc_dedw_reshape_in_vec,
                           worker_0_fc_dedw_reshape_out_vec,
                           1,
                           1,
                           nullptr,
                           0,
                           "reshape",
                           "worker_0_fc_dedw_reshape",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedw_reshape failed!");

    /*************
     * worker_0_fc_dedw node
     * inputs: [worker_0_fc_dedw_tensor_reshape(64, 1, 1, 1000)(dtype=bf16), worker_0_avgpool_output(64, 1, 1,
     *2048)(dtype=bf16)] output: [worker_0_fc_weight_grad(1, 1, 2048, 1000)(dtype=float32)]
     *************/
    synConvolutionParams worker_0_fc_dedw_kernel_params;
    worker_0_fc_dedw_kernel_params.dH   = 1;
    worker_0_fc_dedw_kernel_params.dW   = 1;
    worker_0_fc_dedw_kernel_params.kH   = 1;
    worker_0_fc_dedw_kernel_params.kW   = 1;
    worker_0_fc_dedw_kernel_params.padT = 0;
    worker_0_fc_dedw_kernel_params.padB = 0;
    worker_0_fc_dedw_kernel_params.padL = 0;
    worker_0_fc_dedw_kernel_params.padR = 0;
    worker_0_fc_dedw_kernel_params.dilH = 1;
    worker_0_fc_dedw_kernel_params.dilW = 1;

    synTensor worker_0_fc_dedw_in_vec[2] = {worker_0_fc_dedw_tensor_reshape, worker_0_avgpool_output};

    // create worker_0_fc_weight_grad tensor
    const unsigned      worker_0_fc_weight_grad_sizes[] = {1, 1, 2048, 1000};
    uint64_t            worker_0_fc_weight_grad_dram    = 0;
    synLaunchTensorInfo worker_0_fc_weight_grad_tr_info = {"worker_0_fc_weight_grad", worker_0_fc_weight_grad_dram};
    synTensor           worker_0_fc_weight_grad =
        createTensor(4U, syn_type_single, worker_0_fc_weight_grad_sizes, true, "worker_0_fc_weight_grad");

    synTensor worker_0_fc_dedw_out_vec[1] = {worker_0_fc_weight_grad};

    status = synNodeCreate(graphHandle,
                           worker_0_fc_dedw_in_vec,
                           worker_0_fc_dedw_out_vec,
                           2,
                           1,
                           (void*)&worker_0_fc_dedw_kernel_params,
                           sizeof(worker_0_fc_dedw_kernel_params),
                           "dedw",
                           "worker_0_fc_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedw failed!");

    /*************
     * worker_0_fc_dedb node
     * inputs: [cross_entropy_loss0_grad_input(64, 1000)(dtype=bf16)]
     * output: [worker_0_fc_bias_grad(1000,)(dtype=float32)]
     *************/
    ns_Reduction::Params worker_0_fc_dedb_kernel_params;
    worker_0_fc_dedb_kernel_params.reductionDimension = 1;

    synTensor worker_0_fc_dedb_in_vec[1] = {cross_entropy_loss0_grad_input};

    // create worker_0_fc_bias_grad tensor
    const unsigned worker_0_fc_bias_grad_sizes[] = {
        1000,
    };
    uint64_t            worker_0_fc_bias_grad_dram    = 0;
    synLaunchTensorInfo worker_0_fc_bias_grad_tr_info = {"worker_0_fc_bias_grad", worker_0_fc_bias_grad_dram};
    synTensor           worker_0_fc_bias_grad =
        createTensor(1U, syn_type_single, worker_0_fc_bias_grad_sizes, true, "worker_0_fc_bias_grad");

    synTensor worker_0_fc_dedb_out_vec[1] = {worker_0_fc_bias_grad};

    status = synNodeCreate(graphHandle,
                           worker_0_fc_dedb_in_vec,
                           worker_0_fc_dedb_out_vec,
                           1,
                           1,
                           (void*)&worker_0_fc_dedb_kernel_params,
                           sizeof(worker_0_fc_dedb_kernel_params),
                           "reduce_sum_fwd_bf16",
                           "worker_0_fc_dedb",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_fc_dedb failed!");

    /*************
     * worker_0_avgpool_bwd node
     * inputs: [worker_0_fc_grad_input(64, 1, 1, 2048)(dtype=bf16)]
     * output: [worker_0_avgpool_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/
    ns_AveragePooling::Params worker_0_avgpool_bwd_kernel_params;
    worker_0_avgpool_bwd_kernel_params.kernel_w           = 7;
    worker_0_avgpool_bwd_kernel_params.kernel_h           = 7;
    worker_0_avgpool_bwd_kernel_params.stride_w           = 1;
    worker_0_avgpool_bwd_kernel_params.stride_h           = 1;
    worker_0_avgpool_bwd_kernel_params.pad_w_begin        = 0;
    worker_0_avgpool_bwd_kernel_params.pad_w_end          = 0;
    worker_0_avgpool_bwd_kernel_params.pad_h_begin        = 0;
    worker_0_avgpool_bwd_kernel_params.pad_h_end          = 0;
    worker_0_avgpool_bwd_kernel_params.dilation_w         = 1;
    worker_0_avgpool_bwd_kernel_params.dilation_h         = 1;
    worker_0_avgpool_bwd_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    synTensor worker_0_avgpool_bwd_in_vec[1] = {worker_0_fc_grad_input};

    // create worker_0_avgpool_grad_input tensor
    const unsigned worker_0_avgpool_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      worker_0_avgpool_grad_input =
        createTensor(4U, syn_type_bf16, worker_0_avgpool_grad_input_sizes, false, "worker_0_avgpool_grad_input");

    synTensor worker_0_avgpool_bwd_out_vec[1] = {worker_0_avgpool_grad_input};

    status = synNodeCreate(graphHandle,
                           worker_0_avgpool_bwd_in_vec,
                           worker_0_avgpool_bwd_out_vec,
                           1,
                           1,
                           (void*)&worker_0_avgpool_bwd_kernel_params,
                           sizeof(worker_0_avgpool_bwd_kernel_params),
                           "avg_pool_2d_bwd_bf16",
                           "worker_0_avgpool_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_avgpool_bwd failed!");

    /*************
     * layer4_2_relu3_bwd node
     * inputs: [worker_0_avgpool_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_2_relu3_output(64, 7, 7,
     *2048)(dtype=bf16)] output: [layer4_2_relu3_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_2_relu3_bwd_in_vec[2] = {worker_0_avgpool_grad_input, layer4_2_relu3_output};

    // create layer4_2_relu3_grad_input tensor
    const unsigned layer4_2_relu3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_relu3_grad_input_sizes, false, "layer4_2_relu3_grad_input");

    synTensor layer4_2_relu3_bwd_out_vec[1] = {layer4_2_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_2_relu3_bwd_in_vec,
                           layer4_2_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_2_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu3_bwd failed!");

    /*************
     * layer4_2_add_residual_bwd node
     * inputs: [layer4_2_relu3_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_2_add_residual_grad_input0(64, 7, 7, 2048)(dtype=bf16), layer4_2_add_residual_grad_input1(64, 7,
     *7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_2_add_residual_bwd_in_vec[1] = {layer4_2_relu3_grad_input};

    // create layer4_2_add_residual_grad_input0 tensor
    const unsigned layer4_2_add_residual_grad_input0_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer4_2_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer4_2_add_residual_grad_input0");

    // create layer4_2_add_residual_grad_input1 tensor
    const unsigned layer4_2_add_residual_grad_input1_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer4_2_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer4_2_add_residual_grad_input1");

    synTensor layer4_2_add_residual_bwd_out_vec[2] = {layer4_2_add_residual_grad_input0,
                                                      layer4_2_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer4_2_add_residual_bwd_in_vec,
                           layer4_2_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer4_2_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_add_residual_bwd failed!");

    /*************
     * layer4_2_bn3_bwd node
     * inputs: [layer4_2_conv3_output(64, 7, 7, 2048)(dtype=bf16), layer4_2_add_residual_grad_input0(64, 7, 7,
     *2048)(dtype=bf16), layer4_2_bn3_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_2_bn3_saved_var(1, 1, 1,
     *2048)(dtype=float32), layer4_2_bn3_weight[2048](dtype=float32)] output: [layer4_2_bn3_grad_input(64, 7, 7,
     *2048)(dtype=bf16), layer4_2_bn3_bias_grad(2048,)(dtype=float32), layer4_2_bn3_weight_grad(2048,)(dtype=float32)]
     *************/

    synTensor layer4_2_bn3_bwd_in_vec[5] = {layer4_2_conv3_output,
                                            layer4_2_add_residual_grad_input0,
                                            layer4_2_bn3_saved_mean,
                                            layer4_2_bn3_saved_var,
                                            layer4_2_bn3_weight};

    // create layer4_2_bn3_grad_input tensor
    const unsigned layer4_2_bn3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_bn3_grad_input_sizes, false, "layer4_2_bn3_grad_input");

    // create layer4_2_bn3_bias_grad tensor
    const unsigned layer4_2_bn3_bias_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_2_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_bn3_bias_grad_tr_info = {"layer4_2_bn3_bias_grad", layer4_2_bn3_bias_grad_dram};
    synTensor           layer4_2_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer4_2_bn3_bias_grad_sizes, true, "layer4_2_bn3_bias_grad");

    // create layer4_2_bn3_weight_grad tensor
    const unsigned layer4_2_bn3_weight_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_2_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_bn3_weight_grad_tr_info = {"layer4_2_bn3_weight_grad", layer4_2_bn3_weight_grad_dram};
    synTensor           layer4_2_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer4_2_bn3_weight_grad_sizes, true, "layer4_2_bn3_weight_grad");

    synTensor layer4_2_bn3_bwd_out_vec[3] = {layer4_2_bn3_grad_input, layer4_2_bn3_bias_grad, layer4_2_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_2_bn3_bwd_in_vec,
                           layer4_2_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_2_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn3_bwd failed!");

    /*************
     * layer4_2_conv3_dedx node
     * inputs: [layer4_2_bn3_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_2_conv3_weight[1, 1, 512, 2048](dtype=bf16)]
     * output: [layer4_2_conv3_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_2_conv3_dedx_kernel_params;
    layer4_2_conv3_dedx_kernel_params.dH   = 1;
    layer4_2_conv3_dedx_kernel_params.dW   = 1;
    layer4_2_conv3_dedx_kernel_params.kH   = 1;
    layer4_2_conv3_dedx_kernel_params.kW   = 1;
    layer4_2_conv3_dedx_kernel_params.padT = 0;
    layer4_2_conv3_dedx_kernel_params.padB = 0;
    layer4_2_conv3_dedx_kernel_params.padL = 0;
    layer4_2_conv3_dedx_kernel_params.padR = 0;
    layer4_2_conv3_dedx_kernel_params.dilH = 1;
    layer4_2_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer4_2_conv3_dedx_in_vec[2] = {layer4_2_bn3_grad_input, layer4_2_conv3_weight};

    // create layer4_2_conv3_grad_input tensor
    const unsigned layer4_2_conv3_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_conv3_grad_input_sizes, false, "layer4_2_conv3_grad_input");

    synTensor layer4_2_conv3_dedx_out_vec[1] = {layer4_2_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv3_dedx_in_vec,
                           layer4_2_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_2_conv3_dedx_kernel_params,
                           sizeof(layer4_2_conv3_dedx_kernel_params),
                           "dedx",
                           "layer4_2_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv3_dedx failed!");

    /*************
     * layer4_2_conv3_dedw node
     * inputs: [layer4_2_bn3_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_2_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_2_conv3_weight_grad(1, 1, 512, 2048)(dtype=float32)]
     *************/
    synConvolutionParams layer4_2_conv3_dedw_kernel_params;
    layer4_2_conv3_dedw_kernel_params.dH   = 1;
    layer4_2_conv3_dedw_kernel_params.dW   = 1;
    layer4_2_conv3_dedw_kernel_params.kH   = 1;
    layer4_2_conv3_dedw_kernel_params.kW   = 1;
    layer4_2_conv3_dedw_kernel_params.padT = 0;
    layer4_2_conv3_dedw_kernel_params.padB = 0;
    layer4_2_conv3_dedw_kernel_params.padL = 0;
    layer4_2_conv3_dedw_kernel_params.padR = 0;
    layer4_2_conv3_dedw_kernel_params.dilH = 1;
    layer4_2_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer4_2_conv3_dedw_in_vec[2] = {layer4_2_bn3_grad_input, layer4_2_relu2_output};

    // create layer4_2_conv3_weight_grad tensor
    const unsigned      layer4_2_conv3_weight_grad_sizes[] = {1, 1, 512, 2048};
    uint64_t            layer4_2_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_conv3_weight_grad_tr_info = {"layer4_2_conv3_weight_grad",
                                                              layer4_2_conv3_weight_grad_dram};
    synTensor           layer4_2_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer4_2_conv3_weight_grad_sizes, true, "layer4_2_conv3_weight_grad");

    synTensor layer4_2_conv3_dedw_out_vec[1] = {layer4_2_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv3_dedw_in_vec,
                           layer4_2_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_2_conv3_dedw_kernel_params,
                           sizeof(layer4_2_conv3_dedw_kernel_params),
                           "dedw",
                           "layer4_2_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv3_dedw failed!");

    /*************
     * layer4_2_relu2_bwd node
     * inputs: [layer4_2_conv3_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_2_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_2_relu2_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_2_relu2_bwd_in_vec[2] = {layer4_2_conv3_grad_input, layer4_2_relu2_output};

    // create layer4_2_relu2_grad_input tensor
    const unsigned layer4_2_relu2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_relu2_grad_input_sizes, false, "layer4_2_relu2_grad_input");

    synTensor layer4_2_relu2_bwd_out_vec[1] = {layer4_2_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_2_relu2_bwd_in_vec,
                           layer4_2_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_2_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu2_bwd failed!");

    /*************
     * layer4_2_bn2_bwd node
     * inputs: [layer4_2_conv2_output(64, 7, 7, 512)(dtype=bf16), layer4_2_relu2_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_2_bn2_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_2_bn2_saved_var(1, 1, 1, 512)(dtype=float32),
     *layer4_2_bn2_weight[512](dtype=float32)] output: [layer4_2_bn2_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_2_bn2_bias_grad(512,)(dtype=float32), layer4_2_bn2_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer4_2_bn2_bwd_in_vec[5] = {layer4_2_conv2_output,
                                            layer4_2_relu2_grad_input,
                                            layer4_2_bn2_saved_mean,
                                            layer4_2_bn2_saved_var,
                                            layer4_2_bn2_weight};

    // create layer4_2_bn2_grad_input tensor
    const unsigned layer4_2_bn2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_bn2_grad_input_sizes, false, "layer4_2_bn2_grad_input");

    // create layer4_2_bn2_bias_grad tensor
    const unsigned layer4_2_bn2_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_bn2_bias_grad_tr_info = {"layer4_2_bn2_bias_grad", layer4_2_bn2_bias_grad_dram};
    synTensor           layer4_2_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer4_2_bn2_bias_grad_sizes, true, "layer4_2_bn2_bias_grad");

    // create layer4_2_bn2_weight_grad tensor
    const unsigned layer4_2_bn2_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_bn2_weight_grad_tr_info = {"layer4_2_bn2_weight_grad", layer4_2_bn2_weight_grad_dram};
    synTensor           layer4_2_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer4_2_bn2_weight_grad_sizes, true, "layer4_2_bn2_weight_grad");

    synTensor layer4_2_bn2_bwd_out_vec[3] = {layer4_2_bn2_grad_input, layer4_2_bn2_bias_grad, layer4_2_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_2_bn2_bwd_in_vec,
                           layer4_2_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_2_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn2_bwd failed!");

    /*************
     * layer4_2_conv2_dedx node
     * inputs: [layer4_2_bn2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_2_conv2_weight[3, 3, 512, 512](dtype=bf16)]
     * output: [layer4_2_conv2_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_2_conv2_dedx_kernel_params;
    layer4_2_conv2_dedx_kernel_params.dH   = 1;
    layer4_2_conv2_dedx_kernel_params.dW   = 1;
    layer4_2_conv2_dedx_kernel_params.kH   = 3;
    layer4_2_conv2_dedx_kernel_params.kW   = 3;
    layer4_2_conv2_dedx_kernel_params.padT = 1;
    layer4_2_conv2_dedx_kernel_params.padB = 1;
    layer4_2_conv2_dedx_kernel_params.padL = 1;
    layer4_2_conv2_dedx_kernel_params.padR = 1;
    layer4_2_conv2_dedx_kernel_params.dilH = 1;
    layer4_2_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer4_2_conv2_dedx_in_vec[2] = {layer4_2_bn2_grad_input, layer4_2_conv2_weight};

    // create layer4_2_conv2_grad_input tensor
    const unsigned layer4_2_conv2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_conv2_grad_input_sizes, false, "layer4_2_conv2_grad_input");

    synTensor layer4_2_conv2_dedx_out_vec[1] = {layer4_2_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv2_dedx_in_vec,
                           layer4_2_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_2_conv2_dedx_kernel_params,
                           sizeof(layer4_2_conv2_dedx_kernel_params),
                           "dedx",
                           "layer4_2_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv2_dedx failed!");

    /*************
     * layer4_2_conv2_dedw node
     * inputs: [layer4_2_bn2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_2_relu1_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_2_conv2_weight_grad(3, 3, 512, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer4_2_conv2_dedw_kernel_params;
    layer4_2_conv2_dedw_kernel_params.dH   = 1;
    layer4_2_conv2_dedw_kernel_params.dW   = 1;
    layer4_2_conv2_dedw_kernel_params.kH   = 3;
    layer4_2_conv2_dedw_kernel_params.kW   = 3;
    layer4_2_conv2_dedw_kernel_params.padT = 1;
    layer4_2_conv2_dedw_kernel_params.padB = 1;
    layer4_2_conv2_dedw_kernel_params.padL = 1;
    layer4_2_conv2_dedw_kernel_params.padR = 1;
    layer4_2_conv2_dedw_kernel_params.dilH = 1;
    layer4_2_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer4_2_conv2_dedw_in_vec[2] = {layer4_2_bn2_grad_input, layer4_2_relu1_output};

    // create layer4_2_conv2_weight_grad tensor
    const unsigned      layer4_2_conv2_weight_grad_sizes[] = {3, 3, 512, 512};
    uint64_t            layer4_2_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_conv2_weight_grad_tr_info = {"layer4_2_conv2_weight_grad",
                                                              layer4_2_conv2_weight_grad_dram};
    synTensor           layer4_2_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer4_2_conv2_weight_grad_sizes, true, "layer4_2_conv2_weight_grad");

    synTensor layer4_2_conv2_dedw_out_vec[1] = {layer4_2_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv2_dedw_in_vec,
                           layer4_2_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_2_conv2_dedw_kernel_params,
                           sizeof(layer4_2_conv2_dedw_kernel_params),
                           "dedw",
                           "layer4_2_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv2_dedw failed!");

    /*************
     * layer4_2_relu1_bwd node
     * inputs: [layer4_2_conv2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_2_relu1_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_2_relu1_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_2_relu1_bwd_in_vec[2] = {layer4_2_conv2_grad_input, layer4_2_relu1_output};

    // create layer4_2_relu1_grad_input tensor
    const unsigned layer4_2_relu1_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_relu1_grad_input_sizes, false, "layer4_2_relu1_grad_input");

    synTensor layer4_2_relu1_bwd_out_vec[1] = {layer4_2_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_2_relu1_bwd_in_vec,
                           layer4_2_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_2_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_relu1_bwd failed!");

    /*************
     * layer4_2_bn1_bwd node
     * inputs: [layer4_2_conv1_output(64, 7, 7, 512)(dtype=bf16), layer4_2_relu1_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_2_bn1_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_2_bn1_saved_var(1, 1, 1, 512)(dtype=float32),
     *layer4_2_bn1_weight[512](dtype=float32)] output: [layer4_2_bn1_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_2_bn1_bias_grad(512,)(dtype=float32), layer4_2_bn1_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer4_2_bn1_bwd_in_vec[5] = {layer4_2_conv1_output,
                                            layer4_2_relu1_grad_input,
                                            layer4_2_bn1_saved_mean,
                                            layer4_2_bn1_saved_var,
                                            layer4_2_bn1_weight};

    // create layer4_2_bn1_grad_input tensor
    const unsigned layer4_2_bn1_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_2_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_bn1_grad_input_sizes, false, "layer4_2_bn1_grad_input");

    // create layer4_2_bn1_bias_grad tensor
    const unsigned layer4_2_bn1_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_bn1_bias_grad_tr_info = {"layer4_2_bn1_bias_grad", layer4_2_bn1_bias_grad_dram};
    synTensor           layer4_2_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer4_2_bn1_bias_grad_sizes, true, "layer4_2_bn1_bias_grad");

    // create layer4_2_bn1_weight_grad tensor
    const unsigned layer4_2_bn1_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_2_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_bn1_weight_grad_tr_info = {"layer4_2_bn1_weight_grad", layer4_2_bn1_weight_grad_dram};
    synTensor           layer4_2_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer4_2_bn1_weight_grad_sizes, true, "layer4_2_bn1_weight_grad");

    synTensor layer4_2_bn1_bwd_out_vec[3] = {layer4_2_bn1_grad_input, layer4_2_bn1_bias_grad, layer4_2_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_2_bn1_bwd_in_vec,
                           layer4_2_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_2_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_bn1_bwd failed!");

    /*************
     * layer4_2_conv1_dedx node
     * inputs: [layer4_2_bn1_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_2_conv1_weight[1, 1, 2048, 512](dtype=bf16)]
     * output: [layer4_2_conv1_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_2_conv1_dedx_kernel_params;
    layer4_2_conv1_dedx_kernel_params.dH   = 1;
    layer4_2_conv1_dedx_kernel_params.dW   = 1;
    layer4_2_conv1_dedx_kernel_params.kH   = 1;
    layer4_2_conv1_dedx_kernel_params.kW   = 1;
    layer4_2_conv1_dedx_kernel_params.padT = 0;
    layer4_2_conv1_dedx_kernel_params.padB = 0;
    layer4_2_conv1_dedx_kernel_params.padL = 0;
    layer4_2_conv1_dedx_kernel_params.padR = 0;
    layer4_2_conv1_dedx_kernel_params.dilH = 1;
    layer4_2_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer4_2_conv1_dedx_in_vec[2] = {layer4_2_bn1_grad_input, layer4_2_conv1_weight};

    // create layer4_2_conv1_grad_input tensor
    const unsigned layer4_2_conv1_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_2_conv1_grad_input_sizes, false, "layer4_2_conv1_grad_input");

    synTensor layer4_2_conv1_dedx_out_vec[1] = {layer4_2_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv1_dedx_in_vec,
                           layer4_2_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_2_conv1_dedx_kernel_params,
                           sizeof(layer4_2_conv1_dedx_kernel_params),
                           "dedx",
                           "layer4_2_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv1_dedx failed!");

    /*************
     * layer4_2_conv1_dedw node
     * inputs: [layer4_2_bn1_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_1_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_2_conv1_weight_grad(1, 1, 2048, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer4_2_conv1_dedw_kernel_params;
    layer4_2_conv1_dedw_kernel_params.dH   = 1;
    layer4_2_conv1_dedw_kernel_params.dW   = 1;
    layer4_2_conv1_dedw_kernel_params.kH   = 1;
    layer4_2_conv1_dedw_kernel_params.kW   = 1;
    layer4_2_conv1_dedw_kernel_params.padT = 0;
    layer4_2_conv1_dedw_kernel_params.padB = 0;
    layer4_2_conv1_dedw_kernel_params.padL = 0;
    layer4_2_conv1_dedw_kernel_params.padR = 0;
    layer4_2_conv1_dedw_kernel_params.dilH = 1;
    layer4_2_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer4_2_conv1_dedw_in_vec[2] = {layer4_2_bn1_grad_input, layer4_1_relu3_output};

    // create layer4_2_conv1_weight_grad tensor
    const unsigned      layer4_2_conv1_weight_grad_sizes[] = {1, 1, 2048, 512};
    uint64_t            layer4_2_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_2_conv1_weight_grad_tr_info = {"layer4_2_conv1_weight_grad",
                                                              layer4_2_conv1_weight_grad_dram};
    synTensor           layer4_2_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer4_2_conv1_weight_grad_sizes, true, "layer4_2_conv1_weight_grad");

    synTensor layer4_2_conv1_dedw_out_vec[1] = {layer4_2_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_2_conv1_dedw_in_vec,
                           layer4_2_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_2_conv1_dedw_kernel_params,
                           sizeof(layer4_2_conv1_dedw_kernel_params),
                           "dedw",
                           "layer4_2_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_conv1_dedw failed!");

    /*************
     * layer4_2_add_residual_fwd1 node
     * inputs: [layer4_2_conv1_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_2_add_residual_grad_input1(64, 7, 7,
     *2048)(dtype=bf16)] output: [layer4_2_residual_upstream_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_2_add_residual_fwd1_in_vec[2] = {layer4_2_conv1_grad_input, layer4_2_add_residual_grad_input1};

    // create layer4_2_residual_upstream_grad_input tensor
    const unsigned layer4_2_residual_upstream_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_2_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer4_2_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer4_2_residual_upstream_grad_input");

    synTensor layer4_2_add_residual_fwd1_out_vec[1] = {layer4_2_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_2_add_residual_fwd1_in_vec,
                           layer4_2_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer4_2_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_2_add_residual_fwd1 failed!");

    /*************
     * layer4_1_relu3_bwd node
     * inputs: [layer4_2_residual_upstream_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_1_relu3_output(64, 7, 7,
     *2048)(dtype=bf16)] output: [layer4_1_relu3_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_1_relu3_bwd_in_vec[2] = {layer4_2_residual_upstream_grad_input, layer4_1_relu3_output};

    // create layer4_1_relu3_grad_input tensor
    const unsigned layer4_1_relu3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_relu3_grad_input_sizes, false, "layer4_1_relu3_grad_input");

    synTensor layer4_1_relu3_bwd_out_vec[1] = {layer4_1_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_1_relu3_bwd_in_vec,
                           layer4_1_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_1_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_relu3_bwd failed!");

    /*************
     * layer4_1_add_residual_bwd node
     * inputs: [layer4_1_relu3_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_1_add_residual_grad_input0(64, 7, 7, 2048)(dtype=bf16), layer4_1_add_residual_grad_input1(64, 7,
     *7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_1_add_residual_bwd_in_vec[1] = {layer4_1_relu3_grad_input};

    // create layer4_1_add_residual_grad_input0 tensor
    const unsigned layer4_1_add_residual_grad_input0_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer4_1_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer4_1_add_residual_grad_input0");

    // create layer4_1_add_residual_grad_input1 tensor
    const unsigned layer4_1_add_residual_grad_input1_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer4_1_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer4_1_add_residual_grad_input1");

    synTensor layer4_1_add_residual_bwd_out_vec[2] = {layer4_1_add_residual_grad_input0,
                                                      layer4_1_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer4_1_add_residual_bwd_in_vec,
                           layer4_1_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer4_1_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_add_residual_bwd failed!");

    /*************
     * layer4_1_bn3_bwd node
     * inputs: [layer4_1_conv3_output(64, 7, 7, 2048)(dtype=bf16), layer4_1_add_residual_grad_input0(64, 7, 7,
     *2048)(dtype=bf16), layer4_1_bn3_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_1_bn3_saved_var(1, 1, 1,
     *2048)(dtype=float32), layer4_1_bn3_weight[2048](dtype=float32)] output: [layer4_1_bn3_grad_input(64, 7, 7,
     *2048)(dtype=bf16), layer4_1_bn3_bias_grad(2048,)(dtype=float32), layer4_1_bn3_weight_grad(2048,)(dtype=float32)]
     *************/

    synTensor layer4_1_bn3_bwd_in_vec[5] = {layer4_1_conv3_output,
                                            layer4_1_add_residual_grad_input0,
                                            layer4_1_bn3_saved_mean,
                                            layer4_1_bn3_saved_var,
                                            layer4_1_bn3_weight};

    // create layer4_1_bn3_grad_input tensor
    const unsigned layer4_1_bn3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_bn3_grad_input_sizes, false, "layer4_1_bn3_grad_input");

    // create layer4_1_bn3_bias_grad tensor
    const unsigned layer4_1_bn3_bias_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_1_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_bn3_bias_grad_tr_info = {"layer4_1_bn3_bias_grad", layer4_1_bn3_bias_grad_dram};
    synTensor           layer4_1_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer4_1_bn3_bias_grad_sizes, true, "layer4_1_bn3_bias_grad");

    // create layer4_1_bn3_weight_grad tensor
    const unsigned layer4_1_bn3_weight_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_1_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_bn3_weight_grad_tr_info = {"layer4_1_bn3_weight_grad", layer4_1_bn3_weight_grad_dram};
    synTensor           layer4_1_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer4_1_bn3_weight_grad_sizes, true, "layer4_1_bn3_weight_grad");

    synTensor layer4_1_bn3_bwd_out_vec[3] = {layer4_1_bn3_grad_input, layer4_1_bn3_bias_grad, layer4_1_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_1_bn3_bwd_in_vec,
                           layer4_1_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_1_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_bn3_bwd failed!");

    /*************
     * layer4_1_conv3_dedx node
     * inputs: [layer4_1_bn3_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_1_conv3_weight[1, 1, 512, 2048](dtype=bf16)]
     * output: [layer4_1_conv3_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_1_conv3_dedx_kernel_params;
    layer4_1_conv3_dedx_kernel_params.dH   = 1;
    layer4_1_conv3_dedx_kernel_params.dW   = 1;
    layer4_1_conv3_dedx_kernel_params.kH   = 1;
    layer4_1_conv3_dedx_kernel_params.kW   = 1;
    layer4_1_conv3_dedx_kernel_params.padT = 0;
    layer4_1_conv3_dedx_kernel_params.padB = 0;
    layer4_1_conv3_dedx_kernel_params.padL = 0;
    layer4_1_conv3_dedx_kernel_params.padR = 0;
    layer4_1_conv3_dedx_kernel_params.dilH = 1;
    layer4_1_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer4_1_conv3_dedx_in_vec[2] = {layer4_1_bn3_grad_input, layer4_1_conv3_weight};

    // create layer4_1_conv3_grad_input tensor
    const unsigned layer4_1_conv3_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_conv3_grad_input_sizes, false, "layer4_1_conv3_grad_input");

    synTensor layer4_1_conv3_dedx_out_vec[1] = {layer4_1_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv3_dedx_in_vec,
                           layer4_1_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_1_conv3_dedx_kernel_params,
                           sizeof(layer4_1_conv3_dedx_kernel_params),
                           "dedx",
                           "layer4_1_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv3_dedx failed!");

    /*************
     * layer4_1_conv3_dedw node
     * inputs: [layer4_1_bn3_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_1_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_1_conv3_weight_grad(1, 1, 512, 2048)(dtype=float32)]
     *************/
    synConvolutionParams layer4_1_conv3_dedw_kernel_params;
    layer4_1_conv3_dedw_kernel_params.dH   = 1;
    layer4_1_conv3_dedw_kernel_params.dW   = 1;
    layer4_1_conv3_dedw_kernel_params.kH   = 1;
    layer4_1_conv3_dedw_kernel_params.kW   = 1;
    layer4_1_conv3_dedw_kernel_params.padT = 0;
    layer4_1_conv3_dedw_kernel_params.padB = 0;
    layer4_1_conv3_dedw_kernel_params.padL = 0;
    layer4_1_conv3_dedw_kernel_params.padR = 0;
    layer4_1_conv3_dedw_kernel_params.dilH = 1;
    layer4_1_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer4_1_conv3_dedw_in_vec[2] = {layer4_1_bn3_grad_input, layer4_1_relu2_output};

    // create layer4_1_conv3_weight_grad tensor
    const unsigned      layer4_1_conv3_weight_grad_sizes[] = {1, 1, 512, 2048};
    uint64_t            layer4_1_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_conv3_weight_grad_tr_info = {"layer4_1_conv3_weight_grad",
                                                              layer4_1_conv3_weight_grad_dram};
    synTensor           layer4_1_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer4_1_conv3_weight_grad_sizes, true, "layer4_1_conv3_weight_grad");

    synTensor layer4_1_conv3_dedw_out_vec[1] = {layer4_1_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv3_dedw_in_vec,
                           layer4_1_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_1_conv3_dedw_kernel_params,
                           sizeof(layer4_1_conv3_dedw_kernel_params),
                           "dedw",
                           "layer4_1_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv3_dedw failed!");

    /*************
     * layer4_1_relu2_bwd node
     * inputs: [layer4_1_conv3_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_1_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_1_relu2_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_1_relu2_bwd_in_vec[2] = {layer4_1_conv3_grad_input, layer4_1_relu2_output};

    // create layer4_1_relu2_grad_input tensor
    const unsigned layer4_1_relu2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_relu2_grad_input_sizes, false, "layer4_1_relu2_grad_input");

    synTensor layer4_1_relu2_bwd_out_vec[1] = {layer4_1_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_1_relu2_bwd_in_vec,
                           layer4_1_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_1_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_relu2_bwd failed!");

    /*************
     * layer4_1_bn2_bwd node
     * inputs: [layer4_1_conv2_output(64, 7, 7, 512)(dtype=bf16), layer4_1_relu2_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_1_bn2_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_1_bn2_saved_var(1, 1, 1, 512)(dtype=float32),
     *layer4_1_bn2_weight[512](dtype=float32)] output: [layer4_1_bn2_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_1_bn2_bias_grad(512,)(dtype=float32), layer4_1_bn2_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer4_1_bn2_bwd_in_vec[5] = {layer4_1_conv2_output,
                                            layer4_1_relu2_grad_input,
                                            layer4_1_bn2_saved_mean,
                                            layer4_1_bn2_saved_var,
                                            layer4_1_bn2_weight};

    // create layer4_1_bn2_grad_input tensor
    const unsigned layer4_1_bn2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_bn2_grad_input_sizes, false, "layer4_1_bn2_grad_input");

    // create layer4_1_bn2_bias_grad tensor
    const unsigned layer4_1_bn2_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_bn2_bias_grad_tr_info = {"layer4_1_bn2_bias_grad", layer4_1_bn2_bias_grad_dram};
    synTensor           layer4_1_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer4_1_bn2_bias_grad_sizes, true, "layer4_1_bn2_bias_grad");

    // create layer4_1_bn2_weight_grad tensor
    const unsigned layer4_1_bn2_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_bn2_weight_grad_tr_info = {"layer4_1_bn2_weight_grad", layer4_1_bn2_weight_grad_dram};
    synTensor           layer4_1_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer4_1_bn2_weight_grad_sizes, true, "layer4_1_bn2_weight_grad");

    synTensor layer4_1_bn2_bwd_out_vec[3] = {layer4_1_bn2_grad_input, layer4_1_bn2_bias_grad, layer4_1_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_1_bn2_bwd_in_vec,
                           layer4_1_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_1_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_bn2_bwd failed!");

    /*************
     * layer4_1_conv2_dedx node
     * inputs: [layer4_1_bn2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_1_conv2_weight[3, 3, 512, 512](dtype=bf16)]
     * output: [layer4_1_conv2_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_1_conv2_dedx_kernel_params;
    layer4_1_conv2_dedx_kernel_params.dH   = 1;
    layer4_1_conv2_dedx_kernel_params.dW   = 1;
    layer4_1_conv2_dedx_kernel_params.kH   = 3;
    layer4_1_conv2_dedx_kernel_params.kW   = 3;
    layer4_1_conv2_dedx_kernel_params.padT = 1;
    layer4_1_conv2_dedx_kernel_params.padB = 1;
    layer4_1_conv2_dedx_kernel_params.padL = 1;
    layer4_1_conv2_dedx_kernel_params.padR = 1;
    layer4_1_conv2_dedx_kernel_params.dilH = 1;
    layer4_1_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer4_1_conv2_dedx_in_vec[2] = {layer4_1_bn2_grad_input, layer4_1_conv2_weight};

    // create layer4_1_conv2_grad_input tensor
    const unsigned layer4_1_conv2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_conv2_grad_input_sizes, false, "layer4_1_conv2_grad_input");

    synTensor layer4_1_conv2_dedx_out_vec[1] = {layer4_1_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv2_dedx_in_vec,
                           layer4_1_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_1_conv2_dedx_kernel_params,
                           sizeof(layer4_1_conv2_dedx_kernel_params),
                           "dedx",
                           "layer4_1_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv2_dedx failed!");

    /*************
     * layer4_1_conv2_dedw node
     * inputs: [layer4_1_bn2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_1_relu1_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_1_conv2_weight_grad(3, 3, 512, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer4_1_conv2_dedw_kernel_params;
    layer4_1_conv2_dedw_kernel_params.dH   = 1;
    layer4_1_conv2_dedw_kernel_params.dW   = 1;
    layer4_1_conv2_dedw_kernel_params.kH   = 3;
    layer4_1_conv2_dedw_kernel_params.kW   = 3;
    layer4_1_conv2_dedw_kernel_params.padT = 1;
    layer4_1_conv2_dedw_kernel_params.padB = 1;
    layer4_1_conv2_dedw_kernel_params.padL = 1;
    layer4_1_conv2_dedw_kernel_params.padR = 1;
    layer4_1_conv2_dedw_kernel_params.dilH = 1;
    layer4_1_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer4_1_conv2_dedw_in_vec[2] = {layer4_1_bn2_grad_input, layer4_1_relu1_output};

    // create layer4_1_conv2_weight_grad tensor
    const unsigned      layer4_1_conv2_weight_grad_sizes[] = {3, 3, 512, 512};
    uint64_t            layer4_1_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_conv2_weight_grad_tr_info = {"layer4_1_conv2_weight_grad",
                                                              layer4_1_conv2_weight_grad_dram};
    synTensor           layer4_1_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer4_1_conv2_weight_grad_sizes, true, "layer4_1_conv2_weight_grad");

    synTensor layer4_1_conv2_dedw_out_vec[1] = {layer4_1_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv2_dedw_in_vec,
                           layer4_1_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_1_conv2_dedw_kernel_params,
                           sizeof(layer4_1_conv2_dedw_kernel_params),
                           "dedw",
                           "layer4_1_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv2_dedw failed!");

    /*************
     * layer4_1_relu1_bwd node
     * inputs: [layer4_1_conv2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_1_relu1_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_1_relu1_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_1_relu1_bwd_in_vec[2] = {layer4_1_conv2_grad_input, layer4_1_relu1_output};

    // create layer4_1_relu1_grad_input tensor
    const unsigned layer4_1_relu1_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_relu1_grad_input_sizes, false, "layer4_1_relu1_grad_input");

    synTensor layer4_1_relu1_bwd_out_vec[1] = {layer4_1_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_1_relu1_bwd_in_vec,
                           layer4_1_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_1_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_relu1_bwd failed!");

    /*************
     * layer4_1_bn1_bwd node
     * inputs: [layer4_1_conv1_output(64, 7, 7, 512)(dtype=bf16), layer4_1_relu1_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_1_bn1_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_1_bn1_saved_var(1, 1, 1, 512)(dtype=float32),
     *layer4_1_bn1_weight[512](dtype=float32)] output: [layer4_1_bn1_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_1_bn1_bias_grad(512,)(dtype=float32), layer4_1_bn1_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer4_1_bn1_bwd_in_vec[5] = {layer4_1_conv1_output,
                                            layer4_1_relu1_grad_input,
                                            layer4_1_bn1_saved_mean,
                                            layer4_1_bn1_saved_var,
                                            layer4_1_bn1_weight};

    // create layer4_1_bn1_grad_input tensor
    const unsigned layer4_1_bn1_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_1_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_bn1_grad_input_sizes, false, "layer4_1_bn1_grad_input");

    // create layer4_1_bn1_bias_grad tensor
    const unsigned layer4_1_bn1_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_bn1_bias_grad_tr_info = {"layer4_1_bn1_bias_grad", layer4_1_bn1_bias_grad_dram};
    synTensor           layer4_1_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer4_1_bn1_bias_grad_sizes, true, "layer4_1_bn1_bias_grad");

    // create layer4_1_bn1_weight_grad tensor
    const unsigned layer4_1_bn1_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_1_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_bn1_weight_grad_tr_info = {"layer4_1_bn1_weight_grad", layer4_1_bn1_weight_grad_dram};
    synTensor           layer4_1_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer4_1_bn1_weight_grad_sizes, true, "layer4_1_bn1_weight_grad");

    synTensor layer4_1_bn1_bwd_out_vec[3] = {layer4_1_bn1_grad_input, layer4_1_bn1_bias_grad, layer4_1_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_1_bn1_bwd_in_vec,
                           layer4_1_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_1_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_bn1_bwd failed!");

    /*************
     * layer4_1_conv1_dedx node
     * inputs: [layer4_1_bn1_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_1_conv1_weight[1, 1, 2048, 512](dtype=bf16)]
     * output: [layer4_1_conv1_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_1_conv1_dedx_kernel_params;
    layer4_1_conv1_dedx_kernel_params.dH   = 1;
    layer4_1_conv1_dedx_kernel_params.dW   = 1;
    layer4_1_conv1_dedx_kernel_params.kH   = 1;
    layer4_1_conv1_dedx_kernel_params.kW   = 1;
    layer4_1_conv1_dedx_kernel_params.padT = 0;
    layer4_1_conv1_dedx_kernel_params.padB = 0;
    layer4_1_conv1_dedx_kernel_params.padL = 0;
    layer4_1_conv1_dedx_kernel_params.padR = 0;
    layer4_1_conv1_dedx_kernel_params.dilH = 1;
    layer4_1_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer4_1_conv1_dedx_in_vec[2] = {layer4_1_bn1_grad_input, layer4_1_conv1_weight};

    // create layer4_1_conv1_grad_input tensor
    const unsigned layer4_1_conv1_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_1_conv1_grad_input_sizes, false, "layer4_1_conv1_grad_input");

    synTensor layer4_1_conv1_dedx_out_vec[1] = {layer4_1_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv1_dedx_in_vec,
                           layer4_1_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_1_conv1_dedx_kernel_params,
                           sizeof(layer4_1_conv1_dedx_kernel_params),
                           "dedx",
                           "layer4_1_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv1_dedx failed!");

    /*************
     * layer4_1_conv1_dedw node
     * inputs: [layer4_1_bn1_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_0_relu3_output(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_1_conv1_weight_grad(1, 1, 2048, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer4_1_conv1_dedw_kernel_params;
    layer4_1_conv1_dedw_kernel_params.dH   = 1;
    layer4_1_conv1_dedw_kernel_params.dW   = 1;
    layer4_1_conv1_dedw_kernel_params.kH   = 1;
    layer4_1_conv1_dedw_kernel_params.kW   = 1;
    layer4_1_conv1_dedw_kernel_params.padT = 0;
    layer4_1_conv1_dedw_kernel_params.padB = 0;
    layer4_1_conv1_dedw_kernel_params.padL = 0;
    layer4_1_conv1_dedw_kernel_params.padR = 0;
    layer4_1_conv1_dedw_kernel_params.dilH = 1;
    layer4_1_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer4_1_conv1_dedw_in_vec[2] = {layer4_1_bn1_grad_input, layer4_0_relu3_output};

    // create layer4_1_conv1_weight_grad tensor
    const unsigned      layer4_1_conv1_weight_grad_sizes[] = {1, 1, 2048, 512};
    uint64_t            layer4_1_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_1_conv1_weight_grad_tr_info = {"layer4_1_conv1_weight_grad",
                                                              layer4_1_conv1_weight_grad_dram};
    synTensor           layer4_1_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer4_1_conv1_weight_grad_sizes, true, "layer4_1_conv1_weight_grad");

    synTensor layer4_1_conv1_dedw_out_vec[1] = {layer4_1_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_1_conv1_dedw_in_vec,
                           layer4_1_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_1_conv1_dedw_kernel_params,
                           sizeof(layer4_1_conv1_dedw_kernel_params),
                           "dedw",
                           "layer4_1_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_conv1_dedw failed!");

    /*************
     * layer4_1_add_residual_fwd1 node
     * inputs: [layer4_1_conv1_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_1_add_residual_grad_input1(64, 7, 7,
     *2048)(dtype=bf16)] output: [layer4_1_residual_upstream_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_1_add_residual_fwd1_in_vec[2] = {layer4_1_conv1_grad_input, layer4_1_add_residual_grad_input1};

    // create layer4_1_residual_upstream_grad_input tensor
    const unsigned layer4_1_residual_upstream_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_1_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer4_1_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer4_1_residual_upstream_grad_input");

    synTensor layer4_1_add_residual_fwd1_out_vec[1] = {layer4_1_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_1_add_residual_fwd1_in_vec,
                           layer4_1_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer4_1_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_1_add_residual_fwd1 failed!");

    /*************
     * layer4_0_relu3_bwd node
     * inputs: [layer4_1_residual_upstream_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_0_relu3_output(64, 7, 7,
     *2048)(dtype=bf16)] output: [layer4_0_relu3_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_0_relu3_bwd_in_vec[2] = {layer4_1_residual_upstream_grad_input, layer4_0_relu3_output};

    // create layer4_0_relu3_grad_input tensor
    const unsigned layer4_0_relu3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_relu3_grad_input_sizes, false, "layer4_0_relu3_grad_input");

    synTensor layer4_0_relu3_bwd_out_vec[1] = {layer4_0_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_0_relu3_bwd_in_vec,
                           layer4_0_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_0_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_relu3_bwd failed!");

    /*************
     * layer4_0_add_residual_bwd node
     * inputs: [layer4_0_relu3_grad_input(64, 7, 7, 2048)(dtype=bf16)]
     * output: [layer4_0_add_residual_grad_input0(64, 7, 7, 2048)(dtype=bf16), layer4_0_add_residual_grad_input1(64, 7,
     *7, 2048)(dtype=bf16)]
     *************/

    synTensor layer4_0_add_residual_bwd_in_vec[1] = {layer4_0_relu3_grad_input};

    // create layer4_0_add_residual_grad_input0 tensor
    const unsigned layer4_0_add_residual_grad_input0_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer4_0_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer4_0_add_residual_grad_input0");

    // create layer4_0_add_residual_grad_input1 tensor
    const unsigned layer4_0_add_residual_grad_input1_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer4_0_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer4_0_add_residual_grad_input1");

    synTensor layer4_0_add_residual_bwd_out_vec[2] = {layer4_0_add_residual_grad_input0,
                                                      layer4_0_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer4_0_add_residual_bwd_in_vec,
                           layer4_0_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer4_0_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_add_residual_bwd failed!");

    /*************
     * layer4_0_bn3_bwd node
     * inputs: [layer4_0_conv3_output(64, 7, 7, 2048)(dtype=bf16), layer4_0_add_residual_grad_input0(64, 7, 7,
     *2048)(dtype=bf16), layer4_0_bn3_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_0_bn3_saved_var(1, 1, 1,
     *2048)(dtype=float32), layer4_0_bn3_weight[2048](dtype=float32)] output: [layer4_0_bn3_grad_input(64, 7, 7,
     *2048)(dtype=bf16), layer4_0_bn3_bias_grad(2048,)(dtype=float32), layer4_0_bn3_weight_grad(2048,)(dtype=float32)]
     *************/

    synTensor layer4_0_bn3_bwd_in_vec[5] = {layer4_0_conv3_output,
                                            layer4_0_add_residual_grad_input0,
                                            layer4_0_bn3_saved_mean,
                                            layer4_0_bn3_saved_var,
                                            layer4_0_bn3_weight};

    // create layer4_0_bn3_grad_input tensor
    const unsigned layer4_0_bn3_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_0_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_bn3_grad_input_sizes, false, "layer4_0_bn3_grad_input");

    // create layer4_0_bn3_bias_grad tensor
    const unsigned layer4_0_bn3_bias_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_0_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_bn3_bias_grad_tr_info = {"layer4_0_bn3_bias_grad", layer4_0_bn3_bias_grad_dram};
    synTensor           layer4_0_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer4_0_bn3_bias_grad_sizes, true, "layer4_0_bn3_bias_grad");

    // create layer4_0_bn3_weight_grad tensor
    const unsigned layer4_0_bn3_weight_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_0_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_bn3_weight_grad_tr_info = {"layer4_0_bn3_weight_grad", layer4_0_bn3_weight_grad_dram};
    synTensor           layer4_0_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer4_0_bn3_weight_grad_sizes, true, "layer4_0_bn3_weight_grad");

    synTensor layer4_0_bn3_bwd_out_vec[3] = {layer4_0_bn3_grad_input, layer4_0_bn3_bias_grad, layer4_0_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_0_bn3_bwd_in_vec,
                           layer4_0_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_0_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_bn3_bwd failed!");

    /*************
     * layer4_0_conv3_dedx node
     * inputs: [layer4_0_bn3_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_0_conv3_weight[1, 1, 512, 2048](dtype=bf16)]
     * output: [layer4_0_conv3_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_0_conv3_dedx_kernel_params;
    layer4_0_conv3_dedx_kernel_params.dH   = 1;
    layer4_0_conv3_dedx_kernel_params.dW   = 1;
    layer4_0_conv3_dedx_kernel_params.kH   = 1;
    layer4_0_conv3_dedx_kernel_params.kW   = 1;
    layer4_0_conv3_dedx_kernel_params.padT = 0;
    layer4_0_conv3_dedx_kernel_params.padB = 0;
    layer4_0_conv3_dedx_kernel_params.padL = 0;
    layer4_0_conv3_dedx_kernel_params.padR = 0;
    layer4_0_conv3_dedx_kernel_params.dilH = 1;
    layer4_0_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer4_0_conv3_dedx_in_vec[2] = {layer4_0_bn3_grad_input, layer4_0_conv3_weight};

    // create layer4_0_conv3_grad_input tensor
    const unsigned layer4_0_conv3_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_0_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_conv3_grad_input_sizes, false, "layer4_0_conv3_grad_input");

    synTensor layer4_0_conv3_dedx_out_vec[1] = {layer4_0_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv3_dedx_in_vec,
                           layer4_0_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_0_conv3_dedx_kernel_params,
                           sizeof(layer4_0_conv3_dedx_kernel_params),
                           "dedx",
                           "layer4_0_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv3_dedx failed!");

    /*************
     * layer4_0_conv3_dedw node
     * inputs: [layer4_0_bn3_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_0_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_0_conv3_weight_grad(1, 1, 512, 2048)(dtype=float32)]
     *************/
    synConvolutionParams layer4_0_conv3_dedw_kernel_params;
    layer4_0_conv3_dedw_kernel_params.dH   = 1;
    layer4_0_conv3_dedw_kernel_params.dW   = 1;
    layer4_0_conv3_dedw_kernel_params.kH   = 1;
    layer4_0_conv3_dedw_kernel_params.kW   = 1;
    layer4_0_conv3_dedw_kernel_params.padT = 0;
    layer4_0_conv3_dedw_kernel_params.padB = 0;
    layer4_0_conv3_dedw_kernel_params.padL = 0;
    layer4_0_conv3_dedw_kernel_params.padR = 0;
    layer4_0_conv3_dedw_kernel_params.dilH = 1;
    layer4_0_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer4_0_conv3_dedw_in_vec[2] = {layer4_0_bn3_grad_input, layer4_0_relu2_output};

    // create layer4_0_conv3_weight_grad tensor
    const unsigned      layer4_0_conv3_weight_grad_sizes[] = {1, 1, 512, 2048};
    uint64_t            layer4_0_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_conv3_weight_grad_tr_info = {"layer4_0_conv3_weight_grad",
                                                              layer4_0_conv3_weight_grad_dram};
    synTensor           layer4_0_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer4_0_conv3_weight_grad_sizes, true, "layer4_0_conv3_weight_grad");

    synTensor layer4_0_conv3_dedw_out_vec[1] = {layer4_0_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv3_dedw_in_vec,
                           layer4_0_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_0_conv3_dedw_kernel_params,
                           sizeof(layer4_0_conv3_dedw_kernel_params),
                           "dedw",
                           "layer4_0_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv3_dedw failed!");

    /*************
     * layer4_0_relu2_bwd node
     * inputs: [layer4_0_conv3_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_0_relu2_output(64, 7, 7, 512)(dtype=bf16)]
     * output: [layer4_0_relu2_grad_input(64, 7, 7, 512)(dtype=bf16)]
     *************/

    synTensor layer4_0_relu2_bwd_in_vec[2] = {layer4_0_conv3_grad_input, layer4_0_relu2_output};

    // create layer4_0_relu2_grad_input tensor
    const unsigned layer4_0_relu2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_0_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_relu2_grad_input_sizes, false, "layer4_0_relu2_grad_input");

    synTensor layer4_0_relu2_bwd_out_vec[1] = {layer4_0_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_0_relu2_bwd_in_vec,
                           layer4_0_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_0_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_relu2_bwd failed!");

    /*************
     * layer4_0_bn2_bwd node
     * inputs: [layer4_0_conv2_output(64, 7, 7, 512)(dtype=bf16), layer4_0_relu2_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_0_bn2_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_0_bn2_saved_var(1, 1, 1, 512)(dtype=float32),
     *layer4_0_bn2_weight[512](dtype=float32)] output: [layer4_0_bn2_grad_input(64, 7, 7, 512)(dtype=bf16),
     *layer4_0_bn2_bias_grad(512,)(dtype=float32), layer4_0_bn2_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer4_0_bn2_bwd_in_vec[5] = {layer4_0_conv2_output,
                                            layer4_0_relu2_grad_input,
                                            layer4_0_bn2_saved_mean,
                                            layer4_0_bn2_saved_var,
                                            layer4_0_bn2_weight};

    // create layer4_0_bn2_grad_input tensor
    const unsigned layer4_0_bn2_grad_input_sizes[] = {64, 7, 7, 512};
    synTensor      layer4_0_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_bn2_grad_input_sizes, false, "layer4_0_bn2_grad_input");

    // create layer4_0_bn2_bias_grad tensor
    const unsigned layer4_0_bn2_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_bn2_bias_grad_tr_info = {"layer4_0_bn2_bias_grad", layer4_0_bn2_bias_grad_dram};
    synTensor           layer4_0_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer4_0_bn2_bias_grad_sizes, true, "layer4_0_bn2_bias_grad");

    // create layer4_0_bn2_weight_grad tensor
    const unsigned layer4_0_bn2_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_bn2_weight_grad_tr_info = {"layer4_0_bn2_weight_grad", layer4_0_bn2_weight_grad_dram};
    synTensor           layer4_0_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer4_0_bn2_weight_grad_sizes, true, "layer4_0_bn2_weight_grad");

    synTensor layer4_0_bn2_bwd_out_vec[3] = {layer4_0_bn2_grad_input, layer4_0_bn2_bias_grad, layer4_0_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_0_bn2_bwd_in_vec,
                           layer4_0_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_0_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_bn2_bwd failed!");

    /*************
     * layer4_0_conv2_dedx node
     * inputs: [layer4_0_bn2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_0_conv2_weight[3, 3, 512, 512](dtype=bf16)]
     * output: [layer4_0_conv2_grad_input(64, 14, 14, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_0_conv2_dedx_kernel_params;
    layer4_0_conv2_dedx_kernel_params.dH   = 2;
    layer4_0_conv2_dedx_kernel_params.dW   = 2;
    layer4_0_conv2_dedx_kernel_params.kH   = 3;
    layer4_0_conv2_dedx_kernel_params.kW   = 3;
    layer4_0_conv2_dedx_kernel_params.padT = 1;
    layer4_0_conv2_dedx_kernel_params.padB = 1;
    layer4_0_conv2_dedx_kernel_params.padL = 1;
    layer4_0_conv2_dedx_kernel_params.padR = 1;
    layer4_0_conv2_dedx_kernel_params.dilH = 1;
    layer4_0_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer4_0_conv2_dedx_in_vec[2] = {layer4_0_bn2_grad_input, layer4_0_conv2_weight};

    // create layer4_0_conv2_grad_input tensor
    const unsigned layer4_0_conv2_grad_input_sizes[] = {64, 14, 14, 512};
    synTensor      layer4_0_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_conv2_grad_input_sizes, false, "layer4_0_conv2_grad_input");

    synTensor layer4_0_conv2_dedx_out_vec[1] = {layer4_0_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv2_dedx_in_vec,
                           layer4_0_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_0_conv2_dedx_kernel_params,
                           sizeof(layer4_0_conv2_dedx_kernel_params),
                           "dedx",
                           "layer4_0_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv2_dedx failed!");

    /*************
     * layer4_0_conv2_dedw node
     * inputs: [layer4_0_bn2_grad_input(64, 7, 7, 512)(dtype=bf16), layer4_0_relu1_output(64, 14, 14, 512)(dtype=bf16)]
     * output: [layer4_0_conv2_weight_grad(3, 3, 512, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer4_0_conv2_dedw_kernel_params;
    layer4_0_conv2_dedw_kernel_params.dH   = 2;
    layer4_0_conv2_dedw_kernel_params.dW   = 2;
    layer4_0_conv2_dedw_kernel_params.kH   = 3;
    layer4_0_conv2_dedw_kernel_params.kW   = 3;
    layer4_0_conv2_dedw_kernel_params.padT = 1;
    layer4_0_conv2_dedw_kernel_params.padB = 1;
    layer4_0_conv2_dedw_kernel_params.padL = 1;
    layer4_0_conv2_dedw_kernel_params.padR = 1;
    layer4_0_conv2_dedw_kernel_params.dilH = 1;
    layer4_0_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer4_0_conv2_dedw_in_vec[2] = {layer4_0_bn2_grad_input, layer4_0_relu1_output};

    // create layer4_0_conv2_weight_grad tensor
    const unsigned      layer4_0_conv2_weight_grad_sizes[] = {3, 3, 512, 512};
    uint64_t            layer4_0_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_conv2_weight_grad_tr_info = {"layer4_0_conv2_weight_grad",
                                                              layer4_0_conv2_weight_grad_dram};
    synTensor           layer4_0_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer4_0_conv2_weight_grad_sizes, true, "layer4_0_conv2_weight_grad");

    synTensor layer4_0_conv2_dedw_out_vec[1] = {layer4_0_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv2_dedw_in_vec,
                           layer4_0_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_0_conv2_dedw_kernel_params,
                           sizeof(layer4_0_conv2_dedw_kernel_params),
                           "dedw",
                           "layer4_0_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv2_dedw failed!");

    /*************
     * layer4_0_relu1_bwd node
     * inputs: [layer4_0_conv2_grad_input(64, 14, 14, 512)(dtype=bf16), layer4_0_relu1_output(64, 14, 14,
     *512)(dtype=bf16)] output: [layer4_0_relu1_grad_input(64, 14, 14, 512)(dtype=bf16)]
     *************/

    synTensor layer4_0_relu1_bwd_in_vec[2] = {layer4_0_conv2_grad_input, layer4_0_relu1_output};

    // create layer4_0_relu1_grad_input tensor
    const unsigned layer4_0_relu1_grad_input_sizes[] = {64, 14, 14, 512};
    synTensor      layer4_0_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_relu1_grad_input_sizes, false, "layer4_0_relu1_grad_input");

    synTensor layer4_0_relu1_bwd_out_vec[1] = {layer4_0_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_0_relu1_bwd_in_vec,
                           layer4_0_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer4_0_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_relu1_bwd failed!");

    /*************
     * layer4_0_bn1_bwd node
     * inputs: [layer4_0_conv1_output(64, 14, 14, 512)(dtype=bf16), layer4_0_relu1_grad_input(64, 14, 14,
     *512)(dtype=bf16), layer4_0_bn1_saved_mean(1, 1, 1, 512)(dtype=float32), layer4_0_bn1_saved_var(1, 1, 1,
     *512)(dtype=float32), layer4_0_bn1_weight[512](dtype=float32)] output: [layer4_0_bn1_grad_input(64, 14, 14,
     *512)(dtype=bf16), layer4_0_bn1_bias_grad(512,)(dtype=float32), layer4_0_bn1_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer4_0_bn1_bwd_in_vec[5] = {layer4_0_conv1_output,
                                            layer4_0_relu1_grad_input,
                                            layer4_0_bn1_saved_mean,
                                            layer4_0_bn1_saved_var,
                                            layer4_0_bn1_weight};

    // create layer4_0_bn1_grad_input tensor
    const unsigned layer4_0_bn1_grad_input_sizes[] = {64, 14, 14, 512};
    synTensor      layer4_0_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_bn1_grad_input_sizes, false, "layer4_0_bn1_grad_input");

    // create layer4_0_bn1_bias_grad tensor
    const unsigned layer4_0_bn1_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_bn1_bias_grad_tr_info = {"layer4_0_bn1_bias_grad", layer4_0_bn1_bias_grad_dram};
    synTensor           layer4_0_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer4_0_bn1_bias_grad_sizes, true, "layer4_0_bn1_bias_grad");

    // create layer4_0_bn1_weight_grad tensor
    const unsigned layer4_0_bn1_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer4_0_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_bn1_weight_grad_tr_info = {"layer4_0_bn1_weight_grad", layer4_0_bn1_weight_grad_dram};
    synTensor           layer4_0_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer4_0_bn1_weight_grad_sizes, true, "layer4_0_bn1_weight_grad");

    synTensor layer4_0_bn1_bwd_out_vec[3] = {layer4_0_bn1_grad_input, layer4_0_bn1_bias_grad, layer4_0_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_0_bn1_bwd_in_vec,
                           layer4_0_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_0_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_bn1_bwd failed!");

    /*************
     * layer4_0_conv1_dedx node
     * inputs: [layer4_0_bn1_grad_input(64, 14, 14, 512)(dtype=bf16), layer4_0_conv1_weight[1, 1, 1024,
     *512](dtype=bf16)] output: [layer4_0_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_0_conv1_dedx_kernel_params;
    layer4_0_conv1_dedx_kernel_params.dH   = 1;
    layer4_0_conv1_dedx_kernel_params.dW   = 1;
    layer4_0_conv1_dedx_kernel_params.kH   = 1;
    layer4_0_conv1_dedx_kernel_params.kW   = 1;
    layer4_0_conv1_dedx_kernel_params.padT = 0;
    layer4_0_conv1_dedx_kernel_params.padB = 0;
    layer4_0_conv1_dedx_kernel_params.padL = 0;
    layer4_0_conv1_dedx_kernel_params.padR = 0;
    layer4_0_conv1_dedx_kernel_params.dilH = 1;
    layer4_0_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer4_0_conv1_dedx_in_vec[2] = {layer4_0_bn1_grad_input, layer4_0_conv1_weight};

    // create layer4_0_conv1_grad_input tensor
    const unsigned layer4_0_conv1_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer4_0_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer4_0_conv1_grad_input_sizes, false, "layer4_0_conv1_grad_input");

    synTensor layer4_0_conv1_dedx_out_vec[1] = {layer4_0_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv1_dedx_in_vec,
                           layer4_0_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_0_conv1_dedx_kernel_params,
                           sizeof(layer4_0_conv1_dedx_kernel_params),
                           "dedx",
                           "layer4_0_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv1_dedx failed!");

    /*************
     * layer4_0_conv1_dedw node
     * inputs: [layer4_0_bn1_grad_input(64, 14, 14, 512)(dtype=bf16), layer3_5_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer4_0_conv1_weight_grad(1, 1, 1024, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer4_0_conv1_dedw_kernel_params;
    layer4_0_conv1_dedw_kernel_params.dH   = 1;
    layer4_0_conv1_dedw_kernel_params.dW   = 1;
    layer4_0_conv1_dedw_kernel_params.kH   = 1;
    layer4_0_conv1_dedw_kernel_params.kW   = 1;
    layer4_0_conv1_dedw_kernel_params.padT = 0;
    layer4_0_conv1_dedw_kernel_params.padB = 0;
    layer4_0_conv1_dedw_kernel_params.padL = 0;
    layer4_0_conv1_dedw_kernel_params.padR = 0;
    layer4_0_conv1_dedw_kernel_params.dilH = 1;
    layer4_0_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer4_0_conv1_dedw_in_vec[2] = {layer4_0_bn1_grad_input, layer3_5_relu3_output};

    // create layer4_0_conv1_weight_grad tensor
    const unsigned      layer4_0_conv1_weight_grad_sizes[] = {1, 1, 1024, 512};
    uint64_t            layer4_0_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_0_conv1_weight_grad_tr_info = {"layer4_0_conv1_weight_grad",
                                                              layer4_0_conv1_weight_grad_dram};
    synTensor           layer4_0_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer4_0_conv1_weight_grad_sizes, true, "layer4_0_conv1_weight_grad");

    synTensor layer4_0_conv1_dedw_out_vec[1] = {layer4_0_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_0_conv1_dedw_in_vec,
                           layer4_0_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_0_conv1_dedw_kernel_params,
                           sizeof(layer4_0_conv1_dedw_kernel_params),
                           "dedw",
                           "layer4_0_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_conv1_dedw failed!");

    /*************
     * layer4_bn_bwd node
     * inputs: [layer4_downsample_output(64, 7, 7, 2048)(dtype=bf16), layer4_0_add_residual_grad_input1(64, 7, 7,
     *2048)(dtype=bf16), layer4_bn_saved_mean(1, 1, 1, 2048)(dtype=float32), layer4_bn_saved_var(1, 1, 1,
     *2048)(dtype=float32), layer4_bn_weight[2048](dtype=float32)] output: [layer4_bn_grad_input(64, 7, 7,
     *2048)(dtype=bf16), layer4_bn_bias_grad(2048,)(dtype=float32), layer4_bn_weight_grad(2048,)(dtype=float32)]
     *************/

    synTensor layer4_bn_bwd_in_vec[5] = {layer4_downsample_output,
                                         layer4_0_add_residual_grad_input1,
                                         layer4_bn_saved_mean,
                                         layer4_bn_saved_var,
                                         layer4_bn_weight};

    // create layer4_bn_grad_input tensor
    const unsigned layer4_bn_grad_input_sizes[] = {64, 7, 7, 2048};
    synTensor      layer4_bn_grad_input =
        createTensor(4U, syn_type_bf16, layer4_bn_grad_input_sizes, false, "layer4_bn_grad_input");

    // create layer4_bn_bias_grad tensor
    const unsigned layer4_bn_bias_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_bn_bias_grad_dram    = 0;
    synLaunchTensorInfo layer4_bn_bias_grad_tr_info = {"layer4_bn_bias_grad", layer4_bn_bias_grad_dram};
    synTensor           layer4_bn_bias_grad =
        createTensor(1U, syn_type_single, layer4_bn_bias_grad_sizes, true, "layer4_bn_bias_grad");

    // create layer4_bn_weight_grad tensor
    const unsigned layer4_bn_weight_grad_sizes[] = {
        2048,
    };
    uint64_t            layer4_bn_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_bn_weight_grad_tr_info = {"layer4_bn_weight_grad", layer4_bn_weight_grad_dram};
    synTensor           layer4_bn_weight_grad =
        createTensor(1U, syn_type_single, layer4_bn_weight_grad_sizes, true, "layer4_bn_weight_grad");

    synTensor layer4_bn_bwd_out_vec[3] = {layer4_bn_grad_input, layer4_bn_bias_grad, layer4_bn_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_bn_bwd_in_vec,
                           layer4_bn_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer4_bn_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_bn_bwd failed!");

    /*************
     * layer4_downsample_dedx node
     * inputs: [layer4_bn_grad_input(64, 7, 7, 2048)(dtype=bf16), layer4_downsample_weight[1, 1, 1024,
     *2048](dtype=bf16)] output: [layer4_downsample_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer4_downsample_dedx_kernel_params;
    layer4_downsample_dedx_kernel_params.dH   = 2;
    layer4_downsample_dedx_kernel_params.dW   = 2;
    layer4_downsample_dedx_kernel_params.kH   = 1;
    layer4_downsample_dedx_kernel_params.kW   = 1;
    layer4_downsample_dedx_kernel_params.padT = 0;
    layer4_downsample_dedx_kernel_params.padB = 0;
    layer4_downsample_dedx_kernel_params.padL = 0;
    layer4_downsample_dedx_kernel_params.padR = 0;
    layer4_downsample_dedx_kernel_params.dilH = 1;
    layer4_downsample_dedx_kernel_params.dilW = 1;

    synTensor layer4_downsample_dedx_in_vec[2] = {layer4_bn_grad_input, layer4_downsample_weight};

    // create layer4_downsample_grad_input tensor
    const unsigned layer4_downsample_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer4_downsample_grad_input =
        createTensor(4U, syn_type_bf16, layer4_downsample_grad_input_sizes, false, "layer4_downsample_grad_input");

    synTensor layer4_downsample_dedx_out_vec[1] = {layer4_downsample_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_downsample_dedx_in_vec,
                           layer4_downsample_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer4_downsample_dedx_kernel_params,
                           sizeof(layer4_downsample_dedx_kernel_params),
                           "dedx",
                           "layer4_downsample_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample_dedx failed!");

    /*************
     * layer4_downsample_dedw node
     * inputs: [layer4_bn_grad_input(64, 7, 7, 2048)(dtype=bf16), layer3_5_relu3_output(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer4_downsample_weight_grad(1, 1, 1024, 2048)(dtype=float32)]
     *************/
    synConvolutionParams layer4_downsample_dedw_kernel_params;
    layer4_downsample_dedw_kernel_params.dH   = 2;
    layer4_downsample_dedw_kernel_params.dW   = 2;
    layer4_downsample_dedw_kernel_params.kH   = 1;
    layer4_downsample_dedw_kernel_params.kW   = 1;
    layer4_downsample_dedw_kernel_params.padT = 0;
    layer4_downsample_dedw_kernel_params.padB = 0;
    layer4_downsample_dedw_kernel_params.padL = 0;
    layer4_downsample_dedw_kernel_params.padR = 0;
    layer4_downsample_dedw_kernel_params.dilH = 1;
    layer4_downsample_dedw_kernel_params.dilW = 1;

    synTensor layer4_downsample_dedw_in_vec[2] = {layer4_bn_grad_input, layer3_5_relu3_output};

    // create layer4_downsample_weight_grad tensor
    const unsigned      layer4_downsample_weight_grad_sizes[] = {1, 1, 1024, 2048};
    uint64_t            layer4_downsample_weight_grad_dram    = 0;
    synLaunchTensorInfo layer4_downsample_weight_grad_tr_info = {"layer4_downsample_weight_grad",
                                                                 layer4_downsample_weight_grad_dram};
    synTensor           layer4_downsample_weight_grad =
        createTensor(4U, syn_type_single, layer4_downsample_weight_grad_sizes, true, "layer4_downsample_weight_grad");

    synTensor layer4_downsample_dedw_out_vec[1] = {layer4_downsample_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer4_downsample_dedw_in_vec,
                           layer4_downsample_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer4_downsample_dedw_kernel_params,
                           sizeof(layer4_downsample_dedw_kernel_params),
                           "dedw",
                           "layer4_downsample_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_downsample_dedw failed!");

    /*************
     * layer4_0_add_residual_fwd1 node
     * inputs: [layer4_0_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16), layer4_downsample_grad_input(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer4_0_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer4_0_add_residual_fwd1_in_vec[2] = {layer4_0_conv1_grad_input, layer4_downsample_grad_input};

    // create layer4_0_residual_upstream_grad_input tensor
    const unsigned layer4_0_residual_upstream_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer4_0_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer4_0_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer4_0_residual_upstream_grad_input");

    synTensor layer4_0_add_residual_fwd1_out_vec[1] = {layer4_0_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer4_0_add_residual_fwd1_in_vec,
                           layer4_0_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer4_0_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer4_0_add_residual_fwd1 failed!");

    /*************
     * layer3_5_relu3_bwd node
     * inputs: [layer4_0_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_5_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_5_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_5_relu3_bwd_in_vec[2] = {layer4_0_residual_upstream_grad_input, layer3_5_relu3_output};

    // create layer3_5_relu3_grad_input tensor
    const unsigned layer3_5_relu3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_relu3_grad_input_sizes, false, "layer3_5_relu3_grad_input");

    synTensor layer3_5_relu3_bwd_out_vec[1] = {layer3_5_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_5_relu3_bwd_in_vec,
                           layer3_5_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_5_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_relu3_bwd failed!");

    /*************
     * layer3_5_add_residual_bwd node
     * inputs: [layer3_5_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_5_add_residual_grad_input0(64, 14, 14, 1024)(dtype=bf16), layer3_5_add_residual_grad_input1(64,
     *14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_5_add_residual_bwd_in_vec[1] = {layer3_5_relu3_grad_input};

    // create layer3_5_add_residual_grad_input0 tensor
    const unsigned layer3_5_add_residual_grad_input0_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_5_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer3_5_add_residual_grad_input0");

    // create layer3_5_add_residual_grad_input1 tensor
    const unsigned layer3_5_add_residual_grad_input1_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_5_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer3_5_add_residual_grad_input1");

    synTensor layer3_5_add_residual_bwd_out_vec[2] = {layer3_5_add_residual_grad_input0,
                                                      layer3_5_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer3_5_add_residual_bwd_in_vec,
                           layer3_5_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer3_5_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_add_residual_bwd failed!");

    /*************
     * layer3_5_bn3_bwd node
     * inputs: [layer3_5_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_5_add_residual_grad_input0(64, 14, 14,
     *1024)(dtype=bf16), layer3_5_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_5_bn3_saved_var(1, 1, 1,
     *1024)(dtype=float32), layer3_5_bn3_weight[1024](dtype=float32)] output: [layer3_5_bn3_grad_input(64, 14, 14,
     *1024)(dtype=bf16), layer3_5_bn3_bias_grad(1024,)(dtype=float32), layer3_5_bn3_weight_grad(1024,)(dtype=float32)]
     *************/

    synTensor layer3_5_bn3_bwd_in_vec[5] = {layer3_5_conv3_output,
                                            layer3_5_add_residual_grad_input0,
                                            layer3_5_bn3_saved_mean,
                                            layer3_5_bn3_saved_var,
                                            layer3_5_bn3_weight};

    // create layer3_5_bn3_grad_input tensor
    const unsigned layer3_5_bn3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_bn3_grad_input_sizes, false, "layer3_5_bn3_grad_input");

    // create layer3_5_bn3_bias_grad tensor
    const unsigned layer3_5_bn3_bias_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_5_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_bn3_bias_grad_tr_info = {"layer3_5_bn3_bias_grad", layer3_5_bn3_bias_grad_dram};
    synTensor           layer3_5_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer3_5_bn3_bias_grad_sizes, true, "layer3_5_bn3_bias_grad");

    // create layer3_5_bn3_weight_grad tensor
    const unsigned layer3_5_bn3_weight_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_5_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_bn3_weight_grad_tr_info = {"layer3_5_bn3_weight_grad", layer3_5_bn3_weight_grad_dram};
    synTensor           layer3_5_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer3_5_bn3_weight_grad_sizes, true, "layer3_5_bn3_weight_grad");

    synTensor layer3_5_bn3_bwd_out_vec[3] = {layer3_5_bn3_grad_input, layer3_5_bn3_bias_grad, layer3_5_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_5_bn3_bwd_in_vec,
                           layer3_5_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_5_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_bn3_bwd failed!");

    /*************
     * layer3_5_conv3_dedx node
     * inputs: [layer3_5_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_5_conv3_weight[1, 1, 256,
     *1024](dtype=bf16)] output: [layer3_5_conv3_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_5_conv3_dedx_kernel_params;
    layer3_5_conv3_dedx_kernel_params.dH   = 1;
    layer3_5_conv3_dedx_kernel_params.dW   = 1;
    layer3_5_conv3_dedx_kernel_params.kH   = 1;
    layer3_5_conv3_dedx_kernel_params.kW   = 1;
    layer3_5_conv3_dedx_kernel_params.padT = 0;
    layer3_5_conv3_dedx_kernel_params.padB = 0;
    layer3_5_conv3_dedx_kernel_params.padL = 0;
    layer3_5_conv3_dedx_kernel_params.padR = 0;
    layer3_5_conv3_dedx_kernel_params.dilH = 1;
    layer3_5_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer3_5_conv3_dedx_in_vec[2] = {layer3_5_bn3_grad_input, layer3_5_conv3_weight};

    // create layer3_5_conv3_grad_input tensor
    const unsigned layer3_5_conv3_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_conv3_grad_input_sizes, false, "layer3_5_conv3_grad_input");

    synTensor layer3_5_conv3_dedx_out_vec[1] = {layer3_5_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv3_dedx_in_vec,
                           layer3_5_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_5_conv3_dedx_kernel_params,
                           sizeof(layer3_5_conv3_dedx_kernel_params),
                           "dedx",
                           "layer3_5_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv3_dedx failed!");

    /*************
     * layer3_5_conv3_dedw node
     * inputs: [layer3_5_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_5_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_5_conv3_weight_grad(1, 1, 256, 1024)(dtype=float32)]
     *************/
    synConvolutionParams layer3_5_conv3_dedw_kernel_params;
    layer3_5_conv3_dedw_kernel_params.dH   = 1;
    layer3_5_conv3_dedw_kernel_params.dW   = 1;
    layer3_5_conv3_dedw_kernel_params.kH   = 1;
    layer3_5_conv3_dedw_kernel_params.kW   = 1;
    layer3_5_conv3_dedw_kernel_params.padT = 0;
    layer3_5_conv3_dedw_kernel_params.padB = 0;
    layer3_5_conv3_dedw_kernel_params.padL = 0;
    layer3_5_conv3_dedw_kernel_params.padR = 0;
    layer3_5_conv3_dedw_kernel_params.dilH = 1;
    layer3_5_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer3_5_conv3_dedw_in_vec[2] = {layer3_5_bn3_grad_input, layer3_5_relu2_output};

    // create layer3_5_conv3_weight_grad tensor
    const unsigned      layer3_5_conv3_weight_grad_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_5_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_conv3_weight_grad_tr_info = {"layer3_5_conv3_weight_grad",
                                                              layer3_5_conv3_weight_grad_dram};
    synTensor           layer3_5_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer3_5_conv3_weight_grad_sizes, true, "layer3_5_conv3_weight_grad");

    synTensor layer3_5_conv3_dedw_out_vec[1] = {layer3_5_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv3_dedw_in_vec,
                           layer3_5_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_5_conv3_dedw_kernel_params,
                           sizeof(layer3_5_conv3_dedw_kernel_params),
                           "dedw",
                           "layer3_5_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv3_dedw failed!");

    /*************
     * layer3_5_relu2_bwd node
     * inputs: [layer3_5_conv3_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_5_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_5_relu2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_5_relu2_bwd_in_vec[2] = {layer3_5_conv3_grad_input, layer3_5_relu2_output};

    // create layer3_5_relu2_grad_input tensor
    const unsigned layer3_5_relu2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_relu2_grad_input_sizes, false, "layer3_5_relu2_grad_input");

    synTensor layer3_5_relu2_bwd_out_vec[1] = {layer3_5_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_5_relu2_bwd_in_vec,
                           layer3_5_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_5_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_relu2_bwd failed!");

    /*************
     * layer3_5_bn2_bwd node
     * inputs: [layer3_5_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_5_relu2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_5_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_5_bn2_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_5_bn2_weight[256](dtype=float32)] output: [layer3_5_bn2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_5_bn2_bias_grad(256,)(dtype=float32), layer3_5_bn2_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_5_bn2_bwd_in_vec[5] = {layer3_5_conv2_output,
                                            layer3_5_relu2_grad_input,
                                            layer3_5_bn2_saved_mean,
                                            layer3_5_bn2_saved_var,
                                            layer3_5_bn2_weight};

    // create layer3_5_bn2_grad_input tensor
    const unsigned layer3_5_bn2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_bn2_grad_input_sizes, false, "layer3_5_bn2_grad_input");

    // create layer3_5_bn2_bias_grad tensor
    const unsigned layer3_5_bn2_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_bn2_bias_grad_tr_info = {"layer3_5_bn2_bias_grad", layer3_5_bn2_bias_grad_dram};
    synTensor           layer3_5_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer3_5_bn2_bias_grad_sizes, true, "layer3_5_bn2_bias_grad");

    // create layer3_5_bn2_weight_grad tensor
    const unsigned layer3_5_bn2_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_bn2_weight_grad_tr_info = {"layer3_5_bn2_weight_grad", layer3_5_bn2_weight_grad_dram};
    synTensor           layer3_5_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer3_5_bn2_weight_grad_sizes, true, "layer3_5_bn2_weight_grad");

    synTensor layer3_5_bn2_bwd_out_vec[3] = {layer3_5_bn2_grad_input, layer3_5_bn2_bias_grad, layer3_5_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_5_bn2_bwd_in_vec,
                           layer3_5_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_5_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_bn2_bwd failed!");

    /*************
     * layer3_5_conv2_dedx node
     * inputs: [layer3_5_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_5_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_5_conv2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_5_conv2_dedx_kernel_params;
    layer3_5_conv2_dedx_kernel_params.dH   = 1;
    layer3_5_conv2_dedx_kernel_params.dW   = 1;
    layer3_5_conv2_dedx_kernel_params.kH   = 3;
    layer3_5_conv2_dedx_kernel_params.kW   = 3;
    layer3_5_conv2_dedx_kernel_params.padT = 1;
    layer3_5_conv2_dedx_kernel_params.padB = 1;
    layer3_5_conv2_dedx_kernel_params.padL = 1;
    layer3_5_conv2_dedx_kernel_params.padR = 1;
    layer3_5_conv2_dedx_kernel_params.dilH = 1;
    layer3_5_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer3_5_conv2_dedx_in_vec[2] = {layer3_5_bn2_grad_input, layer3_5_conv2_weight};

    // create layer3_5_conv2_grad_input tensor
    const unsigned layer3_5_conv2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_conv2_grad_input_sizes, false, "layer3_5_conv2_grad_input");

    synTensor layer3_5_conv2_dedx_out_vec[1] = {layer3_5_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv2_dedx_in_vec,
                           layer3_5_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_5_conv2_dedx_kernel_params,
                           sizeof(layer3_5_conv2_dedx_kernel_params),
                           "dedx",
                           "layer3_5_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv2_dedx failed!");

    /*************
     * layer3_5_conv2_dedw node
     * inputs: [layer3_5_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_5_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_5_conv2_weight_grad(3, 3, 256, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_5_conv2_dedw_kernel_params;
    layer3_5_conv2_dedw_kernel_params.dH   = 1;
    layer3_5_conv2_dedw_kernel_params.dW   = 1;
    layer3_5_conv2_dedw_kernel_params.kH   = 3;
    layer3_5_conv2_dedw_kernel_params.kW   = 3;
    layer3_5_conv2_dedw_kernel_params.padT = 1;
    layer3_5_conv2_dedw_kernel_params.padB = 1;
    layer3_5_conv2_dedw_kernel_params.padL = 1;
    layer3_5_conv2_dedw_kernel_params.padR = 1;
    layer3_5_conv2_dedw_kernel_params.dilH = 1;
    layer3_5_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer3_5_conv2_dedw_in_vec[2] = {layer3_5_bn2_grad_input, layer3_5_relu1_output};

    // create layer3_5_conv2_weight_grad tensor
    const unsigned      layer3_5_conv2_weight_grad_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_5_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_conv2_weight_grad_tr_info = {"layer3_5_conv2_weight_grad",
                                                              layer3_5_conv2_weight_grad_dram};
    synTensor           layer3_5_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer3_5_conv2_weight_grad_sizes, true, "layer3_5_conv2_weight_grad");

    synTensor layer3_5_conv2_dedw_out_vec[1] = {layer3_5_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv2_dedw_in_vec,
                           layer3_5_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_5_conv2_dedw_kernel_params,
                           sizeof(layer3_5_conv2_dedw_kernel_params),
                           "dedw",
                           "layer3_5_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv2_dedw failed!");

    /*************
     * layer3_5_relu1_bwd node
     * inputs: [layer3_5_conv2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_5_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_5_relu1_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_5_relu1_bwd_in_vec[2] = {layer3_5_conv2_grad_input, layer3_5_relu1_output};

    // create layer3_5_relu1_grad_input tensor
    const unsigned layer3_5_relu1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_relu1_grad_input_sizes, false, "layer3_5_relu1_grad_input");

    synTensor layer3_5_relu1_bwd_out_vec[1] = {layer3_5_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_5_relu1_bwd_in_vec,
                           layer3_5_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_5_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_relu1_bwd failed!");

    /*************
     * layer3_5_bn1_bwd node
     * inputs: [layer3_5_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_5_relu1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_5_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_5_bn1_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_5_bn1_weight[256](dtype=float32)] output: [layer3_5_bn1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_5_bn1_bias_grad(256,)(dtype=float32), layer3_5_bn1_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_5_bn1_bwd_in_vec[5] = {layer3_5_conv1_output,
                                            layer3_5_relu1_grad_input,
                                            layer3_5_bn1_saved_mean,
                                            layer3_5_bn1_saved_var,
                                            layer3_5_bn1_weight};

    // create layer3_5_bn1_grad_input tensor
    const unsigned layer3_5_bn1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_5_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_bn1_grad_input_sizes, false, "layer3_5_bn1_grad_input");

    // create layer3_5_bn1_bias_grad tensor
    const unsigned layer3_5_bn1_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_bn1_bias_grad_tr_info = {"layer3_5_bn1_bias_grad", layer3_5_bn1_bias_grad_dram};
    synTensor           layer3_5_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer3_5_bn1_bias_grad_sizes, true, "layer3_5_bn1_bias_grad");

    // create layer3_5_bn1_weight_grad tensor
    const unsigned layer3_5_bn1_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_5_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_bn1_weight_grad_tr_info = {"layer3_5_bn1_weight_grad", layer3_5_bn1_weight_grad_dram};
    synTensor           layer3_5_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer3_5_bn1_weight_grad_sizes, true, "layer3_5_bn1_weight_grad");

    synTensor layer3_5_bn1_bwd_out_vec[3] = {layer3_5_bn1_grad_input, layer3_5_bn1_bias_grad, layer3_5_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_5_bn1_bwd_in_vec,
                           layer3_5_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_5_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_bn1_bwd failed!");

    /*************
     * layer3_5_conv1_dedx node
     * inputs: [layer3_5_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_5_conv1_weight[1, 1, 1024,
     *256](dtype=bf16)] output: [layer3_5_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_5_conv1_dedx_kernel_params;
    layer3_5_conv1_dedx_kernel_params.dH   = 1;
    layer3_5_conv1_dedx_kernel_params.dW   = 1;
    layer3_5_conv1_dedx_kernel_params.kH   = 1;
    layer3_5_conv1_dedx_kernel_params.kW   = 1;
    layer3_5_conv1_dedx_kernel_params.padT = 0;
    layer3_5_conv1_dedx_kernel_params.padB = 0;
    layer3_5_conv1_dedx_kernel_params.padL = 0;
    layer3_5_conv1_dedx_kernel_params.padR = 0;
    layer3_5_conv1_dedx_kernel_params.dilH = 1;
    layer3_5_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer3_5_conv1_dedx_in_vec[2] = {layer3_5_bn1_grad_input, layer3_5_conv1_weight};

    // create layer3_5_conv1_grad_input tensor
    const unsigned layer3_5_conv1_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_5_conv1_grad_input_sizes, false, "layer3_5_conv1_grad_input");

    synTensor layer3_5_conv1_dedx_out_vec[1] = {layer3_5_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv1_dedx_in_vec,
                           layer3_5_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_5_conv1_dedx_kernel_params,
                           sizeof(layer3_5_conv1_dedx_kernel_params),
                           "dedx",
                           "layer3_5_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv1_dedx failed!");

    /*************
     * layer3_5_conv1_dedw node
     * inputs: [layer3_5_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_4_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_5_conv1_weight_grad(1, 1, 1024, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_5_conv1_dedw_kernel_params;
    layer3_5_conv1_dedw_kernel_params.dH   = 1;
    layer3_5_conv1_dedw_kernel_params.dW   = 1;
    layer3_5_conv1_dedw_kernel_params.kH   = 1;
    layer3_5_conv1_dedw_kernel_params.kW   = 1;
    layer3_5_conv1_dedw_kernel_params.padT = 0;
    layer3_5_conv1_dedw_kernel_params.padB = 0;
    layer3_5_conv1_dedw_kernel_params.padL = 0;
    layer3_5_conv1_dedw_kernel_params.padR = 0;
    layer3_5_conv1_dedw_kernel_params.dilH = 1;
    layer3_5_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer3_5_conv1_dedw_in_vec[2] = {layer3_5_bn1_grad_input, layer3_4_relu3_output};

    // create layer3_5_conv1_weight_grad tensor
    const unsigned      layer3_5_conv1_weight_grad_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_5_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_5_conv1_weight_grad_tr_info = {"layer3_5_conv1_weight_grad",
                                                              layer3_5_conv1_weight_grad_dram};
    synTensor           layer3_5_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer3_5_conv1_weight_grad_sizes, true, "layer3_5_conv1_weight_grad");

    synTensor layer3_5_conv1_dedw_out_vec[1] = {layer3_5_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_5_conv1_dedw_in_vec,
                           layer3_5_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_5_conv1_dedw_kernel_params,
                           sizeof(layer3_5_conv1_dedw_kernel_params),
                           "dedw",
                           "layer3_5_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_conv1_dedw failed!");

    /*************
     * layer3_5_add_residual_fwd1 node
     * inputs: [layer3_5_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_5_add_residual_grad_input1(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_5_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_5_add_residual_fwd1_in_vec[2] = {layer3_5_conv1_grad_input, layer3_5_add_residual_grad_input1};

    // create layer3_5_residual_upstream_grad_input tensor
    const unsigned layer3_5_residual_upstream_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_5_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer3_5_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer3_5_residual_upstream_grad_input");

    synTensor layer3_5_add_residual_fwd1_out_vec[1] = {layer3_5_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_5_add_residual_fwd1_in_vec,
                           layer3_5_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_5_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_5_add_residual_fwd1 failed!");

    /*************
     * layer3_4_relu3_bwd node
     * inputs: [layer3_5_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_4_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_4_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_4_relu3_bwd_in_vec[2] = {layer3_5_residual_upstream_grad_input, layer3_4_relu3_output};

    // create layer3_4_relu3_grad_input tensor
    const unsigned layer3_4_relu3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_relu3_grad_input_sizes, false, "layer3_4_relu3_grad_input");

    synTensor layer3_4_relu3_bwd_out_vec[1] = {layer3_4_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_4_relu3_bwd_in_vec,
                           layer3_4_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_4_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_relu3_bwd failed!");

    /*************
     * layer3_4_add_residual_bwd node
     * inputs: [layer3_4_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_4_add_residual_grad_input0(64, 14, 14, 1024)(dtype=bf16), layer3_4_add_residual_grad_input1(64,
     *14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_4_add_residual_bwd_in_vec[1] = {layer3_4_relu3_grad_input};

    // create layer3_4_add_residual_grad_input0 tensor
    const unsigned layer3_4_add_residual_grad_input0_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_4_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer3_4_add_residual_grad_input0");

    // create layer3_4_add_residual_grad_input1 tensor
    const unsigned layer3_4_add_residual_grad_input1_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_4_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer3_4_add_residual_grad_input1");

    synTensor layer3_4_add_residual_bwd_out_vec[2] = {layer3_4_add_residual_grad_input0,
                                                      layer3_4_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer3_4_add_residual_bwd_in_vec,
                           layer3_4_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer3_4_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_add_residual_bwd failed!");

    /*************
     * layer3_4_bn3_bwd node
     * inputs: [layer3_4_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_4_add_residual_grad_input0(64, 14, 14,
     *1024)(dtype=bf16), layer3_4_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_4_bn3_saved_var(1, 1, 1,
     *1024)(dtype=float32), layer3_4_bn3_weight[1024](dtype=float32)] output: [layer3_4_bn3_grad_input(64, 14, 14,
     *1024)(dtype=bf16), layer3_4_bn3_bias_grad(1024,)(dtype=float32), layer3_4_bn3_weight_grad(1024,)(dtype=float32)]
     *************/

    synTensor layer3_4_bn3_bwd_in_vec[5] = {layer3_4_conv3_output,
                                            layer3_4_add_residual_grad_input0,
                                            layer3_4_bn3_saved_mean,
                                            layer3_4_bn3_saved_var,
                                            layer3_4_bn3_weight};

    // create layer3_4_bn3_grad_input tensor
    const unsigned layer3_4_bn3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_bn3_grad_input_sizes, false, "layer3_4_bn3_grad_input");

    // create layer3_4_bn3_bias_grad tensor
    const unsigned layer3_4_bn3_bias_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_4_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_bn3_bias_grad_tr_info = {"layer3_4_bn3_bias_grad", layer3_4_bn3_bias_grad_dram};
    synTensor           layer3_4_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer3_4_bn3_bias_grad_sizes, true, "layer3_4_bn3_bias_grad");

    // create layer3_4_bn3_weight_grad tensor
    const unsigned layer3_4_bn3_weight_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_4_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_bn3_weight_grad_tr_info = {"layer3_4_bn3_weight_grad", layer3_4_bn3_weight_grad_dram};
    synTensor           layer3_4_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer3_4_bn3_weight_grad_sizes, true, "layer3_4_bn3_weight_grad");

    synTensor layer3_4_bn3_bwd_out_vec[3] = {layer3_4_bn3_grad_input, layer3_4_bn3_bias_grad, layer3_4_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_4_bn3_bwd_in_vec,
                           layer3_4_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_4_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_bn3_bwd failed!");

    /*************
     * layer3_4_conv3_dedx node
     * inputs: [layer3_4_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_4_conv3_weight[1, 1, 256,
     *1024](dtype=bf16)] output: [layer3_4_conv3_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_4_conv3_dedx_kernel_params;
    layer3_4_conv3_dedx_kernel_params.dH   = 1;
    layer3_4_conv3_dedx_kernel_params.dW   = 1;
    layer3_4_conv3_dedx_kernel_params.kH   = 1;
    layer3_4_conv3_dedx_kernel_params.kW   = 1;
    layer3_4_conv3_dedx_kernel_params.padT = 0;
    layer3_4_conv3_dedx_kernel_params.padB = 0;
    layer3_4_conv3_dedx_kernel_params.padL = 0;
    layer3_4_conv3_dedx_kernel_params.padR = 0;
    layer3_4_conv3_dedx_kernel_params.dilH = 1;
    layer3_4_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer3_4_conv3_dedx_in_vec[2] = {layer3_4_bn3_grad_input, layer3_4_conv3_weight};

    // create layer3_4_conv3_grad_input tensor
    const unsigned layer3_4_conv3_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_conv3_grad_input_sizes, false, "layer3_4_conv3_grad_input");

    synTensor layer3_4_conv3_dedx_out_vec[1] = {layer3_4_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv3_dedx_in_vec,
                           layer3_4_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_4_conv3_dedx_kernel_params,
                           sizeof(layer3_4_conv3_dedx_kernel_params),
                           "dedx",
                           "layer3_4_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv3_dedx failed!");

    /*************
     * layer3_4_conv3_dedw node
     * inputs: [layer3_4_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_4_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_4_conv3_weight_grad(1, 1, 256, 1024)(dtype=float32)]
     *************/
    synConvolutionParams layer3_4_conv3_dedw_kernel_params;
    layer3_4_conv3_dedw_kernel_params.dH   = 1;
    layer3_4_conv3_dedw_kernel_params.dW   = 1;
    layer3_4_conv3_dedw_kernel_params.kH   = 1;
    layer3_4_conv3_dedw_kernel_params.kW   = 1;
    layer3_4_conv3_dedw_kernel_params.padT = 0;
    layer3_4_conv3_dedw_kernel_params.padB = 0;
    layer3_4_conv3_dedw_kernel_params.padL = 0;
    layer3_4_conv3_dedw_kernel_params.padR = 0;
    layer3_4_conv3_dedw_kernel_params.dilH = 1;
    layer3_4_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer3_4_conv3_dedw_in_vec[2] = {layer3_4_bn3_grad_input, layer3_4_relu2_output};

    // create layer3_4_conv3_weight_grad tensor
    const unsigned      layer3_4_conv3_weight_grad_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_4_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_conv3_weight_grad_tr_info = {"layer3_4_conv3_weight_grad",
                                                              layer3_4_conv3_weight_grad_dram};
    synTensor           layer3_4_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer3_4_conv3_weight_grad_sizes, true, "layer3_4_conv3_weight_grad");

    synTensor layer3_4_conv3_dedw_out_vec[1] = {layer3_4_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv3_dedw_in_vec,
                           layer3_4_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_4_conv3_dedw_kernel_params,
                           sizeof(layer3_4_conv3_dedw_kernel_params),
                           "dedw",
                           "layer3_4_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv3_dedw failed!");

    /*************
     * layer3_4_relu2_bwd node
     * inputs: [layer3_4_conv3_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_4_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_4_relu2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_4_relu2_bwd_in_vec[2] = {layer3_4_conv3_grad_input, layer3_4_relu2_output};

    // create layer3_4_relu2_grad_input tensor
    const unsigned layer3_4_relu2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_relu2_grad_input_sizes, false, "layer3_4_relu2_grad_input");

    synTensor layer3_4_relu2_bwd_out_vec[1] = {layer3_4_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_4_relu2_bwd_in_vec,
                           layer3_4_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_4_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_relu2_bwd failed!");

    /*************
     * layer3_4_bn2_bwd node
     * inputs: [layer3_4_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_4_relu2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_4_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_4_bn2_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_4_bn2_weight[256](dtype=float32)] output: [layer3_4_bn2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_4_bn2_bias_grad(256,)(dtype=float32), layer3_4_bn2_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_4_bn2_bwd_in_vec[5] = {layer3_4_conv2_output,
                                            layer3_4_relu2_grad_input,
                                            layer3_4_bn2_saved_mean,
                                            layer3_4_bn2_saved_var,
                                            layer3_4_bn2_weight};

    // create layer3_4_bn2_grad_input tensor
    const unsigned layer3_4_bn2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_bn2_grad_input_sizes, false, "layer3_4_bn2_grad_input");

    // create layer3_4_bn2_bias_grad tensor
    const unsigned layer3_4_bn2_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_bn2_bias_grad_tr_info = {"layer3_4_bn2_bias_grad", layer3_4_bn2_bias_grad_dram};
    synTensor           layer3_4_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer3_4_bn2_bias_grad_sizes, true, "layer3_4_bn2_bias_grad");

    // create layer3_4_bn2_weight_grad tensor
    const unsigned layer3_4_bn2_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_bn2_weight_grad_tr_info = {"layer3_4_bn2_weight_grad", layer3_4_bn2_weight_grad_dram};
    synTensor           layer3_4_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer3_4_bn2_weight_grad_sizes, true, "layer3_4_bn2_weight_grad");

    synTensor layer3_4_bn2_bwd_out_vec[3] = {layer3_4_bn2_grad_input, layer3_4_bn2_bias_grad, layer3_4_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_4_bn2_bwd_in_vec,
                           layer3_4_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_4_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_bn2_bwd failed!");

    /*************
     * layer3_4_conv2_dedx node
     * inputs: [layer3_4_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_4_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_4_conv2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_4_conv2_dedx_kernel_params;
    layer3_4_conv2_dedx_kernel_params.dH   = 1;
    layer3_4_conv2_dedx_kernel_params.dW   = 1;
    layer3_4_conv2_dedx_kernel_params.kH   = 3;
    layer3_4_conv2_dedx_kernel_params.kW   = 3;
    layer3_4_conv2_dedx_kernel_params.padT = 1;
    layer3_4_conv2_dedx_kernel_params.padB = 1;
    layer3_4_conv2_dedx_kernel_params.padL = 1;
    layer3_4_conv2_dedx_kernel_params.padR = 1;
    layer3_4_conv2_dedx_kernel_params.dilH = 1;
    layer3_4_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer3_4_conv2_dedx_in_vec[2] = {layer3_4_bn2_grad_input, layer3_4_conv2_weight};

    // create layer3_4_conv2_grad_input tensor
    const unsigned layer3_4_conv2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_conv2_grad_input_sizes, false, "layer3_4_conv2_grad_input");

    synTensor layer3_4_conv2_dedx_out_vec[1] = {layer3_4_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv2_dedx_in_vec,
                           layer3_4_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_4_conv2_dedx_kernel_params,
                           sizeof(layer3_4_conv2_dedx_kernel_params),
                           "dedx",
                           "layer3_4_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv2_dedx failed!");

    /*************
     * layer3_4_conv2_dedw node
     * inputs: [layer3_4_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_4_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_4_conv2_weight_grad(3, 3, 256, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_4_conv2_dedw_kernel_params;
    layer3_4_conv2_dedw_kernel_params.dH   = 1;
    layer3_4_conv2_dedw_kernel_params.dW   = 1;
    layer3_4_conv2_dedw_kernel_params.kH   = 3;
    layer3_4_conv2_dedw_kernel_params.kW   = 3;
    layer3_4_conv2_dedw_kernel_params.padT = 1;
    layer3_4_conv2_dedw_kernel_params.padB = 1;
    layer3_4_conv2_dedw_kernel_params.padL = 1;
    layer3_4_conv2_dedw_kernel_params.padR = 1;
    layer3_4_conv2_dedw_kernel_params.dilH = 1;
    layer3_4_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer3_4_conv2_dedw_in_vec[2] = {layer3_4_bn2_grad_input, layer3_4_relu1_output};

    // create layer3_4_conv2_weight_grad tensor
    const unsigned      layer3_4_conv2_weight_grad_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_4_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_conv2_weight_grad_tr_info = {"layer3_4_conv2_weight_grad",
                                                              layer3_4_conv2_weight_grad_dram};
    synTensor           layer3_4_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer3_4_conv2_weight_grad_sizes, true, "layer3_4_conv2_weight_grad");

    synTensor layer3_4_conv2_dedw_out_vec[1] = {layer3_4_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv2_dedw_in_vec,
                           layer3_4_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_4_conv2_dedw_kernel_params,
                           sizeof(layer3_4_conv2_dedw_kernel_params),
                           "dedw",
                           "layer3_4_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv2_dedw failed!");

    /*************
     * layer3_4_relu1_bwd node
     * inputs: [layer3_4_conv2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_4_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_4_relu1_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_4_relu1_bwd_in_vec[2] = {layer3_4_conv2_grad_input, layer3_4_relu1_output};

    // create layer3_4_relu1_grad_input tensor
    const unsigned layer3_4_relu1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_relu1_grad_input_sizes, false, "layer3_4_relu1_grad_input");

    synTensor layer3_4_relu1_bwd_out_vec[1] = {layer3_4_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_4_relu1_bwd_in_vec,
                           layer3_4_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_4_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_relu1_bwd failed!");

    /*************
     * layer3_4_bn1_bwd node
     * inputs: [layer3_4_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_4_relu1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_4_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_4_bn1_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_4_bn1_weight[256](dtype=float32)] output: [layer3_4_bn1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_4_bn1_bias_grad(256,)(dtype=float32), layer3_4_bn1_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_4_bn1_bwd_in_vec[5] = {layer3_4_conv1_output,
                                            layer3_4_relu1_grad_input,
                                            layer3_4_bn1_saved_mean,
                                            layer3_4_bn1_saved_var,
                                            layer3_4_bn1_weight};

    // create layer3_4_bn1_grad_input tensor
    const unsigned layer3_4_bn1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_4_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_bn1_grad_input_sizes, false, "layer3_4_bn1_grad_input");

    // create layer3_4_bn1_bias_grad tensor
    const unsigned layer3_4_bn1_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_bn1_bias_grad_tr_info = {"layer3_4_bn1_bias_grad", layer3_4_bn1_bias_grad_dram};
    synTensor           layer3_4_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer3_4_bn1_bias_grad_sizes, true, "layer3_4_bn1_bias_grad");

    // create layer3_4_bn1_weight_grad tensor
    const unsigned layer3_4_bn1_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_4_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_bn1_weight_grad_tr_info = {"layer3_4_bn1_weight_grad", layer3_4_bn1_weight_grad_dram};
    synTensor           layer3_4_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer3_4_bn1_weight_grad_sizes, true, "layer3_4_bn1_weight_grad");

    synTensor layer3_4_bn1_bwd_out_vec[3] = {layer3_4_bn1_grad_input, layer3_4_bn1_bias_grad, layer3_4_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_4_bn1_bwd_in_vec,
                           layer3_4_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_4_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_bn1_bwd failed!");

    /*************
     * layer3_4_conv1_dedx node
     * inputs: [layer3_4_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_4_conv1_weight[1, 1, 1024,
     *256](dtype=bf16)] output: [layer3_4_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_4_conv1_dedx_kernel_params;
    layer3_4_conv1_dedx_kernel_params.dH   = 1;
    layer3_4_conv1_dedx_kernel_params.dW   = 1;
    layer3_4_conv1_dedx_kernel_params.kH   = 1;
    layer3_4_conv1_dedx_kernel_params.kW   = 1;
    layer3_4_conv1_dedx_kernel_params.padT = 0;
    layer3_4_conv1_dedx_kernel_params.padB = 0;
    layer3_4_conv1_dedx_kernel_params.padL = 0;
    layer3_4_conv1_dedx_kernel_params.padR = 0;
    layer3_4_conv1_dedx_kernel_params.dilH = 1;
    layer3_4_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer3_4_conv1_dedx_in_vec[2] = {layer3_4_bn1_grad_input, layer3_4_conv1_weight};

    // create layer3_4_conv1_grad_input tensor
    const unsigned layer3_4_conv1_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_4_conv1_grad_input_sizes, false, "layer3_4_conv1_grad_input");

    synTensor layer3_4_conv1_dedx_out_vec[1] = {layer3_4_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv1_dedx_in_vec,
                           layer3_4_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_4_conv1_dedx_kernel_params,
                           sizeof(layer3_4_conv1_dedx_kernel_params),
                           "dedx",
                           "layer3_4_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv1_dedx failed!");

    /*************
     * layer3_4_conv1_dedw node
     * inputs: [layer3_4_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_3_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_4_conv1_weight_grad(1, 1, 1024, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_4_conv1_dedw_kernel_params;
    layer3_4_conv1_dedw_kernel_params.dH   = 1;
    layer3_4_conv1_dedw_kernel_params.dW   = 1;
    layer3_4_conv1_dedw_kernel_params.kH   = 1;
    layer3_4_conv1_dedw_kernel_params.kW   = 1;
    layer3_4_conv1_dedw_kernel_params.padT = 0;
    layer3_4_conv1_dedw_kernel_params.padB = 0;
    layer3_4_conv1_dedw_kernel_params.padL = 0;
    layer3_4_conv1_dedw_kernel_params.padR = 0;
    layer3_4_conv1_dedw_kernel_params.dilH = 1;
    layer3_4_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer3_4_conv1_dedw_in_vec[2] = {layer3_4_bn1_grad_input, layer3_3_relu3_output};

    // create layer3_4_conv1_weight_grad tensor
    const unsigned      layer3_4_conv1_weight_grad_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_4_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_4_conv1_weight_grad_tr_info = {"layer3_4_conv1_weight_grad",
                                                              layer3_4_conv1_weight_grad_dram};
    synTensor           layer3_4_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer3_4_conv1_weight_grad_sizes, true, "layer3_4_conv1_weight_grad");

    synTensor layer3_4_conv1_dedw_out_vec[1] = {layer3_4_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_4_conv1_dedw_in_vec,
                           layer3_4_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_4_conv1_dedw_kernel_params,
                           sizeof(layer3_4_conv1_dedw_kernel_params),
                           "dedw",
                           "layer3_4_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_conv1_dedw failed!");

    /*************
     * layer3_4_add_residual_fwd1 node
     * inputs: [layer3_4_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_4_add_residual_grad_input1(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_4_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_4_add_residual_fwd1_in_vec[2] = {layer3_4_conv1_grad_input, layer3_4_add_residual_grad_input1};

    // create layer3_4_residual_upstream_grad_input tensor
    const unsigned layer3_4_residual_upstream_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_4_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer3_4_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer3_4_residual_upstream_grad_input");

    synTensor layer3_4_add_residual_fwd1_out_vec[1] = {layer3_4_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_4_add_residual_fwd1_in_vec,
                           layer3_4_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_4_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_4_add_residual_fwd1 failed!");

    /*************
     * layer3_3_relu3_bwd node
     * inputs: [layer3_4_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_3_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_3_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_3_relu3_bwd_in_vec[2] = {layer3_4_residual_upstream_grad_input, layer3_3_relu3_output};

    // create layer3_3_relu3_grad_input tensor
    const unsigned layer3_3_relu3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_relu3_grad_input_sizes, false, "layer3_3_relu3_grad_input");

    synTensor layer3_3_relu3_bwd_out_vec[1] = {layer3_3_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_3_relu3_bwd_in_vec,
                           layer3_3_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_3_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_relu3_bwd failed!");

    /*************
     * layer3_3_add_residual_bwd node
     * inputs: [layer3_3_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_3_add_residual_grad_input0(64, 14, 14, 1024)(dtype=bf16), layer3_3_add_residual_grad_input1(64,
     *14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_3_add_residual_bwd_in_vec[1] = {layer3_3_relu3_grad_input};

    // create layer3_3_add_residual_grad_input0 tensor
    const unsigned layer3_3_add_residual_grad_input0_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_3_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer3_3_add_residual_grad_input0");

    // create layer3_3_add_residual_grad_input1 tensor
    const unsigned layer3_3_add_residual_grad_input1_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_3_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer3_3_add_residual_grad_input1");

    synTensor layer3_3_add_residual_bwd_out_vec[2] = {layer3_3_add_residual_grad_input0,
                                                      layer3_3_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer3_3_add_residual_bwd_in_vec,
                           layer3_3_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer3_3_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_add_residual_bwd failed!");

    /*************
     * layer3_3_bn3_bwd node
     * inputs: [layer3_3_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_3_add_residual_grad_input0(64, 14, 14,
     *1024)(dtype=bf16), layer3_3_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_3_bn3_saved_var(1, 1, 1,
     *1024)(dtype=float32), layer3_3_bn3_weight[1024](dtype=float32)] output: [layer3_3_bn3_grad_input(64, 14, 14,
     *1024)(dtype=bf16), layer3_3_bn3_bias_grad(1024,)(dtype=float32), layer3_3_bn3_weight_grad(1024,)(dtype=float32)]
     *************/

    synTensor layer3_3_bn3_bwd_in_vec[5] = {layer3_3_conv3_output,
                                            layer3_3_add_residual_grad_input0,
                                            layer3_3_bn3_saved_mean,
                                            layer3_3_bn3_saved_var,
                                            layer3_3_bn3_weight};

    // create layer3_3_bn3_grad_input tensor
    const unsigned layer3_3_bn3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_bn3_grad_input_sizes, false, "layer3_3_bn3_grad_input");

    // create layer3_3_bn3_bias_grad tensor
    const unsigned layer3_3_bn3_bias_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_3_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_bn3_bias_grad_tr_info = {"layer3_3_bn3_bias_grad", layer3_3_bn3_bias_grad_dram};
    synTensor           layer3_3_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer3_3_bn3_bias_grad_sizes, true, "layer3_3_bn3_bias_grad");

    // create layer3_3_bn3_weight_grad tensor
    const unsigned layer3_3_bn3_weight_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_3_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_bn3_weight_grad_tr_info = {"layer3_3_bn3_weight_grad", layer3_3_bn3_weight_grad_dram};
    synTensor           layer3_3_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer3_3_bn3_weight_grad_sizes, true, "layer3_3_bn3_weight_grad");

    synTensor layer3_3_bn3_bwd_out_vec[3] = {layer3_3_bn3_grad_input, layer3_3_bn3_bias_grad, layer3_3_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_3_bn3_bwd_in_vec,
                           layer3_3_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_3_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_bn3_bwd failed!");

    /*************
     * layer3_3_conv3_dedx node
     * inputs: [layer3_3_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_3_conv3_weight[1, 1, 256,
     *1024](dtype=bf16)] output: [layer3_3_conv3_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_3_conv3_dedx_kernel_params;
    layer3_3_conv3_dedx_kernel_params.dH   = 1;
    layer3_3_conv3_dedx_kernel_params.dW   = 1;
    layer3_3_conv3_dedx_kernel_params.kH   = 1;
    layer3_3_conv3_dedx_kernel_params.kW   = 1;
    layer3_3_conv3_dedx_kernel_params.padT = 0;
    layer3_3_conv3_dedx_kernel_params.padB = 0;
    layer3_3_conv3_dedx_kernel_params.padL = 0;
    layer3_3_conv3_dedx_kernel_params.padR = 0;
    layer3_3_conv3_dedx_kernel_params.dilH = 1;
    layer3_3_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer3_3_conv3_dedx_in_vec[2] = {layer3_3_bn3_grad_input, layer3_3_conv3_weight};

    // create layer3_3_conv3_grad_input tensor
    const unsigned layer3_3_conv3_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_conv3_grad_input_sizes, false, "layer3_3_conv3_grad_input");

    synTensor layer3_3_conv3_dedx_out_vec[1] = {layer3_3_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv3_dedx_in_vec,
                           layer3_3_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_3_conv3_dedx_kernel_params,
                           sizeof(layer3_3_conv3_dedx_kernel_params),
                           "dedx",
                           "layer3_3_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv3_dedx failed!");

    /*************
     * layer3_3_conv3_dedw node
     * inputs: [layer3_3_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_3_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_3_conv3_weight_grad(1, 1, 256, 1024)(dtype=float32)]
     *************/
    synConvolutionParams layer3_3_conv3_dedw_kernel_params;
    layer3_3_conv3_dedw_kernel_params.dH   = 1;
    layer3_3_conv3_dedw_kernel_params.dW   = 1;
    layer3_3_conv3_dedw_kernel_params.kH   = 1;
    layer3_3_conv3_dedw_kernel_params.kW   = 1;
    layer3_3_conv3_dedw_kernel_params.padT = 0;
    layer3_3_conv3_dedw_kernel_params.padB = 0;
    layer3_3_conv3_dedw_kernel_params.padL = 0;
    layer3_3_conv3_dedw_kernel_params.padR = 0;
    layer3_3_conv3_dedw_kernel_params.dilH = 1;
    layer3_3_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer3_3_conv3_dedw_in_vec[2] = {layer3_3_bn3_grad_input, layer3_3_relu2_output};

    // create layer3_3_conv3_weight_grad tensor
    const unsigned      layer3_3_conv3_weight_grad_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_3_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_conv3_weight_grad_tr_info = {"layer3_3_conv3_weight_grad",
                                                              layer3_3_conv3_weight_grad_dram};
    synTensor           layer3_3_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer3_3_conv3_weight_grad_sizes, true, "layer3_3_conv3_weight_grad");

    synTensor layer3_3_conv3_dedw_out_vec[1] = {layer3_3_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv3_dedw_in_vec,
                           layer3_3_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_3_conv3_dedw_kernel_params,
                           sizeof(layer3_3_conv3_dedw_kernel_params),
                           "dedw",
                           "layer3_3_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv3_dedw failed!");

    /*************
     * layer3_3_relu2_bwd node
     * inputs: [layer3_3_conv3_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_3_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_3_relu2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_3_relu2_bwd_in_vec[2] = {layer3_3_conv3_grad_input, layer3_3_relu2_output};

    // create layer3_3_relu2_grad_input tensor
    const unsigned layer3_3_relu2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_relu2_grad_input_sizes, false, "layer3_3_relu2_grad_input");

    synTensor layer3_3_relu2_bwd_out_vec[1] = {layer3_3_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_3_relu2_bwd_in_vec,
                           layer3_3_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_3_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_relu2_bwd failed!");

    /*************
     * layer3_3_bn2_bwd node
     * inputs: [layer3_3_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_3_relu2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_3_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_3_bn2_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_3_bn2_weight[256](dtype=float32)] output: [layer3_3_bn2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_3_bn2_bias_grad(256,)(dtype=float32), layer3_3_bn2_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_3_bn2_bwd_in_vec[5] = {layer3_3_conv2_output,
                                            layer3_3_relu2_grad_input,
                                            layer3_3_bn2_saved_mean,
                                            layer3_3_bn2_saved_var,
                                            layer3_3_bn2_weight};

    // create layer3_3_bn2_grad_input tensor
    const unsigned layer3_3_bn2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_bn2_grad_input_sizes, false, "layer3_3_bn2_grad_input");

    // create layer3_3_bn2_bias_grad tensor
    const unsigned layer3_3_bn2_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_bn2_bias_grad_tr_info = {"layer3_3_bn2_bias_grad", layer3_3_bn2_bias_grad_dram};
    synTensor           layer3_3_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer3_3_bn2_bias_grad_sizes, true, "layer3_3_bn2_bias_grad");

    // create layer3_3_bn2_weight_grad tensor
    const unsigned layer3_3_bn2_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_bn2_weight_grad_tr_info = {"layer3_3_bn2_weight_grad", layer3_3_bn2_weight_grad_dram};
    synTensor           layer3_3_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer3_3_bn2_weight_grad_sizes, true, "layer3_3_bn2_weight_grad");

    synTensor layer3_3_bn2_bwd_out_vec[3] = {layer3_3_bn2_grad_input, layer3_3_bn2_bias_grad, layer3_3_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_3_bn2_bwd_in_vec,
                           layer3_3_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_3_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_bn2_bwd failed!");

    /*************
     * layer3_3_conv2_dedx node
     * inputs: [layer3_3_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_3_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_3_conv2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_3_conv2_dedx_kernel_params;
    layer3_3_conv2_dedx_kernel_params.dH   = 1;
    layer3_3_conv2_dedx_kernel_params.dW   = 1;
    layer3_3_conv2_dedx_kernel_params.kH   = 3;
    layer3_3_conv2_dedx_kernel_params.kW   = 3;
    layer3_3_conv2_dedx_kernel_params.padT = 1;
    layer3_3_conv2_dedx_kernel_params.padB = 1;
    layer3_3_conv2_dedx_kernel_params.padL = 1;
    layer3_3_conv2_dedx_kernel_params.padR = 1;
    layer3_3_conv2_dedx_kernel_params.dilH = 1;
    layer3_3_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer3_3_conv2_dedx_in_vec[2] = {layer3_3_bn2_grad_input, layer3_3_conv2_weight};

    // create layer3_3_conv2_grad_input tensor
    const unsigned layer3_3_conv2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_conv2_grad_input_sizes, false, "layer3_3_conv2_grad_input");

    synTensor layer3_3_conv2_dedx_out_vec[1] = {layer3_3_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv2_dedx_in_vec,
                           layer3_3_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_3_conv2_dedx_kernel_params,
                           sizeof(layer3_3_conv2_dedx_kernel_params),
                           "dedx",
                           "layer3_3_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv2_dedx failed!");

    /*************
     * layer3_3_conv2_dedw node
     * inputs: [layer3_3_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_3_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_3_conv2_weight_grad(3, 3, 256, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_3_conv2_dedw_kernel_params;
    layer3_3_conv2_dedw_kernel_params.dH   = 1;
    layer3_3_conv2_dedw_kernel_params.dW   = 1;
    layer3_3_conv2_dedw_kernel_params.kH   = 3;
    layer3_3_conv2_dedw_kernel_params.kW   = 3;
    layer3_3_conv2_dedw_kernel_params.padT = 1;
    layer3_3_conv2_dedw_kernel_params.padB = 1;
    layer3_3_conv2_dedw_kernel_params.padL = 1;
    layer3_3_conv2_dedw_kernel_params.padR = 1;
    layer3_3_conv2_dedw_kernel_params.dilH = 1;
    layer3_3_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer3_3_conv2_dedw_in_vec[2] = {layer3_3_bn2_grad_input, layer3_3_relu1_output};

    // create layer3_3_conv2_weight_grad tensor
    const unsigned      layer3_3_conv2_weight_grad_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_3_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_conv2_weight_grad_tr_info = {"layer3_3_conv2_weight_grad",
                                                              layer3_3_conv2_weight_grad_dram};
    synTensor           layer3_3_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer3_3_conv2_weight_grad_sizes, true, "layer3_3_conv2_weight_grad");

    synTensor layer3_3_conv2_dedw_out_vec[1] = {layer3_3_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv2_dedw_in_vec,
                           layer3_3_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_3_conv2_dedw_kernel_params,
                           sizeof(layer3_3_conv2_dedw_kernel_params),
                           "dedw",
                           "layer3_3_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv2_dedw failed!");

    /*************
     * layer3_3_relu1_bwd node
     * inputs: [layer3_3_conv2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_3_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_3_relu1_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_3_relu1_bwd_in_vec[2] = {layer3_3_conv2_grad_input, layer3_3_relu1_output};

    // create layer3_3_relu1_grad_input tensor
    const unsigned layer3_3_relu1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_relu1_grad_input_sizes, false, "layer3_3_relu1_grad_input");

    synTensor layer3_3_relu1_bwd_out_vec[1] = {layer3_3_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_3_relu1_bwd_in_vec,
                           layer3_3_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_3_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_relu1_bwd failed!");

    /*************
     * layer3_3_bn1_bwd node
     * inputs: [layer3_3_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_3_relu1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_3_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_3_bn1_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_3_bn1_weight[256](dtype=float32)] output: [layer3_3_bn1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_3_bn1_bias_grad(256,)(dtype=float32), layer3_3_bn1_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_3_bn1_bwd_in_vec[5] = {layer3_3_conv1_output,
                                            layer3_3_relu1_grad_input,
                                            layer3_3_bn1_saved_mean,
                                            layer3_3_bn1_saved_var,
                                            layer3_3_bn1_weight};

    // create layer3_3_bn1_grad_input tensor
    const unsigned layer3_3_bn1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_3_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_bn1_grad_input_sizes, false, "layer3_3_bn1_grad_input");

    // create layer3_3_bn1_bias_grad tensor
    const unsigned layer3_3_bn1_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_bn1_bias_grad_tr_info = {"layer3_3_bn1_bias_grad", layer3_3_bn1_bias_grad_dram};
    synTensor           layer3_3_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer3_3_bn1_bias_grad_sizes, true, "layer3_3_bn1_bias_grad");

    // create layer3_3_bn1_weight_grad tensor
    const unsigned layer3_3_bn1_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_3_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_bn1_weight_grad_tr_info = {"layer3_3_bn1_weight_grad", layer3_3_bn1_weight_grad_dram};
    synTensor           layer3_3_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer3_3_bn1_weight_grad_sizes, true, "layer3_3_bn1_weight_grad");

    synTensor layer3_3_bn1_bwd_out_vec[3] = {layer3_3_bn1_grad_input, layer3_3_bn1_bias_grad, layer3_3_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_3_bn1_bwd_in_vec,
                           layer3_3_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_3_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_bn1_bwd failed!");

    /*************
     * layer3_3_conv1_dedx node
     * inputs: [layer3_3_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_3_conv1_weight[1, 1, 1024,
     *256](dtype=bf16)] output: [layer3_3_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_3_conv1_dedx_kernel_params;
    layer3_3_conv1_dedx_kernel_params.dH   = 1;
    layer3_3_conv1_dedx_kernel_params.dW   = 1;
    layer3_3_conv1_dedx_kernel_params.kH   = 1;
    layer3_3_conv1_dedx_kernel_params.kW   = 1;
    layer3_3_conv1_dedx_kernel_params.padT = 0;
    layer3_3_conv1_dedx_kernel_params.padB = 0;
    layer3_3_conv1_dedx_kernel_params.padL = 0;
    layer3_3_conv1_dedx_kernel_params.padR = 0;
    layer3_3_conv1_dedx_kernel_params.dilH = 1;
    layer3_3_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer3_3_conv1_dedx_in_vec[2] = {layer3_3_bn1_grad_input, layer3_3_conv1_weight};

    // create layer3_3_conv1_grad_input tensor
    const unsigned layer3_3_conv1_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_3_conv1_grad_input_sizes, false, "layer3_3_conv1_grad_input");

    synTensor layer3_3_conv1_dedx_out_vec[1] = {layer3_3_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv1_dedx_in_vec,
                           layer3_3_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_3_conv1_dedx_kernel_params,
                           sizeof(layer3_3_conv1_dedx_kernel_params),
                           "dedx",
                           "layer3_3_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv1_dedx failed!");

    /*************
     * layer3_3_conv1_dedw node
     * inputs: [layer3_3_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_2_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_3_conv1_weight_grad(1, 1, 1024, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_3_conv1_dedw_kernel_params;
    layer3_3_conv1_dedw_kernel_params.dH   = 1;
    layer3_3_conv1_dedw_kernel_params.dW   = 1;
    layer3_3_conv1_dedw_kernel_params.kH   = 1;
    layer3_3_conv1_dedw_kernel_params.kW   = 1;
    layer3_3_conv1_dedw_kernel_params.padT = 0;
    layer3_3_conv1_dedw_kernel_params.padB = 0;
    layer3_3_conv1_dedw_kernel_params.padL = 0;
    layer3_3_conv1_dedw_kernel_params.padR = 0;
    layer3_3_conv1_dedw_kernel_params.dilH = 1;
    layer3_3_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer3_3_conv1_dedw_in_vec[2] = {layer3_3_bn1_grad_input, layer3_2_relu3_output};

    // create layer3_3_conv1_weight_grad tensor
    const unsigned      layer3_3_conv1_weight_grad_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_3_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_3_conv1_weight_grad_tr_info = {"layer3_3_conv1_weight_grad",
                                                              layer3_3_conv1_weight_grad_dram};
    synTensor           layer3_3_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer3_3_conv1_weight_grad_sizes, true, "layer3_3_conv1_weight_grad");

    synTensor layer3_3_conv1_dedw_out_vec[1] = {layer3_3_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_3_conv1_dedw_in_vec,
                           layer3_3_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_3_conv1_dedw_kernel_params,
                           sizeof(layer3_3_conv1_dedw_kernel_params),
                           "dedw",
                           "layer3_3_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_conv1_dedw failed!");

    /*************
     * layer3_3_add_residual_fwd1 node
     * inputs: [layer3_3_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_3_add_residual_grad_input1(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_3_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_3_add_residual_fwd1_in_vec[2] = {layer3_3_conv1_grad_input, layer3_3_add_residual_grad_input1};

    // create layer3_3_residual_upstream_grad_input tensor
    const unsigned layer3_3_residual_upstream_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_3_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer3_3_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer3_3_residual_upstream_grad_input");

    synTensor layer3_3_add_residual_fwd1_out_vec[1] = {layer3_3_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_3_add_residual_fwd1_in_vec,
                           layer3_3_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_3_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_3_add_residual_fwd1 failed!");

    /*************
     * layer3_2_relu3_bwd node
     * inputs: [layer3_3_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_2_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_2_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_2_relu3_bwd_in_vec[2] = {layer3_3_residual_upstream_grad_input, layer3_2_relu3_output};

    // create layer3_2_relu3_grad_input tensor
    const unsigned layer3_2_relu3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_relu3_grad_input_sizes, false, "layer3_2_relu3_grad_input");

    synTensor layer3_2_relu3_bwd_out_vec[1] = {layer3_2_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_2_relu3_bwd_in_vec,
                           layer3_2_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_2_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_relu3_bwd failed!");

    /*************
     * layer3_2_add_residual_bwd node
     * inputs: [layer3_2_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_2_add_residual_grad_input0(64, 14, 14, 1024)(dtype=bf16), layer3_2_add_residual_grad_input1(64,
     *14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_2_add_residual_bwd_in_vec[1] = {layer3_2_relu3_grad_input};

    // create layer3_2_add_residual_grad_input0 tensor
    const unsigned layer3_2_add_residual_grad_input0_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_2_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer3_2_add_residual_grad_input0");

    // create layer3_2_add_residual_grad_input1 tensor
    const unsigned layer3_2_add_residual_grad_input1_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_2_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer3_2_add_residual_grad_input1");

    synTensor layer3_2_add_residual_bwd_out_vec[2] = {layer3_2_add_residual_grad_input0,
                                                      layer3_2_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer3_2_add_residual_bwd_in_vec,
                           layer3_2_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer3_2_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_add_residual_bwd failed!");

    /*************
     * layer3_2_bn3_bwd node
     * inputs: [layer3_2_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_2_add_residual_grad_input0(64, 14, 14,
     *1024)(dtype=bf16), layer3_2_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_2_bn3_saved_var(1, 1, 1,
     *1024)(dtype=float32), layer3_2_bn3_weight[1024](dtype=float32)] output: [layer3_2_bn3_grad_input(64, 14, 14,
     *1024)(dtype=bf16), layer3_2_bn3_bias_grad(1024,)(dtype=float32), layer3_2_bn3_weight_grad(1024,)(dtype=float32)]
     *************/

    synTensor layer3_2_bn3_bwd_in_vec[5] = {layer3_2_conv3_output,
                                            layer3_2_add_residual_grad_input0,
                                            layer3_2_bn3_saved_mean,
                                            layer3_2_bn3_saved_var,
                                            layer3_2_bn3_weight};

    // create layer3_2_bn3_grad_input tensor
    const unsigned layer3_2_bn3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_bn3_grad_input_sizes, false, "layer3_2_bn3_grad_input");

    // create layer3_2_bn3_bias_grad tensor
    const unsigned layer3_2_bn3_bias_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_2_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_bn3_bias_grad_tr_info = {"layer3_2_bn3_bias_grad", layer3_2_bn3_bias_grad_dram};
    synTensor           layer3_2_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer3_2_bn3_bias_grad_sizes, true, "layer3_2_bn3_bias_grad");

    // create layer3_2_bn3_weight_grad tensor
    const unsigned layer3_2_bn3_weight_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_2_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_bn3_weight_grad_tr_info = {"layer3_2_bn3_weight_grad", layer3_2_bn3_weight_grad_dram};
    synTensor           layer3_2_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer3_2_bn3_weight_grad_sizes, true, "layer3_2_bn3_weight_grad");

    synTensor layer3_2_bn3_bwd_out_vec[3] = {layer3_2_bn3_grad_input, layer3_2_bn3_bias_grad, layer3_2_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_2_bn3_bwd_in_vec,
                           layer3_2_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_2_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_bn3_bwd failed!");

    /*************
     * layer3_2_conv3_dedx node
     * inputs: [layer3_2_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_2_conv3_weight[1, 1, 256,
     *1024](dtype=bf16)] output: [layer3_2_conv3_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_2_conv3_dedx_kernel_params;
    layer3_2_conv3_dedx_kernel_params.dH   = 1;
    layer3_2_conv3_dedx_kernel_params.dW   = 1;
    layer3_2_conv3_dedx_kernel_params.kH   = 1;
    layer3_2_conv3_dedx_kernel_params.kW   = 1;
    layer3_2_conv3_dedx_kernel_params.padT = 0;
    layer3_2_conv3_dedx_kernel_params.padB = 0;
    layer3_2_conv3_dedx_kernel_params.padL = 0;
    layer3_2_conv3_dedx_kernel_params.padR = 0;
    layer3_2_conv3_dedx_kernel_params.dilH = 1;
    layer3_2_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer3_2_conv3_dedx_in_vec[2] = {layer3_2_bn3_grad_input, layer3_2_conv3_weight};

    // create layer3_2_conv3_grad_input tensor
    const unsigned layer3_2_conv3_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_conv3_grad_input_sizes, false, "layer3_2_conv3_grad_input");

    synTensor layer3_2_conv3_dedx_out_vec[1] = {layer3_2_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv3_dedx_in_vec,
                           layer3_2_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_2_conv3_dedx_kernel_params,
                           sizeof(layer3_2_conv3_dedx_kernel_params),
                           "dedx",
                           "layer3_2_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv3_dedx failed!");

    /*************
     * layer3_2_conv3_dedw node
     * inputs: [layer3_2_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_2_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_2_conv3_weight_grad(1, 1, 256, 1024)(dtype=float32)]
     *************/
    synConvolutionParams layer3_2_conv3_dedw_kernel_params;
    layer3_2_conv3_dedw_kernel_params.dH   = 1;
    layer3_2_conv3_dedw_kernel_params.dW   = 1;
    layer3_2_conv3_dedw_kernel_params.kH   = 1;
    layer3_2_conv3_dedw_kernel_params.kW   = 1;
    layer3_2_conv3_dedw_kernel_params.padT = 0;
    layer3_2_conv3_dedw_kernel_params.padB = 0;
    layer3_2_conv3_dedw_kernel_params.padL = 0;
    layer3_2_conv3_dedw_kernel_params.padR = 0;
    layer3_2_conv3_dedw_kernel_params.dilH = 1;
    layer3_2_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer3_2_conv3_dedw_in_vec[2] = {layer3_2_bn3_grad_input, layer3_2_relu2_output};

    // create layer3_2_conv3_weight_grad tensor
    const unsigned      layer3_2_conv3_weight_grad_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_2_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_conv3_weight_grad_tr_info = {"layer3_2_conv3_weight_grad",
                                                              layer3_2_conv3_weight_grad_dram};
    synTensor           layer3_2_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer3_2_conv3_weight_grad_sizes, true, "layer3_2_conv3_weight_grad");

    synTensor layer3_2_conv3_dedw_out_vec[1] = {layer3_2_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv3_dedw_in_vec,
                           layer3_2_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_2_conv3_dedw_kernel_params,
                           sizeof(layer3_2_conv3_dedw_kernel_params),
                           "dedw",
                           "layer3_2_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv3_dedw failed!");

    /*************
     * layer3_2_relu2_bwd node
     * inputs: [layer3_2_conv3_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_2_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_2_relu2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_2_relu2_bwd_in_vec[2] = {layer3_2_conv3_grad_input, layer3_2_relu2_output};

    // create layer3_2_relu2_grad_input tensor
    const unsigned layer3_2_relu2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_relu2_grad_input_sizes, false, "layer3_2_relu2_grad_input");

    synTensor layer3_2_relu2_bwd_out_vec[1] = {layer3_2_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_2_relu2_bwd_in_vec,
                           layer3_2_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_2_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_relu2_bwd failed!");

    /*************
     * layer3_2_bn2_bwd node
     * inputs: [layer3_2_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_2_relu2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_2_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_2_bn2_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_2_bn2_weight[256](dtype=float32)] output: [layer3_2_bn2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_2_bn2_bias_grad(256,)(dtype=float32), layer3_2_bn2_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_2_bn2_bwd_in_vec[5] = {layer3_2_conv2_output,
                                            layer3_2_relu2_grad_input,
                                            layer3_2_bn2_saved_mean,
                                            layer3_2_bn2_saved_var,
                                            layer3_2_bn2_weight};

    // create layer3_2_bn2_grad_input tensor
    const unsigned layer3_2_bn2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_bn2_grad_input_sizes, false, "layer3_2_bn2_grad_input");

    // create layer3_2_bn2_bias_grad tensor
    const unsigned layer3_2_bn2_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_bn2_bias_grad_tr_info = {"layer3_2_bn2_bias_grad", layer3_2_bn2_bias_grad_dram};
    synTensor           layer3_2_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer3_2_bn2_bias_grad_sizes, true, "layer3_2_bn2_bias_grad");

    // create layer3_2_bn2_weight_grad tensor
    const unsigned layer3_2_bn2_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_bn2_weight_grad_tr_info = {"layer3_2_bn2_weight_grad", layer3_2_bn2_weight_grad_dram};
    synTensor           layer3_2_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer3_2_bn2_weight_grad_sizes, true, "layer3_2_bn2_weight_grad");

    synTensor layer3_2_bn2_bwd_out_vec[3] = {layer3_2_bn2_grad_input, layer3_2_bn2_bias_grad, layer3_2_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_2_bn2_bwd_in_vec,
                           layer3_2_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_2_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_bn2_bwd failed!");

    /*************
     * layer3_2_conv2_dedx node
     * inputs: [layer3_2_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_2_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_2_conv2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_2_conv2_dedx_kernel_params;
    layer3_2_conv2_dedx_kernel_params.dH   = 1;
    layer3_2_conv2_dedx_kernel_params.dW   = 1;
    layer3_2_conv2_dedx_kernel_params.kH   = 3;
    layer3_2_conv2_dedx_kernel_params.kW   = 3;
    layer3_2_conv2_dedx_kernel_params.padT = 1;
    layer3_2_conv2_dedx_kernel_params.padB = 1;
    layer3_2_conv2_dedx_kernel_params.padL = 1;
    layer3_2_conv2_dedx_kernel_params.padR = 1;
    layer3_2_conv2_dedx_kernel_params.dilH = 1;
    layer3_2_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer3_2_conv2_dedx_in_vec[2] = {layer3_2_bn2_grad_input, layer3_2_conv2_weight};

    // create layer3_2_conv2_grad_input tensor
    const unsigned layer3_2_conv2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_conv2_grad_input_sizes, false, "layer3_2_conv2_grad_input");

    synTensor layer3_2_conv2_dedx_out_vec[1] = {layer3_2_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv2_dedx_in_vec,
                           layer3_2_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_2_conv2_dedx_kernel_params,
                           sizeof(layer3_2_conv2_dedx_kernel_params),
                           "dedx",
                           "layer3_2_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv2_dedx failed!");

    /*************
     * layer3_2_conv2_dedw node
     * inputs: [layer3_2_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_2_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_2_conv2_weight_grad(3, 3, 256, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_2_conv2_dedw_kernel_params;
    layer3_2_conv2_dedw_kernel_params.dH   = 1;
    layer3_2_conv2_dedw_kernel_params.dW   = 1;
    layer3_2_conv2_dedw_kernel_params.kH   = 3;
    layer3_2_conv2_dedw_kernel_params.kW   = 3;
    layer3_2_conv2_dedw_kernel_params.padT = 1;
    layer3_2_conv2_dedw_kernel_params.padB = 1;
    layer3_2_conv2_dedw_kernel_params.padL = 1;
    layer3_2_conv2_dedw_kernel_params.padR = 1;
    layer3_2_conv2_dedw_kernel_params.dilH = 1;
    layer3_2_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer3_2_conv2_dedw_in_vec[2] = {layer3_2_bn2_grad_input, layer3_2_relu1_output};

    // create layer3_2_conv2_weight_grad tensor
    const unsigned      layer3_2_conv2_weight_grad_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_2_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_conv2_weight_grad_tr_info = {"layer3_2_conv2_weight_grad",
                                                              layer3_2_conv2_weight_grad_dram};
    synTensor           layer3_2_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer3_2_conv2_weight_grad_sizes, true, "layer3_2_conv2_weight_grad");

    synTensor layer3_2_conv2_dedw_out_vec[1] = {layer3_2_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv2_dedw_in_vec,
                           layer3_2_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_2_conv2_dedw_kernel_params,
                           sizeof(layer3_2_conv2_dedw_kernel_params),
                           "dedw",
                           "layer3_2_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv2_dedw failed!");

    /*************
     * layer3_2_relu1_bwd node
     * inputs: [layer3_2_conv2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_2_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_2_relu1_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_2_relu1_bwd_in_vec[2] = {layer3_2_conv2_grad_input, layer3_2_relu1_output};

    // create layer3_2_relu1_grad_input tensor
    const unsigned layer3_2_relu1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_relu1_grad_input_sizes, false, "layer3_2_relu1_grad_input");

    synTensor layer3_2_relu1_bwd_out_vec[1] = {layer3_2_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_2_relu1_bwd_in_vec,
                           layer3_2_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_2_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_relu1_bwd failed!");

    /*************
     * layer3_2_bn1_bwd node
     * inputs: [layer3_2_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_2_relu1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_2_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_2_bn1_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_2_bn1_weight[256](dtype=float32)] output: [layer3_2_bn1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_2_bn1_bias_grad(256,)(dtype=float32), layer3_2_bn1_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_2_bn1_bwd_in_vec[5] = {layer3_2_conv1_output,
                                            layer3_2_relu1_grad_input,
                                            layer3_2_bn1_saved_mean,
                                            layer3_2_bn1_saved_var,
                                            layer3_2_bn1_weight};

    // create layer3_2_bn1_grad_input tensor
    const unsigned layer3_2_bn1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_2_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_bn1_grad_input_sizes, false, "layer3_2_bn1_grad_input");

    // create layer3_2_bn1_bias_grad tensor
    const unsigned layer3_2_bn1_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_bn1_bias_grad_tr_info = {"layer3_2_bn1_bias_grad", layer3_2_bn1_bias_grad_dram};
    synTensor           layer3_2_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer3_2_bn1_bias_grad_sizes, true, "layer3_2_bn1_bias_grad");

    // create layer3_2_bn1_weight_grad tensor
    const unsigned layer3_2_bn1_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_2_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_bn1_weight_grad_tr_info = {"layer3_2_bn1_weight_grad", layer3_2_bn1_weight_grad_dram};
    synTensor           layer3_2_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer3_2_bn1_weight_grad_sizes, true, "layer3_2_bn1_weight_grad");

    synTensor layer3_2_bn1_bwd_out_vec[3] = {layer3_2_bn1_grad_input, layer3_2_bn1_bias_grad, layer3_2_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_2_bn1_bwd_in_vec,
                           layer3_2_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_2_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_bn1_bwd failed!");

    /*************
     * layer3_2_conv1_dedx node
     * inputs: [layer3_2_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_2_conv1_weight[1, 1, 1024,
     *256](dtype=bf16)] output: [layer3_2_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_2_conv1_dedx_kernel_params;
    layer3_2_conv1_dedx_kernel_params.dH   = 1;
    layer3_2_conv1_dedx_kernel_params.dW   = 1;
    layer3_2_conv1_dedx_kernel_params.kH   = 1;
    layer3_2_conv1_dedx_kernel_params.kW   = 1;
    layer3_2_conv1_dedx_kernel_params.padT = 0;
    layer3_2_conv1_dedx_kernel_params.padB = 0;
    layer3_2_conv1_dedx_kernel_params.padL = 0;
    layer3_2_conv1_dedx_kernel_params.padR = 0;
    layer3_2_conv1_dedx_kernel_params.dilH = 1;
    layer3_2_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer3_2_conv1_dedx_in_vec[2] = {layer3_2_bn1_grad_input, layer3_2_conv1_weight};

    // create layer3_2_conv1_grad_input tensor
    const unsigned layer3_2_conv1_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_2_conv1_grad_input_sizes, false, "layer3_2_conv1_grad_input");

    synTensor layer3_2_conv1_dedx_out_vec[1] = {layer3_2_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv1_dedx_in_vec,
                           layer3_2_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_2_conv1_dedx_kernel_params,
                           sizeof(layer3_2_conv1_dedx_kernel_params),
                           "dedx",
                           "layer3_2_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv1_dedx failed!");

    /*************
     * layer3_2_conv1_dedw node
     * inputs: [layer3_2_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_1_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_2_conv1_weight_grad(1, 1, 1024, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_2_conv1_dedw_kernel_params;
    layer3_2_conv1_dedw_kernel_params.dH   = 1;
    layer3_2_conv1_dedw_kernel_params.dW   = 1;
    layer3_2_conv1_dedw_kernel_params.kH   = 1;
    layer3_2_conv1_dedw_kernel_params.kW   = 1;
    layer3_2_conv1_dedw_kernel_params.padT = 0;
    layer3_2_conv1_dedw_kernel_params.padB = 0;
    layer3_2_conv1_dedw_kernel_params.padL = 0;
    layer3_2_conv1_dedw_kernel_params.padR = 0;
    layer3_2_conv1_dedw_kernel_params.dilH = 1;
    layer3_2_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer3_2_conv1_dedw_in_vec[2] = {layer3_2_bn1_grad_input, layer3_1_relu3_output};

    // create layer3_2_conv1_weight_grad tensor
    const unsigned      layer3_2_conv1_weight_grad_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_2_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_2_conv1_weight_grad_tr_info = {"layer3_2_conv1_weight_grad",
                                                              layer3_2_conv1_weight_grad_dram};
    synTensor           layer3_2_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer3_2_conv1_weight_grad_sizes, true, "layer3_2_conv1_weight_grad");

    synTensor layer3_2_conv1_dedw_out_vec[1] = {layer3_2_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_2_conv1_dedw_in_vec,
                           layer3_2_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_2_conv1_dedw_kernel_params,
                           sizeof(layer3_2_conv1_dedw_kernel_params),
                           "dedw",
                           "layer3_2_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_conv1_dedw failed!");

    /*************
     * layer3_2_add_residual_fwd1 node
     * inputs: [layer3_2_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_2_add_residual_grad_input1(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_2_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_2_add_residual_fwd1_in_vec[2] = {layer3_2_conv1_grad_input, layer3_2_add_residual_grad_input1};

    // create layer3_2_residual_upstream_grad_input tensor
    const unsigned layer3_2_residual_upstream_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_2_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer3_2_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer3_2_residual_upstream_grad_input");

    synTensor layer3_2_add_residual_fwd1_out_vec[1] = {layer3_2_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_2_add_residual_fwd1_in_vec,
                           layer3_2_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_2_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_2_add_residual_fwd1 failed!");

    /*************
     * layer3_1_relu3_bwd node
     * inputs: [layer3_2_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_1_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_1_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_1_relu3_bwd_in_vec[2] = {layer3_2_residual_upstream_grad_input, layer3_1_relu3_output};

    // create layer3_1_relu3_grad_input tensor
    const unsigned layer3_1_relu3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_relu3_grad_input_sizes, false, "layer3_1_relu3_grad_input");

    synTensor layer3_1_relu3_bwd_out_vec[1] = {layer3_1_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_1_relu3_bwd_in_vec,
                           layer3_1_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_1_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_relu3_bwd failed!");

    /*************
     * layer3_1_add_residual_bwd node
     * inputs: [layer3_1_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_1_add_residual_grad_input0(64, 14, 14, 1024)(dtype=bf16), layer3_1_add_residual_grad_input1(64,
     *14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_1_add_residual_bwd_in_vec[1] = {layer3_1_relu3_grad_input};

    // create layer3_1_add_residual_grad_input0 tensor
    const unsigned layer3_1_add_residual_grad_input0_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_1_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer3_1_add_residual_grad_input0");

    // create layer3_1_add_residual_grad_input1 tensor
    const unsigned layer3_1_add_residual_grad_input1_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_1_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer3_1_add_residual_grad_input1");

    synTensor layer3_1_add_residual_bwd_out_vec[2] = {layer3_1_add_residual_grad_input0,
                                                      layer3_1_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer3_1_add_residual_bwd_in_vec,
                           layer3_1_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer3_1_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_add_residual_bwd failed!");

    /*************
     * layer3_1_bn3_bwd node
     * inputs: [layer3_1_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_1_add_residual_grad_input0(64, 14, 14,
     *1024)(dtype=bf16), layer3_1_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_1_bn3_saved_var(1, 1, 1,
     *1024)(dtype=float32), layer3_1_bn3_weight[1024](dtype=float32)] output: [layer3_1_bn3_grad_input(64, 14, 14,
     *1024)(dtype=bf16), layer3_1_bn3_bias_grad(1024,)(dtype=float32), layer3_1_bn3_weight_grad(1024,)(dtype=float32)]
     *************/

    synTensor layer3_1_bn3_bwd_in_vec[5] = {layer3_1_conv3_output,
                                            layer3_1_add_residual_grad_input0,
                                            layer3_1_bn3_saved_mean,
                                            layer3_1_bn3_saved_var,
                                            layer3_1_bn3_weight};

    // create layer3_1_bn3_grad_input tensor
    const unsigned layer3_1_bn3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_bn3_grad_input_sizes, false, "layer3_1_bn3_grad_input");

    // create layer3_1_bn3_bias_grad tensor
    const unsigned layer3_1_bn3_bias_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_1_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_bn3_bias_grad_tr_info = {"layer3_1_bn3_bias_grad", layer3_1_bn3_bias_grad_dram};
    synTensor           layer3_1_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer3_1_bn3_bias_grad_sizes, true, "layer3_1_bn3_bias_grad");

    // create layer3_1_bn3_weight_grad tensor
    const unsigned layer3_1_bn3_weight_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_1_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_bn3_weight_grad_tr_info = {"layer3_1_bn3_weight_grad", layer3_1_bn3_weight_grad_dram};
    synTensor           layer3_1_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer3_1_bn3_weight_grad_sizes, true, "layer3_1_bn3_weight_grad");

    synTensor layer3_1_bn3_bwd_out_vec[3] = {layer3_1_bn3_grad_input, layer3_1_bn3_bias_grad, layer3_1_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_1_bn3_bwd_in_vec,
                           layer3_1_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_1_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_bn3_bwd failed!");

    /*************
     * layer3_1_conv3_dedx node
     * inputs: [layer3_1_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_1_conv3_weight[1, 1, 256,
     *1024](dtype=bf16)] output: [layer3_1_conv3_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_1_conv3_dedx_kernel_params;
    layer3_1_conv3_dedx_kernel_params.dH   = 1;
    layer3_1_conv3_dedx_kernel_params.dW   = 1;
    layer3_1_conv3_dedx_kernel_params.kH   = 1;
    layer3_1_conv3_dedx_kernel_params.kW   = 1;
    layer3_1_conv3_dedx_kernel_params.padT = 0;
    layer3_1_conv3_dedx_kernel_params.padB = 0;
    layer3_1_conv3_dedx_kernel_params.padL = 0;
    layer3_1_conv3_dedx_kernel_params.padR = 0;
    layer3_1_conv3_dedx_kernel_params.dilH = 1;
    layer3_1_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer3_1_conv3_dedx_in_vec[2] = {layer3_1_bn3_grad_input, layer3_1_conv3_weight};

    // create layer3_1_conv3_grad_input tensor
    const unsigned layer3_1_conv3_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_conv3_grad_input_sizes, false, "layer3_1_conv3_grad_input");

    synTensor layer3_1_conv3_dedx_out_vec[1] = {layer3_1_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv3_dedx_in_vec,
                           layer3_1_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_1_conv3_dedx_kernel_params,
                           sizeof(layer3_1_conv3_dedx_kernel_params),
                           "dedx",
                           "layer3_1_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv3_dedx failed!");

    /*************
     * layer3_1_conv3_dedw node
     * inputs: [layer3_1_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_1_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_1_conv3_weight_grad(1, 1, 256, 1024)(dtype=float32)]
     *************/
    synConvolutionParams layer3_1_conv3_dedw_kernel_params;
    layer3_1_conv3_dedw_kernel_params.dH   = 1;
    layer3_1_conv3_dedw_kernel_params.dW   = 1;
    layer3_1_conv3_dedw_kernel_params.kH   = 1;
    layer3_1_conv3_dedw_kernel_params.kW   = 1;
    layer3_1_conv3_dedw_kernel_params.padT = 0;
    layer3_1_conv3_dedw_kernel_params.padB = 0;
    layer3_1_conv3_dedw_kernel_params.padL = 0;
    layer3_1_conv3_dedw_kernel_params.padR = 0;
    layer3_1_conv3_dedw_kernel_params.dilH = 1;
    layer3_1_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer3_1_conv3_dedw_in_vec[2] = {layer3_1_bn3_grad_input, layer3_1_relu2_output};

    // create layer3_1_conv3_weight_grad tensor
    const unsigned      layer3_1_conv3_weight_grad_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_1_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_conv3_weight_grad_tr_info = {"layer3_1_conv3_weight_grad",
                                                              layer3_1_conv3_weight_grad_dram};
    synTensor           layer3_1_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer3_1_conv3_weight_grad_sizes, true, "layer3_1_conv3_weight_grad");

    synTensor layer3_1_conv3_dedw_out_vec[1] = {layer3_1_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv3_dedw_in_vec,
                           layer3_1_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_1_conv3_dedw_kernel_params,
                           sizeof(layer3_1_conv3_dedw_kernel_params),
                           "dedw",
                           "layer3_1_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv3_dedw failed!");

    /*************
     * layer3_1_relu2_bwd node
     * inputs: [layer3_1_conv3_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_1_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_1_relu2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_1_relu2_bwd_in_vec[2] = {layer3_1_conv3_grad_input, layer3_1_relu2_output};

    // create layer3_1_relu2_grad_input tensor
    const unsigned layer3_1_relu2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_relu2_grad_input_sizes, false, "layer3_1_relu2_grad_input");

    synTensor layer3_1_relu2_bwd_out_vec[1] = {layer3_1_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_1_relu2_bwd_in_vec,
                           layer3_1_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_1_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_relu2_bwd failed!");

    /*************
     * layer3_1_bn2_bwd node
     * inputs: [layer3_1_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_1_relu2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_1_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_1_bn2_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_1_bn2_weight[256](dtype=float32)] output: [layer3_1_bn2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_1_bn2_bias_grad(256,)(dtype=float32), layer3_1_bn2_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_1_bn2_bwd_in_vec[5] = {layer3_1_conv2_output,
                                            layer3_1_relu2_grad_input,
                                            layer3_1_bn2_saved_mean,
                                            layer3_1_bn2_saved_var,
                                            layer3_1_bn2_weight};

    // create layer3_1_bn2_grad_input tensor
    const unsigned layer3_1_bn2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_bn2_grad_input_sizes, false, "layer3_1_bn2_grad_input");

    // create layer3_1_bn2_bias_grad tensor
    const unsigned layer3_1_bn2_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_bn2_bias_grad_tr_info = {"layer3_1_bn2_bias_grad", layer3_1_bn2_bias_grad_dram};
    synTensor           layer3_1_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer3_1_bn2_bias_grad_sizes, true, "layer3_1_bn2_bias_grad");

    // create layer3_1_bn2_weight_grad tensor
    const unsigned layer3_1_bn2_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_bn2_weight_grad_tr_info = {"layer3_1_bn2_weight_grad", layer3_1_bn2_weight_grad_dram};
    synTensor           layer3_1_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer3_1_bn2_weight_grad_sizes, true, "layer3_1_bn2_weight_grad");

    synTensor layer3_1_bn2_bwd_out_vec[3] = {layer3_1_bn2_grad_input, layer3_1_bn2_bias_grad, layer3_1_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_1_bn2_bwd_in_vec,
                           layer3_1_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_1_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_bn2_bwd failed!");

    /*************
     * layer3_1_conv2_dedx node
     * inputs: [layer3_1_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_1_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_1_conv2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_1_conv2_dedx_kernel_params;
    layer3_1_conv2_dedx_kernel_params.dH   = 1;
    layer3_1_conv2_dedx_kernel_params.dW   = 1;
    layer3_1_conv2_dedx_kernel_params.kH   = 3;
    layer3_1_conv2_dedx_kernel_params.kW   = 3;
    layer3_1_conv2_dedx_kernel_params.padT = 1;
    layer3_1_conv2_dedx_kernel_params.padB = 1;
    layer3_1_conv2_dedx_kernel_params.padL = 1;
    layer3_1_conv2_dedx_kernel_params.padR = 1;
    layer3_1_conv2_dedx_kernel_params.dilH = 1;
    layer3_1_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer3_1_conv2_dedx_in_vec[2] = {layer3_1_bn2_grad_input, layer3_1_conv2_weight};

    // create layer3_1_conv2_grad_input tensor
    const unsigned layer3_1_conv2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_conv2_grad_input_sizes, false, "layer3_1_conv2_grad_input");

    synTensor layer3_1_conv2_dedx_out_vec[1] = {layer3_1_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv2_dedx_in_vec,
                           layer3_1_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_1_conv2_dedx_kernel_params,
                           sizeof(layer3_1_conv2_dedx_kernel_params),
                           "dedx",
                           "layer3_1_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv2_dedx failed!");

    /*************
     * layer3_1_conv2_dedw node
     * inputs: [layer3_1_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_1_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_1_conv2_weight_grad(3, 3, 256, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_1_conv2_dedw_kernel_params;
    layer3_1_conv2_dedw_kernel_params.dH   = 1;
    layer3_1_conv2_dedw_kernel_params.dW   = 1;
    layer3_1_conv2_dedw_kernel_params.kH   = 3;
    layer3_1_conv2_dedw_kernel_params.kW   = 3;
    layer3_1_conv2_dedw_kernel_params.padT = 1;
    layer3_1_conv2_dedw_kernel_params.padB = 1;
    layer3_1_conv2_dedw_kernel_params.padL = 1;
    layer3_1_conv2_dedw_kernel_params.padR = 1;
    layer3_1_conv2_dedw_kernel_params.dilH = 1;
    layer3_1_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer3_1_conv2_dedw_in_vec[2] = {layer3_1_bn2_grad_input, layer3_1_relu1_output};

    // create layer3_1_conv2_weight_grad tensor
    const unsigned      layer3_1_conv2_weight_grad_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_1_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_conv2_weight_grad_tr_info = {"layer3_1_conv2_weight_grad",
                                                              layer3_1_conv2_weight_grad_dram};
    synTensor           layer3_1_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer3_1_conv2_weight_grad_sizes, true, "layer3_1_conv2_weight_grad");

    synTensor layer3_1_conv2_dedw_out_vec[1] = {layer3_1_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv2_dedw_in_vec,
                           layer3_1_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_1_conv2_dedw_kernel_params,
                           sizeof(layer3_1_conv2_dedw_kernel_params),
                           "dedw",
                           "layer3_1_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv2_dedw failed!");

    /*************
     * layer3_1_relu1_bwd node
     * inputs: [layer3_1_conv2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_1_relu1_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_1_relu1_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_1_relu1_bwd_in_vec[2] = {layer3_1_conv2_grad_input, layer3_1_relu1_output};

    // create layer3_1_relu1_grad_input tensor
    const unsigned layer3_1_relu1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_relu1_grad_input_sizes, false, "layer3_1_relu1_grad_input");

    synTensor layer3_1_relu1_bwd_out_vec[1] = {layer3_1_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_1_relu1_bwd_in_vec,
                           layer3_1_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_1_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_relu1_bwd failed!");

    /*************
     * layer3_1_bn1_bwd node
     * inputs: [layer3_1_conv1_output(64, 14, 14, 256)(dtype=bf16), layer3_1_relu1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_1_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_1_bn1_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_1_bn1_weight[256](dtype=float32)] output: [layer3_1_bn1_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_1_bn1_bias_grad(256,)(dtype=float32), layer3_1_bn1_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_1_bn1_bwd_in_vec[5] = {layer3_1_conv1_output,
                                            layer3_1_relu1_grad_input,
                                            layer3_1_bn1_saved_mean,
                                            layer3_1_bn1_saved_var,
                                            layer3_1_bn1_weight};

    // create layer3_1_bn1_grad_input tensor
    const unsigned layer3_1_bn1_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_1_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_bn1_grad_input_sizes, false, "layer3_1_bn1_grad_input");

    // create layer3_1_bn1_bias_grad tensor
    const unsigned layer3_1_bn1_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_bn1_bias_grad_tr_info = {"layer3_1_bn1_bias_grad", layer3_1_bn1_bias_grad_dram};
    synTensor           layer3_1_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer3_1_bn1_bias_grad_sizes, true, "layer3_1_bn1_bias_grad");

    // create layer3_1_bn1_weight_grad tensor
    const unsigned layer3_1_bn1_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_1_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_bn1_weight_grad_tr_info = {"layer3_1_bn1_weight_grad", layer3_1_bn1_weight_grad_dram};
    synTensor           layer3_1_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer3_1_bn1_weight_grad_sizes, true, "layer3_1_bn1_weight_grad");

    synTensor layer3_1_bn1_bwd_out_vec[3] = {layer3_1_bn1_grad_input, layer3_1_bn1_bias_grad, layer3_1_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_1_bn1_bwd_in_vec,
                           layer3_1_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_1_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_bn1_bwd failed!");

    /*************
     * layer3_1_conv1_dedx node
     * inputs: [layer3_1_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_1_conv1_weight[1, 1, 1024,
     *256](dtype=bf16)] output: [layer3_1_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_1_conv1_dedx_kernel_params;
    layer3_1_conv1_dedx_kernel_params.dH   = 1;
    layer3_1_conv1_dedx_kernel_params.dW   = 1;
    layer3_1_conv1_dedx_kernel_params.kH   = 1;
    layer3_1_conv1_dedx_kernel_params.kW   = 1;
    layer3_1_conv1_dedx_kernel_params.padT = 0;
    layer3_1_conv1_dedx_kernel_params.padB = 0;
    layer3_1_conv1_dedx_kernel_params.padL = 0;
    layer3_1_conv1_dedx_kernel_params.padR = 0;
    layer3_1_conv1_dedx_kernel_params.dilH = 1;
    layer3_1_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer3_1_conv1_dedx_in_vec[2] = {layer3_1_bn1_grad_input, layer3_1_conv1_weight};

    // create layer3_1_conv1_grad_input tensor
    const unsigned layer3_1_conv1_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_1_conv1_grad_input_sizes, false, "layer3_1_conv1_grad_input");

    synTensor layer3_1_conv1_dedx_out_vec[1] = {layer3_1_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv1_dedx_in_vec,
                           layer3_1_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_1_conv1_dedx_kernel_params,
                           sizeof(layer3_1_conv1_dedx_kernel_params),
                           "dedx",
                           "layer3_1_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv1_dedx failed!");

    /*************
     * layer3_1_conv1_dedw node
     * inputs: [layer3_1_bn1_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_0_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_1_conv1_weight_grad(1, 1, 1024, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_1_conv1_dedw_kernel_params;
    layer3_1_conv1_dedw_kernel_params.dH   = 1;
    layer3_1_conv1_dedw_kernel_params.dW   = 1;
    layer3_1_conv1_dedw_kernel_params.kH   = 1;
    layer3_1_conv1_dedw_kernel_params.kW   = 1;
    layer3_1_conv1_dedw_kernel_params.padT = 0;
    layer3_1_conv1_dedw_kernel_params.padB = 0;
    layer3_1_conv1_dedw_kernel_params.padL = 0;
    layer3_1_conv1_dedw_kernel_params.padR = 0;
    layer3_1_conv1_dedw_kernel_params.dilH = 1;
    layer3_1_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer3_1_conv1_dedw_in_vec[2] = {layer3_1_bn1_grad_input, layer3_0_relu3_output};

    // create layer3_1_conv1_weight_grad tensor
    const unsigned      layer3_1_conv1_weight_grad_sizes[] = {1, 1, 1024, 256};
    uint64_t            layer3_1_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_1_conv1_weight_grad_tr_info = {"layer3_1_conv1_weight_grad",
                                                              layer3_1_conv1_weight_grad_dram};
    synTensor           layer3_1_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer3_1_conv1_weight_grad_sizes, true, "layer3_1_conv1_weight_grad");

    synTensor layer3_1_conv1_dedw_out_vec[1] = {layer3_1_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_1_conv1_dedw_in_vec,
                           layer3_1_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_1_conv1_dedw_kernel_params,
                           sizeof(layer3_1_conv1_dedw_kernel_params),
                           "dedw",
                           "layer3_1_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_conv1_dedw failed!");

    /*************
     * layer3_1_add_residual_fwd1 node
     * inputs: [layer3_1_conv1_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_1_add_residual_grad_input1(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_1_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_1_add_residual_fwd1_in_vec[2] = {layer3_1_conv1_grad_input, layer3_1_add_residual_grad_input1};

    // create layer3_1_residual_upstream_grad_input tensor
    const unsigned layer3_1_residual_upstream_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_1_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer3_1_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer3_1_residual_upstream_grad_input");

    synTensor layer3_1_add_residual_fwd1_out_vec[1] = {layer3_1_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_1_add_residual_fwd1_in_vec,
                           layer3_1_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_1_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_1_add_residual_fwd1 failed!");

    /*************
     * layer3_0_relu3_bwd node
     * inputs: [layer3_1_residual_upstream_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_0_relu3_output(64, 14, 14,
     *1024)(dtype=bf16)] output: [layer3_0_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_0_relu3_bwd_in_vec[2] = {layer3_1_residual_upstream_grad_input, layer3_0_relu3_output};

    // create layer3_0_relu3_grad_input tensor
    const unsigned layer3_0_relu3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_relu3_grad_input_sizes, false, "layer3_0_relu3_grad_input");

    synTensor layer3_0_relu3_bwd_out_vec[1] = {layer3_0_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_0_relu3_bwd_in_vec,
                           layer3_0_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_0_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_relu3_bwd failed!");

    /*************
     * layer3_0_add_residual_bwd node
     * inputs: [layer3_0_relu3_grad_input(64, 14, 14, 1024)(dtype=bf16)]
     * output: [layer3_0_add_residual_grad_input0(64, 14, 14, 1024)(dtype=bf16), layer3_0_add_residual_grad_input1(64,
     *14, 14, 1024)(dtype=bf16)]
     *************/

    synTensor layer3_0_add_residual_bwd_in_vec[1] = {layer3_0_relu3_grad_input};

    // create layer3_0_add_residual_grad_input0 tensor
    const unsigned layer3_0_add_residual_grad_input0_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_0_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer3_0_add_residual_grad_input0");

    // create layer3_0_add_residual_grad_input1 tensor
    const unsigned layer3_0_add_residual_grad_input1_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer3_0_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer3_0_add_residual_grad_input1");

    synTensor layer3_0_add_residual_bwd_out_vec[2] = {layer3_0_add_residual_grad_input0,
                                                      layer3_0_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer3_0_add_residual_bwd_in_vec,
                           layer3_0_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer3_0_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_add_residual_bwd failed!");

    /*************
     * layer3_0_bn3_bwd node
     * inputs: [layer3_0_conv3_output(64, 14, 14, 1024)(dtype=bf16), layer3_0_add_residual_grad_input0(64, 14, 14,
     *1024)(dtype=bf16), layer3_0_bn3_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_0_bn3_saved_var(1, 1, 1,
     *1024)(dtype=float32), layer3_0_bn3_weight[1024](dtype=float32)] output: [layer3_0_bn3_grad_input(64, 14, 14,
     *1024)(dtype=bf16), layer3_0_bn3_bias_grad(1024,)(dtype=float32), layer3_0_bn3_weight_grad(1024,)(dtype=float32)]
     *************/

    synTensor layer3_0_bn3_bwd_in_vec[5] = {layer3_0_conv3_output,
                                            layer3_0_add_residual_grad_input0,
                                            layer3_0_bn3_saved_mean,
                                            layer3_0_bn3_saved_var,
                                            layer3_0_bn3_weight};

    // create layer3_0_bn3_grad_input tensor
    const unsigned layer3_0_bn3_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_0_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_bn3_grad_input_sizes, false, "layer3_0_bn3_grad_input");

    // create layer3_0_bn3_bias_grad tensor
    const unsigned layer3_0_bn3_bias_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_0_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_bn3_bias_grad_tr_info = {"layer3_0_bn3_bias_grad", layer3_0_bn3_bias_grad_dram};
    synTensor           layer3_0_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer3_0_bn3_bias_grad_sizes, true, "layer3_0_bn3_bias_grad");

    // create layer3_0_bn3_weight_grad tensor
    const unsigned layer3_0_bn3_weight_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_0_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_bn3_weight_grad_tr_info = {"layer3_0_bn3_weight_grad", layer3_0_bn3_weight_grad_dram};
    synTensor           layer3_0_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer3_0_bn3_weight_grad_sizes, true, "layer3_0_bn3_weight_grad");

    synTensor layer3_0_bn3_bwd_out_vec[3] = {layer3_0_bn3_grad_input, layer3_0_bn3_bias_grad, layer3_0_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_0_bn3_bwd_in_vec,
                           layer3_0_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_0_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_bn3_bwd failed!");

    /*************
     * layer3_0_conv3_dedx node
     * inputs: [layer3_0_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_0_conv3_weight[1, 1, 256,
     *1024](dtype=bf16)] output: [layer3_0_conv3_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_0_conv3_dedx_kernel_params;
    layer3_0_conv3_dedx_kernel_params.dH   = 1;
    layer3_0_conv3_dedx_kernel_params.dW   = 1;
    layer3_0_conv3_dedx_kernel_params.kH   = 1;
    layer3_0_conv3_dedx_kernel_params.kW   = 1;
    layer3_0_conv3_dedx_kernel_params.padT = 0;
    layer3_0_conv3_dedx_kernel_params.padB = 0;
    layer3_0_conv3_dedx_kernel_params.padL = 0;
    layer3_0_conv3_dedx_kernel_params.padR = 0;
    layer3_0_conv3_dedx_kernel_params.dilH = 1;
    layer3_0_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer3_0_conv3_dedx_in_vec[2] = {layer3_0_bn3_grad_input, layer3_0_conv3_weight};

    // create layer3_0_conv3_grad_input tensor
    const unsigned layer3_0_conv3_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_0_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_conv3_grad_input_sizes, false, "layer3_0_conv3_grad_input");

    synTensor layer3_0_conv3_dedx_out_vec[1] = {layer3_0_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv3_dedx_in_vec,
                           layer3_0_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_0_conv3_dedx_kernel_params,
                           sizeof(layer3_0_conv3_dedx_kernel_params),
                           "dedx",
                           "layer3_0_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv3_dedx failed!");

    /*************
     * layer3_0_conv3_dedw node
     * inputs: [layer3_0_bn3_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_0_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_0_conv3_weight_grad(1, 1, 256, 1024)(dtype=float32)]
     *************/
    synConvolutionParams layer3_0_conv3_dedw_kernel_params;
    layer3_0_conv3_dedw_kernel_params.dH   = 1;
    layer3_0_conv3_dedw_kernel_params.dW   = 1;
    layer3_0_conv3_dedw_kernel_params.kH   = 1;
    layer3_0_conv3_dedw_kernel_params.kW   = 1;
    layer3_0_conv3_dedw_kernel_params.padT = 0;
    layer3_0_conv3_dedw_kernel_params.padB = 0;
    layer3_0_conv3_dedw_kernel_params.padL = 0;
    layer3_0_conv3_dedw_kernel_params.padR = 0;
    layer3_0_conv3_dedw_kernel_params.dilH = 1;
    layer3_0_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer3_0_conv3_dedw_in_vec[2] = {layer3_0_bn3_grad_input, layer3_0_relu2_output};

    // create layer3_0_conv3_weight_grad tensor
    const unsigned      layer3_0_conv3_weight_grad_sizes[] = {1, 1, 256, 1024};
    uint64_t            layer3_0_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_conv3_weight_grad_tr_info = {"layer3_0_conv3_weight_grad",
                                                              layer3_0_conv3_weight_grad_dram};
    synTensor           layer3_0_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer3_0_conv3_weight_grad_sizes, true, "layer3_0_conv3_weight_grad");

    synTensor layer3_0_conv3_dedw_out_vec[1] = {layer3_0_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv3_dedw_in_vec,
                           layer3_0_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_0_conv3_dedw_kernel_params,
                           sizeof(layer3_0_conv3_dedw_kernel_params),
                           "dedw",
                           "layer3_0_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv3_dedw failed!");

    /*************
     * layer3_0_relu2_bwd node
     * inputs: [layer3_0_conv3_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_0_relu2_output(64, 14, 14,
     *256)(dtype=bf16)] output: [layer3_0_relu2_grad_input(64, 14, 14, 256)(dtype=bf16)]
     *************/

    synTensor layer3_0_relu2_bwd_in_vec[2] = {layer3_0_conv3_grad_input, layer3_0_relu2_output};

    // create layer3_0_relu2_grad_input tensor
    const unsigned layer3_0_relu2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_0_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_relu2_grad_input_sizes, false, "layer3_0_relu2_grad_input");

    synTensor layer3_0_relu2_bwd_out_vec[1] = {layer3_0_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_0_relu2_bwd_in_vec,
                           layer3_0_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_0_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_relu2_bwd failed!");

    /*************
     * layer3_0_bn2_bwd node
     * inputs: [layer3_0_conv2_output(64, 14, 14, 256)(dtype=bf16), layer3_0_relu2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_0_bn2_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_0_bn2_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_0_bn2_weight[256](dtype=float32)] output: [layer3_0_bn2_grad_input(64, 14, 14,
     *256)(dtype=bf16), layer3_0_bn2_bias_grad(256,)(dtype=float32), layer3_0_bn2_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_0_bn2_bwd_in_vec[5] = {layer3_0_conv2_output,
                                            layer3_0_relu2_grad_input,
                                            layer3_0_bn2_saved_mean,
                                            layer3_0_bn2_saved_var,
                                            layer3_0_bn2_weight};

    // create layer3_0_bn2_grad_input tensor
    const unsigned layer3_0_bn2_grad_input_sizes[] = {64, 14, 14, 256};
    synTensor      layer3_0_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_bn2_grad_input_sizes, false, "layer3_0_bn2_grad_input");

    // create layer3_0_bn2_bias_grad tensor
    const unsigned layer3_0_bn2_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_bn2_bias_grad_tr_info = {"layer3_0_bn2_bias_grad", layer3_0_bn2_bias_grad_dram};
    synTensor           layer3_0_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer3_0_bn2_bias_grad_sizes, true, "layer3_0_bn2_bias_grad");

    // create layer3_0_bn2_weight_grad tensor
    const unsigned layer3_0_bn2_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_bn2_weight_grad_tr_info = {"layer3_0_bn2_weight_grad", layer3_0_bn2_weight_grad_dram};
    synTensor           layer3_0_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer3_0_bn2_weight_grad_sizes, true, "layer3_0_bn2_weight_grad");

    synTensor layer3_0_bn2_bwd_out_vec[3] = {layer3_0_bn2_grad_input, layer3_0_bn2_bias_grad, layer3_0_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_0_bn2_bwd_in_vec,
                           layer3_0_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_0_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_bn2_bwd failed!");

    /*************
     * layer3_0_conv2_dedx node
     * inputs: [layer3_0_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_0_conv2_weight[3, 3, 256, 256](dtype=bf16)]
     * output: [layer3_0_conv2_grad_input(64, 28, 28, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_0_conv2_dedx_kernel_params;
    layer3_0_conv2_dedx_kernel_params.dH   = 2;
    layer3_0_conv2_dedx_kernel_params.dW   = 2;
    layer3_0_conv2_dedx_kernel_params.kH   = 3;
    layer3_0_conv2_dedx_kernel_params.kW   = 3;
    layer3_0_conv2_dedx_kernel_params.padT = 1;
    layer3_0_conv2_dedx_kernel_params.padB = 1;
    layer3_0_conv2_dedx_kernel_params.padL = 1;
    layer3_0_conv2_dedx_kernel_params.padR = 1;
    layer3_0_conv2_dedx_kernel_params.dilH = 1;
    layer3_0_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer3_0_conv2_dedx_in_vec[2] = {layer3_0_bn2_grad_input, layer3_0_conv2_weight};

    // create layer3_0_conv2_grad_input tensor
    const unsigned layer3_0_conv2_grad_input_sizes[] = {64, 28, 28, 256};
    synTensor      layer3_0_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_conv2_grad_input_sizes, false, "layer3_0_conv2_grad_input");

    synTensor layer3_0_conv2_dedx_out_vec[1] = {layer3_0_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv2_dedx_in_vec,
                           layer3_0_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_0_conv2_dedx_kernel_params,
                           sizeof(layer3_0_conv2_dedx_kernel_params),
                           "dedx",
                           "layer3_0_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv2_dedx failed!");

    /*************
     * layer3_0_conv2_dedw node
     * inputs: [layer3_0_bn2_grad_input(64, 14, 14, 256)(dtype=bf16), layer3_0_relu1_output(64, 28, 28,
     *256)(dtype=bf16)] output: [layer3_0_conv2_weight_grad(3, 3, 256, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_0_conv2_dedw_kernel_params;
    layer3_0_conv2_dedw_kernel_params.dH   = 2;
    layer3_0_conv2_dedw_kernel_params.dW   = 2;
    layer3_0_conv2_dedw_kernel_params.kH   = 3;
    layer3_0_conv2_dedw_kernel_params.kW   = 3;
    layer3_0_conv2_dedw_kernel_params.padT = 1;
    layer3_0_conv2_dedw_kernel_params.padB = 1;
    layer3_0_conv2_dedw_kernel_params.padL = 1;
    layer3_0_conv2_dedw_kernel_params.padR = 1;
    layer3_0_conv2_dedw_kernel_params.dilH = 1;
    layer3_0_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer3_0_conv2_dedw_in_vec[2] = {layer3_0_bn2_grad_input, layer3_0_relu1_output};

    // create layer3_0_conv2_weight_grad tensor
    const unsigned      layer3_0_conv2_weight_grad_sizes[] = {3, 3, 256, 256};
    uint64_t            layer3_0_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_conv2_weight_grad_tr_info = {"layer3_0_conv2_weight_grad",
                                                              layer3_0_conv2_weight_grad_dram};
    synTensor           layer3_0_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer3_0_conv2_weight_grad_sizes, true, "layer3_0_conv2_weight_grad");

    synTensor layer3_0_conv2_dedw_out_vec[1] = {layer3_0_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv2_dedw_in_vec,
                           layer3_0_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_0_conv2_dedw_kernel_params,
                           sizeof(layer3_0_conv2_dedw_kernel_params),
                           "dedw",
                           "layer3_0_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv2_dedw failed!");

    /*************
     * layer3_0_relu1_bwd node
     * inputs: [layer3_0_conv2_grad_input(64, 28, 28, 256)(dtype=bf16), layer3_0_relu1_output(64, 28, 28,
     *256)(dtype=bf16)] output: [layer3_0_relu1_grad_input(64, 28, 28, 256)(dtype=bf16)]
     *************/

    synTensor layer3_0_relu1_bwd_in_vec[2] = {layer3_0_conv2_grad_input, layer3_0_relu1_output};

    // create layer3_0_relu1_grad_input tensor
    const unsigned layer3_0_relu1_grad_input_sizes[] = {64, 28, 28, 256};
    synTensor      layer3_0_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_relu1_grad_input_sizes, false, "layer3_0_relu1_grad_input");

    synTensor layer3_0_relu1_bwd_out_vec[1] = {layer3_0_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_0_relu1_bwd_in_vec,
                           layer3_0_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer3_0_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_relu1_bwd failed!");

    /*************
     * layer3_0_bn1_bwd node
     * inputs: [layer3_0_conv1_output(64, 28, 28, 256)(dtype=bf16), layer3_0_relu1_grad_input(64, 28, 28,
     *256)(dtype=bf16), layer3_0_bn1_saved_mean(1, 1, 1, 256)(dtype=float32), layer3_0_bn1_saved_var(1, 1, 1,
     *256)(dtype=float32), layer3_0_bn1_weight[256](dtype=float32)] output: [layer3_0_bn1_grad_input(64, 28, 28,
     *256)(dtype=bf16), layer3_0_bn1_bias_grad(256,)(dtype=float32), layer3_0_bn1_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer3_0_bn1_bwd_in_vec[5] = {layer3_0_conv1_output,
                                            layer3_0_relu1_grad_input,
                                            layer3_0_bn1_saved_mean,
                                            layer3_0_bn1_saved_var,
                                            layer3_0_bn1_weight};

    // create layer3_0_bn1_grad_input tensor
    const unsigned layer3_0_bn1_grad_input_sizes[] = {64, 28, 28, 256};
    synTensor      layer3_0_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_bn1_grad_input_sizes, false, "layer3_0_bn1_grad_input");

    // create layer3_0_bn1_bias_grad tensor
    const unsigned layer3_0_bn1_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_bn1_bias_grad_tr_info = {"layer3_0_bn1_bias_grad", layer3_0_bn1_bias_grad_dram};
    synTensor           layer3_0_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer3_0_bn1_bias_grad_sizes, true, "layer3_0_bn1_bias_grad");

    // create layer3_0_bn1_weight_grad tensor
    const unsigned layer3_0_bn1_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer3_0_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_bn1_weight_grad_tr_info = {"layer3_0_bn1_weight_grad", layer3_0_bn1_weight_grad_dram};
    synTensor           layer3_0_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer3_0_bn1_weight_grad_sizes, true, "layer3_0_bn1_weight_grad");

    synTensor layer3_0_bn1_bwd_out_vec[3] = {layer3_0_bn1_grad_input, layer3_0_bn1_bias_grad, layer3_0_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_0_bn1_bwd_in_vec,
                           layer3_0_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_0_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_bn1_bwd failed!");

    /*************
     * layer3_0_conv1_dedx node
     * inputs: [layer3_0_bn1_grad_input(64, 28, 28, 256)(dtype=bf16), layer3_0_conv1_weight[1, 1, 512, 256](dtype=bf16)]
     * output: [layer3_0_conv1_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_0_conv1_dedx_kernel_params;
    layer3_0_conv1_dedx_kernel_params.dH   = 1;
    layer3_0_conv1_dedx_kernel_params.dW   = 1;
    layer3_0_conv1_dedx_kernel_params.kH   = 1;
    layer3_0_conv1_dedx_kernel_params.kW   = 1;
    layer3_0_conv1_dedx_kernel_params.padT = 0;
    layer3_0_conv1_dedx_kernel_params.padB = 0;
    layer3_0_conv1_dedx_kernel_params.padL = 0;
    layer3_0_conv1_dedx_kernel_params.padR = 0;
    layer3_0_conv1_dedx_kernel_params.dilH = 1;
    layer3_0_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer3_0_conv1_dedx_in_vec[2] = {layer3_0_bn1_grad_input, layer3_0_conv1_weight};

    // create layer3_0_conv1_grad_input tensor
    const unsigned layer3_0_conv1_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer3_0_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer3_0_conv1_grad_input_sizes, false, "layer3_0_conv1_grad_input");

    synTensor layer3_0_conv1_dedx_out_vec[1] = {layer3_0_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv1_dedx_in_vec,
                           layer3_0_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_0_conv1_dedx_kernel_params,
                           sizeof(layer3_0_conv1_dedx_kernel_params),
                           "dedx",
                           "layer3_0_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv1_dedx failed!");

    /*************
     * layer3_0_conv1_dedw node
     * inputs: [layer3_0_bn1_grad_input(64, 28, 28, 256)(dtype=bf16), layer2_3_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer3_0_conv1_weight_grad(1, 1, 512, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer3_0_conv1_dedw_kernel_params;
    layer3_0_conv1_dedw_kernel_params.dH   = 1;
    layer3_0_conv1_dedw_kernel_params.dW   = 1;
    layer3_0_conv1_dedw_kernel_params.kH   = 1;
    layer3_0_conv1_dedw_kernel_params.kW   = 1;
    layer3_0_conv1_dedw_kernel_params.padT = 0;
    layer3_0_conv1_dedw_kernel_params.padB = 0;
    layer3_0_conv1_dedw_kernel_params.padL = 0;
    layer3_0_conv1_dedw_kernel_params.padR = 0;
    layer3_0_conv1_dedw_kernel_params.dilH = 1;
    layer3_0_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer3_0_conv1_dedw_in_vec[2] = {layer3_0_bn1_grad_input, layer2_3_relu3_output};

    // create layer3_0_conv1_weight_grad tensor
    const unsigned      layer3_0_conv1_weight_grad_sizes[] = {1, 1, 512, 256};
    uint64_t            layer3_0_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_0_conv1_weight_grad_tr_info = {"layer3_0_conv1_weight_grad",
                                                              layer3_0_conv1_weight_grad_dram};
    synTensor           layer3_0_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer3_0_conv1_weight_grad_sizes, true, "layer3_0_conv1_weight_grad");

    synTensor layer3_0_conv1_dedw_out_vec[1] = {layer3_0_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_0_conv1_dedw_in_vec,
                           layer3_0_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_0_conv1_dedw_kernel_params,
                           sizeof(layer3_0_conv1_dedw_kernel_params),
                           "dedw",
                           "layer3_0_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_conv1_dedw failed!");

    /*************
     * layer3_bn_bwd node
     * inputs: [layer3_downsample_output(64, 14, 14, 1024)(dtype=bf16), layer3_0_add_residual_grad_input1(64, 14, 14,
     *1024)(dtype=bf16), layer3_bn_saved_mean(1, 1, 1, 1024)(dtype=float32), layer3_bn_saved_var(1, 1, 1,
     *1024)(dtype=float32), layer3_bn_weight[1024](dtype=float32)] output: [layer3_bn_grad_input(64, 14, 14,
     *1024)(dtype=bf16), layer3_bn_bias_grad(1024,)(dtype=float32), layer3_bn_weight_grad(1024,)(dtype=float32)]
     *************/

    synTensor layer3_bn_bwd_in_vec[5] = {layer3_downsample_output,
                                         layer3_0_add_residual_grad_input1,
                                         layer3_bn_saved_mean,
                                         layer3_bn_saved_var,
                                         layer3_bn_weight};

    // create layer3_bn_grad_input tensor
    const unsigned layer3_bn_grad_input_sizes[] = {64, 14, 14, 1024};
    synTensor      layer3_bn_grad_input =
        createTensor(4U, syn_type_bf16, layer3_bn_grad_input_sizes, false, "layer3_bn_grad_input");

    // create layer3_bn_bias_grad tensor
    const unsigned layer3_bn_bias_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_bn_bias_grad_dram    = 0;
    synLaunchTensorInfo layer3_bn_bias_grad_tr_info = {"layer3_bn_bias_grad", layer3_bn_bias_grad_dram};
    synTensor           layer3_bn_bias_grad =
        createTensor(1U, syn_type_single, layer3_bn_bias_grad_sizes, true, "layer3_bn_bias_grad");

    // create layer3_bn_weight_grad tensor
    const unsigned layer3_bn_weight_grad_sizes[] = {
        1024,
    };
    uint64_t            layer3_bn_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_bn_weight_grad_tr_info = {"layer3_bn_weight_grad", layer3_bn_weight_grad_dram};
    synTensor           layer3_bn_weight_grad =
        createTensor(1U, syn_type_single, layer3_bn_weight_grad_sizes, true, "layer3_bn_weight_grad");

    synTensor layer3_bn_bwd_out_vec[3] = {layer3_bn_grad_input, layer3_bn_bias_grad, layer3_bn_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_bn_bwd_in_vec,
                           layer3_bn_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer3_bn_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_bn_bwd failed!");

    /*************
     * layer3_downsample_dedx node
     * inputs: [layer3_bn_grad_input(64, 14, 14, 1024)(dtype=bf16), layer3_downsample_weight[1, 1, 512,
     *1024](dtype=bf16)] output: [layer3_downsample_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer3_downsample_dedx_kernel_params;
    layer3_downsample_dedx_kernel_params.dH   = 2;
    layer3_downsample_dedx_kernel_params.dW   = 2;
    layer3_downsample_dedx_kernel_params.kH   = 1;
    layer3_downsample_dedx_kernel_params.kW   = 1;
    layer3_downsample_dedx_kernel_params.padT = 0;
    layer3_downsample_dedx_kernel_params.padB = 0;
    layer3_downsample_dedx_kernel_params.padL = 0;
    layer3_downsample_dedx_kernel_params.padR = 0;
    layer3_downsample_dedx_kernel_params.dilH = 1;
    layer3_downsample_dedx_kernel_params.dilW = 1;

    synTensor layer3_downsample_dedx_in_vec[2] = {layer3_bn_grad_input, layer3_downsample_weight};

    // create layer3_downsample_grad_input tensor
    const unsigned layer3_downsample_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer3_downsample_grad_input =
        createTensor(4U, syn_type_bf16, layer3_downsample_grad_input_sizes, false, "layer3_downsample_grad_input");

    synTensor layer3_downsample_dedx_out_vec[1] = {layer3_downsample_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_downsample_dedx_in_vec,
                           layer3_downsample_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer3_downsample_dedx_kernel_params,
                           sizeof(layer3_downsample_dedx_kernel_params),
                           "dedx",
                           "layer3_downsample_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_downsample_dedx failed!");

    /*************
     * layer3_downsample_dedw node
     * inputs: [layer3_bn_grad_input(64, 14, 14, 1024)(dtype=bf16), layer2_3_relu3_output(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer3_downsample_weight_grad(1, 1, 512, 1024)(dtype=float32)]
     *************/
    synConvolutionParams layer3_downsample_dedw_kernel_params;
    layer3_downsample_dedw_kernel_params.dH   = 2;
    layer3_downsample_dedw_kernel_params.dW   = 2;
    layer3_downsample_dedw_kernel_params.kH   = 1;
    layer3_downsample_dedw_kernel_params.kW   = 1;
    layer3_downsample_dedw_kernel_params.padT = 0;
    layer3_downsample_dedw_kernel_params.padB = 0;
    layer3_downsample_dedw_kernel_params.padL = 0;
    layer3_downsample_dedw_kernel_params.padR = 0;
    layer3_downsample_dedw_kernel_params.dilH = 1;
    layer3_downsample_dedw_kernel_params.dilW = 1;

    synTensor layer3_downsample_dedw_in_vec[2] = {layer3_bn_grad_input, layer2_3_relu3_output};

    // create layer3_downsample_weight_grad tensor
    const unsigned      layer3_downsample_weight_grad_sizes[] = {1, 1, 512, 1024};
    uint64_t            layer3_downsample_weight_grad_dram    = 0;
    synLaunchTensorInfo layer3_downsample_weight_grad_tr_info = {"layer3_downsample_weight_grad",
                                                                 layer3_downsample_weight_grad_dram};
    synTensor           layer3_downsample_weight_grad =
        createTensor(4U, syn_type_single, layer3_downsample_weight_grad_sizes, true, "layer3_downsample_weight_grad");

    synTensor layer3_downsample_dedw_out_vec[1] = {layer3_downsample_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer3_downsample_dedw_in_vec,
                           layer3_downsample_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer3_downsample_dedw_kernel_params,
                           sizeof(layer3_downsample_dedw_kernel_params),
                           "dedw",
                           "layer3_downsample_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_downsample_dedw failed!");

    /*************
     * layer3_0_add_residual_fwd1 node
     * inputs: [layer3_0_conv1_grad_input(64, 28, 28, 512)(dtype=bf16), layer3_downsample_grad_input(64, 28, 28,
     *512)(dtype=bf16)] output: [layer3_0_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer3_0_add_residual_fwd1_in_vec[2] = {layer3_0_conv1_grad_input, layer3_downsample_grad_input};

    // create layer3_0_residual_upstream_grad_input tensor
    const unsigned layer3_0_residual_upstream_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer3_0_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer3_0_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer3_0_residual_upstream_grad_input");

    synTensor layer3_0_add_residual_fwd1_out_vec[1] = {layer3_0_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer3_0_add_residual_fwd1_in_vec,
                           layer3_0_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer3_0_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer3_0_add_residual_fwd1 failed!");

    /*************
     * layer2_3_relu3_bwd node
     * inputs: [layer3_0_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_3_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_3_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_3_relu3_bwd_in_vec[2] = {layer3_0_residual_upstream_grad_input, layer2_3_relu3_output};

    // create layer2_3_relu3_grad_input tensor
    const unsigned layer2_3_relu3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_relu3_grad_input_sizes, false, "layer2_3_relu3_grad_input");

    synTensor layer2_3_relu3_bwd_out_vec[1] = {layer2_3_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_3_relu3_bwd_in_vec,
                           layer2_3_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_3_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_relu3_bwd failed!");

    /*************
     * layer2_3_add_residual_bwd node
     * inputs: [layer2_3_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_3_add_residual_grad_input0(64, 28, 28, 512)(dtype=bf16), layer2_3_add_residual_grad_input1(64,
     *28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_3_add_residual_bwd_in_vec[1] = {layer2_3_relu3_grad_input};

    // create layer2_3_add_residual_grad_input0 tensor
    const unsigned layer2_3_add_residual_grad_input0_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_3_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer2_3_add_residual_grad_input0");

    // create layer2_3_add_residual_grad_input1 tensor
    const unsigned layer2_3_add_residual_grad_input1_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_3_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer2_3_add_residual_grad_input1");

    synTensor layer2_3_add_residual_bwd_out_vec[2] = {layer2_3_add_residual_grad_input0,
                                                      layer2_3_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer2_3_add_residual_bwd_in_vec,
                           layer2_3_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer2_3_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_add_residual_bwd failed!");

    /*************
     * layer2_3_bn3_bwd node
     * inputs: [layer2_3_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_3_add_residual_grad_input0(64, 28, 28,
     *512)(dtype=bf16), layer2_3_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_3_bn3_saved_var(1, 1, 1,
     *512)(dtype=float32), layer2_3_bn3_weight[512](dtype=float32)] output: [layer2_3_bn3_grad_input(64, 28, 28,
     *512)(dtype=bf16), layer2_3_bn3_bias_grad(512,)(dtype=float32), layer2_3_bn3_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer2_3_bn3_bwd_in_vec[5] = {layer2_3_conv3_output,
                                            layer2_3_add_residual_grad_input0,
                                            layer2_3_bn3_saved_mean,
                                            layer2_3_bn3_saved_var,
                                            layer2_3_bn3_weight};

    // create layer2_3_bn3_grad_input tensor
    const unsigned layer2_3_bn3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_bn3_grad_input_sizes, false, "layer2_3_bn3_grad_input");

    // create layer2_3_bn3_bias_grad tensor
    const unsigned layer2_3_bn3_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_3_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_bn3_bias_grad_tr_info = {"layer2_3_bn3_bias_grad", layer2_3_bn3_bias_grad_dram};
    synTensor           layer2_3_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer2_3_bn3_bias_grad_sizes, true, "layer2_3_bn3_bias_grad");

    // create layer2_3_bn3_weight_grad tensor
    const unsigned layer2_3_bn3_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_3_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_bn3_weight_grad_tr_info = {"layer2_3_bn3_weight_grad", layer2_3_bn3_weight_grad_dram};
    synTensor           layer2_3_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer2_3_bn3_weight_grad_sizes, true, "layer2_3_bn3_weight_grad");

    synTensor layer2_3_bn3_bwd_out_vec[3] = {layer2_3_bn3_grad_input, layer2_3_bn3_bias_grad, layer2_3_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_3_bn3_bwd_in_vec,
                           layer2_3_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_3_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_bn3_bwd failed!");

    /*************
     * layer2_3_conv3_dedx node
     * inputs: [layer2_3_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_3_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_3_conv3_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_3_conv3_dedx_kernel_params;
    layer2_3_conv3_dedx_kernel_params.dH   = 1;
    layer2_3_conv3_dedx_kernel_params.dW   = 1;
    layer2_3_conv3_dedx_kernel_params.kH   = 1;
    layer2_3_conv3_dedx_kernel_params.kW   = 1;
    layer2_3_conv3_dedx_kernel_params.padT = 0;
    layer2_3_conv3_dedx_kernel_params.padB = 0;
    layer2_3_conv3_dedx_kernel_params.padL = 0;
    layer2_3_conv3_dedx_kernel_params.padR = 0;
    layer2_3_conv3_dedx_kernel_params.dilH = 1;
    layer2_3_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer2_3_conv3_dedx_in_vec[2] = {layer2_3_bn3_grad_input, layer2_3_conv3_weight};

    // create layer2_3_conv3_grad_input tensor
    const unsigned layer2_3_conv3_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_conv3_grad_input_sizes, false, "layer2_3_conv3_grad_input");

    synTensor layer2_3_conv3_dedx_out_vec[1] = {layer2_3_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv3_dedx_in_vec,
                           layer2_3_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_3_conv3_dedx_kernel_params,
                           sizeof(layer2_3_conv3_dedx_kernel_params),
                           "dedx",
                           "layer2_3_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv3_dedx failed!");

    /*************
     * layer2_3_conv3_dedw node
     * inputs: [layer2_3_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_3_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_3_conv3_weight_grad(1, 1, 128, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer2_3_conv3_dedw_kernel_params;
    layer2_3_conv3_dedw_kernel_params.dH   = 1;
    layer2_3_conv3_dedw_kernel_params.dW   = 1;
    layer2_3_conv3_dedw_kernel_params.kH   = 1;
    layer2_3_conv3_dedw_kernel_params.kW   = 1;
    layer2_3_conv3_dedw_kernel_params.padT = 0;
    layer2_3_conv3_dedw_kernel_params.padB = 0;
    layer2_3_conv3_dedw_kernel_params.padL = 0;
    layer2_3_conv3_dedw_kernel_params.padR = 0;
    layer2_3_conv3_dedw_kernel_params.dilH = 1;
    layer2_3_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer2_3_conv3_dedw_in_vec[2] = {layer2_3_bn3_grad_input, layer2_3_relu2_output};

    // create layer2_3_conv3_weight_grad tensor
    const unsigned      layer2_3_conv3_weight_grad_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_3_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_conv3_weight_grad_tr_info = {"layer2_3_conv3_weight_grad",
                                                              layer2_3_conv3_weight_grad_dram};
    synTensor           layer2_3_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer2_3_conv3_weight_grad_sizes, true, "layer2_3_conv3_weight_grad");

    synTensor layer2_3_conv3_dedw_out_vec[1] = {layer2_3_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv3_dedw_in_vec,
                           layer2_3_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_3_conv3_dedw_kernel_params,
                           sizeof(layer2_3_conv3_dedw_kernel_params),
                           "dedw",
                           "layer2_3_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv3_dedw failed!");

    /*************
     * layer2_3_relu2_bwd node
     * inputs: [layer2_3_conv3_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_3_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_3_relu2_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_3_relu2_bwd_in_vec[2] = {layer2_3_conv3_grad_input, layer2_3_relu2_output};

    // create layer2_3_relu2_grad_input tensor
    const unsigned layer2_3_relu2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_relu2_grad_input_sizes, false, "layer2_3_relu2_grad_input");

    synTensor layer2_3_relu2_bwd_out_vec[1] = {layer2_3_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_3_relu2_bwd_in_vec,
                           layer2_3_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_3_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_relu2_bwd failed!");

    /*************
     * layer2_3_bn2_bwd node
     * inputs: [layer2_3_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_3_relu2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_3_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_3_bn2_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_3_bn2_weight[128](dtype=float32)] output: [layer2_3_bn2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_3_bn2_bias_grad(128,)(dtype=float32), layer2_3_bn2_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_3_bn2_bwd_in_vec[5] = {layer2_3_conv2_output,
                                            layer2_3_relu2_grad_input,
                                            layer2_3_bn2_saved_mean,
                                            layer2_3_bn2_saved_var,
                                            layer2_3_bn2_weight};

    // create layer2_3_bn2_grad_input tensor
    const unsigned layer2_3_bn2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_bn2_grad_input_sizes, false, "layer2_3_bn2_grad_input");

    // create layer2_3_bn2_bias_grad tensor
    const unsigned layer2_3_bn2_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_bn2_bias_grad_tr_info = {"layer2_3_bn2_bias_grad", layer2_3_bn2_bias_grad_dram};
    synTensor           layer2_3_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer2_3_bn2_bias_grad_sizes, true, "layer2_3_bn2_bias_grad");

    // create layer2_3_bn2_weight_grad tensor
    const unsigned layer2_3_bn2_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_bn2_weight_grad_tr_info = {"layer2_3_bn2_weight_grad", layer2_3_bn2_weight_grad_dram};
    synTensor           layer2_3_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer2_3_bn2_weight_grad_sizes, true, "layer2_3_bn2_weight_grad");

    synTensor layer2_3_bn2_bwd_out_vec[3] = {layer2_3_bn2_grad_input, layer2_3_bn2_bias_grad, layer2_3_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_3_bn2_bwd_in_vec,
                           layer2_3_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_3_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_bn2_bwd failed!");

    /*************
     * layer2_3_conv2_dedx node
     * inputs: [layer2_3_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_3_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_3_conv2_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_3_conv2_dedx_kernel_params;
    layer2_3_conv2_dedx_kernel_params.dH   = 1;
    layer2_3_conv2_dedx_kernel_params.dW   = 1;
    layer2_3_conv2_dedx_kernel_params.kH   = 3;
    layer2_3_conv2_dedx_kernel_params.kW   = 3;
    layer2_3_conv2_dedx_kernel_params.padT = 1;
    layer2_3_conv2_dedx_kernel_params.padB = 1;
    layer2_3_conv2_dedx_kernel_params.padL = 1;
    layer2_3_conv2_dedx_kernel_params.padR = 1;
    layer2_3_conv2_dedx_kernel_params.dilH = 1;
    layer2_3_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer2_3_conv2_dedx_in_vec[2] = {layer2_3_bn2_grad_input, layer2_3_conv2_weight};

    // create layer2_3_conv2_grad_input tensor
    const unsigned layer2_3_conv2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_conv2_grad_input_sizes, false, "layer2_3_conv2_grad_input");

    synTensor layer2_3_conv2_dedx_out_vec[1] = {layer2_3_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv2_dedx_in_vec,
                           layer2_3_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_3_conv2_dedx_kernel_params,
                           sizeof(layer2_3_conv2_dedx_kernel_params),
                           "dedx",
                           "layer2_3_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv2_dedx failed!");

    /*************
     * layer2_3_conv2_dedw node
     * inputs: [layer2_3_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_3_relu1_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_3_conv2_weight_grad(3, 3, 128, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_3_conv2_dedw_kernel_params;
    layer2_3_conv2_dedw_kernel_params.dH   = 1;
    layer2_3_conv2_dedw_kernel_params.dW   = 1;
    layer2_3_conv2_dedw_kernel_params.kH   = 3;
    layer2_3_conv2_dedw_kernel_params.kW   = 3;
    layer2_3_conv2_dedw_kernel_params.padT = 1;
    layer2_3_conv2_dedw_kernel_params.padB = 1;
    layer2_3_conv2_dedw_kernel_params.padL = 1;
    layer2_3_conv2_dedw_kernel_params.padR = 1;
    layer2_3_conv2_dedw_kernel_params.dilH = 1;
    layer2_3_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer2_3_conv2_dedw_in_vec[2] = {layer2_3_bn2_grad_input, layer2_3_relu1_output};

    // create layer2_3_conv2_weight_grad tensor
    const unsigned      layer2_3_conv2_weight_grad_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_3_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_conv2_weight_grad_tr_info = {"layer2_3_conv2_weight_grad",
                                                              layer2_3_conv2_weight_grad_dram};
    synTensor           layer2_3_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer2_3_conv2_weight_grad_sizes, true, "layer2_3_conv2_weight_grad");

    synTensor layer2_3_conv2_dedw_out_vec[1] = {layer2_3_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv2_dedw_in_vec,
                           layer2_3_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_3_conv2_dedw_kernel_params,
                           sizeof(layer2_3_conv2_dedw_kernel_params),
                           "dedw",
                           "layer2_3_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv2_dedw failed!");

    /*************
     * layer2_3_relu1_bwd node
     * inputs: [layer2_3_conv2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_3_relu1_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_3_relu1_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_3_relu1_bwd_in_vec[2] = {layer2_3_conv2_grad_input, layer2_3_relu1_output};

    // create layer2_3_relu1_grad_input tensor
    const unsigned layer2_3_relu1_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_relu1_grad_input_sizes, false, "layer2_3_relu1_grad_input");

    synTensor layer2_3_relu1_bwd_out_vec[1] = {layer2_3_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_3_relu1_bwd_in_vec,
                           layer2_3_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_3_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_relu1_bwd failed!");

    /*************
     * layer2_3_bn1_bwd node
     * inputs: [layer2_3_conv1_output(64, 28, 28, 128)(dtype=bf16), layer2_3_relu1_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_3_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_3_bn1_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_3_bn1_weight[128](dtype=float32)] output: [layer2_3_bn1_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_3_bn1_bias_grad(128,)(dtype=float32), layer2_3_bn1_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_3_bn1_bwd_in_vec[5] = {layer2_3_conv1_output,
                                            layer2_3_relu1_grad_input,
                                            layer2_3_bn1_saved_mean,
                                            layer2_3_bn1_saved_var,
                                            layer2_3_bn1_weight};

    // create layer2_3_bn1_grad_input tensor
    const unsigned layer2_3_bn1_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_3_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_bn1_grad_input_sizes, false, "layer2_3_bn1_grad_input");

    // create layer2_3_bn1_bias_grad tensor
    const unsigned layer2_3_bn1_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_bn1_bias_grad_tr_info = {"layer2_3_bn1_bias_grad", layer2_3_bn1_bias_grad_dram};
    synTensor           layer2_3_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer2_3_bn1_bias_grad_sizes, true, "layer2_3_bn1_bias_grad");

    // create layer2_3_bn1_weight_grad tensor
    const unsigned layer2_3_bn1_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_3_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_bn1_weight_grad_tr_info = {"layer2_3_bn1_weight_grad", layer2_3_bn1_weight_grad_dram};
    synTensor           layer2_3_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer2_3_bn1_weight_grad_sizes, true, "layer2_3_bn1_weight_grad");

    synTensor layer2_3_bn1_bwd_out_vec[3] = {layer2_3_bn1_grad_input, layer2_3_bn1_bias_grad, layer2_3_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_3_bn1_bwd_in_vec,
                           layer2_3_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_3_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_bn1_bwd failed!");

    /*************
     * layer2_3_conv1_dedx node
     * inputs: [layer2_3_bn1_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_3_conv1_weight[1, 1, 512, 128](dtype=bf16)]
     * output: [layer2_3_conv1_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_3_conv1_dedx_kernel_params;
    layer2_3_conv1_dedx_kernel_params.dH   = 1;
    layer2_3_conv1_dedx_kernel_params.dW   = 1;
    layer2_3_conv1_dedx_kernel_params.kH   = 1;
    layer2_3_conv1_dedx_kernel_params.kW   = 1;
    layer2_3_conv1_dedx_kernel_params.padT = 0;
    layer2_3_conv1_dedx_kernel_params.padB = 0;
    layer2_3_conv1_dedx_kernel_params.padL = 0;
    layer2_3_conv1_dedx_kernel_params.padR = 0;
    layer2_3_conv1_dedx_kernel_params.dilH = 1;
    layer2_3_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer2_3_conv1_dedx_in_vec[2] = {layer2_3_bn1_grad_input, layer2_3_conv1_weight};

    // create layer2_3_conv1_grad_input tensor
    const unsigned layer2_3_conv1_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_3_conv1_grad_input_sizes, false, "layer2_3_conv1_grad_input");

    synTensor layer2_3_conv1_dedx_out_vec[1] = {layer2_3_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv1_dedx_in_vec,
                           layer2_3_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_3_conv1_dedx_kernel_params,
                           sizeof(layer2_3_conv1_dedx_kernel_params),
                           "dedx",
                           "layer2_3_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv1_dedx failed!");

    /*************
     * layer2_3_conv1_dedw node
     * inputs: [layer2_3_bn1_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_2_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_3_conv1_weight_grad(1, 1, 512, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_3_conv1_dedw_kernel_params;
    layer2_3_conv1_dedw_kernel_params.dH   = 1;
    layer2_3_conv1_dedw_kernel_params.dW   = 1;
    layer2_3_conv1_dedw_kernel_params.kH   = 1;
    layer2_3_conv1_dedw_kernel_params.kW   = 1;
    layer2_3_conv1_dedw_kernel_params.padT = 0;
    layer2_3_conv1_dedw_kernel_params.padB = 0;
    layer2_3_conv1_dedw_kernel_params.padL = 0;
    layer2_3_conv1_dedw_kernel_params.padR = 0;
    layer2_3_conv1_dedw_kernel_params.dilH = 1;
    layer2_3_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer2_3_conv1_dedw_in_vec[2] = {layer2_3_bn1_grad_input, layer2_2_relu3_output};

    // create layer2_3_conv1_weight_grad tensor
    const unsigned      layer2_3_conv1_weight_grad_sizes[] = {1, 1, 512, 128};
    uint64_t            layer2_3_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_3_conv1_weight_grad_tr_info = {"layer2_3_conv1_weight_grad",
                                                              layer2_3_conv1_weight_grad_dram};
    synTensor           layer2_3_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer2_3_conv1_weight_grad_sizes, true, "layer2_3_conv1_weight_grad");

    synTensor layer2_3_conv1_dedw_out_vec[1] = {layer2_3_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_3_conv1_dedw_in_vec,
                           layer2_3_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_3_conv1_dedw_kernel_params,
                           sizeof(layer2_3_conv1_dedw_kernel_params),
                           "dedw",
                           "layer2_3_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_conv1_dedw failed!");

    /*************
     * layer2_3_add_residual_fwd1 node
     * inputs: [layer2_3_conv1_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_3_add_residual_grad_input1(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_3_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_3_add_residual_fwd1_in_vec[2] = {layer2_3_conv1_grad_input, layer2_3_add_residual_grad_input1};

    // create layer2_3_residual_upstream_grad_input tensor
    const unsigned layer2_3_residual_upstream_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_3_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer2_3_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer2_3_residual_upstream_grad_input");

    synTensor layer2_3_add_residual_fwd1_out_vec[1] = {layer2_3_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_3_add_residual_fwd1_in_vec,
                           layer2_3_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_3_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_3_add_residual_fwd1 failed!");

    /*************
     * layer2_2_relu3_bwd node
     * inputs: [layer2_3_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_2_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_2_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_2_relu3_bwd_in_vec[2] = {layer2_3_residual_upstream_grad_input, layer2_2_relu3_output};

    // create layer2_2_relu3_grad_input tensor
    const unsigned layer2_2_relu3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_relu3_grad_input_sizes, false, "layer2_2_relu3_grad_input");

    synTensor layer2_2_relu3_bwd_out_vec[1] = {layer2_2_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_2_relu3_bwd_in_vec,
                           layer2_2_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_2_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_relu3_bwd failed!");

    /*************
     * layer2_2_add_residual_bwd node
     * inputs: [layer2_2_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_2_add_residual_grad_input0(64, 28, 28, 512)(dtype=bf16), layer2_2_add_residual_grad_input1(64,
     *28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_2_add_residual_bwd_in_vec[1] = {layer2_2_relu3_grad_input};

    // create layer2_2_add_residual_grad_input0 tensor
    const unsigned layer2_2_add_residual_grad_input0_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_2_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer2_2_add_residual_grad_input0");

    // create layer2_2_add_residual_grad_input1 tensor
    const unsigned layer2_2_add_residual_grad_input1_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_2_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer2_2_add_residual_grad_input1");

    synTensor layer2_2_add_residual_bwd_out_vec[2] = {layer2_2_add_residual_grad_input0,
                                                      layer2_2_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer2_2_add_residual_bwd_in_vec,
                           layer2_2_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer2_2_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_add_residual_bwd failed!");

    /*************
     * layer2_2_bn3_bwd node
     * inputs: [layer2_2_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_2_add_residual_grad_input0(64, 28, 28,
     *512)(dtype=bf16), layer2_2_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_2_bn3_saved_var(1, 1, 1,
     *512)(dtype=float32), layer2_2_bn3_weight[512](dtype=float32)] output: [layer2_2_bn3_grad_input(64, 28, 28,
     *512)(dtype=bf16), layer2_2_bn3_bias_grad(512,)(dtype=float32), layer2_2_bn3_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer2_2_bn3_bwd_in_vec[5] = {layer2_2_conv3_output,
                                            layer2_2_add_residual_grad_input0,
                                            layer2_2_bn3_saved_mean,
                                            layer2_2_bn3_saved_var,
                                            layer2_2_bn3_weight};

    // create layer2_2_bn3_grad_input tensor
    const unsigned layer2_2_bn3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_bn3_grad_input_sizes, false, "layer2_2_bn3_grad_input");

    // create layer2_2_bn3_bias_grad tensor
    const unsigned layer2_2_bn3_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_2_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_bn3_bias_grad_tr_info = {"layer2_2_bn3_bias_grad", layer2_2_bn3_bias_grad_dram};
    synTensor           layer2_2_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer2_2_bn3_bias_grad_sizes, true, "layer2_2_bn3_bias_grad");

    // create layer2_2_bn3_weight_grad tensor
    const unsigned layer2_2_bn3_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_2_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_bn3_weight_grad_tr_info = {"layer2_2_bn3_weight_grad", layer2_2_bn3_weight_grad_dram};
    synTensor           layer2_2_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer2_2_bn3_weight_grad_sizes, true, "layer2_2_bn3_weight_grad");

    synTensor layer2_2_bn3_bwd_out_vec[3] = {layer2_2_bn3_grad_input, layer2_2_bn3_bias_grad, layer2_2_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_2_bn3_bwd_in_vec,
                           layer2_2_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_2_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_bn3_bwd failed!");

    /*************
     * layer2_2_conv3_dedx node
     * inputs: [layer2_2_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_2_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_2_conv3_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_2_conv3_dedx_kernel_params;
    layer2_2_conv3_dedx_kernel_params.dH   = 1;
    layer2_2_conv3_dedx_kernel_params.dW   = 1;
    layer2_2_conv3_dedx_kernel_params.kH   = 1;
    layer2_2_conv3_dedx_kernel_params.kW   = 1;
    layer2_2_conv3_dedx_kernel_params.padT = 0;
    layer2_2_conv3_dedx_kernel_params.padB = 0;
    layer2_2_conv3_dedx_kernel_params.padL = 0;
    layer2_2_conv3_dedx_kernel_params.padR = 0;
    layer2_2_conv3_dedx_kernel_params.dilH = 1;
    layer2_2_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer2_2_conv3_dedx_in_vec[2] = {layer2_2_bn3_grad_input, layer2_2_conv3_weight};

    // create layer2_2_conv3_grad_input tensor
    const unsigned layer2_2_conv3_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_conv3_grad_input_sizes, false, "layer2_2_conv3_grad_input");

    synTensor layer2_2_conv3_dedx_out_vec[1] = {layer2_2_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv3_dedx_in_vec,
                           layer2_2_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_2_conv3_dedx_kernel_params,
                           sizeof(layer2_2_conv3_dedx_kernel_params),
                           "dedx",
                           "layer2_2_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv3_dedx failed!");

    /*************
     * layer2_2_conv3_dedw node
     * inputs: [layer2_2_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_2_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_2_conv3_weight_grad(1, 1, 128, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer2_2_conv3_dedw_kernel_params;
    layer2_2_conv3_dedw_kernel_params.dH   = 1;
    layer2_2_conv3_dedw_kernel_params.dW   = 1;
    layer2_2_conv3_dedw_kernel_params.kH   = 1;
    layer2_2_conv3_dedw_kernel_params.kW   = 1;
    layer2_2_conv3_dedw_kernel_params.padT = 0;
    layer2_2_conv3_dedw_kernel_params.padB = 0;
    layer2_2_conv3_dedw_kernel_params.padL = 0;
    layer2_2_conv3_dedw_kernel_params.padR = 0;
    layer2_2_conv3_dedw_kernel_params.dilH = 1;
    layer2_2_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer2_2_conv3_dedw_in_vec[2] = {layer2_2_bn3_grad_input, layer2_2_relu2_output};

    // create layer2_2_conv3_weight_grad tensor
    const unsigned      layer2_2_conv3_weight_grad_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_2_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_conv3_weight_grad_tr_info = {"layer2_2_conv3_weight_grad",
                                                              layer2_2_conv3_weight_grad_dram};
    synTensor           layer2_2_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer2_2_conv3_weight_grad_sizes, true, "layer2_2_conv3_weight_grad");

    synTensor layer2_2_conv3_dedw_out_vec[1] = {layer2_2_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv3_dedw_in_vec,
                           layer2_2_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_2_conv3_dedw_kernel_params,
                           sizeof(layer2_2_conv3_dedw_kernel_params),
                           "dedw",
                           "layer2_2_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv3_dedw failed!");

    /*************
     * layer2_2_relu2_bwd node
     * inputs: [layer2_2_conv3_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_2_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_2_relu2_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_2_relu2_bwd_in_vec[2] = {layer2_2_conv3_grad_input, layer2_2_relu2_output};

    // create layer2_2_relu2_grad_input tensor
    const unsigned layer2_2_relu2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_relu2_grad_input_sizes, false, "layer2_2_relu2_grad_input");

    synTensor layer2_2_relu2_bwd_out_vec[1] = {layer2_2_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_2_relu2_bwd_in_vec,
                           layer2_2_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_2_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_relu2_bwd failed!");

    /*************
     * layer2_2_bn2_bwd node
     * inputs: [layer2_2_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_2_relu2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_2_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_2_bn2_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_2_bn2_weight[128](dtype=float32)] output: [layer2_2_bn2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_2_bn2_bias_grad(128,)(dtype=float32), layer2_2_bn2_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_2_bn2_bwd_in_vec[5] = {layer2_2_conv2_output,
                                            layer2_2_relu2_grad_input,
                                            layer2_2_bn2_saved_mean,
                                            layer2_2_bn2_saved_var,
                                            layer2_2_bn2_weight};

    // create layer2_2_bn2_grad_input tensor
    const unsigned layer2_2_bn2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_bn2_grad_input_sizes, false, "layer2_2_bn2_grad_input");

    // create layer2_2_bn2_bias_grad tensor
    const unsigned layer2_2_bn2_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_bn2_bias_grad_tr_info = {"layer2_2_bn2_bias_grad", layer2_2_bn2_bias_grad_dram};
    synTensor           layer2_2_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer2_2_bn2_bias_grad_sizes, true, "layer2_2_bn2_bias_grad");

    // create layer2_2_bn2_weight_grad tensor
    const unsigned layer2_2_bn2_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_bn2_weight_grad_tr_info = {"layer2_2_bn2_weight_grad", layer2_2_bn2_weight_grad_dram};
    synTensor           layer2_2_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer2_2_bn2_weight_grad_sizes, true, "layer2_2_bn2_weight_grad");

    synTensor layer2_2_bn2_bwd_out_vec[3] = {layer2_2_bn2_grad_input, layer2_2_bn2_bias_grad, layer2_2_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_2_bn2_bwd_in_vec,
                           layer2_2_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_2_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_bn2_bwd failed!");

    /*************
     * layer2_2_conv2_dedx node
     * inputs: [layer2_2_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_2_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_2_conv2_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_2_conv2_dedx_kernel_params;
    layer2_2_conv2_dedx_kernel_params.dH   = 1;
    layer2_2_conv2_dedx_kernel_params.dW   = 1;
    layer2_2_conv2_dedx_kernel_params.kH   = 3;
    layer2_2_conv2_dedx_kernel_params.kW   = 3;
    layer2_2_conv2_dedx_kernel_params.padT = 1;
    layer2_2_conv2_dedx_kernel_params.padB = 1;
    layer2_2_conv2_dedx_kernel_params.padL = 1;
    layer2_2_conv2_dedx_kernel_params.padR = 1;
    layer2_2_conv2_dedx_kernel_params.dilH = 1;
    layer2_2_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer2_2_conv2_dedx_in_vec[2] = {layer2_2_bn2_grad_input, layer2_2_conv2_weight};

    // create layer2_2_conv2_grad_input tensor
    const unsigned layer2_2_conv2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_conv2_grad_input_sizes, false, "layer2_2_conv2_grad_input");

    synTensor layer2_2_conv2_dedx_out_vec[1] = {layer2_2_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv2_dedx_in_vec,
                           layer2_2_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_2_conv2_dedx_kernel_params,
                           sizeof(layer2_2_conv2_dedx_kernel_params),
                           "dedx",
                           "layer2_2_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv2_dedx failed!");

    /*************
     * layer2_2_conv2_dedw node
     * inputs: [layer2_2_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_2_relu1_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_2_conv2_weight_grad(3, 3, 128, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_2_conv2_dedw_kernel_params;
    layer2_2_conv2_dedw_kernel_params.dH   = 1;
    layer2_2_conv2_dedw_kernel_params.dW   = 1;
    layer2_2_conv2_dedw_kernel_params.kH   = 3;
    layer2_2_conv2_dedw_kernel_params.kW   = 3;
    layer2_2_conv2_dedw_kernel_params.padT = 1;
    layer2_2_conv2_dedw_kernel_params.padB = 1;
    layer2_2_conv2_dedw_kernel_params.padL = 1;
    layer2_2_conv2_dedw_kernel_params.padR = 1;
    layer2_2_conv2_dedw_kernel_params.dilH = 1;
    layer2_2_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer2_2_conv2_dedw_in_vec[2] = {layer2_2_bn2_grad_input, layer2_2_relu1_output};

    // create layer2_2_conv2_weight_grad tensor
    const unsigned      layer2_2_conv2_weight_grad_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_2_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_conv2_weight_grad_tr_info = {"layer2_2_conv2_weight_grad",
                                                              layer2_2_conv2_weight_grad_dram};
    synTensor           layer2_2_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer2_2_conv2_weight_grad_sizes, true, "layer2_2_conv2_weight_grad");

    synTensor layer2_2_conv2_dedw_out_vec[1] = {layer2_2_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv2_dedw_in_vec,
                           layer2_2_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_2_conv2_dedw_kernel_params,
                           sizeof(layer2_2_conv2_dedw_kernel_params),
                           "dedw",
                           "layer2_2_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv2_dedw failed!");

    /*************
     * layer2_2_relu1_bwd node
     * inputs: [layer2_2_conv2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_2_relu1_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_2_relu1_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_2_relu1_bwd_in_vec[2] = {layer2_2_conv2_grad_input, layer2_2_relu1_output};

    // create layer2_2_relu1_grad_input tensor
    const unsigned layer2_2_relu1_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_relu1_grad_input_sizes, false, "layer2_2_relu1_grad_input");

    synTensor layer2_2_relu1_bwd_out_vec[1] = {layer2_2_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_2_relu1_bwd_in_vec,
                           layer2_2_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_2_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_relu1_bwd failed!");

    /*************
     * layer2_2_bn1_bwd node
     * inputs: [layer2_2_conv1_output(64, 28, 28, 128)(dtype=bf16), layer2_2_relu1_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_2_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_2_bn1_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_2_bn1_weight[128](dtype=float32)] output: [layer2_2_bn1_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_2_bn1_bias_grad(128,)(dtype=float32), layer2_2_bn1_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_2_bn1_bwd_in_vec[5] = {layer2_2_conv1_output,
                                            layer2_2_relu1_grad_input,
                                            layer2_2_bn1_saved_mean,
                                            layer2_2_bn1_saved_var,
                                            layer2_2_bn1_weight};

    // create layer2_2_bn1_grad_input tensor
    const unsigned layer2_2_bn1_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_2_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_bn1_grad_input_sizes, false, "layer2_2_bn1_grad_input");

    // create layer2_2_bn1_bias_grad tensor
    const unsigned layer2_2_bn1_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_bn1_bias_grad_tr_info = {"layer2_2_bn1_bias_grad", layer2_2_bn1_bias_grad_dram};
    synTensor           layer2_2_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer2_2_bn1_bias_grad_sizes, true, "layer2_2_bn1_bias_grad");

    // create layer2_2_bn1_weight_grad tensor
    const unsigned layer2_2_bn1_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_2_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_bn1_weight_grad_tr_info = {"layer2_2_bn1_weight_grad", layer2_2_bn1_weight_grad_dram};
    synTensor           layer2_2_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer2_2_bn1_weight_grad_sizes, true, "layer2_2_bn1_weight_grad");

    synTensor layer2_2_bn1_bwd_out_vec[3] = {layer2_2_bn1_grad_input, layer2_2_bn1_bias_grad, layer2_2_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_2_bn1_bwd_in_vec,
                           layer2_2_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_2_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_bn1_bwd failed!");

    /*************
     * layer2_2_conv1_dedx node
     * inputs: [layer2_2_bn1_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_2_conv1_weight[1, 1, 512, 128](dtype=bf16)]
     * output: [layer2_2_conv1_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_2_conv1_dedx_kernel_params;
    layer2_2_conv1_dedx_kernel_params.dH   = 1;
    layer2_2_conv1_dedx_kernel_params.dW   = 1;
    layer2_2_conv1_dedx_kernel_params.kH   = 1;
    layer2_2_conv1_dedx_kernel_params.kW   = 1;
    layer2_2_conv1_dedx_kernel_params.padT = 0;
    layer2_2_conv1_dedx_kernel_params.padB = 0;
    layer2_2_conv1_dedx_kernel_params.padL = 0;
    layer2_2_conv1_dedx_kernel_params.padR = 0;
    layer2_2_conv1_dedx_kernel_params.dilH = 1;
    layer2_2_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer2_2_conv1_dedx_in_vec[2] = {layer2_2_bn1_grad_input, layer2_2_conv1_weight};

    // create layer2_2_conv1_grad_input tensor
    const unsigned layer2_2_conv1_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_2_conv1_grad_input_sizes, false, "layer2_2_conv1_grad_input");

    synTensor layer2_2_conv1_dedx_out_vec[1] = {layer2_2_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv1_dedx_in_vec,
                           layer2_2_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_2_conv1_dedx_kernel_params,
                           sizeof(layer2_2_conv1_dedx_kernel_params),
                           "dedx",
                           "layer2_2_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv1_dedx failed!");

    /*************
     * layer2_2_conv1_dedw node
     * inputs: [layer2_2_bn1_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_1_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_2_conv1_weight_grad(1, 1, 512, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_2_conv1_dedw_kernel_params;
    layer2_2_conv1_dedw_kernel_params.dH   = 1;
    layer2_2_conv1_dedw_kernel_params.dW   = 1;
    layer2_2_conv1_dedw_kernel_params.kH   = 1;
    layer2_2_conv1_dedw_kernel_params.kW   = 1;
    layer2_2_conv1_dedw_kernel_params.padT = 0;
    layer2_2_conv1_dedw_kernel_params.padB = 0;
    layer2_2_conv1_dedw_kernel_params.padL = 0;
    layer2_2_conv1_dedw_kernel_params.padR = 0;
    layer2_2_conv1_dedw_kernel_params.dilH = 1;
    layer2_2_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer2_2_conv1_dedw_in_vec[2] = {layer2_2_bn1_grad_input, layer2_1_relu3_output};

    // create layer2_2_conv1_weight_grad tensor
    const unsigned      layer2_2_conv1_weight_grad_sizes[] = {1, 1, 512, 128};
    uint64_t            layer2_2_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_2_conv1_weight_grad_tr_info = {"layer2_2_conv1_weight_grad",
                                                              layer2_2_conv1_weight_grad_dram};
    synTensor           layer2_2_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer2_2_conv1_weight_grad_sizes, true, "layer2_2_conv1_weight_grad");

    synTensor layer2_2_conv1_dedw_out_vec[1] = {layer2_2_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_2_conv1_dedw_in_vec,
                           layer2_2_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_2_conv1_dedw_kernel_params,
                           sizeof(layer2_2_conv1_dedw_kernel_params),
                           "dedw",
                           "layer2_2_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_conv1_dedw failed!");

    /*************
     * layer2_2_add_residual_fwd1 node
     * inputs: [layer2_2_conv1_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_2_add_residual_grad_input1(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_2_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_2_add_residual_fwd1_in_vec[2] = {layer2_2_conv1_grad_input, layer2_2_add_residual_grad_input1};

    // create layer2_2_residual_upstream_grad_input tensor
    const unsigned layer2_2_residual_upstream_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_2_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer2_2_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer2_2_residual_upstream_grad_input");

    synTensor layer2_2_add_residual_fwd1_out_vec[1] = {layer2_2_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_2_add_residual_fwd1_in_vec,
                           layer2_2_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_2_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_2_add_residual_fwd1 failed!");

    /*************
     * layer2_1_relu3_bwd node
     * inputs: [layer2_2_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_1_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_1_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_1_relu3_bwd_in_vec[2] = {layer2_2_residual_upstream_grad_input, layer2_1_relu3_output};

    // create layer2_1_relu3_grad_input tensor
    const unsigned layer2_1_relu3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_relu3_grad_input_sizes, false, "layer2_1_relu3_grad_input");

    synTensor layer2_1_relu3_bwd_out_vec[1] = {layer2_1_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_1_relu3_bwd_in_vec,
                           layer2_1_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_1_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_relu3_bwd failed!");

    /*************
     * layer2_1_add_residual_bwd node
     * inputs: [layer2_1_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_1_add_residual_grad_input0(64, 28, 28, 512)(dtype=bf16), layer2_1_add_residual_grad_input1(64,
     *28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_1_add_residual_bwd_in_vec[1] = {layer2_1_relu3_grad_input};

    // create layer2_1_add_residual_grad_input0 tensor
    const unsigned layer2_1_add_residual_grad_input0_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_1_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer2_1_add_residual_grad_input0");

    // create layer2_1_add_residual_grad_input1 tensor
    const unsigned layer2_1_add_residual_grad_input1_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_1_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer2_1_add_residual_grad_input1");

    synTensor layer2_1_add_residual_bwd_out_vec[2] = {layer2_1_add_residual_grad_input0,
                                                      layer2_1_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer2_1_add_residual_bwd_in_vec,
                           layer2_1_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer2_1_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_add_residual_bwd failed!");

    /*************
     * layer2_1_bn3_bwd node
     * inputs: [layer2_1_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_1_add_residual_grad_input0(64, 28, 28,
     *512)(dtype=bf16), layer2_1_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_1_bn3_saved_var(1, 1, 1,
     *512)(dtype=float32), layer2_1_bn3_weight[512](dtype=float32)] output: [layer2_1_bn3_grad_input(64, 28, 28,
     *512)(dtype=bf16), layer2_1_bn3_bias_grad(512,)(dtype=float32), layer2_1_bn3_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer2_1_bn3_bwd_in_vec[5] = {layer2_1_conv3_output,
                                            layer2_1_add_residual_grad_input0,
                                            layer2_1_bn3_saved_mean,
                                            layer2_1_bn3_saved_var,
                                            layer2_1_bn3_weight};

    // create layer2_1_bn3_grad_input tensor
    const unsigned layer2_1_bn3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_bn3_grad_input_sizes, false, "layer2_1_bn3_grad_input");

    // create layer2_1_bn3_bias_grad tensor
    const unsigned layer2_1_bn3_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_1_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_bn3_bias_grad_tr_info = {"layer2_1_bn3_bias_grad", layer2_1_bn3_bias_grad_dram};
    synTensor           layer2_1_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer2_1_bn3_bias_grad_sizes, true, "layer2_1_bn3_bias_grad");

    // create layer2_1_bn3_weight_grad tensor
    const unsigned layer2_1_bn3_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_1_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_bn3_weight_grad_tr_info = {"layer2_1_bn3_weight_grad", layer2_1_bn3_weight_grad_dram};
    synTensor           layer2_1_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer2_1_bn3_weight_grad_sizes, true, "layer2_1_bn3_weight_grad");

    synTensor layer2_1_bn3_bwd_out_vec[3] = {layer2_1_bn3_grad_input, layer2_1_bn3_bias_grad, layer2_1_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_1_bn3_bwd_in_vec,
                           layer2_1_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_1_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_bn3_bwd failed!");

    /*************
     * layer2_1_conv3_dedx node
     * inputs: [layer2_1_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_1_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_1_conv3_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_1_conv3_dedx_kernel_params;
    layer2_1_conv3_dedx_kernel_params.dH   = 1;
    layer2_1_conv3_dedx_kernel_params.dW   = 1;
    layer2_1_conv3_dedx_kernel_params.kH   = 1;
    layer2_1_conv3_dedx_kernel_params.kW   = 1;
    layer2_1_conv3_dedx_kernel_params.padT = 0;
    layer2_1_conv3_dedx_kernel_params.padB = 0;
    layer2_1_conv3_dedx_kernel_params.padL = 0;
    layer2_1_conv3_dedx_kernel_params.padR = 0;
    layer2_1_conv3_dedx_kernel_params.dilH = 1;
    layer2_1_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer2_1_conv3_dedx_in_vec[2] = {layer2_1_bn3_grad_input, layer2_1_conv3_weight};

    // create layer2_1_conv3_grad_input tensor
    const unsigned layer2_1_conv3_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_conv3_grad_input_sizes, false, "layer2_1_conv3_grad_input");

    synTensor layer2_1_conv3_dedx_out_vec[1] = {layer2_1_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv3_dedx_in_vec,
                           layer2_1_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_1_conv3_dedx_kernel_params,
                           sizeof(layer2_1_conv3_dedx_kernel_params),
                           "dedx",
                           "layer2_1_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv3_dedx failed!");

    /*************
     * layer2_1_conv3_dedw node
     * inputs: [layer2_1_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_1_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_1_conv3_weight_grad(1, 1, 128, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer2_1_conv3_dedw_kernel_params;
    layer2_1_conv3_dedw_kernel_params.dH   = 1;
    layer2_1_conv3_dedw_kernel_params.dW   = 1;
    layer2_1_conv3_dedw_kernel_params.kH   = 1;
    layer2_1_conv3_dedw_kernel_params.kW   = 1;
    layer2_1_conv3_dedw_kernel_params.padT = 0;
    layer2_1_conv3_dedw_kernel_params.padB = 0;
    layer2_1_conv3_dedw_kernel_params.padL = 0;
    layer2_1_conv3_dedw_kernel_params.padR = 0;
    layer2_1_conv3_dedw_kernel_params.dilH = 1;
    layer2_1_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer2_1_conv3_dedw_in_vec[2] = {layer2_1_bn3_grad_input, layer2_1_relu2_output};

    // create layer2_1_conv3_weight_grad tensor
    const unsigned      layer2_1_conv3_weight_grad_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_1_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_conv3_weight_grad_tr_info = {"layer2_1_conv3_weight_grad",
                                                              layer2_1_conv3_weight_grad_dram};
    synTensor           layer2_1_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer2_1_conv3_weight_grad_sizes, true, "layer2_1_conv3_weight_grad");

    synTensor layer2_1_conv3_dedw_out_vec[1] = {layer2_1_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv3_dedw_in_vec,
                           layer2_1_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_1_conv3_dedw_kernel_params,
                           sizeof(layer2_1_conv3_dedw_kernel_params),
                           "dedw",
                           "layer2_1_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv3_dedw failed!");

    /*************
     * layer2_1_relu2_bwd node
     * inputs: [layer2_1_conv3_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_1_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_1_relu2_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_1_relu2_bwd_in_vec[2] = {layer2_1_conv3_grad_input, layer2_1_relu2_output};

    // create layer2_1_relu2_grad_input tensor
    const unsigned layer2_1_relu2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_relu2_grad_input_sizes, false, "layer2_1_relu2_grad_input");

    synTensor layer2_1_relu2_bwd_out_vec[1] = {layer2_1_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_1_relu2_bwd_in_vec,
                           layer2_1_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_1_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_relu2_bwd failed!");

    /*************
     * layer2_1_bn2_bwd node
     * inputs: [layer2_1_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_1_relu2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_1_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_1_bn2_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_1_bn2_weight[128](dtype=float32)] output: [layer2_1_bn2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_1_bn2_bias_grad(128,)(dtype=float32), layer2_1_bn2_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_1_bn2_bwd_in_vec[5] = {layer2_1_conv2_output,
                                            layer2_1_relu2_grad_input,
                                            layer2_1_bn2_saved_mean,
                                            layer2_1_bn2_saved_var,
                                            layer2_1_bn2_weight};

    // create layer2_1_bn2_grad_input tensor
    const unsigned layer2_1_bn2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_bn2_grad_input_sizes, false, "layer2_1_bn2_grad_input");

    // create layer2_1_bn2_bias_grad tensor
    const unsigned layer2_1_bn2_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_bn2_bias_grad_tr_info = {"layer2_1_bn2_bias_grad", layer2_1_bn2_bias_grad_dram};
    synTensor           layer2_1_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer2_1_bn2_bias_grad_sizes, true, "layer2_1_bn2_bias_grad");

    // create layer2_1_bn2_weight_grad tensor
    const unsigned layer2_1_bn2_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_bn2_weight_grad_tr_info = {"layer2_1_bn2_weight_grad", layer2_1_bn2_weight_grad_dram};
    synTensor           layer2_1_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer2_1_bn2_weight_grad_sizes, true, "layer2_1_bn2_weight_grad");

    synTensor layer2_1_bn2_bwd_out_vec[3] = {layer2_1_bn2_grad_input, layer2_1_bn2_bias_grad, layer2_1_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_1_bn2_bwd_in_vec,
                           layer2_1_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_1_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_bn2_bwd failed!");

    /*************
     * layer2_1_conv2_dedx node
     * inputs: [layer2_1_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_1_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_1_conv2_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_1_conv2_dedx_kernel_params;
    layer2_1_conv2_dedx_kernel_params.dH   = 1;
    layer2_1_conv2_dedx_kernel_params.dW   = 1;
    layer2_1_conv2_dedx_kernel_params.kH   = 3;
    layer2_1_conv2_dedx_kernel_params.kW   = 3;
    layer2_1_conv2_dedx_kernel_params.padT = 1;
    layer2_1_conv2_dedx_kernel_params.padB = 1;
    layer2_1_conv2_dedx_kernel_params.padL = 1;
    layer2_1_conv2_dedx_kernel_params.padR = 1;
    layer2_1_conv2_dedx_kernel_params.dilH = 1;
    layer2_1_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer2_1_conv2_dedx_in_vec[2] = {layer2_1_bn2_grad_input, layer2_1_conv2_weight};

    // create layer2_1_conv2_grad_input tensor
    const unsigned layer2_1_conv2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_conv2_grad_input_sizes, false, "layer2_1_conv2_grad_input");

    synTensor layer2_1_conv2_dedx_out_vec[1] = {layer2_1_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv2_dedx_in_vec,
                           layer2_1_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_1_conv2_dedx_kernel_params,
                           sizeof(layer2_1_conv2_dedx_kernel_params),
                           "dedx",
                           "layer2_1_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv2_dedx failed!");

    /*************
     * layer2_1_conv2_dedw node
     * inputs: [layer2_1_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_1_relu1_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_1_conv2_weight_grad(3, 3, 128, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_1_conv2_dedw_kernel_params;
    layer2_1_conv2_dedw_kernel_params.dH   = 1;
    layer2_1_conv2_dedw_kernel_params.dW   = 1;
    layer2_1_conv2_dedw_kernel_params.kH   = 3;
    layer2_1_conv2_dedw_kernel_params.kW   = 3;
    layer2_1_conv2_dedw_kernel_params.padT = 1;
    layer2_1_conv2_dedw_kernel_params.padB = 1;
    layer2_1_conv2_dedw_kernel_params.padL = 1;
    layer2_1_conv2_dedw_kernel_params.padR = 1;
    layer2_1_conv2_dedw_kernel_params.dilH = 1;
    layer2_1_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer2_1_conv2_dedw_in_vec[2] = {layer2_1_bn2_grad_input, layer2_1_relu1_output};

    // create layer2_1_conv2_weight_grad tensor
    const unsigned      layer2_1_conv2_weight_grad_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_1_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_conv2_weight_grad_tr_info = {"layer2_1_conv2_weight_grad",
                                                              layer2_1_conv2_weight_grad_dram};
    synTensor           layer2_1_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer2_1_conv2_weight_grad_sizes, true, "layer2_1_conv2_weight_grad");

    synTensor layer2_1_conv2_dedw_out_vec[1] = {layer2_1_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv2_dedw_in_vec,
                           layer2_1_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_1_conv2_dedw_kernel_params,
                           sizeof(layer2_1_conv2_dedw_kernel_params),
                           "dedw",
                           "layer2_1_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv2_dedw failed!");

    /*************
     * layer2_1_relu1_bwd node
     * inputs: [layer2_1_conv2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_1_relu1_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_1_relu1_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_1_relu1_bwd_in_vec[2] = {layer2_1_conv2_grad_input, layer2_1_relu1_output};

    // create layer2_1_relu1_grad_input tensor
    const unsigned layer2_1_relu1_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_relu1_grad_input_sizes, false, "layer2_1_relu1_grad_input");

    synTensor layer2_1_relu1_bwd_out_vec[1] = {layer2_1_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_1_relu1_bwd_in_vec,
                           layer2_1_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_1_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_relu1_bwd failed!");

    /*************
     * layer2_1_bn1_bwd node
     * inputs: [layer2_1_conv1_output(64, 28, 28, 128)(dtype=bf16), layer2_1_relu1_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_1_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_1_bn1_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_1_bn1_weight[128](dtype=float32)] output: [layer2_1_bn1_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_1_bn1_bias_grad(128,)(dtype=float32), layer2_1_bn1_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_1_bn1_bwd_in_vec[5] = {layer2_1_conv1_output,
                                            layer2_1_relu1_grad_input,
                                            layer2_1_bn1_saved_mean,
                                            layer2_1_bn1_saved_var,
                                            layer2_1_bn1_weight};

    // create layer2_1_bn1_grad_input tensor
    const unsigned layer2_1_bn1_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_1_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_bn1_grad_input_sizes, false, "layer2_1_bn1_grad_input");

    // create layer2_1_bn1_bias_grad tensor
    const unsigned layer2_1_bn1_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_bn1_bias_grad_tr_info = {"layer2_1_bn1_bias_grad", layer2_1_bn1_bias_grad_dram};
    synTensor           layer2_1_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer2_1_bn1_bias_grad_sizes, true, "layer2_1_bn1_bias_grad");

    // create layer2_1_bn1_weight_grad tensor
    const unsigned layer2_1_bn1_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_1_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_bn1_weight_grad_tr_info = {"layer2_1_bn1_weight_grad", layer2_1_bn1_weight_grad_dram};
    synTensor           layer2_1_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer2_1_bn1_weight_grad_sizes, true, "layer2_1_bn1_weight_grad");

    synTensor layer2_1_bn1_bwd_out_vec[3] = {layer2_1_bn1_grad_input, layer2_1_bn1_bias_grad, layer2_1_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_1_bn1_bwd_in_vec,
                           layer2_1_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_1_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_bn1_bwd failed!");

    /*************
     * layer2_1_conv1_dedx node
     * inputs: [layer2_1_bn1_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_1_conv1_weight[1, 1, 512, 128](dtype=bf16)]
     * output: [layer2_1_conv1_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_1_conv1_dedx_kernel_params;
    layer2_1_conv1_dedx_kernel_params.dH   = 1;
    layer2_1_conv1_dedx_kernel_params.dW   = 1;
    layer2_1_conv1_dedx_kernel_params.kH   = 1;
    layer2_1_conv1_dedx_kernel_params.kW   = 1;
    layer2_1_conv1_dedx_kernel_params.padT = 0;
    layer2_1_conv1_dedx_kernel_params.padB = 0;
    layer2_1_conv1_dedx_kernel_params.padL = 0;
    layer2_1_conv1_dedx_kernel_params.padR = 0;
    layer2_1_conv1_dedx_kernel_params.dilH = 1;
    layer2_1_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer2_1_conv1_dedx_in_vec[2] = {layer2_1_bn1_grad_input, layer2_1_conv1_weight};

    // create layer2_1_conv1_grad_input tensor
    const unsigned layer2_1_conv1_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_1_conv1_grad_input_sizes, false, "layer2_1_conv1_grad_input");

    synTensor layer2_1_conv1_dedx_out_vec[1] = {layer2_1_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv1_dedx_in_vec,
                           layer2_1_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_1_conv1_dedx_kernel_params,
                           sizeof(layer2_1_conv1_dedx_kernel_params),
                           "dedx",
                           "layer2_1_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv1_dedx failed!");

    /*************
     * layer2_1_conv1_dedw node
     * inputs: [layer2_1_bn1_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_0_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_1_conv1_weight_grad(1, 1, 512, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_1_conv1_dedw_kernel_params;
    layer2_1_conv1_dedw_kernel_params.dH   = 1;
    layer2_1_conv1_dedw_kernel_params.dW   = 1;
    layer2_1_conv1_dedw_kernel_params.kH   = 1;
    layer2_1_conv1_dedw_kernel_params.kW   = 1;
    layer2_1_conv1_dedw_kernel_params.padT = 0;
    layer2_1_conv1_dedw_kernel_params.padB = 0;
    layer2_1_conv1_dedw_kernel_params.padL = 0;
    layer2_1_conv1_dedw_kernel_params.padR = 0;
    layer2_1_conv1_dedw_kernel_params.dilH = 1;
    layer2_1_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer2_1_conv1_dedw_in_vec[2] = {layer2_1_bn1_grad_input, layer2_0_relu3_output};

    // create layer2_1_conv1_weight_grad tensor
    const unsigned      layer2_1_conv1_weight_grad_sizes[] = {1, 1, 512, 128};
    uint64_t            layer2_1_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_1_conv1_weight_grad_tr_info = {"layer2_1_conv1_weight_grad",
                                                              layer2_1_conv1_weight_grad_dram};
    synTensor           layer2_1_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer2_1_conv1_weight_grad_sizes, true, "layer2_1_conv1_weight_grad");

    synTensor layer2_1_conv1_dedw_out_vec[1] = {layer2_1_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_1_conv1_dedw_in_vec,
                           layer2_1_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_1_conv1_dedw_kernel_params,
                           sizeof(layer2_1_conv1_dedw_kernel_params),
                           "dedw",
                           "layer2_1_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_conv1_dedw failed!");

    /*************
     * layer2_1_add_residual_fwd1 node
     * inputs: [layer2_1_conv1_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_1_add_residual_grad_input1(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_1_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_1_add_residual_fwd1_in_vec[2] = {layer2_1_conv1_grad_input, layer2_1_add_residual_grad_input1};

    // create layer2_1_residual_upstream_grad_input tensor
    const unsigned layer2_1_residual_upstream_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_1_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer2_1_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer2_1_residual_upstream_grad_input");

    synTensor layer2_1_add_residual_fwd1_out_vec[1] = {layer2_1_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_1_add_residual_fwd1_in_vec,
                           layer2_1_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_1_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_1_add_residual_fwd1 failed!");

    /*************
     * layer2_0_relu3_bwd node
     * inputs: [layer2_1_residual_upstream_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_0_relu3_output(64, 28, 28,
     *512)(dtype=bf16)] output: [layer2_0_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_0_relu3_bwd_in_vec[2] = {layer2_1_residual_upstream_grad_input, layer2_0_relu3_output};

    // create layer2_0_relu3_grad_input tensor
    const unsigned layer2_0_relu3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_relu3_grad_input_sizes, false, "layer2_0_relu3_grad_input");

    synTensor layer2_0_relu3_bwd_out_vec[1] = {layer2_0_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_0_relu3_bwd_in_vec,
                           layer2_0_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_0_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_relu3_bwd failed!");

    /*************
     * layer2_0_add_residual_bwd node
     * inputs: [layer2_0_relu3_grad_input(64, 28, 28, 512)(dtype=bf16)]
     * output: [layer2_0_add_residual_grad_input0(64, 28, 28, 512)(dtype=bf16), layer2_0_add_residual_grad_input1(64,
     *28, 28, 512)(dtype=bf16)]
     *************/

    synTensor layer2_0_add_residual_bwd_in_vec[1] = {layer2_0_relu3_grad_input};

    // create layer2_0_add_residual_grad_input0 tensor
    const unsigned layer2_0_add_residual_grad_input0_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_0_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer2_0_add_residual_grad_input0");

    // create layer2_0_add_residual_grad_input1 tensor
    const unsigned layer2_0_add_residual_grad_input1_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer2_0_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer2_0_add_residual_grad_input1");

    synTensor layer2_0_add_residual_bwd_out_vec[2] = {layer2_0_add_residual_grad_input0,
                                                      layer2_0_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer2_0_add_residual_bwd_in_vec,
                           layer2_0_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer2_0_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_add_residual_bwd failed!");

    /*************
     * layer2_0_bn3_bwd node
     * inputs: [layer2_0_conv3_output(64, 28, 28, 512)(dtype=bf16), layer2_0_add_residual_grad_input0(64, 28, 28,
     *512)(dtype=bf16), layer2_0_bn3_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_0_bn3_saved_var(1, 1, 1,
     *512)(dtype=float32), layer2_0_bn3_weight[512](dtype=float32)] output: [layer2_0_bn3_grad_input(64, 28, 28,
     *512)(dtype=bf16), layer2_0_bn3_bias_grad(512,)(dtype=float32), layer2_0_bn3_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer2_0_bn3_bwd_in_vec[5] = {layer2_0_conv3_output,
                                            layer2_0_add_residual_grad_input0,
                                            layer2_0_bn3_saved_mean,
                                            layer2_0_bn3_saved_var,
                                            layer2_0_bn3_weight};

    // create layer2_0_bn3_grad_input tensor
    const unsigned layer2_0_bn3_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_0_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_bn3_grad_input_sizes, false, "layer2_0_bn3_grad_input");

    // create layer2_0_bn3_bias_grad tensor
    const unsigned layer2_0_bn3_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_0_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_bn3_bias_grad_tr_info = {"layer2_0_bn3_bias_grad", layer2_0_bn3_bias_grad_dram};
    synTensor           layer2_0_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer2_0_bn3_bias_grad_sizes, true, "layer2_0_bn3_bias_grad");

    // create layer2_0_bn3_weight_grad tensor
    const unsigned layer2_0_bn3_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_0_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_bn3_weight_grad_tr_info = {"layer2_0_bn3_weight_grad", layer2_0_bn3_weight_grad_dram};
    synTensor           layer2_0_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer2_0_bn3_weight_grad_sizes, true, "layer2_0_bn3_weight_grad");

    synTensor layer2_0_bn3_bwd_out_vec[3] = {layer2_0_bn3_grad_input, layer2_0_bn3_bias_grad, layer2_0_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_0_bn3_bwd_in_vec,
                           layer2_0_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_0_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_bn3_bwd failed!");

    /*************
     * layer2_0_conv3_dedx node
     * inputs: [layer2_0_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_0_conv3_weight[1, 1, 128, 512](dtype=bf16)]
     * output: [layer2_0_conv3_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_0_conv3_dedx_kernel_params;
    layer2_0_conv3_dedx_kernel_params.dH   = 1;
    layer2_0_conv3_dedx_kernel_params.dW   = 1;
    layer2_0_conv3_dedx_kernel_params.kH   = 1;
    layer2_0_conv3_dedx_kernel_params.kW   = 1;
    layer2_0_conv3_dedx_kernel_params.padT = 0;
    layer2_0_conv3_dedx_kernel_params.padB = 0;
    layer2_0_conv3_dedx_kernel_params.padL = 0;
    layer2_0_conv3_dedx_kernel_params.padR = 0;
    layer2_0_conv3_dedx_kernel_params.dilH = 1;
    layer2_0_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer2_0_conv3_dedx_in_vec[2] = {layer2_0_bn3_grad_input, layer2_0_conv3_weight};

    // create layer2_0_conv3_grad_input tensor
    const unsigned layer2_0_conv3_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_0_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_conv3_grad_input_sizes, false, "layer2_0_conv3_grad_input");

    synTensor layer2_0_conv3_dedx_out_vec[1] = {layer2_0_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv3_dedx_in_vec,
                           layer2_0_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_0_conv3_dedx_kernel_params,
                           sizeof(layer2_0_conv3_dedx_kernel_params),
                           "dedx",
                           "layer2_0_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv3_dedx failed!");

    /*************
     * layer2_0_conv3_dedw node
     * inputs: [layer2_0_bn3_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_0_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_0_conv3_weight_grad(1, 1, 128, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer2_0_conv3_dedw_kernel_params;
    layer2_0_conv3_dedw_kernel_params.dH   = 1;
    layer2_0_conv3_dedw_kernel_params.dW   = 1;
    layer2_0_conv3_dedw_kernel_params.kH   = 1;
    layer2_0_conv3_dedw_kernel_params.kW   = 1;
    layer2_0_conv3_dedw_kernel_params.padT = 0;
    layer2_0_conv3_dedw_kernel_params.padB = 0;
    layer2_0_conv3_dedw_kernel_params.padL = 0;
    layer2_0_conv3_dedw_kernel_params.padR = 0;
    layer2_0_conv3_dedw_kernel_params.dilH = 1;
    layer2_0_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer2_0_conv3_dedw_in_vec[2] = {layer2_0_bn3_grad_input, layer2_0_relu2_output};

    // create layer2_0_conv3_weight_grad tensor
    const unsigned      layer2_0_conv3_weight_grad_sizes[] = {1, 1, 128, 512};
    uint64_t            layer2_0_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_conv3_weight_grad_tr_info = {"layer2_0_conv3_weight_grad",
                                                              layer2_0_conv3_weight_grad_dram};
    synTensor           layer2_0_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer2_0_conv3_weight_grad_sizes, true, "layer2_0_conv3_weight_grad");

    synTensor layer2_0_conv3_dedw_out_vec[1] = {layer2_0_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv3_dedw_in_vec,
                           layer2_0_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_0_conv3_dedw_kernel_params,
                           sizeof(layer2_0_conv3_dedw_kernel_params),
                           "dedw",
                           "layer2_0_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv3_dedw failed!");

    /*************
     * layer2_0_relu2_bwd node
     * inputs: [layer2_0_conv3_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_0_relu2_output(64, 28, 28,
     *128)(dtype=bf16)] output: [layer2_0_relu2_grad_input(64, 28, 28, 128)(dtype=bf16)]
     *************/

    synTensor layer2_0_relu2_bwd_in_vec[2] = {layer2_0_conv3_grad_input, layer2_0_relu2_output};

    // create layer2_0_relu2_grad_input tensor
    const unsigned layer2_0_relu2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_0_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_relu2_grad_input_sizes, false, "layer2_0_relu2_grad_input");

    synTensor layer2_0_relu2_bwd_out_vec[1] = {layer2_0_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_0_relu2_bwd_in_vec,
                           layer2_0_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_0_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_relu2_bwd failed!");

    /*************
     * layer2_0_bn2_bwd node
     * inputs: [layer2_0_conv2_output(64, 28, 28, 128)(dtype=bf16), layer2_0_relu2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_0_bn2_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_0_bn2_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_0_bn2_weight[128](dtype=float32)] output: [layer2_0_bn2_grad_input(64, 28, 28,
     *128)(dtype=bf16), layer2_0_bn2_bias_grad(128,)(dtype=float32), layer2_0_bn2_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_0_bn2_bwd_in_vec[5] = {layer2_0_conv2_output,
                                            layer2_0_relu2_grad_input,
                                            layer2_0_bn2_saved_mean,
                                            layer2_0_bn2_saved_var,
                                            layer2_0_bn2_weight};

    // create layer2_0_bn2_grad_input tensor
    const unsigned layer2_0_bn2_grad_input_sizes[] = {64, 28, 28, 128};
    synTensor      layer2_0_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_bn2_grad_input_sizes, false, "layer2_0_bn2_grad_input");

    // create layer2_0_bn2_bias_grad tensor
    const unsigned layer2_0_bn2_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_bn2_bias_grad_tr_info = {"layer2_0_bn2_bias_grad", layer2_0_bn2_bias_grad_dram};
    synTensor           layer2_0_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer2_0_bn2_bias_grad_sizes, true, "layer2_0_bn2_bias_grad");

    // create layer2_0_bn2_weight_grad tensor
    const unsigned layer2_0_bn2_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_bn2_weight_grad_tr_info = {"layer2_0_bn2_weight_grad", layer2_0_bn2_weight_grad_dram};
    synTensor           layer2_0_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer2_0_bn2_weight_grad_sizes, true, "layer2_0_bn2_weight_grad");

    synTensor layer2_0_bn2_bwd_out_vec[3] = {layer2_0_bn2_grad_input, layer2_0_bn2_bias_grad, layer2_0_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_0_bn2_bwd_in_vec,
                           layer2_0_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_0_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_bn2_bwd failed!");

    /*************
     * layer2_0_conv2_dedx node
     * inputs: [layer2_0_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_0_conv2_weight[3, 3, 128, 128](dtype=bf16)]
     * output: [layer2_0_conv2_grad_input(64, 56, 56, 128)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_0_conv2_dedx_kernel_params;
    layer2_0_conv2_dedx_kernel_params.dH   = 2;
    layer2_0_conv2_dedx_kernel_params.dW   = 2;
    layer2_0_conv2_dedx_kernel_params.kH   = 3;
    layer2_0_conv2_dedx_kernel_params.kW   = 3;
    layer2_0_conv2_dedx_kernel_params.padT = 1;
    layer2_0_conv2_dedx_kernel_params.padB = 1;
    layer2_0_conv2_dedx_kernel_params.padL = 1;
    layer2_0_conv2_dedx_kernel_params.padR = 1;
    layer2_0_conv2_dedx_kernel_params.dilH = 1;
    layer2_0_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer2_0_conv2_dedx_in_vec[2] = {layer2_0_bn2_grad_input, layer2_0_conv2_weight};

    // create layer2_0_conv2_grad_input tensor
    const unsigned layer2_0_conv2_grad_input_sizes[] = {64, 56, 56, 128};
    synTensor      layer2_0_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_conv2_grad_input_sizes, false, "layer2_0_conv2_grad_input");

    synTensor layer2_0_conv2_dedx_out_vec[1] = {layer2_0_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv2_dedx_in_vec,
                           layer2_0_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_0_conv2_dedx_kernel_params,
                           sizeof(layer2_0_conv2_dedx_kernel_params),
                           "dedx",
                           "layer2_0_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv2_dedx failed!");

    /*************
     * layer2_0_conv2_dedw node
     * inputs: [layer2_0_bn2_grad_input(64, 28, 28, 128)(dtype=bf16), layer2_0_relu1_output(64, 56, 56,
     *128)(dtype=bf16)] output: [layer2_0_conv2_weight_grad(3, 3, 128, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_0_conv2_dedw_kernel_params;
    layer2_0_conv2_dedw_kernel_params.dH   = 2;
    layer2_0_conv2_dedw_kernel_params.dW   = 2;
    layer2_0_conv2_dedw_kernel_params.kH   = 3;
    layer2_0_conv2_dedw_kernel_params.kW   = 3;
    layer2_0_conv2_dedw_kernel_params.padT = 1;
    layer2_0_conv2_dedw_kernel_params.padB = 1;
    layer2_0_conv2_dedw_kernel_params.padL = 1;
    layer2_0_conv2_dedw_kernel_params.padR = 1;
    layer2_0_conv2_dedw_kernel_params.dilH = 1;
    layer2_0_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer2_0_conv2_dedw_in_vec[2] = {layer2_0_bn2_grad_input, layer2_0_relu1_output};

    // create layer2_0_conv2_weight_grad tensor
    const unsigned      layer2_0_conv2_weight_grad_sizes[] = {3, 3, 128, 128};
    uint64_t            layer2_0_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_conv2_weight_grad_tr_info = {"layer2_0_conv2_weight_grad",
                                                              layer2_0_conv2_weight_grad_dram};
    synTensor           layer2_0_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer2_0_conv2_weight_grad_sizes, true, "layer2_0_conv2_weight_grad");

    synTensor layer2_0_conv2_dedw_out_vec[1] = {layer2_0_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv2_dedw_in_vec,
                           layer2_0_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_0_conv2_dedw_kernel_params,
                           sizeof(layer2_0_conv2_dedw_kernel_params),
                           "dedw",
                           "layer2_0_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv2_dedw failed!");

    /*************
     * layer2_0_relu1_bwd node
     * inputs: [layer2_0_conv2_grad_input(64, 56, 56, 128)(dtype=bf16), layer2_0_relu1_output(64, 56, 56,
     *128)(dtype=bf16)] output: [layer2_0_relu1_grad_input(64, 56, 56, 128)(dtype=bf16)]
     *************/

    synTensor layer2_0_relu1_bwd_in_vec[2] = {layer2_0_conv2_grad_input, layer2_0_relu1_output};

    // create layer2_0_relu1_grad_input tensor
    const unsigned layer2_0_relu1_grad_input_sizes[] = {64, 56, 56, 128};
    synTensor      layer2_0_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_relu1_grad_input_sizes, false, "layer2_0_relu1_grad_input");

    synTensor layer2_0_relu1_bwd_out_vec[1] = {layer2_0_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_0_relu1_bwd_in_vec,
                           layer2_0_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer2_0_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_relu1_bwd failed!");

    /*************
     * layer2_0_bn1_bwd node
     * inputs: [layer2_0_conv1_output(64, 56, 56, 128)(dtype=bf16), layer2_0_relu1_grad_input(64, 56, 56,
     *128)(dtype=bf16), layer2_0_bn1_saved_mean(1, 1, 1, 128)(dtype=float32), layer2_0_bn1_saved_var(1, 1, 1,
     *128)(dtype=float32), layer2_0_bn1_weight[128](dtype=float32)] output: [layer2_0_bn1_grad_input(64, 56, 56,
     *128)(dtype=bf16), layer2_0_bn1_bias_grad(128,)(dtype=float32), layer2_0_bn1_weight_grad(128,)(dtype=float32)]
     *************/

    synTensor layer2_0_bn1_bwd_in_vec[5] = {layer2_0_conv1_output,
                                            layer2_0_relu1_grad_input,
                                            layer2_0_bn1_saved_mean,
                                            layer2_0_bn1_saved_var,
                                            layer2_0_bn1_weight};

    // create layer2_0_bn1_grad_input tensor
    const unsigned layer2_0_bn1_grad_input_sizes[] = {64, 56, 56, 128};
    synTensor      layer2_0_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_bn1_grad_input_sizes, false, "layer2_0_bn1_grad_input");

    // create layer2_0_bn1_bias_grad tensor
    const unsigned layer2_0_bn1_bias_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_bn1_bias_grad_tr_info = {"layer2_0_bn1_bias_grad", layer2_0_bn1_bias_grad_dram};
    synTensor           layer2_0_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer2_0_bn1_bias_grad_sizes, true, "layer2_0_bn1_bias_grad");

    // create layer2_0_bn1_weight_grad tensor
    const unsigned layer2_0_bn1_weight_grad_sizes[] = {
        128,
    };
    uint64_t            layer2_0_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_bn1_weight_grad_tr_info = {"layer2_0_bn1_weight_grad", layer2_0_bn1_weight_grad_dram};
    synTensor           layer2_0_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer2_0_bn1_weight_grad_sizes, true, "layer2_0_bn1_weight_grad");

    synTensor layer2_0_bn1_bwd_out_vec[3] = {layer2_0_bn1_grad_input, layer2_0_bn1_bias_grad, layer2_0_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_0_bn1_bwd_in_vec,
                           layer2_0_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_0_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_bn1_bwd failed!");

    /*************
     * layer2_0_conv1_dedx node
     * inputs: [layer2_0_bn1_grad_input(64, 56, 56, 128)(dtype=bf16), layer2_0_conv1_weight[1, 1, 256, 128](dtype=bf16)]
     * output: [layer2_0_conv1_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_0_conv1_dedx_kernel_params;
    layer2_0_conv1_dedx_kernel_params.dH   = 1;
    layer2_0_conv1_dedx_kernel_params.dW   = 1;
    layer2_0_conv1_dedx_kernel_params.kH   = 1;
    layer2_0_conv1_dedx_kernel_params.kW   = 1;
    layer2_0_conv1_dedx_kernel_params.padT = 0;
    layer2_0_conv1_dedx_kernel_params.padB = 0;
    layer2_0_conv1_dedx_kernel_params.padL = 0;
    layer2_0_conv1_dedx_kernel_params.padR = 0;
    layer2_0_conv1_dedx_kernel_params.dilH = 1;
    layer2_0_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer2_0_conv1_dedx_in_vec[2] = {layer2_0_bn1_grad_input, layer2_0_conv1_weight};

    // create layer2_0_conv1_grad_input tensor
    const unsigned layer2_0_conv1_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer2_0_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer2_0_conv1_grad_input_sizes, false, "layer2_0_conv1_grad_input");

    synTensor layer2_0_conv1_dedx_out_vec[1] = {layer2_0_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv1_dedx_in_vec,
                           layer2_0_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_0_conv1_dedx_kernel_params,
                           sizeof(layer2_0_conv1_dedx_kernel_params),
                           "dedx",
                           "layer2_0_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv1_dedx failed!");

    /*************
     * layer2_0_conv1_dedw node
     * inputs: [layer2_0_bn1_grad_input(64, 56, 56, 128)(dtype=bf16), layer1_2_relu3_output(64, 56, 56,
     *256)(dtype=bf16)] output: [layer2_0_conv1_weight_grad(1, 1, 256, 128)(dtype=float32)]
     *************/
    synConvolutionParams layer2_0_conv1_dedw_kernel_params;
    layer2_0_conv1_dedw_kernel_params.dH   = 1;
    layer2_0_conv1_dedw_kernel_params.dW   = 1;
    layer2_0_conv1_dedw_kernel_params.kH   = 1;
    layer2_0_conv1_dedw_kernel_params.kW   = 1;
    layer2_0_conv1_dedw_kernel_params.padT = 0;
    layer2_0_conv1_dedw_kernel_params.padB = 0;
    layer2_0_conv1_dedw_kernel_params.padL = 0;
    layer2_0_conv1_dedw_kernel_params.padR = 0;
    layer2_0_conv1_dedw_kernel_params.dilH = 1;
    layer2_0_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer2_0_conv1_dedw_in_vec[2] = {layer2_0_bn1_grad_input, layer1_2_relu3_output};

    // create layer2_0_conv1_weight_grad tensor
    const unsigned      layer2_0_conv1_weight_grad_sizes[] = {1, 1, 256, 128};
    uint64_t            layer2_0_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_0_conv1_weight_grad_tr_info = {"layer2_0_conv1_weight_grad",
                                                              layer2_0_conv1_weight_grad_dram};
    synTensor           layer2_0_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer2_0_conv1_weight_grad_sizes, true, "layer2_0_conv1_weight_grad");

    synTensor layer2_0_conv1_dedw_out_vec[1] = {layer2_0_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_0_conv1_dedw_in_vec,
                           layer2_0_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_0_conv1_dedw_kernel_params,
                           sizeof(layer2_0_conv1_dedw_kernel_params),
                           "dedw",
                           "layer2_0_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_conv1_dedw failed!");

    /*************
     * layer2_bn_bwd node
     * inputs: [layer2_downsample_output(64, 28, 28, 512)(dtype=bf16), layer2_0_add_residual_grad_input1(64, 28, 28,
     *512)(dtype=bf16), layer2_bn_saved_mean(1, 1, 1, 512)(dtype=float32), layer2_bn_saved_var(1, 1, 1,
     *512)(dtype=float32), layer2_bn_weight[512](dtype=float32)] output: [layer2_bn_grad_input(64, 28, 28,
     *512)(dtype=bf16), layer2_bn_bias_grad(512,)(dtype=float32), layer2_bn_weight_grad(512,)(dtype=float32)]
     *************/

    synTensor layer2_bn_bwd_in_vec[5] = {layer2_downsample_output,
                                         layer2_0_add_residual_grad_input1,
                                         layer2_bn_saved_mean,
                                         layer2_bn_saved_var,
                                         layer2_bn_weight};

    // create layer2_bn_grad_input tensor
    const unsigned layer2_bn_grad_input_sizes[] = {64, 28, 28, 512};
    synTensor      layer2_bn_grad_input =
        createTensor(4U, syn_type_bf16, layer2_bn_grad_input_sizes, false, "layer2_bn_grad_input");

    // create layer2_bn_bias_grad tensor
    const unsigned layer2_bn_bias_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_bn_bias_grad_dram    = 0;
    synLaunchTensorInfo layer2_bn_bias_grad_tr_info = {"layer2_bn_bias_grad", layer2_bn_bias_grad_dram};
    synTensor           layer2_bn_bias_grad =
        createTensor(1U, syn_type_single, layer2_bn_bias_grad_sizes, true, "layer2_bn_bias_grad");

    // create layer2_bn_weight_grad tensor
    const unsigned layer2_bn_weight_grad_sizes[] = {
        512,
    };
    uint64_t            layer2_bn_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_bn_weight_grad_tr_info = {"layer2_bn_weight_grad", layer2_bn_weight_grad_dram};
    synTensor           layer2_bn_weight_grad =
        createTensor(1U, syn_type_single, layer2_bn_weight_grad_sizes, true, "layer2_bn_weight_grad");

    synTensor layer2_bn_bwd_out_vec[3] = {layer2_bn_grad_input, layer2_bn_bias_grad, layer2_bn_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_bn_bwd_in_vec,
                           layer2_bn_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer2_bn_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_bn_bwd failed!");

    /*************
     * layer2_downsample_dedx node
     * inputs: [layer2_bn_grad_input(64, 28, 28, 512)(dtype=bf16), layer2_downsample_weight[1, 1, 256, 512](dtype=bf16)]
     * output: [layer2_downsample_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer2_downsample_dedx_kernel_params;
    layer2_downsample_dedx_kernel_params.dH   = 2;
    layer2_downsample_dedx_kernel_params.dW   = 2;
    layer2_downsample_dedx_kernel_params.kH   = 1;
    layer2_downsample_dedx_kernel_params.kW   = 1;
    layer2_downsample_dedx_kernel_params.padT = 0;
    layer2_downsample_dedx_kernel_params.padB = 0;
    layer2_downsample_dedx_kernel_params.padL = 0;
    layer2_downsample_dedx_kernel_params.padR = 0;
    layer2_downsample_dedx_kernel_params.dilH = 1;
    layer2_downsample_dedx_kernel_params.dilW = 1;

    synTensor layer2_downsample_dedx_in_vec[2] = {layer2_bn_grad_input, layer2_downsample_weight};

    // create layer2_downsample_grad_input tensor
    const unsigned layer2_downsample_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer2_downsample_grad_input =
        createTensor(4U, syn_type_bf16, layer2_downsample_grad_input_sizes, false, "layer2_downsample_grad_input");

    synTensor layer2_downsample_dedx_out_vec[1] = {layer2_downsample_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_downsample_dedx_in_vec,
                           layer2_downsample_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer2_downsample_dedx_kernel_params,
                           sizeof(layer2_downsample_dedx_kernel_params),
                           "dedx",
                           "layer2_downsample_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_downsample_dedx failed!");

    /*************
     * layer2_downsample_dedw node
     * inputs: [layer2_bn_grad_input(64, 28, 28, 512)(dtype=bf16), layer1_2_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer2_downsample_weight_grad(1, 1, 256, 512)(dtype=float32)]
     *************/
    synConvolutionParams layer2_downsample_dedw_kernel_params;
    layer2_downsample_dedw_kernel_params.dH   = 2;
    layer2_downsample_dedw_kernel_params.dW   = 2;
    layer2_downsample_dedw_kernel_params.kH   = 1;
    layer2_downsample_dedw_kernel_params.kW   = 1;
    layer2_downsample_dedw_kernel_params.padT = 0;
    layer2_downsample_dedw_kernel_params.padB = 0;
    layer2_downsample_dedw_kernel_params.padL = 0;
    layer2_downsample_dedw_kernel_params.padR = 0;
    layer2_downsample_dedw_kernel_params.dilH = 1;
    layer2_downsample_dedw_kernel_params.dilW = 1;

    synTensor layer2_downsample_dedw_in_vec[2] = {layer2_bn_grad_input, layer1_2_relu3_output};

    // create layer2_downsample_weight_grad tensor
    const unsigned      layer2_downsample_weight_grad_sizes[] = {1, 1, 256, 512};
    uint64_t            layer2_downsample_weight_grad_dram    = 0;
    synLaunchTensorInfo layer2_downsample_weight_grad_tr_info = {"layer2_downsample_weight_grad",
                                                                 layer2_downsample_weight_grad_dram};
    synTensor           layer2_downsample_weight_grad =
        createTensor(4U, syn_type_single, layer2_downsample_weight_grad_sizes, true, "layer2_downsample_weight_grad");

    synTensor layer2_downsample_dedw_out_vec[1] = {layer2_downsample_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer2_downsample_dedw_in_vec,
                           layer2_downsample_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer2_downsample_dedw_kernel_params,
                           sizeof(layer2_downsample_dedw_kernel_params),
                           "dedw",
                           "layer2_downsample_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_downsample_dedw failed!");

    /*************
     * layer2_0_add_residual_fwd1 node
     * inputs: [layer2_0_conv1_grad_input(64, 56, 56, 256)(dtype=bf16), layer2_downsample_grad_input(64, 56, 56,
     *256)(dtype=bf16)] output: [layer2_0_residual_upstream_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer2_0_add_residual_fwd1_in_vec[2] = {layer2_0_conv1_grad_input, layer2_downsample_grad_input};

    // create layer2_0_residual_upstream_grad_input tensor
    const unsigned layer2_0_residual_upstream_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer2_0_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer2_0_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer2_0_residual_upstream_grad_input");

    synTensor layer2_0_add_residual_fwd1_out_vec[1] = {layer2_0_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer2_0_add_residual_fwd1_in_vec,
                           layer2_0_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer2_0_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer2_0_add_residual_fwd1 failed!");

    /*************
     * layer1_2_relu3_bwd node
     * inputs: [layer2_0_residual_upstream_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_2_relu3_output(64, 56, 56,
     *256)(dtype=bf16)] output: [layer1_2_relu3_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_2_relu3_bwd_in_vec[2] = {layer2_0_residual_upstream_grad_input, layer1_2_relu3_output};

    // create layer1_2_relu3_grad_input tensor
    const unsigned layer1_2_relu3_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_relu3_grad_input_sizes, false, "layer1_2_relu3_grad_input");

    synTensor layer1_2_relu3_bwd_out_vec[1] = {layer1_2_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_2_relu3_bwd_in_vec,
                           layer1_2_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_2_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_relu3_bwd failed!");

    /*************
     * layer1_2_add_residual_bwd node
     * inputs: [layer1_2_relu3_grad_input(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_2_add_residual_grad_input0(64, 56, 56, 256)(dtype=bf16), layer1_2_add_residual_grad_input1(64,
     *56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_2_add_residual_bwd_in_vec[1] = {layer1_2_relu3_grad_input};

    // create layer1_2_add_residual_grad_input0 tensor
    const unsigned layer1_2_add_residual_grad_input0_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer1_2_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer1_2_add_residual_grad_input0");

    // create layer1_2_add_residual_grad_input1 tensor
    const unsigned layer1_2_add_residual_grad_input1_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer1_2_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer1_2_add_residual_grad_input1");

    synTensor layer1_2_add_residual_bwd_out_vec[2] = {layer1_2_add_residual_grad_input0,
                                                      layer1_2_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer1_2_add_residual_bwd_in_vec,
                           layer1_2_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer1_2_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_add_residual_bwd failed!");

    /*************
     * layer1_2_bn3_bwd node
     * inputs: [layer1_2_conv3_output(64, 56, 56, 256)(dtype=bf16), layer1_2_add_residual_grad_input0(64, 56, 56,
     *256)(dtype=bf16), layer1_2_bn3_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_2_bn3_saved_var(1, 1, 1,
     *256)(dtype=float32), layer1_2_bn3_weight[256](dtype=float32)] output: [layer1_2_bn3_grad_input(64, 56, 56,
     *256)(dtype=bf16), layer1_2_bn3_bias_grad(256,)(dtype=float32), layer1_2_bn3_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer1_2_bn3_bwd_in_vec[5] = {layer1_2_conv3_output,
                                            layer1_2_add_residual_grad_input0,
                                            layer1_2_bn3_saved_mean,
                                            layer1_2_bn3_saved_var,
                                            layer1_2_bn3_weight};

    // create layer1_2_bn3_grad_input tensor
    const unsigned layer1_2_bn3_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_bn3_grad_input_sizes, false, "layer1_2_bn3_grad_input");

    // create layer1_2_bn3_bias_grad tensor
    const unsigned layer1_2_bn3_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_2_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_bn3_bias_grad_tr_info = {"layer1_2_bn3_bias_grad", layer1_2_bn3_bias_grad_dram};
    synTensor           layer1_2_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer1_2_bn3_bias_grad_sizes, true, "layer1_2_bn3_bias_grad");

    // create layer1_2_bn3_weight_grad tensor
    const unsigned layer1_2_bn3_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_2_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_bn3_weight_grad_tr_info = {"layer1_2_bn3_weight_grad", layer1_2_bn3_weight_grad_dram};
    synTensor           layer1_2_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer1_2_bn3_weight_grad_sizes, true, "layer1_2_bn3_weight_grad");

    synTensor layer1_2_bn3_bwd_out_vec[3] = {layer1_2_bn3_grad_input, layer1_2_bn3_bias_grad, layer1_2_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_2_bn3_bwd_in_vec,
                           layer1_2_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_2_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_bn3_bwd failed!");

    /*************
     * layer1_2_conv3_dedx node
     * inputs: [layer1_2_bn3_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_2_conv3_weight[1, 1, 64, 256](dtype=bf16)]
     * output: [layer1_2_conv3_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_2_conv3_dedx_kernel_params;
    layer1_2_conv3_dedx_kernel_params.dH   = 1;
    layer1_2_conv3_dedx_kernel_params.dW   = 1;
    layer1_2_conv3_dedx_kernel_params.kH   = 1;
    layer1_2_conv3_dedx_kernel_params.kW   = 1;
    layer1_2_conv3_dedx_kernel_params.padT = 0;
    layer1_2_conv3_dedx_kernel_params.padB = 0;
    layer1_2_conv3_dedx_kernel_params.padL = 0;
    layer1_2_conv3_dedx_kernel_params.padR = 0;
    layer1_2_conv3_dedx_kernel_params.dilH = 1;
    layer1_2_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer1_2_conv3_dedx_in_vec[2] = {layer1_2_bn3_grad_input, layer1_2_conv3_weight};

    // create layer1_2_conv3_grad_input tensor
    const unsigned layer1_2_conv3_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_conv3_grad_input_sizes, false, "layer1_2_conv3_grad_input");

    synTensor layer1_2_conv3_dedx_out_vec[1] = {layer1_2_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv3_dedx_in_vec,
                           layer1_2_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_2_conv3_dedx_kernel_params,
                           sizeof(layer1_2_conv3_dedx_kernel_params),
                           "dedx",
                           "layer1_2_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv3_dedx failed!");

    /*************
     * layer1_2_conv3_dedw node
     * inputs: [layer1_2_bn3_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_2_relu2_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_2_conv3_weight_grad(1, 1, 64, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer1_2_conv3_dedw_kernel_params;
    layer1_2_conv3_dedw_kernel_params.dH   = 1;
    layer1_2_conv3_dedw_kernel_params.dW   = 1;
    layer1_2_conv3_dedw_kernel_params.kH   = 1;
    layer1_2_conv3_dedw_kernel_params.kW   = 1;
    layer1_2_conv3_dedw_kernel_params.padT = 0;
    layer1_2_conv3_dedw_kernel_params.padB = 0;
    layer1_2_conv3_dedw_kernel_params.padL = 0;
    layer1_2_conv3_dedw_kernel_params.padR = 0;
    layer1_2_conv3_dedw_kernel_params.dilH = 1;
    layer1_2_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer1_2_conv3_dedw_in_vec[2] = {layer1_2_bn3_grad_input, layer1_2_relu2_output};

    // create layer1_2_conv3_weight_grad tensor
    const unsigned      layer1_2_conv3_weight_grad_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_2_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_conv3_weight_grad_tr_info = {"layer1_2_conv3_weight_grad",
                                                              layer1_2_conv3_weight_grad_dram};
    synTensor           layer1_2_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer1_2_conv3_weight_grad_sizes, true, "layer1_2_conv3_weight_grad");

    synTensor layer1_2_conv3_dedw_out_vec[1] = {layer1_2_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv3_dedw_in_vec,
                           layer1_2_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_2_conv3_dedw_kernel_params,
                           sizeof(layer1_2_conv3_dedw_kernel_params),
                           "dedw",
                           "layer1_2_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv3_dedw failed!");

    /*************
     * layer1_2_relu2_bwd node
     * inputs: [layer1_2_conv3_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_2_relu2_output(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_2_relu2_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_2_relu2_bwd_in_vec[2] = {layer1_2_conv3_grad_input, layer1_2_relu2_output};

    // create layer1_2_relu2_grad_input tensor
    const unsigned layer1_2_relu2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_relu2_grad_input_sizes, false, "layer1_2_relu2_grad_input");

    synTensor layer1_2_relu2_bwd_out_vec[1] = {layer1_2_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_2_relu2_bwd_in_vec,
                           layer1_2_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_2_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_relu2_bwd failed!");

    /*************
     * layer1_2_bn2_bwd node
     * inputs: [layer1_2_conv2_output(64, 56, 56, 64)(dtype=bf16), layer1_2_relu2_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_2_bn2_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_2_bn2_saved_var(1, 1, 1,
     *64)(dtype=float32), layer1_2_bn2_weight[64](dtype=float32)] output: [layer1_2_bn2_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_2_bn2_bias_grad(64,)(dtype=float32), layer1_2_bn2_weight_grad(64,)(dtype=float32)]
     *************/

    synTensor layer1_2_bn2_bwd_in_vec[5] = {layer1_2_conv2_output,
                                            layer1_2_relu2_grad_input,
                                            layer1_2_bn2_saved_mean,
                                            layer1_2_bn2_saved_var,
                                            layer1_2_bn2_weight};

    // create layer1_2_bn2_grad_input tensor
    const unsigned layer1_2_bn2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_bn2_grad_input_sizes, false, "layer1_2_bn2_grad_input");

    // create layer1_2_bn2_bias_grad tensor
    const unsigned layer1_2_bn2_bias_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_bn2_bias_grad_tr_info = {"layer1_2_bn2_bias_grad", layer1_2_bn2_bias_grad_dram};
    synTensor           layer1_2_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer1_2_bn2_bias_grad_sizes, true, "layer1_2_bn2_bias_grad");

    // create layer1_2_bn2_weight_grad tensor
    const unsigned layer1_2_bn2_weight_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_bn2_weight_grad_tr_info = {"layer1_2_bn2_weight_grad", layer1_2_bn2_weight_grad_dram};
    synTensor           layer1_2_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer1_2_bn2_weight_grad_sizes, true, "layer1_2_bn2_weight_grad");

    synTensor layer1_2_bn2_bwd_out_vec[3] = {layer1_2_bn2_grad_input, layer1_2_bn2_bias_grad, layer1_2_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_2_bn2_bwd_in_vec,
                           layer1_2_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_2_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_bn2_bwd failed!");

    /*************
     * layer1_2_conv2_dedx node
     * inputs: [layer1_2_bn2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_2_conv2_weight[3, 3, 64, 64](dtype=bf16)]
     * output: [layer1_2_conv2_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_2_conv2_dedx_kernel_params;
    layer1_2_conv2_dedx_kernel_params.dH   = 1;
    layer1_2_conv2_dedx_kernel_params.dW   = 1;
    layer1_2_conv2_dedx_kernel_params.kH   = 3;
    layer1_2_conv2_dedx_kernel_params.kW   = 3;
    layer1_2_conv2_dedx_kernel_params.padT = 1;
    layer1_2_conv2_dedx_kernel_params.padB = 1;
    layer1_2_conv2_dedx_kernel_params.padL = 1;
    layer1_2_conv2_dedx_kernel_params.padR = 1;
    layer1_2_conv2_dedx_kernel_params.dilH = 1;
    layer1_2_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer1_2_conv2_dedx_in_vec[2] = {layer1_2_bn2_grad_input, layer1_2_conv2_weight};

    // create layer1_2_conv2_grad_input tensor
    const unsigned layer1_2_conv2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_conv2_grad_input_sizes, false, "layer1_2_conv2_grad_input");

    synTensor layer1_2_conv2_dedx_out_vec[1] = {layer1_2_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv2_dedx_in_vec,
                           layer1_2_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_2_conv2_dedx_kernel_params,
                           sizeof(layer1_2_conv2_dedx_kernel_params),
                           "dedx",
                           "layer1_2_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv2_dedx failed!");

    /*************
     * layer1_2_conv2_dedw node
     * inputs: [layer1_2_bn2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_2_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_2_conv2_weight_grad(3, 3, 64, 64)(dtype=float32)]
     *************/
    synConvolutionParams layer1_2_conv2_dedw_kernel_params;
    layer1_2_conv2_dedw_kernel_params.dH   = 1;
    layer1_2_conv2_dedw_kernel_params.dW   = 1;
    layer1_2_conv2_dedw_kernel_params.kH   = 3;
    layer1_2_conv2_dedw_kernel_params.kW   = 3;
    layer1_2_conv2_dedw_kernel_params.padT = 1;
    layer1_2_conv2_dedw_kernel_params.padB = 1;
    layer1_2_conv2_dedw_kernel_params.padL = 1;
    layer1_2_conv2_dedw_kernel_params.padR = 1;
    layer1_2_conv2_dedw_kernel_params.dilH = 1;
    layer1_2_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer1_2_conv2_dedw_in_vec[2] = {layer1_2_bn2_grad_input, layer1_2_relu1_output};

    // create layer1_2_conv2_weight_grad tensor
    const unsigned      layer1_2_conv2_weight_grad_sizes[] = {3, 3, 64, 64};
    uint64_t            layer1_2_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_conv2_weight_grad_tr_info = {"layer1_2_conv2_weight_grad",
                                                              layer1_2_conv2_weight_grad_dram};
    synTensor           layer1_2_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer1_2_conv2_weight_grad_sizes, true, "layer1_2_conv2_weight_grad");

    synTensor layer1_2_conv2_dedw_out_vec[1] = {layer1_2_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv2_dedw_in_vec,
                           layer1_2_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_2_conv2_dedw_kernel_params,
                           sizeof(layer1_2_conv2_dedw_kernel_params),
                           "dedw",
                           "layer1_2_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv2_dedw failed!");

    /*************
     * layer1_2_relu1_bwd node
     * inputs: [layer1_2_conv2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_2_relu1_output(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_2_relu1_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_2_relu1_bwd_in_vec[2] = {layer1_2_conv2_grad_input, layer1_2_relu1_output};

    // create layer1_2_relu1_grad_input tensor
    const unsigned layer1_2_relu1_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_relu1_grad_input_sizes, false, "layer1_2_relu1_grad_input");

    synTensor layer1_2_relu1_bwd_out_vec[1] = {layer1_2_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_2_relu1_bwd_in_vec,
                           layer1_2_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_2_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_relu1_bwd failed!");

    /*************
     * layer1_2_bn1_bwd node
     * inputs: [layer1_2_conv1_output(64, 56, 56, 64)(dtype=bf16), layer1_2_relu1_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_2_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_2_bn1_saved_var(1, 1, 1,
     *64)(dtype=float32), layer1_2_bn1_weight[64](dtype=float32)] output: [layer1_2_bn1_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_2_bn1_bias_grad(64,)(dtype=float32), layer1_2_bn1_weight_grad(64,)(dtype=float32)]
     *************/

    synTensor layer1_2_bn1_bwd_in_vec[5] = {layer1_2_conv1_output,
                                            layer1_2_relu1_grad_input,
                                            layer1_2_bn1_saved_mean,
                                            layer1_2_bn1_saved_var,
                                            layer1_2_bn1_weight};

    // create layer1_2_bn1_grad_input tensor
    const unsigned layer1_2_bn1_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_2_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_bn1_grad_input_sizes, false, "layer1_2_bn1_grad_input");

    // create layer1_2_bn1_bias_grad tensor
    const unsigned layer1_2_bn1_bias_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_bn1_bias_grad_tr_info = {"layer1_2_bn1_bias_grad", layer1_2_bn1_bias_grad_dram};
    synTensor           layer1_2_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer1_2_bn1_bias_grad_sizes, true, "layer1_2_bn1_bias_grad");

    // create layer1_2_bn1_weight_grad tensor
    const unsigned layer1_2_bn1_weight_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_2_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_bn1_weight_grad_tr_info = {"layer1_2_bn1_weight_grad", layer1_2_bn1_weight_grad_dram};
    synTensor           layer1_2_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer1_2_bn1_weight_grad_sizes, true, "layer1_2_bn1_weight_grad");

    synTensor layer1_2_bn1_bwd_out_vec[3] = {layer1_2_bn1_grad_input, layer1_2_bn1_bias_grad, layer1_2_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_2_bn1_bwd_in_vec,
                           layer1_2_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_2_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_bn1_bwd failed!");

    /*************
     * layer1_2_conv1_dedx node
     * inputs: [layer1_2_bn1_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_2_conv1_weight[1, 1, 256, 64](dtype=bf16)]
     * output: [layer1_2_conv1_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_2_conv1_dedx_kernel_params;
    layer1_2_conv1_dedx_kernel_params.dH   = 1;
    layer1_2_conv1_dedx_kernel_params.dW   = 1;
    layer1_2_conv1_dedx_kernel_params.kH   = 1;
    layer1_2_conv1_dedx_kernel_params.kW   = 1;
    layer1_2_conv1_dedx_kernel_params.padT = 0;
    layer1_2_conv1_dedx_kernel_params.padB = 0;
    layer1_2_conv1_dedx_kernel_params.padL = 0;
    layer1_2_conv1_dedx_kernel_params.padR = 0;
    layer1_2_conv1_dedx_kernel_params.dilH = 1;
    layer1_2_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer1_2_conv1_dedx_in_vec[2] = {layer1_2_bn1_grad_input, layer1_2_conv1_weight};

    // create layer1_2_conv1_grad_input tensor
    const unsigned layer1_2_conv1_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_2_conv1_grad_input_sizes, false, "layer1_2_conv1_grad_input");

    synTensor layer1_2_conv1_dedx_out_vec[1] = {layer1_2_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv1_dedx_in_vec,
                           layer1_2_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_2_conv1_dedx_kernel_params,
                           sizeof(layer1_2_conv1_dedx_kernel_params),
                           "dedx",
                           "layer1_2_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv1_dedx failed!");

    /*************
     * layer1_2_conv1_dedw node
     * inputs: [layer1_2_bn1_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_1_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_2_conv1_weight_grad(1, 1, 256, 64)(dtype=float32)]
     *************/
    synConvolutionParams layer1_2_conv1_dedw_kernel_params;
    layer1_2_conv1_dedw_kernel_params.dH   = 1;
    layer1_2_conv1_dedw_kernel_params.dW   = 1;
    layer1_2_conv1_dedw_kernel_params.kH   = 1;
    layer1_2_conv1_dedw_kernel_params.kW   = 1;
    layer1_2_conv1_dedw_kernel_params.padT = 0;
    layer1_2_conv1_dedw_kernel_params.padB = 0;
    layer1_2_conv1_dedw_kernel_params.padL = 0;
    layer1_2_conv1_dedw_kernel_params.padR = 0;
    layer1_2_conv1_dedw_kernel_params.dilH = 1;
    layer1_2_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer1_2_conv1_dedw_in_vec[2] = {layer1_2_bn1_grad_input, layer1_1_relu3_output};

    // create layer1_2_conv1_weight_grad tensor
    const unsigned      layer1_2_conv1_weight_grad_sizes[] = {1, 1, 256, 64};
    uint64_t            layer1_2_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_2_conv1_weight_grad_tr_info = {"layer1_2_conv1_weight_grad",
                                                              layer1_2_conv1_weight_grad_dram};
    synTensor           layer1_2_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer1_2_conv1_weight_grad_sizes, true, "layer1_2_conv1_weight_grad");

    synTensor layer1_2_conv1_dedw_out_vec[1] = {layer1_2_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_2_conv1_dedw_in_vec,
                           layer1_2_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_2_conv1_dedw_kernel_params,
                           sizeof(layer1_2_conv1_dedw_kernel_params),
                           "dedw",
                           "layer1_2_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_conv1_dedw failed!");

    /*************
     * layer1_2_add_residual_fwd1 node
     * inputs: [layer1_2_conv1_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_2_add_residual_grad_input1(64, 56, 56,
     *256)(dtype=bf16)] output: [layer1_2_residual_upstream_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_2_add_residual_fwd1_in_vec[2] = {layer1_2_conv1_grad_input, layer1_2_add_residual_grad_input1};

    // create layer1_2_residual_upstream_grad_input tensor
    const unsigned layer1_2_residual_upstream_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_2_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer1_2_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer1_2_residual_upstream_grad_input");

    synTensor layer1_2_add_residual_fwd1_out_vec[1] = {layer1_2_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_2_add_residual_fwd1_in_vec,
                           layer1_2_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer1_2_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_2_add_residual_fwd1 failed!");

    /*************
     * layer1_1_relu3_bwd node
     * inputs: [layer1_2_residual_upstream_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_1_relu3_output(64, 56, 56,
     *256)(dtype=bf16)] output: [layer1_1_relu3_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_1_relu3_bwd_in_vec[2] = {layer1_2_residual_upstream_grad_input, layer1_1_relu3_output};

    // create layer1_1_relu3_grad_input tensor
    const unsigned layer1_1_relu3_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_relu3_grad_input_sizes, false, "layer1_1_relu3_grad_input");

    synTensor layer1_1_relu3_bwd_out_vec[1] = {layer1_1_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_1_relu3_bwd_in_vec,
                           layer1_1_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_1_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_relu3_bwd failed!");

    /*************
     * layer1_1_add_residual_bwd node
     * inputs: [layer1_1_relu3_grad_input(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_1_add_residual_grad_input0(64, 56, 56, 256)(dtype=bf16), layer1_1_add_residual_grad_input1(64,
     *56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_1_add_residual_bwd_in_vec[1] = {layer1_1_relu3_grad_input};

    // create layer1_1_add_residual_grad_input0 tensor
    const unsigned layer1_1_add_residual_grad_input0_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer1_1_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer1_1_add_residual_grad_input0");

    // create layer1_1_add_residual_grad_input1 tensor
    const unsigned layer1_1_add_residual_grad_input1_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer1_1_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer1_1_add_residual_grad_input1");

    synTensor layer1_1_add_residual_bwd_out_vec[2] = {layer1_1_add_residual_grad_input0,
                                                      layer1_1_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer1_1_add_residual_bwd_in_vec,
                           layer1_1_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer1_1_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_add_residual_bwd failed!");

    /*************
     * layer1_1_bn3_bwd node
     * inputs: [layer1_1_conv3_output(64, 56, 56, 256)(dtype=bf16), layer1_1_add_residual_grad_input0(64, 56, 56,
     *256)(dtype=bf16), layer1_1_bn3_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_1_bn3_saved_var(1, 1, 1,
     *256)(dtype=float32), layer1_1_bn3_weight[256](dtype=float32)] output: [layer1_1_bn3_grad_input(64, 56, 56,
     *256)(dtype=bf16), layer1_1_bn3_bias_grad(256,)(dtype=float32), layer1_1_bn3_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer1_1_bn3_bwd_in_vec[5] = {layer1_1_conv3_output,
                                            layer1_1_add_residual_grad_input0,
                                            layer1_1_bn3_saved_mean,
                                            layer1_1_bn3_saved_var,
                                            layer1_1_bn3_weight};

    // create layer1_1_bn3_grad_input tensor
    const unsigned layer1_1_bn3_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_bn3_grad_input_sizes, false, "layer1_1_bn3_grad_input");

    // create layer1_1_bn3_bias_grad tensor
    const unsigned layer1_1_bn3_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_1_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_bn3_bias_grad_tr_info = {"layer1_1_bn3_bias_grad", layer1_1_bn3_bias_grad_dram};
    synTensor           layer1_1_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer1_1_bn3_bias_grad_sizes, true, "layer1_1_bn3_bias_grad");

    // create layer1_1_bn3_weight_grad tensor
    const unsigned layer1_1_bn3_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_1_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_bn3_weight_grad_tr_info = {"layer1_1_bn3_weight_grad", layer1_1_bn3_weight_grad_dram};
    synTensor           layer1_1_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer1_1_bn3_weight_grad_sizes, true, "layer1_1_bn3_weight_grad");

    synTensor layer1_1_bn3_bwd_out_vec[3] = {layer1_1_bn3_grad_input, layer1_1_bn3_bias_grad, layer1_1_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_1_bn3_bwd_in_vec,
                           layer1_1_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_1_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_bn3_bwd failed!");

    /*************
     * layer1_1_conv3_dedx node
     * inputs: [layer1_1_bn3_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_1_conv3_weight[1, 1, 64, 256](dtype=bf16)]
     * output: [layer1_1_conv3_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_1_conv3_dedx_kernel_params;
    layer1_1_conv3_dedx_kernel_params.dH   = 1;
    layer1_1_conv3_dedx_kernel_params.dW   = 1;
    layer1_1_conv3_dedx_kernel_params.kH   = 1;
    layer1_1_conv3_dedx_kernel_params.kW   = 1;
    layer1_1_conv3_dedx_kernel_params.padT = 0;
    layer1_1_conv3_dedx_kernel_params.padB = 0;
    layer1_1_conv3_dedx_kernel_params.padL = 0;
    layer1_1_conv3_dedx_kernel_params.padR = 0;
    layer1_1_conv3_dedx_kernel_params.dilH = 1;
    layer1_1_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer1_1_conv3_dedx_in_vec[2] = {layer1_1_bn3_grad_input, layer1_1_conv3_weight};

    // create layer1_1_conv3_grad_input tensor
    const unsigned layer1_1_conv3_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_conv3_grad_input_sizes, false, "layer1_1_conv3_grad_input");

    synTensor layer1_1_conv3_dedx_out_vec[1] = {layer1_1_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv3_dedx_in_vec,
                           layer1_1_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_1_conv3_dedx_kernel_params,
                           sizeof(layer1_1_conv3_dedx_kernel_params),
                           "dedx",
                           "layer1_1_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv3_dedx failed!");

    /*************
     * layer1_1_conv3_dedw node
     * inputs: [layer1_1_bn3_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_1_relu2_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_1_conv3_weight_grad(1, 1, 64, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer1_1_conv3_dedw_kernel_params;
    layer1_1_conv3_dedw_kernel_params.dH   = 1;
    layer1_1_conv3_dedw_kernel_params.dW   = 1;
    layer1_1_conv3_dedw_kernel_params.kH   = 1;
    layer1_1_conv3_dedw_kernel_params.kW   = 1;
    layer1_1_conv3_dedw_kernel_params.padT = 0;
    layer1_1_conv3_dedw_kernel_params.padB = 0;
    layer1_1_conv3_dedw_kernel_params.padL = 0;
    layer1_1_conv3_dedw_kernel_params.padR = 0;
    layer1_1_conv3_dedw_kernel_params.dilH = 1;
    layer1_1_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer1_1_conv3_dedw_in_vec[2] = {layer1_1_bn3_grad_input, layer1_1_relu2_output};

    // create layer1_1_conv3_weight_grad tensor
    const unsigned      layer1_1_conv3_weight_grad_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_1_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_conv3_weight_grad_tr_info = {"layer1_1_conv3_weight_grad",
                                                              layer1_1_conv3_weight_grad_dram};
    synTensor           layer1_1_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer1_1_conv3_weight_grad_sizes, true, "layer1_1_conv3_weight_grad");

    synTensor layer1_1_conv3_dedw_out_vec[1] = {layer1_1_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv3_dedw_in_vec,
                           layer1_1_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_1_conv3_dedw_kernel_params,
                           sizeof(layer1_1_conv3_dedw_kernel_params),
                           "dedw",
                           "layer1_1_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv3_dedw failed!");

    /*************
     * layer1_1_relu2_bwd node
     * inputs: [layer1_1_conv3_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_1_relu2_output(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_1_relu2_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_1_relu2_bwd_in_vec[2] = {layer1_1_conv3_grad_input, layer1_1_relu2_output};

    // create layer1_1_relu2_grad_input tensor
    const unsigned layer1_1_relu2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_relu2_grad_input_sizes, false, "layer1_1_relu2_grad_input");

    synTensor layer1_1_relu2_bwd_out_vec[1] = {layer1_1_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_1_relu2_bwd_in_vec,
                           layer1_1_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_1_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_relu2_bwd failed!");

    /*************
     * layer1_1_bn2_bwd node
     * inputs: [layer1_1_conv2_output(64, 56, 56, 64)(dtype=bf16), layer1_1_relu2_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_1_bn2_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_1_bn2_saved_var(1, 1, 1,
     *64)(dtype=float32), layer1_1_bn2_weight[64](dtype=float32)] output: [layer1_1_bn2_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_1_bn2_bias_grad(64,)(dtype=float32), layer1_1_bn2_weight_grad(64,)(dtype=float32)]
     *************/

    synTensor layer1_1_bn2_bwd_in_vec[5] = {layer1_1_conv2_output,
                                            layer1_1_relu2_grad_input,
                                            layer1_1_bn2_saved_mean,
                                            layer1_1_bn2_saved_var,
                                            layer1_1_bn2_weight};

    // create layer1_1_bn2_grad_input tensor
    const unsigned layer1_1_bn2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_bn2_grad_input_sizes, false, "layer1_1_bn2_grad_input");

    // create layer1_1_bn2_bias_grad tensor
    const unsigned layer1_1_bn2_bias_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_bn2_bias_grad_tr_info = {"layer1_1_bn2_bias_grad", layer1_1_bn2_bias_grad_dram};
    synTensor           layer1_1_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer1_1_bn2_bias_grad_sizes, true, "layer1_1_bn2_bias_grad");

    // create layer1_1_bn2_weight_grad tensor
    const unsigned layer1_1_bn2_weight_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_bn2_weight_grad_tr_info = {"layer1_1_bn2_weight_grad", layer1_1_bn2_weight_grad_dram};
    synTensor           layer1_1_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer1_1_bn2_weight_grad_sizes, true, "layer1_1_bn2_weight_grad");

    synTensor layer1_1_bn2_bwd_out_vec[3] = {layer1_1_bn2_grad_input, layer1_1_bn2_bias_grad, layer1_1_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_1_bn2_bwd_in_vec,
                           layer1_1_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_1_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_bn2_bwd failed!");

    /*************
     * layer1_1_conv2_dedx node
     * inputs: [layer1_1_bn2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_1_conv2_weight[3, 3, 64, 64](dtype=bf16)]
     * output: [layer1_1_conv2_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_1_conv2_dedx_kernel_params;
    layer1_1_conv2_dedx_kernel_params.dH   = 1;
    layer1_1_conv2_dedx_kernel_params.dW   = 1;
    layer1_1_conv2_dedx_kernel_params.kH   = 3;
    layer1_1_conv2_dedx_kernel_params.kW   = 3;
    layer1_1_conv2_dedx_kernel_params.padT = 1;
    layer1_1_conv2_dedx_kernel_params.padB = 1;
    layer1_1_conv2_dedx_kernel_params.padL = 1;
    layer1_1_conv2_dedx_kernel_params.padR = 1;
    layer1_1_conv2_dedx_kernel_params.dilH = 1;
    layer1_1_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer1_1_conv2_dedx_in_vec[2] = {layer1_1_bn2_grad_input, layer1_1_conv2_weight};

    // create layer1_1_conv2_grad_input tensor
    const unsigned layer1_1_conv2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_conv2_grad_input_sizes, false, "layer1_1_conv2_grad_input");

    synTensor layer1_1_conv2_dedx_out_vec[1] = {layer1_1_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv2_dedx_in_vec,
                           layer1_1_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_1_conv2_dedx_kernel_params,
                           sizeof(layer1_1_conv2_dedx_kernel_params),
                           "dedx",
                           "layer1_1_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv2_dedx failed!");

    /*************
     * layer1_1_conv2_dedw node
     * inputs: [layer1_1_bn2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_1_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_1_conv2_weight_grad(3, 3, 64, 64)(dtype=float32)]
     *************/
    synConvolutionParams layer1_1_conv2_dedw_kernel_params;
    layer1_1_conv2_dedw_kernel_params.dH   = 1;
    layer1_1_conv2_dedw_kernel_params.dW   = 1;
    layer1_1_conv2_dedw_kernel_params.kH   = 3;
    layer1_1_conv2_dedw_kernel_params.kW   = 3;
    layer1_1_conv2_dedw_kernel_params.padT = 1;
    layer1_1_conv2_dedw_kernel_params.padB = 1;
    layer1_1_conv2_dedw_kernel_params.padL = 1;
    layer1_1_conv2_dedw_kernel_params.padR = 1;
    layer1_1_conv2_dedw_kernel_params.dilH = 1;
    layer1_1_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer1_1_conv2_dedw_in_vec[2] = {layer1_1_bn2_grad_input, layer1_1_relu1_output};

    // create layer1_1_conv2_weight_grad tensor
    const unsigned      layer1_1_conv2_weight_grad_sizes[] = {3, 3, 64, 64};
    uint64_t            layer1_1_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_conv2_weight_grad_tr_info = {"layer1_1_conv2_weight_grad",
                                                              layer1_1_conv2_weight_grad_dram};
    synTensor           layer1_1_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer1_1_conv2_weight_grad_sizes, true, "layer1_1_conv2_weight_grad");

    synTensor layer1_1_conv2_dedw_out_vec[1] = {layer1_1_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv2_dedw_in_vec,
                           layer1_1_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_1_conv2_dedw_kernel_params,
                           sizeof(layer1_1_conv2_dedw_kernel_params),
                           "dedw",
                           "layer1_1_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv2_dedw failed!");

    /*************
     * layer1_1_relu1_bwd node
     * inputs: [layer1_1_conv2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_1_relu1_output(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_1_relu1_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_1_relu1_bwd_in_vec[2] = {layer1_1_conv2_grad_input, layer1_1_relu1_output};

    // create layer1_1_relu1_grad_input tensor
    const unsigned layer1_1_relu1_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_relu1_grad_input_sizes, false, "layer1_1_relu1_grad_input");

    synTensor layer1_1_relu1_bwd_out_vec[1] = {layer1_1_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_1_relu1_bwd_in_vec,
                           layer1_1_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_1_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_relu1_bwd failed!");

    /*************
     * layer1_1_bn1_bwd node
     * inputs: [layer1_1_conv1_output(64, 56, 56, 64)(dtype=bf16), layer1_1_relu1_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_1_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_1_bn1_saved_var(1, 1, 1,
     *64)(dtype=float32), layer1_1_bn1_weight[64](dtype=float32)] output: [layer1_1_bn1_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_1_bn1_bias_grad(64,)(dtype=float32), layer1_1_bn1_weight_grad(64,)(dtype=float32)]
     *************/

    synTensor layer1_1_bn1_bwd_in_vec[5] = {layer1_1_conv1_output,
                                            layer1_1_relu1_grad_input,
                                            layer1_1_bn1_saved_mean,
                                            layer1_1_bn1_saved_var,
                                            layer1_1_bn1_weight};

    // create layer1_1_bn1_grad_input tensor
    const unsigned layer1_1_bn1_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_1_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_bn1_grad_input_sizes, false, "layer1_1_bn1_grad_input");

    // create layer1_1_bn1_bias_grad tensor
    const unsigned layer1_1_bn1_bias_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_bn1_bias_grad_tr_info = {"layer1_1_bn1_bias_grad", layer1_1_bn1_bias_grad_dram};
    synTensor           layer1_1_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer1_1_bn1_bias_grad_sizes, true, "layer1_1_bn1_bias_grad");

    // create layer1_1_bn1_weight_grad tensor
    const unsigned layer1_1_bn1_weight_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_1_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_bn1_weight_grad_tr_info = {"layer1_1_bn1_weight_grad", layer1_1_bn1_weight_grad_dram};
    synTensor           layer1_1_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer1_1_bn1_weight_grad_sizes, true, "layer1_1_bn1_weight_grad");

    synTensor layer1_1_bn1_bwd_out_vec[3] = {layer1_1_bn1_grad_input, layer1_1_bn1_bias_grad, layer1_1_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_1_bn1_bwd_in_vec,
                           layer1_1_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_1_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_bn1_bwd failed!");

    /*************
     * layer1_1_conv1_dedx node
     * inputs: [layer1_1_bn1_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_1_conv1_weight[1, 1, 256, 64](dtype=bf16)]
     * output: [layer1_1_conv1_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_1_conv1_dedx_kernel_params;
    layer1_1_conv1_dedx_kernel_params.dH   = 1;
    layer1_1_conv1_dedx_kernel_params.dW   = 1;
    layer1_1_conv1_dedx_kernel_params.kH   = 1;
    layer1_1_conv1_dedx_kernel_params.kW   = 1;
    layer1_1_conv1_dedx_kernel_params.padT = 0;
    layer1_1_conv1_dedx_kernel_params.padB = 0;
    layer1_1_conv1_dedx_kernel_params.padL = 0;
    layer1_1_conv1_dedx_kernel_params.padR = 0;
    layer1_1_conv1_dedx_kernel_params.dilH = 1;
    layer1_1_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer1_1_conv1_dedx_in_vec[2] = {layer1_1_bn1_grad_input, layer1_1_conv1_weight};

    // create layer1_1_conv1_grad_input tensor
    const unsigned layer1_1_conv1_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_1_conv1_grad_input_sizes, false, "layer1_1_conv1_grad_input");

    synTensor layer1_1_conv1_dedx_out_vec[1] = {layer1_1_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv1_dedx_in_vec,
                           layer1_1_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_1_conv1_dedx_kernel_params,
                           sizeof(layer1_1_conv1_dedx_kernel_params),
                           "dedx",
                           "layer1_1_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv1_dedx failed!");

    /*************
     * layer1_1_conv1_dedw node
     * inputs: [layer1_1_bn1_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_0_relu3_output(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_1_conv1_weight_grad(1, 1, 256, 64)(dtype=float32)]
     *************/
    synConvolutionParams layer1_1_conv1_dedw_kernel_params;
    layer1_1_conv1_dedw_kernel_params.dH   = 1;
    layer1_1_conv1_dedw_kernel_params.dW   = 1;
    layer1_1_conv1_dedw_kernel_params.kH   = 1;
    layer1_1_conv1_dedw_kernel_params.kW   = 1;
    layer1_1_conv1_dedw_kernel_params.padT = 0;
    layer1_1_conv1_dedw_kernel_params.padB = 0;
    layer1_1_conv1_dedw_kernel_params.padL = 0;
    layer1_1_conv1_dedw_kernel_params.padR = 0;
    layer1_1_conv1_dedw_kernel_params.dilH = 1;
    layer1_1_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer1_1_conv1_dedw_in_vec[2] = {layer1_1_bn1_grad_input, layer1_0_relu3_output};

    // create layer1_1_conv1_weight_grad tensor
    const unsigned      layer1_1_conv1_weight_grad_sizes[] = {1, 1, 256, 64};
    uint64_t            layer1_1_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_1_conv1_weight_grad_tr_info = {"layer1_1_conv1_weight_grad",
                                                              layer1_1_conv1_weight_grad_dram};
    synTensor           layer1_1_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer1_1_conv1_weight_grad_sizes, true, "layer1_1_conv1_weight_grad");

    synTensor layer1_1_conv1_dedw_out_vec[1] = {layer1_1_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_1_conv1_dedw_in_vec,
                           layer1_1_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_1_conv1_dedw_kernel_params,
                           sizeof(layer1_1_conv1_dedw_kernel_params),
                           "dedw",
                           "layer1_1_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_conv1_dedw failed!");

    /*************
     * layer1_1_add_residual_fwd1 node
     * inputs: [layer1_1_conv1_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_1_add_residual_grad_input1(64, 56, 56,
     *256)(dtype=bf16)] output: [layer1_1_residual_upstream_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_1_add_residual_fwd1_in_vec[2] = {layer1_1_conv1_grad_input, layer1_1_add_residual_grad_input1};

    // create layer1_1_residual_upstream_grad_input tensor
    const unsigned layer1_1_residual_upstream_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_1_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer1_1_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer1_1_residual_upstream_grad_input");

    synTensor layer1_1_add_residual_fwd1_out_vec[1] = {layer1_1_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_1_add_residual_fwd1_in_vec,
                           layer1_1_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer1_1_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_1_add_residual_fwd1 failed!");

    /*************
     * layer1_0_relu3_bwd node
     * inputs: [layer1_1_residual_upstream_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_0_relu3_output(64, 56, 56,
     *256)(dtype=bf16)] output: [layer1_0_relu3_grad_input(64, 56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_0_relu3_bwd_in_vec[2] = {layer1_1_residual_upstream_grad_input, layer1_0_relu3_output};

    // create layer1_0_relu3_grad_input tensor
    const unsigned layer1_0_relu3_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_relu3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_relu3_grad_input_sizes, false, "layer1_0_relu3_grad_input");

    synTensor layer1_0_relu3_bwd_out_vec[1] = {layer1_0_relu3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_0_relu3_bwd_in_vec,
                           layer1_0_relu3_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_0_relu3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu3_bwd failed!");

    /*************
     * layer1_0_add_residual_bwd node
     * inputs: [layer1_0_relu3_grad_input(64, 56, 56, 256)(dtype=bf16)]
     * output: [layer1_0_add_residual_grad_input0(64, 56, 56, 256)(dtype=bf16), layer1_0_add_residual_grad_input1(64,
     *56, 56, 256)(dtype=bf16)]
     *************/

    synTensor layer1_0_add_residual_bwd_in_vec[1] = {layer1_0_relu3_grad_input};

    // create layer1_0_add_residual_grad_input0 tensor
    const unsigned layer1_0_add_residual_grad_input0_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_add_residual_grad_input0         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer1_0_add_residual_grad_input0_sizes,
                                                               false,
                                                               "layer1_0_add_residual_grad_input0");

    // create layer1_0_add_residual_grad_input1 tensor
    const unsigned layer1_0_add_residual_grad_input1_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_add_residual_grad_input1         = createTensor(4U,
                                                               syn_type_bf16,
                                                               layer1_0_add_residual_grad_input1_sizes,
                                                               false,
                                                               "layer1_0_add_residual_grad_input1");

    synTensor layer1_0_add_residual_bwd_out_vec[2] = {layer1_0_add_residual_grad_input0,
                                                      layer1_0_add_residual_grad_input1};

    status = synNodeCreate(graphHandle,
                           layer1_0_add_residual_bwd_in_vec,
                           layer1_0_add_residual_bwd_out_vec,
                           1,
                           2,
                           nullptr,
                           0,
                           "add_bwd_bf16",
                           "layer1_0_add_residual_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_add_residual_bwd failed!");

    /*************
     * layer1_0_bn3_bwd node
     * inputs: [layer1_0_conv3_output(64, 56, 56, 256)(dtype=bf16), layer1_0_add_residual_grad_input0(64, 56, 56,
     *256)(dtype=bf16), layer1_0_bn3_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_0_bn3_saved_var(1, 1, 1,
     *256)(dtype=float32), layer1_0_bn3_weight[256](dtype=float32)] output: [layer1_0_bn3_grad_input(64, 56, 56,
     *256)(dtype=bf16), layer1_0_bn3_bias_grad(256,)(dtype=float32), layer1_0_bn3_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer1_0_bn3_bwd_in_vec[5] = {layer1_0_conv3_output,
                                            layer1_0_add_residual_grad_input0,
                                            layer1_0_bn3_saved_mean,
                                            layer1_0_bn3_saved_var,
                                            layer1_0_bn3_weight};

    // create layer1_0_bn3_grad_input tensor
    const unsigned layer1_0_bn3_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_0_bn3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_bn3_grad_input_sizes, false, "layer1_0_bn3_grad_input");

    // create layer1_0_bn3_bias_grad tensor
    const unsigned layer1_0_bn3_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_0_bn3_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_bn3_bias_grad_tr_info = {"layer1_0_bn3_bias_grad", layer1_0_bn3_bias_grad_dram};
    synTensor           layer1_0_bn3_bias_grad =
        createTensor(1U, syn_type_single, layer1_0_bn3_bias_grad_sizes, true, "layer1_0_bn3_bias_grad");

    // create layer1_0_bn3_weight_grad tensor
    const unsigned layer1_0_bn3_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_0_bn3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_bn3_weight_grad_tr_info = {"layer1_0_bn3_weight_grad", layer1_0_bn3_weight_grad_dram};
    synTensor           layer1_0_bn3_weight_grad =
        createTensor(1U, syn_type_single, layer1_0_bn3_weight_grad_sizes, true, "layer1_0_bn3_weight_grad");

    synTensor layer1_0_bn3_bwd_out_vec[3] = {layer1_0_bn3_grad_input, layer1_0_bn3_bias_grad, layer1_0_bn3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_0_bn3_bwd_in_vec,
                           layer1_0_bn3_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_0_bn3_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn3_bwd failed!");

    /*************
     * layer1_0_conv3_dedx node
     * inputs: [layer1_0_bn3_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_0_conv3_weight[1, 1, 64, 256](dtype=bf16)]
     * output: [layer1_0_conv3_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_0_conv3_dedx_kernel_params;
    layer1_0_conv3_dedx_kernel_params.dH   = 1;
    layer1_0_conv3_dedx_kernel_params.dW   = 1;
    layer1_0_conv3_dedx_kernel_params.kH   = 1;
    layer1_0_conv3_dedx_kernel_params.kW   = 1;
    layer1_0_conv3_dedx_kernel_params.padT = 0;
    layer1_0_conv3_dedx_kernel_params.padB = 0;
    layer1_0_conv3_dedx_kernel_params.padL = 0;
    layer1_0_conv3_dedx_kernel_params.padR = 0;
    layer1_0_conv3_dedx_kernel_params.dilH = 1;
    layer1_0_conv3_dedx_kernel_params.dilW = 1;

    synTensor layer1_0_conv3_dedx_in_vec[2] = {layer1_0_bn3_grad_input, layer1_0_conv3_weight};

    // create layer1_0_conv3_grad_input tensor
    const unsigned layer1_0_conv3_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_conv3_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_conv3_grad_input_sizes, false, "layer1_0_conv3_grad_input");

    synTensor layer1_0_conv3_dedx_out_vec[1] = {layer1_0_conv3_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv3_dedx_in_vec,
                           layer1_0_conv3_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_0_conv3_dedx_kernel_params,
                           sizeof(layer1_0_conv3_dedx_kernel_params),
                           "dedx",
                           "layer1_0_conv3_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv3_dedx failed!");

    /*************
     * layer1_0_conv3_dedw node
     * inputs: [layer1_0_bn3_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_0_relu2_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_0_conv3_weight_grad(1, 1, 64, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer1_0_conv3_dedw_kernel_params;
    layer1_0_conv3_dedw_kernel_params.dH   = 1;
    layer1_0_conv3_dedw_kernel_params.dW   = 1;
    layer1_0_conv3_dedw_kernel_params.kH   = 1;
    layer1_0_conv3_dedw_kernel_params.kW   = 1;
    layer1_0_conv3_dedw_kernel_params.padT = 0;
    layer1_0_conv3_dedw_kernel_params.padB = 0;
    layer1_0_conv3_dedw_kernel_params.padL = 0;
    layer1_0_conv3_dedw_kernel_params.padR = 0;
    layer1_0_conv3_dedw_kernel_params.dilH = 1;
    layer1_0_conv3_dedw_kernel_params.dilW = 1;

    synTensor layer1_0_conv3_dedw_in_vec[2] = {layer1_0_bn3_grad_input, layer1_0_relu2_output};

    // create layer1_0_conv3_weight_grad tensor
    const unsigned      layer1_0_conv3_weight_grad_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_0_conv3_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_conv3_weight_grad_tr_info = {"layer1_0_conv3_weight_grad",
                                                              layer1_0_conv3_weight_grad_dram};
    synTensor           layer1_0_conv3_weight_grad =
        createTensor(4U, syn_type_single, layer1_0_conv3_weight_grad_sizes, true, "layer1_0_conv3_weight_grad");

    synTensor layer1_0_conv3_dedw_out_vec[1] = {layer1_0_conv3_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv3_dedw_in_vec,
                           layer1_0_conv3_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_0_conv3_dedw_kernel_params,
                           sizeof(layer1_0_conv3_dedw_kernel_params),
                           "dedw",
                           "layer1_0_conv3_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv3_dedw failed!");

    /*************
     * layer1_0_relu2_bwd node
     * inputs: [layer1_0_conv3_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_0_relu2_output(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_0_relu2_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_0_relu2_bwd_in_vec[2] = {layer1_0_conv3_grad_input, layer1_0_relu2_output};

    // create layer1_0_relu2_grad_input tensor
    const unsigned layer1_0_relu2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_relu2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_relu2_grad_input_sizes, false, "layer1_0_relu2_grad_input");

    synTensor layer1_0_relu2_bwd_out_vec[1] = {layer1_0_relu2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_0_relu2_bwd_in_vec,
                           layer1_0_relu2_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_0_relu2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu2_bwd failed!");

    /*************
     * layer1_0_bn2_bwd node
     * inputs: [layer1_0_conv2_output(64, 56, 56, 64)(dtype=bf16), layer1_0_relu2_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_0_bn2_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn2_saved_var(1, 1, 1,
     *64)(dtype=float32), layer1_0_bn2_weight[64](dtype=float32)] output: [layer1_0_bn2_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_0_bn2_bias_grad(64,)(dtype=float32), layer1_0_bn2_weight_grad(64,)(dtype=float32)]
     *************/

    synTensor layer1_0_bn2_bwd_in_vec[5] = {layer1_0_conv2_output,
                                            layer1_0_relu2_grad_input,
                                            layer1_0_bn2_saved_mean,
                                            layer1_0_bn2_saved_var,
                                            layer1_0_bn2_weight};

    // create layer1_0_bn2_grad_input tensor
    const unsigned layer1_0_bn2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_bn2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_bn2_grad_input_sizes, false, "layer1_0_bn2_grad_input");

    // create layer1_0_bn2_bias_grad tensor
    const unsigned layer1_0_bn2_bias_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn2_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_bn2_bias_grad_tr_info = {"layer1_0_bn2_bias_grad", layer1_0_bn2_bias_grad_dram};
    synTensor           layer1_0_bn2_bias_grad =
        createTensor(1U, syn_type_single, layer1_0_bn2_bias_grad_sizes, true, "layer1_0_bn2_bias_grad");

    // create layer1_0_bn2_weight_grad tensor
    const unsigned layer1_0_bn2_weight_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_bn2_weight_grad_tr_info = {"layer1_0_bn2_weight_grad", layer1_0_bn2_weight_grad_dram};
    synTensor           layer1_0_bn2_weight_grad =
        createTensor(1U, syn_type_single, layer1_0_bn2_weight_grad_sizes, true, "layer1_0_bn2_weight_grad");

    synTensor layer1_0_bn2_bwd_out_vec[3] = {layer1_0_bn2_grad_input, layer1_0_bn2_bias_grad, layer1_0_bn2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_0_bn2_bwd_in_vec,
                           layer1_0_bn2_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_0_bn2_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn2_bwd failed!");

    /*************
     * layer1_0_conv2_dedx node
     * inputs: [layer1_0_bn2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_0_conv2_weight[3, 3, 64, 64](dtype=bf16)]
     * output: [layer1_0_conv2_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_0_conv2_dedx_kernel_params;
    layer1_0_conv2_dedx_kernel_params.dH   = 1;
    layer1_0_conv2_dedx_kernel_params.dW   = 1;
    layer1_0_conv2_dedx_kernel_params.kH   = 3;
    layer1_0_conv2_dedx_kernel_params.kW   = 3;
    layer1_0_conv2_dedx_kernel_params.padT = 1;
    layer1_0_conv2_dedx_kernel_params.padB = 1;
    layer1_0_conv2_dedx_kernel_params.padL = 1;
    layer1_0_conv2_dedx_kernel_params.padR = 1;
    layer1_0_conv2_dedx_kernel_params.dilH = 1;
    layer1_0_conv2_dedx_kernel_params.dilW = 1;

    synTensor layer1_0_conv2_dedx_in_vec[2] = {layer1_0_bn2_grad_input, layer1_0_conv2_weight};

    // create layer1_0_conv2_grad_input tensor
    const unsigned layer1_0_conv2_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_conv2_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_conv2_grad_input_sizes, false, "layer1_0_conv2_grad_input");

    synTensor layer1_0_conv2_dedx_out_vec[1] = {layer1_0_conv2_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv2_dedx_in_vec,
                           layer1_0_conv2_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_0_conv2_dedx_kernel_params,
                           sizeof(layer1_0_conv2_dedx_kernel_params),
                           "dedx",
                           "layer1_0_conv2_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv2_dedx failed!");

    /*************
     * layer1_0_conv2_dedw node
     * inputs: [layer1_0_bn2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_0_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_0_conv2_weight_grad(3, 3, 64, 64)(dtype=float32)]
     *************/
    synConvolutionParams layer1_0_conv2_dedw_kernel_params;
    layer1_0_conv2_dedw_kernel_params.dH   = 1;
    layer1_0_conv2_dedw_kernel_params.dW   = 1;
    layer1_0_conv2_dedw_kernel_params.kH   = 3;
    layer1_0_conv2_dedw_kernel_params.kW   = 3;
    layer1_0_conv2_dedw_kernel_params.padT = 1;
    layer1_0_conv2_dedw_kernel_params.padB = 1;
    layer1_0_conv2_dedw_kernel_params.padL = 1;
    layer1_0_conv2_dedw_kernel_params.padR = 1;
    layer1_0_conv2_dedw_kernel_params.dilH = 1;
    layer1_0_conv2_dedw_kernel_params.dilW = 1;

    synTensor layer1_0_conv2_dedw_in_vec[2] = {layer1_0_bn2_grad_input, layer1_0_relu1_output};

    // create layer1_0_conv2_weight_grad tensor
    const unsigned      layer1_0_conv2_weight_grad_sizes[] = {3, 3, 64, 64};
    uint64_t            layer1_0_conv2_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_conv2_weight_grad_tr_info = {"layer1_0_conv2_weight_grad",
                                                              layer1_0_conv2_weight_grad_dram};
    synTensor           layer1_0_conv2_weight_grad =
        createTensor(4U, syn_type_single, layer1_0_conv2_weight_grad_sizes, true, "layer1_0_conv2_weight_grad");

    synTensor layer1_0_conv2_dedw_out_vec[1] = {layer1_0_conv2_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv2_dedw_in_vec,
                           layer1_0_conv2_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_0_conv2_dedw_kernel_params,
                           sizeof(layer1_0_conv2_dedw_kernel_params),
                           "dedw",
                           "layer1_0_conv2_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv2_dedw failed!");

    /*************
     * layer1_0_relu1_bwd node
     * inputs: [layer1_0_conv2_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_0_relu1_output(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_0_relu1_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_0_relu1_bwd_in_vec[2] = {layer1_0_conv2_grad_input, layer1_0_relu1_output};

    // create layer1_0_relu1_grad_input tensor
    const unsigned layer1_0_relu1_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_relu1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_relu1_grad_input_sizes, false, "layer1_0_relu1_grad_input");

    synTensor layer1_0_relu1_bwd_out_vec[1] = {layer1_0_relu1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_0_relu1_bwd_in_vec,
                           layer1_0_relu1_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "layer1_0_relu1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_relu1_bwd failed!");

    /*************
     * layer1_0_bn1_bwd node
     * inputs: [layer1_0_conv1_output(64, 56, 56, 64)(dtype=bf16), layer1_0_relu1_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn1_saved_var(1, 1, 1,
     *64)(dtype=float32), layer1_0_bn1_weight[64](dtype=float32)] output: [layer1_0_bn1_grad_input(64, 56, 56,
     *64)(dtype=bf16), layer1_0_bn1_bias_grad(64,)(dtype=float32), layer1_0_bn1_weight_grad(64,)(dtype=float32)]
     *************/

    synTensor layer1_0_bn1_bwd_in_vec[5] = {layer1_0_conv1_output,
                                            layer1_0_relu1_grad_input,
                                            layer1_0_bn1_saved_mean,
                                            layer1_0_bn1_saved_var,
                                            layer1_0_bn1_weight};

    // create layer1_0_bn1_grad_input tensor
    const unsigned layer1_0_bn1_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_bn1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_bn1_grad_input_sizes, false, "layer1_0_bn1_grad_input");

    // create layer1_0_bn1_bias_grad tensor
    const unsigned layer1_0_bn1_bias_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_bn1_bias_grad_tr_info = {"layer1_0_bn1_bias_grad", layer1_0_bn1_bias_grad_dram};
    synTensor           layer1_0_bn1_bias_grad =
        createTensor(1U, syn_type_single, layer1_0_bn1_bias_grad_sizes, true, "layer1_0_bn1_bias_grad");

    // create layer1_0_bn1_weight_grad tensor
    const unsigned layer1_0_bn1_weight_grad_sizes[] = {
        64,
    };
    uint64_t            layer1_0_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_bn1_weight_grad_tr_info = {"layer1_0_bn1_weight_grad", layer1_0_bn1_weight_grad_dram};
    synTensor           layer1_0_bn1_weight_grad =
        createTensor(1U, syn_type_single, layer1_0_bn1_weight_grad_sizes, true, "layer1_0_bn1_weight_grad");

    synTensor layer1_0_bn1_bwd_out_vec[3] = {layer1_0_bn1_grad_input, layer1_0_bn1_bias_grad, layer1_0_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_0_bn1_bwd_in_vec,
                           layer1_0_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_0_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_bn1_bwd failed!");

    /*************
     * layer1_0_conv1_dedx node
     * inputs: [layer1_0_bn1_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_0_conv1_weight[1, 1, 64, 64](dtype=bf16)]
     * output: [layer1_0_conv1_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_0_conv1_dedx_kernel_params;
    layer1_0_conv1_dedx_kernel_params.dH   = 1;
    layer1_0_conv1_dedx_kernel_params.dW   = 1;
    layer1_0_conv1_dedx_kernel_params.kH   = 1;
    layer1_0_conv1_dedx_kernel_params.kW   = 1;
    layer1_0_conv1_dedx_kernel_params.padT = 0;
    layer1_0_conv1_dedx_kernel_params.padB = 0;
    layer1_0_conv1_dedx_kernel_params.padL = 0;
    layer1_0_conv1_dedx_kernel_params.padR = 0;
    layer1_0_conv1_dedx_kernel_params.dilH = 1;
    layer1_0_conv1_dedx_kernel_params.dilW = 1;

    synTensor layer1_0_conv1_dedx_in_vec[2] = {layer1_0_bn1_grad_input, layer1_0_conv1_weight};

    // create layer1_0_conv1_grad_input tensor
    const unsigned layer1_0_conv1_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_conv1_grad_input =
        createTensor(4U, syn_type_bf16, layer1_0_conv1_grad_input_sizes, false, "layer1_0_conv1_grad_input");

    synTensor layer1_0_conv1_dedx_out_vec[1] = {layer1_0_conv1_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv1_dedx_in_vec,
                           layer1_0_conv1_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_0_conv1_dedx_kernel_params,
                           sizeof(layer1_0_conv1_dedx_kernel_params),
                           "dedx",
                           "layer1_0_conv1_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv1_dedx failed!");

    /*************
     * layer1_0_conv1_dedw node
     * inputs: [layer1_0_bn1_grad_input(64, 56, 56, 64)(dtype=bf16), worker_0_maxpool_output(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_0_conv1_weight_grad(1, 1, 64, 64)(dtype=float32)]
     *************/
    synConvolutionParams layer1_0_conv1_dedw_kernel_params;
    layer1_0_conv1_dedw_kernel_params.dH   = 1;
    layer1_0_conv1_dedw_kernel_params.dW   = 1;
    layer1_0_conv1_dedw_kernel_params.kH   = 1;
    layer1_0_conv1_dedw_kernel_params.kW   = 1;
    layer1_0_conv1_dedw_kernel_params.padT = 0;
    layer1_0_conv1_dedw_kernel_params.padB = 0;
    layer1_0_conv1_dedw_kernel_params.padL = 0;
    layer1_0_conv1_dedw_kernel_params.padR = 0;
    layer1_0_conv1_dedw_kernel_params.dilH = 1;
    layer1_0_conv1_dedw_kernel_params.dilW = 1;

    synTensor layer1_0_conv1_dedw_in_vec[2] = {layer1_0_bn1_grad_input, worker_0_maxpool_output};

    // create layer1_0_conv1_weight_grad tensor
    const unsigned      layer1_0_conv1_weight_grad_sizes[] = {1, 1, 64, 64};
    uint64_t            layer1_0_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_0_conv1_weight_grad_tr_info = {"layer1_0_conv1_weight_grad",
                                                              layer1_0_conv1_weight_grad_dram};
    synTensor           layer1_0_conv1_weight_grad =
        createTensor(4U, syn_type_single, layer1_0_conv1_weight_grad_sizes, true, "layer1_0_conv1_weight_grad");

    synTensor layer1_0_conv1_dedw_out_vec[1] = {layer1_0_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_0_conv1_dedw_in_vec,
                           layer1_0_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_0_conv1_dedw_kernel_params,
                           sizeof(layer1_0_conv1_dedw_kernel_params),
                           "dedw",
                           "layer1_0_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_conv1_dedw failed!");

    /*************
     * layer1_bn_bwd node
     * inputs: [layer1_downsample_output(64, 56, 56, 256)(dtype=bf16), layer1_0_add_residual_grad_input1(64, 56, 56,
     *256)(dtype=bf16), layer1_bn_saved_mean(1, 1, 1, 256)(dtype=float32), layer1_bn_saved_var(1, 1, 1,
     *256)(dtype=float32), layer1_bn_weight[256](dtype=float32)] output: [layer1_bn_grad_input(64, 56, 56,
     *256)(dtype=bf16), layer1_bn_bias_grad(256,)(dtype=float32), layer1_bn_weight_grad(256,)(dtype=float32)]
     *************/

    synTensor layer1_bn_bwd_in_vec[5] = {layer1_downsample_output,
                                         layer1_0_add_residual_grad_input1,
                                         layer1_bn_saved_mean,
                                         layer1_bn_saved_var,
                                         layer1_bn_weight};

    // create layer1_bn_grad_input tensor
    const unsigned layer1_bn_grad_input_sizes[] = {64, 56, 56, 256};
    synTensor      layer1_bn_grad_input =
        createTensor(4U, syn_type_bf16, layer1_bn_grad_input_sizes, false, "layer1_bn_grad_input");

    // create layer1_bn_bias_grad tensor
    const unsigned layer1_bn_bias_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_bn_bias_grad_dram    = 0;
    synLaunchTensorInfo layer1_bn_bias_grad_tr_info = {"layer1_bn_bias_grad", layer1_bn_bias_grad_dram};
    synTensor           layer1_bn_bias_grad =
        createTensor(1U, syn_type_single, layer1_bn_bias_grad_sizes, true, "layer1_bn_bias_grad");

    // create layer1_bn_weight_grad tensor
    const unsigned layer1_bn_weight_grad_sizes[] = {
        256,
    };
    uint64_t            layer1_bn_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_bn_weight_grad_tr_info = {"layer1_bn_weight_grad", layer1_bn_weight_grad_dram};
    synTensor           layer1_bn_weight_grad =
        createTensor(1U, syn_type_single, layer1_bn_weight_grad_sizes, true, "layer1_bn_weight_grad");

    synTensor layer1_bn_bwd_out_vec[3] = {layer1_bn_grad_input, layer1_bn_bias_grad, layer1_bn_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_bn_bwd_in_vec,
                           layer1_bn_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "layer1_bn_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_bn_bwd failed!");

    /*************
     * layer1_downsample_dedx node
     * inputs: [layer1_bn_grad_input(64, 56, 56, 256)(dtype=bf16), layer1_downsample_weight[1, 1, 64, 256](dtype=bf16)]
     * output: [layer1_downsample_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/
    synConvolutionParams layer1_downsample_dedx_kernel_params;
    layer1_downsample_dedx_kernel_params.dH   = 1;
    layer1_downsample_dedx_kernel_params.dW   = 1;
    layer1_downsample_dedx_kernel_params.kH   = 1;
    layer1_downsample_dedx_kernel_params.kW   = 1;
    layer1_downsample_dedx_kernel_params.padT = 0;
    layer1_downsample_dedx_kernel_params.padB = 0;
    layer1_downsample_dedx_kernel_params.padL = 0;
    layer1_downsample_dedx_kernel_params.padR = 0;
    layer1_downsample_dedx_kernel_params.dilH = 1;
    layer1_downsample_dedx_kernel_params.dilW = 1;

    synTensor layer1_downsample_dedx_in_vec[2] = {layer1_bn_grad_input, layer1_downsample_weight};

    // create layer1_downsample_grad_input tensor
    const unsigned layer1_downsample_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_downsample_grad_input =
        createTensor(4U, syn_type_bf16, layer1_downsample_grad_input_sizes, false, "layer1_downsample_grad_input");

    synTensor layer1_downsample_dedx_out_vec[1] = {layer1_downsample_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_downsample_dedx_in_vec,
                           layer1_downsample_dedx_out_vec,
                           2,
                           1,
                           (void*)&layer1_downsample_dedx_kernel_params,
                           sizeof(layer1_downsample_dedx_kernel_params),
                           "dedx",
                           "layer1_downsample_dedx",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_downsample_dedx failed!");

    /*************
     * layer1_downsample_dedw node
     * inputs: [layer1_bn_grad_input(64, 56, 56, 256)(dtype=bf16), worker_0_maxpool_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_downsample_weight_grad(1, 1, 64, 256)(dtype=float32)]
     *************/
    synConvolutionParams layer1_downsample_dedw_kernel_params;
    layer1_downsample_dedw_kernel_params.dH   = 1;
    layer1_downsample_dedw_kernel_params.dW   = 1;
    layer1_downsample_dedw_kernel_params.kH   = 1;
    layer1_downsample_dedw_kernel_params.kW   = 1;
    layer1_downsample_dedw_kernel_params.padT = 0;
    layer1_downsample_dedw_kernel_params.padB = 0;
    layer1_downsample_dedw_kernel_params.padL = 0;
    layer1_downsample_dedw_kernel_params.padR = 0;
    layer1_downsample_dedw_kernel_params.dilH = 1;
    layer1_downsample_dedw_kernel_params.dilW = 1;

    synTensor layer1_downsample_dedw_in_vec[2] = {layer1_bn_grad_input, worker_0_maxpool_output};

    // create layer1_downsample_weight_grad tensor
    const unsigned      layer1_downsample_weight_grad_sizes[] = {1, 1, 64, 256};
    uint64_t            layer1_downsample_weight_grad_dram    = 0;
    synLaunchTensorInfo layer1_downsample_weight_grad_tr_info = {"layer1_downsample_weight_grad",
                                                                 layer1_downsample_weight_grad_dram};
    synTensor           layer1_downsample_weight_grad =
        createTensor(4U, syn_type_single, layer1_downsample_weight_grad_sizes, true, "layer1_downsample_weight_grad");

    synTensor layer1_downsample_dedw_out_vec[1] = {layer1_downsample_weight_grad};

    status = synNodeCreate(graphHandle,
                           layer1_downsample_dedw_in_vec,
                           layer1_downsample_dedw_out_vec,
                           2,
                           1,
                           (void*)&layer1_downsample_dedw_kernel_params,
                           sizeof(layer1_downsample_dedw_kernel_params),
                           "dedw",
                           "layer1_downsample_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_downsample_dedw failed!");

    /*************
     * layer1_0_add_residual_fwd1 node
     * inputs: [layer1_0_conv1_grad_input(64, 56, 56, 64)(dtype=bf16), layer1_downsample_grad_input(64, 56, 56,
     *64)(dtype=bf16)] output: [layer1_0_residual_upstream_grad_input(64, 56, 56, 64)(dtype=bf16)]
     *************/

    synTensor layer1_0_add_residual_fwd1_in_vec[2] = {layer1_0_conv1_grad_input, layer1_downsample_grad_input};

    // create layer1_0_residual_upstream_grad_input tensor
    const unsigned layer1_0_residual_upstream_grad_input_sizes[] = {64, 56, 56, 64};
    synTensor      layer1_0_residual_upstream_grad_input         = createTensor(4U,
                                                                   syn_type_bf16,
                                                                   layer1_0_residual_upstream_grad_input_sizes,
                                                                   false,
                                                                   "layer1_0_residual_upstream_grad_input");

    synTensor layer1_0_add_residual_fwd1_out_vec[1] = {layer1_0_residual_upstream_grad_input};

    status = synNodeCreate(graphHandle,
                           layer1_0_add_residual_fwd1_in_vec,
                           layer1_0_add_residual_fwd1_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "add_fwd_bf16",
                           "layer1_0_add_residual_fwd1",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for layer1_0_add_residual_fwd1 failed!");

    /*************
     * worker_0_maxpool_bwd node
     * inputs: [layer1_0_residual_upstream_grad_input(64, 56, 56, 64)(dtype=bf16), worker_0_maxpoolmax_indices(64, 56,
     *56, 64)(dtype=uint16)] output: [worker_0_maxpool_grad_input(64, 112, 112, 64)(dtype=bf16)]
     *************/
    ns_SpatialReduction::Params worker_0_maxpool_bwd_kernel_params;
    worker_0_maxpool_bwd_kernel_params.kernel_w           = 3;
    worker_0_maxpool_bwd_kernel_params.kernel_h           = 3;
    worker_0_maxpool_bwd_kernel_params.stride_w           = 2;
    worker_0_maxpool_bwd_kernel_params.stride_h           = 2;
    worker_0_maxpool_bwd_kernel_params.pad_w_begin        = 1;
    worker_0_maxpool_bwd_kernel_params.pad_w_end          = 1;
    worker_0_maxpool_bwd_kernel_params.pad_h_begin        = 1;
    worker_0_maxpool_bwd_kernel_params.pad_h_end          = 1;
    worker_0_maxpool_bwd_kernel_params.dilation_w         = 1;
    worker_0_maxpool_bwd_kernel_params.dilation_h         = 1;
    worker_0_maxpool_bwd_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    synTensor worker_0_maxpool_bwd_in_vec[2] = {layer1_0_residual_upstream_grad_input, worker_0_maxpoolmax_indices};

    // create worker_0_maxpool_grad_input tensor
    const unsigned worker_0_maxpool_grad_input_sizes[] = {64, 112, 112, 64};
    synTensor      worker_0_maxpool_grad_input =
        createTensor(4U, syn_type_bf16, worker_0_maxpool_grad_input_sizes, false, "worker_0_maxpool_grad_input");

    synTensor worker_0_maxpool_bwd_out_vec[1] = {worker_0_maxpool_grad_input};

    status = synNodeCreate(graphHandle,
                           worker_0_maxpool_bwd_in_vec,
                           worker_0_maxpool_bwd_out_vec,
                           2,
                           1,
                           (void*)&worker_0_maxpool_bwd_kernel_params,
                           sizeof(worker_0_maxpool_bwd_kernel_params),
                           "maxpool_2d_bwd_bf16",
                           "worker_0_maxpool_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_maxpool_bwd failed!");

    /*************
     * worker_0_relu_bwd node
     * inputs: [worker_0_maxpool_grad_input(64, 112, 112, 64)(dtype=bf16), worker_0_relu_output(64, 112, 112,
     *64)(dtype=bf16)] output: [worker_0_relu_grad_input(64, 112, 112, 64)(dtype=bf16)]
     *************/

    synTensor worker_0_relu_bwd_in_vec[2] = {worker_0_maxpool_grad_input, worker_0_relu_output};

    // create worker_0_relu_grad_input tensor
    const unsigned worker_0_relu_grad_input_sizes[] = {64, 112, 112, 64};
    synTensor      worker_0_relu_grad_input =
        createTensor(4U, syn_type_bf16, worker_0_relu_grad_input_sizes, false, "worker_0_relu_grad_input");

    synTensor worker_0_relu_bwd_out_vec[1] = {worker_0_relu_grad_input};

    status = synNodeCreate(graphHandle,
                           worker_0_relu_bwd_in_vec,
                           worker_0_relu_bwd_out_vec,
                           2,
                           1,
                           nullptr,
                           0,
                           "relu_bwd_bf16",
                           "worker_0_relu_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_relu_bwd failed!");

    /*************
     * worker_0_bn1_bwd node
     * inputs: [worker_0_conv1_output(64, 112, 112, 64)(dtype=bf16), worker_0_relu_grad_input(64, 112, 112,
     *64)(dtype=bf16), worker_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), worker_0_bn1_saved_var(1, 1, 1,
     *64)(dtype=float32), worker_0_bn1_weight[64](dtype=float32)] output: [worker_0_bn1_grad_input(64, 112, 112,
     *64)(dtype=bf16), worker_0_bn1_bias_grad(64,)(dtype=float32), worker_0_bn1_weight_grad(64,)(dtype=float32)]
     *************/

    synTensor worker_0_bn1_bwd_in_vec[5] = {worker_0_conv1_output,
                                            worker_0_relu_grad_input,
                                            worker_0_bn1_saved_mean,
                                            worker_0_bn1_saved_var,
                                            worker_0_bn1_weight};

    // create worker_0_bn1_grad_input tensor
    const unsigned worker_0_bn1_grad_input_sizes[] = {64, 112, 112, 64};
    synTensor      worker_0_bn1_grad_input =
        createTensor(4U, syn_type_bf16, worker_0_bn1_grad_input_sizes, false, "worker_0_bn1_grad_input");

    // create worker_0_bn1_bias_grad tensor
    const unsigned worker_0_bn1_bias_grad_sizes[] = {
        64,
    };
    uint64_t            worker_0_bn1_bias_grad_dram    = 0;
    synLaunchTensorInfo worker_0_bn1_bias_grad_tr_info = {"worker_0_bn1_bias_grad", worker_0_bn1_bias_grad_dram};
    synTensor           worker_0_bn1_bias_grad =
        createTensor(1U, syn_type_single, worker_0_bn1_bias_grad_sizes, true, "worker_0_bn1_bias_grad");

    // create worker_0_bn1_weight_grad tensor
    const unsigned worker_0_bn1_weight_grad_sizes[] = {
        64,
    };
    uint64_t            worker_0_bn1_weight_grad_dram    = 0;
    synLaunchTensorInfo worker_0_bn1_weight_grad_tr_info = {"worker_0_bn1_weight_grad", worker_0_bn1_weight_grad_dram};
    synTensor           worker_0_bn1_weight_grad =
        createTensor(1U, syn_type_single, worker_0_bn1_weight_grad_sizes, true, "worker_0_bn1_weight_grad");

    synTensor worker_0_bn1_bwd_out_vec[3] = {worker_0_bn1_grad_input, worker_0_bn1_bias_grad, worker_0_bn1_weight_grad};

    status = synNodeCreate(graphHandle,
                           worker_0_bn1_bwd_in_vec,
                           worker_0_bn1_bwd_out_vec,
                           5,
                           3,
                           nullptr,
                           0,
                           "batch_norm_bwd_bf16",
                           "worker_0_bn1_bwd",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_bn1_bwd failed!");

    /*************
     * worker_0_conv1_dedw node
     * inputs: [worker_0_bn1_grad_input(64, 112, 112, 64)(dtype=bf16), input[64, 224, 224, 3](dtype=bf16)]
     * output: [worker_0_conv1_weight_grad(7, 7, 3, 64)(dtype=float32)]
     *************/
    synConvolutionParams worker_0_conv1_dedw_kernel_params;
    worker_0_conv1_dedw_kernel_params.dH   = 2;
    worker_0_conv1_dedw_kernel_params.dW   = 2;
    worker_0_conv1_dedw_kernel_params.kH   = 7;
    worker_0_conv1_dedw_kernel_params.kW   = 7;
    worker_0_conv1_dedw_kernel_params.padT = 3;
    worker_0_conv1_dedw_kernel_params.padB = 3;
    worker_0_conv1_dedw_kernel_params.padL = 3;
    worker_0_conv1_dedw_kernel_params.padR = 3;
    worker_0_conv1_dedw_kernel_params.dilH = 1;
    worker_0_conv1_dedw_kernel_params.dilW = 1;

    synTensor worker_0_conv1_dedw_in_vec[2] = {worker_0_bn1_grad_input, input};

    // create worker_0_conv1_weight_grad tensor
    const unsigned      worker_0_conv1_weight_grad_sizes[] = {7, 7, 3, 64};
    uint64_t            worker_0_conv1_weight_grad_dram    = 0;
    synLaunchTensorInfo worker_0_conv1_weight_grad_tr_info = {"worker_0_conv1_weight_grad",
                                                              worker_0_conv1_weight_grad_dram};
    synTensor           worker_0_conv1_weight_grad =
        createTensor(4U, syn_type_single, worker_0_conv1_weight_grad_sizes, true, "worker_0_conv1_weight_grad");

    synTensor worker_0_conv1_dedw_out_vec[1] = {worker_0_conv1_weight_grad};

    status = synNodeCreate(graphHandle,
                           worker_0_conv1_dedw_in_vec,
                           worker_0_conv1_dedw_out_vec,
                           2,
                           1,
                           (void*)&worker_0_conv1_dedw_kernel_params,
                           sizeof(worker_0_conv1_dedw_kernel_params),
                           "dedw",
                           "worker_0_conv1_dedw",
                           nullptr,
                           nullptr);
    ASSERT_TRUE(status == synSuccess && "synNodeCreate for worker_0_conv1_dedw failed!");

    // generate graph
    LaunchInfo launchInfo;
    compileGraph(graphHandle);

    return;

    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    graph_inputs.push_back(input_tr_info);
    graph_inputs.push_back(worker_0_conv1_weight_tr_info);
    graph_inputs.push_back(worker_0_bn1_weight_tr_info);
    graph_inputs.push_back(worker_0_bn1_bias_tr_info);
    graph_inputs.push_back(worker_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(worker_0_bn1_running_var_tr_info);
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
    graph_inputs.push_back(layer1_1_conv1_weight_tr_info);
    graph_inputs.push_back(layer1_1_bn1_weight_tr_info);
    graph_inputs.push_back(layer1_1_bn1_bias_tr_info);
    graph_inputs.push_back(layer1_1_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer1_1_bn1_running_var_tr_info);
    graph_inputs.push_back(layer1_1_conv2_weight_tr_info);
    graph_inputs.push_back(layer1_1_bn2_weight_tr_info);
    graph_inputs.push_back(layer1_1_bn2_bias_tr_info);
    graph_inputs.push_back(layer1_1_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer1_1_bn2_running_var_tr_info);
    graph_inputs.push_back(layer1_1_conv3_weight_tr_info);
    graph_inputs.push_back(layer1_1_bn3_weight_tr_info);
    graph_inputs.push_back(layer1_1_bn3_bias_tr_info);
    graph_inputs.push_back(layer1_1_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer1_1_bn3_running_var_tr_info);
    graph_inputs.push_back(layer1_2_conv1_weight_tr_info);
    graph_inputs.push_back(layer1_2_bn1_weight_tr_info);
    graph_inputs.push_back(layer1_2_bn1_bias_tr_info);
    graph_inputs.push_back(layer1_2_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer1_2_bn1_running_var_tr_info);
    graph_inputs.push_back(layer1_2_conv2_weight_tr_info);
    graph_inputs.push_back(layer1_2_bn2_weight_tr_info);
    graph_inputs.push_back(layer1_2_bn2_bias_tr_info);
    graph_inputs.push_back(layer1_2_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer1_2_bn2_running_var_tr_info);
    graph_inputs.push_back(layer1_2_conv3_weight_tr_info);
    graph_inputs.push_back(layer1_2_bn3_weight_tr_info);
    graph_inputs.push_back(layer1_2_bn3_bias_tr_info);
    graph_inputs.push_back(layer1_2_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer1_2_bn3_running_var_tr_info);
    graph_inputs.push_back(layer2_0_conv1_weight_tr_info);
    graph_inputs.push_back(layer2_0_bn1_weight_tr_info);
    graph_inputs.push_back(layer2_0_bn1_bias_tr_info);
    graph_inputs.push_back(layer2_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer2_0_bn1_running_var_tr_info);
    graph_inputs.push_back(layer2_0_conv2_weight_tr_info);
    graph_inputs.push_back(layer2_0_bn2_weight_tr_info);
    graph_inputs.push_back(layer2_0_bn2_bias_tr_info);
    graph_inputs.push_back(layer2_0_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer2_0_bn2_running_var_tr_info);
    graph_inputs.push_back(layer2_0_conv3_weight_tr_info);
    graph_inputs.push_back(layer2_0_bn3_weight_tr_info);
    graph_inputs.push_back(layer2_0_bn3_bias_tr_info);
    graph_inputs.push_back(layer2_0_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer2_0_bn3_running_var_tr_info);
    graph_inputs.push_back(layer2_downsample_weight_tr_info);
    graph_inputs.push_back(layer2_bn_weight_tr_info);
    graph_inputs.push_back(layer2_bn_bias_tr_info);
    graph_inputs.push_back(layer2_bn_running_mean_tr_info);
    graph_inputs.push_back(layer2_bn_running_var_tr_info);
    graph_inputs.push_back(layer2_1_conv1_weight_tr_info);
    graph_inputs.push_back(layer2_1_bn1_weight_tr_info);
    graph_inputs.push_back(layer2_1_bn1_bias_tr_info);
    graph_inputs.push_back(layer2_1_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer2_1_bn1_running_var_tr_info);
    graph_inputs.push_back(layer2_1_conv2_weight_tr_info);
    graph_inputs.push_back(layer2_1_bn2_weight_tr_info);
    graph_inputs.push_back(layer2_1_bn2_bias_tr_info);
    graph_inputs.push_back(layer2_1_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer2_1_bn2_running_var_tr_info);
    graph_inputs.push_back(layer2_1_conv3_weight_tr_info);
    graph_inputs.push_back(layer2_1_bn3_weight_tr_info);
    graph_inputs.push_back(layer2_1_bn3_bias_tr_info);
    graph_inputs.push_back(layer2_1_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer2_1_bn3_running_var_tr_info);
    graph_inputs.push_back(layer2_2_conv1_weight_tr_info);
    graph_inputs.push_back(layer2_2_bn1_weight_tr_info);
    graph_inputs.push_back(layer2_2_bn1_bias_tr_info);
    graph_inputs.push_back(layer2_2_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer2_2_bn1_running_var_tr_info);
    graph_inputs.push_back(layer2_2_conv2_weight_tr_info);
    graph_inputs.push_back(layer2_2_bn2_weight_tr_info);
    graph_inputs.push_back(layer2_2_bn2_bias_tr_info);
    graph_inputs.push_back(layer2_2_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer2_2_bn2_running_var_tr_info);
    graph_inputs.push_back(layer2_2_conv3_weight_tr_info);
    graph_inputs.push_back(layer2_2_bn3_weight_tr_info);
    graph_inputs.push_back(layer2_2_bn3_bias_tr_info);
    graph_inputs.push_back(layer2_2_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer2_2_bn3_running_var_tr_info);
    graph_inputs.push_back(layer2_3_conv1_weight_tr_info);
    graph_inputs.push_back(layer2_3_bn1_weight_tr_info);
    graph_inputs.push_back(layer2_3_bn1_bias_tr_info);
    graph_inputs.push_back(layer2_3_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer2_3_bn1_running_var_tr_info);
    graph_inputs.push_back(layer2_3_conv2_weight_tr_info);
    graph_inputs.push_back(layer2_3_bn2_weight_tr_info);
    graph_inputs.push_back(layer2_3_bn2_bias_tr_info);
    graph_inputs.push_back(layer2_3_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer2_3_bn2_running_var_tr_info);
    graph_inputs.push_back(layer2_3_conv3_weight_tr_info);
    graph_inputs.push_back(layer2_3_bn3_weight_tr_info);
    graph_inputs.push_back(layer2_3_bn3_bias_tr_info);
    graph_inputs.push_back(layer2_3_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer2_3_bn3_running_var_tr_info);
    graph_inputs.push_back(layer3_0_conv1_weight_tr_info);
    graph_inputs.push_back(layer3_0_bn1_weight_tr_info);
    graph_inputs.push_back(layer3_0_bn1_bias_tr_info);
    graph_inputs.push_back(layer3_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer3_0_bn1_running_var_tr_info);
    graph_inputs.push_back(layer3_0_conv2_weight_tr_info);
    graph_inputs.push_back(layer3_0_bn2_weight_tr_info);
    graph_inputs.push_back(layer3_0_bn2_bias_tr_info);
    graph_inputs.push_back(layer3_0_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer3_0_bn2_running_var_tr_info);
    graph_inputs.push_back(layer3_0_conv3_weight_tr_info);
    graph_inputs.push_back(layer3_0_bn3_weight_tr_info);
    graph_inputs.push_back(layer3_0_bn3_bias_tr_info);
    graph_inputs.push_back(layer3_0_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer3_0_bn3_running_var_tr_info);
    graph_inputs.push_back(layer3_downsample_weight_tr_info);
    graph_inputs.push_back(layer3_bn_weight_tr_info);
    graph_inputs.push_back(layer3_bn_bias_tr_info);
    graph_inputs.push_back(layer3_bn_running_mean_tr_info);
    graph_inputs.push_back(layer3_bn_running_var_tr_info);
    graph_inputs.push_back(layer3_1_conv1_weight_tr_info);
    graph_inputs.push_back(layer3_1_bn1_weight_tr_info);
    graph_inputs.push_back(layer3_1_bn1_bias_tr_info);
    graph_inputs.push_back(layer3_1_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer3_1_bn1_running_var_tr_info);
    graph_inputs.push_back(layer3_1_conv2_weight_tr_info);
    graph_inputs.push_back(layer3_1_bn2_weight_tr_info);
    graph_inputs.push_back(layer3_1_bn2_bias_tr_info);
    graph_inputs.push_back(layer3_1_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer3_1_bn2_running_var_tr_info);
    graph_inputs.push_back(layer3_1_conv3_weight_tr_info);
    graph_inputs.push_back(layer3_1_bn3_weight_tr_info);
    graph_inputs.push_back(layer3_1_bn3_bias_tr_info);
    graph_inputs.push_back(layer3_1_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer3_1_bn3_running_var_tr_info);
    graph_inputs.push_back(layer3_2_conv1_weight_tr_info);
    graph_inputs.push_back(layer3_2_bn1_weight_tr_info);
    graph_inputs.push_back(layer3_2_bn1_bias_tr_info);
    graph_inputs.push_back(layer3_2_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer3_2_bn1_running_var_tr_info);
    graph_inputs.push_back(layer3_2_conv2_weight_tr_info);
    graph_inputs.push_back(layer3_2_bn2_weight_tr_info);
    graph_inputs.push_back(layer3_2_bn2_bias_tr_info);
    graph_inputs.push_back(layer3_2_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer3_2_bn2_running_var_tr_info);
    graph_inputs.push_back(layer3_2_conv3_weight_tr_info);
    graph_inputs.push_back(layer3_2_bn3_weight_tr_info);
    graph_inputs.push_back(layer3_2_bn3_bias_tr_info);
    graph_inputs.push_back(layer3_2_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer3_2_bn3_running_var_tr_info);
    graph_inputs.push_back(layer3_3_conv1_weight_tr_info);
    graph_inputs.push_back(layer3_3_bn1_weight_tr_info);
    graph_inputs.push_back(layer3_3_bn1_bias_tr_info);
    graph_inputs.push_back(layer3_3_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer3_3_bn1_running_var_tr_info);
    graph_inputs.push_back(layer3_3_conv2_weight_tr_info);
    graph_inputs.push_back(layer3_3_bn2_weight_tr_info);
    graph_inputs.push_back(layer3_3_bn2_bias_tr_info);
    graph_inputs.push_back(layer3_3_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer3_3_bn2_running_var_tr_info);
    graph_inputs.push_back(layer3_3_conv3_weight_tr_info);
    graph_inputs.push_back(layer3_3_bn3_weight_tr_info);
    graph_inputs.push_back(layer3_3_bn3_bias_tr_info);
    graph_inputs.push_back(layer3_3_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer3_3_bn3_running_var_tr_info);
    graph_inputs.push_back(layer3_4_conv1_weight_tr_info);
    graph_inputs.push_back(layer3_4_bn1_weight_tr_info);
    graph_inputs.push_back(layer3_4_bn1_bias_tr_info);
    graph_inputs.push_back(layer3_4_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer3_4_bn1_running_var_tr_info);
    graph_inputs.push_back(layer3_4_conv2_weight_tr_info);
    graph_inputs.push_back(layer3_4_bn2_weight_tr_info);
    graph_inputs.push_back(layer3_4_bn2_bias_tr_info);
    graph_inputs.push_back(layer3_4_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer3_4_bn2_running_var_tr_info);
    graph_inputs.push_back(layer3_4_conv3_weight_tr_info);
    graph_inputs.push_back(layer3_4_bn3_weight_tr_info);
    graph_inputs.push_back(layer3_4_bn3_bias_tr_info);
    graph_inputs.push_back(layer3_4_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer3_4_bn3_running_var_tr_info);
    graph_inputs.push_back(layer3_5_conv1_weight_tr_info);
    graph_inputs.push_back(layer3_5_bn1_weight_tr_info);
    graph_inputs.push_back(layer3_5_bn1_bias_tr_info);
    graph_inputs.push_back(layer3_5_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer3_5_bn1_running_var_tr_info);
    graph_inputs.push_back(layer3_5_conv2_weight_tr_info);
    graph_inputs.push_back(layer3_5_bn2_weight_tr_info);
    graph_inputs.push_back(layer3_5_bn2_bias_tr_info);
    graph_inputs.push_back(layer3_5_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer3_5_bn2_running_var_tr_info);
    graph_inputs.push_back(layer3_5_conv3_weight_tr_info);
    graph_inputs.push_back(layer3_5_bn3_weight_tr_info);
    graph_inputs.push_back(layer3_5_bn3_bias_tr_info);
    graph_inputs.push_back(layer3_5_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer3_5_bn3_running_var_tr_info);
    graph_inputs.push_back(layer4_0_conv1_weight_tr_info);
    graph_inputs.push_back(layer4_0_bn1_weight_tr_info);
    graph_inputs.push_back(layer4_0_bn1_bias_tr_info);
    graph_inputs.push_back(layer4_0_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer4_0_bn1_running_var_tr_info);
    graph_inputs.push_back(layer4_0_conv2_weight_tr_info);
    graph_inputs.push_back(layer4_0_bn2_weight_tr_info);
    graph_inputs.push_back(layer4_0_bn2_bias_tr_info);
    graph_inputs.push_back(layer4_0_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer4_0_bn2_running_var_tr_info);
    graph_inputs.push_back(layer4_0_conv3_weight_tr_info);
    graph_inputs.push_back(layer4_0_bn3_weight_tr_info);
    graph_inputs.push_back(layer4_0_bn3_bias_tr_info);
    graph_inputs.push_back(layer4_0_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer4_0_bn3_running_var_tr_info);
    graph_inputs.push_back(layer4_downsample_weight_tr_info);
    graph_inputs.push_back(layer4_bn_weight_tr_info);
    graph_inputs.push_back(layer4_bn_bias_tr_info);
    graph_inputs.push_back(layer4_bn_running_mean_tr_info);
    graph_inputs.push_back(layer4_bn_running_var_tr_info);
    graph_inputs.push_back(layer4_1_conv1_weight_tr_info);
    graph_inputs.push_back(layer4_1_bn1_weight_tr_info);
    graph_inputs.push_back(layer4_1_bn1_bias_tr_info);
    graph_inputs.push_back(layer4_1_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer4_1_bn1_running_var_tr_info);
    graph_inputs.push_back(layer4_1_conv2_weight_tr_info);
    graph_inputs.push_back(layer4_1_bn2_weight_tr_info);
    graph_inputs.push_back(layer4_1_bn2_bias_tr_info);
    graph_inputs.push_back(layer4_1_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer4_1_bn2_running_var_tr_info);
    graph_inputs.push_back(layer4_1_conv3_weight_tr_info);
    graph_inputs.push_back(layer4_1_bn3_weight_tr_info);
    graph_inputs.push_back(layer4_1_bn3_bias_tr_info);
    graph_inputs.push_back(layer4_1_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer4_1_bn3_running_var_tr_info);
    graph_inputs.push_back(layer4_2_conv1_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn1_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn1_bias_tr_info);
    graph_inputs.push_back(layer4_2_bn1_running_mean_tr_info);
    graph_inputs.push_back(layer4_2_bn1_running_var_tr_info);
    graph_inputs.push_back(layer4_2_conv2_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn2_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn2_bias_tr_info);
    graph_inputs.push_back(layer4_2_bn2_running_mean_tr_info);
    graph_inputs.push_back(layer4_2_bn2_running_var_tr_info);
    graph_inputs.push_back(layer4_2_conv3_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn3_weight_tr_info);
    graph_inputs.push_back(layer4_2_bn3_bias_tr_info);
    graph_inputs.push_back(layer4_2_bn3_running_mean_tr_info);
    graph_inputs.push_back(layer4_2_bn3_running_var_tr_info);
    graph_inputs.push_back(worker_0_fc_weight_tr_info);
    graph_inputs.push_back(worker_0_fc_bias_tr_info);
    graph_inputs.push_back(target_tr_info);

    graph_outputs.push_back(cross_entropy_loss0_output_tr_info);
    graph_outputs.push_back(worker_0_fc_weight_grad_tr_info);
    graph_outputs.push_back(worker_0_fc_bias_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer4_2_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer4_2_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer4_2_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer4_2_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer4_1_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_1_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer4_1_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_1_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer4_1_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer4_1_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer4_1_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer4_1_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer4_1_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer4_0_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_0_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer4_0_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer4_0_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer4_0_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer4_0_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer4_0_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer4_0_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer4_0_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer4_bn_weight_grad_tr_info);
    graph_outputs.push_back(layer4_bn_bias_grad_tr_info);
    graph_outputs.push_back(layer4_downsample_weight_grad_tr_info);
    graph_outputs.push_back(layer3_5_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_5_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer3_5_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_5_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_5_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer3_5_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_5_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_5_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer3_5_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_4_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_4_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer3_4_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_4_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_4_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer3_4_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_4_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_4_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer3_4_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_3_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_3_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer3_3_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_3_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_3_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer3_3_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_3_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_3_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer3_3_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_2_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_2_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer3_2_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_2_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_2_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer3_2_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_2_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_2_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer3_2_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_1_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_1_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer3_1_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_1_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_1_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer3_1_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_1_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_1_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer3_1_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_0_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_0_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer3_0_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer3_0_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_0_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer3_0_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer3_0_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_0_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer3_0_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer3_bn_weight_grad_tr_info);
    graph_outputs.push_back(layer3_bn_bias_grad_tr_info);
    graph_outputs.push_back(layer3_downsample_weight_grad_tr_info);
    graph_outputs.push_back(layer2_3_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_3_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer2_3_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_3_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_3_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer2_3_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_3_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_3_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer2_3_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_2_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_2_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer2_2_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_2_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_2_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer2_2_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_2_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_2_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer2_2_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_1_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_1_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer2_1_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_1_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_1_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer2_1_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_1_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_1_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer2_1_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_0_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_0_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer2_0_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer2_0_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_0_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer2_0_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer2_0_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_0_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer2_0_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer2_bn_weight_grad_tr_info);
    graph_outputs.push_back(layer2_bn_bias_grad_tr_info);
    graph_outputs.push_back(layer2_downsample_weight_grad_tr_info);
    graph_outputs.push_back(layer1_2_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer1_2_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer1_2_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer1_2_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer1_2_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer1_2_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer1_2_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer1_2_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer1_2_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer1_1_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer1_1_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer1_1_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer1_1_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer1_1_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer1_1_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer1_1_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer1_1_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer1_1_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer1_0_bn3_weight_grad_tr_info);
    graph_outputs.push_back(layer1_0_bn3_bias_grad_tr_info);
    graph_outputs.push_back(layer1_0_conv3_weight_grad_tr_info);
    graph_outputs.push_back(layer1_0_bn2_weight_grad_tr_info);
    graph_outputs.push_back(layer1_0_bn2_bias_grad_tr_info);
    graph_outputs.push_back(layer1_0_conv2_weight_grad_tr_info);
    graph_outputs.push_back(layer1_0_bn1_weight_grad_tr_info);
    graph_outputs.push_back(layer1_0_bn1_bias_grad_tr_info);
    graph_outputs.push_back(layer1_0_conv1_weight_grad_tr_info);
    graph_outputs.push_back(layer1_bn_weight_grad_tr_info);
    graph_outputs.push_back(layer1_bn_bias_grad_tr_info);
    graph_outputs.push_back(layer1_downsample_weight_grad_tr_info);
    graph_outputs.push_back(worker_0_bn1_weight_grad_tr_info);
    graph_outputs.push_back(worker_0_bn1_bias_grad_tr_info);
    graph_outputs.push_back(worker_0_conv1_weight_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    cleanup();
}