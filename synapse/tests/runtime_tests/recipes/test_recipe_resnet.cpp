#include "test_recipe_resnet.hpp"

#include "defs.h"

#include "perf_lib_layer_params.h"

#include "synapse_api.h"
#include "../infra/test_types.hpp"

#include "test_tensors_container.hpp"
#include "test_utils.h"

#include <iostream>
#include <vector>

const unsigned inputTensorsPerLaunch  = 6;
const unsigned outputTensorsPerLaunch = 3;

TestRecipeResnet::TestRecipeResnet(synDeviceType deviceType, bool isSfg)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeResnet>(isSfg ? "sfg" : "notsfg"),
                 deviceType,
                 6 /* inputTensorsAmount   */,
                 0 /* innerTensorsAmount   */,
                 3 /* outputTensorsAmount  */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    m_tensorNames = {// Inputs
                     "layer1_0_conv1_output",
                     "layer1_0_bn1_weight",
                     "layer1_0_bn1_bias",
                     "layer1_0_bn1_running_mean",
                     "layer1_0_bn1_running_var",
                     "layer1_0_conv2_weight",
                     // Outputs
                     "layer1_0_bn1_saved_mean",
                     "layer1_0_bn1_saved_var",
                     "layer1_0_conv2_output"};

    m_isSfg = isSfg;

    unsigned dims                 = 4U;
    uint64_t tensorSizeInElements = 64 * 56 * 56 * 64;

    m_tensorInfoVecInputs[0].m_tensorName = "layer1_0_conv1_output";
    m_tensorInfoVecInputs[0].m_dimsAmount = dims;
    m_tensorInfoVecInputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[0].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[0].m_dataType);
    //
    const TSize layer1_0_conv1_output_sizes[] = {64, 56, 56, 64};
    std::copy(layer1_0_conv1_output_sizes,
              layer1_0_conv1_output_sizes + dims,
              m_tensorInfoVecInputs[0].m_tensorDimsSize);

    dims                 = 1U;
    tensorSizeInElements = 64;

    m_tensorInfoVecInputs[1].m_tensorName = "layer1_0_bn1_weight";
    m_tensorInfoVecInputs[1].m_dimsAmount = dims;
    m_tensorInfoVecInputs[1].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[1].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[1].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[1].m_dataType);
    //
    const TSize layer1_0_bn1_weight_sizes[] = {64};
    std::copy(layer1_0_bn1_weight_sizes, layer1_0_bn1_weight_sizes + dims, m_tensorInfoVecInputs[1].m_tensorDimsSize);

    m_tensorInfoVecInputs[2].m_tensorName = "layer1_0_bn1_bias";
    m_tensorInfoVecInputs[2].m_dimsAmount = dims;
    m_tensorInfoVecInputs[2].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[2].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[2].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[2].m_dataType);
    //
    const TSize layer1_0_bn1_bias_sizes[] = {64};
    std::copy(layer1_0_bn1_bias_sizes, layer1_0_bn1_bias_sizes + dims, m_tensorInfoVecInputs[2].m_tensorDimsSize);

    m_tensorInfoVecInputs[3].m_tensorName = "layer1_0_bn1_running_mean";
    m_tensorInfoVecInputs[3].m_dimsAmount = dims;
    m_tensorInfoVecInputs[3].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[3].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[3].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[3].m_dataType);
    //
    const TSize layer1_0_bn1_running_mean_sizes[] = {64};
    std::copy(layer1_0_bn1_running_mean_sizes,
              layer1_0_bn1_running_mean_sizes + dims,
              m_tensorInfoVecInputs[3].m_tensorDimsSize);

    m_tensorInfoVecInputs[4].m_tensorName = "layer1_0_bn1_running_var";
    m_tensorInfoVecInputs[4].m_dimsAmount = dims;
    m_tensorInfoVecInputs[4].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[4].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[4].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[4].m_dataType);
    //
    const TSize layer1_0_bn1_running_var_sizes[] = {64};
    std::copy(layer1_0_bn1_running_var_sizes,
              layer1_0_bn1_running_var_sizes + dims,
              m_tensorInfoVecInputs[4].m_tensorDimsSize);

    dims                 = 4U;
    tensorSizeInElements = 64 * 64 * 3 * 3;

    m_tensorInfoVecInputs[5].m_tensorName = "layer1_0_conv2_weight";
    m_tensorInfoVecInputs[5].m_dimsAmount = dims;
    m_tensorInfoVecInputs[5].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[5].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[5].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[5].m_dataType);
    //
    const TSize layer1_0_conv2_weight_sizes[] = {3, 3, 64, 64};
    std::copy(layer1_0_conv2_weight_sizes,
              layer1_0_conv2_weight_sizes + dims,
              m_tensorInfoVecInputs[5].m_tensorDimsSize);

    dims                 = 1U;
    tensorSizeInElements = 64;

    m_tensorInfoVecOutputs[0].m_tensorName = "layer1_0_bn1_saved_mean";
    m_tensorInfoVecOutputs[0].m_dimsAmount = dims;
    m_tensorInfoVecOutputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_dataType   = syn_type_single;
    m_tensorInfoVecOutputs[0].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);
    //
    const TSize layer1_0_bn1_saved_mean_sizes[] = {64};
    std::copy(layer1_0_bn1_saved_mean_sizes,
              layer1_0_bn1_saved_mean_sizes + dims,
              m_tensorInfoVecOutputs[0].m_tensorDimsSize);

    m_tensorInfoVecOutputs[1].m_tensorName = "layer1_0_bn1_saved_var";
    m_tensorInfoVecOutputs[1].m_dimsAmount = dims;
    m_tensorInfoVecOutputs[1].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecOutputs[1].m_dataType   = syn_type_single;
    m_tensorInfoVecOutputs[1].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[1].m_dataType);
    //
    const TSize layer1_0_bn1_saved_var_sizes[] = {64};
    std::copy(layer1_0_bn1_saved_var_sizes,
              layer1_0_bn1_saved_var_sizes + dims,
              m_tensorInfoVecOutputs[1].m_tensorDimsSize);

    dims                 = 4U;
    tensorSizeInElements = 64 * 56 * 56 * 64;

    m_tensorInfoVecOutputs[2].m_tensorName = "layer1_0_conv2_output";
    m_tensorInfoVecOutputs[2].m_dimsAmount = dims;
    m_tensorInfoVecOutputs[2].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecOutputs[2].m_dataType   = syn_type_single;
    m_tensorInfoVecOutputs[2].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[2].m_dataType);
    //
    const TSize layer1_0_conv2_output_sizes[] = {64, 56, 56, 64};
    std::copy(layer1_0_conv2_output_sizes,
              layer1_0_conv2_output_sizes + dims,
              m_tensorInfoVecOutputs[2].m_tensorDimsSize);
}

void TestRecipeResnet::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    LOG_ERR(SYN_RT_TEST, "This method ({}) must be overriden", __FUNCTION__);
    ASSERT_TRUE(false) << "This method (" << __func__ << ") must be overriden";
}

synStatus TestRecipeResnet::generateSfg()
{
    uint64_t  tensorIdExecOrder[m_numOfExternalTensors];
    synStatus status;
    if (m_isSfg)
    {
        status = synTensorExtExtractExecutionOrder(m_recipeHandle, m_numOfExternalTensors, tensorIdExecOrder);
        HB_ASSERT(status == synSuccess, "Failed to extract tensor execution order");
    }

    std::vector<const char*> tensorNames;
    for (auto& name : m_tensorNames)
    {
        tensorNames.push_back(name.c_str());
    }

    uint64_t inputTensorIds[inputTensorsPerLaunch], outputTensorIds[outputTensorsPerLaunch];
    status = synTensorRetrieveIds(m_recipeHandle, tensorNames.data(), inputTensorIds, inputTensorsPerLaunch);
    HB_ASSERT(status == synSuccess, "Failed to extract input synTensorRetrieveIds");
    status = synTensorRetrieveIds(m_recipeHandle,
                                  tensorNames.data() + inputTensorsPerLaunch,
                                  outputTensorIds,
                                  outputTensorsPerLaunch);
    HB_ASSERT(status == synSuccess, "Failed to extract output synTensorRetrieveIds");

    std::unique_ptr<uint64_t[]> tensorIds(new uint64_t[getTensorInfoVecSize()]);
    status = synTensorRetrieveIds(m_recipeHandle, tensorNames.data(), tensorIds.get(), getTensorInfoVecSize());
    HB_ASSERT(status == synSuccess, "Failed to extract total synTensorRetrieveIds");

    size_t tensorCounter = 0;
    m_tensorExtIdx.resize(m_numOfExternalTensors);
    for (size_t i = 0; i < inputTensorsPerLaunch; i++)
    {
        m_tensorInfoVecInputs[i].m_tensorId = inputTensorIds[i];
        if (m_isSfg && tensorCounter < m_numOfExternalTensors)
        {
            for (size_t execTensorIndex = 0; execTensorIndex < m_numOfExternalTensors; execTensorIndex++)
            {
                if (m_tensorInfoVecInputs[i].m_tensorId == tensorIdExecOrder[execTensorIndex])
                {
                    LOG_DEBUG(SYN_RT_TEST,
                              "tensorName {}, idx {} order {} id {}",
                              m_tensorInfoVecInputs[i].m_tensorName,
                              i,
                              execTensorIndex,
                              m_tensorInfoVecInputs[i].m_tensorId);
                    m_tensorExtIdx[execTensorIndex] = i;
                    tensorCounter++;
                }
            }
        }
    }

    for (size_t i = 0; i < outputTensorsPerLaunch; i++)
    {
        m_tensorInfoVecOutputs[i].m_tensorId = outputTensorIds[i];
        if (m_isSfg && tensorCounter < m_numOfExternalTensors)
        {
            for (size_t execTensorIndex = 0; execTensorIndex < m_numOfExternalTensors; execTensorIndex++)
            {
                if (m_tensorInfoVecOutputs[i].m_tensorId == tensorIdExecOrder[execTensorIndex])
                {
                    LOG_DEBUG(SYN_RT_TEST,
                              "tensorName {}, idx {} order {} id {}",
                              m_tensorInfoVecOutputs[i].m_tensorName,
                              i + inputTensorsPerLaunch,
                              execTensorIndex,
                              m_tensorInfoVecOutputs[i].m_tensorId);
                    m_tensorExtIdx[execTensorIndex] = i + inputTensorsPerLaunch;
                    tensorCounter++;
                }
            }
        }
    }
    return synSuccess;
}

void TestRecipeResnet::compileGraph()
{
    synStatus status(synSuccess);

    _createGraphHandle();

    /*************
     * layer1_0_bn1 node
     * inputs: [layer1_0_conv1_output[64, 56, 56, 64](dtype=float32), layer1_0_bn1_bias[64](dtype=float32),
     *          layer1_0_bn1_weight[64](dtype=float32), layer1_0_bn1_running_mean[64](dtype=float32),
     *          layer1_0_bn1_running_var[64](dtype=float32)] output: [layer1_0_bn1_output(64, 56, 56,
     *64)(dtype=float32), layer1_0_bn1_saved_mean(1, 1, 1, 64)(dtype=float32), layer1_0_bn1_saved_var(1, 1, 1,
     *64)(dtype=float32)]
     *************/
    ns_BatchNormKernel::Params layer1_0_bn1_kernel_params;
    layer1_0_bn1_kernel_params.momentum    = 0.1;
    layer1_0_bn1_kernel_params.threshold.f = 1e-05;
    layer1_0_bn1_kernel_params.epsilon     = 1e-05;

    // create layer1_0_conv1_output tensor
    const TSize          layer1_0_conv1_output_sizes[] = {64, 56, 56, 64};
    TestTensorsContainer layer1_0_conv1_output(1);
    createTrainingTensor(layer1_0_conv1_output,
                         0,
                         4U,
                         syn_type_single,
                         layer1_0_conv1_output_sizes,
                         true,
                         "layer1_0_conv1_output",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    // create layer1_0_bn1_bias tensor
    const TSize          layer1_0_bn1_bias_sizes[] = {64};
    TestTensorsContainer layer1_0_bn1_bias(1);
    createTrainingTensor(layer1_0_bn1_bias,
                         0,
                         1U,
                         syn_type_single,
                         layer1_0_bn1_bias_sizes,
                         true,
                         "layer1_0_bn1_bias",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    // create layer1_0_bn1_weight tensor
    const TSize          layer1_0_bn1_weight_sizes[] = {64};
    TestTensorsContainer layer1_0_bn1_weight(1);
    createTrainingTensor(layer1_0_bn1_weight,
                         0,
                         1U,
                         syn_type_single,
                         layer1_0_bn1_weight_sizes,
                         true,
                         "layer1_0_bn1_weight",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    // create layer1_0_bn1_running_mean tensor
    const TSize          layer1_0_bn1_running_mean_sizes[] = {64};
    TestTensorsContainer layer1_0_bn1_running_mean(1);
    createTrainingTensor(layer1_0_bn1_running_mean,
                         0,
                         1U,
                         syn_type_single,
                         layer1_0_bn1_running_mean_sizes,
                         true,
                         "layer1_0_bn1_running_mean",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    // create layer1_0_bn1_running_var tensor
    const TSize          layer1_0_bn1_running_var_sizes[] = {64};
    TestTensorsContainer layer1_0_bn1_running_var(1);
    createTrainingTensor(layer1_0_bn1_running_var,
                         0,
                         1U,
                         syn_type_single,
                         layer1_0_bn1_running_var_sizes,
                         true,
                         "layer1_0_bn1_running_var",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    synTensor layer1_0_bn1_in_vec[5] = {layer1_0_conv1_output.tensor(0),
                                        layer1_0_bn1_bias.tensor(0),
                                        layer1_0_bn1_weight.tensor(0),
                                        layer1_0_bn1_running_mean.tensor(0),
                                        layer1_0_bn1_running_var.tensor(0)};

    // create layer1_0_bn1_output tensor
    const TSize          layer1_0_bn1_output_sizes[] = {64, 56, 56, 64};
    TestTensorsContainer layer1_0_bn1_output(1);
    createTrainingTensor(layer1_0_bn1_output,
                         0,
                         4U,
                         syn_type_single,
                         layer1_0_bn1_output_sizes,
                         false,
                         "layer1_0_bn1_output",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    // create layer1_0_bn1_saved_mean tensor
    const TSize          layer1_0_bn1_saved_mean_sizes[] = {64};
    TestTensorsContainer layer1_0_bn1_saved_mean(1);
    createTrainingTensor(layer1_0_bn1_saved_mean,
                         0,
                         1U,
                         syn_type_single,
                         layer1_0_bn1_saved_mean_sizes,
                         true,
                         "layer1_0_bn1_saved_mean",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    // create layer1_0_bn1_saved_var tensor
    const TSize          layer1_0_bn1_saved_var_sizes[] = {64};
    TestTensorsContainer layer1_0_bn1_saved_var(1);
    createTrainingTensor(layer1_0_bn1_saved_var,
                         0,
                         1U,
                         syn_type_single,
                         layer1_0_bn1_saved_var_sizes,
                         true,
                         "layer1_0_bn1_saved_var",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);
    if (m_isSfg)
    {
        status = synTensorSetExternal(layer1_0_bn1_saved_mean.tensor(0), m_isSfg);
        ASSERT_EQ(status, synSuccess) << "Failed to synGraphCreate";

        bool resIsExternal = false;
        status             = synTensorGetExternal(layer1_0_bn1_saved_mean.tensor(0), &resIsExternal);
        ASSERT_EQ(status, synSuccess) << "Failed to synGraphCreate";
        ASSERT_EQ(m_isSfg, resIsExternal) << "Failed to synGraphCreate";

        resIsExternal = !m_isSfg;
        status        = synTensorSetExternal(layer1_0_bn1_saved_var.tensor(0), m_isSfg);
        ASSERT_EQ(status, synSuccess) << "Failed to synGraphCreate";
        status = synTensorGetExternal(layer1_0_bn1_saved_var.tensor(0), &resIsExternal);
        ASSERT_EQ(status, synSuccess) << "Failed to synGraphCreate";
        ASSERT_EQ(m_isSfg, resIsExternal) << "Failed to synGraphCreate";
    }

    synTensor layer1_0_bn1_out_vec[3] = {layer1_0_bn1_output.tensor(0),
                                         layer1_0_bn1_saved_mean.tensor(0),
                                         layer1_0_bn1_saved_var.tensor(0)};

    status = synNodeCreate(m_graphHandle,
                           layer1_0_bn1_in_vec,
                           layer1_0_bn1_out_vec,
                           5,
                           3,
                           (void*)&layer1_0_bn1_kernel_params,
                           sizeof(layer1_0_bn1_kernel_params),
                           "batch_norm_fwd_f32",
                           "layer1_0_bn1",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "synNodeCreate for layer1_0_bn1 failed!";

    /*************
     * layer1_0_relu1 node
     * inputs: [layer1_0_bn1_output(64, 56, 56, 64)(dtype=float32)]
     * output: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=float32)]
     *************/

    synTensor layer1_0_relu1_in_vec[1] = {layer1_0_bn1_output.tensor(0)};

    // create layer1_0_relu1_output tensor
    const TSize          layer1_0_relu1_output_sizes[] = {64, 56, 56, 64};
    TestTensorsContainer layer1_0_relu1_output(1);
    createTrainingTensor(layer1_0_relu1_output,
                         0,
                         4U,
                         syn_type_single,
                         layer1_0_relu1_output_sizes,
                         false,
                         "layer1_0_relu1_output",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    status = synNodeCreate(m_graphHandle,
                           layer1_0_relu1_in_vec,
                           layer1_0_relu1_output.tensors(),
                           1,
                           1,
                           nullptr,
                           0,
                           "relu_fwd_f32",
                           "layer1_0_relu1",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "synNodeCreate for layer1_0_relu1 failed!";

    /*************
     * layer1_0_conv2 node
     * inputs: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=float32), layer1_0_conv2_weight[3, 3, 64,
     *64](dtype=float32)] output: [layer1_0_conv2_output(64, 56, 56, 64)(dtype=float32)]
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
    const TSize          layer1_0_conv2_weight_sizes[] = {64, 64, 3, 3};  // {3, 3, 64, 64};
    TestTensorsContainer layer1_0_conv2_weight(1);
    createTrainingTensor(layer1_0_conv2_weight,
                         0,
                         4U,
                         syn_type_single,
                         layer1_0_conv2_weight_sizes,
                         true,
                         "layer1_0_conv2_weight",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    synTensor layer1_0_conv2_in_vec[4] = {layer1_0_relu1_output.tensor(0),
                                          layer1_0_conv2_weight.tensor(0),
                                          nullptr,
                                          nullptr};

    // create layer1_0_conv2_output tensor
    const TSize          layer1_0_conv2_output_sizes[] = {64, 56, 56, 64};
    TestTensorsContainer layer1_0_conv2_output(1);
    createTrainingTensor(layer1_0_conv2_output,
                         0,
                         4U,
                         syn_type_single,
                         layer1_0_conv2_output_sizes,
                         true,
                         "layer1_0_conv2_output",
                         m_graphHandle,
                         nullptr /* pGivenSectionHandle */,
                         false /* isConstSection */,
                         0 /* offset */,
                         nullptr /* hostBuffer */,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    synTensor layer1_0_conv2_out_vec[1] = {layer1_0_conv2_output.tensor(0)};
    if (m_isSfg)
    {
        status = synTensorSetExternal(layer1_0_conv2_output.tensor(0), m_isSfg);
        ASSERT_EQ(status, synSuccess) << "Failed to synGraphCreate";

        bool resIsExternal = false;
        status             = synTensorGetExternal(layer1_0_conv2_output.tensor(0), &resIsExternal);
        ASSERT_EQ(status, synSuccess) << "Failed to synGraphCreate";
        ASSERT_TRUE(resIsExternal) << "resIsExternal is false";
    }

    status = synNodeCreate(m_graphHandle,
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
    ASSERT_EQ(status, synSuccess) << "synNodeCreate for layer1_0_conv2 failed!";

    _graphCompile();

    _destroyGraphHandle();
}
