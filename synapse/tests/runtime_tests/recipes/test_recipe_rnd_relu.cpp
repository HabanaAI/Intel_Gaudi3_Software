#include "test_recipe_rnd_relu.hpp"
#include "synapse_api.h"
#include "../infra/test_types.hpp"
#include "perf_lib_layer_params.h"
#include "test_tensors_container.hpp"
#include <vector>

TestRecipeRndRelu::TestRecipeRndRelu(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeRndRelu>(),
                 deviceType,
                 1 /* inputTensorsAmount   */,
                 0 /* innerTensorsAmount   */,
                 1 /* outputTensorsAmount  */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    const uint32_t dimSize                         = 16;
    const uint32_t numOfDims                       = 4;
    const uint32_t vecNumItems                     = dimSize * dimSize * dimSize * dimSize;
    const uint32_t vecSize                         = vecNumItems * sizeof(float);
    unsigned       sizesTensor[SYN_MAX_TENSOR_DIM] = {dimSize, dimSize, dimSize, dimSize};

    // Init m_tensorInfoVecInputs
    m_tensorInfoVecInputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecInputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_tensorSize = vecSize;
    m_tensorInfoVecInputs[0].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[0].m_tensorName = "middle";
    std::copy(sizesTensor, sizesTensor + numOfDims, m_tensorInfoVecInputs[0].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    m_tensorInfoVecOutputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecOutputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_tensorSize = vecSize;
    m_tensorInfoVecOutputs[0].m_dataType   = syn_type_single;
    m_tensorInfoVecOutputs[0].m_tensorName = "output";
    std::copy(sizesTensor, sizesTensor + numOfDims, m_tensorInfoVecOutputs[0].m_tensorDimsSize);
}

void TestRecipeRndRelu::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    return;
}

void TestRecipeRndRelu::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors
    //
    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         4U,
                         syn_type_single,
                         m_tensorInfoVecInputs[0].m_tensorDimsSize,
                         false,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr,
                         false,
                         0,
                         nullptr,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         4U,
                         syn_type_single,
                         m_tensorInfoVecOutputs[0].m_tensorDimsSize,
                         true,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr,
                         false,
                         0,
                         nullptr,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    ns_RandomUniform::Params params;
    params.high = 100;
    params.low  = -100;
    params.seed = std::rand();

    // Create DMA node
    status = synNodeCreate(m_graphHandle,
                           nullptr,
                           m_inputTensorsContainer.tensors(),
                           0,
                           1,
                           &params,
                           sizeof(params),
                           "random_uniform_fwd_f32",
                           "random_f32",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";

    status = synNodeCreate(m_graphHandle,
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           nullptr,
                           0,
                           "relu_fwd_f32",
                           "relu_f32",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}