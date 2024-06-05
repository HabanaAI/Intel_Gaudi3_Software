#include "test_recipe_addf32.hpp"

#include "synapse_api.h"
#include "../infra/test_types.hpp"
#include "test_utils.h"
#include "test_tensors_container.hpp"

#include <vector>

TestRecipeAddf32::TestRecipeAddf32(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeAddf32>(eagerMode ? "eager" : "graph", sizes),
                 deviceType,
                 2 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 eagerMode),
  m_sizes(sizes)
{
    // Init m_launchInfoInputs
    m_tensorInfoVecInputs[0].m_dimsAmount  = sizes.size();
    m_tensorInfoVecInputs[0].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[0].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[0].m_tensorSize  = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<> {}) * 4;
    m_tensorInfoVecInputs[0].m_tensorName  = "input1";
    std::copy(sizes.begin(), sizes.end(), m_tensorInfoVecInputs[0].m_tensorDimsSize);

    m_tensorInfoVecInputs[1]              = m_tensorInfoVecInputs[0];
    m_tensorInfoVecInputs[1].m_tensorName = "input2";
    std::copy(sizes.begin(), sizes.end(), m_tensorInfoVecInputs[1].m_tensorDimsSize);

    // Init m_launchInfoOutputs
    m_tensorInfoVecOutputs[0].m_dimsAmount  = sizes.size();
    m_tensorInfoVecOutputs[0].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[0].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[0].m_tensorSize  = m_tensorInfoVecInputs[0].m_tensorSize;
    m_tensorInfoVecOutputs[0].m_tensorName  = "output";
    std::copy(sizes.begin(), sizes.end(), m_tensorInfoVecOutputs[0].m_tensorDimsSize);
}

void TestRecipeAddf32::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    const float* hostDataIn1 = rLaunchTensorMemory.getInputHostBuffer<const float>(0);
    const float* hostDataIn2 = rLaunchTensorMemory.getInputHostBuffer<const float>(1);
    const float* hostDataOut = rLaunchTensorMemory.getOutputHostBuffer<const float>(0);

    const size_t refOutputLen =
        m_tensorInfoVecOutputs[0].m_tensorSize / dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);

    for (size_t i = 0; i < refOutputLen; ++i)
    {
        float expected = hostDataIn1[i] + hostDataIn2[i];
        float actual   = hostDataOut[i];

        EXPECT_EQ(expected, actual) << "result missmatch on index: " << i << ", expected: " << expected
                                    << ", actual: " << actual << ", in1: " << hostDataIn1[i]
                                    << ", in2: " << hostDataIn2[i];
    }
}

void TestRecipeAddf32::_graphCreation()
{
    synStatus status(synSuccess);

    // create layer1_0_conv1_output tensor
    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecInputs[0],
                         true /* isPersist */,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);

    createTrainingTensor(m_inputTensorsContainer,
                         1 /* tensorIndex */,
                         m_tensorInfoVecInputs[1],
                         true /* isPersist */,
                         m_tensorInfoVecInputs[1].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);

    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecOutputs[0],
                         true /* isPersist */,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);

    status = synNodeCreate(m_graphHandle,
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           nullptr,
                           0,
                           "add_f32",
                           "add",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create addf32 node";
}
