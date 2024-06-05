#include "test_recipe_hcl.hpp"
#include "test_tensors_container.hpp"
#include "../infra/test_types.hpp"
#include "node_factory.h"
#include "synapse_api.h"

TestRecipeHcl::TestRecipeHcl(synDeviceType deviceType, bool isSfgGraph)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeHcl>(isSfgGraph ? "sfg" : ""),
                 deviceType,
                 1 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 2 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    const unsigned numOfDims                          = 4;
    const TSize    N                                  = 16;
    const TSize    H                                  = 16;
    const TSize    W                                  = 16;
    const TSize    B                                  = 16;
    const TSize    tensorDimSizes[SYN_MAX_TENSOR_DIM] = {N, W, H, B};
    const uint64_t tensorSizeInElements               = N * W * H * B;

    // Inputs:
    //
    // Tensor-0
    m_tensorInfoVecInputs[0].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[0].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[0].m_tensorName  = "In1";
    m_tensorInfoVecInputs[0].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[0].m_dataType);
    //
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[0].m_tensorDimsSize);

    // Outputs:
    //
    // Tensor-0
    m_tensorInfoVecOutputs[0].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[0].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[0].m_tensorName  = "Out1";
    m_tensorInfoVecOutputs[0].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);
    if (isSfgGraph)
    {
        m_tensorInfoVecOutputs[0].m_isSfg = true;
        m_numOfExternalTensors++;
    }
    //
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[0].m_tensorDimsSize);
    //
    // Tensor-1
    m_tensorInfoVecOutputs[1].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[1].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[1].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[1].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[1].m_tensorName  = "Out2";
    m_tensorInfoVecOutputs[1].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[1].m_dataType);
    //
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[1].m_tensorDimsSize);
}

void TestRecipeHcl::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors creation
    for (unsigned i = 0; i < m_inputTensorsContainer.size(); i++)
    {
        createTrainingTensor(m_inputTensorsContainer,
                             i /* tensorIndex */,
                             m_tensorInfoVecInputs[i],
                             true /* isPersist */,
                             m_tensorInfoVecInputs[i].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* hostBuffer */);
    }

    for (unsigned i = 0; i < m_outputTensorsContainer.size(); i++)
    {
        createTrainingTensor(m_outputTensorsContainer,
                             i /* tensorIndex */,
                             m_tensorInfoVecOutputs[i],
                             true /* isPersist */,
                             m_tensorInfoVecOutputs[i].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* hostBuffer */);
    }

    // Create add_bwd_f32 node
    status = synNodeCreate(m_graphHandle,
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           nullptr,
                           0,
                           "add_bwd_f32",
                           "addBwdNode",  // guid and node name
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}

void TestRecipeHcl::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    for (uint64_t i = 0; i < 2; i++)
    {
        const float* hostDataIn  = rLaunchTensorMemory.getInputHostBuffer<const float>(0);
        const float* hostDataOut = rLaunchTensorMemory.getOutputHostBuffer<const float>(i);

        const size_t refOutputLen =
            m_tensorInfoVecOutputs[i].m_tensorSize / dataTypeSizeInBytes(m_tensorInfoVecOutputs[i].m_dataType);

        for (size_t index = 0; index < refOutputLen; index++)
        {
            ASSERT_EQ(hostDataIn[index], hostDataOut[index]) << "Result validation failed";
        }
    }
}