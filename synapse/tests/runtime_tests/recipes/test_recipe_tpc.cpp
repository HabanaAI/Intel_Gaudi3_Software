#include "test_recipe_tpc.hpp"

#include "node_factory.h"

#include "synapse_api.h"
#include "test_tensor_init.hpp"
#include "../infra/test_types.hpp"

#include "test_tensors_container.hpp"
#include "test_utils.h"

#include <vector>

TestRecipeTpc::TestRecipeTpc(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeTpc>(),
                 deviceType,
                 2 /* inputTensorsAmount   */,
                 0 /* innerTensorsAmount   */,
                 1 /* outputTensorsAmount  */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    const unsigned numOfDims                          = 4;
    const unsigned N                                  = 4;
    const unsigned H                                  = 4;
    const unsigned W                                  = 4;
    const unsigned B                                  = 1;
    const TSize    tensorDimSizes[SYN_MAX_TENSOR_DIM] = {N, W, H, B};
    const uint64_t tensorSizeInElements               = N * W * H * B;

    // Init m_tensorInfoVecInputs
    // Tensor-0
    unsigned tensorIndex                             = 0;  // Per type(input / output)
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "In1";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Tensor-1
    tensorIndex++;
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "In2";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    // Tensor-0
    tensorIndex                                       = 0;  // Per type(input / output)
    m_tensorInfoVecOutputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorName  = "Out";
    m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);
    //
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
}

void TestRecipeTpc::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    const float* hostDataIn1 = rLaunchTensorMemory.getInputHostBuffer<const float>(0);
    const float* hostDataIn2 = rLaunchTensorMemory.getInputHostBuffer<const float>(1);
    const float* hostDataOut = rLaunchTensorMemory.getOutputHostBuffer<const float>(0);

    const size_t refOutputLen =
        m_tensorInfoVecOutputs[0].m_tensorSize / dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);

    for (size_t index = 0; index < refOutputLen; index++)
    {
        ASSERT_EQ((hostDataIn1[index] + hostDataIn2[index]), hostDataOut[index]) << "Result validation failed";
    }
}

void TestRecipeTpc::_graphCreation()
{
    synStatus status(synSuccess);

    for (unsigned i = 0; i < 2; i++)
    {
        createTrainingTensor(m_inputTensorsContainer,
                             i,
                             m_tensorInfoVecInputs[i],
                             true /* isPersist */,
                             m_tensorInfoVecInputs[i].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* hostBuffer */);
    }
    //
    // Single output presistent-tensor
    {
        createTrainingTensor(m_outputTensorsContainer,
                             0 /* tensorIndex */,
                             m_tensorInfoVecOutputs[0],
                             true /* isPersist */,
                             m_tensorInfoVecOutputs[0].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* hostBuffer */);
    }

    // Create add_f32 node
    status = synNodeCreate(m_graphHandle,
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           nullptr,
                           0,
                           "add_fwd_f32",
                           "addNode",  // guid and node name
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}
