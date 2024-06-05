#include "test_recipe_gemm.hpp"
#include "test_tensors_container.hpp"
#include "node_factory.h"
#include "synapse_api.h"
#include "../infra/test_types.hpp"
#include "../utils/cpu_calculator.h"
#include <vector>

TestRecipeGemm::TestRecipeGemm(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeGemm>(eagerMode ? "eager" : "graph", sizes),
                 deviceType,
                 2 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 eagerMode),
  m_sizes(sizes),
  m_uniqueNodeId(0)
{
    const unsigned numOfDims            = m_sizes.size();
    const uint64_t tensorSizeInElements = std::accumulate(m_sizes.begin(), m_sizes.end(), 1, std::multiplies<> {});

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

    std::copy(m_sizes.begin(), m_sizes.end(), m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Tensor-1
    tensorIndex++;
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "In2";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);

    std::copy(m_sizes.begin(), m_sizes.end(), m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

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

    std::copy(m_sizes.begin(), m_sizes.end(), m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
}

void TestRecipeGemm::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    // Tensor-Descriptor
    synTensorDescriptor aDesc =
        getTensorDescriptor(m_tensorInfoVecInputs[0].m_dataType,
                            m_tensorInfoVecInputs[0].m_tensorDimsSize,
                            m_tensorInfoVecInputs[0].m_dimsAmount,
                            m_tensorInfoVecInputs[0].m_tensorName.c_str(),
                            nullptr /* strides */,
                            (void*)rLaunchTensorMemory.m_tensorInfoVecInputs[0].getTestDeviceBuffer().getBuffer(),
                            false /* isQuantized */,
                            m_tensorInfoVecInputs[0].m_tensorMinDimsSize,
                            m_tensorInfoVecInputs[0].m_tensorType);

    synTensorDescriptor bDesc =
        getTensorDescriptor(m_tensorInfoVecInputs[1].m_dataType,
                            m_tensorInfoVecInputs[1].m_tensorDimsSize,
                            m_tensorInfoVecInputs[1].m_dimsAmount,
                            m_tensorInfoVecInputs[1].m_tensorName.c_str(),
                            nullptr /* strides */,
                            (void*)rLaunchTensorMemory.m_tensorInfoVecInputs[1].getTestDeviceBuffer().getBuffer(),
                            false /* isQuantized */,
                            m_tensorInfoVecInputs[1].m_tensorMinDimsSize,
                            m_tensorInfoVecInputs[1].m_tensorType);

    synTensorDescriptor cDesc =
        getTensorDescriptor(m_tensorInfoVecOutputs[0].m_dataType,
                            m_tensorInfoVecOutputs[0].m_tensorDimsSize,
                            m_tensorInfoVecOutputs[0].m_dimsAmount,
                            m_tensorInfoVecOutputs[0].m_tensorName.c_str(),
                            nullptr /* strides */,
                            (void*)rLaunchTensorMemory.m_tensorInfoVecOutputs[0].getTestDeviceBuffer().getBuffer(),
                            false /* isQuantized */,
                            m_tensorInfoVecOutputs[0].m_tensorMinDimsSize,
                            m_tensorInfoVecOutputs[0].m_tensorType);

    const float* hostDataIn1 = rLaunchTensorMemory.getInputHostBuffer<const float>(0);
    const float* hostDataIn2 = rLaunchTensorMemory.getInputHostBuffer<const float>(1);
    const float* hostDataOut = rLaunchTensorMemory.getOutputHostBuffer<const float>(0);

    const size_t refOutputLen =
        m_tensorInfoVecOutputs[0].m_tensorSize / dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);
    std::vector<float> refOutput(refOutputLen, 0);

    synGEMMParams params;
    params.transpose_a = false;
    params.transpose_b = false;

    calculateGemm(aDesc,
                  (char*)hostDataIn1,
                  bDesc,
                  (char*)hostDataIn2,
                  cDesc,
                  (char*)refOutput.data(),
                  params,
                  ERepefenceOp::REFERENCE_OP_FWD,
                  m_deviceType);

    for (size_t index = 0; index < refOutputLen; index++)
    {
        ASSERT_NEAR(refOutput[index], hostDataOut[index], 0.001) << "Result validation failed";
    }
}

void TestRecipeGemm::_graphCreation()
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
    //
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

    // Create GEMM node
    status = synNodeCreateWithId(m_graphHandle,
                                 m_inputTensorsContainer.tensors(),
                                 m_outputTensorsContainer.tensors(),
                                 m_inputTensorsContainer.size(),
                                 m_outputTensorsContainer.size(),
                                 nullptr,
                                 0,
                                 "gemm",      // GUID
                                 "gemmNode",  // Node name
                                 &m_uniqueNodeId,
                                 nullptr,
                                 nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}