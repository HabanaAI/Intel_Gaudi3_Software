#include "test_recipe_gemm_addf32.hpp"
#include "test_tensors_container.hpp"
#include "node_factory.h"
#include "synapse_api.h"
#include "../infra/test_types.hpp"
#include "../../utils/cpu_calculator.h"
#include "test_utils.h"
#include <vector>

TestRecipeGemmAddf32::TestRecipeGemmAddf32(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeGemmAddf32>(eagerMode ? "eager" : "graph", sizes),
                 deviceType,
                 2 /* inputTensorsAmount */,
                 1 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 eagerMode /* eagerMode */),
  m_sizes(sizes)
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
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "in1";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);

    std::copy(m_sizes.begin(), m_sizes.end(), m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Tensor-1
    tensorIndex++;
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "in2";
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
    m_tensorInfoVecOutputs[tensorIndex].m_tensorName  = "out";
    m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);

    std::copy(m_sizes.begin(), m_sizes.end(), m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
}

void TestRecipeGemmAddf32::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
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

    for (size_t i = 0; i < refOutputLen; ++i)
    {
        float expected = refOutput[i] + refOutput[i];
        ASSERT_TRUE(float_eq(expected, hostDataOut[i])) << "Result validation failed";
    }
}

void TestRecipeGemmAddf32::_graphCreation()
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

    TensorInfo tensor_info;
    tensor_info              = m_tensorInfoVecInputs[0];
    tensor_info.m_tensorName = "outputTpc";

    createTrainingTensor(m_innerTensorsContainer,
                         0 /* tensorIndex */,
                         tensor_info,
                         false /* isPersist */,
                         tensor_info.m_tensorName,
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
                           m_innerTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_innerTensorsContainer.size(),
                           nullptr,
                           0,
                           "gemm",
                           "gemm",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create gemm node";

    std::vector<synTensor> inner_tensors_handles = {m_innerTensorsContainer.tensor(0),
                                                    m_innerTensorsContainer.tensor(0)};
    status                                       = synNodeCreate(m_graphHandle,
                           inner_tensors_handles.data(),
                           m_outputTensorsContainer.tensors(),
                           inner_tensors_handles.size(),
                           m_outputTensorsContainer.size(),
                           nullptr,
                           0,
                           "add_f32",
                           "add",
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create gemm node";
}