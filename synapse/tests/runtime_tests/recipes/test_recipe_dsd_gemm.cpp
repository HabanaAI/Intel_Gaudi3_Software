#include "test_recipe_dsd_gemm.hpp"
#include "test_tensors_container.hpp"
#include "node_factory.h"
#include "scoped_configuration_change.h"
#include "synapse_api.h"
#include "../infra/test_types.hpp"
#include "../utils/cpu_calculator.h"
#include "test_utils.h"
#include <vector>

TestRecipeDsdGemm::TestRecipeDsdGemm(synDeviceType        deviceType,
                                     bool                 isDynamic,
                                     bool                 isSharedInputSection,
                                     std::array<TSize, 2> op1Max,
                                     std::array<TSize, 2> op1Min,
                                     std::array<TSize, 2> op2Max,
                                     std::array<TSize, 2> op2Min,
                                     std::array<TSize, 2> outMax,
                                     std::array<TSize, 2> outMin)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeDsdGemm>(GCFG_ENABLE_PROFILER.value() ? "profiler" : "",
                                                         isDynamic ? "op1Dynamic" : "",
                                                         isSharedInputSection ? "sharedInputSection" : "",
                                                         op1Max,
                                                         op1Min,
                                                         op2Max,
                                                         op2Min,
                                                         outMax,
                                                         outMin),
                 deviceType,
                 2 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    unsigned numOfDims = 2;

    // OP_1: [256, 128],
    // OP_2: [128 -> 256, 256]
    //  => Output: [128 -> 256, 128]
    synTensorType op1TensorType = DATA_TENSOR;
    if (isDynamic)
    {
        op1TensorType = DATA_TENSOR_DYNAMIC;
    }

    /* dims = [W, H] */
    TSize input1TensorDimSizes[SYN_MAX_TENSOR_DIM] = {op1Max[0], op1Max[1]};
    TSize input1MinDimSizes[SYN_MAX_TENSOR_DIM]    = {op1Min[0], op1Min[1]};
    TSize input2TensorDimSizes[SYN_MAX_TENSOR_DIM] = {op2Max[0], op2Max[1]};
    TSize input2MinDimSizes[SYN_MAX_TENSOR_DIM]    = {op2Min[0], op2Min[1]};
    TSize outputTensorDimSizes[SYN_MAX_TENSOR_DIM] = {outMax[0], outMax[1]};
    TSize outputMinDimSizes[SYN_MAX_TENSOR_DIM]    = {outMin[0], outMin[1]};

    unsigned tensorIndex = 0;
    //
    // Init m_tensorInfoVecInputs
    // Tensor-0
    tensorIndex                                      = 0;  // Per type(input / output)
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = op1TensorType;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "In1";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        input1TensorDimSizes[0] * input1TensorDimSizes[1] *
        dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    std::copy(input1TensorDimSizes,
              input1TensorDimSizes + numOfDims,
              m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);
    std::copy(input1MinDimSizes, input1MinDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorMinDimsSize);
    //
    // Tensor-1
    tensorIndex++;
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR_DYNAMIC;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "In2";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        input2TensorDimSizes[0] * input2TensorDimSizes[1] *
        dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    std::copy(input2TensorDimSizes,
              input2TensorDimSizes + numOfDims,
              m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);
    std::copy(input2MinDimSizes, input2MinDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorMinDimsSize);
    m_tensorInfoVecInputs[tensorIndex].m_sectionOffset =
        isSharedInputSection ? m_tensorInfoVecInputs[tensorIndex - 1].m_tensorSize : 0;

    // Init m_tensorInfoVecOutputs
    // Tensor-0
    tensorIndex                                       = 0;  // Per type(input / output)
    m_tensorInfoVecOutputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorType  = DATA_TENSOR_DYNAMIC;
    m_tensorInfoVecOutputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorName  = "Out";
    m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
        outputTensorDimSizes[0] * outputTensorDimSizes[1] *
        dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);
    std::copy(outputTensorDimSizes,
              outputTensorDimSizes + numOfDims,
              m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
    std::copy(outputMinDimSizes,
              outputMinDimSizes + numOfDims,
              m_tensorInfoVecOutputs[tensorIndex].m_tensorMinDimsSize);
}

void TestRecipeDsdGemm::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    size_t refOutputLen = m_op1ActualSize * m_op2ActualSize;
    ASSERT_TRUE(refOutputLen != 0) << "m_op1ActualSize (" << m_op1ActualSize << ") or m_op2ActualSize ("
                                   << m_op2ActualSize << ") is equal to 0";

    std::unique_ptr<float[]> refOutput(new float[refOutputLen]);

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

    aDesc.m_sizes[1] = m_op1ActualSize;
    bDesc.m_sizes[0] = m_op2ActualSize;
    cDesc.m_sizes[0] = m_op2ActualSize;
    cDesc.m_sizes[1] = m_op1ActualSize;

    calculateGemm(aDesc,
                  (char*)hostDataIn1,
                  bDesc,
                  (char*)hostDataIn2,
                  cDesc,
                  (char*)refOutput.get(),
                  m_gemmParams,
                  ERepefenceOp::REFERENCE_OP_FWD,
                  m_deviceType);

    for (size_t i = 0; i < refOutputLen; i++)
    {
        ASSERT_TRUE(float_eq(refOutput.get()[i], hostDataOut[i], 0.01))
            << "Result validation failed"
            << ". Index " << i << ", Reference " << refOutput.get()[i] << ", Actual " << hostDataOut[i];
    }
}

void TestRecipeDsdGemm::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors creation
    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecInputs[0],
                         true /* isPersist */,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);

    synSectionHandle firstInputSection = m_inputTensorsContainer.section(0);
    createTrainingTensor(m_inputTensorsContainer,
                         1 /* tensorIndex */,
                         m_tensorInfoVecInputs[1],
                         true /* isPersist */,
                         m_tensorInfoVecInputs[1].m_tensorName,
                         m_graphHandle,
                         m_tensorInfoVecInputs[1].m_sectionOffset != 0 ? &firstInputSection : nullptr,
                         nullptr /* hostBuffer */);

    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecOutputs[0],
                         true /* isPersist */,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);

    // Create GEMM node
    status = synNodeCreate(m_graphHandle,
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           &m_gemmParams,
                           sizeof(m_gemmParams),
                           NodeFactory::gemmNodeTypeName,
                           "" /* nodeMame       */,
                           nullptr /* inputLayouts   */,
                           nullptr /* outputLayouts  */);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}