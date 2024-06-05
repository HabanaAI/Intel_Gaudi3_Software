#include "test_recipe_dma_gemm.hpp"
#include "synapse_api.h"
#include "node_factory.h"
#include "test_tensors_container.hpp"
#include "../infra/test_types.hpp"
#include "../utils/cpu_calculator.h"

TestRecipeDmaGemm::TestRecipeDmaGemm(synDeviceType deviceType, TSize hDim, TSize bDim)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeDmaGemm>(std::array {hDim, bDim}),
                 deviceType,
                 2 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 2 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    m_matrixSize = hDim;

    const unsigned numOfDims                          = 4;
    const TSize    N                                  = 1;
    const TSize    H                                  = hDim;
    const TSize    W                                  = 1;
    const TSize    B                                  = bDim;
    const TSize    tensorDimSizes[SYN_MAX_TENSOR_DIM] = {N, W, H, B};
    const uint64_t tensorSizeInElements               = N * W * H * B;

    // Init m_tensorInfoVecInputs
    // Tensor-0 - DMA input
    unsigned tensorIndex                             = 0;  // Per type(input / output)
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    m_tensorInfoVecInputs[tensorIndex].m_tensorName = "DMA_in";
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Tensor-1 - Convolution input
    tensorIndex++;
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    m_tensorInfoVecInputs[tensorIndex].m_tensorName = "Conv_operand_b";
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    // Tensor-0 - DMA output
    tensorIndex                                       = 0;  // Per type(input / output)
    m_tensorInfoVecOutputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);
    m_tensorInfoVecOutputs[tensorIndex].m_tensorName = "DMA_out";
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);

    // Tensor-1 - Convolution output
    tensorIndex++;
    m_tensorInfoVecOutputs[tensorIndex].m_dimsAmount = numOfDims;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecOutputs[tensorIndex].m_dataType   = syn_type_single;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);
    m_tensorInfoVecOutputs[tensorIndex].m_tensorName = "Conv_result";
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
}

void TestRecipeDmaGemm::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    const float* hostDataIn1 = rLaunchTensorMemory.getInputHostBuffer<const float>(0);   // DMA input
    const float* hostDataIn2 = rLaunchTensorMemory.getInputHostBuffer<const float>(1);   // Conv input
    const float* hostDataOut = rLaunchTensorMemory.getOutputHostBuffer<const float>(0);  // Conv output

    const size_t refOutputLen =
        m_tensorInfoVecOutputs[1].m_tensorSize / dataTypeSizeInBytes(m_tensorInfoVecOutputs[1].m_dataType);

    for (size_t index = 0; index < refOutputLen; index++)
    {
        float expectedResult = *hostDataIn1 * *hostDataIn2 * m_matrixSize;

        synTensorDescriptor          aDesc      = synTensorDescriptor(syn_type_single,
                                                        SYN_MAX_TENSOR_DIM,
                                                        (unsigned*)(getTensorInfoOutput(0)->m_tensorDimsSize),
                                                        nullptr);
        synTensorDescriptor          bDesc      = synTensorDescriptor(syn_type_single,
                                                        SYN_MAX_TENSOR_DIM,
                                                        (unsigned*)(getTensorInfoInput(1)->m_tensorDimsSize),
                                                        nullptr);
        synTensorDescriptor          cDesc      = synTensorDescriptor(syn_type_single,
                                                        SYN_MAX_TENSOR_DIM,
                                                        (unsigned*)(getTensorInfoOutput(1)->m_tensorDimsSize),
                                                        nullptr);
        const synConvolution3DParams convParams = {};
        CoordArray                   outIndex   = {};
        ERepefenceOp                 op         = REFERENCE_OP_FWD;

        bool mmeOpRes = checkMmeOp(aDesc,
                                   (char*)hostDataIn1,
                                   bDesc,
                                   (char*)hostDataIn2,
                                   cDesc,
                                   (char*)hostDataOut,
                                   convParams,
                                   op,
                                   outIndex,
                                   m_deviceType,
                                   &expectedResult);

        ASSERT_EQ(mmeOpRes, true) << "Mismatch at index " << index << " Expected: " << expectedResult
                                  << " Result: " << *hostDataOut << " operand-A " << *hostDataIn1 << " operand-B "
                                  << *hostDataIn2;

        hostDataIn1++;
        hostDataIn2++;
        hostDataOut++;
    }
}

void TestRecipeDmaGemm::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors creation
    //
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

    // Create dma node
    synTensor dmaInputTensorInfo  = m_inputTensorsContainer.tensor(0);   // DMA inpute tensor
    synTensor dmaOutputTensorInfo = m_outputTensorsContainer.tensor(0);  // DMA output tensor
    status                        = synNodeCreate(m_graphHandle,
                           &dmaInputTensorInfo,
                           &dmaOutputTensorInfo,
                           1,
                           1,
                           nullptr,
                           0,
                           NodeFactory::memcpyNodeTypeName,
                           "dma_node",
                           nullptr,
                           nullptr);
    HB_ASSERT(status == synSuccess, "Failed to DMA synNodeCreate");

    // Create gemm node
    std::vector<synTensor> gemmInputTensorInfo = {
        m_outputTensorsContainer.tensor(0),
        m_inputTensorsContainer.tensor(1)};                               // DMA output tensor, Conv input tensor
    synTensor gemmOutputTensorInfo = m_outputTensorsContainer.tensor(1);  // Conv output tensor
    status                         = synNodeCreate(m_graphHandle,
                           gemmInputTensorInfo.data(),
                           &gemmOutputTensorInfo,
                           2,
                           1,
                           nullptr,
                           0,
                           "gemm",
                           "gemmnode",
                           nullptr,
                           nullptr);
    HB_ASSERT(status == synSuccess, "Failed to GEMM synNodeCreate");
}
