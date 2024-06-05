#include "test_recipe_dsd_dma.hpp"
#include "test_tensors_container.hpp"
#include "node_factory.h"
#include "synapse_api.h"
#include "../infra/test_types.hpp"

TestRecipeDsdDma::TestRecipeDsdDma(synDeviceType        deviceType,
                                   std::array<TSize, 4> inputMax,
                                   std::array<TSize, 4> inputMin,
                                   std::array<TSize, 4> outputMax,
                                   std::array<TSize, 4> outputMin)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeDsdDma>(inputMax, inputMin, outputMax, outputMin),
                 deviceType,
                 1 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    unsigned numOfDims = 4;
    /* dims = [W, H] */
    unsigned inputTensorDimSizes[SYN_MAX_TENSOR_DIM]  = {inputMax[0], inputMax[1], inputMax[2], inputMax[3]};
    unsigned inputMinDimSizes[SYN_MAX_TENSOR_DIM]     = {inputMin[0], inputMin[1], inputMin[2], inputMin[3]};
    unsigned outputTensorDimSizes[SYN_MAX_TENSOR_DIM] = {outputMax[0], outputMax[1], outputMax[2], outputMax[3]};
    unsigned outputMinDimSizes[SYN_MAX_TENSOR_DIM]    = {outputMin[0], outputMin[1], outputMin[2], outputMin[3]};

    // Init m_tensorInfoVecInputs
    // Input tensor-0
    unsigned tensorIndex                             = 0;  // Per type(input / output)
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR_DYNAMIC;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "input";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        inputTensorDimSizes[0] * inputTensorDimSizes[1] * inputTensorDimSizes[2] * inputTensorDimSizes[3] *
        dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    std::copy(inputTensorDimSizes,
              inputTensorDimSizes + numOfDims,
              m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);
    std::copy(inputMinDimSizes, inputMinDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorMinDimsSize);

    // Init m_tensorInfoVecOutputs
    // Output tensor-0
    tensorIndex                                       = 0;  // Per type(input / output)
    m_tensorInfoVecOutputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorType  = DATA_TENSOR_DYNAMIC;
    m_tensorInfoVecOutputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorName  = "output";
    m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
        outputTensorDimSizes[0] * outputTensorDimSizes[1] * outputTensorDimSizes[1] * outputTensorDimSizes[1] *
        dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);
    std::copy(outputTensorDimSizes,
              outputTensorDimSizes + numOfDims,
              m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
    std::copy(outputMinDimSizes,
              outputMinDimSizes + numOfDims,
              m_tensorInfoVecOutputs[tensorIndex].m_tensorMinDimsSize);
}

void TestRecipeDsdDma::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    float* inBuffer  = (float*)(rLaunchTensorMemory.m_tensorInfoVecInputs[0].getTestHostBuffer().getBuffer());
    float* outBuffer = (float*)(rLaunchTensorMemory.m_tensorInfoVecOutputs[0].getTestHostBuffer().getBuffer());

    uint64_t tensorBatchSizeElements = 1;
    for (int i = 0; i < m_tensorInfoVecInputs[0].m_dimsAmount - 1; i++)
    {
        tensorBatchSizeElements *= m_tensorInfoVecInputs[0].m_tensorMinDimsSize[i];
    }

    const uint64_t tensorSizeElements = tensorBatchSizeElements * m_op1ActualSize /* actualBatch */;

    // Test by the actual batch size.
    for (int i = 0; i < tensorSizeElements; i++)
    {
        ASSERT_EQ(outBuffer[i], inBuffer[i])
            << "outBuffer: " << outBuffer[i] << " inBuffer: " << inBuffer[i] << " index: " << i;
    }
}

void TestRecipeDsdDma::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors
    createTrainingTensor(m_inputTensorsContainer,
                         0,
                         m_tensorInfoVecInputs[0],
                         true,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr,
                         nullptr);

    createTrainingTensor(m_outputTensorsContainer,
                         0,
                         m_tensorInfoVecOutputs[0],
                         true,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr,
                         nullptr);

    // Create DMA node
    status = synNodeCreate(m_graphHandle,
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           // user params
                           nullptr,
                           0,
                           // guid and node name
                           NodeFactory::memcpyNodeTypeName,
                           "dma_node",
                           // input/output layouts
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}