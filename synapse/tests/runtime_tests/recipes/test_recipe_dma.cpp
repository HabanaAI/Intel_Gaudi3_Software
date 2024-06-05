#include "test_recipe_dma.hpp"
#include "test_tensors_container.hpp"
#include "node_factory.h"
#include "synapse_api.h"
#include "../infra/test_types.hpp"

TestRecipeDma::TestRecipeDma(synDeviceType deviceType,
                             TSize         zDim,
                             TSize         wDim,
                             float         initValue,
                             bool          isConstTensor,
                             synDataType   dataType,
                             unsigned      threadIndex,
                             unsigned      iterationIndex)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeDma>(std::array {zDim, wDim}, dataType),
                 deviceType,
                 1 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    m_inputName  = std::string("input_") + std::to_string(threadIndex) + '_' + std::to_string(iterationIndex);
    m_outputName = std::string("output_") + std::to_string(threadIndex) + '_' + std::to_string(iterationIndex);

    unsigned    numOfDims = 4;
    const TSize Z         = zDim;
    const TSize W         = wDim;
    const TSize H         = 1U;
    const TSize batch     = 1U;

    TSize    tensorDimSizes[SYN_MAX_TENSOR_DIM] = {Z, W, H, batch};
    uint64_t tensorSize                         = Z * W * H * batch * dataTypeSizeInBytes(dataType);

    m_constInputTensorInitVal = initValue;

    // Init m_tensorInfoVecInputs
    m_tensorInfoVecInputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecInputs[0].m_dataType   = dataType;
    m_tensorInfoVecInputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_tensorSize = tensorSize;
    m_tensorInfoVecInputs[0].m_isConst    = isConstTensor;
    m_tensorInfoVecInputs[0].m_sectionType =
        isConstTensor ? TestSectionType::CONST_TENSOR_SECTION : TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[0].m_tensorName = m_inputName;
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[0].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    m_tensorInfoVecOutputs[0].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[0].m_dataType    = dataType;
    m_tensorInfoVecOutputs[0].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_tensorSize  = tensorSize;
    m_tensorInfoVecOutputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[0].m_tensorName  = m_outputName;
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[0].m_tensorDimsSize);
}

void TestRecipeDma::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    ASSERT_EQ(m_tensorInfoVecInputs.size(), 1) << "Invalid amount of Inputs";
    ASSERT_EQ(m_tensorInfoVecOutputs.size(), 1) << "Invalid amount of Outputs";

    const uint8_t* hostDataIn  = rLaunchTensorMemory.getInputHostBuffer<const uint8_t>(0);
    const uint8_t* hostDataOut = rLaunchTensorMemory.getOutputHostBuffer<const uint8_t>(0);

    const size_t refOutputLen = m_tensorInfoVecOutputs[0].m_tensorSize;

    bool isConstTensor = m_tensorInfoVecInputs[0].m_isConst;

    for (size_t i = 0; i < refOutputLen; i++)
    {
        ASSERT_EQ(isConstTensor ? ((uint8_t*)(&m_constInputTensorInitVal))[i % 4] : hostDataIn[i], hostDataOut[i])
            << "Result validation failed";
    }
}

void TestRecipeDma::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors
    bool   isConstTensor = m_tensorInfoVecInputs[0].m_isConst;
    float* hostBuffer    = nullptr;

    if (isConstTensor)
    {
        unsigned zDim = m_tensorInfoVecInputs[0].m_tensorDimsSize[0];

        hostBuffer = new float[zDim];
        for (int j = 0; j < zDim; j++)
        {
            hostBuffer[j] = m_constInputTensorInitVal;
        }
    }

    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecInputs[0],
                         true,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         isConstTensor ? (void*)hostBuffer : nullptr);

    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecOutputs[0],
                         true,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);

    // Create DMA node
    status = synNodeCreate(m_graphHandle,
                           // input/output tensor vectors
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           // input/output tensor vector sizes
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

    delete[] hostBuffer;
}