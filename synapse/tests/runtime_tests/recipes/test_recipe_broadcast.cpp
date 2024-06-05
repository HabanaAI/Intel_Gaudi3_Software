#include "test_recipe_broadcast.hpp"
#include "../infra/test_types.hpp"
#include "test_utils.h"
#include "test_tensors_container.hpp"

TestRecipeBroadcast::TestRecipeBroadcast(synDeviceType deviceType,
                                         TestSizes     inSize,
                                         TestSizes     outSize,
                                         uint32_t      dims,
                                         bool          isFcdMultipleDimsBroadcast)
: TestRecipeBase(
      makeUniqueRecipeName<TestRecipeBroadcast>(isFcdMultipleDimsBroadcast ? "fcd" : "regular", inSize, outSize),
      deviceType,
      1 /* inputTensorsAmount */,
      0 /* innerTensorsAmount */,
      1 /* outputTensorsAmount */,
      0 /* uniqueSectionsAmount */,
      false /* eagerMode */)
{
    uint64_t numOfInputElements = 1;
    for (int i = 0; i < dims; i++)
    {
        numOfInputElements *= inSize[i];
    }

    uint64_t numOfOutputElements = 1;
    for (int i = 0; i < dims; i++)
    {
        numOfOutputElements *= outSize[i];
    }

    // Init m_tensorInfoVecInputs
    m_tensorInfoVecInputs[0].m_dimsAmount = dims;
    m_tensorInfoVecInputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_dataType   = syn_type_bf16;
    m_tensorInfoVecInputs[0].m_tensorSize =
        numOfInputElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[0].m_dataType);
    m_tensorInfoVecInputs[0].m_tensorName = "input";
    //
    std::copy(inSize.begin(), inSize.end(), m_tensorInfoVecInputs[0].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    m_tensorInfoVecOutputs[0].m_dimsAmount = dims;
    m_tensorInfoVecOutputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_dataType   = syn_type_bf16;
    m_tensorInfoVecOutputs[0].m_tensorSize =
        numOfOutputElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);
    m_tensorInfoVecOutputs[0].m_tensorName = "output";
    //
    std::copy(outSize.begin(), outSize.end(), m_tensorInfoVecOutputs[0].m_tensorDimsSize);

    m_isFcdMultipleDimsBroadcast = isFcdMultipleDimsBroadcast;
}

void TestRecipeBroadcast::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    ASSERT_EQ(m_tensorInfoVecInputs.size(), 1) << "Invalid amount of Inputs";
    ASSERT_EQ(m_tensorInfoVecOutputs.size(), 1) << "Invalid amount of Outputs";

    uint64_t outputElementsAmount =
        m_tensorInfoVecOutputs[0].m_tensorSize / sizeof(m_tensorInfoVecOutputs[0].m_dataType);

    uint16_t* pInputBuffer  = (uint16_t*)rLaunchTensorMemory.m_tensorInfoVecInputs[0].getTestHostBuffer().getBuffer();
    uint16_t* pOutputBuffer = (uint16_t*)rLaunchTensorMemory.m_tensorInfoVecOutputs[0].getTestHostBuffer().getBuffer();

    if (m_isFcdMultipleDimsBroadcast)
    {
        uint16_t braodcastSize = m_tensorInfoVecOutputs[0].m_tensorDimsSize[0];

        ASSERT_EQ(m_tensorInfoVecInputs[0].m_tensorSize, m_tensorInfoVecOutputs[0].m_tensorSize / braodcastSize)
            << "Tensor-size mismatch";

        for (unsigned i = 0; i < outputElementsAmount; i++)
        {
            ASSERT_EQ(pOutputBuffer[i], pInputBuffer[i / braodcastSize]) << "Wrong output - FCD";
        }
    }
    else
    {
        for (unsigned i = 0; i < outputElementsAmount; i++)
        {
            ASSERT_EQ(pOutputBuffer[i], pInputBuffer[0]) << "Wrong output";
        }
    }
}

void TestRecipeBroadcast::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors
    //
    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecInputs[0],
                         true /* isPersist */,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);
    //
    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecOutputs[0],
                         true /* isPersist */,
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
                           "broadcast",
                           "broadcast",
                           // input/output layouts
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}