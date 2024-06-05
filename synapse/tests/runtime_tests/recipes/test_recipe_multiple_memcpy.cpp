#include "test_recipe_multiple_memcpy.hpp"

#include "defs.h"

#include "synapse_api.h"

#include "test_tensors_container.hpp"

TestRecipeMultipleMemcpy::TestRecipeMultipleMemcpy(synDeviceType deviceType, unsigned numNodes)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeMultipleMemcpy>(numNodes),
                 deviceType,
                 numNodes /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    m_numNodes = numNodes;

    const uint32_t dimSize                      = 16;
    unsigned       numOfDims                    = 4;
    TSize          dimSizes[SYN_MAX_TENSOR_DIM] = {dimSize, dimSize, dimSize, dimSize};
    uint64_t       tensorSize                   = dimSize * dimSize * dimSize * dimSize * sizeof(float);

    int inputIter = 0;
    for (auto& input : m_tensorInfoVecInputs)
    {
        std::string name = inputIter == 0 ? "input" : "mid" + std::to_string(inputIter);

        input.m_dimsAmount = numOfDims;
        input.m_tensorSize = tensorSize;
        input.m_tensorName = name;
        input.m_dataType   = syn_type_float;
        std::copy(dimSizes, dimSizes + numOfDims, input.m_tensorDimsSize);
        ++inputIter;
    }

    // Init m_launchInfoOutputs
    m_tensorInfoVecOutputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecOutputs[0].m_tensorSize = tensorSize;
    m_tensorInfoVecOutputs[0].m_tensorName = "output";
    m_tensorInfoVecOutputs[0].m_dataType   = syn_type_float;
    std::copy(dimSizes, dimSizes + numOfDims, m_tensorInfoVecOutputs[0].m_tensorDimsSize);
}

void TestRecipeMultipleMemcpy::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors
    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecInputs[0].m_dimsAmount,
                         syn_type_float,
                         m_tensorInfoVecInputs[0].m_tensorDimsSize,
                         true /* isPersist */,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         false,
                         0 /* offset */,
                         nullptr,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    for (int i = 0; i < m_numNodes - 1; i++)
    {
        createTrainingTensor(m_inputTensorsContainer,
                             i + 1 /* tensorIndex */,
                             m_tensorInfoVecInputs[i + 1].m_dimsAmount,
                             syn_type_float,
                             m_tensorInfoVecInputs[i + 1].m_tensorDimsSize,
                             true /* isPersist */,
                             m_tensorInfoVecInputs[i + 1].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             false,
                             0 /* offset */,
                             nullptr,
                             DATA_TENSOR,
                             nullptr /* minTensorSize */);
    }

    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecOutputs[0].m_dimsAmount,
                         syn_type_float,
                         m_tensorInfoVecOutputs[0].m_tensorDimsSize,
                         true /* isPersist */,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         false,
                         0 /* offset */,
                         nullptr,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    for (int i = 0; i < m_numNodes - 1; i++)
    {
        // Create DMA node
        status = synNodeCreate(m_graphHandle,
                               &m_inputTensorsContainer.tensor(i),
                               &m_inputTensorsContainer.tensor(i + 1),
                               1 /* numberInputs   */,
                               1 /* numberOutputs  */,
                               nullptr /* pUserParams    */,
                               0 /* userParamsSize */,
                               "memcpy" /* nodeGuid       */,
                               "" /* nodeMame       */,
                               nullptr /* inputLayouts   */,
                               nullptr /* outputLayouts  */);
        HB_ASSERT(status == synSuccess, "Failed to synNodeCreate");
    }

    // Create DMA node
    status = synNodeCreate(m_graphHandle,
                           &m_inputTensorsContainer.tensor(m_numNodes - 1),
                           m_outputTensorsContainer.tensors(),
                           1 /* numberInputs   */,
                           1 /* numberOutputs  */,
                           nullptr /* pUserParams    */,
                           0 /* userParamsSize */,
                           "memcpy" /* nodeGuid       */,
                           "" /* nodeMame       */,
                           nullptr /* inputLayouts   */,
                           nullptr /* outputLayouts  */);
    HB_ASSERT(status == synSuccess, "Failed to synNodeCreate");
}

void TestRecipeMultipleMemcpy::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    LOG_ERR(SYN_RT_TEST, "This method ({}) should be overriden", __FUNCTION__);
    ASSERT_TRUE(false) << "This method (" << __func__ << ") must be overriden";
}
