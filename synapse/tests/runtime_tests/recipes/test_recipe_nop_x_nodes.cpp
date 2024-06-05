#include "test_recipe_nop_x_nodes.hpp"
#include "test_tensors_container.hpp"
#include "synapse_api.h"

#include "test_utils.h"

#include <iostream>
#include <vector>

TestRecipeNopXNodes::TestRecipeNopXNodes(synDeviceType deviceType, unsigned nodes)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeNopXNodes>(m_nodes),
                 deviceType,
                 1 /* inputTensorsAmount */,
                 nodes - 1 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */),
  m_nodes(nodes)
{
    const unsigned numOfDims = 4;
    const uint32_t dimSize   = 16;
    TSize dimSizes[HABANA_DIM_MAX] = {dimSize, dimSize, dimSize, dimSize};
    const uint64_t tensorSize = dimSize * dimSize * dimSize * dimSize * sizeof(uint8_t);

    // Init m_tensorInfoVecInputs
    m_tensorInfoVecInputs[0].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[0].m_tensorSize  = tensorSize;
    m_tensorInfoVecInputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[0].m_tensorName  = "input";
    m_tensorInfoVecInputs[0].m_dataType    = syn_type_uint8;
    m_tensorInfoVecInputs[0].m_tensorType  = DATA_TENSOR;
    std::copy(dimSizes, dimSizes + numOfDims, m_tensorInfoVecInputs[0].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    m_tensorInfoVecOutputs[0].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[0].m_tensorSize  = tensorSize;
    m_tensorInfoVecOutputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[0].m_tensorName  = "output";
    m_tensorInfoVecOutputs[0].m_dataType    = syn_type_uint8;
    m_tensorInfoVecOutputs[0].m_tensorType  = DATA_TENSOR;
    std::copy(dimSizes, dimSizes + numOfDims, m_tensorInfoVecOutputs[0].m_tensorDimsSize);
}

void TestRecipeNopXNodes::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    LOG_ERR(SYN_TEST, "This method ({}) must be overriden", __FUNCTION__);
    ASSERT_TRUE(false) << "This method (" << __func__ << ") must be overriden";
}

void TestRecipeNopXNodes::_graphCreation()
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

    const synTensor* lastNodeInputTensor = m_inputTensorsContainer.tensors();

    for (int i = 0; i < m_nodes - 1; i++)
    {
        std::string name("mid" + std::to_string(i));
        createTrainingTensor(m_innerTensorsContainer,
                             i /* tensorIndex */,
                             m_tensorInfoVecOutputs[0],
                             false /* isPersist */,
                             name.c_str(),
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* minTensorSize */);

        status = synNodeCreate(m_graphHandle,
                               lastNodeInputTensor,
                               &m_innerTensorsContainer.tensors()[i],
                               1 /* numberInputs   */,
                               1 /* numberOutputs  */,
                               nullptr /* pUserParams    */,
                               0 /* userParamsSize */,
                               "nop" /* nodeGuid       */,
                               "nop" /* nodeMame       */,
                               nullptr /* inputLayouts   */,
                               nullptr /* outputLayouts  */);
        ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";

        lastNodeInputTensor = &m_innerTensorsContainer.tensors()[i];
    }

    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecOutputs[0],
                         true /* isPersist */,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* minTensorSize */);

    status = synNodeCreate(m_graphHandle,
                           lastNodeInputTensor,
                           m_outputTensorsContainer.tensors(),
                           1 /* numberInputs   */,
                           1 /* numberOutputs  */,
                           nullptr /* pUserParams    */,
                           0 /* userParamsSize */,
                           "nop" /* nodeGuid       */,
                           "nop" /* nodeName       */,
                           nullptr /* inputLayouts   */,
                           nullptr /* outputLayouts  */);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}
