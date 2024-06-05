#include "test_recipe_split_shape_node.hpp"
#include "synapse_api.h"
#include "test_utils.h"
#include "test_tensors_container.hpp"

TestRecipeSplitShapeNode::TestRecipeSplitShapeNode(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeSplitShapeNode>(),
                 deviceType,
                 1 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 3 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false)
{
    unsigned dimSizes[] = {5, 10, 15, 20};
    unsigned numOfDims  = 4;
    unsigned tensorSize = 5 * 10 * 15 * 20 * dataTypeSizeInBytes(syn_type_single);

    // Init m_tensorInfoVecInputs
    m_tensorInfoVecInputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecInputs[0].m_tensorSize = tensorSize;
    m_tensorInfoVecInputs[0].m_tensorType = SHAPE_TENSOR;
    m_tensorInfoVecInputs[0].m_dataType   = syn_type_single;
    m_tensorInfoVecInputs[0].m_tensorName = "input";
    std::copy(dimSizes, dimSizes + numOfDims, m_tensorInfoVecInputs[0].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    int i = 0;
    for (auto& tensorInfo : m_tensorInfoVecOutputs)
    {
        tensorInfo.m_dimsAmount = numOfDims;
        tensorInfo.m_tensorSize = tensorSize;
        tensorInfo.m_tensorType = SHAPE_TENSOR;
        tensorInfo.m_dataType   = syn_type_single;
        tensorInfo.m_tensorName = "output" + std::to_string(i);
        std::copy(dimSizes, dimSizes + numOfDims, tensorInfo.m_tensorDimsSize);
        i++;
    }
}

void TestRecipeSplitShapeNode::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    LOG_ERR(SYN_TEST, "This method ({}) must be overriden", __FUNCTION__);
    ASSERT_TRUE(false) << "This method (" << __func__ << ") must be overriden";
}

void TestRecipeSplitShapeNode::_graphCreation()
{
    // Tensors
    TestTensorsContainer inputTensor(1 /* numOfTensors */), outputTensor(3 /* numOfTensors */);

    createTrainingTensor(inputTensor,
                         0 /* tensorIndex */,
                         m_tensorInfoVecInputs[0].m_dimsAmount,
                         m_tensorInfoVecInputs[0].m_dataType,
                         m_tensorInfoVecInputs[0].m_tensorDimsSize,
                         false /* isPersist */,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         false,
                         0 /* offset */,
                         nullptr,
                         SHAPE_TENSOR,
                         nullptr /* minTensorSize */);

    for (int i = 0; i < m_tensorInfoVecOutputs.size(); ++i)
    {
        createTrainingTensor(outputTensor,
                             i /* tensorIndex */,
                             m_tensorInfoVecOutputs[i].m_dimsAmount,
                             m_tensorInfoVecOutputs[i].m_dataType,
                             m_tensorInfoVecOutputs[i].m_tensorDimsSize,
                             false /* isPersist */,
                             m_tensorInfoVecOutputs[i].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             false,
                             0 /* offset */,
                             nullptr,
                             SHAPE_TENSOR,
                             nullptr /* minTensorSize */);
    }

    // Create split_shape node
    synSplitParams splitParams;
    splitParams.axis = 2;

    synStatus status = synNodeCreate(m_graphHandle,
                                     inputTensor.tensors(),
                                     outputTensor.tensors(),
                                     1 /* numberInputs   */,
                                     3 /* numberOutputs  */,
                                     &splitParams /* pUserParams    */,
                                     sizeof(splitParams) /* userParamsSize */,
                                     "split_shape" /* nodeGuid       */,
                                     "split_shape" /* nodeMame       */,
                                     nullptr /* inputLayouts   */,
                                     nullptr /* outputLayouts  */);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}