#include "test_recipe_dynamic_split.hpp"

#include "node_factory.h"

#include "synapse_common_types.h"

TestRecipeDynamicSplit::TestRecipeDynamicSplit(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeDynamicSplit>(),
                 deviceType,
                 2 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 3 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */),
  m_uniqueNodeId(0)
{
    const unsigned numOfDims            = 3;
    uint64_t       tensorSizeInElements = 0;
    unsigned       dim0(0), dim1(0), dim2(0);
    unsigned       tensorDimSizes[SYN_MAX_TENSOR_DIM] = {0};

    // Init m_tensorInfoVecInputs
    // Tensor-0
    unsigned tensorIndex = 0;  // Per type(input / output)
    //
    dim0 = 15;
    dim1 = 40;
    dim2 = 384;
    //
    tensorDimSizes[0]    = dim0;
    tensorDimSizes[1]    = dim1;
    tensorDimSizes[2]    = dim2;
    tensorSizeInElements = dim0 * dim1 * dim2;
    //
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "in";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    //
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);
    //
    // Tensor-1
    tensorIndex++;
    //
    dim0 = 5;
    dim1 = 40;
    dim2 = 2;
    //
    tensorDimSizes[0]    = dim0;
    tensorDimSizes[1]    = dim1;
    tensorDimSizes[2]    = dim2;
    tensorSizeInElements = dim0 * dim1 * dim2;
    //
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = HOST_SHAPE_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::CONST_TENSOR_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "shape";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    //
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    //
    dim0              = 15;
    dim1              = 40;
    dim2              = 128;
    tensorDimSizes[0] = dim0;
    tensorDimSizes[1] = dim1;
    tensorDimSizes[2] = dim2;
    //
    tensorSizeInElements = dim0 * dim1 * dim2;
    //
    for (tensorIndex = 0; tensorIndex < m_tensorInfoVecOutputs.size(); tensorIndex++)
    {
        m_tensorInfoVecOutputs[tensorIndex].m_dimsAmount  = numOfDims;
        m_tensorInfoVecOutputs[tensorIndex].m_tensorType  = DATA_TENSOR_DYNAMIC;
        m_tensorInfoVecOutputs[tensorIndex].m_dataType    = syn_type_single;
        m_tensorInfoVecOutputs[tensorIndex].m_sectionType = TestSectionType::CONST_TENSOR_SECTION;
        m_tensorInfoVecOutputs[tensorIndex].m_tensorName  = "output" + std::to_string(tensorIndex);
        m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
            tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);
        std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
        std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[tensorIndex].m_tensorMinDimsSize);
    }
}

void TestRecipeDynamicSplit::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    ASSERT_TRUE(false) << "Validation had not been implemented";
}

void TestRecipeDynamicSplit::_graphCreation()
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

    // Create Split node
    synSplitParams initSplitParams = {0};
    status                         = synNodeCreateWithId(m_graphHandle,
                                 m_inputTensorsContainer.tensors(),
                                 m_outputTensorsContainer.tensors(),
                                 m_inputTensorsContainer.size(),
                                 m_outputTensorsContainer.size(),
                                 (void*)&initSplitParams,
                                 sizeof(synSplitParams),
                                 NodeFactory::dynamicSplitNodeTypeName,
                                 "split",
                                 &m_uniqueNodeId,
                                 nullptr,
                                 nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create " << NodeFactory::dynamicSplitNodeTypeName << "Node";
}