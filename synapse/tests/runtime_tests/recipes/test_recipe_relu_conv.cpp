#include "test_recipe_relu_conv.hpp"

#include "node_factory.h"

#include "synapse_api.h"
#include "test_tensor_init.hpp"
#include "../infra/test_types.hpp"

#include "test_tensors_container.hpp"
#include "test_utils.h"

#include <vector>

TestRecipeReluConv::TestRecipeReluConv(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeReluConv>(),
                 deviceType,
                 2 /* inputTensorsAmount   */,
                 1 /* innerTensorsAmount   */,
                 1 /* outputTensorsAmount  */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    const unsigned           numOfDims            = 4;
    const unsigned           N                    = 1;
    const unsigned           H                    = 1000;
    const unsigned           W                    = 1;
    const unsigned           B                    = 1000;
    const TSize              tensorDimSizes[4]    = {1000, 1000, 1, 1};
    const uint64_t           tensorSizeInElements = N * W * H * B;
    std::vector<std::string> tensorsNamesInput    = {"relu_in", "Conv_operand_b"};

    unsigned tensorIndex = 0, inputTensorsAmt = m_inputTensorsContainer.size();  // Per type(input / output)

    // Init m_tensorInfoVecInputs
    for (; tensorIndex < inputTensorsAmt; tensorIndex++)
    {
        m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
        m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
        m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
        m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
        m_tensorInfoVecInputs[tensorIndex].m_tensorName  = tensorsNamesInput.at(tensorIndex);
        m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
            tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
        std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);
    }

    m_tensorInfoVecOutputs[0].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[0].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[0].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[0].m_tensorName  = "Conv_result";
    m_tensorInfoVecOutputs[0].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[0].m_tensorDimsSize);
}

void TestRecipeReluConv::_graphCreation()
{
    unsigned inputTensorsAmt = m_inputTensorsContainer.size();

    synStatus status(synSuccess);

    for (unsigned i = 0; i < inputTensorsAmt; i++)
    {
        createTrainingTensor(m_inputTensorsContainer,
                             i,
                             m_tensorInfoVecInputs[i],
                             true /* isPersist */,
                             m_tensorInfoVecInputs[i].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* hostBuffer */);
    }

    // Single inner tensor
    {
        TensorInfo tensor_info;
        tensor_info              = m_tensorInfoVecInputs[0];
        tensor_info.m_tensorName = "relu_out";

        createTrainingTensor(m_innerTensorsContainer,
                             0 /* tensorIndex */,
                             4U,
                             syn_type_single,
                             tensor_info.m_tensorDimsSize,
                             false,
                             tensor_info.m_tensorName,
                             m_graphHandle,
                             nullptr,
                             false,
                             0,
                             nullptr,
                             DATA_TENSOR,
                             nullptr /* minTensorSize */);
    }

    // Single output presistent-tensor
    {
        createTrainingTensor(m_outputTensorsContainer,
                             0 /* tensorIndex */,
                             m_tensorInfoVecOutputs[0],
                             true /* isPersist */,
                             m_tensorInfoVecOutputs[0].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* hostBuffer */);
    }

    /*
     *
     *  adding a node to copy the input tensor in (input is relu_in, output tensor is relu_out)
     *      synNodeCreate()
     *
     */
    // relu node
    const char*    reluNodeGuid         = "relu_fwd_f32";  // NodeFactory::memcpyNodeTypeName;
    char           reluNodeName[]       = "relu_node";
    const unsigned reluNodeNumOfInputs  = 1;
    const unsigned reluNodeNumOfOutputs = 1;

    status = synNodeCreate(m_graphHandle,
                           &m_inputTensorsContainer.tensor(0),
                           &m_innerTensorsContainer.tensor(0),
                           reluNodeNumOfInputs,
                           reluNodeNumOfOutputs,
                           nullptr,
                           0,
                           reluNodeGuid,
                           reluNodeName,
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create internal relu Node";

    /*
     *
     *   now add a convoution (gemm) node on the inputs (relu_out and Conv_operand_b)
     *         it will store the output in Conv_result
     *          synNodeCreate()
     */
    char           addNodeGuid[]       = "gemm";
    char           addNodeName[]       = "gemmnode";
    const unsigned addNodeNumOfInputs  = 2;
    const unsigned addNodeNumOfOutputs = 1;

    std::vector<synTensor> tensors_handles = {m_innerTensorsContainer.tensor(0), m_inputTensorsContainer.tensor(1)};

    status = synNodeCreate(m_graphHandle,
                           tensors_handles.data(),
                           m_outputTensorsContainer.tensors(),
                           addNodeNumOfInputs,
                           addNodeNumOfOutputs,
                           nullptr,
                           0,
                           addNodeGuid,
                           addNodeName,
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create Add Node";
}