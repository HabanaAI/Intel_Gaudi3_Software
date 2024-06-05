#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include <gtest/gtest.h>
#include "compilation_hal_reader.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"

using namespace std;

class PASSES : public GraphOptimizerTest
{
};

TEST_F(PASSES, einsum_test_extraction)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    /* This test verifies that Einsum multinode is being extracted properly - to the expected nodes */
    GaudiGraph g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    TSize i = 3;
    TSize b = 5;
    TSize n = 6;
    TSize d = 2;
    TSize j = 4;

    TSize input1Dims[] = {d, n, b, i};
    TSize input2Dims[] = {d, n, b, j};
    TSize outputDims[] = {n, b, j};

    int32_t input1_data_buff[d][n][b][i];
    int32_t input2_data_buff[d][n][b][j];
    int32_t output_data_buff[n][b][j];

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr input1Tensor =
        TensorPtr(new Tensor(4, input1Dims, syn_type_float, reinterpret_cast<char*>(input1_data_buff)));
    TensorPtr input2Tensor =
        TensorPtr(new Tensor(4, input2Dims, syn_type_float, reinterpret_cast<char*>(input2_data_buff)));
    TensorPtr outputTensor =
        TensorPtr(new Tensor(3, outputDims, syn_type_float, reinterpret_cast<char*>(output_data_buff)));

    input1Tensor->setMemoryDescriptor(persistentMemoryDesc);
    input1Tensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    input2Tensor->setMemoryDescriptor(persistentMemoryDesc);
    input2Tensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    outputTensor->setMemoryDescriptor(persistentMemoryDesc);
    outputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    synEinsumParams einsumParams {"ibnd,jbnd->jbn"};

    NodePtr einsumNode =
        NodeFactory::createNode({input1Tensor, input2Tensor}, {outputTensor}, &einsumParams, "Einsum", "einsum");

    GraphEditor::addNode(g, einsumNode);
    ASSERT_TRUE(extractMultiNodes(g));
    ASSERT_TRUE(extractDataMovementMultiNodes(g));

    const NodeVector& nodes = g.getExeSortedNodes();
    NodePtr    currNode;
    TPCNodePtr tpcNode;
    /* expecting:
     * 1. transpose - 1
     * 2. reduceSum - 1
     * 3. bacth matmul - 1
     * 4. reshape - 3
     * */
    unsigned countTranspose = 0, countReduceSum = 0, countBmm = 0, countReshape = 0;

    for (const NodePtr& node : nodes)
    {
        tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        if (tpcNode != nullptr)
        {
            if (tpcNode->getGUID() == "reduce_sum_fwd_f32")
            {
                countReduceSum++;
            }
            else
            {
                ASSERT_TRUE(false) << "Unexpected TPC node with guid " + tpcNode->getGUID();
            }
        }
        else
        {
            switch (node->getNodeType())
            {
                case Node::TYPE_LOGICAL_TRANSPOSE:
                    countTranspose++;
                    break;
                case Node::TYPE_BATCH_GEMM:
                    countBmm++;
                    break;
                case Node::TYPE_INTERNAL_RESHAPE:
                    countReshape++;
                    break;
                case Node::TYPE_DMA:
                case Node::TYPE_MEMCOPY:
                case Node::TYPE_INTERNAL_FLATTEN:
                case Node::TYPE_TRANSPOSED_SHAPE_NODE:
                    /* ignore nodes */
                    break;
                default:
                    ASSERT_TRUE(false) << "Unexpected Node type " + std::to_string(node->getNodeType()) +
                                              ", name is:" + node->getNodeName();
            }
        }
    }

    ASSERT_EQ(countTranspose, 1);
    ASSERT_EQ(countReshape, 4);
    ASSERT_EQ(countBmm, 1);
    ASSERT_EQ(countReduceSum, 1);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";
}