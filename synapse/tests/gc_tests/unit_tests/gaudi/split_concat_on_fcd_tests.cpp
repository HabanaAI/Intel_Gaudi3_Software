#include "compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "synapse_common_types.h"

class SplitConcatOnFcdTest : public GraphOptimizerTest
{
};

TEST_F(SplitConcatOnFcdTest, split_concat_on_fcd)
{
    setGlobalConfForTest(GCFG_OPTIMIZE_SPLIT_CONCAT_ON_FCD, "true");

    GaudiGraph graph;
    CompilationHalReaderSetter compHalReaderSetter(&graph);
    SizeArray  fullSize   = {10, 512, 4, 62};
    SizeArray  split1Size = {5, 512, 4, 62};
    SizeArray  split2Size = {1, 512, 4, 62};
    SizeArray  split3Size = {4, 512, 4, 62};

    pTensor input = pTensor(new Tensor(4, fullSize.data(), syn_type_float));

    pTensor split1 = pTensor(new Tensor(4, split1Size.data(), syn_type_float));
    pTensor split2 = pTensor(new Tensor(4, split2Size.data(), syn_type_float));
    pTensor split3 = pTensor(new Tensor(4, split3Size.data(), syn_type_float));

    pTensor relu1 = pTensor(new Tensor(4, split1Size.data(), syn_type_float));
    pTensor relu2 = pTensor(new Tensor(4, split2Size.data(), syn_type_float));
    pTensor relu3 = pTensor(new Tensor(4, split3Size.data(), syn_type_float));

    pTensor output = pTensor(new Tensor(4, fullSize.data(), syn_type_float));

    synSplitParams splitParams;
    splitParams.axis = 0;
    pNode splitNode  = NodeFactory::createNode({input},
                                              {split1, split2, split3},
                                              &splitParams,
                                              NodeFactory::splitNodeTypeName,
                                              "SPLIT");

    pNode relu1Node = NodeFactory::createNode({split1}, {relu1}, nullptr, 0, "relu_fwd_f32", "RELU1");
    pNode relu2Node = NodeFactory::createNode({split2}, {relu2}, nullptr, 0, "relu_fwd_f32", "RELU2");
    pNode relu3Node = NodeFactory::createNode({split3}, {relu3}, nullptr, 0, "relu_fwd_f32", "RELU3");

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    pNode concatNode  = NodeFactory::createNode({relu1, relu2, relu3},
                                               {output},
                                               &concatParams,
                                               NodeFactory::concatenateNodeTypeName,
                                               "CONCAT");

    ASSERT_TRUE(GraphEditor::addNode(graph, splitNode));
    ASSERT_TRUE(GraphEditor::addNode(graph, relu1Node));
    ASSERT_TRUE(GraphEditor::addNode(graph, relu2Node));
    ASSERT_TRUE(GraphEditor::addNode(graph, relu3Node));
    ASSERT_TRUE(GraphEditor::addNode(graph, concatNode));

    const auto& preNodes         = graph.getExeSortedNodes();
    unsigned    numOfReluNodes   = std::count_if(preNodes.begin(), preNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_USER);
    });
    unsigned    numOfSplitNodes  = std::count_if(preNodes.begin(), preNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_INTERNAL_SPLIT);
    });
    unsigned    numOfConcatNodes = std::count_if(preNodes.begin(), preNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_INTERNAL_CONCAT);
    });
    unsigned    numOfDmaNodes    = std::count_if(preNodes.begin(), preNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_DMA);
    });

    ASSERT_EQ(numOfReluNodes, 3);
    ASSERT_EQ(numOfSplitNodes, 1);
    ASSERT_EQ(numOfConcatNodes, 1);
    ASSERT_EQ(numOfDmaNodes, 0);

    extractMultiNodes(graph);
    extractDataMovementMultiNodes(graph);

    const auto& postNodes = graph.getExeSortedNodes();
    numOfReluNodes        = std::count_if(postNodes.begin(), postNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_USER);
    });
    numOfSplitNodes       = std::count_if(postNodes.begin(), postNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_INTERNAL_SPLIT);
    });
    numOfConcatNodes      = std::count_if(postNodes.begin(), postNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_INTERNAL_CONCAT);
    });
    numOfDmaNodes         = std::count_if(postNodes.begin(), postNodes.end(), [](const NodePtr& n) {
        return (n->getNodeType() == Node::TYPE_DMA);
    });

    ASSERT_EQ(numOfReluNodes, 3);
    ASSERT_EQ(numOfSplitNodes, 1);
    ASSERT_EQ(numOfConcatNodes, 1);
    ASSERT_EQ(numOfDmaNodes,
              8);  // One transpose before and after each RELU + one transpose before Split + one transpose after Concat
}
