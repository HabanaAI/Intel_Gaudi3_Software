#include <memory>

#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "liveness_analysis.h"
#include "node_factory.h"
#include "tensor.h"
#include "graph_optimizer_test.h"

class LaTest : public GraphOptimizerTest {};

TEST_F(LaTest, liveness_analysis_test)
{
    TSize sizes[] = {1,1,1};
    TSize concatSizes[] = {2,1,1};
    // Create original graph

    // Tensor creation
    TensorPtr tensor_0(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr tensor_1(new Tensor(3, sizes, syn_type_fixed));
    TensorPtr tensor_2(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr tensor_3(new Tensor(3, sizes, syn_type_fixed));
    TensorPtr tensor_4(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr tensor_5(new Tensor(3, sizes, syn_type_fixed));

    TensorVector concatinput;
    concatinput.push_back(tensor_3);
    concatinput.push_back(tensor_5);

    TensorPtr tensor_6(new Tensor(3, concatSizes, syn_type_fixed));
    TensorPtr tensor_7(new Tensor(3, sizes, syn_type_fixed));

    GaudiGraph origGraph;
    unsigned expandDim = 0;
    auto node_0 = NodeFactory::createNode({tensor_0}, {tensor_1}, &expandDim, NodeFactory::memcpyNodeTypeName, "");
    auto node_1 = NodeFactory::createNode({tensor_2}, {tensor_3}, &expandDim, NodeFactory::memcpyNodeTypeName, "");
    auto node_2 = NodeFactory::createNode({tensor_2}, {tensor_5}, &expandDim, NodeFactory::memcpyNodeTypeName, "");

    GraphEditor::addNode(origGraph, node_0);
    GraphEditor::addNode(origGraph, node_1);
    GraphEditor::addNode(origGraph, node_2);

    // TODO [SW-20488] Note that this is only done for the side effect
    // of creating an exec order since AllLivenessAnalysis uses the const
    // variation of getExeSortedNodes.
    UNUSED(origGraph.getExeSortedNodes());
    pLivenessAnalysis ls(new AllLivenessAnalysis(&origGraph));

    // wasRealTensorEncounteredBeforeNode
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_0), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_1), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_2), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_3), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_0, tensor_5), false);

    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_0), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_1), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_2), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_3), false);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_1, tensor_5), false);

    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_0), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_1), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_2), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_3), true);
    EXPECT_EQ(ls->wasRealTensorEncounteredBeforeNode(node_2, tensor_5), false);

    // isRealTensorAliveAfterNode
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_0), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_1), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_2), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_3), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_0, tensor_5), false);

    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_0), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_1), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_2), true);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_3), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_1, tensor_5), false);

    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_0), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_1), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_2), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_3), false);
    EXPECT_EQ(ls->isRealTensorAliveAfterNode(node_2, tensor_5), false);
}
