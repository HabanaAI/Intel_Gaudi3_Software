#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"

namespace gaudi
{

class MemcpyToCastTest : public GraphOptimizerTest
{
};

// The test calls directly the selectMemcpyEngine pass. Since the input and output data types are different, a tpc node
// of cast is inserted to the graph instead of the original (semantic) memcpy node
// original graph: [inTensor]-->(memcpyNode)--[outTensor]
// modified graph: [inTensor]-->((TPC)cast_f32_to_bf16)--[outTensor]
TEST_F(MemcpyToCastTest, f32_to_bf16)
    {
        GaudiGraph g;
        TSize sizes[] = {1, 1, 1};
        pTensor t0(new Tensor(3, sizes, syn_type_float));
        pTensor t1(new Tensor(3, sizes, syn_type_bf16));
        auto n = NodeFactory::createNode({t0}, {t1}, nullptr, NodeFactory::memcpyNodeTypeName, "node_memcpy");
        GraphEditor::addNode(g, n);
        bool ret = selectMemcpyEngine(g);

        //validations:
        ASSERT_TRUE(ret) << "selectMemcpyEngine failed";
        ASSERT_EQ(g.getExeSortedNodes().size(), 1);
        auto nodeAfter = g.getExeSortedNodes().back();
        ASSERT_TRUE(GaudiGraph::runsOnTPC(nodeAfter));
        std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(nodeAfter);
        ASSERT_NE(tpcNode, nullptr);
        ASSERT_TRUE(tpcNode->getGUID() == "cast_f32_to_bf16");
        ASSERT_EQ(tpcNode->getInputs().size(), 1);
        ASSERT_EQ(tpcNode->getOutputs().size(), 1);
        ASSERT_EQ(tpcNode->getInput(0)->getElementType(), syn_type_float);
        ASSERT_EQ(tpcNode->getOutput(0)->getElementType(), syn_type_bf16);
    }

} // namespace gaudi