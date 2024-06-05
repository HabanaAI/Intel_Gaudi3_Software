
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"

class HandleIdentityCastNodesTest : public GraphOptimizerTest
{
protected:
    virtual void SetUp()
    {
        GraphOptimizerTest::SetUp();
        GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(true);
    }
};

TEST_F(HandleIdentityCastNodesTest, handle_identity_cast_nodes)
{
    // handleIdentityCastNodes requires GCFG_SYNAPSE_DATA_TYPE_SELECTION to be on
    GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(true);
    Gaudi2Graph g;
    g.setInferenceMode(true);
    bool      ret;

    std::list<std::string> removeNodeNames;
    std::list<std::string> keepNodeNames;

    const TSize sizes[] = {1, 2, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t4 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t5 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t6 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t7 = pTensor(new Tensor(4U, sizes, syn_type_int32));

    pNode tanh1 = NodeFactory::createNode({t1}, {t2}, nullptr, "tanh", "tanh1");
    pNode cast1 = NodeFactory::createNode({t2}, {t3}, nullptr, "cast_i8_to_i16", "cast1");
    pNode cast2 = NodeFactory::createNode({t3}, {t4}, nullptr, "cast_i16_to_i16", "cast2");
    pNode cast3 = NodeFactory::createNode({t4}, {t5}, nullptr, "cast_f32_to_f32", "cast3");
    pNode tanh2 = NodeFactory::createNode({t5}, {t6}, nullptr, "tanh", "tanh2");
    pNode cast4 = NodeFactory::createNode({t6}, {t7}, nullptr, "cast_f32_to_f32", "cast4");
    GraphEditor::addNode(g, tanh1);
    GraphEditor::addNode(g, cast1);
    GraphEditor::addNode(g, cast2);
    GraphEditor::addNode(g, cast3);
    GraphEditor::addNode(g, tanh2);
    GraphEditor::addNode(g, cast4);

    keepNodeNames.push_back("tanh1");
    keepNodeNames.push_back("cast1");
    removeNodeNames.push_back("cast2");
    removeNodeNames.push_back("cast3");
    keepNodeNames.push_back("tanh2");
    removeNodeNames.push_back("cast4");

    // run the pass
    ret = handleIdentityCastNodes(g);
    ASSERT_TRUE(ret) << "failed to run handle identity cast nodes pass";

    const NodeVector& sortedNodes = g.getExeSortedNodes();
    for (const NodePtr& node : sortedNodes)
    {
        // Checking all nodes in removeNodeNames were removed
        ASSERT_EQ(std::find(removeNodeNames.begin(), removeNodeNames.end(), node->getNodeName()),
                  removeNodeNames.end()) << "Did not remove node " << node->getNodeName();

        keepNodeNames.remove(node->getNodeName());
    }

    ASSERT_EQ(keepNodeNames.size(), 0) << "Nodes which should have been kept, were removed";
}
