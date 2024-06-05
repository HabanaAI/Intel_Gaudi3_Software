#include "gaudi2_graph.h"
#include "slicer/key_nodes_finder.h"
#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"
#include "node_factory.h"
#include "types.h"

using namespace gc::layered_brain;

class MultiMmeBundleKeyNodesFinderTest : public GraphOptimizerTest
{
protected:
    NodePtr addTPCNode(const SizeVector& sizes, const TensorPtr& input, const TensorPtr& output, bool isInBundle = true)
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        for (auto i = 0; i < sizes.size(); i++)
        {
            nodeParams.dims.emplace_back(sizes.at(i), 1, 0);
        }
        nodeParams.transpose = false;
        NodePtr node         = TPCCustomIndexSpaceNode::create(nodeParams, input, output);
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addGEMMNode(TSize heightA, TSize commonDim, TSize widthB, const TensorPtr& inputA = nullptr)
    {
        synGEMMParams      params(false, false);
        const std::vector<TSize> sizesA   = {commonDim, heightA};
        const std::vector<TSize> sizesB   = {widthB, commonDim};
        const std::vector<TSize> sizesOut = {widthB, heightA};

        TensorVector inputs;
        inputs.push_back(inputA ? inputA : std::make_shared<Tensor>(sizesA.size(), sizesA.data(), syn_type_float));
        inputs.push_back(std::make_shared<Tensor>(sizesB.size(), sizesB.data(), syn_type_float));
        TensorVector outputs;
        outputs.push_back(std::make_shared<Tensor>(sizesOut.size(), sizesOut.data(), syn_type_float));
        NodePtr node = NodeFactory::createNode(inputs,
                                               outputs,
                                               &params,
                                               NodeFactory::gemmNodeTypeName,
                                               "GEMM" + std::to_string(m_nodeId++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        m_bundleNodes.push_back(node);
        return node;
    }

    void testMultiMME(const NodeVector& expectedSortedNodes) const
    {
        MultiMmeBundleKeyNodesFinder keyNodesFinder(m_graph, m_bundleNodes);
        ASSERT_EQ(keyNodesFinder.getSortedKeyNodes(), expectedSortedNodes);
    }

    Gaudi2Graph    m_graph;
    NodeVector     m_bundleNodes;
    unsigned       m_nodeId = 0;
};

TEST_F(MultiMmeBundleKeyNodesFinderTest, single_mme_bundle)
{
    // Create a bundle with single MME node.
    const TSize commonDim = 128, height = 128, width = 128;
    NodePtr     mme = addGEMMNode(height, commonDim, width);

    // Expect a single key node.
    testMultiMME({mme});
}

TEST_F(MultiMmeBundleKeyNodesFinderTest, multi_mme_bundle_with_different_num_of_consumers)
{
    // Create a shared MME bundle - 3 MMEs consumes the output of a TPC producer.
    // Each MME has different number of consumers in bundle.
    const TSize commonDim = 128, height = 128, width = 128;

    NodePtr producer = addTPCNode({commonDim, height}, nullptr, nullptr);
    NodePtr mme0     = addGEMMNode(height, commonDim, width, producer->getOutput(0));
    NodePtr mme1     = addGEMMNode(height, commonDim, width, producer->getOutput(0));
    NodePtr mme2     = addGEMMNode(height, commonDim, width, producer->getOutput(0));

    // mme0 has 3 consumers - all of them are not bundled
    addTPCNode({width, height}, mme0->getOutput(0), nullptr, false);
    addTPCNode({width, height}, mme0->getOutput(0), nullptr, false);
    addTPCNode({width, height}, mme0->getOutput(0), nullptr, false);

    // mme1 has 2 consumers in bundle
    addTPCNode({width, height}, mme1->getOutput(0), nullptr);
    addTPCNode({width, height}, mme1->getOutput(0), nullptr);

    // mme2 has a single consumer in bundle
    addTPCNode({width, height}, mme2->getOutput(0), nullptr);

    // Expect 3 key nodes - sorted by number of consumers in the bundle.
    testMultiMME({mme1, mme2, mme0});
}

TEST_F(MultiMmeBundleKeyNodesFinderTest, multi_mme_bundle_with_num_consumers_tie_breaker_by_operands_size)
{
    // Create a shared MME bundle - 3 MMEs consumes the output of a TPC producer.
    // Each MME has a single consumer in bundle and different width size.
    const TSize commonDim = 128, height = 128, width0 = 300, width1 = 400, width2 = 200;

    NodePtr tpcProducer = addTPCNode({commonDim, height}, nullptr, nullptr);
    NodePtr mme0        = addGEMMNode(height, commonDim, width0, tpcProducer->getOutput(0));
    NodePtr mme1        = addGEMMNode(height, commonDim, width1, tpcProducer->getOutput(0));
    NodePtr mme2        = addGEMMNode(height, commonDim, width2, tpcProducer->getOutput(0));

    // Add consumer for each MME node.
    addTPCNode({width0, height}, mme0->getOutput(0), nullptr);
    addTPCNode({width1, height}, mme1->getOutput(0), nullptr);
    addTPCNode({width2, height}, mme2->getOutput(0), nullptr);

    // Expect 3 key nodes - sorted by operands size.
    testMultiMME({mme1, mme0, mme2});
}