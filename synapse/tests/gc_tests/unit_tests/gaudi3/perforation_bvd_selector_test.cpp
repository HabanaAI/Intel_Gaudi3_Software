#include "perforation_test.h"
#include "slicer/perforation_bvd_selector.h"

using namespace gc::layered_brain;

class PerforationBVDSelectorTest : public PerforationTest
{
protected:
    PerforationPerNode selectPerforationPerNode(const std::map<NodePtr, PerforationCandidates>& candidates) const
    {
        PerforationBVDSelector bvdSelector(m_graph, m_bundleNodes);
        return bvdSelector.selectPerforationPerNode(candidates);
    }
};

TEST_F(PerforationBVDSelectorTest, no_available_candidates)
{
    createSingleMMEGraph();

    std::map<NodePtr, PerforationCandidates> candidates;
    for (const auto& node : m_bundleNodes)
    {
        PerforationCandidates nodeCandidates;
        candidates[node] = nodeCandidates;
    }
    const auto& perforationPerNode = selectPerforationPerNode(candidates);
    ASSERT_EQ(perforationPerNode.size(), m_bundleNodes.size());
    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(perforationPerNode.find(node) != perforationPerNode.end());
        ASSERT_FALSE(perforationPerNode.at(node).has_value());
    }
}

TEST_F(PerforationBVDSelectorTest, mme_with_preferred_perforation_no_candidates_for_tpc_nodes)
{
    createSingleMMEGraph();

    std::map<NodePtr, PerforationCandidates> candidates;
    for (const auto& node : m_bundleNodes)
    {
        PerforationCandidates nodeCandidates;
        if (HabanaGraph::runsOnMME(node))
        {
            nodeCandidates.mmeCandidate        = 2;
            nodeCandidates.preferredCandidates = {0, 1};
            nodeCandidates.validCandidates     = {3, 2};
        }
        candidates[node] = nodeCandidates;
    }
    const auto& perforationPerNode = selectPerforationPerNode(candidates);
    ASSERT_EQ(perforationPerNode.size(), m_bundleNodes.size());
    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(perforationPerNode.find(node) != perforationPerNode.end());
        if (HabanaGraph::runsOnMME(node))
        {
            ASSERT_TRUE(perforationPerNode.at(node).has_value());
            ASSERT_EQ(perforationPerNode.at(node).value(), candidates.at(node).mmeCandidate.value());
        }
        else
        {
            ASSERT_FALSE(perforationPerNode.at(node).has_value());
        }
    }
}

TEST_F(PerforationBVDSelectorTest, mme_with_preferred_perforation_and_available_candidates_for_tpc_nodes)
{
    createSingleMMEGraph();

    std::map<NodePtr, PerforationCandidates> candidates;
    for (const auto& node : m_bundleNodes)
    {
        PerforationCandidates nodeCandidates;
        if (HabanaGraph::runsOnMME(node))
        {
            nodeCandidates.mmeCandidate        = 2;
            nodeCandidates.preferredCandidates = {0, 1};
            nodeCandidates.validCandidates     = {3, 2};
        }
        else
        {
            nodeCandidates.preferredCandidates = {0, 1, 2};
            nodeCandidates.validCandidates     = {3, 0, 1, 2};
        }
        candidates[node] = nodeCandidates;
    }
    const auto& perforationPerNode = selectPerforationPerNode(candidates);
    ASSERT_EQ(perforationPerNode.size(), m_bundleNodes.size());
    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(perforationPerNode.find(node) != perforationPerNode.end());
        ASSERT_TRUE(perforationPerNode.at(node).has_value());
        ASSERT_EQ(perforationPerNode.at(node).value(), candidates.at(m_gemm).mmeCandidate.value());
    }
}

TEST_F(PerforationBVDSelectorTest, perforation_dim_switching)
{
    createSingleMMEGraph();
    // Add another TPC after the TPC MME consumer.
    const auto& finalNodes = m_graph.getFinalNodes();
    ASSERT_EQ(finalNodes.size(), 1);
    NodePtr tpc =
        TPCCustomIndexSpaceNode::createSliceableNode(finalNodes.front()->getOutput(0), createTensor({4096, 4096}));
    addNodeToGraph(tpc);

    std::map<NodePtr, PerforationCandidates> candidates;
    const auto&                              nodes = m_graph.getTopoSortedNodes();
    ASSERT_EQ(nodes.size(), 4);

    candidates[nodes[0]].preferredCandidates = {2, 5, 3};  // TPC producer
    candidates[nodes[1]].mmeCandidate        = {5};        // MME
    candidates[nodes[2]].validCandidates     = {1};        // TPC consumer
    candidates[nodes[3]].preferredCandidates = {5, 1};     // Last TPC consumer

    const auto& perforationPerNode = selectPerforationPerNode(candidates);
    ASSERT_EQ(perforationPerNode.size(), m_bundleNodes.size());

    // Perforation dim for MME node set according to MME candidate.
    ASSERT_TRUE(perforationPerNode.find(nodes[1]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[1]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[1]).value(), candidates[nodes[1]].mmeCandidate.value());

    // Perforation dim for TPC producer set according to MME since it has this BVD in its preferred candidates.
    ASSERT_TRUE(perforationPerNode.find(nodes[0]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[0]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[0]).value(), candidates[nodes[1]].mmeCandidate.value());

    // Perforation dim for TPC consumer is switched since it doesn't have the MME candidate in its valid candidates.
    ASSERT_TRUE(perforationPerNode.find(nodes[2]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[2]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[2]).value(), candidates[nodes[2]].validCandidates.front());

    // Perforation dim for last TPC node set according to previous TPC (and not according to MME).
    ASSERT_TRUE(perforationPerNode.find(nodes[3]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[3]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[3]).value(), candidates[nodes[2]].validCandidates.front());
}

TEST_F(PerforationBVDSelectorTest, multiple_mme_with_preferred_perforation)
{
    createMultiMMEGraph();

    std::map<NodePtr, PerforationCandidates> candidates;
    const auto&                              nodes = m_graph.getTopoSortedNodes();
    ASSERT_EQ(nodes.size(), 4);

    candidates[nodes[0]].preferredCandidates = {2, 5, 3};  // TPC producer
    candidates[nodes[1]].mmeCandidate        = {5};        // First MME
    candidates[nodes[2]].preferredCandidates = {4, 5};     // TPC consumer
    candidates[nodes[3]].mmeCandidate        = {2};        // Second MME - after the TPC consumer
    candidates[nodes[3]].preferredCandidates = {5};

    const auto& perforationPerNode = selectPerforationPerNode(candidates);
    ASSERT_EQ(perforationPerNode.size(), m_bundleNodes.size());

    // Perforation dim for MME node set according to MME candidate.
    ASSERT_TRUE(perforationPerNode.find(nodes[1]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[1]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[1]).value(), candidates[nodes[1]].mmeCandidate.value());

    // Perforation dim for TPC producer set according to MME since it has this BVD in its preferred candidates.
    ASSERT_TRUE(perforationPerNode.find(nodes[0]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[0]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[0]).value(), candidates[nodes[1]].mmeCandidate.value());

    // Perforation dim for TPC consumer set according to MME since it has this BVD in its preferred candidates.
    ASSERT_TRUE(perforationPerNode.find(nodes[2]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[2]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[2]).value(), candidates[nodes[1]].mmeCandidate.value());

    // Perforation dim for last MME node set according to MME candidate.
    ASSERT_TRUE(perforationPerNode.find(nodes[3]) != perforationPerNode.end());
    ASSERT_TRUE(perforationPerNode.at(nodes[3]).has_value());
    ASSERT_EQ(perforationPerNode.at(nodes[3]).value(), candidates[nodes[3]].mmeCandidate.value());
}

TEST_F(PerforationBVDSelectorTest, shared_input_multi_mme)
{
    createSharedInputMultiMMEGraph();

    std::map<NodePtr, PerforationCandidates> candidates;
    NodeVector                               mmeNodes;
    for (const auto& node : m_graph.getNodes())
    {
        if (HabanaGraph::runsOnMME(node))
        {
            mmeNodes.push_back(node);
        }
        candidates[node].preferredCandidates = {0, 1, 2};  // All nodes have BVDs 0,1,2 in their preferred candidates
    }
    // Update MME candidates for MME nodes
    ASSERT_EQ(mmeNodes.size(), 3);
    candidates[mmeNodes[0]].mmeCandidate = 0;
    candidates[mmeNodes[1]].mmeCandidate = 1;
    candidates[mmeNodes[2]].mmeCandidate = 2;

    const auto& perforationPerNode = selectPerforationPerNode(candidates);
    ASSERT_EQ(perforationPerNode.size(), m_bundleNodes.size());

    // Perforation dim for MME nodes set according to MME candidate.
    // Perforation dim for TPC nodes set according to their MME producer/consumer.
    for (const auto& node : m_graph.getNodes())
    {
        ASSERT_TRUE(perforationPerNode.find(node) != perforationPerNode.end());
        ASSERT_TRUE(perforationPerNode.at(node).has_value());
        if (HabanaGraph::runsOnMME(node))
        {
            ASSERT_EQ(perforationPerNode.at(node).value(), candidates.at(node).mmeCandidate.value());
        }
        else
        {
            auto producers = m_graph.getNodeProducers(node);
            if (!producers.empty())  // MME producer
            {
                ASSERT_EQ(producers.size(), 1);
                ASSERT_TRUE(HabanaGraph::runsOnMME(*producers.begin()));
                ASSERT_EQ(perforationPerNode.at(node).value(), candidates.at(*producers.begin()).mmeCandidate.value());
            }
            else  // MME consumer
            {
                auto consumers = m_graph.getNodeConsumers(node);
                ASSERT_EQ(consumers.size(), 1);
                ASSERT_TRUE(HabanaGraph::runsOnMME(*consumers.begin()));
                ASSERT_EQ(perforationPerNode.at(node).value(), candidates.at(*consumers.begin()).mmeCandidate.value());
            }
        }
    }
}