#include "bundle_view.h"
#include "perforation_test.h"
#include "slicer/perforation_bvd_candidates_finder.h"
#include "scoped_configuration_change.h"

using namespace gc::layered_brain;

class PerforationBVDCandidatesFinderTest : public PerforationTest
{
protected:
    std::map<NodePtr, PerforationCandidates>
    findPerforationCandidates(const StrategyPtr& strategy, const ReducedBVDsPerNode& reducedBVDsPerNode) const
    {
        PerforationBVDCandidatesFinder bvdCandidatesFinder(m_bundleNodes, m_bundleViews, NUM_DCORES);
        return bvdCandidatesFinder.findPerforationCandidates(strategy, reducedBVDsPerNode);
    }
};

TEST_F(PerforationBVDCandidatesFinderTest, no_available_candidates_all_bvds_unsliceable)
{
    createUnsliceableGraph();
    createBundleViews();
    StrategyPtr strategy   = createDefaultStrategy();
    const auto& candidates = findPerforationCandidates(strategy, {});
    ASSERT_EQ(candidates.size(), m_bundleNodes.size());
    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(candidates.find(node) != candidates.end());
        ASSERT_FALSE(candidates.at(node).mmeCandidate.has_value());
        ASSERT_TRUE(candidates.at(node).preferredCandidates.empty());
        ASSERT_TRUE(candidates.at(node).validCandidates.empty());
    }
}

TEST_F(PerforationBVDCandidatesFinderTest, preferred_and_valid_candidates)
{
    createSingleMMEGraph();
    createBundleViews();
    StrategyPtr strategy = createDefaultStrategy();

    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 3);
    BundleViewId wBVD         = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 0);
    BundleViewId hBVD         = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 1);
    BundleViewId commonDimBVD = m_bundleViews->getBVDForTensorDim(m_gemm->getInput(0), 0);
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        ASSERT_GE(m_bundleViews->getBundleView(bvd).resolution, NUM_DCORES);
    }

    const auto& candidates = findPerforationCandidates(strategy, {});
    ASSERT_EQ(candidates.size(), m_bundleNodes.size());
    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(candidates.find(node) != candidates.end());
        ASSERT_FALSE(candidates.at(node).mmeCandidate.has_value());
        if (HabanaGraph::runsOnMME(node))  // GEMM
        {
            // Preferred candidates are sorted by number of occurrences in bundle nodes.
            // H BVD is common to all bundle nodes - should be first.
            ASSERT_EQ(candidates.at(node).preferredCandidates.size(), 3);
            ASSERT_EQ(candidates.at(node).preferredCandidates.front(), hBVD);
            // Valid candidates are sorted according to distance from FCD - external dims should appear first.
            // H BVD is mapped to dim=1 in both operand A and output.
            // Common dim BVD is mapped to dim=1 in operand B.
            // W BVD is mapped to dim=0 in both operand B and output.
            std::vector<BundleViewId> expectedValidCandidates = {hBVD, commonDimBVD, wBVD};
            ASSERT_EQ(candidates.at(node).validCandidates, expectedValidCandidates);
        }
        else  // TPC producer / consumer
        {
            // H BVD is mapped to dim=1 in both input and output, the second BVD is mapped to dim=0 in both input
            // and output.
            std::vector<BundleViewId> expectedCandidates = {hBVD,
                                                            m_bundleViews->getBVDForTensorDim(node->getOutput(0), 0)};
            ASSERT_EQ(candidates.at(node).preferredCandidates, expectedCandidates);
            ASSERT_EQ(candidates.at(node).validCandidates, expectedCandidates);
        }
    }
}

TEST_F(PerforationBVDCandidatesFinderTest, preferred_and_valid_candidates_with_tpc_reduction_dims)
{
    createSingleMMEGraph();
    createBundleViews();
    StrategyPtr strategy = createDefaultStrategy();

    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 3);
    BundleViewId wBVD         = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 0);
    BundleViewId hBVD         = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 1);
    BundleViewId commonDimBVD = m_bundleViews->getBVDForTensorDim(m_gemm->getInput(0), 0);
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        ASSERT_GE(m_bundleViews->getBundleView(bvd).resolution, NUM_DCORES);
    }

    ReducedBVDsPerNode reducedBVDsPerNode {};
    const auto&        tpcProducer = m_graph.getTensorProducer(m_gemm->getInput(0));
    ASSERT_TRUE(tpcProducer);
    reducedBVDsPerNode[tpcProducer] = {hBVD};
    const auto& candidates          = findPerforationCandidates(strategy, reducedBVDsPerNode);
    ASSERT_EQ(candidates.size(), m_bundleNodes.size());
    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(candidates.find(node) != candidates.end());
        ASSERT_FALSE(candidates.at(node).mmeCandidate.has_value());
        if (HabanaGraph::runsOnMME(node))  // GEMM
        {
            // H BVD is reduction dim - should be omitted from preferred candidates.
            ASSERT_EQ(candidates.at(node).preferredCandidates.size(), 2);
            ASSERT_TRUE(std::find(candidates.at(node).preferredCandidates.begin(),
                                  candidates.at(node).preferredCandidates.end(),
                                  hBVD) == candidates.at(node).preferredCandidates.end());
            // Valid candidates are sorted according to distance from FCD - external dims should appear first.
            std::vector<BundleViewId> expectedValidCandidates = {hBVD, commonDimBVD, wBVD};
            ASSERT_EQ(candidates.at(node).validCandidates, expectedValidCandidates);
        }
        else  // TPC producer / consumer
        {
            // H BVD is reduction dim - should be omitted from preferred candidates.
            std::vector<BundleViewId> expectedPreferredCandidates = {
                m_bundleViews->getBVDForTensorDim(node->getOutput(0), 0)};
            ASSERT_EQ(candidates.at(node).preferredCandidates, expectedPreferredCandidates);
            if (node == tpcProducer)
            {
                // H BVD is reduced for the producer - should be filtered out from valid candidates list.
                std::vector<BundleViewId> expectedValidCandidatesForProducer = {
                    m_bundleViews->getBVDForTensorDim(node->getOutput(0), 0)};
                ASSERT_EQ(candidates.at(node).validCandidates, expectedValidCandidatesForProducer);
            }
            else
            {
                // H BVD is not reduced for the consumer and mapped to dim=1 in both input and output,
                // the second BVD is mapped to dim=0 in both input and output.
                std::vector<BundleViewId> expectedValidCandidatesForConsumer = {
                    hBVD,
                    m_bundleViews->getBVDForTensorDim(node->getOutput(0), 0)};
                ASSERT_EQ(candidates.at(node).validCandidates, expectedValidCandidatesForConsumer);
            }
        }
    }
}

TEST_F(PerforationBVDCandidatesFinderTest, mme_candidate)
{
    createSingleMMEGraph();
    createBundleViews();
    StrategyPtr strategy = createDefaultStrategy();

    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 3);
    BundleViewId perforationBVD = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 1);
    strategy->getMmeSolution()->QORs.at(m_gemm)->solutionRequirements.perforationDimVec.push_back(perforationBVD);

    const auto& candidates = findPerforationCandidates(strategy, {});
    ASSERT_EQ(candidates.size(), m_bundleNodes.size());
    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(candidates.find(node) != candidates.end());
        if (HabanaGraph::runsOnMME(node))  // GEMM
        {
            ASSERT_TRUE(candidates.at(node).mmeCandidate.has_value());
            ASSERT_EQ(candidates.at(node).mmeCandidate.value(), perforationBVD);
        }
        else
        {
            ASSERT_FALSE(candidates.at(node).mmeCandidate.has_value());
        }
    }
}

TEST_F(PerforationBVDCandidatesFinderTest, perforation_utilization_threshold)
{
    ScopedConfigurationChange perforationUtilizationThreshold("PERFORATION_UTILIZATION_THRESHOLD", "0.87");

    // Create elementwise node -> For each dim: granularity = 1, bvd resolution = dim size
    const std::vector<TSize> shape = {16, 10, 7, 5};
    NodePtr node = TPCCustomIndexSpaceNode::createSliceableNode(createTensor(shape), createTensor(shape));
    addNodeToGraph(node);
    createBundleViews();
    StrategyPtr strategy = createDefaultStrategy();

    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 4);
    const auto& candidates = findPerforationCandidates(strategy, {});
    ASSERT_EQ(candidates.size(), 1);
    ASSERT_TRUE(candidates.find(node) != candidates.end());
    const auto& nodeCandidates = candidates.at(node);

    ASSERT_EQ(NUM_DCORES, 4);
    // Preferred candidates should contain the BVDs with perforation utilization >= threshold (0.87)
    ASSERT_EQ(nodeCandidates.preferredCandidates.size(), 2);
    // dim 0 size is 16 -> perforation utilization = (16 / 16) = 1 > 0.87
    BundleViewId dim0BVD = m_bundleViews->getBVDForTensorDim(node->getInput(0), 0);
    ASSERT_TRUE(std::find(nodeCandidates.preferredCandidates.begin(),
                          nodeCandidates.preferredCandidates.end(),
                          dim0BVD) != nodeCandidates.preferredCandidates.end());
    // dim 2 size is 7 -> perforation utilization = (7 / 8) = 0.875 > 0.87
    BundleViewId dim2BVD = m_bundleViews->getBVDForTensorDim(node->getInput(0), 2);
    ASSERT_TRUE(std::find(nodeCandidates.preferredCandidates.begin(),
                          nodeCandidates.preferredCandidates.end(),
                          dim2BVD) != nodeCandidates.preferredCandidates.end());
    // dim 1 size is 10 -> perforation utilization = (10 / 12) = 0.83 < 0.87
    // dim 3 size is 5 -> perforation utilization = (5 / 8) = 0.625 < 0.87

    // Valid candidates should contain all the BVDs (BVD resolution > NUM_DCORES)
    ASSERT_EQ(nodeCandidates.validCandidates.size(), m_bundleViews->getNumOfBundleViews());
}