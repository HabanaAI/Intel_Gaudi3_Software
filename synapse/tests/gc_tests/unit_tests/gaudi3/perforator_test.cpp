#include "perforation_test.h"
#include "slicer/perforator.h"

using namespace gc::layered_brain;

class PerforatorTest : public PerforationTest
{
protected:
    void selectPerforationForStrategy(const StrategyPtr& strategy) const
    {
        Perforator perforator(m_graph, m_bundleNodes, m_bundleViews);
        return perforator.selectPerforationForStrategy(strategy);
    }
};

TEST_F(PerforatorTest, single_mme_bundle)
{
    createSingleMMEGraph();
    createBundleViews();
    StrategyPtr  strategy  = createDefaultStrategy();
    BundleViewId slicedBVD = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 1);
    strategy->setBVDMultiplier(slicedBVD, BVDMultiplier(NUM_DCORES));
    strategy->getMmeSolution()->QORs[m_gemm]->solutionRequirements.perforationDimVec.push_back(slicedBVD);
    selectPerforationForStrategy(strategy);

    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(strategy->getPerforationBVDForNode(node).has_value());
    }
}

TEST_F(PerforatorTest, multi_mme_bundle)
{
    createMultiMMEGraph();
    createBundleViews();
    StrategyPtr  strategy  = createDefaultStrategy();
    BundleViewId slicedBVD = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 1);
    strategy->setBVDMultiplier(slicedBVD, BVDMultiplier(NUM_DCORES));
    strategy->getMmeSolution()->QORs[m_gemm]->solutionRequirements.perforationDimVec.push_back(slicedBVD);
    selectPerforationForStrategy(strategy);

    for (const auto& node : m_bundleNodes)
    {
        if (HabanaGraph::runsOnMME(node) && (node != m_gemm))
        {
            // Second MME should not be perforated - perforationDimVec in MME solution is empty
            ASSERT_FALSE(strategy->getPerforationBVDForNode(node).has_value());
        }
        else
        {
            ASSERT_TRUE(strategy->getPerforationBVDForNode(node).has_value());
        }
    }
}

TEST_F(PerforatorTest, shared_input_multi_mme_bundle)
{
    createSharedInputMultiMMEGraph();
    createBundleViews();
    StrategyPtr  strategy  = createDefaultStrategy();
    BundleViewId slicedBVD = m_bundleViews->getBVDForTensorDim(m_gemm->getOutput(0), 1);
    strategy->setBVDMultiplier(slicedBVD, BVDMultiplier(NUM_DCORES));
    strategy->getMmeSolution()->QORs[m_gemm]->solutionRequirements.perforationDimVec.push_back(slicedBVD);
    selectPerforationForStrategy(strategy);

    for (const auto& node : m_bundleNodes)
    {
        if (HabanaGraph::runsOnMME(node) && (node != m_gemm))
        {
            // Second MME should not be perforated - perforationDimVec in MME solution is empty
            ASSERT_FALSE(strategy->getPerforationBVDForNode(node).has_value());
        }
        else
        {
            ASSERT_TRUE(strategy->getPerforationBVDForNode(node).has_value());
        }
    }
}

TEST_F(PerforatorTest, tpc_only_bundle)
{
    createTPCOnlyGraph();
    createBundleViews();
    StrategyPtr  strategy  = createDefaultStrategy();
    BundleViewId slicedBVD = m_bundleViews->getBVDForTensorDim((*m_graph.getNodes().begin())->getOutput(0), 0);
    strategy->setBVDMultiplier(slicedBVD, BVDMultiplier(NUM_DCORES));
    selectPerforationForStrategy(strategy);

    for (const auto& node : m_bundleNodes)
    {
        ASSERT_TRUE(strategy->getPerforationBVDForNode(node).has_value());
    }
}

TEST_F(PerforatorTest, unsliceable_bundle)
{
    createUnsliceableGraph();
    createBundleViews();
    StrategyPtr strategy = createDefaultStrategy();
    selectPerforationForStrategy(strategy);

    for (const auto& node : m_bundleNodes)
    {
        ASSERT_FALSE(strategy->getPerforationBVDForNode(node).has_value());
    }
}
