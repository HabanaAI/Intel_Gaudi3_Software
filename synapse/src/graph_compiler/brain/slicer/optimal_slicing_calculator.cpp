#include "optimal_slicing_calculator.h"
#include "perforator.h"
#include "strategy_inflator.h"

using namespace gc::layered_brain;

// Initial implementation - assumes a single strategy, already inflated.
StrategyPtr OptimalSlicingCalculator::getOptimalStrategy(const BundleViewContainerPtr& bundleViews,
                                                         const StrategyVector&         strategies,
                                                         const NodeVector&             bundleNodes)
{
    SET_TEMP_LOG_CONTEXT("OptimalSlicingCalculator");

    LOG_TRACE(LB_SLICER,
              "Calculate optimal slicing, total number of strategies {}, total number of BVDs {}",
              strategies.size(),
              bundleViews->getNumOfBundleViews());

    StrategyPtr strategy = selectStrategy(strategies);

    // A strategy might not include all BVDs, assign multiplier for missing dimensions.
    strategy->fillMissingMultipliers(bundleViews, 1UL);

    LOG_DEBUG(LB_SLICER, "Select perforation for strategy {}", strategy->index());
    Perforator perforator(m_graph, bundleNodes, bundleViews);
    perforator.selectPerforationForStrategy(strategy);

    LOG_DEBUG(LB_SLICER, "Inflate selected strategy {}", strategy->index());
    inflateStrategy(bundleViews, strategy);

    HB_ASSERT(isStrategyValid(bundleViews, strategy),
              "Expected a single sliced BVD at most and number of slices < {}",
              m_maxNumOfSlices);

    return strategy;
}

StrategyPtr OptimalSlicingCalculator::selectStrategy(const StrategyVector& strategies) const
{
    HB_ASSERT(!strategies.empty(), "Expected at least a single strategy");
    return *std::max_element(strategies.begin(), strategies.end(), [&](const StrategyPtr& s1, const StrategyPtr& s2) {
        HB_ASSERT_PTR(s1->getMmeSolution());
        HB_ASSERT_PTR(s2->getMmeSolution());
        HB_ASSERT(!s1->getMmeSolution()->QORs.empty() && !s2->getMmeSolution()->QORs.empty(),
                  "Expected at least one MME in each strategy");
        // Sort by utilization of the first MME
        return s1->getMmeSolution()->QORs.begin()->second->perfAttr.mmeUtilization <
               s2->getMmeSolution()->QORs.begin()->second->perfAttr.mmeUtilization;
    });
}

void OptimalSlicingCalculator::inflateStrategy(const BundleViewContainerPtr& bundleViews,
                                               const StrategyPtr&            strategy) const
{
    StrategyInflator           inflator(bundleViews);
    std::vector<InflationType> inflationPriorities = {InflationType::INFLATE_FOR_UTILIZATION,
                                                      InflationType::INFLATE_FOR_BW,
                                                      InflationType::INFLATE_FOR_NUM_SLICES};
    HB_ASSERT_PTR(strategy->getMmeSolution());
    HB_ASSERT(!strategy->getMmeSolution()->QORs.empty(), "Expected at least one MME in the strategy");
    for (auto inflationType : inflationPriorities)
    {
        NodePtr nodeToInflate = (inflationType == InflationType::INFLATE_FOR_NUM_SLICES)
                                    ? nullptr
                                    : strategy->getMmeSolution()->QORs.begin()->first;  // Inflate by first MME
        while (!isStrategyValid(bundleViews, strategy))
        {
            bool successfulInflation = inflator.inflateOneStep(inflationType, strategy, nodeToInflate);
            LOG_DEBUG(LB_SLICER,
                      "\t One step inflation (type: {}) {}",
                      inflationType,
                      successfulInflation ? "succeeded" : "failed");
            if (!successfulInflation)
            {
                break;  // Move to next inflation type
            }
        }
    }
}

bool OptimalSlicingCalculator::isStrategyValid(const BundleViewContainerPtr& bundleViews,
                                               const StrategyPtr&            strategy) const
{
    return (getNumOfSlicedBVDs(bundleViews, strategy) <= 1) &&
           (getNumOfSlices(bundleViews, strategy) <= m_maxNumOfSlices);
}

unsigned OptimalSlicingCalculator::getNumOfSlicedBVDs(const BundleViewContainerPtr& bundleViews,
                                                      const StrategyPtr&            strategy) const
{
    unsigned numOfSlicedBVDs = 0;
    for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
    {
        const auto& multiplier = strategy->getBVDMultiplier(bvdId);
        if ((multiplier.isSliced()) && (multiplier.getMultiplier() < bundleViews->getBundleView(bvdId).resolution))
        {
            numOfSlicedBVDs++;
        }
    }
    return numOfSlicedBVDs;
}

uint64_t OptimalSlicingCalculator::getNumOfSlices(const BundleViewContainerPtr& bundleViews,
                                                  const StrategyPtr&            strategy) const
{
    uint64_t numSlices = 1;
    for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
    {
        const auto& multiplier = strategy->getBVDMultiplier(bvdId);
        if (multiplier.isSliced())
        {
            numSlices *= div_round_up(bundleViews->getBundleView(bvdId).resolution, multiplier.getMultiplier());
        }
    }
    return numSlices;
}