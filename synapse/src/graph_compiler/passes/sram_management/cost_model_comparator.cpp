#include "cost_model_comparator.h"
#include "passes/sram_management/engine_cost_model.h"
#include "passes/sram_management/strategy_cost_model.h"

bool StrategyCostModelComparator::operator()(const SlicingStrategyPtr& a, const SlicingStrategyPtr& b)
{
    // Comparison function - returns ​true if a is less than b.
    const auto& aCost = getStrategyCost(a);
    const auto& bCost = getStrategyCost(b);
    if (aCost.timeNano != bCost.timeNano)
    {
        return aCost.timeNano < bCost.timeNano;
    }
    HB_ASSERT(bCost.hbmTrafficBytes != 0, "Invalid HBM traffic (0 bytes) for strategy");
    const auto trafficDiff = std::abs(1.0 - (double)aCost.hbmTrafficBytes / (double)bCost.hbmTrafficBytes);
    if ((trafficDiff > SlicingBrain::knobs.hbmTrafficDiffThreshold) ||
        (a->getSlicingData().getWalkingDir() == b->getSlicingData().getWalkingDir()))
    {
        return aCost.hbmTrafficBytes < bCost.hbmTrafficBytes;
    }
    // Prefer strategies with Left-To-Right walking direction (TODO: SW-44181)
    return a->getSlicingData().getWalkingDir() == StrategySlicingData::WalkingDir::LeftToRight;
}

gaudi::CostModel::Cost StrategyCostModelComparator::getStrategyCost(const SlicingStrategyPtr& strategy)
{
    HB_ASSERT(m_bundle->type() == BundleType::MME, "Invalid bundle type for cost-model strategy comparator");
    if (strategy->getCost().is_set())
    {
        strategy->getCost().value();
    }

    // The original strategy is cloned, the valid candidates will be added to the new one.
    const pMmeSlicingStrategy& strategyWithStitchedCandidates = cloneStrategyAndStitchCandidates(strategy);

    const auto& cost = calculateStrategyCost(strategyWithStitchedCandidates);
    strategy->setCost(cost);

    return cost;
}

pMmeSlicingStrategy
StrategyCostModelComparator::cloneStrategyAndStitchCandidates(const SlicingStrategyPtr& origStrategy)
{
    // The original strategy is cloned, the valid candidates will be added to the new one.
    pMmeSlicingStrategy mmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(origStrategy->clone(false));

    for (auto& candidate : mmeStrategy->getMmeSlicingData().getRoleCandidates())
    {
        if (candidate && candidate->nodeToStitch)
        {
            bool res = true;
            switch (candidate->role)
            {
                case BundleExpansion::WideInputProducer:
                case BundleExpansion::NarrowInputProducer:
                case BundleExpansion::SlaveInputProducer:
                    if (candidate->reshapeNode)
                    {
                        res = m_slicingBrains.m_reshapeBrain->addProducerToStrategy(candidate, mmeStrategy);
                        HB_ASSERT(res,
                                  "Cost model for bundle {}: Failed to add producer reshape {} to strategy",
                                  m_bundle->index(),
                                  candidate->reshapeNode->getNodeName());
                    }
                    res = m_slicingBrains.m_tpcSlaveBrain->addProducerToStrategy(candidate, mmeStrategy);
                    HB_ASSERT(res,
                              "Cost model for bundle {}: Failed to add producer {} to strategy",
                              m_bundle->index(),
                              candidate->nodeToStitch->getNodeName());
                    break;
                case BundleExpansion::SlaveOutputConsumer:
                case BundleExpansion::OutputConsumer:
                    if (candidate->reshapeNode)
                    {
                        res = m_slicingBrains.m_reshapeBrain->addConsumerToStrategy(candidate, mmeStrategy);
                        HB_ASSERT(res,
                                  "Cost model for bundle {}: Failed to add consumer reshape {} to strategy",
                                  m_bundle->index(),
                                  candidate->reshapeNode->getNodeName());
                    }
                    res = m_slicingBrains.m_tpcSlaveBrain->addConsumerToStrategy(candidate, mmeStrategy);
                    HB_ASSERT(res,
                              "Cost model for bundle {}: Failed to add consumer {} to strategy",
                              m_bundle->index(),
                              candidate->nodeToStitch->getNodeName());
                    break;
                case BundleExpansion::SharedInputConsumer:
                    m_slicingBrains.m_mmeSlaveBrain.addSharedOperandMme(candidate, mmeStrategy);
                    break;
                default:
                    HB_ASSERT(false, "Cost model for bundle {}: Unexpected candidate role", m_bundle->index());
                    break;
            }
        }
    }
    return mmeStrategy;
}

gaudi::CostModel::Cost StrategyCostModelComparator::calculateStrategyCost(const pMmeSlicingStrategy& strategy) const
{
    gaudi::MMECostModel         mmeCM(*m_graph.getHALReader());
    gaudi::TPCCostModel         tpcCM(*m_graph.getHALReader());
    gaudi::DmaFetchCostModel    fetchCM(m_graph);
    gaudi::DmaEvictionCostModel evictCM(m_graph, m_bundle, strategy);
    gaudi::StrategyCostModel    strategyCost(m_graph, m_bundle, strategy, m_slicingBrains);
    strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);
    return strategyCost.getAggregatedCost();
}