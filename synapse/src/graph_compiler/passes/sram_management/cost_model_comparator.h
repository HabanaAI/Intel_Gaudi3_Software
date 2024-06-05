#pragma once

#include "habana_graph.h"
#include "mme_slicing_strategy.h"
#include "slicing_brain.h"
#include "passes/sram_management/cost_model.h"
#include "passes/sram_management/strategy_cost_model.h"
#include <unordered_map>

// Cost model based strategies comparator.
// The comparison will be done based on execution time and HBM traffic as a tie breaker.
class StrategyCostModelComparator
{
public:
    StrategyCostModelComparator(const pBundle&     bundle,
                                const HabanaGraph& graph,
                                const AllBrains&   slicingBrains,
                                bool               resetCache)
    : m_graph(graph), m_bundle(bundle), m_slicingBrains(slicingBrains)
    {
        if (resetCache)
        {
            gaudi::StrategyCostModel::reset();
        }
    }

    bool operator()(const SlicingStrategyPtr& a, const SlicingStrategyPtr& b);

private:
    gaudi::CostModel::Cost getStrategyCost(const SlicingStrategyPtr& strategy);
    pMmeSlicingStrategy    cloneStrategyAndStitchCandidates(const SlicingStrategyPtr& origStrategy);
    gaudi::CostModel::Cost calculateStrategyCost(const pMmeSlicingStrategy& strategy) const;

    const HabanaGraph&                                             m_graph;
    const pBundle&                                                 m_bundle;
    const AllBrains&                                               m_slicingBrains;
};
