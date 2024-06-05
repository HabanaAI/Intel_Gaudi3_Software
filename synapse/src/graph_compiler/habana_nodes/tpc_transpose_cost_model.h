#pragma once

#include "dma_cost_model.h"
#include "habana_graph.h"
#include "transpose_nodes_creator.h"
#include "tpc_node.h"

class TpcTransposeCostModel : public TransposeCostModel
{
public:
    uint64_t getCost(const TensorPtr& input, const TransposePermutationArray& permutation) const override
    {
        HB_ASSERT(false,
                  "TpcTransposeCostModel::getCost(const TensorPtr& input, const TransposePermutationArray& "
                  "permutation) - Not implemented");
        return 0;
    }

    uint64_t getCost(const NodeVector& extractedNodes) const override
    {
        uint64_t cost = 0;
        for (const auto& node : extractedNodes)
        {
            if (!node->isTranspose()) continue;
            if (node->isLogicalOperation())
            {
                cost += getLogicalTransposeCost(node);
            }
            else
            {
                HB_ASSERT(HabanaGraph::runsOnTPC(node), "Expected TPC Node");
                const auto&                             tpcNode     = static_cast<TPCNode&>(*node);
                std::optional<TPCNode::CostModelResult> tpcNodeCost = tpcNode.getCostModelResult();
                cost += tpcNodeCost ? tpcNodeCost->asicCycles : 0;  // tpc_asic_cycles
            }
        }

        return cost;
    }

private:
    uint64_t getLogicalTransposeCost(const NodePtr& node) const
    {
        HB_ASSERT(false, "TpcTransposeCostModel::getLogicalTransposeCost - Not implemented");
        return 0;
    }
};