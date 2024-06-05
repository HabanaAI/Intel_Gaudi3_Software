#pragma once

#include "habana_graph.h"
#include "types.h"
#include "brain_data.h"
#include "strategy.h"
#include "passes/sram_management/pipeline_management/node_solver.h"

namespace gc::layered_brain
{
struct StrategyContainer
{
    StrategyContainer() = default;
    StrategyContainer(const StrategyPtr& strategy, const NodePtr& node, const MmeSolutionPtr& mmeSolution)
    {
        strategies.push_back(strategy);
        nodes.push_back(node);
        mmeSolutions.push_back(mmeSolution);
    }

    StrategyVector       strategies;    // Layered-brain strategies
    NodeVector           nodes;         // Solved nodes
    MmeSolutionContainer mmeSolutions;  // MME-brain strategies
};

// The key node solver is responsible to generate a list of strategies for a single op.
class KeyNodeSolver
{
public:
    explicit KeyNodeSolver(const HabanaGraph& graph, const NodePtr& keyNode, uint64_t maxTileSize)
    : m_graph(graph), m_keyNode(keyNode), m_maxTileSize(maxTileSize)
    {
    }

    virtual StrategyContainer getSlicingStrategies(const BundleViewContainerPtr& bundleViews,
                                                   const StrategyContainer&      existingStrategies) = 0;

    virtual ~KeyNodeSolver() {}

protected:
    void logStrategies(const StrategyContainer& strategies, const BundleViewContainerPtr& bundleViews) const;

    const HabanaGraph& m_graph;
    const NodePtr      m_keyNode;
    const uint64_t m_maxTileSize;  // Max SRAM budget per tile (bytes) - the solver is not allowed to provide a strategy
                                   // in which a tile size exceeds that number.
};

}  // namespace gc::layered_brain