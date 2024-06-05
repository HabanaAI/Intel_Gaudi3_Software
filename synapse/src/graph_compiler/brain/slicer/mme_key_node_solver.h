#pragma once

#include "habana_graph.h"
#include "types.h"
#include "brain_data.h"
#include "strategy.h"
#include "key_node_solver.h"

namespace gc::layered_brain
{
// The MME key node solver is responsible to generate a list of MME strategies for a single MME op.
// As the list needs to be acceptable by all MME ops in bundle, for repeated MME ops in bundle, it
// gets a list of options from previous MME ops.
// The solver for current op must find solutions that are full multiple (1 or more) of existing
// solutions from previous MME ops, to ensure compatibility.
// Some dimensions may be free of constrain (when they are not part of previous MME ops).
// Every MME strategy defines tile size for the 3 I/O tensors, i.e. for every dimension
// of every tensor of the op, MME solver provides a multiplier (scalar value) for every strategy.
class MMEKeyNodeSolver : public KeyNodeSolver
{
public:
    explicit MMEKeyNodeSolver(const HabanaGraph& graph,
                              const NodePtr&     keyNode,
                              uint64_t           maxTileSize,
                              uint64_t           minCommonDimForPartials)
    : KeyNodeSolver(graph, keyNode, maxTileSize), m_minCommonDimForPartials(minCommonDimForPartials)
    {
    }

    StrategyContainer getSlicingStrategies(const BundleViewContainerPtr& bundleViews,
                                           const StrategyContainer&      existingStrategies) override;

protected:
    StrategyContainer createStrategiesFromMmeSolutions(const BundleViewContainerPtr& bundleViews,
                                                       const MmeSolutionContainer&   mmeSolutions,
                                                       const StrategyContainer&      existingStrategies) const;

    const uint64_t m_minCommonDimForPartials;  // Min common dim to use when slicing on common dimension (elements).
};

}  // namespace gc::layered_brain