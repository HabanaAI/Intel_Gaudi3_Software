#pragma once

#include "habana_graph.h"
#include "types.h"
#include "brain_data.h"
#include "strategy.h"
#include "key_node_solver.h"
#include "sram_management/pipeline_management/node_solver.h"

namespace gc::layered_brain
{
// Temp implementation using existing PM NodeSolver as MME brain.
class LegacyMMEKeyNodeSolver : public KeyNodeSolver
{
public:
    explicit LegacyMMEKeyNodeSolver(const HabanaGraph& graph,
                                    const NodePtr&     keyNode,
                                    uint64_t           maxTileSize,
                                    uint64_t           minCommonDimForPartials)
    : KeyNodeSolver(graph, keyNode, maxTileSize), m_minCommonDimForPartials(minCommonDimForPartials)
    {
    }

    StrategyContainer getSlicingStrategies(const BundleViewContainerPtr& bundleViews,
                                           const StrategyContainer&      existingStrategies) override;

protected:
    BundleSolutionConstraints prepareMmeBrainRequest(const BundleViewContainerPtr& bundleViews) const;
    NodeStrategyPtr           sendRequestToMmeBrain(const BundleSolutionConstraints& constraints) const;
    StrategyContainer         processMmeBrainResponse(const BundleViewContainerPtr& bundleViews,
                                                      const NodeStrategyPtr&        nodeStrategy) const;
    StrategyContainer         handleNoStrategy(const BundleViewContainerPtr& bundleViews) const;

    const uint64_t m_minCommonDimForPartials;  // Min common dim to use when slicing on common dimension (elements).
};

}  // namespace gc::layered_brain