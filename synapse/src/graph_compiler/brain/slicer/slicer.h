#pragma once

#include "habana_graph.h"
#include "perforator.h"
#include "common_tile_size_calculator.h"
#include "types.h"
#include "brain_data.h"
#include "strategy.h"
#include "key_node_solver.h"
#include "strategy_inflator.h"
#include "slicing_details.h"

namespace gc::layered_brain
{
// The slicer uses the outcome of the ISMR and Bundler layers and further interacts with the MME brain to get list
// of possible strategies of slicing for the bundle, so they will execute in separate parts and can later be scheduled
// in parallel across different engines.
// Then, it decides for every dimension whether it should be sliced and what should be the granularity of slicing for
// that dimension. Once slice sizes are selected, the slicer is responsible to create a graph representation of the
// sliced bundle which can be further processed by the following scheduling and cache/mem management layers.
class Slicer
{
public:
    static constexpr uint64_t MAX_TILE_SIZE               = 4 * 1024 * 1024;
    static constexpr uint64_t MIN_COMMON_DIM_FOR_PARTIALS = 2 * 1024;

    Slicer(const HabanaGraph& g,
           const BundleIdx    bundleIdx,
           const NodeVector&  bundleNodes,
           SlicingPolicy      slicingPolicy           = SlicingPolicy::ENOUGH_REUSE,
           uint64_t           maxTileSize             = MAX_TILE_SIZE,
           uint64_t           minCommonDimForPartials = MIN_COMMON_DIM_FOR_PARTIALS);

    // Forward progress APIs
    HabanaGraphPtr getSlicedBundle();

    // Iterative APIs
    StrategyVector getStrategies() const;
    HabanaGraphPtr sliceBundleByStrategy(const StrategyPtr& strategy, bool dryRun = false) const;
    bool inflateStrategy(InflationType inflationType, const StrategyPtr& strategy, const NodePtr& node = nullptr) const;
    ConstSlicingDetailsPtr getSlicingDetails(const StrategyPtr& strategy) const;

private:
    std::pair<TileSizePerTensor, TileSizePerNode> getMinCommonTilesSizes() const;
    BundleViewContainerPtr                        getInitialBundleViews() const;
    StrategyVector                                generateInitialStrategies() const;
    StrategyVector filterInternalConflictStrategies(const StrategyVector& origStrategies) const;
    StrategyPtr                                   calcSlicingStrategy() const;

    HabanaGraphPtr getSlicedGraphFromStrategy(const StrategyPtr& strategy, bool dryRun) const;

    std::unique_ptr<KeyNodeSolver> getKeyNodeSolver(const NodePtr& keyNode) const;

    const HabanaGraph&  m_graph;
    const BundleIdx     m_bundleIdx;
    const NodeVector    m_bundleNodes;
    const NodeVector    m_keyNodes;
    const SlicingPolicy m_slicingPolicy;
    const uint64_t      m_maxTileSize;         // Max SRAM budget per tile (bytes).
    const uint64_t m_minCommonDimForPartials;  // Min common dim to use when slicing on common dimension (elements).

    BundleViewContainerPtr            m_bundleViews;
    std::unique_ptr<StrategyInflator> m_inflator;
    std::unique_ptr<Perforator>       m_perforator;
};

}  // namespace gc::layered_brain