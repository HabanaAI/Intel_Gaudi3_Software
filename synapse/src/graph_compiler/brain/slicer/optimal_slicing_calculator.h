#pragma once

#include "strategy.h"
#include "brain_data.h"

namespace gc::layered_brain
{
// The optimal slicing calculator is responsible to select a strategy that optimizes the required metrics (defined in
// the global policy). It decides on multipliers such that tiles sizes are not “too large” and are expected to fit in
// SRAM after graph slicing and placement. High level idea: choose strategy and inflate tiles such that every tile is
// maxed below the thresholds.
class OptimalSlicingCalculator
{
public:
    explicit OptimalSlicingCalculator(const HabanaGraph& graph,
                                      SlicingPolicy      slicingPolicy,
                                      uint64_t           maxTileSize,
                                      uint64_t           maxBwGBps      = 0, /* no limit */
                                      uint64_t           maxNumOfSlices = 128)
    : m_graph(graph),
      m_slicingPolicy(slicingPolicy),
      m_maxTileSize(maxTileSize),
      m_maxBwGBps(maxBwGBps),
      m_maxNumOfSlices(maxNumOfSlices)
    {
    }
    StrategyPtr getOptimalStrategy(const BundleViewContainerPtr& bundleViews,
                                   const StrategyVector&         strategies,
                                   const NodeVector&             bundleNodes);

private:
    StrategyPtr selectStrategy(const StrategyVector& strategies) const;
    void        inflateStrategy(const BundleViewContainerPtr& bundleViews, const StrategyPtr& strategy) const;
    bool        isStrategyValid(const BundleViewContainerPtr& bundleViews, const StrategyPtr& strategy) const;
    unsigned    getNumOfSlicedBVDs(const BundleViewContainerPtr& bundleViews, const StrategyPtr& strategy) const;
    uint64_t    getNumOfSlices(const BundleViewContainerPtr& bundleViews, const StrategyPtr& strategy) const;

    const HabanaGraph&  m_graph;
    const SlicingPolicy m_slicingPolicy;
    const uint64_t      m_maxTileSize;
    const uint64_t      m_maxBwGBps;
    const uint64_t      m_maxNumOfSlices;
};

}  // namespace gc::layered_brain