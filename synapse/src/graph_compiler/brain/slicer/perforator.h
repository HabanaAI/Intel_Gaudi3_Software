#pragma once

#include "bundle_view.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "perforation_bvd_candidates_finder.h"
#include "perforation_bvd_selector.h"
#include "strategy.h"
#include "types.h"

namespace gc::layered_brain
{
// The perforator is responsible to select a BVD for DCORE partition for every node in the bundle.
// The alogorithm is split to 2 stages:
// 1) Find perforation candidates per node.
// 2) Select perforation BVD for each node.
class Perforator
{
public:
    Perforator(const HabanaGraph& graph, const NodeVector& bundleNodes, const BundleViewContainerPtr& bundleViews)
    : m_bundleNodes(bundleNodes),
      m_bundleViews(bundleViews),
      m_numDcores(graph.getHALReader()->getNumDcores()),
      m_bvdCandidatesFinder(bundleNodes, bundleViews, m_numDcores),
      m_bvdSelector(graph, bundleNodes)
    {
    }

    void selectPerforationForStrategy(const StrategyPtr& strategy) const;

private:
    const NodeVector               m_bundleNodes;
    const BundleViewContainerPtr   m_bundleViews;
    const unsigned                 m_numDcores;
    PerforationBVDCandidatesFinder m_bvdCandidatesFinder;
    PerforationBVDSelector         m_bvdSelector;
};

}  // namespace gc::layered_brain