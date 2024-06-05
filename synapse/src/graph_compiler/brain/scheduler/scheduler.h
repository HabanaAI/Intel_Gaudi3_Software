#pragma once

#include "habana_graph.h"
#include "layered_brain.h"

namespace gc::layered_brain
{
class SlicedNodesScheduler
{
public:
    explicit SlicedNodesScheduler(HabanaGraph& g) : m_graph(g) {}
    bool handleAllBundles(bool dryRun);
    bool scheduleBundle(const BundleNodes& bundleNodes, bool dryRun);

private:
    HabanaGraph& m_graph;

    Bundles getBundles() const;
    bool    alreadyScheduled(const BundleNodes& bundleNodes) const;
    void    scheduleMemsetNodes(const BundleNodes& bundleNodes) const;
    void    setOpIdxForMemset(const NodePtr& memsetNode) const;
};

}  // namespace gc::layered_brain