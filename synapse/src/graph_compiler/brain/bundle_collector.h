#pragma once
#include "habana_graph.h"
#include "layered_brain.h"

namespace gc::layered_brain
{
class BundleCollector
{
public:
    explicit BundleCollector(HabanaGraph& graph);
    Bundles getAllBundles() const;

    static size_t nofLBBundles(const HabanaGraph& g);

private:
    NodeVector m_sortedNodes;
};

}  // namespace gc::layered_brain