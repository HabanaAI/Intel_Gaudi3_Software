#include "bundle_collector.h"
#include "brain_data.h"

using namespace gc::layered_brain;

BundleCollector::BundleCollector(HabanaGraph& graph) : m_sortedNodes(graph.getNumNodes())
{
    const auto& sortedNodes = graph.getExeSortedNodes();
    std::copy(sortedNodes.begin(), sortedNodes.end(), m_sortedNodes.begin());
}

Bundles BundleCollector::getAllBundles() const
{
    Bundles bundles;
    for (const NodePtr& n : m_sortedNodes)
    {
        const auto& bundleIdx = getBundleIndex(n);
        if (bundleIdx)
        {
            // The collector returns the nodes in the order they appear in a execution schedule sorted graph. The order
            // is important for memory analysis and management.
            bundles[*bundleIdx].push_back(n);
        }
    }
    return bundles;
}

size_t BundleCollector::nofLBBundles(const HabanaGraph& g)
{
    if (!g.getLayeredBrainData()) return 0;
    return g.getLayeredBrainData()->m_bundleData.size();
}