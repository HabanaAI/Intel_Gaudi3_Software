#pragma once

#include "habana_graph.h"
#include "layered_brain.h"

namespace gc::layered_brain
{
class MemoryManager
{
public:
    explicit MemoryManager(HabanaGraph& g) : m_graph(g) {}
    bool handleAllBundles();

private:
    HabanaGraph& m_graph;

    Bundles getBundles() const;
    bool    manageBundleMemory(const BundleNodes& bundleNodes);
};

}  // namespace gc::layered_brain