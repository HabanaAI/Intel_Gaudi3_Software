#pragma once

#include "layered_brain.h"
#include "bundle_memory_manager_interfaces.h"

namespace gc::layered_brain
{
// The heart of the memory manager - applys the memory management policy over the given bundle.
class BundleMemoryManager
{
public:
    BundleMemoryManager(HabanaGraph& graph, const BundleNodes&);
    bool placeTiles();

private:
    HabanaGraph&  m_graph;
    BundleNodes   m_nodes;
    MemoryUsageDB m_db;

    // Generate placement DB
    void preProcess();
    // Process each step and set placement and spill fill directives in the DB
    void setPlacementDirectives();
    // Execute the directives set in the DB (graph change)
    bool executeDirectives();

    std::unique_ptr<BundleMemoryInsulator>         getInsulator(const BundleNodes& bundleNodes);
    std::unique_ptr<BundleMemoryPreProcessor>      getPreprocessor(const BundleNodes& bundleNodes);
    std::unique_ptr<BundleSRAMAllocator>           getAllocator();
    std::unique_ptr<BundleMemoryPlacer>            getPlacer();
    std::unique_ptr<BundleMemoryDirectiveExecutor> getExecutor();
};

}  // namespace gc::layered_brain