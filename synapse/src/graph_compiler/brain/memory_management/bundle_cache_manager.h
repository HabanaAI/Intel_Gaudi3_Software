#pragma once

#include "brain_data.h"
#include "habana_graph.h"
#include "layered_brain.h"
#include "cache_management_apis.h"
#include "bundle_cache_state.h"
#include "memory_usage_db.h"

namespace gc::layered_brain
{
class BundleCacheManager
{
public:
    BundleCacheManager(HabanaGraph& graph, const BundleNodes& nodes);

    // Sets cache directive to all bundle nodes annotations
    // In dry run mode, returns false on the first cache directive assignment failure.
    // Otherwise, always succeeds but may emit warning on directive assignment failure.
    bool setCacheDirectives(bool dryRun = false);

private:
    HabanaGraph&        m_graph;
    const BundleNodes&  m_nodes;
    const BundleIndex   m_bundleIdx;
    const MemoryUsageDB m_db;
    BundleCacheState    m_cacheState;
    BundleData&         m_bundleData;

    std::unique_ptr<CacheRequirementsAnalyzerIfc> m_reqAnalyzer;
    std::unique_ptr<NodeCacheSetterIfc>           m_nodeCacheSetter;

    BundleCacheState::Capacity cacheBudget() const;

    BundleIndex   findBundleIdx() const;
    MemoryUsageDB buildMemDB() const;
    BundleData&   findBundleData();

    // Sets the cache directives for each of the node's accesses.
    // Returns whether all accesses were cached as needed.
    bool setNodeCacheDirectives(size_t nodeIdx);

    NodeCacheSetterIfc*           getNodeCacheSetter();
    CacheRequirementsAnalyzerIfc* getRequirementsAnalyzer();

    void setBundleCacheUsage();
};

}  // namespace gc::layered_brain