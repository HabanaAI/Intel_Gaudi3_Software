#include "memory_management.h"

#include "brain_data.h"
#include "habana_graph.h"
#include "bundle_collector.h"
#include "bundle_memory_manager.h"
#include "bundle_cache_manager.h"

using namespace gc::layered_brain;

bool MemoryManager::handleAllBundles()
{
    for (const auto& bundleIdxAndNodes : getBundles())
    {
        const auto* lbData = m_graph.getLayeredBrainData();
        HB_ASSERT_PTR(lbData);

        if (!lbData->isLayeredBrainBundle(bundleIdxAndNodes.first))
        {
            LOG_DEBUG(LAYERED_BRAIN,
                      "Bundle {} does not have layered brain data - skipping mem mgmt for it.",
                      bundleIdxAndNodes.first);
            continue;
        }

        bool res = manageBundleMemory(bundleIdxAndNodes.second);
        CHECK_RET_FALSE(res, "Failed to manage bundle {} memory.", bundleIdxAndNodes.first);
    }
    return true;
}

Bundles MemoryManager::getBundles() const
{
    BundleCollector bc {m_graph};
    return bc.getAllBundles();
}

bool MemoryManager::manageBundleMemory(const BundleNodes& bundleNodes)
{
    if (m_graph.getHALReader()->isCacheSupported())
    {
        LOG_DEBUG(LAYERED_BRAIN, "Cache system - using cache management");
        BundleCacheManager bcm {m_graph, bundleNodes};
        return bcm.setCacheDirectives();
    }
    else
    {
        LOG_DEBUG(LAYERED_BRAIN, "SRAM system - using SRAM management");
        BundleMemoryManager bmm {m_graph, bundleNodes};
        return bmm.placeTiles();
    }
}

bool bundleMemoryManagement(HabanaGraph& g)
{
    if (!GCFG_ENABLE_BUNDLE_MEMORY_MANAGEMENT.value() || BundleCollector::nofLBBundles(g) == 0)
    {
        LOG_DEBUG(LAYERED_BRAIN,
                  "Bundle memory management pass disabled. Mem-Management flag: {}, LB bundles: {}",
                  GCFG_ENABLE_BUNDLE_MEMORY_MANAGEMENT.value(),
                  BundleCollector::nofLBBundles(g));
        return true;
    }

    return MemoryManager(g).handleAllBundles();
}
