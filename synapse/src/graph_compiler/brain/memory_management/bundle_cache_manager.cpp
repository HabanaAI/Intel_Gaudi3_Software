#include "bundle_cache_manager.h"
#include "memory_management/bundle_memory_preprocessor.h"
#include "memory_management/cache_requirements_analyzer.h"
#include "memory_management/cache_requirements_profiler.h"
#include "node_cache_setter.h"
#include "brain_data.h"

using namespace gc::layered_brain;

BundleCacheManager::BundleCacheManager(HabanaGraph& graph, const BundleNodes& nodes)
: m_graph(graph),
  m_nodes(nodes),
  m_bundleIdx(findBundleIdx()),
  m_db(buildMemDB()),
  m_cacheState(cacheBudget()),
  m_bundleData(findBundleData())
{
}

BundleCacheState::Capacity BundleCacheManager::cacheBudget() const
{
    return m_graph.getHALReader()->getSRAMSizeInBytes() * GCFG_FRAGMENTATION_COMPENSATION_FACTOR.value();
}

bool BundleCacheManager::setCacheDirectives(bool dryRun)
{
    TempLogContextSetter logContext(fmt::format("CacheManagement bundle#{}", m_bundleIdx));

    for (size_t nodeIdx = 0; nodeIdx < m_nodes.size(); nodeIdx++)
    {
        bool success = setNodeCacheDirectives(nodeIdx);
        if (!success)
        {
            if (dryRun)
            {
                LOG_DEBUG(LB_CACHE_MNGR,
                          "Failed to set cache directives for operation {} ({})",
                          nodeIdx,
                          m_nodes[nodeIdx]->getNodeName());
                return false;
            }
            else
            {
                LOG_DEBUG_AND_PERF(LB_CACHE_MNGR,
                                   "Failed to set cache directives for node {}",
                                   m_nodes[nodeIdx]->getNodeName());
            }
        }
    }
    HB_ASSERT(m_cacheState.totalLive() == 0,
              "Undexpected un-freed cache at the end of the bundle cache allocation ({}B)",
              m_cacheState.totalLive());
    setBundleCacheUsage();
    return true;
}

bool BundleCacheManager::setNodeCacheDirectives(size_t nodeIdx)
{
    HB_ASSERT(nodeIdx < m_nodes.size(), "Unexpected node index: {} in bundle with {} nodes", nodeIdx, m_nodes.size());

    auto* reqAnalizer = getRequirementsAnalyzer();
    return getNodeCacheSetter()->setDirectives(nodeIdx, reqAnalizer);
}

CacheRequirementsAnalyzerIfc* BundleCacheManager::getRequirementsAnalyzer()
{
    if (!m_reqAnalyzer)
    {
        CacheRequirementProfilerPtr profiler {
            new CacheRequirementProfiler(m_graph, m_db, m_bundleData.getFinalStrategy())};
        m_reqAnalyzer.reset(new CacheRequirementsAnalyzer(profiler));
    }
    return m_reqAnalyzer.get();
}

NodeCacheSetterIfc* BundleCacheManager::getNodeCacheSetter()
{
    if (!m_nodeCacheSetter)
    {
        // In Cache based systems, the cache capacity is currently held in HAL under SRAM size. HCL reserved is already
        // excluded.
        m_nodeCacheSetter.reset(new NodeCacheSetter(m_graph,
                                                    m_nodes,
                                                    m_db,
                                                    m_cacheState,
                                                    m_bundleData.getPipelineDepth()));
    }
    return m_nodeCacheSetter.get();
}

MemoryUsageDB BundleCacheManager::buildMemDB() const
{
    return POCBundleMemoryPreProcessor(m_graph, m_nodes).buildMemUsageDB();
}

BundleData& BundleCacheManager::findBundleData()
{
    auto* lbData = m_graph.getLayeredBrainData();
    HB_ASSERT_PTR(lbData);

    return lbData->m_bundleData.at(m_bundleIdx);
}

void BundleCacheManager::setBundleCacheUsage()
{
    auto maxCacheUsage = m_cacheState.maxLiveCapacity();
    LOG_DEBUG(LB_CACHE_MNGR,
              "Max cache usage for bundle {}: {}B ({:.2}MB)",
              m_bundleIdx,
              maxCacheUsage,
              bToMb(maxCacheUsage));

    m_bundleData.setMaxCacheUsage(maxCacheUsage);
}

BundleIndex BundleCacheManager::findBundleIdx() const
{
    HB_ASSERT(!m_nodes.empty(), "Unexpected empty bundle");
    auto idx = getBundleIndex(m_nodes.front());
    HB_ASSERT(idx, "Unexpected unbundled node: {}", m_nodes.front()->getNodeName());
    return *idx;
}
