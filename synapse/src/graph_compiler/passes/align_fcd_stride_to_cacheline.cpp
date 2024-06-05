#include "cacheline_aligner.h"
#include "brain_data.h"
#include "bundle_collector.h"

using namespace gc::layered_brain;
enum class AlignMode
{
    DISABLED = 0,
    FULL_CACHELINE,
    SMALL_FCD_HALF_CACHELINE,
};

static bool isEnabled()
{
    const auto alignMode = static_cast<AlignMode>(GCFG_ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE.value());
    switch (alignMode)
    {
        case AlignMode::DISABLED:
        {
            return false;
        }
        case AlignMode::FULL_CACHELINE:
        case AlignMode::SMALL_FCD_HALF_CACHELINE:
        {
            return true;
        }
        default:
        {
            HB_ASSERT(false, "Unknown align mode: {}", alignMode);
            return false;
        }
    }
}

static std::unique_ptr<CachelineAligner> getFCDStrideCachelineAligner(HabanaGraph&                  g,
                                                                      const BPTClonePersistenceMap& bptClonePersistence)
{
    const auto alignMode = static_cast<AlignMode>(GCFG_ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE.value());
    switch (alignMode)
    {
        case AlignMode::DISABLED:
        {
            HB_ASSERT(false, "FCD stride cacheline alignment is disabled");
            return nullptr;
        }
        case AlignMode::FULL_CACHELINE:
        {
            return std::make_unique<CachelineAligner>(g, bptClonePersistence);
        }
        case AlignMode::SMALL_FCD_HALF_CACHELINE:
        {
            return std::make_unique<SmallFCDHalfCachelineAligner>(g, bptClonePersistence);
        }
        default:
        {
            HB_ASSERT(false, "Unknown align mode: {}", alignMode);
            return nullptr;
        }
    }
}

static BPTClonePersistenceMap aggregateBPTClonePersistence(const LayeredBrainData& brainData)
{
    BPTClonePersistenceMap bptClonePersistence;
    for (const auto& bundleDataItem : brainData.m_bundleData)
    {
        bptClonePersistence.insert(bundleDataItem.second.getBPTClonePersistenceMap().begin(),
                                   bundleDataItem.second.getBPTClonePersistenceMap().end());
    }
    return bptClonePersistence;
}

bool alignBPTFCDStrideToCacheLine(HabanaGraph& g)
{
    if (isEnabled() && BundleCollector::nofLBBundles(g) > 0)
    {
        const auto pLayeredBrainData = g.getLayeredBrainData();
        HB_ASSERT_PTR(pLayeredBrainData);
        const auto bptClonePersistence(aggregateBPTClonePersistence(*pLayeredBrainData));
        auto       aligner = getFCDStrideCachelineAligner(g, bptClonePersistence);
        HB_ASSERT_PTR(aligner);
        aligner->run();
    }
    return true;
}
