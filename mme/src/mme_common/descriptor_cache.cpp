#include "include/mme_common/descriptor_cache.h"
#include "include/gaudi2/mme_descriptor_generator.h"
#include "include/gaudi3/mme_descriptor_generator.h"
#include "mme_common/mme_descriptor_cache_utils.h"
#include "utils/logger.h"

using namespace MmeCommon;

template<typename KeyType, typename ValueType>
std::vector<std::string> DescriptorsCache<KeyType, ValueType>::getDebugInfo()
{
    size_t cachesHit = 0;
    size_t descsGenerated = 0;
    size_t descsMapSize = 0;
    size_t cacheSizeLimit = 0;
    {
        std::unique_lock lock(m_mutex);
        cachesHit = m_cachesHit;
        descsGenerated = m_descsGenerated;
        descsMapSize = m_descCacheMap.size();
        cacheSizeLimit = m_cacheSizeLimit;
    }
    return {fmt::format("Actual cache size {}, Max cache size {}", descsMapSize, cacheSizeLimit),
            fmt::format("Number of descriptors generated from cache is {}, out of {}",
                        cachesHit,
                        cachesHit + descsGenerated)};
}

// instantiate getDebugInfo for gaudi2/gaudi3
template std::vector<std::string> DescriptorsCache<MmeCommon::MmeLayerParams, gaudi3::MmeActivation>::getDebugInfo();
template std::vector<std::string> DescriptorsCache<MmeCommon::MmeLayerParams, Gaudi2::MmeActivation>::getDebugInfo();
