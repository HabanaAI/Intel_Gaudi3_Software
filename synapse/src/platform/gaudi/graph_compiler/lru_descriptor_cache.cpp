#include <vector>

#include "descriptor_generator.h"
#include "log_manager.h"
#include "habana_global_conf.h"

#include "include/mme_common/mme_common_enum.h"
#include "include/gaudi/gaudi_utils.h"
#include "habana_global_conf.h"
#include "mme_logger.h"

namespace gaudi
{
static MmeCommon::MmeLayerParams createKeyToCache(const MmeCommon::MmeLayerParams& params)
{
    // Create key to the descriptor cache

    MmeCommon::MmeLayerParams keyParams = params;
    /// TODO create general key with addition don't cares params (like paddingValue)
    keyParams.tracing.ctxId      = 0;
    keyParams.controls.atomicAdd = false;

    return keyParams;
}

static void setActivationParams(const MmeCommon::MmeLayerParams& params, std::list<MmeActivation>& activations)
{
    // Fix activation params that weren't inserted to the key according the the original params

    MmeCommon::MmeTensorView c = params.y;
    switch (params.opType)
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
            c = params.y;
            break;
        case MmeCommon::e_mme_dedx:
            c = params.x;
            break;
        case MmeCommon::e_mme_dedw:
            c = params.w;
            break;
        default:
            HB_ASSERT(false, "unsupported operation type");
    }

    bool                    reductionEn = params.controls.atomicAdd;
    MmeCommon::EMmeDataType dt          = c.elementType;
    MmeCommon::RoundingMode rm          = params.controls.roundingMode;
    uint32_t                dw          = getUserDataVal(reductionEn, dt, rm);

    for (auto& act : activations)
    {
        // Fix ctxId
        uint16_t ctxId                     = params.tracing.ctxId;
        act.getDesc(0).perfEvtL[0].value  = ctxId;
        act.getDesc(0).perfEvtL[1].value  = ctxId;
        act.getDesc(0).perfEvtO[0].value  = ctxId;
        act.getDesc(0).perfEvtO[1].value  = ctxId;
        act.getDesc(0).perfEvtS.value     = ctxId;
        act.getDesc(1).perfEvtL[0].value = ctxId;
        act.getDesc(1).perfEvtL[1].value = ctxId;
        act.getDesc(1).perfEvtO[0].value = ctxId;
        act.getDesc(1).perfEvtO[1].value = ctxId;
        act.getDesc(1).perfEvtS.value    = ctxId;

        // Fix attomicAdd
        act.getDesc(0).axiUserData.dw  = dw;
        act.getDesc(1).axiUserData.dw = dw;
    }
}

void DescriptorsCache::DescriptorsCacheInit(int size)
{
    m_csize = size;
    m_DesCacheMap.clear();
    m_dqDesCache.clear();
    m_cacheDesCount  = 0;
    m_newDesGenCount = 0;
    if (size == 0)
    {
        m_DesCacheEnabled = false;
        LOG_INFO(MME_DESC_CACHE, "DescriptorsCache disable");
    }
    else
    {
        m_DesCacheEnabled = true;
        LOG_INFO(MME_DESC_CACHE, "DescriptorsCache enabled, DescriptorsCache max size {}", size);
    }
}
bool DescriptorsCache::isElementInDesCache(const MmeCommon::MmeLayerParams& params, descriptorCacheIt& it)
{
    MmeCommon::MmeLayerParams              keyParams = createKeyToCache(params);
    std::unique_lock<std::recursive_mutex> lk(m_mutex);
    // multithreading protection
    it = m_DesCacheMap.find(keyParams);
    return (it != m_DesCacheMap.end());
}

static std::string nodeTypeToStr(MmeCommon::EMmeOpType opType)
{
    switch (opType)
    {
        case MmeCommon::e_mme_fwd:
            return "fwd";
        case MmeCommon::e_mme_dedx:
            return "dedx";
        case MmeCommon::e_mme_dedw:
            return "dedw";
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
            return "gemm";
        default:
            return "unknown";
    }
}

static void compareActivations(const std::list<gaudi::MmeActivation>& newStackActivations,
                               const std::list<gaudi::MmeActivation>& oldStackActivations,
                               const MmeCommon::MmeLayerParams&       layerParams,
                               std::string                            genCodeStr)
{
    compareActivationSizes(newStackActivations, oldStackActivations, genCodeStr);

    if (newStackActivations.size() == oldStackActivations.size())
    {
        auto a_it = newStackActivations.begin();
        auto b_it = oldStackActivations.begin();
        for (int actIdx = 0; (actIdx < newStackActivations.size()); actIdx++)
        {
            compareDescriptors(a_it->getDesc(0), b_it->getDesc(0), "act" + std::to_string(actIdx) + "_desc0");
            compareDescriptors(a_it->getDesc(1), b_it->getDesc(1), "act" + std::to_string(actIdx) + "_desc1");
            b_it = std::next(b_it, 1);
            a_it = std::next(a_it, 1);
        }
    }
    else
    {
        LOG_INFO(MME_DESC_CACHE, "Cannot compare the activation lists because their sizes are different");
    }
}

void DescriptorsCache::generateDescriptorsCache(const MmeCommon::MmeLayerParams& params,
                                                std::list<MmeActivation>&        activations)
{
    // multithreading protection
    std::unique_lock<std::recursive_mutex> lk(m_mutex);
    descriptorCacheIt                      cachedDesIter;
    bool                                   isInCache = isElementInDesCache(params, cachedDesIter);
    if (!isInCache)
    {
        // Update new descriptors generation count
        m_newDesGenCount++;
        HB_ASSERT(cachedDesIter == m_DesCacheMap.end(), "Probable threads race condition");
        lk.unlock();
        // not present in cache, generate new descriptors
        bool useOldDescriptors = GCFG_MME_ENABLE_USE_OLD_DESCRIPTORS.value() && !params.isGemmOperation() &&
                                 !MmeDescriptorGenerator::isZeroCD(params);
        bool compareAllNewDescriptors = GCFG_MME_ENABLE_COMPARE_NEW_VS_OLD_DESCRIPTORS.value();

        if (useOldDescriptors)
        {
            generateDescriptors(params, activations);
        }
        else
        {
            auto descGenerator = gaudi::MmeDescriptorGenerator::createMmeDescGenerator(params);
            descGenerator->mmeGenerateActivations();
            activations = descGenerator->getMmeActivations();
            MmeLogger mmeLogger;
            mmeLogger.printDebugInfoGaudi(&*descGenerator);
        }

        if (compareAllNewDescriptors)  // if set, we need to create the other descs as well
        {
            std::list<gaudi::MmeActivation> refActivations;

            if (useOldDescriptors)
            {
                auto descGenerator = gaudi::MmeDescriptorGenerator::createMmeDescGenerator(params);
                descGenerator->mmeGenerateActivations();
                refActivations = descGenerator->getMmeActivations();
            }
            else
            {
                generateDescriptors(params, refActivations);
            }

            // Compare. The new activations are always the first, old activations always the second
            std::string genCodeStr = nodeTypeToStr(params.opType);
            compareActivations(useOldDescriptors ? refActivations : activations,
                               useOldDescriptors ? activations : refActivations,
                               params,
                               genCodeStr);
        }

        lk.lock();
        MmeCommon::MmeLayerParams keyParams = createKeyToCache(params);
        // Not present in cache, insert it
        auto retPair = m_DesCacheMap.insert(std::make_pair(keyParams, activations));
        if (retPair.second == false)
        {
            // Insertion failed. Key was present
            m_dqDesCache.remove(retPair.first);
        }
        // update cachedDesIter for later moving it to front of m_dqDesCache
        cachedDesIter = retPair.first;
        // cache is full
        if (m_dqDesCache.size() == m_csize)
        {
            // delete least recently used element
            auto last = m_dqDesCache.back();
            // Pops the last elmeent
            m_dqDesCache.pop_back();
            // Erase in cache map
            m_DesCacheMap.erase(last);
            LOG_INFO(MME_DESC_CACHE, "Cache is full, delete least recently used element");
        }
    }
    else
    {
        HB_ASSERT(cachedDesIter != m_DesCacheMap.end(), "impropper iterator, Probable threads race condition");
        // Update descriptors generation from cache count
        m_cacheDesCount++;
        // present in cache
        m_dqDesCache.remove(cachedDesIter);
        // Similar descriptor found in cache, generate descriptors using the cache descriptor
        activations = cachedDesIter->second;
        setActivationParams(params, activations);
    }
    // update reference
    m_dqDesCache.push_front(cachedDesIter);
    HB_ASSERT(m_dqDesCache.size() == m_DesCacheMap.size(), "Descriptors size mismatch");
}
void DescriptorsCache::printDesCacheStats()
{
    if (!LOG_LEVEL_AT_LEAST_INFO(MME_DESC_CACHE)) return;

    // multithreading protection
    std::unique_lock<std::recursive_mutex> lk(m_mutex);
    if (isDesCacheEnabled())
    {
        uint32_t totalDesGenCount = m_cacheDesCount + m_newDesGenCount;
        LOG_INFO(MME_DESC_CACHE, "Statistics:");
        LOG_INFO(MME_DESC_CACHE, "Actual cache size {}, Max cache size {}", m_DesCacheMap.size(), getDesCacheMaxSize());
        LOG_INFO(MME_DESC_CACHE,
                 "Descriptors generation from cache count - {} out of - {}",
                 m_cacheDesCount,
                 totalDesGenCount);
        LOG_INFO(MME_DESC_CACHE,
                 "New descriptors generation count - {} out of - {}",
                 m_newDesGenCount,
                 totalDesGenCount);
    }
}
DescriptorsCache::DescriptorsCache()
{
    if (GCFG_ENABLE_MME_DESCRIPTOR_CACHE.value())
    {
        DescriptorsCacheInit(GCFG_MME_DESCRIPTORS_CACHE_SIZE.value());
    }
}
}
