#include "include/mme_common/mme_common_enum.h"
#include "mme_unit_test.h"
#include "mme_common/mme_descriptor_cache_utils.h"
#include "include/gaudi2/mme_descriptor_generator.h"
#include <thread>
#include <atomic>

class DescriptorCacheForTest : public MmeCommon::DescriptorsCache<MmeCommon::MmeLayerParams, Gaudi2::MmeActivation>
{
public:
    DescriptorCacheForTest(size_t cacheSizeLimit) : DescriptorsCache(cacheSizeLimit) {}
    bool isEnabled() const { return DescriptorsCache::isEnabled(); }
    size_t size() const { return DescriptorsCache::size(); }
};

class MmeGaudi2DescriptorCacheTest : public MMEUnitTest
{
};

TEST_F(MmeGaudi2DescriptorCacheTest, mme_descriptor_cache_zero_sized)
{
    MmeCommon::MmeLayerParams params(MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2));
    Gaudi2::ActivationVec activations = {Gaudi2::MmeActivation(2)};

    DescriptorCacheForTest emptyCache(0);
    ASSERT_FALSE(emptyCache.isEnabled()) << "cache isn't supposed to be enabled";
    ASSERT_FALSE(emptyCache.add(params, activations)) << "added an entry to a zero sized cache";
}

TEST_F(MmeGaudi2DescriptorCacheTest, mme_descriptor_cache_check_evictions)
{
    MmeCommon::MmeLayerParams params(MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2));
    Gaudi2::ActivationVec activations = {Gaudi2::MmeActivation(1)};
    activations[0].fcdView.viewBase = 10;
    activations[0].fcdView.viewOrigSize = 100;
    activations[0].fcdView.viewSize = 3;

    constexpr unsigned CACHE_SIZE = 1000;
    DescriptorCacheForTest cache(CACHE_SIZE);

    // adding first entry to the cache
    ASSERT_TRUE(cache.isEnabled()) << "cache is not enabled";
    ASSERT_TRUE(cache.add(params, activations)) << "failed to add entry to cache";
    ASSERT_TRUE(cache.size() == 1) << "wrong cache size";
    ASSERT_TRUE(cache.contains(params)) << "inserted key not in cache";
    ASSERT_TRUE(cache.get(params)->at(0).fcdView == activations[0].fcdView) << "inserted value not in cache";
    // verify we cloned the data and roi shared pointers
    ASSERT_FALSE(cache.get(params)->data() == activations.data()) << "didn't make a deep copy of activations";
    ASSERT_FALSE(cache.get(params)->data()->roiX.subRois.get() == activations.data()->roiX.subRois.get())
        << "didn't make a deep copy of subRois";

    // insert new different key,value pairs to cache
    for (int numOfAdded = 1; numOfAdded < CACHE_SIZE + 5; numOfAdded++)
    {
        params.spBase = numOfAdded * 10;
        activations[0].numSignals = numOfAdded;
        ASSERT_FALSE(cache.contains(params)) << "spBase is ignored by equality operator";
        ASSERT_TRUE(cache.add(params, activations)) << "failed to add new entry to cache";
        ASSERT_TRUE(cache.get(params)->at(0).numSignals == numOfAdded) << "wrong value found in cache";
        if (numOfAdded >= CACHE_SIZE)
        {
            ASSERT_TRUE(cache.size() == CACHE_SIZE) << "unexpected cache size after new entry addition to a full cache";
        }
        else
        {
            ASSERT_TRUE(cache.size() == numOfAdded + 1)
                << "unexpected cache size after new entry addition to non full cache";
        };
    }
    // check eviction worked correctly
    for (int numOfAdded = 0; numOfAdded < 5; numOfAdded++)
    {
        params.spBase = numOfAdded * 10;
        ASSERT_FALSE(cache.contains(params)) << "key with spBase: " << params.spBase << " should be evicted ";
    }
    params.spBase = 50;
    ASSERT_TRUE(cache.contains(params)) << "entry shouldn't be evicted according to LRU logic but was evicted";
    cache.get(params);
    params.spBase = 155;
    ASSERT_TRUE(cache.add(params, activations)) << "failed to add existing entry to cache";
    ASSERT_TRUE(cache.contains(params)) << "newly added entry is not in cache";
    ASSERT_TRUE(cache.size() == CACHE_SIZE) << "wrong cache size after fill";
}

TEST_F(MmeGaudi2DescriptorCacheTest, mme_descriptor_cache_multi_threaded)
{
    constexpr unsigned CACHE_SIZE = 1000;
    DescriptorCacheForTest cache(CACHE_SIZE);

    auto threadOp = [&cache]() {
        static std::atomic<unsigned> index;
        unsigned threadIndex = index.fetch_add(1);
        MmeCommon::MmeLayerParams threadParams = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2);
        Gaudi2::ActivationVec threadActivations = {Gaudi2::MmeActivation(1)};
        threadParams.spSize = threadIndex * 9;
        threadActivations[0].numSignals = threadIndex;
        ASSERT_TRUE(cache.add(threadParams, threadActivations)) << "failed to add new entry to cache";
        auto activationsPtr = cache.get(threadParams);
        ASSERT_TRUE(activationsPtr->at(0).numSignals == threadIndex) << "added entry value is missing from cache";
    };

    // check concurrent access
    std::vector<std::thread> threads;
    threads.reserve(CACHE_SIZE);

    for (int i = 0; i < CACHE_SIZE; ++i)
    {
        threads.emplace_back(threadOp);
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    MmeCommon::MmeLayerParams params = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2);
    for (int i = 0; i < CACHE_SIZE; ++i)
    {
        params.spSize = i * 9;
        ASSERT_TRUE(cache.contains(params)) << "added entry key is missing from cache";
        ASSERT_TRUE(cache.get(params)->at(0).numSignals == i) << "added entry value is missing from cache";
    }
}

TEST_F(MmeGaudi2DescriptorCacheTest, mme_descriptor_cache_no_eviction)
{
    MmeCommon::MmeLayerParams params(MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2));
    Gaudi2::ActivationVec activations = {Gaudi2::MmeActivation(2)};
    activations[0].fcdView.viewBase = 10;
    activations[0].fcdView.viewOrigSize = 100;
    activations[0].fcdView.viewSize = 3;

    constexpr unsigned CACHE_SIZE = 10000;
    DescriptorCacheForTest cache(CACHE_SIZE);

    ASSERT_TRUE(cache.size() == 0) << "initial cache should be empty";

    // filling the cache and then checking same key re-addition fails
    ASSERT_TRUE(cache.add(params, activations)) << "failed to add entry to cache";
    for (int numOfAdded = 1; numOfAdded < CACHE_SIZE; numOfAdded++)
    {
        params.spBase = numOfAdded * 10;
        ASSERT_FALSE(cache.contains(params)) << "spBase is ignored by equality operator";
        ASSERT_TRUE(cache.add(params, activations)) << "failed to add entry to cache";
        ASSERT_TRUE(cache.size() == numOfAdded + 1) << "unexpected cache size";
    }
    ASSERT_FALSE(cache.add(params, activations)) << "added existing entry to cache";
    for (int numOfAdded = 1; numOfAdded <= 1; numOfAdded++)
    {
        params.spBase = numOfAdded * 10;
        ASSERT_TRUE(cache.contains(params)) << "entry key is missing from cache";
        ASSERT_FALSE(cache.add(params, activations)) << "added existing entry to cache";
        ASSERT_TRUE(cache.size() == CACHE_SIZE) << "wrong cache size after fill";
    }
    params.spBase = 0;
    ASSERT_TRUE(cache.contains(params)) << "first entry is missing from cache";
}

TEST_F(MmeGaudi2DescriptorCacheTest, mme_descriptor_cache_eviction_ownership_claim)
{
    MmeCommon::MmeLayerParams oldParams(MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2));
    Gaudi2::ActivationVec oldActivations = {Gaudi2::MmeActivation(2)};

    constexpr unsigned CACHE_SIZE = 1;
    DescriptorCacheForTest cache(CACHE_SIZE);

    ASSERT_TRUE(cache.size() == 0) << "initial cache should be empty";

    ASSERT_TRUE(cache.add(oldParams, oldActivations)) << "failed to add first entry to cache";
    auto oldActivationsPtr = cache.get(oldParams);
    ASSERT_TRUE(cache.size() == 1 && cache.contains(oldParams)) << "cache is missing first entry";

    MmeCommon::MmeLayerParams newParams(MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2));
    Gaudi2::ActivationVec newActivations = {Gaudi2::MmeActivation(6)};
    newParams.opType = MmeCommon::e_mme_dedw;

    ASSERT_TRUE(cache.add(newParams, newActivations)) << "failed to add second entry to cache";
    auto newActivationsPtr = cache.get(newParams);
    ASSERT_TRUE(cache.contains(newParams) && !cache.contains(oldParams)) << "cache did not evict first entry";
    ASSERT_TRUE(cache.size() == 1) << "incorrect cache utilization";
    ASSERT_TRUE(newActivationsPtr->size() == newActivations.size()) << "wrong new activations returned from cache";
    ASSERT_TRUE(oldActivationsPtr->size() == oldActivations.size()) << "wrong old activations returned from cache";
}

TEST_F(MmeGaudi2DescriptorCacheTest, mme_descriptor_cache_check_evictions_with_retrievals)
{
    MmeCommon::MmeLayerParams params(MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2));
    Gaudi2::ActivationVec activations = {Gaudi2::MmeActivation(1)};

    constexpr unsigned CACHE_SIZE = 1000;
    DescriptorCacheForTest cache(CACHE_SIZE);
    // fill the cache
    for (int numOfAdded = 0; numOfAdded < CACHE_SIZE; numOfAdded++)
    {
        params.spBase = numOfAdded;
        activations[0].numSignals = 2 * numOfAdded;
        ASSERT_TRUE(cache.add(params, activations)) << "failed to add new entry to cache";
    }
    ASSERT_TRUE(cache.size() == CACHE_SIZE) << "wrong cache size after fill";

    // retrieve the oldest and middle entries to improve position in LRU
    params.spBase = 0;
    ASSERT_TRUE(cache.get(params) != nullptr) << "first inserted entry is not in cache";
    params.spBase = CACHE_SIZE / 2;
    ASSERT_TRUE(cache.get(params) != nullptr) << "middle inserted entry is not in cache";

    // evict all cache entries up to two recently used
    for (int numOfAdded = 0; numOfAdded < CACHE_SIZE - 2; numOfAdded++)
    {
        params.spBase = numOfAdded + CACHE_SIZE;
        activations[0].numSignals = numOfAdded;
        ASSERT_TRUE(cache.add(params, activations)) << "failed to add new entry to cache";
    }
    ASSERT_TRUE(cache.size() == CACHE_SIZE) << "wrong cache size after fill";

    // verify the two promoted entries from earlier are in cache
    params.spBase = 0;
    ASSERT_TRUE(cache.contains(params)) << "first inserted entry is not in cache";
    params.spBase = CACHE_SIZE / 2;
    ASSERT_TRUE(cache.contains(params)) << "middle inserted entry is not in cache";
    params.spBase = CACHE_SIZE - 1;
    ASSERT_FALSE(cache.contains(params)) << "entry should have been evicted";
    params.spBase = 1;
    ASSERT_FALSE(cache.contains(params)) << "entry should have been evicted";
}