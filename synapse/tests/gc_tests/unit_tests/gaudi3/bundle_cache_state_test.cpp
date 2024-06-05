#include "memory_management/bundle_cache_state.h"

#include "../gaudi2/layered_brain_test.h"

using namespace gc::layered_brain;

class BundleCacheStateTest : public LayeredBrainTest
{
};

TEST_F(BundleCacheStateTest, tensors_are_not_cached_by_default)
{
    BundleCacheState cacheState({});
    EXPECT_FALSE(cacheState.isCached(newTensor()));
}

TEST_F(BundleCacheStateTest, cache_state_should_indicate_cached_tensors)
{
    TensorPtr        cachedSlice = newTensor();
    BundleCacheState cacheState(20);
    cacheState.cache(cachedSlice, 20);
    EXPECT_TRUE(cacheState.isCached(cachedSlice));
}

TEST_F(BundleCacheStateTest, cache_state_should_indicate_tensor_cache_occupancy)
{
    TensorPtr        t = newTensor();
    BundleCacheState cacheState(20);
    cacheState.cache(t, 14);
    EXPECT_EQ(14, cacheState.capacity(t));
}

TEST_F(BundleCacheStateTest, cache_state_should_have_limited_capacity)
{
    BundleCacheState cacheState(100);
    EXPECT_EQ(100, cacheState.totalFree());
}

TEST_F(BundleCacheStateTest, caching_tensors_should_reduce_free_budget)
{
    BundleCacheState cacheState(100);
    TensorPtr        s0 = newTensor();
    TensorPtr        s1 = newTensor();

    cacheState.cache(s0, 20);
    EXPECT_EQ(100 - 20, cacheState.totalFree());
    cacheState.cache(s1, 30);
    EXPECT_EQ(100 - 20 - 30, cacheState.totalFree());
}

TEST_F(BundleCacheStateTest, cache_state_should_allow_over_subscription)
{
    BundleCacheState cacheState(100);
    TensorPtr        s0 = newTensor();
    TensorPtr        s1 = newTensor();

    cacheState.cache(s0, 60);
    EXPECT_EQ(100 - 60, cacheState.totalFree());
    cacheState.cache(s1, 50);
    EXPECT_EQ(100 - 60 - 50, cacheState.totalFree());
    EXPECT_GT(0, cacheState.totalFree());
}

TEST_F(BundleCacheStateTest, cache_state_should_not_indicate_released_tensors_as_cached)
{
    BundleCacheState cacheState(30);
    TensorPtr        slice = newTensor();

    cacheState.cache(slice, 20);
    cacheState.release(slice);

    EXPECT_FALSE(cacheState.isCached(slice));
}

TEST_F(BundleCacheStateTest, releasing_tensors_should_not_free_budget)
{
    BundleCacheState cacheState(30);
    TensorPtr        slice = newTensor();

    cacheState.cache(slice, 20);
    cacheState.release(slice);

    // The released tensor should still be counted in the "occupied" budget until it's reclaimed
    EXPECT_EQ(10, cacheState.totalFree());
}

TEST_F(BundleCacheStateTest, reclaiming_tensors_should_increase_free_budget)
{
    BundleCacheState cacheState(100);
    TensorPtr        s0 = newTensor();
    TensorPtr        s1 = newTensor();

    cacheState.cache(s0, 20);
    cacheState.cache(s1, 30);

    cacheState.reclaim(s1);

    // occupied 20 out of 100, then another 30, then reclaimed 30, so left with a total occupied of 80
    EXPECT_EQ(80, cacheState.totalFree());
}

TEST_F(BundleCacheStateTest, cache_state_should_have_no_accessor_to_a_tensor_by_default)
{
    BundleCacheState cacheState(100);
    TensorPtr        s0 = newTensor();
    TensorPtr        s1 = newTensor();

    cacheState.cache(s0, 10);
    cacheState.cache(s1, 20);

    EXPECT_EQ(BundleCacheState::Accessors {}, cacheState.accesses(s0));
    EXPECT_EQ(BundleCacheState::Accessors {}, cacheState.accesses(s1));
}

TEST_F(BundleCacheStateTest, cache_state_should_allow_adding_accessors_to_tensors)
{
    BundleCacheState cacheState(100);
    TensorPtr        s0 = newTensor();
    TensorPtr        s1 = newTensor();

    cacheState.cache(s0, 10);
    cacheState.cache(s1, 20);

    cacheState.addAccess(s0, 0);
    cacheState.addAccess(s0, 1);
    cacheState.addAccess(s1, 2);
    cacheState.addAccess(s1, 3);

    BundleCacheState::Accessors s0expectedAccessors = {0, 1};
    BundleCacheState::Accessors s1expectedAccessors = {2, 3};

    EXPECT_EQ(s0expectedAccessors, cacheState.accesses(s0));
    EXPECT_EQ(s1expectedAccessors, cacheState.accesses(s1));
}

TEST_F(BundleCacheStateTest, cache_state_should_list_reclaim_candidates)
{
    BundleCacheState cacheState(100);
    TensorVector     tensors;

    // When caching 10 tensors:
    for (int i = 0; i < 10; i++)
    {
        tensors.push_back(newTensor());
        cacheState.cache(tensors.back(), 10);
    }
    // And accessing them in reverse order (tensors[9] has access-0 ... tensors[0] has access-9)
    for (int i = 9; i >= 0; i--)
    {
        cacheState.addAccess(tensors[i], 9 - i);
    }

    // Before releasing, expect no candidates
    auto candidates = cacheState.lruReclaimCandidates();
    EXPECT_TRUE(candidates.empty());

    cacheState.release(tensors[5]);
    cacheState.release(tensors[2]);
    cacheState.release(tensors[6]);

    // After releasing, expect them listed in LRU order (by least recently accessed)
    candidates = cacheState.lruReclaimCandidates();
    ASSERT_EQ(3, candidates.size());
    auto it = candidates.begin();
    EXPECT_EQ(tensors[6], (*it++)->tensor);
    EXPECT_EQ(tensors[5], (*it++)->tensor);
    EXPECT_EQ(tensors[2], (*it++)->tensor);
}

TEST_F(BundleCacheStateTest, cache_state_should_provide_reclaim_potential)
{
    BundleCacheState cacheState(100);
    TensorVector     tensors;

    // When caching 10 tensors:
    for (int i = 0; i < 10; i++)
    {
        tensors.push_back(newTensor());
        cacheState.cache(tensors.back(), 10);
    }

    EXPECT_EQ(0, cacheState.maxReclaim());
    cacheState.release(tensors[3]);
    EXPECT_EQ(10, cacheState.maxReclaim());
    cacheState.release(tensors[9]);
    EXPECT_EQ(20, cacheState.maxReclaim());
    cacheState.release(tensors[7]);
    EXPECT_EQ(30, cacheState.maxReclaim());
}

TEST_F(BundleCacheStateTest, cache_state_should_not_list_reclaimed_tensors_as_candidates)
{
    BundleCacheState cacheState(100);

    TensorPtr t0 = newTensor();
    TensorPtr t1 = newTensor();

    cacheState.cache(t0, 20);
    cacheState.addAccess(t0, 0);
    cacheState.cache(t1, 30);
    cacheState.addAccess(t1, 1);

    cacheState.release(t0);
    cacheState.release(t1);

    cacheState.reclaim(t0);

    auto candidates = cacheState.lruReclaimCandidates();

    // t0 was already reclaimed, so expect to see only t1 in the candidates
    ASSERT_EQ(1, candidates.size());
    EXPECT_EQ(t1, candidates.front()->tensor);
}

TEST_F(BundleCacheStateTest, cache_state_should_track_max_living_tensors_capacity)
{
    BundleCacheState cacheState(100);  // live = 0
    EXPECT_EQ(0, cacheState.maxLiveCapacity());

    cacheState.cache(newTensor(), 10);  // live = 10; new max!
    EXPECT_EQ(10, cacheState.maxLiveCapacity());

    TensorPtr t0 = newTensor();
    cacheState.cache(t0, 20);  // live = 10 + 20; new max!
    cacheState.release(t0);    // live = 10 (release reduces live, not maxLive)
    EXPECT_EQ(30, cacheState.maxLiveCapacity());

    TensorPtr t1 = newTensor();
    cacheState.cache(t1, 10);  // live = 20
    EXPECT_EQ(30, cacheState.maxLiveCapacity());

    cacheState.reclaim(t1);    // live = 10 (reclaim also reduces live)
    cacheState.cache(t1, 25);  // live = 35; new max!
    EXPECT_EQ(35, cacheState.maxLiveCapacity());

    cacheState.release(t1);    // live = 10
    cacheState.reclaim(t1);    // live = 10 (reclaim after release does not reduce live)
    cacheState.cache(t1, 30);  // live = 40; new max!
    EXPECT_EQ(40, cacheState.maxLiveCapacity());
}