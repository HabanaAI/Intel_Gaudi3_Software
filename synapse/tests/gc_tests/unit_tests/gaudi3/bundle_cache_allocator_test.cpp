#include "memory_management/bundle_cache_allocator.h"

#include "../gaudi2/layered_brain_test.h"

using namespace gc::layered_brain;

using Result = BundleCacheAllocator::Result;
using Deps   = Result::Deps;

class BundleCacheAllocatorTest : public LayeredBrainTest
{
protected:
    BundleCacheState     m_cacheState {100};
    BundleCacheAllocator m_cacheAllocator {m_cacheState};
};

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_succeed_for_trivial_allocations)
{
    Result res = m_cacheAllocator.allocate(newTensor(), 10);
    EXPECT_TRUE(res.successful);
    EXPECT_TRUE(res.dependencies.empty());
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_fail_when_cant_fit_in_cache)
{
    Result res = m_cacheAllocator.allocate(newTensor(), 110);
    EXPECT_FALSE(res.successful);
    EXPECT_EQ(10, res.missingCapacity);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_fail_to_over_subscribe)
{
    TensorPtr t   = newTensor();
    Result    res = m_cacheAllocator.allocate(t, 80);
    EXPECT_TRUE(res.successful);
    res = m_cacheAllocator.allocate(newTensor(), 40);
    EXPECT_FALSE(res.successful);
    EXPECT_EQ(20, res.missingCapacity);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_fail_to_free_un_cached_tensors)
{
    EXPECT_FALSE(m_cacheAllocator.free(newTensor()));
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_succeed_to_free_cached_tensors)
{
    TensorPtr t   = newTensor();
    Result    res = m_cacheAllocator.allocate(t, 80);
    EXPECT_TRUE(res.successful);
    EXPECT_TRUE(m_cacheAllocator.free(t));
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_reuse_freed_tensors_budget)
{
    // When cache and free 80% of the cache for accessor index 0
    TensorPtr t   = newTensor();
    Result    res = m_cacheAllocator.allocate(t, 80);
    EXPECT_TRUE(res.successful);
    m_cacheState.addAccess(t, 0);
    EXPECT_TRUE(m_cacheAllocator.free(t));

    // Expect successful allocation of 40% of the cache, depending on operation index 0 being over
    res = m_cacheAllocator.allocate(newTensor(), 40);
    EXPECT_TRUE(res.successful);
    EXPECT_EQ(Deps {0}, res.dependencies);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_reuse_minimum_freed_tensors_by_lru)
{
    // When cache and free 80% of the cache for accessor index 1 and then 0
    TensorPtr t0  = newTensor();
    TensorPtr t1  = newTensor();
    Result    res = m_cacheAllocator.allocate(t0, 40);
    m_cacheState.addAccess(t0, 1);
    EXPECT_TRUE(res.successful);
    res = m_cacheAllocator.allocate(t1, 40);
    m_cacheState.addAccess(t1, 0);

    EXPECT_TRUE(m_cacheAllocator.free(t0));
    EXPECT_TRUE(m_cacheAllocator.free(t1));

    // Expect successful allocation of 40% of the cache, depending on operation index 0 being over.
    // This test checks that although an allocation for operation with index 1 was made and freed first, it is not
    // being reused for the new allocation since freeing the allocation made for the accessor with index 0 is enough.
    res = m_cacheAllocator.allocate(newTensor(), 40);
    EXPECT_TRUE(res.successful);
    EXPECT_EQ(Deps {0}, res.dependencies);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_not_double_reuse_reclaimed_budget)
{
    // When cache and free 80% of the cache for accessor index 0
    TensorPtr t0  = newTensor();
    TensorPtr t1  = newTensor();
    Result    res = m_cacheAllocator.allocate(t0, 40);
    m_cacheState.addAccess(t0, 0);
    EXPECT_TRUE(res.successful);
    res = m_cacheAllocator.allocate(t1, 40);
    m_cacheState.addAccess(t1, 0);

    EXPECT_EQ(20, m_cacheState.totalFree());

    EXPECT_TRUE(m_cacheAllocator.free(t0));
    EXPECT_TRUE(m_cacheAllocator.free(t1));

    // Freeing does not increase total free in the cache. Needs reclaiming.
    EXPECT_EQ(20, m_cacheState.totalFree());

    // Expect successful allocation of 40% of the cache
    res = m_cacheAllocator.allocate(newTensor(), 40);
    EXPECT_TRUE(res.successful);
    // Expect one entry to have been reclaimed, so total free should remain at 20%
    EXPECT_EQ(20, m_cacheState.totalFree());

    // Expect failure to allocate 80% of the cache
    res = m_cacheAllocator.allocate(newTensor(), 80);
    EXPECT_FALSE(res.successful);
    // 20% is free and 40% needs reclaiming, so to allocate 80%, missing 20% capacity:
    EXPECT_EQ(20, res.missingCapacity);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_not_lose_track_of_dependencies_in_case_of_allocation_failure)
{
    // This test makes sure the allocator doesn't reclaim the cache budget pre-maturely in cases where the allocation
    // failes even with reclaiming. If the reclaim was done pre-maturely, the dependencies would be lost.

    // When cache and free 80% of the cache for accessor index 0
    TensorPtr t0  = newTensor();
    TensorPtr t1  = newTensor();
    Result    res = m_cacheAllocator.allocate(t0, 30);
    m_cacheState.addAccess(t0, 0);
    EXPECT_TRUE(res.successful);
    res = m_cacheAllocator.allocate(t1, 50);
    m_cacheState.addAccess(t1, 1);

    EXPECT_TRUE(m_cacheAllocator.free(t0));
    EXPECT_TRUE(m_cacheAllocator.free(t1));

    // Expect failure to allocate 130% of the cache
    res = m_cacheAllocator.allocate(newTensor(), 130);
    EXPECT_FALSE(res.successful);
    EXPECT_EQ(30, res.missingCapacity);

    // Expect successful allocation of 60% of the cache, depending on operation index 0 and 1 being over
    res          = m_cacheAllocator.allocate(newTensor(), 60);
    Deps expDeps = {0, 1};
    EXPECT_TRUE(res.successful);
    EXPECT_EQ(expDeps, res.dependencies);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_succeed_reallocating_tensors)
{
    TensorPtr t = newTensor();
    for (int i = 0; i < 10; i++)
    {
        Result res = m_cacheAllocator.allocate(t, 50);
        EXPECT_TRUE(res.successful) << "failure in iteration " << i;
    }
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_not_reallocate_over_capacity)
{
    TensorPtr t = newTensor();
    m_cacheAllocator.allocate(t, 50);
    Result res = m_cacheAllocator.allocate(t, 110);
    EXPECT_FALSE(res.successful);
    EXPECT_EQ(10, res.missingCapacity);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_retain_accessors_when_reallocating)
{
    // When allocating a tensor with accessors, then re-allocating it and releasing it
    TensorPtr t0 = newTensor();
    m_cacheAllocator.allocate(t0, 50);
    m_cacheState.addAccess(t0, 0);
    m_cacheAllocator.allocate(t0, 80);
    m_cacheAllocator.free(t0);

    // Expect the accessors to be dependencies of a new user of the same budget
    Result res = m_cacheAllocator.allocate(newTensor(), 100);
    EXPECT_TRUE(res.successful);
    EXPECT_EQ(Deps {0}, res.dependencies);
}

TEST_F(BundleCacheAllocatorTest, cache_allocator_should_not_release_upon_reallocation_failure)
{
    // When failing to re-allocated a tensor
    TensorPtr t = newTensor();
    m_cacheAllocator.allocate(t, 50);
    Result res = m_cacheAllocator.allocate(t, 101);
    ASSERT_FALSE(res.successful);

    // Expect it to stay cached with the old capacity
    ASSERT_TRUE(m_cacheState.isCached(t));
    EXPECT_EQ(50, m_cacheState.capacity(t));
}