#include "memory_management/bucket_allocator.h"

#include "graph_optimizer_test.h"

#include "memory_management/heap_allocator.h"
#include "memory_management/multi_buckets_allocator.h"
#include "gtest/gtest.h"

class BucketMemoryTest : public GraphOptimizerTest
{
};

TEST_F(BucketMemoryTest, buckets_allocation)
{
    static constexpr uint32_t MEMORY_SIZE = 4 * 1024 * 1024;
    static constexpr uint32_t BUCKET_SIZE = MEMORY_SIZE / 32;
    static constexpr uint32_t ALIGNMENT   = 128;

    HeapAllocator mainAllocator("BucketTest");
    mainAllocator.Init(MEMORY_SIZE);
    BucketAllocator bucketAlloc(BUCKET_SIZE, mainAllocator);
    auto            address1 = bucketAlloc.allocate(ALIGNMENT, 0);
    ASSERT_TRUE(address1.is_set());
    auto address2 = bucketAlloc.allocate(ALIGNMENT, 0);
    ASSERT_TRUE(address2.is_set());
    auto address3 = bucketAlloc.allocate(ALIGNMENT, 0);
    ASSERT_TRUE(address3.is_set());
    uint64_t remainSpace = mainAllocator.getMaxFreeContiguous();
    bucketAlloc.deallocate(address1.value());
    ASSERT_EQ(remainSpace, mainAllocator.getMaxFreeContiguous());
    auto address4 = bucketAlloc.allocate(ALIGNMENT, 0);
    ASSERT_TRUE(address4.is_set());
    ASSERT_EQ(address1.value(), address4.value());
    ASSERT_EQ(remainSpace, mainAllocator.getMaxFreeContiguous());
    bucketAlloc.allocate(ALIGNMENT, 0);
    ASSERT_EQ(remainSpace - BUCKET_SIZE, mainAllocator.getMaxFreeContiguous());
}

TEST_F(BucketMemoryTest, multi_buckets_allocation)
{
    static constexpr uint64_t MEMORY_SIZE = 2 * 1024 * 1024;
    static constexpr uint64_t BUCKET_SIZE = 32 * 1024;
    static constexpr uint32_t ALIGNMENT   = 1;
    HeapAllocator             mainAlloc("MULTI_BUCKET_TEST");
    mainAlloc.Init(MEMORY_SIZE);
    MultiBucketsAllocator multi({BUCKET_SIZE}, mainAlloc);

    auto address1 = multi.Allocate(BUCKET_SIZE / 2, ALIGNMENT);
    ASSERT_TRUE(address1.is_set());
    auto address2 = multi.Allocate(BUCKET_SIZE / 2, ALIGNMENT);
    ASSERT_TRUE(address2.is_set());
    ASSERT_EQ(multi.GetCurrentlyUsed(), BUCKET_SIZE * 2);
    // Bucket stays at the bucket pool
    multi.Free(address1.value());
    multi.Allocate(BUCKET_SIZE + 1, ALIGNMENT);
    ASSERT_EQ(multi.GetCurrentlyUsed(), BUCKET_SIZE * 3 + 1);
    // Uses the bucket that was free before
    multi.Allocate(BUCKET_SIZE / 2, ALIGNMENT);
    ASSERT_EQ(multi.GetCurrentlyUsed(), BUCKET_SIZE * 3 + 1);
}