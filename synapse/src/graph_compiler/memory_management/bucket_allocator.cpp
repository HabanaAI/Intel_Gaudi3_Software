#include "memory_management/bucket_allocator.h"
#include "defs.h"
#include "memory_allocator.h"

BucketAllocator::BucketAllocator(uint32_t bucketSize, MemoryAllocator& mainAllocator)
: m_bucketSize(bucketSize), m_mainAllocator(mainAllocator)
{
}

Settable<deviceAddrOffset> BucketAllocator::allocate(uint64_t alignment, uint64_t offset)
{
    Settable<deviceAddrOffset> ret;
    if (!m_freeBuckets.empty())
    {
        ret.set(m_freeBuckets.back());
        m_freeBuckets.pop_back();
    }
    else
    {
        ret = m_mainAllocator.Allocate(m_bucketSize, alignment, offset);
    }
    if (ret.is_set())
    {
        m_allocated.insert(ret.value());
    }
    return ret;
}

void BucketAllocator::deallocate(deviceAddrOffset address)
{
    if (m_allocated.erase(address) > 0)
    {
        m_freeBuckets.push_back(address);
    }
    else
    {
        HB_ASSERT(0, "Release bucket address which doesn't exist, Bucket size: {}", m_bucketSize);
    }
}

void BucketAllocator::releaseAllFreeBuckets()
{
    for (const auto& block : m_freeBuckets)
    {
        m_mainAllocator.Free(block);
    }
    m_freeBuckets.clear();
}

uint32_t BucketAllocator::getNumBuckets() const
{
    return m_freeBuckets.size() + m_allocated.size();
}