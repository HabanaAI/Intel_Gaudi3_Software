#include "multi_buckets_allocator.h"

#include "defs.h"
#include <memory>

MultiBucketsAllocator::MultiBucketsAllocator(std::vector<uint32_t>&& bucketsSize, MemoryAllocator& mainAllocator)
: m_mainAllocator(mainAllocator)
{
    for (uint32_t size : bucketsSize)
    {
        m_buckets.emplace(size, std::unique_ptr<BucketAllocator>(new BucketAllocator(size, mainAllocator)));
    }
}

void MultiBucketsAllocator::Init(uint64_t memorySize, deviceAddrOffset base)
{
    m_mainAllocator.Init(memorySize, base);
}

Settable<deviceAddrOffset> MultiBucketsAllocator::Allocate(uint64_t size,
                                                           uint64_t alignment,
                                                           uint64_t offset,
                                                           bool     allowFailure,
                                                           uint64_t requestedAddress)
{
    HB_ASSERT(requestedAddress == 0, "Request address is not expected with this allocator");
    Settable<deviceAddrOffset> ret;
    // Find the right bucket to use
    auto bucketIter = m_buckets.lower_bound(size);
    while (!ret.is_set() && bucketIter != m_buckets.end())
    {
        ret = bucketIter->second->allocate(alignment, offset);
        if (ret.is_set())
        {
            size = bucketIter->first;
        }
        // Try allocate on a bigger bucket if the smaller bucket fail
        ++bucketIter;
    }

    if (!ret.is_set())
    {
        // In case no bucket was used, try allocate from the main allocator
        ret = m_mainAllocator.Allocate(size, alignment, offset, allowFailure, requestedAddress);
        if (!ret.is_set())
        {
            LOG_WARN(HEAP_ALLOC,
                     "Reached max memory size on bucket allocator, trying to release all buckets and re-allocate");
            releaseAllBuckets();
            if (m_mainAllocator.getMaxFreeContiguous() >= size)
            {
                // Try reallocate after releasing buckets kept memory
                return Allocate(size, alignment, offset, allowFailure, requestedAddress);
            }
        }
    }

    if (ret.is_set())
    {
        m_addressToSize[ret.value()] = size;
    }

    return ret;
}

void MultiBucketsAllocator::Free(deviceAddrOffset ptr)
{
    auto bucketIter = m_buckets.find(m_addressToSize[ptr]);

    if (bucketIter != m_buckets.end())
    {
        bucketIter->second->deallocate(ptr);
    }
    else
    {
        m_mainAllocator.Free(ptr);
    }
}

uint64_t MultiBucketsAllocator::GetCurrentlyUsed() const
{
    return m_mainAllocator.GetCurrentlyUsed();
}

uint64_t MultiBucketsAllocator::getMaxFreeContiguous() const
{
    return m_mainAllocator.getMaxFreeContiguous();
}

std::unique_ptr<MemoryAllocator> MultiBucketsAllocator::Clone()
{
    HB_ASSERT(0, "Multi bucket allocator shouldn't be cloned");
    return nullptr;
}

void MultiBucketsAllocator::SetPrintStatus(bool isAllowed)
{
    m_mainAllocator.SetPrintStatus(isAllowed);
}

bool MultiBucketsAllocator::IsAllocated(deviceAddrOffset ptr) const
{
    return m_mainAllocator.IsAllocated(ptr);
}

MemoryAllocatorType MultiBucketsAllocator::getMemAllocatorType() const
{
    return MULTI_BUCKET_ALLOCATOR;
}

uint64_t MultiBucketsAllocator::GetMemorySize() const
{
    return m_mainAllocator.GetMemorySize();
}

deviceAddrOffset MultiBucketsAllocator::GetMemoryBase() const
{
    return m_mainAllocator.GetMemoryBase();
}

void MultiBucketsAllocator::releaseAllBuckets()
{
    for (auto& bucket : m_buckets)
    {
        bucket.second->releaseAllFreeBuckets();
    }
}