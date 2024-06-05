#pragma once

#include "settable.h"
#include "types.h"

namespace SynapseInternals
{
    class MemoryAllocator;
}

using SynapseInternals::MemoryAllocator;

class BucketAllocator
{
public:
    BucketAllocator(uint32_t bucketSize, MemoryAllocator& mainAllocator);

    Settable<deviceAddrOffset> allocate(uint64_t alignment, uint64_t offset);
    void                       deallocate(deviceAddrOffset address);

    void releaseAllFreeBuckets();

    uint32_t getNumBuckets() const;

private:
    uint32_t         m_bucketSize;
    MemoryAllocator& m_mainAllocator;

    std::set<deviceAddrOffset>    m_allocated;
    std::vector<deviceAddrOffset> m_freeBuckets;
};
