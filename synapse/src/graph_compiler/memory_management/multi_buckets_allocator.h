#pragma once

#include "types.h"

#include "bucket_allocator.h"
#include "memory_allocator.h"
#include <map>
#include <unordered_map>

class MultiBucketsAllocator : public MemoryAllocator
{
public:
    MultiBucketsAllocator(std::vector<uint32_t>&& bucketsSize, MemoryAllocator& mainAllocator);

    void                             Init(uint64_t memorySize, deviceAddrOffset base = 0) override;
    Settable<deviceAddrOffset>       Allocate(uint64_t size,
                                              uint64_t alignment,
                                              uint64_t offset           = 0,
                                              bool     allowFailure     = false,
                                              uint64_t requestedAddress = 0) override;
    void                             Free(deviceAddrOffset ptr) override;
    uint64_t                         GetCurrentlyUsed() const override;
    uint64_t                         getMaxFreeContiguous() const override;
    std::unique_ptr<MemoryAllocator> Clone() override;
    void                             SetPrintStatus(bool isAllowed) override;
    bool                             IsAllocated(deviceAddrOffset ptr) const override;
    MemoryAllocatorType              getMemAllocatorType() const override;
    uint64_t                         GetMemorySize() const override;
    deviceAddrOffset                 GetMemoryBase() const override;

private:
    void releaseAllBuckets();

    MemoryAllocator&                                     m_mainAllocator;
    std::map<uint32_t, std::unique_ptr<BucketAllocator>> m_buckets;
    std::unordered_map<deviceAddrOffset, uint64_t>       m_addressToSize;
};