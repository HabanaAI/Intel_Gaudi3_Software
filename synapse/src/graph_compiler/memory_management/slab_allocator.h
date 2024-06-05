#ifndef _SLAB_ALLOCATOR_H_
#define _SLAB_ALLOCATOR_H_

#include "memory_allocator.h"

class SlabAllocator : public MemoryAllocatorBase
{
public:
    explicit SlabAllocator(const std::string& name);
    virtual ~SlabAllocator();

    virtual void                             Init(uint64_t memorySize, deviceAddrOffset base = 0) override;
    virtual Settable<deviceAddrOffset>       Allocate(uint64_t size,
                                                      uint64_t alignment,
                                                      uint64_t offset           = 0,
                                                      bool     allowFailure     = false,
                                                      uint64_t requestedAddress = 0) override;
    virtual void                             Free(deviceAddrOffset ptr) override;
    virtual uint64_t                         GetCurrentlyUsed() const override;
    virtual uint64_t                         getMaxFreeContiguous() const override;
    virtual std::unique_ptr<MemoryAllocator> Clone() override;
    virtual bool                             IsAllocated(deviceAddrOffset ptr) const override;

protected:
    deviceAddrOffset m_nextBlock;
};

#endif  //_SLAB_ALLOCATOR_H_
