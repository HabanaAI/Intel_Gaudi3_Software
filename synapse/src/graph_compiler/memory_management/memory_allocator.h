#ifndef _MEMORY_ALLOCATOR_H_
#define _MEMORY_ALLOCATOR_H_

#include <cstdint>
#include <memory>
#include "infra/settable.h"
#include "types.h"
#include "allocators_utils.h"

typedef enum
{
    MEMORY_SLAB_ALLOCATOR,
    MEMORY_HEAP_ALLOCATOR,
    MEMORY_HEAP_NON_CYCLIC_ALLOCATOR,
    MULTI_BUCKET_ALLOCATOR,
} MemoryAllocatorType;

// Interface for classes implementing SRAM allocation algorithms
//
// The name MemoryAllocator conflicts with at least one other library!
// Put it in a namespace and make available with `using`.
//
namespace SynapseInternals
{

class MemoryAllocator
{
public:
    MemoryAllocator() {}
    virtual ~MemoryAllocator() {}
    virtual void                             Init(uint64_t memorySize, deviceAddrOffset base = 0) = 0;
    virtual Settable<deviceAddrOffset>       Allocate(uint64_t size,
                                                      uint64_t alignment,
                                                      uint64_t offset           = 0,
                                                      bool     allowFailure     = false,
                                                      uint64_t requestedAddress = 0)              = 0;
    virtual Settable<deviceAddrOffset>       Allocate(uint64_t size,
                                                      uint64_t alignment,
                                                      Lifetime tensorLifetime,
                                                      uint64_t offset           = 0,
                                                      bool     allowFailure     = false,
                                                      uint64_t requestedAddress = 0)
    {
        return Allocate(size, alignment, offset, allowFailure, requestedAddress);
    };
    virtual void                             Free(deviceAddrOffset ptr)                           = 0;
    virtual uint64_t                         GetCurrentlyUsed() const                             = 0;
    virtual uint64_t                         getMaxFreeContiguous() const                         = 0;
    virtual std::unique_ptr<MemoryAllocator> Clone()                                              = 0;
    virtual void                             SetPrintStatus(bool isAllowed)                       = 0;
    virtual bool                             IsAllocated(deviceAddrOffset ptr) const              = 0;
    virtual MemoryAllocatorType              getMemAllocatorType() const                          = 0;
    virtual uint64_t                         GetMemorySize() const                                = 0;
    virtual deviceAddrOffset                 GetMemoryBase() const                                = 0;

protected:
};

class MemoryAllocatorBase : public MemoryAllocator
{
public:
    explicit MemoryAllocatorBase(MemoryAllocatorType type, const std::string& name);
    virtual void                             Init(uint64_t memorySize, deviceAddrOffset base = 0) override;
    virtual Settable<deviceAddrOffset>       Allocate(uint64_t size,
                                                      uint64_t alignment,
                                                      uint64_t offset,
                                                      bool     allowFailure     = false,
                                                      uint64_t requestedAddress = 0) override = 0;
    virtual void                             Free(deviceAddrOffset ptr) override              = 0;
    virtual uint64_t                         GetCurrentlyUsed() const override                = 0;
    virtual std::unique_ptr<MemoryAllocator> Clone() override                                 = 0;
    virtual void                SetPrintStatus(bool isAllowed) override { m_isPrintStatusAllowed = isAllowed; };
    virtual bool                IsAllocated(deviceAddrOffset ptr) const override = 0;
    virtual MemoryAllocatorType getMemAllocatorType() const override { return m_memAllocatorType; }
    virtual uint64_t            GetMemorySize() const override { return m_memorySize; }
    virtual deviceAddrOffset    GetMemoryBase() const override { return m_base; }

protected:
    deviceAddrOffset    m_base;
    uint64_t            m_memorySize;
    MemoryAllocatorType m_memAllocatorType;
    bool                m_isPrintStatusAllowed;

    const std::string m_name;
};

std::unique_ptr<MemoryAllocator> createAllocator(MemoryAllocatorType type, const std::string& name);

} // SynapseInternals

using SynapseInternals::MemoryAllocator;
using SynapseInternals::MemoryAllocatorBase;
using SynapseInternals::createAllocator;


#endif
