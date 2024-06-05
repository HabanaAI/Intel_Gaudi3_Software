#include "infra/defs.h"
#include "utils.h"

#include "slab_allocator.h"

SlabAllocator::SlabAllocator(const std::string& name) : MemoryAllocatorBase(MEMORY_SLAB_ALLOCATOR, name), m_nextBlock(0)
{
}

SlabAllocator::~SlabAllocator() {}

void SlabAllocator::Init(uint64_t memorySize, deviceAddrOffset base)
{
    MemoryAllocatorBase::Init(memorySize, base);
}

Settable<deviceAddrOffset> SlabAllocator::Allocate(uint64_t size,
                                                   uint64_t alignment,
                                                   uint64_t offset /* = 0*/,
                                                   bool     allowFailure /* = false*/,
                                                   uint64_t requestedAddress /* = 0 */)
{
    HB_ASSERT(!allowFailure, "DRAM allocation failure is not supported");
    Settable<deviceAddrOffset> addr;
    uint64_t                   allocationSize = size + offset;
    uint64_t                   pad            = 0;
    if (m_nextBlock % alignment != 0)
    {
        pad = (alignment - (m_nextBlock % alignment));
    }
    allocationSize += pad;

    if (m_nextBlock + allocationSize > m_base + m_memorySize)
    {
        LOG_ERR(GC, "Memory Overrun! (reached 0x{:x})", m_base + m_memorySize);
        return addr;
    }

    addr = m_base + m_nextBlock + pad + offset;
    m_nextBlock += allocationSize;

    LOG_DEBUG(GC, "Allocate SLAB at addr 0x{:x}, size 0x{:x}", addr.value(), allocationSize);

    return addr;
}

void SlabAllocator::Free(deviceAddrOffset ptr)
{
    // Empty implementation
    UNUSED(ptr);
}

uint64_t SlabAllocator::GetCurrentlyUsed() const
{
    return m_nextBlock;
}

uint64_t SlabAllocator::getMaxFreeContiguous() const
{
    return m_memorySize - GetCurrentlyUsed();
}

std::unique_ptr<MemoryAllocator> SlabAllocator::Clone()
{
    return std::unique_ptr<MemoryAllocator> {new SlabAllocator(*this)};
}

bool SlabAllocator::IsAllocated(deviceAddrOffset ptr) const
{
    return (m_nextBlock > ptr);
}
