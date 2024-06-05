#include "heap_allocator.h"
#include "infra/defs.h"

#include "slab_allocator.h"
#include "utils.h"

MemoryAllocatorBase::MemoryAllocatorBase(MemoryAllocatorType type, const std::string& name)
: m_base(0), m_memorySize(0), m_memAllocatorType(type), m_isPrintStatusAllowed(false), m_name(name)
{
}

void MemoryAllocatorBase::Init(uint64_t memorySize, deviceAddrOffset base)
{
    m_memorySize = memorySize;
    m_base       = base;
}

std::unique_ptr<MemoryAllocator> SynapseInternals::createAllocator(MemoryAllocatorType type, const std::string& name)
{
    switch (type)
    {
        case MEMORY_SLAB_ALLOCATOR:
            return std::unique_ptr<MemoryAllocator> {new SlabAllocator(name)};

        case MEMORY_HEAP_ALLOCATOR:
            return std::unique_ptr<MemoryAllocator> {new HeapAllocator(name)};

        case MEMORY_HEAP_NON_CYCLIC_ALLOCATOR:
        {
            return std::unique_ptr<MemoryAllocator> {new HeapAllocator(name, 0, false)};
        }

        default:
            break;
    }
    HB_ASSERT(false, "Requested invalid allocator type {}", type);
    return nullptr;
}
