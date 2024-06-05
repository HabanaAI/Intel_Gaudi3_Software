#include "recipe_allocator.h"
#include "defs.h"
#include "log_manager.h"
#include "memory_allocator_utils.hpp"

//-----------------------------------------------------------------------------
//                            RecipeAllocator
//-----------------------------------------------------------------------------

RecipeAllocator::~RecipeAllocator()
{
    freeAll();
}

char* RecipeAllocator::allocate(uint64_t sizeInBytes, bool shouldBeMappedToDevice)
{
    char* buf = nullptr;

    if (sizeInBytes)
    {
        const bool mappedAlloc = !m_ignoreDeviceMapping && shouldBeMappedToDevice;
        if (mappedAlloc)
        {
            buf = (char*)MemoryAllocatorUtils::alloc_memory_to_be_mapped_to_device(sizeInBytes);
        }
        else
        {
            buf = new char[sizeInBytes];
        }

        m_data.push_back(RecipeAllocation {buf, sizeInBytes, mappedAlloc});
    }

    return buf;
}

void RecipeAllocator::freeAll() noexcept
{
    // release all allocated memory for the recipe
    LOG_DEBUG(RECIPE_GEN, "{}: freeing {} recipe elements", HLLOG_FUNC, m_data.size());
    for (auto& allocation : m_data)
    {
        if (allocation.allocatedWithMmap)
        {
            MemoryAllocatorUtils::free_memory(allocation.pData, allocation.size);
        }
        else
        {
            delete[](char*)(allocation.pData);
        }
    }
    m_data.clear();
}

bool RecipeAllocator::freeSingleEntry(char* entryBaseAddress)
{
    if (entryBaseAddress == nullptr)
    {
        LOG_ERR(RECIPE_GEN, "{}: Got nullptr address", HLLOG_FUNC);
        return false;
    }

    LOG_TRACE(RECIPE_GEN, "{}: freeing single entry ({:#x})", HLLOG_FUNC, (uint64_t)entryBaseAddress);

    // Find the entry
    auto entryIt = std::find_if(m_data.begin(),
                                m_data.end(),
                                [&](const RecipeAllocation &r) { return entryBaseAddress == r.pData; });
    if (entryIt == m_data.end())
    {
        LOG_WARN(RECIPE_GEN,
                 "{}: Entry not found in m_data vector (base-address {:#x})",
                 HLLOG_FUNC,
                 (uint64_t)entryBaseAddress);
        return false;
    }

    // In case entry is mapped, unmap it
    if (entryIt->allocatedWithMmap)
    {
        MemoryAllocatorUtils::free_memory(entryIt->pData, entryIt->size);
    }

    // Remove the entry (the lifecycle-management is passed to the user)
    m_data.erase(entryIt);

    return true;
}

RecipeBlockAllocator::RecipeBlockAllocator(RecipeAllocator& recipeAllocator, size_t poolSz)
: m_recipeAllocator(recipeAllocator), m_size(poolSz), m_used(poolSz)
{
}

std::byte* RecipeBlockAllocator::alloc(size_t size)
{
    // if the new alloc can be pooled with anything else given the m_size limit
    // (exactly size bytes would fit in a new pool but not pool with anything)
    if (size < m_size)
    {
        HB_DEBUG_VALIDATE(m_used <= m_size);
        // safe way to check since m_used <= m_size! :)
        if (size > m_size - m_used)
        {
            // get a new block from the allocator since there's not enough room in the current one
            m_ptr  = reinterpret_cast<std::byte*>(m_recipeAllocator.allocate(m_size, /*shouldBeMappedToDevice*/ false));
            m_used = 0;
        }
        std::byte* res = m_ptr + m_used;
        m_used += size;
        return res;
    }

    // use a dedicated alloc since `size` is too big to be pooled with anything
    // without affecting the current pool.
    return reinterpret_cast<std::byte*>(m_recipeAllocator.allocate(size, /*shouldBeMappedToDevice*/ false));
}
