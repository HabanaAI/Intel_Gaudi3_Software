#include "dumb_allocator.hpp"

#include <cassert>
#include <defs.h>
#include <stdexcept>

const uint64_t DumbAllocator::ALIGN;
/*
 ***************************************************************************************************
 *   @brief DumbAllocator() - constructor
 *
 *   @param  startAddr, size- start address of the memory to use and its size
 *   @param  numChunks - Memory area is split into numChunks chunks
 *   @return None
 *
 ***************************************************************************************************
 */
DumbAllocator::DumbAllocator(uint8_t* startAddr, uint64_t size, uint32_t numChunks)
: m_startAddr(startAddr), m_size(size), m_numChunks(numChunks), m_chunkSize(m_size / numChunks)
{
    m_used.resize(numChunks);
    HB_ASSERT((m_chunkSize % ALIGN) == 0, "Not aligned to {:x}", ALIGN);
}

/*
 ***************************************************************************************************
 *   @brief allocate() - loop over the chunks, find the first free one. Verify needed size is
 *                       smaller than chunk size
 *
 *   @param  size- need size
 *   @return Address of chunk, nullptr if no free chunk
 *
 ***************************************************************************************************
 */
uint8_t* DumbAllocator::allocate(uint64_t size)
{
    if (size > m_chunkSize)
    {
        throw std::runtime_error("out of mapped memory");  // using throw so can run tessts in debug mode
    }

    for (int i = 0; i < m_numChunks; i++)
    {
        if (m_used[i] == false)
        {
            m_used[i] = true;
            return m_startAddr + i * m_chunkSize;
        }
    }
    return nullptr;
}

/*
 ***************************************************************************************************
 *   @brief free() - translate the address to the chunk number and mark this chunk as free
 *
 *   @param  addr - address to be freed
 *   @return None
 *
 ***************************************************************************************************
 */
void DumbAllocator::free(uint8_t* addr)
{
    uint64_t offset = addr - m_startAddr;

    assert((offset % m_chunkSize) == 0);

    uint32_t chunk = offset / m_chunkSize;

    assert(chunk < m_numChunks);
    assert(m_used[chunk] == true);

    m_used[chunk] = false;
}
