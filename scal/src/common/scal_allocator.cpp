#include <cassert>
#include <iostream> // temp
#include "logger.h"
#include "scal_allocator.h"

ScalHeapAllocator::ScalHeapAllocator(const std::string& name) : m_name(name)
{
}

void ScalHeapAllocator::setSize(uint64_t size)
{
    assert(m_slabs.empty());
    m_slabs[0] = 0;
    m_slabs[size] = 1; // mark end of heap
    m_totalSize = size;
    m_freeSize  = size;
}

uint64_t ScalHeapAllocator::alloc(uint64_t size, uint64_t alignment)
{
    if (size == 0) return 0;
    if (alignment == 0) return Scal::Allocator::c_bad_alloc;
    std::unique_lock<std::mutex> lock(m_mutex);
    // check for a free slab
    if (!m_slabs.empty())
    {
        std::map<uint64_t, uint64_t>::const_iterator first = m_slabs.cbegin();
        std::map<uint64_t, uint64_t>::const_iterator second = first;
        ++second;
        // iterate over all pairs to find a free slab
        for (; second != m_slabs.cend(); ++first, ++second)
        {
            uint64_t address = first->first + first->second;// addr+size
            uint64_t allocationSize = size;
            uint64_t pad            = 0;
            if (address % alignment != 0)
            {
                pad = (alignment - (address % alignment));
            }
            allocationSize += pad;
            if (second->first - address >= allocationSize)
            {
                // allocate
                m_slabs[address + pad] = size;
                m_freeSize -= size;
                return address + pad;
            }
        }
    }
    LOG_ERR_F(SCAL, "failed to allocate {}. total size {} free size {}", size, m_totalSize, m_freeSize);
    if(HLLOG_LEVEL_AT_LEAST_TRACE(SCAL))
    {
        for (auto iter = m_slabs.cbegin(); iter != m_slabs.cend(); ++iter)
        {
            LOG_TRACE(SCAL, "m_slabs[{}] = {}", iter->first, iter->second);
        }
    }
    return Scal::Allocator::c_bad_alloc;
}

void ScalHeapAllocator::free(uint64_t ptr)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_freeSize += m_slabs[ptr];
    if (ptr == 0)
    {
        if (m_slabs[0] == 0)
        {
            assert(false);
        }
        m_slabs[0] = 0;
    }
    else
    {
        unsigned erased = m_slabs.erase(ptr);
        if (erased == 0)
        {
            assert(0);// otherwise release build complains
        }
    }
}

void ScalHeapAllocator::getInfo(uint64_t& totalSize, uint64_t& freeSize)
{
    totalSize = m_totalSize;
    freeSize = m_freeSize;
}
