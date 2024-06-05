#pragma once

// eager includes (relative to src/eager/lib/)
#include "utils/general_defs.h"

// std includes
#include <iterator>

namespace eager_mode
{
// Info required to perform sequential iteration over a non-random access container e.g. list.
// This is a bullet prof solution for any logic that receives an index of item as an input.
// Alternative solution is to convert the non-random access data structure into random access one e.g. an array.
// However, this representation avoids extra memory allocation to store the pointers into a random access data
// structure.
template<class ContainerIterator>
class SequentialIterTracker final
{
public:
    SequentialIterTracker() = default;
    SequentialIterTracker(const ContainerIterator& firstIter, size_t totalItems)
    : m_isInitialized(true), m_firstIter(firstIter), m_iter(firstIter), m_totalItems(totalItems)
    {
    }
    inline void                     initialize(const ContainerIterator& firstIter, size_t totalItems);
    ContainerIterator&              getIter(size_t index);
    inline const ContainerIterator& getIter(size_t index) const
    {
        return const_cast<SequentialIterTracker*>(this)->getIter(index);
    }

private:
    bool              m_isInitialized = false;
    unsigned          m_lastDescId    = -2;  // As we +1 later on and check against actual descriptor
    ContainerIterator m_firstIter;
    ContainerIterator m_iter;
    size_t            m_totalItems;
};

template<class ContainerIterator>
void SequentialIterTracker<ContainerIterator>::initialize(const ContainerIterator& firstIter, size_t totalItems)
{
    m_isInitialized = true;
    m_firstIter = m_iter = firstIter;
    m_totalItems         = totalItems;
}

template<class ContainerIterator>
ContainerIterator& SequentialIterTracker<ContainerIterator>::getIter(size_t index)
{
    EAGER_ASSERT(m_isInitialized, "Sequential iteration tracker is not initialized");
    if (m_lastDescId != index)
    {
        EAGER_ASSERT(index < m_totalItems, "The given MME desc index is out of bound");
        if (m_lastDescId + 1 == index)
        {
            ++m_lastDescId;
            ++m_iter;
        }
        else
        {
            m_lastDescId = 0;
            m_iter       = m_firstIter;
            if (m_lastDescId != index)
            {
                m_lastDescId = index;
                std::advance(m_iter, index);
                EAGER_ASSERT(0, "Trying non-sequential access to a non-random-access container");
            }
        }
        EAGER_ASSERT(m_lastDescId == index, "Unable to select the proper item");
    }
    return m_iter;
}

}  // namespace eager_mode