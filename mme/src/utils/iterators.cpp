#include "include/utils/iterators.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// MultiLoopIterator::Iterator
////////////////////////////////////////////////////////////////////////////////////////////////////

// Calculate current actual values (by considering reversed hint) of all iterators and associate each one with its key.
MultiLoopIterator::ItersType MultiLoopIterator::Iterator::operator*() const
{
    ItersType retVal;
    for (const Iterator* cur = this; cur != nullptr; cur = cur->m_nextIter.get())
    {
        retVal[cur->m_key] = (cur->m_isReversed ? (cur->m_itersNr - cur->m_current - 1) : cur->m_current);
    }
    return retVal;
}

// Increment 'm_current' of all iterators.
// Note that calculation doesn't take into account the counting direction (reverse or not).
MultiLoopIterator::Iterator& MultiLoopIterator::Iterator::operator++()
{
    if ((++m_current == m_itersNr) &&
        (m_nextIter != nullptr) &&  // Increment and check if the iterator reaches its limit
        ((++*m_nextIter).m_current != m_nextIter->m_itersNr))  // Propagate the carry to iterators of the outer loops
    {
        m_current = 0;  // Reset the saturated iterator, as carry was propagated and the counting is not done yet
    }
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MultiLoopIterator
////////////////////////////////////////////////////////////////////////////////////////////////////

// Build a linked list of given vector of iterators
MultiLoopIterator::MultiLoopIterator(const std::vector<Iterator>& iterators)
{
    for (const Iterator& iter : iterators)
    {
        m_iterators = std::make_shared<Iterator>(iter, m_iterators);
    }
}
