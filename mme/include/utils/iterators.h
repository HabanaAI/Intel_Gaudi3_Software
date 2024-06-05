#ifndef MME__ITERATORS_H
#define MME__ITERATORS_H

#include <map>
#include <memory>
#include <vector>
#include "include/mme_assert.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// MultiLoopIterator
////////////////////////////////////////////////////////////////////////////////////////////////////

// A utility to manage multiple nested loops with various number of iterations and reverse-counting options
// via single range-based for loop.
// Within 'MultiLoopIterator' there is 'Iterator'. Both comply with standard definition of range-based for loops.
// Current values are elements of std::map that have 'm_key' as key and 'm_current' as value.
class MultiLoopIterator
{
public:
    using ItersType = std::map<unsigned, unsigned>;

    class Iterator
    {
    public:
        Iterator(unsigned key, unsigned itersNr, bool isReversed = false)
        : m_key(key), m_itersNr(itersNr), m_isReversed(isReversed)
        {
        }
        Iterator(const Iterator& iter, std::shared_ptr<Iterator> nextIter) : Iterator(iter) { m_nextIter = nextIter; }
        ItersType operator*() const;
        Iterator& operator++();
        friend bool operator!=(const Iterator& itr1, const Iterator& itr2)
        {
            return (itr1.m_current != itr1.m_itersNr);
        }
        bool isFirst(unsigned iter) const { return m_isReversed ? (iter == m_itersNr - 1) : (iter == 0); }
        bool isLast(unsigned iter) const { return m_isReversed ? (iter == 0) : (iter == m_itersNr - 1); }

    private:
        const unsigned m_key;  // Unique key to be attached to current value (returned by operator*)
        const unsigned m_itersNr;  // Number of iterations
        const bool m_isReversed;  // Counting direction hint - count forward (false) or backward (true)
        std::shared_ptr<Iterator> m_nextIter = nullptr;  // Pointer to iterator of next inner loop
        unsigned m_current = 0;  // Current iteration
    };

    MultiLoopIterator(const std::vector<Iterator>& iterators);
    Iterator begin()
    {
        MME_ASSERT_PTR(m_iterators);
        return *m_iterators;
    }
    Iterator end() { return begin(); }  // Implementation don't care about end()

private:
    std::shared_ptr<Iterator> m_iterators = nullptr;  // Pointer to iterator of the very outer loop
};

#endif //MME__ITERATORS_H
