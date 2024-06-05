#include "sliced_operand_traversal.h"

#include "slicing_utils.h"
#include "types_exception.h"
#include "utils.h"

const DimVector SlicedOperandTraversalPattern::LEFT_TO_RIGHT_1D = {WEIGHT_DIM_K};
const DimVector SlicedOperandTraversalPattern::LEFT_TO_RIGHT_2D = {WEIGHT_DIM_K, DIM_W};
const DimVector SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D = {DIM_W, WEIGHT_DIM_K};
const DimVector SlicedOperandTraversalPattern::LEFT_TO_RIGHT_4D = {WEIGHT_DIM_K, DIM_B};

SlicedOperandTraversalPattern::SlicedOperandTraversalPattern(pSlicedOperand slicedOperand,
                                                             DimVector      dimOrder,
                                                             bool           snakeTraversal,
                                                             unsigned       numOfCommonDimSlices)
: m_operand(std::move(slicedOperand)),
  m_dimOrder(std::move(dimOrder)),
  m_snakeTraversal(snakeTraversal),
  m_numOfCommonDimSlices(numOfCommonDimSlices)
{
    if (m_dimOrder.size() < 2)
    {
        m_snakeTraversal = false;
    }
    validateDimOrder();
}

MultiOperandSliceIterator SlicedOperandTraversalPattern::begin() const
{
    OperandSliceIterator beginIterator(m_operand, m_dimOrder, m_snakeTraversal, m_numOfCommonDimSlices);
    if (m_masterInputSliceChangeDim.is_set())
    {
        beginIterator.setDimForSliceChangeIndication(m_masterInputSliceChangeDim.value());
    }
    OperandSliceIterator endIterator   = OperandSliceIterator::getEndIterator(beginIterator);

    MultiOperandSliceIterator iterator(beginIterator, endIterator);

    for (auto& traversalPattern : m_slaveTraversalPatterns)
    {
        iterator.addOperandIterator(traversalPattern.begin());
    }
    return iterator;
}

MultiOperandSliceIterator SlicedOperandTraversalPattern::end() const
{
    return begin().getEndIterator();
}

void SlicedOperandTraversalPattern::validateDimOrder() const
{
    if (m_dimOrder.size() > m_operand->originalTensor->getDim())
    {
        throw SynapseException(
            fmt::format("Walking pattern on {} has too many dimensions", m_operand->originalTensor->getName()));
    }
    std::vector<unsigned> dimListCopy(m_dimOrder.begin(), m_dimOrder.end());
    std::sort(dimListCopy.begin(), dimListCopy.end());
    for (auto it = ++dimListCopy.begin(); it != dimListCopy.end(); it++)
    {
        if (*(it - 1) == *it)
        {
            throw SynapseException(
                fmt::format("Walking pattern on {} has duplicate dimensions", m_operand->originalTensor->getName()));
        }
    }
    if (dimListCopy.back() > m_operand->originalTensor->getDim())
    {
        throw SynapseException(
            fmt::format("Walking pattern on {} lists a dimension which is not available in the tensor",
                        m_operand->originalTensor->getName()));
    }
}

void SlicedOperandTraversalPattern::setInputSliceChangeDim(unsigned int interestingDimForSliceChangeSignal)
{
    m_masterInputSliceChangeDim.set(interestingDimForSliceChangeSignal);
}

void SlicedOperandTraversalPattern::setDimOrder(DimVector&& dimOrder)
{
    m_dimOrder = std::move(dimOrder);
}

const pSlicedOperand& SlicedOperandTraversalPattern::getOperand() const
{
    return m_operand;
}

void OperandSliceIterator::CoordCyclicCounter::advance()
{
    m_countRestarted = false;
    coord += m_advanceDir;
    if (coord == m_currentLimit)
    {
        m_countRestarted = true;
        restart();
    }
}

/* Snake traversal looks like the following chart:
 * >------------>
 *              |
 * <------------<
 * |
 * >------------>
 * The snake traversal will return the following order:
 * Main dimension represented by the 2nd index
 * (0,0), (0,1), (0,2), (1,2), (1,1), (1,0)
 *  ====================
 * |      |      |      |
 * |  0,0 | 0,1  | 0,2  |
 * |======|======|======| Notice: When (0,2) --> (1,2) the main dimension remains the same
 * |      |      |      |
 * |  1,0 | 1,1  | 1,2  |
 *  ====================
 *  To support it, the first traversed coordinate is counted start to end, then when restarted, swith to end to start
 *  and so on. (0, 1, 2, 2 ,1, 0, 0, 1, 2, ....)
 * */
void OperandSliceIterator::CoordCyclicCounter::restart()
{
    if (m_snake)
    {
        m_advanceDir *= -1;
        m_currentLimit = (m_advanceDir < 0)
                         ? -1                    // Advancing from end to start
                         : coordLimit;           // Advancing from start to end
        coord += m_advanceDir;
    }
    else
    {
        coord = 0;
    }
}

OperandSliceIterator::OperandSliceIterator(const pSlicedOperand& slicedOperand,
                                           DimVector             dimOrder,
                                           bool                  snakePattern,
                                           unsigned              numOfCommonDimSlices)
: m_dimOrder(std::move(dimOrder)),
  m_commonDimCounter(0 /*don't care*/, numOfCommonDimSlices, snakePattern),
  m_next(slicedOperand),
  m_snakePattern(snakePattern)
{
    init();
}

void OperandSliceIterator::init()
{
    bool addSnake = m_snakePattern; // Optionally: get rid of this. Everything should be snake if m_snakePattern is true.
    m_dimCounters.reserve(m_dimOrder.size());
    for (auto dim : m_dimOrder)
    {
        int limit = SlicedOperandUtils::nofSlices(m_next.operand, dim);
        m_dimCounters.emplace_back(dim, limit, addSnake);
        addSnake = false;
    }

    if (m_dimOrder.size() > 1)
    {
        // If the 2nd dimension traversed change, it means a wide slice changed.
        // This is the default for more than single dim traversal.
        m_dimensionForSliceChangeIndication = *std::next(m_dimOrder.begin());
    }
    else
    {
        // No wide slices changes, either never signal, or there are common dim slices, so always signal.
        m_dimensionForSliceChangeIndication = -1;
    }
}

OperandSliceIterator& OperandSliceIterator::operator++()
{
    advance();
    return *this;
}

OperandSliceIterator OperandSliceIterator::operator++(int)
{
    OperandSliceIterator ret = *this;
    advance();
    return ret;
}

void OperandSliceIterator::advance()
{
    HB_ASSERT(!ended(), "Trying to advance the iterator past the end");

    // The default wideSliceChange is false, unless the common dim is sliced.
    m_inputSliceChanged = (m_commonDimCounter.coordLimit > 1);

    // Common dim advancement
    m_commonDimCounter.advance();
    if (m_commonDimCounter.lastAdvanceRestartedCount())
    {
        // Common dimension restarted => need to advance output dimensions
        auto dimCounterIter = m_dimCounters.begin();
        for (; dimCounterIter != m_dimCounters.end(); ++dimCounterIter)
        {
            dimCounterIter->advance();
            if (m_dimensionForSliceChangeIndication == dimCounterIter->dim)
            {
                // The signaled dimension was advanced
                m_inputSliceChanged = true;
            }
            if (!dimCounterIter->lastAdvanceRestartedCount())
            {
                // If the current dimension didn't restart, no need to advance to the next.
                break;
            }
        }
        updateNext();
        if (dimCounterIter == m_dimCounters.end())
        {
            moveToPastTheEndElement();
        }
    }
}

void OperandSliceIterator::updateNext()
{
    for (auto dimIter : m_dimCounters)
    {
        m_next.coordinates[dimIter.dim] = dimIter.coord;
    }
}

/* Iterate to the lastSlice+1. The iterator will point to the past-the-end element in the sequence */
void OperandSliceIterator::moveToPastTheEndElement()
{
    m_commonDimCounter.coord = m_commonDimCounter.coordLimit;
    for (auto& dimIter : m_dimCounters)
    {
        dimIter.coord = dimIter.coordLimit;
    }
    updateNext();
}

// Iterators are equal if they iterate on the same tensor and are currently dereferenced into the same slice.
bool OperandSliceIterator::operator==(const OperandSliceIterator& other) const
{
    // Can't compare anything if other iterates another operand.
    if (m_next.operand != other.m_next.operand) return false;

    HB_ASSERT(m_snakePattern == other.m_snakePattern,
              "Compared iterators must agree on snake-ness");

    return (m_next.coordinates == other.m_next.coordinates &&
            m_commonDimCounter.coord == other.m_commonDimCounter.coord );
}

bool OperandSliceIterator::operator!=(const OperandSliceIterator& other) const
{
    return !(*this == other);
}

SliceRefCommonDimIdxPair OperandSliceIterator::operator*()
{
    return {std::make_shared<SliceReference>(m_next), m_commonDimCounter.coord};
}

OperandSliceIterator OperandSliceIterator::getEndIterator(const OperandSliceIterator& iterator)
{
    OperandSliceIterator end = iterator;
    end.moveToPastTheEndElement();
    return end;
}

/***********************************************************************************************************************
 ***************************************MultiOperandSliceIterator implementation****************************************
 ***********************************************************************************************************************/

MultiOperandSliceIterator::MultiOperandSliceIterator(const OperandSliceIterator& iteratorBegin,
                                                     const OperandSliceIterator& iteratorEnd)
: m_currentIteratorIdx(0)
{
    m_iterators.emplace_back(iteratorBegin, iteratorEnd, iteratorBegin.inputSliceChanged());
}

MultiOperandSliceIterator::MultiOperandSliceIterator(OperandSliceIterator&& iteratorBegin,
                                                     OperandSliceIterator&& iteratorEnd)
: m_currentIteratorIdx(0)
{
    m_iterators.emplace_back(std::move(iteratorBegin), std::move(iteratorEnd), iteratorBegin.inputSliceChanged());
}

void MultiOperandSliceIterator::addOperandIterator(const OperandSliceIterator& iteratorBegin, const OperandSliceIterator& iteratorEnd)
{
   m_iterators.emplace_back(iteratorBegin, iteratorEnd, iteratorBegin.inputSliceChanged());
}

void MultiOperandSliceIterator::addOperandIterator(const MultiOperandSliceIterator& iterator)
{
    for(auto& it : iterator.m_iterators)
    {
        m_iterators.push_back(it);
    }
}

MultiOperandSliceIterator& MultiOperandSliceIterator::operator++()
{
    advance();
    return *this;
}

MultiOperandSliceIterator MultiOperandSliceIterator::operator++(int)
{
    MultiOperandSliceIterator ret = *this;
    advance();
    return ret;
}

bool MultiOperandSliceIterator::operator==(const MultiOperandSliceIterator& other) const
{
    // check iterator list size first
    if (m_iterators.size() != other.m_iterators.size()) return false;

    // check equality for each iterator in the list.
    unsigned thisItIdx = 0, otherItIdx = 0;
    unsigned iteratorsAtEnd = 0;

    while (thisItIdx < m_iterators.size())
    {
        const OperandSliceIterator& thisIterator  = m_iterators[thisItIdx].m_iterator;
        const OperandSliceIterator& otherIterator = other.m_iterators[thisItIdx].m_iterator;

        if (thisIterator != otherIterator)
        {
            return false;
        }

        if (thisIterator == m_iterators[thisItIdx].m_endIterator)
        {
            iteratorsAtEnd++;
        }

        thisItIdx++;
        otherItIdx++;
    }

    // no need to check current iterator idx when comparing to end iterator.
    if (iteratorsAtEnd != m_iterators.size())
    {
        return m_currentIteratorIdx == other.m_currentIteratorIdx;
    }

    return true;
}

bool MultiOperandSliceIterator::operator!=(const MultiOperandSliceIterator& other) const
{
    return !(*this == other);
}

SliceRefCommonDimIdxPair MultiOperandSliceIterator::operator*()
{
    return *getCurrentIterator();
}

void MultiOperandSliceIterator::advance()
{
    /* increment the current iterator */
    getCurrentIterator()++;

    /* Suspend iterator that reached new line */
    if (getCurrentIterator().inputSliceChanged() ||
        m_iterators.at(m_currentIteratorIdx).m_iterator == m_iterators.at(m_currentIteratorIdx).m_endIterator)
    {
        /* Suspend the iterator until it's time to start a new input slice (or indefinitely if it's ended) */
        m_iterators.at(m_currentIteratorIdx).m_suspended = true;
    }

    /* Find the next iterator */
    m_currentIteratorIdx = getNextAvailableIterator();
}

MultiOperandSliceIterator::MultiOperandSliceIterator(const MultiOperandSliceIterator& other)
{
    for (auto& it : other.m_iterators)
    {
        m_iterators.push_back(it);
    }

    m_currentIteratorIdx = other.m_currentIteratorIdx;

}

MultiOperandSliceIterator MultiOperandSliceIterator::getEndIterator() const
{
    unsigned idx = 0;
    MultiOperandSliceIterator endIt(m_iterators[idx].m_endIterator, m_iterators[idx].m_endIterator);

    idx++;

    while (idx < m_iterators.size())
    {
        endIt.m_iterators.emplace_back(m_iterators[idx].m_endIterator, m_iterators[idx].m_endIterator, true);
        idx++;
    }

    return endIt;
}

OperandSliceIterator MultiOperandSliceIterator::getEndIterator(const OperandSliceIterator& iterator) const
{
    return OperandSliceIterator::getEndIterator(iterator);
}

unsigned MultiOperandSliceIterator::getNextAvailableIterator()
{
    unsigned              suspendedIterators = 0;
    unsigned              currIteratorIdx    = m_currentIteratorIdx;

    do
    {
        if (suspendedIterators == m_iterators.size())
        {
            /* All iterators have finished a narrow line.
             * Move the currIterator to the beginning and reset iterators suspension.
             * Iterator that reached the end will remain suspended.
             * */
            currIteratorIdx = 0;

            if (*this == getEndIterator()) return currIteratorIdx;


            /* Since we are starting a new traversal - need to reset all iterators suspension */
            resetIteratorsSuspension();
        }
        else
        {
            currIteratorIdx++;
        }

        /* Cyclic iteration over the list. */
        if (currIteratorIdx == m_iterators.size())
        {
            currIteratorIdx = 0;
        }

        if (m_iterators.at(currIteratorIdx).m_suspended)
        {
            suspendedIterators++;
        }

    } while(m_iterators.at(currIteratorIdx).m_suspended);

    return currIteratorIdx;
}

void MultiOperandSliceIterator::resetIteratorsSuspension()
{
    for (OperandSliceIteratorWrapper& iteratorWrapper : m_iterators)
    {
        if (iteratorWrapper.m_iterator != iteratorWrapper.m_endIterator)
        {
            iteratorWrapper.m_suspended = false;
        }
    }
}
