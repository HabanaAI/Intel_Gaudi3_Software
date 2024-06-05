#include "coordinate_traversal.h"
#include "defs.h"
#include "utils.inl"

CoordIterator::CoordCyclicCounter::AdvanceStatus CoordIterator::CoordCyclicCounter::advance()
{
    AdvanceStatus ret = STEP;
    coord += m_advanceDir;
    if (coord == m_currentLimit)
    {
        ret = m_snake ? TURN_AROUND : WRAP_AROUND;
        restart();
    }
    return ret;
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
void CoordIterator::CoordCyclicCounter::restart()
{
    if (m_snake)
    {
        m_advanceDir *= -1;
        m_currentLimit = (m_advanceDir < 0) ? -1           // Advancing from end to start
                                            : coordLimit;  // Advancing from start to end
        coord += m_advanceDir;
    }
    else
    {
        coord = 0;
    }
}

CoordIterator::CoordIterator(Coordinate limits, DimVector dimOrder, DimVector snakePatternDims)
: m_dimOrder(std::move(dimOrder)),
  m_limits(std::move(limits)),
  m_next(m_limits.size(), 0),
  m_snakePatternDims(snakePatternDims)
{
    init();
}

void CoordIterator::init()
{
    // Find how many dims have limit > 1. Those dims advance between coordinates and need counters
    unsigned numCounters = std::count_if(m_limits.begin(), m_limits.end(), [&](uint64_t limit) { return limit > 1; });
    m_dimCounters.reserve(numCounters);
    // Add the counters for the dims that require order, by the requested order
    for (auto dim : m_dimOrder)
    {
        addCounterForDim(dim);
    }
    // Add the counters for the rest of the dims which require a counter
    for (unsigned dim = 0; dim < m_limits.size(); dim++)
    {
        // If the dim is not in dim order, but its limit is larger than 1, it requires a counter
        if (std::find(m_dimOrder.begin(), m_dimOrder.end(), dim) == m_dimOrder.end())
        {
            addCounterForDim(dim);
        }
    }
}

void CoordIterator::addCounterForDim(uint8_t dim)
{
    bool addSnake = std::find(m_snakePatternDims.begin(), m_snakePatternDims.end(), dim) != m_snakePatternDims.end();
    m_dimCounters.emplace_back(dim, m_limits[dim], addSnake);
}

CoordIterator& CoordIterator::operator++()
{
    advance();
    return *this;
}

CoordIterator CoordIterator::operator++(int)
{
    CoordIterator ret = *this;
    advance();
    return ret;
}

void CoordIterator::advance()
{
    HB_ASSERT(!ended(), "Trying to advance the iterator past the end");

    auto dimCounterIter = m_dimCounters.begin();
    for (; dimCounterIter != m_dimCounters.end(); ++dimCounterIter)
    {
        CoordIterator::CoordCyclicCounter::AdvanceStatus status = dimCounterIter->advance();
        if (status == CoordIterator::CoordCyclicCounter::STEP)
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

void CoordIterator::updateNext()
{
    for (auto dimIter : m_dimCounters)
    {
        m_next[dimIter.dim] = dimIter.coord;
    }
}

bool CoordIterator::ended() const
{
    for (auto dimIter : m_dimCounters)
    {
        // if any of the coordinate passed its limit - it is beyond the last element
        if (dimIter.coord == dimIter.coordLimit)
        {
            return true;
        }
    }
    return false;
}

/* Iterate to the lastSlice+1. The iterator will point to the past-the-end element in the sequence */
void CoordIterator::moveToPastTheEndElement()
{
    for (auto& dimIter : m_dimCounters)
    {
        dimIter.coord = dimIter.coordLimit;
    }
    updateNext();
}

// Iterators are equal if they iterate on the same tensor and are currently dereferenced into the same slice.
bool CoordIterator::operator==(const CoordIterator& other) const
{
    // Can't compare anything if other iterates another operand.
    if (m_limits != other.m_limits) return false;

    HB_ASSERT(m_snakePatternDims == other.m_snakePatternDims, "Compared iterators must agree on snake-ness");

    return (m_next == other.m_next);
}

bool CoordIterator::operator!=(const CoordIterator& other) const
{
    return !(*this == other);
}

Coordinate CoordIterator::operator*()
{
    return m_next;
}

CoordIterator CoordIterator::getEndIterator(const CoordIterator& iterator)
{
    CoordIterator end = iterator;
    end.moveToPastTheEndElement();
    return end;
}

CoordinateTraversalPattern::CoordinateTraversalPattern(Coordinate limits,
                                                       DimVector  dimOrder,
                                                       DimVector  snakeTraversalDims)
: m_limits(std::move(limits)), m_dimOrder(std::move(dimOrder)), m_snakeTraversalDims(std::move(snakeTraversalDims))
{
    validateDimOrder();
}

CoordIterator CoordinateTraversalPattern::begin() const
{
    CoordIterator beginIterator(m_limits, m_dimOrder, m_snakeTraversalDims);

    return beginIterator;
}

CoordIterator CoordinateTraversalPattern::end() const
{
    return CoordIterator::getEndIterator(begin());
}

void CoordinateTraversalPattern::validateDimOrder() const
{
    // check that all dims in dim order are smaller than the coord size
    for (auto& dim : m_dimOrder)
    {
        HB_ASSERT(dim < m_limits.size(), "dim {} is out of bounds ({})", dim, m_limits.size());
    }
    // check that all dims in snake dims are smaller than the coord size
    for (auto& dim : m_snakeTraversalDims)
    {
        HB_ASSERT(dim < m_limits.size(), "dim {} is out of bounds ({})", dim, m_limits.size());
    }
    // check that the dims in dim order are unique
    HB_ASSERT(areAllElementsUnique(m_dimOrder), "dims in dims order must be unique");
}
