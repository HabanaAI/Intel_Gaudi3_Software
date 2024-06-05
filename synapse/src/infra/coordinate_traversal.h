#pragma once

#include "types.h"

using Coordinate = llvm_vecsmall::SmallVector<uint64_t, HABANA_DIM_MAX>;

// Provides methods to iterate coordinates indices according to a requested dim order, with the given limits.
// The caller provides the limits vector, with limit per coordinate, in the natural order of coordinate dims (0,
// 1, 2....), and the dim order - a subset of dims to set which coordinate dim should advance first.
// The vectors size for limits represent the rank of a coordinate - how many dims it has.
// Snake pattern dims is a subset of the dims which need to advance in snake pattern.

class CoordIterator
{
public:
    CoordIterator(Coordinate limits, DimVector dimOrder, DimVector snakePatternDims);
    virtual ~CoordIterator()                  = default;
    CoordIterator(const CoordIterator& other) = default;
    CoordIterator(CoordIterator&& other)      = default;

    CoordIterator& operator++();     // prefix inc
    CoordIterator  operator++(int);  // postfix inc
    bool           operator==(const CoordIterator& other) const;
    bool           operator!=(const CoordIterator& other) const;
    Coordinate     operator*();

    /* Returns an iterator referring to the past-the-end element. */
    static CoordIterator getEndIterator(const CoordIterator& iterator);

protected:
    // Count the coordinates of a dimension values. Restarts once advance is called after the last coordinate.
    // In snake walking pattern, restarted counter will reverse direction and count from end to start.
    class CoordCyclicCounter
    {
    public:
        const unsigned dim;
        int            coord = 0;
        const int      coordLimit;

        CoordCyclicCounter(unsigned dimension, int limit, bool snake = true)
        : dim(dimension), coordLimit(limit), m_snake(snake), m_currentLimit(limit)
        {
        }

        enum AdvanceStatus
        {
            STEP,
            WRAP_AROUND,
            TURN_AROUND
        };

        AdvanceStatus advance();

    private:
        const bool m_snake;
        int        m_currentLimit;
        int        m_advanceDir     = 1;

        void restart();
    };

    const DimVector                 m_dimOrder; /* The pattern deduced from the dim order */
    Coordinate                      m_limits;
    std::vector<CoordCyclicCounter> m_dimCounters;
    Coordinate                      m_next;
    DimVector                       m_snakePatternDims;

    void init();
    void addCounterForDim(uint8_t dim);
    void advance();
    void moveToPastTheEndElement();
    void updateNext();
    bool ended() const;
};

class CoordinateTraversalPattern
{
public:
    // TODO - accept dimOrder as any container with order
    CoordinateTraversalPattern(Coordinate limits, DimVector dimOrder, DimVector snakeTraversalDims);
    CoordIterator begin() const;
    CoordIterator end() const;

private:
    Coordinate m_limits;
    DimVector  m_dimOrder;
    DimVector  m_snakeTraversalDims;
    void       validateDimOrder() const;
};