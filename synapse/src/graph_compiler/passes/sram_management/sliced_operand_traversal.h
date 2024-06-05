#pragma once

#include "bundle.h"

class MultiOperandSliceIterator;

// Represents the sliced tensor as a traverse-able object. Each "element" in the traversal is a slice of the tensor.
// The order of the slices is determined by the order of dimensions given when the traverser is constructed.
class SlicedOperandTraversalPattern
{
public:
    static const DimVector LEFT_TO_RIGHT_1D;
    static const DimVector LEFT_TO_RIGHT_2D;
    static const DimVector TOP_TO_BOTTOM_2D;
    static const DimVector LEFT_TO_RIGHT_4D;

    SlicedOperandTraversalPattern(pSlicedOperand slicedOperand,
                                  DimVector      dimOrder,
                                  bool           snakeTraversal       = false,
                                  unsigned       numOfCommonDimSlices = 1);
    MultiOperandSliceIterator begin() const;
    MultiOperandSliceIterator end() const;
    unsigned numCommonDimSlices() const { return m_numOfCommonDimSlices; }
    void addSlave(const SlicedOperandTraversalPattern& slaveTraversalPattern) {m_slaveTraversalPatterns.push_back(slaveTraversalPattern);};
    void setInputSliceChangeDim(unsigned interestingDimForSliceChangeSignal);
    void                      setDimOrder(DimVector&& dimOrder);
    const pSlicedOperand& getOperand() const;
private:
    pSlicedOperand                             m_operand;
    DimVector                                  m_dimOrder;
    bool                                       m_snakeTraversal;
    unsigned                                   m_numOfCommonDimSlices;
    Settable<unsigned>                         m_masterInputSliceChangeDim;
    std::vector<SlicedOperandTraversalPattern> m_slaveTraversalPatterns;
    void validateDimOrder() const;
};

using SliceRefCommonDimIdxPair = std::pair<pSliceReference, unsigned>;
// Iterates over the slices of a sliced tensor
class OperandSliceIterator
{
public:
    OperandSliceIterator(const pSlicedOperand& slicedOperand,
                         DimVector             dimOrder,
                         bool                  snakePattern         = false,
                         unsigned              numOfCommonDimSlices = 1);
    virtual ~OperandSliceIterator() = default;
    OperandSliceIterator(const OperandSliceIterator& other) = default;
    OperandSliceIterator(OperandSliceIterator&& other)      = default;

    OperandSliceIterator& operator++();      // prefix inc
    OperandSliceIterator operator++(int);    // postfix inc
    bool operator==(const OperandSliceIterator& other) const;
    bool operator!=(const OperandSliceIterator& other) const;
    SliceRefCommonDimIdxPair operator*();

    unsigned getTotalOfCommonDimSlices()                  const { return m_commonDimCounter.coordLimit; }
    bool     inputSliceChanged()                          const { return m_inputSliceChanged; }
    void     setDimForSliceChangeIndication(unsigned dim)       { m_dimensionForSliceChangeIndication = dim; }

    /* Returns an iterator referring to the past-the-end element. */
    static OperandSliceIterator getEndIterator(const OperandSliceIterator& iterator);
private:

    // Count the coordinates of a dimension slices. Restarts once advance is called after the last coordinate.
    // In snake walking pattern, restarted counter will reverse direction and count from end to start.
    class CoordCyclicCounter
    {
    public:
        const unsigned dim;
        int      coord = 0;
        const int      coordLimit;

        CoordCyclicCounter(unsigned dimension, int limit, bool snake=true)
        : dim(dimension), coordLimit(limit), m_snake(snake), m_currentLimit(limit) {}

        void advance();
        bool lastAdvanceRestartedCount() const { return m_countRestarted; }

    private:
        const bool m_snake;
        int  m_currentLimit;
        int  m_advanceDir = 1;
        bool m_countRestarted = false;

        void restart();
    };

    const DimVector                 m_dimOrder; /* The pattern deduced from the dim order   */
    CoordCyclicCounter              m_commonDimCounter;
    std::vector<CoordCyclicCounter> m_dimCounters;
    SliceReference m_next;
    bool m_snakePattern;
    bool m_inputSliceChanged = false; /* indicates if the iterator new position require input slice change */

    unsigned m_dimensionForSliceChangeIndication; /* When num of common dim slices is and this dimension changes, */
                                                  /* the 'inputSliceChanged' signal will be set                     */

    void init();
    void advance();
    void moveToPastTheEndElement();
    void updateNext();

    bool ended() const {  return m_commonDimCounter.coord == m_commonDimCounter.coordLimit; }
};

// Composite class that holds several OperandSliceIterators and increment them in a round robin.
// When an iterator needs to change wide slice , it waits until all the other iterators also needs to change the wide slice.
class MultiOperandSliceIterator
{
public:
    MultiOperandSliceIterator(const OperandSliceIterator& iteratorBegin, const OperandSliceIterator& iteratorEnd);
    MultiOperandSliceIterator(OperandSliceIterator&& iteratorBegin, OperandSliceIterator&& iteratorEnd);
    explicit MultiOperandSliceIterator(OperandSliceIterator iterator);
    virtual ~MultiOperandSliceIterator() = default;
    MultiOperandSliceIterator(const MultiOperandSliceIterator& other);
    // adds new operand iterator to the round robin
    void addOperandIterator(const OperandSliceIterator& iteratorBegin, const OperandSliceIterator& iteratorEnd);
    void addOperandIterator(const MultiOperandSliceIterator& iterator);
    // get iterator pointing to the end position.
    MultiOperandSliceIterator getEndIterator() const;
    OperandSliceIterator getEndIterator(const OperandSliceIterator& iterator) const;
    OperandSliceIterator& getCurrentIterator() { return m_iterators.at(m_currentIteratorIdx).m_iterator;}
    // regular iterator methods
    MultiOperandSliceIterator& operator++();   // prefix increment
    MultiOperandSliceIterator operator++(int); // postfix increment
    bool operator==(const MultiOperandSliceIterator& other) const;
    bool operator!=(const MultiOperandSliceIterator& other) const;
    SliceRefCommonDimIdxPair operator*();
private:
    class OperandSliceIteratorWrapper
    {
    public:
        OperandSliceIteratorWrapper(OperandSliceIterator iterator, OperandSliceIterator iteratorEnd, bool suspended)
        : m_iterator(std::move(iterator)), m_endIterator(std::move(iteratorEnd)), m_suspended(suspended)
        {

        }
        OperandSliceIterator m_iterator;
        OperandSliceIterator m_endIterator;
        bool                 m_suspended;
    };
    /* Reset suspension of all iterators, iterators that have already finished the iteration will be skipped*/
    void resetIteratorsSuspension();
    /* Advance on the next iterator */
    virtual void advance();
    /* Find the next iterator in the list that can be incremented without wide slice change.
     * If all iterators needs a wide slice change - start incrementing the list front.*/
    unsigned getNextAvailableIterator();

    std::vector<OperandSliceIteratorWrapper> m_iterators;          /* Operand iterators, their equivalent end iterators
                                                                    * and suspension status*/
    unsigned                                 m_currentIteratorIdx; /* points to the current iterator in the vector. */
};
