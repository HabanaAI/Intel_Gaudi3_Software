#pragma once

#include "strategy_slicing_data.h"

class HabanaGraph;

/**
 * Given a start operand and chunk size to a specific dim
 * This class update all operands chunk size in a backward mapping way for the specific dim
 */
class BackwardOperandChunkSetter
{
public:
    BackwardOperandChunkSetter(const HabanaGraph& graph, const StrategySlicingData& slicingData);

    void setDimensionChunk(const pSlicedOperand& startOperand, uint32_t dim, uint32_t chunk);

private:
    const HabanaGraph&                m_graph;
    const StrategySlicingData& m_slicingData;
    std::map<pTensor, pSlicedOperand> m_tensorToOperand;
};
