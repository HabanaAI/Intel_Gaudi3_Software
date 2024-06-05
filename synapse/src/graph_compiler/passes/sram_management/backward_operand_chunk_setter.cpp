#include "backward_operand_chunk_setter.h"
#include "habana_graph.h"

BackwardOperandChunkSetter::BackwardOperandChunkSetter(const HabanaGraph& graph, const StrategySlicingData& slicingData)
: m_graph(graph), m_slicingData(slicingData)
{
    for (const auto& operand : m_slicingData.bundleTensors)
    {
        m_tensorToOperand[operand->originalTensor] = operand;
    }
    m_tensorToOperand[m_slicingData.masterOperand->originalTensor] = m_slicingData.masterOperand;

}

void BackwardOperandChunkSetter::setDimensionChunk(const pSlicedOperand& startOperand, uint32_t dim, uint32_t chunk)
{
    std::list<std::pair<pSlicedOperand, uint32_t>> operandsToSet;
    operandsToSet.push_back(std::make_pair(startOperand, dim));
    while (!operandsToSet.empty())
    {
        pSlicedOperand operand = operandsToSet.front().first;
        dim = operandsToSet.front().second;
        operandsToSet.pop_front();
        if (operand->originalTensor->getDim() < dim) continue;

        operand->chunkDimensions[dim] = chunk;
        // Walk thorough all node outputs
        NodePtr producer = m_graph.getTensorProducer(operand->originalTensor);
        if (!producer) continue;
        for (const TensorPtr& output : producer->getOutputs())
        {
            auto operandIter = m_tensorToOperand.find(output);
            if (operandIter != m_tensorToOperand.end() &&
                operandIter->second != operand &&
                operandIter->second->originalTensor->getDim() > dim)
            {
                operandIter->second->chunkDimensions[dim] = chunk;
            }
        }
        pSliceReference operandSliceRef = std::make_shared<SliceReference>(operand);
        // Change the changed dim coord, to know the new dim for the input operand
        operandSliceRef->coordinates[dim] = 1;
        // Add bwd mapped inputs
        const auto& inputs = m_slicingData.getInputsForSlice(operandSliceRef);
        for (const auto& inRef : inputs)
        {
            auto oneIter = std::find(inRef->coordinates.begin(), inRef->coordinates.end(), 1);
            if (oneIter != inRef->coordinates.end())
            {
                uint32_t dim = std::distance(inRef->coordinates.begin(), oneIter);
                operandsToSet.push_back(std::make_pair(inRef->operand, dim));
            }
        }
    }
}
