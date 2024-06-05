#pragma once

#include "access_pattern.h"
#include "sram_management/slicing_utils.h"

using namespace gc::access_pattern;

// Map the given operands sliced dimensions to the dimensions of all other operands of the node with the given access
// pattern
class SlicedDimsProjector
{
    using ProjectedDims = std::set<Dim>;

public:
    template<typename SlicedOperandsContainer>
    SlicedDimsProjector(const NodeAccessPatternPtr& nodeAccessPattern, const SlicedOperandsContainer& slicedOperands)
    : m_accessPattern(nodeAccessPattern)
    {
        for (const auto& slicedOp : slicedOperands)
        {
            const auto& slicedDims                       = SlicedOperandUtils::getSlicedDims(slicedOp);
            m_tensorSlicedDims[slicedOp->originalTensor] = MultiDims(slicedDims.begin(), slicedDims.end());
        }
    }

    MultiDims getSlicedDims(const TensorPtr& tensor) const
    {
        auto iter = m_tensorSlicedDims.find(tensor);
        if (iter != m_tensorSlicedDims.end())
        {
            return iter->second;
        }

        // Not in the given sliced operands. Project and collect which operand dimensions are sliced.
        // I don't see it being requested more then once a.t.m, so not caching this value.
        ProjectedDims tensorSlicedDims;
        for (const auto& tensorAndSlicingDims : m_tensorSlicedDims)
        {
            tensorSlicedDims.merge(
                projectSlicedTensor(tensor, tensorAndSlicingDims.first, tensorAndSlicingDims.second));
        }
        return MultiDims {tensorSlicedDims.begin(), tensorSlicedDims.end()};
    }

private:
    ProjectedDims projectSlicedTensor(const TensorPtr& queriedTensor,
                                      const TensorPtr& givenTensor,
                                      const MultiDims& givenSlicedDims) const
    {
        ProjectedDims pd;
        for (Dim slicedDim : givenSlicedDims)
        {
            pd.merge(projectSlicedDim(queriedTensor, givenTensor, slicedDim));
        }
        return pd;
    }

    ProjectedDims
    projectSlicedDim(const TensorPtr& queriedTensor, const TensorPtr& givenTensor, Dim givenTensorDim) const
    {
        MultiDims projectedDims =
            m_accessPattern->getTensorMatchingSlicedDims(queriedTensor, givenTensor, givenTensorDim);
        return ProjectedDims {projectedDims.begin(), projectedDims.end()};
    }

    NodeAccessPatternPtr                     m_accessPattern;
    std::unordered_map<TensorPtr, MultiDims> m_tensorSlicedDims;
};
