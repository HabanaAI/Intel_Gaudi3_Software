#pragma once

#include "node.h"
#include "types.h"
#include <unordered_map>

// Hold the specific slice details of an operation slice
// This is a partial stub. In the future we'll likely want more details like the index space offset and pointer to the
// full index space and/or coordinates of the slice in the sliced operation.
class OperationSlice
{
public:
    using OffsetArray = ::OffsetArray;

    // Reserve capacity in tensorOffsets for spill/fill fusion for every tensor
    explicit OperationSlice(const Node* thisNode)
    : m_thisNode(thisNode), m_tensorOffsets(thisNode->getOperands().size() * 2)
    {
    }

    struct TensorSliceOffset
    {
        TensorPtr   origTensor;
        OffsetArray sliceOffset;
    };

    void addTensorSliceOffset(const TensorPtr&   sliceTensor,
                              const TensorPtr&   origTensor,
                              const OffsetArray& sliceOffsetPerDim);

    const OffsetArray&      getTensorSliceOffset(const TensorPtr& sliceTensor) const;
    OffsetArray::value_type getTensorSliceOffsetInDim(const TensorPtr& sliceTensor, unsigned dim) const;

    TensorPtr getOriginalTensor(const TensorPtr& sliceTensor) const;

private:
    const Node*                    m_thisNode;
    std::vector<TensorSliceOffset> m_tensorOffsets;

    size_t tensorIdx(const TensorPtr& sliceTensor) const;
};