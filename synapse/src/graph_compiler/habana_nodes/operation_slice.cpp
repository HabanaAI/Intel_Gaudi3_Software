#include "operation_slice.h"
#include "defs.h"
#include "log_manager.h"
#include "utils.inl"
#include "tensor.h"

void OperationSlice::addTensorSliceOffset(const TensorPtr&   sliceTensor,
                                          const TensorPtr&   origTensor,
                                          const OffsetArray& sliceOffsetPerDim)
{
    LOG_TRACE(OP_SLICE,
              "OperationSlice: Adding slice offset for slice tensor: {} offset: [{}] in original tensor: {}",
              sliceTensor->getName(),
              toString(sliceOffsetPerDim.begin(), sliceOffsetPerDim.end(), ','),
              origTensor->getName());

    auto idx = tensorIdx(sliceTensor);
    HB_ASSERT(idx < m_tensorOffsets.size(), "tensorOffset size is too small to add fusion tensor");

    m_tensorOffsets[idx] = TensorSliceOffset {origTensor, sliceOffsetPerDim};
}

const OperationSlice::OffsetArray& OperationSlice::getTensorSliceOffset(const TensorPtr& sliceTensor) const
{
    return m_tensorOffsets.at(tensorIdx(sliceTensor)).sliceOffset;
}

OperationSlice::OffsetArray::value_type OperationSlice::getTensorSliceOffsetInDim(const TensorPtr& sliceTensor,
                                                                                  unsigned         dim) const
{
    return getTensorSliceOffset(sliceTensor).at(dim);
}

TensorPtr OperationSlice::getOriginalTensor(const TensorPtr& sliceTensor) const
{
    const TensorPtr& savedOrigTensor = m_tensorOffsets.at(tensorIdx(sliceTensor)).origTensor;
    if (!savedOrigTensor)
    {
        // In case the slice creator did not add slice information to all the operands, some operands are mapped to a
        // null TensorSliceInfo. This is possible where the slice node uses the original tensor, which means that
        // sliceTensor == originalTensor, so:
        return sliceTensor;
    }
    return savedOrigTensor;
}

size_t OperationSlice::tensorIdx(const TensorPtr& sliceTensor) const
{
    TensorVector allOperands = m_thisNode->getOperands();

    auto sliceTensorIter = std::find(allOperands.begin(), allOperands.end(), sliceTensor);
    HB_ASSERT(sliceTensorIter != allOperands.end(),
              "Tensor slice {} wasn't found in slice node {} operands.",
              sliceTensor->getName(),
              m_thisNode->getNodeName());

    return std::distance(allOperands.begin(), sliceTensorIter);
}