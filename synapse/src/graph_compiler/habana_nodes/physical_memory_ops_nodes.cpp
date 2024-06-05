#include "physical_memory_ops_nodes.h"
#include "calculate_tensor_roi_linear_ranges.h"

template <class BASE>
void PhysicalMemoryOpNode<BASE>::calculateLinearRanges(TensorROI& tRoi, const NodePtr& n, bool isInput) const
{
    CalculateTensorROIsLinearRanges::calculateMemoryRanges(tRoi, n, isInput);
}

// SERIALIZE and DESERIALIZE patch up their actual memory layout during runtime.
// they are compiled with max sizes, so for every activation the maximum address that we might access is correct
// However, the lowest address for each activation is known only during runtime.
// Therefor, we assume the worst - use the tensor base address for every activation.
template <class BASE>
void PhysicalMemoryOpNode<BASE>::fixLinearRangesToRealParentStart(TensorROI& tRoi) const
{
    uint64_t parentAddress = Tensor::getRealTensor(tRoi.m_parentTensor)->getTensorOffset();
    for (auto& subRoi : *tRoi.m_overlapRoi.subRois)
    {
        for (auto& r : subRoi.ranges)
        {
            r.start(parentAddress);
        }
    }
}

// use the full linear address range of the parent tensor
template <class BASE>
void PhysicalMemoryOpNode<BASE>::applyFullViewLinearRange(TensorROI& tRoi) const
{
    tRoi.m_overlapRoi.subRois->resize(1);

    uint64_t            startAddress = Tensor::getRealTensor(tRoi.m_parentTensor)->getTensorOffset();
    uint64_t            endAddress   = startAddress + Tensor::getRealTensor(tRoi.m_parentTensor)->getDenseSizeInBytes();
    DataRange<uint64_t> r(startAddress, endAddress);
    tRoi.m_overlapRoi.subRois->back().ranges.push_back(r);
}

// explicit instantiations: there will be only these two
template class PhysicalMemoryOpNode<DMAMemcpyNode>;
template class PhysicalMemoryOpNode<TPCMemcpyNode>;
