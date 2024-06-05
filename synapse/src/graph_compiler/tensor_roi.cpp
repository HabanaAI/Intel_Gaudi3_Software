#include <cstring>
#include <algorithm>

#include "tensor_roi.h"
#include "utils.h"

typedef DataRange<uint64_t> Range;

uint64_t TensorROILayout::getByteOffset(TOffset coord[Tensor::c_tensorMaxNDim], unsigned elementSizeInBits)
{
    HB_ASSERT(tensorDim <= Tensor::c_tensorMaxNDim, "tensor dimension is bigger than maximum dimensions");
    int64_t  sizeInBits = static_cast<int64_t>(coord[0]) * elementSizeInBits;
    uint64_t offset     = baseAddress + safeBitsToByte(sizeInBits);
    for (unsigned i = 1; i < tensorDim; ++i)
    {
        offset += coord[i] * spatialStrides[i-1];
    }
    return offset;
}

void TensorROILayout::getStartIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const
{
    memcpy(idx, m_baseOffset, sizeof(m_baseOffset));
}

void TensorROILayout::getEndIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const
{
    TOffset sOffset[Tensor::c_tensorMaxNDim];
    getStartIndex(sOffset);
    for (unsigned dim = 0; dim < Tensor::c_tensorMaxNDim; ++dim)
    {
        idx[dim] = sOffset[dim] + m_size[dim];
    }
}

TensorSizesVector TensorROI::getDimSizesInElements() const
{
    TensorSizesVector dimSizes(std::begin(m_layout.m_size), std::begin(m_layout.m_size) + m_layout.tensorDim);
    return dimSizes;
}

TensorStridesVector TensorROI::getStridesNoFcdDim() const
{
    TensorStridesVector dimStrides(std::begin(m_layout.spatialStrides),
                                   std::begin(m_layout.spatialStrides) + m_layout.tensorDim);
    return dimStrides;
}

TensorStridesVector TensorROI::getStridesWithFcdDim() const
{
    TensorStridesVector dimStrides = {m_parentTensor->getElementSizeInBytes()};
    dimStrides.insert(dimStrides.end(),
                      std::begin(m_layout.spatialStrides),
                      std::begin(m_layout.spatialStrides) + m_layout.tensorDim);
    return dimStrides;
}

TensorROILayout& TensorROI::getLayout()
{
    return m_layout;
}

const TensorROILayout& TensorROI::getLayout() const
{
    return m_layout;
}

OverlapRoi& TensorROI::getOverlapROI()
{
    return m_overlapRoi;
}

const OverlapRoi& TensorROI::getOverlapROI() const
{
    return m_overlapRoi;
}

void TensorROI::getStartIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const
{
    m_layout.getStartIndex(idx);
    for (unsigned dim = 0; dim < Tensor::c_tensorMaxNDim; ++dim)
    {
        idx[dim] = std::max<TOffset>(0, idx[dim]);
    }
}
void TensorROI::getEndIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const
{
    TOffset sOffset[Tensor::c_tensorMaxNDim];
    getStartIndex(sOffset);
    m_layout.getStartIndex(idx);
    for (unsigned dim = 0; dim < Tensor::c_tensorMaxNDim; ++dim)
    {
        idx[dim] = std::min<TOffset>(m_parentTensor->getSizeInElements(dim), sOffset[dim] + idx[dim]);
    }
}

void TensorROI::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;

    LOG_DEBUG(GC, "ROI for Tensor {}", m_parentTensor->getName());
    TOffset sOffset[Tensor::c_tensorMaxNDim], eOffset[Tensor::c_tensorMaxNDim];
    getStartIndex(sOffset);
    getEndIndex(eOffset);
    LOG_DEBUG(GC, "From {},{},{},{},{} to {},{},{},{},{} [FCD on the right]", sOffset[4], sOffset[3], sOffset[2], sOffset[1], sOffset[0],
                                                                                eOffset[4], eOffset[3], eOffset[2], eOffset[1], eOffset[0]);
    LOG_DEBUG(GC, "ROI is located at 0x{:x} in {}", m_layout.baseAddress, m_layout.inSram ? "SRAM" : "DRAM");
    LOG_DEBUG(GC, "Roi strides: {}, {}, {}, {}", m_layout.spatialStrides[3], m_layout.spatialStrides[2], m_layout.spatialStrides[1], m_layout.spatialStrides[0]);
    LOG_DEBUG(GC, "ROI is made up of {} sub-ROIS", m_overlapRoi.subRois->size());
    for (const auto& tsr : *m_overlapRoi.subRois)
    {
        LOG_DEBUG(GC, "Sub-ROI signal index is {}", tsr.relSoIdx);
        LOG_DEBUG(GC, "Sub-ROI is made up of {} linear ranges", tsr.ranges.size());
        for (const Range& r : tsr.ranges)
        {
            LOG_DEBUG(GC, "Start: 0x{:x}, Size: {}", r.start(), r.size());
        }
    }
}
