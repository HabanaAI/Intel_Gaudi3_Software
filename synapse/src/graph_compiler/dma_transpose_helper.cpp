#include "dma_transpose_helper.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <memory>

#include "synapse_common_types.h"
#include "tensor.h"
#include "tensor_roi.h"
#include "utils.h"

class DmaTransposeAdapter
{
public:
    DmaTransposeAdapter(const TensorROILayout& srcLayout,
                        const NSizeArray&      srcSizes,
                        TensorROI&             dst,
                        synDataType            elementType,
                        TSize                  maximalDestDim0Bytes)
    : m_elementType(elementType),
      m_elementSizeInBits(dataTypeToSizeInBits(m_elementType)),
      m_maximalDestDim0Bytes(maximalDestDim0Bytes)
    {
        m_srcSize0                 = srcLayout.m_size[0];
        m_srcW                     = srcLayout.m_size[1];
        m_srcH                     = srcLayout.m_size[2];
        m_srcN                     = srcLayout.m_size[3];
        m_dstC                     = srcSizes[0];
        m_dstW                     = srcSizes[1];
        m_dstH                     = srcSizes[2];
        m_dstN                     = srcSizes[3];
        m_dstV                     = srcSizes[4];
        TensorROILayout& dstLayout = dst.getLayout();
        m_dstSizes                 = dstLayout.m_size.data();
        m_dstSrides                = dstLayout.spatialStrides;
    }

    TSize SRC_SIZE_0() const { return safeBitsToByte(m_srcSize0 * m_elementSizeInBits); }

    TSize maximalDestDim0Bytes() const { return m_maximalDestDim0Bytes; }

    TSize w() const { return m_srcW; }

    TSize h() const { return m_srcH; }

    TSize n() const { return m_srcN; }

    // Might be a little limiting, since
    // there isn't much control over the parent tensor.
    TSize C() const { return m_dstC; }

    TSize W() const { return m_dstW; }

    TSize H() const { return m_dstH; }

    TSize N() const { return m_dstN; }

    TSize V() const { return m_dstV; }

    void DST_SIZE_(unsigned i, TSize value)
    {
        TSize size;

        if (i == 0)
        {
            // layout normalized to elementSize;
            size = BITS_PER_BYTE * value / m_elementSizeInBits;
            m_dstSizes[i] = size;
        }
        else
        {
            m_dstSizes[i] = value;
        }
    }

    void DST_STRIDE_(unsigned i, TStride value)
    {
        m_dstSrides[i - 1] = value;
    }

    unsigned ELEMENT_SIZE_IN_BITS() const { return m_elementSizeInBits; }

private:
    TSize*           m_dstSizes;
    TStride*         m_dstSrides;
    synDataType      m_elementType;
    unsigned         m_elementSizeInBits;
    TSize            m_srcSize0;
    TSize            m_srcW;
    TSize            m_srcH;
    TSize            m_srcN;
    TSize            m_dstC;
    TSize            m_dstW;
    TSize            m_dstH;
    TSize            m_dstN;
    TSize            m_dstV;
    TSize            m_maximalDestDim0Bytes;
};

[[maybe_unused]] static void contiguousCNVHW(DmaTransposeAdapter& a)
{
    TStride dstCNVHWBytes = safeBitsToByte(a.ELEMENT_SIZE_IN_BITS() * a.N() * a.V() * a.H() * a.W());
    a.DST_SIZE_(0, a.maximalDestDim0Bytes());
    a.DST_STRIDE_(1, dstCNVHWBytes);
    a.DST_SIZE_(1, BITS_PER_BYTE * a.SRC_SIZE_0() / a.ELEMENT_SIZE_IN_BITS());
    a.DST_STRIDE_(2, a.maximalDestDim0Bytes());
    a.DST_SIZE_(2, dstCNVHWBytes / a.maximalDestDim0Bytes());
    a.DST_STRIDE_(3, 0);
    a.DST_SIZE_(3, 1);
    a.DST_STRIDE_(4, 0);
    a.DST_SIZE_(4, 1);
}

static void CNHW(DmaTransposeAdapter& a)
{
    TStride srcWBytes  = safeBitsToByte(a.w() * a.ELEMENT_SIZE_IN_BITS());
    TStride dstWBytes  = safeBitsToByte(a.ELEMENT_SIZE_IN_BITS() * a.W());
    TStride dstHWBytes = dstWBytes * a.H();
    a.DST_SIZE_(0, std::min(srcWBytes, a.maximalDestDim0Bytes()));
    a.DST_STRIDE_(1, dstHWBytes * a.N());
    a.DST_SIZE_(1, BITS_PER_BYTE * a.SRC_SIZE_0() / a.ELEMENT_SIZE_IN_BITS());
    a.DST_STRIDE_(2, a.maximalDestDim0Bytes());
    a.DST_SIZE_(2, std::max(1ul, srcWBytes / (a.maximalDestDim0Bytes())));
    a.DST_STRIDE_(3, dstWBytes);
    a.DST_SIZE_(3, a.h());
    a.DST_STRIDE_(4, dstHWBytes);
    a.DST_SIZE_(4, a.n());
}

static void NHCW(DmaTransposeAdapter& a)
{
    TStride srcWBytes  = safeBitsToByte(a.w() * a.ELEMENT_SIZE_IN_BITS());
    TStride dstWBytes  = safeBitsToByte(a.ELEMENT_SIZE_IN_BITS() * a.W());
    TStride dstCWBytes = dstWBytes * a.C();
    a.DST_SIZE_(0, std::min(srcWBytes, a.maximalDestDim0Bytes()));
    a.DST_STRIDE_(1, dstWBytes);
    a.DST_SIZE_(1, BITS_PER_BYTE * a.SRC_SIZE_0() / a.ELEMENT_SIZE_IN_BITS());
    a.DST_STRIDE_(2, a.maximalDestDim0Bytes());
    a.DST_SIZE_(2, std::max(1ull, a.w() / (BITS_PER_BYTE * a.maximalDestDim0Bytes() / a.ELEMENT_SIZE_IN_BITS())));
    a.DST_STRIDE_(3, dstCWBytes);
    a.DST_SIZE_(3, a.h());
    a.DST_STRIDE_(4, dstCWBytes * a.H());
    a.DST_SIZE_(4, a.n());
}

static void NCHW(DmaTransposeAdapter& a)
{
    TStride srcWBytes  = safeBitsToByte(a.w() * a.ELEMENT_SIZE_IN_BITS());
    TStride dstWBytes  = safeBitsToByte(a.ELEMENT_SIZE_IN_BITS() * a.W());
    TStride dstHWBytes = dstWBytes * a.H();
    a.DST_SIZE_(0, std::min(srcWBytes, a.maximalDestDim0Bytes()));
    a.DST_STRIDE_(1, dstHWBytes);
    a.DST_SIZE_(1, BITS_PER_BYTE * a.SRC_SIZE_0() / a.ELEMENT_SIZE_IN_BITS());
    a.DST_STRIDE_(2, a.maximalDestDim0Bytes());
    a.DST_SIZE_(2, std::max(1ul, srcWBytes / (a.maximalDestDim0Bytes())));
    a.DST_STRIDE_(3, dstWBytes);
    a.DST_SIZE_(3, a.h());
    a.DST_STRIDE_(4, dstHWBytes * a.C());
    a.DST_SIZE_(4, a.n());
}

[[maybe_unused]] static void HNCW(DmaTransposeAdapter& a)
{
    TStride srcWBytes  = safeBitsToByte(a.w() * a.ELEMENT_SIZE_IN_BITS());
    TStride dstWBytes  = safeBitsToByte(a.ELEMENT_SIZE_IN_BITS() * a.W());
    TStride dstHWBytes = dstWBytes * a.H();
    a.DST_SIZE_(0, std::min(srcWBytes, a.maximalDestDim0Bytes()));
    a.DST_STRIDE_(1, dstHWBytes);
    a.DST_SIZE_(1, safeBitsToByte(a.SRC_SIZE_0() / a.ELEMENT_SIZE_IN_BITS()));
    a.DST_STRIDE_(2, a.maximalDestDim0Bytes());
    a.DST_SIZE_(2, std::max(1UL, srcWBytes) / 128UL);
    a.DST_STRIDE_(3, dstWBytes);
    a.DST_SIZE_(3, a.h());
    a.DST_STRIDE_(4, dstHWBytes * a.C());
    a.DST_SIZE_(4, a.n());
}

void DmaTransposeHelper::checkInOut(synDataType            inElementType,
                                    const TensorROILayout& inLayout,
                                    const TensorROI&       out) const
{
    HB_ASSERT(dataTypeToSizeInBits(inElementType) == m_elementSizeInBits, "Inconsistent element size");
    HB_ASSERT(dataTypeToSizeInBits(out.m_parentTensor->getElementType()) == m_elementSizeInBits,
              "Inconsistent element size");
    TSize maxSrcDim0Size = maxSrcDim0();
    HB_ASSERT(inLayout.m_size[0] <= maxSrcDim0Size,
              "Source dimension 1 ({}) must be less than {}",
              inLayout.m_size[0],
              maxSrcDim0Size);
    checkNumlines(numLines(inLayout.m_size.data()));
}

void DmaTransposeHelper::checkInOut(const TensorROI& in, const TensorROI& out) const
{
    checkInOut(in.m_parentTensor->getElementType(), in.getLayout(), out);
}

void DmaTransposeHelper::transposeByCIndex(uint32_t               cIndex,
                                           synDataType            inElementType,
                                           const TensorROILayout& inLayout,
                                           const NSizeArray&      inSizes,
                                           TensorROI&             out) const
{
    checkInOut(inElementType, inLayout, out);
    DmaTransposeAdapter a(inLayout, inSizes, out, m_elementType, maximalDestDim0Bytes());

    switch (cIndex)
    {
        case 1:
            NHCW(a);
            break;
        case 2:
            NCHW(a);
            break;
        case 3:
            CNHW(a);
            break;
        default:
        {
            HB_ASSERT(cIndex == 3, "Unsupported cIndex: {}", cIndex);
            break;
        }
    }
}
