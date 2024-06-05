#pragma once
#include <cstdint>

#include "infra/defs.h"
#include "dma_transpose_engine_params.h"
#include "utils.h"

class TensorROI;
class TensorROILayout;
class DmaTransposeHelper
{
public:
    static constexpr uint64_t MAX_LINES = 128;
    DmaTransposeHelper(synDataType dType, DmaTransposeEngineParams teParams)
    : m_elementType(dType), m_elementSizeInBits(dataTypeToSizeInBits(m_elementType)), m_teParams(teParams)
    {
    }

    unsigned maxSrcDim0() const { return BITS_PER_BYTE * m_teParams.maxSrc0 / m_elementSizeInBits; }

    uint64_t maximalDestElementsDim0() const
    {
        return std::min<uint64_t>(MAX_LINES, BITS_PER_BYTE * m_teParams.maxDst0 / m_elementSizeInBits);
    }

    uint64_t chunkSizeDim1() const { return maximalDestElementsDim0(); }

    uint64_t maximalDestDim0Bytes() const
    {
        return m_elementSizeInBits == 4 ? m_teParams.maxDst0 / 2 : m_teParams.maxDst0;
    }

    void transposeByCIndex(uint32_t               cIndex,
                           synDataType            inElementType,
                           const TensorROILayout& inLayout,
                           const NSizeArray&      inSizes,
                           TensorROI&             out) const;

    void checkInOut(synDataType inElementType, const TensorROILayout& inLayout, const TensorROI& out) const;
    void checkInOut(const TensorROI& in, const TensorROI& out) const;

    uint64_t numLines(const TSize* sizes, uint32_t maxDim = SYN_MAX_TENSOR_DIM) const
    {
        // When checking numLines, 4 bit is counted as 8.
        return bitsToByte(multiplyElements(sizes + 1, sizes + maxDim) *
                          std::max<uint64_t>(m_elementSizeInBits, BITS_PER_BYTE));
    }

    uint64_t getValidNumLinesRequiredTwoExponent() const
    {
        // When checking numLines, 4 bit is counted as 8.
        return __builtin_ffsll(m_teParams.numLinesDivisor /
                               bitsToByte(std::max<uint64_t>(m_elementSizeInBits, BITS_PER_BYTE))) -
               1;
    }

    bool isValidNumLines(const TSize* sizes, uint32_t maxDim = SYN_MAX_TENSOR_DIM) const
    {
        auto numlines = numLines(sizes, maxDim);
        return numlines % m_teParams.numLinesDivisor == 0 && numlines != 0;
    }

    const DmaTransposeEngineParams& params() const { return m_teParams; }

private:

    void checkNumlines(uint64_t numLines) const
    {
        HB_ASSERT(numLines % m_teParams.numLinesDivisor == 0 || numLines < m_teParams.numLinesDivisor,
              "NUM_LINES must be a multiplication of {} or less",
              m_teParams.numLinesDivisor);
    }

    const synDataType              m_elementType;
    const unsigned                 m_elementSizeInBits;
    const DmaTransposeEngineParams m_teParams;
};
