#include <algorithm>
#include <numeric>
#include <cstring>

#include "transpose_permutation_handler_base.h"
#include "node.h"
#include "node_roi.h"
#include "tensor.h"
#include "utils.h"


TransposePermutationDim operator--(TransposePermutationDim& lhs)
{
    lhs = (TransposePermutationDim)((unsigned int)lhs - 1);
    return lhs;
}

/**
 * The dim converter returns the correct dim by the transpose permutation
 */
class DimToOriginalConvertor
{
public:
    DimToOriginalConvertor(const TransposePermutationArray& originalPermDictionary)
    : m_originalPermDictionary (originalPermDictionary)
    {
    }

    unsigned int operator() (unsigned int dim) const
    {
        return m_originalPermDictionary[dim];
    }

private:
    const TransposePermutationArray& m_originalPermDictionary;
};

/**
 * Returns the dim as is
 */
class DimPassThrough
{
public:
    unsigned int operator() (unsigned int dim) const
    {
        return dim;
    }
};

/**
 * Accumulate sizes listed in listToAcc
 * The correct dimension is calculated by DimPassThrough or DimToOriginalConvertor
 */
template<class DimConverter>
int32_t accumulateSize(const Tensor& tensor,
                       const std::list<TransposePermutationDim>& listToAcc,
                       const DimConverter& converter)
{
    return std::accumulate(listToAcc.begin(),
                           listToAcc.end(),
                           1,
                           [&tensor, &converter](unsigned int acc, unsigned int dim){ return tensor.getSizeInElements(converter(dim)) * acc; });
}

void TransposePermutationHandlerBase::calculateHeightAndWidthDims()
{
    auto dimIter = m_permutation.rbegin();

    while (dimIter != m_permutation.rend())
    {
        m_matrixHeightDims.push_front(*dimIter);
        if (*dimIter == TPD_Channel)
        {
            ++dimIter;
            break;
        }
        ++dimIter;
    }
    while(dimIter != m_permutation.rend())
    {
        m_matrixWidthDims.push_front(*dimIter);
        ++dimIter;
    }
}

TransposePermutationHandlerBase::TransposePermutationHandlerBase(const TransposePermutationArray& permutation)
: m_permutation(permutation)
{
    // Used only for FCD transpose
    HB_ASSERT(permutation[0] != TPD_Channel, "used only for FCD transpose");

    m_originalPermDictionary.resize(m_permutation.size());
    for (unsigned int i = 0; i< m_permutation.size(); ++i)
    {
        m_originalPermDictionary[m_permutation[i]] = static_cast<TransposePermutationDim>(i);
    }

    calculateHeightAndWidthDims();
}


uint32_t TransposePermutationHandlerBase::IfmSpatialSize(const Tensor& IFM) const
{
    return accumulateSize(IFM, m_matrixWidthDims, DimPassThrough());
}

uint32_t TransposePermutationHandlerBase::IfmSpatialStrides(const Tensor& IFM) const
{
    return IFM.getStrideInElements(m_matrixWidthDims.back() + 1);
}

uint32_t TransposePermutationHandlerBase::IfmFcdSize(const Tensor& IFM) const
{
    return accumulateSize(IFM, m_matrixHeightDims, DimPassThrough());
}

uint32_t TransposePermutationHandlerBase::IfmFcdStrides(const Tensor& IFM) const
{
    return IFM.getStrideInElements(m_matrixHeightDims.back() + 1);
}

uint32_t TransposePermutationHandlerBase::OfmSpatialSize(const Tensor& OFM) const
{
    return accumulateSize(OFM, m_matrixHeightDims, DimToOriginalConvertor(m_originalPermDictionary));
}

uint32_t TransposePermutationHandlerBase::OfmSpatialStrides(const Tensor& OFM) const
{
    return OFM.getStrideInElements(m_originalPermDictionary[m_matrixHeightDims.back()] + 1);
}

uint32_t TransposePermutationHandlerBase::OfmFcdStrides(const Tensor& OFM) const
{
    return OFM.getStrideInElements(m_originalPermDictionary[m_matrixWidthDims.back()] + 1);
}

uint32_t TransposePermutationHandlerBase::OfmFcdSize(const Tensor& OFM) const
{
    return accumulateSize(OFM, m_matrixWidthDims, DimToOriginalConvertor(m_originalPermDictionary));
}
