#include "agu_iterator.h"
#include "mme_assert.h"

namespace Gaudi2
{
void AguIterator::initialize(unsigned int initialFirstIdx, unsigned int initialSecondIdx, unsigned int initialThirdIdx)
{
    m_currentAgu = {initialFirstIdx, initialSecondIdx, initialThirdIdx};
    initialized = true;
}

bool AguIterator::shouldSkipSpatialStep(unsigned int pairIdx) const
{
    return (m_totalSizes[pairsIdxInArray] != 1) && (m_currentAgu[pairsIdxInArray] != pairIdx);
}

Mme::MmeAguCoreDesc* AguIterator::getNext(unsigned coreOffset, unsigned portOffset)
{
    // sanity check
    MME_ASSERT(initialized, "m_currentAgu is not initialized");
    // increment m_currentAgu
    std::array<unsigned, 3> step = {1, 0, 0};
    m_wasFirstDimStep = m_currentAgu[0] + 1 < m_totalSizes[0];
    bool ret = counter(m_totalSizes.data(), step.data(), m_totalSizes.size(), m_currentAgu.data(), m_currentAgu.data());
    unsigned sbId = m_sbIndices[m_currentAgu[sbIdxInArray] + portOffset];
    unsigned coreId = m_currentAgu[coresIdxInArray] + coreOffset;
    return ret ? &(m_isInput ? m_desc->aguIn : m_desc->aguOut)[sbId][coreId] : nullptr;
}

const std::array<unsigned, 3>& AguIterator::getCurrentAgu() const
{
    return m_currentAgu;
}

bool AguIterator::counter(const uint32_t* totalSizes,
                          const uint32_t* steps,
                          const unsigned dimNr,
                          const uint32_t* src,
                          uint32_t* dst)
{
    unsigned carry = 0;
    for (unsigned dim = 0; dim < dimNr; ++dim)
    {
        unsigned totalInDim = src[dim] + steps[dim] + carry;
        dst[dim] = totalInDim % totalSizes[dim];
        carry = totalInDim / totalSizes[dim];
    }

    return carry == 0;
}

SpatialAguIterator::SpatialAguIterator(Mme::Desc* desc,
                                       unsigned pairsNr,
                                       unsigned coresPerPairNr,
                                       unsigned portsPerCoreNr,
                                       const std::vector<unsigned int>& sbIndices)
: AguIterator(desc, pairsNr, coresPerPairNr, portsPerCoreNr, false, sbIndices)
{
    sbIdxInArray = 2;
    coresIdxInArray = 1;
    pairsIdxInArray = 0;
}

DenseAguIterator::DenseAguIterator(Mme::Desc* desc,
                                   unsigned int portsPerCoreNr,
                                   unsigned int coresPerPairNr,
                                   unsigned int pairsNr,
                                   const std::vector<unsigned int>& sbIndices)
: AguIterator(desc, portsPerCoreNr, coresPerPairNr, pairsNr, false, sbIndices)
{
    sbIdxInArray = 0;
    coresIdxInArray = 1;
    pairsIdxInArray = 2;
}

BatchAguIterator::BatchAguIterator(Mme::Desc* desc,
                                   unsigned int portsPerCoreNr,
                                   unsigned int coresPerPairNr,
                                   unsigned int pairsNr,
                                   const std::vector<unsigned int>& sbIndices)
: AguIterator(desc, portsPerCoreNr, coresPerPairNr, pairsNr, false, sbIndices)
{
    sbIdxInArray = 0;
    coresIdxInArray = 1;
    pairsIdxInArray = 2;
}

}  // namespace Gaudi2