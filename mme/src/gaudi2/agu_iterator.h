#pragma once
#include "gaudi2/mme.h"
#include <vector>
#include <array>

namespace Gaudi2
{
class AguIterator
{
public:
    AguIterator(Mme::Desc* aguDescBase,
                unsigned firstNr,
                unsigned secondNr,
                unsigned thirdNr,
                bool isInput,
                const std::vector<unsigned>& sbIndices)
    : m_desc(aguDescBase), m_totalSizes {firstNr, secondNr, thirdNr}, m_isInput(isInput), m_sbIndices(sbIndices)
    {
    }
    void initialize(unsigned initialFirstIdx, unsigned initialSecondIdx, unsigned initialThirdIdx);
    Mme::MmeAguCoreDesc* getNext(unsigned coreOffset = 0, unsigned portOffset = 0);
    bool shouldSkipSpatialStep(unsigned pairIdx) const;
    const std::array<unsigned, 3>& getCurrentAgu() const;
    const bool getWasFirstDimStep() const { return m_wasFirstDimStep; }
    void setIsInput(bool val) { m_isInput = val; }
    bool isEvenPort() { return ((m_currentAgu[sbIdxInArray] % 2) == 0); }

protected:
    unsigned sbIdxInArray;
    unsigned coresIdxInArray;
    unsigned pairsIdxInArray;

private:
    bool counter(const uint32_t* totalSizes,
                 const uint32_t* steps,
                 const unsigned dimNr,
                 const uint32_t* src,
                 uint32_t* dst);
    Mme::Desc* m_desc;
    std::array<unsigned, 3> m_totalSizes;
    std::array<unsigned, 3> m_currentAgu;
    const std::vector<unsigned>& m_sbIndices;
    bool initialized = false;
    bool m_isInput = true;
    bool m_wasFirstDimStep = false;
};

class SpatialAguIterator : public AguIterator
{
public:
    SpatialAguIterator(Mme::Desc* desc,
                       unsigned pairsNr,
                       unsigned coresPerPairNr,
                       unsigned portsPerCoreNr,
                       const std::vector<unsigned>& sbIndices);
};

class DenseAguIterator : public AguIterator
{
public:
    DenseAguIterator(Mme::Desc* desc,
                     unsigned portsPerCoreNr,
                     unsigned coresPerPairNr,
                     unsigned pairsNr,
                     const std::vector<unsigned>& sbIndices);
};

class BatchAguIterator : public AguIterator
{
public:
    BatchAguIterator(Mme::Desc* desc,
                     unsigned portsPerCoreNr,
                     unsigned coresPerPairNr,
                     unsigned pairsNr,
                     const std::vector<unsigned>& sbIndices);
};

}  // namespace Gaudi2