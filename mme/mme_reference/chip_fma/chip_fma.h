#pragma once

#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{
class ChipFma
{
public:
    static std::unique_ptr<ChipFma> getChipFma(ChipType chipType,
                                               EMmeDataType inputA,
                                               EMmeDataType inputB,
                                               uint8_t expBiasA = 0,
                                               uint8_t expBiasB = 0,
                                               MmeCommon::RoundingMode rmEU = RoundingMode::RoundToNearest,
                                               MmeCommon::RoundingMode rmACC = RoundingMode::RoundToNearest,
                                               bool clipFp = false,
                                               bool clipFpInfIn = false,
                                               MmeCommon::InfNanMode infNanModeA = e_mme_full_inf_nan,
                                               MmeCommon::InfNanMode infNanModeB = e_mme_full_inf_nan);

    ChipFma(EMmeDataType inputA,
            EMmeDataType inputB,
            uint8_t expBiasA,
            uint8_t expBiasB,
            MmeCommon::RoundingMode rmEU = RoundingMode::RoundToNearest,
            MmeCommon::RoundingMode rmACC = RoundingMode::RoundToNearest,
            bool clipFp = false,
            bool clipFpInfIn = false,
            MmeCommon::InfNanMode infNanModeA = e_mme_full_inf_nan,
            MmeCommon::InfNanMode infNanModeB = e_mme_full_inf_nan)
    : m_typeA(inputA),
      m_typeB(inputB),
      m_expBiasA(expBiasA),
      m_expBiasB(expBiasB),
      m_roundingModeEU(rmEU),
      m_roundingModeACC(rmACC),
      m_clipFp(clipFp),
      m_clipFpInfIn(clipFpInfIn),
      m_infNanModeA(infNanModeA),
      m_infNanModeB(infNanModeB) {};

    virtual ~ChipFma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned size) const = 0;

protected:
    const EMmeDataType m_typeA;
    const EMmeDataType m_typeB;
    const RoundingMode m_roundingModeEU;
    const RoundingMode m_roundingModeACC;
    const uint8_t m_expBiasA;
    const uint8_t m_expBiasB;
    const InfNanMode m_infNanModeA;
    const InfNanMode m_infNanModeB;
    const bool m_clipFp;
    const bool m_clipFpInfIn;

private:
    static const unsigned c_fma_tree_addr_width = 8;
};
}  // namespace MmeCommon
