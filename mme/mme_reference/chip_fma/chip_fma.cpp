#include "chip_fma.h"
#include "gaudi_fma.h"
#include "gaudi2_fma.h"
#include "gaudi3_fma.h"

namespace MmeCommon
{
std::unique_ptr<ChipFma> ChipFma::getChipFma(ChipType chipType,
                                             EMmeDataType inputA,
                                             EMmeDataType inputB,
                                             uint8_t expBiasA,
                                             uint8_t expBiasB,
                                             MmeCommon::RoundingMode rmEU,
                                             MmeCommon::RoundingMode rmACC,
                                             bool clipFp,
                                             bool clipFpInfIn,
                                             InfNanMode infNanModeA,
                                             InfNanMode infNanModeB)
{
    switch (chipType)
    {
        default:
            MME_ASSERT(0, "chip not supported by mme reference yet");
        case e_mme_Gaudi:
            return gaudi::GaudiFma::getGaudiFma(inputA, inputB, expBiasA, expBiasB, rmEU);
        case e_mme_Gaudi2:
            return Gaudi2::Gaudi2Fma::getGaudi2Fma(inputA, inputB, expBiasA, expBiasB, clipFp);
        case e_mme_Gaudi3:
            return gaudi3::Gaudi3Fma::getGaudi3Fma(inputA,
                                                   inputB,
                                                   expBiasA,
                                                   expBiasB,
                                                   rmACC,
                                                   clipFp,
                                                   clipFpInfIn,
                                                   infNanModeA,
                                                   infNanModeB);
    }
}
}  // namespace MmeCommon
