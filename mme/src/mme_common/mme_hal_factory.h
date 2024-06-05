#ifndef MME__HAL_FACTORY_H
#define MME__HAL_FACTORY_H

#include "include/mme_assert.h"
#include "src/gaudi/gaudi_mme_hal_reader.h"
#include "src/gaudi2/gaudi2_mme_hal_reader.h"
#include "src/gaudi3/gaudi3_mme_hal_reader.h"

namespace MmeCommon
{
static inline const MmeHalReader& getMmeHal(ChipType chipType)
{
    switch (chipType)
    {
        case e_mme_Gaudi:
            return gaudi::MmeHalReader::getInstance();
        case e_mme_Gaudi2:
            return Gaudi2::MmeHalReader::getInstance();
        case e_mme_Gaudi3:
            return gaudi3::MmeHalReader::getInstance();
        default:
            MME_ASSERT(0, "chip not supported by mmeHal yey");
            break;
    }

    return Gaudi2::MmeHalReader::getInstance();
}

}  // namespace MmeCommon

#endif //MME__HAL_FACTORY_H
