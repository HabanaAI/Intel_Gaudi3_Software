#ifndef MME__GAUDI3_DEFS_H
#define MME__GAUDI3_DEFS_H

#include "include/mme_common/mme_common_enum.h"

namespace gaudi3
{
struct Gaudi3MmeLayerParams : MmeCommon::MmeLayerParams
{
    Gaudi3MmeLayerParams() : MmeCommon::MmeLayerParams()
    {
        this->controls.roundingMode = MmeCommon::RoundToZero;
        this->controls.accRoundingMode = MmeCommon::RoundToZero;
        this->strategy.mmeLimit = 8;

        this->strategy.geometry = MmeCommon::e_mme_geometry_nr;
        this->strategy.pattern = MmeCommon::e_mme_patterns_nr;
    }
};
}  // namespace gaudi3

#endif //MME__GAUDI3_DEFS_H
