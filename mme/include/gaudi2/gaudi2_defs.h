#ifndef MME__GAUDI2_DEFS_H
#define MME__GAUDI2_DEFS_H

#include "include/mme_common/mme_common_enum.h"

namespace Gaudi2
{
struct Gaudi2MmeLayerParams : MmeCommon::MmeLayerParams
{
    Gaudi2MmeLayerParams() : MmeCommon::MmeLayerParams()
    {
        this->controls.roundingMode = MmeCommon::RoundToNearest;
        this->controls.accRoundingMode = MmeCommon::RoundToNearest;
        this->strategy.mmeLimit = 2;

        this->strategy.geometry = MmeCommon::e_mme_geometry_nr;
        this->strategy.pattern = MmeCommon::e_mme_patterns_nr;
    }
};
}  // namespace Gaudi2

#endif //MME__GAUDI2_DEFS_H
