#pragma once

#include "include/mme_common/mme_common_enum.h"

namespace gaudi
{
struct GaudiMmeLayerParams : MmeCommon::MmeLayerParams
{
    GaudiMmeLayerParams() : MmeCommon::MmeLayerParams()
    {
        MmeCommon::MmeLayerParams::x.elementType = MmeCommon::e_type_bf16;
        MmeCommon::MmeLayerParams::y.elementType = MmeCommon::e_type_bf16;
        MmeCommon::MmeLayerParams::w.elementType = MmeCommon::e_type_bf16;
        MmeCommon::MmeLayerParams::strategy.unrollEn = true;
    }
};

}  // namespace gaudi