#pragma once

#include "gaudi/gaudi_defs.h"
#include "gaudi2/gaudi2_defs.h"
#include "gaudi3/gaudi3_defs.h"

namespace MmeCommon
{
struct DefaultMmeLayerParams : MmeLayerParams
{
    DefaultMmeLayerParams() : MmeLayerParams() {}
};

static MmeLayerParams getMmeLayerParams(ChipType chipType)
{
    switch (chipType)
    {
        case e_mme_Gaudi:
            return gaudi::GaudiMmeLayerParams();
        case e_mme_Gaudi2:
            return Gaudi2::Gaudi2MmeLayerParams();
        case e_mme_Gaudi3:
            return gaudi3::Gaudi3MmeLayerParams();
        default:
            break;
    }

    MME_ASSERT(0, "Unsupported chip type (only Gaudis are supported)");
    return DefaultMmeLayerParams();
}

}  // namespace MmeCommon