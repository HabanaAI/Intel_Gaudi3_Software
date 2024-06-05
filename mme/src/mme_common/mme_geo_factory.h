#ifndef MME__GEO_FACTORY_H
#define MME__GEO_FACTORY_H

#include "src/gaudi/gaudi_geo_attr.h"
#include "src/gaudi2/gaudi2_geo_attr.h"
#include "src/gaudi3/gaudi3_geo_attr.h"

namespace MmeCommon
{
inline upCommonGeoAttr getGeoAttr(ChipType chipType, const MmeCommon::MmeLayerParams& params)
{
    switch (chipType)
    {
        default:
            return std::make_unique<gaudi::GaudiGeoAttr>(params);
        case e_mme_Gaudi2:
            return std::make_unique<Gaudi2::Gaudi2GeoAttr>(params);
        case e_mme_Gaudi3:
            return std::make_unique<gaudi3::Gaudi3GeoAttr>(params);
    }
}
}  // namespace MmeCommon

#endif //MME__GEO_FACTORY_H
