#pragma once

#include <string>

/*
 General mme geometry for gc per platform.
 */
enum MmeGeometry
{
    gaudi_geometry_4wx1h = 0x0,
    gaudi_geometry_2wx2h,
    gaudi_geometry_1wx4h,
};

std::string geometry2String(const MmeGeometry &geometry);

static const MmeGeometry GAUDI_GEOMETRY[3] = {gaudi_geometry_1wx4h, gaudi_geometry_2wx2h, gaudi_geometry_4wx1h};
