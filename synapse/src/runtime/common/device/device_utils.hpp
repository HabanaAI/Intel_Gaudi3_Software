#pragma once

#include "synapse_api_types.h"
#include "synapse_common_types.h"

class DeviceInterface;

synStatus extractDeviceAttributes(const synDeviceAttribute* deviceAttr,
                                  const unsigned            querySize,
                                  uint64_t*                 retVal,
                                  const synDeviceInfo       deviceInfo,
                                  uint32_t*                 pCurrentClockRate,
                                  DeviceInterface*          pDevice);