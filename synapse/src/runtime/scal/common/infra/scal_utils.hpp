#pragma once

#include <cstdint>

namespace ScalUtils
{
uint8_t convertLogicalEngineIdTypeToScalEngineGroupType(uint8_t logicalEngineId);

bool isCompareAfterDownLoad();
bool isCompareAfterDownloadPost();
bool isCompareAfterLaunch();

}  // namespace ScalUtils
