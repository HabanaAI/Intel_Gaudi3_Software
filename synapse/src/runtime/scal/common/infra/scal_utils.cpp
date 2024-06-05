#include "scal_utils.hpp"

#include "defs.h"
#include "habana_global_conf_runtime.h"
#include "recipe.h"
#include "scal.h"
#include "scal_types.hpp"

#include <limits>

namespace ScalUtils
{
uint8_t convertLogicalEngineIdTypeToScalEngineGroupType(uint8_t logicalEngineId)
{
    uint8_t engineGrpType = std::numeric_limits<uint8_t>::max();

    switch (logicalEngineId)
    {
        case Recipe::EngineType::TPC:
        {
            engineGrpType = SCAL_TPC_COMPUTE_GROUP;
            break;
        }
        case Recipe::EngineType::MME:
        {
            engineGrpType = SCAL_MME_COMPUTE_GROUP;
            break;
        }
        case Recipe::EngineType::DMA:
        {
            engineGrpType = SCAL_EDMA_COMPUTE_GROUP;
            break;
        }
        case Recipe::EngineType::ROT:
        {
            engineGrpType = SCAL_RTR_COMPUTE_GROUP;
            break;
        }
        case Recipe::EngineType::CME:
        {
            engineGrpType = SCAL_CME_GROUP;
            break;
        }
        default:
        {
            HB_ASSERT(false, "Invalid logicalEngineId {}", logicalEngineId);
        }
    }
    return engineGrpType;
}

/*
 ***************************************************************************************************
 *   @brief isCompareAfterDownLoad() - utility to parse global conf value
 *                                     read the recipe from HBM and compare to given recipe before
 *                                     doing the launch
 *   @return bool
 ***************************************************************************************************
 */
bool isCompareAfterDownLoad()
{
    return (GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value() & COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD) ==
           COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD;
}

/*
 ***************************************************************************************************
 *   @brief isCompareAfterDownloadPost() - utility to parse global conf value
 *   @return bool
 ***************************************************************************************************
 */
bool isCompareAfterDownloadPost()
{
    return (GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value() & COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD_POST) ==
           COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD_POST;
}

/*
 ***************************************************************************************************
 *   @brief isCompareAfterLaunch() - utility to parse global conf value
 *   @return bool
 ***************************************************************************************************
 */
bool isCompareAfterLaunch()
{
    return (GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value() & COMPARE_RECIPE_ON_DEVICE_AFTER_LAUNCH) ==
           COMPARE_RECIPE_ON_DEVICE_AFTER_LAUNCH;
}

}  // namespace ScalUtils
