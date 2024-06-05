#pragma once

#include "synapse_common_types.h"
#include "synapse_api_types.h"

struct recipe_t;
struct DeviceAgnosticRecipeInfo;
struct basicRecipeInfo;
struct SignalFromGraphInfo;
struct RecipeTensorsInfo;

class DeviceAgnosticRecipeProcessor
{
public:
    static synStatus process(const basicRecipeInfo&    rBasicRecipeInfo,
                             DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo);

    static synStatus recipeGetAttribute(uint64_t*                 retVal,
                                        const synRecipeAttribute* recipeAttr,
                                        const unsigned            querySize,
                                        const synRecipeHandle     recipeHandle);

private:
    static void parseRecipe(const basicRecipeInfo& rBasicRecipeInfo, synDeviceType& rDeviceType);

    static synStatus verifyRecipe(const basicRecipeInfo& rBasicRecipeInfo);

    static void getTopologyWorkspaceSize(const basicRecipeInfo& rBasicRecipeInfo,
                                         synDeviceType          deviceType,
                                         uint64_t&              rWorkspaceSize);

    static bool _extractSfgInfo(const RecipeTensorsInfo& recipeTensorInfo,
                                const recipe_t*          recipe,
                                SignalFromGraphInfo&     rSignalFromGraphInfo);

    static bool isScalArchitecture(synDeviceType deviceType);
};
