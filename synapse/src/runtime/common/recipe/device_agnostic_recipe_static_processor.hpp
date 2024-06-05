#pragma once

#include "synapse_common_types.h"
#include "recipe_package_types.hpp"

struct recipe_t;
struct DeviceAgnosticRecipeStaticInfo;

class DeviceAgnosticRecipeStaticProcessor
{
public:
    static synStatus
    process(const recipe_t& rRecipe, DeviceAgnosticRecipeStaticInfo& rRecipeInfo, synDeviceType deviceType);

private:
    static bool calculatePatchableBufferSizeAndChunksAmount(const recipe_t&                 rRecipe,
                                                            DeviceAgnosticRecipeStaticInfo& rRecipeInfo,
                                                            uint64_t                        dcSizeCommand);

    // Stores  blob-index, blob-size pair for the internal-queues
    static void setWorkCompletionProgramIndex(const recipe_t&                 rRecipe,
                                              DeviceAgnosticRecipeStaticInfo& rRecipeInfo,
                                              synDeviceType                   deviceType);

    static bool validateWorkCompletionQueue(const recipe_t& rRecipe, synDeviceType deviceType);

    static uint32_t getAmountOfEnginesInArbGroup(synDeviceType deviceType);
};