#pragma once

#include "log_manager.h"
#include "recipe_handle_impl.hpp"
#include "synapse_common_types.h"

struct DeviceAgnosticRecipeInfo;
struct basicRecipeInfo;

class RecipeLogger
{
public:
    static void dumpRecipe(InternalRecipeHandle*           pRecipeHandle,
                           bool                            log,
                           bool                            screen,
                           synapse::LogManager::LogType    logger);

    static void dumpRecipe(const DeviceAgnosticRecipeInfo& deviceAgnosticInfo,
                           const basicRecipeInfo&          basicRecipeInfo,
                           uint64_t                        id,
                           bool                            log,
                           bool                            screen,
                           synapse::LogManager::LogType    logger);

    static void dfaDumpRecipe(const InternalRecipeHandle* internalRecipeHandle,
                              bool                        isScalDev,
                              const std::string&          callerMsg);

    static void dfaDumpRecipe(const DeviceAgnosticRecipeInfo& deviceAgnosticInfo,
                              const basicRecipeInfo&          basicRecipeInfo,
                              bool                            isScalDev,
                              uint64_t                        id,
                              const std::string&              callerMsg);

private:
    static void dumpSyncScheme(const basicRecipeInfo& recipeInfo,
                               bool                   isScalDev,
                               uint64_t               id,
                               const std::string&     callerMsg);
};
