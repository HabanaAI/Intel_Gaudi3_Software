#pragma once

// Common hpp file for the definition (and not just the declaration) of the Recipe-Handle struct

#include <cstdint>
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"

struct InternalRecipeHandle
{
    static void createRecipeHandle(InternalRecipeHandle*& rpRecipeHandle);

    static void destroyRecipeHandle(InternalRecipeHandle* pRecipeHandle);

    basicRecipeInfo basicRecipeHandle;

    DeviceAgnosticRecipeInfo deviceAgnosticRecipeHandle;

    uint64_t recipeSeqNum;
};
