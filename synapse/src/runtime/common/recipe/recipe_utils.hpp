#pragma once

#include "recipe.h"
#include "settable.h"
#include "basic_recipe_info.hpp"

namespace RecipeUtils
{
    Settable<uint64_t> getConfVal(const recipe_t* recipe, gc_conf_t::recipeCompileParams param);
    bool               isRecipeTpcValid(const recipe_t* recipe);

    bool        isKernelPrintf(const InternalRecipeHandle& rInternalRecipeHandle);
    bool        isKernelPrintf(const basicRecipeInfo& recipeInfo);
    inline bool isDsd(const basicRecipeInfo& recipeInfo)
    {
        return (recipeInfo.shape_plan_recipe != nullptr);
    }
    bool isSfg(const InternalRecipeHandle& pInternalRecipeHandle);

    uint64_t getTotalKernelPrintfSize(const basicRecipeInfo& recipeInfo);

    uint64_t getKernelPrintOffsetInWs(const InternalRecipeHandle* recipeHandle);

    bool isIH2DRecipe(const recipe_t* pRecipe);
}  // namespace RecipeUtils
