#include "recipe_handle_impl.hpp"
#include "defs.h"
#include "recipe_allocator.h"

void InternalRecipeHandle::createRecipeHandle(InternalRecipeHandle*& rpRecipeHandle)
{
    rpRecipeHandle                        = new InternalRecipeHandle();
    basicRecipeInfo& basicRecipeHandle    = rpRecipeHandle->basicRecipeHandle;
    basicRecipeHandle.recipe              = nullptr;
    basicRecipeHandle.shape_plan_recipe   = nullptr;
    basicRecipeHandle.recipeDebugInfo     = nullptr;
    basicRecipeHandle.recipeDebugInfoSize = 0;
    basicRecipeHandle.recipeAllocator     = nullptr;
}

void InternalRecipeHandle::destroyRecipeHandle(InternalRecipeHandle* pRecipeHandle)
{
    if (pRecipeHandle->basicRecipeHandle.recipeAllocator != nullptr)
    {
        pRecipeHandle->basicRecipeHandle.recipeAllocator->freeAll();

        delete pRecipeHandle->basicRecipeHandle.recipeAllocator;
        pRecipeHandle->basicRecipeHandle.recipeAllocator = nullptr;
    }
    else
    {
        if (pRecipeHandle->basicRecipeHandle.recipe != nullptr)
        {
            delete pRecipeHandle->basicRecipeHandle.recipe;
            pRecipeHandle->basicRecipeHandle.recipe = nullptr;
        }
        if (pRecipeHandle->basicRecipeHandle.shape_plan_recipe != nullptr)
        {
            delete pRecipeHandle->basicRecipeHandle.shape_plan_recipe;
            pRecipeHandle->basicRecipeHandle.shape_plan_recipe = nullptr;
        }
        if (pRecipeHandle->basicRecipeHandle.recipeDebugInfo != nullptr)
        {
            delete[] pRecipeHandle->basicRecipeHandle.recipeDebugInfo;
            pRecipeHandle->basicRecipeHandle.recipeDebugInfo = nullptr;
        }
    }

    delete pRecipeHandle;
}
