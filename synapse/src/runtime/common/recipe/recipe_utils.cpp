#include "recipe_utils.hpp"

#include "recipe_handle_impl.hpp"
#include "define_synapse_common.hpp"
#include "habana_global_conf.h"

#include "defs.h"

namespace RecipeUtils
{
Settable<uint64_t> getConfVal(const recipe_t* recipe, gc_conf_t::recipeCompileParams confId)
{
    if (recipe == nullptr)
    {
        return Settable<uint64_t>();
    }
    for (int i = 0; i < recipe->recipe_conf_nr; i++)
    {
        if (recipe->recipe_conf_params[i].conf_id == confId)
        {
            return recipe->recipe_conf_params[i].conf_value;
        }
    }
    return Settable<uint64_t>();  // not found
}

#define STR(v) #v
#define LOG_NUM_MISMATCH(recipeConfValue, gcfgValue, fmt)                                                              \
    if (recipeConfValue.is_set())                                                                                      \
    {                                                                                                                  \
        LOG_ERR(SYN_API,                                                                                               \
                "recipe conf value: " STR(recipeConfValue) " {" fmt                                                    \
                                                           "} does not match GCFG value " STR(gcfgValue) " {" fmt "}", \
                recipeConfValue.value(),                                                                               \
                gcfgValue);                                                                                            \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LOG_ERR(SYN_API, "recipe " STR(recipeConfValue) " not set");                                                   \
    }

bool isRecipeTpcValid(const recipe_t* recipe)
{
    auto recipeTpcMask = getConfVal(recipe, gc_conf_t::TPC_ENGINE_MASK);
    if (!recipeTpcMask.is_set() ||
        (recipeTpcMask.value() & GCFG_TPC_ENGINES_ENABLED_MASK.value()) != recipeTpcMask.value())
    {
        LOG_NUM_MISMATCH(recipeTpcMask, GCFG_TPC_ENGINES_ENABLED_MASK.value(), ":x");
        return false;
    }

    return true;
}

bool isKernelPrintf(const InternalRecipeHandle& rInternalRecipeHandle)
{
    return isKernelPrintf(rInternalRecipeHandle.basicRecipeHandle);
}

bool isKernelPrintf(const basicRecipeInfo& recipeInfo)
{
    return (recipeInfo.recipe->debug_profiler_info.printf_addr_nr > 0);
}

bool isSfg(const InternalRecipeHandle& pInternalRecipeHandle)
{
    return (pInternalRecipeHandle.deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getNumberOfExternalTensors() > 0);
}

uint64_t getTotalKernelPrintfSize(const basicRecipeInfo& recipeInfo)
{
    return (recipeInfo.recipe->debug_profiler_info.printf_addr_nr * GCFG_TPC_PRINTF_TENSOR_SIZE.value());
}

uint64_t getKernelPrintOffsetInWs(const InternalRecipeHandle* recipeHandle)
{
    return recipeHandle->basicRecipeHandle.recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE];
}

bool isIH2DRecipe(const recipe_t* pRecipe)
{
    HB_ASSERT_PTR(pRecipe);

    return (pRecipe->h2di_tensors_nr != 0);
}

}  // namespace RecipeUtils
