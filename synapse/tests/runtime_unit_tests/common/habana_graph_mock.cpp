#include "habana_graph_mock.hpp"
#include "recipe_allocator.h"
#include "recipe.h"

HabanaGraphMock::HabanaGraphMock(synDeviceType deviceType, const std::vector<uint64_t>& workspaceSizes)
: m_deviceType(deviceType), mpMemoryAllocator(nullptr), m_workspaceSizes(workspaceSizes)
{
}

recipe_t* HabanaGraphMock::serializeDataPlane(RecipeAllocator* recipeAlloc) const
{
    recipe_t* pRecipe = reinterpret_cast<recipe_t*>(recipeAlloc->allocate(sizeof(recipe_t)));
    memset(pRecipe, 0, sizeof(recipe_t));
    initializeRecipe(*recipeAlloc, *pRecipe);
    return pRecipe;
}

bool HabanaGraphMock::compile()
{
    return true;
}

void HabanaGraphMock::initializeRecipe(RecipeAllocator& rRecipeAlloc, recipe_t& rRecipe) const
{
    rRecipe.recipe_conf_nr = 1;
    rRecipe.recipe_conf_params =
        reinterpret_cast<gc_conf_t*>(rRecipeAlloc.allocate(rRecipe.recipe_conf_nr * sizeof(gc_conf_t)));
    rRecipe.recipe_conf_params[0].conf_id    = gc_conf_t::DEVICE_TYPE;
    rRecipe.recipe_conf_params[0].conf_value = m_deviceType;

    rRecipe.workspace_nr = m_workspaceSizes.size();
    rRecipe.workspace_sizes =
        reinterpret_cast<uint64_t*>(rRecipeAlloc.allocate(rRecipe.workspace_nr * sizeof(uint64_t)));
    for (size_t workspaceIndex = 0; workspaceIndex < m_workspaceSizes.size(); workspaceIndex++)
    {
        rRecipe.workspace_sizes[workspaceIndex] = m_workspaceSizes[workspaceIndex];
    }
}
