#pragma once

// eager includes (relative to src/eager/lib/)
#include "eager_recipe_memory_allocator.h"
#include "node_info/tensor_info.h"
#include "program_data_blob_manager.h"
#include "recipe_gen/recipe_defs.h"

// std includes
#include <string_view>
#include <utility>

struct recipe_t;
class RecipeAllocator;

namespace eager_mode
{
class Node2DescContainer;

class EagerRecipeGenerator
{
public:
    bool generate(std::string_view              recipeName,
                  const Node2DescContainer&     descriptors,
                  const ProgramDataBlobManager& programDataBlobManager,
                  WorkspaceSizesType            workspaceSize,
                  WorkspaceSizesType            programDataSize,
                  const EagerTensorsSet&        tensorsSet,
                  std::optional<RecipeIdType>   recipeDebugId,
                  bool                          nopKernelAdded);
    ~EagerRecipeGenerator();

    bool generateEmptyRecipe(std::string_view recipeName, synDeviceType deviceType, const EagerTensorsSet& tensorsSet);

    RecipeAllocator* consumeRecipeAllocator() { return std::exchange(m_recipeAllocator, nullptr); }
    const recipe_t*  getRecipe() const { return m_recipe; }

private:
    EagerRecipeMemoryAllocator* m_recipeAllocator {};
    recipe_t*        m_recipe {};
};

}  // namespace eager_mode