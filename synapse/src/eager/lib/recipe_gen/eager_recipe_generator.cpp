#include "eager_recipe_generator.h"

// eager includes (relative to src/eager/lib/)
#include "chip_info.h"
#include "desc_gen/node2desc.h"
#include "eager_recipe_memory_allocator.h"
#include "recipe_gen/eager_recipe_allocator.h"
#include "recipe_gen/recipe_instantiation.h"
#include "utils/general_defs.h"
#include "utils/memory_utils.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

// std includes
#include <cstdint>

namespace eager_mode
{
EagerRecipeGenerator::~EagerRecipeGenerator()
{
    delete m_recipeAllocator;
}

bool EagerRecipeGenerator::generate(std::string_view              recipeName,
                                    const Node2DescContainer&     descriptors,
                                    const ProgramDataBlobManager& programDataBlobManager,
                                    WorkspaceSizesType            workspaceSize,
                                    WorkspaceSizesType            programDataSize,
                                    const EagerTensorsSet&        tensorsSet,
                                    std::optional<RecipeIdType>   recipeDebugId,
                                    bool                          nopKernelAdded)
{
    EAGER_ASSERT(m_recipeAllocator == nullptr, "Trying to allocate recipe allocator twice");
    const RecipeHalBase& recipeHal = ChipInfo::getRecipeHal(descriptors.getFirstDescGen().getChipType());

    bool canUseCloneFastPath = false;
    if (descriptors.isSingleActivation())
    {
        const auto& desc = descriptors.getFirstDescGen();
        if (desc.getPatchableTensorsNr() == tensorsSet.getPersistentNr())
        {
            canUseCloneFastPath =
                (desc.getEngineType() == EngineType::TPC) || (desc.getEngineType() == EngineType::MME);
        }
    }

    const bool isProgramDataBlobsCopyRequired = programDataBlobManager.isProgramDataBlobCopyRequired();

    m_recipeAllocator = new EagerRecipeMemoryAllocator;
    EagerRecipeAllocator allocator(programDataBlobManager,
                                   programDataSize,
                                   *m_recipeAllocator,
                                   recipeHal,
                                   descriptors,
                                   tensorsSet.getPersistentNr(),
                                   canUseCloneFastPath,
                                   isProgramDataBlobsCopyRequired,
                                   recipeDebugId.has_value());
    m_recipe = allocator.allocateAndInit((recipeName.size() + 1) + tensorsSet.getNamesSizeOfPersistentTensors());
    if (m_recipe == nullptr)
    {
        return false;
    }

    RecipeInstantiation ins(*m_recipe,
                            recipeHal,
                            descriptors,
                            allocator.getStringBufAllocator(),
                            canUseCloneFastPath);
    ins.instantiateGlobalInfo(recipeName,
                              workspaceSize,
                              programDataSize,
                              programDataBlobManager.getProgramDataBlobs(),
                              tensorsSet,
                              isProgramDataBlobsCopyRequired,
                              recipeDebugId,
                              nopKernelAdded);
    ins.instantiateNodeSpecificInfo();
    return true;
}

bool EagerRecipeGenerator::generateEmptyRecipe(std::string_view       recipeName,
                                               synDeviceType          deviceType,
                                               const EagerTensorsSet& tensorsSet)
{
    EAGER_ASSERT(m_recipeAllocator == nullptr, "Trying to allocate recipe allocator twice");
    m_recipeAllocator = new EagerRecipeMemoryAllocator;

    const size_t namesStrLen = (recipeName.size() + 1) + tensorsSet.getNamesSizeOfPersistentTensors();

    size_t totalAlloc = 0;
    planAlloc<recipe_t>(totalAlloc, 1);
    planAlloc<decltype(*recipe_t::workspace_sizes)>(totalAlloc, workspaceSizesNr);
    planAlloc<decltype(*recipe_t::recipe_conf_params)>(totalAlloc, confParamsNr);
    planAlloc<decltype(*recipe_t::tensors)>(totalAlloc, tensorsSet.getPersistentNr());
    planAlloc<char>(totalAlloc, namesStrLen);

    // Note that allocator's using "new" which has a sufficiently large alignment
    std::byte* buf =
        reinterpret_cast<std::byte*>(m_recipeAllocator->allocate(totalAlloc, /*shouldBeMappedToDevice*/ false));
    EAGER_ASSERT(size_t(buf) % alignof(recipe_t) == 0, "expected default alignment of new to be sufficiently large");

    m_recipe  = doPlacement<recipe_t>(buf, 1);
    *m_recipe = {};

    // This is how we express that the recipe_t is empty as an early out for parsing
    EAGER_ASSERT(m_recipe->node_nr == 0, "");

    m_recipe->workspace_nr    = workspaceSizesNr;
    m_recipe->workspace_sizes = doPlacement<decltype(*recipe_t::workspace_sizes)>(buf, m_recipe->workspace_nr);

    static_assert(workspaceSizesNr == 3);
    static_assert(MEMORY_ID_RESERVED_FOR_WORKSPACE == 0);
    static_assert(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA == 1);
    static_assert(MEMORY_ID_RESERVED_FOR_PROGRAM == 2);
    m_recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE]    = 0;
    m_recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] = 0;
    m_recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM]      = 0;

    m_recipe->recipe_conf_nr     = confParamsNr;
    m_recipe->recipe_conf_params = doPlacement<decltype(*recipe_t::recipe_conf_params)>(buf, m_recipe->recipe_conf_nr);

    static_assert(confParamsNr == 2);
    m_recipe->recipe_conf_params[0] = {.conf_value = deviceType, .conf_id = gc_conf_t::DEVICE_TYPE};
    m_recipe->recipe_conf_params[1] = {.conf_value = (uint64_t)-1, .conf_id = gc_conf_t::TPC_ENGINE_MASK};

    m_recipe->persist_tensors_nr = tensorsSet.getPersistentNr();
    m_recipe->tensors            = doPlacement<decltype(*recipe_t::tensors)>(buf, tensorsSet.getPersistentNr());

    auto*              tensorNamesBuf = reinterpret_cast<std::byte*>(doPlacement<char>(buf, namesStrLen));
    DataBuf            dataBuf(tensorNamesBuf, namesStrLen);
    StringBufAllocator sba {dataBuf};

    m_recipe->nameSize = recipeName.size() + 1;
    m_recipe->name     = sba.cloneAllocStr(recipeName);
    RecipeInstantiation::createPersistentTensorsInfo(*m_recipe, tensorsSet, sba);

    return true;
}

}  // namespace eager_mode
