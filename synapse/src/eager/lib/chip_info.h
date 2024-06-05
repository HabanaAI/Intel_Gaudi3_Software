#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "desc_gen/sync_scheme_manager_base.h"
#include "utils/general_defs.h"

namespace eager_mode
{
class EagerGraph;
class EagerNode;
class RecipeHalBase;
class EagerMmeBrainBase;

class ChipInfo
{
public:
    using RecipeHalArray = std::array<std::unique_ptr<RecipeHalBase>, static_cast<unsigned>(ChipType::CHIPS_NR)>;

    static bool isValidForEager(ChipType chipType);
    static DescGeneratorBasePtr
    createDescGenerator(EagerGraph& graph, const EagerNode& node, ChipType chipType, EngineType engineType);
    static const SyncSchemeManagerBase& getSyncSchemeManager(ChipType chipType);
    static const EagerMmeBrainBase&     getEagerMmeBrain(ChipType chipType);
    static const RecipeHalBase&         getRecipeHal(ChipType chipType)
    {
        const unsigned idx = static_cast<unsigned>(chipType);
        EAGER_ASSERT(idx < getRecipeHalArr().size(), "getRecipeHal unsupported chip type");
        return *getRecipeHalArr()[idx];
    }

    // It's a parallel Execution Knob:
    // Control the execution mode (parallel or serial) of nodes in the Eager graph
    // Execution Modes: -1: Pure serial execution, 0: Pure parallel execution
    // Threshold Value: Set another value to check if any tensor exceeds it
    // When two or more engines are associated with nodes containing tensors exceeding the threshold,
    // parallel execution will be enabled; otherwise, it will be disabled.
    static uint64_t getTensorSizeThresholdForParallelExec(ChipType chipType);

private:
    static const RecipeHalArray& getRecipeHalArr();
};
}  // namespace eager_mode
