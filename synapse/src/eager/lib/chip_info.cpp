#include "chip_info.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "chip_specification/gaudi2/desc_gen/desc_factory.h"
#include "chip_specification/gaudi2/desc_gen/desc_gen_hal.h"
#include "chip_specification/gaudi2/desc_gen/sync_scheme_manager.h"
#include "chip_specification/gaudi2/general_defs.h"
#include "chip_specification/gaudi2/mme_brain.h"
#include "chip_specification/gaudi2/recipe/recipe_hal.h"

// synapse-internal gaudi3-specific includes (relative to src/)
#include "chip_specification/gaudi3/desc_gen/desc_factory.h"
#include "chip_specification/gaudi3/desc_gen/desc_gen_hal.h"
#include "chip_specification/gaudi3/desc_gen/sync_scheme_manager.h"
#include "chip_specification/gaudi3/general_defs.h"
#include "chip_specification/gaudi3/mme_brain.h"
#include "chip_specification/gaudi3/recipe/recipe_hal.h"

namespace eager_mode
{
gaudi2_spec_info::DescGeneratorHal gaudi2DescGenHal;
gaudi3_spec_info::DescGeneratorHal gaudi3DescGenHal;

// Note: Wrapped to avoid uncatchable exception
const ChipInfo::RecipeHalArray& ChipInfo::getRecipeHalArr()
{
    static const ChipInfo::RecipeHalArray RECIPE_HAL_ARR = {std::make_unique<gaudi2_spec_info::RecipeHal>(),
                                                            std::make_unique<gaudi3_spec_info::RecipeHal>()};
    return RECIPE_HAL_ARR;
}

bool ChipInfo::isValidForEager(ChipType chipType)
{
    switch (chipType)
    {
        case ChipType::GAUDI2:
        case ChipType::GAUDI3:
            return true;
        default:
            return false;
    }
}

DescGeneratorBasePtr
ChipInfo::createDescGenerator(EagerGraph& graph, const EagerNode& node, ChipType chipType, EngineType engineType)
{
    switch (chipType)
    {
        case ChipType::GAUDI2:
            return gaudi2_spec_info::DescFactory::createDescGenerator(graph, node, engineType);
        case ChipType::GAUDI3:
            return gaudi3_spec_info::DescFactory::createDescGenerator(graph, node, engineType);
        default:
            break;
    }
    EAGER_ASSERT(false, "unsupported device");
    return nullptr;
}

const SyncSchemeManagerBase& ChipInfo::getSyncSchemeManager(ChipType chipType)
{
    switch (chipType)
    {
        default:
            EAGER_ASSERT(false, "getSyncSchemeManager called for unsupported device");
            // fallthrough
        case ChipType::GAUDI2:
        {
            static constexpr gaudi2_spec_info::SyncSchemeManager GAUDI2_SYNC_SCHEME_MANAGER(gaudi2DescGenHal);
            return GAUDI2_SYNC_SCHEME_MANAGER;
        }
        case ChipType::GAUDI3:
        {
            static constexpr gaudi3_spec_info::SyncSchemeManager GAUDI3_SYNC_SCHEME_MANAGER(gaudi3DescGenHal);
            return GAUDI3_SYNC_SCHEME_MANAGER;
        }
    }
}

const EagerMmeBrainBase& ChipInfo::getEagerMmeBrain(ChipType chipType)
{
    switch (chipType)
    {
        default:
            EAGER_ASSERT(false, "getEagerMmeBrain called for unsupported device");
            // fallthrough (it doesn't matter what we return here)
        case ChipType::GAUDI2:
        {
            static constexpr gaudi2_spec_info::EagerMmeBrain GAUDI2_BRAIN;
            return GAUDI2_BRAIN;
        }
        case ChipType::GAUDI3:
        {
            static constexpr gaudi3_spec_info::EagerMmeBrain GAUDI3_BRAIN;
            return GAUDI3_BRAIN;
        }
    }
}

uint64_t ChipInfo::getTensorSizeThresholdForParallelExec(ChipType chipType)
{
    const std::string& userVal = GCFG_ENABLE_EAGER_PARALLEL_EXECUTION.value();
    if (userVal == "auto")
    {
        switch (chipType)
        {
            case ChipType::GAUDI2:
                return gaudi2_spec_info::TENSOR_SIZE_THRESHOLD_FOR_EAGER_PARALLEL_EXECUTION;
            case ChipType::GAUDI3:
                return gaudi3_spec_info::TENSOR_SIZE_THRESHOLD_FOR_EAGER_PARALLEL_EXECUTION;
            default:
                EAGER_ASSERT_0;
        }
    }
    else if (userVal == "enable")
    {
        return 0;
    }
    if (unlikely(userVal != "disable"))
    {
        EAGER_REPORT_ERROR("Invalid GCFG value ({}) for parallel execution", userVal);
    }
    return -1;
}

}  // namespace eager_mode
