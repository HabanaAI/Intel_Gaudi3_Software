#include <gtest/gtest.h>
#include "hpp/synapse.hpp"
#include "recipe.h"
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_processor.hpp"
#include "define_synapse_common.hpp"
#include "runtime/common/habana_global_conf_runtime.h"

class UTDeviceAgnosticRecipeProcessor : public ::testing::Test
{
};

TEST_F(UTDeviceAgnosticRecipeProcessor, checkProcessGetTopologyWorkspaceSize)
{
    // Initialization
    syn::Context context;

    recipe_t recipe {0};
    // Conf
    gc_conf_t confArr[] {{synDeviceGaudi2, gc_conf_t::DEVICE_TYPE},
                         {GCFG_TPC_ENGINES_ENABLED_MASK.value(), gc_conf_t::TPC_ENGINE_MASK}};
    recipe.recipe_conf_nr     = sizeof(confArr) / sizeof(gc_conf_t);
    recipe.recipe_conf_params = confArr;
    // Workspace sizes
    uint64_t workspaceSizeArr[] {0x100, 0x200, 0x400};
    recipe.workspace_nr    = sizeof(workspaceSizeArr) / sizeof(uint64_t);
    recipe.workspace_sizes = workspaceSizeArr;
    uint64_t workspaceSizeExpected = workspaceSizeArr[MEMORY_ID_RESERVED_FOR_WORKSPACE];

    DeviceAgnosticRecipeInfo deviceAgnosticRecipeInfo;
    synStatus                status =
        DeviceAgnosticRecipeProcessor::process({&recipe, nullptr, nullptr, 0, nullptr}, deviceAgnosticRecipeInfo);
    ASSERT_EQ(status, synSuccess) << "Failed to process";
    ASSERT_EQ(deviceAgnosticRecipeInfo.m_deviceType, synDeviceGaudi2) << "device type failed";
    ASSERT_EQ(deviceAgnosticRecipeInfo.m_workspaceSize, workspaceSizeExpected) << "workspace size failed";

}
