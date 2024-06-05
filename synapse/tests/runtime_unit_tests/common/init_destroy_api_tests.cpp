#include "scoped_configuration_change.h"
#include <gtest/gtest.h>
#include "synapse_api.h"
#include "habana_global_conf_runtime.h"
#include "recipe.h"
#include "recipe_allocator.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/recipe/recipe_manager.hpp"
#include "habana_graph_mock.hpp"

TEST(InitializeDestroyTest, ref_counting_check)
{
    // Disable recipe validation as this test does not suppose to check recipe correctness only api call
    GCFG_RUNTIME_SKIP_RECIPE_VALIDATION.setValue(true);
    // create a recipe
    recipe_t recipe     = {0};
    recipe.workspace_nr = 1;
    uint64_t workspaceSizes[1] {128};
    recipe.workspace_sizes     = workspaceSizes;
    const char* recipeFileName = "/tmp/tmp_recipe.txt";

    InternalRecipeHandle handle {};
    handle.basicRecipeHandle.recipe = &recipe;

    synRecipeHandle recipeHandle = &handle;

    // synapse api call must fail before the first synInitialize and after the last synDestroy

    synStatus status = synRecipeSerialize(recipeHandle, recipeFileName);
    ASSERT_EQ(status, synUninitialized) << "succeeded but must fail before syInitialize";

    // Initialization
    status = synInitialize();
    ASSERT_EQ(status, synSuccess) << "Failed to initialize Synapse!";

    status = synRecipeSerialize(recipeHandle, recipeFileName);
    ASSERT_EQ(status, synSuccess) << "Failed in syn api";

    status = synRecipeSerialize(recipeHandle, recipeFileName);
    ASSERT_EQ(status, synSuccess) << "Failed in syn api";

    status = synDestroy();
    ASSERT_EQ(status, synSuccess) << "Failed to synDestroy";

    status = synRecipeSerialize(recipeHandle, recipeFileName);
    ASSERT_EQ(status, synUninitialized) << "succeeded but must fail after synDestroy";

    // Initialization
    status = synInitialize();
    ASSERT_EQ(status, synSuccess) << "Failed to initialize Synapse!";

    status = synRecipeSerialize(recipeHandle, recipeFileName);
    ASSERT_EQ(status, synSuccess) << "Failed in syn api";

    status = synDestroy();
    ASSERT_EQ(status, synSuccess) << "Failed to synDestroy";

    status = synDestroy();
    ASSERT_EQ(status, synUninitialized) << "succeeded but must fail after synDestroy";
}

TEST(InitializeDestroyTest, double_init_test)
{
    const char*     disableDoubleSynInitEnvVarName = "HABANA_DISABLE_DOUBLE_SYN_INITIALIZE";
    ScopedEnvChange disableDoubleSynInit(disableDoubleSynInitEnvVarName, "0");
    ASSERT_EQ(synInitialize(), synSuccess);
    ASSERT_EQ(synInitialize(), synSuccess) << "double initialization is OK";
    ASSERT_EQ(synDestroy(), synSuccess);

    unsetenv(disableDoubleSynInitEnvVarName);
    ASSERT_EQ(synInitialize(), synSuccess);
    ASSERT_EQ(synInitialize(), synSuccess) << "double initialization is OK";
    ASSERT_EQ(synDestroy(), synSuccess);

    int v = setenv(disableDoubleSynInitEnvVarName, "1", 1);
    ASSERT_EQ(v, 0);
    ASSERT_EQ(synInitialize(), synSuccess);
    ASSERT_EQ(synInitialize(), synAlreadyInitialized) << "double initialization is disabled";
    ASSERT_EQ(synDestroy(), synSuccess);
}

TEST(InitializeDestroyTest, double_destroy_test)
{
    const char*     disableSynDestroyEnvVarName = "HABANA_DISABLE_DOUBLE_SYN_DESTROY";
    ScopedEnvChange disableSynDestroy(disableSynDestroyEnvVarName, "0");
    ASSERT_EQ(synInitialize(), synSuccess);
    ASSERT_EQ(synDestroy(), synSuccess);
    ASSERT_EQ(synDestroy(), synSuccess) << "double destroy is enabled";

    unsetenv(disableSynDestroyEnvVarName);
    ASSERT_EQ(synInitialize(), synSuccess);
    ASSERT_EQ(synDestroy(), synSuccess);
    ASSERT_EQ(synDestroy(), synUninitialized) << "double destroy is disabled";

    int v = setenv(disableSynDestroyEnvVarName, "1", 1);
    ASSERT_EQ(v, 0);
    ASSERT_EQ(synInitialize(), synSuccess);
    ASSERT_EQ(synDestroy(), synSuccess);
    ASSERT_EQ(synDestroy(), synUninitialized) << "double destroy is disabled";
}

TEST(InitializeDestroyTest, wrong_oder_test)
{
    // Disable recipe validation as this test does not suppose to check recipe correctness only api call
    GCFG_RUNTIME_SKIP_RECIPE_VALIDATION.setValue(true);
    // create a recipe
    recipe_t recipe     = {0};
    recipe.workspace_nr = 1;
    uint64_t workspaceSizes[1] {128};
    recipe.workspace_sizes     = workspaceSizes;
    const char* recipeFileName = "/tmp/tmp_recipe.txt";

    InternalRecipeHandle handle {};
    handle.basicRecipeHandle.recipe = &recipe;

    synRecipeHandle recipeHandle = &handle;

    // synapse api call must fail before the first synInitialize and after the last synDestroy
    ASSERT_EQ(synRecipeSerialize(recipeHandle, recipeFileName), synUninitialized)
        << "succeeded but must fail before syInitialize";

    ASSERT_EQ(synDestroy(), synUninitialized) << "succeeded but must fail after synDestroy";
}