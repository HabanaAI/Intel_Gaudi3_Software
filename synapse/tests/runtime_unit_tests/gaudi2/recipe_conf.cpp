#include <vector>
#include "synapse_test.hpp"
#include "test_dummy_recipe.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "habana_global_conf.h"

TEST(UTrecipeConfParser, basic)
{
    TestDummyRecipe dummyRecipe;

    std::vector<gc_conf_t> conf;

    recipe_t* recipe           = dummyRecipe.getRecipe();
    recipe->recipe_conf_params = conf.data();
    recipe->recipe_conf_nr     = conf.size();
    // Nothing set yet, make sure we get false
    auto val = RecipeUtils::getConfVal(recipe, gc_conf_t::DEVICE_TYPE);
    ASSERT_EQ(val.is_set(), false);

    // push few params
    conf.push_back({0, gc_conf_t::TIME_STAMP});
    conf.push_back({0xFF, gc_conf_t::TPC_ENGINE_MASK});
    conf.push_back({synDeviceGaudi2, gc_conf_t::DEVICE_TYPE});
    conf.push_back({55, gc_conf_t::MME_NUM_OF_ENGINES});
    recipe->recipe_conf_params = conf.data();
    recipe->recipe_conf_nr     = conf.size();

    val = RecipeUtils::getConfVal(recipe, gc_conf_t::DEVICE_TYPE);
    ASSERT_EQ(val.is_set(), true);
    ASSERT_EQ(val.value(), synDeviceGaudi2);
}

TEST(UTrecipeConfParser, isRecipeTpcValid)
{
    std::vector<gc_conf_t> conf;
    conf.push_back({GCFG_TPC_ENGINES_ENABLED_MASK.value(), gc_conf_t::TPC_ENGINE_MASK});

    TestDummyRecipe dummyRecipe;
    recipe_t*   recipe         = dummyRecipe.getRecipe();
    recipe->recipe_conf_params = conf.data();
    recipe->recipe_conf_nr     = conf.size();

    ASSERT_TRUE(RecipeUtils::isRecipeTpcValid(recipe));
}
