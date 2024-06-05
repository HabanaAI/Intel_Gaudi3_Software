#include "recipe_runner.h"
#include <iostream>

namespace json_tests
{
RecipeRunner::RecipeRunner(const ArgParser& args)
: RunTypedDeviceTest(args), m_recipeFilePath(args.getValue<std::string>(an_recipe_file))
{
}

void RecipeRunner::run()
{
    JT_LOG_INFO("Load recipe: " << m_recipeFilePath);
    auto recipe = syn::Recipe(m_recipeFilePath);
    runGraph(0, recipe, nullptr, nullptr);
    dumpStats();
}
}  // namespace json_tests