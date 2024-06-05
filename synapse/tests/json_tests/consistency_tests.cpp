#include "consistency_tests.h"
#include "graph_loader.h"
#include <iostream>
#include "infra/recipe/recipe_compare.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "habana_global_conf.h"

namespace json_tests
{
ConsistencyTest::ConsistencyTest(const ArgParser& args)
: TypedDeviceTest(args),
  JsonTest(args),
  m_testIterations(args.getValueOrDefault(an_test_iter, 1)),
  m_keepGoing(args.getValueOrDefault(an_keep_going, false))
{
    setup();
}

void ConsistencyTest::setup()
{
    if (m_testIterations < 1)
    {
        throw std::runtime_error("NUMBER_OF_TEST_ITERATIONS must be >= 1");
    }

    m_jsonFileLoader->loadEnv(m_ctx);

    GCFG_ENABLE_PROFILER.setValue(false);
    JT_LOG_INFO("Disabling profiler for compilation consistency check");
}

syn::Recipe ConsistencyTest::compileGraph(const size_t graphIndex)
{
    auto gl = m_jsonFileLoader->getGraphLoader(m_ctx, m_deviceType, CompilationMode::Graph, graphIndex);

    const auto& graph = gl.getGraph();
    return graph.compile(gl.getName());
}

bool ConsistencyTest::checkConsistency(const size_t graphIndex)
{
    using namespace RecipeCompare;

    syn::Recipe baseRecipe = compileGraph(graphIndex);

    bool isRecipeConsistent = true;
    for (int iter = 0; iter < m_testIterations; ++iter)
    {
        // Log iterations
        if (m_testIterations > 1)
        {
            JT_LOG_INFO("Iter " << (iter + 1) << " out of " << m_testIterations << "...");
        }

        syn::Recipe currRecipe = compileGraph(graphIndex);
        if (*currRecipe.handle()->basicRecipeHandle.recipe != *baseRecipe.handle()->basicRecipeHandle.recipe)
        {
            isRecipeConsistent = false;
            break;
        }
    }

    if (isRecipeConsistent)
    {
        JT_LOG_INFO("All recipe compilations tested are consistent in graph number " << graphIndex);
    }
    else
    {
        JT_LOG_ERR("Some of the recipe compilations were inconsistent in graph number " << graphIndex);
    }

    return isRecipeConsistent;
}

void ConsistencyTest::run()
{
    bool isModelRecipesConsistent = true;

    for (const auto& graphIndex : m_graphsIndices)
    {
        bool isConsistent = checkConsistency(graphIndex);

        isModelRecipesConsistent &= isConsistent;

        if (!isConsistent && !m_keepGoing)
        {
            break;
        }
    }

    if (!isModelRecipesConsistent)
    {
        throw std::runtime_error("Inconsistency found");
    }
}
}  // namespace json_tests