#include "performance_playback_tests.h"

namespace json_tests
{
PerformancePlaybackTest::PerformancePlaybackTest(const ArgParser& args)
: PerformanceBaseTest(args, args.getValueOrDefault(an_run, false))
{
}

void PerformancePlaybackTest::run()
{
    runTest(m_run);
    dumpStats();
}

void PerformancePlaybackTest::runTest(bool run)
{
    const std::string model = m_jsonFileLoader->getModelName();
    const std::string msg   = run ? "Compile and run" : "Compile";
    JT_LOG_INFO(msg << " model: " << model);

    // map graph index to compilation error
    std::map<int, std::string> failedGraphs;

    const float step         = m_testIterations / 20.f;
    float       nextBoundary = 0;
    for (int iter = 0; iter < m_testIterations; ++iter)
    {
        if (m_testIterations > 1)
        {
            if (!m_quietMode)
            {
                JT_LOG_INFO("Iter " << (iter + 1) << " out of " << m_testIterations << "...");
            }
            else if (iter == 0 || iter == m_testIterations - 1 || iter + 1 > nextBoundary)
            {
                nextBoundary += step;
                JT_LOG_INFO("Iter " << (iter + 1) << " out of " << m_testIterations << "...");
            }
        }

        for (const auto& i : m_graphsIndices)
        {
            if (failedGraphs.count(i) == 0)
            {
                const auto& jsonGraph    = m_jsonFileLoader->getGraph(i);
                executeGraph(i, model, run, failedGraphs, jsonGraph);
            }
        }
    }

    if (failedGraphs.empty())
    {
        JT_LOG_INFO(msg << " model: " << model << " finished successfully");
    }
    else
    {
        JT_LOG_ERR(msg << " model: " << model << " finished with the following compilation failures: ");
        for (const auto& kv : failedGraphs)
        {
            JT_LOG_ERR(msg << " - graph #" << kv.first << ": " << kv.second);
        }
    }
}

void PerformancePlaybackTest::executeGraph(size_t                      graphIndex,
                                           const std::string&          model,
                                           bool                        run,
                                           std::map<int, std::string>& failedGraphs,
                                           const nlohmann_hcl::json& jsonGraph)
{
    auto        graphName = json_utils::get(jsonGraph, "name");
    GraphParams graphParams;
    RecipeParams       recipeParams;
    JT_LOG_INFO_NON_QUIET("Compile graph: " << graphIndex << " (" << graphName << ")");

    // parse the json file and prepare the arguments for the later Synapse API calls.
    // We wish all the allocations and execution environment added overhead to take
    // place in this phase so that the later phase of Synapse API calls will not contain
    // unrelated noise for the time measurments and callgrind profiling.
    graphParams.idx  = graphIndex;
    graphParams.name = graphName;
    fillTensorAndSectionCreationParams(jsonGraph, graphParams);
    fillNodeCreationParams(jsonGraph, graphParams);
    if (m_run)
    {
        fillRecipeParams(graphParams, recipeParams);
    }

    // Synapse API phase

    const auto hostTimeStartTime = std::chrono::high_resolution_clock::now();

    // Graph creation
    synGraphHandle currentGraph = createGraph();
    createSections(currentGraph, graphParams);
    createTensors(currentGraph, graphParams);
    createNodes(currentGraph, graphParams);
    setBlockingNodes(currentGraph, graphParams);

    synGraphHandle  currentDuplicateGraph = nullptr;
    synRecipeHandle currentRecipe         = nullptr;
    bool            compilationSuccessful = false;
    try
    {
        // Graph duplicate
        currentDuplicateGraph = duplicateGraph(currentGraph, graphParams);
        // Graph compilation of duplicate
        currentRecipe         = compileGraph(currentDuplicateGraph, graphName);
        compilationSuccessful = true;
    }
    catch (const std::runtime_error& re)
    {
        cleanup(&currentGraph,
                currentDuplicateGraph ? &currentDuplicateGraph : nullptr,
                nullptr,
                graphParams,
                recipeParams);
        if (!m_keepGoing) throw;
        failedGraphs.emplace(graphIndex, re.what());
        currentRecipe = nullptr;
    }

    if (compilationSuccessful)
    {
        JT_LOG_INFO_NON_QUIET("Compile graph: " << graphIndex << " finished successfully");
    }
    else
    {
        JT_LOG_ERR("Compile graph: " << graphIndex << " finished with an error: " << failedGraphs.at(graphIndex));
        // TODO: maybe worth keeping a list of failures in case they change (or the iter of the failure)
        m_stats["failedGraphs"][graphIndex] = failedGraphs[graphIndex];
        return;
    }

    if (m_run)
    {
        executeRecipe(currentRecipe, recipeParams);
    }

    // Synapse resources cleanup
    cleanup(&currentGraph, &currentDuplicateGraph, &currentRecipe, graphParams, recipeParams);

    if (compilationSuccessful)
    {
        const auto hostTimeEndTime = std::chrono::high_resolution_clock::now();
        m_totalHostTime =
            std::chrono::duration_cast<std::chrono::nanoseconds>(hostTimeEndTime - hostTimeStartTime).count();

        // record stats for json
        recordJsonStats(graphIndex);
        resetStats();
    }
}

}  // namespace json_tests
