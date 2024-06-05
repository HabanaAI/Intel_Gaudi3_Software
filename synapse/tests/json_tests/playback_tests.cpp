#include "playback_tests.h"

#include "base_test.h"
#include "graph_loader.h"
#include "hpp/syn_graph.hpp"
#include "json_utils.h"
#include "utils/data_provider.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace json_tests
{
PlaybackTest::PlaybackTest(const ArgParser& args)
: RunTypedDeviceTest(args),
  JsonTest(args),
  m_run(args.getValueOrDefault(an_run, false)),
  m_iterationsFilter(args.getValues<uint64_t>(an_run_iter_filter)),
  m_compilationMode(stringToCompilationMode(args.getValue<std::string>(an_compilation_mode))),
  m_testIterations(args.getValueOrDefault(an_test_iter, 1))
{
    setup();
}

void PlaybackTest::setup()
{
    if (m_testIterations < 1)
    {
        throw std::runtime_error("NUMBER_OF_TEST_ITERATIONS must be >= 1");
    }

    m_jsonFileLoader->loadEnv(m_ctx);
}

void PlaybackTest::run()
{
    runTest(m_run);
    dumpStats();
}

void PlaybackTest::runTest(bool run)
{
    const std::string model = m_jsonFileLoader->getModelName();
    const std::string msg   = run ? "Compile and run" : "Compile";

    JT_LOG_INFO(msg << " model: " << model);

    const float step         = m_testIterations / 20.f;
    float       nextBoundary = 0;

    std::map<int, std::string> failedGraphs;
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

        const auto iterBegin = std::chrono::steady_clock::now();
        for (const auto& i : m_graphsIndices)
        {
            if (failedGraphs.count(i)) continue;

            // TODO: avoid copy of the graph loader
            auto gl = m_jsonFileLoader->getGraphLoader(m_ctx, m_deviceType, m_compilationMode, i, m_dataFilePath);

            // set given configurations for each graph separately
            std::vector<std::unique_ptr<ScopedConfig>> configsSetter;
            loadGraphConfigs(gl.getConfig(), configsSetter);

            if (!m_groups.empty() && std::find(m_groups.begin(), m_groups.end(), gl.getGroup()) == m_groups.end())
                continue;

            syn::Recipe recipe;
            try
            {
                recipe = compileGraph(i, gl);
                if (!m_recipeFolderPath.empty())
                {
                    recipe.serialize(fmt::format("{}/{}.recipe", m_recipeFolderPath, gl.getName()));
                }
            }
            catch (const std::runtime_error& re)
            {
                if (!m_keepGoing) throw;
                failedGraphs.emplace(i, re.what());
                continue;
            }

            if (run)
            {
                const std::shared_ptr<DataProvider>           dataProvider     = m_constDataOnly ? nullptr : getDataProvider(gl);
                const std::shared_ptr<DataComparator::Config> comparatorConfig = getComparatorConfig(gl.getJsonGraph());

                const auto nonDataIterations = getNonDataIterations(dataProvider);
                for (const auto& it : nonDataIterations)
                {
                    if (!m_iterationsFilter.empty() &&
                        std::find(m_iterationsFilter.begin(), m_iterationsFilter.end(), it) == m_iterationsFilter.end())
                        continue;
                    runIteration(i, recipe, dataProvider, it);
                }
                const auto dataIterations = getDataIterations(dataProvider);
                for (const auto& it : dataIterations)
                {
                    if (!m_iterationsFilter.empty() &&
                        std::find(m_iterationsFilter.begin(), m_iterationsFilter.end(), it) == m_iterationsFilter.end())
                        continue;
                    runIteration(i, recipe, dataProvider, it, comparatorConfig);
                }
            }
        }
        if (!m_statsFilePath.empty())
        {
            const auto iterEnd = std::chrono::steady_clock::now();
            const auto delta   = std::chrono::duration_cast<std::chrono::nanoseconds>(iterEnd - iterBegin).count();
            m_stats["iters"].push_back(delta);
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
        // TODO: maybe worth keeping a list of failures in case they change (or the iter of the failure)
        m_stats["failedGraphs"] = failedGraphs;
    }
}

void PlaybackTest::loadGraphConfigs(nlohmann_hcl::json                          graphConfig,
                                    std::vector<std::unique_ptr<ScopedConfig>>& configsSetter)
{
    // set configurations for a given graph.
    for (auto it = graphConfig.begin(); it != graphConfig.end(); ++it)
    {
        JT_LOG_INFO("Set config: " << it.key() << ", value: " << it.value());
        configsSetter.push_back(std::make_unique<ScopedConfig>(m_ctx, it.key(), it.value()));
    }
}

}  // namespace json_tests
