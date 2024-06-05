#include "config_compare_tests.h"

#include "utils/data_comparator.h"
#include "utils/data_provider.h"
#include "base_test.h"
#include "gc_tests/platform_tests/infra/gc_tests_utils.h"

#include <iostream>

namespace json_tests
{
ConfigCompareTest::ConfigCompareTest(const ArgParser& args) : PlaybackTest(args)
{
    if (!m_device)
    {
        m_device = Launcher::acquireDevice(m_ctx, m_deviceType, m_optionalDeviceTypes);
    }

    m_run = true;
    if (m_dataFilePath.empty())
    {
        m_useSyntheticData = true;
    }
    initRunsConfigurations(args);
}

void ConfigCompareTest::generateRunsConfigurationsFromConfigFile(const std::string& configFilePath)
{
    auto jsonConfigFileData = json_utils::jsonFromFile(configFilePath);
    for (const auto& jsonRunConfs : jsonConfigFileData)
    {
        std::vector<HabanaGlobalConfig> confs;
        for (auto it = jsonRunConfs.begin(); it != jsonRunConfs.end(); ++it)
        {
            if (!it.value().is_string())
            {
                confs.push_back({it.key(), json_utils::toString(it.value())});
            }
            else
            {
                confs.push_back({it.key(), it.value()});
            }
        }
        m_runsConfs.push_back(confs);
    }
}

void ConfigCompareTest::generateRunsConfigurationsFromConfigsList(const std::vector<std::string>& configCompareValues)
{
    for (size_t i = 0; i < configCompareValues.size(); i += 2)
    {
        m_runsConfs.push_back({{configCompareValues[i], configCompareValues[i + 1]}});
    }
}

void ConfigCompareTest::initRunsConfigurations(const ArgParser& args)
{
    auto configCompareValues   = args.getValues<std::string>(an_config_compare_values);
    auto configCompareFilePath = getOptionalArg<std::string>(args, an_config_compare_file);
    if (!configCompareValues.empty() && configCompareFilePath.has_value())
    {
        throw std::runtime_error(
            "config compare test supports only one mode at a time: --config-compare-values or --config-compare-file");
    }
    else if (configCompareValues.size() > 0 && configCompareValues.size() % 2 == 0)
    {
        generateRunsConfigurationsFromConfigsList(configCompareValues);
    }
    else if (configCompareFilePath.has_value())
    {
        generateRunsConfigurationsFromConfigFile(configCompareFilePath.value());
    }
    else
    {
        throw std::runtime_error("config compare test arguments are not as expected");
    }

    if (m_runsConfs.size() == 1)
    {
        m_runsConfs.push_back({});
    }
}

static std::string getConfsString(const std::vector<HabanaGlobalConfig>& v)
{
    if (v.empty())
    {
        return "default";
    }
    else
    {
        std::stringstream ss;
        std::string       del;
        for (const auto& config : v)
        {
            ss << fmt::format("{}{}={}", del, config.configName, config.configValue);
            del = " ";
        }
        return ss.str();
    }
}

void ConfigCompareTest::runTest(bool run)
{
    const std::string model = m_jsonFileLoader->getModelName();
    const std::string msg   = run ? "Compile and run" : "Compile";
    JT_LOG_INFO(msg << " model: " << model);
    ScopedConfig expConfigSetter(m_ctx, "ENABLE_EXPERIMENTAL_FLAGS", "1");
    for (const auto& i : m_graphsIndices)
    {
        // compile and run the graph with different configurations
        std::vector<Result>                     refResultsVec;
        std::string                             refConfsString;
        std::shared_ptr<DataComparator::Config> compConfig = nullptr;

        for (const auto& runConfs : m_runsConfs)
        {
            std::vector<std::unique_ptr<ScopedConfig>> runConfigsSetter;
            if (runConfs.size() > 0)
            {
                for (const auto& runConf : runConfs)
                {
                    JT_LOG_INFO("Set config: " << runConf.configName << ", value: " << runConf.configValue);
                    runConfigsSetter.push_back(
                        std::make_unique<ScopedConfig>(m_ctx, runConf.configName, runConf.configValue));
                }
            }
            else
            {
                JT_LOG_INFO("Set configs to default values");
            }

            auto gl = m_jsonFileLoader->getGraphLoader(m_ctx, m_deviceType, m_compilationMode, i);

            if (compConfig == nullptr)
            {
                compConfig = getComparatorConfig(gl.getJsonGraph());
            }

            if (!m_groups.empty() && std::find(m_groups.begin(), m_groups.end(), gl.getGroup()) == m_groups.end())
                continue;

            const auto recipe         = compileGraph(i, gl);
            const auto dataProvider   = getDataProvider(gl);
            const auto currResultsVec = runRecipe(i, recipe, dataProvider);
            if (refResultsVec.empty())
            {
                refResultsVec  = currResultsVec;
                refConfsString = getConfsString(runConfs);
            }
            else
            {
                // compare results
                if (refResultsVec.size() != currResultsVec.size())
                {
                    std::string msg = "The number of results between the runs does not match";
                    if (compConfig->breakOnFirstError) throw std::runtime_error(msg);
                    JT_LOG_ERR(msg);
                    continue;
                }
                JT_LOG_INFO("Compare results: current run configurations: "
                            << getConfsString(runConfs) << ", ref run configurations: " << refConfsString);
                for (size_t j = 0; j < refResultsVec.size(); ++j)
                {
                    validateData(i,
                                 refResultsVec[j].dataIteration,
                                 compConfig,
                                 refResultsVec[j].dataCollector,
                                 currResultsVec[j].dataCollector);
                }
            }
        }
    }
    JT_LOG_INFO(msg << " model: " << model << " finished successfully");
}

std::vector<Result> ConfigCompareTest::runRecipe(const size_t                         index,
                                                 const syn::Recipe&                   recipe,
                                                 const std::shared_ptr<DataProvider>& dataProvider)
{
    std::vector<Result> results;
    if (!dataProvider)
    {
        throw std::runtime_error("Data provider is required for ConfigCompareTest");
    }
    std::set<uint64_t> dataIterations = dataProvider->getAllDataIterations();
    if (dataIterations.empty())
    {
        throw std::runtime_error("Data provider does not contain any tensors which is mandatory for ConfigCompareTest");
    }
    for (const auto& dataIter : dataIterations)
    {
        if (!m_iterationsFilter.empty() &&
            std::find(m_iterationsFilter.begin(), m_iterationsFilter.end(), dataIter) == m_iterationsFilter.end())
            continue;
        results.push_back(runIteration(index, recipe, dataProvider, dataIter, false));
    }
    return results;
}

Result ConfigCompareTest::runIteration(const size_t                         index,
                                       const syn::Recipe&                   recipe,
                                       const std::shared_ptr<DataProvider>& dataProvider,
                                       uint64_t                             dataIteration,
                                       bool                                 metadataOnly)
{
    JT_LOG_INFO_NON_QUIET(fmt::format("Run graph: {} , data iteration: {}{}",
                                      index,
                                      dataIteration,
                                      (metadataOnly ? " (metadata only)" : "")));

    HB_ASSERT(dataProvider, "null data provider on graph index: {}", index);

    std::shared_ptr<DataCollector> dataCollector;
    dataCollector = std::make_shared<DataCollector>(recipe);

    const auto  beginLaunch = std::chrono::steady_clock::now();
    syn::Device tmpDevice   = m_device ? m_device : Launcher::acquireDevice(m_ctx, m_deviceType, m_optionalDeviceTypes);

    const Launcher::Result res = Launcher::launch(tmpDevice,
                                                  recipe,
                                                  m_runIterations,
                                                  dataProvider,
                                                  dataCollector,
                                                  m_timeMeasurement,
                                                  m_keepGoing);
    for (const auto& w : res.warnings)
    {
        JT_LOG_WARN(w);
    }
    if (!m_statsFilePath.empty())
    {
        const auto endLaunch = std::chrono::steady_clock::now();
        const auto delta     = std::chrono::duration_cast<std::chrono::nanoseconds>(endLaunch - beginLaunch).count();
        const auto indexStr  = std::to_string(index);
        m_stats["graphs"][indexStr]["hostRuntimeE2E"].push_back(delta);
        if (!res.durations.empty())
        {
            m_stats["graphs"][indexStr]["deviceRuntime"].push_back(res.durations);
        }
    }
    if (res.durations.empty())
    {
        if (!m_quietMode)
        {
            JT_LOG_WARN("failed to measure execution time");
        }
        else if (!m_warnedAboutMeasureFailureAlready)
        {
            JT_LOG_WARN("failed to measure execution time (Will not be repeated due to JT_QUIET_MODE)");
            m_warnedAboutMeasureFailureAlready = true;
        }
    }
    else
    {
        JT_LOG_INFO_NON_QUIET("Run graph: " << index << " , data iteration: " << dataIteration
                                            << " finished successfully");

        double average = res.durations.front();
        if (res.durations.size() > 1)
        {
            average = accumulate(res.durations.begin() + 1, res.durations.end(), 0.0) / (res.durations.size() - 1);
        }
        m_totalDuration += average;
        JT_LOG_INFO_NON_QUIET("Average run time of " << (res.durations.size() == 1 ? 1 : res.durations.size() - 1)
                                                     << " iterations: " << average * 1e-6 << "[ms]");
    }

    return {dataIteration, dataCollector};
}

void ConfigCompareTest::validateData(const size_t                                   index,
                                     const uint64_t                                 dataIteration,
                                     const std::shared_ptr<DataComparator::Config>& compConfig,
                                     const std::shared_ptr<DataContainer>           referenceData,
                                     const std::shared_ptr<DataContainer>           actualData)
{
    if (actualData == nullptr || referenceData == nullptr)
    {
        throw std::runtime_error("Missing data for validation");
    }
    JT_LOG_INFO_NON_QUIET("Compare results of iteration: " << dataIteration);
    DataComparator dc(actualData, referenceData, *compConfig);
    auto           res = dc.compare();
    if (!res.errors.empty())
    {
        for (const auto& t : res.errors)
        {
            JT_LOG_ERR("Data mismatch in data iteration: " << dataIteration << " on output tensor: " << t);
        }
        if (compConfig->breakOnFirstError)
        {
            throw std::runtime_error("wrong output data found");
        }
    }
    if (!res.warnings.empty())
    {
        for (const auto& t : res.warnings)
        {
            JT_LOG_WARN("Data mismatch in data iteration: " << dataIteration << " on output tensor: " << t);
        }
    }
    JT_LOG_INFO_NON_QUIET("Compare results of graph: " << index << " , data iteration: " << dataIteration
                                                       << " finished "
                                                       << (res.errors.empty() ? "successfully" : "with errors"));
}
}  // namespace json_tests