#include "base_test.h"

#include "file_loader.h"
#include "graph_loader.h"
#include "json_test_debug_model.h"
#include "json_utils.h"
#include "utils/data_collector.h"
#include "utils/data_comparator.h"
#include "utils/data_container.h"
#include "utils/launcher.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <optional>
#include <string>

namespace json_tests
{
synDeviceType deviceTypeFromString(const std::string& name)
{
    if (name == "greco") return synDeviceGreco;
    if (name == "gaudi") return synDeviceGaudi;
    if (name == "gaudiM") return synDeviceGaudi;
    if (name == "gaudi2") return synDeviceGaudi2;
    if (name == "gaudi3") return synDeviceGaudi3;
    throw std::runtime_error(fmt::format("Invalid device type: {}", name));
}

TypedDeviceTest::TypedDeviceTest(const ArgParser& args)
: BaseTest(),
  m_deviceType(deviceTypeFromString(args.getValue<std::string>(an_device_type))),
  m_statsFilePath(args.getValueOrDefault(an_stats_file, std::string())),
  m_quietMode(args.getValueOrDefault(an_quiet, false)),
  m_keepGoing(args.getValueOrDefault(an_keep_going, false))
{
}

syn::Recipe TypedDeviceTest::compileGraph(const size_t index, const JsonGraphLoader& gl)
{
    JT_LOG_INFO_NON_QUIET("Compile graph: " << index << " (" << gl.getName() << ")");
    const auto& graph = gl.getGraph();

    const auto  compileBegin = std::chrono::steady_clock::now();
    syn::Recipe recipe       = graph.compile(gl.getName());
    if (!m_statsFilePath.empty())
    {
        const auto compileEnd = std::chrono::steady_clock::now();
        const auto delta      = std::chrono::duration_cast<std::chrono::nanoseconds>(compileEnd - compileBegin).count();
        m_stats["graphs"][std::to_string(index)]["compileTime"].push_back(delta);
        m_stats["graphs"][std::to_string(index)]["workspaceSize"] = recipe.getWorkspaceSize();
    }

    JT_LOG_INFO_NON_QUIET("Compile graph: " << index << " finished successfully");
    return recipe;
}

RunTypedDeviceTest::RunTypedDeviceTest(const ArgParser& args)
: TypedDeviceTest(args),
  m_useSyntheticData(args.getValueOrDefault(an_synthetic_data, false)),
  m_runIterations(args.getValueOrDefault(an_run_iter, 1)),
  m_dataFilePath(args.getValueOrDefault(an_data_file, std::string())),
  m_constDataOnly(args.getValueOrDefault(an_const_data_only, false)),
  m_comparatorConfigFilePath(args.getValueOrDefault(an_comp_config_file, std::string())),
  m_timeMeasurement(Launcher::timeMeasurementFromString(args.getValue<std::string>(an_time_measurement))),
  m_totalDuration(0)
{
    if (m_runIterations < 1)
    {
        throw std::runtime_error("NUMBER_OF_RUN_ITERATIONS must be >= 1");
    }
    if (m_useSyntheticData && !m_dataFilePath.empty())
    {
        throw std::runtime_error("using synthetic data is not allowed when loading tensors data from file");
    }
    if (m_runIterations > 1 && !m_dataFilePath.empty())
    {
        JT_LOG_WARN("only one iteration is allowed when using tensors data capture, reducing to 1");
        m_runIterations = 1;
    }
    if (!args.getValueOrDefault(an_reset_device, false) && args.getValueOrDefault(an_run, false))
    {
        m_device = Launcher::acquireDevice(m_ctx, m_deviceType, m_optionalDeviceTypes);
    }
}

static void validateData(const DataComparator::Status& status, const uint64_t dataIteration, bool throwOnError)
{
    if (!status.errors.empty())
    {
        for (const auto& t : status.errors)
        {
            JT_LOG_ERR("Data mismatch in data iteration: " << dataIteration << " on output tensor: " << t);
        }
        if (throwOnError)
        {
            throw std::runtime_error("wrong output data found");
        }
    }

    if (!status.warnings.empty())
    {
        for (const auto& t : status.warnings)
        {
            JT_LOG_WARN("Data mismatch in data iteration: " << dataIteration << " on output tensor: " << t);
        }
    }
}

void RunTypedDeviceTest::runIteration(const size_t                                   index,
                                      const syn::Recipe&                             recipe,
                                      const std::shared_ptr<DataProvider>&           dataProvider,
                                      uint64_t                                       dataIteration,
                                      const std::shared_ptr<DataComparator::Config>& compConfig)
{
    JT_LOG_INFO_NON_QUIET(fmt::format("Run graph: {} , data iteration: {}{}",
                                      index,
                                      dataIteration,
                                      (compConfig ? "" : " (metadata only)")));

    std::shared_ptr<DataCollector> dataCollector;
    Launcher::OnTensor             onOutputTensor;
    std::optional<std::string>     compareStatus;
    std::optional<DataComparator>  dataComparator;

    if (dataProvider)
    {
        dataProvider->setDataIteration(dataIteration);
        if (compConfig)
        {
            JT_LOG_INFO_NON_QUIET("Compare results of iteration: " << dataIteration);

            dataCollector  = std::make_shared<DataCollector>(recipe);
            dataComparator = DataComparator(dataCollector, dataProvider, *compConfig);
            compareStatus  = "finished successfully";
            onOutputTensor = [&](const std::string& name, const syn::HostBuffer& buffer) {
                dataCollector->setBuffer(name, buffer);
                auto sts = dataComparator.value().compare(name);
                if (!sts.errors.empty()) compareStatus = "finished with errors";
                validateData(sts, dataIteration, compConfig->breakOnFirstError);
                dataCollector->removeBuffer(name);
            };
        }
    }

    const auto  beginLaunch = std::chrono::steady_clock::now();
    syn::Device tmpDevice   = m_device ? m_device : Launcher::acquireDevice(m_ctx, m_deviceType, m_optionalDeviceTypes);

    const Launcher::Result res = Launcher::launch(tmpDevice,
                                                  recipe,
                                                  m_runIterations,
                                                  dataProvider,
                                                  onOutputTensor,
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

    if (compareStatus.has_value())
    {
        JT_LOG_INFO_NON_QUIET("Compare results of graph: " << index << " , data iteration: " << dataIteration << " "
                                                           << compareStatus.value());
    }
}

std::shared_ptr<DataProvider> RunTypedDeviceTest::getDataProvider(const JsonGraphLoader& gl)
{
    std::shared_ptr<DataProvider> dataProvider = nullptr;
    if (!m_dataFilePath.empty())
    {
        dataProvider = std::static_pointer_cast<DataProvider>(
            std::make_shared<CapturedDataProvider>(m_dataFilePath, gl.getName(), gl.getRecipeId(), gl.getGroup()));
    }
    else if (m_useSyntheticData)
    {
        dataProvider =
            std::static_pointer_cast<DataProvider>(std::make_shared<RandDataProvider>(0.5, 1.0, gl.getTensors()));
    }
    return dataProvider;
}

std::shared_ptr<DataComparator::Config> RunTypedDeviceTest::getComparatorConfig(const nlohmann_hcl::json& graph)
{
    if (m_comparatorConfigFilePath.empty())
    {
        return std::make_shared<DataComparator::Config>();
    }
    return std::make_shared<DataComparator::Config>(m_comparatorConfigFilePath, graph);
}

std::set<uint64_t> RunTypedDeviceTest::getDataIterations(const std::shared_ptr<DataProvider>& dataProvider) const
{
    return dataProvider ? dataProvider->getDataIterations() : std::set<uint64_t> {};
}

std::set<uint64_t> RunTypedDeviceTest::getNonDataIterations(const std::shared_ptr<DataProvider>& dataProvider) const
{
    return dataProvider ? dataProvider->getNonDataIterations() : std::set<uint64_t> {0};
}

void RunTypedDeviceTest::runGraph(const size_t                                   index,
                                  const syn::Recipe&                             recipe,
                                  const std::shared_ptr<DataProvider>&           dataProvider,
                                  const std::shared_ptr<DataComparator::Config>& compConfig)
{
    if (dataProvider)
    {
        std::set<uint64_t> nonDataIterations = dataProvider->getNonDataIterations();
        for (const auto& dataIter : nonDataIterations)
        {
            runIteration(index, recipe, dataProvider, dataIter);
        }
        std::set<uint64_t> dataIterations = dataProvider->getDataIterations();
        for (const auto& dataIter : dataIterations)
        {
            runIteration(index, recipe, dataProvider, dataIter, compConfig);
        }
    }
    else
    {
        runIteration(index, recipe, nullptr, 0);
    }
}

void RunTypedDeviceTest::dumpStats() const
{
    if (!m_statsFilePath.empty())
    {
        json_utils::jsonToFile(m_stats, m_statsFilePath);
    }
    JT_LOG_INFO_NON_QUIET("Total run time of " << m_totalDuration * 1e-6 << "[ms]");
}

JsonTest::JsonTest(const ArgParser& args)
: m_excludeGraphs(args.getValueOrDefault(an_exclude_graphs, false)),
  m_jsonFilePath(args.getValue<std::string>(an_json_file)),
  m_recipeFolderPath(args.getValueOrDefault<std::string>(an_serialize_recipe, "")),
  m_graphsIndices(args.getValues<uint64_t>(an_graphs_indices)),
  m_groups(args.getValues<uint64_t>(an_groups)),
  m_jsonFileLoader(JsonFileLoader::createFromJsonFile(m_jsonFilePath))
{
    const auto graphCount = m_jsonFileLoader->getNumOfGraphs();
    if (m_graphsIndices.empty())
    {
        m_graphsIndices.resize(graphCount);
        std::iota(m_graphsIndices.begin(), m_graphsIndices.end(), 0);
    }
    else if (m_excludeGraphs)
    {
        std::sort(m_graphsIndices.begin(), m_graphsIndices.end());
        m_graphsIndices.erase(std::unique(m_graphsIndices.begin(), m_graphsIndices.end()), m_graphsIndices.end());

        uint64_t              idx = 0;
        std::vector<uint64_t> tmp;
        tmp.reserve(graphCount);
        for (uint64_t i = 0; i < graphCount; ++i)
        {
            if (i == m_graphsIndices[idx]) ++idx;
            else
                tmp.push_back(i);
        }
        m_graphsIndices = tmp;
    }
}

std::optional<CompilationMode> stringToCompilationMode(const std::string& s)
{
    if (s == "graph")
    {
        return CompilationMode::Graph;
    }
    if (s == "eager")
    {
        return CompilationMode::Eager;
    }
    return std::nullopt;
}

}  // namespace json_tests
