#include "model_loader.h"

#include "base_test.h"
#include "infra/recipe/recipe_compare.hpp"
#include "hpp/syn_device.hpp"
#include "json_utils.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "utils/launcher.h"

#include <iostream>
#include <optional>

#define CHECK_THROW(cond, msg)                                                                                         \
    if (!(cond)) throw std::runtime_error(msg);
#define CHECK_STATUS_THROW(cmd, msg) CHECK_THROW(synSuccess == cmd, msg)

using SynEvents = std::vector<std::pair<synEventHandle, synEventHandle>>;

static CompilationMode compilationModeFromString(const std::string& str)
{
    if (str == "eager") return CompilationMode::Eager;
    if (str == "graph") return CompilationMode::Graph;
    throw std::runtime_error("Invalid compilation mode string: " + str);
}

static synDeviceType deviceTypeFromString(const std::string& str)
{
    if (str == "greco") return synDeviceGreco;
    if (str == "gaudi") return synDeviceGaudi;
    if (str == "gaudiM") return synDeviceGaudi;
    if (str == "gaudi2") return synDeviceGaudi2;
    if (str == "gaudi3") return synDeviceGaudi3;
    throw std::runtime_error("Invalid device type string: " + str);
}

static std::shared_ptr<DataProvider> getDataProvider(const std::optional<std::string>& dataFilePath,
                                                     const syn::Device&                device,
                                                     const std::string&                name,
                                                     uint16_t                          recipeId)
{
    std::shared_ptr<DataProvider> dataProvider = nullptr;
    if (dataFilePath.has_value())
    {
        dataProvider = std::static_pointer_cast<DataProvider>(
            std::make_shared<CapturedDataProvider>(dataFilePath.value(), name, recipeId, 0));
    }
    return dataProvider;
}

ModelLoader::ModelLoader(const std::string& comObjFilePath, const std::string& root)
: m_comObjFilePath(comObjFilePath), m_root(root)
{
    m_comObjectJson        = json_utils::jsonFromFile(m_comObjFilePath);
    const auto& config     = json_utils::get(m_comObjectJson, "config");
    std::string name       = json_utils::get(config, "name");
    std::string workFolder = json_utils::get(config, "work_folder");

    m_iterations                  = json_utils::get(config, "iterations");
    m_modelName                   = json_utils::get(config, "model");
    m_jsonFilePath                = json_utils::get(config, "model_file");
    m_tensorsDataFilePath         = json_utils::get_opt<std::string>(config, "model_data_file");
    m_dataComparatorFilePath      = json_utils::get_opt<std::string>(config, "data_comparator_config_file");
    m_releaseDevice               = json_utils::get(config, "release_device");
    m_groups                      = json_utils::get(config, "groups", std::vector<uint64_t> {0});
    m_compilationMode             = compilationModeFromString(json_utils::get(config, "compilation_mode"));
    m_compilationConsistencyIters = json_utils::get(config, "compile_consistency_iters", 0);

    m_recipeFolderPath =
        fmt::format("{}/{}",
                    workFolder,
                    json_utils::get(config, "recipe_folder", fmt::format("{}/{}/recipes", name, m_modelName)));

    m_timeMeasurement = Launcher::timeMeasurementFromString(
        json_utils::get(config,
                        "time_measurement",
                        Launcher::timeMeasurementToString(Launcher::TimeMeasurement::EVENETS)));
    m_deviceType = deviceTypeFromString(m_comObjectJson["config"]["device"]);
    // allow fallback to gaudi/gaudiM since both can run with the same recipe
    m_jsonFileLoader = JsonFileLoader::createFromJsonFile(m_jsonFilePath);
    m_jsonFileLoader->loadEnv(m_ctx);
}

ModelLoader::~ModelLoader()
{
    m_comObjectJson[m_root] = m_comObject.serialize();
    json_utils::jsonToFile(m_comObjectJson, m_comObjFilePath);
}

std::string ModelLoader::getError()
{
    return m_comObject.errorMessage.has_value() ? m_comObject.errorMessage.value() : "unknown error";
}

CompileModelLoader::CompileModelLoader(const std::string& comObjFilePath) : ModelLoader(comObjFilePath, "compile") {}

bool CompileModelLoader::run()
{
    size_t graphIndex = 0;
    try
    {
        m_comObject.timeUnits = "ms";
        std::string test      = m_modelName;

        const size_t nGraphs = m_jsonFileLoader->getNumOfGraphs();

        if (m_compilationConsistencyIters)
        {
            JT_LOG_INFO("Run consistency check over " << (m_compilationConsistencyIters + 1)
                                                      << " compilations of model: " << m_modelName);
        }

        for (; graphIndex < nGraphs; ++graphIndex)
        {
            const std::string recipeName     = test + ".g_" + std::to_string(graphIndex);
            const std::string recipeFileName = fmt::format("{}.recipe", recipeName);
            const std::string recipeFilePath = fmt::format("{}/{}", m_recipeFolderPath, recipeFileName);
            const auto&       jsonGraph      = m_jsonFileLoader->getGraph(graphIndex);
            const uint64_t    group          = json_utils::get<uint64_t>(jsonGraph, "group", 0);

            if (std::find(m_groups.begin(), m_groups.end(), group) == m_groups.end()) continue;

            auto& res       = m_comObject.graphs.emplace_back();
            res.indexInFile = graphIndex;

            auto gl = m_jsonFileLoader->getGraphLoader(m_ctx,
                                                       m_deviceType,
                                                       m_compilationMode,
                                                       graphIndex,
                                                       m_tensorsDataFilePath.value_or(""));

            JT_LOG_INFO("Compile model: " << m_modelName << ", graph: " << graphIndex << ", name: " << gl.getName());

            const auto  start  = std::chrono::steady_clock::now();
            syn::Recipe recipe = gl.getGraph().compile(recipeName);
            const auto  stop   = std::chrono::steady_clock::now();

            res.name          = gl.getName();
            res.recipeFile    = recipeFileName;
            res.time          = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
            res.workspaceSize = recipe.getWorkspaceSize();
            const std::vector<synRecipeAttribute> attributes = {RECIPE_ATTRIBUTE_NUM_PERSISTENT_TENSORS,
                                                                RECIPE_ATTRIBUTE_PERSISTENT_TENSORS_SIZE};
            auto                                  attrs      = recipe.getAttributes(attributes);
            res.numPersistentTensors                         = attrs[0];
            res.persistentTensorsSize                        = attrs[1];

            recipe.serialize(recipeFilePath);

            JT_LOG_INFO("Compile model: " << m_modelName << ", graph: " << graphIndex << " finished successfully");

            for (int iter = 0; iter < m_compilationConsistencyIters; ++iter)
            {
                using namespace RecipeCompare;
                auto currGl = m_jsonFileLoader->getGraphLoader(m_ctx,
                                                               m_deviceType,
                                                               m_compilationMode,
                                                               graphIndex,
                                                               m_tensorsDataFilePath.value_or(""));

                syn::Recipe currRecipe = currGl.getGraph().compile(recipeName);
                if (*(currRecipe.handle()->basicRecipeHandle.recipe) != *(recipe.handle()->basicRecipeHandle.recipe))
                {
                    recipe.serialize(recipeFilePath + ".consistency_failure");
                    m_comObject.graphs[graphIndex].errors.push_back(
                        fmt::format("Found inconsistent recipe in graph number {}", graphIndex));
                    break;
                }
            }
        }

        m_comObject.status = "pass";
        return true;
    }
    catch (const std::exception& e)
    {
        m_comObject.status       = "fail";
        m_comObject.errorMessage = fmt::format("graph index {}: {}", graphIndex, e.what());
        return false;
    }
}

RunModelLoader::RunModelLoader(const std::string& comObjFilePath) : ModelLoader(comObjFilePath, "run") {}

bool RunModelLoader::run()
{
    size_t graphIndexInFile = 0;
    try
    {
        syn::Device device =
            m_releaseDevice ? syn::Device() : Launcher::acquireDevice(m_ctx, m_deviceType, m_optionalDeviceTypes);
        auto compile = json_utils::get(m_comObjectJson, "compile");
        auto graphs  = json_utils::get(compile, "graphs");
        if (graphs.empty())
        {
            throw std::runtime_error("No graphs found to run");
        }
        m_comObject.graphs.resize(graphs.size());
        m_comObject.timeUnits = "ns";
        for (size_t i = 0; i < graphs.size(); ++i)
        {
            std::string graphId         = std::to_string(i);
            auto        graph           = json_utils::get(graphs, graphId);
            graphIndexInFile            = json_utils::get(graph, "index_in_file");
            const std::string name      = json_utils::get(graph, "name");
            const auto&       jsonGraph = m_jsonFileLoader->getGraph(i);
            const uint16_t    recipeId  = json_utils::get(jsonGraph, "recipe_id", -1);
            JT_LOG_INFO(
                fmt::format("Run model: {}, graph index: {}, graph name: {}", m_modelName, graphIndexInFile, name));
            const std::string recipeFileName = json_utils::get(graph, "recipe_file");
            const std::string recipeFilePath = fmt::format("{}/{}", m_recipeFolderPath, recipeFileName);
            syn::Recipe       recipe(recipeFilePath);

            m_comObject.graphs[i].indexInFile = graphIndexInFile;
            m_comObject.graphs[i].name        = name;

            syn::Device tmpDevice =
                device ? device : Launcher::acquireDevice(m_ctx, m_deviceType, m_optionalDeviceTypes);
            auto dataProvider  = getDataProvider(m_tensorsDataFilePath, tmpDevice, name, recipeId);
            auto dataCollector = m_tensorsDataFilePath ? std::make_shared<DataCollector>(recipe) : nullptr;
            std::optional<DataComparator> dataComparator;

            Launcher::OnTensor onOutputTensor;
            if (dataCollector)
            {
                JT_LOG_INFO(
                    fmt::format("Comparing tensors data, model: {} , graph: {} compare config file: {}",
                                m_modelName,
                                i,
                                m_dataComparatorFilePath.has_value() ? m_dataComparatorFilePath.value() : "default"));
                DataComparator::Config config = m_dataComparatorFilePath.has_value()
                                                    ? DataComparator::Config(m_dataComparatorFilePath.value(), {})
                                                    : DataComparator::Config();
                dataComparator                = DataComparator(dataCollector, dataProvider, std::move(config));
                onOutputTensor                = [&](const std::string& name, const syn::HostBuffer& buffer) {
                    dataCollector->setBuffer(name, buffer);
                    auto sts                     = dataComparator.value().compare(name);
                    m_comObject.graphs[i].errors = sts.errors;
                    m_comObject.graphs[i].warnings.insert(m_comObject.graphs[i].warnings.end(),
                                                          sts.warnings.begin(),
                                                          sts.warnings.end());
                    dataCollector->removeBuffer(name);
                };
            }

            const Launcher::Result perfResults =
                Launcher::launch(tmpDevice, recipe, m_iterations, dataProvider, onOutputTensor, m_timeMeasurement);

            m_comObject.graphs[i].times = perfResults.durations;
            m_comObject.graphs[i].warnings.insert(m_comObject.graphs[i].warnings.end(),
                                                  perfResults.warnings.begin(),
                                                  perfResults.warnings.end());

            for (const auto& w : perfResults.warnings)
            {
                JT_LOG_WARN(w);
            }
            JT_LOG_INFO("Run model: " << m_modelName << ", graph: " << graphIndexInFile << " finished successfully");
        }

        m_comObject.status = "pass";

        return true;
    }
    catch (const std::exception& e)
    {
        m_comObject.status       = "fail";
        m_comObject.errorMessage = fmt::format("graph index {}: {}", graphIndexInFile, e.what());
        return false;
    }
}