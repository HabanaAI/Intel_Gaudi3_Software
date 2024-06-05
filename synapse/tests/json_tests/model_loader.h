#pragma once

#include "file_loader.h"
#include "graph_loader.h"
#include "hpp/syn_context.hpp"
#include "utils/launcher.h"
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct GraphInfo
{
    std::optional<std::string> name;
    std::optional<size_t>      indexInFile;
    std::optional<std::string> recipeFile;
    std::optional<uint64_t>    workspaceSize;
    std::optional<uint64_t>    numPersistentTensors;
    std::optional<uint64_t>    persistentTensorsSize;
    std::optional<double>      time;
    std::vector<double>        times;
    std::vector<std::string>   warnings;
    std::vector<std::string>   errors;
    virtual nlohmann_hcl::json serialize()
    {
        nlohmann_hcl::json block;
        if (name.has_value()) block["name"] = name.value();
        if (indexInFile.has_value()) block["index_in_file"] = indexInFile.value();
        if (recipeFile.has_value()) block["recipe_file"] = recipeFile.value();
        if (workspaceSize.has_value()) block["workspace_size"] = workspaceSize.value();
        if (numPersistentTensors.has_value()) block["num_persistent_tensors"] = numPersistentTensors.value();
        if (persistentTensorsSize.has_value()) block["persistent_tensors_size"] = persistentTensorsSize.value();
        if (time.has_value()) block["time"] = time.value();
        if (!times.empty()) block["times"] = times;
        if (!warnings.empty()) block["warnings"] = warnings;
        if (!errors.empty()) block["errors"] = errors;
        return block;
    }
};

struct ComObject
{
    // supported com object fields
    std::optional<std::string> status;
    std::optional<std::string> errorMessage;
    std::vector<GraphInfo>     graphs;
    std::optional<std::string> timeUnits;
    std::optional<std::string> performanceUnits;
    std::optional<std::string> accuracyUnits;
    virtual nlohmann_hcl::json serialize()
    {
        nlohmann_hcl::json block;
        if (status.has_value()) block["status"] = status.value();
        if (errorMessage.has_value()) block["error"] = errorMessage.value();
        if (timeUnits.has_value()) block["time_units"] = timeUnits.value();
        if (performanceUnits.has_value()) block["performance_units"] = performanceUnits.value();
        if (accuracyUnits.has_value()) block["accuracy_units"] = accuracyUnits.value();
        for (size_t i = 0; i < graphs.size(); ++i)
        {
            std::string graphName      = std::to_string(i);
            block["graphs"][graphName] = graphs[i].serialize();
        }
        return block;
    }
};

class ModelLoader
{
public:
    ModelLoader(const std::string& comObjFilePath, const std::string& root);
    virtual ~ModelLoader();

    virtual bool        run() = 0;
    virtual std::string getError();

protected:
    ::synDeviceType         m_deviceType;
    std::set<synDeviceType> m_optionalDeviceTypes;

    std::vector<uint64_t>      m_groups;
    std::string                m_modelName;
    std::string                m_jsonFilePath;
    std::optional<std::string> m_tensorsDataFilePath;
    std::optional<std::string> m_dataComparatorFilePath;

    nlohmann_hcl::json        m_comObjectJson;
    std::string               m_comObjFilePath;
    std::string               m_recipeFolderPath;
    Launcher::TimeMeasurement m_timeMeasurement;
    CompilationMode           m_compilationMode;
    uint32_t                  m_compilationConsistencyIters;

    const std::string m_root;
    JsonFileLoaderPtr m_jsonFileLoader;
    unsigned          m_iterations {1};
    ComObject         m_comObject;
    bool              m_releaseDevice;
    syn::Context      m_ctx;
};

class CompileModelLoader : public ModelLoader
{
public:
    CompileModelLoader(const std::string& comObjFilePath);
    virtual ~CompileModelLoader() = default;

    bool run() override;
};

class RunModelLoader : public ModelLoader
{
public:
    RunModelLoader(const std::string& comObjFilePath);
    virtual ~RunModelLoader() = default;

    bool run() override;
};