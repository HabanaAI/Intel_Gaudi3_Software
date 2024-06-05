

#pragma once

#include "graph_loader.h"
#include "json.hpp"
#include "json_utils.h"
#include "hpp/syn_context.hpp"

class JsonFileLoader;
using JsonFileLoaderPtr = std::shared_ptr<JsonFileLoader>;

class JsonFileLoader
{
public:
    // creation methods
    static JsonFileLoaderPtr createFromJsonContent(const std::string& jsonContent);
    static JsonFileLoaderPtr createFromJsonFile(const std::string& jsonFilePath);

    // getters
    JsonGraphLoader           getGraphLoader(syn::Context&                  ctx,
                                             synDeviceType                  deviceType,
                                             std::optional<CompilationMode> compilationMode,
                                             unsigned                       graphIndex,
                                             const std::string&             constTensorsFilePath = "") const;
    const nlohmann_hcl::json& getJsonData() const;
    const nlohmann_hcl::json& getGraphs() const;
    const nlohmann_hcl::json& getGraph(unsigned graphIndex) const;
    size_t                    getNumOfGraphs() const;
    const nlohmann_hcl::json  getConfig() const;
    uint32_t                  getFileVersion() const;
    const std::string         getModelName() const;

    void loadEnv(syn::Context& ctx) const;
    virtual ~JsonFileLoader() = default;
private:
    JsonFileLoader(nlohmann_hcl::json&& jsonData) : m_jsonData(std::move(jsonData)) {};

    const nlohmann_hcl::json m_jsonData;
};
