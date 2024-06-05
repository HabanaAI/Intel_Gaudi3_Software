#include "file_loader.h"
#include "graph_loader.h"
#include "hpp/syn_context.hpp"
#include <cstddef>

// creation methods
JsonFileLoaderPtr JsonFileLoader::createFromJsonContent(const std::string& jsonContent)
{
    return std::shared_ptr<JsonFileLoader>(new JsonFileLoader(json_utils::Json::parse(jsonContent)));
}

JsonFileLoaderPtr JsonFileLoader::createFromJsonFile(const std::string& jsonFilePath)
{
    return std::shared_ptr<JsonFileLoader>(new JsonFileLoader(json_utils::jsonFromFile(jsonFilePath)));
}

// getters
JsonGraphLoader JsonFileLoader::getGraphLoader(syn::Context&                  ctx,
                                               synDeviceType                  deviceType,
                                               std::optional<CompilationMode> compilationMode,
                                               unsigned                       graphIndex,
                                               const std::string&             constTensorsFilePath) const
{
    return JsonGraphLoader(ctx, deviceType, compilationMode, getGraph(graphIndex), constTensorsFilePath);
}

const nlohmann_hcl::json& JsonFileLoader::getJsonData() const
{
    return m_jsonData;
}

const nlohmann_hcl::json& JsonFileLoader::getGraphs() const
{
    return json_utils::get(m_jsonData, "graphs");
}

const nlohmann_hcl::json& JsonFileLoader::getGraph(unsigned graphIndex) const
{
    const auto& graphs = getGraphs();
    if (graphIndex >= graphs.size())
    {
        throw std::runtime_error("Graph index out of range, index: " + std::to_string(graphIndex) +
                                 ", graphs count: " + std::to_string(graphs.size()));
    }
    return graphs[graphIndex];
}

size_t JsonFileLoader::getNumOfGraphs() const
{
    return getGraphs().size();
}

const nlohmann_hcl::json JsonFileLoader::getConfig() const
{
    return json_utils::get(m_jsonData, "config", nlohmann_hcl::json());
}

uint32_t JsonFileLoader::getFileVersion() const
{
    return json_utils::get<uint32_t>(m_jsonData, "version", 0);
}

const std::string JsonFileLoader::getModelName() const
{
    return json_utils::get<std::string>(m_jsonData, "name", "");
}

void JsonFileLoader::loadEnv(syn::Context& ctx) const
{
    nlohmann_hcl::json globalConfig = json_utils::get(m_jsonData, "global_config", nlohmann_hcl::json());

    if (auto it = globalConfig.find("ENABLE_EXPERIMENTAL_FLAGS"); it != globalConfig.end())
    {
        if (std::getenv(it.key().c_str()) == nullptr)
        {
            ctx.setConfiguration(it.key(), it.value());
        }
    }

    for (auto it = globalConfig.begin(); it != globalConfig.end(); ++it)
    {
        if (it.key() == std::string_view{"POD_SIZE"}) continue;

        try
        {
            if (std::getenv(it.key().c_str()) == nullptr)
            {
                ctx.setConfiguration(it.key(), it.value());
            }
        }
        catch (const syn::Exception& e)
        {
        }
    }
}
