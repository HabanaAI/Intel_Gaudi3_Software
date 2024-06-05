#pragma once

#include "graph_compiler/habana_global_conf.h"
#include "json_utils.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace graph_serialize
{
using Json = json_utils::Json;

// This class is the backend of serializing/deserializing various strategies into/from JSON file
class StrategySerializer final
{
public:
    static StrategySerializer& getInstance()
    {
        static StrategySerializer instance;
        return instance;
    }

    bool isImportingEnabled() const { return !m_importPath.empty(); }
    bool isExportingEnabled() const { return !m_exportPath.empty(); }

    const Json& getSerializationInfo(const std::string& strategyName,
                                     const std::string& graphName,
                                     const std::string& nodeName) const
    {
        if (m_importedData.count(graphName) != 0)
        {
            const Json& graphData = m_importedData.at(graphName);
            if (graphData.count(strategyName) != 0)
            {
                const Json& strategyData = graphData.at(strategyName);
                if (strategyData.count(nodeName) != 0) return strategyData.at(nodeName);
            }
        }
        const static Json defaultRes;
        return defaultRes;
    }

    Json& createNewSerializationInfo(const std::string& strategyName,
                                     const std::string& graphName,
                                     const std::string& nodeName)
    {
        auto graphIter = m_dataToExport.find(graphName);
        if (graphIter == m_dataToExport.end())
        {
            m_dataToExport[graphName] = {};
            graphIter                 = m_dataToExport.find(graphName);
        }
        auto strategyIter = graphIter->second.find(strategyName);
        if (strategyIter == graphIter->second.end())
        {
            graphIter->second[strategyName] = {};
            strategyIter                    = graphIter->second.find(strategyName);
        }
        auto nodeIter = strategyIter->second.find(nodeName);
        if (nodeIter == strategyIter->second.end())
        {
            strategyIter->second[nodeName] = {};
            nodeIter                       = strategyIter->second.find(nodeName);
        }
        else
        {
            nodeIter->second.clear();
        }
        return nodeIter->second;
    }

private:
    StrategySerializer()
    : m_importPath(GCFG_PATH_TO_IMPORT_STRATEGIES.value()),
      m_exportPath(GCFG_PATH_TO_EXPORT_STRATEGIES.value()),
      m_importedData(isImportingEnabled() ? json_utils::jsonFromFile(m_importPath) : Json())
    {
    }

    // In destructor do the serialization into file
    ~StrategySerializer()
    {
        if (isExportingEnabled())
        {
            json_utils::jsonToFile(m_dataToExport, m_exportPath, /*indent*/ 4);
        }
    }

    // Disable copy constructor and copy assignment operator to enforce single instance
    StrategySerializer(const StrategySerializer&) = delete;
    StrategySerializer& operator=(const StrategySerializer&) = delete;

private:
    const std::string m_importPath;
    const std::string m_exportPath;
    Json              m_importedData;

    using NodesTree    = std::map<std::string /*nodeName*/, Json>;
    using StrategyTree = std::map<std::string /*strategyName*/, NodesTree>;
    using GraphsTree   = std::map<std::string /*graphName*/, StrategyTree>;
    GraphsTree m_dataToExport;
};

}  // namespace graph_serialize