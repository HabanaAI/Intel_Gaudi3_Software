#pragma once

#include "graph_serializer/graph_serializer.h"
#include "json.hpp"
#include "types.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace graph_serialize
{

struct SerializeGraphInfo
{
    SerializeGraphInfo(const std::string& name, nlohmann_hcl::json&& json, const std::vector<size_t>& indices)
    : recipeName(name), jsonData(std::move(json)), tensorIndices(indices)
    {
    }
    std::string         recipeName;
    nlohmann_hcl::json  jsonData;
    std::vector<size_t> tensorIndices;
};

class GraphSerializerImpl : public GraphSerializer
{
public:
    GraphSerializerImpl(const Config& config);
    virtual ~GraphSerializerImpl() = default;

    void         addGraph(const synGraphHandle graph, CompilationMode compileMode) override;
    void         removeGraph(const synGraphHandle graph) override;
    const Graph& getGraph(const synGraphHandle graph) const override;
    void         addNode(const synGraphHandle graph, const Node& node) override;

    void setBlockingNodes(const synGraphHandle          graph,
                          const synNodeId               nodeId,
                          const std::vector<synNodeId>& blocking) override;
    void setDeterministic(const synGraphHandle graph, const synNodeId nodeId, const bool useDeterministic) override;
    void
    setRoundingMode(const synGraphHandle graph, const synNodeId nodeId, const synRoundingMode roundingMode) override;
    void setParams(const synGraphHandle graphHandle,
                   const synNodeId      nodeId,
                   const void*          userParams,
                   const unsigned       paramsSize) override;

    /*DEPRECATED*/
    void setGraphAttributes(synGraphHandle           graphHandle,
                            const synGraphAttribute* attributes,
                            const uint64_t*          values,
                            const uint32_t           size) override;

    void setGraphAttributesV2(synGraphHandle              graphHandle,
                              const synGraphAttribute*    attributes,
                              const synGraphAttributeVal* values,
                              const uint32_t              size) override;

    void generateGraph(const synGraphHandle graph, const std::string& recipeName, const std::string& tensorsFilePath);

protected:
    mutable std::mutex                        m_mutex;
    std::unordered_map<synGraphHandle, Graph> m_graphs;
    Config                                    m_config;
    std::vector<SerializeGraphInfo>           m_serializeGraphsInfo;
};

class SplitGraphSerializer : public GraphSerializerImpl
{
public:
    SplitGraphSerializer(const Config& config);
    virtual ~SplitGraphSerializer() = default;

    uint32_t serialize(const synGraphHandle graph,
                       const std::string&   recipeName,
                       bool                 isRecording,
                       const std::string&   uniqueId) override;
    void     mapRecipeToGraphJson(synRecipeHandle* pRecipeHandle, const std::string& uniqueId) override;
    void     recordGraph(const synRecipeHandle recipeHandle, uint16_t recipeId) override;
    void     postCompilationUpdate(uint32_t graphIndex, std::optional<uint16_t> recipeId) override;

private:
    std::string getJsonFilePath(const std::string& recipeName);

    std::string m_folderPath;

    typedef struct RecordAttributes
    {
        std::string        name;
        nlohmann_hcl::json graphJson;
        std::string        filePath;

        RecordAttributes(const std::string& name, nlohmann_hcl::json& jsonGrpah, const std::string& filePath)
        : name(name), graphJson(jsonGrpah), filePath(filePath)
        {
        }
    } RecordAttributes;

    std::unordered_map<std::string, std::shared_ptr<RecordAttributes>> m_uniqueIdToRecAttrsMap;
    std::unordered_map<synRecipeHandle, std::string>                   m_recipeGraphMap;
};

class UnifiedGraphSerializer : public GraphSerializerImpl
{
public:
    UnifiedGraphSerializer(const Config& config);
    virtual ~UnifiedGraphSerializer();

    uint32_t serialize(const synGraphHandle graph,
                       const std::string&   recipeName,
                       bool                 isRecording,
                       const std::string&   uniqueId) override;
    void     mapRecipeToGraphJson(synRecipeHandle* pRecipeHandle, const std::string& uniqueId) override;
    void     recordGraph(const synRecipeHandle recipeHandle, uint16_t recipeId) override;
    void     postCompilationUpdate(uint32_t graphIndex, std::optional<uint16_t> recipeId) override;

private:
    void writeThread();
    void generateModel();

    std::string                     m_lockFilePath;
    std::mutex                      m_writeMutex;
    std::condition_variable         m_writeCv;
    bool                            m_writing = true;
    std::vector<nlohmann_hcl::json> m_graphsDataWithRecipeId;
    std::thread                     m_writeThread;

    std::vector<uint16_t>                               m_recipesIds;
    std::unordered_map<std::string, nlohmann_hcl::json> m_uniqueIdToGraphJsonMap;
    std::unordered_map<synRecipeHandle, std::string>    m_recipeGraphMap;
};
}  // namespace graph_serialize