#pragma once

#include "compiler_types.h"
#include "types.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

class synSingletonInterface;

namespace graph_serialize
{
struct Node
{
    Node(const synGraphHandle graphHandle,
         const synTensor*     inputs,
         const synTensor*     outputs,
         const uint32_t       sizeInputs,
         const uint32_t       sizeOutputs,
         const void*          userParams,
         const unsigned       paramsSize,
         const char*          guid,
         const char**         inputLayouts,
         const char**         outputLayouts,
         const std::string&   name,
         synNodeId            nodeUniqueId);

    // Special constructor used for graph duplication
    Node(synGraphHandle      newGraphHandle,
         const Node&         orgNode,
         synNodeId           newNodeUniqueId,
         synTensorHandleMap* tensorsMap,
         uint32_t            numTensors);

    uint64_t                       graphId;
    std::vector<Tensor*>           inputTensors;
    std::vector<Tensor*>           outputTensors;
    std::vector<std::string>       inputLayouts;
    std::vector<std::string>       outputLayouts;
    std::vector<char>              userParams;
    std::string                    guid;
    std::string                    name;
    synNodeId                      nodeUniqueId;
    std::optional<bool>            deterministic {};
    std::optional<synRoundingMode> roundingMode {};
    std::vector<std::string>       blockingNodes {};
};

struct Graph
{
    uint32_t                                        graphId;
    CompilationMode                                 graphCompilationMode;
    std::vector<Node>                               nodes;
    std::unordered_map<synNodeId, uint64_t>         nodeIdToIndex;
    std::set<std::string>                           nodesNames;
    std::unordered_map<synGraphAttribute, uint64_t> graphAttributes;
};

enum SerializeType
{
    SPLIT,
    UNIFIED
};

struct Config
{
    SerializeType           type;
    uint64_t                rankId;
    std::string             filePath;
    std::string             tensorsFilePath;
    std::optional<uint64_t> filterByElements;
};

class GraphSerializer
{
public:
    GraphSerializer();
    virtual ~GraphSerializer() = default;

    static std::unique_ptr<GraphSerializer> createGraphSerializer(const Config& config);

    virtual uint32_t serialize(const synGraphHandle graph,
                               const std::string&   recipeName,
                               bool                 isRecording,
                               const std::string&   unique_id)                                                    = 0;
    virtual void     addGraph(const synGraphHandle graph, CompilationMode compileMode = CompilationMode::Graph) = 0;
    virtual void     removeGraph(const synGraphHandle graph)                                                    = 0;

    virtual const Graph& getGraph(const synGraphHandle graph) const            = 0;
    virtual void         addNode(const synGraphHandle graph, const Node& node) = 0;

    virtual void
    setBlockingNodes(const synGraphHandle graph, const synNodeId nodeId, const std::vector<synNodeId>& blocking) = 0;

    virtual void setDeterministic(const synGraphHandle graph, const synNodeId nodeId, const bool useDeterministic) = 0;

    virtual void
    setRoundingMode(const synGraphHandle graph, const synNodeId nodeId, const synRoundingMode roundingMode) = 0;

    virtual void setParams(const synGraphHandle graphHandle,
                           const synNodeId      nodeId,
                           const void*          userParams,
                           const unsigned       paramsSize) = 0;

    /*DEPRECATED*/
    virtual void setGraphAttributes(synGraphHandle           graphHandle,
                                    const synGraphAttribute* attributes,
                                    const uint64_t*          values,
                                    const uint32_t           size) = 0;

    virtual void setGraphAttributesV2(synGraphHandle              graphHandle,
                                      const synGraphAttribute*    attributes,
                                      const synGraphAttributeVal* values,
                                      const uint32_t              size) = 0;

    virtual void mapRecipeToGraphJson(synRecipeHandle* pRecipeHandle, const std::string& uniqueId) = 0;
    virtual void recordGraph(const synRecipeHandle recipeHandle, uint16_t recipeId)                = 0;
    virtual void postCompilationUpdate(uint32_t graphIndex, std::optional<uint16_t> recipeId)      = 0;
};
}  // namespace graph_serialize