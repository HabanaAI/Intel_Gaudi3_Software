#include "serializer_base.h"
#include "common_type_utils.h"
#include "data_type_utils.h"
#include "habana_graph.h"
#include "serialize_utils.h"
#include "compilation_hal_reader.h"
#include "types.h"
#include <cstdint>
#include <string>

using namespace graph_serializer;
using Json = nlohmann_hcl::json;

Json SerializerBase::serializeGraph(const HabanaGraph& graph, const std::string& graphName, uint32_t index)
{
    const auto& sortedNodes  = getNodes(graph);
    const auto& graphTensors = graph.getTensors();

    CompilationHalReaderSetter compHalReaderSetter(&graph);

    std::vector<Json> serializedNodes;
    serializedNodes.reserve(sortedNodes.size());
    for (const auto& n : sortedNodes)
    {
        serializedNodes.emplace_back(serializeNode(graph, n, index));
    }

    std::vector<Json> serializedTensors;
    serializedTensors.reserve(graphTensors.size());
    for (const auto& t : graphTensors)
    {
        serializedTensors.emplace_back(serializeTensor(t));
    }

    // sorting the tensors by name can make file comparison simpler
    std::sort(serializedTensors.begin(), serializedTensors.end(), [](const Json& a, const Json& b) {
        return a.at("name") < b.at("name");
    });

    Json graphData;
    graphData["id"]      = index;
    graphData["nodes"]   = serializedNodes;
    graphData["tensors"] = serializedTensors;
    graphData["name"]    = graphName;

    graph.dumpTpcNodesDataToJson(index);

    return graphData;
}

void SerializerBase::serialize(const HabanaGraph& graph, const fs::path& filePath, const std::string& name, bool append)
{
    const std::string modelName = fs::path(filePath).filename().replace_extension();

    Json model            = append ? json_utils::jsonFromFile(filePath) : Json();
    auto serializedGraphs = model["graphs"];
    Json serializedGraph  = serializeGraph(graph, name, serializedGraphs.size());

    serializedGraphs.push_back(serializedGraph);
    model["graphs"]  = serializedGraphs;
    model["name"]    = modelName;
    model["version"] = version();

    json_utils::jsonToFile(model, filePath);
}