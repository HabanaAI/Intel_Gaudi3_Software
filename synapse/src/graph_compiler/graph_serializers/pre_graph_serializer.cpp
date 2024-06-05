#include "pre_graph_serializer.h"
#include "common_type_utils.h"
#include "graph_serializers/serialize_utils.h"
#include "data_type_utils.h"
#include "graph_serializers/serializer.h"
#include "habana_graph.h"

using namespace graph_serializer;
using Json = nlohmann_hcl::json;

uint32_t PreSerializer::version() const
{
    return 2;
}

std::string PreSerializer::name() const
{
    return "";
}

bool PreSerializer::supports(GraphState state) const
{
    return state == GraphState::PRE_COMPILE;
}

NodeVector PreSerializer::getNodes(const HabanaGraph& graph) const
{
    const auto& nodes = graph.getNodes();
    return NodeVector(nodes.begin(), nodes.end());
}

Json PreSerializer::serializeTensor(const TensorPtr& t) const
{
    return ::serializeTensor(t.get());
}

Json PreSerializer::serializeNode(const HabanaGraph& graph, const NodePtr& node, uint32_t graphIndex)
{
    Json ret;
    ret["name"]        = node->getNodeName();
    ret["guid"]        = node->getGUID();
    ret["graph_index"] = graphIndex;

    ret["params"]         = node->getParamsRawData();
    ret["input_tensors"]  = getTensorsNames(node->getInputs(), false);
    ret["output_tensors"] = getTensorsNames(node->getOutputs(), false);
    ret["blocking_nodes"] = getNodeNames(graph.getBlockingNodes(node));

    return ret;
}