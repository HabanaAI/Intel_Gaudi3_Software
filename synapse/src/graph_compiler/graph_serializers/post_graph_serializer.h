#pragma once

#include "serializer_base.h"
#include <unordered_map>

namespace graph_serializer
{
class PostSerializer : public SerializerBase
{
public:
    uint32_t    version() const override;
    std::string name() const override;
    bool        supports(GraphState state) const override;

protected:
    NodeVector       getNodes(const HabanaGraph& graph) const override;
    json_utils::Json serializeTensor(const TensorPtr& t) const override;
    json_utils::Json serializeTensorBaseInfo(const TensorPtr& t) const;
    json_utils::Json serializeNodeBaseInfo(const NodePtr& node) const;
    json_utils::Json serializeNode(const HabanaGraph& graph, const NodePtr& node, uint32_t graphIndex) override;
    json_utils::Json serializeGraph(const HabanaGraph& graph, const std::string& graphName, uint32_t index) override;
    json_utils::Json serializeFusedGraph(const HabanaGraph& graph, const std::string& fusedNode, NodeList nodes);

private:
    unsigned                                     m_numBPs = 0;
    std::unordered_map<std::string, std::string> m_tensorsIO;
};
}  // namespace graph_serializer
