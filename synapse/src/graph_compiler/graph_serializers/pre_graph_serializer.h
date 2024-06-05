#pragma once

#include "serializer_base.h"

namespace graph_serializer
{
class PreSerializer : public SerializerBase
{
public:
    uint32_t    version() const override;
    std::string name() const override;
    bool        supports(GraphState state) const override;

protected:
    NodeVector       getNodes(const HabanaGraph& graph) const override;
    json_utils::Json serializeTensor(const TensorPtr& t) const override;
    json_utils::Json serializeNode(const HabanaGraph& graph, const NodePtr& node, uint32_t graphIndex) override;
};
}  // namespace graph_serializer