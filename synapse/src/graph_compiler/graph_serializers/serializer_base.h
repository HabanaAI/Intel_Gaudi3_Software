#pragma once

#include "serializer.h"
#include "json_utils.h"

namespace graph_serializer
{
class SerializerBase : public Serializer
{
public:
    void serialize(const HabanaGraph& graph, const fs::path& file, const std::string& name, bool append) override;

protected:
    virtual NodeVector       getNodes(const HabanaGraph& graph) const                                          = 0;
    virtual json_utils::Json serializeTensor(const TensorPtr& t) const                                         = 0;
    virtual json_utils::Json serializeNode(const HabanaGraph& graph, const NodePtr& node, uint32_t graphIndex) = 0;

    virtual json_utils::Json serializeGraph(const HabanaGraph& graph, const std::string& graphName, uint32_t index);
};
}  // namespace graph_serializer