#pragma once

#include "filesystem.h"
#include "types.h"
#include <memory>

class HabanaGraph;

namespace graph_serializer
{
enum class Serializers
{
    PRE_GRAPH = 0,
    POST_GRAPH,
    SERIALIZERS_MAX
};

enum class GraphState
{
    PRE_COMPILE = 0,
    POST_COMPILE,
    POST_PASS,
    GRAPH_STATE_MAX
};

std::string toString(Serializers s);

std::string toString(GraphState s);

std::string_view getGraphStatePath(GraphState s);

class Serializer
{
public:
    virtual ~Serializer() = default;

    virtual uint32_t version() const = 0;

    virtual std::string name() const = 0;

    virtual bool supports(GraphState state) const = 0;

    virtual void serialize(const HabanaGraph& graph, const fs::path& file, const std::string& name, bool append) = 0;
};

std::shared_ptr<Serializer> create(Serializers serializer);

}  // namespace graph_serializer