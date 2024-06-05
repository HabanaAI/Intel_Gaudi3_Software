#include "serializer.h"

#include "habana_global_conf.h"
#include "post_graph_serializer.h"
#include "pre_graph_serializer.h"
#include "types_exception.h"
#include "types.h"  //TODO

#include <memory>

namespace graph_serializer
{
std::shared_ptr<Serializer> create(Serializers serializer)
{
    switch (serializer)
    {
        case Serializers::PRE_GRAPH:
            return std::make_shared<PreSerializer>();
        case Serializers::POST_GRAPH:
            return std::make_shared<PostSerializer>();
        case Serializers::SERIALIZERS_MAX:
            SynapseStatusException(fmt::format("{} Invalid Serializer type: {}", __FUNCTION__, toString(serializer)));
            break;
    }
    return nullptr;
}

std::string toString(Serializers s)
{
    switch (s)
    {
        case Serializers::PRE_GRAPH:
            return "PRE_GRAPH";
        case Serializers::POST_GRAPH:
            return "POST_GRAPH";
        case Serializers::SERIALIZERS_MAX:
            return "SERIALIZERS_MAX";
    }
    return "unknown";
}

std::string toString(GraphState s)
{
    switch (s)
    {
        case GraphState::PRE_COMPILE:
            return "pre";
        case GraphState::POST_COMPILE:
            return "post";
        case GraphState::POST_PASS:
            return "passes";
        case GraphState::GRAPH_STATE_MAX:
            return "invalid_state";
    }
    return "unknown";
}

std::string_view getGraphStatePath(GraphState s)
{
    switch (s)
    {
        case GraphState::PRE_COMPILE:
            return GCFG_DUMP_PRE_GRAPHS.value();
        case GraphState::POST_COMPILE:
            return GCFG_DUMP_POST_GRAPHS.value();
        case GraphState::POST_PASS:
            return GCFG_DUMP_PASSES_GRAPHS.value();
        case GraphState::GRAPH_STATE_MAX:
            return "";
    }
    return "";
}
}  // namespace graph_serializer