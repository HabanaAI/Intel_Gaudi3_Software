#pragma once

// eager includes (relative to src/eager/lib/)
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node.h"

// std includes
#include <variant>

class Node;

namespace gaudi2
{
class SyncSchemeFwContext;
}

namespace gaudi3
{
class SyncSchemeFwContext;
}

namespace eager_mode
{
using SyncSchemeFwContextPtrVariant =
    std::variant<std::monostate, gaudi2::SyncSchemeFwContext*, gaudi3::SyncSchemeFwContext*>;

class DescGeneratorHal
{
public:
    constexpr static unsigned LOGICAL_QUEUE_MAX_ID = 5;

    virtual ~DescGeneratorHal() = default;  // Designate destructor to be virtual
    virtual unsigned deviceTypeToLogicalQueue(EngineType engineType, const Node& node) const = 0;
    virtual unsigned safeIncrement(unsigned logicalId, unsigned sigVal, unsigned incAmount) const = 0;
};

}  // namespace eager_mode