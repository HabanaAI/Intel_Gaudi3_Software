#pragma once

#include "graph_compiler/node_cost_model.h"

namespace gaudi2
{
class NodeCostModel : public ::NodeCostModel
{
public:
    NodeCostModel(HabanaGraph& g) : ::NodeCostModel(g) {}
    std::optional<std::pair<EngineType, double>> getNodeExpectedDuration(const NodePtr& node) override;
};
}  // namespace gaudi2