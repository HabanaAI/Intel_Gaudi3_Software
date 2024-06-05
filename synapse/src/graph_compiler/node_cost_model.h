#pragma once

#include "types.h"

#include <optional>
#include <utility>

class HabanaGraph;

class NodeCostModel
{
public:
    enum class EngineType
    {
        TPC,
        MME,
        DMA,
        // more can be added here

        // must be last:
        INVALID,
        ENGINES_NR = INVALID  // specifies total number of engine types in this enum
    };

    NodeCostModel(HabanaGraph& g) : m_graph(g) {}
    virtual ~NodeCostModel()                                                                          = default;
    virtual std::optional<std::pair<EngineType, double>> getNodeExpectedDuration(const NodePtr& node) = 0;

protected:
    HabanaGraph& m_graph;
};
