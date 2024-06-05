#pragma once

#include "signal_out_from_graph.h"

class HabanaGraph;

class Gaudi3SignalOutFromGraph : public SignalOutFromGraph
{
public:
    Gaudi3SignalOutFromGraph() : SignalOutFromGraph() {}
    bool executePass(HabanaGraph& g) override;
};
