#pragma once

#include "signal_out_from_graph.h"

class HabanaGraph;

class Gaudi2SignalOutFromGraph : public SignalOutFromGraph
{
public:
    Gaudi2SignalOutFromGraph() : SignalOutFromGraph() {}

private:
    void init(HabanaGraph& g) override;
    void setInitialSyncValues(HabanaGraph& g) override;
    void addSigOutGroupMonitors(HabanaGraph& g, const NodeSet& producers, const TensorPtr& t, int index) override;
};
