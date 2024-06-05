#pragma once

#include "habana_graph.h"

//A debug implementation of graph, running on the CPU

class SimGraph : public HabanaGraph
{
public:
    SimGraph();
    SimGraph(const SimGraph& other);
    SimGraph& operator=(const SimGraph& other);
    ~SimGraph();

    bool                       compile() override;
    bool execute() override;
    virtual NodeVector         getTopoSortedNodes() const { return Graph::getTopoSortedNodes(); }

    virtual HabanaGraphPtr clone(bool cloneAllocators = false, bool keepMappings = false) const override;

    virtual bool     isPersistentTensor(const pTensor& tensor) const override { return false; }
    virtual bool     isUserManagedDram(const pTensor& tensor) const override { return false; }

    virtual synDeviceType    getDeviceType() const override { return synDeviceGaudi; }

    HabanaGraphPtr createEmptyGraph() const override;

};
