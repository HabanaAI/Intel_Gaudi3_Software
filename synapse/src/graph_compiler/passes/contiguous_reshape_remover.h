#pragma once

#include "habana_graph.h"

class ContiguousReshapeRemover
{
public:
    explicit ContiguousReshapeRemover(HabanaGraph& graph) : m_graph(graph) {}
    virtual ~ContiguousReshapeRemover() = default;
    //Remove all reshapes of producers of a reshape node.
    void removeContiguousReshapesForNode(pNode node);
    // Remove all contiguous reshapes for the member graph.
    void removeContiguousReshapesForGraph();

    bool fuseProducerAndConsumerReshape(pNode producer, pNode consumer);

private:
    HabanaGraph& m_graph;
};
