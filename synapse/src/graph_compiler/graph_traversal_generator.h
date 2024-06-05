#pragma once

#include "habana_graph.h"

/**
    generator object that allows traversing over a graph in topological order using a tie breaker for choosing the next
    free node.
    Supports both forward and reverse topological order.
*/
class GraphTraversalGenerator
{
public:
    using GraphTraversalComparator = std::function<bool(const NodePtr& a, const NodePtr& b)>;

    // initialize object using a strict weak ordering comparator.
    GraphTraversalGenerator(const HabanaGraph&       g,
                            bool                     reverse = false,
                            GraphTraversalComparator comp    = NodeComparator());
    // get next node in topological order
    NodePtr getNext();
    // check if no more nodes are available in topological order
    bool empty() const;

private:
    void validateDone() const;

    using ConnectivityMap = std::unordered_map<NodePtr, NodeSet>;
    ConnectivityMap m_blockedNodes;

    const HabanaGraph&          m_g;
    std::map<NodePtr, unsigned> m_inDegrees;
    NodeList                    m_freeNodes;
    GraphTraversalComparator    m_comp;
};