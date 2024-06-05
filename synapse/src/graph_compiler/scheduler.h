#pragma once

#include "types.h"
#include "graph.h"

class NodeScheduleComparator
{
public:
    virtual bool operator()(const NodePtr& n1, const NodePtr& n2) const;
};

class Scheduler
{
public:
    explicit Scheduler(const Graph* graph) : m_graph(graph) {}

    virtual NodeList scheduleNodes();

    virtual ~Scheduler() = default;

protected:
    bool areNodeProducersFree(const pNode& node, SortableNodeMap<int>& inDegrees);

    NodeList scheduleNodesDefault();

    void handleProducersDMANodes(NodeList& schedule, pNode& node,
                                 std::set<pNode, NodeScheduleComparator>& freeDMANodes,
                                 NodeSet& usedDMANodes,
                                 SortableNodeMap<int>& inDegrees,
                                 SortableNodeMap<NodeSet>& barrierList);

    void updateConsumerDegree(const pNode& child,
                              SortableNodeMap<int>& inDegrees,
                              std::set<pNode, NodeScheduleComparator>& freeNodes,
                              std::set<pNode, NodeScheduleComparator>& freeDMANodes);

    const Graph* m_graph;
};
