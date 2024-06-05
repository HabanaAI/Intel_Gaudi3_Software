#include <utility>

#pragma once

#include "scheduler.h"


class DfsNodeScheduleComparator : public NodeScheduleComparator
{
public:
    explicit DfsNodeScheduleComparator(NodePtr node, const std::map<NodePtr, int>& maxPath) : m_node(std::move(node)),
                                                                                              m_maxPath(maxPath)
    {};

    bool operator()(const NodePtr& n1, const NodePtr& n2) const override;

private:
    NodePtr m_node;
    const std::map<NodePtr, int>& m_maxPath;
};

class MemoryOrientedScheduler : public Scheduler
{
public:

    explicit MemoryOrientedScheduler(const HabanaGraph* graph);

    NodeList scheduleNodes() override;

protected:
    NodeList dfsSchedule();

private:
    NodeList dfsScheduleFromRoot(const NodePtr& root, int rootCount, NodeSet& visitedConsumers);
    void pushForwardDramSpills(NodeList& schedule);
    void postponeDramFills(NodeList& schedule);
    bool allConsumersVisited(const NodePtr& n);
    void gatherLogicalInputs(NodeList& schedule, int rootCount);

    void verifySchedule(const NodeList& schedule) const;

    static const int NEVER_VISITED = -1;
    std::map<NodePtr, int> m_whenVisited;  // when was this node visited (at what root)
    std::map<NodePtr, NodeSet> m_barrierConsumers;    // treat barriers like consumers
    std::map<NodePtr, int> m_maxPathLengthFromNode;   // the length of longest path from node to any output
};
