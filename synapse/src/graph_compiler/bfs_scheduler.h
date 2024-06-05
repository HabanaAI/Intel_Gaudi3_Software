#pragma once

#include "scheduler.h"
#include "node.h"

class FreeNodesContainer
{
public:
    using ScheduleComparator = std::function<bool(const NodePtr& a, const NodePtr& b)>;
    using FreeNodesSet       = std::set<NodePtr, ScheduleComparator>;

    explicit FreeNodesContainer(const ScheduleComparator& cmp) : m_freeNodes(cmp) {}
    FreeNodesContainer() : FreeNodesContainer(defaultCompare) {}
    virtual ~FreeNodesContainer() = default;

    virtual void                 insert(const NodePtr& n) { m_freeNodes.insert(n); }
    virtual void                 erase(const NodePtr& n) { m_freeNodes.erase(n); };
    virtual NodePtr              getNext() { return *m_freeNodes.begin(); };
    virtual bool                 empty() const { return m_freeNodes.empty(); };
    FreeNodesSet::const_iterator cbegin() const { return m_freeNodes.cbegin(); }
    FreeNodesSet::const_iterator cend() const { return m_freeNodes.cend(); }
    static bool                  defaultCompare(const NodePtr& n1, const NodePtr& n2);

protected:
    FreeNodesSet m_freeNodes;
};

class BfsScheduler : public Scheduler
{
public:
    explicit BfsScheduler(const Graph* graph) : Scheduler(graph) {}
    NodeList scheduleNodes() override;

protected:
    virtual NodeList getTopoSortedNodes(FreeNodesContainer& freeNodes) const;
    virtual NodeSet  getBlockingNodes(const NodePtr& n) const;
    virtual NodeSet  getBlockedNodes(const NodePtr& n) const;
    void             initializeFreeNodesAndInDegree(FreeNodesContainer&               freeNodes,
                                                    std::unordered_map<NodePtr, int>& inDegrees) const;
};
