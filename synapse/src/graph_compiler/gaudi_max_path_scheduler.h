#pragma once

#include "log_manager.h"
#include "max_path_scheduler.h"
#include <stack>

class GaudiMaxPathScheduler : public MaxPathScheduler
{
public:
    using ConnectivityFunc = std::function<NodeSet(const NodePtr& n)>;

    GaudiMaxPathScheduler(const Graph*      graph,
                          const ConnectivityFunc& getBlockingNodes,
                          const ConnectivityFunc& getBlockedNodes)
    : MaxPathScheduler(graph, FreeNodesContainer::defaultCompare),
      m_getBlockingNodes(getBlockingNodes),
      m_getBlockedNodes(getBlockedNodes)
    {
    }

    NodeList scheduleNodes() override;

protected:
    NodeSet getBlockingNodes(const NodePtr& n) const override { return m_getBlockingNodes(n); }
    NodeSet getBlockedNodes(const NodePtr& n) const override { return m_getBlockedNodes(n); }

private:
    ConnectivityFunc m_getBlockingNodes;
    ConnectivityFunc m_getBlockedNodes;
};

class GaudiMaxPathFreeNodesContainer : public MaxPathFreeNodesContainer
{
public:
    explicit GaudiMaxPathFreeNodesContainer(const MaxPathNodeComparator& cmp) : MaxPathFreeNodesContainer(cmp) {}

    virtual void insert(const NodePtr& n) override;
    virtual void erase(const NodePtr& n) override;

protected:
    // given n1 and n2 are both free to run, this function returns true if n1 should block n2
    bool shouldAddInternalDependencyRead(const NodePtr& n1, const NodePtr& n2) const;
    bool shouldAddInternalDependencyWrite(const NodePtr& n1, const NodePtr& n2) const;
    void addInternalDependency(const NodePtr& blocking, const NodePtr& blocked);
    void removeInternalDependencies(const NodePtr& blockingNode);

    // holds a mapping between free nodes, and the nodes that are blocked by them
    std::unordered_map<NodePtr, NodeSet> m_internalNodeDependencies;
    // holds all blocked nodes, and the amount of free nodes that are blocking them
    std::unordered_map<NodePtr, int> m_internalInDegrees;
    // holds all the free nodes with extra dependencies, and the operand type (input/output) that caused it.
    // we will not allow a single node to have 2 different dependencies with different operands,
    // otherwise, we might get a cycle (see [SW-179003])
    std::unordered_set<NodePtr> m_readDepNodes;
    std::unordered_set<NodePtr> m_writeDepNodes;
};

class GaudiDfsScheduler : public Scheduler
{
public:
    GaudiDfsScheduler(const Graph* graph, const FreeNodesContainer::ScheduleComparator& comp = FreeNodesContainer::defaultCompare)
    : Scheduler(graph), m_comp(comp)
    {
    }

    NodeList scheduleNodes() override;

private:
    NodeList getTopoSortedNodes(FreeNodesContainer& freeNodes) const;
    void     initializeFreeNodesAndInDegree(FreeNodesContainer&               freeNodes) const;
    void     scheduleWithDfsFromFreeNodes(FreeNodesContainer&                freeNodes,
                                          std::unordered_map<NodePtr, bool>& visitedMap,
                                          NodeList&                          ret) const;
    void     scheduleWithDfsFromFreeNodesUtil(const NodePtr&                     currNode,
                                              std::unordered_map<NodePtr, bool>& visitedMap,
                                              std::stack<NodePtr>&               stack) const;

    const FreeNodesContainer::ScheduleComparator& m_comp;
};
