#pragma once

#include "bfs_scheduler.h"

using MaxPathMap            = std::unordered_map<NodePtr, unsigned>;
using MaxPathTieBreakerFunc = std::function<bool(const NodePtr& a, const NodePtr& b)>;

class MaxPathScheduler : public BfsScheduler
{
public:
    MaxPathScheduler(const Graph* graph, const MaxPathTieBreakerFunc& comp) : BfsScheduler(graph), m_comp(comp) {}

    NodeList scheduleNodes() override;

    MaxPathMap createMaxPathMap(bool useRealConnectingTensorAsWeights = false) const;

protected:
    const MaxPathTieBreakerFunc m_comp;

private:
    unsigned getRealConnectingTensorSumOfWeights(const NodePtr& producer, const NodePtr& consumer) const;
    void     createMaxPathEdgeWeightIsRealTensors(MaxPathMap& maxPath, const NodeList& topoSortedNodes) const;
    void     createMaxPathEdgeWeightIsOne(MaxPathMap& maxPath, const NodeList& topoSortedNodes) const;
};

class MaxPathNodeComparator
{
public:
    MaxPathNodeComparator(MaxPathMap maxPath, const MaxPathTieBreakerFunc& comp)
    : m_maxPath(std::move(maxPath)), m_tieBreakComp(comp)
    {
    }

    std::optional<bool> compareMaxPath(const NodePtr& n1, const NodePtr& n2) const;
    virtual bool operator()(const NodePtr& n1, const NodePtr& n2) const;

private:
    const MaxPathMap            m_maxPath;
    const MaxPathTieBreakerFunc m_tieBreakComp;
};

class MaxPathFreeNodesContainer : public FreeNodesContainer
{
public:
    MaxPathFreeNodesContainer(const MaxPathNodeComparator& cmp);

protected:
    const MaxPathNodeComparator& m_maxPathComparator;
};
