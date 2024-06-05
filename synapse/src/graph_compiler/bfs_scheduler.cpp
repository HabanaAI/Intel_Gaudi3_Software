#include "bfs_scheduler.h"
#include "habana_graph.h"


bool FreeNodesContainer::defaultCompare(const NodePtr& n1, const NodePtr& n2)
{
    if (GCFG_ENABLE_PARENT_ID_SCHEDULE.value())
    {
        if (n1->getParentId() > n2->getParentId()) return false;
        if (n1->getParentId() < n2->getParentId()) return true;
    }

    return n1->getId() < n2->getId();
}

NodeSet BfsScheduler::getBlockingNodes(const NodePtr& n) const
{
    return m_graph->getNodeProducers(n);
}

NodeSet BfsScheduler::getBlockedNodes(const NodePtr& n) const
{
    return m_graph->getNodeConsumers(n);
}

NodeList BfsScheduler::scheduleNodes()
{
    FreeNodesContainer freeNodes;
    return getTopoSortedNodes(freeNodes);
}

void BfsScheduler::initializeFreeNodesAndInDegree(FreeNodesContainer&               freeNodes,
                                                  std::unordered_map<NodePtr, int>& inDegrees) const
{
    const NodeSet& allNodes = m_graph->getNodes();

    for (const NodePtr& node : allNodes)
    {
        if (!node) continue;
        int degree      = static_cast<int>(getBlockingNodes(node).size());
        inDegrees[node] = degree;
        if (degree == 0)
        {
            freeNodes.insert(node);
        }
    }
}

NodeList BfsScheduler::getTopoSortedNodes(FreeNodesContainer& freeNodes) const
{
    // simple BFS schedule
    NodeList                         ret;
    std::unordered_map<NodePtr, int> inDegrees;
    initializeFreeNodesAndInDegree(freeNodes, inDegrees);

    while (!freeNodes.empty())
    {
        NodePtr nextNode = freeNodes.getNext();
        freeNodes.erase(nextNode);

        ret.push_back(nextNode);
        for (const NodePtr& consumer : getBlockedNodes(nextNode))
        {
            if (--inDegrees[consumer] == 0)
            {
                freeNodes.insert(consumer);
            }
        }
    }

    for (const auto& inDeg : inDegrees)
    {
        if (inDeg.second != 0)
        {
            HB_ASSERT(inDeg.second == 0, "not finished deg for {}", inDeg.first->getNodeName());
        }
    }
    HB_ASSERT(ret.size() == m_graph->getNodes().size(), "didn't schedule all nodes...");

    return ret;
}