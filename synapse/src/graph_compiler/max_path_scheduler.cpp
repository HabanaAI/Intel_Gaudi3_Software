#include "max_path_scheduler.h"
#include "habana_graph.h"

std::optional<bool> MaxPathNodeComparator::compareMaxPath(const NodePtr& n1, const NodePtr& n2) const
{
    if (n1 == nullptr) return false;
    if (n2 == nullptr) return true;

    // logical nodes first, so that all their consumers will be considered.
    bool isLogical1 = (n1->isLogicalOperation() && !n1->isDebug());
    bool isLogical2 = (n2->isLogicalOperation() && !n2->isDebug());
    if (isLogical1 && !isLogical2) return true;
    if (!isLogical1 && isLogical2) return false;

    // maximum path length from graph exit. this will make the schedule "zigzag".
    int n1MaxPath = m_maxPath.at(n1);
    int n2MaxPath = m_maxPath.at(n2);
    if (n1MaxPath > n2MaxPath) return true;
    if (n1MaxPath < n2MaxPath) return false;

    return std::nullopt;
}

bool MaxPathNodeComparator::operator()(const NodePtr& n1, const NodePtr& n2) const
{
    return compareMaxPath(n1, n2).value_or(m_tieBreakComp(n1, n2));
}

MaxPathFreeNodesContainer::MaxPathFreeNodesContainer(const MaxPathNodeComparator& cmp)
: FreeNodesContainer(cmp), m_maxPathComparator(cmp)
{
}

unsigned MaxPathScheduler::getRealConnectingTensorSumOfWeights(const NodePtr& producer, const NodePtr& consumer) const
{
    if (producer->isLogicalOperation()) return 0;
    unsigned realConnectingTensorSumOfWeights = 0;
    for (const auto& output : producer->getOutputs())
    {
        for (const auto& outputConsumer : m_graph->getTensorConsumers(output))
        {
            // check if the producer output's consumer equals to consumer argument.
            if (outputConsumer == consumer)
            {
                realConnectingTensorSumOfWeights += output->getTotalSizeInBytes();
            }
        }
    }
    return realConnectingTensorSumOfWeights;
}

void MaxPathScheduler::createMaxPathEdgeWeightIsRealTensors(MaxPathMap& maxPath, const NodeList& topoSortedNodes) const
{
    for (auto it = topoSortedNodes.rbegin(); it != topoSortedNodes.rend(); it++)
    {
        const NodePtr& n = *it;
        maxPath[n]       = 0;
        for (const NodePtr& consumer : getBlockedNodes(n))
        {
            maxPath[n] = std::max(maxPath[n], maxPath[consumer] + getRealConnectingTensorSumOfWeights(n, consumer));
        }
    }
}

void MaxPathScheduler::createMaxPathEdgeWeightIsOne(MaxPathMap& maxPath, const NodeList& topoSortedNodes) const
{
    for (auto it = topoSortedNodes.rbegin(); it != topoSortedNodes.rend(); it++)
    {
        const NodePtr& n = *it;
        maxPath[n]       = 1;
        for (const NodePtr& consumer : getBlockedNodes(n))
        {
            maxPath[n] = std::max(maxPath[n], maxPath[consumer] + 1);
        }
        if ((n)->isLogicalOperation() && !(n)->isDebug())
        {
            maxPath[n]--;
        }
    }
}

MaxPathMap MaxPathScheduler::createMaxPathMap(bool useRealConnectingTensorAsWeights) const
{
    // create max path - the maximum path in graph from node to exit.
    MaxPathMap         maxPath;
    FreeNodesContainer freeNodes;
    NodeList           preOrder = BfsScheduler::getTopoSortedNodes(freeNodes);
    if (useRealConnectingTensorAsWeights)
    {
        LOG_DEBUG(SCHEDULER, "{}: calculate max paths such that edge weight is sum of real tensors", __FUNCTION__);
        createMaxPathEdgeWeightIsRealTensors(maxPath, preOrder);
    }
    else
    {
        LOG_DEBUG(SCHEDULER, "{}: calculate max paths such that edge weight is one", __FUNCTION__);
        createMaxPathEdgeWeightIsOne(maxPath, preOrder);
    }
    return maxPath;
}

NodeList MaxPathScheduler::scheduleNodes()
{
    MaxPathNodeComparator     maxPathCompare(createMaxPathMap(), m_comp);
    MaxPathFreeNodesContainer freeNodes(maxPathCompare);
    NodeList                  ret = BfsScheduler::getTopoSortedNodes(freeNodes);
    return ret;
}
