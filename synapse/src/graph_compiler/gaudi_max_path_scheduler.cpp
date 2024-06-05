#include "gaudi_max_path_scheduler.h"
#include "passes/handle_memory_reuse.h"

bool GaudiMaxPathFreeNodesContainer::shouldAddInternalDependencyRead(const NodePtr& n1, const NodePtr& n2) const
{
    if (m_maxPathComparator.compareMaxPath(n1, n2) || m_maxPathComparator.compareMaxPath(n2, n1)) return false;
    // If both nodes are using the same real input tensor - schedule the node with the lower offset first.
    // for better pipelining
    if (m_writeDepNodes.find(n1) != m_writeDepNodes.end()) return false;
    if (m_writeDepNodes.find(n2) != m_writeDepNodes.end()) return false;
    return MemoryReuseHandler::hasLowerReadingOffset(n1, n2);
}

bool GaudiMaxPathFreeNodesContainer::shouldAddInternalDependencyWrite(const NodePtr& n1, const NodePtr& n2) const
{
    if (m_maxPathComparator.compareMaxPath(n1, n2) || m_maxPathComparator.compareMaxPath(n2, n1)) return false;
    // If both nodes are using the same real output tensor - schedule the node with the lower offset first.
    // for better pipelining
    if (m_readDepNodes.find(n1) != m_readDepNodes.end()) return false;
    if (m_readDepNodes.find(n2) != m_readDepNodes.end()) return false;
    return MemoryReuseHandler::hasLowerWritingOffset(n1, n2);
}

void GaudiMaxPathFreeNodesContainer::addInternalDependency(const NodePtr& blocking, const NodePtr& blocked)
{
    bool shouldAddReadDep  = shouldAddInternalDependencyRead(blocking, blocked);
    bool shouldAddWriteDep = shouldAddInternalDependencyWrite(blocking, blocked);
    if (!shouldAddReadDep && !shouldAddWriteDep) return;

    (shouldAddReadDep ? m_readDepNodes : m_writeDepNodes).insert(blocking);
    (shouldAddReadDep ? m_readDepNodes : m_writeDepNodes).insert(blocked);

    // add dependency between 'other' and 'node'
    m_internalNodeDependencies[blocking].insert(blocked);
    if (++m_internalInDegrees[blocked] == 1)  // increase the inDegree of 'node'
    {
        m_freeNodes.erase(blocked);  // if newly blocked, make it unavailable
    }
}

void GaudiMaxPathFreeNodesContainer::removeInternalDependencies(const NodePtr& blockingNode)
{
    const auto it = m_internalNodeDependencies.find(blockingNode);
    if (it != m_internalNodeDependencies.end())
    {
        for (const NodePtr& blocked : it->second)  // all nodes that were blocked by 'blockingNode'
        {
            if (--m_internalInDegrees.at(blocked) == 0)  // if not blocked anymore, make it read to be scheduled again
            {
                m_freeNodes.insert(blocked);
            }
        }
        m_internalNodeDependencies.erase(it);
    }
}

/* whenever a 'new' free node is inserted, go over all available free nodes and block any 'other' node that needs to be
   scheduled after the 'new' free node */
void GaudiMaxPathFreeNodesContainer::insert(const NodePtr& node)
{
    m_internalInDegrees[node] = 0;

    // check if 'node' should be blocked from entering
    for (auto it : m_internalInDegrees)
    {
        const NodePtr& other = it.first;
        {
            addInternalDependency(other, node);
            addInternalDependency(node, other);
        }
    }

    if (m_internalInDegrees.at(node) == 0)  // if not blocked by anyone, make it available for selection
    {
        m_freeNodes.insert(node);
    }
}

/* whenever a free node is scheduled, go over all 'blocked' free nodes, and un-block them. if there are no more
   blockers, make them schedulable again */
void GaudiMaxPathFreeNodesContainer::erase(const NodePtr& n)
{
    bool res = m_freeNodes.erase(n);
    HB_ASSERT(res, "free node {} not found!", n->getNodeName());
    res = m_internalInDegrees.erase(n);
    HB_ASSERT(res, "internal in degree of free node {} not found!", n->getNodeName());
    m_readDepNodes.erase(n);
    m_writeDepNodes.erase(n);
    removeInternalDependencies(n);  // remove all internal dependencies of erased node
}

NodeList GaudiMaxPathScheduler::scheduleNodes()
{
    MaxPathNodeComparator          maxPathCompare(createMaxPathMap(), FreeNodesContainer::defaultCompare);
    GaudiMaxPathFreeNodesContainer freeNodes(maxPathCompare);
    NodeList                       ret = MaxPathScheduler::getTopoSortedNodes(freeNodes);
    return ret;
}

NodeList GaudiDfsScheduler::scheduleNodes()
{
    FreeNodesContainer freeNodes(m_comp);
    return getTopoSortedNodes(freeNodes);
}

// A recursive function used by topologicalSort
void GaudiDfsScheduler::scheduleWithDfsFromFreeNodesUtil(const NodePtr&                     currNode,
                                                         std::unordered_map<NodePtr, bool>& visitedMap,
                                                         std::stack<NodePtr>&               stack) const
{
    // Mark the current node as visited
    HB_ASSERT(visitedMap.find(currNode) != visitedMap.end(),
              "Expecting to find {} in the visited map",
              currNode->getNodeName());
    visitedMap.at(currNode) = true;

    // Recur for all the vertices adjacent to this vertex
    for (const auto& consumer : m_graph->getNodeConsumers(currNode, Node::TENSOR_TYPE_ALL))
    {
        HB_ASSERT(visitedMap.find(consumer) != visitedMap.end(),
                  "Expecting to find {} in the visited map",
                  consumer->getNodeName());
        if (!visitedMap.at(consumer))
        {
            scheduleWithDfsFromFreeNodesUtil(consumer, visitedMap, stack);
        }
    }

    // Push current vertex to stack which stores result
    stack.push(currNode);
}

void GaudiDfsScheduler::scheduleWithDfsFromFreeNodes(FreeNodesContainer&                freeNodes,
                                                     std::unordered_map<NodePtr, bool>& visitedMap,
                                                     NodeList&                          ret) const
{
    LOG_TRACE(SCHEDULER, "{}", HLLOG_FUNC);

    // Execute DFS, use the freeNodes as roots
    std::stack<NodePtr> dfsStack;
    while (!freeNodes.empty())
    {
        const NodePtr& nextNode = freeNodes.getNext();
        LOG_TRACE(SCHEDULER, "Start dfs from: {}", nextNode->getNodeName());
        scheduleWithDfsFromFreeNodesUtil(nextNode, visitedMap, dfsStack);
        freeNodes.erase(nextNode);
    }

    while (!dfsStack.empty())
    {
        LOG_DEBUG(SCHEDULER, "Schedule {}", dfsStack.top()->getNodeName());
        ret.push_back(dfsStack.top());
        dfsStack.pop();
    }
}

void GaudiDfsScheduler::initializeFreeNodesAndInDegree(FreeNodesContainer& freeNodes) const
{
    const NodeSet& allNodes = m_graph->getNodes();

    for (const NodePtr& node : allNodes)
    {
        if (!node) continue;
        int degree = static_cast<int>(m_graph->getNodeProducers(node, Node::TENSOR_TYPE_ALL).size());
        if (degree == 0)
        {
            freeNodes.insert(node);
        }
    }
}

NodeList GaudiDfsScheduler::getTopoSortedNodes(FreeNodesContainer& freeNodes) const
{
    // Initialize data structures
    NodeList ret;
    initializeFreeNodesAndInDegree(freeNodes);
    std::unordered_map<NodePtr, bool> visitedMap;
    for (const auto& node : m_graph->getNodes())
    {
        visitedMap.emplace(node, false);
    }

    scheduleWithDfsFromFreeNodes(freeNodes, visitedMap, ret);

    HB_ASSERT(ret.size() == m_graph->getNodes().size(), "Didn't schedule all nodes...");
    return ret;
}