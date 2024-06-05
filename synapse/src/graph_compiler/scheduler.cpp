#include "scheduler.h"
#include "habana_graph.h"

NodeList Scheduler::scheduleNodes()
{
    return scheduleNodesDefault();
}

bool NodeScheduleComparator::operator()(const NodePtr& n1, const NodePtr& n2) const
{
    //ensure everything comes before null
    if (n1 == nullptr) return false;
    if (n2 == nullptr) return true;

    //Ranges precede all
    if (n1->getNodeAnnotation().rangeIndex < n2->getNodeAnnotation().rangeIndex) return true;
    if (n2->getNodeAnnotation().rangeIndex < n1->getNodeAnnotation().rangeIndex) return false;

    //Slices come next
    if (n1->getNodeAnnotation().sliceIndex < n2->getNodeAnnotation().sliceIndex) return true;
    if (n2->getNodeAnnotation().sliceIndex < n1->getNodeAnnotation().sliceIndex) return false;

    //sort constants and DMA nodes before other nodes
    unsigned n1Inputs = n1->getNumInputs();
    unsigned n2Inputs = n2->getNumInputs();
    if ((n1Inputs == 0) && (n2Inputs > 0)) return true;
    if ((n1Inputs > 0) && (n2Inputs == 0)) return false;

    //sort DMA up nodes before other nodes
    unsigned n1Outputs = n1->getNumOutputs();
    unsigned n2Outputs = n2->getNumOutputs();
    if ((n1Outputs == 0) && (n2Outputs > 0)) return true;
    if ((n1Outputs > 0) && (n2Outputs == 0)) return false;

    //Prefer other nodes over memcpy
    //Todo: this needs to be dictated differently, by the optimizer, this is a static priority mechanism
    bool n1Memcpy = isMemcpy(*n1);
    bool n2Memcpy = isMemcpy(*n2);
    if (!n1Memcpy && n2Memcpy) return true;
    if (n1Memcpy && !n2Memcpy) return false;

    //fall-through path, just break ties by order of creation for consistency between compilations
    return n1->getId() < n2->getId();
}

NodeList Scheduler::scheduleNodesDefault()
{
    NodeList schedule;

    const NodeSet& allNodes = m_graph->getNodes();
    std::set<NodePtr, NodeScheduleComparator> freeNodes;
    NodeSet usedDMANodes;
    std::set<NodePtr, NodeScheduleComparator> freeDMANodes;
    SortableNodeMap<int> inDegrees;
    SortableNodeMap<NodeSet> barrierList;

    LOG_TRACE(SCHEDULER, "Generating execution schedule for {} nodes", allNodes.size());

    for (const NodePtr& node : allNodes)
    {
        unsigned degree = 0;
        for (const NodePtr& producerNode : m_graph->getNodeProducers(node))
        {
            if (producerNode == nullptr) continue;

            // DMA nodes should not affect the execution order
            // DMA nodes will be inserted right before their tensors are used
            if (HabanaGraph::isNonActivationDMA(producerNode)) continue;

            ++degree;
        }
        const std::list<NodePtr>& barriers = node->getNodeAnnotation().memorySpaceInfo.barriers;
        for (const NodePtr& singleBarrier : barriers)
        {
            ++degree;
            barrierList[singleBarrier].insert(node);
        }

        inDegrees[node] = degree;
    }
    HB_ASSERT(inDegrees.size() == allNodes.size(), "Unreachable node in graph");

    for (const auto& it : inDegrees)
    {
        if (it.second == 0 && areNodeProducersFree(it.first, inDegrees))
        {
            // DMA down nodes should not be used as a root nodes of the graph
            if (!HabanaGraph::isNonActivationDMA(it.first) || it.first->getNumInputs() != 0)
            {
                LOG_TRACE(SCHEDULER, "Node {} is a root node", it.first->getNodeName());
                freeNodes.insert(it.first);
            }
            else
            {
                freeDMANodes.insert(it.first);
            }
        }
    }

    while (!freeNodes.empty())
    {
        auto it = freeNodes.begin();
        NodePtr node = *it;
        // Handling the case in which the current node depends on tensor given by a DMA.
        // Inserting the DMA right before it and updating the barriers
        handleProducersDMANodes(schedule, node, freeDMANodes, usedDMANodes, inDegrees, barrierList);

        schedule.push_back(node);
        freeNodes.erase(it);
        LOG_TRACE(SCHEDULER, "Scheduling {}", node->getNodeName());

        for (const NodePtr& child : m_graph->getNodeConsumers(node))
        {
            updateConsumerDegree(child, inDegrees, freeNodes, freeDMANodes);
        }
        for (const NodePtr& child : barrierList[node])
        {
            updateConsumerDegree(child, inDegrees, freeNodes, freeDMANodes);
        }
    }

    HB_ASSERT(freeDMANodes.empty(), "Some DMA nodes were not scheduled");

    for (const auto& it : inDegrees)
    {
        HB_ASSERT(it.second == 0, "Unreachable node in graph: {}", it.first->getNodeName());
    }
    HB_ASSERT(schedule.size() == allNodes.size(), "Unreachable node in graph");
    return schedule;
}

bool Scheduler::areNodeProducersFree(const NodePtr& node, SortableNodeMap<int>& inDegrees)
{
    for (const NodePtr& producerNode : m_graph->getNodeProducers(node))
    {
        if (producerNode == nullptr) continue;
        if (inDegrees[producerNode] != 0)
        {
            return false;
        }
    }
    return true;
}

void Scheduler::updateConsumerDegree(const NodePtr& child,
                                     SortableNodeMap<int>& inDegrees,
                                     std::set<NodePtr, NodeScheduleComparator>& freeNodes,
                                     std::set<NodePtr, NodeScheduleComparator>& freeDMANodes)
{
    int newRefCount = --inDegrees[child];
    HB_ASSERT(newRefCount >= 0, "Node has more paths reaching it than its in-degree");
    if (newRefCount == 0 && areNodeProducersFree(child, inDegrees))
    {
        if (!HabanaGraph::isNonActivationDMA(child) || child->getNumInputs() != 0)
        {
            LOG_TRACE(SCHEDULER, "Inserting {} into free list", child->getNodeName());
            freeNodes.insert(child);
        }
        else
        {
            LOG_TRACE(SCHEDULER, "Inserting {} into free DMA list", child->getNodeName());
            freeDMANodes.insert(child);

            for (const NodePtr& childConsumer: m_graph->getNodeConsumers(child))
            {
                if (inDegrees[childConsumer] == 0 && areNodeProducersFree(childConsumer, inDegrees))
                {
                    LOG_TRACE(SCHEDULER, "Inserting {} into free list", childConsumer->getNodeName());
                    freeNodes.insert(childConsumer);
                }
            }
        }
    }
}

void Scheduler::handleProducersDMANodes(NodeList& schedule,
                                        NodePtr& node,
                                        std::set<NodePtr, NodeScheduleComparator>& freeDMANodes,
                                        NodeSet& usedDMANodes,
                                        SortableNodeMap<int>& inDegrees,
                                        SortableNodeMap<NodeSet>& barrierList)
{
    for (const NodePtr& producerNode : m_graph->getNodeProducers(node))
    {
        if (producerNode == nullptr || !HabanaGraph::isNonActivationDMA(producerNode)) continue;
        if (usedDMANodes.find(producerNode) != usedDMANodes.end())
        {
            LOG_TRACE(SCHEDULER, "Already scheduled DMA node {}", producerNode->getNodeName());
            continue;
        }
        auto producer_it = freeDMANodes.find(producerNode);
        HB_ASSERT(producer_it != freeDMANodes.end(), "DMA node should be available");
        usedDMANodes.insert(producerNode);
        schedule.push_back(producerNode);
        freeDMANodes.erase(producer_it);
        LOG_TRACE(SCHEDULER, "Scheduling {}", producerNode->getNodeName());

        // Update barriered DMA nodes
        for (const NodePtr& child : barrierList[producerNode])
        {
            HB_ASSERT(HabanaGraph::isNonActivationDMA(child) && child->getNumInputs() == 0,
                      "DMA can be a barrier only for a DMA down node");
            updateConsumerDegree(child, inDegrees, freeDMANodes, freeDMANodes);
        }
    }
}