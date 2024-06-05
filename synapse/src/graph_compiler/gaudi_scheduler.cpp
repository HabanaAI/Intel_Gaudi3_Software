#include "gaudi_scheduler.h"
#include "defs.h"
#include "habana_graph.h"
#include "graph_editor.h"
#include "node.h"
#include "node_annotation.h"
#include "strided_insert_node.h"
#include "handle_memory_reuse.h"
#include "gaudi_max_path_scheduler.h"
#include "types.h"
#include <stack>
#include "brain_conf.h"

namespace gaudi
{
    struct BundleScheduleComparator
    {
        bool operator()(const BundleInfo& b1, const BundleInfo& b2) const
        {
            if ((b1.bundleEngine == BundleEngine::ENGINE_MME) && (b2.bundleEngine != BundleEngine::ENGINE_MME))
            {
                return true;
            }
            if ((b1.bundleEngine != BundleEngine::ENGINE_MME) && (b2.bundleEngine == BundleEngine::ENGINE_MME))
            {
                return false;
            }
            return b1.bundleIndex < b2.bundleIndex;
        }
    };  // BundleScheduleComparator

    class GaudiSchedulerFreeNodes : public FreeNodesContainer
    {
        struct InternalDependencyData
        {
            NodeVector blockedNodes;
            unsigned   numOfBlockingNodes = 0;
        };

    public:
        GaudiSchedulerFreeNodes() : FreeNodesContainer(NodeScheduleComparator()) {}

        virtual void    insert(const NodePtr& n) override;
        virtual void    erase(const NodePtr& n) override;
        virtual NodePtr getNext() override;

    private:
        std::unordered_map<NodePtr, InternalDependencyData> m_internalDependencyData;
        const NodeScheduleComparator                        m_comparator = NodeScheduleComparator();
    };

    // Return the first nodes that not blocked
    NodePtr GaudiSchedulerFreeNodes::getNext()
    {
        if (empty()) return nullptr;
        auto next = std::find_if(m_freeNodes.begin(), m_freeNodes.end(), [&](const NodePtr& n) {
            return m_internalDependencyData[n].numOfBlockingNodes == 0;
        });
        HB_ASSERT(next != m_freeNodes.end(), "Can't find free node");
        return *next;
    }

    // If two logical nodes are "tied" (NodeScheduleComparator::smaller(a,b) == TIE),
    // and they have the same real output tensor - schedule the node with the lower offset first.
    // This will enable better pipelining between the producer and the consumers of the nodes.
    void GaudiSchedulerFreeNodes::insert(const NodePtr& n)
    {
        m_internalDependencyData[n] = InternalDependencyData();
        if (n->isLogicalOperation())
        {
            for (const auto& n2 : m_freeNodes)
            {
                if (!n2->isLogicalOperation() || m_comparator.compareBundlePriority(n, n2).has_value()) continue;
                if (MemoryReuseHandler::hasLowerWritingOffset(n, n2))
                {
                    m_internalDependencyData[n].blockedNodes.push_back(n2);
                    ++m_internalDependencyData[n2].numOfBlockingNodes;
                }
                else if (MemoryReuseHandler::hasLowerWritingOffset(n2, n))
                {
                    m_internalDependencyData[n2].blockedNodes.push_back(n);
                    ++m_internalDependencyData[n].numOfBlockingNodes;
                }
            }
        }
        m_freeNodes.insert(n);
    }

    // Remove node and release the "LowerWritingOffset" relations with the blocked nodes
    void GaudiSchedulerFreeNodes::erase(const NodePtr& n)
    {
        auto        it             = m_internalDependencyData.find(n);
        const auto& additionalData = it->second;
        HB_ASSERT(additionalData.numOfBlockingNodes == 0,
                  "Try to erase node with {} blocking nodes",
                  additionalData.numOfBlockingNodes);
        for (const auto& blockedNode : additionalData.blockedNodes)
        {
            HB_ASSERT(m_internalDependencyData[blockedNode].numOfBlockingNodes != 0,
                      "Unexpected blocking/blocking relation");
            --m_internalDependencyData[blockedNode].numOfBlockingNodes;
        }
        m_internalDependencyData.erase(it);
        m_freeNodes.erase(n);
    }

} // namespace gaudi

void GaudiScheduler::generateExecutionSchedule(NodeList& schedule) const
{
    const NodeSet& allNodes = m_graph->getNodes();
    using BundledNodeSet    = gaudi::GaudiSchedulerFreeNodes;

    //Otherwise do topological sort using DFS.
    //The free nodes will be inserted according to their depth since we use a stack.
    LOG_DEBUG(SCHEDULER, "Generating execution schedule for {} nodes", allNodes.size());
    BundledNodeSet freeNodes;
    std::map<BundleInfo, BundledNodeSet, gaudi::BundleScheduleComparator>     bundleFreeNodes;
    std::map<BundleIdx, uint32_t>          bundleInDegrees;  // Number of bundle dependencies
    Settable<BundleInfo>                   currentBundle;
    std::map<NodePtr, int, NodeComparator> inDegrees;
    for (const NodePtr& node : allNodes)
    {
        unsigned degree = 0;
        for (const NodePtr& producer : getBlockingNodes(node))
        {
            if (producer == nullptr) continue;
            ++degree;
            if (node->getNodeAnnotation().bundleInfo.is_set() &&
                (!producer->getNodeAnnotation().bundleInfo.is_set() ||
                    producer->getNodeAnnotation().bundleInfo->bundleIndex !=
                    node->getNodeAnnotation().bundleInfo->bundleIndex))
            {
                // input to bundled node from outside the bundle
                bundleInDegrees[node->getNodeAnnotation().bundleInfo->bundleIndex]++;
            }
        }

        inDegrees[node] = degree;
        if (degree == 0)
        {
            if (node->getNodeAnnotation().bundleInfo.is_set())
            {
                bundleFreeNodes[node->getNodeAnnotation().bundleInfo.value()].insert(node);
            }
            else
            {
                freeNodes.insert(node);
            }
        }
    }
    HB_ASSERT(inDegrees.size() == allNodes.size(),
                "{} Unreachable nodes in graph",
                allNodes.size() - inDegrees.size());

    while (true)
    {
        // If the last node inserted belong to a bundle, continue with all bundle insertion
        if (!currentBundle.is_set())
        {
            // should work on bundle but no current active bundle is found
            for (auto& IdxbundleSet : bundleFreeNodes)
            {
                if (!IdxbundleSet.second.empty() && (bundleInDegrees[IdxbundleSet.first.bundleIndex] == 0))
                {
                    //only set if the current bundle index is NOT scalar pipe!
                    currentBundle.set(IdxbundleSet.first);
                    auto firstNodeInBundle = bundleFreeNodes.at(currentBundle.value()).getNext();
                    //if current bundle is scalar pipe - skip it
                    if (firstNodeInBundle->getNodeAnnotation().bundleInfo->bundleType == BundleType::SCALAR_PIPE && !freeNodes.empty())
                    {
                        currentBundle.unset();
                    }
                    else
                    {
                        LOG_DEBUG(SCHEDULER, "Starting to schedule bundle {}", currentBundle.value().bundleIndex);
                    }
                    break;
                }
            }
        }

        if (freeNodes.empty() && !currentBundle.is_set())
        {
            if (bundleFreeNodes.empty())
            {
                break; // Nothing left to do
            }
            else
            {
                if (!bundleFreeNodes.empty())
                {
                    currentBundle.set(bundleFreeNodes.begin()->first);
                    LOG_WARN(SCHEDULER,
                            "Cyclic bundle dependency detected, bundle {} will not be grouped in execution schedule",
                            currentBundle.value().bundleIndex);
                }
            }
        }
        auto&   workingSet = currentBundle.is_set() ? bundleFreeNodes.at(currentBundle.value())
                                                    : freeNodes;
        NodePtr node       = workingSet.getNext();
        workingSet.erase(node);

        schedule.push_back(node);
        LOG_DEBUG(SCHEDULER, "Scheduling {}", node->getNodeName());

        for (const NodePtr& child : getBlockedNodes(node))
        {
            int newRefCount = --inDegrees[child];
            if (child->getNodeAnnotation().bundleInfo.is_set() &&
                     (!node->getNodeAnnotation().bundleInfo.is_set() ||
                      node->getNodeAnnotation().bundleInfo->bundleIndex !=
                          child->getNodeAnnotation().bundleInfo->bundleIndex))
            {
                // Scheduled input to a bundled node from outside the bundle
                bundleInDegrees[child->getNodeAnnotation().bundleInfo->bundleIndex]--;
            }
            HB_ASSERT(newRefCount >= 0, "Node has more paths reaching it than its in-degree");
            if (newRefCount == 0)
            {
                LOG_TRACE(SCHEDULER, "Inserting {} into free list", child->getNodeName());
                if (child->getNodeAnnotation().bundleInfo.is_set())
                {
                    bundleFreeNodes[child->getNodeAnnotation().bundleInfo.value()].insert(child);
                }
                else
                {
                    freeNodes.insert(child);
                }
            }
        }
        if (currentBundle.is_set() && bundleFreeNodes.at(currentBundle.value()).empty())
        {
            LOG_TRACE(SCHEDULER, "Bundle {} scheduling done.", currentBundle.value().bundleIndex);
            bundleFreeNodes.erase(currentBundle.value());
            currentBundle.unset();
        }
    }
    for (const auto& it : inDegrees)
    {
        HB_ASSERT(it.second == 0, "Unreachable node {} in graph", it.first->getNodeName());
    }

    // If we insert probe nodes, we want them executed immediately after the op they are probing
    if (GCFG_ENABLE_NAN_INF_PROBE.value() == true)
    {
        auto                                   probeNodes = 0;
        std::map<NodePtr, NodeList ::iterator> nodeToProbeConsumerPlace;
        for (auto it = schedule.begin(); it != schedule.end();)
        {
            for (const auto& consumer : m_graph->getNodeConsumers(*it))
            {
                if (consumer->getGUID().find("probe_nan") != std::string::npos)
                {
                    nodeToProbeConsumerPlace.emplace(consumer, it);
                }
            }

            if (nodeToProbeConsumerPlace.count(*it))
            {
                auto tmp_it = it++;
                ++probeNodes;
                schedule.splice(std::next(nodeToProbeConsumerPlace.at(*tmp_it)), schedule, tmp_it);
            }
            else
            {
                ++it;
            }
        }
        HB_ASSERT(probeNodes == nodeToProbeConsumerPlace.size(), "Error scheduling proneNodes");
    }
    HB_ASSERT(schedule.size() == allNodes.size(),
                "{} Unreachable nodes in graph",
                schedule.size() - allNodes.size());
}

// Get the first bundle node, if the memset is outside the given node bundle, or the same node if inapplicable
NodePtr GaudiScheduler::replaceBlockedNodeWithFirstInBundle(const NodePtr& blocked, const NodePtr& memsetNode) const
{
    const auto& memsetBundleInfo  = memsetNode->getNodeAnnotation().bundleInfo;
    const auto& blockedBundleInfo = blocked->getNodeAnnotation().bundleInfo;
    if (blockedBundleInfo.is_set() &&
        (!memsetBundleInfo.is_set() || memsetBundleInfo->bundleIndex != blockedBundleInfo->bundleIndex))
    {
        // blocked is bundled, and memset is outside this bundle (in diff bundle or not bundled)
        auto it = m_bundleIdxToFirstNode.find(blockedBundleInfo->bundleIndex);
        HB_ASSERT(it != m_bundleIdxToFirstNode.end(), "{} bundle not found!", __func__);
        return it->second;
    }
    else
    {
        return blocked;
    }
}

void GaudiScheduler::fixupMemsetBundleOpIndex(const NodePtr& memset, const NodePtr& reductionProducer)
{
    auto&       memsetBundleInfo   = memset->getNodeAnnotation().bundleInfo;
    const auto& producerBundleInfo = reductionProducer->getNodeAnnotation().bundleInfo;
    if (!memsetBundleInfo.is_set() || !producerBundleInfo.is_set()) return;
    if (memsetBundleInfo->bundleIndex != producerBundleInfo->bundleIndex) return;
    memsetBundleInfo->operationIndex = producerBundleInfo->operationIndex;
}

//  memsets can be scheduled anywhere before their first dependent node,
//  but we want to move them as late as possible to avoid tensor long liveness.
//  the main algorithm is:
//  1) go over the schedule in order:
//    1.1) when we find a memset node we save its position in a map from each dependent node to memset(s) position(s)
//    1.2) when we find a node that exists in the above map, we save its position,
//         and remove the dependent nodes of the memset(s) from the map,
//         this guarantees that any time we find a node in the map it is the first dependent node
//  2) move all memsets to their new location in reverse order
//     to support cases that memset depends on another memset
void GaudiScheduler::optimizeMemsetsLocation(NodeList& schedule) const
{
    // we save map from node to list of memsets locations in the scheduling
    std::map<NodePtr, std::list<NodeList::iterator>>               dependenciesMap;
    std::vector<std::pair<NodeList::iterator, NodeList::iterator>> positions;

    // #1
    for (NodeList::iterator pos = schedule.begin(); pos != schedule.end(); ++pos)
    {
        const NodePtr& node = *pos;
        // #1.1
        if (node->isMemset() && shouldOptimizeMemset(node))
        {
            for (const NodePtr& dependantNode : getBlockedNodes(node))
            {
                const NodePtr blocked = replaceBlockedNodeWithFirstInBundle(dependantNode, node);
                auto          it      = dependenciesMap.find(blocked);
                if (it == dependenciesMap.end())
                {
                    dependenciesMap.emplace(blocked, std::list<NodeList::iterator>({pos}));
                }
                else
                {
                    it->second.push_back(pos);
                }
            }
        }
        // #1.2
        auto it = dependenciesMap.find(node);
        if (it != dependenciesMap.end())
        {
            for (const NodeList::iterator& memsetPos : it->second)
            {
                const auto& memsetNode = *memsetPos;
                LOG_DEBUG(SCHEDULER,
                          "optimize {} location: inserting before {}",
                          memsetNode->getNodeName(),
                          node->getNodeName());
                for (const NodePtr& dependedNode : getBlockedNodes(*memsetPos))
                {
                    // the node will handle later to avoid iterator invalidate
                    const NodePtr blocked = replaceBlockedNodeWithFirstInBundle(dependedNode, *memsetPos);
                    if (node == blocked) continue;
                    auto nodeMemsets = dependenciesMap.find(blocked);
                    if (nodeMemsets == dependenciesMap.end())
                    {  // memset was already removed from dependency map
                        LOG_TRACE(SCHEDULER,
                                  "{} not exists in dependencies map of {}",
                                  blocked->getNodeName(),
                                  memsetNode->getNodeName());
                        continue;
                    }
                    if (nodeMemsets->second.size() == 1)
                    {
                        dependenciesMap.erase(nodeMemsets);
                    }
                    else
                    {
                        nodeMemsets->second.remove(memsetPos);  // remove by value
                    }
                }
                positions.emplace_back(pos, memsetPos);
            }
            dependenciesMap.erase(it);
        }
    }
    // #2
    for (auto it = positions.rbegin(); it != positions.rend(); ++it)  // reverse order
    {
        schedule.splice(it->first, schedule, it->second);
    }
}

bool GaudiScheduler::shouldOptimizeMemset(const NodePtr& memset) const
{
    // handle unbundled memsets, or bundled memsets if the LB scheduler doesn't handle them
    return !memset->getNodeAnnotation().bundleInfo.is_set() ||
           !GCFG_ENABLE_LAYERED_BRAIN_SCHEDULER_OPTIMIZE_MEMSET.value();
}

// Create a map from bundle-id to first node in bundle.
std::map<GaudiScheduler::BundleIdx, NodePtr> GaudiScheduler::findFirstNodePerBundle(const Graph& g)
{
    std::map<BundleIdx, NodePtr> bundleIdxToFirstNode;
    for (const NodePtr& node : g.getNodes())
    {
        const auto& bundleInfo = node->getNodeAnnotation().bundleInfo;
        if (bundleInfo.is_set())
        {
            BundleIdx   bundleIdx = bundleInfo->bundleIndex;
            const auto& it        = bundleIdxToFirstNode.find(bundleIdx);
            if ((it == bundleIdxToFirstNode.end()) ||
                (bundleInfo->operationIndex < it->second->getNodeAnnotation().bundleInfo->operationIndex))
            {
                bundleIdxToFirstNode[bundleIdx] = node;
            }
        }
    }
    return bundleIdxToFirstNode;
}

void GaudiScheduler::createFirstNodePerBundleMapping()
{
    m_bundleIdxToFirstNode = findFirstNodePerBundle(*m_graph);
}

// modified "getNodeProducers" that include the implicit memset-reduction dependencies
NodeSet GaudiScheduler::getBlockingNodes(const NodePtr& node) const
{
    return m_graph->getNodeProducers(node, Node::TENSOR_TYPE_ALL);
}

// modified "getNodeConsumers" that include the implicit memset-reduction dependencies
NodeSet GaudiScheduler::getBlockedNodes(const NodePtr& node) const
{
    return m_graph->getNodeConsumers(node, Node::TENSOR_TYPE_ALL);
}

bool GaudiScheduler::isValidGraphForMaxPath() const
{
    for (const NodePtr& n : m_graph->getNodes())
    {
        if (!n) continue;
        bool isMME          = HabanaGraph::runsOnMME(n);
        bool isPartOfBundle = n->getNodeAnnotation().bundleInfo.is_set();
        // TODO [SW-104068] - Fix condition for multibuffer check
        // Max-Path scheduling can extend tensors' liveness, which can cause multibuffered tensors to not fit
        // the given buffering level
        const TensorVector& operands = n->getOperands();
        bool areTensorsMultibuffered = std::any_of(operands.begin(), operands.end(), [](const TensorPtr& t) {
            return t && t->getTensorAnnotation().nonPersistentSectionInfo.sectionId.is_set();
        });
        if (isMME || isPartOfBundle || areTensorsMultibuffered) return false;
    }
    return true;
}

NodeList GaudiScheduler::scheduleNodes()
{
    NodeList schedule;

    // Create a map from bundle-id to first node in bundle (according to operation idx)
    createFirstNodePerBundleMapping();

    if (GCFG_ENABLE_MAX_PATH_SCHEDULE.value() && isValidGraphForMaxPath())
    {
        GaudiMaxPathScheduler::ConnectivityFunc getBlocking = [&](const NodePtr& n) { return getBlockingNodes(n); };
        GaudiMaxPathScheduler::ConnectivityFunc getBlocked  = [&](const NodePtr& n) { return getBlockedNodes(n); };
        schedule = GaudiMaxPathScheduler(m_graph, getBlocking, getBlocked).scheduleNodes();
    }
    else
    {
        generateExecutionSchedule(schedule);
    }
    optimizeMemsetsLocation(schedule);
    return schedule;
}