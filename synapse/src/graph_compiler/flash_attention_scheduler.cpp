#include "gaudi_max_path_scheduler.h"
#include "bundle_plane_graph.h"
#include "flash_attention_scheduler.h"

std::optional<bool> FlashAttentionComparator::compareFlashAttentionChains(const NodePtr& n1, const NodePtr& n2) const
{
    if (n1 == nullptr) return false;
    if (n2 == nullptr) return true;

    // Prioritize flash attention chainId with lower value if exists
    if (n1->getNodeAnnotation().flashAttentionInfo.has_value() && n2->getNodeAnnotation().flashAttentionInfo.has_value())
    {
        const auto& chainNumN1 = n1->getNodeAnnotation().flashAttentionInfo->chainInfo;
        const auto& chainNumN2 = n2->getNodeAnnotation().flashAttentionInfo->chainInfo;
        if (chainNumN1.has_value() && !chainNumN2.has_value()) return true;
        if (!chainNumN1.has_value() && chainNumN2.has_value()) return false;
        if (chainNumN1.has_value() && chainNumN2.has_value())
        {
            if (chainNumN1.value() > chainNumN2.value()) return true;
            if (chainNumN1.value() < chainNumN2.value()) return false;
        }
    }

    return std::nullopt;
}

bool FlashAttentionComparator::operator()(const NodePtr& n1, const NodePtr& n2) const
{
    return compareFlashAttentionChains(n1,n2).value_or(FreeNodesContainer::defaultCompare(n1, n2));
}

void FlashAttentionScheduler::initFlashAttentionInfo(HabanaGraph& g, const NodePtr& node)
{
    // Skip nodes that already have FA info and nodes that are not FA
    auto parentId = node->getParentId();
    if (node->getNodeAnnotation().flashAttentionInfo.has_value() || !g.getGraphAnnotation().flashAttentionDb.isRegistered(parentId)) return;
    // Set original node id as parent id
    node->getNodeAnnotation().flashAttentionInfo = FlashAttentionInfo(parentId);

    // Set chain info
    const auto& nodeName       = node->getNodeName();
    std::size_t innerGuidStart = nodeName.rfind("sdpa");
    const auto  corePos        = nodeName.find("core", innerGuidStart);
    if (corePos != std::string::npos)
    {
        const size_t chainNumEnd = nodeName.find('/', innerGuidStart);
        if (chainNumEnd != std::string::npos)
        {
            std::size_t chainNumStart     = nodeName.find_last_of('_', chainNumEnd);
            unsigned    chainNumStrLength = chainNumEnd - chainNumStart - 1;
            std::string chainNumStr       = nodeName.substr(chainNumStart + 1, chainNumStrLength);
            unsigned    chainId           = std::stoi(chainNumStr);
            node->getNodeAnnotation().flashAttentionInfo->chainInfo = chainId;
            LOG_TRACE(FLASH_ATTENTION, "{}: {} is in chain {}", __func__, node->getNodeName(), chainId);

            // Add chainId to the known chainId's set for this parentId
            g.getGraphAnnotation().flashAttentionDb.registerChainForNode(parentId, chainId);
        }
    }
}

void FlashAttentionScheduler::updateChainInfo(const NodePtr& node, unsigned chainId)
{
    HB_ASSERT(node->getNodeAnnotation().flashAttentionInfo.has_value(),
              "FlashAttentionChainInfo is a trait of FlashAttention nodes only! {} not a FA node",
              node->getNodeName());
    HB_ASSERT(!node->getNodeAnnotation().flashAttentionInfo->chainInfo.has_value(),
              "ChainInfo shouldn't be modified once its set: {} -> {} not allowed",
              node->getNodeAnnotation().flashAttentionInfo->chainInfo.value(),
              chainId);
    node->getNodeAnnotation().flashAttentionInfo->chainInfo = chainId;
}

std::optional<unsigned> FlashAttentionScheduler::getChainInfo(const NodePtr& node) const
{
    if (node->getNodeAnnotation().flashAttentionInfo.has_value() &&
        node->getNodeAnnotation().flashAttentionInfo->chainInfo.has_value())
    {
        return node->getNodeAnnotation().flashAttentionInfo->chainInfo.value();
    }
    return {};
}

std::optional<unsigned> FlashAttentionScheduler::getChainIdFromConnectedNodes(const NodePtr& node,
                                                                              const NodeSet& connectedNodes) const
{
    if (!connectedNodes.empty())
    {
        std::set<unsigned> nodesChainIds;
        for (const NodePtr& connectedNode : connectedNodes)
        {
            // Skip consumers that are outside of node's FA
            if (connectedNode->getParentId() != node->getParentId()) continue;

            // Collect the chainIds of all the consumers
            if (getChainInfo(connectedNode).has_value())
            {
                nodesChainIds.insert(getChainInfo(connectedNode).value());
            }
        }
        if (nodesChainIds.size() == 1)
        {
            return *nodesChainIds.begin();
        }
    }
    return std::nullopt;
}

void FlashAttentionScheduler::setChainIdBwd(const NodeVector& topoSortedNodes)
{
    // Iterate the topological sort in reverse order to set chain id from consumers
    for (auto it = topoSortedNodes.rbegin(); it != topoSortedNodes.rend(); it++)
    {
        const auto& node     = *it;
        if (node->getNodeAnnotation().flashAttentionInfo.has_value() && !getChainInfo(node).has_value())
        {
            const auto& consumers = m_graph->getNodeConsumers(node, Node::TENSOR_TYPE_ALL);
            std::optional<unsigned> consumerChainId = getChainIdFromConnectedNodes(node, consumers);
            if (consumerChainId.has_value())
            {
                LOG_TRACE(FLASH_ATTENTION, "{}: Adding {} to chain {}", __FUNCTION__, node->getNodeName(), consumerChainId.value());
                updateChainInfo(node, consumerChainId.value());
            }
        }
    }
}

void FlashAttentionScheduler::setChainIdFwd(const NodeVector& topoSortedNodes)
{
    // Iterate the topological sort in regular/chronological order to set chain id from producers
    for (auto it = topoSortedNodes.begin(); it != topoSortedNodes.end(); it++)
    {
        const auto& node     = *it;
        if (node->getNodeAnnotation().flashAttentionInfo.has_value() && !getChainInfo(node).has_value())
        {
            const auto& producers = m_graph->getNodeProducers(node, Node::TENSOR_TYPE_ALL);
            std::optional<unsigned> producerChainId = getChainIdFromConnectedNodes(node, producers);
            if (producerChainId.has_value())
            {
                LOG_TRACE(FLASH_ATTENTION, "{}: Adding {} to chain {}", __FUNCTION__, node->getNodeName(), producerChainId.value());
                updateChainInfo(node, producerChainId.value());
            }
        }
    }
}

void FlashAttentionScheduler::setChainIdIfPossible()
{
    const NodeVector& topoSortedNodes = m_graph->getTopoSortedNodes();
    setChainIdBwd(topoSortedNodes);
    setChainIdFwd(topoSortedNodes);
}

bool FlashAttentionScheduler::trySetAsSink(const NodePtr& node)
{
    const auto& consumers = m_graph->getNodeConsumers(node);
    bool isSink = std::all_of(consumers.begin(), consumers.end(), [&node](const NodePtr& consumer){
        return consumer->getParentId() != node->getParentId();
    });
    if (isSink)
    {
        m_faToSourcesAndSinks[node->getParentId()].sinks.insert(node);
        LOG_TRACE(FLASH_ATTENTION, "Setting {} as sink of FA id {}", node->getNodeName(), node->getParentId());
        return true;
    }
    return false;
}

bool FlashAttentionScheduler::trySetAsSource(const NodePtr& node)
{
    const auto& producers = m_graph->getNodeProducers(node);
    bool isSource = std::all_of(producers.begin(), producers.end(), [&node](const NodePtr& producer){
        return producer->getParentId() != node->getParentId();
    });
    if (isSource)
    {
        m_faToSourcesAndSinks[node->getParentId()].sources.insert(node);
        LOG_TRACE(FLASH_ATTENTION, "Setting {} as source of FA id {}", node->getNodeName(), node->getParentId());
    }
    return isSource;
}

void FlashAttentionScheduler::setBlockingFaMap(const NodePtr& node)
{
    for (const auto& producer : m_graph->getNodeProducers(node))
    {
        // Add node FA as blocked by producer's FA
        if (producer->getNodeAnnotation().flashAttentionInfo.has_value() && (producer->getParentId() != node->getParentId()))
        {
            m_faToBlockedFas[producer->getParentId()].insert(node->getParentId());
            LOG_TRACE(FLASH_ATTENTION, "Marking FA id {} as blocking of FA id {}", producer->getParentId(), node->getParentId());
        }
    }
}

void FlashAttentionScheduler::initFlashAttentionSubgraphMetadata(const NodePtr& node)
{
    if (!trySetAsSource(node))
    {
        trySetAsSink(node);
    }
    setBlockingFaMap(node);
}

void FlashAttentionScheduler::setCrossFaDependencies()
{
    // Add control edges between different original flash attention nodes to prevent mixed schedule between them
    for (const auto& [blockingId, blockedIds] : m_faToBlockedFas)
    {
        for (const auto& blockedId : blockedIds)
        {
            // Sources and sinks of BLOCKING flashAttn
            const auto& blockingSinks = m_faToSourcesAndSinks[blockingId].sinks;
            // Sources and sinks of BLOCKED flashAttn
            const auto& blockedSources = m_faToSourcesAndSinks[blockedId].sources;
            // Add a control edge between each sink of the blocking FA to each source of the blocked FA
            for (const auto& blockingSink : blockingSinks)
            {
                for (const auto& blockedSource : blockedSources)
                {
                    m_graph->addControlDependency(blockingSink, blockedSource, Tensor::ControlEdgeType::SCHEDULE);
                    LOG_TRACE(FLASH_ATTENTION, "Added schedule ctrl edge between {} and {}", blockingSink->getNodeName(), blockedSource->getNodeName());
                }
            }
        }
    }
}

void tracePrint(unsigned id, const NodeSet& nodes)
{
    if (LOG_LEVEL_AT_LEAST_TRACE(FLASH_ATTENTION))
    {
        LOG_TRACE(FLASH_ATTENTION, "Flash attention node id {} contains {} nodes:", id, nodes.size());
        for (const auto& n : nodes)
        {
            LOG_TRACE(FLASH_ATTENTION, "\t{}", n->getNodeName());
        }
    }
}

void FlashAttentionScheduler::createSchedule()
{
    const NodeVector&    topoSortedNodes  = m_graph->getTopoSortedNodes();
    FlashAttnIdToNodeMap flashAttnIdToNodes;

    // Init flash attention info for all nodes
    for (auto it = topoSortedNodes.begin(); it != topoSortedNodes.end(); it++)
    {
        const auto& node     = *it;
        unsigned    parentId = node->getParentId();

        // Skip nodes that don't belong to any flash attention/don't need GC scheduling
        if (!m_graph->getGraphAnnotation().flashAttentionDb.isRegistered(parentId)) continue;

        flashAttnIdToNodes[parentId].insert(node);
        initFlashAttentionInfo(*m_graph, node);
        initFlashAttentionSubgraphMetadata(node);
    }

    setChainIdIfPossible();

    // Schedule each FlashAttention subgraph
    LOG_DEBUG(FLASH_ATTENTION, "Creating schedule for {} FlashAttention nodes", flashAttnIdToNodes.size());
    for (const auto& flashAttnIdAndNodes : flashAttnIdToNodes)
    {
        tracePrint(flashAttnIdAndNodes.first, flashAttnIdAndNodes.second);

        m_graph->constructBPGraph(true /*useAnnotation*/, [&flashAttnIdAndNodes, this](const NodePtr& n) {
            return n->getParentId() == flashAttnIdAndNodes.first && getChainInfo(n).has_value();
        });
        const auto& bpgSchedule =
            GaudiDfsScheduler(m_graph->getBPGraph()->getBundlePlaneGraph(), FlashAttentionComparator()).scheduleNodes();
        const auto& flashAttnSchedule = BundlePlane::createOrigNodesScheduleFromBpgSchedule(bpgSchedule);

        LOG_DEBUG(FLASH_ATTENTION,
                  "Set the schedule of flash attention id {}, with {} nodes",
                  flashAttnIdAndNodes.first,
                  flashAttnSchedule.size());
        unsigned idx    = 0;
        auto     prevIt = flashAttnSchedule.begin();
        for (auto it = std::next(prevIt); it != flashAttnSchedule.end(); ++it)
        {
            m_graph->addControlDependency(*prevIt, *it, Tensor::ControlEdgeType::SCHEDULE);
            prevIt = it;
            LOG_DEBUG(FLASH_ATTENTION, "\t{} exec index: {}", (*it)->getNodeName(), idx++);
        }
    }
    m_graph->discardBPGraph();
    setCrossFaDependencies();
}

void FlashAttentionScheduler::optimizeScheduleForMemory()
{
    createSchedule();
    m_graph->invalidateExecutionSchedule();
    m_graph->setScheduleFlashAttention();
}

bool scheduleFlashAttentionNodes(HabanaGraph& g)
{
    if (!GCFG_ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE.value()) return true;
    FlashAttentionScheduler(&g).optimizeScheduleForMemory();
    return true;
}