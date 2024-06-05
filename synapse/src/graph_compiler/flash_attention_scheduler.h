#pragma once

#include "habana_graph.h"

class FlashAttentionScheduler
{
    using FlashAttnIdToNodeMap = std::unordered_map<unsigned, NodeSet>;
    struct FaSourcesAndSinks
    {
        std::unordered_set<NodePtr> sources;
        std::unordered_set<NodePtr> sinks;
    };

public:
    explicit FlashAttentionScheduler(HabanaGraph* graph) : m_graph(graph) {}
    void optimizeScheduleForMemory();
    static void initFlashAttentionInfo(HabanaGraph& g, const NodePtr& node);

private:
    void                    setChainIdIfPossible();
    bool                    trySetAsSink(const NodePtr& node);
    bool                    trySetAsSource(const NodePtr& node);
    void                    initFlashAttentionSubgraphMetadata(const NodePtr& node);
    void                    setCrossFaDependencies();
    void                    createSchedule();
    std::optional<unsigned> getChainInfo(const NodePtr& node) const;
    void                    updateChainInfo(const NodePtr& node, unsigned chainId);
    std::optional<unsigned> getChainIdFromConnectedNodes(const NodePtr& node, const NodeSet& connectedNodes) const;
    void                    setChainIdFwd(const NodeVector& topoSortedNodes);
    void                    setChainIdBwd(const NodeVector& topoSortedNodes);
    void                    setBlockingFaMap(const NodePtr& node);

    HabanaGraph*                                     m_graph;
    std::map<unsigned, std::unordered_set<unsigned>> m_faToBlockedFas;
    std::map<unsigned, FaSourcesAndSinks>            m_faToSourcesAndSinks;
    std::unordered_set<unsigned>                     m_chainIds;
};

class FlashAttentionComparator
{
public:
    FlashAttentionComparator() {}
    bool operator()(const NodePtr& n1, const NodePtr& n2) const;

private:
    std::optional<bool> compareFlashAttentionChains(const NodePtr& n1, const NodePtr& n2) const;
};
