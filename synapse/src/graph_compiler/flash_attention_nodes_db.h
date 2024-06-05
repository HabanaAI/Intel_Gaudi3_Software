#pragma once

#include "types.h"

class FlashAttentionDb
{
    using ChainIdx = unsigned;

public:
    FlashAttentionDb() {};
    void registerId(synNodeId id, const std::unordered_set<ChainIdx>& chains = {});
    void registerChainForNode(synNodeId id, ChainIdx chain);
    void removeUnslicedFlashAttentionNodes();
    bool isRegistered(synNodeId id) const;

private:
    std::map<synNodeId, std::unordered_set<ChainIdx>> m_flashAttentionOrigIdsToChains;
};