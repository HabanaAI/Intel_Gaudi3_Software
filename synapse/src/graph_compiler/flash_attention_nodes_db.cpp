#include "flash_attention_nodes_db.h"
#include "defs.h"
#include "log_manager.h"

void FlashAttentionDb::registerId(synNodeId id, const std::unordered_set<ChainIdx>& chains)
{
    m_flashAttentionOrigIdsToChains.insert({id, chains});
}

bool FlashAttentionDb::isRegistered(synNodeId id) const
{
    return m_flashAttentionOrigIdsToChains.find(id) != m_flashAttentionOrigIdsToChains.end();
}

void FlashAttentionDb::registerChainForNode(synNodeId id, ChainIdx chain)
{
    const auto& it = m_flashAttentionOrigIdsToChains.find(id);
    HB_ASSERT(it != m_flashAttentionOrigIdsToChains.end(), "Id {} is not FA", id);
    it->second.insert(chain);
}

void FlashAttentionDb::removeUnslicedFlashAttentionNodes()
{
    for (auto it = m_flashAttentionOrigIdsToChains.begin(); it != m_flashAttentionOrigIdsToChains.end();)
    {
        const auto& chains = it->second;
        if (chains.size() <= 1)
        {
            LOG_TRACE(FLASH_ATTENTION,
                      "Flash attention id {} is a single chain, no special handling is needed",
                      it->first);
            it = m_flashAttentionOrigIdsToChains.erase(it);
        }
        else
        {
            it++;
        }
    }
}