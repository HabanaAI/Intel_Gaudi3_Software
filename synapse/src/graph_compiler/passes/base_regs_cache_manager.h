#pragma once

#include "compiler_types.h"
#include "habana_graph.h"

template<size_t TQueueNum, unsigned int TCacheCap>
class BaseRegsCacheManager
{
public:
    BaseRegsCacheManager(HabanaGraph& g) : m_graph(g) {}

    void go()
    {
        // eager mode must run with base regs cache enabled
        if (GCFG_DISABLE_BASE_REGISTERS_CACHE.value() && m_graph.getCompilationMode() != CompilationMode::Eager) return;

        for (const auto& node : m_graph.getExeSortedNodes())
        {
            if (node->isLogicalOperation()) continue;
            uint64_t            cacheIdx      = getCacheIndex(node);
            std::set<uint64_t>& stagedCache   = m_stagedCaches[cacheIdx];
            CommitedCache&      commitedCache = m_commitedCaches[cacheIdx];
            TensorVector        nodeOperands  = node->getOperands();
            bool                commitedOnce  = false;

            // The first node of each logical engine is cache updater
            if (commitedCache.getUpdatingNode() == nullptr)
            {
                commitedCache.setUpdatingNode(node);

                // For TPC logical engine, ensure the program data section is always in the cache since it is
                // needed for the kernel address (and possibly also aux tensors and coeff table), so if this
                // is the first TPC node that we encounter, add the program data section to the staged cache.
                if (HabanaGraph::runsOnTPC(node))
                {
                    stagedCache.insert(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
                }
            }

            unsigned operandIdx = 0;
            unsigned increment  = 0;
            for (; operandIdx < nodeOperands.size(); operandIdx += increment)
            {
                increment        = 1;
                TensorPtr tensor = nodeOperands[operandIdx];
                if (tensor == nullptr) continue;
                tensor = tensor->getRealTensor(tensor);
                if (!tensor->isDramOffsetSet()) continue;
                uint64_t tSection = getMemoryIDFromVirtualAddress(tensor->getDramOffset());

                std::set<uint64_t>::iterator itr;
                bool                         inserted = false;
                std::tie(itr, inserted)               = stagedCache.insert(tSection);
                if (inserted && stagedCache.size() > TCacheCap)
                {
                    // Cache overflow. Remove the just inserted entry and commit the staged cache.
                    stagedCache.erase(itr);

                    // If we already commited once in this node (which means that this node uses more sections than the
                    // cache full capacity), we should stop processing this node. In this case the address registers
                    // that use sections which are not loaded to the cache will fall-back to legacy commands with
                    // patch-points.
                    if (commitedOnce) break;

                    commitStagedCache(stagedCache, commitedCache);
                    stagedCache.clear();
                    commitedCache.setAllEntriesAsFree();
                    commitedCache.setUpdatingNode(node);  // mark current node as cache updater for the next commit
                    commitedOnce = true;

                    // Immediately stage the program data section for TPC nodes (for the kernels)
                    if (HabanaGraph::runsOnTPC(node))
                    {
                        stagedCache.insert(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
                    }

                    // Start process this node from the beginning; sections that were just commited for
                    // this node will be staged again and will reuse the entries from the commited cache.
                    operandIdx = 0;
                    increment  = 0;
                }
            }
        }

        // Commit residual staged cache of each logical engine
        for (unsigned i = 0; i < m_stagedCaches.size(); i++)
        {
            commitStagedCache(m_stagedCaches[i], m_commitedCaches[i]);
        }
    }

protected:
    // Internal representation for comitted cache
    class CommitedCache
    {
    public:
        static const unsigned INVALID_INDEX = std::numeric_limits<unsigned>::max();
        CommitedCache()
        {
            m_values.fill(std::numeric_limits<uint64_t>::max());
            m_used.fill(false);
        }

        void setEntryAsUsed(unsigned index) { m_used.at(index) = true; }
        void setEntryValue(unsigned index, uint64_t value)
        {
            m_values.at(index) = value;
            setEntryAsUsed(index);
        }
        void     setAllEntriesAsFree() { m_used.fill(false); }
        unsigned getIndexOfFreeSpot() const
        {
            // return lowest free spot first
            auto freeSpotItr = std::find(m_used.begin(), m_used.end(), false);
            return (freeSpotItr == m_used.end()) ? INVALID_INDEX : (freeSpotItr - m_used.begin());
        }
        unsigned getIndexOfValue(uint64_t value) const
        {
            auto valueItr = std::find(m_values.begin(), m_values.end(), value);
            return (valueItr == m_values.end()) ? INVALID_INDEX : (valueItr - m_values.begin());
        }

        const NodePtr& getUpdatingNode() const { return m_updatingNode; }
        void           setUpdatingNode(NodePtr n) { m_updatingNode = n; }

    private:
        std::array<uint64_t, TCacheCap> m_values;
        std::array<bool, TCacheCap>     m_used;  // flag indicating if an entry in m_values is in-use or free

        NodePtr m_updatingNode;
    };  // class CommitedCache

    virtual uint64_t getCacheIndex(const NodePtr& node) = 0;

    void commitStagedCache(std::set<uint64_t>& stagedCache, CommitedCache& commitedCache)
    {
        // First pass, find entries that already exist in the commited cache and reuse them
        auto itr = stagedCache.begin();
        while (itr != stagedCache.end())
        {
            auto     currItr    = itr++;  // move itr as we may be deleting the current element
            unsigned sectionIdx = commitedCache.getIndexOfValue(*currItr);
            if (sectionIdx != CommitedCache::INVALID_INDEX)
            {
                commitedCache.setEntryAsUsed(sectionIdx);
                stagedCache.erase(currItr);
            }
        }

        // Second pass, find free spot for the remaining new entries, commit them and put them on the updater node
        for (uint64_t section : stagedCache)
        {
            unsigned freeSpotIdx = commitedCache.getIndexOfFreeSpot();
            HB_ASSERT(freeSpotIdx != CommitedCache::INVALID_INDEX,
                      "I've been asked to commit more than can possibly fit, bug!");
            commitedCache.setEntryValue(freeSpotIdx, section);  // also mark as used
            commitedCache.getUpdatingNode()->getNodeAnnotation().baseRegsCacheUpdate.push_back({freeSpotIdx, section});
        }
    }

    HabanaGraph&                     m_graph;

    std::array<std::set<uint64_t>, TQueueNum> m_stagedCaches;    // one entry for each logical engine
    std::array<CommitedCache, TQueueNum>      m_commitedCaches;  // one entry for each logical engine
};
