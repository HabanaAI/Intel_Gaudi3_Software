#include "bundle_cache_state.h"

namespace gc::layered_brain
{
BundleCacheState::BundleCacheState(Capacity budget) : m_budget(budget) {}

BundleCacheState::Capacity BundleCacheState::totalFree() const
{
    return m_budget - m_occupied;
}

BundleCacheState::Capacity BundleCacheState::totalLive() const
{
    return m_live;
}

bool BundleCacheState::isCached(const TensorPtr& t) const
{
    return m_tensorEntries.find(t) != m_tensorEntries.end() && tensorEntry(t)->state == Entry::State::ALIVE;
}

bool BundleCacheState::isReclaimCandidate(const TensorPtr& t) const
{
    return m_tensorEntries.find(t) != m_tensorEntries.end() && tensorEntry(t)->state == Entry::State::DEAD;
}

BundleCacheState::Capacity BundleCacheState::capacity(const TensorPtr& t) const
{
    return tensorEntry(t)->size;
}

void BundleCacheState::cache(const TensorPtr& t, Capacity capacity)
{
    HB_ASSERT(m_tensorEntries.find(t) == m_tensorEntries.end(),
              "Re-caching without reclaim of previous allocation of tensor {}",
              t->getName());

    m_occupied += capacity;
    m_live += capacity;
    m_maxLive = std::max(m_live, m_maxLive);

    m_cacheState.emplace_back(t, capacity);
    m_tensorEntries[t] = m_cacheState.size() - 1;

    LOG_TRACE(LB_CACHE_MNGR, "After caching {}, with capacity {}B, cache state:", t->getName(), capacity);
    logCacheState();
}

void BundleCacheState::release(const TensorPtr& t)
{
    auto* entry = tensorEntry(t);
    HB_ASSERT(entry->state == Entry::State::ALIVE,
              "Double release or release after reclaim of tensor {}",
              t->getName());

    m_live -= entry->size;
    entry->state = Entry::State::DEAD;

    LOG_TRACE(LB_CACHE_MNGR, "After releasing {}, cache state:", t->getName());
    logCacheState();
}

void BundleCacheState::reclaim(const TensorPtr& t)
{
    auto* entry = tensorEntry(t);
    HB_ASSERT(entry->state != Entry::State::RECLAIMED, "Double reclaim of tensor {}", t->getName());

    m_occupied -= entry->size;
    if (entry->state == Entry::State::ALIVE)
    {
        m_live -= entry->size;
    }

    entry->state = Entry::State::RECLAIMED;
    m_tensorEntries.erase(t);

    LOG_TRACE(LB_CACHE_MNGR, "After reclaiming {}, cache state:", t->getName());
    logCacheState();
}

BundleCacheState::Entry* BundleCacheState::tensorEntry(const TensorPtr& t)
{
    auto it = m_tensorEntries.find(t);
    HB_ASSERT(it != m_tensorEntries.end(), "Trying to access uncached tensor entry: {}", t ? t->getName() : "<null>");
    auto handle = m_tensorEntries.at(t);
    return &m_cacheState.at(handle);
}

const BundleCacheState::Accessors& BundleCacheState::accesses(const TensorPtr& t) const
{
    return tensorEntry(t)->accesses;
}

void BundleCacheState::addAccess(const TensorPtr& t, size_t accessIdx)
{
    tensorEntry(t)->accesses.push_back(accessIdx);
}

std::vector<const BundleCacheState::Entry*> BundleCacheState::lruReclaimCandidates() const
{
    // Optimization - The LRU order of the entries can be saved and updated "in-flight" in order to
    // save this post-process sorting.
    std::vector<const BundleCacheState::Entry*> candidates;
    for (const Entry& e : m_cacheState)
    {
        if (e.state == Entry::State::DEAD)
        {
            candidates.push_back(&e);
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Entry* lhs, const Entry* rhs) {
        HB_ASSERT(!lhs->accesses.empty(), "unexpected reclaim of unaccessed cached tensor");
        HB_ASSERT(!rhs->accesses.empty(), "unexpected reclaim of unaccessed cached tensor");
        return lhs->accesses.back() < rhs->accesses.back();
    });
    return candidates;
}

BundleCacheState::Capacity BundleCacheState::maxLiveCapacity() const
{
    return m_maxLive;
}

BundleCacheState::Capacity BundleCacheState::maxReclaim() const
{
    return m_occupied - m_live;
}

void BundleCacheState::logCacheState() const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(LB_CACHE_MNGR)) return;
    std::vector<std::string> entryLogs;
    for (const Entry& e : m_cacheState)
    {
        entryLogs.push_back(logEntry(e));
    }
    LOG_TRACE(LB_CACHE_MNGR,
              "\n======================== Cache State records: ========================\n"
              "{}"
              "\n"
              "------------------------------------------------------------------------\n"
              "Live: {} B, Occupied: {} B, Max Live: {} B"
              "\n======================== Cache State End =============================",
              fmt::join(entryLogs.begin(), entryLogs.end(), "\n"),
              m_live,
              m_occupied,
              m_maxLive);
}

std::string BundleCacheState::logEntry(const Entry& e)
{
    return fmt::format("{:<9} | {:<80} | {:>10} B", stateStr(e.state), e.tensor->getName(), e.size);
}

std::string BundleCacheState::stateStr(const Entry::State& s)
{
    std::string str;
    switch (s)
    {
        case Entry::State::ALIVE:
            str = "ALIVE";
            break;
        case Entry::State::DEAD:
            str = "DEAD";
            break;
        case Entry::State::RECLAIMED:
            str = "RECLAIMED";
            break;
    }
    return str;
}

}  // namespace gc::layered_brain