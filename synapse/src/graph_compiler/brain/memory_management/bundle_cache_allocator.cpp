#include "bundle_cache_allocator.h"
#include "layered_brain.h"

namespace gc::layered_brain
{
const BundleCacheAllocator::Result BundleCacheAllocator::allocate(const TensorPtr& t, Capacity cap)
{
    if (!m_cacheState.isCached(t) && !m_cacheState.isReclaimCandidate(t))
    {
        LOG_DEBUG(LB_CACHE_MNGR, "Allocate new entry with capacity: {}B, tensor: {}", cap, t->getName());
        return allocateNew(t, cap);
    }
    else
    {
        LOG_DEBUG(LB_CACHE_MNGR, "Re-allocate entry with capacity: {}B, tensor: {}", cap, t->getName());
        return reAllocate(t, cap);
    }
}

const BundleCacheAllocator::Result BundleCacheAllocator::allocateNew(const TensorPtr& t, Capacity cap)
{
    Result::Deps deps;
    if (potentialCapacity() < cap)
    {
        return Result::failure(cap - potentialCapacity());
    }
    if (m_cacheState.totalFree() < cap)
    {
        auto reclaimRes = reclaim(cap - m_cacheState.totalFree());
        HB_ASSERT(reclaimRes.has_value(), "Unexpected failure to reclaim enough cache space.");
        deps = *reclaimRes;
    }
    m_cacheState.cache(t, cap);
    return Result::success(deps);
}

const BundleCacheAllocator::Result BundleCacheAllocator::reAllocate(const TensorPtr& t, Capacity cap)
{
    Capacity capToAllocate = cap;
    if (m_cacheState.isCached(t))
    {
        // If the tensor is alive in cache, no need to allocate extra capacity on top of what's already allocated to it.
        // Note that Capacity is a signed type and in case 'cap' is smaller than the current allocation, the extra
        // capacity will be freed.
        capToAllocate -= m_cacheState.capacity(t);
    }

    // Optimization - in case nothing is needed, prevent adding unnecessary entries
    if (capToAllocate == 0) return Result::success(Result::NO_DEPENDENCIES);

    if (potentialCapacity() < capToAllocate) return Result::failure(capToAllocate - potentialCapacity());

    auto accesses = m_cacheState.accesses(t);
    m_cacheState.reclaim(t);
    Result res = allocateNew(t, cap);

    // Expect allocation to always succeed since the available capacity was ensured above by checking that
    // totalFree + maxReclaim >= extraCap
    HB_ASSERT(res.successful, "Unexpected re-allocation failure");

    for (auto acc : accesses)
    {
        m_cacheState.addAccess(t, acc);
    }
    return res;
}

std::optional<BundleCacheAllocator::Result::Deps> BundleCacheAllocator::reclaim(BundleCacheAllocator::Capacity cap)
{
    if (m_cacheState.maxReclaim() < cap)
    {
        LOG_DEBUG(LB_CACHE_MNGR,
                  "Not enough potential reclaiming capacity. Max available: {}B, needed: {}B",
                  m_cacheState.maxReclaim(),
                  cap);
        return std::nullopt;
    }

    LOG_DEBUG(LB_CACHE_MNGR, "Reclaiming at least {}B (out of available: {})", cap, m_cacheState.maxReclaim());

    Capacity     reclaimed = 0;
    Result::Deps deps;
    for (const auto* entry : m_cacheState.lruReclaimCandidates())
    {
        deps.insert(deps.end(), entry->accesses.begin(), entry->accesses.end());
        reclaimed += entry->size;

        LOG_DEBUG(LB_CACHE_MNGR,
                  "  Reclaimed capcity: {}B (total so far: {}B) allocated for {}.",
                  entry->size,
                  reclaimed,
                  entry->tensor->getName());

        m_cacheState.reclaim(entry->tensor);

        if (reclaimed >= cap) break;
    }
    return deps;
}

bool BundleCacheAllocator::free(const TensorPtr& t)
{
    if (!m_cacheState.isCached(t)) return false;

    LOG_DEBUG(LB_CACHE_MNGR, "Free entry for tensor {}", t->getName());
    m_cacheState.release(t);
    return true;
}

}  // namespace gc::layered_brain