#pragma once

#include "bundle_cache_state.h"

namespace gc::layered_brain
{
// This class is responsible for the logic of allocating cache, reclaiming freed cache and re-using the budget while
// providing the set of dependencies that must be ensured to prevent cache thrashing.
class BundleCacheAllocator
{
public:
    using Capacity = BundleCacheState::Capacity;

    // Result from an allocation attempt. If successful, the user must ensure dependency between the new user of
    // the allocated budget and the slice nodes with operation index in the 'dependency' set.
    struct Result
    {
        using Deps = llvm_vecsmall::SmallVector<size_t, 4>;

        bool     successful;
        Deps     dependencies;     // In case of success
        Capacity missingCapacity;  // In case of failure

        static constexpr std::initializer_list<Deps::value_type> NO_DEPENDENCIES = {};

        static Result failure(Capacity missingCap)
        {
            return Result {.successful = false, .dependencies = {}, .missingCapacity = missingCap};
        }
        static Result success(const Deps& deps)
        {
            return Result {.successful = true, .dependencies = deps, .missingCapacity = 0};
        }
    };

    explicit BundleCacheAllocator(BundleCacheState& cacheState) : m_cacheState(cacheState) {}

    // Try to allocate 'cap' bytes in the cache. If some cache needs to be reclaimed in order to do so, returns the
    // operation indices of the reclaimed budget accessors. The user of this class is responsible to ensure dependency
    // between the accessors of the reclaimed budget and the accessors of 't'.
    const Result allocate(const TensorPtr& t, Capacity cap);

    // If 't' is cached, releases it and returns true. Otherwise ('t' is not cached) return false.
    bool free(const TensorPtr& t);

private:
    BundleCacheState& m_cacheState;

    const Result                allocateNew(const TensorPtr& t, Capacity cap);
    const Result                reAllocate(const TensorPtr& t, Capacity cap);
    std::optional<Result::Deps> reclaim(Capacity cap);
    inline Capacity potentialCapacity() const { return m_cacheState.totalFree() + m_cacheState.maxReclaim(); }
};

}  // namespace gc::layered_brain