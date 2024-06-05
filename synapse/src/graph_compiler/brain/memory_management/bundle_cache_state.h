#pragma once

#include "tensor.h"
#include "types.h"

namespace gc::layered_brain
{
// Tracks the state of the cache as tensors are cached and released. As in cache, there is no enforcement of the budget
// and over-subscription is allowed. The user of this object is responsible to prevent over-subscription by checking the
// available free space before caching and reclaiming released tensors in appropriate times when cache-thrashing is
// ensured to be prevented.
class BundleCacheState
{
public:
    // using signed capacity since over-subscription is allowed, so the 'free' space may become negative.
    using Capacity  = int64_t;
    using Accessors = llvm_vecsmall::SmallVector<size_t, 4>;

    struct Entry
    {
        Entry(const TensorPtr& t, Capacity sz) : tensor(t), size(sz) {}

        TensorPtr tensor;    // Whose data is cached
        Capacity  size;      // in bytes
        Accessors accesses;  // by order of addition

        enum class State
        {
            ALIVE,
            DEAD,
            RECLAIMED,
        } state {State::ALIVE};  // All entries are created alive.
    };

    explicit BundleCacheState(Capacity budget);

    // Free cache capacity out of the total budget
    Capacity totalFree() const;

    // Tensor actively in use in the cache (not freed nor reclaimed)
    Capacity totalLive() const;

    // Count capacity 'cap' as occupied by the tensor 't'.
    // Notice that the capacity is different from the tensor size because:
    // a. per-core copy may require up to 4x the cache capacity for each datum
    // b. partial access to the tensor may also be cached and a caller may optimize the state to reflect this.
    void cache(const TensorPtr& t, Capacity cap);

    // Mark 't' as no longer cached.
    // Notice that the capacity saved for 't' is not released until the tensor is reclaimed.
    // This should be used to ensure synchronization between users of 't' and any following user of t's cache budget.
    void release(const TensorPtr& t);

    // Release the capacity occupied by 't'. It's not mandatory to release(t) before reclaiming it. The user is expected
    // to ensure the reclaimed capacity is reused only after 't' is no longer required in cache (sync any accessor of
    // the reclaimed budget with accessors of 't').
    void reclaim(const TensorPtr& t);

    // Is there any cache capacity already allocated for 't' that wasn't already freed or reclaimed.
    bool isCached(const TensorPtr& t) const;

    // Is there any cache capacity already allocated for 't' that was freed but not reclaimed.
    bool isReclaimCandidate(const TensorPtr& t) const;

    // The capacity occupied by 't'
    Capacity capacity(const TensorPtr& t) const;

    // Accesses are bundle nodes that write/read data to/from the cache allocated for 't'. They are represented by their
    // operation index (their order in the bundle execution schedule).
    const Accessors& accesses(const TensorPtr& t) const;
    void             addAccess(const TensorPtr& t, size_t accessIdx);

    // List freed entries in the cache state that weren't reclaimed yet, in Least Recently Used order.
    std::vector<const Entry*> lruReclaimCandidates() const;

    // Returns the currently tracked maximal capacity used by simultaneously non freed or reclaimed tensors.
    Capacity maxLiveCapacity() const;

    // Return the currently reclaim-able capacity
    Capacity maxReclaim() const;

private:
    using StateTable    = std::vector<Entry>;
    using EntryHandle   = StateTable::size_type;
    using TensorEntries = std::unordered_map<TensorPtr, EntryHandle>;

    const Capacity m_budget;
    Capacity       m_occupied = 0;
    Capacity       m_live     = 0;
    Capacity       m_maxLive  = 0;

    // Optimization option - the entries can be saved directly in the tensor map and the indirect access can be saved,
    // but for initial debugging, it's useful to see the reclaimed entries after they were deleted, so the first version
    // uses both a map to handles and an indirect access.
    StateTable    m_cacheState;
    TensorEntries m_tensorEntries;

    Entry*       tensorEntry(const TensorPtr& t);
    const Entry* tensorEntry(const TensorPtr& t) const { return const_cast<BundleCacheState*>(this)->tensorEntry(t); }

    void               logCacheState() const;
    static std::string logEntry(const Entry& e);
    static std::string stateStr(const Entry::State& st);
};

}  // namespace gc::layered_brain