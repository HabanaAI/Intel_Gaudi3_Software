#pragma once

#include "tensor.h"

#include <tuple>
#include <unordered_map>
#include <vector>

class HabanaGraph;

class NonPersistentSectionAllocTracker
{
public:
    NonPersistentSectionAllocTracker(HabanaGraph& g, bool sram);

    // NOTE: The user should invoke verifyAllDone() before the d'tor.
    ~NonPersistentSectionAllocTracker();

    // mark the base address of the section once it is allocated.
    // use -1 when rolling back to unallocated state
    void setSectionBaseAddr(uint64_t sectionId, uint64_t addr);

    // Mark tensor as allocated, returns true if it's the first tensor within
    // the non-persistent section and it requires an actual allocation.
    // If false, offset is the tensors new offset (-1 otherwise).
    bool markAsAlloc(const TensorPtr& tensor, /*OUT*/ size_t& offset);

    // Mark tensor as freed or unallocated if rollback. Return true if either
    // all freed or unallocated after allowing non-persistent section memory release.
    //
    // Note that freed tensors may not be re-allocated unless rollback'd
    bool markAsFree(const TensorPtr& tensor, bool rollback);

    void handleUnallocatedTensorFallback(const TensorPtr& tensor);

    // If tensor belongs to a non-persistent section, mark it as planned for allocation and
    // return if its non-persistent section was already allocated or if it's been planned.
    bool trackPlannedNonPersistentSectionTensor(const TensorPtr& tensor);

    void discardPlannedTensors();

    // Check internal consistency and that all non-persistent sections are unalloc or free
    void verifyAllDone() const;

    // Verbose LOG_TRACE level print of the entire m_tensor2status state
    void tracePrint(const std::string& label, const bool* res, bool showAll) const;

private:
    enum class TensorStatus
    {
        unallocated,
        allocated,
        freed
    };
    std::unordered_map<TensorPtr, TensorStatus> m_tensor2status;

    struct NonPersistentSectionDesc
    {
        NonPersistentSectionDesc() = default;
        NonPersistentSectionDesc(TensorVector nonPersistentSectionTensors);

        size_t                 totalUnallocated = 0;
        size_t                 totalAllocated   = 0;
        size_t                 totalFreed       = 0;
        uint64_t               addr             = -1;  // this value is ignored in == operator
        TensorVector           tensors;

        bool operator==(const NonPersistentSectionDesc& o) const;
    };
    // A per non-persistent section summary of m_tensor2status which can be completely
    // regenerated from it (As is done in verifyDataInSync) but is kept
    // up-to-date to avoid searching all non-persistent section tensors in m_tensor2status
    // when asking whether there's any un/allocated/freed tensors and when
    // doing resetting a specific non-persistent section to unallocated/ looking for an
    // allocated tensor to recover the non-persistent section offset from it.
    std::unordered_map<uint64_t, NonPersistentSectionDesc> m_nonPersistentSectionId2desc;

    // Since epoch planning is done before allocations are made, we keep track
    // locally of additional non-persistent section id's which we plan to allocate in the
    // current epoch, so that for all tensors but the 1st,
    // writeSpaceForTensor = 0.
    std::vector<uint64_t> m_currentPlannedNonPersistentSectionIds;

    // Check that m_nonPersistentSectionId2desc and m_tensor2status are in sync
    bool verifyDataInSync() const;

    const TensorStatus& getTensorStatus(const TensorPtr& t) const;
    TensorStatus&       getTensorStatus(const TensorPtr& t);
};
