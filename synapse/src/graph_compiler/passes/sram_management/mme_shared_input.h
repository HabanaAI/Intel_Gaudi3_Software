#pragma once

#include "mme_slicing_strategy.h"
#include "habana_graph.h"
#include "bundle.h"
#include "slicing_utils.h"

using SlaveNodeToSharedTensor = SortableNodeMap<pTensor>;

// Class responsible for finding valid candidates for shared MME input stitching
class SharedMMEInputCandidateHandler final
{
public:
    SharedMMEInputCandidateHandler() = default;
    ~SharedMMEInputCandidateHandler() = default;
    // find an mme candidate with shared input.
    std::list<pBundleExpansion> findSharedMMEInputCandidate(const pMmeSlicingStrategy& strategy, const HabanaGraph& graph) const;
    // make sure there is enough room in SRAM for this candidate.
    static bool isCandidateValidForStrategy(const pBundleExpansion& candidate,
                                            const pMmeSlicingStrategy& strategy);
    // find the relevant candidate from the given strategy valid or non-valid.
    static pBundleExpansion getCandidateFromStrategy(const MmeSlicingStrategy* strategy, bool& isValidForRole);
    // slice the candidate slave node correctly, and adjust the sram capacity accordingly.
    static uint64_t getCandidateAdditionalCapacity(const pBundleExpansion& candidate);

private:
    SlaveNodeToSharedTensor findSharedMMEInputConsumers(const pNode& masterNode, const HabanaGraph& graph) const;
};
