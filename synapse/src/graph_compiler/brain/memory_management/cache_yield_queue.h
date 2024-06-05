#pragma once

#include "cache_management_apis.h"

namespace gc::layered_brain
{
// This is a wrapper around a set of yielding candidates with random access by cache key (real tensor).
// The set is sorted by yielding priority and can be filtered to provide only candidates that fit some criteria.
// Each candidate saves the details for its sorting and the details on how to "retroactively" release it, if the user
// chooses to do so. "Retroactive" here is only in terms of compilation phase - the bundle operations are processed
// one by one, but yielding goes back to an operation that was already processed and changes its cache directives. In
// runtime the operation would simply release the cache during its execution, no retroactiveness involved.
class CacheYieldQueue
{
public:
    using ReleaseReq = CacheRequirementsAnalyzerIfc::RequirementDetails;
    using Key        = TensorPtr;

    // Details of a buffer that can be yielded if needed.
    struct Candidate
    {
        // Details needed to retroactively release a buffer when deciding to yield it
        struct ReleaseRecipe
        {
            NodePtr    node;
            size_t     inputIdx;
            ReleaseReq requirement;
        };

        Key           cacheKey;
        size_t        threadIdx;
        ReleaseRecipe releaseRecipe;
    };

    struct ReleaseCompare
    {
        bool operator()(const Candidate::ReleaseRecipe& lhs, const Candidate::ReleaseRecipe& rhs) const
        {
            HB_ASSERT(lhs.node != rhs.node || lhs.inputIdx != rhs.inputIdx,
                      "Unexpected compare of the same cache access of node: {} input: {}",
                      lhs.node->getNodeName(),
                      lhs.inputIdx);

            auto lhsOperationIdx = lhs.node->getNodeAnnotation().bundleInfo->operationIndex;
            auto rhsOperationIdx = rhs.node->getNodeAnnotation().bundleInfo->operationIndex;
            if (lhsOperationIdx != rhsOperationIdx)
            {
                return lhsOperationIdx < rhsOperationIdx;
            }
            else if (lhs.requirement.capacity != rhs.requirement.capacity)
            {
                // In the same operation index, prefer yielding the buffer with the smallest capacity
                return lhs.requirement.capacity < rhs.requirement.capacity;
            }
            return false;  // can't decide based on release recipe
        }
    };

    // This comparator is responsible to sort the candidates by their release priority.
    struct CandidateCompare
    {
        ReleaseCompare releaseCompare;
        TensorComparator tensorCompare;
        bool           operator()(const Candidate& lhs, const Candidate& rhs) const
        {
            if (lhs.cacheKey == rhs.cacheKey) return false;
            if (releaseCompare(lhs.releaseRecipe, rhs.releaseRecipe) !=
                releaseCompare(rhs.releaseRecipe, lhs.releaseRecipe))
            {
                return releaseCompare(lhs.releaseRecipe, rhs.releaseRecipe);
            }
            // Last resort tie breaker (threadIdx is not compared, but since threads run in parallel, they shouldn't be
            // used to prioritize yielding candidates, only, maybe, as filter)
            return tensorCompare(lhs.cacheKey, rhs.cacheKey);
        }
    };

    using CandidateQueue = std::set<Candidate, CandidateCompare>;

    void addCandidate(const NodePtr& node, const Key& cacheKey, size_t inputIdx, const ReleaseReq& requirements)
    {
        erase(cacheKey);  // First clear any previous entry of this buffer.

        const auto& thread = node->getNodeAnnotation().bundleInfo->threadIndex;
        if (!thread)
        {
            LOG_WARN(LB_CACHE_MNGR,
                     "Buffer with key: {}, was accessed from a thread-less operation: {} so it can't be added as a "
                     "candidate for cache yielding.",
                     cacheKey->getName(),
                     node->getNodeName());
            return;
        }

        auto [iter, inserted] =
            m_candidates.insert(Candidate {cacheKey, *thread, Candidate::ReleaseRecipe {node, inputIdx, requirements}});
        HB_ASSERT(inserted, "Unexpected duplicate entry in yielding candidates queue: {}", cacheKey->getName());
        m_candidateByTensor[cacheKey] = iter;
    }

    CandidateQueue::const_iterator begin() const { return m_candidates.begin(); }
    CandidateQueue::const_iterator end() const { return m_candidates.end(); }

    const Candidate& get(const Key& cacheKey) const
    {
        auto iter = m_candidateByTensor.find(cacheKey);
        HB_ASSERT(iter != m_candidateByTensor.end(), "Unexpected key to yield candidates queue");
        return *iter->second;
    }

    void erase(const Key& cacheKey)
    {
        auto iter = m_candidateByTensor.find(cacheKey);
        if (iter != m_candidateByTensor.end())
        {
            m_candidates.erase(iter->second);
            m_candidateByTensor.erase(iter);
        }
    }

    struct YieldingOptions
    {
        uint64_t         availableCapacity = 0;
        std::vector<Key> sortedYieldingKeys {};
    };

    // Provide the yielding options from candidates that pass the provided filter
    template<typename Filter>
    YieldingOptions yieldingOptions(const Filter& candidatesFilter) const
    {
        YieldingOptions ret;
        for (const auto& c : m_candidates)
        {
            if (candidatesFilter(c))
            {
                ret.availableCapacity += c.releaseRecipe.requirement.capacity;
                ret.sortedYieldingKeys.push_back(c.cacheKey);
            }
        }
        return ret;
    }

private:
    CandidateQueue                                          m_candidates;
    std::unordered_map<TensorPtr, CandidateQueue::iterator> m_candidateByTensor;
};
}  // namespace gc::layered_brain