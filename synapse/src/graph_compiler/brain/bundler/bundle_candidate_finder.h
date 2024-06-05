#pragma once

#include <optional>
#include "node.h"
#include "layered_brain_bundle.h"
#include "bundle_break_rules.h"

namespace gc::layered_brain::bundler
{
class CandidateFinder;
using CandidateFinderPtr = std::shared_ptr<CandidateFinder>;
using ExpansionCandidate = NodePtr;

class CandidateFinder
{
public:
    CandidateFinder(const HabanaGraph& graph);
    virtual ~CandidateFinder() = default;

    /**
     * @brief Yields a pair <candidate_i-1, candidate_i> allowing the user to
     *        construct a bundle expansion chain step-by-step by trimming current chain
     *        up to candidate_i-1 (excluding it) as it is the current effective head of the chain.
     *        Trivial expansion:
     *        1st candidate: <nullptr    , candidate_0> , chain: [c0]
     *        2nd candidate: <candidate_0, candidate_1> , chain: [c0,c1]
     *        3nd candidate: <candidate_1, candidate_2> , chain: [c0,c1,c2]
     *        4th candidate: <candidate_2, candidate_3> , chain: [c0,c1,c2,c3]
     *        and so on until exhaustion.
     *
     *        Expansion with trim:
     *        1st candidate: <nullptr    , candidate_0> , chain: [c0]
     *        2nd candidate: <candidate_0, candidate_1> , chain: [c0,c1]
     *        3rd candidate: <candidate_1, candidate_2> , chain: [c0,c1,c2]
     *        4th candidate: <candidate_1, candidate_3> , chain: [c0,c1,c3] # c2 trimmed
     *        5th candidate: <candidate_0, candidate_6> , chain: [c0,c6]    # c1,c3 trimmed
     *        6th candidate: <candidate_6, candidate_7> , chain: [c0,c6,c7]
     *        and so on until exhaustion.
     */
    virtual std::optional<std::pair<ExpansionCandidate, ExpansionCandidate>> next() = 0;

    /**
     * @brief Reject the last candidate returned
     */
    virtual void rejectCandidate() = 0;

protected:
    const HabanaGraph& m_graph;
};

}  // namespace gc::layered_brain::bundler
