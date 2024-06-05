#pragma once
#include "types.h"

#include <string_view>
#include <memory>
#include <vector>
#include "bundle_candidate_finder.h"
#include "layered_brain_bundle.h"

namespace gc::layered_brain::bundler
{
class BundleExpander
{
public:
    using ExpanderEnumValueType = uint8_t;

    enum class Type : ExpanderEnumValueType
    {
        TPC_PRODUCERS,
        FIRST_TPC_PRODUCER,
        FIRST_TPC_CONSUMER,
        ITERATIVE_TPC_PRODUCERS,
        ITERATIVE_TPC_CONSUMERS,
        NO_PRODUCERS_TPC_CONSUMER,
        NO_PRODUCERS_ITERATIVE_TPC_CONSUMERS,
        NO_CONSUMERS_ITERATIVE_TPC_PRODUCERS,
        ITERATIVE_FIRST_MME_PRODUCER,
        ITERATIVE_FIRST_MME_CONSUMER,
        NUM_TYPES
    };

    BundleExpander(const BundlePtr& bundle, HabanaGraph& graph, CandidateFinderPtr finder);
    virtual ~BundleExpander()         = default;
    virtual BundlePtr expand()        = 0;
    virtual Type      getType() const = 0;
    bool              isProducerExpander() const;

protected:
    BundlePtr          m_bundle;
    HabanaGraph&       m_graph;
    CandidateFinderPtr m_candidateFinder;
    const RuleLibrary& m_ruleLibrary;
};
using BundleExpanderPtr = std::shared_ptr<BundleExpander>;
std::string_view toString(BundleExpander::Type type);

}  // namespace gc::layered_brain::bundler
