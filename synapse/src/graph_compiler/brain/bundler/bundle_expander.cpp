#include "bundle_expander.h"
#include "bundle_candidate_finder.h"
#include "bundle_break_rules.h"

namespace gc::layered_brain::bundler
{
BundleExpander::BundleExpander(const BundlePtr& bundle, HabanaGraph& graph, CandidateFinderPtr finder)
: m_bundle(bundle), m_graph(graph), m_candidateFinder(finder), m_ruleLibrary(RuleLibrary::instance())
{
    HB_ASSERT_PTR(m_candidateFinder);
    HB_ASSERT_PTR(m_bundle);
}

std::string_view toString(BundleExpander::Type type)
{
    switch (type)
    {
        case BundleExpander::Type::FIRST_TPC_CONSUMER:
            return "FirstTpcConsumerExpander";
        case BundleExpander::Type::NO_PRODUCERS_TPC_CONSUMER:
            return "NoProducersFirstTpcConsumerExpander";
        case BundleExpander::Type::FIRST_TPC_PRODUCER:
            return "FirstTpcProducerExpander";
        case BundleExpander::Type::TPC_PRODUCERS:
            return "TpcProducersExpander";
        case BundleExpander::Type::ITERATIVE_TPC_PRODUCERS:
            return "TpcProducersIterativeExpander";
        case BundleExpander::Type::ITERATIVE_TPC_CONSUMERS:
            return "TpcConsumersIterativeExpander";
        case BundleExpander::Type::NO_PRODUCERS_ITERATIVE_TPC_CONSUMERS:
            return "NoProducersTpcConsumersIterativeExpander";
        case BundleExpander::Type::NO_CONSUMERS_ITERATIVE_TPC_PRODUCERS:
            return "NoConsumersTpcProducersIterativeExpander";
        case BundleExpander::Type::ITERATIVE_FIRST_MME_PRODUCER:
            return "IterativeFirstMmeProducerExpander";
        case BundleExpander::Type::ITERATIVE_FIRST_MME_CONSUMER:
            return "IterativeFirstMmeConsumerExpander";
        default:
            HB_ASSERT(false, "Unexpected expander type {}", type);
            return "";
    }
}

bool BundleExpander::isProducerExpander() const
{
    const auto expanderType = getType();
    switch (expanderType)
    {
        case BundleExpander::Type::FIRST_TPC_PRODUCER:
        case BundleExpander::Type::TPC_PRODUCERS:
        case BundleExpander::Type::ITERATIVE_TPC_PRODUCERS:
        case BundleExpander::Type::NO_CONSUMERS_ITERATIVE_TPC_PRODUCERS:
            return true;
        case BundleExpander::Type::FIRST_TPC_CONSUMER:
        case BundleExpander::Type::NO_PRODUCERS_TPC_CONSUMER:
        case BundleExpander::Type::ITERATIVE_TPC_CONSUMERS:
        case BundleExpander::Type::NO_PRODUCERS_ITERATIVE_TPC_CONSUMERS:
            return false;
        default:
            HB_ASSERT(false, "Unexpected expander type {}", expanderType);
            return false;
    }
}

}  // namespace gc::layered_brain::bundler