#include "bundle_expanders.h"
#include "bundle_candidate_finders.h"
#include "bundle_break_rules.h"
#include "log_manager.h"

namespace gc::layered_brain::bundler
{
OperandExpander::OperandExpander(const TensorPtr&            operand,
                                 const BundlePtr&            bundle,
                                 HabanaGraph&                graph,
                                 const CandidateFinderPtr&   finder,
                                 const std::vector<RuleType> validityRules,
                                 const std::vector<RuleType> stopRules)
: BundleExpander(bundle, graph, finder),
  m_operand(operand),
  m_validityRules(m_ruleLibrary.getRules(validityRules)),
  m_stopRules(m_ruleLibrary.getRules(stopRules))
{
    HB_ASSERT_PTR(m_operand);
}

std::vector<ExpansionCandidate>& OperandExpander::getCandidates()
{
    return m_expansionCandidates;
}

const std::vector<ExpansionCandidate>& OperandExpander::getCandidates() const
{
    return const_cast<const std::vector<ExpansionCandidate>&>(const_cast<OperandExpander*>(this)->getCandidates());
}

RuleContext OperandExpander::getCurCtx(const NodePtr& candidate) const
{
    RuleContext ctx(m_graph,
                    candidate,
                    m_bundle,
                    std::make_shared<decltype(m_expansionCandidates)>(m_expansionCandidates));
    return ctx;
}

bool OperandExpander::stop() const
{
    RuleContext ctx = getCurCtx(nullptr /*no specific candidate*/);
    for (const auto& rule : m_stopRules)
    {
        if (rule->apply(ctx))
        {
            LOG_DEBUG(LB_BUNDLER, "Candidate hit stop rule: {}", rule->name());
            return true;
        }
    }
    return false;
}

bool OperandExpander::valid(const NodePtr& candidate) const
{
    HB_ASSERT_PTR(candidate);
    LOG_DEBUG(LB_BUNDLER,
              "Running validity rules for candidate {}:{}",
              candidate->getNodeTypeStr(),
              candidate->getNodeName());

    RuleContext ctx = getCurCtx(candidate);
    for (const auto& rule : m_validityRules)
    {
        if (!rule->apply(ctx))
        {
            LOG_DEBUG(LB_BUNDLER, "Candidate failed rule: {}", rule->name());
            return false;
        }
    }
    return true;
}

void OperandExpander::adjustCandidatesContainer(const NodePtr& candidate, const NodePtr& link)
{
    auto& candidatesContainer = getCandidates();
    HB_ASSERT_PTR(candidate);
    while (!candidatesContainer.empty() && candidatesContainer.back() != link)
    {
        LOG_TRACE(LB_BUNDLER,
                  "{}: Removing {}[{}] from candidates container:",
                  HLLOG_FUNC,
                  candidatesContainer.back()->getNodeName(),
                  candidatesContainer.back()->getNodeTypeStr());
        candidatesContainer.pop_back();
    }
}
void OperandExpander::addCandidatesToBundle()
{
    // add nodes directly to bundle by default
    addBundleNodes();
}

void OperandExpander::addBundleNodes()
{
    m_bundle->add(getCandidates());
}
void OperandExpander::addBundleCandidates()
{
    m_bundle->addCandidate(getCandidates());
}

bool OperandExpander::skipRejectedTraversalPath(const NodePtr& link) const
{
    // When bundle candidates from a previous expansion are rejected by the runner,
    // Expander's candidate finder might suggest an invalid expansion in following expansion step
    // in sense of the expansion candidate being linked to previously rejected candidates.
    // In such cases, skip the invalid expansion and signal candidate finder to reject the traversal direction.
    return link != nullptr &&
           std::find(getCandidates().begin(), getCandidates().end(), link) == getCandidates().end() &&
           m_bundle->getNodes().find(link) == m_bundle->getNodes().end();
}

BundlePtr OperandExpander::expand()
{
    SET_TEMP_LOG_CONTEXT(fmt::format("{} Bundle#{}", toString(getType()), std::to_string(m_bundle->index())));
    if (!canDeploy())
    {
        LOG_DEBUG(LB_BUNDLER, "Cannot deploy");
        return nullptr;
    }

    LOG_DEBUG(LB_BUNDLER, "Expansion from operand {}", m_operand->getName());
    for (auto state = m_candidateFinder->next(); state.has_value(); state = m_candidateFinder->next())
    {
        const auto [link, candidate] = state.value();
        // Link may be null when expanding from the expander's starting operand.
        HB_ASSERT_PTR(candidate);

        // Adjust candidates list according to finder state
        adjustCandidatesContainer(candidate, link);

        if (skipRejectedTraversalPath(link))
        {
            LOG_DEBUG(LB_BUNDLER,
                      "Expansion link {}[{}] required but not in bundle, reject candidate {}[{}]",
                      link->getNodeName(),
                      link->getNodeTypeStr(),
                      candidate->getNodeName(),
                      candidate->getNodeTypeStr());
            m_candidateFinder->rejectCandidate();
            continue;
        }

        if (!valid(candidate))
        {
            // Reached invalid candidate, change expansion direction
            LOG_DEBUG(LB_BUNDLER,
                      "Reject expansion candidate: {}[{}]",
                      candidate->getNodeName(),
                      candidate->getNodeTypeStr());
            m_candidateFinder->rejectCandidate();
            continue;
        }

        LOG_TRACE(LB_BUNDLER,
                  "Append {}[{}] to candidates container",
                  candidate->getNodeName(),
                  candidate->getNodeTypeStr());
        getCandidates().push_back(candidate);

        // Check whether to yield or not
        if (stop())
        {
            addCandidatesToBundle();
            getCandidates().clear();
            return m_bundle;
        }
    }

    LOG_DEBUG(LB_BUNDLER, "Expander exhausted");
    return nullptr;
}

void OperandExpander::injectDeployRules(const std::vector<RuleType>& deployRules)
{
    std::transform(deployRules.begin(),
                   deployRules.end(),
                   std::back_inserter(m_deployRules),
                   [this](const RuleType& rule) { return m_ruleLibrary.getRule(rule); });
}

bool OperandExpander::runDeployRules() const
{
    RuleContext ctx = getCurCtx(nullptr /*no specific candidate*/);
    for (const auto& rule : m_deployRules)
    {
        if (!rule->apply(ctx))
        {
            return false;
        }
    }
    return true;
}

bool OperandExpander::canDeploy() const
{
    if (m_deployed.has_value())
    {
        // deploy rules had been checked, return cached state
        return m_deployed.value();
    }
    else
    {
        // run deploy rules to determine whether the expander can run.
        // cache result for future invocations
        m_deployed = runDeployRules();
        return m_deployed.value();
    }
}

FirstTpcProducerExpander::FirstTpcProducerExpander(const TensorPtr& operand,
                                                   const BundlePtr& bundle,
                                                   HabanaGraph&     graph)
: OperandExpander(operand, bundle, graph, getFinder(operand, graph), getValidityRules(), getStopRules())
{
}

const std::vector<RuleType> FirstTpcProducerExpander::getValidityRules() const
{
    return {RuleType::NOT_RUNS_ON_MME,
            RuleType::NOT_RUNS_ON_DMA,
            RuleType::NOT_BUNDLED,
            RuleType::NOT_NDIM,
            RuleType::HAS_ACCESS_PATTERN,
            (GCFG_ENABLE_DSD_WITH_LB.value() ? RuleType::NOT_SERIALIZE_DESERIALIZE_NODE : RuleType::NOT_DYNAMIC_SHAPE),
            RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
            RuleType::NO_BINDING_INPUT_REUSE,
            RuleType::NO_SHARED_INPUT_IN_BUNDLE,
            RuleType::NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
            RuleType::HAS_UNIQUE_INPUTS,
            RuleType::SUPPORTED_LOGICAL_OPERATION,
            RuleType::NO_PRODUCER_BPG_CYCLE,
            RuleType::NOT_MAX_CANDIDATES};
}

const std::vector<RuleType> FirstTpcProducerExpander::getStopRules() const
{
    return {RuleType::LAST_CANDIDATE_TPC};
}

const CandidateFinderPtr FirstTpcProducerExpander::getFinder(const TensorPtr& operand, HabanaGraph& graph) const
{
    return std::make_shared<DfsProducersFinder>(operand, graph);
}

NoProducersFirstTpcConsumerExpander::NoProducersFirstTpcConsumerExpander(const TensorPtr& operand,
                                                                         const BundlePtr& bundle,
                                                                         HabanaGraph&     graph)
: FirstTpcConsumerExpander(operand, bundle, graph)
{
    injectDeployRules({RuleType::NO_MME_PRODUCERS_BUNDLED});
}

TpcProducersExpander::TpcProducersExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph)
: OperandExpander(operand, bundle, graph, getFinder(operand, graph), getValidityRules(), getStopRules())
{
}

const std::vector<RuleType> TpcProducersExpander::getValidityRules() const
{
    return {RuleType::NOT_RUNS_ON_MME,
            RuleType::NOT_RUNS_ON_DMA,
            RuleType::NOT_BUNDLED,
            RuleType::NOT_NDIM,
            RuleType::HAS_ACCESS_PATTERN,
            (GCFG_ENABLE_DSD_WITH_LB.value() ? RuleType::NOT_SERIALIZE_DESERIALIZE_NODE : RuleType::NOT_DYNAMIC_SHAPE),
            RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
            RuleType::NO_BINDING_INPUT_REUSE,
            RuleType::NO_SHARED_INPUT_IN_BUNDLE,
            RuleType::NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
            RuleType::HAS_UNIQUE_INPUTS,
            RuleType::SUPPORTED_LOGICAL_OPERATION,
            RuleType::NO_PRODUCER_BPG_CYCLE,
            RuleType::NOT_MAX_CANDIDATES};
}

const std::vector<RuleType> TpcProducersExpander::getStopRules() const
{
    return {RuleType::EXACTLY_2_TPC_PRODUCERS};
}

const CandidateFinderPtr TpcProducersExpander::getFinder(const TensorPtr& operand, HabanaGraph& graph) const
{
    return std::make_shared<DfsProducersFinder>(operand, graph);
}

TpcProducersIterativeExpander::TpcProducersIterativeExpander(const TensorPtr& operand,
                                                             const BundlePtr& bundle,
                                                             HabanaGraph&     graph)
: OperandExpander(operand, bundle, graph, getFinder(operand, graph), getValidityRules(), getStopRules())
{
}

const std::vector<RuleType> TpcProducersIterativeExpander::getValidityRules() const
{
    return {RuleType::NOT_RUNS_ON_MME,
            RuleType::NOT_RUNS_ON_DMA,
            RuleType::NOT_BUNDLED,
            RuleType::NOT_NDIM,
            RuleType::HAS_ACCESS_PATTERN,
            (GCFG_ENABLE_DSD_WITH_LB.value() ? RuleType::NOT_SERIALIZE_DESERIALIZE_NODE : RuleType::NOT_DYNAMIC_SHAPE),
            RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
            RuleType::NO_BINDING_INPUT_REUSE,
            RuleType::NO_SHARED_INPUT_IN_BUNDLE,
            RuleType::NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
            RuleType::SUPPORTED_LOGICAL_OPERATION,
            RuleType::HAS_UNIQUE_INPUTS,
            RuleType::NO_PRODUCER_BPG_CYCLE};
}

const std::vector<RuleType> TpcProducersIterativeExpander::getStopRules() const
{
    return {RuleType::LAST_CANDIDATE_TPC};
}

const CandidateFinderPtr TpcProducersIterativeExpander::getFinder(const TensorPtr& operand, HabanaGraph& graph) const
{
    return std::make_shared<DfsProducersFinder>(operand, graph);
}

void TpcProducersIterativeExpander::addCandidatesToBundle()
{
    addBundleCandidates();
}

FirstTpcConsumerExpander::FirstTpcConsumerExpander(const TensorPtr& operand,
                                                   const BundlePtr& bundle,
                                                   HabanaGraph&     graph)
: OperandExpander(operand, bundle, graph, getFinder(operand, graph), getValidityRules(), getStopRules())
{
}

const std::vector<RuleType> FirstTpcConsumerExpander::getValidityRules() const
{
    return {RuleType::NOT_BUNDLED,
            RuleType::NOT_RUNS_ON_MME,
            RuleType::NOT_RUNS_ON_DMA,
            RuleType::NOT_NDIM,
            RuleType::HAS_ACCESS_PATTERN,
            (GCFG_ENABLE_DSD_WITH_LB.value() ? RuleType::NOT_SERIALIZE_DESERIALIZE_NODE : RuleType::NOT_DYNAMIC_SHAPE),
            RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
            RuleType::NO_BINDING_INPUT_REUSE,
            RuleType::NO_SHARED_INPUT_IN_BUNDLE,
            RuleType::NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
            RuleType::HAS_UNIQUE_INPUTS,
            RuleType::SUPPORTED_LOGICAL_OPERATION,
            RuleType::NO_CONSUMER_BPG_CYCLE,
            RuleType::NOT_MAX_CANDIDATES};
}

const std::vector<RuleType> FirstTpcConsumerExpander::getStopRules() const
{
    return {RuleType::LAST_CANDIDATE_TPC};
}

const CandidateFinderPtr FirstTpcConsumerExpander::getFinder(const TensorPtr& operand, HabanaGraph& graph) const
{
    HB_ASSERT_PTR(operand);
    return std::make_shared<DfsConsumersFinder>(operand, graph);
}

TpcConsumersIterativeExpander::TpcConsumersIterativeExpander(const TensorPtr& operand,
                                                             const BundlePtr& bundle,
                                                             HabanaGraph&     graph)
: OperandExpander(operand, bundle, graph, getFinder(operand, graph), getValidityRules(), getStopRules())
{
}

const std::vector<RuleType> TpcConsumersIterativeExpander::getValidityRules() const
{
    return {RuleType::NOT_RUNS_ON_MME,
            RuleType::NOT_RUNS_ON_DMA,
            RuleType::NOT_BUNDLED,
            RuleType::NOT_NDIM,
            RuleType::HAS_ACCESS_PATTERN,
            (GCFG_ENABLE_DSD_WITH_LB.value() ? RuleType::NOT_SERIALIZE_DESERIALIZE_NODE : RuleType::NOT_DYNAMIC_SHAPE),
            RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
            RuleType::NO_BINDING_INPUT_REUSE,
            RuleType::NO_SHARED_INPUT_IN_BUNDLE,
            RuleType::NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
            RuleType::SUPPORTED_LOGICAL_OPERATION,
            RuleType::HAS_UNIQUE_INPUTS,
            RuleType::NO_CONSUMER_BPG_CYCLE};
}

const std::vector<RuleType> TpcConsumersIterativeExpander::getStopRules() const
{
    return {RuleType::LAST_CANDIDATE_TPC};
}

const CandidateFinderPtr TpcConsumersIterativeExpander::getFinder(const TensorPtr& operand, HabanaGraph& graph) const
{
    return std::make_shared<DfsConsumersFinder>(operand, graph);
}

void TpcConsumersIterativeExpander::addCandidatesToBundle()
{
    addBundleCandidates();
}

NoProducersTpcConsumersIterativeExpander::NoProducersTpcConsumersIterativeExpander(const TensorPtr& operand,
                                                                                   const BundlePtr& bundle,
                                                                                   HabanaGraph&     graph)
: TpcConsumersIterativeExpander(operand, bundle, graph)
{
    injectDeployRules({RuleType::NO_MME_PRODUCERS_BUNDLED});
}

NoConsumersTpcProducersIterativeExpander::NoConsumersTpcProducersIterativeExpander(const TensorPtr& operand,
                                                                                   const BundlePtr& bundle,
                                                                                   HabanaGraph&     graph)
: TpcProducersIterativeExpander(operand, bundle, graph)
{
    injectDeployRules({RuleType::NO_MME_CONSUMERS_BUNDLED});
}

IterativeFirstMmeProducerExpander::IterativeFirstMmeProducerExpander(const TensorPtr& operand,
                                                                     const BundlePtr& bundle,
                                                                     HabanaGraph&     graph)
: OperandExpander(operand, bundle, graph, getFinder(operand, graph), getValidityRules(), getStopRules())
{
}

const std::vector<RuleType> IterativeFirstMmeProducerExpander::getValidityRules() const
{
    return {RuleType::NOT_RUNS_ON_TPC,
            RuleType::NOT_RUNS_ON_DMA,
            RuleType::NOT_BUNDLED,
            RuleType::NOT_NDIM,
            RuleType::HAS_ACCESS_PATTERN,
            RuleType::NOT_DYNAMIC_SHAPE,
            RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
            RuleType::NO_BINDING_INPUT_REUSE,
            RuleType::NO_SHARED_INPUT_IN_BUNDLE,
            RuleType::NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
            RuleType::HAS_UNIQUE_INPUTS,
            RuleType::SUPPORTED_LOGICAL_OPERATION,
            RuleType::SUPPORTED_ATTENTION_SEED,
            RuleType::NO_PRODUCER_BPG_CYCLE,
            RuleType::NOT_MAX_CANDIDATES,
            RuleType::MME_OUTPUTS_LARGER_THAN_INPUTS};
}

const std::vector<RuleType> IterativeFirstMmeProducerExpander::getStopRules() const
{
    return {RuleType::LAST_CANDIDATE_MME};
}

const CandidateFinderPtr IterativeFirstMmeProducerExpander::getFinder(const TensorPtr& operand,
                                                                      HabanaGraph&     graph) const
{
    HB_ASSERT_PTR(operand);
    return std::make_shared<DfsProducersFinder>(operand, graph);
}

void IterativeFirstMmeProducerExpander::addCandidatesToBundle()
{
    addBundleCandidates();
}

IterativeFirstMmeConsumerExpander::IterativeFirstMmeConsumerExpander(const TensorPtr& operand,
                                                                     const BundlePtr& bundle,
                                                                     HabanaGraph&     graph)
: OperandExpander(operand, bundle, graph, getFinder(operand, graph), getValidityRules(), getStopRules())
{
}

const std::vector<RuleType> IterativeFirstMmeConsumerExpander::getValidityRules() const
{
    return {RuleType::NOT_BUNDLED,
            RuleType::NOT_RUNS_ON_TPC,
            RuleType::NOT_RUNS_ON_DMA,
            RuleType::NOT_NDIM,
            RuleType::HAS_ACCESS_PATTERN,
            RuleType::NOT_DYNAMIC_SHAPE,
            RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
            RuleType::NO_BINDING_INPUT_REUSE,
            RuleType::NO_SHARED_INPUT_IN_BUNDLE,
            RuleType::NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
            RuleType::HAS_UNIQUE_INPUTS,
            RuleType::SUPPORTED_LOGICAL_OPERATION,
            RuleType::SUPPORTED_ATTENTION_SEED,
            RuleType::NO_CONSUMER_BPG_CYCLE,
            RuleType::NOT_MAX_CANDIDATES,
            RuleType::MME_INPUTS_LARGER_THAN_OUTPUTS};
}

const std::vector<RuleType> IterativeFirstMmeConsumerExpander::getStopRules() const
{
    return {RuleType::LAST_CANDIDATE_MME};
}

const CandidateFinderPtr IterativeFirstMmeConsumerExpander::getFinder(const TensorPtr& operand,
                                                                      HabanaGraph&     graph) const
{
    HB_ASSERT_PTR(operand);
    return std::make_shared<DfsConsumersFinder>(operand, graph);
}

void IterativeFirstMmeConsumerExpander::addCandidatesToBundle()
{
    addBundleCandidates();
}

}  // namespace gc::layered_brain::bundler
