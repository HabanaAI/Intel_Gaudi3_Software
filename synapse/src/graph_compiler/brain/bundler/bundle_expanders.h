#pragma once

#include "bundle_candidate_finder.h"
#include "habana_graph.h"
#include "node.h"
#include "bundle_expander.h"

namespace gc::layered_brain::bundler
{
class OperandExpander : public BundleExpander
{
public:
    OperandExpander(const TensorPtr&            operand,
                    const BundlePtr&            bundle,
                    HabanaGraph&                graph,
                    const CandidateFinderPtr&   finder,
                    const std::vector<RuleType> validityRules,
                    const std::vector<RuleType> stopRules);
    virtual ~OperandExpander() = default;
    BundlePtr    expand() override;
    virtual Type getType() const override = 0;

protected:
    /**
     * @brief Append additional rules to the existing deploy rules.
     *        The deploy rules determine whether the expander will
     *        attempt epanding the bundle or not run.
     */
    void injectDeployRules(const std::vector<RuleType>& deployRules);

    void                                   adjustCandidatesContainer(const NodePtr& candidate, const NodePtr& link);
    RuleContext                            getCurCtx(const NodePtr& candidate) const;
    std::vector<ExpansionCandidate>&       getCandidates();
    const std::vector<ExpansionCandidate>& getCandidates() const;
    virtual void                           addCandidatesToBundle();
    void                                   addBundleNodes();
    void                                   addBundleCandidates();
    bool                                   skipRejectedTraversalPath(const NodePtr& link) const;

    /**
     * @brief Execute rules to determine whether expander should yield or
     *        keep trying to expand.
     */
    bool stop() const;

    /**
     * @brief Execute rules to determine whether a bundle expansion candidate is valid.
     *
     */
    bool valid(const NodePtr& candidate) const;

    const TensorPtr            m_operand;
    const std::vector<RulePtr> m_validityRules;
    const std::vector<RulePtr> m_stopRules;

private:
    /**
     * @brief Either execute deploy rules to check whether bundle expander can attempt expanding
     *        the bundle or return previously cached result.
     */
    bool canDeploy() const;

    /**
     * @brief Execute deploy rules to determine whether expander will attempt running.
     *
     */
    bool runDeployRules() const;

    std::vector<ExpansionCandidate> m_expansionCandidates;
    std::vector<RulePtr>            m_deployRules;
    mutable std::optional<bool>     m_deployed;
};

class FirstTpcProducerExpander : public OperandExpander
{
public:
    FirstTpcProducerExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    Type getType() const override { return Type::FIRST_TPC_PRODUCER; };

protected:
    const std::vector<RuleType> getValidityRules() const;
    const std::vector<RuleType> getStopRules() const;
    const CandidateFinderPtr    getFinder(const TensorPtr& operand, HabanaGraph& graph) const;
};

class TpcProducersExpander : public OperandExpander
{
public:
    TpcProducersExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    Type getType() const override { return Type::TPC_PRODUCERS; };

private:
    const std::vector<RuleType> getValidityRules() const;
    const std::vector<RuleType> getStopRules() const;
    const CandidateFinderPtr    getFinder(const TensorPtr& operand, HabanaGraph& graph) const;
};

class TpcProducersIterativeExpander : public OperandExpander
{
public:
    TpcProducersIterativeExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    Type getType() const override { return Type::ITERATIVE_TPC_PRODUCERS; };

private:
    const std::vector<RuleType> getValidityRules() const;
    const std::vector<RuleType> getStopRules() const;
    const CandidateFinderPtr    getFinder(const TensorPtr& operand, HabanaGraph& graph) const;
    void                        addCandidatesToBundle() override;
};

class FirstTpcConsumerExpander : public OperandExpander
{
public:
    FirstTpcConsumerExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    Type getType() const override { return Type::FIRST_TPC_CONSUMER; };

protected:
    const std::vector<RuleType> getValidityRules() const;
    const std::vector<RuleType> getStopRules() const;
    const CandidateFinderPtr    getFinder(const TensorPtr& operand, HabanaGraph& graph) const;
};

class NoProducersFirstTpcConsumerExpander : public FirstTpcConsumerExpander
{
public:
    NoProducersFirstTpcConsumerExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    virtual Type getType() const override { return Type::NO_PRODUCERS_TPC_CONSUMER; };
};

class TpcConsumersIterativeExpander : public OperandExpander
{
public:
    TpcConsumersIterativeExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    Type getType() const override { return Type::ITERATIVE_TPC_CONSUMERS; };

protected:
    const std::vector<RuleType> getValidityRules() const;
    const std::vector<RuleType> getStopRules() const;
    const CandidateFinderPtr    getFinder(const TensorPtr& operand, HabanaGraph& graph) const;
    void                        addCandidatesToBundle() override;
};

class NoProducersTpcConsumersIterativeExpander : public TpcConsumersIterativeExpander
{
public:
    NoProducersTpcConsumersIterativeExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    virtual Type getType() const override { return Type::NO_PRODUCERS_ITERATIVE_TPC_CONSUMERS; };
};

class NoConsumersTpcProducersIterativeExpander : public TpcProducersIterativeExpander
{
public:
    NoConsumersTpcProducersIterativeExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    virtual Type getType() const override { return Type::NO_CONSUMERS_ITERATIVE_TPC_PRODUCERS; };
};

class IterativeFirstMmeProducerExpander : public OperandExpander
{
public:
    IterativeFirstMmeProducerExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    Type getType() const override { return Type::ITERATIVE_FIRST_MME_PRODUCER; };

protected:
    const std::vector<RuleType> getValidityRules() const;
    const std::vector<RuleType> getStopRules() const;
    const CandidateFinderPtr    getFinder(const TensorPtr& operand, HabanaGraph& graph) const;
    void                        addCandidatesToBundle() override;
};

class IterativeFirstMmeConsumerExpander : public OperandExpander
{
public:
    IterativeFirstMmeConsumerExpander(const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph);
    Type getType() const override { return Type::ITERATIVE_FIRST_MME_CONSUMER; };

protected:
    const std::vector<RuleType> getValidityRules() const;
    const std::vector<RuleType> getStopRules() const;
    const CandidateFinderPtr    getFinder(const TensorPtr& operand, HabanaGraph& graph) const;
    void                        addCandidatesToBundle() override;
};

}  // namespace gc::layered_brain::bundler
