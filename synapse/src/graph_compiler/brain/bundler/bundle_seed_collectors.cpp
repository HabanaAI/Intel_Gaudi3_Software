#include "bundle_seed_collectors.h"
#include "bundle_expanders.h"
#include "bundler/bundle_expander.h"
#include "habana_graph.h"
#include "bundle_break_rules.h"
#include "layered_brain_bundle.h"
#include "log_manager.h"
#include "brain_conf.h"
#include <algorithm>
#include <iterator>
namespace gc::layered_brain::bundler
{
bool MmeNodeCollector::bundleOnlyIsolatedMmeNodes()
{
    // when in hybrid mode and multi seed collection is disabled, leave multi mme bundles for pipeline management
    return GCFG_ENABLE_LB_HYBRID_MODE.value() && !GCFG_ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS.value();
}

MmeNodeCollector::MmeNodeCollector(HabanaGraph& graph)
: SeedCollector(graph),
  m_commonRules(m_ruleLibrary.getRules({RuleType::NOT_NDIM,
                                        RuleType::NOT_BUNDLED,
                                        RuleType::HAS_ACCESS_PATTERN,
                                        (GCFG_ENABLE_DSD_WITH_LB.value() ? RuleType::NOT_SERIALIZE_DESERIALIZE_NODE : RuleType::NOT_DYNAMIC_SHAPE),
                                        RuleType::RUNS_ON_MME,
                                        RuleType::HAS_UNIQUE_INPUTS}))
{
    if (bundleOnlyIsolatedMmeNodes())
    {
        injectBreakRules({RuleType::IS_ISOLATED});
    }
}

void MmeNodeCollector::injectBreakRules(const std::vector<RuleType>& rules)
{
    for (const auto& rule : rules)
    {
        m_breakRules.push_back(m_ruleLibrary.getRule(rule));
    }
}

bool MmeNodeCollector::passesCommonRules(const NodePtr& n) const
{
    HB_ASSERT_PTR(n);
    LOG_DEBUG(LB_BUNDLER, "Running common rules for candidate node: {} ", n->getNodeName());
    RuleContext ctx(m_graph, n);
    for (const auto& rule : m_commonRules)
    {
        if (!rule->apply(ctx))
        {
            LOG_DEBUG(LB_BUNDLER,
                      "Candidate {}:{} failed common rule: {}",
                      n->getNodeTypeStr(),
                      n->getNodeName(),
                      rule->name());
            return false;
        }
    }
    return true;
}

bool MmeNodeCollector::passesBreakRules(const NodePtr& n, const BundlePtr& bundleSeed) const
{
    HB_ASSERT_PTR(n);
    LOG_DEBUG(LB_BUNDLER, "Running break rules for candidate node: {} ", n->getNodeName());
    auto        dummyCandidatesVector = std::make_shared<std::vector<NodePtr>>();
    RuleContext ctx(m_graph, n, bundleSeed, dummyCandidatesVector);
    for (const auto& rule : m_breakRules)
    {
        if (!rule->apply(ctx))
        {
            LOG_DEBUG(LB_BUNDLER,
                      "Candidate {}:{} failed break rule: {}",
                      n->getNodeTypeStr(),
                      n->getNodeName(),
                      rule->name());
            return false;
        }
    }
    return true;
}

static void printExpanders(const TensorPtr& operand, const NodePtr& node, const BundleExpanders& expanders)
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(LB_BUNDLER)) return;
    std::stringstream ss;
    for (const auto& ex : expanders)
    {
        ss << fmt::format("{}{}", toString(ex->getType()), ex != expanders.back() ? ", " : "");
    }
    LOG_DEBUG(LB_BUNDLER,
              "Expanding {}:{} operand {} with expanders: {}",
              node->getNodeTypeStr(),
              node->getNodeName(),
              operand->getName(),
              ss.str());
}

template<typename... T>
static void
appendExpanders(BundleExpanders& expanders, const TensorPtr& operand, const BundlePtr& bundle, HabanaGraph& graph)
{
    BundleExpanders expandersToAdd = {std::make_shared<T>(operand, bundle, graph)...};
    expanders.splice(expanders.end(), expandersToAdd);
}

template<bool IsIterative>
BundleExpanders MmeNodeCollector::getExpanders(const NodeSet& seed, const BundlePtr& bundle) const
{
    using ConsumerExpander = std::conditional_t<IsIterative, TpcConsumersIterativeExpander, FirstTpcConsumerExpander>;
    using ConditionalConsumerExpander =
        std::conditional_t<IsIterative, NoProducersTpcConsumersIterativeExpander, NoProducersFirstTpcConsumerExpander>;
    HB_ASSERT_PTR(bundle);
    HB_ASSERT(!seed.empty(), "Expecting non-empty seed");
    BundleExpanders               expanders;
    std::unordered_set<TensorPtr> visitedOperands;
    for (const auto& mme : seed)
    {
        for (const auto& operand : mme->getInputs())
        {
            if (!operand) continue;
            const auto visited = !visitedOperands.insert(operand).second;
            if (visited) continue;
            BundleExpanders inputExpanders;
            if constexpr (IsIterative)
            {
                if (shouldAddConditionalExpander(mme))
                {
                    appendExpanders<NoConsumersTpcProducersIterativeExpander>(inputExpanders, operand, bundle, m_graph);
                }
                else
                {
                    appendExpanders<TpcProducersIterativeExpander>(inputExpanders, operand, bundle, m_graph);
                }
            }
            else
            {
                appendExpanders<TpcProducersExpander, FirstTpcProducerExpander>(inputExpanders,
                                                                                operand,
                                                                                bundle,
                                                                                m_graph);
            }

            printExpanders(operand, mme, inputExpanders);
            expanders.splice(expanders.end(), inputExpanders);
        }

        HB_ASSERT(mme->getNumOutputs() > 0,
                  "Expecting at least one output for node {}[{}]",
                  mme->getNodeName(),
                  mme->getNodeTypeStr());
        const auto& outputA = mme->getOutput(0);
        const auto  visited = !visitedOperands.insert(outputA).second;
        if (visited) continue;
        BundleExpanders consumerExpanders;
        if (shouldAddConditionalExpander(mme))
        {
            appendExpanders<ConditionalConsumerExpander>(consumerExpanders, outputA, bundle, m_graph);
        }
        else
        {
            appendExpanders<ConsumerExpander>(consumerExpanders, outputA, bundle, m_graph);
        }

        if (!consumerExpanders.empty())
        {
            printExpanders(outputA, mme, consumerExpanders);
            expanders.insert(expanders.end(), consumerExpanders.begin(), consumerExpanders.end());
        }
    }
    applyExpandersOrder(expanders);
    return expanders;
}

void MmeNodeCollector::applyExpandersOrder(BundleExpanders& expanders) const
{
    // sort expander list giving precedence to consumers/producers based on cfg
    expanders.sort([](const BundleExpanderPtr& lhs, const BundleExpanderPtr& rhs) {
        return GCFG_ENABLE_LB_PREFER_CONSUMERS.value() ? lhs->isProducerExpander() < rhs->isProducerExpander()
                                                       : lhs->isProducerExpander() > rhs->isProducerExpander();
    });
}

bool MmeNodeCollector::shouldAddConditionalExpander(const NodePtr& n) const
{
    RuleContext       ctx(m_graph, n);
    static const auto isConvFamilyRule = m_ruleLibrary.getRule(RuleType::IS_CONV);
    static const auto isBgemmRule      = m_ruleLibrary.getRule(RuleType::IS_BATCH_GEMM);
    static const auto isGemmRule       = m_ruleLibrary.getRule(RuleType::IS_GEMM);
    HB_ASSERT(isConvFamilyRule->apply(ctx) || isBgemmRule->apply(ctx) || isGemmRule->apply(ctx),
              "Unexpected bundle seed {}",
              n->getNodeName());

    if (GCFG_LIMIT_CONV_BUNDLES_EXPANSION.value() && isConvFamilyRule->apply(ctx))
    {
        return true;
    }
    if (GCFG_LIMIT_GEMM_BUNDLES_EXPANSION.value() && (isBgemmRule->apply(ctx) || isGemmRule->apply(ctx)))
    {
        return true;
    }
    return false;
}

void MmeNodeCollector::logSeed(const BundlePtr& seed) const
{
    HB_ASSERT_PTR(seed);
    if (LOG_LEVEL_AT_LEAST_DEBUG(LB_BUNDLER))
    {
        LOG_DEBUG(LB_BUNDLER,
                  "Collected {} bundle seed index: {}, nodes ({}):",
                  toString(getType()),
                  seed->index(),
                  seed->getNodes().size());
        for (const auto& n : seed->getNodes())
        {
            LOG_DEBUG(LB_BUNDLER, "\t{} [{}]", n->getNodeName(), n->getNodeTypeStr());
        }
    }
}

std::vector<BundleAndExpanders> MmeNodeCollector::collect(bool iterative)
{
    if (getType() == Type::SINGLE_CONV && !GCFG_ENABLE_LAYERED_BRAIN_CONV_BUNDLING.value())
    {
        // Skipping conv bundle seed collection
        return {};
    }
    const std::string logContext(toString(getType()));
    SET_TEMP_LOG_CONTEXT(logContext);
    std::vector<BundleAndExpanders> ret;
    std::unordered_set<NodePtr>     visitedNodes;

    const auto nodes = m_graph.getNodes();
    for (const auto& n : nodes)
    {
        HB_ASSERT_PTR(n);
        const bool visited = !visitedNodes.insert(n).second;
        if (visited) continue;

        if (!passesCommonRules(n))
        {
            continue;
        }

        if (!passesBreakRules(n))
        {
            continue;
        }

        auto            bundle = Bundle::create(m_graph);
        BundleExpanders expanders {};
        if (iterative)
        {
            // add seed as bundle candidate
            bundle->addCandidate(n);
            expanders = getExpanders</*IsIterative*/ true>(bundle->getNodes(), bundle);
        }
        else
        {
            // add seed directly as a valid bundle node
            bundle->add(n);
            expanders = getExpanders</*IsIterative*/ false>(bundle->getNodes(), bundle);
        }
        ret.push_back(std::make_pair(bundle, expanders));
        logSeed(bundle);
    }
    return ret;
}

SingleBatchGemmCollector::SingleBatchGemmCollector(HabanaGraph& graph) : MmeNodeCollector(graph)
{
    injectBreakRules({RuleType::SUPPORTED_BATCH_GEMM});
}

SingleGemmCollector::SingleGemmCollector(HabanaGraph& graph) : MmeNodeCollector(graph)
{
    injectBreakRules({RuleType::SUPPORTED_GEMM});
}

SingleConvCollector::SingleConvCollector(HabanaGraph& graph) : MmeNodeCollector(graph)
{
    injectBreakRules({RuleType::IS_CONV});
}

MultiMmeCollector::MultiMmeCollector(HabanaGraph& graph) : MmeNodeCollector(graph)
{
    // TODO [SW-148708]: remove IS_VALID_SEED_SIZE once ticket is done
    injectBreakRules({RuleType::IS_VALID_SEED_SIZE,
                      RuleType::NO_ANCESTORS_IN_BUNDLE,
                      RuleType::SINGLE_SHARED_MME_OPERAND_IN_BUNDLE,
                      RuleType::NO_SHARED_MME_OPERAND_BPG_CYCLE});
}

std::vector<MultiMmeCollector::MmeGroup> MultiMmeCollector::gatherMmeGroups() const
{
    std::vector<MmeGroup> mmeGroups;
    uint64_t              groupId = 0;
    for (const auto& t : m_graph.getTensors())
    {
        if (!t || !t->isDataTensor()) continue;
        const auto&          consumers = m_graph.getTensorConsumers(t);
        std::vector<NodePtr> groupNodes;
        std::copy_if(consumers.begin(), consumers.end(), std::back_inserter(groupNodes), [](const auto& consumer) {
            if (!consumer) return false;
            return HabanaGraph::runsOnMME(consumer);
        });
        if (groupNodes.size() > 1)
        {
            mmeGroups.emplace_back(t, groupNodes, groupId++);
        }
    }
    return mmeGroups;
}

std::vector<BundleAndExpanders> MultiMmeCollector::collect(bool iterative)
{
    const std::string logContext(toString(getType()));
    SET_TEMP_LOG_CONTEXT(logContext);
    std::vector<BundleAndExpanders> ret {};

    if (!GCFG_ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS.value())
    {
        LOG_INFO(LB_BUNDLER, "Multi-MME seed collection is disabled");
        return ret;
    }

    std::vector<MmeGroup> mmeGroups = gatherMmeGroups();
    BundlePtr             bundle(nullptr);  // reuse bundles to reduce wasted bundleIndices
    while (!mmeGroups.empty())
    {
        // sort in ascending order w.r.t the num of MME sharing mmeGroups.sharedOperand
        std::stable_sort(mmeGroups.begin(), mmeGroups.end(), [](const MmeGroup& lhs, const MmeGroup& rhs) {
            return lhs.m_nodes.size() < rhs.m_nodes.size();
        });

        std::unordered_set<MmeGroup::Id> garbage;
        for (auto& mmeGroup : mmeGroups)
        {
            if (mmeGroup.m_nodes.size() <= 1)
            {
                // lazily throw small mme groups
                garbage.insert(mmeGroup.m_id);
                continue;
            }

            LOG_DEBUG(LB_BUNDLER,
                      "Handle mme group id: {}, size: {}, shared operand: {}",
                      mmeGroup.m_id,
                      mmeGroup.m_nodes.size(),
                      mmeGroup.m_sharedOperand->getName());
            if (!bundle) bundle = Bundle::create(m_graph);
            for (const auto& mme : mmeGroup.m_nodes)
            {
                if (passesCommonRules(mme) && passesBreakRules(mme, bundle))
                {
                    LOG_DEBUG(LB_BUNDLER, "pass common and break rules");
                    bundle->addCandidate(mme);
                }
            }

            // discard group if nothing was salvaged from it
            if (bundle->getNodes().empty())
            {
                LOG_DEBUG(LB_BUNDLER, "Empty seed, discard mme group");
                garbage.insert(mmeGroup.m_id);
                continue;
            }

            // remove seed node/s from mme group nodes
            auto& groupNodes = mmeGroup.m_nodes;
            groupNodes.erase(std::remove_if(groupNodes.begin(),
                                            groupNodes.end(),
                                            [&bundle](const auto& n) {
                                                return bundle->getNodes().find(n) != bundle->getNodes().end();
                                            }),
                             groupNodes.end());

            // if single mme bundle, leave for single node collectors or a different seed composition
            if (bundle->getNodes().size() == 1)
            {
                LOG_DEBUG(LB_BUNDLER, "Reject single node seed");
                bundle->rejectCandidates();
                continue;
            }

            // legit multi mme seed, create expanders for it
            BundleExpanders expanders {};
            if (iterative)
            {
                // bundle seed nodes remain candidates until passing initial composition evaluation
                expanders = getExpanders</*IsIterative*/ true>(bundle->getNodes(), bundle);
            }
            else
            {
                // in fwd progress candidates are immediately accepted to the bundle
                bundle->acceptCandidates();
                expanders = getExpanders</*IsIterative*/ false>(bundle->getNodes(), bundle);
            }

            logSeed(bundle);
            ret.emplace_back(bundle, expanders);
            bundle.reset();  // trigger new bundle creation after committing on a seed
        }

        // garbage collection of irrelevant groups
        mmeGroups.erase(
            std::remove_if(mmeGroups.begin(),
                           mmeGroups.end(),
                           [&garbage](const auto& group) { return garbage.find(group.m_id) != garbage.end(); }),
            mmeGroups.end());
    }
    return ret;
}

MultiGemmCollector::MultiGemmCollector(HabanaGraph& graph) : MultiMmeCollector(graph)
{
    injectBreakRules({RuleType::SUPPORTED_MULTI_GEMM_SEED});
}

AttentionCollector::AttentionCollector(HabanaGraph& graph)
: SeedCollector(graph),
  m_tpcSeedRules(m_ruleLibrary.getRules({RuleType::RUNS_ON_TPC,
                                         RuleType::NOT_BUNDLED,
                                         RuleType::NOT_NDIM,
                                         RuleType::HAS_ACCESS_PATTERN,
                                         RuleType::NOT_DYNAMIC_SHAPE,
                                         RuleType::NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
                                         RuleType::NO_BINDING_INPUT_REUSE,
                                         RuleType::HAS_UNIQUE_INPUTS}))
{
}

std::vector<BundleAndExpanders> AttentionCollector::collect(bool iterative)
{
    const std::string logContext(toString(getType()));
    SET_TEMP_LOG_CONTEXT(logContext);
    std::vector<BundleAndExpanders> ret {};

    if (!GCFG_ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS.value())
    {
        LOG_INFO(LB_BUNDLER, "Attention bundle seed collection is disabled");
        return ret;
    }

    BundlePtr bundle(nullptr);  // Reuse bundles to reduce wasted bundle-indices

    for (const auto& tpcSeed : m_graph.getNodes())
    {
        if (!tpcSeed || !passesTpcSeedRules(tpcSeed)) continue;

        if (!bundle)
        {
            bundle = Bundle::create(m_graph);
        }

        if (getAttentionBundle(bundle, tpcSeed))
        {
            logSeed(bundle);
            BundleExpanders expanders = {};
            ret.emplace_back(bundle, expanders);
            bundle.reset();  // Trigger a new bundle creation after committing on a seed
        }
    }
    return ret;
}

bool AttentionCollector::getAttentionBundle(const BundlePtr& bundle, const NodePtr& tpcSeed) const
{
    static constexpr unsigned NUM_MME_NODES_IN_ATTENTION_SEED = 2;

    for (const auto& input : tpcSeed->getInputs())
    {
        if (!input) continue;

        for (const auto& output : tpcSeed->getOutputs())
        {
            if (!output) continue;

            HB_ASSERT(bundle->getNodes().empty(), "Expected an empty bundle");

            bundle->addCandidate(tpcSeed);

            const auto producerExpander = std::make_shared<IterativeFirstMmeProducerExpander>(input, bundle, m_graph);
            const auto consumerExpander = std::make_shared<IterativeFirstMmeConsumerExpander>(output, bundle, m_graph);

            if (producerExpander->expand() && consumerExpander->expand())
            {
                const auto& bundleNodes = bundle->getNodes();
                const auto  numMMEs     = std::count_if(bundleNodes.begin(), bundleNodes.end(), [](const auto& n) {
                    return n && HabanaGraph::runsOnMME(n);
                });
                HB_ASSERT(numMMEs == NUM_MME_NODES_IN_ATTENTION_SEED,
                          "Expected {} MME nodes in attention seed",
                          NUM_MME_NODES_IN_ATTENTION_SEED);

                // Found a seed of MME -> [logicals...] -> TPC -> [logicals...] -> MME
                return true;
            }

            bundle->rejectCandidates();  // Try to expand from a different operand
        }
    }
    return false;
}

void AttentionCollector::logSeed(const BundlePtr& seed) const
{
    HB_ASSERT_PTR(seed);
    if (LOG_LEVEL_AT_LEAST_DEBUG(LB_BUNDLER))
    {
        LOG_DEBUG(LB_BUNDLER,
                  "Collected {} bundle seed index: {}, nodes ({}):",
                  toString(getType()),
                  seed->index(),
                  seed->getNodes().size());
        for (const auto& n : seed->getNodes())
        {
            LOG_DEBUG(LB_BUNDLER, "\t{} [{}]", n->getNodeName(), n->getNodeTypeStr());
        }
    }
}

bool AttentionCollector::passesTpcSeedRules(const NodePtr& n) const
{
    HB_ASSERT_PTR(n);
    LOG_DEBUG(LB_BUNDLER, "Running TPC seed rules for candidate node: {} ", n->getNodeName());
    RuleContext ctx(m_graph, n);
    for (const auto& rule : m_tpcSeedRules)
    {
        if (!rule->apply(ctx))
        {
            LOG_DEBUG(LB_BUNDLER,
                      "Candidate {}:{} failed rule: {}",
                      n->getNodeTypeStr(),
                      n->getNodeName(),
                      rule->name());
            return false;
        }
    }
    return true;
}

}  // namespace gc::layered_brain::bundler
