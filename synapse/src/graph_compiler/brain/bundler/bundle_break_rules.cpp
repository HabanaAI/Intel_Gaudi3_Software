#include "bundle_break_rules.h"
#include "bundle_paths_validation.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "physical_memory_ops_nodes.h"
#include "log_manager.h"
#include "utils.h"
#include <algorithm>
#include <memory>
#include <unordered_set>
#include "brain/brain_conf.h"

namespace gc::layered_brain::bundler
{
template<typename RuleType>
static RulePtr makeRulePtr()
{
    return std::make_shared<RuleType>();
}

template<typename... RuleType>
static std::vector<RulePtr> makeRulePtrs()
{
    return {makeRulePtr<RuleType>()...};
}

#define DEFINE_SINGLE_RULE(_singleRule, _type, _func)                                                                  \
    class _singleRule : public Rule                                                                                    \
    {                                                                                                                  \
    public:                                                                                                            \
        _singleRule() : Rule(_func, #_type, RuleType::_type) {}                                                        \
                                                                                                                       \
    protected:                                                                                                         \
    }

#define DEFINE_MULTI_AND_RULE(_multiRule, _type, _rules...)                                                            \
    class _multiRule : public CompositeAndRule                                                                         \
    {                                                                                                                  \
    public:                                                                                                            \
        _multiRule() : CompositeAndRule(_rules, #_type, RuleType::_type) {}                                            \
                                                                                                                       \
    protected:                                                                                                         \
    }

#define DEFINE_MULTI_OR_RULE(_multiRule, _type, _rules...)                                                             \
    class _multiRule : public CompositeOrRule                                                                          \
    {                                                                                                                  \
    public:                                                                                                            \
        _multiRule() : CompositeOrRule(_rules, #_type, RuleType::_type) {}                                             \
                                                                                                                       \
    protected:                                                                                                         \
    }

typedef bool (*RuleFunction)(const RuleContext&);
class Rule : public IRule
{
public:
    Rule(RuleFunction f, const std::string& name, RuleType type) : IRule(name, type), m_rule(f) {}

    bool apply(const RuleContext& ctx) const override
    {
        LOG_TRACE(LB_BUNDLER, "Apply rule {}", name());
        return m_rule(ctx);
    }

    virtual ~Rule() = default;

protected:
    const RuleFunction m_rule;
};

class ICompositeRule : public IRule
{
public:
    ICompositeRule(const std::vector<RulePtr>& rules, const std::string& name, RuleType type)
    : IRule(name, type), m_rules(rules)
    {
    }
    virtual ~ICompositeRule()                        = default;
    virtual bool apply(const RuleContext& ctx) const = 0;

protected:
    const std::vector<RulePtr> m_rules;
};

class CompositeOrRule : public ICompositeRule
{
public:
    CompositeOrRule(const std::vector<RulePtr>& rules, const std::string& name, RuleType type)
    : ICompositeRule(rules, name, type)
    {
    }
    virtual ~CompositeOrRule() = default;

    bool apply(const RuleContext& ctx) const override
    {
        const std::string logCtx = fmt::format("CompositeOrRule: {}", name());
        SET_TEMP_LOG_CONTEXT(logCtx);
        return std::any_of(m_rules.begin(), m_rules.end(), [&ctx](const auto& rule) { return rule->apply(ctx); });
    }
};

class CompositeAndRule : public ICompositeRule
{
public:
    CompositeAndRule(const std::vector<RulePtr>& rules, const std::string& name, RuleType type)
    : ICompositeRule(rules, name, type)
    {
    }

    virtual ~CompositeAndRule() = default;
    bool apply(const RuleContext& ctx) const override
    {
        const std::string logCtx = fmt::format("CompositeAndRule: {}", name());
        SET_TEMP_LOG_CONTEXT(logCtx);
        return std::all_of(m_rules.begin(), m_rules.end(), [&ctx](const auto& rule) { return rule->apply(ctx); });
    }
};

// internal rule/helper function forward declarations
static bool noMMEConsumersBundled(const RuleContext& ctx);
static bool noMMEProducersBundled(const RuleContext& ctx);
template<std::size_t TpcNr>
bool        hasNrTpcCandidates(const RuleContext& ctx);
static bool lastCandidateTpc(const RuleContext& ctx);
static bool lastCandidateMme(const RuleContext& ctx);
template<std::size_t NrCandidates>
bool notExceedsNrCandidates(const RuleContext& ctx);
template<std::size_t NrSeedNodes>
static bool notExceedsNrSeedNodes(const RuleContext& ctx);
static bool noConsumerBPGCycle(const RuleContext& ctx);
static bool noSharedMMEOperandBPGCycle(const RuleContext& ctx);
static bool isConsumerBPGCycle(const RuleContext& ctx);
static bool noProducerBPGCycle(const RuleContext& ctx);
static bool isProducerBPGCycle(const RuleContext& ctx);
template<std::size_t MaxSharedOperands>
static bool notExceedsNrSharedMMEOperands(const RuleContext& ctx);
static bool noMultipleOutputsInBundle(const RuleContext& ctx);
static bool noSharedInputInBundle(const RuleContext& ctx);
static bool isSharedInputInBundle(const RuleContext& ctx);
static bool isIsolated(const RuleContext& ctx);
static bool noAncestorsInBundle(const RuleContext& ctx);
static bool isBatchGemm(const RuleContext& ctx);
static bool isValidBatchGemm(const RuleContext& ctx);
static bool isGemm(const RuleContext& ctx);
static bool isConv(const RuleContext& ctx);
static bool notRunsOnDMA(const RuleContext& ctx);
static bool runsOnMME(const RuleContext& ctx);
static bool notRunsOnMME(const RuleContext& ctx);
static bool notRunsOnTPC(const RuleContext& ctx);
static bool runsOnTPC(const RuleContext& ctx);
static bool isUniqueInputs(const RuleContext& ctx);
static bool supportedLogicalOperation(const RuleContext& ctx);
static bool isLogicalOperation(const RuleContext& ctx);
static bool isNoOverlapOrOffsetInAccessPattern(const RuleContext& ctx);
static bool isOverlapOrOffsetInAccessPattern(const RuleContext& ctx);
static bool isNotBindingInputReuse(const RuleContext& ctx);
static bool isBindingInputReuse(const RuleContext& ctx);
static bool hasAccessPattern(const RuleContext& ctx);
static bool isNotBundled(const RuleContext& ctx);
static bool isBundled(const RuleContext& ctx);
static bool isLowRank(const RuleContext& ctx);
static bool isStaticShapeNode(const RuleContext& ctx);
static bool noTPCConsumers(const RuleContext& ctx);
static bool notSerializeDeserializeNode(const RuleContext& ctx);
static bool mmeInputsLargerThanOutputs(const RuleContext& ctx);
static bool mmeOutputsLargerThanInputs(const RuleContext& ctx);

// clang-format off

// Basic rule definitions
DEFINE_SINGLE_RULE(LowRank, NOT_NDIM, isLowRank);
DEFINE_SINGLE_RULE(IsStaticShapeNode, NOT_DYNAMIC_SHAPE, isStaticShapeNode);
DEFINE_SINGLE_RULE(IsNotBundled, NOT_BUNDLED, isNotBundled);
DEFINE_SINGLE_RULE(IsBundled, BUNDLED, isBundled);
DEFINE_SINGLE_RULE(HasAccessPattern, HAS_ACCESS_PATTERN, hasAccessPattern);
DEFINE_SINGLE_RULE(IsBindingInputReuse, BINDING_INPUT_REUSE, isBindingInputReuse);
DEFINE_SINGLE_RULE(IsNotBindingInputReuse, NO_BINDING_INPUT_REUSE, isNotBindingInputReuse);
DEFINE_SINGLE_RULE(IsOverlapOrOffsetInAccessPattern, OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN, isOverlapOrOffsetInAccessPattern);
DEFINE_SINGLE_RULE(NoOverlapOrOffsetInAccessPattern, NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN, isNoOverlapOrOffsetInAccessPattern);
DEFINE_SINGLE_RULE(HasUniqueInputs, HAS_UNIQUE_INPUTS, isUniqueInputs);
DEFINE_SINGLE_RULE(SupportedLogicalOperation, SUPPORTED_LOGICAL_OPERATION, supportedLogicalOperation);
DEFINE_SINGLE_RULE(IsLogicalOperation, IS_LOGICAL_OPERATION, isLogicalOperation);
DEFINE_SINGLE_RULE(RunsOnTPC, RUNS_ON_TPC, runsOnTPC);
DEFINE_SINGLE_RULE(NotRunsOnTPC, NOT_RUNS_ON_TPC, notRunsOnTPC);
DEFINE_SINGLE_RULE(RunsOnMME, RUNS_ON_MME, runsOnMME);
DEFINE_SINGLE_RULE(NotRunsOnMME, NOT_RUNS_ON_MME, notRunsOnMME);
DEFINE_SINGLE_RULE(NotRunsOnDMA, NOT_RUNS_ON_DMA, notRunsOnDMA);
DEFINE_SINGLE_RULE(IsGemm, IS_GEMM, isGemm);
DEFINE_SINGLE_RULE(IsConv, IS_CONV, isConv);
DEFINE_SINGLE_RULE(IsBatchGemm, IS_BATCH_GEMM, isBatchGemm);
DEFINE_SINGLE_RULE(ValidBatchGemm, VALID_BATCH_GEMM, isValidBatchGemm);
DEFINE_SINGLE_RULE(Isolated, IS_ISOLATED, isIsolated);
DEFINE_SINGLE_RULE(NoAncestorsInBundle, NO_ANCESTORS_IN_BUNDLE, noAncestorsInBundle);
DEFINE_SINGLE_RULE(SharedInputInBundle, HAS_SHARED_INPUT_IN_BUNDLE, isSharedInputInBundle);
DEFINE_SINGLE_RULE(SingleSharedMMEOperandInBundle, SINGLE_SHARED_MME_OPERAND_IN_BUNDLE, notExceedsNrSharedMMEOperands<1>);
DEFINE_SINGLE_RULE(NoSharedInputInBundle, NO_SHARED_INPUT_IN_BUNDLE, noSharedInputInBundle);
DEFINE_SINGLE_RULE(NoMultipleOutputsInBundle, NO_MULTIPLE_OUTPUTS_IN_BUNDLE, noMultipleOutputsInBundle);
DEFINE_SINGLE_RULE(NoProducerBPGCycle, NO_PRODUCER_BPG_CYCLE, noProducerBPGCycle);
DEFINE_SINGLE_RULE(NoConsumerBPGCycle, NO_CONSUMER_BPG_CYCLE, noConsumerBPGCycle);
DEFINE_SINGLE_RULE(NoSharedMMEOperandBPGCycle, NO_SHARED_MME_OPERAND_BPG_CYCLE, noSharedMMEOperandBPGCycle);
DEFINE_SINGLE_RULE(NotMaxCandidates, NOT_MAX_CANDIDATES, notExceedsNrCandidates<7>);
DEFINE_SINGLE_RULE(IsValidSeedSize, IS_VALID_SEED_SIZE, notExceedsNrSeedNodes<3>);
DEFINE_SINGLE_RULE(LastCandidateTpc, LAST_CANDIDATE_TPC, lastCandidateTpc);
DEFINE_SINGLE_RULE(LastCandidateMme, LAST_CANDIDATE_MME, lastCandidateMme);
DEFINE_SINGLE_RULE(TwoTpcProducers, EXACTLY_2_TPC_PRODUCERS, hasNrTpcCandidates<2>);
DEFINE_SINGLE_RULE(NoMMEProducersBundled, NO_MME_PRODUCERS_BUNDLED, noMMEProducersBundled);
DEFINE_SINGLE_RULE(NoMMEConsumersBundled, NO_MME_CONSUMERS_BUNDLED, noMMEConsumersBundled);
DEFINE_SINGLE_RULE(NoTPCConsumers, NO_TPC_CONSUMERS, noTPCConsumers);
DEFINE_SINGLE_RULE(NotSerializeDeserializeNode, NOT_SERIALIZE_DESERIALIZE_NODE, notSerializeDeserializeNode);
DEFINE_SINGLE_RULE(MmeInputsLargerThanOutputs, MME_INPUTS_LARGER_THAN_OUTPUTS, mmeInputsLargerThanOutputs);
DEFINE_SINGLE_RULE(MmeOutputsLargerThanInputs, MME_OUTPUTS_LARGER_THAN_INPUTS, mmeOutputsLargerThanInputs);

// Composite rule definitions
DEFINE_MULTI_AND_RULE(SupportedGemm,       SUPPORTED_GEMM,         makeRulePtrs<IsGemm, HasUniqueInputs>());
DEFINE_MULTI_AND_RULE(SupportedBatchGemm,  SUPPORTED_BATCH_GEMM,   makeRulePtrs<IsBatchGemm, HasUniqueInputs, ValidBatchGemm>());
DEFINE_MULTI_OR_RULE(SupportedMultiGemmSeed, SUPPORTED_MULTI_GEMM_SEED, makeRulePtrs<SupportedGemm, SupportedBatchGemm, IsConv>());
DEFINE_MULTI_OR_RULE(SupportedAttentionSeed, SUPPORTED_ATTENTION_SEED, makeRulePtrs<SupportedGemm, SupportedBatchGemm, IsLogicalOperation>());

// clang-format on

static bool noMMEProducersBundled(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.bundle);
    const auto& bundleNodes = ctx.bundle->getNodes();
    for (const auto& n : bundleNodes)
    {
        if (!HabanaGraph::runsOnMME(n)) continue;
        for (const auto& t : n->getInputs())
        {
            const auto& producer = ctx.graph.getTensorProducer(t);
            if (producer && bundleNodes.find(producer) != bundleNodes.end())
            {
                return false;
            }
        }
    }
    return true;
}

static bool noMMEConsumersBundled(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.bundle);
    const auto& bundleNodes = ctx.bundle->getNodes();
    for (const auto& n : bundleNodes)
    {
        if (!HabanaGraph::runsOnMME(n)) continue;
        for (const auto& t : n->getOutputs())
        {
            const auto& consumers = ctx.graph.getTensorConsumers(t);
            if (std::any_of(consumers.begin(), consumers.end(), [&bundleNodes](const auto& n) {
                    return bundleNodes.find(n) != bundleNodes.end();
                }))
            {
                return false;
            }
        }
    }
    return true;
}

static bool noTPCConsumers(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    const auto& realConsumers = ctx.graph.getNodeRealConsumers(ctx.node, Node::TENSOR_TYPE_DATA);
    return std::all_of(realConsumers.begin(), realConsumers.end(), [&](const NodePtr& n) {
        return !ctx.graph.runsOnTPC(n);
    });
}

static uint64_t maxTensorSizeInElements(const TensorVector& tensors)
{
    uint64_t maxSize = 0;
    for (const auto& t : tensors)
    {
        if (!t) continue;
        maxSize = std::max(maxSize, t->getDenseSizeInElements());
    }
    return maxSize;
}

static bool mmeInputsLargerThanOutputs(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);

    if (!HabanaGraph::runsOnMME(ctx.node)) return true;

    const auto maxInputSize  = maxTensorSizeInElements(ctx.node->getInputs());
    const auto maxOutputSize = maxTensorSizeInElements(ctx.node->getOutputs());

    return maxInputSize > maxOutputSize;
}

static bool mmeOutputsLargerThanInputs(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);

    if (!HabanaGraph::runsOnMME(ctx.node)) return true;

    const auto maxInputSize  = maxTensorSizeInElements(ctx.node->getInputs());
    const auto maxOutputSize = maxTensorSizeInElements(ctx.node->getOutputs());

    return maxOutputSize > maxInputSize;
}

static bool isLowRank(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    return !ctx.node->hasHighRankOperand();
}

static bool isStaticShapeNode(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    return !ctx.node->isDynamicShape();
}

static bool isBundled(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    if (ctx.node->getNodeAnnotation().bundleInfo.is_set())
    {
        LOG_DEBUG(LB_BUNDLER,
                  "Node: {} already belongs to bundle {}",
                  ctx.node->getNodeName(),
                  ctx.node->getNodeAnnotation().bundleInfo->bundleIndex);
        return true;
    }
    return false;
}

static bool isNotBundled(const RuleContext& ctx)
{
    return !isBundled(ctx);
}

static bool hasAccessPattern(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");

    const auto nodeAp = ctx.node->getNodeAccessPattern();
    if (!nodeAp)
    {
        LOG_DEBUG(LB_BUNDLER,
                  "Node: {}, type: {} does not have an access pattern",
                  ctx.node->getNodeName(),
                  ctx.node->getNodeTypeStr());
        return false;
    }

    // In rare cases, not all tensors are mapped to a node access pattern. Currently the known case is when a fuser
    // fuses several nodes with shape tensors and the resulting node has less outputs than shape tensors.
    for (const auto& ts : {ctx.node->getInputs(), ctx.node->getOutputs()})
    {
        for (const auto& t : ts)
        {
            if (t && !nodeAp->hasAccessPattern(t))
            {
                LOG_DEBUG(LB_BUNDLER,
                          "Node: {}, type: {}, operand: {} has no access pattern",
                          ctx.node->getNodeName(),
                          ctx.node->getNodeTypeStr(),
                          t->getName());
                return false;
            }
        }
    }
    return true;
}

static bool isBindingInputReuse(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    if (ctx.node->hasBindingInputReuse())
    {
        LOG_DEBUG(LB_BUNDLER, "{} has bindining input reuse", ctx.node->getNodeName());
        return true;
    }
    return false;
}

static bool isNotBindingInputReuse(const RuleContext& ctx)
{
    return !isBindingInputReuse(ctx);
}

static bool isOverlapOrOffsetInAccessPattern(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    const auto& node = ctx.node;
    const auto  ap   = node->getNodeAccessPattern();
    HB_ASSERT_PTR(ap);
    const auto& resolution = ap->getNodeResolution();
    for (const auto& t : node->getOperands())
    {
        if (!t || t->isShapeTensor()) continue;
        const auto& granularity = ap->getTensorGranularity(t);
        const auto& overlap     = ap->getTensorOverlap(t);
        access_pattern::TensorTile fullTensorTile(t->getDim(),
                                                  t->getAllNSizesInElements(),
                                                  access_pattern::TensorTile::Offset(t->getDim(), 0));
        const auto&                nodeTile = ap->getNodeTile(t, fullTensorTile);
        for (auto dim = 0; dim < t->getDim(); ++dim)
        {
            const auto indexSpaceDim = ap->getIndexSpaceDim(t, dim);
            auto       dimResolution = resolution.at(indexSpaceDim);
            if (overlap.geometry[dim] != 0 && dimResolution > 1)
            {
                LOG_DEBUG(LB_BUNDLER,
                          "node {}, type {} has overlap {} on tensor "
                          "{}, dim {}",
                          node->getNodeName(),
                          node->getNodeTypeStr(),
                          overlap.geometry[dim],
                          t->getName(),
                          dim);
                return true;
            }

            if (granularity.offset[dim] != 0)
            {
                LOG_DEBUG(LB_BUNDLER,
                          "node {}, type {} has offset {} on tensor {} dim {}",
                          node->getNodeName(),
                          node->getNodeTypeStr(),
                          granularity.offset[dim],
                          t->getName(),
                          dim);
                return true;
            }

            if (nodeTile.geometry.at(indexSpaceDim) != dimResolution)
            {
                // There are index-space elements that are not mapped to any tensor element.
                // In this case, mapping the tensor tile to a node tile may produce wrong results.
                // For example: a pad node with 17 index space elements, input tensor of 64 elements,
                // output tensor of 65 elements, each index space element maps to 4 tensor elements.
                // The kernel adds pad of 1 at the end so the last index space maps to padding area
                // in the input and within the tensor in the output.
                LOG_DEBUG(LB_BUNDLER,
                          "node {}, type {} has padding on tensor {} dim {}",
                          node->getNodeName(),
                          node->getNodeTypeStr(),
                          t->getName(),
                          dim);
                return true;
            }
        }
    }
    return false;
}

static bool isNoOverlapOrOffsetInAccessPattern(const RuleContext& ctx)
{
    return !isOverlapOrOffsetInAccessPattern(ctx);
}

static bool isUniqueInputs(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    const auto inputs = ctx.node->getInputs();

    std::unordered_set<TensorPtr> inputSet {};
    inputSet.reserve(inputs.size());
    for (const auto& input : inputs)
    {
        if (!input || !input->isDataTensor()) continue;
        bool visited = !inputSet.insert(input).second;
        if (visited)
        {
            LOG_DEBUG(LB_BUNDLER,
                      "node: {}, type: {} - input tensor {} has multiple operand roles",
                      ctx.node->getNodeName(),
                      ctx.node->getNodeTypeStr(),
                      input->getName());
            return false;
        }
    }
    return true;
}

static bool supportedLogicalOperation(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);

    if (!ctx.node->isLogicalOperation()) return true;  // Not logical => no problem

    const auto numInputs  = ctx.node->getNumInputsDataTensors();
    const auto numOutputs = ctx.node->getNumOutputsDataTensors();

    // Logical operations require bundling all their producers and consumers (otherwise, the forks and joins would force
    // memcopies). Since currently this is not guaranteed, this is only possible if the logical operation has a single
    // non-shape input and single non-shape output.
    return numInputs == 1 && numOutputs == 1;
}

static bool isLogicalOperation(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    return ctx.node->isLogicalOperation();
}

static bool runsOnTPC(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node, "");
    return HabanaGraph::runsOnTPC(ctx.node);
}

static bool notRunsOnTPC(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node, "");
    return !runsOnTPC(ctx);
}

static bool runsOnMME(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node, "");
    return HabanaGraph::runsOnMME(ctx.node);
}

static bool notRunsOnMME(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node, "");
    return !runsOnMME(ctx);
}

static bool notRunsOnDMA(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node, "");
    return !ctx.node->isDma();
}

static bool isGemm(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    if (!Node::isGemmNode(ctx.node))
    {
        LOG_DEBUG(LB_BUNDLER,
                  "node: {}, type: {} is not a gemm node",
                  ctx.node->getNodeName(),
                  ctx.node->getNodeTypeStr());
        return false;
    }
    return true;
}

static bool isConv(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    if (!GCFG_ENABLE_LAYERED_BRAIN_CONV_BUNDLING.value())
    {
        // if conv bundling is disabled, prevent multi seed collector from gathering such nodes
        return false;
    }

    if (ctx.node->getNodeType() != Node::TYPE_CONVOLUTION && ctx.node->getNodeType() != Node::TYPE_DEDW &&
        ctx.node->getNodeType() != Node::TYPE_DEDX && ctx.node->getNodeType() != Node::TYPE_TRANSPOSED_DEDX)
    {
        LOG_DEBUG(LB_BUNDLER,
                  "node: {}, type: {} is not a fwd/bwd convolution node",
                  ctx.node->getNodeName(),
                  ctx.node->getNodeTypeStr());
        return false;
    }
    return true;
}

static bool isValidBatchGemm(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    // [SW-85839] Ranks of inputs must be equal
    const auto* pBgemm = dynamic_cast<BatchGemmNode*>(ctx.node.get());
    HB_ASSERT(pBgemm != nullptr, "expecting successful ptr cast");

    // TODO [SW-75607][SW-120978] - Handle bgemm without batch dimensions
    if (pBgemm->getOutput(0)->getDim() <= DIM_GEMM_BATCH)
    {
        LOG_DEBUG(SRAM_SLICE, "batch-gemm without any batch dimensions is not supported yet.");
        return false;
    }

    // TODO [SW-120978]: Support broadcast bgemm with inequal input operand ranks
    const auto& operandA               = pBgemm->getInput(0);
    const auto& operandB               = pBgemm->getInput(1);
    bool        equalInputOperandRanks = operandA->getDim() == operandB->getDim();
    if (!equalInputOperandRanks)
    {
        LOG_DEBUG(LB_BUNDLER,
                  "unsupported bgemm {} with inputOperandA.rank {} != inputOperandB.rank {}",
                  pBgemm->getNodeName(),
                  operandA->getDim(),
                  operandB->getDim());
        return false;
    }

    return true;
}

static bool isBatchGemm(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    const auto type = ctx.node->getNodeType();
    if (type != Node::TYPE_BATCH_GEMM && type != Node::TYPE_MASKED_BATCH_GEMM)
    {
        LOG_DEBUG(LB_BUNDLER,
                  "node: {}, type: {} is not a masked batch gemm / batch gemm node",
                  ctx.node->getNodeName(),
                  ctx.node->getNodeTypeStr());
        return false;
    }
    return true;
}

static bool isIsolated(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr, "");
    const auto& graph  = ctx.graph;
    const auto& n      = ctx.node;
    const auto& inputs = n->getInputs();
    for (const auto& t : inputs)
    {
        if (!t) continue;
        for (const auto& consumer : graph.getTensorConsumers(t))
        {
            if (!consumer || consumer == n || !HabanaGraph::runsOnMME(consumer)) continue;
            if (!graph.isAncestor(consumer, n) && !graph.isAncestor(n, consumer)) return false;
        }
    }
    return true;
}

static bool noAncestorsInBundle(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    HB_ASSERT_PTR(ctx.bundle);

    const auto& graph     = ctx.graph;
    const auto& candidate = ctx.node;
    const auto& bundle    = ctx.bundle;

    for (const auto& n : bundle->getNodes())
    {
        if (n == candidate) continue;
        if (graph.isAncestor(candidate, n) || graph.isAncestor(n, candidate))
        {
            return false;
        }
    }
    return true;
}

static bool isSharedInputInBundle(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr && ctx.bundle != nullptr && ctx.curCandidates != nullptr, "");
    auto bundleCopy = ctx.bundle->getNodes();
    bundleCopy.insert(ctx.curCandidates->begin(), ctx.curCandidates->end());
    const auto& graph = ctx.graph;
    for (const TensorPtr& candidateInput : ctx.node->getInputs())
    {
        if (!candidateInput) continue;
        if (candidateInput->isNonScratchpadAuxTensor())
        {
            // There is no problem sharing aux tensors between producers as long as they are not scratch-pad aux tensors
            continue;
        }
        for (const NodePtr& inputConsumer : graph.getTensorConsumers(candidateInput))
        {
            if (bundleCopy.find(inputConsumer) != bundleCopy.end())
            {
                LOG_DEBUG(LB_BUNDLER,
                          "bundle candidate {} shares input tensor {} with node {} "
                          "which is already in the bundle",
                          ctx.node->getNodeName(),
                          candidateInput->getName(),
                          inputConsumer->getNodeName());
                return true;
            }
        }
    }
    return false;
}

template<std::size_t MaxSharedOperands>
static bool notExceedsNrSharedMMEOperands(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    HB_ASSERT_PTR(ctx.bundle);
    const auto& graph           = ctx.graph;
    const auto& candidate       = ctx.node;
    auto        bundleNodesCopy = ctx.bundle->getNodes();
    bundleNodesCopy.insert(candidate);

    std::unordered_set<TensorPtr> sharedInputs;
    for (const auto& n : bundleNodesCopy)
    {
        if (!HabanaGraph::runsOnMME(n)) continue;
        for (const auto& input : n->getInputs())
        {
            if (!input || !input->isDataTensor()) continue;

            const auto consumers = graph.getTensorConsumers(input);
            // count mme consumers of candidate input in {candidate, bundled mme nodes}
            const auto nMmeConsumers =
                std::count_if(consumers.begin(), consumers.end(), [&bundleNodesCopy](const auto& n) {
                    return n && HabanaGraph::runsOnMME(n) && bundleNodesCopy.find(n) != bundleNodesCopy.end();
                });
            if (nMmeConsumers > 1)
            {
                sharedInputs.insert(input);
            }
        }
    }

    return sharedInputs.size() <= MaxSharedOperands;
}

static bool noSharedInputInBundle(const RuleContext& ctx)
{
    return !isSharedInputInBundle(ctx);
}

static bool noMultipleOutputsInBundle(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr && ctx.bundle != nullptr && ctx.curCandidates != nullptr, "");
    const auto& cand       = ctx.node;
    const auto& graph      = ctx.graph;
    auto        bundleCopy = ctx.bundle->getNodes();
    bundleCopy.insert(ctx.curCandidates->begin(), ctx.curCandidates->end());
    const auto& outputs = cand->getOutputs();
    const auto  bundledOutputNr =
        std::count_if(outputs.begin(), outputs.end(), [&graph, &cand, &bundleCopy](const auto& t) {
            const auto outConsumers = graph.getTensorConsumers(t);
            return std::any_of(outConsumers.begin(), outConsumers.end(), [&cand, &bundleCopy](const auto& consumer) {
                return bundleCopy.find(consumer) != bundleCopy.end();
            });
        });
    return bundledOutputNr <= 1;
}

static bool isProducerBPGCycle(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr && ctx.bundle != nullptr && ctx.curCandidates != nullptr, "");
    const auto& prod       = ctx.node;
    auto        bundleCopy = ctx.bundle->getNodes();
    bundleCopy.insert(ctx.curCandidates->begin(), ctx.curCandidates->end());
    const auto& graph = ctx.graph;

    std::vector<TensorPtr> stitchCandidates;
    for (const auto& out : prod->getOutputs())
    {
        if (!out) continue;
        const auto outConsumers = graph.getTensorConsumers(out);

        if (std::none_of(outConsumers.begin(), outConsumers.end(), [&bundleCopy](const auto& consumer) {
                return bundleCopy.find(consumer) != bundleCopy.end();
            }))
        {
            continue;
        }
        stitchCandidates.push_back(out);
    };

    const auto closesBPGCycle = [&](const TensorPtr& t) {
        BundlePathsValidation validator(graph);
        bool                  validPath = validator.validateProducerPaths(prod, t, bundleCopy, {});
        if (!validPath)
        {
            LOG_DEBUG(LB_BUNDLER,
                      "adding candidate: {}, type: {} with stitched tensor: {} to the bundle introduces a BPG cycle",
                      prod->getNodeName(),
                      prod->getNodeType(),
                      t->getName());
            return true;
        }
        return false;
    };

    return std::any_of(stitchCandidates.begin(), stitchCandidates.end(), closesBPGCycle);
}

static bool noProducerBPGCycle(const RuleContext& ctx)
{
    return !isProducerBPGCycle(ctx);
}

static bool isConsumerBPGCycle(const RuleContext& ctx)
{
    HB_ASSERT(ctx.node != nullptr && ctx.bundle != nullptr && ctx.curCandidates != nullptr, "");
    const auto& consumer   = ctx.node;
    auto        bundleCopy = ctx.bundle->getNodes();
    bundleCopy.insert(ctx.curCandidates->begin(), ctx.curCandidates->end());
    const auto& graph = ctx.graph;

    std::vector<TensorPtr> stitchCandidates;
    for (const auto& in : consumer->getInputs())
    {
        if (!in) continue;
        const auto inProducer = graph.getTensorProducer(in);
        if (!inProducer || bundleCopy.find(inProducer) == bundleCopy.end()) continue;
        stitchCandidates.push_back(in);
    };

    const auto closesBPGCycle = [&](const TensorPtr& t) {
        BundlePathsValidation validator(graph);
        bool                  validPath = validator.validateConsumerPaths(consumer, t, bundleCopy);
        if (!validPath)
        {
            LOG_DEBUG(LB_BUNDLER,
                      "adding candidate: {}, type: {} with stitched tensor: {} to the bundle introduces a BPG cycle",
                      consumer->getNodeName(),
                      consumer->getNodeType(),
                      t->getName());
            return true;
        }
        return false;
    };

    return std::any_of(stitchCandidates.begin(), stitchCandidates.end(), closesBPGCycle);
}

static bool noConsumerBPGCycle(const RuleContext& ctx)
{
    return !isConsumerBPGCycle(ctx);
}

static bool noSharedMMEOperandBPGCycle(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    HB_ASSERT_PTR(ctx.bundle);
    const auto& graph       = ctx.graph;
    const auto& candidate   = ctx.node;
    const auto& bundle      = ctx.bundle;
    const auto& bundleNodes = bundle->getNodes();

    TensorPtr sharedMMEOperand;
    for (const auto& t : candidate->getInputs())
    {
        if (!t || !t->isDataTensor()) continue;
        const auto& consumers = graph.getTensorConsumers(t);
        if (std::any_of(consumers.begin(), consumers.end(), [bundleNodes](const auto& n) {
                return HabanaGraph::runsOnMME(n) && bundleNodes.find(n) != bundleNodes.end();
            }))
        {
            sharedMMEOperand = t;
            break;
        }
    }
    if (!sharedMMEOperand) return true;
    BundlePathsValidation validator(graph);

    if (!validator.validateConsumerPaths(candidate, sharedMMEOperand, bundleNodes))
    {
        LOG_DEBUG(LB_BUNDLER,
                  "adding candidate: {}, type: {} with shared operand: {} to the seed introduces a BPG cycle",
                  candidate->getNodeName(),
                  candidate->getNodeType(),
                  sharedMMEOperand->getName());
        return false;
    }
    return true;
}

template<std::size_t MaxNrSeedNodes>
bool notExceedsNrSeedNodes(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    HB_ASSERT_PTR(ctx.bundle);
    return ctx.bundle->getNodes().size() + 1 /* candidate */ <= MaxNrSeedNodes;
}

template<std::size_t NrCandidates>
bool notExceedsNrCandidates(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.curCandidates);
    unsigned countCandidate = ctx.node ? 1 : 0;
    return (ctx.curCandidates->size() + countCandidate) <= NrCandidates;
}

template<>
bool notExceedsNrCandidates<0>(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.curCandidates);
    return ctx.curCandidates->empty();
}

static bool lastCandidateTpc(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.curCandidates);
    return !ctx.curCandidates->empty() && HabanaGraph::runsOnTPC(ctx.curCandidates->back());
}

static bool lastCandidateMme(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.curCandidates);
    return !ctx.curCandidates->empty() && HabanaGraph::runsOnMME(ctx.curCandidates->back());
}

template<std::size_t TpcNr>
bool hasNrTpcCandidates(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.curCandidates);
    const auto& candidates = *ctx.curCandidates;
    return TpcNr == std::count_if(candidates.begin(), candidates.end(), [](const auto& candidate) {
               return candidate && HabanaGraph::runsOnTPC(candidate);
           });
}

static bool notSerializeDeserializeNode(const RuleContext& ctx)
{
    HB_ASSERT_PTR(ctx.node);
    auto memOpNode = std::dynamic_pointer_cast<TPCPhysicalMemoryOpNode>(ctx.node);
    if (memOpNode &&
            (memOpNode->getDynamicMemoryOpType() == DMA_OP_SERIALIZE ||
             memOpNode->getDynamicMemoryOpType() == DMA_OP_DESERIALIZE))
        return false;
    return true;
}

// clang-format off
RuleLibrary::RuleLibrary()
{
    // single rules
    appendRule(makeRulePtr<LowRank>());
    appendRule(makeRulePtr<IsStaticShapeNode>());
    appendRule(makeRulePtr<IsNotBundled>());
    appendRule(makeRulePtr<IsBundled>());
    appendRule(makeRulePtr<HasAccessPattern>());
    appendRule(makeRulePtr<IsBindingInputReuse>());
    appendRule(makeRulePtr<IsNotBindingInputReuse>());
    appendRule(makeRulePtr<IsOverlapOrOffsetInAccessPattern>());
    appendRule(makeRulePtr<NoOverlapOrOffsetInAccessPattern>());
    appendRule(makeRulePtr<HasUniqueInputs>());
    appendRule(makeRulePtr<SupportedLogicalOperation>());
    appendRule(makeRulePtr<IsLogicalOperation>());
    appendRule(makeRulePtr<RunsOnTPC>());
    appendRule(makeRulePtr<NotRunsOnTPC>());
    appendRule(makeRulePtr<RunsOnMME>());
    appendRule(makeRulePtr<NotRunsOnMME>());
    appendRule(makeRulePtr<NotRunsOnDMA>());
    appendRule(makeRulePtr<IsGemm>());
    appendRule(makeRulePtr<IsConv>());
    appendRule(makeRulePtr<IsBatchGemm>());
    appendRule(makeRulePtr<ValidBatchGemm>());
    appendRule(makeRulePtr<Isolated>());
    appendRule(makeRulePtr<NoAncestorsInBundle>());
    appendRule(makeRulePtr<NoMultipleOutputsInBundle>());
    appendRule(makeRulePtr<NoSharedInputInBundle>());
    appendRule(makeRulePtr<SingleSharedMMEOperandInBundle>());
    appendRule(makeRulePtr<SharedInputInBundle>());
    appendRule(makeRulePtr<NoProducerBPGCycle>());
    appendRule(makeRulePtr<NoConsumerBPGCycle>());
    appendRule(makeRulePtr<NoSharedMMEOperandBPGCycle>());
    appendRule(makeRulePtr<NotMaxCandidates>());
    appendRule(makeRulePtr<IsValidSeedSize>());
    appendRule(makeRulePtr<LastCandidateTpc>());
    appendRule(makeRulePtr<LastCandidateMme>());
    appendRule(makeRulePtr<TwoTpcProducers>());
    appendRule(makeRulePtr<NoMMEProducersBundled>());
    appendRule(makeRulePtr<NoMMEConsumersBundled>());
    appendRule(makeRulePtr<NoTPCConsumers>());
    appendRule(makeRulePtr<NotSerializeDeserializeNode>());
    appendRule(makeRulePtr<MmeInputsLargerThanOutputs>());
    appendRule(makeRulePtr<MmeOutputsLargerThanInputs>());

    // multi rules
    appendRule(makeRulePtr<SupportedGemm>());
    appendRule(makeRulePtr<SupportedBatchGemm>());
    appendRule(makeRulePtr<SupportedMultiGemmSeed>());
    appendRule(makeRulePtr<SupportedAttentionSeed>());
}
// clang-format on

const RuleLibrary& RuleLibrary::instance()
{
    static RuleLibrary lib;
    return lib;
}

RulePtr RuleLibrary::getRule(RuleType type) const
{
    const auto it = m_factoryMap.find(type);
    HB_ASSERT(it != m_factoryMap.end(), "Missing rule for type {}", type);
    return it->second;
}

std::vector<RulePtr> RuleLibrary::getRules(const std::vector<RuleType>& types) const
{
    std::vector<RulePtr> rules;
    for (const auto& type : types)
    {
        const auto rule = getRule(type);
        if (rule) rules.push_back(rule);
    }
    return rules;
}

void RuleLibrary::appendRule(const RulePtr& rule)
{
    HB_ASSERT_PTR(rule);
    m_factoryMap.insert(std::make_pair(rule->type(), rule));
}

}  // namespace gc::layered_brain::bundler
