#include "bundle_evaluator.h"
#include "brain_data.h"
#include "defs.h"
#include "habana_graph.h"
#include "compilation_hal_reader.h"
#include "brain_conf.h"
#include <algorithm>
#include <limits>

namespace gc::layered_brain
{
std::string_view toString(const EvalResult& type)
{
    std::string_view ret;
    switch (type)
    {
        case EvalResult::UNEVALUATED:
            return "UNEVALUATED";
        case EvalResult::LOW:
            return "LOW";
        case EvalResult::VALID:
            return "VALID";
        case EvalResult::HIGH:
            return "HIGH";
        default:
            HB_ASSERT(false,
                      "Unexpected evaluation result type {}",
                      std::to_string(static_cast<EvalResultValue>(type)));
            return "";
    }
}

bool Evaluation::allMetricsValid() const
{
    return mmeBw.valid() && mmeUtil.valid() && (cacheUtil.valid() || cacheUtil.unevaluated()) && nofSlices.valid() &&
           (GCFG_ENABLE_EVALUATE_PERFORATION_UTIL.value()
                ? (perforationUtil.valid() || perforationUtil.unevaluated())
                : (perforationMultiplier.valid() || perforationMultiplier.unevaluated()));
}

std::string Evaluation::evalString() const
{
    return fmt::format("[nofSlices: {}, {}][mmeUtil: {:>3.2f} %, {}][cacheUtil: {:>3.2f} MB, {}][mmeBw: {:>4} GB/s, "
                       "{}][perforationMultiplier: {}, {}][perforationUtil: {:>3.2f} %, {}]",
                       nofSlices.value,
                       toString(nofSlices.eval),
                       mmeUtil.value * 100.0f,
                       toString(mmeUtil.eval),
                       bToMb(cacheUtil.value),
                       toString(cacheUtil.eval),
                       mmeBw.value,
                       toString(mmeBw.eval),
                       perforationMultiplier.value,
                       toString(perforationMultiplier.eval),
                       perforationUtil.value * 100.0f,
                       toString(perforationUtil.eval));
}

BundleEvaluator::BundleEvaluator(const BundlePtr&              bundle,
                                 const ConstSlicingDetailsPtr& slicingDetails,
                                 const MaxUtilPerMME&          maxUtilPerMME)
: m_bundle(bundle), m_slicingDetails(slicingDetails), m_mmeToMaxUtil(maxUtilPerMME)
{
    HB_ASSERT_PTR(bundle);
    HB_ASSERT_PTR(slicingDetails);
    acquireBundledMmeNodes(bundle);
}

MaxUtilPerMME BundleEvaluator::collectMaxUtils(const BundlePtr& bundle, const StrategyVector& strategies)
{
    MaxUtilPerMME maxUtilPerMME;
    HB_ASSERT(!strategies.empty(), "Expecting at least one slicing strategy");
    for (const auto& n : bundle->getNodes())
    {
        if (!HabanaGraph::runsOnMME(n)) continue;

        float maxUtil = 0.0f;
        for (const auto& s : strategies)
        {
            const auto qor = s->getNodeQORs(n);
            HB_ASSERT_PTR(qor);
            const auto util = qor->perfAttr.maxUtilization;
            maxUtil         = std::max(util, maxUtil);
        }
        maxUtilPerMME.insert(std::make_pair(n, maxUtil));
        LOG_DEBUG(LB_EVALUATOR,
                  "Max util for MME {} [{}]: {:>3.2f} %",
                  n->getNodeName(),
                  n->getNodeTypeStr(),
                  maxUtil * 100.0f);
    }
    return maxUtilPerMME;
}

const NodeSet& BundleEvaluator::getBundledMmeNodes() const
{
    return m_bundleMmeNodes;
}

EvalResult BundleEvaluator::evaluateNofSlices(uint64_t nofSlices)
{
    EvalResult res = EvalResult::UNEVALUATED;
    if (nofSlices < GCFG_LAYERED_BRAIN_BUNDLE_MIN_NOF_SLICES.value())
    {
        res = EvalResult::LOW;
    }
    else if (nofSlices <= GCFG_LAYERED_BRAIN_BUNDLE_MAX_NOF_SLICES.value())
    {
        res = EvalResult::VALID;
    }
    else  // nofSlices.value > GCFG_LAYERED_BRAIN_BUNDLE_MAX_NOF_SLICES.value()
    {
        res = EvalResult::HIGH;
    }
    return res;
}

ConstEvalPtr BundleEvaluator::postSlicingEvaluation(const HabanaGraph& slicedGraph) const
{
    SET_TEMP_LOG_CONTEXT(fmt::format("{} Bundle#{}", HLLOG_FUNC, m_bundle->index()));
    const ConstEvalPtr preSlicingEval = preSlicingEvaluation();
    const EvalPtr      eval           = std::make_shared<Evaluation>(*preSlicingEval);
    const auto&        bundleData     = getBundleData(slicedGraph);
    handleCacheUtil(eval->cacheUtil, bundleData.maxCacheUsageBytes(), slicedGraph.getHALReader()->getSRAMSizeInBytes());
    return eval;
}

ConstEvalPtr BundleEvaluator::preSlicingEvaluation() const
{
    SET_TEMP_LOG_CONTEXT(fmt::format("{} Bundle#{}", HLLOG_FUNC, m_bundle->index()));
    auto preSlicingEval = std::make_shared<Evaluation>();
    handleNofSlices(preSlicingEval->nofSlices);
    handleMmeBw(preSlicingEval->mmeBw);
    handleMmeUtil(preSlicingEval->mmeUtil);
    handlePerforationUtil(preSlicingEval->perforationUtil);
    handlePerforationMultiplier(preSlicingEval->perforationMultiplier);
    return preSlicingEval;
}

NodeSet BundleEvaluator::acquireBundledMmeNodes(const BundlePtr& bundle)
{
    HB_ASSERT_PTR(bundle);
    for (const auto& n : bundle->getNodes())
    {
        if (HabanaGraph::runsOnMME(n))
        {
            m_bundleMmeNodes.insert(n);
        }
    }
    return m_bundleMmeNodes;
}

const BundleData& BundleEvaluator::getBundleData(const HabanaGraph& slicedGraph) const
{
    const auto* lbData(slicedGraph.getLayeredBrainData());
    HB_ASSERT_PTR(lbData);
    HB_ASSERT(lbData->isLayeredBrainBundle(m_bundle->index()),
              "bundleId {} is not a layered brain bundle",
              m_bundle->index());
    const auto it = lbData->m_bundleData.find(m_bundle->index());
    HB_ASSERT(it != lbData->m_bundleData.end(), "Expecting bundle data for bundle {}", m_bundle->index());
    return it->second;
}

void BundleEvaluator::handleNofSlices(Evaluation::BundleMetric<uint64_t>& nofSlices) const
{
    nofSlices.value = m_slicingDetails->getNofSlices();
    nofSlices.eval  = evaluateNofSlices(nofSlices.value);
    LOG_TRACE(LB_EVALUATOR, "Number of bundle slices: {}, result: {}", nofSlices.value, toString(nofSlices.eval));
}

EvalResult BundleEvaluator::evaluateMmeBw(float bw) const
{
    EvalResult res;
    if (bw <= GCFG_LAYERED_BRAIN_MAX_VALID_MME_BW.value())
    {
        res = EvalResult::VALID;
    }
    else
    {
        res = EvalResult::HIGH;
    }
    return res;
}

void BundleEvaluator::handleMmeBw(Evaluation::BundleMetric<float>& mmeBw) const
{
    if (getBundledMmeNodes().empty())
    {
        // no reason in evaluating if there aren't any MMEs in bundle
        mmeBw.eval = EvalResult::UNEVALUATED;
        return;
    }

    mmeBw.value = 0;
    NodePtr maxBwMme;
    for (const auto& mme : getBundledMmeNodes())
    {
        const auto curBw = m_slicingDetails->getMmeBw(mme);
        if (curBw >= mmeBw.value)
        {
            mmeBw.value = curBw;
            mmeBw.eval  = evaluateMmeBw(curBw);
            maxBwMme    = mme;
        }

        if (!mmeBw.valid())
        {
            // stop on first offending mme
            mmeBw.offendingNode = mme;
            break;
        }
    }

    HB_ASSERT(!mmeBw.unevaluated(), "Expecting MME BW to be evaluated");

    LOG_TRACE(LB_EVALUATOR,
              "{} MME bandwidth: {} GB/s, result: {}, node: {}[{}]",
              mmeBw.valid() ? "Highest valid" : "Offending",
              mmeBw.value,
              toString(mmeBw.eval),
              mmeBw.valid() ? maxBwMme->getNodeName() : mmeBw.offendingNode->getNodeName(),
              mmeBw.valid() ? maxBwMme->getNodeTypeStr() : mmeBw.offendingNode->getNodeTypeStr());
}

EvalResult BundleEvaluator::evaluateMmeUtil(const NodePtr& mme, float util) const
{
    EvalResult res;
    HB_ASSERT_PTR(mme);
    const auto it = m_mmeToMaxUtil.find(mme);
    HB_ASSERT(it != m_mmeToMaxUtil.end(),
              "Expecting {} [{}] in mme to max util map",
              mme->getNodeName(),
              mme->getNodeTypeStr());
    const float maxMmeUtil = it->second;
    HB_ASSERT(maxMmeUtil > 0, "Unexpected max util 0 for {} [{}]", mme->getNodeName(), mme->getNodeTypeStr());
    const float utilRelativeToMax = util / maxMmeUtil;
    LOG_DEBUG(LB_EVALUATOR,
              "{} [{}] util/maxUtil: {:>4.3f}, min valid: {:>4.3f}",
              mme->getNodeName(),
              mme->getNodeTypeStr(),
              utilRelativeToMax,
              GCFG_LAYERED_BRAIN_MIN_VALID_MME_UTIL_RATIO.value());
    if (utilRelativeToMax >= GCFG_LAYERED_BRAIN_MIN_VALID_MME_UTIL_RATIO.value())
    {
        res = EvalResult::VALID;
    }
    else
    {
        res = EvalResult::LOW;
    }
    return res;
}

void BundleEvaluator::handleCacheUtil(Evaluation::BundleMetric<uint64_t>& cacheUtil,
                                      uint64_t                            cacheUsed,
                                      uint64_t                            cacheSize) const
{
    cacheUtil.value = cacheUsed;
    cacheUtil.eval  = cacheUtil.value < cacheSize ? EvalResult::VALID : EvalResult::HIGH;
    LOG_TRACE(LB_EVALUATOR,
              "Cache utilization: {:>4.2f}/{} MB, result: {}",
              bToMb(cacheUtil.value),
              bToMb(cacheSize),
              toString(cacheUtil.eval));
}

void BundleEvaluator::handleMmeUtil(Evaluation::BundleMetric<float>& mmeUtil) const
{
    static constexpr float MAX_MME_UTIL = 1.0;
    if (getBundledMmeNodes().empty())
    {
        // no reason in evaluating if there aren't any MMEs in bundle
        mmeUtil.eval = EvalResult::UNEVALUATED;
        return;
    }

    mmeUtil.value = MAX_MME_UTIL;
    NodePtr minUtilMme;
    for (const auto& mme : getBundledMmeNodes())
    {
        const auto curUtil = m_slicingDetails->getMmeUtil(mme);
        if (curUtil <= mmeUtil.value)
        {
            mmeUtil.value = curUtil;
            mmeUtil.eval  = evaluateMmeUtil(mme, curUtil);
            minUtilMme    = mme;
        }

        if (!mmeUtil.valid())
        {
            mmeUtil.offendingNode = mme;
            break;
        }
    }

    HB_ASSERT(!mmeUtil.unevaluated(), "Expecting MME BW to be evaluated");

    LOG_TRACE(LB_EVALUATOR,
              "{} MME utilization: {:>3.2f}%, result: {}, node: {}[{}]",
              mmeUtil.valid() ? "Lowest valid" : "Offending",
              mmeUtil.value * 100.0f,
              toString(mmeUtil.eval),
              mmeUtil.valid() ? minUtilMme->getNodeName() : mmeUtil.offendingNode->getNodeName(),
              mmeUtil.valid() ? minUtilMme->getNodeTypeStr() : mmeUtil.offendingNode->getNodeTypeStr());
}

void BundleEvaluator::handlePerforationMultiplier(Evaluation::BundleMetric<uint64_t>& perforationMultiplier) const
{
    perforationMultiplier.eval = EvalResult::UNEVALUATED;
    for (const auto& n : m_bundle->getNodes())
    {
        if (n->isLogicalOperation()) continue;
        const auto multiplier = m_slicingDetails->getNodePerforationBvdMultiplier(n);
        if (!multiplier.has_value()) continue;
        perforationMultiplier.value = multiplier.value();
        if (multiplier.value() % CompilationHalReader::getHalReader()->getNumDcores() == 0)
        {
            perforationMultiplier.eval = EvalResult::VALID;
        }
        else
        {
            perforationMultiplier.offendingNode = n;
            perforationMultiplier.eval          = EvalResult::LOW;
            return;
        }
    }
}

void BundleEvaluator::handlePerforationUtil(Evaluation::BundleMetric<float>& perforationUtil) const
{
    perforationUtil.eval = EvalResult::UNEVALUATED;
    for (const auto& n : m_bundle->getNodes())
    {
        if (n->isLogicalOperation()) continue;
        const auto perfUtil = m_slicingDetails->getNodePerforationUtil(n);
        if (!perfUtil.has_value()) continue;
        perforationUtil.value = perfUtil.value();
        if (perforationUtil.value >= GCFG_PERFORATION_UTILIZATION_THRESHOLD.value())
        {
            perforationUtil.eval = EvalResult::VALID;
        }
        else
        {
            perforationUtil.offendingNode = n;
            perforationUtil.eval          = EvalResult::LOW;
            return;
        }
    }
}
}  // namespace gc::layered_brain