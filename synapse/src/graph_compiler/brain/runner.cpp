#include "runner.h"

#include "brain_data.h"
#include "bundle_evaluator.h"
#include "bundler/bundle_seed_collector.h"
#include "bundler/bundlers.h"
#include "log_manager.h"
#include "scheduler/scheduler.h"
#include "slicer/slicer.h"
#include "memory_management/bundle_cache_manager.h"

namespace gc::layered_brain
{
// Controlled by global conf. This enum just give names to the available modes.
enum BrainFlowConfig
{
    SINGLE_PASS = 0,
    ITERATIVE   = 1,
};

class ScopedLogSuppression final
{
public:
    ScopedLogSuppression();
    ~ScopedLogSuppression();

    ScopedLogSuppression(const ScopedLogSuppression& other) = delete;
    ScopedLogSuppression(ScopedLogSuppression&& other)      = delete;
    ScopedLogSuppression& operator=(const ScopedLogSuppression& other) = delete;
    ScopedLogSuppression& operator=(ScopedLogSuppression&& other) = delete;

private:
    std::map<synapse::LogManager::LogType, int /*prev log level*/> m_origLoggingLevels;

    static constexpr int SUPPRESSED_LOG_LEVEL = 4 /*error*/;
};

ScopedLogSuppression::ScopedLogSuppression()
{
    if (GCFG_ENABLE_LAYERED_BRAIN_FULL_LOGGING.value()) return;
    auto& logMngr = synapse::LogManager::instance();
    for (uint32_t loggerIdx = 0; loggerIdx < hl_logger::getNbLoggers<synapse::LogManager::LogType>(); ++loggerIdx)
    {
        const synapse::LogManager::LogType logger = static_cast<synapse::LogManager::LogType>(loggerIdx);
        const auto                         level  = logMngr.get_log_level(logger);
        if (level < SUPPRESSED_LOG_LEVEL)
        {
            m_origLoggingLevels[logger] = level;
            logMngr.set_log_level(logger, SUPPRESSED_LOG_LEVEL);
        }
    }
}

ScopedLogSuppression::~ScopedLogSuppression()
{
    auto& logMngr = synapse::LogManager::instance();
    for (const auto& [loggerIdx, prevLevel] : m_origLoggingLevels)
    {
        const synapse::LogManager::LogType logger = static_cast<synapse::LogManager::LogType>(loggerIdx);
        logMngr.set_log_level(logger, prevLevel);
    }
}

namespace lb_timing
{
using clock_t = std::chrono::steady_clock;
inline clock_t::time_point now()
{
    return clock_t::now();
}
inline double elapsedMili(const clock_t::time_point& since)
{
    return 1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(now() - since).count();
}
}  // namespace lb_timing

// The "main" method of the layered brain
void Runner::run()
{
    m_graph.setLayeredBrainData(std::make_unique<LayeredBrainData>());
    switch (GCFG_LAYERED_BRAIN_FLOW_MODE.value())
    {
        case BrainFlowConfig::SINGLE_PASS:
            runFwdProgress();
            break;
        case BrainFlowConfig::ITERATIVE:
            runIterative();
            break;
        default:
            HB_ASSERT(false, "Unsupported layered brain flow mode: {}", GCFG_LAYERED_BRAIN_FLOW_MODE.value());
    }
}

void Runner::runFwdProgress()
{
    LOG_DEBUG(LAYERED_BRAIN, "Executing single pass layered brain");
    std::map<BundleIndex, BundleNodes> allBundles = generateBundles();

    for (const auto& [idx, bigNodes] : allBundles)
    {
        auto slicedGraph = sliceBundle(bigNodes, idx);
        HB_ASSERT(slicedGraph, "Fwd progress brain unexpectedly failed to slice bundle {}", idx);

        BundleNodes sliceNodes {slicedGraph->getNodes().begin(), slicedGraph->getNodes().end()};

        bool successfulSwap = finalizeBundleSlicing(*slicedGraph, sliceNodes, bigNodes);
        HB_ASSERT(successfulSwap, "Fwd progress brain unexpectedly failed to insert sliced bundle {}", idx);
    }
    // memory management is done in a separate passes.
}

std::map<BundleIndex, BundleNodes> Runner::generateBundles() const
{
    BPGraphContext bpgCtx(m_graph);
    MmeBundler     bundler(m_graph);
    const auto&    allBundles = bundler.generateBundles();
    HB_ASSERT(m_graph.getBPGraph()->getBundlePlaneGraph()->isAcyclicGraph(), "Cycle detected between bundles");
    return allBundles;
}

HabanaGraphPtr Runner::sliceBundle(const BundleNodes& bigNodes, BundleIndex bundleIdx) const
{
    return Slicer(m_graph, bundleIdx, bigNodes).getSlicedBundle();
}

bool Runner::scheduleBundle(HabanaGraph& slicedGraph, bool dryRun) const
{
    BundleNodes sliceNodes;
    for (const NodePtr& n : slicedGraph.getNodes())
    {
        if (n->getNodeAnnotation().bundleInfo.is_set())
        {
            sliceNodes.push_back(n);
        }
    }
    return SlicedNodesScheduler(slicedGraph).scheduleBundle(sliceNodes, dryRun);
}

bool Runner::handlePartialWrites(HabanaGraph& slicedGraph) const
{
    return handlePartialsWrites(slicedGraph);
}

bool Runner::setCacheDirectives(HabanaGraph& slicedGraph, bool dryRun) const
{
    BundleNodes        sortedSlicedNodes = slicedGraph.getExeSortedNodes();
    BundleCacheManager cacheMgr(slicedGraph, sortedSlicedNodes);
    return cacheMgr.setCacheDirectives(dryRun);
}

bool Runner::finalizeBundleSlicing(HabanaGraph& slicedGraph, BundleNodes& sliceNodes, const BundleNodes& bigNodes)
{
    if (!swapBigAndSlicedNodes(slicedGraph, sliceNodes, bigNodes))
    {
        return false;
    }
    moveBundleDataToFullGraph(slicedGraph);
    return true;
}

bool Runner::swapBigAndSlicedNodes(HabanaGraph& slicedGraph, BundleNodes& sliceNodes, const BundleNodes& bigNodes)
{
    GraphEditor::removeNodes(slicedGraph, sliceNodes);
    // GraphEditor promises that the replace is either successful or no change happens to the graph, so no need to
    // assert here.
    return REPLACE_NODE_SUCCESS == GraphEditor::replaceNodes(m_graph, bigNodes, sliceNodes);
}

// Move the layered brain data from the sliced graph into the full graph
void Runner::moveBundleDataToFullGraph(HabanaGraph& slicedGraph)
{
    HB_ASSERT_PTR(slicedGraph.getLayeredBrainData());
    HB_ASSERT(slicedGraph.getLayeredBrainData()->m_bundleData.size() == 1,
              "Expected the data of a single bundle in the sliced graph, found: {}",
              slicedGraph.getLayeredBrainData()->m_bundleData.size());

    auto& bundleData = *(slicedGraph.getLayeredBrainData()->m_bundleData.begin());
    // stamp bigNodes annotations with mme chosen strategy
    bundleData.second.getFinalStrategy()->getMmeSolution()->chooseSolution();
    m_graph.getLayeredBrainData()->m_bundleData.insert(std::move(bundleData));
}

void Runner::runIterative()
{
    LOG_DEBUG(LAYERED_BRAIN, "Executing iterative layered brain pass");
    BPGraphContext bpgCtx(m_graph);
    const BundlerPtr& bundler         = std::make_unique<MmeBundler>(m_graph);

    auto runnerStartTime = lb_timing::now();
    auto expandedBundles = getExpandedBundles(bundler);
    LOG_DEBUG(LAYERED_BRAIN, "<LB_TIMING> Bundle expansion duration: {}[ms]", lb_timing::elapsedMili(runnerStartTime));

    bundler->logGraphBundlingStatus();

    auto expansionStartTime = lb_timing::now();
    for (auto& bundle : expandedBundles)
    {
        LOG_DEBUG(LAYERED_BRAIN, "Finalizing bundle index {}", bundle->index());
        const auto unslicedNodes = bundle->getNodesCopy<BundleNodes>();
        Slicer     slicer(m_graph, bundle->index(), unslicedNodes);
        const auto chosenStrategy = solveBundle(slicer, bundle);
        HB_ASSERT_PTR(chosenStrategy);
        auto slicedGraph = slicer.sliceBundleByStrategy(chosenStrategy);
        HB_ASSERT(slicedGraph->isConnectedGraph(), "Expecting sliced graph {} to be connected", bundle->index());
        BundleNodes slicedNodes(slicedGraph->getNodes().begin(), slicedGraph->getNodes().end());

        bptContaminationCanaryCheck(slicedGraph);

        const bool swapSlicedSuccess = finalizeBundleSlicing(*slicedGraph, slicedNodes, unslicedNodes);
        HB_ASSERT(swapSlicedSuccess, "Failed to insert sliced bundle into the graph");
    }
    LOG_DEBUG(LAYERED_BRAIN, "<LB_TIMING> Bundle solving duration: {}[ms]", lb_timing::elapsedMili(expansionStartTime));
}

std::vector<bundler::BundleAndExpanders> Runner::getInitialBundles(const BundlerPtr& bundler)
{
    std::vector<bundler::BundleAndExpanders> initialBundles {};
    LOG_DEBUG(LAYERED_BRAIN, "Collect initial bundle candidates");
    for (auto& seedCollector : bundler->getSeedCollectors())
    {
        for (auto& seedCandidateAndExpanders : seedCollector->collect(true /*iterative mode*/))
        {
            auto& bundleSeedCandidate = seedCandidateAndExpanders.first;
            if (evaluateComposition(bundleSeedCandidate))
            {
                LOG_DEBUG(LAYERED_BRAIN, "Candidate bundle {} passed initial evaluation", bundleSeedCandidate->index());
                bundleSeedCandidate->acceptCandidates();
                initialBundles.push_back(std::move(seedCandidateAndExpanders));
            }
            else
            {
                LOG_DEBUG(LAYERED_BRAIN, "Candidate bundle {} failed initial evaluation", bundleSeedCandidate->index());
                if (bundleSeedCandidate->getNodes().size() == 1)
                {
                    // TODO [SW-146447]: initial bundles with one node shouldn't fail initial evaluate composition
                    const auto& bundleNodes        = bundleSeedCandidate->getNodes();
                    const auto  getBundledNodeName = [](const auto& n) {
                        return fmt::format("{}[{}]", n->getNodeName(), n->getNodeTypeStr());
                    };
                    LOG_WARN_AND_PERF(LAYERED_BRAIN,
                                      "Candidate bundle {} failed initial composition, nodes: {}",
                                      bundleSeedCandidate->index(),
                                      toString(bundleNodes, ',', getBundledNodeName));
                }
                bundleSeedCandidate->rejectCandidates();
            }
        }
    }
    return initialBundles;
}

std::vector<BundlePtr> Runner::getExpandedBundles(const BundlerPtr& bundler)
{
    auto initialBundlesAndExpanders = getInitialBundles(bundler);
    auto expandedBundles            = expandInitialBundles(bundler, initialBundlesAndExpanders);
    return expandedBundles;
}

std::vector<BundlePtr> Runner::expandInitialBundles(const BundlerPtr&                               bundler,
                                                    const std::vector<bundler::BundleAndExpanders>& bundlesAndExpanders)
{
    using BundleAndExpandersList = std::list<bundler::BundleAndExpanders>;
    std::vector<BundlePtr>                    expandedBundles {};
    std::unordered_map<BundleIndex, unsigned> bundleIdxToStepNr;
    BundleAndExpandersList bundleAndExpandersList(bundlesAndExpanders.begin(), bundlesAndExpanders.end());

    // init step map
    std::for_each(bundlesAndExpanders.begin(),
                  bundlesAndExpanders.end(),
                  [&bundleIdxToStepNr](const auto& bundleAndExpanders) {
                      bundleIdxToStepNr.insert(std::make_pair(bundleAndExpanders.first->index(), 0));
                  });

    // round robin expansion of expandable bundles
    // at the end of each iteration, remove finalized bundles
    while (!bundleAndExpandersList.empty())
    {
        std::vector<BundleAndExpandersList::const_iterator> finalizedBundles;
        const auto finalizeBundle = [&finalizedBundles, &expandedBundles](const auto& it) {
            finalizedBundles.push_back(it);
            expandedBundles.push_back(it->first);
        };
        for (auto it = bundleAndExpandersList.begin(); it != bundleAndExpandersList.end(); ++it)
        {
            auto& [bundle, expanders] = *it;
            HB_ASSERT_PTR(bundle);
            unsigned& stepNr = bundleIdxToStepNr.at(bundle->index());
            SET_TEMP_LOG_CONTEXT(fmt::format("ExpandInitialBundle #{}, step {}", bundle->index(), stepNr));
            LOG_DEBUG(LAYERED_BRAIN, "Start expansion step #{}", stepNr);
            auto       expansionStartTime = lb_timing::now();
            const bool success = bundler->expansionStep(bundle, expanders);
            if (success)
            {
                LOG_DEBUG(LAYERED_BRAIN, "Expansion step success");
                if (evaluateComposition(bundle))
                {
                    LOG_DEBUG(LAYERED_BRAIN, "Expansion step eval success");
                    bundle->acceptCandidates();
                }
                else
                {
                    LOG_DEBUG(LAYERED_BRAIN, "Expansion step eval failure");
                    bundle->rejectCandidates();
                }
            }
            else
            {
                LOG_DEBUG(LAYERED_BRAIN, "Expansion step failed");
                // prevent bundle from expanding in the next iteration of outer loop
                finalizeBundle(it);
            }
            ++stepNr;
            LOG_DEBUG(LAYERED_BRAIN,
                      "<LB_TIMING> Bundle {} expansion step duration: {}[ms]",
                      bundle->index(),
                      lb_timing::elapsedMili(expansionStartTime));
        }
        std::for_each(finalizedBundles.begin(), finalizedBundles.end(), [&bundleAndExpandersList](const auto& it) {
            bundleAndExpandersList.erase(it);
        });
    }
    return expandedBundles;
}

bool Runner::inflateToReduceNofSlices(Slicer&             slicer,
                                      const BundlePtr&    bundle,
                                      StrategyPtr&        strategy,
                                      const ConstEvalPtr& evaluation) const
{
    // When reducing num of bundle slices, attempt inflating perforation BVDs
    // to a valid multiplier before inflating for num slices (IFN).
    // When reaching a valid perforation BVD multiplier (= divides evenly between dcores),
    // it gets locked such that succeeding IFNs are promised to not compromise the multiplier.
    bool inflateSuccess = false;
    LOG_DEBUG(LAYERED_BRAIN, "Inflate for optimization (resources exhaustion)");

    // Even though perforation multiplier may not be the effective perforation metric,
    // it is used below to attempt reaching an even work distribution between dcores due to it being valid
    // only once BVD multiplier splits evenly between dcores.
    if (GCFG_ENABLE_EVALUATE_PERFORATION_UTIL.value() && !evaluation->perforationMultiplier.valid() &&
        !evaluation->perforationMultiplier.unevaluated())
    {
        inflateSuccess = slicer.inflateStrategy(InflationType::INFLATE_FOR_PERFORATION,
                                                strategy,
                                                evaluation->perforationMultiplier.offendingNode);
    }
    if (!inflateSuccess)
    {
        inflateSuccess = slicer.inflateStrategy(InflationType::INFLATE_FOR_NUM_SLICES, strategy, nullptr);
    }
    return inflateSuccess;
}

StrategyPtr Runner::findOptimalStrategy(const std::vector<std::pair<ConstEvalPtr, StrategyPtr>>& validStrategies) const
{
    if (validStrategies.empty()) return nullptr;

    // diff in % between val1 and val2
    auto relativeDiff = [](auto val1, auto val2) -> double {
        double ratio = static_cast<double>(val1) / static_cast<double>(val2);
        return std::abs(1.0 - ratio);
    };

    // Returns true if lhs "<" rhs (lhs is worse then rhs)
    const auto strategyCmp = [&relativeDiff](const std::pair<ConstEvalPtr, StrategyPtr>& lhs,
                                             const std::pair<ConstEvalPtr, StrategyPtr>& rhs) {
        const auto& lhsEval = lhs.first;
        const auto& rhsEval = rhs.first;

        // higher mme util is better
        if (relativeDiff(lhsEval->mmeUtil.value, rhsEval->mmeUtil.value) >= .05)
        {
            return lhsEval->mmeUtil.value < rhsEval->mmeUtil.value;
        }

        // lower mme bw is better
        if (relativeDiff(lhsEval->mmeBw.value, rhsEval->mmeBw.value) >= .05)
        {
            return lhsEval->mmeBw.value > rhsEval->mmeBw.value;
        }

        // higher number of perforated nodes is better
        const auto numPerforatedNodesLhs = lhs.second->getNumPerforatedNodes();
        const auto numPerforatedNodesRhs = rhs.second->getNumPerforatedNodes();
        if (numPerforatedNodesLhs != numPerforatedNodesRhs)
        {
            return numPerforatedNodesLhs < numPerforatedNodesRhs;
        }

        // Prefer strategies that are not sliced on common-dim
        const auto isSlicedOnCDLhs = lhs.second->isSlicedOnCommonDim();
        const auto isSlicedOnCDRhs = rhs.second->isSlicedOnCommonDim();
        if (isSlicedOnCDLhs != isSlicedOnCDRhs)  // One of the strategies sliced on CD, the other is not
        {
            // LHS "<" RHS if it's the one with the CD slicing (RHS is better)
            return isSlicedOnCDLhs;
        }

        // Lower $ usage is better
        if (relativeDiff(lhsEval->cacheUtil.value, rhsEval->cacheUtil.value) >= .05)
        {
            return lhsEval->cacheUtil.value > rhsEval->cacheUtil.value;
        }

        // tie breaker by strategy index
        return lhs.second->index() < rhs.second->index();
    };

    LOG_DEBUG(LAYERED_BRAIN, "Selecting optimal strategy out of {} valid strategies", validStrategies.size());

    const auto optimalIt = std::max_element(validStrategies.begin(), validStrategies.end(), strategyCmp);

    LOG_DEBUG(LAYERED_BRAIN,
              "FinalEvaluation: strategy: {}, nofSlices: {}, mmeUtil: {:>3.2f} %, cacheUtil: {:>3.2f}/{} MB, mmeBw: {} "
              "GB/s, pipeline depth: {}",
              optimalIt->second->index(),
              optimalIt->first->nofSlices.value,
              100 * (optimalIt->first->mmeUtil.value),
              bToMb(optimalIt->first->cacheUtil.value),
              bToMb(m_graph.getHALReader()->getSRAMSizeInBytes()),
              optimalIt->first->mmeBw.value,
              optimalIt->second->getPipelineDepth());
    optimalIt->second->log();

    return optimalIt->second;
}

std::pair<ConstEvalPtr, StrategyPtr> Runner::getMaxInflatedStrategy(const BundlePtr&     bundle,
                                                                    Slicer&              slicer,
                                                                    StrategyPtr&         strategy,
                                                                    const MaxUtilPerMME& maxUtilPerMME) const
{
    std::pair<ConstEvalPtr, StrategyPtr> lastValid(nullptr, nullptr);

    unsigned nStep          = 0;
    const auto minPipelineDepth = GCFG_LAYERED_BRAIN_SCHEDULER_MIN_PIPELINE_DEPTH.value();
    const auto maxPipelineDepth = GCFG_LAYERED_BRAIN_SCHEDULER_MAX_PIPELINE_DEPTH.value();
    HB_ASSERT(minPipelineDepth <= maxPipelineDepth,
              "Invalid scheduler pipeline depth - min: {} max: {}",
              minPipelineDepth,
              maxPipelineDepth);

    for (auto pipelineDepth = minPipelineDepth; pipelineDepth <= maxPipelineDepth; pipelineDepth++)
    {
        strategy->setPipelineDepth(pipelineDepth);
        // Returns true if strategy is already valid (without inflation)
        bool inflateSuccess = preSlicingInflateForValidity(bundle, slicer, strategy, maxUtilPerMME);
        while (inflateSuccess)
        {
            SET_TEMP_LOG_CONTEXT(fmt::format("maxInflate bundle#{}, strategy#{}, step#{}, pipeline-depth:{}",
                                             bundle->index(),
                                             strategy->index(),
                                             nStep,
                                             pipelineDepth));
            auto evaluation = evaluateStep(slicer, bundle, strategy, maxUtilPerMME);
            if (!evaluation)
            {
                LOG_DEBUG(LAYERED_BRAIN, "Evaluate step failed, exiting inflation loop");
                break;
            }

            LOG_DEBUG(LAYERED_BRAIN, "Evaluation: {}", evaluation->evalString());
            ++nStep;  // increment step count after a successful eval step
            if (evaluation->allMetricsValid())
            {
                LOG_DEBUG(LAYERED_BRAIN, "All bundle stats valid");
                lastValid = std::make_pair(evaluation, strategy->clone());
                // The strategy is valid and saved, try another inflation.
                inflateSuccess = inflateToReduceNofSlices(slicer, bundle, strategy, evaluation);
                // Strategy is not changed if inflation fails, ready to be evaluated with next pipeline-depth.
            }
            else
            {
                // Strategy was valid, but now invalid (assert below), move to the next pipeline-depth.
                HB_ASSERT(lastValid.second != nullptr,
                          "Expecting at least one valid inflated strategy for bundle: {}, strategy: {}",
                          bundle->index(),
                          strategy->index());
                strategy = lastValid.second;  // continue with the last valid strategy
                break;
            }
        }
    }
    return lastValid;
}

StrategyPtr Runner::solveBundle(Slicer& slicer, const BundlePtr& bundle) const
{
    HB_ASSERT_PTR(bundle);
    using EvalAndStrategy = std::pair<ConstEvalPtr, StrategyPtr>;
    std::vector<EvalAndStrategy> validStrategies;
    auto                         solveStart = lb_timing::now();
    auto                         strategies = slicer.getStrategies();
    const auto&                  maxUtilPerMME = BundleEvaluator::collectMaxUtils(bundle, strategies);
    for (auto& s : strategies)
    {
        LOG_DEBUG(LAYERED_BRAIN, "solveBundle #{}, strategy#{}", bundle->index(), s->index());
        if (const auto [eval, strategy] = getMaxInflatedStrategy(bundle, slicer, s, maxUtilPerMME); eval && strategy)
        {
            validStrategies.push_back({eval, strategy});
        }
    }
    LOG_DEBUG(LAYERED_BRAIN,
              "<LB_TIMING> Bundle {} strategies max inflation duration: {}[ms]",
              bundle->index(),
              lb_timing::elapsedMili(solveStart));
    SET_TEMP_LOG_CONTEXT(fmt::format("FindOptimalStrategy Bundle#{}", bundle->index()))
    return findOptimalStrategy(validStrategies);
}

bool Runner::evaluateComposition(const BundlePtr& bundle) const
{
    HB_ASSERT_PTR(bundle);
    const auto unslicedNodes = bundle->getNodesCopy<BundleNodes>();
    Slicer     slicer(m_graph, bundle->index(), unslicedNodes);
    auto       strategies = slicer.getStrategies();
    const auto& maxUtilPerMME = BundleEvaluator::collectMaxUtils(bundle, strategies);
    SET_TEMP_LOG_CONTEXT(fmt::format("EvaluateComposition Bundle#{}", bundle->index()))
    for (auto& strategy : strategies)
    {
        SET_TEMP_LOG_CONTEXT(
            fmt::format("EvaluateComposition Bundle#{}, Strategy#{}", bundle->index(), strategy->index()))
        if (preSlicingInflateForValidity(bundle, slicer, strategy, maxUtilPerMME))
        {
            // Strategy is valid pre-slicing, now check whether it fits $
            auto evaluation = evaluateStep(slicer, bundle, strategy, maxUtilPerMME);
            if (evaluation)
            {
                HB_ASSERT(evaluation->allMetricsValid(), "Expecting all metrics valid for bundle {}", bundle->index());
                LOG_DEBUG(LAYERED_BRAIN, "Evaluation: {}", evaluation->evalString());
                return true;
            }
        }
        // skip to the next strategy
        LOG_DEBUG(LAYERED_BRAIN, "Skipping to next strategy");
    }
    LOG_DEBUG(LAYERED_BRAIN, "No valid strategy for current composition (#strategies: {})", strategies.size());
    return false;
}

bool Runner::preSlicingInflateForValidity(const BundlePtr&     bundle,
                                          Slicer&              slicer,
                                          StrategyPtr&         strategy,
                                          const MaxUtilPerMME& maxUtilPerMME) const
{
    SET_TEMP_LOG_CONTEXT(fmt::format("{} Bundle#{}", HLLOG_FUNC, bundle->index()));
    bool         inflateSuccess = true;
    ConstEvalPtr preSlicingEval;
    const BundleEvaluator evaluator(bundle, slicer.getSlicingDetails(strategy), maxUtilPerMME);
    while (inflateSuccess)
    {
        preSlicingEval = evaluator.preSlicingEvaluation();
        if (preSlicingEval->allMetricsValid()) break;
        /**
         * Pre-slicing inflation priority (as long as nofSlices isn't low):
         * - mme utilization
         * - mme bandwidth
         * - perforation multiplier / util
         * - number of bundle slices
         */
        const auto& mmeUtil               = preSlicingEval->mmeUtil;
        const auto& mmeBw                 = preSlicingEval->mmeBw;
        const auto& nofSlices             = preSlicingEval->nofSlices;
        const auto& perforationMultiplier = preSlicingEval->perforationMultiplier;
        const auto& perforationUtil       = preSlicingEval->perforationUtil;

        if (nofSlices.eval == EvalResult::LOW)
        {
            // break inflation loop and skip to the next strategy
            LOG_DEBUG(LAYERED_BRAIN, "LOW slice count, skipping to next strategy");
            inflateSuccess = false;
        }
        else if (mmeUtil.eval == EvalResult::LOW)
        {
            LOG_DEBUG(LAYERED_BRAIN, "Inflate for mme util");
            HB_ASSERT_PTR(mmeUtil.offendingNode);
            inflateSuccess =
                slicer.inflateStrategy(InflationType::INFLATE_FOR_UTILIZATION, strategy, mmeUtil.offendingNode);
        }
        else if (mmeBw.eval == EvalResult::HIGH)
        {
            LOG_DEBUG(LAYERED_BRAIN, "Inflate for mme bw");
            HB_ASSERT_PTR(mmeBw.offendingNode);
            inflateSuccess = slicer.inflateStrategy(InflationType::INFLATE_FOR_BW, strategy, mmeBw.offendingNode);
        }
        else if (GCFG_ENABLE_EVALUATE_PERFORATION_UTIL.value() ? perforationUtil.eval == EvalResult::LOW
                                                               : perforationMultiplier.eval == EvalResult::LOW)
        {
            LOG_DEBUG(LAYERED_BRAIN, "Inflate for perforation");
            const auto& offendingNode = GCFG_ENABLE_EVALUATE_PERFORATION_UTIL.value()
                                            ? perforationUtil.offendingNode
                                            : perforationMultiplier.offendingNode;
            HB_ASSERT_PTR(offendingNode);
            inflateSuccess = slicer.inflateStrategy(InflationType::INFLATE_FOR_PERFORATION, strategy, offendingNode);
        }
        else if (nofSlices.eval == EvalResult::HIGH)
        {
            LOG_DEBUG(LAYERED_BRAIN, "Inflate for num of bundle slices");
            inflateSuccess = inflateToReduceNofSlices(slicer, bundle, strategy, preSlicingEval);
        }
        else
        {
            // Reaching here means there is an evaluation discrepancy: evaluation is invalid in terms of
            // Evaluation::allMetricsValid() while non of the evaluation metrics are considered invalid
            // based on the conditions above.
            HB_ASSERT(false, "Bundle {} evaluation discrepancy", bundle->index());
        }
    }
    return preSlicingEval->allMetricsValid();
}

class GenericPassesRunner
{
public:
    GenericPassesRunner(const HabanaGraph& origGraph, HabanaGraphPtr& slicedGraph) : m_slicedGraph(slicedGraph) {}

    bool runGenericPasses(PassId stopBeforePass) const
    {
        ScopedLogSuppression suppressLogging;
        LOG_DEBUG(LAYERED_BRAIN, "Running generic layered brain passes");
        return m_slicedGraph->runPartialPasses(stopBeforePass);
    }

protected:
    HabanaGraphPtr& m_slicedGraph;
};

ConstEvalPtr Runner::evaluateStep(Slicer&              slicer,
                                  const BundlePtr&     bundle,
                                  const StrategyPtr&   strategy,
                                  const MaxUtilPerMME& maxUtilPerMME) const
{
    constexpr bool performDryRun = true;  // evaluation step is done in dry run mode.
    // slice
    auto slicedGraph = slicer.sliceBundleByStrategy(strategy, performDryRun);
    HB_ASSERT_PTR(slicedGraph);

    GenericPassesRunner passesRunner(m_graph, slicedGraph);

    // run generic passes until partials writes handler
    bool runPassesSuccess = passesRunner.runGenericPasses(PASS_ID_HANDLE_PARTIALS_WRITES);
    HB_ASSERT(runPassesSuccess, "Expecting generic passes to run successfully");
    // partials writes handling
    const bool partialsSuccess = handlePartialWrites(*slicedGraph);
    if (!partialsSuccess)
    {
        LOG_DEBUG(LAYERED_BRAIN, "Evaluation step partial writes failed for bundle {}", bundle->index());
        return nullptr;
    }

    // run generic passes until scheduler
    runPassesSuccess = passesRunner.runGenericPasses(PASS_ID_BUNDLE_NODES_SCHEDULE);
    HB_ASSERT(runPassesSuccess, "Expecting generic passes to run successfully");
    // schedule
    const bool scheduleSuccess = scheduleBundle(*slicedGraph, performDryRun);
    if (!scheduleSuccess)
    {
        LOG_DEBUG(LAYERED_BRAIN, "Evaluation step scheduler failed for bundle {}", bundle->index());
        return nullptr;
    }

    // run generic passes until cache manager
    runPassesSuccess = passesRunner.runGenericPasses(PASS_ID_BUNDLE_MEMORY_MANAGEMENT);
    HB_ASSERT(runPassesSuccess, "Expecting generic passes to run successfully");
    // cache manager
    const bool cacheMgmtSuccess = setCacheDirectives(*slicedGraph, performDryRun);
    if (!cacheMgmtSuccess)
    {
        LOG_DEBUG(LAYERED_BRAIN, "Evaluation step $-management failed for bundle {}", bundle->index());
        return nullptr;
    }

    // evaluate
    return BundleEvaluator(bundle, slicer.getSlicingDetails(strategy), maxUtilPerMME)
        .postSlicingEvaluation(*slicedGraph);
}

void Runner::bptContaminationCanaryCheck(const HabanaGraphPtr& slicedGraph) const
{
    bool contaminated = false;
    for (const auto& t : slicedGraph->getTensors())
    {
        if (!t || !t->isDataTensor()) continue;
        if (t->isRealInAliasing())  // currently this is the only known contamination type (a property that should not
                                    // be set before the brain slices the bundles). If more are found, this condition
                                    // can be extended or other conditions can be added for better logging.
        {
            LOG_CRITICAL(LAYERED_BRAIN, "Real in aliasing after final slicing (tensor: {})", t->getName());
            contaminated = true;
        }
    }
    HB_ASSERT(!contaminated, "Canary check failed - contaminated tensors found in {}", slicedGraph->getRecipeName());
}

}  // namespace gc::layered_brain