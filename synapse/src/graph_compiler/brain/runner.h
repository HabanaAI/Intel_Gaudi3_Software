#pragma once

#include "bundler/bundler.h"
#include "bundler/bundle_seed_collector.h"
#include "habana_graph.h"
#include "layered_brain.h"
#include "bundler/layered_brain_bundle.h"
#include "bundle_evaluator.h"
#include "strategy.h"
#include "slicer/slicer.h"

namespace gc::layered_brain
{
// The layered brain runner performs the outermost flow of the brain - setting it up and calling each layer
// in turn
class Runner
{
public:
    Runner(HabanaGraph& graph) : m_graph(graph) {}
    void run();

private:
    HabanaGraph& m_graph;

    void                               runFwdProgress();
    std::map<BundleIndex, BundleNodes> generateBundles() const;
    HabanaGraphPtr                     sliceBundle(const BundleNodes& bigNodes, BundleIndex bundleIdx) const;
    bool                               scheduleBundle(HabanaGraph& slicedGraph, bool dryRun) const;
    bool                               handlePartialWrites(HabanaGraph& slicedGraph) const;
    bool                               setCacheDirectives(HabanaGraph& slicedGraph, bool dryRun) const;

    bool finalizeBundleSlicing(HabanaGraph& slicedGraph, BundleNodes& sliceNodes, const BundleNodes& bigNodes);
    void runIterative();
    std::vector<bundler::BundleAndExpanders> getInitialBundles(const BundlerPtr& bundler);
    std::vector<BundlePtr>                   getExpandedBundles(const BundlerPtr& bundler);
    std::vector<BundlePtr>                   expandInitialBundles(const BundlerPtr&                               bundler,
                                                                  const std::vector<bundler::BundleAndExpanders>& bundlesAndExpanders);
    bool swapBigAndSlicedNodes(HabanaGraph& slicedGraph, BundleNodes& sliceNodes, const BundleNodes& bigNodes);
    void moveBundleDataToFullGraph(HabanaGraph& slicedGraph);

    ConstEvalPtr evaluateStep(Slicer&              slicer,
                              const BundlePtr&     bundle,
                              const StrategyPtr&   strategy,
                              const MaxUtilPerMME& maxUtilPerMME) const;
    bool         evaluateComposition(const BundlePtr& bundle) const;

    /**
     * @brief Attempts inflating strategy to reach valid evaluator metrics.
     *        Returns true if strategy reached valid metrics otherwise false.
     */
    bool preSlicingInflateForValidity(const BundlePtr&     bundle,
                                      Slicer&              slicer,
                                      StrategyPtr&         strategy,
                                      const MaxUtilPerMME& maxUtilPerMME) const;

    bool inflateToReduceNofSlices(Slicer&             slicer,
                                  const BundlePtr&    bundle,
                                  StrategyPtr&        strategy,
                                  const ConstEvalPtr& evaluation) const;

    StrategyPtr solveBundle(Slicer& slicer, const BundlePtr& bundle) const;
    StrategyPtr findOptimalStrategy(const std::vector<std::pair<ConstEvalPtr, StrategyPtr>>& validStrategies) const;
    std::pair<ConstEvalPtr, StrategyPtr> getMaxInflatedStrategy(const BundlePtr&     bundle,
                                                                Slicer&              slicer,
                                                                StrategyPtr&         strategy,
                                                                const MaxUtilPerMME& maxUtilPerMME) const;
    bool                                 runGenericPasses(HabanaGraphPtr& slicedGraph) const;

    // Canary check that the full graph tensors hadn't been contaminated by the evaluation steps
    void bptContaminationCanaryCheck(const HabanaGraphPtr& slicedGraph) const;
};

}  // namespace gc::layered_brain