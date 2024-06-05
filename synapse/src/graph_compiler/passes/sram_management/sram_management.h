#pragma once

#include "habana_graph.h"
#include "pipeline_management/habana_norms_handler.h"
#include "slicing_brain.h"
#include "reshape_aligner.h"

// Exposing the sram management main class in order to ease testing it.

class SRAMSlicingManager
{
public:
    explicit SRAMSlicingManager(HabanaGraph& graph) : m_graph(graph) {}

    // Main procedure
    bool sliceGraph();

    // Published for testing:

    // Classify the graph nodes to preliminary bundles.
    void generateInitialBundles();
    // Create preliminary strategies to solve (slice) every bundle.
    void generateInitialStrategies();
    // Get the current solving data for each bundle
    const BundleSolvingDataMap& getBundlesSolvingData()
    {
        return m_solvingDataPerBundle;
    }
    const Bundlizer& getBundlizer() { return m_bundlizer; }

private:
    HabanaGraph&       m_graph;
    Bundlizer          m_bundlizer {m_graph};
    AllBrains          m_brains {m_graph};
    ReshapeAligner     m_reshapeAligner {m_graph};
    BundleList         m_mmeBundles;
    BundleList         m_tpcScalarPipeBundles;
    BundleList         m_tpcBundles;
    BundleList         m_rmwSectionBundles;
    BundleList         m_dmaTransposeBundles;
    HabanaNormsHandler m_normsHandler {m_graph, std::make_shared<PatternNodesCollector>()};

    mutable BundleSolvingDataMap m_solvingDataPerBundle;

    // Try to include more nodes in each bundle
    void expandAllBundles();
    std::vector <pBundle> getBundleExpansionOrder() const;
    void expandBundle(pBundle& bundle);
    void logGraphSizeOptimizationStats(bool beforeBundleExpansion);

    // Bundle expansion execution sub functions
    void applyWinningStrategyToBundle(pBundle& bundle, pMmeSlicingStrategy winningStrategy);
    void flattenBundleNodes(pBundle& bundle, pMmeSlicingStrategy winningStrategy);
    void flattenMmeNode(pNode node, std::vector<pSlicedOperand> slicedOperands);
    void addCandidateToBundle(pBundle& bundle, pBundleExpansion& candidate, pMmeSlicingStrategy& strategy);
    void addSharedInput(pBundle& bundle, pBundleExpansion& candidate, pMmeSlicingStrategy& strategy);
    void addTpcProducer(pBundle& bundle, pBundleExpansion& candidate, pMmeSlicingStrategy& strategy);
    void addTpcConsumer(pBundle& bundle, pBundleExpansion& candidate, pMmeSlicingStrategy& strategy);

    // Slice each bundle according to the winning strategy
    void sliceAllBundles();
    void sliceBundles(BundleList& bundles);
    bool sliceBundle(pBundle& bundle);

    // Graph and slicing modifications based on the bundle expansion before actually slicing the graph.
    void preSlicingOptimizations();
    // After MME has a winning strategy, we can decide the slicing for TPC bundles
    void setTpcBundleSlicingChunk();
    bool isTpcBundleSlicingValid(const pBundle& tpcBundle, const StrategySlicingData& slicingData) const;
    // Add optional outputs to operations to implement sram slices evictions (instead of planting memcpy later)
    void fuseEvictions();
    void fuseMMEEvictions(pBundle& bundle);
    void fuseTPCEvictions(pBundle& bundle);
    void fuseBNEvictions(pBundle& bundle);
};
