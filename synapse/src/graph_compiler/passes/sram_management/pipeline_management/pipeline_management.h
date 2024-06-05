#pragma once

#include "pipeline_bundlizer.h"
#include "bundle_solver.h"
#include "pipeline_management/habana_norms_handler.h"

class HabanaGraph;

using SolvedBundles = std::list<pBundle>;

class PipelineSlicingManager
{
public:
    explicit PipelineSlicingManager(HabanaGraph& graph, BundlingPolicy bundlingPolicy);
    bool optimizeGraph();

protected:
    // Partition the graph to clusters, and solve each cluster pipeline optimization
    void deviseGraphPipelineStrategies();
    // Replace each bundle sub-graph with the sliced nodes and additional required nodes
    void applyGraphPipelineStrategies();

    // Find initial clusters (sub graphs), for which the nodes will be pipelined explicitly.
    void createBundles();
    // Find a solution to apply the bundle pipeline on the graph
    void solveBundles();

    // Graph transformations that improve bundling and slicing
    void applyPreBundlingOptimizations();

    // Graph and slicing modifications based on the bundle expansion before actually slicing the graph.
    void applyPreSlicingOptimizations(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution);
    // Add optional outputs to operations to implement sram slices evictions (instead of planting memcpy later)
    void fuseEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution);
    void fuseMMEEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution);
    void fuseTPCEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution);
    void fuseBNEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution);

    void logGraphBundlingStatus();

    // Graph corrections after slicing to leave the pass with the graph in a valid state
    bool applyPostSlicingCorrections();

    BundlingPolicy       m_bundlingPolicy;
    HabanaGraph&         m_graph;
    BundlesInfoContainer m_bundles;
    SolvedBundles        m_solvedBundles;
    HabanaNormsHandler   m_normsHandler;
};
