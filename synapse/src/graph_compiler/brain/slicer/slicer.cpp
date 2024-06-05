#include "slicer.h"
#include "types.h"
#include "bundle_views_collector.h"
#include "key_nodes_finder.h"
#include "mme_key_node_solver.h"
#include "legacy_mme_key_node_solver.h"
#include "optimal_slicing_calculator.h"
#include "sliced_bundle_graph_generator.h"
#include "brain_conf.h"
#include "strategy_filter.h"

using namespace gc::layered_brain;

Slicer::Slicer(const HabanaGraph& g,
               const BundleIdx    bundleIdx,
               const NodeVector&  bundleNodes,
               SlicingPolicy      slicingPolicy,
               uint64_t           maxTileSize,
               uint64_t           minCommonDimForPartials)
: m_graph(g),
  m_bundleIdx(bundleIdx),
  m_bundleNodes(bundleNodes),
  m_keyNodes(MultiMmeBundleKeyNodesFinder(g, bundleNodes).getSortedKeyNodes()),
  m_slicingPolicy(slicingPolicy),
  m_maxTileSize(maxTileSize),
  m_minCommonDimForPartials(minCommonDimForPartials)
{
    m_bundleViews = getInitialBundleViews();
    m_inflator    = std::make_unique<StrategyInflator>(m_bundleViews);
    m_perforator  = std::make_unique<Perforator>(m_graph, m_bundleNodes, m_bundleViews);
}

// TODO: [SW-136617] - LCM calculator should handle non-scratchpad aux tensors internally
std::pair<TileSizePerTensor, TileSizePerNode> Slicer::getMinCommonTilesSizes() const
{
    std::vector<std::pair<TensorPtr, TensorTile::Geometry>> nonLcmTensors {};
    TensorSet                                               bundleTensorsSet {};
    NodeSet                                                 bundleNodesSet(m_bundleNodes.begin(), m_bundleNodes.end());
    for (const auto& n : m_bundleNodes)
    {
        for (const auto& nodeOperand : n->getOperands())
        {
            if (!nodeOperand) continue;
            if (nodeOperand->isNonScratchpadAuxTensor())
            {
                const TensorTile tensorGranularity = n->getNodeAccessPattern()->getTensorGranularity(nodeOperand);
                const SizeArray  sizes             = nodeOperand->getAllSizesInElements();
                for (Dim dim = 0; dim < nodeOperand->getDim(); dim++)
                {
                    HB_ASSERT(tensorGranularity.geometry.at(dim) == sizes[dim],
                              "Expected operand {} dim {} to be all required",
                              nodeOperand->getName(),
                              dim);
                }
                nonLcmTensors.push_back(std::make_pair(nodeOperand, tensorGranularity.geometry));
            }
            else
            {
                bundleTensorsSet.insert(nodeOperand);
            }
        }
    }

    // Set granularities for tensors that weren't included in the LCM calculation
    // and since these are aux tensors they're all required on all dims
    auto [tensorTiles, nodeTiles] =
        CommonTileSizeCalculator::getMinCommonTilesSizes(bundleNodesSet, bundleTensorsSet, m_graph);
    for (const auto& [t, geometry] : nonLcmTensors)
    {
        tensorTiles.insert(std::make_pair(t, geometry));
    }
    return std::make_pair(tensorTiles, nodeTiles);
}

BundleViewContainerPtr Slicer::getInitialBundleViews() const
{
    LOG_DEBUG(LB_SLICER, "Create bundle-views for bundle {}", m_bundleIdx);

    // Generate common ISR/ISMR per op/tensor
    const auto& [granularityPerTensor, granularityPerNode] = getMinCommonTilesSizes();

    // Collect all bundle views
    BundleViewsCollector          bundleViewsCollector(m_bundleNodes);
    const BundleViewContainerPtr& bundleViews =
        bundleViewsCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);

    return bundleViews;
}

StrategyVector Slicer::generateInitialStrategies() const
{
    LOG_DEBUG(LB_SLICER, "Generate initial strategies for bundle {}", m_bundleIdx);
    HB_ASSERT_PTR(m_bundleViews);

    StrategyContainer allStrategies;

    // Apply node-solver on any key bundle node
    for (const auto& keyNode : m_keyNodes)
    {
        // Get node slicing strategies (based on existing if there are any)
        auto keyNodeSolver = getKeyNodeSolver(keyNode);
        allStrategies      = keyNodeSolver->getSlicingStrategies(m_bundleViews, allStrategies);
    }

    return allStrategies.strategies;
}

StrategyPtr Slicer::calcSlicingStrategy() const
{
    LOG_DEBUG(LB_SLICER, "Find slicing strategy for bundle {}", m_bundleIdx);

    const auto& initialStrategies = generateInitialStrategies();

    if (initialStrategies.empty())
    {
        return nullptr;
    }

    // Select a strategy that optimizes the required metrics
    OptimalSlicingCalculator optimalSlicingCalculator(m_graph, m_slicingPolicy, m_maxTileSize);
    StrategyPtr              winningStrategy =
        optimalSlicingCalculator.getOptimalStrategy(m_bundleViews, initialStrategies, m_bundleNodes);

    return winningStrategy;
}

HabanaGraphPtr Slicer::getSlicedGraphFromStrategy(const StrategyPtr& strategy, bool dryRun) const
{
    HB_ASSERT_PTR(m_bundleViews);
    HB_ASSERT_PTR(strategy);
    LOG_DEBUG(LB_SLICER, "Create sliced graph for bundle {} from strategy {}", m_bundleIdx, strategy->index());
    SlicedBundleGraphGenerator slicedGraphGenerator(m_graph,
                                                    m_bundleIdx,
                                                    m_bundleNodes,
                                                    m_bundleViews,
                                                    strategy,
                                                    dryRun);
    // In dry run - create a tmp graph that is not connected to the original graph to prevent any changes from leaking
    // from the tmp to the original before the bundle is finalized.
    HabanaGraphPtr slicedBundleGraph = slicedGraphGenerator.createSlicedGraph();
    HB_ASSERT_PTR(slicedBundleGraph);
    LOG_DEBUG(LB_SLICER, "Bundle sliced successfully");

    slicedBundleGraph->setRecipeName(
        fmt::format("{}_slicedGraph_Bundle#{}_Strategy#{}", m_graph.getRecipeName(), m_bundleIdx, strategy->index()));

    return slicedBundleGraph;
}

HabanaGraphPtr Slicer::getSlicedBundle()
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Slicer Bundle#{}", m_bundleIdx));

    const StrategyPtr& strategy = calcSlicingStrategy();
    if (!strategy)
    {
        LOG_WARN(LB_SLICER, "No solution found for bundle {}", m_bundleIdx);
        return nullptr;
    }

    return getSlicedGraphFromStrategy(strategy, false);
}

StrategyVector Slicer::getStrategies() const
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Slicer Bundle#{}", m_bundleIdx));

    const auto& strategies = generateInitialStrategies();

    for (const auto& strategy : strategies)
    {
        // At this point a strategy may not have multipliers for all BVDs -
        // assign multiplier for missing dimensions.
        strategy->fillMissingMultipliers(m_bundleViews, 1UL);

        // Select perforation BVD per bundle node.
        m_perforator->selectPerforationForStrategy(strategy);
    }

    if (GCFG_ENABLE_CONFLICT_BASED_STRATEGY_TRIMMING.value() && !strategies.empty())
    {
        auto filteredStrategies = filterInternalConflictStrategies(strategies);
        if (!filteredStrategies.empty())
        {
            return filteredStrategies;
        }
        else
        {
            LOG_WARN_AND_PERF(LB_SLICER, "All strategies for bundle {} have internal conflicts", m_bundleIdx);
            // Falling back to returning the internally conflicted strategies.
        }
    }
    return strategies;
}

StrategyVector Slicer::filterInternalConflictStrategies(const StrategyVector& origStrategies) const
{
    StrategyVector filtered;

    std::copy_if(origStrategies.begin(),
                 origStrategies.end(),
                 std::back_inserter(filtered),
                 slicer::StrategyFilter(m_bundleViews, m_keyNodes));
    return filtered;
}

HabanaGraphPtr Slicer::sliceBundleByStrategy(const StrategyPtr& strategy, bool dryRun) const
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Slicer Bundle#{}", m_bundleIdx));

    return getSlicedGraphFromStrategy(strategy, dryRun);
}

bool Slicer::inflateStrategy(InflationType inflationType, const StrategyPtr& strategy, const NodePtr& node) const
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Slicer Bundle#{}", m_bundleIdx));

    HB_ASSERT_PTR(m_inflator);
    return m_inflator->inflateOneStep(inflationType, strategy, node);
}

ConstSlicingDetailsPtr Slicer::getSlicingDetails(const StrategyPtr& strategy) const
{
    return std::make_shared<const SlicingDetails>(m_bundleViews, strategy);
}

std::unique_ptr<KeyNodeSolver> Slicer::getKeyNodeSolver(const NodePtr& keyNode) const
{
    if (GCFG_ENABLE_MME_BRAIN.value())
    {
        return std::make_unique<MMEKeyNodeSolver>(m_graph, keyNode, m_maxTileSize, m_minCommonDimForPartials);
    }
    return std::make_unique<LegacyMMEKeyNodeSolver>(m_graph, keyNode, m_maxTileSize, m_minCommonDimForPartials);
}