#include "pipeline_management.h"
#include "bundle.h"
#include "bundle_plane_graph.h"
#include "habana_graph.h"
#include "defs.h"
#include "sram_management/bundle_slicer.h"
#include "sram_management/sram_management.h"  // to call legacy sram management if feature is disabled
#include "sram_management/batch_norm_eviction_fuser.h"
#include "pair_grads.h"
#include <memory>
#include "brain_conf.h"
#include "snapshot_pre_slicing_sizes.h"

PipelineSlicingManager::PipelineSlicingManager(HabanaGraph& graph, BundlingPolicy bundlingPolicy)
: m_bundlingPolicy(bundlingPolicy), m_graph(graph), m_normsHandler(graph, std::make_shared<PatternNodesCollector>())
{
}

bool PipelineSlicingManager::optimizeGraph()
{
    // TODO SW-76732 - support slicing for pipeline even without SRAM
    if (GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.value() == 0UL)
    {
        return true;
    }

    if (GCFG_ENABLE_LAYERED_PIPELINE_BRAIN.value() && !GCFG_ENABLE_LB_HYBRID_MODE.value())
    {
        // LB enabled and hybrid mode is off => skip pipeline management
        return true;
    }

    applyPreBundlingOptimizations();
    deviseGraphPipelineStrategies();
    applyGraphPipelineStrategies();
    return applyPostSlicingCorrections();
}

void PipelineSlicingManager::applyPreBundlingOptimizations()
{
    // Break normalization undirected cycles as they are not supported atm
    m_normsHandler.findAndRemoveSliceNormNodes();

    // Pair more dedx/dedw to improve bwd pass bundles
    bool res = GradAReshapedGradBPairTransformer {m_graph}.optimizeGradPairs();
    HB_ASSERT(res, "Grad pair optimization failed unexpectedly.");
}


void PipelineSlicingManager::deviseGraphPipelineStrategies()
{
    createBundles();
    solveBundles();
}

void PipelineSlicingManager::createBundles()
{
    // Bundle plane graph lifetime should be only through bundles creation, before graph slicing
    BPGraphContext bpgCtxt(m_graph);

    // Generate RMW bundles to mark the nodes of a single RMW section to run together,
    // so the scheduling of the nodes is done together to shorten the lifetime of the SRAM memory usage.
    // It also keeps from mixing SRAM usage for RMW with bundles SRAM usage.
    Bundlizer(m_graph).bundleNodesWithRMWSectionOperand();

    switch (m_bundlingPolicy)
    {
        case BUNDLE_BY_TRANSFORMER_PATTERNS:
            m_bundles = MantaRayBundlizer(m_graph).generateBundles();
            break;
        case BUNDLE_BY_VISION_PATTERNS:
            m_bundles = TPCExpansionsAndSharedMMEBundlizer(m_graph).generateBundles();
            break;
        default:
            HB_ASSERT(false, "Invalid bundling criteria");
    }

    logGraphBundlingStatus();
}

void PipelineSlicingManager::solveBundles()
{
    for (BundleAndSolver& bundleAndSolverPair : m_bundles)
    {
        BundleSolverPtr solver = bundleAndSolverPair.second;
        HB_ASSERT_PTR(solver);
        BundleStrategyPtr solution = solver->solveBundle();
        if (solution)  // if a solution is found
        {
            PipelineBundlePtr& bundle = bundleAndSolverPair.first;
            applyPreSlicingOptimizations(bundle, solution);
            solver->fillBundleSolution(solution);
            m_solvedBundles.push_back(bundle);
        }
    }
}

void PipelineSlicingManager::applyGraphPipelineStrategies()
{
    while (!m_solvedBundles.empty())
    {
        PipelineBundlePtr& bundle = m_solvedBundles.front();
        BundleSlicer::sliceBundle(*bundle, m_graph);
        // After slicing the bundle, each bundle may hold thousands of operations.
        // Delete it in order to keep the memory footprint lower.
        m_solvedBundles.pop_front();
    }
}

void PipelineSlicingManager::logGraphBundlingStatus()
{
    // This method is a little heavy if not actually logging so check in advance
    if (!LOG_LEVEL_AT_LEAST_INFO(SRAM_SLICE)) return;

    // clear execution schedule cache to force re-calculation of it (for accurate logging)
    m_graph.invalidateExecutionSchedule();
    // getExeSortedNodes may generate logs itself, so calling it before the "headline"
    const NodeVector& exeSched = m_graph.getExeSortedNodes();

    SLC_INFO("Graph Bundling Status:");
    for (const pNode& node : exeSched)
    {
        std::string bundleId = node->getNodeAnnotation().bundleInfo.is_set()
                                   ? std::to_string(node->getNodeAnnotation().bundleInfo->bundleIndex)
                                   : "N/A";
        SLC_INFO("Bundle-ID: {:>3} Node: {} [{}]", bundleId, node->getNodeName(), node->getNodeTypeStr());
    }
}

static void
setPerforationAnnotation(const HabanaGraph& graph, const PipelineBundlePtr& bundle, const BundleStrategyPtr& strategy);

void PipelineSlicingManager::applyPreSlicingOptimizations(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution)
{
    fuseEvictions(bundle, solution);

    // TODO [SW-87423]: temporary WA for Gaudi3 until layered brain is completely implemented and solely used for it.
    if (GCFG_ENABLE_PIPELINE_MANAGEMENT_SRAM_OVERRIDE.value())
    {
        LOG_DEBUG(SRAM_SLICE, "Overriding SRAM decisions for bundle {} solution:", bundle->index());
        solution->printLog(1, synapse::LogManager::LogType::SRAM_SLICE);
        setPerforationAnnotation(m_graph, bundle, solution);
        for (const auto& slicedOperand : solution->getSlicingData().getSlicedOperands())
        {
            slicedOperand->resideInSRAM       = false;
            slicedOperand->alignWithCacheLine = false;
        }
    }
}

void PipelineSlicingManager::fuseEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution)
{
    // Fuse eviction of an intermediate tensor in the bundle which is generated slice by slice in SRAM and is also
    // needed in HBM (because it is persistent or it is read by an external consumer). This method tries to have the
    // bundled producer generate the same output in 2 places: one as intermediate tensor for the bundled consumer(s) and
    // one for the rest of the graph or the user to read.

    if (GCFG_ENABLE_BUNDLE_EVICTION_FUSING.value())
    {
        fuseMMEEvictions(bundle, solution);
        fuseTPCEvictions(bundle, solution);
    }
}

void PipelineSlicingManager::fuseMMEEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution)
{
    if (m_graph.getHALReader()->isDuplicateMmeOutputSupported())
    {
        // TODO [SW-52941]: In Gaudi2, when an MME output is generated in SRAM and is required outside the bundle (or is
        // persistent), add another output to the MME that generates the slices directly in HBM (subject to the MME
        // limitations. See ticket).
    }
}

void PipelineSlicingManager::fuseTPCEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution)
{
    // TODO - In the future, we may want to use the TPC fuser to fuse memcpy where an eviction from SRAM to HBM is
    // needed. Currently only batch norm is supported, via its optional output, not TPC fuser.
    fuseBNEvictions(bundle, solution);
}

void PipelineSlicingManager::fuseBNEvictions(PipelineBundlePtr& bundle, const BundleStrategyPtr& solution)
{
    HB_ASSERT_PTR(solution);
    BatchNormStagesEvictionFuser bnsef(m_graph, bundle, solution);
    bnsef.fuseEvictions(true,
                        false /*Not enabled yet, need regression to be investigated SW-92875*/,
                        false /*BN1_BWD eviction fusion not implemented yet in new pass*/);
}

static BundlingPolicy findBestBundlingPolicyForGraph(const HabanaGraph& graph)
{
    constexpr std::array policyStr = {"Transformer", "Vision"};

    if (GCFG_PIPELINE_MANAGEMENT_FORCE_BUNDLIZER.value() > 0)
    {
        switch (GCFG_PIPELINE_MANAGEMENT_FORCE_BUNDLIZER.value())
        {
            case 1:
                return BUNDLE_BY_VISION_PATTERNS;
            case 2:
                return BUNDLE_BY_TRANSFORMER_PATTERNS;
            default:
                HB_ASSERT(false,
                          "unexpected value for pipeline management force bundlizer flag: {}",
                          GCFG_PIPELINE_MANAGEMENT_FORCE_BUNDLIZER.value());
        }
    }

    uint64_t convType(0), matmulType(0);

    for (const NodePtr& node : graph.getNodes())
    {
        if (node)
        {
            // Using dynamic_cast, thus new conv nodes will be added transparently
            if (std::dynamic_pointer_cast<ConvBaseNode>(node))
            {
                ++convType;
            }
            else if (Node::isGemmNode(node) || Node::isBatchGemmNode(node))
            {
                ++matmulType;
            }
        }
    }

    // select policy based on type dominance
    BundlingPolicy bundlingPolicy;
    if (convType >= matmulType)
    {
        bundlingPolicy = BUNDLE_BY_VISION_PATTERNS;
    }
    else
    {
        bundlingPolicy = BUNDLE_BY_TRANSFORMER_PATTERNS;
    }

    LOG_INFO(SRAM_SLICE, "Selected bundling policy: {}", policyStr[bundlingPolicy]);
    return bundlingPolicy;
}

// Pipeline slicing pass "main"
bool sliceGraphForPipeline(HabanaGraph& g)
{
    // Pipeline management disabled falls-back to Gaudi 1 algorithm.
    auto pipelineMgmtEnabled = GCFG_ENABLE_PIPELINE_MANAGEMENT.value();
    LOG_INFO(SRAM_SLICE, "Pipeline management enabled: {}", pipelineMgmtEnabled ? "Yes" : "No");

    if (pipelineMgmtEnabled == false)
    {
        // legacy SRAM management
        SRAMSlicingManager sramSlicingManager(g);
        return sramSlicingManager.sliceGraph();
    }

    // Shapshot pre-slicing tensor sizes for later validation
    snapshotPreSlicingSizes(g);

    SlicingBrain           dummyBrain(g);  // required to intialize SlicingBrain::knobs
    PipelineSlicingManager pipelineManager(g, findBestBundlingPolicyForGraph(g));
    return pipelineManager.optimizeGraph();
}

bool PipelineSlicingManager::applyPostSlicingCorrections()
{
    // re-connect normalization undirected cycles post slicing
    bool res = m_normsHandler.handleRemovedSliceNormNodes();
    if (!res)
    {
        LOG_ERR(SRAM_SLICE, "Failed to re-connect undirected normalization cycles.");
        return false;
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Temporary hack for pipeline management over Gaudi3
// Add perforation annotation over the big nodes and big tensors of the bundle.
//
class LitePerforationAnnotator
{
public:
    LitePerforationAnnotator(const HabanaGraph& graph, const PipelineBundlePtr& bundle)
    : m_nodes(bundle->getNodes().begin(), bundle->getNodes().end()), m_tensors(allIntermediateOperands(graph))
    {
        m_nodeGranularity = CommonTileSizeCalculator::getMinCommonTilesSizes(m_nodes, m_tensors, graph).second;
    }

    void annotateByStrategy(const BundleStrategyPtr& strategy) const
    {
        for (const TensorPtr& t : m_tensors)
        {
            SLC_DEBUG("Annotating tensor {} with perforation information", t->getName());
            annotateTensor(t, strategy);
        }

        for (const NodePtr& n : m_nodes)
        {
            SLC_DEBUG("Annotating node {} with perforation information", n->getNodeName());
            annotateNode(n, strategy);
        }
    }

private:
    const NodeSet   m_nodes;
    const TensorSet m_tensors;
    TileSizePerNode m_nodeGranularity;

    TensorSet allIntermediateOperands(const HabanaGraph& graph)
    {
        TensorSet operands;
        for (const NodePtr& n : m_nodes)
        {
            // Add all intermediate tensors - produced and consumed by a bundle node
            for (const TensorPtr& t : n->getOutputs())
            {
                if (!t || t->isShapeTensor()) continue;
                for (const NodePtr& cons : graph.getTensorConsumers(t))
                {
                    if (isBundled(cons)) operands.insert(t);
                }
            }
            // Add shared input for multiple MME nodes - even if not produced in the bundle, it is required to connect
            // the MME nodes subgraph. Shared input for non 2 MMEs might close a directed cycle in the bundle sub-graph,
            // which currently fails the common granularity calculator.
            if (HabanaGraph::runsOnMME(n))
            {
                for (const TensorPtr& t : n->getInputs())
                {
                    if (!t || t->isShapeTensor()) continue;
                    for (const NodePtr& cons : graph.getTensorConsumers(t))
                    {
                        // look for shared input MME consumer in the bundle
                        if (cons != n && HabanaGraph::runsOnMME(cons) && isBundled(cons)) operands.insert(t);
                    }
                }
            }
        }
        return operands;
    }

    bool isBundled(const NodePtr& node) { return std::find(m_nodes.begin(), m_nodes.end(), node) != m_nodes.end(); }

    void annotateTensor(const TensorPtr& tensor, const BundleStrategyPtr& strategy) const
    {
        HB_ASSERT(tensor && !tensor->isShapeTensor(), "{}: Unexpected invalid tensor to annotate", __func__);
        const auto& slicedOperand = strategy->getSlicingData().getSlicedOperand(tensor);
        HB_ASSERT_PTR(slicedOperand);
        tensor->getTensorAnnotation().perforation =
            LitePerforationLocalityHints {slicedOperand->resideInSRAM,
                                          !SlicedOperandUtils::isTriviallySliced(slicedOperand)};
    }

    void annotateNode(const NodePtr& node, const BundleStrategyPtr& strategy) const
    {
        // Only MME and TPC nodes are interesting to perforate
        if (HabanaGraph::runsOnTPC(node) ||
            (HabanaGraph::runsOnMME(node) && node->getNodeType() != Node::TYPE_INTERNAL_TRANSPOSE))
        {
            for (const TensorPtr& t : node->getInputs())
            {
                if (annotateNodeByTensor(node, t, strategy)) return;
            }
            for (const TensorPtr& t : node->getOutputs())
            {
                if (annotateNodeByTensor(node, t, strategy)) return;
            }
        }
    }

    bool annotateNodeByTensor(const NodePtr& node, const TensorPtr& tensor, const BundleStrategyPtr& strategy) const
    {
        if (!tensor || tensor->isShapeTensor()) return false;

        const auto& ap         = node->getNodeAccessPattern();
        const auto& sliced     = strategy->getSlicingData().getSlicedOperand(tensor);

        for (size_t tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
        {
            if (sliced->chunkDimensions[tensorDim] < sliced->finalShape[tensorDim])
            {
                // All non-transpose MME and TPC nodes support getIndexSpaceDim
                size_t idxSpaceDim = ap->getIndexSpaceDim(tensor, tensorDim);
                SLC_DEBUG("Found sliced index space dimension from tensor '{}' dim: {}", tensor->getName(), tensorDim);
                node->getNodeAnnotation().perforation =
                    LitePerforationHints {idxSpaceDim, m_nodeGranularity.at(node).at(idxSpaceDim)};
                return true;
            }
        }

        return false;
    }
};

static void
setPerforationAnnotation(const HabanaGraph& graph, const PipelineBundlePtr& bundle, const BundleStrategyPtr& strategy)
{
    if (!GCFG_ENABLE_BRAIN_LOCALITY_HINTS_ANNOTATION.value()) return;

    SLC_INFO("setting perforation annotation to bundle {}", bundle->getName());
    LitePerforationAnnotator(graph, bundle).annotateByStrategy(strategy);
}
// End of lite perforation hacks
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
