#include "sliced_bundle_graph_generator.h"
#include "access_pattern.h"
#include "pass_manager.h"

using namespace gc::layered_brain;

HabanaGraphPtr SlicedBundleGraphGenerator::createEmptySlicedGraph() const
{
    auto slicedGraph = m_graph.createEmptyGraph();
    // Copy pass manager from the original graph here to start collecting predicates that trigger from building the
    // sliced graph
    auto passManager = m_graph.clonePassManager();
    slicedGraph->setPassManager(passManager);
    // Copy memory coherence data - so that control edges will be copied along with the nodes.
    slicedGraph->getGraphAnnotation().memoryCoherence = m_graph.getGraphAnnotation().memoryCoherence;
    slicedGraph->getGraphAnnotation().partialGraph = true;
    slicedGraph->setLayeredBrainData(std::make_unique<LayeredBrainData>());
    return slicedGraph;
}

// Find the bundle engine - needed by the graph scheduler later.
BundleEngine SlicedBundleGraphGenerator::getBundleEngine() const
{
    bool runsOnMme = false;
    bool runsOnTpc = false;
    bool runsOnDma = false;
    for (const auto& node : m_bundleNodes)
    {
        if (HabanaGraph::runsOnTPC(node))
        {
            runsOnTpc = true;
        }
        else if (HabanaGraph::runsOnMME(node))
        {
            runsOnMme = true;
        }
        else if (node->isDma())
        {
            runsOnDma = true;
        }
    }
    BundleEngine bundleEngine = runsOnMme && runsOnTpc ? BundleEngine::ENGINE_MME_TPC
                                : runsOnMme            ? BundleEngine::ENGINE_MME
                                : runsOnTpc            ? BundleEngine::ENGINE_TPC
                                : runsOnDma            ? BundleEngine::ENGINE_DMA
                                                       : BundleEngine::ENGINE_UNDEFINED;
    HB_ASSERT(bundleEngine != BundleEngine::ENGINE_UNDEFINED, "Bundle {} has no MME/TPC/DMA nodes", m_bundleIdx);
    return bundleEngine;
}

void SlicedBundleGraphGenerator::addSlicedNodesToSlicedGraph() const
{
    BundleEngine bundleEngine = getBundleEngine();
    LOG_DEBUG(LB_SLICER, "Add sliced bundle nodes to sliced graph, bundle engine is set to : {}", bundleEngine);
    for (const auto& node : m_slicedNodes)
    {
        LOG_DEBUG(LB_SLICER, "\t Add node {} (type {}) to sliced graph", node->getNodeName(), node->getNodeTypeStr());
        HB_ASSERT(node->getNodeAnnotation().bundleInfo.is_set(),
                  "Adding bundle node {} without bundle info set",
                  node->getNodeName());
        node->getNodeAnnotation().bundleInfo->bundleEngine = bundleEngine;
        bool res                                           = GraphEditor::addNode(*m_slicedGraph, node);
        HB_ASSERT(res, "Failed to add node {} to sliced graph", node->getNodeName());
    }
}

// Create a slice for the tensor (or reuse if it was already created), and attach it to the sliced node.
// Reduction/BPT data is collected as well to avoid a second iteration later.
void SlicedBundleGraphGenerator::replaceOperandWithSlicedTensor(const NodePtr&   origNode,
                                                                const NodePtr&   slicedNode,
                                                                const BVDCoord&  nodeBVDCoord,
                                                                const TensorPtr& origTensor,
                                                                unsigned         tensorIdx,
                                                                bool             isInput)
{
    BaseClass::replaceOperandWithSlicedTensor(origNode, slicedNode, nodeBVDCoord, origTensor, tensorIdx, isInput);

    HB_ASSERT_PTR(m_slicedGraph);
    HB_ASSERT_PTR(m_slicedGraph->getLayeredBrainData());

    const auto&      tensorBVDCoord = m_bvdCoordsGenerator.projectBVDCoordOnTensor(origTensor, nodeBVDCoord);
    const TensorPtr& slicedTensor   = isInput ? slicedNode->getInput(tensorIdx) : slicedNode->getOutput(tensorIdx);
    // Save the BVD coordinates of the slice to update the bundle data after the join node is added
    m_slicesCoords[slicedTensor] = tensorBVDCoord;
}

HabanaGraphPtr SlicedBundleGraphGenerator::createSlicedGraph()
{
    SET_TEMP_LOG_CONTEXT(fmt::format("SlicedBundleGraphGenerator_bundle#{}", m_bundleIdx));

    LOG_TRACE(LB_SLICER, "Create sliced graph for bundle {}", m_bundleIdx);

    m_slicedGraph = createEmptySlicedGraph();
    BaseClass::createSlicedNodes();
    addSlicedNodesToSlicedGraph();
    cacheSlicerReductions();
    projectCoordsOnReductionInputs();
    updateBundleData();
    if (m_dryRun)
    {
        insulateSlicedGraph();
    }

    LOG_TRACE(LB_SLICER, "Sliced graph for bundle {} contains {} nodes", m_bundleIdx, m_slicedGraph->getNumNodes());

    return std::move(m_slicedGraph);
}

void SlicedBundleGraphGenerator::cacheSlicerReductions()
{
    for (const auto& n : m_slicedNodes)
    {
        // Skip nodes that aren't reduction
        if (n->getNodeType() != Node::TYPE_INTERNAL_REDUCTION) continue;
        // Skip reductions that have inputs we cannot project bvd coords onto,
        // but keep reductions with memset and multiple real producers
        if (!std::all_of(n->getInputs().begin(), n->getInputs().end(), [this, n](const TensorPtr& t) {
                const auto& prod = m_slicedGraph->getTensorProducer(t);
                return prod && (m_slicedNodeToCoord.find(prod) != m_slicedNodeToCoord.end() ||
                                (prod->isMemset() && n->getNumInputs() > 2));
            }))
        {
            LOG_TRACE(LB_SLICER, "Skipping unsupported reduction node {}", n->getNodeName());
            continue;
        }
        LOG_TRACE(LB_SLICER, "Caching slicer reduction node {}", n->getNodeName());
        m_slicerReductions.insert(n);
    }
}

void SlicedBundleGraphGenerator::projectCoordsOnReductionInputs()
{
    for (const auto& reduction : m_slicerReductions)
    {
        HB_ASSERT(m_slicedNodes.find(reduction) != m_slicedNodes.end(),
                  "Expecting reduction node {} in sliced nodes container",
                  reduction->getNodeName());
        for (const auto& reductionInput : reduction->getInputs())
        {
            const auto& prod = m_slicedGraph->getTensorProducer(reductionInput);
            HB_ASSERT_PTR(prod);
            // memset producer isn't expected to have coordinate, and we don't want to schedule this input separately
            if (prod->isMemset()) continue;
            const auto& prodBVDCoordIt = m_slicedNodeToCoord.find(prod);
            HB_ASSERT(prodBVDCoordIt != m_slicedNodeToCoord.end(),
                      "Expecting a bvd coord entry for node {}[{}]",
                      prod->getNodeName(),
                      prod->getNodeTypeStr());
            m_slicesCoords.emplace(reductionInput, prodBVDCoordIt->second);
        }
    }
}

void SlicedBundleGraphGenerator::updateBundleData() const
{
    HB_ASSERT_PTR(m_slicedGraph->getLayeredBrainData());
    auto& bundleData = m_slicedGraph->getLayeredBrainData()->m_bundleData[m_bundleIdx];
    HB_ASSERT_PTR(m_bundleViews);
    bundleData.setBundleViews(m_bundleViews);
    bundleData.setFinalStrategy(m_strategy);
    const auto& numOfSlicesPerBVD = m_bvdCoordsGenerator.getNumOfSlicesPerBVD();
    if (areAllElementsEqual(numOfSlicesPerBVD.begin(), numOfSlicesPerBVD.end(), 1))
    {
        LOG_WARN(LB_SLICER, "Bundle {} is not sliced", m_bundleIdx);
    }

    bundleData.setNumOfSlicesPerBVD(numOfSlicesPerBVD);
    for (const auto& n : m_slicedNodes)
    {
        if (Node::isJoinNode(n) || m_slicerReductions.find(n) != m_slicerReductions.end())
        {
            HB_ASSERT(n->getNumOutputs() == 1,
                      "Expected a single output (found: {}) for node {}[{}]",
                      n->getNumOutputs(),
                      n->getNodeName(),
                      n->getNodeTypeStr());
            std::vector<BVDCoord> inputsCoords;
            inputsCoords.resize(n->getNumInputs());
            for (unsigned inputIdx = 0; inputIdx < n->getNumInputs(); inputIdx++)
            {
                const auto& input = n->getInput(inputIdx);
                HB_ASSERT_PTR(input);
                // leave memset zeros tensor without BVD coord, so it won't enter its own slice set.
                // the memset node should be scheduled through ctrl dep and not on its own.
                const auto& prod = m_slicedGraph->getTensorProducer(input);
                HB_ASSERT_PTR(prod);
                if (prod->isMemset()) continue;
                const auto coordsIt = m_slicesCoords.find(input);
                HB_ASSERT(coordsIt != m_slicesCoords.end(),
                          "Expecting a bvd coord entry for {}[{}] slice {}",
                          n->getNodeTypeStr(),
                          n->getNodeName(),
                          input->getName());
                inputsCoords.at(inputIdx) = coordsIt->second;
            }
            bundleData.addRouteEndInputsCoords(n, inputsCoords);
        }
    }
}

// Replace the BPTs with clones, so that any change to the sliced graph will not contaminate the original graph.
void SlicedBundleGraphGenerator::insulateSlicedGraph()
{
    HB_ASSERT_PTR(m_slicedGraph->getLayeredBrainData());
    auto&      bundleData  = m_slicedGraph->getLayeredBrainData()->m_bundleData[m_bundleIdx];
    const auto origTensors = m_slicedGraph->getTensors();  // save a copy since the loop will alter the graph's tensors
    for (const auto& orig : origTensors)
    {
        if (m_bptHandler.isBPT(orig))
        {
            const TensorPtr replacement = getBPTReplacement(orig);
            LOG_DEBUG(LB_SLICER, "Replacing {} in the sliced graph with {}", orig->getName(), replacement->getName());
            GraphEditor::replaceTensor(*m_slicedGraph, orig, replacement);
            bundleData.addBPTClonePersistence(replacement, orig->isUserManagedDram());
        }
    }
}

TensorPtr SlicedBundleGraphGenerator::getBPTReplacement(const TensorPtr& origBPT)
{
    // Cloning method:
    // Address and data should be irrelevant to the brain and the generic passes it uses.
    // Persistence can't be copied, since the sliced graph uses different section namespace. Need to re-assign section
    // IDs that are allocated in the sliced graph (done below).
    // Name will be suffixed with some unique identifier which should enable easier log analysis when looking at
    // different evaluation steps for different bundles and bundle compositions that interract with the original tensor.
    TensorPtr replacement = origBPT->clone(false /*copy address*/, false /*copy data*/, false /*copy persistence*/);

    // non-persistent BPT may lead to errors or inconsistencies in the operations of generic passes on the graph, so
    // setting all of them as persistents in the sliced graph.
    const auto tmpSectionId =
        m_slicedGraph->getCodeGenerator()->getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);
    replacement->setMemoryDescriptor(synMemoryDescriptor {true});
    replacement->setMemorySectionID(tmpSectionId);

    LOG_DEBUG(LB_SLICER,
              "Created: {} and set as persistent in sliced graph (section: {})",
              replacement->getName(),
              tmpSectionId);
    return replacement;
}

NodeSet SlicedBundleGraphGenerator::getRequireCastNodes(const StrategyPtr& strategy) const
{
    NodeSet requireCastNodes;
    if (strategy->getMmeSolution())
    {
        for (const auto& [node, solutionParams] : strategy->getMmeSolution()->QORs)
        {
            if (solutionParams->solutionRequirements.requiresCast)
            {
                requireCastNodes.insert(node);
            }
        }
    }
    return requireCastNodes;
}

NodeSet SlicedBundleGraphGenerator::getRequireMemsetNodes(const StrategyPtr& strategy) const
{
    NodeSet requireMemsetNodes;
    if (strategy->getMmeSolution())
    {
        for (const auto& [node, solutionParams] : strategy->getMmeSolution()->QORs)
        {
            if (solutionParams->solutionRequirements.requiresMemset)
            {
                requireMemsetNodes.insert(node);
            }
        }
    }
    return requireMemsetNodes;
}
