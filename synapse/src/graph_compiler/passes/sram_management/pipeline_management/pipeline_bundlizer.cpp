#include "pipeline_bundlizer.h"
#include "access_pattern.h"
#include "bundle.h"
#include "bundle_plane_graph.h"
#include "defs.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "log_manager.h"
#include "mme_node.h"
#include "node.h"
#include "sram_management/bundle_paths_validation.h"
#include "sram_management/pipeline_management/node_solver.h"
#include "sram_management/slicing_brain.h"
#include "mme_brain_proxy.h"
#include "bundlizer_utils.h"
#include "tensor.h"
#include "type_utils.h"
#include "types.h"
#include <algorithm>
#include <optional>
#include <stack>

using namespace gc::access_pattern;

static uint64_t getOutputSizeForPartials(const NodePtr& n)
{
    const TensorPtr& output = n->getOutput(0);
    return output->getDenseSizeInElements() *
           dataTypeSizeInBytes(SlicedOperandUtils::getTypeForPartials(output->getElementType()));
}

static std::pair<NodeSet, TensorSet>
getChainsNodesAndTensors(const TPCExpansionsAndSingleMMEBundleSolver::PipelineMultiChain& chains)
{
    NodeSet   nodes;
    TensorSet tensors;

    for (const auto& chain : chains)
    {
        for (const auto& pipelineNode : chain)
        {
            nodes.insert(pipelineNode.node);
            tensors.insert(pipelineNode.connectingTensor);
        }
    }
    return {nodes, tensors};
}

void PipelineBundlizer::addNodeToBundle(const NodePtr& node, PipelineBundlePtr& bundle)
{
    BundleInfo info(bundle->index(), bundle->type());
    bundle->addNode(node);
    HB_ASSERT(!node->getNodeAnnotation().bundleInfo.is_set(),
              "Cannot add a node {} to bundle {} if it's already belong to bundle {}",
              node->getNodeName(),
              bundle->index(),
              node->getNodeAnnotation().bundleInfo->bundleIndex);
    node->getNodeAnnotation().bundleInfo.set(info);
    m_graph.getBPGraph()->addNodeToBundle(node, info);
    LOG_DEBUG(SRAM_SLICE, "Add node {} to bundle {} (type {})", node->getNodeName(), bundle->index(), bundle->type());
}

void PipelineBundlizer::removeNodeFromBundle(const NodePtr& node, PipelineBundlePtr& bundle)
{
    bundle->removeNode(node);
    m_graph.getBPGraph()->unbundleNode(node);
    node->getNodeAnnotation().bundleInfo.unset();
    LOG_DEBUG(SRAM_SLICE,
              "Remove node {} from bundle {} (type {})",
              node->getNodeName(),
              bundle->index(),
              bundle->type());
}

void PipelineBundlizer::removeBundle(PipelineBundlePtr bundle)
{
    const auto bundleNodesVectorCopy = bundle->getNodes();
    for (const NodePtr& n : bundleNodesVectorCopy)
    {
        n->getNodeAnnotation().bundleInfo.unset();
        bundle->removeNode(n);
    }
    if (!bundleNodesVectorCopy.empty())
    {
        m_graph.getBPGraph()->removeBundle(bundleNodesVectorCopy[0]);
    }
}

TensorGranularity PipelineBundlizer::getMmeSharedInputGranularity(const MMENodeSet&        mmeNodes,
                                                                  const TensorPtr&         input,
                                                                  const std::vector<Dim>&  slicedDims,
                                                                  const TensorGranularity& granularity) const
{
    TensorGranularity sharedOperandGranularity(granularity);
    // Get mme nodes min slice sizes (not clipped) LCM, to find the min granularity capacity multiplier
    for (Dim sliceDim : slicedDims)
    {
        unsigned mmeSlicedDimSizeLCM = 1;
        for (const MMENodePtr& n : mmeNodes)
        {
            // Calculate the common dim size alignment for all MME nodes, which takes MME utilization into account,
            // without any slicing granularity constraint (thus call with granularity = 1).
            // Apply the slicing granularity constraint on the common alignment after it is accumulated.
            unsigned minSlicedDimSize = MmeNodeSolver::getMinSlicedDimSizeInElements(n, input, sliceDim, 1);
            LOG_TRACE(SRAM_SLICE,
                      "Min slice dim size {} elements for {} on dim {}",
                      minSlicedDimSize,
                      n->getNodeName(),
                      sliceDim);
            mmeSlicedDimSizeLCM = std::lcm(minSlicedDimSize, mmeSlicedDimSizeLCM);
        }
        // Now apply the dim slicing granularity constraint on the common (LCM) size.
        mmeSlicedDimSizeLCM = MmeNodeSolver::alignDimSizeToGranularity(mmeSlicedDimSizeLCM, granularity[sliceDim]);
        LOG_TRACE(SRAM_SLICE,
                  "{}: Sliced dim MME util granularity {} -> aligned to bundle granularity: {}",
                  HLLOG_FUNC,
                  granularity[sliceDim],
                  mmeSlicedDimSizeLCM);
        sharedOperandGranularity[sliceDim] = mmeSlicedDimSizeLCM;
    }
    return sharedOperandGranularity;
}

uint64_t PipelineBundlizer::getSlicedTensorMinSize(const TensorPtr&            t,
                                                   const TensorTile::Geometry& granularity,
                                                   const std::vector<Dim>&     slicedDims) const
{
    NSizeArray sliceSizes = t->getAllNSizesInElements();
    // Update the sliced dims size to the granularity. Limit the slice size to tensor size
    std::map<unsigned, TSize> sizePerSlicedDim;
    for (auto slicedDim : slicedDims)
    {
        sizePerSlicedDim.emplace(slicedDim, std::min(granularity[slicedDim], (unsigned long)(sliceSizes[slicedDim])));
    }
    unsigned minSliceSizeInBytes = SlicedOperandUtils::getTensorSliceSizeInBytes(t, sizePerSlicedDim);
    LOG_TRACE(SRAM_SLICE,
              "{}: Min slice size for tensor {}: [{}] ==> {} MB",
              HLLOG_FUNC,
              t->getName(),
              toString(sliceSizes.begin(), std::next(sliceSizes.begin(), t->getDim()), ','),
              bToMb(minSliceSizeInBytes));

    return minSliceSizeInBytes;
}

bool PipelineBundlizer::isSupportedLogicalNodeType(const NodePtr& node) const
{
    HB_ASSERT(node->isLogicalOperation(), "This function is not valid for non logical nodes");
    switch (node->getNodeType())
    {
        case Node::TYPE_INTERNAL_RESHAPE:
        case Node::TYPE_STATIC_RESHAPE:
        case Node::TYPE_INTERNAL_PACKING:
        case Node::TYPE_LOGICAL_TRANSPOSE:
        case Node::TYPE_INTERNAL_EXPAND_DIMS:
        case Node::TYPE_SQUEEZE_NODE:
            return true;
        default:
            return false;
    }
}

BundlesInfoContainer TPCExpansionsAndSingleMMEBundlizer::generateBundles()
{
    LOG_TRACE(SRAM_SLICE, "TPCExpansionsAndSingleMMEBundlizer::{}", HLLOG_FUNC);
    auto const& bundles = generateMmeWithTpcExpansionsBundles();
    HB_ASSERT(validateBundles(bundles), "Failed bundles validation");
    return bundles;
}

NodeVector TPCExpansionsAndSingleMMEBundlizer::getNodesBundleCreationOrder(const NodeSet& nodes)
{
    return NodeVector(nodes.begin(), nodes.end());
}

bool TPCExpansionsAndSingleMMEBundlizer::validateBundles(const BundlesInfoContainer& bundlesAndBundleSolvers) const
{
    for (const BundleAndSolver& bundleAndSolver : bundlesAndBundleSolvers)
    {
        const auto& bundleSolver = bundleAndSolver.second;
        if (!bundleSolver->validateBundle())
        {
            // error logged from within validateBundle()
            return false;
        }
    }
    return true;
}

bool TPCExpansionsAndSingleMMEBundlizer::isSupportedLogicalNode(const NodePtr& node) const
{
    bool supported(isSupportedLogicalNodeType(node));
    if (node->getNodeType() == Node::TYPE_LOGICAL_TRANSPOSE)
    {
        supported &= false;
    }
    return supported;
}

BundlesInfoContainer TPCExpansionsAndSingleMMEBundlizer::generateMmeWithTpcExpansionsBundles()
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);

    BundlesInfoContainer bundles;

    for (const pNode& n : getNodesBundleCreationOrder(m_graph.getNodes()))
    {
        if (n->getNodeAnnotation().bundleInfo.is_set())
        {
            LOG_DEBUG(SRAM_SLICE, "node {} already belongs to bundle!", n->getNodeName());
            continue;
        }
        if (HabanaGraph::runsOnMME(n))
        {
            std::optional<BundleAndSolver> nodeBundleAndSolver = getMmeWithTpcExpansionsBundle(n);
            if (nodeBundleAndSolver)
            {
                bundles.push_back(*nodeBundleAndSolver);
            }
        }
    }

    return bundles;
}

TPCExpansionsAndSingleMMEBundlizer::PipelineChain
TPCExpansionsAndSingleMMEBundlizer::getProducerChain(const NodePtr&   mmeNode,
                                                     const TensorPtr& input,
                                                     const NodeSet&   otherNodesInBundle) const
{
    if (!NodeSolver::doesNodeSupportSlicing(mmeNode))
    {
        return {};
    }
    unsigned inputIdx = mmeNode->getInputIndexOfTensor(input);
    if (!NodeSolver::isInputSliceable(mmeNode, inputIdx))
    {
        return {};
    }
    // TODO [SW-156603]: resolve this issue: this function creates producer chain based on single slice dim,
    // in mantaray bundles this producer chain is only used to decide which input to slice, it means that the created
    // producer chain has an influence on bundles that sliced on more than one dim (currently just mantaray).
    auto slicingDims = NodeSolver::getInputSlicingDims(mmeNode, inputIdx);
    HB_ASSERT(!slicingDims.empty(), "supported node must have slicing dim");
    Dim           sliceDim  = slicingDims.front();  // currently vision bundlizer supports only a single dim
    PipelinedNode candidate = createProducerCandidate(input, {sliceDim});
    return expandProducerChain(mmeNode, candidate, {}, otherNodesInBundle);
}

bool TPCExpansionsAndSingleMMEBundlizer::producerChainsIntersect(const PipelineMultiChain& producerChains)
{
    if (producerChains.size() < 2) return false;  // One or less chains cannot intersect.

    TensorSet producedTensors;
    for (const PipelineChain& prodChain : producerChains)
    {
        for (const PipelinedNode& producer : prodChain)
        {
            const TensorPtr& tensor = producer.connectingTensor;
            if (producedTensors.find(tensor) != producedTensors.end())
            {
                LOG_DEBUG(SRAM_SLICE,
                          "Tensor {} appeared twice in the producer chains => the chains intersect.",
                          tensor->getName());
                return true;
            }
            producedTensors.insert(tensor);
        }
    }

    return false;
}

TPCExpansionsAndSingleMMEBundlizer::PipelinedNode
TPCExpansionsAndSingleMMEBundlizer::createProducerCandidate(const TensorPtr&        tensor,
                                                            const std::vector<Dim>& slicingDims) const
{
    // The caller is responsible to make sure the producer node is not null.
    return PipelinedNode(m_graph.getTensorProducer(tensor), tensor, slicingDims);
}

void TPCExpansionsAndSingleMMEBundlizer::clipProducerChain(PipelineChain& currentChain, unsigned maxTPCs) const
{
    while (currentChain.size() > MAX_CHAIN_SIZE)
    {
        currentChain.pop_back();
    }
    unsigned nTPCs = 0;
    for (const PipelinedNode& p : currentChain)
    {
        if ((p.node != nullptr) && (HabanaGraph::runsOnTPC(p.node)))
        {
            ++nTPCs;
        }
    }
    if (maxTPCs == 0)
    {
        maxTPCs = nTPCs;
    }

    while (!currentChain.empty())
    {
        const auto& it = currentChain.rbegin();
        HB_ASSERT(it->node != nullptr, "Malformed producer chain");
        if (it->node->isLogicalOperation())
        {
            currentChain.pop_back();
            continue;
        }
        if ((nTPCs > maxTPCs) && (HabanaGraph::runsOnTPC(it->node)))
        {
            currentChain.pop_back();
            --nTPCs;
            continue;
        }
        break;
    }
}

TPCExpansionsAndSingleMMEBundlizer::PipelineChain
TPCExpansionsAndSingleMMEBundlizer::expandProducerChain(const NodePtr&       finalConsumer,
                                                        const PipelinedNode& nextCandidate,
                                                        PipelineChain        currentChain,
                                                        const NodeSet&       otherNodesInBundle) const
{
    // Before returning from this function - clipProducerChain is assumed to be called on currentChain
    const NodePtr&   producerNode     = nextCandidate.node;
    const TensorPtr& producedTensor   = nextCandidate.connectingTensor;

    LOG_DEBUG(SRAM_SLICE,
              "{}: Expanding with producer: {} (GUID: {})",
              HLLOG_FUNC,
              producerNode ? producerNode->getNodeName() : "N/A",
              producerNode ? producerNode->getGUID() : "N/A");
    LOG_DEBUG(SRAM_SLICE, "    Connecting tensor: {}", producedTensor ? producedTensor->getName() : "N/A");

    bool canIncludeCurrentNode    = true;
    bool canExtendPastCurrentNode = true;

    if (!producerNode || producerIsChainBreaker(nextCandidate, currentChain, finalConsumer, otherNodesInBundle) ||
        currentChain.size() > TPCExpansionsAndSingleMMEBundlizer::MAX_CHAIN_SIZE)
    {
        canIncludeCurrentNode = false;
    }

    if (canIncludeCurrentNode)
    {
        currentChain.emplace_back(nextCandidate);
    }

    canExtendPastCurrentNode = canIncludeCurrentNode && !isProducerChainBoundary(nextCandidate.node);

    if (canExtendPastCurrentNode)
    {
        // Try extending the chain on the first input that has matching slicingDim
        for (const TensorPtr& nextProducedTensor : producerNode->getInputs())
        {
            std::vector<Dim> inputSlicingDims = NodeSolver::getTensorMatchingSlicedDims(producerNode,
                                                                                        producedTensor,
                                                                                        nextCandidate.slicingDims,
                                                                                        nextProducedTensor);
            // In proucer expansion, we want the slicing dims of the candidate to be mapped 1:1 to its producer slicing
            // dims.
            if (inputSlicingDims.size() == nextCandidate.slicingDims.size())
            {
                PipelinedNode candidate = createProducerCandidate(nextProducedTensor, inputSlicingDims);
                auto expandedChain = expandProducerChain(finalConsumer, candidate, currentChain, otherNodesInBundle);
                if (expandedChain.size() > currentChain.size())
                {
                    // expansion successful for this input
                    return expandedChain;
                }
                else
                {
                    LOG_DEBUG(SRAM_SLICE,
                              "{}: Can't expand producer chain towards tensor {}. moving on to next input.",
                              HLLOG_FUNC,
                              nextProducedTensor ? nextProducedTensor->getName() : "nullptr");
                }
            }
            else
            {
                LOG_DEBUG(SRAM_SLICE,
                          "{}: Could not project slicing dims onto input of node {}. Chain moving on to next input.",
                          HLLOG_FUNC,
                          producerNode->getNodeName());
            }
        }
    }

    clipProducerChain(currentChain);
    return currentChain;
}

bool TPCExpansionsAndSingleMMEBundlizer::producerIsChainBreaker(const PipelinedNode& nextCandidate,
                                                                const PipelineChain& currentChain,
                                                                const NodePtr&       finalConsumer,
                                                                const NodeSet&       otherNodesInBundle) const
{
    if (commonChainBreakerChecks(nextCandidate.node)) return true;

    if (areAssociatedWithDifferentFlashAttention(nextCandidate.node, finalConsumer))
    {
        LOG_DEBUG(SRAM_SLICE,
                  "Candidate {} and consumer {} are associated with different FA",
                  __func__,
                  nextCandidate.node->getNodeName(),
                  finalConsumer->getNodeName());
        return true;
    }

    // Check bundling doesn't create circles in the bundle-plane graph
    if (producerCyclesChainBreaker(nextCandidate, currentChain, finalConsumer, otherNodesInBundle))
    {
        LOG_DEBUG(SRAM_SLICE, "{} - creates a circle in BP graph", nextCandidate.node->getNodeName());
        return true;
    }

    if (producerSharedInputChainBreaker(nextCandidate, currentChain, finalConsumer, otherNodesInBundle)) return true;
    if (producerMultipleConnectingTensorsChainBreaker(nextCandidate, currentChain, finalConsumer)) return true;

    if (HabanaGraph::runsOnTPC(nextCandidate.node))
    {
        return isTPCChainBreaker(nextCandidate);
    }
    else if (nextCandidate.node->isLogicalOperation())
    {
        return isLogicalChainBreaker(nextCandidate,
                                     nextCandidate.node->getInput(0));  // TODO [SW-74163] should it always be input[0]?
    }
    // Currently, only TPC and logical ops are supported
    LOG_DEBUG(SRAM_SLICE, "{}: {} - unsupported op", HLLOG_FUNC, nextCandidate.node->getNodeName());
    return true;
}

bool TPCExpansionsAndSingleMMEBundlizer::areAssociatedWithDifferentFlashAttention(const NodePtr& nextCandidate,
                                                                    const NodePtr&       finalConsumer) const
{
    bool candidateRegistered = m_graph.getGraphAnnotation().flashAttentionDb.isRegistered(nextCandidate->getParentId());
    bool consumerRegistered  = m_graph.getGraphAnnotation().flashAttentionDb.isRegistered(finalConsumer->getParentId());
    bool isDiffParentId      = (nextCandidate->getParentId() != finalConsumer->getParentId());
    return (candidateRegistered != consumerRegistered) || (candidateRegistered && isDiffParentId);
}

bool TPCExpansionsAndSingleMMEBundlizer::producerCyclesChainBreaker(const PipelinedNode& nextCandidate,
                                                                    const PipelineChain& currentChain,
                                                                    const NodePtr&       finalConsumer,
                                                                    const NodeSet&       otherNodesInBundle) const
{
    const auto& producer         = nextCandidate.node;
    const auto& connectingTensor = nextCandidate.connectingTensor;
    BundlePathsValidation pathsValidation(m_graph);
    NodeSet               acceptedNodes = otherNodesInBundle;
    acceptedNodes.insert(finalConsumer);
    std::for_each(currentChain.begin(), currentChain.end(), [&](const PipelinedNode& p) {
        acceptedNodes.insert(p.node);
    });
    return !pathsValidation.validateProducerPaths(producer, connectingTensor, acceptedNodes, {});
}

// Sharing bundle intermediate input between 2 producers or between a producer and the final consumer is not supported.
// Current issues with it:
// 1. LCM (common granularity) calculator may produce the wrong granularity for the shared bundle intermediate tensor.
//    BPTs are not included in the LCM calc.
// 2. 2 different SlicedOperands may be generated for it and there is no telling which one belongs to which operation
//    (i.e getSlicedOperand will always return the first one added to the strategy)
bool TPCExpansionsAndSingleMMEBundlizer::producerSharedInputChainBreaker(const PipelinedNode& nextCandidate,
                                                                         const PipelineChain& currentChain,
                                                                         const NodePtr&       finalConsumer,
                                                                         const NodeSet&       otherNodesInBundle) const
{
    std::unordered_set<NodePtr> currentNodes(otherNodesInBundle.begin(), otherNodesInBundle.end());
    currentNodes.insert(finalConsumer);
    for (const PipelinedNode& pn : currentChain)
    {
        currentNodes.insert(pn.node);
    }

    for (const TensorPtr& candidateInput : nextCandidate.node->getInputs())
    {
        if (!candidateInput) continue;
        if (candidateInput->isNonScratchpadAuxTensor())
        {
            // There is no problem sharing aux tensors between producers as long as they are not scratch-pad aux tensors
            continue;
        }
        const NodePtr& inputProducer        = m_graph.getTensorProducer(candidateInput);
        bool           isBundleIntermediate = currentNodes.find(inputProducer) != currentNodes.end();
        for (const NodePtr& inputConsumer : m_graph.getTensorConsumers(candidateInput))
        {
            // Block only intermediate tensors, as double creation of the tensor in the bundle for
            // them isn't handled yet
            if (currentNodes.find(inputConsumer) != currentNodes.end() && isBundleIntermediate)
            {
                LOG_DEBUG(SRAM_SLICE,
                          "{}: Chain candidate {} shares an input {} with {} which is already in the chain or the "
                          "bundle. Such sharing is not supported.",
                          HLLOG_FUNC,
                          nextCandidate.node->getNodeName(),
                          candidateInput->getName(),
                          inputConsumer->getNodeName());
                return true;
            }
        }
    }
    return false;  // No need to break
}

bool TPCExpansionsAndSingleMMEBundlizer::producerMultipleConnectingTensorsChainBreaker(
    const PipelinedNode& nextCandidate,
    const PipelineChain& currentChain,
    const NodePtr&       finalConsumer) const
{
    const NodePtr&   producer         = nextCandidate.node;
    const TensorPtr& connectingTensor = nextCandidate.connectingTensor;
    const NodePtr&   consumer         = currentChain.empty() ? finalConsumer : currentChain.back().node;
    HB_ASSERT_PTR(consumer);
    for (const TensorPtr& output : producer->getOutputs())
    {
        const auto& inputs = consumer->getInputs();
        if (output && output != connectingTensor && !output->isAuxTensor() &&
            (std::find(inputs.begin(), inputs.end(), output) != inputs.end()))
        {
            LOG_DEBUG(SRAM_SLICE,
                      "{}: Chain candidate {} shares multiple outputs with bundle consumer {}",
                      HLLOG_FUNC,
                      producer->getNodeName(),
                      consumer->getNodeName());
            return true;
        }
    }
    return false;
}

bool TPCExpansionsAndSingleMMEBundlizer::commonChainBreakerChecks(const NodePtr& candidateNode) const
{
    // Check that the node is not bundled
    if (candidateNode->getNodeAnnotation().bundleInfo.is_set())
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{} is already bundled (bundle index {}), so it breaks the producer chain.",
                  candidateNode->getNodeName(),
                  candidateNode->getNodeAnnotation().bundleInfo->bundleIndex);
        return true;
    }

    for (const TensorPtr& tensor : candidateNode->getOperands())
    {
        // TODO [SW-63304]: support high rank
        if (!tensor) continue;
        if (tensor->getDim() > SYN_MAX_TENSOR_DIM)
        {
            LOG_DEBUG(SRAM_SLICE,
                      "{} has high rank operand ({} - rank:{}). Slicing it is not supported.",
                      candidateNode->getNodeName(),
                      tensor->getName(),
                      tensor->getDim());
            return true;
        }
    }
    NodeAccessPatternPtr nodeAp = candidateNode->getNodeAccessPattern();
    if (nodeAp == nullptr)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{} has no access pattern - can't create producer/consumer chain with it.",
                  candidateNode->getNodeName());
        return true;
    }

    for (const TensorPtr& tensor : candidateNode->getOperands())
    {
        if (!tensor) continue;
        if (!nodeAp->hasAccessPattern(tensor))
        {
            LOG_DEBUG(SRAM_SLICE,
                      "Tensor:{} has no access pattern in node: {} - can't create producer/consumer chain with it.",
                      tensor->getName(),
                      candidateNode->getNodeName());
            return true;
        }
    }

    if (candidateNode->hasBindingInputReuse())
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{} has bindgining input re-use - bundling it is not supported ATM.",
                  candidateNode->getNodeName());
        return true;
    }

    // No chain break needed
    return false;
}

class TPCProducersAndSingleMMEExpansionChecker : public BaseBundleChainExpansionChecker
{
public:
    TPCProducersAndSingleMMEExpansionChecker(const NodePtr&          node,
                                             const TensorPtr&        connectingTensor,
                                             const std::vector<Dim>& dims)
    : BaseBundleChainExpansionChecker(node, connectingTensor, dims) {};
    virtual ~TPCProducersAndSingleMMEExpansionChecker() = default;
    const ExpansionBreakerContainer getExpansionBreakers() const override
    {
        ExpansionBreakerContainer tpcExpansionBreakers;

        tpcExpansionBreakers.push_back(std::make_unique<GranularityCoverBundleExpansionBreaker>(m_candidateNode,
                                                                                                m_connectingTensor,
                                                                                                m_slicingDims));
        tpcExpansionBreakers.push_back(
            std::make_unique<OverlapExpansionBreaker>(m_candidateNode, m_connectingTensor, m_slicingDims));
        tpcExpansionBreakers.push_back(
            std::make_unique<OffsetExpansionBreaker>(m_candidateNode, m_connectingTensor, m_slicingDims));
        tpcExpansionBreakers.push_back(std::make_unique<NonStichedOffsetOverlapExpansionBreaker>(
            m_candidateNode,
            m_connectingTensor,
            m_slicingDims));  // Block stitching of TPC nodes with offset/overlap on the non-stitched operands until it
                              // will be supported (TODO: SW-99608)

        return tpcExpansionBreakers;
    }
};

bool TPCExpansionsAndSingleMMEBundlizer::isTPCChainBreaker(const PipelinedNode& nextCandidate)
{
    if (nextCandidate.connectingTensor->getTotalElements() == 1)
    {
        // it is useless to have scalar producer because it cant be sliced and also not take much sram resources.
        // Continuing the chain to bigger producer tensors can't be helpful because reduction cant be a producer.
        LOG_DEBUG(SRAM_SLICE,
                  "{} - connecting tensor {} is scalar",
                  nextCandidate.node->getNodeName(),
                  nextCandidate.connectingTensor->getName());
        return true;
    }
    else if (!nextCandidate.slicingDims.empty())
    {
        TPCProducersAndSingleMMEExpansionChecker chainExpansionChecker(nextCandidate.node,
                                                                       nextCandidate.connectingTensor,
                                                                       nextCandidate.slicingDims);
        return chainExpansionChecker.isChainBreaker();
    }

    // No chain break needed
    return false;
}

bool TPCExpansionsAndSingleMMEBundlizer::isLogicalChainBreaker(const PipelinedNode& nextCandidate,
                                                               const TensorPtr&     nextTensor) const
{
    const auto& candidateNode = nextCandidate.node;

    if (!isSupportedLogicalNode(candidateNode))
    {
        LOG_DEBUG(SRAM_SLICE, "{} - can't stitch node type", candidateNode->getNodeName());
        return true;
    }

    const TensorPtr& givenTensor = nextCandidate.connectingTensor;
    // Common validation made sure there is node access pattern to the candidate node
    auto matchingSlicingDims =
        NodeSolver::getTensorMatchingSlicedDims(candidateNode, givenTensor, nextCandidate.slicingDims, nextTensor);
    // if both vectors are empty (tensor unsliced) the node is valid. Otherwise, the slicing dims should be mapped 1:1
    if (matchingSlicingDims.size() != nextCandidate.slicingDims.size())
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{} - logical op can't map tensor {} slicing dims {} to a single slicing dim each in tensor {}",
                  candidateNode->getNodeName(),
                  givenTensor->getName(),
                  toString(nextCandidate.slicingDims, ','),
                  nextTensor->getName());
        return true;
    }

    // No chain break needed
    return false;
}

bool TPCExpansionsAndSingleMMEBundlizer::isProducerChainBoundary(const NodePtr& producer) const
{
    // Currently support chains which start with a TPC node.
    return HabanaGraph::runsOnTPC(producer);
}

bool TPCExpansionsAndSingleMMEBundlizer::isConsumerChainBoundary(const NodePtr& producer) const
{
    // Currently support chains which end with a TPC node.
    return HabanaGraph::runsOnTPC(producer);
}

// Returns the input index which is best to be placed in SRAM
// bothReadOnce is set to true if both operands are small enough to be read once from HBM.
// In this case it's useful to place in SRAM only if ther's a producer
unsigned TPCExpansionsAndSingleMMEBundlizer::selectSramInputIndexByBw(const NodePtr& mmeConsumer,
                                                                      bool&          bothReadOnce) const
{
    MmeCommon::PerfAttr perfAttr = MmeBrainProxy::getRecommendedConfigMmePerf(mmeConsumer);

    // Compare the number of bytes expected to be read from HBM from each operand,
    // to select which operand is best to place in SRAM
    unsigned sizeA       = mmeConsumer->getInput(0)->getDenseSizeInBytes();
    unsigned sizeB       = mmeConsumer->getInput(1)->getDenseSizeInBytes();
    unsigned totalBytesA = perfAttr.fetchNrA * sizeA;
    unsigned totalBytesB = perfAttr.fetchNrB * sizeB;
    bothReadOnce         = (perfAttr.fetchNrA == 1 && perfAttr.fetchNrB == 1);
    unsigned ret         = 0;
    if (totalBytesA == totalBytesB)
    {
        ret = sizeA > sizeB ? 0 : 1;
    }
    else
    {
        ret = totalBytesA > totalBytesB ? 0 : 1;
    }
    LOG_DEBUG(SRAM_SLICE, "input {} of {} reads most bytes", ret, mmeConsumer->getNodeName());
    return ret;
}

std::optional<BundleAndSolver> TPCExpansionsAndSingleMMEBundlizer::getMmeWithTpcExpansionsBundle(const NodePtr& mmeNode)
{
    PipelineMultiChain producerChains;
    bool               producerFound = false;
    NodeSet            otherNodesInBundle;
    if (mmeNode->getNodeAnnotation().bundleInfo.is_set())
    {
        // Already bundled
        return {};
    }
    if (!NodeSolver::doesNodeSupportSlicing(mmeNode))
    {
        LOG_TRACE(SRAM_SLICE, "Cannot bundle node {} because it can't be sliced", mmeNode->getNodeName());
        return {};
    }
    LOG_DEBUG(SRAM_SLICE, "Create new bundle for {}", mmeNode->getNodeName());
    for (int inputIdx : {0, 1})
    {
        LOG_TRACE(SRAM_SLICE, "get Producer Chain for index {}", inputIdx);
        // [CID: 41558] False positive - coverity ignores std::set default c'tor
        PipelineChain pc = getProducerChain(mmeNode, mmeNode->getInput(inputIdx), otherNodesInBundle);
        if (!pc.empty())
        {
            std::for_each(pc.begin(), pc.end(), [&](const PipelinedNode& p) { otherNodesInBundle.insert(p.node); });
            producerChains.push_back(std::move(pc));
            producerFound = true;
        }
    }

    if (producerChainsIntersect(producerChains))
    {
        // In BERT this only happens when the chains are identical. In the future we may want a wiser decision between
        // the chains.
        producerChains.resize(1);
        otherNodesInBundle.clear();
        std::for_each(producerChains.front().begin(), producerChains.front().end(), [&](const PipelinedNode& p) {
            otherNodesInBundle.insert(p.node);
        });
    }

    // TODO [SW-73851]: move this into the bundle solver
    bool     bothReadOnce;
    unsigned bestBwIndex = selectSramInputIndexByBw(mmeNode, bothReadOnce);
    HB_ASSERT(bestBwIndex == 0 || bestBwIndex == 1, "MME Input index is expected to be 0 or 1");

    // Consumer might be added later, so we can't know at this stage if MME is bundled alone.
    // Future improvment can be to call it again and re-select the sliced input if consumers weren't added.
    TensorPtr slicedInput = getMmeInputToSlice(mmeNode, producerChains, false);
    // NodeSolver::doesNodeSupportSlicing() above should filter out nodes that both of their input operands
    // can't be sliced
    HB_ASSERT_PTR(slicedInput);

    LOG_TRACE(SRAM_SLICE, "get Consumer Chain for output index 0");
    unsigned int  slicedInputIdx = mmeNode->getInputIndexOfTensor(slicedInput);
    auto          slicingDims    = NodeSolver::getInputSlicingDims(mmeNode, slicedInputIdx);
    HB_ASSERT(!slicingDims.empty(), "supported node must have slicing dim");
    Dim           sliceDim = slicingDims.front();  // currently support consumer chain with a single slicing dim
    PipelineChain consumerChain =
        getConsumerChain(mmeNode, mmeNode->getOutput(0), slicedInput, {sliceDim}, otherNodesInBundle);

    if (shouldSliceOnSpatialDim(mmeNode, slicedInput))
    {
        LOG_TRACE(SRAM_SLICE,
                  "Create bundle with spatial slicing for node {}, discarding producers and consumers",
                  mmeNode->getNodeName());

        // bundle the node alone to allow it to slice on spatial dim
        // not bundling with consumers, as spatial dims granularity of the consumers is not considered
        return createBundle(mmeNode, {}, {}, slicedInput);
    }

    // Bundling is preferable if there are several nodes to bundle, or if the mme node reads its input more than once
    if (producerFound || !bothReadOnce || !consumerChain.empty())
    {
        return createBundle(mmeNode, producerChains, consumerChain, slicedInput);
    }
    return {};
}

TPCExpansionsAndSingleMMEBundlizer::PipelineChain
TPCExpansionsAndSingleMMEBundlizer::getConsumerChain(const NodePtr&          mmeNode,
                                                     const TensorPtr&        output,
                                                     const TensorPtr&        slicedInput,
                                                     const std::vector<Dim>& inputSlicingDims,
                                                     const NodeSet&          otherNodesInBundle) const
{
    std::vector<Dim> outputSlicingDims =
        NodeSolver::getTensorMatchingSlicedDims(mmeNode, slicedInput, inputSlicingDims, output);
    if (outputSlicingDims.empty())
    {
        LOG_DEBUG(SRAM_SLICE,
                  "MME node {} - no matching output slicing dims for input[{}]",
                  mmeNode->getNodeName(),
                  mmeNode->getInputIndexOfTensor(slicedInput));
        return {};
    }
    if (outputSlicingDims.size() > 1)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "MME node {} - consumer chain supported for a single slicing dim",
                  mmeNode->getNodeName());
        return {};
    }

    Dim mmeSlicingDim = outputSlicingDims.front();
    if (output->getSizeInElements(mmeSlicingDim) <= 1)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "MME node {} - output tensor dim size is too small to slice",
                  mmeNode->getNodeName());
        return {};
    }

    PipelineChain consumerCandidates = createConsumerCandidates(output, {mmeSlicingDim});
    PipelineChain consumers;

    for (const auto& consumerCandidate : consumerCandidates)
    {
        consumers = expandConsumerChain(mmeNode, consumerCandidate, {}, otherNodesInBundle);
        if (!consumers.empty())
        {
            // TODO SW-75981 : current implementation takes the first available consumer chain.
            // Instead we can decide based on tensor size / chain length.
            return consumers;
        }
    }

    return {};
}

TPCExpansionsAndSingleMMEBundlizer::PipelineChain
TPCExpansionsAndSingleMMEBundlizer::createConsumerCandidates(const TensorPtr&        tensor,
                                                             const std::vector<Dim>& slicingDims) const
{
    PipelineChain consumers;
    for (const auto& consumer : m_graph.getTensorConsumers(tensor))
    {
        consumers.push_back(PipelinedNode(consumer, tensor, slicingDims));
    }
    return consumers;
}

bool TPCExpansionsAndSingleMMEBundlizer::consumerIsChainBreaker(const PipelinedNode& nextCandidate,
                                                                const PipelineChain& currentChain,
                                                                const NodePtr&       firstProducer,
                                                                const NodeSet&       otherNodesInBundle) const
{
    if (commonChainBreakerChecks(nextCandidate.node)) return true;

    if (areAssociatedWithDifferentFlashAttention(nextCandidate.node, firstProducer))
    {
        LOG_DEBUG(SRAM_SLICE,
                  "Candidate {} and producer {} are associated with different FA",
                  __func__,
                  nextCandidate.node->getNodeName(),
                  firstProducer->getNodeName());
        return true;
    }
    // Check bundling doesn't create circles in the bundle-plane graph
    if (consumerCyclesChainBreaker(nextCandidate, currentChain, firstProducer, otherNodesInBundle))
    {
        LOG_DEBUG(SRAM_SLICE, "{} - creates a circle in BP graph", nextCandidate.node->getNodeName());
        return true;
    }

    // check that all of the slicing dims (if any) have supported granularity for slicing
    if (consumerGranularityChainBreaker(nextCandidate))
    {
        LOG_DEBUG(SRAM_SLICE,
                  "Can't create consumer chain with {} - invalid granularity",
                  nextCandidate.node->getNodeName());
        return true;
    }

    if (HabanaGraph::runsOnTPC(nextCandidate.node))
    {
        return isTPCChainBreaker(nextCandidate);
    }
    else if (nextCandidate.node->isLogicalOperation())
    {
        return isLogicalChainBreaker(nextCandidate, nextCandidate.node->getOutput(0));
    }
    // Currently, only TPC and logical ops are supported
    return true;
}

bool TPCExpansionsAndSingleMMEBundlizer::consumerCyclesChainBreaker(const PipelinedNode& nextCandidate,
                                                                    const PipelineChain& currentChain,
                                                                    const NodePtr&       firstProducer,
                                                                    const NodeSet&       otherNodesInBundle) const
{
    const auto&           consumer         = nextCandidate.node;
    const auto&           connectingTensor = nextCandidate.connectingTensor;
    BundlePathsValidation pathsValidation(m_graph);
    NodeSet               acceptedNodes = otherNodesInBundle;
    acceptedNodes.insert(firstProducer);
    std::for_each(currentChain.begin(), currentChain.end(), [&](const PipelinedNode& p) {
        acceptedNodes.insert(p.node);
    });
    return !pathsValidation.validateConsumerPaths(consumer, connectingTensor, acceptedNodes);
}

bool TPCExpansionsAndSingleMMEBundlizer::consumerGranularityChainBreaker(const PipelinedNode& nextCandidate) const
{
    // Common validation made sure there is node access pattern to the consumer (in commonChainBreakerChecks)
    NodeAccessPatternPtr nodeAP = nextCandidate.node->getNodeAccessPattern();
    HB_ASSERT(nodeAP, "Consumer candidate must have access pattern");
    const auto& granularity = nodeAP->getTensorGranularity(nextCandidate.connectingTensor).geometry;
    // check that all of the slicing dims (if any) have supported granularity for slicing
    // TODO: SW-75259 - Support TPC consumer with slicing dim granularity > 1
    // make sure all sliced dims granularity is 1
    auto slicingDims = nextCandidate.slicingDims;
    bool validGranularity =
        std::all_of(slicingDims.begin(), slicingDims.end(), [&](Dim d) { return granularity[d] == 1; });
    if (!validGranularity)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "consumer {} has invalid granularity ({}) for slicing dims ({})",
                  nextCandidate.node->getNodeName(),
                  toString(granularity, ','),
                  toString(slicingDims, ','));
    }
    return !validGranularity;
}

TPCExpansionsAndSingleMMEBundlizer::PipelineChain
TPCExpansionsAndSingleMMEBundlizer::expandConsumerChain(const NodePtr&       firstProducer,
                                                        const PipelinedNode& nextCandidate,
                                                        PipelineChain        currentChain,
                                                        const NodeSet&       otherNodesInBundle) const
{
    const NodePtr&   consumerNode    = nextCandidate.node;
    const TensorPtr& consumedTensor  = nextCandidate.connectingTensor;

    LOG_DEBUG(SRAM_SLICE,
              "{}: Expanding with consumer: {} (GUID: {})",
              HLLOG_FUNC,
              consumerNode ? consumerNode->getNodeName() : "N/A",
              consumerNode ? consumerNode->getGUID() : "N/A");

    if (!consumerNode || consumerIsChainBreaker(nextCandidate, currentChain, firstProducer, otherNodesInBundle) ||
        currentChain.size() > TPCExpansionsAndSingleMMEBundlizer::MAX_CHAIN_SIZE)
    {
        // Chain cannot continue
        return {};
    }

    // Consumer is valid for the chain.
    currentChain.emplace_back(nextCandidate);
    if (isConsumerChainBoundary(consumerNode))
    {
        // Valid chain is ready
        return currentChain;
    }
    else
    {
        // The chain can be extended
        const auto& nextConsumedTensor = consumerNode->getOutput(0);  // TODO: SW-75260 - should it always be output[0]?

        std::vector<Dim> outputSlicingDims = NodeSolver::getTensorMatchingSlicedDims(consumerNode,
                                                                                     consumedTensor,
                                                                                     nextCandidate.slicingDims,
                                                                                     nextConsumedTensor);
        // In consumer expansion, we want the slicing dims of the candidate to be mapped 1:1 to its consumer slicing
        // dims.
        if (outputSlicingDims.size() == nextCandidate.slicingDims.size())
        {
            for (const auto& consumerCandidate : createConsumerCandidates(nextConsumedTensor, outputSlicingDims))
            {
                PipelineChain chain =
                    expandConsumerChain(firstProducer, consumerCandidate, currentChain, otherNodesInBundle);
                if (!chain.empty())
                {
                    // TODO SW-75981 : current implementation takes the first available consumer chain.
                    // Instead we can decide based on tensor size / chain length.
                    return chain;
                }
            }
            return {};
        }
        else
        {
            LOG_DEBUG(SRAM_SLICE,
                      "Could not continue the chain past {}, but the boundary was not reached.",
                      consumerNode->getNodeName());
            return {};
        }
    }
}

// Collect mme node and what can fit in SRAM from the producer chains and consumer chain into a bundle.
std::optional<BundleAndSolver>
TPCExpansionsAndSingleMMEBundlizer::createBundle(const NodePtr&            mmeNode,
                                                 const PipelineMultiChain& producerChains,
                                                 const PipelineChain&      consumerChain,
                                                 const TensorPtr&          mmeInputToSlice)
{
    auto bundle = std::make_shared<PipelineBundle>(UNDEFINED);
    LOG_DEBUG(SRAM_SLICE, "Collecting bundle {}:", bundle->index());
    SET_TEMP_LOG_CONTEXT(fmt::format("Collecting bundle {}", bundle->index()));
    addNodeToBundle(mmeNode, bundle);

    for (const PipelineChain& producerChain : producerChains)
    {
        for (const auto& producer : producerChain)
        {
            addNodeToBundle(producer.node, bundle);
        }
    }

    for (const auto& consumer : consumerChain)
    {
        addNodeToBundle(consumer.node, bundle);
    }

    BundleSolverPtr solver(
        new TPCExpansionsAndSingleMMEBundleSolver(bundle, m_graph, producerChains, consumerChain, mmeInputToSlice));
    return BundleAndSolver({bundle, solver});
}

bool TPCExpansionsAndSingleMMEBundlizer::inputHasProducersChain(const TensorPtr&          input,
                                                                const PipelineMultiChain& producerChains)
{
    const auto& it = std::find_if(producerChains.begin(), producerChains.end(), [&](const PipelineChain& chain) {
        return (chain.front().connectingTensor == input);
    });
    return (it != producerChains.end());
}

bool TPCExpansionsAndSingleMMEBundlizer::shouldSliceOnSpatialDim(const NodePtr&   mmeNode,
                                                                 const TensorPtr& slicedInput,
                                                                 unsigned         doubelBufFactor)
{
    // Check for big input convolution nodes, which can't fit SRAM if only sliced on batch
    const auto conv = std::dynamic_pointer_cast<ConvBaseNode>(mmeNode);
    if (!conv) return false;
    // W is irrelevant for spatial slicing
    if (slicedInput == conv->getWOperand()) return false;
    // Check that single batch size doesn't fit sram
    Dim      batchDim               = conv->is3DConvolution() ? DIM_B_FOR_5D_TENSOR : DIM_B;
    uint64_t singleBatchSizeInBytes = slicedInput->getDenseSizeInBytes() / slicedInput->getSizeInElements(batchDim);
    if (singleBatchSizeInBytes * doubelBufFactor <= SlicingBrain::knobs.maxSRAMCapInBytes) return false;
    // check if slicing is possible for the first spatial dim
    bool spatialSlicingSupported = conv->isSpatialSlicingSupported(batchDim - 1);
    LOG_TRACE(SRAM_SLICE,
              "{}: Spatial slicing is required since single batch size is {} MB, {}supported by node",
              HLLOG_FUNC,
              bToMb(singleBatchSizeInBytes),
              (spatialSlicingSupported ? "" : "not "));
    return spatialSlicingSupported;
}

TensorTile::Geometry
TPCExpansionsAndSingleMMEBundlizer::getInputGranularityWithMinimalBundle(const NodePtr&            mmeNode,
                                                                         const PipelineMultiChain& producerChains,
                                                                         const TensorPtr&          inputToSlice,
                                                                         const std::vector<Dim>&   slicedDims) const
{
    // Calculate the slicing granularity for the MME inputs, assuming the non sliced producers chain is entirely
    // bundled, but only the immediate producer for the sliced input is bundled. This gives placing both operands in
    // SRAM priority over extending the sliced input producers chain beyond the immediate producer.
    PipelineMultiChain chains;
    for (const PipelineChain& producerChain : producerChains)
    {
        bool        isInputToSliceProdChain = producerChain.front().connectingTensor == inputToSlice;
        const auto& chainToConsider =
            isInputToSliceProdChain ? TPCExpansionsAndSingleMMEBundleSolver::getMmeToFirstTpcSubChain(producerChain)
                                    : producerChain;
        chains.push_back(chainToConsider);
    }
    auto [slicedNodes, slicedTensors] = getChainsNodesAndTensors(chains);
    slicedNodes.insert(mmeNode);
    slicedTensors.insert(mmeNode->getInput(0));
    slicedTensors.insert(mmeNode->getInput(1));

    auto slicingGranularity =
        CommonTileSizeCalculator::getMinCommonTilesSizes(slicedNodes, slicedTensors, m_graph).first;
    const auto& granularity = slicingGranularity.at(inputToSlice);
    auto granularityForMmeUtil = granularity;
    for (auto dim : slicedDims)
    {
        HB_ASSERT(dim < granularityForMmeUtil.size(), "dim {} invalid for rank {}", dim, granularityForMmeUtil.size());
        granularityForMmeUtil[dim] =
            MmeNodeSolver::getMinSlicedDimSizeInElements(mmeNode, inputToSlice, dim, granularity.at(dim));
    }
    return granularityForMmeUtil;
}

TSize TPCExpansionsAndSingleMMEBundlizer::predictMinSliceSizeWithMinimalBundle(const NodePtr&            mmeNode,
                                                                               const PipelineMultiChain& producerChains,
                                                                               const unsigned            inputIdx) const
{
    const auto& inputToSlice = mmeNode->getInput(inputIdx);
    const auto  slicingDims  = NodeSolver::getInputSlicingDims(mmeNode, inputIdx);
    HB_ASSERT(!slicingDims.empty(), "Expecting mme input {} to have a slicing dim", inputToSlice->getName());
    std::vector<Dim> sliceDims   = {slicingDims.front()};  // currently assume the bundle will be sliced on a single dim
    auto granularuty = getInputGranularityWithMinimalBundle(mmeNode, producerChains, inputToSlice, sliceDims);
    return getSlicedTensorMinSize(inputToSlice, granularuty, sliceDims);
}

static unsigned getMmeLargerInputIndex(const NodePtr& mmeNode)
{
    const auto& inputs = mmeNode->getInputs();
    unsigned    sizeA  = mmeNode->getInput(0)->getDenseSizeInBytes();
    unsigned    sizeB  = mmeNode->getInput(1)->getDenseSizeInBytes();
    return (sizeA < sizeB) ? 1 : 0;
}

bool TPCExpansionsAndSingleMMEBundlizer::isPlacedInSramUnsliced(const NodePtr&            mmeNode,
                                                                const PipelineMultiChain& producerChains,
                                                                const unsigned            unslicedOpIndex)
{
    // In Vision solver the unsliced operand may be copied to SRAM if it has producers chain or requires alignment
    return inputHasProducersChain(mmeNode->getInput(unslicedOpIndex), producerChains) ||
           !SlicedOperandUtils::isTensorAlignedToMmeCL(mmeNode, unslicedOpIndex);
}

bool TPCExpansionsAndSingleMMEBundlizer::isUnbalancedBundle(const NodePtr&            mmeNode,
                                                            const PipelineMultiChain& producerChains,
                                                            const unsigned            inputIdx)
{
    // There is no good balance between the gain on mme and the regression from not pipelining tpc prod chain.
    // The size condition is trying to detect this imbalance.
    if (!inputHasProducersChain(mmeNode->getInput(inputIdx), producerChains) &&
        inputHasProducersChain(mmeNode->getInput(1 - inputIdx), producerChains))
    {
        if (mmeNode->getInput(inputIdx)->getDenseSizeInBytes() <=
            mmeNode->getInput(1 - inputIdx)->getDenseSizeInBytes())
        {
            return true;
        }
    }
    return false;
}

std::optional<TensorPtr>
TPCExpansionsAndSingleMMEBundlizer::tryPlaceBothOperandsInSram(const NodePtr&            mmeNode,
                                                               const PipelineMultiChain& producerChains)
{
    // Avoid this optimization to prevent regression in pt_bert_large_fp8_1x.
    if (!SlicedOperandUtils::isTensorAlignedToMmeCL(mmeNode, 0) &&
        !SlicedOperandUtils::isTensorAlignedToMmeCL(mmeNode, 1))
    {
        return {};
    }
    // Try to place one of the operands sliced in SRAM, and the other operand unsliced in SRAM. For this
    // calculation assume only the immediate producer of the sliced operand is bundled, as the chain might not fit
    // entirely anyway. This gives priority to placing both operands in SRAM over longer producers chain for the sliced
    // input.
    const unsigned        largerInputIndex  = getMmeLargerInputIndex(mmeNode);
    const unsigned        smallerInputIndex = 1 - largerInputIndex;
    std::vector<unsigned> inputsOrder       = {largerInputIndex, smallerInputIndex};
    for (unsigned inputIdx : inputsOrder)
    {
        // This condition prevents regression in pt_alberta_large_1x.
        // Predict if the other operand is going to be copied to SRAM by the bundle solver (per specific solver type).
        if (!isPlacedInSramUnsliced(mmeNode, producerChains, 1 - inputIdx)) continue;

        // This condition prevents regression in pt_alberta_large_1x, pt_albert_large_1x, pt_mlp_mixer_large_1x.
        if (isUnbalancedBundle(mmeNode, producerChains, inputIdx)) continue;

        const uint64_t slicedMinSize = predictMinSliceSizeWithMinimalBundle(mmeNode, producerChains, inputIdx);
        const auto&    inputToSlice  = mmeNode->getInput(inputIdx);
        // This condition prevents regression in pt_albert_large_1x.
        // Check if operand cant be sliced or is too small to slice.
        if (slicedMinSize == inputToSlice->getTotalSizeInBytes()) continue;

        // TODO: the size of the other operand can be refined if it's sliced.
        const auto unslicedOperandSize = mmeNode->getInput(1 - inputIdx)->getTotalSizeInBytes();
        // Select to slice this input if its slice fits with double buffer to SRAM and the
        // other operand fits to SRAM as well.
        if (slicedMinSize * 2 + unslicedOperandSize <= SlicingBrain::knobs.maxSRAMCapInBytes)
        {
            return inputToSlice;
        }
    }
    return {};
}

std::optional<TensorPtr>
TPCExpansionsAndSingleMMEBundlizer::selectMmeInputToSliceByPerf(const NodePtr&            mmeNode,
                                                                const PipelineMultiChain& producerChains,
                                                                bool allowCopyInputWithoutProducers)
{
    // Conv perf params are still not accurate.
    if (std::dynamic_pointer_cast<ConvBaseNode>(mmeNode) != nullptr) return {};

    MmeCommon::PerfAttr perfAttr = MmeBrainProxy::getRecommendedConfigMmePerf(mmeNode);
    if (allowCopyInputWithoutProducers && producerChains.empty())
    {
        if (perfAttr.fetchNrA == 1 || perfAttr.fetchNrB == 1)
        {
            const auto& inputs        = mmeNode->getInputs();
            TensorPtr   masterOperand = (perfAttr.fetchNrA == 1) ? inputs[1] : inputs[0];
            LOG_INFO(
                SRAM_SLICE,
                "Single MME node bundle, prefer operand that is read multiple times: {}, master operand: {}, fetchNrA: "
                "{}, fetchNrB: {} ",
                mmeNode->getNodeName(),
                masterOperand->getName(),
                perfAttr.fetchNrA,
                perfAttr.fetchNrB);

            return masterOperand;
        }
        // prefer unaligned operand
        auto unalignedInput = selectMmeInputToSliceByCLAlignment(mmeNode, {perfAttr.fetchNrA, perfAttr.fetchNrB});
        if (unalignedInput.has_value()) return unalignedInput;
    }

    // If both operands can fit SRAM when one of them is sliced - prefer this solution. Operand is worth copying to SRAM
    // only if it's fetched more than once by MME. Otherwise the single fetch costs less than the copy.
    if (perfAttr.fetchNrA > 1 && perfAttr.fetchNrB > 1)
    {
        return tryPlaceBothOperandsInSram(mmeNode, producerChains);
    }
    return {};
}

std::optional<TensorPtr>
TPCExpansionsAndSingleMMEBundlizer::selectMmeInputToSliceByCLAlignment(const NodePtr&          mmeNode,
                                                                       std::array<unsigned, 2> fetchNr) const
{
    if (!GCFG_PREFER_SLICING_UNALIGNED_MME_INPUT.value()) return {};
    for (unsigned inputIdx : {0, 1})
    {
        if (!SlicedOperandUtils::isTensorAlignedToMmeCL(mmeNode, inputIdx) && fetchNr.at(inputIdx) > 1)
        {
            LOG_DEBUG(SRAM_SLICE, " {}: Single MME node bundle, unaligned {} selected", HLLOG_FUNC, inputIdx);
            return mmeNode->getInput(inputIdx);
        }
    }
    return {};
}

std::optional<TensorPtr>
TPCExpansionsAndSingleMMEBundlizer::getLargestInputThatFitsSram(const NodePtr& mmeNode, const unsigned largerInputIndex)
{
    // Get the minimal slice size of given input as if the bundle contains only single mme node
    uint64_t slicedMinSize = predictMinSliceSizeWithMinimalBundle(mmeNode, {}, largerInputIndex);

    if (slicedMinSize <= SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        return mmeNode->getInput(largerInputIndex);
    }

    const unsigned smallerInputIndex = 1 - largerInputIndex;
    slicedMinSize = predictMinSliceSizeWithMinimalBundle(mmeNode, {}, smallerInputIndex);
    if (slicedMinSize <= SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{} select smaller input to slice (index: {}) because it fits to SRAM and the larger doesn't",
                  __FUNCTION__,
                  smallerInputIndex);
        return mmeNode->getInput(smallerInputIndex);
    }
    return {};
}

TensorPtr TPCExpansionsAndSingleMMEBundlizer::selectMmeInputToSliceBySize(const NodePtr&            mmeNode,
                                                                          const PipelineMultiChain& producerChains)
{
    const unsigned   largerInputIndex = getMmeLargerInputIndex(mmeNode);
    const auto&      inputs           = mmeNode->getInputs();
    const TensorPtr& largerInput      = inputs[largerInputIndex];
    if (producerChains.empty() || producerChains.size() > 1)
    {
        const bool canSliceInput0OnSlicingDim = NodeSolver::isInputSliceable(mmeNode, 0);
        const bool canSliceInput1OnSlicingDim = NodeSolver::isInputSliceable(mmeNode, 1);

        // If both input operands are slicable on their designated slicing dimension, and won't require spatial slicing,
        // slice the larger operand otherwise, fall back to the slicable operand.

        // Return the larget input that fits to sram, if neither of them fit choose the bigger one.
        if (canSliceInput0OnSlicingDim && canSliceInput1OnSlicingDim)
        {
            return getLargestInputThatFitsSram(mmeNode, largerInputIndex).value_or(largerInput);
        }
        else if (canSliceInput0OnSlicingDim)
        {
            return inputs[0];
        }
        else if (canSliceInput1OnSlicingDim)
        {
            return inputs[1];
        }
        else
        {
            LOG_WARN(SRAM_SLICE,
                     "Both input operands [{}, {}] cannot be sliced on their slicing dimension",
                     inputs[0]->getName(),
                     inputs[1]->getName());
            return nullptr;
        }
    }
    else
    {
        // Only 1 producer chain exists. Take its produced tensor as constraint.
        return producerChains.front().front().connectingTensor;
    }
}

std::optional<TensorPtr>
TPCExpansionsAndSingleMMEBundlizer::selectMmeInputForSpatialSlicing(const NodePtr&            mmeNode,
                                                                    const PipelineMultiChain& producerChains)
{
    TensorPtr        masterOperand(nullptr);
    const unsigned   largerInputIndex = getMmeLargerInputIndex(mmeNode);
    const auto&      inputs           = mmeNode->getInputs();
    const TensorPtr& largerInput      = inputs[largerInputIndex];
    if (shouldSliceOnSpatialDim(mmeNode, largerInput))
    {
        // Even if this operand has producers chain - it won't fit SRAM without spatial slicing.
        // If the other operand has producers chain and it doesn't need spatial slicing - select it.
        // TODO SW-108436 - try to prefer spatial slicing if the other operand has producers but is too small
        const TensorPtr& smallerInput = inputs[1 - largerInputIndex];
        if (inputHasProducersChain(smallerInput, producerChains) && !shouldSliceOnSpatialDim(mmeNode, smallerInput))
        {
            masterOperand = smallerInput;
        }
        else
        {
            // let the node solver slice the larger input spatially, and discard its producers chain even if exists
            masterOperand = largerInput;
        }
        return masterOperand;
    }
    return {};
}

TensorPtr TPCExpansionsAndSingleMMEBundlizer::getMmeInputToSlice(const NodePtr&            mmeNode,
                                                                 const PipelineMultiChain& producerChains,
                                                                 bool allowCopyInputWithoutProducers)
{
    // Check for spatial slicing before examining the producers chains, as the producers might be irrelevant if spatial
    // slicing is required.
    const auto inputForSpatialSlicing = selectMmeInputForSpatialSlicing(mmeNode, producerChains);
    if (inputForSpatialSlicing) return inputForSpatialSlicing.value();
    // no spatial slicing is involved - select by producers chains and sizes
    // TODO SW-108436 - prefer a slicable operand, if one operand has producers but is too small to be sliced

    const auto inputByPerf = selectMmeInputToSliceByPerf(mmeNode, producerChains, allowCopyInputWithoutProducers);
    if (inputByPerf) return inputByPerf.value();

    return selectMmeInputToSliceBySize(mmeNode, producerChains);
}

bool MantaRayBundlizer::isPlacedInSramUnsliced(const NodePtr&            mmeNode,
                                               const PipelineMultiChain& producerChains,
                                               const unsigned            unslicedOpIndex)
{
    // In MantaRay solver the unsliced operand may be copied to SRAM only if it has producers chain.
    return inputHasProducersChain(mmeNode->getInput(unslicedOpIndex), producerChains);
}

bool MantaRayBundlizer::isSupportedLogicalNode(const NodePtr& node) const
{
    bool supported(isSupportedLogicalNodeType(node));
    if (node->getNodeType() == Node::TYPE_LOGICAL_TRANSPOSE)
    {
        supported &= GCFG_ENABLE_BUNDLE_TRANSPOSE.value();
    }
    return supported;
}

bool MantaRayBundlizer::canBundleSharedInputMmeNode(const MMENodePtr& candidate,
                                                    const MMENodeSet& allCandidates,
                                                    const MMENodeSet& committedCandidates,
                                                    const TensorPtr&  sharedInput)
{
    // Check if the candidate node has ancestors in the candidates group
    // Check against the full candidates group as this relation is asymmetric -
    // no risk of loosing both nodes while we could keep one
    for (const MMENodePtr& m : allCandidates)
    {
        if (m == candidate) continue;
        if (m_graph.isAncestor(std::static_pointer_cast<Node>(m), std::static_pointer_cast<Node>(candidate)))
        {
            LOG_TRACE(SRAM_SLICE,
                      "can't bundle {}: descendant of {} (shared input: {})",
                      candidate->getNodeName(),
                      m->getNodeName(),
                      sharedInput->getName());
            return false;
        }
    }
    // Check if the candidate shares more than 1 input with any of the committed candidates
    // Check against the committed candidates as this relation is symmetric and might cause both to be removed
    unsigned candNonSharedInputIndex = 1 - candidate->getInputIndexOfTensor(sharedInput);
    for (const MMENodePtr& m : committedCandidates)
    {
        unsigned committedNonSharedInputIndex = 1 - m->getInputIndexOfTensor(sharedInput);
        if (candidate->getInput(candNonSharedInputIndex) == m->getInput(committedNonSharedInputIndex))
        {
            LOG_DEBUG(SRAM_SLICE,
                      "can't bundle {}: shares 2 inputs with {}",
                      candidate->getNodeName(),
                      m->getNodeName());
            return false;
        }
        if (areAssociatedWithDifferentFlashAttention(candidate, m))
        {
            LOG_DEBUG(SRAM_SLICE,
                      "can't bundle {}: associated with different FA than {}",
                      __func__,
                      candidate->getNodeName(),
                      m->getNodeName());
            return false;
        }
    }

    return true;
}

// Calculates the minimal size of each chain tensor.
// If maxTile is false - returns the sum of all minimal tiles sizes.
// If maxTile is true - returns the max tile size among the chain tensors.
uint64_t MantaRayBundlizer::getSlicedChainGranuleSize(const PipelineChain&     chain,
                                                      const TileSizePerTensor& tiles,
                                                      bool                     maxTile) const
{
    uint64_t chainSizeBytes = 0;
    for (const PipelinedNode& p : chain)
    {
        if (TPCExpansionsAndSingleMMEBundleSolver::ignoreInSramCapacityCalculation(p.node, chain, m_graph))
        {
            LOG_DEBUG(SRAM_SLICE,
                      "{}: Ignoring connecting tensor {} for logical node {}",
                      HLLOG_FUNC,
                      p.connectingTensor->getName(),
                      p.node->getNodeName());
            continue;
        }
        HB_ASSERT(!p.slicingDims.empty(), "Missing slicing dim for node {}", p.node->getNodeName());
        uint64_t sliceSize = getSlicedTensorMinSize(p.connectingTensor, tiles.at(p.connectingTensor), p.slicingDims);
        chainSizeBytes     = maxTile ? std::max(sliceSize, chainSizeBytes) : chainSizeBytes + sliceSize;
    }
    return chainSizeBytes;
}

uint64_t MantaRayBundlizer::getNumGranules(const TensorGranularity& sizes,
                                           const TensorGranularity& granularity,
                                           const std::vector<Dim>&  slicedDims) const
{
    uint64_t numGranules = 1;
    // multiply the number of granules per sliced dim
    for (auto slicedDim : slicedDims)
    {
        TSize dimSize        = sizes[slicedDim];
        TSize dimGranularity = granularity[slicedDim];
        HB_ASSERT(dimSize % dimGranularity == 0, "Slice size is not a mult of its granularity");
        uint64_t numDimGranuls = dimSize / dimGranularity;
        numGranules *= numDimGranuls;
    }
    return numGranules;
}

// Calculate the SRAM usage of the given sliced chains tensors min slices, including the double buffer factor.
// Returns the sum of the tile sizes if sharedChainMaxTile is false, multiplied by double buffer.
// Returns the maximal tile size for the shared input producres chain, if sharedChainMaxTile is true, multiplied by
// multibuf level.
uint64_t MantaRayBundlizer::getSlicedTensorsMinSramUsage(const MantaRaySolver::BundleParams& params,
                                                         const PipelineMultiChain&           slicedChains,
                                                         bool                                sharedChainMaxTile) const
{
    uint64_t  sramSize = 0;
    // Add the chains nodes and tensors to the bundle sliced nodes and tensors, in case some of the chains are not yet
    // committed to the bundle, and the caller just checks their effect on the bundle tensors granularity
    // It's harmless to insert nodes / tensors which are already there
    auto [slicedNodes, slicedTensors] = getChainsNodesAndTensors(slicedChains);
    slicedNodes.insert(params.mmeNodes.begin(), params.mmeNodes.end());
    slicedTensors.insert(params.sharedOperand);

    TileSizePerTensor bundleTiles =
        CommonTileSizeCalculator::getMinCommonTilesSizes(slicedNodes, slicedTensors, m_graph).first;
    HB_ASSERT(bundleTiles.find(params.sharedOperand) != bundleTiles.end(), "shared operand must have granularity");
    if (LOG_LEVEL_AT_LEAST_DEBUG(SRAM_SLICE))
    {
        LOG_DEBUG(SRAM_SLICE, "{}: Bundle tensors tile sizes:", HLLOG_FUNC);
        MantaRaySolver::printTensorsTileGranularity(bundleTiles);
    }

    // Update shared operand granularity based on mme utilization requirements
    const TensorGranularity& sharedOperandGranularity = bundleTiles[params.sharedOperand];
    const TensorGranularity  mmeUtilGranularity       = getMmeSharedInputGranularity(params.mmeNodes,
                                                                              params.sharedOperand,
                                                                              params.sharedOperandSlicingDims,
                                                                              sharedOperandGranularity);

    const unsigned numGranules =
        getNumGranules(mmeUtilGranularity, sharedOperandGranularity, params.sharedOperandSlicingDims);

    bool sharedOperandChainExists = false;
    for (const auto& chain : slicedChains)
    {
        const bool isSharedOperandChain = params.sharedOperand == chain.front().connectingTensor;
        if (isSharedOperandChain)
        {
            sharedOperandChainExists = true;
        }

        // for non shared operand producers chain we place in SRAM only the subchain from the last
        // (closest to MME) TPC producer to the MME, so we need to count only that sub chain sram "cost"
        const auto& effectiveChain = isSharedOperandChain ? chain : MantaRaySolver::getMmeToFirstTpcSubChain(chain);
        // computing max tile or sum of tiles is relevant only for shared multi-buffer, which is enabled only for the
        // shared operand producers chain
        bool       maxTile                = isSharedOperandChain ? sharedChainMaxTile : false;
        const auto slicedChainGranularity = getSlicedChainGranuleSize(effectiveChain, bundleTiles, maxTile);
        unsigned   chainBufLevel          = maxTile ? c_sharedMultiBufLevel : params.doubleBufferFactor;
        uint64_t   chainSize              = slicedChainGranularity * numGranules * chainBufLevel;
        LOG_DEBUG(SRAM_SLICE,
                  "{}: Size: {} MB = Sliced chain granularity {} MB x Num granules {} x Buffer level {}",
                  HLLOG_FUNC,
                  bToMb(chainSize),
                  bToMb(slicedChainGranularity),
                  numGranules,
                  chainBufLevel);
        sramSize += chainSize;
    }
    // if the shared operand has no producers chain - need to add its min size to the sliced tensors capacity
    if (!sharedOperandChainExists)
    {
        uint64_t sharedOperandMinSize =
            getSlicedTensorMinSize(params.sharedOperand, mmeUtilGranularity, params.sharedOperandSlicingDims);

        sramSize += (sharedOperandMinSize * params.doubleBufferFactor);
        LOG_DEBUG(SRAM_SLICE,
                  "{}: adding shared operand without chain Size: {} MB, Buffer level {}",
                  HLLOG_FUNC,
                  bToMb(sharedOperandMinSize),
                  params.doubleBufferFactor);
    }
    // reserve sram for shared operand alignment, if it is possible
    auto alignmentSize = getSharedOperandSliceAlignmentSize(params, mmeUtilGranularity);
    if (alignmentSize.has_value())
    {
        sramSize += (alignmentSize.value() * params.doubleBufferFactor);
        LOG_DEBUG(SRAM_SLICE,
                  "{}: adding shared operand alignment size: {} MB * {} double buffer",
                  HLLOG_FUNC,
                  bToMb(alignmentSize.value()),
                  params.doubleBufferFactor);
    }
    return sramSize;
}

std::optional<TStride> MantaRayBundlizer::getSharedOperandSliceAlignmentSize(const MantaRaySolver::BundleParams& params,
                                                                             const TensorGranularity& granularity) const
{
    if (!GCFG_PREFER_SLICING_UNALIGNED_MME_INPUT.value()) return std::nullopt;

    // multiple sliced dims not supported
    if (params.sharedOperandSlicingDims.size() != 1) return std::nullopt;

    std::map<unsigned, TSize> sizePerDim;
    for (auto slicedDim : params.sharedOperandSlicingDims)
    {
        // Can't calculate in advance the final alignment size if the tensor is sliced on FCD - it is influenced by the
        // slice size.
        if (slicedDim == 0) return std::nullopt;

        sizePerDim.emplace(slicedDim, granularity[slicedDim]);
    }
    const TensorPtr& t           = params.sharedOperand;
    unsigned         bundleIndex = (*params.mmeNodes.begin())->getNodeAnnotation().bundleInfo->bundleIndex;

    // Check if the tensor can be aligned, based on the nodes that were added so far to the bundle.
    if (!TPCExpansionsAndSingleMMEBundleSolver::alignmentAllowedForTensorInBundle(t, m_graph, bundleIndex))
    {
        return std::nullopt;
    }

    return SlicedOperandUtils::getSliceAlignmentSize(t, sizePerDim);
}

bool MantaRayBundlizer::isSharedMultiBufUseful(const PipelineChain& chain)
{
    unsigned numPhysicalNodes = 0;
    for (const PipelinedNode& producer : chain)
    {
        if (!TPCExpansionsAndSingleMMEBundleSolver::ignoreInSramCapacityCalculation(producer.node, chain, m_graph))
        {
            numPhysicalNodes++;
        }
    }
    // Shared multi buf is useless if there are 2 or less nodes, as it holds both nodes outputs concurrently
    // while allocating the max output size between them.
    // TODO [SW-105425]: Limit the solution for up to 4 nodes, otherwise there might be a bug in bundle memcopy
    // scheduler, since it doesn't capture correctly the dependencies between the chain nodes using the same buffers.
    // This limitation can be removed with the memcopy scheduler fix.
    return (numPhysicalNodes > 2 && numPhysicalNodes <= c_sharedMultiBufLevel);
}

// The function trims the given candidate chain such that it fits SRAM capacity, and updates params with:
// sharedOperandProducers - the trimmed chain which fits into the given SRAM capacity
// slicedOperandsSramMinSize - The chain tensors min tiles SRAM capacity
// sharedOperandChainInSharedMultiBuf - Flag if the chain uses shared multi buffer
void MantaRayBundlizer::trimSharedProducersChainToFitSram(MantaRaySolver::BundleParams& params,
                                                          const PipelineChain&          candidateFullChain,
                                                          uint64_t                      sramCap,
                                                          bool                          allowMultiBuf)
{
    LOG_DEBUG(SRAM_SLICE,
              "{}: Chain length {}, sram capacity: {} MB",
              HLLOG_FUNC,
              candidateFullChain.size(),
              bToMb(sramCap));
    HB_ASSERT(!candidateFullChain.empty() && !candidateFullChain.back().node->isLogicalOperation(),
              "Expecting clipped non empty chain");

    PipelineMultiChain slicedChains = getBundledSlicedChains(params);
    // Make sure the current producers chain is the last chain, to make it easier to replace it with the extending
    // chain.
    HB_ASSERT(slicedChains.back().front().connectingTensor == params.sharedOperandProducers.front().connectingTensor,
              "Shared producers chain is expected to be last");

    std::list<PipelinedNode> finalChain {};
    std::list<PipelinedNode> currSubChain {};
    uint64_t                 finalSlicedOperandsSramUse = 0;
    bool                     finalSharedMultiBuf        = false;

    for (const PipelinedNode& producer : candidateFullChain)
    {
        HB_ASSERT_PTR(producer.node);
        currSubChain.push_back(producer);

        if (producer.node->isLogicalOperation())
        {
            // skip to next iteration until reaching a non-logic node
            LOG_TRACE(SRAM_SLICE,
                      "{}: Skipping logical node {}:{}, current sub chain length: {}",
                      HLLOG_FUNC,
                      producer.node->getNodeTypeStr(),
                      producer.node->getNodeName(),
                      currSubChain.size());
            continue;
        }

        PipelineChain currentChain(finalChain.begin(), finalChain.end());
        currentChain.insert(currentChain.end(), currSubChain.begin(), currSubChain.end());
        // a new node added to the chain potentially affects slicing granularity hence recalculate min chain slice given
        // possibly new size constraints
        bool sharedMultiBuf = allowMultiBuf && isSharedMultiBufUseful(currentChain);
        // replace the last chain (for shared operand) with the current chain
        slicedChains.pop_back();
        slicedChains.push_back(currentChain);
        auto slicedOperandsSramUse = getSlicedTensorsMinSramUsage(params, slicedChains, sharedMultiBuf);
        if (tensorsFitSramCapacity(slicedOperandsSramUse, sramCap))
        {
            // commit on current sub chain
            finalChain.splice(finalChain.end(), currSubChain);
            LOG_TRACE(SRAM_SLICE,
                      "{}: Current chain length: {}, sliced tensors SRAM: {} MB, {} shared multi buf",
                      HLLOG_FUNC,
                      finalChain.size(),
                      bToMb(slicedOperandsSramUse),
                      sharedMultiBuf ? "using" : "not using");
            finalSlicedOperandsSramUse = slicedOperandsSramUse;
            finalSharedMultiBuf        = sharedMultiBuf;
        }
        else
        {
            LOG_TRACE(SRAM_SLICE,
                      "{}: Stopping chain expansion without including {}, min required SRAM: {} MB",
                      HLLOG_FUNC,
                      producer.node->getNodeName(),
                      bToMb(slicedOperandsSramUse));
            break;
        }
    }
    params.sharedOperandProducers             = PipelineChain(finalChain.begin(), finalChain.end());
    params.slicedOperandsSramMinSize          = finalSlicedOperandsSramUse;
    params.sharedOperandChainInSharedMultiBuf = finalSharedMultiBuf;
    LOG_DEBUG(
        SRAM_SLICE,
        "{}: Trimmed shared operand producer chain length: {}, sliced tensors SRAM size: {} MB, shared multi buf {}",
        HLLOG_FUNC,
        params.sharedOperandProducers.size(),
        bToMb(finalSlicedOperandsSramUse),
        finalSharedMultiBuf ? "enabled" : "disabled");
}

// Sort the MME nodes to partials and non partials consumers, and add the valid ones (which won't create cycles)
// to the bundle and the bundle params.
void MantaRayBundlizer::addSortedMmeNodesToBundle(const MMENodeSet&             mmeNodes,
                                                  PipelineBundlePtr             bundle,
                                                  MantaRaySolver::BundleParams& params)
{
    // Split mmeNodes to partial/nonpartial consumers
    for (auto iter = mmeNodes.begin(); iter != mmeNodes.end(); iter++)
    {
        if (isSlicedOnCommonDim(*iter, params.sharedOperand, params.sharedOperandSlicingDims))
        {
            params.partialsConsumers.insert(*iter);
        }
        else
        {
            params.nonPartialsConsumers.insert(*iter);
        }
    }

    // Give priority to the nonPartialConsumers to be added to the bundle, then the partialConsumers
    addValidMmeNodesToBundle(params.nonPartialsConsumers, bundle, params.sharedOperand, false);
    addValidMmeNodesToBundle(params.partialsConsumers, bundle, params.sharedOperand, true);
    params.mmeNodes.insert(params.nonPartialsConsumers.begin(), params.nonPartialsConsumers.end());
    params.mmeNodes.insert(params.partialsConsumers.begin(), params.partialsConsumers.end());
}

// Iterate the set of mmeConsumers, and validate that adding consumer to the bundle won't create a cycle in the graph,
// and if it does, skip adding it to the bundle, and erase it from the consumersSet
void MantaRayBundlizer::addValidMmeNodesToBundle(MMENodeSet&        mmeConsumers,
                                                 PipelineBundlePtr& bundle,
                                                 const TensorPtr&   masterOperand,
                                                 bool               partialsConsumers)
{
    for (auto iter = mmeConsumers.begin(); iter != mmeConsumers.end();)
    {
        NodePtr n = std::static_pointer_cast<Node>(*iter);
        BundlePathsValidation pathsValidation(m_graph);
        bool                  validPaths =
            pathsValidation.validateConsumerPaths(n,
                                                  masterOperand,
                                                  NodeSet(bundle->getNodes().begin(), bundle->getNodes().end()));
        uint64_t outputSize = getOutputSizeForPartials(n);
        bool blockPartials = partialsConsumers && (outputSize > SlicingBrain::knobs.maxSRAMCapInBytes);
        if (!validPaths || blockPartials)
        {
            LOG_DEBUG(SRAM_SLICE,
                      "Skip adding node {} (valid paths {}, block partials {})",
                      n->getNodeName(),
                      validPaths,
                      blockPartials);
            iter = mmeConsumers.erase(iter);
        }
        else
        {
            addNodeToBundle(n, bundle);
            iter++;
        }
    }
}

bool MantaRayBundlizer::isSlicedOnCommonDim(const MMENodePtr&       n,
                                            const TensorPtr&        t,
                                            const std::vector<Dim>& slicingDims) const
{
    for (auto slicingDim : slicingDims)
    {
        if (Node::isBatchGemmNode(n) && (slicingDim >= DIM_GEMM_BATCH))
        {
            // When slicing on outer BGEMM dims, the transpose doesn't matter because it applies to inner dims
            continue;
        }

        bool     isTransposed = n->isOperandTransposed(t);
        unsigned inputIdx     = n->getInputIndexOfTensor(t);
        HB_ASSERT((inputIdx == 0 || inputIdx == 1), "Expected to get A or B operand for MME node");
        bool isDimC = inputIdx == 0 ? (slicingDim == DIM_C) : (slicingDim != WEIGHT_DIM_K);
        // transposed and C = not partials, not transposed and not C, same.
        if (isDimC != isTransposed)
        {
            return true;
        }
    }
    return false;
}

bool MantaRayBundlizer::tensorsFitSramCapacity(uint64_t tensorsSizeInBytes, uint64_t availableSramBytes) const
{
    bool fitSram = tensorsSizeInBytes <= availableSramBytes;
    LOG_DEBUG(SRAM_SLICE,
              "Tensors of size {} MB - {} fit available SRAM {} MB",
              bToMb(tensorsSizeInBytes),
              (fitSram ? "can" : "can't"),
              bToMb(availableSramBytes));
    return fitSram;
}

TPCExpansionsAndSingleMMEBundlizer::PipelineChain
MantaRayBundlizer::getNonSharedProducersChain(const MMENodePtr&        n,
                                              const TensorPtr&         input,
                                              const PipelineBundlePtr& bundle,
                                              const std::vector<Dim>&  slicingDims)
{
    const NodePtr& prod = m_graph.getTensorProducer(input);
    NodeSet        bundleNodes(bundle->getNodes().begin(), bundle->getNodes().end());

    PipelinedNode p(prod, input, slicingDims);
    PipelineChain pc = expandProducerChain(n, p, {}, bundleNodes);
    clipProducerChain(pc, nTPCForUnsharedOperand);

    return pc;
}

// Check if t needs to be copied to SRAM without producers chain, to improve n execution time
bool MantaRayBundlizer::shouldCopyToSram(const MMENodePtr& n, const TensorPtr& t)
{
    // If the node can be flattened and the producer probably adds strides
    // Add the memcopy to make the input dense and allow the flattening to happen.
    // Can't know for sure if the input is strided before logical ops pass, so temp support only slice producer
    const NodePtr& prod = m_graph.getTensorProducer(t);
    if (n->canBeConvertedToGEMM() && prod && prod->getNodeType() == Node::TYPE_SLICE)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{}: Copy operand {} to SRAM even if {} not bundled",
                  HLLOG_FUNC,
                  t->getName(),
                  prod->getNodeName());
        return true;
    }
    // Masked bgemm is improved when its masks are in sram.
    // prioritize placing them in sram even without producers.
    if (n->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        const auto inputIndex = n->getInputIndexOfTensor(t);
        if (inputIndex == TENSOR_AUX_BGEMM_MASK_A || inputIndex == TENSOR_AUX_BGEMM_MASK_B)
        {
            return true;
        }
    }
    // TODO SW-76353 - allow placing input in SRAM without a producer if #fetches is large
    return false;
}

void MantaRayBundlizer::addNonSharedProducers(MantaRaySolver::BundleParams& params, PipelineBundlePtr& bundle)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const uint64_t SRAMCap = SlicingBrain::knobs.maxSRAMCapInBytes;

    // Try to add TPC producers for the non shared operands
    // TODO SW-76318 - by priority order instead of unordered
    for (const MMENodePtr& n : params.mmeNodes)
    {
        LOG_DEBUG(SRAM_SLICE, "Looking for non shared producers to node {}", n->getNodeName());
        for (const TensorPtr& t : n->getInputs())
        {
            if (t == params.sharedOperand || !t) continue;
            uint64_t allocatedSram = params.slicedOperandsSramMinSize + params.unslicedOperandsSramSize;
            HB_ASSERT(allocatedSram <= SRAMCap, "committed SRAM must be within capacity");
            uint64_t   availableSram = SRAMCap - allocatedSram;
            const std::vector<Dim> nonSharedOpSlicingDim =
                NodeSolver::getTensorMatchingSlicedDims(n, params.sharedOperand, params.sharedOperandSlicingDims, t);
            PipelineChain pc = getNonSharedProducersChain(n, t, bundle, nonSharedOpSlicingDim);
            if (pc.empty())
            {
                LOG_TRACE(SRAM_SLICE, "Non shared producer chain in sram is empty");
                // TODO [SW-103073]: refine tensor size below to be slice size if sliced
                uint64_t tensorSize = t->getTotalSizeInBytes();
                if (tensorsFitSramCapacity(tensorSize, availableSram) && shouldCopyToSram(n, t))
                {
                    LOG_TRACE(SRAM_SLICE, "Copy to SRAM non shared producer {}", t->getName());
                    params.copyTensorsToSram.insert(t);
                    params.unslicedOperandsSramSize += tensorSize;
                }
                // Keep searching for producers chain for this node
                continue;
            }
            // else - valid producers chain exists - check if can be bundled

            if (nonSharedOpSlicingDim.empty())
            {
                LOG_TRACE(SRAM_SLICE, "Unsliced non shared operand chain for {}", t->getName());
                if (blockNonSharedProducersChain(n, t, params.mmeNodes.size())) continue;

                // Currently, we store only the last operand of non shared producer chain in SRAM.
                // Therefore the chain size in SRAM is computed accordingly.
                const auto&    mmeToFirstTpcSubChain = MantaRaySolver::getMmeToFirstTpcSubChain(pc);
                const uint64_t unslicedChainSize = MantaRaySolver::getUnSlicedChainSize(mmeToFirstTpcSubChain, m_graph);
                if (!tensorsFitSramCapacity(unslicedChainSize, availableSram)) continue;

                LOG_DEBUG(SRAM_SLICE, "Adding unsliced non shared producers chain {} MB", bToMb(unslicedChainSize));
                params.unslicedOperandsSramSize += unslicedChainSize;
                if (!GCFG_ENABLE_LONG_UNSLICED_NON_SHARED_PROD_CHAIN.value())
                {
                    // Non shared producers, when unsliced, must complete before the MME can run. Thus, they are not
                    // pipelined with the MME inside the bundle. If they are not placed in SRAM - there is no benefit in
                    // bundling them.
                    // Also, not bundling them allows them to pipeline with other bundles / free nodes.
                    pc = mmeToFirstTpcSubChain;
                }
            }
            else
            {
                LOG_TRACE(SRAM_SLICE, "Sliced non shared operand chain for {}", t->getName());
                // recalculate sliced operands SRAM capacity with this chain, as the granularity of all sliced tensors
                // may have changed
                PipelineMultiChain slicedChains = getBundledSlicedChains(params);
                // add the candidate chain
                slicedChains.push_back(pc);
                uint64_t slicedOperandsSramUse = getSlicedTensorsMinSramUsage(params, slicedChains, false /* maxTile*/);
                availableSram                  = SRAMCap - params.unslicedOperandsSramSize;
                if (!tensorsFitSramCapacity(slicedOperandsSramUse, availableSram)) continue;

                // Sliced non shared operand min chain fits sram, insert the whole chain to bundle
                LOG_TRACE(SRAM_SLICE,
                          "Adding sliced non shared producers chain, sliced cap {} MB",
                          bToMb(slicedOperandsSramUse));
                params.slicedOperandsSramMinSize = slicedOperandsSramUse;
            }
            // Add the chain nodes to the bundle
            for (const PipelinedNode& pr : pc)
            {
                addNodeToBundle(pr.node, bundle);
            }
            params.nonSharedProducers.push_back(pc);
            // Stop searching for producers chain for this node
            break;
        }
    }
}

void MantaRayBundlizer::extendSharedInputProducers(MantaRaySolver::BundleParams& params,
                                                   PipelineBundlePtr&            bundle,
                                                   const PipelineChain&          sharedProducerChain)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const uint64_t SRAMCap = SlicingBrain::knobs.maxSRAMCapInBytes;

    if (!sharedProducerChain.empty())
    {
        // Trim the chain to fix SRAM capacity. On first try - block shared multi buffer usage,
        // since for tensors with big size differences using shared multi buffer may require
        // more SRAM than the simple placement of all the chain.
        trimSharedProducersChainToFitSram(params,
                                          sharedProducerChain,
                                          SRAMCap - params.unslicedOperandsSramSize,
                                          false /*allowMultiBuf*/);

        if ((sharedProducerChain.size() != params.sharedOperandProducers.size()) &&
            GCFG_ENABLE_SHARED_MULTIBUF_PER_SLICED_CHAIN.value())
        {
            // producers chain was trimmed - try with shared multi buffer
            trimSharedProducersChainToFitSram(params,
                                              sharedProducerChain,
                                              SRAMCap - params.unslicedOperandsSramSize,
                                              true /*allowMultiBuf*/);
        }
    }

    for (const PipelinedNode& p : params.sharedOperandProducers)
    {
        // The first producers sub chain is already in the bundle
        if (std::find(bundle->getNodes().begin(), bundle->getNodes().end(), p.node) == bundle->getNodes().end())
        {
            LOG_TRACE(SRAM_SLICE,
                      "Add node ({}) to shared producers chain: {}",
                      p.node->getNodeTypeStr(),
                      p.node->getNodeName());
            addNodeToBundle(p.node, bundle);
        }
    }
}

bool MantaRayBundlizer::sliceNonSharedProducersChainInSramAllowed(const MantaRaySolver::BundleParams& params) const
{
    // The optimization is allowed for single GEMM bundle with producer chains for both inputs and no consumers.
    return GCFG_ENABLE_SLICING_BOTH_PRODUCER_CHAINS.value() && (params.mmeNodes.size() == 1) &&
           !params.sharedOperandProducers.empty() && !params.nonSharedProducers.empty() &&
           params.mmeOutputConsumers.empty() && Node::isGemmNode(*params.mmeNodes.begin());
}

void MantaRayBundlizer::updateNonSharedProducers(MantaRaySolver::BundleParams& params, PipelineBundlePtr& bundle)
{
    HB_ASSERT(params.mmeNodes.size() == 1, "Expected a single MME node in bundle {}", bundle->index());
    HB_ASSERT(params.nonSharedProducers.size() == 1,
              "Expected a single non shared producers chain in bundle {}",
              bundle->index());
    const auto& mmeNode        = *params.mmeNodes.begin();
    const auto  nonSharedInputIdx        = 1 - mmeNode->getInputIndexOfTensor(params.sharedOperand);
    const auto& nonSharedInput           = mmeNode->getInput(nonSharedInputIdx);
    const auto& nonSharedInputSlicingDims = NodeSolver::getInputSlicingDims(mmeNode, nonSharedInputIdx);
    HB_ASSERT(!nonSharedInputSlicingDims.empty(),
              "MME node {}: no valid slicing dim for operand {}",
              mmeNode->getNodeName(),
              nonSharedInput->getName());

    LOG_DEBUG(SRAM_SLICE,
              "{}: Re-create non shared producers chain for MME node {} (operand {}) with slicing dims {}",
              HLLOG_FUNC,
              mmeNode->getNodeName(),
              nonSharedInput->getName(),
              toString(nonSharedInputSlicingDims, ','));
    // Un-bundle the existing non-shared producers chain - in reverse order
    const auto& producerChain = params.nonSharedProducers.front();
    for (auto producerIt = producerChain.rbegin(); producerIt != producerChain.rend(); ++producerIt)
    {
        removeNodeFromBundle(producerIt->node, bundle);
    }
    // Create a new chain with the first slicing dim and take only the immediate sub chain.
    // slice the non shared operand only on a single dim.
    PipelineChain newNonSharedProducerChain =
        getNonSharedProducersChain(mmeNode, nonSharedInput, bundle, {nonSharedInputSlicingDims.front()});
    newNonSharedProducerChain = MantaRaySolver::getMmeToFirstTpcSubChain(newNonSharedProducerChain);

    if (!newNonSharedProducerChain.empty() &&  // Found a new valid chain that enables slicing on nonSharedOpSlicingDim
        (MantaRaySolver::getUnSlicedChainSize(newNonSharedProducerChain, m_graph) ==
         params.unslicedOperandsSramSize))  // No change in reserved SRAM
    {
        // Replace the existing non-shared producers chain and enable the optimization flag.
        params.nonSharedProducers.clear();
        params.nonSharedProducers.push_back(newNonSharedProducerChain);
        params.sliceNonSharedProducersChain = true;
    }

    // In case a new producers chain found - bundle the new nodes.
    // Otherwise, the old nodes will be re-added to the bundle.
    for (const auto& producer : params.nonSharedProducers.front())
    {
        addNodeToBundle(producer.node, bundle);
    }
}

bool MantaRayBundlizer::sliceOnMoreThanOneDimIsAllowed(const std::vector<Dim>& slicingDims) const
{
    return slicingDims.size() > 1 && GCFG_ENABLE_SLICING_ON_MORE_THAN_ONE_DIM.value();
}

bool MantaRayBundlizer::extendSharedInputProducersIsAllowed(const MantaRaySolver::BundleParams& params) const
{
    // TODO: need a better cost model.
    // The folloiwng issue was observed in gpt2 when slicing batch gemm and its two tpc producers on two batch dims,
    // When slicing two or more consecutive tpc nodes the cost of the syncs can become significant, in this pattern it
    // made the slices overall compute time to be bigger than unsliced.
    // It was a problem since the bundle compute time was tpc bound.
    return params.mmeNodes.size() > 1 || params.sharedOperandSlicingDims.size() == 1;
}

bool MantaRayBundlizer::preferSlicingOnMultipleDims(bool fitsSram, const MantaRaySolver::BundleParams& params) const
{
    if (!fitsSram) return true;
    if (params.doubleBufferFactor == 1) return true;
    return false;
}

std::optional<BundleAndSolver> MantaRayBundlizer::createMantaRayBundle(const MMENodeSet&       mmeNodes,
                                                                       const std::vector<Dim>& slicingDims,
                                                                       const TensorPtr&        masterOperand)
{
    LOG_DEBUG(SRAM_SLICE, "{}: {} MMEs sharing operand {}", HLLOG_FUNC, mmeNodes.size(), masterOperand->getName());
    MantaRaySolver::BundleParams params;
    params.sharedOperand = masterOperand;
    // Initially set the master operand to be sliced on a single dim
    params.sharedOperandSlicingDims = {slicingDims.front()};

    PipelineBundlePtr bundle = std::make_shared<PipelineBundle>(UNDEFINED);
    SET_TEMP_LOG_CONTEXT(fmt::format("Bundle#{}", bundle->index()));

    // Sort the MME nodes to partials and non partials consumers, and add the valid ones (which won't create cycles)
    // to the bundle and the bundle params.
    addSortedMmeNodesToBundle(mmeNodes, bundle, params);
    // In case the bundle had nodes reduced from it so that it was left as a single node - return an empty bundle so
    // that the node will be handled in the single-mme-node-bundle flow
    if (bundle->getNodes().size() == 1 && mmeNodes.size() > 1)
    {
        removeBundle(bundle);
        return {};
    }

    bool fitsSram = tryPlaceSharedOperandInSram(params);
    if (preferSlicingOnMultipleDims(fitsSram, params) && sliceOnMoreThanOneDimIsAllowed(slicingDims))
    {
        // try slicing on more than one dim
        const unsigned numDimsToSlice =
            std::min(GCFG_PIPELINE_MANAGEMENT_NUM_DIMS_TO_SLICE.value(), slicingDims.size());
        params.sharedOperandSlicingDims =
            std::vector<Dim>(slicingDims.begin(), std::next(slicingDims.begin(), numDimsToSlice));
        LOG_DEBUG(SRAM_SLICE, "Attempt slicing on dims: [{}]", toString(params.sharedOperandSlicingDims, ','));
        fitsSram = tryPlaceSharedOperandInSram(params);
    }
    if (!fitsSram)
    {
        LOG_DEBUG(SRAM_SLICE, "Remove bundle for shared input operand doesn't fit SRAM");
        removeBundle(bundle);
        return {};
    }
    // Try to add the shared operand producer to the bundle
    PipelineChain sharedProducerChain = getSharedInputCommonProducersChain(params, bundle);
    bool          sharedProducerAdded = addSharedInputFirstProducer(params, bundle, sharedProducerChain);
    // TODO [SW-156602]: if shared producer wasn't added (sharedProducerAdded = false) and we haven't tried slicing
    // on more than one dim yet, try it now.

    // Try to add TPC producers for the non shared operands
    addNonSharedProducers(params, bundle);
    if (sharedProducerAdded && extendSharedInputProducersIsAllowed(params))
    {
        // Try to extend the shared operand producers chain
        extendSharedInputProducers(params, bundle, sharedProducerChain);
    }
    // try to add consumer to mme output

    addConsumers(params, bundle);

    if (sliceNonSharedProducersChainInSramAllowed(params))
    {
        updateNonSharedProducers(params, bundle);
    }

    // Calculate final min input size, slicing dim alignment for shared mme and tensor granularities.
    updateSlicingGranularityParams(params);

    if (!tryPlacePartialsOutputsInSram(params))
    {
        LOG_WARN(SRAM_SLICE, "Remove bundle for partials MME output doesn't fit SRAM");
        removeBundle(bundle);
        return {};
    }
    // [CID: 45601] False positive - coverity ignores std::set and std::map default c'tor
    BundleSolverPtr solver(new MantaRaySolver(bundle, m_graph, params));
    return BundleAndSolver({bundle, solver});
}

bool MantaRayBundlizer::tryPlaceSharedOperandInSram(MantaRaySolver::BundleParams& params)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const uint64_t SRAMCap = SlicingBrain::knobs.maxSRAMCapInBytes;

    // Calc shared operand min slice size, without any producer constraint, to fit to SRAM
    params.doubleBufferFactor   = 1;  // check if with single buffer the tensor can fit sram
    params.sharedOperandMinSize = getSlicedTensorsMinSramUsage(params, {}, false /* maxTile*/);
    if (!tensorsFitSramCapacity(params.sharedOperandMinSize, SRAMCap))
    {
        LOG_DEBUG(SRAM_SLICE, "{}: Can't place shared operand {} in SRAM", HLLOG_FUNC, params.sharedOperand->getName());
        return false;
    }
    // If operand isn't sliced, double buffer factor stays 1
    if (params.sharedOperandMinSize != params.sharedOperand->getTotalSizeInBytes())
    {
        // Prefer double buffer over bundle expansion. Set factor without producers, and enforce from here
        params.doubleBufferFactor = 2;
        if (params.doubleBufferFactor * params.sharedOperandMinSize > SRAMCap)
        {
            LOG_WARN(SRAM_SLICE, "{}: Bundle cannot double-buffer shared operand in SRAM", HLLOG_FUNC);
            params.doubleBufferFactor = 1;
        }
    }
    // initialize the sliced tensors SRAM usage to include the shared operand with double buffer factor
    params.slicedOperandsSramMinSize = params.sharedOperandMinSize * params.doubleBufferFactor;
    return true;
}

// Returns true if a producer was added
bool MantaRayBundlizer::addSharedInputFirstProducer(MantaRaySolver::BundleParams& params,
                                                    PipelineBundlePtr&            bundle,
                                                    const PipelineChain&          sharedProducerChain)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const uint64_t SRAMCap = SlicingBrain::knobs.maxSRAMCapInBytes;
    // Try to add the last valid sub-chain (single TPC and logicals until the shared input) to the bundle
    const auto firstProducerSubChain = MantaRaySolver::getMmeToFirstTpcSubChain(sharedProducerChain);
    if (firstProducerSubChain.empty())
    {
        LOG_DEBUG(SRAM_SLICE, "{}: Shared operand producres chain is empty", HLLOG_FUNC);
        return false;
    }

    uint64_t slicedOperandsSramUse = getSlicedTensorsMinSramUsage(params, {firstProducerSubChain}, false /* maxTile*/);

    if (!tensorsFitSramCapacity(slicedOperandsSramUse, SRAMCap))
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{}: Shared operand first producre doesn't fit SRAM: min size {} (including double buffer {})",
                  HLLOG_FUNC,
                  slicedOperandsSramUse,
                  params.doubleBufferFactor);
        return false;
    }
    // Add the sub chain to the bundle
    params.sharedOperandProducers             = firstProducerSubChain;
    params.slicedOperandsSramMinSize          = slicedOperandsSramUse;
    params.sharedOperandChainInSharedMultiBuf = false;

    for (const PipelinedNode& p : params.sharedOperandProducers)
    {
        LOG_TRACE(SRAM_SLICE,
                  "Add node ({}) to shared producers chain: {}",
                  p.node->getNodeTypeStr(),
                  p.node->getNodeName());
        addNodeToBundle(p.node, bundle);
    }
    return true;
}

MantaRaySolver::PipelineChain
MantaRayBundlizer::getSharedInputCommonProducersChain(const MantaRaySolver::BundleParams& params,
                                                      const PipelineBundlePtr&            bundle)
{
    const NodePtr& firstConsumer = std::static_pointer_cast<Node>(*params.mmeNodes.begin());
    NodeSet        bundleNodes(bundle->getNodes().begin(), bundle->getNodes().end());
    LOG_TRACE(SRAM_SLICE, "{}: get chain for {}", HLLOG_FUNC, firstConsumer->getNodeName());
    PipelinedNode candidate           = createProducerCandidate(params.sharedOperand, params.sharedOperandSlicingDims);
    PipelineChain masterProducerChain = expandProducerChain(firstConsumer, candidate, {}, bundleNodes);
    for (auto iter = std::next(params.mmeNodes.begin()); iter != params.mmeNodes.end(); ++iter)
    {
        const NodePtr& consumer = std::static_pointer_cast<Node>(*iter);
        LOG_TRACE(SRAM_SLICE, "{}: get chain for {}", HLLOG_FUNC, firstConsumer->getNodeName());
        PipelineChain tmp = expandProducerChain(consumer, candidate, {}, bundleNodes);
        // Keeping the shortest chain as it reflects the intersection of all MME consumers constraints on the allowed
        // nodes to bundle
        if (tmp.size() < masterProducerChain.size())
        {
            masterProducerChain = tmp;
        }
    }
    return masterProducerChain;
}

// Blocking non shared producers for some cases, after a drop of 3% was seen in perf of pt_bert_large_fp8_1x without
// this block
bool MantaRayBundlizer::blockNonSharedProducersChain(const MMENodePtr& mme,
                                                     const TensorPtr&  input,
                                                     unsigned          numMmesInBundle)
{
    // If there's another mme consumer not bundled, which may be able to bundle with this chain and slice it - allow it
    // to bundle with it as sliced chain
    bool anotherMmeConsumerFound = false;
    for (const NodePtr& n : m_graph.getTensorConsumers(input))
    {
        if ((n != mme) && m_graph.runsOnMME(n) && !n->getNodeAnnotation().bundleInfo.is_set() &&
            NodeSolver::doesNodeSupportSlicing(n))
        {
            anotherMmeConsumerFound = true;
            break;
        }
    }
    return (numMmesInBundle == 1 && anotherMmeConsumerFound);
}

void MantaRayBundlizer::addConsumers(MantaRaySolver::BundleParams& params, PipelineBundlePtr& bundle)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);

    const uint64_t SRAMCap = SlicingBrain::knobs.maxSRAMCapInBytes;
    NodeSet        otherNodesInBundle(bundle->getNodes().begin(), bundle->getNodes().end());
    const NodePtr& mmeNode = std::static_pointer_cast<Node>(*params.mmeNodes.begin());
    // Get updated and valid consumer chain
    PipelineChain consumerChain = getConsumerChain(mmeNode,
                                                   mmeNode->getOutput(0),
                                                   params.sharedOperand,
                                                   params.sharedOperandSlicingDims,
                                                   otherNodesInBundle);
    if (consumerChain.empty()) return;
    if (isBundleSupportedForConsumers(params.mmeNodes, consumerChain, bundle, params.doubleBufferFactor))
    {
        PipelineMultiChain slicedChains = getBundledSlicedChains(params);
        slicedChains.push_back(consumerChain);
        auto slicedOperandsSramUse = getSlicedTensorsMinSramUsage(params, slicedChains, false /* maxTile*/);
        if (tensorsFitSramCapacity(slicedOperandsSramUse, SRAMCap - params.unslicedOperandsSramSize))
        {
            LOG_TRACE(SRAM_SLICE, "{}: Adding consumers chain for node {}", HLLOG_FUNC, mmeNode->getNodeName());
            params.mmeOutputConsumers = consumerChain;
            for (auto consumer : consumerChain)
            {
                addNodeToBundle(consumer.node, bundle);
            }
            params.slicedOperandsSramMinSize = slicedOperandsSramUse;
        }
    }
}

bool MantaRayBundlizer::isBundleSupportedForConsumers(MMENodeSet         mmeNodes,
                                                      PipelineChain      consumerChain,
                                                      PipelineBundlePtr& bundle,
                                                      unsigned int       doubleBufferFactor) const
{
    // adding consumers to multiple mme bundle is not supported
    if (mmeNodes.size() > 1) return false;
    // In case of single mme node in bundle which will solve with single buffer, adding consumers will reduce bundle
    // performance
    if (bundle->getNodes().size() == 1 && doubleBufferFactor == 1) return false;
    NodePtr lastConsumedNode = consumerChain.back().node;

    if (!GCFG_ENABLE_BUNDLES_WITH_CONSUMERS_AND_PRODUCERS.value())
    {
        // Legacy behavior - support only bundle with single mme and consumers
        if (bundle->getNodes().size() != 1) return false;
    }

    for (auto output : lastConsumedNode->getOutputs())
    {
        if (consumerCanBeProducer(output, lastConsumedNode)) return false;
    }

    return true;
}

bool MantaRayBundlizer::consumerCanBeProducer(const TensorPtr& consumerOutput,
                                              const NodePtr&   originalConsumerNode) const
{
    // This function checks if the originalConsumerNode can be a producer.
    // It finds the next mme conumers consumed by consumerOutput and checks if originalConsumerNode can be a part of
    // one of the mme's producer chain.
    const NodeSet& mmeConsumers = getNextMmeConsumers(consumerOutput);
    for (const NodePtr& mmeConsumer : mmeConsumers)
    {
        if (mmeConsumer == nullptr || mmeConsumer->getNodeAnnotation().bundleInfo.is_set()) continue;
        for (const TensorPtr& input : mmeConsumer->getInputs())
        {
            if (input == nullptr || input->isShapeTensor()) continue;
            NodeSet             nodeInBundle({mmeConsumer});
            const PipelineChain pc = getProducerChain(mmeConsumer, input, nodeInBundle);
            if (std::any_of(pc.begin(), pc.end(), [&originalConsumerNode](const PipelinedNode& pn) {
                    return pn.node == originalConsumerNode;
                }))
            {
                // there is mme consumer that can bundle originalConsumerNode as producer
                return true;
            }
        }
    }
    return false;
}

NodeSet MantaRayBundlizer::getNextMmeConsumers(const TensorPtr& t) const
{
    // This function return the first mme consumers consumed directly and indirectly by t
    std::stack<NodePtr> nonMmeConsumers;
    NodeSet             visitedNonMmeConsumers;  // needed to avoid recheck mmeConsumers that already checked
    NodeSet             mmeConsumers;

    for (const NodePtr& consumer : m_graph.getTensorConsumers(t))
    {
        if (consumer == nullptr || consumer->isShapeOperation()) continue;
        if (!HabanaGraph::runsOnMME(consumer) || consumer->isLogicalOperation())
        {
            visitedNonMmeConsumers.insert(consumer);
            nonMmeConsumers.push(consumer);
        }
        else
        {
            mmeConsumers.insert(consumer);
        }
    }

    while (!nonMmeConsumers.empty())
    {
        const NodePtr consumer = nonMmeConsumers.top();
        nonMmeConsumers.pop();
        for (const NodePtr& consumersConsumer : m_graph.getNodeConsumers(consumer))
        {
            if (consumersConsumer == nullptr || consumersConsumer->isShapeOperation())
            {
                continue;
            }
            if (!HabanaGraph::runsOnMME(consumersConsumer))
            {
                const auto& res = visitedNonMmeConsumers.insert(consumersConsumer);
                if (res.second)  // add nonMmeConsumers only if it is the first occurrence
                {
                    nonMmeConsumers.push(consumersConsumer);
                }
            }
            else
            {
                mmeConsumers.insert(consumersConsumer);
            }
        }
    }
    return mmeConsumers;
}

// Calculate min input size, slicing dim alignment for shared mme and tensor granularities.
void MantaRayBundlizer::updateSlicingGranularityParams(MantaRaySolver::BundleParams& params) const
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    PipelineMultiChain slicedChains   = getBundledSlicedChains(params);
    auto [slicedNodes, slicedTensors] = getChainsNodesAndTensors(slicedChains);
    slicedNodes.insert(params.mmeNodes.begin(), params.mmeNodes.end());
    slicedTensors.insert(params.sharedOperand);
    params.tileSizes = CommonTileSizeCalculator::getMinCommonTilesSizes(slicedNodes, slicedTensors, m_graph).first;
    const auto it    = params.tileSizes.find(params.sharedOperand);
    HB_ASSERT(it != params.tileSizes.end(), "shared operand must have granularity");
    const TensorGranularity sharedOperandGranularity = it->second;

    auto mmeUtilGranularity = getMmeSharedInputGranularity(params.mmeNodes,
                                                           params.sharedOperand,
                                                           params.sharedOperandSlicingDims,
                                                           sharedOperandGranularity);

    params.sharedOperandMinSize =
        getSlicedTensorMinSize(params.sharedOperand, mmeUtilGranularity, params.sharedOperandSlicingDims);

    auto alignmentSize = getSharedOperandSliceAlignmentSize(params, mmeUtilGranularity);
    if (alignmentSize.has_value())
    {
        // Add single alignment size, as shared operand size is maintained without double buffer
        params.sharedOperandMinSize += alignmentSize.value();
        LOG_TRACE(SRAM_SLICE,
                  "{}: adding shared operand alignment size: {} MB",
                  HLLOG_FUNC,
                  bToMb(alignmentSize.value()));
    }
    // updated multiple MMEs common utilization granularity for the shared operand sliced dims
    for (auto dim : params.sharedOperandSlicingDims)
    {
        params.sharedOperandSlicingDimsAlignment.emplace(dim, mmeUtilGranularity[dim]);
    }
}

bool MantaRayBundlizer::tryPlacePartialsOutputsInSram(MantaRaySolver::BundleParams& params) const
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const uint64_t SRAMCap = SlicingBrain::knobs.maxSRAMCapInBytes;
    uint64_t       allocatedSram = params.unslicedOperandsSramSize + params.slicedOperandsSramMinSize;
    for (const auto& n : params.partialsConsumers)
    {
        uint64_t outputSize = getOutputSizeForPartials(n);
        if (!tensorsFitSramCapacity(outputSize, SRAMCap - allocatedSram))
        {
            LOG_DEBUG(SRAM_SLICE, "partials MME output doesn't fit SRAM: {}", n->getNodeName());
            return false;
        }
        else
        {
            allocatedSram += outputSize;
            // commit on the SRAM capacity for the output tensor
            params.unslicedOperandsSramSize += outputSize;
        }
    }
    return true;
}

TPCExpansionsAndSingleMMEBundleSolver::PipelineMultiChain
MantaRayBundlizer::getBundledSlicedChains(const MantaRaySolver::BundleParams& params) const
{
    PipelineMultiChain slicedChains;
    for (const auto& pc : params.nonSharedProducers)
    {
        HB_ASSERT(!pc.empty(), "bundled producers chains are expected to have nodes");
        if (!pc.front().slicingDims.empty())
        {
            slicedChains.push_back(pc);
        }
    }
    if (!params.mmeOutputConsumers.empty())
    {
        slicedChains.push_back(params.mmeOutputConsumers);
    }
    // must be pushed last, assumed by trimSharedProducersChainToFitSram
    if (!params.sharedOperandProducers.empty())
    {
        slicedChains.push_back(params.sharedOperandProducers);
    }
    return slicedChains;
}

// Given an operand and MME nodes that consume it, find a dimension to split, in order of preference
std::vector<unsigned> MantaRayBundlizer::getSlicingDimForSharedOperand(const TensorPtr&  masterOperand,
                                                                       const MMENodeSet& consumers)
{
    TSize dimHistogram[Tensor::c_tensorMaxDim] = {0};

    static const unsigned dimZeroPenalty = 3;
    for (MMENodePtr n : consumers)
    {
        unsigned masterInputIndex = n->getInputIndexOfTensor(masterOperand);
        auto     slicingDims      = NodeSolver::getInputSlicingDims(n, masterInputIndex);
        HB_ASSERT(!slicingDims.empty(),
                  "MME node {} refuses to slice tensor {} on any dimension",
                  n->getNodeName(),
                  masterOperand->getName());
        // select the first dim to slice for the heuristic selection. it should be the external
        unsigned dim = slicingDims.front();
        HB_ASSERT(dim < Tensor::c_tensorMaxDim, "MME Node wants to slice on N-dim at index {}", dim);
        dimHistogram[dim]++;
    }
    unsigned bestExternalDim      = 0;
    unsigned bestExternalDimScore = 0;
    // Go over the dims from outer to inner, as we prefer to slice on outer
    for (unsigned i = Tensor::c_tensorMaxDim - 1; i > 0; --i)
    {
        if (dimHistogram[i] > bestExternalDimScore)
        {
            bestExternalDim      = i;
            bestExternalDimScore = dimHistogram[i];
        }
    }
    if (dimHistogram[0] == 0)  // no request to slice on FCD
    {
        return {bestExternalDim};
    }
    if (dimHistogram[0] == consumers.size())  // all requested FCD
    {
        return {0};
    }
    // Prefer outer dim with heuristic
    if (dimHistogram[0] > dimZeroPenalty * bestExternalDimScore)
    {
        return {0, bestExternalDim};
    }
    return {bestExternalDim, 0};
}

bool MantaRayBundlizer::isProducerChainBoundary(const NodePtr& producer) const
{
    // TODO SW-76310 - limit the chain by the number of TPC nodes
    // Let chain expansion continue; We will chop it up later
    return false;
}
bool MantaRayBundlizer::isConsumerChainBoundary(const NodePtr& consumer) const
{
    // Currently support consumer chain which end with a single TPC node.
    return HabanaGraph::runsOnTPC(consumer);
}

bool MantaRayBundlizer::consumerGranularityChainBreaker(const PipelinedNode& nextCandidate) const
{
    if (!GCFG_ENABLE_BUNDLES_WITH_CONSUMERS_AND_PRODUCERS.value())
    {
        // behave like vision - make sure all sliced dims granularity is 1
        return TPCExpansionsAndSingleMMEBundlizer::consumerGranularityChainBreaker(nextCandidate);
    }
    // else - any granularity is fine, so the candidate isn't a chain breaker
    return false;
}

BundlesInfoContainer MantaRayBundlizer::generateBundles()
{
    LOG_TRACE(SRAM_SLICE, "MantaRayBundlizer::{}", HLLOG_FUNC);

    BundlesInfoContainer bundles;
    if (GCFG_SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED.value())
    {
        BundlesInfoContainer multiMmeBundles = generateMultiMmeWithTpcProducersBundles();
        bundles.insert(bundles.end(), multiMmeBundles.begin(), multiMmeBundles.end());
    }
    BundlesInfoContainer singleMmeBundles = generateMmeWithTpcExpansionsBundles();
    bundles.insert(bundles.end(), singleMmeBundles.begin(), singleMmeBundles.end());

    HB_ASSERT(validateBundles(bundles), "Failed bundles validation");
    return bundles;
}

BundlesInfoContainer MantaRayBundlizer::generateMultiMmeWithTpcProducersBundles()
{
    BundlesInfoContainer bundles;

    // Find all MME nodes that consume exactly one operand in common
    MMENodeSet                                visitedNodes;
    std::map<TensorPtr, MMENodeSet, TensorComparator> masterOperands;

    for (const TensorPtr& t : m_graph.getTensors())
    {
        if (t->isControlEdge()) continue;

        MMENodeSet mmeConsumers;
        for (const NodePtr& n : m_graph.getTensorConsumers(t))
        {
            if (!m_graph.runsOnMME(n)) continue;
            const MMENodePtr& pMME = std::dynamic_pointer_cast<MmeNode>(n);
            if (visitedNodes.find(pMME) != visitedNodes.end()) continue;
            if (n->getNodeAnnotation().bundleInfo.is_set()) continue;
            if (!NodeSolver::doesNodeSupportSlicing(n)) continue;
            mmeConsumers.insert(pMME);
        }
        // Cull the candidate list to avoid cycles or shared input on more than 1 operand
        // TODO SW-76351: handle a case where they share more than one operand
        MMENodeSet committedConsumers;
        for (const MMENodePtr& n : mmeConsumers)
        {
            if (canBundleSharedInputMmeNode(n, mmeConsumers, committedConsumers, t))
            {
                committedConsumers.insert(n);
            }
            if (committedConsumers.size() == nMMEToBundle)
            {
                break;
            }
        }
        if (committedConsumers.size() > 1)
        {
            visitedNodes.insert(committedConsumers.begin(), committedConsumers.end());
            masterOperands[t] = committedConsumers;
        }
    }

    LOG_TRACE(SRAM_SLICE, "Found {} potential master operands", masterOperands.size());
    if (LOG_LEVEL_AT_LEAST_TRACE(SRAM_SLICE))
    {
        for (const auto& it : masterOperands)
        {
            LOG_TRACE(SRAM_SLICE, "Operand {} with {} consumers", it.first->getName(), it.second.size());
            for (const MMENodePtr& n : it.second)
            {
                LOG_TRACE(SRAM_SLICE, "{}", n->getNodeName());
            }
        }
    }

    // Create bundle around each master operand
    for (const auto& it : masterOperands)
    {
        const TensorPtr&      masterOperand = it.first;
        const MMENodeSet&     mmeConsumers  = it.second;
        std::vector<unsigned> slicingDims   = getSlicingDimForSharedOperand(masterOperand, mmeConsumers);
        HB_ASSERT(!slicingDims.empty(), "Couldn't slice {} on any dim", masterOperand->getName());
        unsigned mmeSlicingDim = slicingDims[0];  // dims are by priority
        LOG_TRACE(SRAM_SLICE, "Trying for shared operand {} sliced on dim {}", masterOperand->getName(), mmeSlicingDim);

        if (masterOperand->getSizeInElements(mmeSlicingDim) <= 1)
        {
            LOG_DEBUG(SRAM_SLICE,
                      "Cannot slice master operand {} on dim {} - too small",
                      masterOperand->getName(),
                      mmeSlicingDim);
            continue;
        }
        std::vector<Dim> slicingDimsPrio = getMultiMmeCommonSlicingDims(mmeConsumers, masterOperand, mmeSlicingDim);
        std::optional<BundleAndSolver> bundle = createMantaRayBundle(mmeConsumers, slicingDimsPrio, masterOperand);
        if (bundle)
        {
            bundles.push_back(*bundle);
        }
    }
    return bundles;
}

std::vector<Dim> MantaRayBundlizer::getSlicingDimsIntersection(const std::vector<Dim>& n1SlicingDims,
                                                               const std::vector<Dim>& slicingDimsIntersectionAcc,
                                                               const NodePtr&          n1) const
{
    // set_intersection requires the dims to be sorted
    HB_ASSERT(std::is_sorted(n1SlicingDims.begin(), n1SlicingDims.end(), std::greater<unsigned>()),
              "Expecting slicing dims to be sorted, {} for node {}",
              toString(n1SlicingDims, ','),
              n1->getNodeName());
    std::vector<Dim> slicingDimsIntersectionCurr;
    std::set_intersection(n1SlicingDims.begin(),
                          n1SlicingDims.end(),
                          slicingDimsIntersectionAcc.begin(),
                          slicingDimsIntersectionAcc.end(),
                          std::back_inserter(slicingDimsIntersectionCurr),
                          std::greater<unsigned>());
    return slicingDimsIntersectionCurr;
}
std::vector<Dim> MantaRayBundlizer::getMultiMmeCommonSlicingDims(const MMENodeSet& mmeConsumers,
                                                                 const TensorPtr&  masterOperand,
                                                                 Dim               selectedSlicingDim)
{
    std::vector<Dim> slicingDimsByPrio = {selectedSlicingDim};
    HB_ASSERT(!mmeConsumers.empty(), "Expecting mme consumers not empty");
    // Add all the common slicing dims between all the mme consumers
    std::vector<Dim> slicingDimsByPrioIntersection(SYN_MAX_TENSOR_DIM);
    std::iota(slicingDimsByPrioIntersection.rbegin(), slicingDimsByPrioIntersection.rend(), 0);
    for (const auto& mmeConsumer : mmeConsumers)
    {
        const unsigned masterInputIndex = mmeConsumer->getInputIndexOfTensor(masterOperand);
        const auto     slicingDims      = NodeSolver::getInputSlicingDims(mmeConsumer, masterInputIndex);
        slicingDimsByPrioIntersection = getSlicingDimsIntersection(slicingDims, slicingDimsByPrioIntersection, mmeConsumer);
    }
    // Local node decisions have an intersection, prefer them over selected slicingDimsByPrio which is initialized by a
    // global bundle decision
    if (!slicingDimsByPrioIntersection.empty())
    {
        slicingDimsByPrio = slicingDimsByPrioIntersection;
        LOG_TRACE(SRAM_SLICE,
                  "{}: Intersection of slicing dims of all mme consumers: {}",
                  __FUNCTION__,
                  toString(slicingDimsByPrioIntersection, ','));
    }
    // Since there is no common dim, go with the global selected slicing dim
    else
    {
        LOG_TRACE(SRAM_SLICE,
                  "{}: Slicing dim is not shared by all mme consumers, choosing to slice dim: {}",
                  __FUNCTION__,
                  toString(slicingDimsByPrio, ','));
    }
    return slicingDimsByPrio;
}

std::optional<BundleAndSolver> MantaRayBundlizer::createBundle(const NodePtr&            mmeNode,
                                                               const PipelineMultiChain& producerChains,
                                                               const PipelineChain&      consumerChain,
                                                               const TensorPtr&          mmeInputToSlice)
{
    MMENodeSet mmeNodes = {std::static_pointer_cast<MmeNode>(mmeNode)};

    TensorPtr masterOperand = getMmeInputToSlice(mmeNode, producerChains, true);
    // Node had already been validated to have at least one slicable operand hence shared operand can't be null
    HB_ASSERT_PTR(masterOperand);
    auto slicingDims = NodeSolver::getInputSlicingDims(mmeNode, mmeNode->getInputIndexOfTensor(masterOperand));
    HB_ASSERT(!slicingDims.empty(), "Trying to bundle MME without valid slicing dim");
    return createMantaRayBundle(mmeNodes, slicingDims, masterOperand);
}

NodeVector TPCExpansionsAndSharedMMEBundlizer::getNodesBundleCreationOrder(const NodeSet& nodes)
{
    NodeVector nodesVec(nodes.begin(), nodes.end());

    // Preferring bundles with dedx, because heuristically in RN50, it improves the BWD pipelining.
    std::stable_sort(nodesVec.begin(), nodesVec.end(), [](const NodePtr& lhs, const NodePtr& rhs) {
        return Node::isDedxNode(lhs) && !Node::isDedxNode(rhs);
    });

    return nodesVec;
}

std::optional<BundleAndSolver>
TPCExpansionsAndSharedMMEBundlizer::createBundle(const NodePtr&            mmeNode,
                                                 const PipelineMultiChain& producerChains,
                                                 const PipelineChain&      consumerChain,
                                                 const TensorPtr&          mmeInputToSlice)
{
    auto bundleAndSolver = TPCExpansionsAndSingleMMEBundlizer::createBundle(mmeNode, producerChains, consumerChain, mmeInputToSlice);

    if (!GCFG_SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED.value())
    {
        return bundleAndSolver;
    }

    auto bundle = bundleAndSolver->first;
    LOG_DEBUG(SRAM_SLICE, "Looking for shared MME for bundle {}", bundle->index());
    bool isSharedMme = false;
    NodePtr sharedInputConsumer;
    if (Node::isDedxNode(mmeNode))
    {
        // Only support stitching of operand 0 (dy)
        TensorPtr sharedOperand = mmeNode->getInput(0);
        // Find dedw consumer for the stitched operand
        const NodeList& sharedOpConsumers = m_graph.getTensorConsumers(sharedOperand);
        auto            sharedInputConsumerIt =
            std::find_if(sharedOpConsumers.begin(), sharedOpConsumers.end(), [&](const NodePtr& consumer) {
                return consumer->getNodeType() == Node::TYPE_DEDW;
            });
        if (sharedInputConsumerIt != sharedOpConsumers.end())
        {
            sharedInputConsumer               = *sharedInputConsumerIt;
            PipelinedNode candidate = PipelinedNode(sharedInputConsumer, sharedOperand, {DIM_B} /* not used */);
            NodeSet       bundleNodes(bundle->getNodes().begin(), bundle->getNodes().end());
            if (consumerCyclesChainBreaker(candidate, {}, mmeNode, bundleNodes))
            {
                LOG_DEBUG(SRAM_SLICE,
                          "{} creates a circle in BP graph, not adding it to bundle",
                          sharedInputConsumer->getNodeName());
            }
            else if (sharedInputConsumer->getInput(1) == mmeNode->getInput(1))
            {
                LOG_DEBUG(SRAM_SLICE,
                          "{} shares two input tensors with {}, not adding it to bundle",
                          sharedInputConsumer->getNodeName(),
                          mmeNode->getNodeName());
            }
            else if (mmeInputToSlice != sharedOperand)
            {
                LOG_DEBUG(SRAM_SLICE,
                          "don't bundle shared mme for node {} if the sliced input is not the shared operand",
                          mmeNode->getNodeName());
            }
            else if (shouldSliceOnSpatialDim(mmeNode, sharedOperand))
            {
                LOG_DEBUG(SRAM_SLICE,
                          "don't bundle shared mme for node {} if the shared operand should be sliced spatially",
                          mmeNode->getNodeName());
            }
            else if (dedwPrefersToCacheX(sharedInputConsumer))
            {
                LOG_DEBUG(SRAM_SLICE,
                          "don't bundle shared mme for node {} if dedw prefers to cache X and can't",
                          mmeNode->getNodeName());
            }
            else
            {
                LOG_DEBUG(SRAM_SLICE,
                          "Found shared MME/slave input consumer: {}, for: {}",
                          sharedInputConsumer->getNodeName(),
                          mmeNode->getNodeName());
                // notice it is important that the dedw is added to the bundle after dedx
                addNodeToBundle(sharedInputConsumer, bundle);
                isSharedMme = true;
            }
        }
    }
    if (isSharedMme)
    {
        HB_ASSERT_PTR(sharedInputConsumer);
        bundleAndSolver->second.reset(new SharedMmeProducerChainBundleSolver(bundle,
                                                                             m_graph,
                                                                             producerChains,
                                                                             consumerChain,
                                                                             mmeNode,
                                                                             sharedInputConsumer,
                                                                             mmeInputToSlice));
    }
    return bundleAndSolver;
}

bool TPCExpansionsAndSharedMMEBundlizer::dedwPrefersToCacheX(const NodePtr& dedw)
{
    return (isDedwBwBoundOnX(dedw) && !canCacheXInJointBundle(dedw));
}

bool TPCExpansionsAndSharedMMEBundlizer::canCacheXInJointBundle(const NodePtr& dedw)
{
    // TODO SW-115193: currently assuming sizes according to spatial slicing, as calculating the actual sizes causes
    // perf degradation in some models
    bool xRequiresSpatialSlicing = shouldSliceOnSpatialDim(dedw, dedw->getInput(TENSOR_X_BWD), 2);
    return !xRequiresSpatialSlicing;
}

bool TPCExpansionsAndSharedMMEBundlizer::isDedwBwBoundOnX(const NodePtr& dedw)
{
    // TODO SW-115193: currently assuming BW according to convolution stride params, as BW estimation causes perf
    // degradation in some models, and no quick cost model was found
    const auto conv = std::dynamic_pointer_cast<ConvBaseNode>(dedw);
    HB_ASSERT_PTR(conv);

    bool convStrided = conv->getConvolutionParams().stride[CONV_STRIDE_DEPTH] *
                           conv->getConvolutionParams().stride[CONV_STRIDE_HEIGHT] *
                           conv->getConvolutionParams().stride[CONV_STRIDE_WIDTH] > 1;
    return convStrided;
}
