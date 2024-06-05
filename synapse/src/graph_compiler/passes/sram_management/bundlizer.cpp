#include "bundlizer.h"
#include "defs.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "pipeline_management/node_projector.h"
#include "slicing_brain.h"
#include "slicing_utils.h"
#include "mme_shared_input.h"
#include "pattern_solvers.h"
#include "bundle_plane_graph.h"
#include "dma_transpose_node.h"
#include <stack>
#include "bundlizer_utils.h"
#include "register_memory_coherence.h"

Bundlizer::Bundlizer(HabanaGraph& graph) : m_graph(graph)
{
    // for some unit tests that initialize the bundlizer without passing through the sram slicer
    if (m_graph.getBPGraph() == nullptr)
    {
        m_graph.constructBPGraph();
    }
}

// Mark the nodes with RMW tensors with bundle ID. The caller doesn't need the resulting bundles list.
// Used also by Pipeline Manager temporarily
void Bundlizer::bundleNodesWithRMWSectionOperand()
{
    BundleList complexGuidBundles;
    generateRMWSectionBundles(complexGuidBundles);
}

// To prevent rmw section users from mixing with SRAM management bundles,
// use bundles to wrap the nodes of each rmw section together.
// This should take care to execute the rmw-section nodes to completion and leave the SRAM clean for MME bundles.
void Bundlizer::generateRMWSectionBundles(BundleList& rmwSectionBundles)
{
    if (!GCFG_ENABLE_COMPLEX_GUID_BUNDLES.value()) return;

    std::unordered_map<uint64_t, NodeVector> rmwSectionBundleNodes;
    for (const NodePtr& node : m_graph.getNodes())
    {
        if (!node->isPartOfRMWSection()) continue;
        const uint64_t RMWSectionId = node->getRMWSectionId();
        rmwSectionBundleNodes[RMWSectionId].push_back(node);
    }

    for (const auto& bundleNodes : rmwSectionBundleNodes)
    {
        pBundle bundle = createBundleFromNodes(bundleNodes.second, BundleType::COMPLEX_GUID);
        rmwSectionBundles.push_back(bundle);
    }
}

bool Bundlizer::shouldEnableScalarPipeBundle(const TPCNodePtr& tpcNode, tpc_lib_api::DeviceId deviceId) const
{
    return (tpcNode->getNumOutputs() > 0) &&
           (tpcNode->getScalarPipeInputsSize(deviceId) > GCFG_MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT.value());
}

void Bundlizer::generateBundles(BundleList& mmeBundles,
                                BundleList& scalarPipeBundles,
                                BundleList& tpcBundles,
                                BundleList& rmwSectionBundles,
                                BundleList& dmaTransposeBundles)
{
    generateRMWSectionBundles(rmwSectionBundles);
    for (const pNode& n : m_graph.getExeSortedNodes())
    {
        if (n->getNodeAnnotation().bundleInfo.is_set()) continue;
        if (n->hasHighRankOperand())  // TODO: SW-120808 - high rank support
        {
            SLC_WARN("Node {} (type {}) has high rank operand - bundling is not supported",
                     n->getNodeName(),
                     n->getNodeTypeStr());
            continue;
        }
        if (HabanaGraph::runsOnMME(n) &&
            (n->getNodeType() != Node::TYPE_INTERNAL_TRANSPOSE) &&  // TODO SW-8245 - support transpose
            (n->getNodeType() != Node::TYPE_MASKED_BATCH_GEMM))
        {
            pBundle bundle(new Bundle(BundleType::MME));
            addNodeToBundle(bundle, n);
            mmeBundles.push_back(bundle);
            SLC_DEBUG("MME Bundle added: {}, MME node name: {}", bundle->getName(), n->getNodeName());
            if (GCFG_ENABLE_TPC_BUNDLES.value())
            {
                pBundle bundle = getTPCBundle(n);
                if (bundle)
                {
                    tpcBundles.push_back(bundle);
                    SLC_DEBUG("TPC Bundle added: {}", bundle->getName());
                }
            }
        }
        else if (n->isDma())
        {
            std::shared_ptr<DMATransposeNode> pTransposeNode = std::dynamic_pointer_cast<DMATransposeNode>(n);
            if (pTransposeNode != nullptr)
            {
                pBundle bundle(new Bundle(BundleType::DMA_TRANSPOSE));
                addNodeToBundle(bundle, pTransposeNode);
                auto dmaTransposeSolver = std::make_shared<DMATransposeSolver>(*m_graph.getHALReader(), bundle);
                if (dmaTransposeSolver->effectiveForBundle())
                {
                    dmaTransposeBundles.push_back(bundle);
                    SLC_DEBUG("dmaTranspose Bundle added: {}", bundle->getName());
                }
                else
                {
                    removeBundle(pTransposeNode);
                }
            }
        }
        else
        {
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(n);
            if (!tpcNode) continue;

            if (shouldEnableScalarPipeBundle(tpcNode, deviceTypeToDeviceID(m_graph.getDeviceType())))
            {
                pBundle bundle(new Bundle(BundleType::SCALAR_PIPE));
                addNodeToBundle(bundle, tpcNode);
                auto scalarPipeSolver = std::make_shared<TPCScalarPipeSolver>(*m_graph.getHALReader(), bundle);
                if (scalarPipeSolver->effectiveForBundle())
                {
                    scalarPipeBundles.push_back(bundle);
                    SLC_DEBUG("TPC Scalar Pipe Bundle added: {}", bundle->getName());
                }
                else
                {
                    removeBundle(tpcNode);
                }
            }
        }
    }
}

BundleList Bundlizer::getMMEBundles()
{
    BundleList mmeBundles;
    BundleList scalarPipeBundles;
    BundleList tpcBundles;
    BundleList rmwSectionBundles;
    BundleList dmaTransposeBundles;
    generateBundles(mmeBundles, scalarPipeBundles, tpcBundles, rmwSectionBundles, dmaTransposeBundles);
    return mmeBundles;
}

BundleList Bundlizer::getTPCScalarPipeBundles()
{
    BundleList mmeBundles;
    BundleList scalarPipeBundles;
    BundleList tpcBundles;
    BundleList rmwSectionBundles;
    BundleList dmaTransposeBundles;
    generateBundles(mmeBundles, scalarPipeBundles, tpcBundles, rmwSectionBundles, dmaTransposeBundles);
    return scalarPipeBundles;
}

bool Bundlizer::isValidNodeForTPCBundle(const NodeList& bundleNodes, pNode& candidate, TensorPtr& connectingTensor)
{
    if (m_graph.getNodeConsumers(candidate).size() == 0) return false;
    for (const NodePtr& node : bundleNodes)
    {
        if (m_graph.getBPGraph()->getNumberOfPaths(node, candidate) > 1) return false;
    }
    if (candidate->hasHighRankOperand())  // TODO: SW-120808 - high rank support
    {
        SLC_WARN("Node {} (type {}) has high rank operand - bundling is not supported",
                 candidate->getNodeName(),
                 candidate->getNodeTypeStr());
        return false;
    }
    if (candidate->isLogicalOperation() &&
        std::dynamic_pointer_cast<ReshapeNode>(candidate) != nullptr)
    {
        // Todo- WA for [SW-44744] Need to handle dynamic split of shape tensor for reshape
        if (candidate->isDynamicShape())
        {
            return false;
        }
        const pTensor& in = candidate->getInput(0);
        const pTensor& out = candidate->getOutput(0);
        // Bundle only if the last dimension in each tensor stays the same size
        return in->getSizeInElements(in->getDim() - 1) == out->getSizeInElements(out->getDim() - 1);
    }
    TPCNodePtr tpcNode  = std::dynamic_pointer_cast<TPCNode>(candidate);
    auto       deviceId = deviceTypeToDeviceID(m_graph.getDeviceType());
    if (!tpcNode)
    {
        return false;
    }
    if (shouldEnableScalarPipeBundle(tpcNode, deviceId))
    {
        return false;
    }
    if (!(tpcNode->isSeparable(deviceId, connectingTensor->getDim() - 1)))
    {
        return false;
    }
    if (tpcNode->hasTransposeOptimization(deviceId))
    {
        return false;
    }
    return true;
}

BundleList Bundlizer::getTPCBundles()
{
    BundleList mmeBundles;
    BundleList scalarPipeBundles;
    BundleList tpcBundles;
    BundleList rmwSectionBundles;
    BundleList dmaTransposeBundles;
    generateBundles(mmeBundles, scalarPipeBundles, tpcBundles, rmwSectionBundles, dmaTransposeBundles);
    return tpcBundles;
}

void Bundlizer::removeLastNodesFromBundle(NodeList& bundleNodes, int counter)
{
    bundleNodes.erase(std::prev(bundleNodes.end(), counter), bundleNodes.end());
}

pBundle Bundlizer::makeBundle(NodeList& bundleNodes, BundleType type)
{
    pBundle bundle(new Bundle(type));
    for (pNode& node : bundleNodes)
    {
        addNodeToBundle(bundle, node);
    }
    return bundle;
}

TensorPtr Bundlizer::getConnectingTensor(const NodePtr& producer, const NodePtr& consumer) const
{
    TensorPtr ret;
    auto consumerInputs = consumer->getInputs();
    for (const pTensor& tensor : producer->getOutputs())
    {
        if (std::find(consumerInputs.begin(), consumerInputs.end(), tensor) != consumerInputs.end())
        {
            return tensor;
        }
    }
    return ret;
}

bool Bundlizer::canAddParallelTpc(const TPCNodePtr& tpcCandidate, const NodeSet& parallelTpcs) const
{
    // Make sure there is no path between the parallel TPC nodes to avoid inter-bundle dependencies.
    for (const auto& parallelTpc : parallelTpcs)
    {
        if ((m_graph.getBPGraph()->getNumberOfPaths(tpcCandidate, parallelTpc) > 0) ||
            (m_graph.getBPGraph()->getNumberOfPaths(parallelTpc, tpcCandidate) > 0))
        {
            return false;
        }
    }
    return true;
}

bool Bundlizer::canConsumeTpcBundle(const pNode&     candidate,
                                    const TensorPtr& connectingTensor,
                                    const pNode&     producer) const
{
    return (HabanaGraph::runsOnMME(candidate) &&
            // The TPC chain should connect to operand A of the MME, as it planed to be sliced on the outer dimension
            (connectingTensor == candidate->getInput(TENSOR_IFM)) &&
            // Prevent create a TPC bundle from FWD sequence produce the bwd MME
            (m_graph.getBPGraph()->getNumberOfPaths(producer, candidate) == 1));
}

pBundle Bundlizer::getTPCBundle(const NodePtr& node)
{
    std::stack<std::tuple<NodePtr, NodePtr, TensorPtr>> nodesToCheck;
    std::stack<int> lastDetour;
    NodeList bundleNodes;
    for (const NodePtr& consumer : m_graph.getNodeConsumers(node))
    {
        if (HabanaGraph::runsOnTPC(consumer) && consumer->getNodeAnnotation().bundleInfo.is_set()) continue;
        TensorPtr connectingTensor = getConnectingTensor(node, consumer);
        nodesToCheck.push(std::make_tuple(node, consumer, connectingTensor));
    }
    while (!nodesToCheck.empty())
    {
        NodePtr prod = std::get<0>(nodesToCheck.top());
        NodePtr curr = std::get<1>(nodesToCheck.top());
        TensorPtr connectingTensor = std::get<2>(nodesToCheck.top());
        nodesToCheck.pop();
        if (isValidNodeForTPCBundle(bundleNodes.empty()? NodeList({prod}) : bundleNodes, curr, connectingTensor))
        {
            bundleNodes.push_back(curr);
            if (lastDetour.size() > 0)
            {
                lastDetour.top()++;
            }
            NodeSet consumers = m_graph.getNodeConsumers(curr);
            for (const NodePtr& consumer : consumers)
            {
                if (HabanaGraph::runsOnTPC(consumer) && consumer->getNodeAnnotation().bundleInfo.is_set()) continue;
                TensorPtr connectingTensor = getConnectingTensor(curr, consumer);
                nodesToCheck.push(std::make_tuple(curr, consumer, connectingTensor));
            }
            for (int i = 0; i < consumers.size() - 1; i++)
            {
                lastDetour.push(0);
            }
        }
        else
        {
            // Add tpc bundle only if its connected to operandA of the MME node - slicing issue
            if ((bundleNodes.size() >= 3) && canConsumeTpcBundle(curr, connectingTensor, prod))
            {
                NodeSet parallelTpcNodes;
                for (const auto& tpcBundleConsumer : m_graph.getTensorConsumers(connectingTensor))
                {
                    if (canConsumeTpcBundle(tpcBundleConsumer, connectingTensor, prod))
                    {
                        // If the second MME operand is produced by element-wise TPC node, add it to bundle
                        const NodePtr& producer = m_graph.getTensorProducer(tpcBundleConsumer->getInput(TENSOR_WEIGHT));
                        if (producer)
                        {
                            std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(producer);
                            if (tpcNode &&
                                // Check if the producer not in another bundle
                                !producer->getNodeAnnotation().bundleInfo.is_set() &&
                                tpcNode->isSeparable(deviceTypeToDeviceID(m_graph.getDeviceType())) &&
                                // Check there's only one connection to prevent cyclic bundle connection
                                (m_graph.getBPGraph()->getNumberOfPaths(producer, tpcBundleConsumer) == 1) &&
                                // Check there is no connection between the parallel TPCs nodes
                                canAddParallelTpc(tpcNode, parallelTpcNodes) &&
                                // if producer's output is both inputs of following MME don't bundle twice
                                (std::find(bundleNodes.begin(), bundleNodes.end(), producer) == bundleNodes.end()))
                            {
                                bundleNodes.push_back(producer);
                                parallelTpcNodes.insert(tpcNode);
                            }
                        }
                    }
                }
                return makeBundle(bundleNodes, BundleType::TPC);
            }
            else if (lastDetour.size() > 0)
            {
                removeLastNodesFromBundle(bundleNodes, lastDetour.top());
                lastDetour.pop();
            }
            else if (bundleNodes.size() > 0)
            {
                bundleNodes.clear();
            }
        }
    }
    return nullptr;
}

bool Bundlizer::addCandidateToBundle(pBundle& bundle, const pBundleExpansion& expansionCandidate)
{
    if (!expansionCandidate->nodeToStitch) return false;
    if (expansionCandidate->reshapeNode)
    {
        addNodeToBundle(bundle, expansionCandidate->reshapeNode);
    }
    addNodeToBundle(bundle, expansionCandidate->nodeToStitch);
    return true;
}

void Bundlizer::registerNodeToBundle(pBundle& bundle, const BundleInfo& info, const NodePtr& node)
{
    HB_ASSERT(!node->getNodeAnnotation().bundleInfo.is_set(),
              "try to insert node {} to bundle {}, but it already belongs to bundle {}",
              node->getNodeName(),
              info.bundleIndex,
              node->getNodeAnnotation().bundleInfo->bundleIndex);
    bundle->addNode(node);
    m_nodeBundleMap.insert({node, bundle});
    node->getNodeAnnotation().bundleInfo.set(info);
}

void Bundlizer::addNodeToBundle(pBundle& bundle, const NodePtr& node)
{
    BundleInfo info(bundle->index(), bundle->type());
    registerNodeToBundle(bundle, info, node);
    m_graph.getBPGraph()->addNodeToBundle(node, info);
}

pBundle Bundlizer::createBundleFromNodes(const NodeVector& nodes, BundleType type)
{
    pBundle    bundle(new Bundle(type));
    BundleInfo info(bundle->index(), type);

    NodeSet intersectingNodes = m_graph.getIntersectingNodes(nodes);
    HB_ASSERT(intersectingNodes.size() >= nodes.size(), "failed to get intersecting nodes");
    if (intersectingNodes.size() > nodes.size())
    {
        SLC_WARN("new bundle includes {} unexpected nodes", intersectingNodes.size() - nodes.size());
        if (LOG_LEVEL_AT_LEAST_DEBUG(SRAM_SLICE))
        {
            SLC_DEBUG("original nodes:");
            for (const NodePtr& n : nodes)
            {
                SLC_DEBUG("\t{}", n->getNodeName());
            }
            SLC_DEBUG("bundled nodes:");
            for (const NodePtr& n : intersectingNodes)
            {
                SLC_DEBUG("\t{}", n->getNodeName());
            }
        }
    }

    for (const NodePtr& node : intersectingNodes)
    {
        registerNodeToBundle(bundle, info, node);
    }
    m_graph.getBPGraph()->createBundleFromNodes({intersectingNodes.begin(), intersectingNodes.end()}, info);
    return bundle;
}

pBundleExpansion Bundlizer::findMmeInputSharingCandidate(const pMmeSlicingStrategy&    strategy,
                                                         const BundleSolvingDataMap&   solvingDataPerBundle,
                                                         const ExpansionCandidatesSet& existingCandidates) const
{
    SharedMMEInputCandidateHandler handler;
    std::list<pBundleExpansion> candidates = handler.findSharedMMEInputCandidate(strategy, m_graph);
    // make sure candidates are not already stitched to another bundle
    candidates.remove_if([&](const pBundleExpansion& candidate)
                         {
                             pBundle b = findBundleByNode(candidate->nodeToStitch);
                             unsigned count = 0;
                             for (auto& node : b->getNodes())
                             {
                                 if (HabanaGraph::runsOnMME(node) || HabanaGraph::runsOnTPC(node)) count++;
                             }
                             // If there is more than one mme/tpc node in the bundle we cannot stitch the node to the
                             // current bundle, as it has already been stitched to a different bundle.
                             bool candidatesBundleAlreadyExpanded = (count > 1);
                             // Also if candidate already been found no need to count it again.
                             bool alreadyFound =
                                 (existingCandidates.find(candidate->nodeToStitch) != existingCandidates.end());
                             // If the slave candidate has no valid strategy, it cannot be stitched as slave MME.
                             bool hasSolution = ((solvingDataPerBundle.count(b) > 0) &&
                                                 !solvingDataPerBundle.at(b).strategies.empty());
                             return candidatesBundleAlreadyExpanded || alreadyFound || !hasSolution;
                         });
    // TODO - [SW-8475] support more then one consumer.
    return candidates.empty() ? nullptr : candidates.front();
}

pBundleExpansion Bundlizer::findWideTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                                  const ExpansionCandidatesSet& existingCandidates) const
{
    return findTpcProducerExpansionCandidate(strategy,
                                             strategy->getMmeSlicingData().getWide(),
                                             BundleExpansion::WideInputProducer,
                                             existingCandidates);
}

pBundleExpansion Bundlizer::findNarrowTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                                    const ExpansionCandidatesSet& existingCandidates) const
{
    return findTpcProducerExpansionCandidate(strategy,
                                             strategy->getMmeSlicingData().getNarrow(),
                                             BundleExpansion::NarrowInputProducer,
                                             existingCandidates);
}

pBundleExpansion Bundlizer::findSlaveTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                                  const ExpansionCandidatesSet& existingCandidates) const
{
    const auto& expansionCandidates = strategy->getMmeSlicingData().getRoleCandidates();
    const auto& slaveInput = expansionCandidates[BundleExpansion::Role::SharedInputConsumer]->slaveOperands.getInput();
    return findTpcProducerExpansionCandidate(strategy,
                                             slaveInput,
                                             BundleExpansion::SlaveInputProducer,
                                             existingCandidates);
}

pBundleExpansion Bundlizer::findTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                              const pSlicedOperand& mmeSlicedInput,
                                                              const BundleExpansion::Role& role,
                                                              const ExpansionCandidatesSet& existingCandidates) const
{
    HB_ASSERT(role == BundleExpansion::WideInputProducer   ||
              role == BundleExpansion::NarrowInputProducer ||
              role == BundleExpansion::SlaveInputProducer, "role mismatch");
    const MmeSlicingStrategy::MmeSlicingData& slicingData = strategy->getMmeSlicingData();
    pBundleExpansion candidate      = std::make_shared<BundleExpansion>();
    pNode            mmeNode        = m_graph.getTensorProducer(slicingData.masterOperand->originalTensor);

    pNode reshapeNode = nullptr;
    pNode tpcProducer = findNonReshapeProducer(mmeSlicedInput->originalTensor, reshapeNode);

    candidate->role             = role;
    candidate->nodeToStitch     = tpcProducer;
    candidate->reshapeNode      = reshapeNode;
    candidate->stitchedOperand  = mmeSlicedInput;
    candidate->bundleNode       = mmeNode;

    if (role == BundleExpansion::SlaveInputProducer)
    {
        // In case we have a slave input producer we need to update the bundleNode (consumer node) to be the
        // slave mme node instead of the master before we check if the producer can be added.
        // Otherwise we will get a no connection between producer and consumer error.
        const auto& slaveCandidate = slicingData.getRoleCandidates()[BundleExpansion::SharedInputConsumer];
        if ((slaveCandidate == nullptr) || (slaveCandidate->nodeToStitch == nullptr))
        {
            return std::make_shared<BundleExpansion>();
        }
        candidate->bundleNode = slaveCandidate->nodeToStitch;
    }

    if ((tpcProducer != nullptr) && (existingCandidates.find(tpcProducer) == existingCandidates.end()) &&
        isNodeEligibleForStitching(tpcProducer, mmeSlicedInput, candidate->reshapeNode) &&
        SlicedOperandUtils::canAlignProducerReshape(candidate))
    {
        return candidate;
    }

    return std::make_shared<BundleExpansion>();
}

pBundleExpansion Bundlizer::findTpcConsumerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                              const ExpansionCandidatesSet& existingCandidates,
                                                              BundleExpansion::Role role /*= BundleExpansion::OutputConsumer*/) const
{
    HB_ASSERT(role == BundleExpansion::OutputConsumer || role == BundleExpansion::SlaveOutputConsumer, "role mismatch");

    const MmeSlicingStrategy::MmeSlicingData& slicingData = strategy->getMmeSlicingData();
    const pSlicedOperand& mmeOut = (role ==  BundleExpansion::OutputConsumer) ?
            slicingData.masterOperand :
            slicingData.getRoleCandidates()[BundleExpansion::Role::SharedInputConsumer]->slaveOperands.getOutput();
    const pNode& mmeNode = m_graph.getTensorProducer(mmeOut->originalTensor);

    pBundleExpansion candidate = std::make_shared<BundleExpansion>();
    candidate->role = role;
    candidate->bundleNode = mmeNode;

    // TODO [SW-7955] Can't consume reduction in a bundle ==> Can't consume dedw
    if (mmeNode->getNodeType() == Node::TYPE_DEDW)
    {
        return candidate;
    }

    if (!strategyIsConsumerStitchingCapable(strategy, mmeOut)) return candidate;

    for (const pNode& consumer : m_graph.getTensorConsumers(mmeOut->originalTensor))
    {
        if (consumer->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            const pNode& reshapeNode = consumer;
            for (const pNode& reshapedConsumer : m_graph.getTensorConsumers(reshapeNode->getOutput(0)))
            {
                if (existingCandidates.find(reshapedConsumer) == existingCandidates.end() &&
                    isNodeEligibleForStitching(reshapedConsumer, mmeOut, reshapeNode) &&
                    // Consumer that is dynamic shape is currently not supported for reshape
                    // alignment - the shape tensor sizes are not trivial
                    !reshapedConsumer->isDynamicShape())
                {
                    candidate->nodeToStitch    = reshapedConsumer;
                    candidate->stitchedOperand = mmeOut;
                    candidate->reshapeNode     = reshapeNode;
                    return candidate;
                }
            }
        }
        if (existingCandidates.find(consumer) == existingCandidates.end() &&
            isNodeEligibleForStitching(consumer, mmeOut, nullptr))
        {
            candidate->nodeToStitch    = consumer;
            candidate->stitchedOperand = mmeOut;
            break;
        }
    }

    return candidate;
}

pBundle Bundlizer::removeBundle(const pNode& node)
{
    pBundle ret = findBundleByNode(node);
    if (ret)
    {
        auto bundleNodes = ret->getNodes();
        for (const pNode& n : bundleNodes)
        {
            m_nodeBundleMap.erase(n);
            n->getNodeAnnotation().bundleInfo.unset();
            ret->removeNode(n);
        }
        m_graph.getBPGraph()->removeBundle(node);
    }
    return ret;
}

bool Bundlizer::strategyIsConsumerStitchingCapable(const pMmeSlicingStrategy& strategy,
                                                   const pSlicedOperand&      consumed) const
{
    // Check if the strategy somehow added the output to SRAM anyway and is double buffered
    if (consumed->resideInSRAM &&
        (strategy->getMetrics().isDoubleBuffered ||
         SlicedOperandUtils::isTriviallySliced(consumed)))
    {
        return true;
    }

    // If not, check that there is room for double buffered output in SRAM
    uint64_t sramCapacityAfterSlicing =
        SlicingBrain::knobs.maxSRAMCapInBytes - strategy->getMetrics().SRAMCapacity;
    uint64_t singleOutputSlice = SlicedOperandUtils::getSliceSizeInBytes(consumed);
    unsigned numOfOutputBuffers =
            SlicedOperandUtils::isTriviallySliced(consumed) ? 1 : 2;
    return (sramCapacityAfterSlicing >= numOfOutputBuffers * singleOutputSlice);
}

pNode Bundlizer::findNonReshapeProducer(const pTensor& tensor, pNode& reshapeNode) const
{
    pNode producer = m_graph.getTensorProducer(tensor);
    if (producer && producer->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
    {
        // Skip the reshape for now
        reshapeNode = producer;
        producer    = m_graph.getTensorProducer(producer->getInput(0));
    }
    return producer;
}

bool Bundlizer::canMMEInputConsumerBeAddedToBundle(const pBundleExpansion& candidate,
                                                   const pMmeSlicingStrategy& strategy) const
{
    return SharedMMEInputCandidateHandler::isCandidateValidForStrategy(candidate, strategy);
}

class BundlizerChainExpansionChecker : public BaseBundleChainExpansionChecker
{
private:
    const pSlicedOperand&   m_operand;

public:
    BundlizerChainExpansionChecker(const NodePtr& node, const pSlicedOperand& operand, Dim dim)
    : BaseBundleChainExpansionChecker(node, operand, {dim}), m_operand(operand) {};
    virtual ~BundlizerChainExpansionChecker() = default;
    const ExpansionBreakerContainer getExpansionBreakers() const override
    {
        ExpansionBreakerContainer expansionBreakers;

        expansionBreakers.push_back(
            std::make_unique<GranularityMultipleExpansionBreaker>(m_candidateNode, m_operand, m_slicingDims));
        expansionBreakers.push_back(
            std::make_unique<OverlapExpansionBreaker>(m_candidateNode, m_connectingTensor, m_slicingDims));
        expansionBreakers.push_back(
            std::make_unique<OffsetExpansionBreaker>(m_candidateNode, m_connectingTensor, m_slicingDims));
        expansionBreakers.push_back(std::make_unique<NonStichedOffsetOverlapExpansionBreaker>(
            m_candidateNode,
            m_connectingTensor,
            m_slicingDims));  // Block stitching of TPC nodes with offset/overlap on the non-stitched operands until it
                              // will be supported (TODO: SW-99608)

        return expansionBreakers;
    }
};

bool Bundlizer::isOperandSlicingValid(const NodePtr& node, const pSlicedOperand& operand)
{
    gc::access_pattern::NodeAccessPatternPtr accessPattern = node->getNodeAccessPattern();
    HB_ASSERT_PTR(accessPattern);
    const TensorTile&       granularity = accessPattern->getTensorGranularity(operand->originalTensor);
    const IntersectionTile& overlap     = accessPattern->getTensorOverlap(operand->originalTensor);
    const DimVector&        slicedDims  = SlicedOperandUtils::getSlicedDims(operand);
    for (const auto& slicedDim : slicedDims)
    {
        BundlizerChainExpansionChecker chainExpansionChecker(node, operand, slicedDim);
        if (chainExpansionChecker.isChainBreaker())
        {
            return false;
        }
    }

    return true;
}

pSlicedOperand Bundlizer::getNextStitchedOperand(const NodePtr&        reshapeNode,
                                                 const pSlicedOperand& stitchedOperand) const
{
    // Project the slicing of the stitched operand (the tensor between the MME and the reshape node)
    // to the next tensor in the chain (the tensor between the reshape and the TPC node).

    AccessPatternNodeSolutionProjector projector(reshapeNode);
    const auto& reshapeNodeStrategy = projector.getNodeStrategy({stitchedOperand}, stitchedOperand->originalTensor);
    const auto&                        nodeSlicingData     = reshapeNodeStrategy->getSlicingData();
    const auto&                        reshapeInputs       = reshapeNode->getInputs();

    pSlicedOperand nextStitchedOperand;
    if (std::find(reshapeInputs.begin(), reshapeInputs.end(), stitchedOperand->originalTensor) != reshapeInputs.end())
    {
        // MME -> Reshape -> TPC
        nextStitchedOperand = nodeSlicingData.getSlicedOperand(reshapeNode->getOutput(0));
    }
    else
    {
        // TPC -> Reshape -> MME
        nextStitchedOperand = nodeSlicingData.getSlicedOperand(reshapeNode->getInput(0));
    }
    HB_ASSERT_PTR(nextStitchedOperand);

    return nextStitchedOperand;
}

bool Bundlizer::isNodeAllowedForSlicing(const TPCNodePtr&     node,
                                        const pSlicedOperand& stitchedOperand,
                                        const NodePtr&        reshapeNode) const
{
    if (GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value())
    {
        // TODO [SW-10295] TPC stitching - can't move reshapes when tpc node is only seperable on some dimensions
        bool isSeparable = node->isSeparable(deviceTypeToDeviceID(m_graph.getDeviceType()));
        if (!isSeparable)
        {
            SLC_DEBUG("MME operand is sliced, but the candidate to stitch to"
                      " this operand ({}) is not separable",
                      node->getNodeName());
        }
        if (!node->isAllowedForStitching(m_graph))
        {
            SLC_DEBUG("TPC kernel is not in the allowed for stitching list");
        }
        if (isSeparable && !node->isAllowedForStitching(m_graph))
        {
            SLC_WARN("TPC kernel can be stitched but it is not in the allowed for stitching list, Node: {}, GUID: {}",
                     node->getNodeName(),
                     std::string(node->getGUIDWithoutDtype()));
        }
        return isSeparable && node->isAllowedForStitching(m_graph);
    }
    else
    {
        pSlicedOperand connectingOperand = stitchedOperand;
        if (reshapeNode)
        {
            if (!isOperandSlicingValid(reshapeNode, connectingOperand))
            {
                SLC_DEBUG(
                    "Reshape node {} can't be stitched to operand {} - slicing is invalid according to index-space",
                    reshapeNode->getNodeName(),
                    connectingOperand->originalTensor->getName());
                return false;
            }
            connectingOperand = getNextStitchedOperand(reshapeNode, connectingOperand);
        }
        if (!isOperandSlicingValid(node, connectingOperand))
        {
            SLC_DEBUG("TPC node {} can't be stitched to operand {} - slicing is invalid according to index-space",
                      node->getNodeName(),
                      connectingOperand->originalTensor->getName());
            return false;
        }
        const auto& memoryCoherence = m_graph.getGraphAnnotation().memoryCoherence;
        if (memoryCoherence && memoryCoherence->overlapsWithOthersInSection(connectingOperand->originalTensor))
        {
            SLC_WARN("TPC node {} can't be stitched to operand {} since it has multiple producers in section",
                     node->getNodeName(),
                     connectingOperand->originalTensor->getName());
            return false;  // For details see SW-45973
        }

        return true;
    }
}

bool Bundlizer::isNodeEligibleForStitchingInternal(const TPCNodePtr&     node,
                                                   const pSlicedOperand& stitchedOperand,
                                                   const NodePtr&        reshapeNode) const
{
    bool stitchReshape = reshapeNode != nullptr;
    SLC_DEBUG("isNodeEligibleForStitching {}, {}", node->getNodeName(), stitchedOperand->toString());
    if (m_nodeBundleMap.count(node) != 0)
    {
        SLC_DEBUG("Candidate is already in a bundle.");
        return false;
    }
    if (stitchedOperand->finalElementType != stitchedOperand->originalTensor->getElementType())
    {
        SLC_DEBUG("MME operand is reduction. Final element type type is different than original");
        return false;
    }
    if (!GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value() && SlicedOperandUtils::isOperandFlattened(stitchedOperand))
    {
        SLC_TRACE("TPC node {} can't be stitched to flattened operand {}",
                  node->getNodeName(),
                  stitchedOperand->originalTensor->getName());
        return false;
    }

    if (node->hasHighRankOperand())  // TODO: SW-120808 - high rank support
    {
        SLC_WARN("Node {} (type {}) has high rank operand - bundling is not supported",
                 node->getNodeName(),
                 node->getNodeTypeStr());
        return false;
    }

    // In case any of the node's outputs has a binding inplace request, stitching is not allowed.
    // If the output is marked as alias to the input before slicing, and since the slicer is not
    // aware of aliasing, it can undo the inplace reuse. In other cases it can change the location of the output tensor
    // (sram/hbm) and cause wrong behavior of the bundle.
    if (node->hasBindingInputReuse())
    {
        SLC_WARN("Node {} requires inplace input reuse, stitching not allowed", node->getNodeName());
        return false;
    }

    // Wide mme input is sliced to multiple chunks => TPC would need to be sliced as well,
    // which is only possible for TPC nodes that are separable on the sliced axis (for now)
    if (SlicedOperandUtils::isTriviallySliced(stitchedOperand))
    {
        SLC_DEBUG("Candidate is trivially sliced");
        return true;
    }
    else
    {
        // No need to align reshape when slicing according to index-space
        if (GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value() &&
            (SlicedOperandUtils::isOperandFlattened(stitchedOperand) || stitchReshape))
        {  // Reshaped operand and non-trivial slicing, means the reshape will be aligned
            if (!GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT.value())
            {
                SLC_TRACE("Disqualifying {} since reshape alignment is required and is globally disabled.",
                          node->getNodeName());
                return false;
            }
            if (node->isFusedKernel())
            {
                // TODO [SW-36266] : When the TPC-fuser support re-compiling based on slice shapes, this limitation can
                // be lifted See: SW-29461
                SLC_DEBUG("Can't reshape fused kernel {}, so can't stitch it to reshaped and sliced operand: {}",
                          node->getNodeName(),
                          stitchedOperand->originalTensor->getName());
                return false;
            }
        }
        // [SW-57121] When user reshapes BN on C dimension, the kernel can't be stitched.
        if (!isBnStitchingAllowed(node, stitchedOperand, reshapeNode))
        {
            return false;
        }

        return isNodeAllowedForSlicing(node, stitchedOperand, reshapeNode);
    }
}

bool Bundlizer::isNodeEligibleForStitching(const NodePtr&        node,
                                           const pSlicedOperand& stitchedOperand,
                                           const NodePtr&        reshapeNode) const
{
    TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
    if (tpcNode == nullptr)
    {
        SLC_DEBUG("Either no candidate or candidate is not TPC.");
        return false;
    }

    bool ret = isNodeEligibleForStitchingInternal(tpcNode, stitchedOperand, reshapeNode);

    SLC_DEBUG("Node: {}, GUID: {}, {} eligible for stitching",
              tpcNode->getNodeName(),
              std::string(tpcNode->getGUIDWithoutDtype()),
              (ret ? "is" : "is not"));

    return ret;
}

void Bundlizer::logGraphBundlingStatus(const BundleSolvingDataMap& solvingDataMap)
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
        const auto    bundleIter = m_nodeBundleMap.find(node);
        const pBundle nodeBundle = (bundleIter == m_nodeBundleMap.end() ? nullptr : bundleIter->second);
        std::string   bundleId   = nodeBundle ? std::to_string(nodeBundle->index()) : "N/A";
        std::string solved = nodeBundle ? (solvingDataMap.at(nodeBundle).strategies.empty() ? "Unsolved" : "Solved")
                                        : "N/A";
        SLC_INFO("Bundle-ID: {:>3} ({:<10}) Node: {} [{}]",
                 bundleId,
                 solved,
                 node->getNodeName(),
                 node->getNodeTypeStr());
    }
}

pBundle Bundlizer::findBundleByNode(const pNode& node) const
{
    pBundle ret;
    auto bundleIter = m_nodeBundleMap.find(node);
    if (bundleIter != m_nodeBundleMap.end())
    {
        ret = bundleIter->second;
    }
    return ret;
}

bool Bundlizer::isBnStitchingAllowed(const TPCNodePtr&     node,
                                     const pSlicedOperand& stitchedOperand,
                                     const NodePtr&        reshapeNode) const
{
    constexpr unsigned int BatchNormDimensions = 4;
    // This entire function can be removed when we disable reshape alignment by default.
    if (!GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT.value())
    {
        return true;
    }
    if (node->getGUIDWithoutDtype().find("batch_norm") != std::string::npos)
    {
        // Todo: [SW-70996] >4D restriction  can be removed when tpc kernels add 5D bn support [SW-70992].
        if (stitchedOperand->originalTensor->getDim() != BatchNormDimensions)
        {
            SLC_TRACE("Disqualifying {} since batch norm doesn't support non 4D tensors.", node->getNodeName());
            return false;
        }

        if (reshapeNode != nullptr)
        {
            if (reshapeNode->getInput(0)->getSizeInElements(DIM_C) !=
                reshapeNode->getOutput(0)->getSizeInElements(DIM_C))
            {
                SLC_TRACE("Disqualifying {} since reshape changes batch norm channels.", node->getNodeName());
                return false;
            }
        }
        if (stitchedOperand->finalShape[DIM_C] != stitchedOperand->originalTensor->getSizeInElements(DIM_C))
        {
            SLC_TRACE("Disqualifying {} since final shape will change batch norm channels.", node->getNodeName());
            return false;
        }
    }
    return true;
}