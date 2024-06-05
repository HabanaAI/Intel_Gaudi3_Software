#include "compilation_hal_reader.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "habana_pass.h"
#include "log_manager.h"
#include "node.h"
#include "shape_node.h"
#include "tensor.h"
#include "tpc_kernel_names.h"
#include "graph_editor.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "types.h"
#include <algorithm>
#include <memory>
#include "fuse_broadcast.h"

void fuseConstantBroadcastConsumer(HabanaGraph& g, TPCNodePtr& tpcNode)
{
    auto constOut = tpcNode->getOutput(0);
    // No fusing if output of const is in user managed DRAM.
    if (constOut->isUserManagedDram())
    {
        LOG_TRACE(FUSE_BROADCAST,
                  "Skipping fusion for node {}, since its output tensor is in user managed DRAM",
                  tpcNode->getNodeName());
        return;
    }
    NodeList constConsumers = g.getTensorConsumers(constOut);
    // No fusing if there are other consumers.
    if (constConsumers.size() != 1)
    {
        LOG_TRACE(FUSE_BROADCAST,
                  "Skipping fusion for node {}, since its output tensor has more than one consumer",
                  tpcNode->getNodeName());
        return;
    }
    if (constConsumers.front()->getNodeType() == Node::TYPE_BROADCAST)
    {
        auto broadCastNode = constConsumers.front();
        // Now we can fuse.
        std::string newNodeName = tpcNode->getNodeName();
        newNodeName += "_fused_broadcast_";
        newNodeName += broadCastNode->getNodeName();
        bool constByDataTensor =
            tpcNode->getNumInputs(Node::TENSOR_TYPE_DATA) > 0 && !tpcNode->getInput(0)->isShapeTensor();
        TensorVector inputs;
        NodePtr      fusedBroadcast;
        UserParams   params     = nullptr;
        unsigned     paramsSize = 0;
        if (constByDataTensor)
        {
            inputs.push_back(tpcNode->getInput(0));
        }
        // Const by params
        else
        {
            params     = tpcNode->getParams();
            paramsSize = tpcNode->getParamsSize();
        }
        // If we have DS we expect shape tensor to come from 2nd Bcast input
        if (broadCastNode->getNumInputs() > 1)
        {
            HB_ASSERT(broadCastNode->getInput(1)->isShapeTensor(),
                      "Expected shape tensor for node {}",
                      broadCastNode->getNodeName());
            inputs.push_back(broadCastNode->getInput(1));
        }
        // For static shapes, The shape input of the tpc node is the shape tensor
        else if (tpcNode->getNumInputs() > 0)
        {
            auto shapeTensor = tpcNode->getInputs().back();
            HB_ASSERT(shapeTensor->isShapeTensor(), "Expected shape tensor for node {}", tpcNode->getNodeName());
            inputs.push_back(shapeTensor);
        }

        auto newNode = NodeFactory::createNode(inputs,
                                               {broadCastNode->getOutput(0)},
                                               params,
                                               paramsSize,
                                               tpcNode->getGUID(),
                                               newNodeName);

        GraphEditor::replaceNodes(g, {tpcNode, broadCastNode}, {newNode});
    }
}

static std::vector<unsigned> getBroadcastedDimensions(const NodePtr& broadcastNode)
{
    std::vector<unsigned> broadcastedDims;
    auto                  broadcastIn  = broadcastNode->getInput(0);
    auto                  broadcastOut = broadcastNode->getOutput(0);
    for (unsigned dim = 0; dim < broadcastNode->getOutput(0)->getDim(); dim++)
    {
        if (broadcastIn->getSizeInElements(dim) != broadcastOut->getSizeInElements(dim))
        {
            broadcastedDims.push_back(dim);
        }
    }
    return broadcastedDims;
}

// Validation methods for producers that can be part of broadcast chain.
static bool validateOnlyOneOutput(const NodePtr& producer)
{
    return (producer->getOutputs().size() == 1);
}
static bool validateOutputIsNotPersistent(const NodePtr& producer)
{
    return !producer->getOutput(0)->isUserManagedDram();
}
static bool validateOnlyOneConsumer(const HabanaGraph& g, const NodePtr& producer)
{
    return g.getNumberOfTensorConsumers(producer->getOutput(0)) == 1;
}

static bool validateProducer(const HabanaGraph& g, const NodePtr& producer)
{
    // Currently, only max sizes are being considered in the pass's validation / edit methods.
    if (producer->isDynamicShape()) return false;

    if (!validateOnlyOneOutput(producer)) return false;
    if (!validateOutputIsNotPersistent(producer))
    {
        LOG_TRACE(FUSE_BROADCAST,
                  "Skipping broadcast fusion since {} output tensor is in user managed DRAM",
                  producer->getNodeName());
        return false;
    }
    if (!validateOnlyOneConsumer(g, producer))
    {
        LOG_TRACE(FUSE_BROADCAST,
                  "Skipping broadcast fusion since {} output tensor has more than one consumer",
                  producer->getNodeName());
        return false;
    }
    return true;
}

static bool validateSourceNode(const NodePtr& sourceNode)
{
    // Currently, only max sizes are being considered in the pass's validation / edit methods.
    if (sourceNode->isDynamicShape()) return false;
    return true;
}

bool TpcBroadcastChainHandler::doesTensorExistsMoreThanOnce(const BroadcastChain& chain)
{
    unsigned    sourceNodeInputIndex = chain.sourceNodeInputIndex;
    const auto& allInputs            = chain.sourceNode->getInputs();
    return std::count(allInputs.begin(), allInputs.end(), chain.sourceNode->getInput(sourceNodeInputIndex)) > 1;
}

// Check that all other sourceNode inputs have non degenerated dims in the broadcasted dims.
bool BroadcastChainHandler::validSizesForFusion(const BroadcastChain& chain)
{
    const auto& sourceNode           = chain.getSourceNode();
    unsigned    sourceNodeInputIndex = chain.sourceNodeInputIndex;
    const auto& broadcastNode        = chain.getBroadcastNode();
    const auto  broadcastedDims      = getBroadcastedDimensions(broadcastNode);
    // check that at least one of the inputs (other than the one at index) has size != 1 in broadcasted dim
    if (!broadcastedDims.empty())
    {
        for (unsigned i = 0; i < sourceNode->getInputs().size(); i++)
        {
            if (i == sourceNodeInputIndex) continue;
            bool inputValidForFusion =
                std::all_of(broadcastedDims.begin(), broadcastedDims.end(), [&sourceNode, &i](unsigned dim) {
                    return sourceNode->getInput(i)->getSizeInElements(dim) != 1;
                });
            if (!inputValidForFusion)
            {
                LOG_TRACE(FUSE_BROADCAST,
                          "Skipping fusion for node {}, fusion is not valid",
                          sourceNode->getNodeName());
                return false;
            }
        }
    }
    return true;
}

bool TpcBroadcastChainHandler::validUtilizationForFusion(const BroadcastChain& chain)
{
    // Avoid fusion if tpc utilization is smaller than the threshold
    auto  tpcVectorSize = CompilationHalReader::getHalReader()->getTpcVectorSize();
    auto  broadcastedIn = chain.sourceNode->getInput(chain.sourceNodeInputIndex);
    auto  fcdInBytes    = broadcastedIn->getSizeInBytes(0);
    float sizeAligned   = std::ceil((float)fcdInBytes / tpcVectorSize) * tpcVectorSize;
    float utilThreshold = GCFG_MAX_TPC_VEC_UTIL_FOR_BROADCAST_TPC_FUSION.value();
    return (fcdInBytes / sizeAligned) > utilThreshold;
}

static bool isReshapeOnlyOnBatchDims(const NodePtr& reshape)
{
    // Check that all reshape operands have batch dims.
    if (reshape->getInput(0)->getDim() <= DIM_GEMM_BATCH || reshape->getOutput(0)->getDim() <= DIM_GEMM_BATCH)
    {
        return false;
    }
    // Check that the spatial dims are equal.
    for (unsigned i = 0; i < DIM_GEMM_BATCH; i++)
    {
        if (reshape->getInput(0)->getSizeInElements(i) != reshape->getOutput(0)->getSizeInElements(i))
        {
            return false;
        }
    }
    return true;
}

bool TpcBroadcastChainHandler::producerCanBeHandled(const NodePtr& producer)
{
    return (producer->getNodeType() == Node::TYPE_BROADCAST) ||
           BroadcastChainHandler::canProducerPropagateBroadcastRemoval(producer);
}

bool BroadcastChainExtendedOperator::isReshapeSupportsReplacement(const HabanaGraph&                    g,
                                                                  const std::shared_ptr<BatchGemmNode>& sourceNode,
                                                                  const NodePtr&                        reshape)
{
    // Currently only reshape on the batch dim is supported.
    if (!isReshapeOnlyOnBatchDims(reshape)) return false;

    const auto& reshapeConsumers = g.getTensorConsumers(reshape->getOutput(0));
    // Check that:
    // a. Reshape is a direct producer to the bgemm.
    // b. Bgemm is symmetric layout. It assures that an inverse reshape can be inserted for the other operand.
    // TODO [SW-170089]: remove isSymmetricLayout.
    return *reshapeConsumers.begin() == sourceNode && sourceNode->isSymmetricLayout();
}

bool BgemmBroadcastChainFusionHandler::canProducerBeRemovedFromBroadcastChain(
    const HabanaGraph&                    g,
    const std::shared_ptr<BatchGemmNode>& sourceNode,
    const NodePtr&                        producer)
{
    // Currenlty we support only one reshape in chain.
    return (isLogicalReshape(producer) &&
            BroadcastChainExtendedOperator::isReshapeSupportsReplacement(g, sourceNode, producer));
}

bool BroadcastChainHandler::canProducerPropagateBroadcastRemoval(const NodePtr& producer)
{
    // For now supporting only cast nodes
    return producer->isCast();
}

bool BgemmBroadcastChainFusionHandler::producerCanBeHandled(const HabanaGraph&                    g,
                                                            const std::shared_ptr<BatchGemmNode>& sourceNode,
                                                            const NodePtr&                        producer)
{
    // We support fusing broadcast to bgemm only if the broadcast dims are batch dims
    if (producer->getNodeType() == Node::TYPE_BROADCAST)
    {
        const auto& broadcastedDims = getBroadcastedDimensions(producer);
        return std::all_of(broadcastedDims.begin(), broadcastedDims.end(), [](unsigned broadcastDim) {
            return broadcastDim >= DIM_GEMM_BATCH;
        });
    }
    if (canProducerPropagateBroadcastRemoval(producer)) return true;
    return canProducerBeRemovedFromBroadcastChain(g, sourceNode, producer);
}

void BroadcastChainOperator::propagateBroadcast(NodePtr& prod, const std::vector<unsigned>& broadcastedDims)
{
    if (prod->isCast())
    {
        // Resize the cast output to be not broadcasted - degenerate all broadcasted dims
        // This will implicitly modify the input for the next node in the chain
        TensorPtr  castOutput = prod->getOutput(0);
        NSizeArray sizes      = castOutput->getNSizesInElements();
        for (unsigned dim : broadcastedDims)
        {
            sizes[dim] = 1;
        }
        castOutput->reshape(castOutput->getDim(), sizes.data(), nullptr);
        castOutput->setName(castOutput->getName() + "_without_broadcast");
    }
    else
    {
        HB_ASSERT(false, "Unsupported node type for broadcast propagation");
    }
}

void PropagateBroadcastAfterSourceNode::resizeNodeOutput(HabanaGraph& g, const NodePtr& node)
{
    TensorPtr  out      = node->getOutput(0)->clone(false, false, false);
    NSizeArray maxSizes = {0};
    NSizeArray minSizes = {0};
    unsigned   newDim   = 1;
    for (const TensorPtr& in : node->getInputs())
    {
        if (!in) continue;
        newDim = std::max(in->getDim(), newDim);
    }
    for (const TensorPtr& in : node->getInputs())
    {
        if (!in) continue;
        for (unsigned dim = 0; dim < newDim; dim++)
        {
            TSize inputDimMaxSize = in->getDim() < dim ? 1 : in->getSizeInElements(dim);
            maxSizes[dim]         = std::max(maxSizes[dim], inputDimMaxSize);
            TSize inputDimMinSize = in->getDim() < dim ? 1 : in->getMinimalSizeInElements(dim);
            minSizes[dim]         = std::max(minSizes[dim], inputDimMinSize);
        }
    }
    out->reshape(newDim, maxSizes.data(), nullptr, minSizes.data());
    GraphEditor::replaceOutput(g, node, 0, out);  // swap node and broadcast
}

bool BroadcastChainManager::validateChainNodes(const BroadcastChain& chain)
{
    for (const NodePtr& n : chain.getProducerChain())
    {
        if (n != m_sourceNode && n->isCast())
        {
            if (n->getInput(0)->getNSizesInElements() != n->getOutput(0)->getNSizesInElements())
            {
                LOG_ERR(FUSE_BROADCAST, "Incorrect tensor sizes after broadcast fusion/propagation");
                return false;
            }
        }
    }
    return true;
}

void FuseBroadcastToSourceNode::printOperationInfoToLog(const NodePtr& sourceNode, const NodePtr& broadcastNode)
{
    // Turn
    // [in0] -> broadcast -> [bcastOut0] -> intermediate nodes (cast) -> [out0] -> node
    // into:
    // [in0] -> intermediate nodes (cast) -> [out0_without_broadcast] -> node
    LOG_DEBUG(FUSE_BROADCAST, "Fusing broadcast: {} with: {}", broadcastNode->getNodeName(), sourceNode->getNodeName());
}

void PropagateBroadcastAfterSourceNode::printOperationInfoToLog(const NodePtr& sourceNode, const NodePtr& broadcastNode)
{
    // Turn
    // [in0] -> broadcast -> [bcastOut0] -> intermediate nodes (cast) -> [out0] -> node
    // into:
    // [in0] -> intermediate nodes (cast) -> [out0_without_broadcast] -> node -> [out0'] -> broadcast -> [out0]
    LOG_DEBUG(FUSE_BROADCAST,
              "Moving broadcast: {} after: {}",
              broadcastNode->getNodeName(),
              sourceNode->getNodeName());
}

bool FuseBroadcastToSourceNode::handleSourceNode(HabanaGraph& g, NodePtr& broadcastNode, NodePtr& sourceNode)
{
    // Assume node's input were already handled in broadcast propagation through the producer chain.
    GraphEditor::removeNode(g, broadcastNode);
    return true;
}

bool PropagateBroadcastAfterSourceNode::handleSourceNode(HabanaGraph& g, NodePtr& broadcastNode, NodePtr& sourceNode)
{
    TensorPtr newBcastOutput = sourceNode->getOutput(0);  // this will be the broadcast output
    resizeNodeOutput(g, sourceNode);                      // create a new output with fit sizes
    GraphEditor::editNode(g, broadcastNode, [&sourceNode, &newBcastOutput](const NodePtr& broadcastNode) {
        broadcastNode->replaceInput(0, sourceNode->getOutput(0));
        broadcastNode->replaceOutput(0, newBcastOutput);
    });
    return true;
}

void BroadcastChainOperator::propagateBroadcastRemovalUntilSourceNode(HabanaGraph&   g,
                                                                      NodeVector&    prodChainWithoutBroadcast,
                                                                      const NodePtr& broadcastNode,
                                                                      NodePtr&       sourceNode,
                                                                      unsigned       sourceNodeInputIndex)
{
    // First, set the broadcast consumer. In case there's a producer chain - it is the last added node.
    NodePtr bcastConsumer = prodChainWithoutBroadcast.empty() ? sourceNode : prodChainWithoutBroadcast.back();
    // Set the input index. In case of a chain - the simple chain checks only input 0 of the nodes
    unsigned inputIndex = prodChainWithoutBroadcast.empty() ? sourceNodeInputIndex : 0;
    GraphEditor::replaceInput(g, bcastConsumer, inputIndex, broadcastNode->getInput(0));

    // Second, propagate the broadcast through the intermediate nodes between the broadcast and the node.
    // The chain is assumed to have a 1:1 mapping of input:output dims (correct for cast).
    // Otherwise the broadcast dims required propagation through access pattern
    for (auto& prod : prodChainWithoutBroadcast)
    {
        LOG_DEBUG(FUSE_BROADCAST, "{}: propagate broadcast through {}", HLLOG_FUNC, prod->getNodeName());
        // Resize the producer output to be not broadcasted according to the new input sizes
        propagateBroadcast(prod, getBroadcastedDimensions(broadcastNode));
    }
}

NodePtr BroadcastChainExtendedOperator::createLogicalReshape(const TensorPtr& sourceTensor,
                                                             const TensorPtr& targetTensor)
{
    return NodeFactory::createInternalNode({sourceTensor},
                                           {targetTensor},
                                           nullptr,
                                           NodeFactory::reshapeNodeTypeName,
                                           fmt::format("{}_reshape", sourceTensor->getName()));
}

bool BroadcastChainOperator::validateChainForOperation(const HabanaGraph& g, const BroadcastChain& chain)
{
    // Validate outputs of producer chain are not persistent because they are edited
    bool allValidated = std::all_of(chain.getProducerChain().begin(),
                                    chain.getProducerChain().end(),
                                    [&g](const NodePtr& producer) { return validateProducer(g, producer); });
    return allValidated && chain.size() >= 2 && chain.getBroadcastNode();
}

bool BroadcastChainOperator::handleOperation(HabanaGraph& g, BroadcastChain& chain)
{
    HB_ASSERT_DEBUG_ONLY(validateChainForOperation(g, chain), "Expecting that broadcast chain validation succeeded");
    NodeVector prodChainWithoutBroadcast(
        chain.getProducerChain().begin(),
        std::next(chain.getProducerChain().begin(), chain.getProducerChain().size() - 1));
    printOperationInfoToLog(chain);

    // Handle broadcast chain in two steps:
    // 1. Propagate broadcast removal until source Node:
    propagateBroadcastRemovalUntilSourceNode(g,
                                             prodChainWithoutBroadcast,
                                             chain.getBroadcastNode(),
                                             chain.sourceNode,
                                             chain.sourceNodeInputIndex);

    // 2. Handle the source node (fuse it to broadcast / propagate broadcast)
    return m_sourceNodeOperator->handleSourceNode(g, chain.getBroadcastNode(), chain.sourceNode);
}

static unsigned countLogicalReshapes(const NodeVector& chainOfNodes)
{
    const auto reshapeCount = std::count_if(chainOfNodes.begin(), chainOfNodes.end(), [](const NodePtr& prod) {
        return isLogicalReshape(prod);
    });
    return reshapeCount;
}

static void setBatchDimsOfTensor(SizeArray& currSizes, const TensorPtr& tensor)
{
    for (int i = DIM_GEMM_BATCH; i < tensor->getDim(); ++i)
    {
        currSizes[i] = tensor->getAllSizesInElements()[i];
    }
}

NodePtr BroadcastChainExtendedOperator::createInverseReshapeForOtherOperand(const NodePtr& origReshape,
                                                                            const NodePtr& sourceNode,
                                                                            unsigned       otherOperandIndex)
{
    const auto& otherOperand              = sourceNode->getInput(otherOperandIndex);
    SizeArray   reshapedOtherOperandSizes = otherOperand->getAllSizesInElements();
    setBatchDimsOfTensor(reshapedOtherOperandSizes, origReshape->getInput(0));
    TensorPtr reshapedOtherOperand = otherOperand->clone(false, false, false);
    reshapedOtherOperand->reshape(origReshape->getInput(0)->getDim(), reshapedOtherOperandSizes.data());
    return createLogicalReshape(otherOperand, reshapedOtherOperand);
}

NodePtr BroadcastChainExtendedOperator::createReshapeForOutputOperand(const NodePtr& origReshape,
                                                                      const NodePtr& sourceNode)
{
    SizeArray newOutputSizes = sourceNode->getOutput(0)->getAllSizesInElements();
    setBatchDimsOfTensor(newOutputSizes, origReshape->getInput(0));
    const auto& currOutput = sourceNode->getOutput(0);
    const auto  newOutput  = currOutput->clone(false, false, false);
    newOutput->reshape(origReshape->getInput(0)->getDim(), newOutputSizes.data());
    return createLogicalReshape(newOutput, currOutput);
}

bool BroadcastChainExtendedOperator::validateChainForOperation(const HabanaGraph&    g,
                                                               const BroadcastChain& chain,
                                                               const NodePtr&        reshape)
{
    const auto& bgemm = std::dynamic_pointer_cast<BatchGemmNode>(chain.getSourceNode());
    return bgemm && validateProducer(g, reshape) && isLogicalReshape(reshape) &&
           BroadcastChainExtendedOperator::isReshapeSupportsReplacement(g, bgemm, reshape);
}

bool BroadcastChainExtendedOperator::handleOperation(HabanaGraph& g, BroadcastChain& chain)
{
    auto reshapeCount = countLogicalReshapes(chain.getProducerChain());
    HB_ASSERT(reshapeCount <= 1, "Expecting that there is only one reshape in the broadcast chain");
    if (reshapeCount == 1)
    {
        const auto& origReshapeIt = std::find_if(chain.getProducerChain().begin(),
                                                 chain.getProducerChain().end(),
                                                 [](const NodePtr& prod) { return isLogicalReshape(prod); });
        HB_ASSERT(origReshapeIt != chain.getProducerChain().end(),
                  "Expecting that there is a reshape in the producer chain");

        const auto& origReshape = *origReshapeIt;
        HB_ASSERT_DEBUG_ONLY(validateChainForOperation(g, chain, origReshape),
                             "Expecting that chain validation is succeeded");

        // Move the reshape out of the producer chain.
        // To keep the subgraph correctness we need to add the following reshapes:
        // 1. An inverse reshape for the bgemm's other operand.
        // 2. Similar reshape for the bgemm output operand.

        // 1.
        auto&       sourceNode = chain.getSourceNode();
        const auto& reshapeOtherOperand =
            createInverseReshapeForOtherOperand(origReshape, sourceNode, 1 - chain.sourceNodeInputIndex);

        // 2.
        const auto& outputReshape = createReshapeForOutputOperand(origReshape, sourceNode);

        // Edit the bgemm
        GraphEditor::editNode(g,
                              sourceNode,
                              [&chain, &origReshape, &reshapeOtherOperand, &outputReshape](const NodePtr& sourceNode) {
                                  sourceNode->replaceInput(chain.sourceNodeInputIndex, origReshape->getInput(0));
                                  sourceNode->replaceInput(1 - chain.sourceNodeInputIndex,
                                                           reshapeOtherOperand->getOutput(0));
                                  sourceNode->replaceOutput(0, outputReshape->getInput(0));
                              });

        printOperationInfoToLog(chain);
        GraphEditor::replaceNodes(g, {origReshape}, {reshapeOtherOperand, outputReshape});

        chain.getProducerChain().erase(origReshapeIt);
    }

    return BroadcastChainOperator::handleOperation(g, chain);
}

// Returns chain of producers ordered from the given source node's producer to broadcast node.
// Searches for broadcast as long as the producers chain is simple and supports broadcast removal (this is checked with
// some general validation rules (validateProducer)). For the first producer we use sourceNodeInputIndex. For the rest
// of the chain it would be input 0.
NodeVector BroadcastChainFinder::getProducerChain(unsigned sourceNodeInputIndex)
{
    NodeVector producersChain;
    auto       producer = m_graph.getTensorProducer(m_sourceNode->getInput(sourceNodeInputIndex));
    while (producer && validateProducer(m_graph, producer) &&
           producersChain.size() <= GCFG_BGEMM_PRODUCER_CHAIN_MAX_SIZE_FOR_FUSING_WITH_BROADCAST.value())
    {
        producersChain.push_back(producer);
        if (producer->getNodeType() == Node::TYPE_BROADCAST)
        {
            // Reached a broadcast with a valid chain - return it as the last node in producersChain
            return producersChain;
        }
        // Get the next producer in the simple chain (on input 0, not searching all inputs - simple chain)
        producer = m_graph.getTensorProducer(producer->getInput(0));
    }
    return {};
}

BroadcastChains BroadcastChainFinder::getChains()
{
    BroadcastChains broadcastChains;
    for (unsigned i = 0; i < m_sourceNode->getNumInputs(); ++i)
    {
        const auto& prodChain = getProducerChain(i);
        if (!prodChain.empty())
        {
            BroadcastChain c {.producerChain = prodChain, .sourceNode = m_sourceNode, .sourceNodeInputIndex = i};
            broadcastChains.push_back(c);
        }
    }
    return broadcastChains;
}

bool BroadcastChainHandler::handleChain()
{
    m_chain.printChain();
    // Return true if chain was handled successfully
    return m_chainOperator->handleOperation(m_graph, m_chain);
}

bool BroadcastChainHandler::canHandle(const HabanaGraph& g, const BroadcastChain& chain)
{
    return std::all_of(chain.getProducerChain().begin(),
                       chain.getProducerChain().end(),
                       [&g](const NodePtr& producer) { return validateProducer(g, producer); }) &&
           chain.size() >= 2 && chain.getBroadcastNode()->getNodeType() == Node::TYPE_BROADCAST;
}

bool TpcBroadcastChainHandler::canHandle(const HabanaGraph& g, const BroadcastChain& chain)
{
    return BroadcastChainHandler::canHandle(g, chain) &&
           std::all_of(chain.getProducerChain().begin(),
                       chain.getProducerChain().end(),
                       TpcBroadcastChainHandler::producerCanBeHandled) &&
           validUtilizationForFusion(chain) && !doesTensorExistsMoreThanOnce(chain);
};

bool TpcBroadcastChainFusionHandler::canHandle(const HabanaGraph& g, const BroadcastChain& chain)
{
    return TpcBroadcastChainHandler::canHandle(g, chain) && BroadcastChainHandler::validSizesForFusion(chain);
};

bool TpcBroadcastChainPropagationHandler::canHandle(const HabanaGraph& g, const BroadcastChain& chain)
{
    if (TpcBroadcastChainHandler::canHandle(g, chain) && !BroadcastChainHandler::validSizesForFusion(chain))
    {
        // Check valid sizes for propagation:
        const auto& sourceNode    = chain.sourceNode;
        const auto& broadcastNode = chain.getBroadcastNode();
        // If the tpc output sizes are not the same as the broadcasted sizes, the moving optimization cannot be done
        // (need to create a new shape tensor for the broadcast)
        if (!broadcastNode->getOutput(0)->compareGeometry(*(sourceNode->getOutput(0))))
        {
            return false;
        }
        // No point of swapping because we will have to broadcast twice the data
        if (sourceNode->getNumOutputsDataTensors() > 1)
        {
            return false;
        }
        return true;
    }
    return false;
};

bool BgemmBroadcastChainFusionHandler::canHandle(const HabanaGraph& g, const BroadcastChain& chain)
{
    const auto& sourceNode = std::dynamic_pointer_cast<BatchGemmNode>(chain.getSourceNode());
    if (!sourceNode) return false;
    return BroadcastChainHandler::canHandle(g, chain) &&
           std::all_of(chain.getProducerChain().begin(),
                       chain.getProducerChain().end(),
                       [&g, &sourceNode](const NodePtr& producer) {
                           return BgemmBroadcastChainFusionHandler::producerCanBeHandled(g, sourceNode, producer);
                       })
           // If there is a reshape, the 1:1 dim mapping between broadcast dims and bgemm other op dims which is assumed
           // here: BroadcastChainHandler::validSizesForFusion isn't correct. And since bgemm symmetric layout was
           // checked earlier (producerCanBeHandled) the fusion is possible. If there isn't a reshape, bgemm symmetric
           // layout wasn't checked, so we need to check that the bgemm output sizes stay the same after broadcast
           // fusion (BroadcastChainHandler::validSizesForFusion).
           //
           // TODO [SW-170089]: when we use access pattern we should use one validation check no matter if the chain
           // contains reshape or not.
           && (chain.containsReshape() || BroadcastChainHandler::validSizesForFusion(chain));
};

BroadcastChainHandlerPtr TpcBroadcastChainFusionHandlerFactory::createForChain(HabanaGraph& g, BroadcastChain& chain)
{
    if (TpcBroadcastChainFusionHandler::canHandle(g, chain))
    {
        return std::make_shared<TpcBroadcastChainFusionHandler>(g, chain);
    }
    return nullptr;
}

BroadcastChainHandlerPtr TpcBroadcastChainPropagationHandlerFactory::createForChain(HabanaGraph&    g,
                                                                                    BroadcastChain& chain)
{
    if (TpcBroadcastChainPropagationHandler::canHandle(g, chain))
    {
        return std::make_shared<TpcBroadcastChainPropagationHandler>(g, chain);
    }
    return nullptr;
}

BroadcastChainHandlerPtr BgemmBroadcastChainFusionHandlerFactory::createForChain(HabanaGraph& g, BroadcastChain& chain)
{
    if (BgemmBroadcastChainFusionHandler::canHandle(g, chain))
    {
        return std::make_shared<BgemmBroadcastChainFusionHandler>(g, chain);
    }
    return nullptr;
}

BroadcastChainHandlersVector BroadcastChainManager::getHandlers(BroadcastChain& chain)
{
    return BroadcastChainHandlerSelector::selectHandlers(m_graph, chain, getFusionHandlersFactories());
}

bool BroadcastChainManager::optimizeChain(BroadcastChain& chain)
{
    const auto& handlers = getHandlers(chain);
    for (auto& handler : handlers)
    {
        if (handler->handleChain())
        {
            HB_ASSERT(validateChainNodes(chain), "Expecting that chain nodes pass validation");
            handler->turnOnPredicates();
            return true;
        }
    }
    return false;
}

void BroadcastChainManager::optimizeChains()
{
    auto chains = BroadcastChainFinder(m_graph, m_sourceNode).getChains();
    for (auto& chain : chains)
    {
        optimizeChain(chain);
    }
}

void BgemmBroadcastChainManager::optimizeChains()
{
    auto chains = BroadcastChainFinder(m_graph, m_sourceNode).getChains();
    for (auto& chain : chains)
    {
        if (optimizeChain(chain))
        {
            // Currently we support optimizing only one chain in bgemm, since
            // an optimization of one producer chain can change the other producer chain.
            return;
        }
    }
}

// fuse broadcast producer to tpc binary elementwise op, or broadcast consumer to constant kernel
bool fuseBroadcast(HabanaGraph& g)
{
    // creating a copy since the graph is modified in the loop
    NodeVector nodeVec = g.getTopoSortedNodes();
    for (auto& node : nodeVec)
    {
        if (!node || !validateSourceNode(node)) continue;
        if (HabanaGraph::runsOnTPC(node))
        {
            if (!GCFG_ENABLE_BROADCAST_TPC_FUSION.value()) return true;
            auto tpcNode = std::static_pointer_cast<TPCNode>(node);

            // TODO [SW-27229] Use gluecode query instead of isBroadcastableOperation
            if (tpcNode->isBroadcastableOperation())
            {
                TpcBroadcastChainManager(g, node).optimizeChains();
            }
            if (tpcNode->getGUIDWithoutDtype() == "constant")
            {
                fuseConstantBroadcastConsumer(g, tpcNode);
            }
        }
        else if (HabanaGraph::runsOnMME(node))
        {
            auto bgemm = std::dynamic_pointer_cast<BatchGemmNode>(node);
            if (GCFG_ENABLE_FUSE_BROADCAST_BGEMM.value() && bgemm)
            {
                BgemmBroadcastChainManager(g, node).optimizeChains();
            }
        }
    }
    return true;
}