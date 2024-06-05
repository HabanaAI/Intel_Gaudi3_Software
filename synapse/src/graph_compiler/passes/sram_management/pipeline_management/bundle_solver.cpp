#include "bundle_solver.h"
#include "access_pattern.h"
#include "bundle_slicer.h"
#include "conv_base_node.h"
#include "defs.h"
#include "habana_graph.h"
#include "log_manager.h"
#include "logical_op_node.h"
#include "node.h"
#include "slicing_utils.h"
#include "sram_management/bundle.h"
#include "sram_management/pipeline_management/node_solver.h"
#include "sram_management/slicing_brain.h"
#include "node_projector.h"
#include "common_tile_size_calculator.h"
#include "sram_management/solution_generator.h"
#include "sram_management/strategy_slicing_data.h"
#include "compilation_hal_reader.h"
#include "transpose_node.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include "handle_logical_operations.h"
#include "utils.h"
#include "mme_brain_proxy.h"

bool BundleSolver::isConnectingTensor(const TensorPtr& t, const std::unordered_set<TensorPtr>& connectingTensors) const
{
    return connectingTensors.find(t) != connectingTensors.end();
}

bool BundleSolver::isBundlePersistentInputTensor(const TensorPtr&                   t,
                                                 const std::unordered_set<NodePtr>& bundleNodes) const
{
    return bundleNodes.find(m_graph.getTensorProducer(t)) == bundleNodes.end();
}

NodeVector BundleSolver::getAdjacentBundleNodes(const TensorPtr&                   t,
                                                const std::unordered_set<NodePtr>& bundleNodes) const
{
    NodeVector adjacentBundleNodes {};

    const auto& tensorConsumers = m_graph.getTensorConsumers(t);
    adjacentBundleNodes.insert(adjacentBundleNodes.end(), tensorConsumers.begin(), tensorConsumers.end());
    adjacentBundleNodes.insert(adjacentBundleNodes.end(), m_graph.getTensorProducer(t));
    adjacentBundleNodes.erase(std::remove_if(adjacentBundleNodes.begin(),
                                             adjacentBundleNodes.end(),
                                             [&bundleNodes](const auto& node) {
                                                 const bool isBundleNode = bundleNodes.find(node) != bundleNodes.end();
                                                 return !isBundleNode;
                                             }),
                              adjacentBundleNodes.end());

    return adjacentBundleNodes;
}

std::unique_ptr<TPCExpansionsAndSingleMMEBundleSolver::ShouldIgnoreNodeInSramCapacityCalculation>
TPCExpansionsAndSingleMMEBundleSolver::ShouldIgnoreNodeInSramCapacityCalculation::getIgnoreNodeMethod()
{
    const auto& ignoreNodeMethod = GCFG_ENABLE_RELAXED_IGNORE_IN_SRAM_CAP_CALC.value()
                                       ? ShouldIgnoreNodeInSramCapacityCalculation::ignoreNonTransposeLogicalsRelaxed
                                       : ShouldIgnoreNodeInSramCapacityCalculation::ignoreNonTransposeLogicals;
    return std::make_unique<ShouldIgnoreNodeInSramCapacityCalculation>(ignoreNodeMethod);
}

bool TPCExpansionsAndSingleMMEBundleSolver::ShouldIgnoreNodeInSramCapacityCalculation::isNodeWithCtrlEdges(
    const NodePtr& n)
{
    const auto nControlEdges = n->getControlInputs().size() + n->getControlOutputs().size();
    return nControlEdges > 0;
}

NodeList TPCExpansionsAndSingleMMEBundleSolver::ShouldIgnoreNodeInSramCapacityCalculation::getLogicOpsSubChain(
    const NodePtr&       n,
    const PipelineChain& chain)
{
    NodeList subChain {};
    if (!n->isLogicalOperation()) return subChain;

    const auto nodeIt = std::find_if(chain.begin(), chain.end(), [&n](const auto& cn) { return cn.node == n; });
    if (nodeIt == chain.end())
    {
        LOG_WARN(SRAM_SLICE, "{}: Input node {} is not in input chain", HLLOG_FUNC, n->getNodeName());
        HB_ASSERT(chain.empty(),
                  "Expecting node {} to be in input pipeline chain or chain to be empty",
                  n->getNodeName());
        return subChain;
    }
    subChain.push_back(n);

    for (auto it = std::next(nodeIt); it < chain.end() && it->node->isLogicalOperation(); ++it)
    {
        subChain.push_back(it->node);
    }

    for (auto it = std::prev(nodeIt); it >= chain.begin() && it->node->isLogicalOperation(); --it)
    {
        subChain.push_front(it->node);
    }

    return subChain;
}

/**
 * @brief Account for a node in sram capacity calculation if one of the below hold:
 *        (1) Node is not logic
 *        (2) Node is a non-pure logic transpose
 */
bool TPCExpansionsAndSingleMMEBundleSolver::ShouldIgnoreNodeInSramCapacityCalculation::ignoreNonTransposeLogicals(
    const NodePtr&       n,
    const PipelineChain& chain,
    const HabanaGraph&   g)
{
    if (!n->isLogicalOperation()) return false;
    const auto logicOp = std::dynamic_pointer_cast<LogicalOpNode>(n);
    HB_ASSERT(logicOp,
              "Expecting node type: {}, name: {} to be a logical operation node",
              n->getNodeTypeStr(),
              n->getNodeName());
    return (logicOp->getNodeType() != Node::TYPE_LOGICAL_TRANSPOSE || logicOp->isPureLogical());
}

/**
 * @brief Account for a non-pure logical node in sram capacity calculation
 *        if one of the conditions below hold:
 *        (1) Node is physical
 *        (2) Node has control edges
 *        (2) Node is a logic transpose in a logic sub chain of more than 2 nodes
 *        (3) Node is a logic transpose in a logic sub chain of exactly 2 nodes and
 *            at least one of the logic nodes in the sub chain is not pure logic or
 *            its alias direction cannot be swapped.
 */
bool TPCExpansionsAndSingleMMEBundleSolver::ShouldIgnoreNodeInSramCapacityCalculation::
    ignoreNonTransposeLogicalsRelaxed(const NodePtr& n, const PipelineChain& chain, const HabanaGraph& g)
{
    if (!n->isLogicalOperation()) return false;
    const auto logicOp = std::dynamic_pointer_cast<LogicalOpNode>(n);
    if (logicOp->isPureLogical()) return true;
    if (isNodeWithCtrlEdges(n)) return false;
    if (n->getNodeType() != Node::TYPE_LOGICAL_TRANSPOSE) return true;

    bool       ignoreNode       = false;
    const auto logicOpsSubChain = getLogicOpsSubChain(n, chain);
    if (logicOpsSubChain.size() < 2)
    {
        ignoreNode = true;
    }
    else if (logicOpsSubChain.size() == 2)
    {
        // Logic transpose has exactly one adjacent logical operation.
        // Assume handle logical ops will optimize the logic sequence
        // such that no memcpy is planted as long as for both nodes the
        // alias directions of both nodes can be swapped or they are pure logical.
        // Assume memcpy is planted when the producer input is user managed, regardless of the logical op direction
        const auto& consumer = std::dynamic_pointer_cast<LogicalOpNode>(logicOpsSubChain.front());
        HB_ASSERT_PTR(consumer);
        const auto& producer = std::dynamic_pointer_cast<LogicalOpNode>(logicOpsSubChain.back());
        HB_ASSERT_PTR(producer);
        const auto canSwapAliasDirection = [&g](const std::shared_ptr<LogicalOpNode>& logic) -> bool {
            return (LogicalOpsHandler::wantBackwardDirectionShouldCallSwapDirection(logic) &&
                    LogicalOpsHandler::isSwapAliasDirectionProfitable(logic, g));
        };
        return (canSwapAliasDirection(producer)) &&
               (LogicalOpsHandler::isForwardNode(*consumer) && !producer->getInput(0)->isUserManagedDram());
    }
    else  // logicOpsSubChain.size() > 2
    {
        ignoreNode = false;
    }

    return ignoreNode;
}

bool TPCExpansionsAndSingleMMEBundleSolver::ignoreInSramCapacityCalculation(const NodePtr&       n,
                                                                            const PipelineChain& chain,
                                                                            const HabanaGraph&   g)
{
    const auto shouldIgnoreNodeInSram = ShouldIgnoreNodeInSramCapacityCalculation::getIgnoreNodeMethod();
    bool       ignore                 = shouldIgnoreNodeInSram->query(n, chain, g);
    LOG_TRACE(SRAM_SLICE,
              "{}: Node {} should{} be ignored in SRAM calc",
              HLLOG_FUNC,
              n->getNodeName(),
              ignore ? "" : " not");
    return ignore;
}

uint64_t TPCExpansionsAndSingleMMEBundleSolver::getUnSlicedChainSize(const PipelineChain& chain, const HabanaGraph& g)
{
    uint64_t chainSizeBytes = 0;
    for (const auto& chainNode : chain)
    {
        const auto& n = chainNode.node;
        if (ignoreInSramCapacityCalculation(n, chain, g))
        {
            LOG_DEBUG(SRAM_SLICE,
                      "{}: Excluding node<{}> {} in chain size calculation",
                      HLLOG_FUNC,
                      n->getNodeTypeStr(),
                      n->getNodeName());
        }
        else
        {
            const auto& ct(chainNode.connectingTensor);
            chainSizeBytes += ct->getDenseSizeInBytes();
        }
    }
    LOG_DEBUG(SRAM_SLICE, "{}: Unsliced producer chain size {} MB", HLLOG_FUNC, bToMb(chainSizeBytes));
    return chainSizeBytes;
}

TPCExpansionsAndSingleMMEBundleSolver::PipelineChain
TPCExpansionsAndSingleMMEBundleSolver::getMmeToFirstTpcSubChain(const PipelineChain& chain)
{
    // MME input/output real producer/consumer chain structure assumption:
    // (Logical)->...->(Logical)->(TPC)
    TPCExpansionsAndSingleMMEBundleSolver::PipelineChain subChain {};
    const auto                                           firstTPCNodeIt =
        std::find_if(chain.begin(), chain.end(), [](const auto& p) { return HabanaGraph::runsOnTPC(p.node); });
    if (firstTPCNodeIt != chain.end())
    {
        // output sub chain should include the first tpc node
        const auto mmeToFirstTpcSubChainEndIt(std::next(firstTPCNodeIt, 1));
        subChain.insert(subChain.begin(), chain.begin(), mmeToFirstTpcSubChainEndIt);
    }
    return subChain;
}

bool TPCExpansionsAndSingleMMEBundleSolver::alignmentAllowedForTensorInBundle(const TensorPtr&   t,
                                                                              const HabanaGraph& graph,
                                                                              unsigned           bundleIndex)
{
    if (!GCFG_SRAM_SLICER_ALIGN_TO_CACHE_LINE.value())
    {
        return false;
    }
    const auto& producer = graph.getTensorProducer(t);
    if (producer && producer->isLogicalOperation() && producer->getNodeAnnotation().bundleInfo.is_set() &&
        producer->getNodeAnnotation().bundleInfo->bundleIndex == bundleIndex)
    {
        // Can't align FCD strides in case of logical producer in the bundle
        LOG_DEBUG(SRAM_SLICE, "Can't align MME input {} due to logical producer", t->getName());
        return false;
    }
    return true;
}

BundleStrategyPtr TPCExpansionsAndSingleMMEBundleSolver::solveBundle()
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Bundle#{}", m_bundle->index()));

    LOG_TRACE(SRAM_SLICE, "{}", __PRETTY_FUNCTION__);

    BundleSolutionConstraints constraints = collectInitialConstraints();
    LOG_DEBUG(
        SRAM_SLICE,
        "Constraints: availableSramBytes {}, canSliceMultipleDims {}, canAlignInputToCL {}, sliceNonMasterOperand {}",
        constraints.availableSramBytes,
        constraints.canSliceMultipleDims,
        constraints.canAlignInputToCL,
        constraints.sliceNonMasterOperand);

    NodeStrategyPtr nodeStrategy = solvePrimeNode(constraints);
    // MME might not have a solution, in which case this bundle is not interesting.
    if (!nodeStrategy)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "No solution for node {} - discard bundle {}",
                  getBundlePrimeMmeNode()->getNodeName(),
                  m_bundle->index());
        return nullptr;
    }
    // If the node solver did not put a produced tensor in SRAM and there's enough SRAM budget to fit it, do so now.
    addUnslicedChainToPrimeNodeStrategy(nodeStrategy);

    tryPlaceInSramForCLAlignment(nodeStrategy);

    // Set the mme node solution to the bundle strategy
    BundleStrategyPtr bundleStrategy = createInitialStrategyForBundle(nodeStrategy);

    // Handle the rest of the nodes in the bundle according to the MME node
    expandInitialStrategy(bundleStrategy);
    bundleStrategy->alignNumBuffers();
    optimizeNumBuffers(bundleStrategy);
    return bundleStrategy;
}

BundleSolutionConstraints TPCExpansionsAndSingleMMEBundleSolver::collectInitialConstraints()
{
    BundleSolutionConstraints constraints;
    NodePtr                   mmeNode = getBundlePrimeMmeNode();

    constraints.canSliceMultipleDims        = canSlicePrimeNodeOnMultipleDims();
    constraints.tensorsInSram               = getInitialTensorsInSRAMConstraint();
    constraints.canAlignInputToCL           = canAlignTensorToCL(m_mmeInputToSlice);
    constraints.slicingGranularityPerTensor = getSlicingGranularityConstraint();
    constraints.availableSramBytes          = getSramBytesConstraint(constraints.slicingGranularityPerTensor);
    updateTensorsInSramConstraint(constraints.tensorsInSram);
    constraints.sharedOperandSlicingDimsAlignment = getSharedOperandSlicingDimsAlignmentConstraint();
    constraints.sliceNonMasterOperand             = canSliceNonMasterOperand();
    constraints.sharedOperandSlicingDims          = getSharedOperandSlicingDims();
    constraints.isSingleNodeBundle                = m_bundle->getNodes().size() == 1;
    return constraints;
}

bool TPCExpansionsAndSingleMMEBundleSolver::canAlignTensorToCL(const TensorPtr& tensorToAlign) const
{
    return alignmentAllowedForTensorInBundle(tensorToAlign, m_graph, m_bundle->index());
}

bool TPCExpansionsAndSingleMMEBundleSolver::canSlicePrimeNodeOnMultipleDims() const
{
    const auto& primeNode(getBundlePrimeMmeNode());
    return NodeSolver::canSliceNodeOnMultipleDims(primeNode,
                                                  getSharedOperandSlicingDims(),
                                                  (m_bundle->getNodes().size() == 1));
}

TensorSet TPCExpansionsAndSingleMMEBundleSolver::getInitialTensorsInSRAMConstraint() const
{
    // Bundle prime node must have at least one input that is slicable on its slicing dimension
    HB_ASSERT_PTR(m_mmeInputToSlice);

    TensorSet constraint {m_mmeInputToSlice};
    if (!m_consumerChain.empty())
    {
        constraint.insert(getBundlePrimeMmeNode()->getOutput(0));
    }
    return constraint;
}

GranularityPerTensor TPCExpansionsAndSingleMMEBundleSolver::getSlicingGranularityConstraint() const
{
    TensorSet nodeTensorsInSram = getInitialTensorsInSRAMConstraint();
    TensorSet slicedTensors({m_mmeInputToSlice});
    NodeSet   slicedNodes({getBundlePrimeMmeNode()});
    if (m_producerChains.empty())
    {
        slicedTensors = nodeTensorsInSram;
    }
    else
    {
        // In calculating of tensors slicing granularity consider all producer chains, in this way
        // the slicing granularity of the master operand's sliced dimension will accurately represent
        // the granularity of all sliced tensors.
        // For example: in case of symmetric BGEMM - both operands will be sliced on batch.
        // TODO: When SW-75259 is done add the consumer chain as well.
        for (const PipelineChain& producerChain : m_producerChains)
        {
            for (const PipelinedNode& producer : producerChain)
            {
                HB_ASSERT_PTR(producer.node);
                HB_ASSERT_PTR(producer.connectingTensor);
                slicedNodes.insert(producer.node);
                slicedTensors.insert(producer.connectingTensor);
            }
        }
    }
    // [CID: 42240] False positive - coverity ignores std::set default c'tor
    return getTensorsSlicingGranularity(slicedNodes, slicedTensors);
}

unsigned TPCExpansionsAndSingleMMEBundleSolver::getSramBytesConstraint(const GranularityPerTensor& slicingGranularity)
{
    return SlicingBrain::knobs.maxSRAMCapInBytes;
}

GranularityPerTensor
TPCExpansionsAndSingleMMEBundleSolver::getTensorsSlicingGranularity(const NodeSet&   nodesInSram,
                                                                    const TensorSet& tensorsInSram) const
{
    if (tensorsInSram.empty())
    {
        return {};
    }

    GranularityPerTensor slicingGranularity;
    if (!nodesInSram.empty())
    {
        slicingGranularity =
            CommonTileSizeCalculator::getMinCommonTilesSizes(nodesInSram, tensorsInSram, m_graph).first;
    }
    else
    {
        for (const TensorPtr& tensor : tensorsInSram)
        {
            if (slicingGranularity.find(tensor) == slicingGranularity.end())
            {
                // Found a tensor that's meant to be used from SRAM but is not a part of a producer chain. Currently
                // slicing granularity is only affected by TPC nodes and since this tensor is not produced, there's no
                // TPC node affecting its granularity => no constrains (for now)
                slicingGranularity[tensor] = TensorTile::Geometry(tensor->getDim(), 1);
            }
        }
    }
    return slicingGranularity;
}

bool TPCExpansionsAndSingleMMEBundleSolver::canSliceNonMasterOperand() const
{
    return false;
}

NodeStrategyPtr
TPCExpansionsAndSingleMMEBundleSolver::solvePrimeNode(const BundleSolutionConstraints& constraints) const
{
    // MME is always the prime for now
    NodePtr mmeNode = getBundlePrimeMmeNode();
    return NodeSolver::solveNode(mmeNode, constraints);
}

void TPCExpansionsAndSingleMMEBundleSolver::addUnslicedChainToPrimeNodeStrategy(
    NodeStrategyPtr& primeNodeStrategy) const
{
    // If the node solver did not put a produced tensor in SRAM and there's enough SRAM budget to fit it, do so now.
    for (const PipelineChain& prodChain : m_producerChains)
    {
        const TensorPtr&      producedTensor = prodChain.front().connectingTensor;
        const pSlicedOperand& slicedOp       = primeNodeStrategy->getSlicingData().getSlicedOperand(producedTensor);
        if (!slicedOp->resideInSRAM)
        {
            uint64_t usedSram = primeNodeStrategy->calculateMetrics().SRAMCapacity;
            LOG_TRACE(SRAM_SLICE, "Prime node SRAM allocation: {} MB", bToMb(usedSram));
            uint64_t remainingSRAM = SlicingBrain::knobs.maxSRAMCapInBytes - usedSram;
            if (producedTensor->getDenseSizeInBytes() <= remainingSRAM)
            {
                LOG_TRACE(SRAM_SLICE,
                          "Unsliced operand SRAM allocation {} MB",
                          bToMb(producedTensor->getDenseSizeInBytes()));
                slicedOp->resideInSRAM = true;
            }
        }
    }
}

void TPCExpansionsAndSingleMMEBundleSolver::tryPlaceInSramForCLAlignment(NodeStrategyPtr& primeNodeStrategy) const
{
    // If the node solver did not put an unaligned MME input in SRAM and there's enough SRAM budget to fit it, do so
    // now.
    const auto&    mmeNode   = getBundlePrimeMmeNode();
    pSlicedOperand slicedOp0 = primeNodeStrategy->getSlicingData().getSlicedOperand(mmeNode->getInput(0));
    pSlicedOperand slicedOp1 = primeNodeStrategy->getSlicingData().getSlicedOperand(mmeNode->getInput(1));
    HB_ASSERT(slicedOp0 && slicedOp1, "Missing sliced operand for MME input");
    std::set<pSlicedOperand> operandsToAlign = {slicedOp0, slicedOp1};
    // The strategy must include only the prime node, otherwise calculateMetrics won't be valid.
    for (const auto& slicedOp : operandsToAlign)
    {
        bool isNotAligned = SlicedOperandUtils::getCacheLineAlignedStrides(slicedOp->chunkDimensions, slicedOp).has_value();
        if (!slicedOp->resideInSRAM && canAlignTensorToCL(slicedOp->originalTensor) && isNotAligned)
        {
            slicedOp->resideInSRAM       = true;
            slicedOp->alignWithCacheLine = true;
            primeNodeStrategy->alignNumBuffers();
            if (primeNodeStrategy->calculateMetrics().SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes)
            {
                // Undo alignment and SRAM placing
                slicedOp->resideInSRAM       = false;
                slicedOp->alignWithCacheLine = false;
            }
            else
            {
                LOG_DEBUG(SRAM_SLICE, "{}: Place {} in SRAM", HLLOG_FUNC, slicedOp->originalTensor->getName());
            }
        }
    }
    primeNodeStrategy->alignNumBuffers();
}

BundleStrategyPtr
TPCExpansionsAndSingleMMEBundleSolver::createInitialStrategyForBundle(const NodeStrategyPtr& nodeStrategy) const
{
    // Set MME node strategy as the basic bundle strategy.
    return nodeStrategy;
}

NodeStrategyPtr TPCExpansionsAndSingleMMEBundleSolver::projectNodeStrategy(const NodePtr&           node,
                                                                           const BundleStrategyPtr& bundleStrategy,
                                                                           const TensorPtr& connectingTensor) const
{
    HB_ASSERT(node->getNodeAccessPattern() != nullptr,
              "Can't project bundle strategy on node {} (GUID: {}) - missing access pattern.",
              node->getNodeName(),
              node->getGUID());

    AccessPatternNodeSolutionProjector projector {node};

    NodeStrategyPtr nodeStrategy = projector.getNodeStrategy(bundleStrategy, connectingTensor);
    HB_ASSERT_PTR(nodeStrategy);

    return nodeStrategy;
}

void TPCExpansionsAndSingleMMEBundleSolver::expandInitialStrategy(BundleStrategyPtr& bundleStrategy)
{
    for (const PipelineChain& producers : m_producerChains)
    {
        const TensorPtr&      mmeInput       = producers.front().connectingTensor;
        const pSlicedOperand& slicedMmeInput = bundleStrategy->getSlicingData().getSlicedOperand(mmeInput);
        bool                  chainInSram    = slicedMmeInput->resideInSRAM;

        for (const PipelinedNode& producer : producers)
        {
            NodeStrategyPtr nodeStrategy =
                projectNodeStrategy(producer.node, bundleStrategy, producer.connectingTensor);

            // Unify the node strategy with the rest of the bundle.
            addProducerNodeSolutionToBundleStrategy(nodeStrategy, bundleStrategy, producer);
            bundleStrategy->getSlicingData().getSlicedOperand(producer.connectingTensor)->resideInSRAM = chainInSram;
        }
    }
    expandInitialStrategyWithConsumers(bundleStrategy);
}

void TPCExpansionsAndSingleMMEBundleSolver::expandInitialStrategyWithConsumers(BundleStrategyPtr& bundleStrategy)
{
    if (!m_consumerChain.empty())
    {
        bool             consumersChainInSram = bundleStrategy->getSlicingData().masterOperand->resideInSRAM;
        const DimVector& outputSlicedDims =
            SlicedOperandUtils::getSlicedDims(bundleStrategy->getSlicingData().masterOperand);
        const auto& consumerChainDims = m_consumerChain.front().slicingDims;
        HB_ASSERT(outputSlicedDims.empty() || std::is_permutation(outputSlicedDims.begin(),
                                                                  outputSlicedDims.end(),
                                                                  consumerChainDims.begin(),
                                                                  consumerChainDims.end()),
                  "Wrong slicing dim for consumer chain in bundle {}",
                  m_bundle->index());
        for (const auto& consumer : m_consumerChain)
        {
            NodeStrategyPtr nodeStrategy =
                projectNodeStrategy(consumer.node, bundleStrategy, consumer.connectingTensor);

            // Unify the node strategy with the rest of the bundle.
            addConsumerNodeSolutionToBundleStrategy(nodeStrategy, bundleStrategy, consumer);
            bundleStrategy->getSlicingData().getSlicedOperand(consumer.connectingTensor)->resideInSRAM =
                consumersChainInSram;
        }
    }
}

void TPCExpansionsAndSingleMMEBundleSolver::addProducerNodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                                                    BundleStrategyPtr&   bundleStrategy,
                                                                                    const PipelinedNode& producer) const
{
    // The MME node strategy is expected to be set as the initial strategy of the bundle.
    // The Logical and TPC nodes are expected to be handled after the MME node, and should cause the bundle strategy
    // expansion.

    HB_ASSERT_PTR(bundleStrategy);

    const NodePtr& node              = producer.node;
    const auto& nodeSlicingData   = nodeStrategy->getSlicingData();
    auto&       bundleSlicingData = bundleStrategy->getSlicingData();

    // Collect the sliced inputs from the node strategy and push them to the bundle strategy
    const auto&                        nodeInputs = node->getInputs();
    const std::vector<pSlicedOperand>& slicedInputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeInputs, nodeSlicingData, nullptr);
    const unsigned numInputs =
        std::count_if(nodeInputs.begin(), nodeInputs.end(), [](const auto& t) { return t != nullptr; });
    HB_ASSERT(slicedInputs.size() == numInputs,
              "Expecting a sliced operand for each input tensor. slicedInputs.size()={}, numInputs={}",
              std::to_string(slicedInputs.size()),
              std::to_string(numInputs));

    // Push the sliced outputs to the bundle strategy and find the connecting operand
    pSlicedOperand stitchedOperand = bundleSlicingData.getSlicedOperand(producer.connectingTensor);
    HB_ASSERT(stitchedOperand, "Connecting tensor wasn't found in bundle strategy");

    const auto&                        nodeOutputs = node->getOutputs();
    const std::vector<pSlicedOperand>& slicedOutputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeOutputs, nodeSlicingData, stitchedOperand);
    HB_ASSERT(slicedOutputs.size() == nodeOutputs.size(),
              "Expecting a sliced operand for each output tensor. slicedOutputs.size()={}, nodeOutputs.size={}",
              std::to_string(slicedOutputs.size()),
              std::to_string(nodeOutputs.size()));
    HB_ASSERT(m_graph.getTensorProducer(stitchedOperand->originalTensor) == node,
              "Expects the stitched operand: {} to be produced by the added node: {}",
              stitchedOperand->originalTensor->getName(),
              node->getNodeName());

    // Map produced MME input slices to TPC input slices.
    bundleSlicingData.setOperandSliceBackwardMapping(
        stitchedOperand,
        AccessPatternSliceMapper::createBwdMapping(node, slicedInputs, slicedOutputs));
}

void TPCExpansionsAndSingleMMEBundleSolver::addConsumerNodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                                                    BundleStrategyPtr&   bundleStrategy,
                                                                                    const PipelinedNode& consumer) const
{
    // The MME node strategy is expected to be set as the initial strategy of the bundle.
    // The Logical and TPC nodes are expected to be handled after the MME node, and should cause the bundle strategy
    // expansion.

    HB_ASSERT_PTR(bundleStrategy);

    const NodePtr&             node              = consumer.node;
    const StrategySlicingData& nodeSlicingData   = nodeStrategy->getSlicingData();
    StrategySlicingData&       bundleSlicingData = bundleStrategy->getSlicingData();

    // Push the sliced inputs to the bundle strategy and find the connecting operand
    pSlicedOperand stitchedOperand = bundleSlicingData.getSlicedOperand(consumer.connectingTensor);
    HB_ASSERT(stitchedOperand, "Connecting tensor wasn't found in bundle strategy");
    const auto&                        nodeInputs = node->getInputs();
    const std::vector<pSlicedOperand>& slicedInputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeInputs, nodeSlicingData, stitchedOperand);
    const unsigned numInputs =
        std::count_if(nodeInputs.begin(), nodeInputs.end(), [](const auto& t) { return t != nullptr; });
    HB_ASSERT(slicedInputs.size() == numInputs,
              "Expecting a sliced operand for each input tensor. slicedInputs.size()={}, numInputs={}",
              std::to_string(slicedInputs.size()),
              std::to_string(numInputs));

    // Collect the sliced outputs from the node strategy and push them to the bundle strategy
    const auto&                        nodeOutputs = node->getOutputs();
    const std::vector<pSlicedOperand>& slicedOutputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeOutputs, nodeSlicingData, nullptr);
    HB_ASSERT(slicedOutputs.size() == nodeOutputs.size(),
              "Expecting a sliced operand for each output tensor. slicedOutputs.size()={}, nodeOutputs.size()={}",
              std::to_string(slicedOutputs.size()),
              std::to_string(nodeOutputs.size()));

    const NodeList& consumers = m_graph.getTensorConsumers(stitchedOperand->originalTensor);
    HB_ASSERT((std::find(consumers.begin(), consumers.end(), node) != consumers.end()),
              "Expects the stitched operand: {} to be consumed by the added node: {}",
              stitchedOperand->originalTensor->getName(),
              node->getNodeName());

    std::list<pSlicedOperand> slicedInputsList(slicedInputs.begin(), slicedInputs.end());
    std::list<pSlicedOperand> slicedOutputsList(slicedOutputs.begin(), slicedOutputs.end());
    // Map consumed MME output slices to TPC output slices
    bundleSlicingData.setOperandSliceForwardMapping(
        stitchedOperand,
        AccessPatternSliceMapper::createFwdMapping(node, slicedInputsList, slicedOutputsList));
}

NodePtr TPCExpansionsAndSingleMMEBundleSolver::getBundlePrimeMmeNode() const
{
    auto iter = std::find_if(m_bundle->getNodes().begin(), m_bundle->getNodes().end(), [&](const NodePtr& n) {
        return HabanaGraph::runsOnMME(n);
    });
    HB_ASSERT(iter != m_bundle->getNodes().end(), "MME bundle is expected to have an MME node");
    return *iter;
}

bool TPCExpansionsAndSingleMMEBundleSolver::isTensorSharedByMMEBundleNodes(const TensorPtr& t) const
{
    return false;
}

void TPCExpansionsAndSingleMMEBundleSolver::fillBundleSolution(const BundleStrategyPtr& strategy)
{
    SolutionGenerator generator(m_graph, m_bundle, strategy);
    generator.fillSolution();
}

void TPCExpansionsAndSingleMMEBundleSolver::gatherConnectingTensors(const PipelineMultiChain&      pipelineChains,
                                                                    std::unordered_set<TensorPtr>& connectingTensors)
{
    for (const auto& pipelineChain : pipelineChains)
    {
        std::for_each(pipelineChain.begin(), pipelineChain.end(), [&connectingTensors](const auto& chainNode) {
            const bool isUniqueConnectingTensor = connectingTensors.insert(chainNode.connectingTensor).second;
            HB_ASSERT(isUniqueConnectingTensor,
                      "Tensor {} appears as a connecting tensor twice",
                      chainNode.connectingTensor->getName());
        });
    }
}

std::unordered_set<TensorPtr> TPCExpansionsAndSingleMMEBundleSolver::getConnectingTensors() const
{
    std::unordered_set<TensorPtr> connectingTensors {};
    gatherConnectingTensors(m_producerChains, connectingTensors);
    gatherConnectingTensors({m_consumerChain}, connectingTensors);
    return connectingTensors;
}

std::map<Dim, TSize> TPCExpansionsAndSingleMMEBundleSolver::getSharedOperandSlicingDimsAlignmentConstraint() const
{
    return {};
}

std::vector<Dim> TPCExpansionsAndSingleMMEBundleSolver::getSharedOperandSlicingDims() const
{
    return {};
}

bool BundleSolver::validateBundle() const
{
    const auto&                 bundleNodes = m_bundle->getNodes();
    std::unordered_set<NodePtr> bundleNodesSet(bundleNodes.begin(), bundleNodes.end());

    std::unordered_set<TensorPtr> connectingTensors(getConnectingTensors());
    std::unordered_set<TensorPtr> visitedTensors {};

    for (const auto& bundleNode : bundleNodes)
    {
        for (const auto& t : bundleNode->getOperands())
        {
            // Check only tensors which appear more than once, otherwise they won't have multiple adjucent nodes
            if (t && !t->isAuxTensor() && !visitedTensors.insert(t).second /* isVisited? */)
            {
                const auto& adjacentNodes = getAdjacentBundleNodes(t, bundleNodesSet);
                if (adjacentNodes.size() > 1)
                {
                    // A bundle tensor with more than one adjacent bundle node is considered valid if it belongs to
                    // one of the following exceptions: shared by mme nodes, connecting tensor, bundle persistent tensor
                    const bool isTensorSharedByMME = isTensorSharedByMMEBundleNodes(t);
                    const bool isBPT               = isBundlePersistentInputTensor(t, bundleNodesSet);
                    const bool isConnecting        = isConnectingTensor(t, connectingTensors);
                    LOG_DEBUG(SRAM_SLICE,
                              "Bundle tensor {} in {} has {} adjacent nodes, tensorSharedByMMENodes={}, "
                              "bundlePersistentTensor={}, connectingTensor={}",
                              t->getName(),
                              m_bundle->getName(),
                              adjacentNodes.size(),
                              isTensorSharedByMME,
                              isBPT,
                              isConnecting);
                    if (!isTensorSharedByMME && !isConnecting && !isBPT)
                    {
                        LOG_ERR(SRAM_SLICE, "Unsupported bundle tensor {} in {}", t->getName(), m_bundle->getName());
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// MantaRaySolver:
// We have: the mme nodes that share an operand
//         a producer chain producing the shared operand
//         0-3 TPC producers chains for the non-shared operands
//         The dim to slice the shared operand on

// We want: to slice the producer chain and the shared operand
//         to place the other inputs completely in SRAM
//         to find the maximal slice size that allows the slices to fit in SRAM

// How:    Calc the available SRAM for prime MME node according to all tensors in SRAM, and the sliced producres chain
//         granulairty
//         Slice the prime MME node by the SRAM capacity constraints
//         Project the slicing of the shared operand to its producers chain and the other MME nodes
//         Map forward to other MME nodes inputs and output
//         Map backwards from MME nodes to their non shared producers chains

NodePtr MantaRaySolver::getBundlePrimeMmeNode() const
{
    HB_ASSERT(!m_nonpartialsConsumers.empty(), "Expected to have at least one MME node that doesn't read partials");

    // Todo: improve selection
    return std::static_pointer_cast<Node>(*m_nonpartialsConsumers.begin());
}

bool MantaRaySolver::isTensorSharedByMMEBundleNodes(const TensorPtr& t) const
{
    return t == m_mmeInputToSlice;
}

std::unordered_set<TensorPtr> MantaRaySolver::getConnectingTensors() const
{
    // 1. Connecting tensors of master (shared) operand producer chain
    std::unordered_set<TensorPtr> connectingTensors(BaseClass::getConnectingTensors());
    // 2. Connecting tensors of the non master operand producer chains
    gatherConnectingTensors(m_nonMasterProducers, connectingTensors);
    return connectingTensors;
}

void MantaRaySolver::printTensorsTileGranularity(const TileSizePerTensor& tensorTileSizes)
{
    for (const auto& tensorAndGeometry : tensorTileSizes)
    {
        LOG_DEBUG(SRAM_SLICE,
                  "\ttensor: {} -> granularity: [{}]",
                  tensorAndGeometry.first->getName(),
                  toString(tensorAndGeometry.second, ','));
    }
}

TensorSet MantaRaySolver::getInitialTensorsInSRAMConstraint() const
{
    TensorSet tensorsInSram {m_mmeInputToSlice};

    return tensorsInSram;
}

GranularityPerTensor MantaRaySolver::getSlicingGranularityConstraint() const
{
    // No need to do anything, tensor tile sizes calc taking into account
    // all sliced nodes has already been performed by the bundlizer
    if (LOG_LEVEL_AT_LEAST_DEBUG(SRAM_SLICE))
    {
        LOG_DEBUG(SRAM_SLICE, "Effective tensor tile map:");
        printTensorsTileGranularity(m_tileSizes);
    }
    return m_tileSizes;
}

unsigned MantaRaySolver::getSramBytesConstraint(const GranularityPerTensor& slicingGranularity)
{
    unsigned availableSramBytes = SlicingBrain::knobs.maxSRAMCapInBytes;

    // Take into account sram committed for unsliced non shared operand chains
    HB_ASSERT(m_unslicedOperandsSramSize <= availableSramBytes, "Unsliced tensors unexpectedly don't fit in sram");
    availableSramBytes -= m_unslicedOperandsSramSize;

    LOG_DEBUG(SRAM_SLICE,
              "{}: Sliced producers chains require min {} MB, available SRAM {} MB",
              HLLOG_FUNC,
              bToMb(m_slicedOperandsSramMinSize),
              bToMb(availableSramBytes));
    HB_ASSERT(m_slicedOperandsSramMinSize > 0, "Expecting minSharedOpChainSramSize > 0");
    HB_ASSERT(m_slicedOperandsSramMinSize <= availableSramBytes,
              "minSharedOpChainSramSize {} MB > available SRAM {} MB",
              bToMb(m_slicedOperandsSramMinSize),
              bToMb(availableSramBytes));

    uint64_t sliceMultiplier = availableSramBytes / m_slicedOperandsSramMinSize;
    LOG_DEBUG(SRAM_SLICE, "{}: Slice multiplier factor: {}", HLLOG_FUNC, sliceMultiplier);

    // Derive the master operand SRAM capacity from the slice multiplier
    uint64_t primeNodeAllowedBytes = m_doubleBufferFactor * sliceMultiplier * m_sharedOperandMinSize;
    LOG_DEBUG(SRAM_SLICE, "Prime node SRAM allocation: {} MB", bToMb(primeNodeAllowedBytes));
    HB_ASSERT(primeNodeAllowedBytes <= availableSramBytes, "Can't fit prime node in sram with all other producers");
    return std::min(primeNodeAllowedBytes, SlicingBrain::knobs.maxSRAMCapInBytes);
}

NodeStrategyPtr MantaRaySolver::solvePrimeNode(const BundleSolutionConstraints& constraints) const
{
    NodeStrategyPtr strategy = BaseClass::solvePrimeNode(constraints);
    if (strategy)
    {
        auto slicedMasterOperand = strategy->getSlicingData().getSlicedOperand(m_mmeInputToSlice);
        HB_ASSERT_PTR(slicedMasterOperand);
        if (!effectiveSramUsage(slicedMasterOperand, m_masterOperandProducers))
        {
            LOG_DEBUG(SRAM_SLICE, "{}: Remove prime node from SRAM", __func__);
            slicedMasterOperand->resideInSRAM       = false;
            slicedMasterOperand->alignWithCacheLine = false;
            slicedMasterOperand->numOfBuffers       = 1;
        }
    }
    return strategy;
}

bool MantaRaySolver::effectiveSramUsage(const pSlicedOperand& slicedOperand, const PipelineChain& producers)
{
    // For SRAM usage to be effective, each slice either have to be produced directly in SRAM, or require enough
    // processing to hide the overhead of filling it with memcpies.

    const bool noProducer = producers.empty();
    const bool tinySlices =
        SlicedOperandUtils::getSliceSizeInBytes(slicedOperand) < GCFG_MIN_SLICE_SIZE_FOR_SRAM_FILLING.value();

    if (noProducer && tinySlices) return false;

    return true;
}

void MantaRaySolver::expandInitialStrategy(BundleStrategyPtr& bundleStrategy)
{
    // Use the strategy SRAM capacity calculator only for the prime node without any producers.
    // It doesn't calculate correctly once logical nodes are added to the strategy
    m_allocatedSram = bundleStrategy->calculateMetrics().SRAMCapacity;
    LOG_TRACE(SRAM_SLICE, "{}: Current bundle SRAM allocation: {} MB", HLLOG_FUNC, bToMb(m_allocatedSram));

    // Add the master operand producer chain for the prime node
    expandStrategyWithSharedProducersChain(bundleStrategy);

    // Add the shared MME nodes
    expandStrategyWithSharedMmeNodes(bundleStrategy);

    // Add the non shared operands chains for all MME nodes
    // Must be done after MME nodes are in the bundle -
    // This function assumes that an output of the node is already in the bundle
    expandStrategyWithNonSharedProducersChains(bundleStrategy);

    expandStrategyWithConsumersChain(bundleStrategy);
    // Set the non shared MME inputs, which have no producers chain, in SRAM.
    // Those tensors were selected by the bundlizer to be copied to SRAM to optimize the MME execution.
    updateCopyToSramMmeInputs(bundleStrategy);

    // Try to set the output of non prime MME nodes which have partials in SRAM.
    // Must be called after the producers are placed in SRAM
    // Update the output operand type to float32 for partials
    updatePartialsSharedMmeOutputs(bundleStrategy);

    tryAlignMmeInputsToCacheline(bundleStrategy);
}

void MantaRaySolver::optimizeNumBuffers(BundleStrategyPtr& bundleStrategy) const
{
    // Remove double buffer from operands which are part of the shared operands producers chain.
    // This param then becomes the "concurrency level" of the BFS order in bundle memcpy scheduler
    for (auto& op : bundleStrategy->getSlicingData().getSlicedOperands())
    {
        if (op->sharedChainMultiBuf)
        {
            op->numOfBuffers = 1;
        }
    }

    if (m_sliceNonSharedProducersChain)
    {
        HB_ASSERT(m_nonMasterProducers.size() == 1, "Expected a single chain for the non master operand");
        for (const PipelinedNode& producer : m_nonMasterProducers.front())
        {
            const auto& producerOutput = bundleStrategy->getSlicingData().getSlicedOperand(producer.connectingTensor);
            HB_ASSERT_PTR(producerOutput);
            if (producerOutput->resideInSRAM)
            {
                // Assign buffer for each slice - this will ensure that all slices will be placed in SRAM concurrently,
                // and enable reuse on the operand without additional DMA copies.
                producerOutput->numOfBuffers = SlicedOperandUtils::nofSlices(producerOutput);
            }
        }
    }
}

bool MantaRaySolver::isValidForAlignment(const pSlicedOperand& slicedOperand)
{
    const auto&                 bundleNodes = m_bundle->getNodes();
    std::unordered_set<NodePtr> bundleNodesSet(bundleNodes.begin(), bundleNodes.end());
    if (isBundlePersistentInputTensor(slicedOperand->originalTensor, bundleNodesSet) && !slicedOperand->resideInSRAM) return false;
    const auto& producer              = m_graph.getTensorProducer(slicedOperand->originalTensor);
    if (producer && !producer->canHandleStridedOutput(m_graph.getHALReader()->getDeviceType())) return false;
    auto alignedStrides = SlicedOperandUtils::getCacheLineAlignedStrides(slicedOperand->chunkDimensions, slicedOperand);
    return alignedStrides.has_value();
}

bool MantaRaySolver::isAlreadyAligned(const pSlicedOperand& slicedOperand)
{
    return slicedOperand->alignWithCacheLine;
}

TSize MantaRaySolver::tryAlignSingleOperandToCacheline(BundleStrategyPtr& bundleStrategy, const pSlicedOperand& slicedOperand)
{
    TSize additionalSramCap = 0;

    unsigned    numBuffers     = bundleStrategy->getMetrics().isDoubleBuffered ? 2 : 1;
    const auto  sliceSize      = calcEffectiveSliceSize(slicedOperand, numBuffers);

    slicedOperand->alignWithCacheLine = true;
    if (slicedOperand->resideInSRAM)
    {
        const auto& bundleNodes       = m_bundle->getNodes();
        const auto& producer          = m_graph.getTensorProducer(slicedOperand->originalTensor);
        bool        logicalProducer   = (producer && producer->isLogicalOperation() &&
                                (std::find(bundleNodes.begin(), bundleNodes.end(), producer) != bundleNodes.end()));
        unsigned    alignedSliceSlize = calcEffectiveSliceSize(slicedOperand, numBuffers);
        additionalSramCap             = logicalProducer ? alignedSliceSlize : (alignedSliceSlize - sliceSize);
    }
    return additionalSramCap;
}

void MantaRaySolver::tryAlignMmeInputsToCacheline(BundleStrategyPtr& bundleStrategy)
{
    if (!GCFG_ALIGN_ALL_MME_INPUTS.value()) return;

    for (const NodePtr& mmeNode : m_bundle->getNodes())
    {
        if (!HabanaGraph::runsOnMME(mmeNode)) continue;

        pSlicedOperand slicedOp0 = bundleStrategy->getSlicingData().getSlicedOperand(mmeNode->getInput(0));
        pSlicedOperand slicedOp1 = bundleStrategy->getSlicingData().getSlicedOperand(mmeNode->getInput(1));
        HB_ASSERT(slicedOp0 && slicedOp1, "Missing sliced operand for MME input");

        for (const auto& slicedOp : {slicedOp0, slicedOp1})
        {
            if (isAlreadyAligned(slicedOp) || !isValidForAlignment(slicedOp)) continue;

            auto additionalSram = tryAlignSingleOperandToCacheline(bundleStrategy, slicedOp);
            if ((m_allocatedSram + additionalSram) > SlicingBrain::knobs.maxSRAMCapInBytes)
            {
                // Undo alignment if the extra sram capacity is too large to fit
                slicedOp->alignWithCacheLine = false;
                continue;
            }
            m_allocatedSram += additionalSram;
            LOG_DEBUG(SRAM_SLICE,
                      "{}: Align {} to CL, additional SRAM: {}, total SRAM: {}",
                      HLLOG_FUNC,
                      slicedOp->originalTensor->getName(),
                      additionalSram,
                      m_allocatedSram);
        }
    }
}

uint64_t
MantaRaySolver::calcEffectiveSliceSize(const pSlicedOperand slicedOp, unsigned numBuffers) const
{
    uint64_t effectiveSliceSize  = SlicedOperandUtils::getSliceSizeInBytes(slicedOp);
    effectiveSliceSize *= numBuffers;
    LOG_TRACE(SRAM_SLICE,
              "Sliced tensor SRAM allocation {} MB, num buffers {} for {}",
              bToMb(effectiveSliceSize),
              numBuffers,
              slicedOp->originalTensor->getName());
    return effectiveSliceSize;
}

void MantaRaySolver::expandStrategyWithSharedProducersChain(BundleStrategyPtr& bundleStrategy)
{
    bool chainInSram = bundleStrategy->getSlicingData().getSlicedOperand(m_mmeInputToSlice)->resideInSRAM;
    // The first real producer output SRAM allocation is already accounted for by the prime node solution
    bool     mmeInputCountWasSkipped = false;
    uint64_t bufSize                 = 0;
    for (const PipelinedNode& producer : m_masterOperandProducers)
    {
        NodeStrategyPtr nodeStrategy = projectNodeStrategy(producer.node, bundleStrategy, producer.connectingTensor);

        // Unify the node strategy with the rest of the bundle.
        addProducerNodeSolutionToBundleStrategy(nodeStrategy, bundleStrategy, producer);
        const pSlicedOperand& producerOutput =
            bundleStrategy->getSlicingData().getSlicedOperand(producer.connectingTensor);
        producerOutput->resideInSRAM = chainInSram;
        producerOutput->sharedChainMultiBuf = m_sharedOperandChainInSharedMultiBuf;

        // Add the tensor SRAM allocation to the total count, if it's not ignoreInSramCapacityCalculation() and not
        // already counted
        if (chainInSram && !ignoreInSramCapacityCalculation(producer.node, m_masterOperandProducers, m_graph))
        {
            // sliced which are part of a shared multi buffer shouldn't count as double buffer, as they are multiplied
            // by the multi buffer level later
            unsigned numBuffers =
                bundleStrategy->getMetrics().isDoubleBuffered && !m_sharedOperandChainInSharedMultiBuf ? 2 : 1;
            const auto sliceSize = calcEffectiveSliceSize(producerOutput, numBuffers);
            if (!mmeInputCountWasSkipped)
            {
                // For single multi buffer we need to remove the allocation counting of the MME input, as it will be
                // replaced by the allocation of the multibuffer, according to the chain max buf size
                if (m_sharedOperandChainInSharedMultiBuf)
                {
                    unsigned strategyBuffers = bundleStrategy->getMetrics().isDoubleBuffered ? 2 : 1;
                    m_allocatedSram -= sliceSize * strategyBuffers;
                    bufSize = sliceSize;
                }
                mmeInputCountWasSkipped = true;
                continue;
            }
            if (!m_sharedOperandChainInSharedMultiBuf)
            {
                uint64_t remainingSRAM = SlicingBrain::knobs.maxSRAMCapInBytes - m_allocatedSram;
                HB_ASSERT(sliceSize <= remainingSRAM,
                          "No place in SRAM for slice (remaining {} MB)",
                          bToMb(remainingSRAM));
                m_allocatedSram += sliceSize;
            }
            else
            {
                LOG_TRACE(SRAM_SLICE, "Max bufSize {} MB with slice {} MB", bToMb(bufSize), bToMb(sliceSize));
                bufSize = std::max(bufSize, sliceSize);
            }
        }
    }
    if (m_sharedOperandChainInSharedMultiBuf)
    {
        uint64_t remainingSRAM = SlicingBrain::knobs.maxSRAMCapInBytes - m_allocatedSram;
        uint64_t chainSize     = c_sharedMultiBufLevel * bufSize;
        LOG_TRACE(SRAM_SLICE,
                  "m_allocatedSram {} MB increased by shared multi buf total size {} MB",
                  bToMb(m_allocatedSram),
                  bToMb(chainSize));
        HB_ASSERT(chainSize <= remainingSRAM,
                  "No place in SRAM for chain {} MB (remaining {} MB)",
                  bToMb(chainSize),
                  bToMb(remainingSRAM));
        m_allocatedSram += chainSize;
    }
}

void MantaRaySolver::expandStrategyWithSharedMmeNodes(BundleStrategyPtr& bundleStrategy) const
{
    for (auto& mmeNode : m_mmeNodes)
    {
        // handle the non-prime MME nodes
        if (mmeNode != getBundlePrimeMmeNode())
        {
            AccessPatternNodeSolutionProjector projector(mmeNode);
            NodeStrategyPtr mmeNodeStrategy = projector.getNodeStrategy(bundleStrategy, m_mmeInputToSlice);
            HB_ASSERT_PTR(mmeNodeStrategy);
            // Unify the node strategy with the rest of the bundle.
            addFwdMappedNodeSolutionToBundleStrategy(mmeNodeStrategy, bundleStrategy, mmeNode, m_mmeInputToSlice);
        }
    }
}
void MantaRaySolver::expandStrategyWithConsumersChain(BundleStrategyPtr& bundleStrategy)
{
    expandInitialStrategyWithConsumers(bundleStrategy);
    // to avoid exposed memcpy we prefer in case of evicted MME output which is not sliced to have consumer chain not to
    // be in sram
    auto                  mmeOutput        = m_mmeNodes.begin()->get()->getOutput(0);
    const pSlicedOperand& mmeOutputOperand = bundleStrategy->getSlicingData().getSlicedOperand(mmeOutput);
    NodeSet               bundleNodes(m_bundle->getNodes().begin(), m_bundle->getNodes().end());
    if (SlicedOperandUtils::isTriviallySliced(mmeOutputOperand) &&
        BundleSlicer::shouldTensorBeEvicted(mmeOutput, m_graph, bundleNodes))
        return;
    // Validate the consumer chain has the sram capacity it needs and place the consumer chain operands in SRAM
    for (auto it = m_consumerChain.begin(); it != m_consumerChain.end(); it++)
    {
        const auto& consumerNode       = *it;

        const pSlicedOperand& consumedOperand =
            bundleStrategy->getSlicingData().getSlicedOperand(consumerNode.connectingTensor);
        unsigned   numBuffers         = bundleStrategy->getMetrics().isDoubleBuffered ? 2 : 1;
        const auto sliceSize          = calcEffectiveSliceSize(consumedOperand, numBuffers);
        consumedOperand->resideInSRAM = true;
        if (!ignoreInSramCapacityCalculation(consumerNode.node, m_consumerChain, m_graph))
        {
            m_allocatedSram += sliceSize;
        }
        HB_ASSERT(m_allocatedSram <= SlicingBrain::knobs.maxSRAMCapInBytes,
                  "No place in SRAM for slice (missing {} MB)",
                  bToMb(m_allocatedSram - SlicingBrain::knobs.maxSRAMCapInBytes));
    }
}

void MantaRaySolver::addFwdMappedNodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                              BundleStrategyPtr&     bundleStrategy,
                                                              const NodePtr&         node,
                                                              const TensorPtr&       connectingTensor) const
{
    HB_ASSERT_PTR(bundleStrategy);
    HB_ASSERT_PTR(nodeStrategy);

    LOG_TRACE(SRAM_SLICE, "Stitching shared input MME Node {}", node->getNodeName());

    const StrategySlicingData& nodeSlicingData   = nodeStrategy->getSlicingData();
    StrategySlicingData&       bundleSlicingData = bundleStrategy->getSlicingData();
    pSlicedOperand             stitchedOperand   = bundleSlicingData.getSlicedOperand(connectingTensor);
    HB_ASSERT(stitchedOperand, "Connecting tensor wasn't found in bundle strategy");

    // Collect the sliced inputs from the node strategy and push them to the bundle strategy
    const auto&                        nodeInputs = node->getInputs();
    const std::vector<pSlicedOperand>& slicedInputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeInputs, nodeSlicingData, stitchedOperand);
    const unsigned numInputs =
        std::count_if(nodeInputs.begin(), nodeInputs.end(), [](const auto& t) { return t != nullptr; });
    HB_ASSERT(slicedInputs.size() == numInputs,
              "Expecting a sliced operand for each input tensor. slicedInputs.size()={}, numInputs={}",
              std::to_string(slicedInputs.size()),
              std::to_string(numInputs));

    // Collect the sliced outputs from the node strategy and push them to the bundle strategy
    const auto& nodeOutputs = node->getOutputs();

    const std::vector<pSlicedOperand>& slicedOutputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeOutputs, nodeSlicingData, nullptr);
    HB_ASSERT(slicedOutputs.size() == nodeOutputs.size(),
              "Expecting a sliced operand for each output tensor. slicedOutputs.size()={}, numOutputs={}",
              std::to_string(slicedOutputs.size()),
              std::to_string(nodeOutputs.size()));

    std::list<pSlicedOperand> slicedInputsList(slicedInputs.begin(), slicedInputs.end());
    std::list<pSlicedOperand> slicedOutputsList(slicedOutputs.begin(), slicedOutputs.end());
    // Map produced MME input slices to other MME input slices and the output slices
    bundleSlicingData.addOperandSliceForwardMapping(
        stitchedOperand,
        AccessPatternSliceMapper::createFwdMapping(node, slicedInputsList, slicedOutputsList));
}

void MantaRaySolver::updateCopyToSramMmeInputs(BundleStrategyPtr& bundleStrategy)
{
    for (const TensorPtr& t : m_copyTensorsToSram)
    {
        pSlicedOperand operand = bundleStrategy->getSlicingData().getSlicedOperand(t);
        // The operand is expected to be added to the strategy by the MME node solver / projection
        HB_ASSERT_PTR(operand);
        if (!operand->resideInSRAM)
        {
            uint64_t remainingSRAM = SlicingBrain::knobs.maxSRAMCapInBytes - m_allocatedSram;
            uint64_t tensorSize    = t->getDenseSizeInBytes();
            HB_ASSERT(tensorSize <= remainingSRAM,
                      "No place in SRAM for shared producer slice (remaining {})",
                      remainingSRAM);
            LOG_DEBUG(SRAM_SLICE, "Place input {} in SRAM - size {} MB", t->getName(), bToMb(tensorSize));
            operand->resideInSRAM = true;
            m_allocatedSram += tensorSize;
        }
    }
}

void MantaRaySolver::updatePartialsSharedMmeOutputs(BundleStrategyPtr& bundleStrategy)
{
    for (auto& mmeNode : m_partialsConsumers)
    {
        // add MME output to sram if it's partial, and its unsliced input (non shared operand)
        pSlicedOperand outputOperand = bundleStrategy->getSlicingData().getSlicedOperand(mmeNode->getOutput(0));
        // Assuming the output is single slice. If more sliced - need to count for double buffer and analyze the inputs
        // differently
        HB_ASSERT(SlicedOperandUtils::nofSlices(outputOperand) == 1, "Assuming output isn't sliced");

        // mmeNode was classified as partial consumer based on the expected slicing dimension, withuot regard to whether
        // the dimension was actually sliced. If the inputs were left unsliced, the output doesn't need any update.
        // (The non-common dimensions can't be sliced since it is asserted before that the output is not sliced)
        pSlicedOperand inputOperand = bundleStrategy->getSlicingData().getSlicedOperand(mmeNode->getInput(0));
        if (SlicedOperandUtils::nofSlices(inputOperand) == 1) continue;
        // The node is sliced on common dim - output data type is set for correct reduction.
        outputOperand->finalElementType = SlicedOperandUtils::getTypeForPartials(outputOperand->finalElementType);
        // slice size is expected to be the full tensor, as this node has partials, so only sliced on common dim
        unsigned outputSliceSize = SlicedOperandUtils::getSliceSizeInBytes(outputOperand);
        uint64_t remainingSRAM   = SlicingBrain::knobs.maxSRAMCapInBytes - m_allocatedSram;
        HB_ASSERT(outputSliceSize <= remainingSRAM,
                  "partials output {} > available SRAM {} for {}",
                  outputSliceSize,
                  remainingSRAM,
                  mmeNode->getNodeName());
        LOG_TRACE(SRAM_SLICE,
                  "Shared MME output SRAM allocation {} MB for {}",
                  bToMb(outputSliceSize),
                  mmeNode->getOutput(0)->getName());
        bundleStrategy->getSlicingData().getSlicedOperand(mmeNode->getOutput(0))->resideInSRAM = true;
        m_allocatedSram += outputSliceSize;
    }
}

void MantaRaySolver::expandStrategyWithNonSharedProducersChains(BundleStrategyPtr& bundleStrategy)
{
    for (const PipelineChain& producers : m_nonMasterProducers)
    {
        for (const PipelinedNode& producer : producers)
        {
            AccessPatternNodeSolutionProjector projector(producer.node);
            NodeStrategyPtr prodNodeStrategy = projector.getNodeStrategy(bundleStrategy, producer.connectingTensor);
            HB_ASSERT_PTR(prodNodeStrategy);

            // Unify the node strategy with the rest of the bundle.
            // This function assumes that an output of the node is already in the bundle, so this must be after the MME
            // nodes were added
            addProducerNodeSolutionToBundleStrategy(prodNodeStrategy, bundleStrategy, producer);
        }
        const TensorPtr&      nonSharedInput = producers.front().connectingTensor;
        const pSlicedOperand& slicedOperand  = bundleStrategy->getSlicingData().getSlicedOperand(nonSharedInput);
        HB_ASSERT_PTR(slicedOperand);  // The operand was added by the above calls
        if (slicedOperand->resideInSRAM) continue;

        auto const chainInSram = getMmeToFirstTpcSubChain(producers);
        // When m_sliceNonSharedProducersChain=true the non shared producer slices are expected to be placed
        // concurrently in SRAM (sram capacity should be calculated according to the unsliced case).
        if (!SlicedOperandUtils::isTriviallySliced(slicedOperand) && !m_sliceNonSharedProducersChain)
        {
            // Sliced non shared operand chain
            for (const PipelinedNode& producer : chainInSram)
            {
                const auto& producerOutput =
                    bundleStrategy->getSlicingData().getSlicedOperand(producer.connectingTensor);
                // Regardless as to whether to ignore in sram, set connecting tensor in sram
                producerOutput->resideInSRAM = true;

                if (ignoreInSramCapacityCalculation(producer.node, chainInSram, m_graph)) continue;
                unsigned   numBuffers    = bundleStrategy->getMetrics().isDoubleBuffered ? 2 : 1;
                const auto sliceSize     = calcEffectiveSliceSize(producerOutput, numBuffers);

                uint64_t remainingSRAM = SlicingBrain::knobs.maxSRAMCapInBytes - m_allocatedSram;
                HB_ASSERT(sliceSize <= remainingSRAM,
                          "No place in SRAM for node {} ({}) output, slice chunk shape [{}], slice size: {} MB "
                          "(remaining SRAM {} MB)",
                          producer.node->getNodeName(),
                          producer.node->getGUID(),
                          toString(producerOutput->chunkDimensions, ','),
                          bToMb(sliceSize),
                          bToMb(remainingSRAM));

                m_allocatedSram += sliceSize;
            }
        }
        else
        {
            // non shared operand is unsliced
            const auto chainInSramSize = getUnSlicedChainSize(chainInSram, m_graph);
            HB_ASSERT(m_allocatedSram + chainInSramSize <= SlicingBrain::knobs.maxSRAMCapInBytes,
                        "No place in sram for non shared input {} producer chain. SRAM left {} MB, chain size {} MB",
                        producers.front().connectingTensor->getName(),
                        bToMb(SlicingBrain::knobs.maxSRAMCapInBytes - m_allocatedSram),
                        bToMb(chainInSramSize));

            std::for_each(chainInSram.begin(), chainInSram.end(), [&bundleStrategy](const auto& producer) {
                bundleStrategy->getSlicingData().getSlicedOperand(producer.connectingTensor)->resideInSRAM = true;
            });
            LOG_DEBUG(SRAM_SLICE,
                        "Producer chain with {} nodes starting with {} has {} chain nodes in sram consuming {} MB",
                        producers.size(),
                        producers.front().node->getNodeName(),
                        chainInSram.size(),
                        bToMb(chainInSramSize));

            m_allocatedSram += chainInSramSize;
        }
    }
}

void MantaRaySolver::fillBundleSolution(const BundleStrategyPtr& strategy)
{
    if (!m_consumerChain.empty())
    {
        // Using the original solgen because it handles fwd mapping in a way which is correct for the consumers, however
        // it is not corret for shared input mme fwd mapping
        HB_ASSERT(m_mmeNodes.size() == 1, "Default solgen doesn't support shared mme fwd mapping");
        SolutionGenerator generator(m_graph, m_bundle, strategy);
        generator.fillSolution();
    }
    else
    {
        // Using MantaRay solgen to handle correctly shared input mme fwd mapping, but consumer fwd mapping is broken
        // (Keep traversing the bwd mapping after the shared mme fwd mapping)
        HB_ASSERT(m_consumerChain.empty(), "Mantaray solgen doesn't support consumer fwd mapping");
        SolutionGeneratorMantaRay generator(m_graph, m_bundle, strategy);
        generator.fillSolution();
    }
}

std::map<Dim, TSize> MantaRaySolver::getSharedOperandSlicingDimsAlignmentConstraint() const
{
    return m_sharedOperandSlicingDimsAlignment;
}

std::vector<Dim> MantaRaySolver::getSharedOperandSlicingDims() const
{
    return m_sharedOperandSlicingDims;
}

bool MantaRaySolver::canSliceNonMasterOperand() const
{
    return m_sliceNonSharedProducersChain;
}

// Reserve enough sram for the bundle operands. The actual placement of the tensors in sram is done in
// addSharedMMENodeSolutionToBundleStrategy.
unsigned SharedMmeProducerChainBundleSolver::getSramBytesConstraint(const GranularityPerTensor& slicingGranularity)
{
    unsigned doubleBufferFactor = 2;
    uint64_t remainingSram      = SlicingBrain::knobs.maxSRAMCapInBytes;

    // Calculate slicedDim minimal size. The minimal size is the product of the tensor's non-sliced dims with the
    // minimal size for the sliced dim, taking into account the dim alignment and granularity
    auto slicedDims = NodeSolver::getInputSlicingDims(m_primeNode, 0);
    HB_ASSERT(slicedDims.size() == 1,
              "Node {} should have a single slicing dim for input 0 when calculating constraints",
              m_primeNode->getNodeName());
    Dim      slicedDim = slicedDims.front();
    unsigned slicedDimMinSize =
        MmeNodeSolver::getMinSlicedDimSizeInElements(m_primeNode,
                                                     m_primeNode->getInput(0),
                                                     slicedDim,
                                                     slicingGranularity.at(m_primeNode->getInput(0))[slicedDim]);

    // First priority is fitting the shared operand in sram with double buffer. If it doesn't fit, fallback to single
    // buffer
    TensorPtr slaveShared   = m_slaveNode->getInput(0);
    std::map<unsigned, TSize> sizePerSlicedDim = {{slicedDim, slicedDimMinSize}};
    uint64_t sharedMinSize = SlicedOperandUtils::getTensorSliceSizeInBytes(slaveShared, sizePerSlicedDim);
    // It is possible that even the smallest shared input slice doesn't fit SRAM, since the bundlizer didn't check for
    // this. Currently the node solvers would just fail to find a strategy for those nodes, and the entire bundle won't
    // be sliced. Return the max SRAM capacity just for the prime node input, to let the shared MME behave like single
    // MME bundle.
    // TODO SW-91156 - check if the perf is worth blocking such bundles
    if (sharedMinSize > SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        LOG_WARN(SRAM_SLICE,
                 "Shared operand min size {} is larger than max SRAM capacity - let the node solver deal with it",
                 sharedMinSize);
        return SlicingBrain::knobs.maxSRAMCapInBytes;
    }
    if (sharedMinSize * doubleBufferFactor > remainingSram)
    {
        doubleBufferFactor = 1;
    }
    LOG_DEBUG(SRAM_SLICE, "Shared operand min size {}, doubleBufferFactor {}", sharedMinSize, doubleBufferFactor);

    // totalMinSize holds the sum of minimal sizes of all the operands that are sliced
    uint64_t totalMinSize       = sharedMinSize;
    uint64_t outputPrimeMinSize = 0;
    // Calculate dedx output with doubleBufferFactor. We want to fit it in sram only in case there is a consumer chain
    // in the bundle.
    if (!m_consumerChain.empty())
    {
        TensorPtr outputPrime = m_primeNode->getOutput(0);
        HB_ASSERT(m_consumerChain.front().slicingDims.size() == 1, "single consumer slicing dim is supported");
        unsigned                  outputSlicedDim        = m_consumerChain.front().slicingDims.front();
        std::map<unsigned, TSize> sizePerSlicedDimOutput = {{outputSlicedDim, slicedDimMinSize}};
        outputPrimeMinSize = SlicedOperandUtils::getTensorSliceSizeInBytes(outputPrime, sizePerSlicedDimOutput);
        if ((totalMinSize + outputPrimeMinSize) * doubleBufferFactor <= remainingSram)
        {
            m_isPrimeOutputInSram = true;
            LOG_DEBUG(SRAM_SLICE,
                      "prime node output set in SRAM, operand min size {} (before double buffer)",
                      outputPrimeMinSize);
            totalMinSize += outputPrimeMinSize;
        }
    }
    // All other operands should fit in single/double buffer, depending on the shared operand doulbeBufferFactor
    TensorPtr                 slaveNonShared        = m_slaveNode->getInput(1);
    std::map<unsigned, TSize> sizePerSlicedDimInput = {{slicedDim, slicedDimMinSize}};
    uint64_t                  nonSharedSlaveMinSize =
        SlicedOperandUtils::getTensorSliceSizeInBytes(slaveNonShared, sizePerSlicedDimInput);
    if ((totalMinSize + nonSharedSlaveMinSize) * doubleBufferFactor <= remainingSram)
    {
        m_isSlaveNonSharedInSram = true;
        LOG_DEBUG(SRAM_SLICE,
                  "slave non shared input set in SRAM, operand min size {} (before double buffer)",
                  nonSharedSlaveMinSize);
        totalMinSize += nonSharedSlaveMinSize;
    }
    // After calculating the totalMinSize of all the sliced operands, we want to reduce their size with double
    // buffer from the remainingSram, for the rest of the operands placing.
    remainingSram -= (totalMinSize * doubleBufferFactor);
    // nonSlicedOpsSize is for all the operands that node_solver does not place in sram by itself
    uint64_t nonSlicedOpsSize = 0;
    // Dedw output is in float because of the reduction, no double buffer since it's not sliced
    TensorPtr slaveOutput     = m_slaveNode->getOutput(0);
    uint64_t  slaveOutputSize = slaveOutput->getDenseSizeInElements() * dataTypeSizeInBytes(syn_type_float);
    if (slaveOutputSize <= remainingSram)
    {
        m_isSlaveOutputInSram = true;
        LOG_DEBUG(SRAM_SLICE, "slave output set in SRAM, operand size {}", slaveOutputSize);
        remainingSram -= slaveOutputSize;
        nonSlicedOpsSize += slaveOutputSize;
    }
    // Reserve non shared operand of dedx to sram, no double buffer since it's not sliced
    TensorPtr nonSharedPrime     = m_primeNode->getInput(1);
    uint64_t  nonSharedPrimeSize = nonSharedPrime->getDenseSizeInBytes();
    if (nonSharedPrimeSize <= remainingSram)
    {
        m_isPrimeNonSharedInSram = true;
        LOG_DEBUG(SRAM_SLICE, "prime node non shared input set in SRAM, operand size {}", nonSharedPrimeSize);
        remainingSram -= nonSharedPrimeSize;
        nonSlicedOpsSize += nonSharedPrimeSize;
    }
    unsigned multiplier =
        (SlicingBrain::knobs.maxSRAMCapInBytes - nonSlicedOpsSize) / (totalMinSize * doubleBufferFactor);
    HB_ASSERT(multiplier >= 1, "Unexpected multiplier value");
    // Return the allowed sram capacity required for the prime node operands for the node solver to fit the slices
    uint64_t primeNodeOperandsMinSize = sharedMinSize;
    if (m_isPrimeOutputInSram)
    {
        primeNodeOperandsMinSize += outputPrimeMinSize;
    }
    return ((primeNodeOperandsMinSize)*multiplier * doubleBufferFactor);
}

void SharedMmeProducerChainBundleSolver::addSharedMMENodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                                                  BundleStrategyPtr&     bundleStrategy,
                                                                                  const NodePtr&         node) const
{
    // Assert all assumtions: this code is intended only for Resnet shared input bundles of Dedx as master, and dedw as
    // slave (shared operand is dY).
    NodePtr primeNode = getBundlePrimeMmeNode();
    HB_ASSERT(Node::isDedxNode(primeNode), "DEDX must be the bundle master for shared mme input");
    HB_ASSERT(node->getNodeType() == Node::TYPE_DEDW, "Only DEDW stitching is supported");
    HB_ASSERT_PTR(bundleStrategy);
    HB_ASSERT_PTR(nodeStrategy);

    LOG_TRACE(SRAM_SLICE, "Stitching MME Node - {} to master node - {}", node->getNodeName(), primeNode->getNodeName());

    const auto&    nodeSlicingData     = nodeStrategy->getSlicingData();
    auto&          bundleSlicingData   = bundleStrategy->getSlicingData();
    pSlicedOperand slaveSharedOperand  = nodeSlicingData.getSlicedOperand(node->getInput(0));
    pSlicedOperand masterSharedOperand = bundleSlicingData.getSlicedOperand(primeNode->getInput(0));
    pSlicedOperand masterNonSharedOperand = bundleSlicingData.getSlicedOperand(primeNode->getInput(1));

    HB_ASSERT(slaveSharedOperand->originalTensor == masterSharedOperand->originalTensor,
              "Input at index 0 (dY) is expected to be the shared operand");

    // Add the slave operands to the bundle strategy
    pSlicedOperand slaveOutputOperand    = nodeSlicingData.getSlicedOperand(node->getOutput(0));
    pSlicedOperand slaveNonSharedOperand = nodeSlicingData.getSlicedOperand(node->getInput(1));
    bundleSlicingData.bundleTensors.push_back(slaveNonSharedOperand);
    bundleSlicingData.bundleTensors.push_back(slaveOutputOperand);

    if (masterSharedOperand->resideInSRAM)
    {
        slaveNonSharedOperand->resideInSRAM  = m_isSlaveNonSharedInSram;
        masterNonSharedOperand->resideInSRAM = m_isPrimeNonSharedInSram;
        slaveOutputOperand->resideInSRAM     = m_isSlaveOutputInSram;
        // masterOutputOperand is placed in sram (if needed) in node_solver
    }
    // Map the slave master operand to its inputs
    auto mapping = MMESliceMapper::mapOutputToInputs(node,
                                                     masterSharedOperand,
                                                     slaveNonSharedOperand,
                                                     slaveOutputOperand,
                                                     nullptr);
    bundleStrategy->getSlicingData().setOperandSliceBackwardMapping(slaveOutputOperand, mapping);

    // Taken from slicing_brain.cpp: transform input indexes to proper traversal pattern (dim
    // list)
    DimVector       slaveTraversalPattern;
    MmeDimController dimController(node);
    for (auto& index : {1, 0})  // {narrow operand index, wide operand index}
    {
        DimVector dimList;
        dimList = (index == 0) ? dimController.heightOutput() : dimController.widthOutput();
        std::reverse(
            dimList.begin(),
            dimList.end());  // In case more than one dimension is not degenerate (filter>1x1), take the outer one.
        // filter out degenerated dims.
        for (auto& dim : dimList)
        {
            // This function might be called before flattening the bundle nodes and adding
            // the candidates to the bundle (cost-model path).
            // Therefore we need to use finalShape (which represents the flattened operand sizes)
            // and not the original sizes of the slave node output tensor.
            if (slaveOutputOperand->finalShape[dim] == 1) continue;
            slaveTraversalPattern.push_back(dim);
            // only 1 dim for each input is allowed.
            break;
        }
    }
    auto& slaveTraversalPatternPtr = bundleStrategy->getSlicingData().getSlavesPatterns();
    slaveTraversalPatternPtr.push_back(
        SlicedOperandTraversalPattern(slaveOutputOperand,
                                      slaveTraversalPattern, /* TODO - is it ok to take the dedx traversal pattern aka
                                                               bundleStrategy->getSlicingData().traversalPattern?*/
                                      bundleStrategy->getSlicingData().isSnakeWalkingPatternEnabled(),
                                      SlicedOperandUtils::nofSlices(slaveNonSharedOperand)));
}

void SharedMmeProducerChainBundleSolver::expandInitialStrategy(BundleStrategyPtr& bundleStrategy)
{
    BaseClass::expandInitialStrategy(bundleStrategy);
    for (auto& node : m_bundle->getNodes())
    {
        if (HabanaGraph::runsOnMME(node))
        {
            // If the bundle is dedx and n is dedw, stitch it
            if (Node::isDedxNode(getBundlePrimeMmeNode()) && node->getNodeType() == Node::TYPE_DEDW)
            {
                SharedMMENodeSolutionProjector projector(node);

                NodeStrategyPtr mmeNodeStrategy =
                    projector.getNodeStrategy(bundleStrategy, node->getInput(TENSOR_DEDY));
                HB_ASSERT_PTR(mmeNodeStrategy);
                addSharedMMENodeSolutionToBundleStrategy(mmeNodeStrategy, bundleStrategy, node);
            }
        }
    }
}

bool SharedMmeProducerChainBundleSolver::isTensorSharedByMMEBundleNodes(const TensorPtr& t) const
{
    return t == m_primeNode->getInput(TENSOR_DEDY);
}

// getInitialTensorsInSRAMConstraint first decides if primeOutput is in sram.
// Then getSramBytesConstraint can change that decision in case it doesn't fit.
void SharedMmeProducerChainBundleSolver::updateTensorsInSramConstraint(TensorSet& tensorsInSram)
{
    if (!m_isPrimeOutputInSram)
    {
        NodePtr mmeNode = getBundlePrimeMmeNode();
        tensorsInSram.erase(mmeNode->getOutput(0));
    }
}