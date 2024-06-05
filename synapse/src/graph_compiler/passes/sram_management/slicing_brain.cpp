#include "slicing_brain.h"

#include "batch_gemm_solvers.h"
#include "batch_slicing_solver.h"
#include "common_dim_slicing_solver.h"
#include "dedx_node.h"
#include "defs.h"
#include "flatten_mme.h"
#include "graph_size_optimization_solver.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "mme_shared_input.h"
#include "passes/non_bundle_sram_tensor_comp.h"
#include "pattern_solvers.h"
#include "slicing_utils.h"
#include "spatial_slicing_solver.h"
#include "sram_management/pipeline_management/node_projector.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "tpc_bundle_solver.h"

SlicingBrain::SlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal) : m_graph {graph}
{
    if (initKnobsFromHal)
    {
        initKnobsValues();
    }
}

thread_local SlicingBrain::Knobs SlicingBrain::knobs;

void SlicingBrain::initKnobsValues()
{
    uint64_t nonBundleSramCap = NonBundleSramTensorComp::getGraphNonBundleTensorSramSize(&m_graph);

    HB_ASSERT(m_graph.getHALReader()->getSRAMSizeInBytes() > nonBundleSramCap, "SRAM capping too big.");

    auto          calculatedSRAMSize = m_graph.getHALReader()->getSRAMSizeInBytes() - nonBundleSramCap;
    constexpr int align              = 256 * 1024;
    if ((calculatedSRAMSize > align) && ((calculatedSRAMSize % align) != 0))
    {
        calculatedSRAMSize = round_to_multiple(calculatedSRAMSize - align, align);
    }

    knobs.maxSRAMCapInBytes = std::min(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.value(), calculatedSRAMSize);

    HB_ASSERT(knobs.maxSRAMCapInBytes < 0xFFFFFFFFFFFFFFFF,
              "SRAM size for slicing default value of 0xFFFFFFFFFFFFFFFF should never be used.");

    knobs.maxWideSliceSizeFactor_nonCommon2D        = 32768;
    knobs.maxWideSliceSizeFactor_nonCommon4D        = 32768;
    knobs.maxNarrowSliceSize                        = 32768;
    knobs.minCDSizeForPartials                      = 512;
    knobs.graphSizeOptimizationMultiplicationFactor = GCFG_SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_FACTOR.value();
    knobs.freqGHz = static_cast<double>(m_graph.getHALReader()->getClockFreqMHz()) / 1000.0;
    knobs.snakeWalkingTraversal                     = true;
    knobs.aggProcessingTimePipeliningFactor         = 1.1;
    knobs.hbmAvailableBWGBps                        = static_cast<double>(m_graph.getHALReader()->getHbmBwGBps());
    knobs.allowMultipleSolvers                      = GCFG_SRAM_SLICER_MULTIPLE_SOLVERS_ENABLED.value();
    knobs.numOfSlicesThreshold                      = 64;
    knobs.hbmTrafficDiffThreshold                   = 0.05;
}

MMESlicingBrain::MMESlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal /*= true*/)
: SlicingBrain(graph, initKnobsFromHal)
{
}

SlicingStrategyList MMESlicingBrain::getSolutionStrategies(const pBundle& bundle) const
{
    const auto& solvers = getSolversForBundle(bundle);
    if (solvers.empty()) return {};
    SlicingStrategyList strategies;
    for (const auto& solver : solvers)
    {
        solver->createAllStrategies();
        solver->AddStrategiesForGraphSizeOptimization();
        strategies.splice(strategies.end(),
                          GCFG_SRAM_SLICER_COST_MODEL_ENABLED.value() ? solver->getReducedStrategyList()
                                                                      : solver->getStrategies());
    }
    return strategies;
}

MMESlicingBrain::SolverList MMESlicingBrain::getSolversForBundle(const pBundle& bundle) const
{
    SolverList solvers;
    SLC_TRACE("Looking for solver for bundle: {}", bundle->getName());

    auto bgemmSolver = std::make_shared<BatchGemmSolver>(*m_graph.getHALReader(), bundle);
    auto nonCD2DSolver = std::make_shared<NonCD2DSolver>(*m_graph.getHALReader(), bundle);
    auto bgemmTinySolver = std::make_shared<BatchTinyGemmSolver>(*m_graph.getHALReader(), bundle);
    auto commonDimSolver = std::make_shared<CommonDimSlicingSolver>(*m_graph.getHALReader(), bundle);
    if (isBundleBatchGemm(bundle))
    {
        if (nonCD2DSolver->effectiveForBundle())
        {
            // Asymmetric(=Full broadcast) BGEMMs can be solved by the 2D solver.
            SLC_TRACE("- Non Common Dim (2d) solver was added");
            solvers.push_back(nonCD2DSolver);
        }
        if (bgemmTinySolver->effectiveForBundle())
        {
            // If the GEMMs are tiny, this solver should provide a better solution which the strategy comparison may
            // not realize. Currently the solvers are mutually exclusive, but they can be tested together in the future.
            SLC_TRACE("- BatchTinyGemm solver was added");
            solvers.push_back(bgemmTinySolver);
        }
        else if (bgemmSolver->effectiveForBundle())
        {
            // Symmetric batch gemms can be solved by this solver
            SLC_TRACE("- BatchGemm solver was added");
            solvers.push_back(bgemmSolver);
        }
        if (solvers.empty() && commonDimSolver->effectiveForBundle())
        {
            // Asymmetric(=Full broadcast) BGEMMs with large common dim can be solved by this solver, should be used
            // only if no other solver has found a solution
            SLC_TRACE("- Partials solver was added");
            solvers.push_back(commonDimSolver);
        }
        // Avoid adding any other solvers
        return solvers;
    }

    auto trivialSolver = std::make_shared<TrivialSolver>(*m_graph.getHALReader(), bundle);
    auto nonCD4dSolver = std::make_shared<NonCommon4DSolver>(*m_graph.getHALReader(), bundle);

    if (trivialSolver->effectiveForBundle())
    {
        SLC_TRACE("- Trivial solver was added");
        solvers.push_back(trivialSolver);
    }

    if (nonCD2DSolver->effectiveForBundle())
    {
        SLC_TRACE("- Non Common Dim (2d) solver was added");
        solvers.push_back(nonCD2DSolver);
    }

    if (nonCD4dSolver->effectiveForBundle())
    {
        SLC_TRACE("- Non Common Dim (4d) solver was added");
        solvers.push_back(nonCD4dSolver);
    }

    if (!isBundleGemm(bundle) || // Add common dim solver for any bundle which is not GEMM, or
        solvers.empty())         // for GEMM bundle, if there is no other solver for it
    {
        // make sure this solver is added last, to keep the GEMM condition correct
        if (commonDimSolver->effectiveForBundle())
        {
            SLC_TRACE("- Partials solver was added");
            solvers.push_back(commonDimSolver);
        }
    }
    if (solvers.empty())
    {
        SLC_TRACE("- Couldn't find effective solution");
        return {};
    }
    if (!knobs.allowMultipleSolvers)
    {
        SLC_TRACE("(allowMultipleSolvers=false) Only the first solver will be used!");
        return {solvers.front()};
    }
    else // default case - allow multiple solvers
    {
        return solvers;
    }
}

SlicingStrategyList TPCSlicingBrain::getSolutionStrategies(const pBundle& bundle) const
{
    auto solver = getSolverForBundle(bundle);
    if (!solver) return {};

    solver->createAllStrategies();

    return solver->getStrategies();
}

std::shared_ptr<Solver> TPCSlicingBrain::getSolverForBundle(const pBundle& bundle) const
{
    SLC_TRACE("Looking for solver for bundle: {}", bundle->getName());
    auto scalarPipeSolver = std::make_shared<TPCScalarPipeSolver>(*m_graph.getHALReader(), bundle);
    if (scalarPipeSolver->effectiveForBundle())
    {
        SLC_TRACE("- Scalar Pipe solution was selected");
        return scalarPipeSolver;
    }

    auto tpcSolver = std::make_shared<TpcBundleSolver>(*m_graph.getHALReader(), bundle);
    if (tpcSolver->effectiveForBundle())
    {
        SLC_TRACE("- TPC bundle solution was selected");
        return tpcSolver;
    }

    SLC_TRACE("- Couldn't find effective solution");
    return nullptr;
}

SlicingStrategyList DmaTransposeSlicingBrain::getSolutionStrategies(const pBundle& bundle) const
{
    auto solver = getSolverForBundle(bundle);
    if (!solver) return {};

    solver->createAllStrategies();

    return solver->getStrategies();
}

std::shared_ptr<Solver> DmaTransposeSlicingBrain::getSolverForBundle(const pBundle& bundle) const
{
    auto dmaTransposeSolver = std::make_shared<DMATransposeSolver>(*m_graph.getHALReader(), bundle);
    if (dmaTransposeSolver->effectiveForBundle())
    {
        SLC_TRACE("- Dma Transpose solution was selected");
        return dmaTransposeSolver;
    }
    return nullptr;
}

bool SlaveSlicingBrain::addProducerToStrategy(pBundleExpansion& expansionCandidate,
                                              pMmeSlicingStrategy& strategy) const
{
    NodePtr nodeToStitch = getNodeToBeStitched(expansionCandidate);
    if (!validateNodeToBeStitched(nodeToStitch)) return false;

    const pSlicedOperand& stitchedOperand = expansionCandidate->stitchedOperand;
    if (!stitchedOperand || m_graph.getTensorProducer(stitchedOperand->originalTensor) != nodeToStitch)
    {
        LOG_ERR(SRAM_SLICE, "Slave Slicing Brain: expects the stitched operand to be produced by the added node");
        return false;
    }

    SLC_TRACE("Slave Slicing Brain: adding node {} to solution strategy.", nodeToStitch->getNodeName());

    auto& bundleTensors = strategy->getSlicingData().bundleTensors;

    // When the stitched node is reshape - the shape of the stitched operand (output of the reshape) is different than
    // the inputs of the reshape
    bool allowReshape = allowOperandReshape();

    // Add inputs to strategy
    std::list<pSlicedOperand> slicedInputs =
        getSlicedOperandsByStitchedOperand(nodeToStitch->getInputs(), stitchedOperand, allowReshape);
    bundleTensors.insert(bundleTensors.end(), slicedInputs.begin(), slicedInputs.end());

    // Add outputs to strategy
    std::list<pSlicedOperand> slicedOutputs =
        getSlicedOperandsByStitchedOperand(nodeToStitch->getOutputs(), stitchedOperand, false);
    bundleTensors.insert(bundleTensors.end(), slicedOutputs.begin(), slicedOutputs.end());

    // Map produced MME input slices to TPC input slices.
    strategy->getSlicingData().setOperandSliceBackwardMapping(
        stitchedOperand, mapOutputToInputs(nodeToStitch, slicedInputs, stitchedOperand));

    updateProducerExpansionCandidateIfNeeded(expansionCandidate, slicedInputs);
    return true;
}

bool SlaveSlicingBrain::addConsumerToStrategy(pBundleExpansion& expansionCandidate,
                                              pMmeSlicingStrategy& strategy) const
{
    if (!validateConsumerCandidate(expansionCandidate)) return false;

    NodePtr nodeToStitch = getNodeToBeStitched(expansionCandidate);
    pSlicedOperand& stitchedOperand = expansionCandidate->stitchedOperand;

    SLC_TRACE("Slave Slicing Brain: adding node {} to solution strategy.", nodeToStitch->getNodeName());

    // TODO [SW-8591] - when mutliple strategies are implemented, expects this to become and assert
    //  (we should select a strategy to expand only if the nodeToStitch is already planned to be in SRAM)
    stitchedOperand->resideInSRAM = true;
    stitchedOperand->numOfBuffers = SlicedOperandUtils::isTriviallySliced(stitchedOperand) ? 1 : 2;

    auto& bundleTensors = strategy->getSlicingData().bundleTensors;

    // When the stitched node is reshape - the shape of the stitched operand (input of the reshape) is different than
    // the outputs of the reshape
    bool allowReshape = allowOperandReshape();

    // Add inputs to strategy
    std::list<pSlicedOperand> slicedInputs =
        getSlicedOperandsByStitchedOperand(nodeToStitch->getInputs(), stitchedOperand, false);
    bundleTensors.insert(bundleTensors.end(), slicedInputs.begin(), slicedInputs.end());

    // Add outputs to strategy
    std::list<pSlicedOperand> slicedOutputs =
        getSlicedOperandsByStitchedOperand(nodeToStitch->getOutputs(), stitchedOperand, allowReshape);
    bundleTensors.insert(bundleTensors.end(), slicedOutputs.begin(), slicedOutputs.end());

    // Map non-stitched inputs and outputs to a stitched input slice
    strategy->getSlicingData().setOperandSliceForwardMapping(
        stitchedOperand, mapSlicedOperandForward(nodeToStitch, stitchedOperand, slicedInputs, slicedOutputs));

    updateConsumerExpansionCandidateIfNeeded(expansionCandidate, slicedOutputs);
    return true;
}

std::list<pSlicedOperand> SlaveSlicingBrain::getSlicedOperandsByStitchedOperand(const TensorVector&   operands,
                                                                                const pSlicedOperand& stitchedOperand,
                                                                                bool allowReshape) const
{
    std::list<pSlicedOperand> slicedOperands;
    for (const pTensor& operand : operands)
    {
        if (operand == stitchedOperand->originalTensor) continue;
        pSlicedOperand nextSlicedOperand = std::make_shared<SlicedOperand>(operand);
        setOperandChunkSizeByStitchedOperand(nextSlicedOperand, stitchedOperand, allowReshape);
        setOperandMemoryLocation(nextSlicedOperand, stitchedOperand);
        slicedOperands.push_back(nextSlicedOperand);
    }
    return slicedOperands;
}

void SlaveSlicingBrain::setOperandChunkSizeByStitchedOperand(pSlicedOperand&       slicedOperand,
                                                             const pSlicedOperand& slicedStitchedOperand,
                                                             bool                  allowReshape)
{
    if (SlicedOperandUtils::isTriviallySliced(slicedStitchedOperand))
    {
        // output is not sliced, no need to slice inputs either.
        return;
    }
    if (slicedOperand->originalTensor->getTensorType() == INPUT_DESCRIBING_SHAPE_TENSOR)
    {
        // no need to slice INPUT_DESCRIBING_SHAPE_TENSOR, it should be the same for all inputs.
        return;
    }

    bool sameSizes = slicedOperand->originalTensor->getAllSizesInElements() ==
                     slicedStitchedOperand->originalTensor->getAllSizesInElements();
    bool stitchedOperandReshaped =
        slicedStitchedOperand->originalTensor->getAllSizesInElements() != slicedStitchedOperand->finalShape;
    bool sameNumOfElements = slicedOperand->originalTensor->getDenseSizeInElements() ==
                             slicedStitchedOperand->originalTensor->getDenseSizeInElements();

    if (slicedOperand->finalShape == slicedStitchedOperand->finalShape)
    {
        slicedOperand->chunkDimensions = slicedStitchedOperand->chunkDimensions;
    }
    else if ((stitchedOperandReshaped && sameSizes) ||  // The sliced stitched operand was flattened/reshaped, the
                                                        // sliced operand has the same original size.
             (allowReshape && sameNumOfElements))       // The stitched node is reshape - the shape of the
                                                        // slicedStitchedOperand and the slicedOperand is different,
                                                        // same number of elements.
    {
        slicedOperand->finalShape      = slicedStitchedOperand->finalShape;
        slicedOperand->chunkDimensions = slicedStitchedOperand->chunkDimensions;
    }
    else if (slicedOperand->finalShape[0] == slicedStitchedOperand->finalShape[0])
    {
        slicedOperand->chunkDimensions[0] = slicedStitchedOperand->chunkDimensions[0];
    }
}

bool SlaveSlicingBrain::validateConsumerCandidate(const pBundleExpansion& expansionCandidate) const
{
    NodePtr nodeToStitch = getNodeToBeStitched(expansionCandidate);
    if (!validateNodeToBeStitched(nodeToStitch)) return false;

    const pSlicedOperand& stitchedOperand = expansionCandidate->stitchedOperand;
    if (!stitchedOperand)
    {
        LOG_ERR(SRAM_SLICE, "Slave Slicing Brain: stitched operand is not found. Something went wrong");
        HB_ASSERT(false, "Slave Slicing Brain: stitched operand is nullptr. Something went wrong");
        return false;
    }
    NodeList consumers = m_graph.getTensorConsumers(stitchedOperand->originalTensor);
    auto consumerIter = std::find(consumers.begin(), consumers.end(), nodeToStitch);
    if (consumerIter == consumers.end())
    {
        LOG_ERR(SRAM_SLICE, "Slave Slicing Brain: consumer stitching: trying to stitch a node that is not the stitched operand consumer.");
        HB_ASSERT(false, "Slave Slicing Brain: stitched operand is not consumed by stitched node. Something went wrong");
        return false;
    }
    return true;
}

bool SlaveSlicingBrain::allowOperandReshape() const
{
    return false;
}

pBackwardSliceMapping SlaveSlicingBrain::mapOutputToInputs(const NodePtr& node,
                                                           const std::list<pSlicedOperand>& inputs,
                                                           const pSlicedOperand& output) const
{
    HB_ASSERT(0, "Not implemented - not to be used");
    return pBackwardSliceMapping();
}

///////////////////
// MMESlaveBrain //
///////////////////
void MMESlaveBrain::addSharedOperandMme(const pBundleExpansion& expansionCandidate, pMmeSlicingStrategy& strategy) const
{
    pSlicedOperand nonSharedInputOperand = expansionCandidate->slaveOperands.getInput();
    pSlicedOperand slaveOutputOperand = expansionCandidate->slaveOperands.getOutput();

    SLC_TRACE("Stitching MME Node - {} to master node - {}", expansionCandidate->nodeToStitch->getNodeName(), expansionCandidate->bundleNode->getNodeName());
    strategy->getMmeSlicingData().bundleTensors.push_back(nonSharedInputOperand);
    strategy->getMmeSlicingData().bundleTensors.push_back(slaveOutputOperand);
    auto shapeSlaveOperand = expansionCandidate->slaveOperands.getShapeOperand();
    if(shapeSlaveOperand)
    {
        strategy->getMmeSlicingData().bundleTensors.push_back(shapeSlaveOperand);
    }
    strategy->calculateMetrics();
    SLC_TRACE("New Strategy - ");
    strategy->printLog();

    unsigned inputIndex = expansionCandidate->nodeToStitch->getInputIndexOfTensor(nonSharedInputOperand->originalTensor);
    const pSlicedOperand& opA = (inputIndex == 0) ? nonSharedInputOperand : expansionCandidate->stitchedOperand;
    const pSlicedOperand& opB = (inputIndex == 0) ? expansionCandidate->stitchedOperand : nonSharedInputOperand;

    std::shared_ptr<BackwardSliceMapping> mapping;
    if (expansionCandidate->nodeToStitch->isBatchGemm())
    {
        mapping = MMETriviallyBatchedSliceMapper::mapOutputToInputs(expansionCandidate->nodeToStitch,
                                                                    opA, opB,
                                                                    slaveOutputOperand);
    }
    else
    {
        mapping = MMESliceMapper::mapOutputToInputs(expansionCandidate->nodeToStitch,
                                                    opA, opB,
                                                    slaveOutputOperand,
                                                    shapeSlaveOperand);
    }

    strategy->getMmeSlicingData().setOperandSliceBackwardMapping(slaveOutputOperand, std::move(mapping));
    strategy->getMmeSlicingData().addSlaveTraversalPattern(expansionCandidate);
}


NodePtr MMESlaveBrain::getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const
{
    return expansionCandidate->nodeToStitch;
}

bool MMESlaveBrain::validateNodeToBeStitched(const NodePtr& masterNode,
                                             const pBundle& masterBundle,
                                             const pBundle& slaveBundle)
{
    HB_ASSERT(slaveBundle, "could not find bundle of the slave node");
    auto it = std::find(slaveBundle->getNodes().begin(), slaveBundle->getNodes().end(), masterNode);
    if (it != slaveBundle->getNodes().end())
    {
        SLC_DEBUG("Master node of bundle {} is already stitched to bundle {}", masterBundle->index(), slaveBundle->index());
        return false;
    }
    // TODO - [SW-10870] check for this earlier.
    if (slaveBundle->getNodes().size() != 1)
    {
        SLC_DEBUG("Slave node has already other nodes stitched to it.");
        return false;
    }
    return true;
}

bool MMESlaveBrain::validateNodeToBeStitched(const NodePtr& node) const
{
    return HabanaGraph::runsOnMME(node);
}

pBundleExpansion MMESlaveBrain::adjustCandidateToStrategy(const pBundleExpansion& candidate,
                                                          const pMmeSlicingStrategy& strategy) const
{
    pBundleExpansion adjustedCandidate = std::make_shared<BundleExpansion>(*candidate);
    createStitchedOperands(adjustedCandidate, strategy);
    adjustedCandidate->additionalSRAMCapacity =
        m_candidateHandler.getCandidateAdditionalCapacity(adjustedCandidate);
    return adjustedCandidate;
}

bool MMESlaveBrain::needToPerformSamePositionStitching(const pBundleExpansion& candidate) const
{
    // left-right stitching is done when shared operand is on the same input index for both nodes and not transposed (or transposed on both)
    // OR shared operand is on different input index regarding both nodes and transposed on only one of them.
    bool isSameInputIdx, isTransposedOnOneNode;
    const NodePtr& masterNode = candidate->bundleNode;
    const NodePtr& slaveNode = candidate->nodeToStitch;
    const pTensor& sharedInput = candidate->stitchedOperand->originalTensor;
    // check if shared input is on the same position for master and slave node
    isSameInputIdx = isSameInputIndex(sharedInput, masterNode, slaveNode);
    // check if shared input is transposed.
    isTransposedOnOneNode = isSharedInputTransposed(sharedInput, masterNode) != isSharedInputTransposed(sharedInput, slaveNode);

    return (isSameInputIdx != isTransposedOnOneNode);
}

bool MMESlaveBrain::isSameInputIndex(const pTensor& sharedInput,
                                     const NodePtr& masterNode,
                                     const NodePtr& slaveNode) const
{
    unsigned masterIndex = masterNode->getInputIndexOfTensor(sharedInput);
    unsigned slaveIndex = slaveNode->getInputIndexOfTensor(sharedInput);
    return masterIndex == slaveIndex;
}

bool MMESlaveBrain::isSharedInputTransposed(const pTensor& sharedInput, const NodePtr& node) const
{
    bool isSharedInputTransposed = false;
    unsigned inputIdx = node->getInputIndexOfTensor(sharedInput);

    switch(node->getNodeType())
    {
    case Node::TYPE_DEDX :
        // W input is transposed.
        isSharedInputTransposed = (inputIdx == 1);
        break;
    case Node::TYPE_DEDW :
        // X input is transposed
        isSharedInputTransposed = (inputIdx == 0);
        break;
    case Node::TYPE_GEMM :
    case Node::TYPE_BATCH_GEMM:
    case Node::TYPE_GEMM_DEDW:
    case Node::TYPE_GEMM_DEDX:
    case Node::TYPE_BATCH_GEMM_DEDW:
    case Node::TYPE_BATCH_GEMM_DEDX:
    {
        auto gemmNode = dynamic_cast<GEMMNode*>(node.get());
        HB_ASSERT(gemmNode, "node is not a GEMMNode");
        isSharedInputTransposed = (inputIdx == 0) ? gemmNode->getGEMMParams().transpose_a :
                                  gemmNode->getGEMMParams().transpose_b;
        break;
    }
    default :
        isSharedInputTransposed = false;
    }
    return isSharedInputTransposed;
}

void MMESlaveBrain::doBatchGemmStitching(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const
{
    const pSlicedOperand& sharedOperand = candidate->stitchedOperand;
    pSlicedOperand slaveNonSharedOperand = candidate->slaveOperands.getInput();

    slaveNonSharedOperand->numOfBuffers = 2;

    unsigned initialSramCap = strategy->calculateMetrics().SRAMCapacity;
    // get non-shared operand slicing dim
    for (unsigned dim = DIM_GEMM_BATCH; dim < slaveNonSharedOperand->originalTensor->getDim(); ++dim)
    {
        if (SlicedOperandUtils::isSlicedOnDimension(sharedOperand, dim))
        {
            slaveNonSharedOperand->chunkDimensions[dim] = sharedOperand->chunkDimensions[dim];
        }
    }
    unsigned sramCap = initialSramCap +
                       (SlicedOperandUtils::getSliceSizeInBytes(slaveNonSharedOperand) *
                       slaveNonSharedOperand->numOfBuffers);
    // verifying if the required slice fits the SRAM. Double buffer is taking into account in sramCap value.
    if (sramCap <= SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        // found correct slicing - we can now set the operand to reside in SRAM
        slaveNonSharedOperand->resideInSRAM = true;
        // re-set num of buffers in case solution is trivial
        if (SlicedOperandUtils::isTriviallySliced(slaveNonSharedOperand))
            slaveNonSharedOperand->numOfBuffers = 1;

        SLC_TRACE("Found slicing for slave non-shared input - new chunk dimensions - {}",
                  toString(slaveNonSharedOperand->chunkDimensions.begin(),
                  slaveNonSharedOperand->chunkDimensions.end(),
                  'x'));
    }
}

void MMESlaveBrain::doSamePositionStitching(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const
{
    const pSlicedOperand& sharedOperand = candidate->stitchedOperand;
    pSlicedOperand slaveNonSharedOperand = candidate->slaveOperands.getInput();
    /* An example for a scenario that handled by the following block-
     * when the shared operand is sliced on the common dim but the topology should be stitched on the same position.*/
    if (SlicedOperandUtils::isSlicedOnCommonDim(sharedOperand, candidate->nodeToStitch))
    {
        SLC_TRACE("Using different-position stitching to determine non-shared operand slicing");
        doDifferentPositionStitching(candidate, strategy);
        return;
    }
    // TODO - [SW-11654] merge this with NonCommon2DSolver class.
    slaveNonSharedOperand->numOfBuffers = 2;

    // get non-shared operand slicing dim
    unsigned inputIndex = candidate->nodeToStitch->getInputIndexOfTensor(slaveNonSharedOperand->originalTensor);
    MmeDimController controller(candidate->nodeToStitch);
    const DimVector& inputSlicingDims =
        (inputIndex == 0) ? controller.nonCommonDimOperandA() : controller.nonCommonDimOperandB();
    // find the outer dim with size > 1 - this will provide the slicing dim for both 4d and 2d tensors.
    Settable<unsigned> nonSharedOpSlicingDim =
        SlicedOperandUtils::getFirstNonDegeneratedDim(inputSlicingDims, slaveNonSharedOperand);
    if (!nonSharedOpSlicingDim.is_set())
    {
        // in case all dims are of size 1 - no slicing will be done so any value is OK.
        nonSharedOpSlicingDim.set(inputSlicingDims.front());
    }
    unsigned geometrySizeInElements = (strategy->getMmeSlicingData().getWide() == sharedOperand)
                                          ? strategy->getMMENarrowGeometryInElements()
                                          : strategy->getMMEWideGeometryInElements();
    unsigned axisSizeInElements =
        slaveNonSharedOperand->originalTensor->getSizeInElements(nonSharedOpSlicingDim.value());
    // Set minimum slice size by default
    unsigned sliceSize = std::min(axisSizeInElements, geometrySizeInElements);
    // Check that this is a 4d convolution operation- it can't be flattened
    // Otherwise, it might be batch-gemm, but it's sliced on batch dim so it's not an issue.
    ConvBaseNode* pConv = dynamic_cast<ConvBaseNode*>(candidate->nodeToStitch.get());
    if (pConv && !pConv->canBeConvertedToGEMM())
    {
        auto spatialDims         = pConv->getSpatialDims();
        bool isSpatialSlicingDim = std::any_of(spatialDims.begin(), spatialDims.end(), [&](uint32_t dim) {
            return SlicedOperandUtils::isSlicedOnDimension(sharedOperand, dim);
        });
        // If shared operand is x or y and operand is sliced on spatial dimension
        if (sharedOperand->originalTensor != pConv->getWOperand() && isSpatialSlicingDim)
        {
            // When the convolution is 4d, x and y operands slicing is dependent (because of conv params).
            // If the shared op is x or y, and it's sliced on a spatial dim, the slave non-shared operand or output
            // slicing depends on the shared operand slicing. Since we don't want to go into the 4d slicing logic
            // here, we block this candidate.
            return;
        }
        isSpatialSlicingDim = std::any_of(spatialDims.begin(), spatialDims.end(), [&](uint32_t dim) {
            return nonSharedOpSlicingDim.value() == dim;
        });
        // If the non-shared operand is X or Y and, and non-shared slicing dimension is spatial
        if (slaveNonSharedOperand->originalTensor != pConv->getWOperand() && isSpatialSlicingDim)
        {
            // Keep non-shared operand trivially sliced
            sliceSize                           = axisSizeInElements;
            slaveNonSharedOperand->numOfBuffers = 1;
        }
        if (slaveNonSharedOperand->originalTensor == pConv->getWOperand())
        {
            // DimController doesn't set Q/R/S as slicing dims
            HB_ASSERT(nonSharedOpSlicingDim.value() < WEIGHT_DIM_S, "Slicing dimension can't be Q/R/S");
        }
    }
    slaveNonSharedOperand->chunkDimensions[nonSharedOpSlicingDim.value()] = sliceSize;
    unsigned initialSramCap = strategy->calculateMetrics().SRAMCapacity;
    unsigned sramCap = initialSramCap +
                       SlicedOperandUtils::getSliceSizeInBytes(slaveNonSharedOperand) *
                       slaveNonSharedOperand->numOfBuffers;

    // Try to increase as much as possible as long as the slave non-shared operand slice is smaller than the
    // master non-shared operand slice (or 2 geometries to ensure enough reuse),
    // to enable continuous operation of the MME without DMA bubbles.
    if (sramCap <= SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        const auto& masterNonSharedOperand = (strategy->getMmeSlicingData().getWide() == sharedOperand)
                                                 ? strategy->getMmeSlicingData().getNarrow()
                                                 : strategy->getMmeSlicingData().getWide();

        const auto masterNonSharedOperandSliceSize =
            SlicedOperandUtils::getSliceSizeInBytes(masterNonSharedOperand);

        while ((sliceSize < SlicingBrain::knobs.maxNarrowSliceSize) && (sliceSize < axisSizeInElements) &&
               ((SlicedOperandUtils::getSliceSizeInBytes(slaveNonSharedOperand) <
                 masterNonSharedOperandSliceSize) ||
                (sliceSize < 2 * geometrySizeInElements)))
        {
            sliceSize += geometrySizeInElements;
            slaveNonSharedOperand->chunkDimensions[nonSharedOpSlicingDim.value()] = sliceSize;
            sramCap = initialSramCap +
                      SlicedOperandUtils::getSliceSizeInBytes(slaveNonSharedOperand) *
                      slaveNonSharedOperand->numOfBuffers;
            if (sramCap > SlicingBrain::knobs.maxSRAMCapInBytes || sliceSize > axisSizeInElements)
            {
                // roll back one step
                sliceSize -= geometrySizeInElements;
                slaveNonSharedOperand->chunkDimensions[nonSharedOpSlicingDim.value()] = sliceSize;
                break;
            }
        }
        // found correct slicing - we can now set the operand to reside in SRAM
        slaveNonSharedOperand->resideInSRAM = true;
        // re-set num of buffers in case solution is trivial
        if (SlicedOperandUtils::isTriviallySliced(slaveNonSharedOperand))
            slaveNonSharedOperand->numOfBuffers = 1;

        SLC_TRACE("Found slicing for slave non-shared input - new chunk dimensions - {}",
                  toString(slaveNonSharedOperand->chunkDimensions.begin(), slaveNonSharedOperand->chunkDimensions.end(), 'x'));
    }
}

void MMESlaveBrain::doDifferentPositionStitching(const pBundleExpansion& candidate,
                                                 const pMmeSlicingStrategy& strategy) const
{
    const pSlicedOperand& sharedOperand = candidate->stitchedOperand;
    pSlicedOperand slaveNonSharedOperand = candidate->slaveOperands.getInput();
    if (!SlicedOperandUtils::isSlicedOnCommonDim(sharedOperand, candidate->nodeToStitch) &&
        candidate->nodeToStitch->getNodeType() != Node::TYPE_DEDW)
    {
        // in cases where the shared operand is sliced on the non-common dim regarding the slave node and the master node -
        // we would need to find slicing on the non-common dim of the slave non-shared operand - similar to same position stitching
        // for example - when stitching dedx node as slave to dedw master - the shared operand dedy is sliced on the non-common dim regarding the dedx.
        // cases where the slave node is DEDW and the shared operand is trivially solved was proven to be inefficient when performing same-position stitching.
        // in cases where the shared operand is sliced on the non-common dim regarding the slave node but not regarding the master node -
        // we should not slice slave non-shared operand
        if (SlicedOperandUtils::isSlicedOnCommonDim(sharedOperand, candidate->bundleNode))
        {
            slaveNonSharedOperand->numOfBuffers = 1;
            slaveNonSharedOperand->resideInSRAM = true;
        }
        else
        {
            SLC_TRACE("Using same-position stitching to determine non-shared operand slicing");
            doSamePositionStitching(candidate, strategy);
        }
    }
    else
    {
        // Check that this is a 4d convolution operation- it can't be flattened
        // Otherwise, it might be batch-gemm, but it's sliced on batch dim so it's not an issue.
        ConvBaseNode* pConv = dynamic_cast<ConvBaseNode*>(candidate->nodeToStitch.get());
        if (pConv && !pConv->canBeConvertedToGEMM())
        {
            auto spatialDims         = pConv->getSpatialDims();
            bool isSpatialSlicingDim = std::any_of(spatialDims.begin(), spatialDims.end(), [&](uint32_t dim) {
                return SlicedOperandUtils::isSlicedOnDimension(sharedOperand, dim);
            });
            // If shared operand is x or y and operand is sliced on spatial dimension, block
            if (sharedOperand->originalTensor != pConv->getWOperand() && isSpatialSlicingDim)
            {
                return;
            }
        }
        slaveNonSharedOperand->numOfBuffers = 2;
        // slice the slave non-shared operand on the common dim
        // slicing is determined according to the shared operand slicing
        MmeDimController dimController(candidate->nodeToStitch);
        unsigned inputIdx = candidate->nodeToStitch->getInputIndexOfTensor(slaveNonSharedOperand->originalTensor);
        const DimVector& nonSharedOperandCommonDims =
            (inputIdx == 0) ? dimController.commonDimOperandA() : dimController.commonDimOperandB();
        const DimVector& sharedOperandCommonDims =
            (inputIdx == 0) ? dimController.commonDimOperandB() : dimController.commonDimOperandA();
        HB_ASSERT(nonSharedOperandCommonDims.size() == sharedOperandCommonDims.size(), "size mismatch");

        auto nonSharedDim = nonSharedOperandCommonDims.begin();
        auto sharedDim = sharedOperandCommonDims.begin();
        for (unsigned i = 0; i < sharedOperandCommonDims.size(); i++)
        {
            if (SlicedOperandUtils::isSlicedOnDimension(sharedOperand, *sharedDim))
            {
                slaveNonSharedOperand->chunkDimensions[*nonSharedDim] = sharedOperand->chunkDimensions[*sharedDim];
            }
            nonSharedDim++;
            sharedDim++;
        }
        slaveNonSharedOperand->resideInSRAM = true;
        SLC_TRACE("Found slicing for slave non-shared input - new chunk dimensions - {}",
                  toString(slaveNonSharedOperand->chunkDimensions.begin(), slaveNonSharedOperand->chunkDimensions.end(), 'x'));
    }
}

void MMESlaveBrain::createSlicedOutputOperand(const pSlicedOperand& sharedOperand,
                                              const pSlicedOperand& nonSharedOperand,
                                              pSlicedOperand& slaveOutputOperand,
                                              const NodePtr& slaveNode) const
{
    // in case the operands are sliced on the common-dim - partials are created and output should reside in SRAM,
    // otherwise - need to check if slicing is needed.
    bool partialsCreated = SlicedOperandUtils::isSlicedOnCommonDim(nonSharedOperand, slaveNode);
    if (partialsCreated)
    {
        slaveOutputOperand->resideInSRAM = true;
        slaveOutputOperand->finalElementType = syn_type_float;
    }
    else if (slaveNode->isBatchGemm())
    {
        for (unsigned dim = DIM_GEMM_BATCH; dim < slaveOutputOperand->originalTensor->getDim(); ++dim)
        {
            slaveOutputOperand->chunkDimensions[dim] = sharedOperand->chunkDimensions[dim];
        }
    }
    else
    {
        // get output slicing dim
        MmeDimController controller(slaveNode);
        for (auto& operand : {sharedOperand, nonSharedOperand})
        {
            unsigned inputIndex = slaveNode->getInputIndexOfTensor(operand->originalTensor);
            const DimVector& outputSlicingDims =
                (inputIndex == 0) ? controller.heightOutput() : controller.widthOutput();
            Settable<unsigned> slicingDim = SlicedOperandUtils::getFirstNonDegeneratedDim(outputSlicingDims, slaveOutputOperand);
            if (slicingDim.is_set())
            {
                for (unsigned dim=0; dim < slaveOutputOperand->originalTensor->getDim(); ++dim)
                {
                    // operand is only sliced on 1 dimension
                    if (SlicedOperandUtils::isSlicedOnDimension(operand, dim))
                    {
                        slaveOutputOperand->chunkDimensions[slicingDim.value()] = operand->chunkDimensions[dim];
                        break;
                    }
                }
            }
        }
    }
}

void MMESlaveBrain::createStitchedOperands(const pBundleExpansion& candidate,
                                           const pMmeSlicingStrategy& strategy) const
{
    resetSlaveOperandsOfCandidate(candidate, strategy);
    SLC_TRACE("Creating the stitched operands for candidate - {}", candidate->nodeToStitch->getNodeName());
    pSlicedOperand nonSharedInput = candidate->slaveOperands.getInput();
    pSlicedOperand slaveOutput = candidate->slaveOperands.getOutput();

    bool shouldFlatten =
        SlicedOperandUtils::shouldAnyOperandBeFlattened(strategy->getMmeSlicingData().getSlicedOperands());
    if (shouldFlatten)  // The master operands are flattened - should flatten slave as well
    {
        // This is validated in findSharedMMEInputConsumers - if the master node is flattenable, the slave node is
        // flattenable as well.
        HB_ASSERT(MMENodeFlattener::canFlattenMMENode(candidate->nodeToStitch), "Slave MME node can't be flattened");
        for (auto& operandTupple : candidate->slaveOperands)
        {
            auto& operand            = operandTupple.second;
            operand->chunkDimensions = MMENodeFlattener::getFlattenShape(operand->originalTensor);
            operand->finalShape      = operand->chunkDimensions;
        }
    }

    if (candidate->nodeToStitch->isBatchGemm() && candidate->bundleNode->isBatchGemm())
    {
        SLC_TRACE("MME shared input candidate {} suitable for Batch Gemm stitching",
                  candidate->nodeToStitch->getNodeName());
        doBatchGemmStitching(candidate, strategy);
    }
    else if (needToPerformSamePositionStitching(candidate))
    {
        SLC_TRACE("MME shared input candidate {} suitable for same-position stitching",
                  candidate->nodeToStitch->getNodeName());
        doSamePositionStitching(candidate, strategy);
    }
    else
    {
        SLC_TRACE("MME shared input candidate {} suitable for different-position stitching",
                  candidate->nodeToStitch->getNodeName());
        doDifferentPositionStitching(candidate, strategy);
    }

    HB_ASSERT_PTR(nonSharedInput);
    if (nonSharedInput->resideInSRAM)
    {
        // non shared input is already sliced, need to only update the slave output operand
        createSlicedOutputOperand(candidate->stitchedOperand,
                                  nonSharedInput,
                                  slaveOutput,
                                  candidate->nodeToStitch);
        // align the shape operand to the output

        auto slaveOperandsShape = candidate->slaveOperands.getShapeOperand();
        if(slaveOperandsShape)
        {
            auto shapeTensor = slaveOperandsShape->originalTensor;
            *slaveOperandsShape = *slaveOutput;
            slaveOperandsShape->originalTensor = shapeTensor;
            slaveOperandsShape->resideInSRAM = false;
        }
    }
    else
        SLC_TRACE("Didnt found suitable slicing for candidate's non-shared operand");
}

void MMESlaveBrain::resetSlaveOperandsOfCandidate(const pBundleExpansion& candidate,
                                                  const pMmeSlicingStrategy& strategy) const
{
    // make sure we are working on the correct stitched operand
    for (auto& operand : strategy->getMmeSlicingData().getSlicedOperands())
    {
        if (operand->originalTensor == candidate->stitchedOperand->originalTensor)
        {
            candidate->stitchedOperand = operand;
        }
    }
    // reset the state of the slave operands to be ready for slicing calculations.
    for (auto& slaveOperandTupple : candidate->slaveOperands)
    {
        auto slaveOperand = slaveOperandTupple.second;
        slaveOperand->resideInSRAM = false;
        slaveOperand->chunkDimensions = slaveOperand->finalShape;
    }
}


/////////////////////
// TPCSlaveBrain //
/////////////////////
NodePtr TPCSlaveBrain::getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const
{
    return expansionCandidate->nodeToStitch;
}

bool TPCSlaveBrain::validateNodeToBeStitched(const NodePtr& node) const
{
    if (!HabanaGraph::runsOnTPC(node))
    {
        LOG_ERR(SRAM_SLICE, "TPC-Brain: expects the expansion candidate to be a TPC node");
        return false;
    }
    return true;
}

pBackwardSliceMapping TPCSlaveBrain::mapOutputToInputs(const NodePtr& node, const std::list<pSlicedOperand>& inputs,
                                                         const pSlicedOperand& output) const
{
    return TPCSliceMapper::mapOutputToInputs(node, inputs, output);
}

pForwardSliceMapping SlaveSlicingBrain::mapSlicedOperandForward(const NodePtr&                     node,
                                                                const pSlicedOperand&            keyInput,
                                                                const std::list<pSlicedOperand>& allInputs,
                                                                const std::list<pSlicedOperand>& allOutputs) const
{
    // When mapping forward, the keyInput is not included in allInputs, complete the list.
    std::list<pSlicedOperand> allInputsWithKey;
    auto slicedInputsIter = allInputs.begin();
    for (const pTensor& nodeInput : node->getInputs())
    {
        if (nodeInput == keyInput->originalTensor)
        {
            allInputsWithKey.push_back(keyInput);
        }
        else
        {
            allInputsWithKey.push_back(*slicedInputsIter++);
        }
    }
    return TrivialSliceMapper::mapSlicedOperandForward(allInputsWithKey, allOutputs);
}

NodePtr ReshapeSlicingBrain::getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const
{
    return expansionCandidate->reshapeNode;
}

bool ReshapeSlicingBrain::validateNodeToBeStitched(const NodePtr& node) const
{
    if (node == nullptr) // it's OK, i.e. in case the reshapes were moved out of th bundle before - so just return
    {
        return false;
    }
    if (node->getNodeType() != Node::TYPE_INTERNAL_RESHAPE)
    {
        LOG_ERR(SRAM_SLICE, "Reshape Slicing Brain: expects the expansion candidate to be a reshape node");
        return false;
    }
    return true;
}

pBackwardSliceMapping ReshapeSlicingBrain::mapOutputToInputs(const NodePtr& node, const std::list<pSlicedOperand>& inputs,
                                                     const pSlicedOperand& output) const
{
    return TrivialSliceMapper::mapOutputToInputs(node, inputs, output);
}

void ReshapeSlicingBrain::updateProducerExpansionCandidateIfNeeded(pBundleExpansion& expansionCandidate,
                                                                   const std::list<pSlicedOperand>& slicedInputs) const
{
    expansionCandidate->stitchedOperand = slicedInputs.front();
}

void ReshapeSlicingBrain::updateConsumerExpansionCandidateIfNeeded(pBundleExpansion& expansionCandidate,
                                                                   const std::list<pSlicedOperand>& slicedOutputs) const
{
    expansionCandidate->stitchedOperand = slicedOutputs.front();
}

void ReshapeSlicingBrain::setOperandMemoryLocation(pSlicedOperand        slicedOperand,
                                                   const pSlicedOperand& stitchedOperand) const
{
    if (slicedOperand->originalTensor->isShapeTensor())
    {
        // Shape tensor has no address, specifically, it is not residing in SRAM.
        slicedOperand->resideInSRAM = false;
    }
    else
    {
        // Non-shape tensors in both the reshape sides are alias of each other, so they both reside in SRAM
        // and have the same alignment.
        slicedOperand->resideInSRAM = true;
        slicedOperand->alignWithCacheLine = stitchedOperand->alignWithCacheLine;
        slicedOperand->numOfBuffers = stitchedOperand->numOfBuffers;
    }
}

bool ReshapeSlicingBrain::allowOperandReshape() const
{
    return true;
}

bool AccessPatternSlicingBrain::addProducerToStrategy(pBundleExpansion&    expansionCandidate,
                                                      pMmeSlicingStrategy& strategy) const
{
    NodePtr nodeToStitch = getNodeToBeStitched(expansionCandidate);
    if (!validateNodeToBeStitched(nodeToStitch)) return false;

    const pSlicedOperand& stitchedOperand = expansionCandidate->stitchedOperand;
    if (!stitchedOperand || m_graph.getTensorProducer(stitchedOperand->originalTensor) != nodeToStitch)
    {
        LOG_ERR(SRAM_SLICE, "AccessPatternSlicingBrain: Expects the stitched operand to be produced by the added node");
        return false;
    }

    SLC_TRACE("AccessPatternSlicingBrain: Adding node {} to solution strategy.", nodeToStitch->getNodeName());

    AccessPatternNodeSolutionProjector projector(nodeToStitch);
    const auto& tpcNodeStrategy = projector.getNodeStrategy(strategy, stitchedOperand->originalTensor);
    const auto&                        nodeSlicingData   = tpcNodeStrategy->getSlicingData();
    auto&                              bundleSlicingData = strategy->getSlicingData();

    // Collect the sliced inputs and outputs from the node strategy and push them to the bundle strategy
    const std::vector<pSlicedOperand>& slicedInputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeToStitch->getInputs(), nodeSlicingData, nullptr);
    const std::vector<pSlicedOperand>& slicedOutputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeToStitch->getOutputs(), nodeSlicingData, stitchedOperand);

    // Map produced MME input slices to TPC input slices.
    bundleSlicingData.setOperandSliceBackwardMapping(
        stitchedOperand,
        AccessPatternSliceMapper::createBwdMapping(nodeToStitch, slicedInputs, slicedOutputs));

    return true;
}

bool AccessPatternSlicingBrain::addConsumerToStrategy(pBundleExpansion&    expansionCandidate,
                                                      pMmeSlicingStrategy& strategy) const
{
    if (!validateConsumerCandidate(expansionCandidate)) return false;

    NodePtr         nodeToStitch    = getNodeToBeStitched(expansionCandidate);
    pSlicedOperand& stitchedOperand = expansionCandidate->stitchedOperand;

    SLC_TRACE("AccessPatternSlicingBrain: Adding node {} to solution strategy.", nodeToStitch->getNodeName());

    stitchedOperand->resideInSRAM = true;
    stitchedOperand->numOfBuffers = SlicedOperandUtils::isTriviallySliced(stitchedOperand) ? 1 : 2;

    AccessPatternNodeSolutionProjector projector(nodeToStitch);
    const auto& tpcNodeStrategy = projector.getNodeStrategy(strategy, stitchedOperand->originalTensor);
    const auto&                        nodeSlicingData   = tpcNodeStrategy->getSlicingData();
    auto&                              bundleSlicingData = strategy->getSlicingData();

    // Collect the sliced inputs and outputs from the node strategy and push them to the bundle strategy
    const std::vector<pSlicedOperand>& slicedInputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeToStitch->getInputs(), nodeSlicingData, stitchedOperand);
    const std::vector<pSlicedOperand>& slicedOutputs =
        bundleSlicingData.addNodeOperandsToStrategy(nodeToStitch->getOutputs(), nodeSlicingData, nullptr);

    std::list<pSlicedOperand> slicedInputsList(slicedInputs.begin(), slicedInputs.end());
    std::list<pSlicedOperand> slicedOutputsList(slicedOutputs.begin(), slicedOutputs.end());
    // Map consumed MME output slices to TPC output slices
    bundleSlicingData.setOperandSliceForwardMapping(
        stitchedOperand,
        AccessPatternSliceMapper::createFwdMapping(nodeToStitch, slicedInputsList, slicedOutputsList));

    return true;
}

NodePtr TPCAccessPatternSlicingBrain::getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const
{
    return expansionCandidate->nodeToStitch;
}

bool TPCAccessPatternSlicingBrain::validateNodeToBeStitched(const NodePtr& node) const
{
    if (!HabanaGraph::runsOnTPC(node))
    {
        LOG_ERR(SRAM_SLICE, "TPCAccessPatternSlicingBrain: The expansion candidate should be a TPC node");
        return false;
    }
    return true;
}

bool ReshapeAccessPatternSlicingBrain::addProducerToStrategy(pBundleExpansion&    expansionCandidate,
                                                             pMmeSlicingStrategy& strategy) const
{
    if (!getNodeToBeStitched(expansionCandidate))
    {
        return true;  // No reshape
    }

    if (!AccessPatternSlicingBrain::addProducerToStrategy(expansionCandidate, strategy))
    {
        return false;
    }

    updateNextStitchedOperand(expansionCandidate, strategy, getNodeToBeStitched(expansionCandidate)->getInput(0));

    return true;
}

bool ReshapeAccessPatternSlicingBrain::addConsumerToStrategy(pBundleExpansion&    expansionCandidate,
                                                             pMmeSlicingStrategy& strategy) const
{
    if (!getNodeToBeStitched(expansionCandidate))
    {
        return true;  // No reshape
    }

    if (!AccessPatternSlicingBrain::addConsumerToStrategy(expansionCandidate, strategy))
    {
        return false;
    }

    updateNextStitchedOperand(expansionCandidate, strategy, getNodeToBeStitched(expansionCandidate)->getOutput(0));

    return true;
}

void ReshapeAccessPatternSlicingBrain::updateNextStitchedOperand(pBundleExpansion&          expansionCandidate,
                                                                 const pMmeSlicingStrategy& strategy,
                                                                 const TensorPtr&           nextStitchedTensor) const
{
    auto nextStitchedOperand = strategy->getSlicingData().getSlicedOperand(nextStitchedTensor);
    HB_ASSERT_PTR(nextStitchedOperand);

    updateOperandMemoryLocation(nextStitchedOperand, expansionCandidate->stitchedOperand);

    // Move the stitched operand to the tensor between the TPC and the reshape node.
    expansionCandidate->stitchedOperand = nextStitchedOperand;
}

NodePtr ReshapeAccessPatternSlicingBrain::getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const
{
    return expansionCandidate->reshapeNode;
}

bool ReshapeAccessPatternSlicingBrain::validateNodeToBeStitched(const NodePtr& node) const
{
    if (!node || (node->getNodeType() != Node::TYPE_INTERNAL_RESHAPE))
    {
        LOG_ERR(SRAM_SLICE, "ReshapeAccessPatternSlicingBrain: The expansion candidate should be a reshape node");
        return false;
    }
    return true;
}

void ReshapeAccessPatternSlicingBrain::updateOperandMemoryLocation(pSlicedOperand        slicedOperand,
                                                                   const pSlicedOperand& stitchedOperand) const
{
    // The input and the output of the reshape node are alias of each other, so they both reside in SRAM
    // and have the same alignment.
    slicedOperand->resideInSRAM       = true;
    slicedOperand->alignWithCacheLine = stitchedOperand->alignWithCacheLine;
    slicedOperand->numOfBuffers       = stitchedOperand->numOfBuffers;
}