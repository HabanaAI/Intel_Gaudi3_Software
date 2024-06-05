#include "node_solver.h"
#include "bundle.h"
#include "math_utils.h"
#include "mme_brain_proxy.h"
#include "compilation_hal_reader.h"
#include "utils.h"
#include "post_slicing_op_handler.h"
#include "pipeline_management/node_projector.h"
#include "sram_management/pipeline_management/bundle_solver.h"
#include "sram_management/slicing_brain.h"
#include "node_tensor_accessor.h"

// Returns single dim mapping for each givenSlicingDim, or empty result if one dim has no single dim mapping
std::vector<Dim> NodeSolver::getTensorMatchingSlicedDims(const NodePtr&          node,
                                                         const TensorPtr&        givenTensor,
                                                         const std::vector<Dim>& givenSlicingDims,
                                                         const TensorPtr&        queriedTensor)
{
    NodeAccessPatternPtr accessPattern = node->getNodeAccessPattern();
    HB_ASSERT_PTR(accessPattern);
    std::vector<Dim> queriedSlicingDims;
    for (Dim givenSlicingDim : givenSlicingDims)
    {
        MultiDims matchingSlicingDims =
            accessPattern->getTensorMatchingSlicedDims(queriedTensor, givenTensor, givenSlicingDim);

        HB_ASSERT(!HabanaGraph::runsOnMME(node) || matchingSlicingDims.size() <= 1,
                  "Invalid number of matching slicing dims for MME node: {}",
                  matchingSlicingDims.size());
        // If the given tensor slicing dim is not mapped to any queried tensor dim, continue to the next dim. Let the
        // caller decide if there should be 1:1 matching, or partial matching. For example, partial matching is good
        // enough for mapping two operands of the same batch-gemm, where one of them is sliced on spatial non common dim
        // + batch dim
        if (matchingSlicingDims.empty()) continue;
        // If some dimensions packing mapped the given tensor slicing dim to more than 1 queried tensor slicing dim, the requested value is
        // undefined, and we fail the entire matching
        if (matchingSlicingDims.size() > 1) return {};

        queriedSlicingDims.push_back(matchingSlicingDims.front());
    }
    return queriedSlicingDims;
}

bool NodeSolver::doesNodeSupportSlicing(const NodePtr& node)
{
    std::unique_ptr<MmeNodeSolver> mmeSolver = MmeNodeSolver::getSpecificMmeSolver(node);
    // TODO SW-101527 Support MME transpose bundle for gaudi3 pipeline manager pass
    if (node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE) return false;

    // Avoid bundling tiny mme nodes, since slicing them can make them even more latency-bound
    MmeCommon::PerfAttr perfAttr = MmeBrainProxy::getRecommendedConfigMmePerf(node);
    if (perfAttr.expectedComputeCycles <= GCFG_MIN_CYCLES_FOR_MME_SLICING.value())
    {
        LOG_DEBUG(SRAM_SLICE,
                  "{}: Node {} has expected num of compute cycles {} which is less than {} (minimal cycles for slicing)",
                  __func__,
                  node->getNodeName(),
                  perfAttr.expectedComputeCycles,
                  GCFG_MIN_CYCLES_FOR_MME_SLICING.value());
        return false;
    }

    HB_ASSERT_PTR(mmeSolver);  // Only supported node types are expected to check if the node can be sliced
    bool supported = mmeSolver->isNodeSupported();
    return supported && (isInputSliceable(node, 0) || isInputSliceable(node, 1));
}

bool NodeSolver::isInputSliceable(const NodePtr& node, unsigned inputIdx)
{
    auto slicingDims = NodeSolver::getInputSlicingDims(node, inputIdx);
    // Nodes must have at least one operand slicable on its slicing dimension
    return std::any_of(slicingDims.begin(), slicingDims.end(), [&](Dim d) {
        return node->getInput(inputIdx)->getSizeInElements(d) > 1;
    });
}

std::vector<Dim> NodeSolver::getInputSlicingDims(const NodePtr& node, unsigned inputIdx)
{
    std::unique_ptr<MmeNodeSolver> mmeSolver = MmeNodeSolver::getSpecificMmeSolver(node);
    HB_ASSERT_PTR(mmeSolver);  // This function is expected only for supported node types
    return mmeSolver->getInputSlicingDims(inputIdx);
}

bool NodeSolver::canSliceNodeOnMultipleDims(const NodePtr&          mmeNode,
                                            const std::vector<Dim>& sharedOperandSlicingDims,
                                            const bool              isSingleNodeBundle)
{
    std::unique_ptr<MmeNodeSolver> mmeSolver = MmeNodeSolver::getSpecificMmeSolver(mmeNode);
    HB_ASSERT_PTR(mmeSolver);  // This function is expected only for supported node type
    return mmeSolver->canSliceNodeOnMultipleDims(sharedOperandSlicingDims, isSingleNodeBundle);
}

TSize MmeNodeSolver::getMinSlicedDimSizeInElements(const NodePtr&   node,
                                                   const TensorPtr& slicedInput,
                                                   unsigned         slicingDim,
                                                   const TSize      dimGranularity)
{
    TSize dimSize = slicedInput->getSizeInElements(slicingDim);
    if (dimSize <= dimGranularity) return dimSize;

    std::unique_ptr<MmeNodeSolver> mmeSolver = MmeNodeSolver::getSpecificMmeSolver(node);
    HB_ASSERT_PTR(mmeSolver);  // This function is expected only for supported node types
    return mmeSolver->calcMinSlicedDimSizeInElements(slicedInput, slicingDim, dimGranularity);
}

TSize MmeNodeSolver::calcMinSlicedDimSizeInElements(const TensorPtr& slicedInput,
                                                    unsigned         inputSlicingDim,
                                                    const TSize      dimGranularity)
{
    unsigned           inputIdx = m_node->getInputIndexOfTensor(slicedInput);
    std::optional<Dim> outputSliceDim;
    if (!m_dimController.isCommonDimForOperand(inputSlicingDim, inputIdx))
    {
        auto outputDims = getTensorMatchingSlicedDims(m_node, slicedInput, {inputSlicingDim}, m_node->getOutput(0));
        HB_ASSERT(outputDims.size() == 1, "a single non common dim is expected to project to a single output dim");
        outputSliceDim = outputDims.front();
    }
    // Any slice size, including the minimal slice size, must be aligned to granularity and at least 1 geometry size
    // For non common dim - this function is called only for non filtered output dims (1:1 mapping of input to output
    // elements), thus the alignment is valid both for output and for input tensors.
    TSize dimAlignment = getDimAlignmentInElements(inputIdx, outputSliceDim, dimGranularity);
    // Not clipping to actual dim size, otherwise we lose the ability to multiply granularity by the caller
    return dimAlignment;
}

NodeStrategyPtr
NodeSolver::solveNode(const NodePtr& mmeNode, const BundleSolutionConstraints& constraints)
{
    std::unique_ptr<NodeSolver> mmeSolver = MmeNodeSolver::getSpecificMmeSolver(mmeNode);
    HB_ASSERT_PTR(mmeSolver);  // This function is expected only for supported node types
    return mmeSolver->solveNode(constraints);
}

std::unique_ptr<MmeNodeSolver> MmeNodeSolver::getSpecificMmeSolver(const NodePtr& node)
{
    std::unique_ptr<MmeNodeSolver> mmeSolver;
    if (node->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        mmeSolver = std::make_unique<MaskedBatchGemmNodeSolver>(node);
    }
    else if (Node::isBatchGemmNode(node))
    {
        mmeSolver = std::make_unique<BatchGemmNodeSolver>(node);
    }
    else if (Node::isGemmNode(node))
    {
        mmeSolver = std::make_unique<GemmNodeSolver>(node);
    }
    else if (node->getNodeType() == Node::TYPE_CONVOLUTION)
    {
        mmeSolver = std::make_unique<ConvNodeSolver>(node);
    }
    else if (Node::isDedxNode(node))
    {
        mmeSolver = std::make_unique<DedxNodeSolver>(node);
    }
    else if (node->getNodeType() == Node::TYPE_DEDW)
    {
        mmeSolver = std::make_unique<DedwNodeSolver>(node);
    }
    return mmeSolver;
}

TSize MmeNodeSolver::getCommonDimMinSliceSize(const pSlicedOperand& slicedOperandA,
                                              Dim                   slicingDim,
                                              TSize                 slicingDimGranularity) const
{
    HB_ASSERT(false, "Common dim chunk size for MME node with unspecific type is unexpected");
    return 0;
};

void MmeNodeSolver::setTraversalPatternAndMapping(StrategySlicingData& data)
{
    HB_ASSERT(false, "Traversal patterns and mapping for MME node with unspecific type is unexpected");
}

// Return the number of outputSliceDim elements that fit in one MME geometry
TSize MmeNodeSolver::getOutDimAlignmentForMmeUtil(unsigned slicedInputIdx, unsigned dim)
{
    // For simple axis, which has a single dim, or the mme geometry is relevant for the dim (like bgemm concurrency)
    // axis with multiple dims should override this func
    unsigned mmeGeometryAxisElements = getMmeGeometrySizeInElementsToAlign(slicedInputIdx, dim);
    return mmeGeometryAxisElements;
}

// Returns the MME geometry axis size in elements, which corresponds to the given input index.
// The geometry is selected based on the MME node recommended configuration
// Generally for  MME node - operand A matches output height, operand B matches output width
unsigned MmeNodeSolver::getMmeGeometrySizeInElementsToAlign(unsigned slicedInputIdx, Dim slicedDim)
{
    // TODO SW-71550 - look for the smallest geometry with the same utilization
    return MmeBrainProxy::getRecommendedGeometryAxisElements(m_node, slicedInputIdx);
}

TSize MmeNodeSolver::getOperandAxisElements(const pSlicedOperand& operand, unsigned slicedInputIdx, unsigned slicedDim)
{
    return operand->finalShape[slicedDim];
}

// Checks if slicing the operand on the given slicing dimension to any size will surely reduce utilization
bool MmeNodeSolver::slicingReducesUtilization(SlicingStrategyPtr& strategy,
                                              unsigned            outputSliceDim,
                                              unsigned            slicedInputIdx)
{
    pSlicedOperand& outputOperand      = strategy->getSlicingData().masterOperand;
    TSize           outputAxisElements = getOperandAxisElements(outputOperand, slicedInputIdx, outputSliceDim);

    // No need to slice - if the full operand fits SRAM, and
    // The axis is already smaller than MME geometry, or trying to maximize slice size and use logical ROI
    return (outputAxisElements <= getMmeGeometrySizeInElementsToAlign(slicedInputIdx, outputSliceDim));
}

bool MmeNodeSolver::sliceNodeOnSingleDim(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints)
{
    const unsigned slicedInputIdx = getInputIndexToSlice(constraints.tensorsInSram, m_node);
    const auto     slicingDims    = getInputSlicingDims(slicedInputIdx);
    HB_ASSERT(!slicingDims.empty(),
              "Unexpected failure to find a dimension on which to slice {}",
              m_node->getInput(slicedInputIdx)->getName());

    const unsigned inputSliceDim = slicingDims.front();  // slicing here on single dim
    const bool     fitsSram      = sliceNodeOnSingleNonCommonDim(strategy, inputSliceDim, constraints);
    if (fitsSram && constraints.sliceNonMasterOperand)
    {
        // Slice the other operand without placing in SRAM - doesn't change SRAM capacity
        const unsigned nonMasterInputIdx = 1 - getInputIndexToSlice(constraints.tensorsInSram, m_node);
        sliceNonMasterOperand(strategy,
                              nonMasterInputIdx,
                              getSlicingGranularity(nonMasterInputIdx, constraints.slicingGranularityPerTensor));
    }
    return fitsSram;
}

bool MmeNodeSolver::sliceNodeOnMultipleDims(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints)
{
    return false;
}

bool MmeNodeSolver::sramPlacementIsHighPrio(const NodeStrategyPtr&           strategy,
                                            const BundleSolutionConstraints& constraints) const
{
    const auto& slicingData = strategy->getSlicingData();
    bool in0SlicedOnMultDims =
        SlicedOperandUtils::getNumOfSlicedDims(slicingData.getSlicedOperand(m_node->getInput(0))) > 1;
    bool in1SlicedOnMultDims =
        SlicedOperandUtils::getNumOfSlicedDims(slicingData.getSlicedOperand(m_node->getInput(1))) > 1;
    return in0SlicedOnMultDims || in1SlicedOnMultDims || !constraints.isSingleNodeBundle;
}

NodeStrategyPtr MmeNodeSolver::solveNode(const BundleSolutionConstraints& constraints)
{
    LOG_TRACE(SRAM_SLICE, "{}: {}", __PRETTY_FUNCTION__, m_node->getNodeName());
    HB_ASSERT(HabanaGraph::runsOnMME(m_node), "Only MME nodes are supported by this solver");
    if (!isNodeSupported())
    {
        return nullptr;
    }
    NodeStrategyPtr   strategy       = createInitialStrategy();
    bool              fitsSram       = sliceNodeOnSingleDim(strategy, constraints);
    if (!fitsSram && constraints.canSliceMultipleDims)
    {
        // Try to slice on more dims. Reset the strategy as the function slices from scratch
        resetStrategySlicingData(strategy);
        fitsSram = sliceNodeOnMultipleDims(strategy, constraints);
    }
    // TODO [SW-158384]: resolve the regression that occurs when reducePerformanceToFitSram is called for single node
    // bundles that cannot be sliced on multiple dims
    if (!fitsSram && sramPlacementIsHighPrio(strategy, constraints))
    {
        // Try without CL alignment / single buffer
        fitsSram = reducePerformanceToFitSram(strategy, constraints.availableSramBytes);
    }
    if (!fitsSram)
    {
        LOG_WARN(SRAM_SLICE, "{}: Can't slice node to fit SRAM {}", HLLOG_FUNC, m_node->getNodeName());
        resetStrategySlicingData(strategy);
    }
    unsigned slicedInputIdx = getInputIndexToSlice(constraints.tensorsInSram, m_node);
    projectSlicingOnNodeOperands(strategy, slicedInputIdx);
    finalizeStrategy(strategy);
    validateStrategy(strategy);
    return strategy;
}

NodeStrategyPtr MmeNodeSolver::createInitialStrategy()
{
    NodeStrategyPtr strategy = SlicingStrategy::createStrategy(*CompilationHalReader::getHalReader(), m_node);
    // TODO - is there a better way to do that? modify StrategySlicingData c'tor?
    if (m_node->getInput(0) == m_node->getInput(1))
    {
        StrategySlicingData& data = strategy->getSlicingData();
        HB_ASSERT(data.bundleTensors.size() == 2 || (Node::isDedxNode(m_node) && data.bundleTensors.size() == 3),
                  "Unexpected number of bundle tensor created ({}) for node {} (type: {})",
                  data.bundleTensors.size(),
                  m_node->getNodeName(),
                  m_node->getNodeTypeStr());
        data.bundleTensors.pop_back();
    }
    setTraversalPatternAndMapping(strategy->getSlicingData());
    return strategy;
}

void MmeNodeSolver::finalizeStrategy(NodeStrategyPtr& strategy)
{
    strategy->alignNumBuffers();
    strategy->alignWalkingPattern();
    strategy->calculateMetrics();
}

void MmeNodeSolver::validateStrategy(const SlicingStrategyPtr& strategy)
{
    validateSlicesSize(strategy);
}

void MmeNodeSolver::validateSlicesSize(const SlicingStrategyPtr& strategy)
{
    for (const auto& t : m_node->getOperands())
    {
        if (!t) continue;

        pSlicedOperand operand = strategy->getSlicingData().getSlicedOperand(t);
        HB_ASSERT_PTR(operand);
        // validate slice sizes are within the original tensor bounds
        for (auto dim = 0; dim < t->getDim(); dim++)
        {
            HB_ASSERT(operand->chunkDimensions[dim] <= operand->finalShape[dim],
                      "slice size is larger than original dim size {} > {} for {} dim {}",
                      operand->chunkDimensions[dim],
                      operand->finalShape[dim],
                      t->getName(),
                      dim);
        }
    }
}

std::vector<unsigned> MmeNodeSolver::getInputsIdicesInSram(const TensorSet& tensorsInSram, const NodePtr& node)
{
    for (auto t : tensorsInSram)
        LOG_DEBUG(SRAM_SLICE, "{}: Tensor in SRAM: {}", HLLOG_FUNC, t->getName());
    std::vector<unsigned> inputIdxInSramList;
    for (unsigned inputIdx = 0; inputIdx < node->getNumInputs(); inputIdx++)
    {
        const TensorPtr& input = node->getInput(inputIdx);
        if (inputIdx == 1 && input == node->getInput(0)) continue;  // No need to count the same tensor twice

        if (tensorsInSram.find(input) != tensorsInSram.end())
        {
            inputIdxInSramList.push_back(inputIdx);
        }
    }
    return inputIdxInSramList;
}

bool MmeNodeSolver::isOutputInSram(const TensorSet& tensorsInSram)
{
    return (tensorsInSram.find(m_node->getOutput(0)) != tensorsInSram.end());
}

unsigned MmeNodeSolver::getInputIndexToSlice(const TensorSet& tensorsInSram, const NodePtr& node)
{
    // Get the sliced operand from the constraints - check if any of the operands is in SRAM
    std::vector<unsigned> sramInputsIndices = getInputsIdicesInSram(tensorsInSram, node);
    HB_ASSERT(sramInputsIndices.size() == 1,
              "Assuming exactly one input is placed in SRAM for MME. Found: ({}).",
              toString(sramInputsIndices, ','));
    return sramInputsIndices.front();
}

TensorGranularity MmeNodeSolver::getSlicingGranularity(unsigned                    slicedInputIdx,
                                                       const GranularityPerTensor& slicingGranularityMap)
{
    TensorPtr slicedInput = m_node->getInput(slicedInputIdx);
    HB_ASSERT(slicingGranularityMap.find(slicedInput) != slicingGranularityMap.end(),
              "Tensor in SRAM must have granularity data. Missing granularity for node {} tenosr {}",
              m_node->getNodeName(),
              slicedInput->getName());
    return slicingGranularityMap.at(slicedInput);
}

// Calculates the sliced dim granularity, taking MME util and the given granularity into account.
// If outSliceDim is valid, calculates for this output dim, otherwise calc for the input common dim
// If the caller calls this function on a non filtered dim (1:1 mapping of input to output), the returned
// value is valid for the corresponding input tensor dim as well.
TSize MmeNodeSolver::getDimAlignmentInElements(unsigned inputIdx, std::optional<Dim> outSliceDim, TSize dimGranularity)
{
    TSize elementsForMmeUtil = 0;
    if (!outSliceDim)
    {
        elementsForMmeUtil = getMmeMinCDInElementsForPartials();
    }
    else
    {
        // Get output dim size for best MME utilization.
        // The size is the same for input dim size of non filtered slicing dim (1:1 mapping)
        elementsForMmeUtil = getOutDimAlignmentForMmeUtil(inputIdx, *outSliceDim);
    }
    TSize dimAlignment = alignDimSizeToGranularity(elementsForMmeUtil, dimGranularity);
    LOG_DEBUG(SRAM_SLICE,
              "Dim {}: elementsForMmeUtil {} slicingGranularity {} => dimAlignment {}",
              (outSliceDim ? std::to_string(*outSliceDim) : "CD"),
              elementsForMmeUtil,
              dimGranularity,
              dimAlignment);
    return dimAlignment;
}

TSize MmeNodeSolver::alignDimSizeToGranularity(TSize dimSize, TSize granularity)
{
    // TODO SW-72188
    // select inputs aligned size by projecting the output aligned size on the input. output aligned size should
    // be selected as maximal MME utilization, which may be achieved by aligning to more than 1 geometry. Need to find
    // the util max for several geometries given the slicing granularity requirement. We can also try to align to
    // geometry the final slice instead of the minimal slice as granularity, and optimize the utilization of the entire
    // solution, including the last slice, which might be smaller. If there are only a few slices it might be important.
    TSize dimAlignment = round_to_multiple(dimSize, granularity);
    return dimAlignment;
}

// Returns true if the operand is placed successfully in SRAM
bool MmeNodeSolver::sliceNodeOnSingleNonCommonDim(SlicingStrategyPtr&              strategy,
                                                  Dim                              inputSliceDim,
                                                  const BundleSolutionConstraints& constraints)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const unsigned slicedInputIdx           = getInputIndexToSlice(constraints.tensorsInSram, m_node);
    const unsigned c_minSliceNumForPipeline = GCFG_NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING.value();

    pSlicedOperand  inputOperand  = strategy->getSlicingData().getSlicedOperand(m_node->getInput(slicedInputIdx));
    pSlicedOperand& outputOperand = strategy->getSlicingData().masterOperand;
    // set the input in SRAM before calculateMetrics is called
    strategy->setInputIsInSRAM(slicedInputIdx, true);
    strategy->setOutputIsInSRAM(isOutputInSram(constraints.tensorsInSram));

    TSize inputDimSizeInElements = inputOperand->finalShape[inputSliceDim];
    if (inputDimSizeInElements == 1)
    {
        bool fitsSRAM = strategy->calculateMetrics().SRAMCapacity <= constraints.availableSramBytes;
        LOG_DEBUG(SRAM_SLICE,
                  "Input operand {} is not sliceable on slicing dim {} (has a single element), fitsSRAM={}",
                  inputOperand->originalTensor->getName(),
                  inputSliceDim,
                  fitsSRAM);
        return fitsSRAM;
    }
    auto slicedInput = m_node->getInput(slicedInputIdx);
    auto outputDims  = getTensorMatchingSlicedDims(m_node, slicedInput, {inputSliceDim}, m_node->getOutput(0));
    HB_ASSERT(outputDims.size() == 1, "a single non common dim is expected to project to a single output dim");
    unsigned outputSliceDim = outputDims.front();
    LOG_DEBUG(SRAM_SLICE, "{}: inputSliceDim {}, outputSliceDim {}", HLLOG_FUNC, inputSliceDim, outputSliceDim);

    // No need to slice - if the full operand fits SRAM, and the axis is already smaller than MME geometry
    if (strategy->calculateMetrics().SRAMCapacity <= constraints.availableSramBytes &&
        slicingReducesUtilization(strategy, outputSliceDim, slicedInputIdx))
    {
        LOG_DEBUG(SRAM_SLICE, "Operand fits completely in SRAM and smaller than MME geometry, no slicing");
        return true;
    }

    // Any slice size, including the minimal slice size, must be aligned to granularity and at least 1 geometry size
    // In case sliceSizeAlignment was set by the caller, the node has to consider external nodes constraints,
    // so it should use the given alignment. If not set - calculate its own required alignment.
    const auto& slicingGranularity = getSlicingGranularity(slicedInputIdx, constraints.slicingGranularityPerTensor);
    TSize       dimAlignment       = 0;
    const auto& bundleAlignment    = constraints.sharedOperandSlicingDimsAlignment;
    if (bundleAlignment.empty())
    {
        dimAlignment = getDimAlignmentInElements(slicedInputIdx, outputSliceDim, slicingGranularity[inputSliceDim]);
    }
    else
    {
        HB_ASSERT(bundleAlignment.find(inputSliceDim) != bundleAlignment.end(),
                  "Bundle dim granularity exists, but missing for dim {}",
                  inputSliceDim);
        dimAlignment = bundleAlignment.at(inputSliceDim);
    }

    // Limit slicing factor to make sure slice includes at least 1 element
    unsigned slicingFactor    = std::min(inputDimSizeInElements, (TSize)c_minSliceNumForPipeline);
    TSize    alignedSliceSize = 0;

    // Set the flag before calling to calculateMetrics
    inputOperand->alignWithCacheLine = constraints.canAlignInputToCL;

    do
    {
        TSize sliceDimSizeInElements    = std::floor((float)inputDimSizeInElements / (float)slicingFactor);
        alignedSliceSize                = round_to_multiple(sliceDimSizeInElements, dimAlignment);
        LOG_DEBUG(SRAM_SLICE,
                  "sliceDimSizeInElements {}, alignedSliceSize {}, slicingFactor {}",
                  sliceDimSizeInElements,
                  alignedSliceSize,
                  slicingFactor);
        inputOperand->chunkDimensions[inputSliceDim]   = std::min(alignedSliceSize, inputDimSizeInElements);
        HB_ASSERT(inputOperand->chunkDimensions[inputSliceDim] <= outputOperand->finalShape[outputSliceDim],
                  "projecting input tensor on output tensor excceeds output tensor size");
        outputOperand->chunkDimensions[outputSliceDim] = inputOperand->chunkDimensions[inputSliceDim];
        // Set double buffer after the operand is sliced, otherwise it is ignored for trivial slicing
        strategy->setDoubleBuffer(true);
        slicingFactor++;
    } while (strategy->calculateMetrics().SRAMCapacity > constraints.availableSramBytes &&
             alignedSliceSize > dimAlignment);

    auto requiredBytes = strategy->calculateMetrics().SRAMCapacity;
    bool fitsSram      = requiredBytes <= constraints.availableSramBytes;
    LOG_DEBUG(SRAM_SLICE, "{}: {} MB {} fit SRAM", HLLOG_FUNC, bToMb(requiredBytes), (fitsSram ? "does" : "doesn't"));
    return fitsSram;
}

bool MmeNodeSolver::reducePerformanceToFitSram(NodeStrategyPtr& strategy, unsigned availableSram)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);

    if (strategy->calculateMetrics().SRAMCapacity <= availableSram)
    {
        LOG_TRACE(SRAM_SLICE, "Strategy requires {} MB - fits SRAM", bToMb(strategy->getMetrics().SRAMCapacity));
        return true;
    }
    if (retryWithoutCLAlign(strategy, availableSram)) return true;

    if (retryWithoutDoubleBuffer(strategy, availableSram)) return true;
    // Resume double buffer for additional retries
    strategy->setDoubleBuffer(true);
    strategy->alignNumBuffers();

    if (retryWithoutOutputInSram(strategy, availableSram)) return true;

    // Finally try again with single buffer and output not in SRAM
    if (retryWithoutDoubleBuffer(strategy, availableSram)) return true;

    LOG_WARN(SRAM_SLICE, "Can't fit strategy of node {} in SRAM", m_node->getNodeName());
    return false;
}

bool MmeNodeSolver::retryWithoutCLAlign(NodeStrategyPtr& strategy, unsigned availableSram)
{
    pSlicedOperand inputOperand0 = strategy->getSlicingData().getSlicedOperand(m_node->getInput(0));
    pSlicedOperand inputOperand1 = strategy->getSlicingData().getSlicedOperand(m_node->getInput(1));
    if (inputOperand0->alignWithCacheLine || inputOperand1->alignWithCacheLine)
    {
        LOG_WARN(SRAM_SLICE,
                 "Can't fit {} MB operand in SRAM with CL alignment",
                 bToMb(strategy->getMetrics().SRAMCapacity));

        // Try without input CL alignment
        inputOperand0->alignWithCacheLine = false;
        inputOperand1->alignWithCacheLine = false;
        if (strategy->calculateMetrics().SRAMCapacity <= availableSram)
        {
            LOG_DEBUG(SRAM_SLICE,
                      "Strategy without CL alignment requires {} MB, fits SRAM",
                      bToMb(strategy->getMetrics().SRAMCapacity));
            return true;
        }
    }
    return false;
}

bool MmeNodeSolver::retryWithoutOutputInSram(NodeStrategyPtr& strategy, unsigned availableSram)
{
    pSlicedOperand outputOperand = strategy->getSlicingData().masterOperand;
    if (outputOperand->resideInSRAM)
    {
        LOG_WARN(SRAM_SLICE,
                 "Can't fit {} MB operands in SRAM with out operand in SRAM",
                 bToMb(strategy->calculateMetrics().SRAMCapacity));

        // Try without output in SRAM
        strategy->setOutputIsInSRAM(false);
        if (strategy->calculateMetrics().SRAMCapacity <= availableSram)
        {
            LOG_DEBUG(SRAM_SLICE,
                      "Strategy without output in SRAM requires {} MB, fits SRAM",
                      bToMb(strategy->calculateMetrics().SRAMCapacity));
            return true;
        }
    }
    return false;
}

bool MmeNodeSolver::retryWithoutDoubleBuffer(NodeStrategyPtr& strategy, unsigned availableSram)
{
    LOG_WARN(SRAM_SLICE,
             "Can't fit {} MB operand to SRAM with double buffer",
             bToMb(strategy->getMetrics().SRAMCapacity));

    // TODO SW-108841 - try to enable single buf strategy for dedw and get rid of singleBufferStrategyAllowed
    if (singleBufferStrategyAllowed())
    {
        // Try with single buffer
        strategy->setDoubleBuffer(false);
        if (strategy->calculateMetrics().SRAMCapacity <= availableSram)
        {
            LOG_DEBUG(SRAM_SLICE,
                      "Strategy with single buffer requires {} MB, fits SRAM",
                      bToMb(strategy->getMetrics().SRAMCapacity));
            return true;
        }
    }
    return false;
}

bool MmeNodeSolver::isDuplicateInputMme()
{
    return m_node->getInput(0) == m_node->getInput(1);
}

bool MmeNodeSolver::isNodeSupported()
{
    if (isDuplicateInputMme())
    {
        LOG_WARN(SRAM_SLICE,
                 "Unsupported node - both its inputs are the same tensor for node {}",
                 m_node->getNodeName());
        return false;
    }
    return true;
}

// Returns true if the operand is placed successfully in SRAM
bool MmeNodeSolver::sliceNodeOnSingleCommonDim(SlicingStrategyPtr&              strategy,
                                               const BundleSolutionConstraints& constraints)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const unsigned slicedInputIdx = getInputIndexToSlice(constraints.tensorsInSram, m_node);
    // This value is used to stop trying to double the size of the slices. After doubling, it is assumed that the new
    // slice size is <= the full tensor size. For that reason, the gcfg should not allow doubling the size when there
    // are less then 3 slices (slices may not have the same size, so 2 slices where the first is bigger will still raise
    // an assert). It is recommended to leave this gcfg on 4 or above.
    const unsigned c_minSliceNumForPipeline = std::max(3ul, GCFG_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING.value());

    pSlicedOperand operandA     = strategy->getSlicingData().getSlicedOperand(m_node->getInput(0));
    pSlicedOperand operandB     = strategy->getSlicingData().getSlicedOperand(m_node->getInput(1));
    auto           slicingDims0 = getInputSlicingDims(0);
    auto           slicingDims1 = getInputSlicingDims(1);
    HB_ASSERT(!slicingDims0.empty() && !slicingDims1.empty(),
              "Unexpected failure to find a dimension on which to slice inputs");
    Dim   inputASliceDim = slicingDims0.front();  // slicing here on single dim
    Dim   inputBSliceDim = slicingDims1.front();
    auto& slicingData    = strategy->getSlicingData();

    // set the input and output in SRAM before calculateMetrics is called
    strategy->setInputIsInSRAM(slicedInputIdx, true);
    strategy->setOutputIsInSRAM(true);

    // No need to slice - if the full operand fits SRAM.
    // In case the output is FP32 slice anyway for better pipelining.
    if ((strategy->calculateMetrics().SRAMCapacity <= constraints.availableSramBytes) &&
        (slicingData.masterOperand->originalTensor->getElementType() != syn_type_float))
    {
        LOG_DEBUG(SRAM_SLICE, "Operand fits completely in SRAM, no slicing");
        return true;
    }
    const auto& slicingGranularity = getSlicingGranularity(slicedInputIdx, constraints.slicingGranularityPerTensor);
    TSize sliceChunkSize = getCommonDimMinSliceSize(operandA, inputASliceDim, slicingGranularity[inputASliceDim]);

    LOG_TRACE(SRAM_SLICE,
              "Slice common dim:  OperandA on dim: {}, operandB on dim: {}, to chunk size  {}",
              inputASliceDim,
              inputBSliceDim,
              sliceChunkSize);

    pSlicedOperand operandInSram = strategy->getSlicingData().getSlicedOperand(m_node->getInput(slicedInputIdx));
    // Set the flag before calling to calculateMetrics
    operandInSram->alignWithCacheLine = constraints.canAlignInputToCL;

    pSlicedOperand& sliceOpA                  = slicingData.bundleTensors[0];
    sliceOpA->chunkDimensions[inputASliceDim] = sliceChunkSize;

    pSlicedOperand& sliceOpB                  = slicingData.bundleTensors[1];
    sliceOpB->chunkDimensions[inputBSliceDim] = sliceChunkSize;

    // Set double buffer after the operand is sliced, otherwise it is ignored for trivial slicing
    strategy->setDoubleBuffer(true);

    // Reduction is performed in high precision data type.
    slicingData.masterOperand->finalElementType = SlicedOperandUtils::getTypeForPartials(slicingData.masterOperand->finalElementType);

    TSize nextSliceChunkSize = sliceChunkSize;
    while ((strategy->calculateMetrics().SRAMCapacity < constraints.availableSramBytes) &&
           (SlicedOperandUtils::nofSlices(sliceOpA) >= c_minSliceNumForPipeline))
    {
        sliceChunkSize = nextSliceChunkSize;  // Save the valid slice size from prev. iteration

        // Try larger slice size
        nextSliceChunkSize = sliceChunkSize * 2;
        // sliceChunkSize was chosen to be a multiple of the granularity (see getCommonDimMinSliceSize).
        HB_ASSERT(nextSliceChunkSize % slicingGranularity[inputASliceDim] == 0,
                  "Invalid slice chunk size, not a multiple of the slicing granularity");
        // We have at least 4 slices (started from >= 8 slices) - chunk size must be smaller than full tensor dim size.
        HB_ASSERT(nextSliceChunkSize <= sliceOpA->finalShape[inputASliceDim],
                  "Invalid slice chunk size, exceeds original size");
        HB_ASSERT(nextSliceChunkSize <= sliceOpB->finalShape[inputBSliceDim],
                  "Invalid slice chunk size, exceeds original size");
        sliceOpA->chunkDimensions[inputASliceDim] = nextSliceChunkSize;
        sliceOpB->chunkDimensions[inputBSliceDim] = nextSliceChunkSize;
    }

    // Set chunkDimensions to the last valid slice size
    sliceOpA->chunkDimensions[inputASliceDim] = sliceChunkSize;
    sliceOpB->chunkDimensions[inputBSliceDim] = sliceChunkSize;
    slicingData.numCommonDimSlices            = SlicedOperandUtils::nofSlices(sliceOpA);

    auto requiredBytes = strategy->calculateMetrics().SRAMCapacity;
    bool fitsSram      = requiredBytes <= constraints.availableSramBytes;
    LOG_DEBUG(SRAM_SLICE, "{}: {} MB {} fit SRAM", HLLOG_FUNC, bToMb(requiredBytes), (fitsSram ? "does" : "doesn't"));
    return fitsSram;
}

void MmeNodeSolver::resetStrategySlicingData(SlicingStrategyPtr& strategy)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    StrategySlicingData& slicingData = strategy->getSlicingData();

    pSlicedOperand operandA = slicingData.getSlicedOperand(m_node->getInput(0));
    pSlicedOperand operandB = slicingData.getSlicedOperand(m_node->getInput(1));
    operandA->resetSlicingData();
    operandB->resetSlicingData();
    slicingData.masterOperand->resetSlicingData();
    slicingData.numCommonDimSlices = 1;
    strategy->setDoubleBuffer(false);
}

TSize MmeNodeSolver::getMmeMinCDInElementsForPartials() const
{
    // When slicing on the common dim we modify the output type to be fp32 (for accuracy reasons),
    // thus it is used in the call to getMmeMinCDInElements.
    return std::max(CompilationHalReader::getHalReader()->getMmeMinCDInElements(
                        m_node->getInput(0)->getElementType(),
                        SlicedOperandUtils::getTypeForPartials(m_node->getOutput(0)->getElementType())),
                    SlicingBrain::knobs.minCDSizeForPartials);
}

bool MmeNodeSolver::canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                               const bool              isSingleNodeBundle)
{
    return false;
}

TSize DedwNodeSolver::getCommonDimMinSliceSize(const pSlicedOperand& slicedOperandA,
                                               Dim                   slicingDim,
                                               TSize                 slicingDimGranularity) const
{
    // multiply the common dims up to the sliced dim, excluding it
    const auto& chunkDims  = slicedOperandA->chunkDimensions;
    const auto& commonDims = m_dimController.commonDimOperandA();
    HB_ASSERT(std::find(commonDims.begin(), commonDims.end(), slicingDim) != commonDims.end(),
              "invalid common slicing dim");
    TSize cdSizeForSingleSlicedDimElement =
        multiplyElements(chunkDims.data() + commonDims.front(), chunkDims.data() + slicingDim);
    // Prefer min CD slice size above SlicingBrain::knobs.minCDSizeForPartials - div to get the number of elements from
    // the slicing dim which sum to the required minimum.
    // TODO SW-108711 - the min CD size should be at least 1KB for gaudi2. Find a better knob value and set it for
    // gaudi2.
    const TSize desiredMin = div_round_up(getMmeMinCDInElementsForPartials(), cdSizeForSingleSlicedDimElement);
    // Align desiredMin to granularity
    const TSize alignedMinToGranularity = round_to_multiple(desiredMin, slicingDimGranularity);
    // Limit by the actual tensor size
    const TSize sliceChunk = std::min(alignedMinToGranularity, slicedOperandA->finalShape[slicingDim]);
    return sliceChunk;
}

void DedwNodeSolver::setTraversalPatternAndMapping(StrategySlicingData& data)
{
    // DeDw swaps operands, so the default traverse pattern should be top to bottom
    data.traversalPattern = SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D;
    data.setOutputSliceBackwardMapping(MMESliceMapper::mapOutputToInputs(m_node,
                                                                         data.getSlicedOperand(m_node->getInput(0)),
                                                                         data.getSlicedOperand(m_node->getInput(1)),
                                                                         data.masterOperand,
                                                                         nullptr));
}

std::vector<Dim> DedwNodeSolver::getInputSlicingDims(unsigned inputIdx)
{
    HB_ASSERT((inputIdx == 0 || inputIdx == 1), "MME node producer is expected for operands A or B");
    unsigned sliceDim = m_dimController.commonDimsForOperand(inputIdx).back();
    HB_ASSERT(sliceDim < m_node->getInput(inputIdx)->getDim(), "slicing dim out of bounds");
    return {sliceDim};
}

bool DedwNodeSolver::sliceNodeOnSingleDim(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints)
{
    return sliceNodeOnSingleCommonDim(strategy, constraints);
}

bool DedwNodeSolver::canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                                const bool              isSingleNodeBundle)
{
    return isSingleNodeBundle;
}

bool DedwNodeSolver::sliceNodeOnMultipleDims(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const unsigned slicedInputIdx = getInputIndexToSlice(constraints.tensorsInSram, m_node);

    auto&          slicingData = strategy->getSlicingData();
    pSlicedOperand dyOperand   = slicingData.getSlicedOperand(m_node->getInput(TENSOR_DEDY));
    pSlicedOperand xOperand    = slicingData.getSlicedOperand(m_node->getInput(TENSOR_X_BWD));

    // check dY common dim min size, as X common dim is divided by a function of the stride and the kernel
    if (!canSliceSpatially(dyOperand))
    {
        return false;
    }

    Dim batchDim = m_dimController.batchDim().back();
    Dim sliceDim = batchDim - 1;  // first spatial dim - same for both DEDW inputs

    TSize sliceSize = getMinSpatialDimSize(dyOperand, sliceDim);
    if (sliceSize >= dyOperand->finalShape[sliceDim])
    {
        LOG_DEBUG(SRAM_SLICE, "Min slice size isn't smaller than dim original size - can't slice");
        return false;
    }

    // Slice the external common dim (batch) to minimal size
    dyOperand->chunkDimensions[batchDim] = 1;
    xOperand->chunkDimensions[batchDim]  = 1;

    // This value is used to stop trying to double the size of the slices. After doubling, it is assumed that the new
    // slice size is <= the full tensor size. For that reason, the gcfg should not allow doubling the size when there
    // are less then 3 slices (slices may not have the same size, so 2 slices where the first is bigger will still raise
    // an assert). It is recommended to leave this gcfg on 4 or above.
    TSize minSliceNumForPipeline =
        std::max(GCFG_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING.value() / dyOperand->finalShape[batchDim], 3UL);

    // Set the input and output in SRAM, and CL align flag before calculateMetrics is called
    strategy->setInputIsInSRAM(slicedInputIdx, true);
    strategy->setOutputIsInSRAM(true);  // for partials calc optimization
    slicingData.getSlicedOperand(m_node->getInput(slicedInputIdx))->alignWithCacheLine = constraints.canAlignInputToCL;

    // Slice the operands to the minimal slice size
    sliceCommonSpatialDim(sliceDim, sliceSize, dyOperand, xOperand);

    // Set double buffer after the operand is sliced, otherwise it is ignored for trivial slicing
    strategy->setDoubleBuffer(true);
    // Reduction is performed in high precision data type.
    slicingData.masterOperand->finalElementType = SlicedOperandUtils::getTypeForPartials(slicingData.masterOperand->finalElementType);
    // Try to reduce the number of slices as long as they fit SRAM and enough slices remain
    TSize nextSliceSize = sliceSize;
    while ((strategy->calculateMetrics().SRAMCapacity < constraints.availableSramBytes) &&
           (SlicedOperandUtils::nofSlices(dyOperand) >= minSliceNumForPipeline) &&
           nextSliceSize < dyOperand->finalShape[sliceDim])
    {
        // Save the valid slice size from prev. iteration
        sliceSize = nextSliceSize;
        // Try larger slice size
        nextSliceSize = sliceSize * 2;
        sliceCommonSpatialDim(sliceDim, nextSliceSize, dyOperand, xOperand);
    }

    // Slice to the last valid slice size
    sliceCommonSpatialDim(sliceDim, sliceSize, dyOperand, xOperand);
    slicingData.numCommonDimSlices = SlicedOperandUtils::nofSlices(dyOperand);

    // Set the post slicing handler to handle the sliced dedw nodes padding
    xOperand->postSlicingHandler = std::make_shared<PostSlicingConvHandler>();

    auto requiredBytes = strategy->calculateMetrics().SRAMCapacity;
    bool fitsSram      = requiredBytes <= constraints.availableSramBytes;
    LOG_DEBUG(SRAM_SLICE, "{}: {} MB {} fit SRAM", HLLOG_FUNC, bToMb(requiredBytes), (fitsSram ? "does" : "doesn't"));
    return fitsSram;
}

bool DedwNodeSolver::canSliceSpatially(const pSlicedOperand& inputOperand)
{
    Dim batchDim = m_dimController.batchDim().back();
    Dim sliceDim = batchDim - 1;  // first spatial dim - same for both DEDW inputs

    if (!m_conv->isSpatialSlicingSupported(sliceDim))
    {
        LOG_TRACE(SRAM_SLICE, "{}: Spatial slice unsupported for dim {}", HLLOG_FUNC, sliceDim);
        return false;
    }

    // Make sure the common dim can be sliced on batch to 1 and also slice next dim. If the common dim is too small for
    // partials utilization and batch can't be sliced to 1 - can't slice the spatial dim as well.
    TSize mmeMinBatchSize = getCommonDimMinSliceSize(inputOperand, batchDim, 1);
    if (mmeMinBatchSize > 1)
    {
        LOG_DEBUG(SRAM_SLICE, "Batch can't be sliced to 1 - the common dim is too small to slice on spatial");
        return false;
    }
    return true;
}

TSize DedwNodeSolver::getMinSpatialDimSize(const pSlicedOperand& dyOperand, Dim spatialSliceDim)
{
    TSize mmeMinSliceSize = getCommonDimMinSliceSize(dyOperand, spatialSliceDim, 1);
    // Minimal dY slice size is set to be larger than the padding before the dimension – to make sure the first X slice
    // size is not 0. Minimal dY slice size is set to be larger than the overlap – to make sure a slice includes new
    // lines beyond the previous slice.
    const ConvParamsIndices convIdx       = ConvBaseNode::dimIndexToConvParamsIndices(spatialSliceDim);
    int                     paddingBefore = m_conv->getConvolutionParams().padding[convIdx.paddingBeforeIndex];
    // TODO SW-108439 - padding is for x - need to project to get better min
    int overlap = m_conv->getInputROIOverlapForDim(TENSOR_X_BWD, spatialSliceDim);
    // Overlap/offset might be negative, and add 1 to make sure the slice is larger
    TSize convMinSliceSize = std::max(std::max(overlap, paddingBefore), 0) + 1;
    TSize minSliceSize     = std::max(mmeMinSliceSize, convMinSliceSize);

    LOG_DEBUG(SRAM_SLICE, "Slice dim {}: mme min {}, conv min {}", spatialSliceDim, mmeMinSliceSize, convMinSliceSize);

    return minSliceSize;
}

void DedwNodeSolver::sliceCommonSpatialDim(unsigned        slicingDim,
                                           TSize           dySliceSize,
                                           pSlicedOperand& dyOperand,
                                           pSlicedOperand& xOperand)
{
    if (dySliceSize > dyOperand->finalShape[slicingDim])
    {
        return;
    }

    LOG_DEBUG(SRAM_SLICE, "{}: Slice dim {}: sliceSize {}", HLLOG_FUNC, slicingDim, dySliceSize);

    const ConvParamsIndices convIdx = ConvBaseNode::dimIndexToConvParamsIndices(slicingDim);

    int paddingBefore = m_conv->getConvolutionParams().padding[convIdx.paddingBeforeIndex];
    int paddingAfter  = m_conv->getConvolutionParams().padding[convIdx.paddingAfterIndex];

    // Slice dY operand on the first spatial dimension
    dyOperand->chunkDimensions[slicingDim] = dySliceSize;
    // Find the corresponding X operand slice size, based on the convolution parameters and the output slice size.
    TensorShape dyShape(dyOperand->originalTensor->getDim(), dyOperand->chunkDimensions);
    TensorShape xShape = m_conv->getXOperandShape(dyShape);
    // X operand slice size might be larger than X tensor H dim, since getXOperandShape doesn't clip the result with the
    // tensor actual dimensions. In this case just set to the original size. It is expected to happen when there are
    // only 2 slices, and the 2nd X slice is the overlap + padding.
    if (xShape.getSize(slicingDim) > xOperand->finalShape[slicingDim])
    {
        int padding = paddingAfter + paddingBefore;
        HB_ASSERT(xShape.getSize(slicingDim) <= xOperand->finalShape[slicingDim] + padding,
                  "CommonDimSlicingSolver: X opernad slicing doesn't match dY operand slicing");
    }
    xOperand->chunkDimensions[slicingDim] = xShape.getSize(slicingDim);

    // Set the X operand spatial dim slice overlap and offset
    int overlap                                = m_conv->getInputROIOverlapForDim(TENSOR_X_BWD, slicingDim);
    xOperand->overlapElementsCount[slicingDim] = overlap;
    xOperand->offsetBefore[slicingDim]         = paddingBefore;
    xOperand->offsetAfter[slicingDim]          = paddingAfter;
    xOperand->requiresTensorView               = true;

    // Set the minimal slice size that is required to generate output, will be used to calculate number of slices
    xOperand->minValidSliceSize[slicingDim] = m_conv->getDimActualKernelSize(slicingDim);

    auto dyNumSlices = SlicedOperandUtils::nofSlices(dyOperand, slicingDim);
    auto xNumSlices  = SlicedOperandUtils::nofSlices(xOperand, slicingDim);
    HB_ASSERT(dyNumSlices == xNumSlices, "Wrong number of spatial slices: dy {} X {}", dyNumSlices, xNumSlices);
}

void ConvNodeSolver::setTraversalPatternAndMapping(StrategySlicingData& data)
{
    data.traversalPattern = SlicedOperandTraversalPattern::LEFT_TO_RIGHT_4D;
    if (std::static_pointer_cast<ConvBaseNode>(m_node)->is3DConvolution())
    {
        data.traversalPattern.push_back(DIM_B_FOR_5D_TENSOR);
    }
    const auto& shapeTensor       = m_node->getInput(TENSOR_SHAPE_DEDX);
    const auto& shapeBundleTensor = shapeTensor ? data.bundleTensors[TENSOR_SHAPE_DEDX] : nullptr;
    data.setOutputSliceBackwardMapping(MMESliceMapper::mapOutputToInputs(m_node,
                                                                         data.bundleTensors[0],
                                                                         data.bundleTensors[1],
                                                                         data.masterOperand,
                                                                         shapeBundleTensor));
}

TSize ConvNodeSolver::getOperandAxisElements(const pSlicedOperand& operand, unsigned slicedInputIdx, unsigned slicedDim)
{
    auto     nonCDDims          = m_dimController.getNonCommonAxis(slicedInputIdx);
    TSize    outputAxisElements = 1;
    for (auto dim : nonCDDims)
    {
        outputAxisElements *= m_node->getInput(slicedInputIdx)->getSizeInElements(dim);
    }
    return outputAxisElements;
}

std::vector<Dim> ConvNodeSolver::getInputSlicingDims(unsigned inputIdx)
{
    HB_ASSERT((inputIdx == 0 || inputIdx == 1), "MME node producer is expected for operands A or B");
    unsigned sliceDim = m_dimController.nonCommonDimsForOperand(inputIdx).back();
    HB_ASSERT(sliceDim < m_node->getInput(inputIdx)->getDim(), "slicing dim out of bounds");
    return {sliceDim};
}

// Return the number of outputSliceDim elements that fit in one MME geometry, but at least 1.
TSize ConvNodeSolver::getOutDimAlignmentForMmeUtil(unsigned slicedInputIdx, unsigned outputSliceDim)
{
    unsigned mmeGeometryAxisElements = getMmeGeometrySizeInElementsToAlign(slicedInputIdx, outputSliceDim);
    auto     axisDims = (slicedInputIdx == 0) ? m_dimController.heightOutput() : m_dimController.widthOutput();
    HB_ASSERT(std::find(axisDims.begin(), axisDims.end(), outputSliceDim) != axisDims.end(),
              "Invalid output slicing dim {}",
              outputSliceDim);
    // Accumulate the dims sizes up to the slicing dim
    TSize outputAxisElements = 1;
    for (unsigned i = 0; i < axisDims.size(); i++)
    {
        Dim dim = axisDims[i];
        if (dim == outputSliceDim) break;

        outputAxisElements *= m_node->getOutput(0)->getSizeInElements(dim);
    }
    LOG_DEBUG(SRAM_SLICE, "mme geometry {}, output axis {}", mmeGeometryAxisElements, outputAxisElements);
    // TODO SW-72188 - align to multiple geometries
    return std::max((TSize)mmeGeometryAxisElements / outputAxisElements, 1UL);
}

bool ConvNodeSolver::canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                                const bool              isSingleNodeBundle)
{
    return isSingleNodeBundle;
}

bool ConvNodeSolver::sliceNodeOnMultipleDims(SlicingStrategyPtr& strategy, const BundleSolutionConstraints& constraints)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);

    // Slice the input with spatial dims (X or dY)
    unsigned spatialInputIdx = TENSOR_IFM;  // same as TENSOR_DEDY

    Dim batchDim = m_dimController.batchDim().back();
    Dim sliceDim = batchDim - 1;  // first spatial dim - same for input and output in conv/dedx

    if (!m_conv->isSpatialSlicingSupported(sliceDim))
    {
        LOG_TRACE(SRAM_SLICE, "{}: Spatial slice unsupported for dim {}", HLLOG_FUNC, sliceDim);
        return false;
    }

    pSlicedOperand  inputOperand  = strategy->getSlicingData().getSlicedOperand(m_node->getInput(spatialInputIdx));
    pSlicedOperand& outputOperand = strategy->getSlicingData().masterOperand;

    // Slice the external dim (batch) to minimal size
    inputOperand->chunkDimensions[batchDim]  = 1;
    outputOperand->chunkDimensions[batchDim] = 1;

    // Set the input in SRAM and alignment before calculateMetrics is called
    strategy->setInputIsInSRAM(spatialInputIdx, true);
    inputOperand->alignWithCacheLine = constraints.canAlignInputToCL;

    TSize minSliceNumForPipeline =
        std::max(GCFG_NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING.value() / inputOperand->finalShape[batchDim], 2UL);

    // Find the required dim alignment
    TSize convDimGranularity = getSpatialDimGranularity(sliceDim);
    TSize dimAlignment       = getDimAlignmentInElements(spatialInputIdx, sliceDim, convDimGranularity);
    TSize outMinSize         = round_to_multiple(m_conv->getMinSpatialDimOutputROI(sliceDim), dimAlignment);
    TSize origOutSize        = outputOperand->finalShape[sliceDim];
    if (outMinSize >= origOutSize)
    {
        LOG_TRACE(SRAM_SLICE, "{}: out size {} >= tensor size {}, can't slice", HLLOG_FUNC, outMinSize, origOutSize);
        return false;
    }

    // Limit slicing factor to make sure slice includes at least 1 element
    float slicingFactor = std::min(origOutSize, minSliceNumForPipeline);
    LOG_DEBUG(SRAM_SLICE, "Calc slice size for dim {} of size {}, min size {}", sliceDim, origOutSize, outMinSize);

    OffsetArray slicePadBefore = {0}, slicePadAfter = {0};
    TSize       alignedOutSliceSize = 0;
    do
    {
        alignedOutSliceSize = round_to_multiple(((float)origOutSize / slicingFactor), dimAlignment);
        alignedOutSliceSize = std::max(alignedOutSliceSize, outMinSize);
        LOG_DEBUG(SRAM_SLICE, "alignedOutSize {}, slicingFactor {}", alignedOutSliceSize, slicingFactor);
        TSize outputSliceSize = std::min(alignedOutSliceSize, origOutSize);
        bool  inputSliced     = false;
        std::tie(inputSliced, slicePadBefore[sliceDim], slicePadAfter[sliceDim]) =
            sliceNonCommonSpatialDim(sliceDim, outputSliceSize, inputOperand, outputOperand);
        // Set double buffer after the operand is sliced, otherwise it is ignored for trivial slicing
        if (inputSliced)
        {
            strategy->setDoubleBuffer(true);
        }
        slicingFactor++;
    } while (strategy->calculateMetrics().SRAMCapacity > constraints.availableSramBytes &&
             alignedOutSliceSize > outMinSize);

    // Set the post slicing handler to handle the sliced nodes padding
    pSlicedOperand xOperand      = strategy->getSlicingData().getSlicedOperand(m_conv->getXOperand());
    xOperand->postSlicingHandler = std::make_shared<PostSlicingConvHandler>(slicePadBefore, slicePadAfter);

    // Update traversal pattern: walk left on the output width, then down on output height, from inner dimension
    // to outer
    strategy->getSlicingData().traversalPattern = {m_dimController.widthOutput().front(), sliceDim, batchDim};

    // try to place W in SRAM opportunistically
    strategy->setInputIsInSRAM(TENSOR_WEIGHT, true);
    if (strategy->calculateMetrics().SRAMCapacity > constraints.availableSramBytes)
    {
        // rollback
        strategy->setInputIsInSRAM(TENSOR_WEIGHT, false);
    }
    else
    {
        LOG_DEBUG(SRAM_SLICE, "{}: managed to place W in SRAM", HLLOG_FUNC);
    }

    auto requiredBytes = strategy->calculateMetrics().SRAMCapacity;
    bool fitsSram      = requiredBytes <= constraints.availableSramBytes;
    LOG_DEBUG(SRAM_SLICE, "{}: {} MB {} fit SRAM", HLLOG_FUNC, bToMb(requiredBytes), (fitsSram ? "does" : "doesn't"));
    return fitsSram;
}

TSize ConvNodeSolver::getSpatialDimGranularity(unsigned slicingDim) const
{
    return 1;
}

std::tuple<bool, int, int> ConvNodeSolver::sliceNonCommonSpatialDim(unsigned        slicingDim,
                                                                    TSize           outputSliceSize,
                                                                    pSlicedOperand& inputOperand,
                                                                    pSlicedOperand& outputOperand)
{
    // Slice the output operand
    TSize origOutputSize                       = outputOperand->chunkDimensions[slicingDim];
    outputOperand->chunkDimensions[slicingDim] = outputSliceSize;
    TSize inputSliceSize =
        getInputSliceSize(outputOperand->chunkDimensions, outputOperand->originalTensor->getDim(), slicingDim);

    // inputSliceSize returned unclipped and might be larger than input tensor size
    if (inputSliceSize >= inputOperand->chunkDimensions[slicingDim])
    {
        // restore output size
        outputOperand->chunkDimensions[slicingDim] = origOutputSize;
        LOG_DEBUG(SRAM_SLICE, "Input operand isn't sliced for output slice size {}", outputSliceSize);
        return std::make_tuple(false, 0, 0);
    }
    // Slice the input operand
    inputOperand->chunkDimensions[slicingDim] = inputSliceSize;
    // Calc the overlap of the input slice
    int overlap = m_conv->getInputROIOverlapForDim(TENSOR_IFM, slicingDim);

    inputOperand->overlapElementsCount[slicingDim] = overlap;
    auto [slicePadBefore, slicePadAfter] =
        setSlicedOperandsPadding(slicingDim, overlap, outputSliceSize, inputOperand, outputOperand);

    // Set the minimal valid last slice size, to clip X operand lines, which are not enough to create another Y operand
    // line. In dedx - those lines can't be calculated. The last slice must include enough lines to complete the padding
    // before lines to at least actual kernel size. Otherwise there wasn't kernel calculation with the last lines in
    // fwd. In fwd - slicePadBefore is 0, and the min is set to the actual kernel size.
    pSlicedOperand xOperand = (inputOperand->originalTensor == m_conv->getXOperand()) ? inputOperand : outputOperand;
    xOperand->minValidSliceSize[slicingDim] = m_conv->getDimActualKernelSize(slicingDim) - slicePadBefore;

    unsigned inNumSlices  = SlicedOperandUtils::nofSlices(inputOperand, slicingDim);
    unsigned outNumSlices = SlicedOperandUtils::nofSlices(outputOperand, slicingDim);
    // Make sure there are enough input slices to create output slices. The solution is built based on the output
    // slices, and throws extra input slice if required.
    HB_ASSERT(inNumSlices >= outNumSlices,
              "Spatial slicing - wrong number of slices in: {} out: {}",
              inNumSlices,
              outNumSlices);

    inputOperand->requiresTensorView = true;  // the opernad with the opverlap
    xOperand->requiresTensorView     = true;  // X operand may always have non trivial slicing

    LOG_TRACE(SRAM_SLICE,
              "input size {}, output size {}, overlap {}, slicePadBefore {}, slicePadAfter {}",
              inputSliceSize,
              outputSliceSize,
              overlap,
              slicePadBefore,
              slicePadAfter);
    return std::make_tuple(true, slicePadBefore, slicePadAfter);
}

// Calculates the input size of the sliced dimension, given the output slice size.
// The input size returned unclipped, and might be larger than the given input tensor dim size.
// TODO handle paddingType here <===== PADDING_TYPE
TSize ConvNodeSolver::getInputSliceSize(const SizeArray& outputSliceSizes, unsigned rank, unsigned slicedDim) const
{
    TensorShape outputShape(rank, outputSliceSizes, SizeArray {0});
    // getInputShape for conv and dedx expect output index 0 and input index 0, which have different names.
    // TENSOR_IFM = TENSOR_DEDY = 0, TENSOR_OFM = TENSOR_DEDW = TENSOR_DEDX = 0,
    TensorShape inputShape = m_conv->getInputShape(outputShape, TENSOR_OFM, TENSOR_IFM);
    // The function doesn't clip the result, which may be larger than the real input in case of padding. pass it
    // unclipped.
    return inputShape.getSize(slicedDim);
}

std::tuple<int, int> ConvNodeSolver::setSlicedOperandsPadding(unsigned        slicingDim,
                                                              int             overlap,
                                                              TSize           outputSliceSize,
                                                              pSlicedOperand& inputOperand,
                                                              pSlicedOperand& outputOperand) const
{
    // Set the convolution padding before, so the first slice size and other slices offset will be adjusted accordingly.
    // the X operand is the operand which requires them.
    pSlicedOperand    xOperand = (inputOperand->originalTensor == m_conv->getXOperand()) ? inputOperand : outputOperand;
    ConvParamsIndices convIdx  = ConvBaseNode::dimIndexToConvParamsIndices(slicingDim);
    xOperand->offsetBefore[slicingDim] = m_conv->getConvolutionParams().padding[convIdx.paddingBeforeIndex];
    xOperand->offsetAfter[slicingDim]  = m_conv->getConvolutionParams().padding[convIdx.paddingAfterIndex];
    return std::make_tuple(0, 0);
}

TSize DedxNodeSolver::getSpatialDimGranularity(unsigned slicingDim) const
{
    const synConvolution3DParamsV2& convParams = m_conv->getConvolutionParams();
    ConvParamsIndices               convIdx    = ConvBaseNode::dimIndexToConvParamsIndices(slicingDim);
    unsigned                        stride     = convParams.stride[convIdx.spatialIndex];
    return stride;
}

std::tuple<int, int> DedxNodeSolver::setSlicedOperandsPadding(unsigned        slicingDim,
                                                              int             overlap,
                                                              TSize           outputSliceSize,
                                                              pSlicedOperand& inputOperand,
                                                              pSlicedOperand& outputOperand) const
{
    int slicePadBefore = 0;
    int slicePadAfter  = 0;
    // Overlap in dy means the first slice starts in a negative offset, but it's not really part of the tensor.
    // Need to reduce this offset from the dy slices like the padding is reduced from X slices.
    inputOperand->offsetBefore[slicingDim] = overlap;
    // Calc the X operand middle slices padding
    m_conv->getXStrideAlignedROIPaddingForDim(slicingDim, outputSliceSize, slicePadBefore, slicePadAfter);
    // Don't count the last slice if it's padding slice
    outputOperand->countPaddingOnlySlice = false;

    ConvNodeSolver::setSlicedOperandsPadding(slicingDim, overlap, outputSliceSize, inputOperand, outputOperand);
    return std::make_tuple(slicePadBefore, slicePadAfter);
}

void DedxNodeSolver::projectSlicingOnNodeOperands(NodeStrategyPtr& strategy, unsigned slicedInputIdx)
{
    auto outputOperand = strategy->getSlicingData().masterOperand;
    HB_ASSERT_PTR(outputOperand);
    runOnTensorsForType<Node::USAGE_INPUT>(m_node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& tensor) {
        if (tensor && tensor->getTensorType() == OUTPUT_DESCRIBING_SHAPE_TENSOR)
        {
            pSlicedOperand shapeOperand = strategy->getSlicingData().getSlicedOperand(tensor);
            HB_ASSERT_PTR(shapeOperand);
            shapeOperand->copyShapeData(*outputOperand);
            shapeOperand->resideInSRAM    = false;
        }
    });
}

bool GemmNodeSolver::isNodeSupported()
{
    // This solver doesn't support GEMM dedx / dedw, and they still don't have access pattern
    if (m_node->getNodeType() != Node::TYPE_GEMM)
    {
        LOG_WARN(SRAM_SLICE,
                 "GemmNodeSolver: Only Gemm node type is supported. Got: {} (node: {}).",
                 m_node->getNodeTypeStr(),
                 m_node->getNodeName());
        return false;
    }
    return MmeNodeSolver::isNodeSupported();
}

void GemmNodeSolver::setTraversalPatternAndMapping(StrategySlicingData& data)
{
    data.setOutputSliceBackwardMapping(MMESliceMapper::mapOutputToInputs(m_node,
                                                                         data.getSlicedOperand(m_node->getInput(0)),
                                                                         data.getSlicedOperand(m_node->getInput(1)),
                                                                         data.masterOperand,
                                                                         nullptr));
}

void GemmNodeSolver::sliceNonMasterOperand(SlicingStrategyPtr&      strategy,
                                           unsigned                 nonMasterInputIdx,
                                           const TensorGranularity& nonMasterGranularity)
{
    const auto& nonMasterInput = m_node->getInput(nonMasterInputIdx);
    const auto& sliceDims      = getInputSlicingDims(nonMasterInputIdx);
    HB_ASSERT(!sliceDims.empty(),
              "Node {} can't be sliced on input {}",
              m_node->getNodeName(),
              nonMasterInput->getName());
    const auto& inputSliceDim = sliceDims.front();  // slicing here on single dim
    if (nonMasterInput->getSizeInElements(inputSliceDim) == 1) return;

    auto outputDims = getTensorMatchingSlicedDims(m_node, nonMasterInput, {inputSliceDim}, m_node->getOutput(0));
    HB_ASSERT(outputDims.size() == 1, "a single non common dim is expected to project to a single output dim");
    const unsigned outputSliceDim = outputDims.front();

    const auto elementsForMmeUtil =
        MmeBrainProxy::getRecommendedGeometryAxisElementsForNonMasterOperand(m_node, 1 - nonMasterInputIdx);
    const TSize dimAlignment = alignDimSizeToGranularity(elementsForMmeUtil, nonMasterGranularity[inputSliceDim]);
    const TSize fullDimSize  = nonMasterInput->getSizeInElements(inputSliceDim);

    const unsigned sliceNumForPipeline    = GCFG_NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING.value();
    const auto     slicingFactor          = std::min(fullDimSize, (TSize)sliceNumForPipeline);
    const TSize    sliceDimSizeInElements = std::floor((float)fullDimSize / (float)slicingFactor);
    const TSize    alignedSliceSize       = round_to_multiple(sliceDimSizeInElements, dimAlignment);

    if (alignedSliceSize >= fullDimSize) return;  // input can't be sliced

    // Input can't be sliced evenly - since the non master operand slices are expected to be placed in SRAM concurrently
    // in multi-buffer with buffer for each slice, if the last slice is smaller - it might exceed SRAM capacity since
    // the size for the multi-buffer is determined by (max slice size * num buffers).
    if (fullDimSize % alignedSliceSize != 0) return;

    // Input is not aligned to CL - additional SRAM will be required for alignment -> might exceed reserved SRAM.
    std::map<unsigned, TSize> sizePerDim = {{inputSliceDim, alignedSliceSize}};
    if (SlicedOperandUtils::getSliceAlignmentSize(nonMasterInput, sizePerDim) != 0) return;

    LOG_DEBUG(SRAM_SLICE,
              "{}: MME node: {}, slice non master input {} (idx {}) on dim {} to size {} (elementsForMmeUtil={} "
              "granularity={})",
              HLLOG_FUNC,
              m_node->getNodeName(),
              nonMasterInput->getName(),
              nonMasterInputIdx,
              inputSliceDim,
              alignedSliceSize,
              elementsForMmeUtil,
              nonMasterGranularity[inputSliceDim]);

    auto slicedInputOperand = strategy->getSlicingData().getSlicedOperand(nonMasterInput);
    HB_ASSERT_PTR(slicedInputOperand);
    auto slicedOutputOperand = strategy->getSlicingData().getSlicedOperand(m_node->getOutput(0));
    HB_ASSERT_PTR(slicedOutputOperand);

    slicedInputOperand->chunkDimensions[inputSliceDim]   = alignedSliceSize;
    slicedOutputOperand->chunkDimensions[outputSliceDim] = alignedSliceSize;

    // Set traversal pattern according to the operand that should be reused in SRAM, to avoid additional DMA mem-copies.
    strategy->getSlicingData().traversalPattern = (nonMasterInputIdx == 0)
                                                      ? SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D
                                                      : SlicedOperandTraversalPattern::LEFT_TO_RIGHT_2D;
}

std::vector<Dim> GemmNodeSolver::getInputSlicingDims(unsigned inputIdx)
{
    HB_ASSERT((inputIdx == 0 || inputIdx == 1), "MME node producer is expected for operands A or B");
    unsigned sliceDim = m_dimController.nonCommonDimsForOperand(inputIdx).back();
    HB_ASSERT(sliceDim < m_node->getInput(inputIdx)->getDim(), "slicing dim out of bounds");
    return {sliceDim};
}

bool BatchGemmNodeSolver::isNodeSupported()
{
    if (m_node->getNodeType() != Node::TYPE_BATCH_GEMM)
    {
        LOG_WARN(SRAM_SLICE,
                 "BatchGemmNodeSolver: Only BatchGemm node types are supported. Got: {} (node: {}).",
                 m_node->getNodeTypeStr(),
                 m_node->getNodeName());
        return false;
    }
    return isOperandsLayoutSupported();
}

bool BatchGemmNodeSolver::isOperandsLayoutSupported()
{
    // [SW-85839] Ranks of inputs must be equal
    if (m_node->getInput(0)->getDim() != m_node->getInput(1)->getDim())
    {
        LOG_WARN(SRAM_SLICE, "BatchGemmNodeSolver: only equal rank bgemm is supported");
        return false;
    }

    if (BatchGemmNode::allBatchDimsDegenerated(toSizeVector(m_node->getOutput(0))) && isDuplicateInputMme())
    {
        LOG_WARN(SRAM_SLICE,
                 "BatchGemmNodeSolver: all batch dims are degenerated with duplicate input operands is not supported");
        return false;
    }

    return true;
}

void BatchGemmNodeSolver::setTraversalPatternAndMapping(StrategySlicingData& data)
{
    for (auto i = (unsigned)DIM_GEMM_BATCH; i < data.masterOperand->originalTensor->getDim(); ++i)
    {
        data.traversalPattern.push_back(i);
    }
    std::vector<pSlicedOperand> slicedInputs;
    // In case of 2 inputs which are the same tensor - the single operand will be
    // insert twice to the vector, which is required for the mapper to work correctly.
    for (const TensorPtr& input : m_node->getInputs())
    {
        if (!input) continue;
        const pSlicedOperand& operand = data.getSlicedOperand(input);
        HB_ASSERT_PTR(operand);
        slicedInputs.push_back(operand);
    }
    std::vector<pSlicedOperand> slicedOutputs = {data.masterOperand};

    data.setOutputSliceBackwardMapping(AccessPatternSliceMapper::createBwdMapping(m_node, slicedInputs, slicedOutputs));
}

std::vector<Dim> BatchGemmNodeSolver::clearLeadingOnesAndReverseBatchDims(const SizeArray& sizes, const DimVector& batchDim) const
{
    // Find the first non degenerated batch dim
    auto nonDegeneratedIt =
        std::find_if(batchDim.rbegin(), batchDim.rend(), [&sizes](const uint8_t& dim) { return sizes.at(dim) != 1; });
    // sliceDims are dims from the outer first non-degenerated dim, until the inner batch dim
    std::vector<Dim> sliceDims(nonDegeneratedIt, batchDim.rend());
    return sliceDims;
}

std::vector<Dim> BatchGemmNodeSolver::getInputSlicingDims(unsigned inputIdx)
{
    // TODO [SW-68598] no need to check once all bgemms are supported by the solver.
    if (!isNodeSupported()) return {};

    HB_ASSERT((inputIdx == 0 || inputIdx == 1 || m_node->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM),
              "MME node producer is expected for operands A or B");

    DimVector inputBatchDims;
    for (uint32_t dim = DIM_GEMM_BATCH; dim < m_node->getInput(inputIdx)->getDim(); ++dim)
    {
        inputBatchDims.push_back(dim);
    }
    // Take non degenerated batch dims as slicing dims
    std::vector<Dim> sliceDims = clearLeadingOnesAndReverseBatchDims(m_node->getInput(inputIdx)->getAllSizesInElements(), inputBatchDims);

    // Add non common dim in lower priority after batch dims. Only valid for bgemm nodes that don't have duplicate input
    // tensors
    if (!isDuplicateInputMme())
    {
        sliceDims.push_back(m_dimController.nonCommonDimsForOperand(inputIdx).back());
    }

    HB_ASSERT(!sliceDims.empty(), "no valid slicing dims set");
    return sliceDims;
}

// Returns the MME geometry axis size in elements, which corresponds to the given input index.
// The geometry is selected based on the MME node recommended configuration
// In bgemm - it's the concurrency level, which is the same for A and B
unsigned BatchGemmNodeSolver::getMmeGeometrySizeInElementsToAlign(unsigned slicedInputIdx, Dim slicedDim)
{
    if (slicedDim < DIM_GEMM_BATCH)
    {
        return MmeNodeSolver::getMmeGeometrySizeInElementsToAlign(slicedInputIdx, slicedDim);
    }
    // else - sliced on batch dim - return required concurrency
    unsigned recommendedConcurrency = MmeBrainProxy::getRecommendedGeometryConcurrency(m_node);

    const TensorPtr& input = m_node->getInput(slicedInputIdx);
    // Find the innermost batch dim that can satisfy the concurrency requirement (=is divisible by it).
    // All the other batch dims are allowed to be sliced in a single element granularity.
    Dim innermostBatchDim = 0;
    for (Dim dim = DIM_GEMM_BATCH; dim < input->getDim(); dim++)
    {
        if (input->getSizeInElements(dim) % recommendedConcurrency == 0)
        {
            innermostBatchDim = dim;
            break;
        }
    }
    if (innermostBatchDim != 0 && innermostBatchDim != slicedDim)
    {
        return 1;
    }

    return recommendedConcurrency;
}

void BatchGemmNodeSolver::projectSlicingOnNodeOperands(SlicingStrategyPtr& strategy, unsigned slicedInputIdx)
{
    pSlicedOperand slicedInputOperand = strategy->getSlicingData().getSlicedOperand(m_node->getInput(slicedInputIdx));
    AccessPatternNodeSolutionProjector projector {m_node};

    NodeStrategyPtr projectedStrategy =
        projector.getNodeStrategy({slicedInputOperand}, slicedInputOperand->originalTensor);

    for (auto& t : m_node->getInputs())
    {
        // Avoid projecting on the same operand in case both inputs are the same tensor
        if (!t || t == slicedInputOperand->originalTensor) continue;

        pSlicedOperand operand = strategy->getSlicingData().getSlicedOperand(t);
        HB_ASSERT_PTR(operand);

        pSlicedOperand projectedOperand = projectedStrategy->getSlicingData().getSlicedOperand(t);
        HB_ASSERT_PTR(projectedOperand);
        operand->chunkDimensions = projectedOperand->chunkDimensions;
        LOG_DEBUG(SRAM_SLICE,
                  "Projected chunkDimensions [{}] for input {}",
                  toString(operand->chunkDimensions, ','),
                  t->getName());
    }
}

bool BatchGemmNodeSolver::canSliceNodeOnMultipleDims(const std::vector<Dim>& sharedOperandSlicingDims,
                                                     const bool              isSingleNodeBundle)
{
    return sharedOperandSlicingDims.size() > 1;
}

bool BatchGemmNodeSolver::sliceNodeOnMultipleDims(SlicingStrategyPtr&              strategy,
                                                  const BundleSolutionConstraints& constraints)
{
    LOG_TRACE(SRAM_SLICE, "{}", HLLOG_FUNC);
    const unsigned slicedInputIdx = getInputIndexToSlice(constraints.tensorsInSram, m_node);
    const auto&    slicingDims    = constraints.sharedOperandSlicingDims;
    HB_ASSERT(slicingDims.size() > 1,
              "Expecting for 2 or more slicing dims",
              m_node->getInput(slicedInputIdx)->getName());
    for (const auto sliceDim : slicingDims)
    {
        if (sliceNodeOnSingleNonCommonDim(strategy, sliceDim, constraints))
        {
            // The node was sliced on the current iteration's sliceDim and all previous iterations' sliceDims
            return true;
        }
    }
    return false;
}

bool MaskedBatchGemmNodeSolver::isNodeSupported()
{
    if (m_node->getNodeType() != Node::TYPE_MASKED_BATCH_GEMM)
    {
        LOG_WARN(SRAM_SLICE,
                 "MaskedBatchGemmNodeSolver: Only MaskedBatchGemm node types are supported. Got: {} (node: {}).",
                 m_node->getNodeTypeStr(),
                 m_node->getNodeName());
        return false;
    }

    if (m_dimController.batchDim().size() != 2)
    {
        LOG_WARN(SRAM_SLICE,
                 "MaskedBatchGemmNodeSolver: masked bgemm is expected to have 2 batch dims (node: {}).",
                 m_node->getNodeName());
        return false;
    }

    return isOperandsLayoutSupported();
}