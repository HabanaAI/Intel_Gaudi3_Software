#include "defs.h"
#include "mme_dim_controller.h"
#include "habana_graph.h"
#include "conv_base_node.h"
#include "node.h"
#include "sram_management/bundle.h"
#include "utils.h"
#include "slicing_utils.h"
#include "common_dim_slicing_solver.h"
#include "post_slicing_op_handler.h"
#include "flatten_mme.h"
#include <memory>

static bool gemmLikeMMENode(const NodePtr& n, const pSlicedOperand& opA, const pSlicedOperand& opB)
{
    const auto* mmeNode = dynamic_cast<MmeNode*>(n.get());
    HB_ASSERT_PTR(mmeNode);

    return mmeNode->canBeConvertedToGEMM() && SlicedOperandUtils::isFinalShape2D(opA) &&
           SlicedOperandUtils::isFinalShape2D(opB);
}

bool CommonDimSlicingSolver::effectiveForBundle()
{
    if (!isSupportedNodeType())
    {
        SLC_TRACE("CommonDimSlicingSolver:{}: Supported nodes: dedw / GEMM. Actual node:{}!",
                  HLLOG_FUNC,
                  m_mmeNode->getNodeType());
        return false;
    }

    pTensor operandA = m_mmeNode->getInput(0);
    pTensor operandB = m_mmeNode->getInput(1);
    pTensor out      = m_mmeNode->getOutput(0);

    const DimVector& commonDimOperandA = m_dimController.commonDimOperandA();
    const SizeArray& sizeOperandA = operandA->getAllSizesInElements();

    uint64_t commonDimSize = multiplyElements(sizeOperandA.data() + commonDimOperandA.front(),
                                              sizeOperandA.data() + commonDimOperandA.back() + 1);

    if (commonDimSize < SlicingBrain::knobs.minCDSizeForPartials)
    {
        SLC_TRACE("CommonDimSlicingSolver:{}: common dim size is too small {} < {}",
                  HLLOG_FUNC,
                  commonDimSize,
                  SlicingBrain::knobs.minCDSizeForPartials);
        return false;
    }
    // make sure the solver can actually slice the tensors to fit to SRAM
    createAllStrategies();
    return !getStrategies().empty();
}

void CommonDimSlicingSolver::createStrategiesWithBiggerSpatialSlices(const Settable<unsigned>& spatialSlicedDim,
                                                                     const SlicingStrategyPtr& initialStrategy)
{
    if (!spatialSlicedDim.is_set())
    {
        return;
    }
    const auto* pConv = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    HB_ASSERT_PTR(pConv);  // Verified before the spatial slicing is applied
    if (!pConv->canBeConvertedToGEMM())
    {
        // When the convolution is not flattenable, the operands slice sizes are dependent
        return;
    }
    const unsigned     multiplicationFactor = 2;
    const unsigned     slicingDim           = spatialSlicedDim.value();
    SlicingStrategyPtr prevStrategy         = initialStrategy;
    while (prevStrategy->recalculateSramCapacity().SRAMCapacity < SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        SlicingStrategyPtr newStrategy = prevStrategy->clone(true);

        pSlicedOperand& dyOperand = newStrategy->getSlicingData().bundleTensors[0];
        pSlicedOperand& xOperand  = newStrategy->getSlicingData().bundleTensors[1];

        dyOperand->chunkDimensions[slicingDim] *= multiplicationFactor;
        xOperand->chunkDimensions[slicingDim] *= multiplicationFactor;

        if ((dyOperand->chunkDimensions[slicingDim] > dyOperand->finalShape[slicingDim]) ||
            (xOperand->chunkDimensions[slicingDim] > xOperand->finalShape[slicingDim]))
        {
            break;
        }

        if (newStrategy->recalculateSramCapacity().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
        {
            SLC_DEBUG("CommonDimSlicingSolver: - Create new strategy with bigger slices on spatial dim {}, dy slice "
                      "size: {}, x slice size: {}",
                      slicingDim,
                      dyOperand->chunkDimensions[slicingDim],
                      xOperand->chunkDimensions[slicingDim]);
            addStrategy(newStrategy);
        }

        prevStrategy = newStrategy;
    }
}

void CommonDimSlicingSolver::createAllStrategies()
{
    if (!getStrategies().empty())
    {
        SLC_TRACE("CommonDimSlicingSolver:{}: Strategies were already created by effectiveForBundle", HLLOG_FUNC);
        return;
    }

    pMmeSlicingStrategy   strategy = MmeSlicingStrategy::createStrategyForMMENode(m_halReader, m_mmeNode);
    const pSlicedOperand& opA      = strategy->getMmeSlicingData().bundleTensors[0];
    const pSlicedOperand& opB      = strategy->getMmeSlicingData().bundleTensors[1];
    const pSlicedOperand& outputOp = strategy->getMmeSlicingData().masterOperand;

    setFinalShapeInOperands(opA, opB, outputOp, m_mmeNode);

    // set all inputs in SRAM
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1,true).setOutputIsInSRAM(true);
    auto& slicingData = strategy->getMmeSlicingData();

    if (m_mmeNode->getNodeType() == Node::TYPE_DEDW)
    {
        // DeDw swaps operands, so the default traverse pattern should be top to bottom
        slicingData.traversalPattern = SlicedOperandTraversalPattern::TOP_TO_BOTTOM_2D;
    }

    unsigned sliceAxisA;
    unsigned sliceAxisB;
    std::tie(sliceAxisA, sliceAxisB) = getSlicingDims(opA, opB);
    bool slicingBatch = std::any_of(m_dimController.batchDim().begin(), m_dimController.batchDim().end(), [=](unsigned dim){return dim == sliceAxisA;});
    uint32_t   sliceChunkSize = getCommonDimSliceChunkSize(m_mmeNode, opA, slicingBatch);

    SLC_TRACE("Slice common dim:  OperandA on dim: {}, operandB on dim: {}, to chunk size  {}", sliceAxisA, sliceAxisB, sliceChunkSize);

    pSlicedOperand& sliceOpA = slicingData.bundleTensors[0];
    sliceOpA->chunkDimensions[sliceAxisA] = sliceChunkSize;

    pSlicedOperand& sliceOpB = slicingData.bundleTensors[1];
    sliceOpB->chunkDimensions[sliceAxisB] = sliceChunkSize;

    auto numSliceOpA = SlicedOperandUtils::nofSlices(sliceOpA, sliceAxisA);
    auto numSliceOpB = SlicedOperandUtils::nofSlices(sliceOpB, sliceAxisB);
    HB_ASSERT(numSliceOpA == numSliceOpB, "CommonDimSlicingSolver: wrong number of slices");

    slicingData.masterOperand->finalElementType = syn_type_float;

    if (strategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        addStrategy(strategy);
        return;
    }

    if (canApplySpatialSlicing(strategy))
    {
        unsigned slicingDim            = m_dimController.commonDimOperandA().back();
        unsigned maxSpatialDimsToSlice = GCFG_SRAM_SLICER_CONV_MULTI_SPATIAL_DIMS_SLICE_ENABLED.value() ? 2 : 1;
        for (unsigned numSpatialDimsToSlice = 1; numSpatialDimsToSlice <= maxSpatialDimsToSlice;
             numSpatialDimsToSlice++)
        {
            // Advance the slicing dim to the next spatial dim.
            slicingDim = getNextSpatialSlicingDim(m_dimController.commonDimOperandA(), slicingDim);
            Settable<unsigned> actualSlicedDim = applySpatialSlicing(strategy, slicingDim);

            if (strategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
            {
                addStrategy(strategy);
                // When a valid strategy is found, create clones with bigger slice sizes.
                // applyNonCommonSlicing also adds strategies but we avoid enlarging their slice sizes,
                // assuming that if we had to slice the non-common dim the slice sizes are big enough already.
                createStrategiesWithBiggerSpatialSlices(actualSlicedDim, strategy);
                return;
            }

            // TODO: [SW-30959] DSD - unblock snake pattern on spatial slicing of dEdW
            if (actualSlicedDim.is_set() && (sliceOpA->originalTensor->isDynamicDim(actualSlicedDim.value()) ||
                                             sliceOpB->originalTensor->isDynamicDim(actualSlicedDim.value())))
            {
                slicingData.setSnakeWalkingPattern(false);
                strategy->setDoubleBuffer(false);
            }

            // Try to slice on non-common dim before moving to the next spatial dim (proved better performance).
            // applyNonCommonSlicing doesn't change the given strategy, so we can use it in the next iteration,
            // the prev. spatial dim is already sliced
            applyNonCommonSlicing(strategy);

            if (!getStrategies().empty()) break;  // Valid strategy found - no need to slice on the next spatial dim
        }
    }
    else
    {
        applyNonCommonSlicing(strategy);
    }
}

bool CommonDimSlicingSolver::canApplySpatialSlicing(const pMmeSlicingStrategy& strategy)
{
    // Curently supports only dedw nodes
    if (m_mmeNode->getNodeType() != Node::TYPE_DEDW) return false;

    const pSlicedOperand& opA = strategy->getMmeSlicingData().bundleTensors[0];
    const pSlicedOperand& opB = strategy->getMmeSlicingData().bundleTensors[1];

    // Gemm-like node doesn't need "spatial" slicing.
    if (gemmLikeMMENode(m_mmeNode, opA, opB)) return false;

    // Both Operand A and B are 4D/5D
    if ((opA->originalTensor->getDim() < 4) || (opB->originalTensor->getDim() < 4)) return false;

    for (auto batchDimension : m_dimController.batchDim())
    {
        if (opA->chunkDimensions[batchDimension] != 1 || opB->chunkDimensions[batchDimension] != 1)
        {
            return false;
        }
    }

    auto convNode = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    HB_ASSERT_PTR(convNode);
    HB_ASSERT(m_dimController.batchDim().size() == 1, "Single batch dim expected");
    const uint32_t batchDim = m_dimController.batchDim().front();
    return convNode->isSpatialSlicingSupported(batchDim - 1);
}

static unsigned findMinimalSpatialSliceSize(unsigned              spatialSliceAxis,
                                            int                   offset,
                                            int                   overlap,
                                            const pSlicedOperand& slicedOperand,
                                            const DimVector&      commonDims)
{
    // Minimal dY slice size is set to be larger than the padding before the dimension – to make sure the first X slice size is not 0.
    // Minimal dY slice size is set to be larger than the overlap – to make sure a slice includes new lines beyond the previous slice.
    // Slice size should be bigger than minSliceSize.
    unsigned minSliceSize = std::max(std::max(overlap, offset), 0); // Overlap/offset might be negative

    // Find the minimal slice size, which creates a slice size larger than the MME minimal common dim size for partials.
    // Start from minSliceSize+1 and increase until the MME minimal size requirement is satisfied.
    SizeArray chunkDims = slicedOperand->chunkDimensions;
    unsigned commonDimSize = 0;

    do
    {
        minSliceSize++;
        chunkDims[spatialSliceAxis] = minSliceSize;
        commonDimSize = multiplyElements(chunkDims.data() + commonDims.front(), chunkDims.data() + commonDims.back() + 1);
    } while ((commonDimSize < SlicingBrain::knobs.minCDSizeForPartials) && (minSliceSize < slicedOperand->finalShape[spatialSliceAxis]));

    return minSliceSize;
}

// Returns the first sliced spatial dim - if sliced on spatial.
Settable<unsigned> CommonDimSlicingSolver::applySpatialSlicing(const pMmeSlicingStrategy& strategy,
                                                               unsigned                   spatialSliceAxis)
{
    // If no strategy was found for batch slicing, try to slice on spatial dims
    const auto* pConv = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    HB_ASSERT(pConv != nullptr, "CommonDimSlicingSolver: wrong type for MME node when applying spatial slicing"); // Verified before the slicing is applied

    SLC_DEBUG("CommonDimSlicingSolver:{} - Try to apply spatial slicing on dim {}", HLLOG_FUNC, spatialSliceAxis);

    // Calc minimal spatial slicing size for the first spatial dimension (D or H)
    pSlicedOperand& dyOperand = strategy->getMmeSlicingData().bundleTensors[0]; // dY operand

    const ConvParamsIndices convIdx = ConvBaseNode::dimIndexToConvParamsIndices(spatialSliceAxis);
    int paddingAfter = pConv->getConvolutionParams().padding[convIdx.paddingAfterIndex];
    int paddingBefore = pConv->getConvolutionParams().padding[convIdx.paddingBeforeIndex];
    unsigned actualKernelSize = pConv->getDimActualKernelSize(spatialSliceAxis);
    if(paddingAfter >= actualKernelSize)
    {
        SLC_TRACE("No solution for spatial slicing - padding after too big");
        // No solution - Return unset object
        return Settable<unsigned>();
    }
    if (paddingBefore < 0 || paddingAfter < 0)
    {
        // TODO - SW-25558 - handle negative padding
        SLC_TRACE("No solution for spatial slicing - padding before or after is negative");
        // No solution - Return unset object
        return Settable<unsigned>();
    }
    int overlap = pConv->getInputROIOverlapForDim(TENSOR_X_BWD, spatialSliceAxis);
    auto operandASliceSize = findMinimalSpatialSliceSize(spatialSliceAxis,
                                                         paddingBefore,
                                                         overlap,
                                                         dyOperand,
                                                         m_dimController.commonDimOperandA());
    if (operandASliceSize >= dyOperand->finalShape[spatialSliceAxis])
    {
        SLC_TRACE("No solution for spatial slicing");
        // No solution - Return unset object
        return Settable<unsigned>();
    }
    dyOperand->chunkDimensions[spatialSliceAxis] =
        operandASliceSize;  // Slice dY operand on the first spatial dimension

    // Find the corresponding B operand slice size, based on the convolution parameters and the output slice size.
    pSlicedOperand& xOperand = strategy->getMmeSlicingData().bundleTensors[1]; // X operand
    TensorShape dyShape(dyOperand->originalTensor->getDim(), dyOperand->chunkDimensions);
    TensorShape xShape = pConv->getXOperandShape(dyShape);
    // X operand slice size might be larger than X tensor H dim, since getXOperandShape doesn't clip the result with the
    // tensor actual dimensions. In this case just set to the original size. It is expected to happen when there are
    // only 2 slices, and the 2nd X slice is the overlap + padding.
    if (xShape.getSize(spatialSliceAxis) > xOperand->chunkDimensions[spatialSliceAxis])
    {
        int padding = paddingAfter + paddingBefore;
        HB_ASSERT(xShape.getSize(spatialSliceAxis) <= xOperand->chunkDimensions[spatialSliceAxis] + padding,
                  "CommonDimSlicingSolver: X opernad slicing doesn't match dY operand slicing");
    }
    xOperand->chunkDimensions[spatialSliceAxis] = xShape.getSize(spatialSliceAxis);

    // Set the X operand spatial dim slice overlap and offset
    xOperand->overlapElementsCount[spatialSliceAxis] = overlap;
    xOperand->offsetBefore[spatialSliceAxis] = paddingBefore;
    xOperand->offsetAfter[spatialSliceAxis] = paddingAfter;
    xOperand->requiresTensorView = true;

    // Set the minimal slice size that is required to generate output, will be used to calculate number of slices
    xOperand->minValidSliceSize[spatialSliceAxis] = pConv->getDimActualKernelSize(spatialSliceAxis);

    // Set the post slicing handler to handle the sliced dedw nodes padding
    xOperand->postSlicingHandler = std::make_shared<PostSlicingConvHandler>();

    auto numSliceOpA = SlicedOperandUtils::nofSlices(dyOperand, spatialSliceAxis);
    auto numSliceOpB = SlicedOperandUtils::nofSlices(xOperand, spatialSliceAxis);
    HB_ASSERT(numSliceOpA == numSliceOpB, "CommonDimSlicingSolver: wrong number of slices");

    SLC_DEBUG("CommonDimSlicingSolver:{} - slicing spatial dim {}, dy slice size: {}, x slice size: {}, x overlap: {}, "
              "x offsetBefore: {}, x offsetAfter: {}",
              HLLOG_FUNC,
              spatialSliceAxis,
              dyOperand->chunkDimensions[spatialSliceAxis],
              xOperand->chunkDimensions[spatialSliceAxis],
              xOperand->overlapElementsCount[spatialSliceAxis],
              xOperand->offsetBefore[spatialSliceAxis],
              xOperand->offsetAfter[spatialSliceAxis]);

    // Disable GraphSizeOptimizationSolver which enlarges the slices sizes, since slicing on spatial makes the operands
    // slice sizes dependent.
    strategy->getMmeSlicingData().enableGraphSizeOptimization = false;

    return spatialSliceAxis;
}

void CommonDimSlicingSolver::applyNonCommonSlicing(const pMmeSlicingStrategy& strategy)
{
    SLC_TRACE("Applying non-common dim slicing");
    // If no strategy was found for batch and spatial slicing, try to slice on non-common dim

    DimVector leftToRightTraversalPattern = {m_dimController.widthOutput().front(),
                                             m_dimController.heightOutput().front()};
    for (auto geometry : GAUDI_GEOMETRY)
    {
        pMmeSlicingStrategy s = std::static_pointer_cast<MmeSlicingStrategy>(strategy->clone(true /* reset alignment */));
        // set geometry and left to right walking pattern
        s->setOutputTraversalPattern(leftToRightTraversalPattern).setGeometry(geometry);
        // reduce non-common slice size in a square like manner
        doNonCommonDimSlicing(s);

        if (s->calculateMetrics().SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes)
        {
            // try to add single buffer.
            addSingleBufferStrategyIfFits(s);
        }
        else
        {
            addStrategy(s);
        }
    }
}

/* To get the slicing dims we use DimController which returns a list of all common dims
 * When tensor is 2D (originally or flattened) we want the inner dim as all other dims are flat(=1)
 * When tensor is 4D we want the batch dim which is the outer dimension in the list.
 * return: pair - {slice dim of operand A, sliced dim of operand B}  (only common dimension slicing)
 */
std::pair<unsigned, unsigned> CommonDimSlicingSolver::getSlicingDims(const pSlicedOperand& opA,
                                                                     const pSlicedOperand& opB) const
{
    unsigned sliceAxisA;
    unsigned sliceAxisB;
    if (gemmLikeMMENode(m_mmeNode, opA, opB))
    {
        sliceAxisA = m_dimController.commonDimOperandA().front();
        sliceAxisB = m_dimController.commonDimOperandB().front();
    }
    else
    {
        sliceAxisA = m_dimController.commonDimOperandA().back();
        sliceAxisB = m_dimController.commonDimOperandB().back();
    }
    return {sliceAxisA, sliceAxisB};
}

void CommonDimSlicingSolver::doNonCommonDimSlicing(pMmeSlicingStrategy& strategy)
{
    pSlicedOperand& wideOperand   = strategy->getMmeSlicingData().getWide();
    pSlicedOperand& narrowOperand = strategy->getMmeSlicingData().getNarrow();
    pSlicedOperand& outputOperand = strategy->getMmeSlicingData().masterOperand;

    unsigned wideSlicingDim         = strategy->getMmeSlicingData().getWideNonCommonSlicingDims().back();
    unsigned outputWideSlicingDim   = strategy->getMmeSlicingData().getWideOutputSlicingDims().back();
    unsigned narrowSlicingDim       = strategy->getMmeSlicingData().getNarrowNonCommonSlicingDims().back();
    unsigned outputNarrowSlicingDim = strategy->getMmeSlicingData().getNarrowOutputSlicingDims().back();

    /* Align the wide to mme geometry */
    if (wideOperand->chunkDimensions[wideSlicingDim] > strategy->getMMEWideGeometryInElements())
    {
        unsigned alignedWide = strategy->alignToMMEWide(wideOperand->chunkDimensions[wideSlicingDim], true);

        wideOperand->chunkDimensions[wideSlicingDim]         = alignedWide;
        outputOperand->chunkDimensions[outputWideSlicingDim] = alignedWide;
    }

    /* Align the narrow to mme geometry */
    if (narrowOperand->chunkDimensions[narrowSlicingDim] > strategy->getMMENarrowGeometryInElements())
    {
        unsigned alignedNarrow = strategy->alignToMMENarrow(narrowOperand->chunkDimensions[narrowSlicingDim], true);

        narrowOperand->chunkDimensions[narrowSlicingDim]       = alignedNarrow;
        outputOperand->chunkDimensions[outputNarrowSlicingDim] = alignedNarrow;
    }

    bool reduceWide = true;

    while (strategy->calculateMetrics().SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        if (reduceWide && wideOperand->chunkDimensions[wideSlicingDim] > strategy->getMMEWideGeometryInElements())
        {
            unsigned sliceSize = wideOperand->chunkDimensions[wideSlicingDim] - strategy->getMMEWideGeometryInElements();

            wideOperand->chunkDimensions[wideSlicingDim]         = sliceSize;
            outputOperand->chunkDimensions[outputWideSlicingDim] = sliceSize;
        }
        else if (narrowOperand->chunkDimensions[narrowSlicingDim] > strategy->getMMENarrowGeometryInElements())
        {
            unsigned sliceSize = narrowOperand->chunkDimensions[narrowSlicingDim] - strategy->getMMENarrowGeometryInElements();

            narrowOperand->chunkDimensions[narrowSlicingDim]       = sliceSize;
            outputOperand->chunkDimensions[outputNarrowSlicingDim] = sliceSize;
        }
        reduceWide = !reduceWide;

        /* check reduction is possible on both operands */
        if (wideOperand->chunkDimensions[wideSlicingDim] <= strategy->getMMEWideGeometryInElements() &&
            narrowOperand->chunkDimensions[narrowSlicingDim] <= strategy->getMMENarrowGeometryInElements())
        {
            break;
        }
    }
}

uint32_t CommonDimSlicingSolver::getCommonDimSliceChunkSize(const pNode&          mmeNode,
                                                            const pSlicedOperand& slicedOperandA,
                                                            bool                  isSlicedOnBatch) const
{
    if (!isSlicedOnBatch)
    {
        return SlicingBrain::knobs.minCDSizeForPartials;
    }

    HB_ASSERT(m_dimController.batchDim().size() == 1, "Single batch dim expected");
    const uint32_t nonCommonDimOperA = m_dimController.nonCommonDimOperandA().front();
    const pTensor& operandA          = mmeNode->getInput(0);

    const uint32_t commonDimSize = operandA->getDenseSizeInElements() / slicedOperandA->finalShape[nonCommonDimOperA];

    uint32_t batchDimSize = 1;
    for (auto batchDim : m_dimController.batchDim())
    {
        batchDimSize*= slicedOperandA->originalTensor->getSizeInElements(batchDim);
    }
    const uint32_t singleBatchCdSize = commonDimSize / batchDimSize;

    // Prefer min CD slice size above SlicingBrain::knobs.minCDSizeForPartials but limit to available batchDimSize
    const uint32_t desiredMin = div_round_up(SlicingBrain::knobs.minCDSizeForPartials, singleBatchCdSize);
    const uint32_t sliceChunk = std::min(desiredMin, batchDimSize);
    return sliceChunk;
}

bool CommonDimSlicingSolver::isSupportedNodeType()
{
    if (m_mmeNode->getNodeType() == Node::TYPE_DEDW)
    {
        return true;
    }
    if (GCFG_SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED.value() && isBundleGemm(getBundle()))
    {
        return true;
    }
    if (isBundleBatchGemm(getBundle()) && MMENodeFlattener::canFlattenMMENode(m_mmeNode))
    {
        // Batch gemm can be flattened if it's full broadcast on operand B only, which also means it's equivalent to
        // Gemm
        return true;
    }
    return false;
}
