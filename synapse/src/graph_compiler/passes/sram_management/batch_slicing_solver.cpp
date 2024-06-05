#include "batch_slicing_solver.h"

#include "flatten_mme.h"
#include "habana_graph.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "post_slicing_op_handler.h"
#include "slicing_utils.h"

NonCommon4DSolver::NonCommon4DSolver(const HalReader&           halReader,
                                     const pBundle&             b,
                                     const pMmeSlicingStrategy& initialStrategy)
: NonCDSolver(halReader, b, initialStrategy)
{
}

bool NonCommon4DSolver::effectiveForBundle()
{
    if (!m_mmeNode)
    {
        SLC_TRACE("NonCommon4DSolver:{}: did not find MME node in bundle: {}!", HLLOG_FUNC, getBundle()->getName());
        return false;
    }

    // This solver handles only convolution nodes, as it expects a 4D shape (at least)
    auto* pConv = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    if (pConv == nullptr)
    {
        SLC_TRACE("NonCommon4DSolver:{}: MME node is not of ConvBase type in bundle: {}!", HLLOG_FUNC, getBundle()->getName());
        return false;
    }

    //TODO SW-18666 Need to allow 4D solver for flattenable nodes.
    if (MMENodeFlattener::canFlattenMMENode(m_mmeNode))
    {
        SLC_TRACE("NonCommon4DSolver:{}: MME node can be flattened in bundle: {}!", HLLOG_FUNC, getBundle()->getName());
        return false;
    }

    unsigned wideSliceDim = m_dimController.nonCommonDimOperandA().back();
    unsigned wideBatchDim = m_dimController.batchDim().back();
    if (wideSliceDim != wideBatchDim)
    {
        SLC_TRACE("NonCommon4DSolver:{}: (bundle {}) operand A last non common dimension {} is not batch, this solver"
                  " is not effective",
                  HLLOG_FUNC,
                  getBundle()->getName(),
                  wideSliceDim);
        return false;
    }
    if (m_mmeNode->getOutput(0)->getDim() < 4)
    {
        SLC_TRACE("NonCommon4DSolver:{}: (bundle {}) output is not 4d, this solver is not effective",
                  HLLOG_FUNC,
                  getBundle()->getName());
        return false;
    }
    // make sure the solver can actually slice the tensors to fit to SRAM
    createAllStrategies();
    return (getStrategies().size() > 0);
}

DimVector NonCommon4DSolver::getWalkingPattern(unsigned numHeightSlicingDims)
{
    HB_ASSERT(numHeightSlicingDims <= m_dimController.heightOutput().size(),
              "Trying to slice on more spatial dims than exist");
    // Set walking pattern left to right, since operand A (X or dy) is assumed to be larger.
    DimVector walkingPattern;
    // walk left - on the output width
    walkingPattern.push_back(m_dimController.widthOutput().front());
    // walk down - on output height, from inner dimension to outer
    auto lastHeightDimsIter = std::prev(m_dimController.heightOutput().end(), numHeightSlicingDims);
    walkingPattern.insert(walkingPattern.end(), lastHeightDimsIter, m_dimController.heightOutput().end());

    return walkingPattern;
}

Settable<unsigned> NonCommon4DSolver::getNarrowSlicingDim(const pMmeSlicingStrategy& strategy) const
{
    return {m_dimController.nonCommonDimOperandB().front()};
}

Settable<unsigned> NonCommon4DSolver::getWideSlicingDim(const pMmeSlicingStrategy& strategy) const
{
    return {m_dimController.nonCommonDimOperandA().back()};
}

bool NonCommon4DSolver::computeSlicingForBatch(unsigned int batch, pMmeSlicingStrategy &strategy)
{
    SLC_TRACE("NonCommon4DSolver:{}: {}", HLLOG_FUNC, batch);

    pSlicedOperand&  outputOperand = strategy->getMmeSlicingData().masterOperand;
    pSlicedOperand&  wideOperand   = strategy->getMmeSlicingData().getWide();
    unsigned wideSlicingDim        = getWideSlicingDim(strategy).value();
    unsigned outputWideSlicingDim  = getWideOutputSlicingDim(strategy);

    HB_ASSERT(wideSlicingDim == m_dimController.batchDim().back(), "This solver requires batch dim to be valid");

    /* Slice the wide operand on the batch dimension */
    wideOperand->chunkDimensions[wideSlicingDim]         = batch;
    outputOperand->chunkDimensions[outputWideSlicingDim] = batch;

    return computeSlicingForNarrow(strategy);
}

bool NonCommon4DSolver::sliceSpatialDim(unsigned        wideSlicingDim,
                                        TSize           outputSliceSize,
                                        pSlicedOperand& wideOperand,
                                        pSlicedOperand& outputOperand,
                                        OffsetArray&    slicePadBefore,
                                        OffsetArray&    slicePadAfter)
{
    // Slice the output operand
    TSize origOutputSize                           = outputOperand->chunkDimensions[wideSlicingDim];
    outputOperand->chunkDimensions[wideSlicingDim] = outputSliceSize;
    TSize inputSliceSize = getInputSliceSize(outputOperand->chunkDimensions,
                                             outputOperand->originalTensor->getDim(),
                                             wideSlicingDim,
                                             wideOperand->chunkDimensions[wideSlicingDim]);

    if (inputSliceSize >= wideOperand->chunkDimensions[wideSlicingDim])
    {
        // restore output size
        outputOperand->chunkDimensions[wideSlicingDim] = origOutputSize;
        SLC_TRACE("NonCommon4DSolver:{} - Input slice size is equal to the original size for output slice size {} - no slicing", HLLOG_FUNC, outputSliceSize);
        return false;
    }
    // Slice the wide operand
    wideOperand->chunkDimensions[wideSlicingDim] = inputSliceSize;
    // Calc the overlap of the input slice
    auto* pConv                                       = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    int overlap = pConv->getInputROIOverlapForDim(TENSOR_IFM, wideSlicingDim);
    wideOperand->overlapElementsCount[wideSlicingDim] = overlap;

    // Set the convolution padding before, so the first slice size and other slices offset will be adjusted accordingly.
    // the X operand is the operand which requires them.
    pSlicedOperand xOperand = (wideOperand->originalTensor == pConv->getXOperand()) ? wideOperand : outputOperand;
    if (pConv->getNodeType() == Node::TYPE_DEDX)
    {
        // Overlap in dy means the first slice starts in a negative offset, but it's not really part of the tensor.
        // Need to reduce this offset from the dy slices like the padding is reduced from X slices.
        wideOperand->offsetBefore[wideSlicingDim] = overlap;
        // Calc the X operand middle slices padding
        pConv->getXStrideAlignedROIPaddingForDim(wideSlicingDim, outputSliceSize, slicePadBefore[wideSlicingDim], slicePadAfter[wideSlicingDim]);
        // Don't count the last slice if it's padding slice
        outputOperand->countPaddingOnlySlice = false;
    }
    ConvParamsIndices convIdx = ConvBaseNode::dimIndexToConvParamsIndices(wideSlicingDim);
    xOperand->offsetBefore[wideSlicingDim] = pConv->getConvolutionParams().padding[convIdx.paddingBeforeIndex];
    xOperand->offsetAfter[wideSlicingDim] = pConv->getConvolutionParams().padding[convIdx.paddingAfterIndex];
    // Set the minimal valid last slice size, to clip X operand lines, which are not enough to create another Y operand line.
    // In dedx - those lines can't be calculated. The last slice must include enough lines to complete the padding before lines
    //           to at least actual kernel size. Otherwise there wasn't kernel calculation with the last lines in fwd.
    // In fwd - slicePadBefore is 0, and the min is set to the actual kernel size.
    xOperand->minValidSliceSize[wideSlicingDim] = pConv->getDimActualKernelSize(wideSlicingDim) - slicePadBefore[wideSlicingDim];

    unsigned wideNumSlices = SlicedOperandUtils::nofSlices(wideOperand, wideSlicingDim);
    unsigned outNumSlices = SlicedOperandUtils::nofSlices(outputOperand, wideSlicingDim);

    // For dedx nodes, calculate the expected last slice size for the current dimension, to determine whether there are
    // leftover tensor elements that are not part of any slice. Motivation: see [SW-112633] for an instance where it
    // occurs.
    if (pConv->getNodeType() == Node::TYPE_DEDX)
    {
        unsigned expectedLastSliceSize =
            xOperand->originalTensor->getSizeInElements(wideSlicingDim) -                            // whole dimension size
            ((xOperand->chunkDimensions[wideSlicingDim] - xOperand->offsetBefore[wideSlicingDim]) +  // first slice size
             ((outNumSlices - 2) * xOperand->chunkDimensions[wideSlicingDim])                        // middle slices size
            );

        if (expectedLastSliceSize > xOperand->chunkDimensions[wideSlicingDim])
        {
            // Set how much we need to extend the last slice in order for all the slices to cover all the elements in
            // the tensor
            xOperand->extraLeftoverAfter[wideSlicingDim] =
                expectedLastSliceSize - xOperand->chunkDimensions[wideSlicingDim];
        }
    }

    // Make sure there are enough wide slices to create output slices. The solution is built based on the output slices,
    // and throws extra input slice if required.
    HB_ASSERT(wideNumSlices >= outNumSlices, "NonCommon4DSolver: wrong number of slices");

    wideOperand->requiresTensorView = true; // the opernad with the opverlap
    xOperand->requiresTensorView = true; // X operand may always have non trivial slicing

    SLC_TRACE("NonCommon4DSolver:{} - input slice {}, output slice {}, overlap {}, slicePadBefore {}, slicePadAfter {}",
              HLLOG_FUNC,
              inputSliceSize,
              outputSliceSize,
              overlap,
              slicePadBefore[wideSlicingDim],
              slicePadAfter[wideSlicingDim]);
    return true;
}

// Returns true if the strategy is sliced to fit SRAM. Sets validSlicing to true if the strategy is sliced to valid
// sizes, regardless if it fits SRAM, or false if the slice sizes can't be used.
bool NonCommon4DSolver::computeSlicingForSpatial(TSize                lastDimOutputSliceSize,
                                                 pMmeSlicingStrategy& strategy,
                                                 unsigned             numSpatialDimsToSlice,
                                                 bool&                validSlicing)
{
    SLC_TRACE("NonCommon4DSolver:{} - output slice size {} for {} spatial dims",
              HLLOG_FUNC,
              lastDimOutputSliceSize,
              numSpatialDimsToSlice);

    pSlicedOperand& outputOperand        = strategy->getMmeSlicingData().masterOperand;
    pSlicedOperand& wideOperand          = strategy->getMmeSlicingData().getWide();
    unsigned        wideSlicingDim       = getWideSlicingDim(strategy).value();
    unsigned        outputWideSlicingDim = getWideOutputSlicingDim(strategy);
    OffsetArray     slicePadBefore = {0}, slicePadAfter = {0};
    auto*           pConv = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());

    // Slice the batch dim to minimal size
    HB_ASSERT(wideSlicingDim == m_dimController.batchDim().back(), "This solver requires batch dim to be valid");
    wideOperand->chunkDimensions[wideSlicingDim]         = 1;
    outputOperand->chunkDimensions[outputWideSlicingDim] = 1;

    // Slice the spatial dims - from outer to inner dims.
    // The first outer dims are sliced to minimal size. The last dim is sliced to lastDimOutputSliceSize
    while (numSpatialDimsToSlice > 0)
    {
        // move to the next dim to slice
        wideSlicingDim       = getNextSpatialSlicingDim(m_dimController.nonCommonDimOperandA(), wideSlicingDim);
        outputWideSlicingDim = getNextSpatialSlicingDim(m_dimController.heightOutput(), outputWideSlicingDim);
        HB_ASSERT(wideSlicingDim == outputWideSlicingDim, "Slicing on non matching spatial dims");

        // Slice on the first spatial dimensions (besides the last) to the minimal possible size, the last requested
        // dimension according to lastDimOutputSliceSize
        TSize minOutSize =
            std::min(pConv->getMinSpatialDimOutputROI(wideSlicingDim), outputOperand->chunkDimensions[wideSlicingDim]);
        TSize sliceOutSize = (numSpatialDimsToSlice > 1) ? minOutSize : lastDimOutputSliceSize;
        bool     inputSliced =
            sliceSpatialDim(wideSlicingDim, sliceOutSize, wideOperand, outputOperand, slicePadBefore, slicePadAfter);
        if (!inputSliced && numSpatialDimsToSlice == 1)
        {
            // Input wasn't sliced for the last spatial dim. For the first spatial dims it's ok that they are not
            // sliced. For the last spatial dim - there's no solution with this slicing size.
            validSlicing = false;
            return false;
        }
        numSpatialDimsToSlice--;
    }
    validSlicing = true;

    // Set the post slicing handler to handle the sliced nodes padding
    pSlicedOperand xOperand      = (wideOperand->originalTensor == pConv->getXOperand()) ? wideOperand : outputOperand;
    xOperand->postSlicingHandler = std::make_shared<PostSlicingConvHandler>(slicePadBefore, slicePadAfter);

    // Block the optimization which enlarges the slices sizes, since slicing on spatial dim makes the operands slice
    // sizes dependent.
    strategy->getMmeSlicingData().enableGraphSizeOptimization = false;

    return computeSlicingForNarrow(strategy);
}

bool NonCommon4DSolver::computeSlicingForNarrow(pMmeSlicingStrategy& strategy)
{
    pSlicedOperand&  narrowOperand = strategy->getMmeSlicingData().getNarrow();
    pSlicedOperand&  outputOperand = strategy->getMmeSlicingData().masterOperand;
    unsigned narrowSlicingDim = getNarrowSlicingDim(strategy).value();
    unsigned outputNarrowSlicingDim = getNarrowOutputSlicingDim(strategy);
    // in the relevant nodes this is a single dim.
    TSize narrowAxisSize = SlicedOperandUtils::getNarrowFullAxisSize(*strategy);
    TSize gNarrow = strategy->getMMENarrowGeometryInElements();

    if (narrowAxisSize <= gNarrow)
    {
        /* In this case narrow isn't sliced */
        return strategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes;
    }

    int sliceSize = 0;
    uint64_t SRAMCapacity = 0;

    // The initial function "if" returns if this condition isn't fulfilled. Asserting in case of future changes
    HB_ASSERT(narrowAxisSize > gNarrow, "Narrow axis size is expected to be larger than geometry");
    /* Gradually increase the narrow size, consider both SRAM capacity and and reuse factor */
    do
    {
        /* If the minimum is narrowAxisSize a single iteration will be preformed */
        sliceSize += gNarrow;
        narrowOperand->chunkDimensions[narrowSlicingDim] = sliceSize;
        outputOperand->chunkDimensions[outputNarrowSlicingDim] = sliceSize;

        /* Compute the SRAM capacity based on the latest slicing */
        SRAMCapacity = strategy->calculateMetrics().SRAMCapacity;

        /* Roll back to the last working slice size */
        if (SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes ||
            sliceSize > narrowAxisSize)
        {
            sliceSize                                       -= gNarrow;
            narrowOperand->chunkDimensions[narrowSlicingDim] = sliceSize;
            outputOperand->chunkDimensions[outputNarrowSlicingDim] = sliceSize;
            if (sliceSize <= 0)
            {
                /* Even with the smallest slice of the narrow there is no space in SRAM for current batch*/
                // Set to min slice size, to keep the strategy valid for the calling function to try single buffer
                narrowOperand->chunkDimensions[narrowSlicingDim]       = gNarrow;
                outputOperand->chunkDimensions[outputNarrowSlicingDim] = gNarrow;
                return false;
            }
            break;
        }
        SLC_TRACE("NonCommon4DSolver:{}: narrow/output slice size: {}", HLLOG_FUNC, sliceSize);

    }while (narrowAxisSize > sliceSize &&
            SRAMCapacity < SlicingBrain::knobs.maxSRAMCapInBytes &&
            sliceSize + gNarrow < SlicingBrain::knobs.maxNarrowSliceSize);

    return true;
}

std::string geometry2String(const MmeGeometry &geometry)
{
    switch (geometry)
    {
        case gaudi_geometry_4wx1h:
            return "4Wx1H";
        case gaudi_geometry_2wx2h:
            return "2Wx2H";
        case gaudi_geometry_1wx4h:
            return "1Wx4H";
        default:
            SLC_ERR("Invalid geometry!");
            break;
    }

    return "Invalid Geometry!";
}

void NonCommon4DSolver::createStrategiesForGeometry(DimVector& walkingPattern, const MmeGeometry& geometry)
{
    SLC_TRACE("NonCommon4DSolver:{}: {}", HLLOG_FUNC, geometry2String(geometry));

    bool strategyAdded = false; // can't use getStrategies(), which returns for all geometries

    // Find strategies with sliced batch
    pMmeSlicingStrategy strategy     = getInitialStrategy(walkingPattern, geometry);

    // calc the bounds for the batch slicing loop
    unsigned       batch           = 1;
    pSlicedOperand wideOperand     = strategy->getMmeSlicingData().getWide();
    unsigned       wideSliceDim    = getWideSlicingDim(strategy).value();
    unsigned       origBatch       = wideOperand->chunkDimensions[wideSliceDim];
    TSize          wideFullSize    = SlicedOperandUtils::getSliceSizeInElements(wideOperand);
    TSize          wideCDSize      = calculateDimsSize(wideOperand->chunkDimensions, m_dimController.commonDimOperandA());
    TSize          wideSpatialSize = (wideFullSize / wideCDSize) / origBatch;

    /* slice wide    - batch dim, then h dim
     * slice narrow  - k     dim
     * Iterate over the possible batches/spatial dim sizes (for wide operand).
     * For each batch+spatial dim compute all possible narrow slicing possibilities */
    do
    {
        if (computeSlicingForBatch(batch, strategy))
        {
            addStrategy(strategy);
            strategyAdded = true;
        }
        else
        {
            SLC_TRACE(
                "NonCommon4DSolver:{}: No strategies for batch size {}, stop searching batch slicing for geometry {}",
                HLLOG_FUNC,
                batch,
                geometry2String(geometry));
            break;
        }

        strategy = getInitialStrategy(walkingPattern, geometry);

        batch++;
    } while (batch <= origBatch &&
             batch * wideSpatialSize <= (float) SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon4D)/* HBM reuse factor */;

    // If strategy was found - no need to slice on spatial
    if (strategyAdded)
    {
        return;
    }
    // else - try to slice on spatial dims
    unsigned maxSpatialDimsToSlice = GCFG_SRAM_SLICER_CONV_MULTI_SPATIAL_DIMS_SLICE_ENABLED.value() ? 2 : 1;
    pMmeSlicingStrategy lastValidStrategy;

    for (unsigned numSpatialDimsToSlice = 1; numSpatialDimsToSlice <= maxSpatialDimsToSlice; numSpatialDimsToSlice++)
    {
        // Advance the wide operand dim to the next spatial dim. Use this dim index for input and for output,
        // as they must be aligned. computeSlicingForSpatial will assert this assumption.
        wideSliceDim = getNextSpatialSlicingDim(m_dimController.nonCommonDimOperandA(), wideSliceDim);
        if (!isSpatialSlicingSupported(wideSliceDim))
        {
            SLC_TRACE("NonCommon4DSolver:{}: Slicing dim {} is not supported, stop searching for geometry {}",
                      HLLOG_FUNC,
                      wideSliceDim,
                      geometry2String(geometry));
            return;
        }

        // set walking pattern, with output height dimensions slicing on the requested number of spatial plus 1 for
        // batch
        DimVector walkingPatternLeftToRight = getWalkingPattern(numSpatialDimsToSlice + 1);
        strategy                          = getInitialStrategy(walkingPatternLeftToRight, geometry);

        // calc the bounds for the spatial slicing loop.
        // the output is sliced first to select the input slices size, so the loop checks the output size.
        TSize origOutputDimSize = strategy->getSlicingData().masterOperand->chunkDimensions[wideSliceDim];
        TSize maxSliceSize = 0, minSliceSize = 0;
        int   step = 0;
        getOutputSpatialDimSlicingBoundsAndStep(origOutputDimSize, wideSliceDim, maxSliceSize, step, minSliceSize);

        // loop on the spatial dim size until the inputs can fit to SRAM
        bool  validSlicing = false;
        TSize sliceDimSize = maxSliceSize;  // TODO SW-25560 - reverse order for multiple strategies
        do
        {
            if (computeSlicingForSpatial(sliceDimSize, strategy, numSpatialDimsToSlice, validSlicing))
            {
                addStrategy(strategy);
                strategyAdded = true;
                SLC_TRACE("NonCommon4DSolver:{}: Added strategy with double buffer for geometry {}",
                          HLLOG_FUNC,
                          geometry2String(geometry));

                // for now create a single strategy. TODO SW-25560 - create more strategies
                return;
            }
            // else - the strategy doesn't fit SRAM. If it has valid slicing - keep it to later try with single buffer
            if (validSlicing)
            {
                lastValidStrategy = strategy;
            }
            strategy = getInitialStrategy(walkingPatternLeftToRight, geometry);

            sliceDimSize -= step;
        } while (sliceDimSize >= minSliceSize);
    }
    // try last strategy with single buffer
    if (lastValidStrategy)
    {
        lastValidStrategy->setDoubleBuffer(false);
        if (lastValidStrategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
        {
            addStrategy(lastValidStrategy);
            strategyAdded = true;
            SLC_TRACE("NonCommon4DSolver:{}: Added strategy with single buffer for geometry {}",
                      HLLOG_FUNC,
                      geometry2String(geometry));

            // for now create a single strategy. TODO SW-25560 - create more strategies
            return;
        }
    }
    if(!strategyAdded)
    {
        SLC_TRACE("NonCommon4DSolver:{}: No strategies were found, stop searching for geometry {}",
                  HLLOG_FUNC,
                  geometry2String(geometry));
    }
}

void NonCommon4DSolver::createAllStrategies()
{
    SLC_TRACE("NonCommon4DSolver:{}", HLLOG_FUNC);

    if (getStrategies().size() > 0)
    {
        SLC_TRACE("NonCommon4DSolver:{}: Strategies were already created by effectiveForBundle", HLLOG_FUNC);
        return;
    }

    /* Walk left to right  */
    DimVector walkingPattern = getWalkingPattern();

    if (walkingPattern.empty())
    {
        return;
    }

    /* Run over MME geometries*/
    for (auto geometry : GAUDI_GEOMETRY)
    {
        createStrategiesForGeometry(walkingPattern, geometry);
    }
}

TSize NonCommon4DSolver::calculateDimsSize(const SizeArray& sizes, const DimVector& dims)
{
    TSize fullSize = 1;
    for (auto& dim : dims)
    {
        fullSize *= sizes[dim];
    }
    return fullSize;
}

// Returns the parameters for slicing the dimension -
// The initial size to slice, the step to get the next sizes to try, and the limit size to stop searching
// Currently sets a descending loop from original tensor size to the lower bound of the slice size.
void NonCommon4DSolver::getOutputSpatialDimSlicingBoundsAndStep(TSize    origDimSize,
                                                                unsigned dim,
                                                                TSize&   startSize,
                                                                int&     step,
                                                                TSize&   limitSize)
{
    // Set default spatial slicing params - currently set to start from largest and make is smaller.
    // TODO SW-25560 - reverse for multiple strategies
    startSize = origDimSize;
    step = 1;
    limitSize = 1;

    auto* pConv = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    HB_ASSERT(pConv != nullptr, "4D solver handles only conv types nodes");

    // calc minimal slice size for the spatial dimension
    // set the min dim size of the output according to minimal input size (larger than overlap and initial offest)
    // the input is assumed to be no smaller than the output, so this is a lower bound
    limitSize = pConv->getMinSpatialDimOutputROI(dim);
    // make sure the original dim size is a boundary to the loop.
    // since the limit is set to Y operand the same way as to X operand, it might be smaller than the spatial limit.
    // in this case the loop will run only once.
    limitSize = std::min(limitSize, origDimSize);

    // calc the initial size and step to change dimension size, such that the dedx slices are aligned to stride
    const synConvolution3DParamsV2& convParams = pConv->getConvolutionParams();
    ConvParamsIndices convIdx = ConvBaseNode::dimIndexToConvParamsIndices(dim);
    unsigned stride = convParams.stride[convIdx.spatialIndex];
    if ((pConv->getNodeType() == Node::TYPE_DEDX) && (stride != 1))
    {
        // set the init size and step to enforce output slice size start is stride-aligned.
        startSize = origDimSize - (origDimSize % stride);
        step = stride; // keep the output slices stride-aligned
    }
}

// Calculates the input size of the sliced dimension, given the output slice size.
// TODO handle paddingType here <===== PADDING_TYPE
TSize NonCommon4DSolver::getInputSliceSize(const SizeArray& outputSliceSizes, unsigned numDims, unsigned slicedDim, TSize maxInputSize)
{
    auto* pConv = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    // Map the first slice
    CoordArray base = {0};
    TensorShape outputShape(numDims, outputSliceSizes, base);
    // getInputShape for conv and dedx expect output index 0 and input index 0, which have different names.
    // TENSOR_IFM = TENSOR_DEDY = 0, TENSOR_OFM = TENSOR_DEDW = TENSOR_DEDX = 0,
    TensorShape inputShape = pConv->getInputShape(outputShape, TENSOR_OFM, TENSOR_IFM);
    // The function doesn't clip the result, which may be larger than the real input in case of padding. clip it.
    TSize inputSliceSize = std::min(inputShape.getSize(slicedDim), maxInputSize);
    return inputSliceSize;
}

bool NonCommon4DSolver::isSpatialSlicingSupported(unsigned dim)
{
    auto* conv = dynamic_cast<ConvBaseNode*>(m_mmeNode.get());
    HB_ASSERT_PTR(conv);
    return conv->isSpatialSlicingSupported(dim);
}
