#include "mme_slicing_strategy.h"
#include "spatial_slicing_solver.h"
#include "types.h"
#include "utils.h"
#include "habana_graph.h"
#include "slicing_utils.h"
#include "reuse_limit_calculator.h"
#include "flatten_mme.h"

NonCDSolver::NonCDSolver(const HalReader& halReader, const pBundle& b) : NonCDSolver(halReader, b, nullptr) {}

NonCDSolver::NonCDSolver(const HalReader& halReader, const pBundle& b, const pMmeSlicingStrategy& initialStrategy)
: MmeBundleSolver(halReader, b),
  m_initialStrategy(initialStrategy),
  c_sliceSizeFactor(ReuseLimitCalculator(halReader, b->getNodes().front()).getLimit())
{}

pMmeSlicingStrategy NonCDSolver::getInitialStrategy(DimVector& traversalPattern, MmeGeometry geometry)
{
    pMmeSlicingStrategy s = nullptr;
    if (m_initialStrategy)
    {
        s = std::static_pointer_cast<MmeSlicingStrategy>(m_initialStrategy->clone(true /* reset alignment */));
    }
    else
    {
        s = MmeSlicingStrategy::createStrategyForMMENode(m_halReader, m_mmeNode);
        s->setOutputIsInSRAM(false).setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
    }
    s->setOutputTraversalPattern(traversalPattern).setGeometry(geometry).setDoubleBuffer(true);
    return s;
}

unsigned NonCDSolver::getNarrowOutputSlicingDim(pMmeSlicingStrategy& strategy)
{
    return strategy->getMmeSlicingData().traversalPattern.front();
}

unsigned NonCDSolver::getWideOutputSlicingDim(pMmeSlicingStrategy& strategy)
{
    return strategy->getMmeSlicingData().traversalPattern.back();
}

bool NonCD2DSolver::effectiveForBundle()
{
    // check there is something to be sliced
    if (m_mmeNode->getOutput(0)->getDenseSizeInElements() <= 1)
    {
        SLC_TRACE("NonCD2DSolver:{}: output dense size = 1, this solver is not effective", HLLOG_FUNC);
        return false;
    }
    // check that the tensor is 2D.
    const pTensor& inputA = m_mmeNode->getInput(TENSOR_IFM);
    const pTensor& inputB = m_mmeNode->getInput(TENSOR_WEIGHT);
    if ((!SlicedOperandUtils::isTensor2D(inputA) ||
         !SlicedOperandUtils::isTensor2D(inputB)) &&      // one of the inputs isn't 2D and
        !MMENodeFlattener::canFlattenMMENode(m_mmeNode))  // the node can't be flattened
    {
        SLC_TRACE("NonCD2DSolver:{}: inputs are not 2D and node can't be flattened.", HLLOG_FUNC);
        return false;
    }
    // make sure the solver can actually slice the tensors to fit to SRAM
    createAllStrategies();
    return (getStrategies().size() > 0);
}

NonCD2DSolver::NonCD2DSolver(const HalReader& halReader, const pBundle& b, const pMmeSlicingStrategy& initialStrategy)
: NonCDSolver(halReader, b, initialStrategy)
{
}

bool NonCD2DSolver::shouldFlatten() const
{
    //If we have an initial strategy, Flattening or not should already be there.
    return m_initialStrategy == nullptr;
}

void NonCD2DSolver::createAllStrategies()
{
    if (getStrategies().size() > 0)
    {
        SLC_TRACE("NonCD2DSolver:{}: Strategies were already created by effectiveForBundle", HLLOG_FUNC);
        return;
    }

    HB_ASSERT(m_mmeNode, "should have MME Node in the bundle in this stage");

    SLC_INFO("NonCD2DSolver::{} - bundle {} ({}), reuse limit: {}",
             HLLOG_FUNC,
             getBundle()->index(),
             m_mmeNode->getNodeName(),
             c_sliceSizeFactor);

    DimVector dimOrder = getOutputSlicingDimList();
    // sort
    std::sort(dimOrder.begin(), dimOrder.end());
    for (auto geometry : GAUDI_GEOMETRY)
    {
        do
        {
            pMmeSlicingStrategy s = getInitialStrategy(dimOrder, geometry);
            if(shouldFlatten())
            {
                setFinalShapeInOperands(s->getMmeSlicingData().bundleTensors[0],
                                        s->getMmeSlicingData().bundleTensors[1],
                                        s->getMmeSlicingData().masterOperand,
                                        m_mmeNode);
            }
            // fix the narrow slice to MMe geometry.
            setNarrowSlice(s);
            // find the optimal slicing on the wide input (according to the walking pattern).
            findWideSlicing(s);
            do
            {
                if (s->calculateMetrics().SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes)
                {
                    addSingleBufferStrategyIfFits(s);
                    break;
                }
                else
                {
                    findNarrowSlicing(s);
                    addStrategy(s);
                    s = findStrategyWithSmallerWideSlicing(s);
                }
            } while(s);
        } while (std::next_permutation(dimOrder.begin(), dimOrder.end()));
    }
}

void NonCD2DSolver::setNarrowSlice(pMmeSlicingStrategy& strategy)
{
    // set narrow slice to initial value - MME geometry
    pSlicedOperand& narrowOperand = strategy->getMmeSlicingData().getNarrow();
    Settable<unsigned> narrowSlicingDim = getNarrowSlicingDim(strategy);
    if (!narrowSlicingDim.is_set()) return; //no slicing is needed
    unsigned narrowAxisSize = SlicedOperandUtils::getNarrowFullAxisSize(*strategy);
    unsigned sliceSize = std::min(narrowAxisSize, strategy->getMMENarrowGeometryInElements());
    narrowOperand->chunkDimensions[narrowSlicingDim.value()] = sliceSize;
    strategy->getMmeSlicingData().masterOperand->chunkDimensions[getNarrowOutputSlicingDim(strategy)] = sliceSize;
}

void NonCD2DSolver::findWideSlicing(pMmeSlicingStrategy& strategy)
{
    // after the narrow input is set with an initial value - start slicing wide input to more & more slices
    // find the first slicing strategy that fits in the SRAM capacity.
    pSlicedOperand& wideOperand = strategy->getMmeSlicingData().getWide();
    pSlicedOperand& outputOperand = strategy->getMmeSlicingData().masterOperand;
    unsigned wideAxisSizeInElements = SlicedOperandUtils::getWideFullAxisSize(*strategy);
    unsigned slicingFactor = std::ceil((float)wideAxisSizeInElements / c_sliceSizeFactor);
    if (strategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes &&
        wideAxisSizeInElements < strategy->getMMEWideGeometryInElements())
    {
        // no need for slicing
        return;
    }
    // try to find the optimal slicing on the wide side
    float    alignedWideSliceSizeInElements;
    Settable<unsigned> wideSlicingDim = getWideSlicingDim(strategy);
    if (!wideSlicingDim.is_set()) return; // no slicing is needed
    do
    {
        unsigned wideSliceSizeInElements = std::floor((float)wideAxisSizeInElements / (float)slicingFactor);
        alignedWideSliceSizeInElements = strategy->alignToMMEWide(wideSliceSizeInElements, false);
        wideOperand->chunkDimensions[wideSlicingDim.value()] = std::min((unsigned)alignedWideSliceSizeInElements, wideAxisSizeInElements);
        outputOperand->chunkDimensions[getWideOutputSlicingDim(strategy)] = std::min((unsigned)alignedWideSliceSizeInElements, wideAxisSizeInElements);
        slicingFactor++;
    } while (strategy->calculateMetrics().SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes &&
             alignedWideSliceSizeInElements > strategy->getMMEWideGeometryInElements());
}

pMmeSlicingStrategy NonCD2DSolver::findStrategyWithSmallerWideSlicing(const pMmeSlicingStrategy& origStrategy)
{
    pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(origStrategy->clone(true /* reset alignment */));

    pSlicedOperand& wideOperand = strategy->getMmeSlicingData().getWide();
    pSlicedOperand& outputOperand = strategy->getMmeSlicingData().masterOperand;
    const DimVector& wideOutputSlicingDims  = strategy->getMmeSlicingData().getWideOutputSlicingDims();
    unsigned wideAxisSizeInElements = SlicedOperandUtils::getWideFullAxisSize(*strategy);
    auto wideSlicingDim = getWideSlicingDim(strategy);
    if (!wideSlicingDim.is_set() || wideAxisSizeInElements < strategy->getMMEWideGeometryInElements())
    {
        // no need for slicing
        return nullptr;
    }
    unsigned slicedWideAxisSizeInElements = multiplyElements(outputOperand->chunkDimensions.data() + wideOutputSlicingDims.front(),
                                                             outputOperand->chunkDimensions.data() + wideOutputSlicingDims.back() + 1);
    unsigned slicingFactor = std::ceil((float)wideAxisSizeInElements /
                                       (float)slicedWideAxisSizeInElements);
    float alignedWideSliceSizeInElements;
    slicingFactor *= 2;
    // no need for 2 slices in double buffer if trivial solution fits.
    if (slicingFactor == 2) slicingFactor++;
    unsigned wideSliceSizeInElements = std::floor((float)wideAxisSizeInElements / (float)slicingFactor);
    alignedWideSliceSizeInElements = strategy->alignToMMEWide(wideSliceSizeInElements, false);
    // check that alignment didnt put us back on the prev slicing factor
    if (std::ceil((float)wideAxisSizeInElements / (float)alignedWideSliceSizeInElements) < slicingFactor)
    {
        return nullptr;
    }

    wideOperand->chunkDimensions[wideSlicingDim.value()] = std::min((unsigned)alignedWideSliceSizeInElements, wideAxisSizeInElements);
    outputOperand->chunkDimensions[getWideOutputSlicingDim(strategy)] = std::min((unsigned)alignedWideSliceSizeInElements, wideAxisSizeInElements);

    // check if valid
    bool     smallerThanMME   = alignedWideSliceSizeInElements < strategy->getMMEWideGeometryInElements();
    auto     narrowSlicingDim  = getNarrowSlicingDim(strategy);
    bool     smallerThanNarrow = false;
    if (narrowSlicingDim.is_set())
    {
        smallerThanNarrow = (alignedWideSliceSizeInElements <
                             strategy->getMmeSlicingData().getNarrow()->chunkDimensions[narrowSlicingDim.value()]);
    }

    if (smallerThanMME || smallerThanNarrow)
    {
        // No smaller slicing possible
        return nullptr;
    }
    return strategy;
}

void NonCD2DSolver::findNarrowSlicing(pMmeSlicingStrategy& strategy)
{
    // slicing strategy on the wide input is set - now increase the narrow input slice size to the possible maximum -
    // until it doesnt fit in the SRAM cap or the slice size reaches a pre-defined value.
    Settable<unsigned> narrowSlicingDim = getNarrowSlicingDim(strategy);
    if (narrowSlicingDim.is_set()) // slicing on narrow is needed.
    {
        unsigned narrowOutputSlicingDim = getNarrowOutputSlicingDim(strategy);
        pSlicedOperand& narrowOperand = strategy->getMmeSlicingData().getNarrow();
        pSlicedOperand& outputOperand = strategy->getMmeSlicingData().masterOperand;
        // now the wide slice is fixed - start increasing narrow slicing to maximum.
        unsigned narrowSliceSize = narrowOperand->chunkDimensions[narrowSlicingDim.value()];
        unsigned wideSliceSize = getWideSlicingDim(strategy).is_set()  ?
                strategy->getMmeSlicingData().getWide()->chunkDimensions[getWideSlicingDim(strategy).value()]
                : 1;
        unsigned narrowAxisSizeInElements = narrowOperand->finalShape[narrowSlicingDim.value()];
        while (narrowSliceSize < c_sliceSizeFactor &&
               narrowSliceSize < narrowAxisSizeInElements &&
               narrowSliceSize <= wideSliceSize)
        {
            narrowSliceSize += strategy->getMMENarrowGeometryInElements();
            narrowOperand->chunkDimensions[narrowSlicingDim.value()]       = narrowSliceSize;
            outputOperand->chunkDimensions[narrowOutputSlicingDim] = narrowSliceSize;
            if (strategy->calculateMetrics().SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes ||
                narrowSliceSize > narrowAxisSizeInElements)
            {
                // roll back to the last working slice size
                narrowSliceSize -= strategy->getMMENarrowGeometryInElements();
                narrowOperand->chunkDimensions[narrowSlicingDim.value()] = narrowSliceSize;
                outputOperand->chunkDimensions[narrowOutputSlicingDim] = narrowSliceSize;
                break;
            }
        }
    }
}

DimVector NonCD2DSolver::getOutputSlicingDimList()
{
    DimVector ret;
    TensorPtr output     = m_mmeNode->getOutput(0);
    SizeArray finalShape = MMENodeFlattener::getFlattenShape(output);

    for (const auto& dim : m_dimController.widthOutput())
    {
        // skip flatten dims
        if (finalShape[dim] != 1)
        {
            ret.push_back(dim);
        }
    }
    for (const auto& dim : m_dimController.heightOutput())
    {
        // skip flatten dims
        if (finalShape[dim] != 1)
        {
            ret.push_back(dim);
        }
    }
    HB_ASSERT (ret.size() <= 2, "Should slice output on 2 dims maximum");
    return ret;
}

Settable<unsigned> NonCD2DSolver::getNarrowSlicingDim(const pMmeSlicingStrategy& strategy) const
{
    MmeSlicingStrategy::MmeSlicingData& data = strategy->getMmeSlicingData();
    for (auto& dim : data.getNarrowNonCommonSlicingDims())
    {
        // skip flatten dims
        if (data.getNarrow()->finalShape[dim] == 1)
            continue;
        return {dim};
    }
    // all dim sizes are equal to 1 - no slicing is needed
    return {};
}

Settable<unsigned> NonCD2DSolver::getWideSlicingDim(const pMmeSlicingStrategy& strategy) const
{
    MmeSlicingStrategy::MmeSlicingData& data = strategy->getMmeSlicingData();
    for (auto& dim : data.getWideNonCommonSlicingDims())
    {
        // skip flatten dims
        if (data.getWide()->finalShape[dim] == 1)
            continue;
        return {dim};
    }
    // all dim sizes are equal to 1 - no slicing is needed
    return {};
}
