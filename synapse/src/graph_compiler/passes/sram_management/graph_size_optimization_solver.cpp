#include "graph_size_optimization_solver.h"
#include "defs.h"
#include "slicing_utils.h"

GraphSizeOptimizationSolver::GraphSizeOptimizationSolver(const HalReader&          halReader,
                                                         const pBundle&            bundle,
                                                         const SlicingStrategyPtr& initialStrategy)
: MmeBundleSolver(halReader, bundle), m_initialStrategy(initialStrategy)
{
}

void GraphSizeOptimizationSolver::createAllStrategies()
{
    SLC_TRACE("GraphSizeOptimizationSolver:{}", HLLOG_FUNC);
    findSlicedOperandAndDims();
    for (const slicedOperandAndDimList& sameProportionSlicedOpAndDims : m_slicedOperandAndDims)
    {
        // the name "sameProportionSlicedOpAndDims" just to indicate they are going to be increased together, keeping
        // the original proportion between them
        createStrategiesWithBiggerSlices(sameProportionSlicedOpAndDims);
    }
}

// prepare the lists of operands+dims that were sliced by the original solver, so that each list will represent a
// group of operands that the slice size in those dims can be increased together by this solver.
// currently it creates maximum 3 lists : common dims, non common dims, and the union of both.
void GraphSizeOptimizationSolver::findSlicedOperandAndDims()
{
    slicedOperandAndDimList nonCommonOpAndDims;
    slicedOperandAndDimList commonOpAndDims; // i.e. when slicing both inputs on batch in the common solver

    const StrategySlicingData& slicingData = m_initialStrategy->getSlicingData();

    for (const auto& slicedOpAndDim : slicingData.getSlicedOperandsAndDims())
    {
        const pSlicedOperand& slicedOp = slicedOpAndDim.first;
        unsigned dim = slicedOpAndDim.second;
        if (slicedOp == slicingData.masterOperand) // output is always "non common"
        {
            nonCommonOpAndDims.push_back({slicedOpAndDim});
        }
        else
        {
            if ((slicedOp == slicingData.bundleTensors[0] && m_dimController.isCommonDimOperandA(dim)) ||
                (slicedOp == slicingData.bundleTensors[1] && m_dimController.isCommonDimOperandB(dim)))
            {
                commonOpAndDims.push_back({slicedOpAndDim});
            }
            else
            {
                nonCommonOpAndDims.push_back({slicedOpAndDim});
            }
        }
    }

    if (!nonCommonOpAndDims.empty())
    {
        m_slicedOperandAndDims.push_back(nonCommonOpAndDims);
    }
    if (!commonOpAndDims.empty())
    {
        m_slicedOperandAndDims.push_back(commonOpAndDims);
    }

    if (!nonCommonOpAndDims.empty() && !commonOpAndDims.empty())
    {
        // now, since every list inside m_slicedOperandAndDims will be handled as a group of (operators,dimensions) that
        // should have their slice size increased together, and we don't want to give up the option to increase the
        // slice size on multiple lists together, let's add another list to m_slicedOperandAndDims containing a union
        // of the other two lists
        slicedOperandAndDimList allSlicedOpAndDims;
        allSlicedOpAndDims.swap(nonCommonOpAndDims);
        allSlicedOpAndDims.splice(allSlicedOpAndDims.end(), commonOpAndDims);
        m_slicedOperandAndDims.push_back(allSlicedOpAndDims);
    }
}

void GraphSizeOptimizationSolver::createStrategiesWithBiggerSlices(const slicedOperandAndDimList& sameProportionSlicedOpAndDims)
{
    SlicingStrategyPtr prevStrategy = m_initialStrategy;
    const int maxLoops = 10000;
    int loopCounter = 0;
    for (; loopCounter < maxLoops; ++loopCounter) // essentially "while(true)", but safer
    {
        // reset alignment in order not to miss an opportunity to add a strategy to the bundle (when the new strategy is
        // added, we will try again to align the tensors if possible under sram size restrictions)
        SlicingStrategyPtr strategy = prevStrategy->clone(true /* reset alignment */);
        const StrategySlicingData& slicingData = strategy->getSlicingData();
        const std::vector<pSlicedOperand>& slicedOperands = slicingData.getSlicedOperands();
        for (const slicedOperandAndDim& singleOpAndDim : sameProportionSlicedOpAndDims)
        {
            auto iterMatchingOperand = find_if(slicedOperands.begin(), slicedOperands.end(),
                                               [&singleOpAndDim] (const pSlicedOperand& op)
                                               { return op->originalTensor == singleOpAndDim.first->originalTensor; } );
            HB_ASSERT(iterMatchingOperand != slicedOperands.end(), "failed to find matching operand");
            SlicedOperand& operand = **iterMatchingOperand;
            unsigned dim = singleOpAndDim.second;
            operand.chunkDimensions[dim] = std::ceil(operand.chunkDimensions[dim] *
                                                     SlicingBrain::knobs.graphSizeOptimizationMultiplicationFactor);
            if (operand.chunkDimensions[dim] > operand.finalShape[dim])
            {   // since we only increase the slice sizes, if this one doesn't fit - there's no point to keep trying
                // making more strategies from the given list
                return;
            }
        }
        if (strategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
        {
            strategy->setGraphSizeOptimized(true);
            addStrategy(strategy, false);
        }
        else
        {
            // if this slicing doesn't fit in sram, larger ones will surely not fit - return
            return;
        }
        prevStrategy = strategy;
    }
    HB_ASSERT(loopCounter < maxLoops, "infinite loop in GraphSizeOptimizationSolver!");
}
