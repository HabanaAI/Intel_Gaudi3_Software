#include "mme_slicing_strategy.h"
#include "pattern_solvers.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "slicing_utils.h"
#include "graph_size_optimization_solver.h"
#include <unordered_set>
#include "flatten_mme.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "dma_transpose_node.h"

void Solver::addStrategy(SlicingStrategyPtr s, bool printLog /*=true */)
{
    // Finalize strategy metrics and indicators.
    s->alignNumBuffers();
    s->alignWalkingPattern();
    s->calculateMetrics();
    s->tryAlignToCacheLine();
    s->alignShapeTensorsSlicing(getBundle());
    if (validateStrategy(s))
    {
        if (printLog)
        {
            SLC_TRACE("Adding Strategy - ");
            s->printLog(0, synapse::LogManager::LogType::SRAM_SLICE);
        }
        m_strategies.push_back(s);
    }
    else
    {
        HB_ASSERT(false, "Strategy did not pass validation");
    }
}

pNode Solver::getFirstMMENodeFromBundle(const Bundle& bundle)
{
    for (const pNode& n : bundle.getNodes())
    {
        if (HabanaGraph::runsOnMME(n))
        {
            return n;
        }
    }
    return nullptr;
}

pNode Solver::getFirstTPCNodeFromBundle(const Bundle& bundle)
{
    for (const pNode& n : bundle.getNodes())
    {
        if (HabanaGraph::runsOnTPC(n))
        {
            return n;
        }
    }
    return nullptr;
}

void Solver::addStrategies(const SlicingStrategyList& newStrategies)
{
    m_strategies.insert(m_strategies.end(), newStrategies.begin(), newStrategies.end());
}

void Solver::addSingleBufferStrategyIfFits(SlicingStrategyPtr strategy)
{
    // check if single buffer strategy with the current slicing can fit in SRAM capacity
    strategy->setDoubleBuffer(false);
    if (strategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        addStrategy(strategy);
    }
}

void Solver::addStrategiesIfNotExist(const SlicingStrategyList& newStrategies)
{
    std::unordered_set<SlicingStrategyPtr, SlicingStrategy::Hasher, SlicingStrategy::IsEqual> strategySet;
    strategySet.insert(m_strategies.begin(), m_strategies.end());

    // explicit iteration in order to log new (unique) added strategies
    for (const SlicingStrategyPtr& newStrategy : newStrategies)
    {
        if (strategySet.find(newStrategy) == strategySet.end())
        {
            addStrategy(newStrategy);
            strategySet.insert(newStrategy);
        }
    }
}

bool Solver::validateStrategy(const SlicingStrategyPtr& s)
{
    HB_ASSERT(s->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes,
              "strategy SRAM capcity ({}) exceeds the max capacity ({})",
              s->calculateMetrics().SRAMCapacity,
              SlicingBrain::knobs.maxSRAMCapInBytes);

    // validate slices are not larger then original tensor dim size.
    for (const auto& operand : s->getSlicingData().getSlicedOperands())
    {
        SizeArray originalTensorSizes = operand->finalShape;
        pTensor   originalTensor      = operand->originalTensor;
        for (unsigned i = 0; i < originalTensorSizes.size(); i++)
        {
            auto maxPadding = operand->offsetAfter[i] + operand->offsetBefore[i];
            CHECK_RET_FALSE(operand->chunkDimensions[i] <= originalTensorSizes[i] + maxPadding,
                            "Slice size is larger then original size for tensor - {}",
                            originalTensor->getName());
        }
    }
    return true;
}

// (common code for all other solvers) - once the "official" solvers finish creating their strategies,
// we call this function in order to extend the list of strategies by adding more strategies that have larger slices
// than the original ones (this way the final graph could be smaller)
void Solver::AddStrategiesForGraphSizeOptimization()
{
    if (GCFG_SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_ENABLED.value())
    {
        if (isBundleBatchGemm(getBundle()))
        {
            SLC_DEBUG("Skipping strategy slice size increase of {} due to node being one of the 3 BatchGemm Types!",
                      getBundle()->getNodes().front()->getNodeName());
            return;
        }

        SLC_DEBUG("{} - trying to add more strategies with bigger slices", HLLOG_FUNC);
        SlicingStrategyList& strategies = getStrategies();
        size_t sizeBefore = strategies.size();
        SlicingStrategyList newStrategies;
        for (const auto& strategy : strategies)
        {
            if (strategy->getSlicingData().enableGraphSizeOptimization)
            {
                GraphSizeOptimizationSolver solver(m_halReader, getBundle(), strategy);
                solver.createAllStrategies();
                newStrategies.splice(newStrategies.end(), solver.getStrategies());
            }
        }
        SLC_TRACE("{} - going to merge {} new strategies into existing list", HLLOG_FUNC, newStrategies.size());
        // it is possible that the new created strategy will match a strategy already created by the original solver (in
        // example the 4d solver can create strategies s1,s2 such as s2 has twice the slice sizes od s1), but we don't
        // really want the final strategy list to have duplicates, so we merge them:
        addStrategiesIfNotExist(newStrategies);
        size_t sizeAfter = getStrategies().size();
        size_t addedStrategies = sizeAfter - sizeBefore;
        if (addedStrategies > 0)
        {
            SLC_DEBUG("{} - added {} additional strategies (from {} to {})",
                      HLLOG_FUNC,
                      addedStrategies,
                      sizeBefore,
                      sizeAfter);
        }
        else
        {
            SLC_DEBUG("{} - no additional strategies were added", HLLOG_FUNC);
        }
    }
}

SlicingStrategyList Solver::getUniqueStrategies()
{
    const auto& allStrategies = getStrategies();
    if (GCFG_SRAM_SLICER_COST_MODEL_ENABLED.value())
    {
        // The cost-model ignores some of the slicing parameters (for example: MME geometry),
        // and therefore some of the strategies can be eliminated to reduce the compilation time.
        std::unordered_set<SlicingStrategyPtr, SlicingStrategy::Hasher, SlicingStrategy::IsEqual> uniqueStrategies(
            1 /* minimal number of buckets to use on initialization */,
            SlicingStrategy::Hasher(false),
            SlicingStrategy::IsEqual(false));
        uniqueStrategies.insert(allStrategies.begin(), allStrategies.end());
        return SlicingStrategyList(uniqueStrategies.begin(), uniqueStrategies.end());
    }
    return allStrategies;
}

SlicingStrategyList Solver::getReducedStrategyList()
{
    SlicingStrategyList sortedStrategies = getUniqueStrategies();

    struct
    {
        bool operator()(const SlicingStrategyPtr& a, const SlicingStrategyPtr& b) const
        {
            pMmeSlicingStrategy aMmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(a);
            pMmeSlicingStrategy bMmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(b);
            return SlicedOperandUtils::nofSlices(aMmeStrategy->getMmeSlicingData().masterOperand) <
                   SlicedOperandUtils::nofSlices(bMmeStrategy->getMmeSlicingData().masterOperand);
        }
    } StrategyComparatorByNumOfSlices;
    // Sort the strategies by number of output slices, strategies with less slices will be first.
    sortedStrategies.sort(StrategyComparatorByNumOfSlices);

    // Trying to reduce number of strategies to improve the compilation time.
    // Iterate over the sorted strategies list until numOfSlicesThreshold is reached and there is still enough room
    // in SRAM for stitching and for pre-fetching the next bundle.
    // The next strategies will be removed.
    auto iter = sortedStrategies.begin();
    while (iter != sortedStrategies.end())
    {
        const pMmeSlicingStrategy& s = std::static_pointer_cast<MmeSlicingStrategy>(*iter);
        ++iter;
        const auto wideSliceSize = SlicedOperandUtils::getSliceSizeInBytes(s->getMmeSlicingData().getWide());
        const auto narrowSliceSize = SlicedOperandUtils::getSliceSizeInBytes(s->getMmeSlicingData().getNarrow());
        const auto outSliceSize = SlicedOperandUtils::getSliceSizeInBytes(s->getMmeSlicingData().masterOperand);
        const auto numOfOutputSlices = SlicedOperandUtils::nofSlices(s->getMmeSlicingData().masterOperand);
        // Additional SRAM size for output consumer and slave MME stitching, * 2 for double buffer.
        const auto sramAdditionalSizeForStitching = (outSliceSize + std::max(wideSliceSize, narrowSliceSize)) * 2;
        const auto totalSram                      = s->calculateMetrics().SRAMCapacity + sramAdditionalSizeForStitching;
        // Ensure both short prefix/suffix and enough room for stitching and pre-fetching the next bundle.
        if ((numOfOutputSlices > SlicingBrain::knobs.numOfSlicesThreshold) &&
            (totalSram < (SlicingBrain::knobs.maxSRAMCapInBytes / 2)))
        {
            break;
        }
    }
    if (iter != sortedStrategies.end())
    {
        // Delete the strategies from iter (include) to the end of the list.
        sortedStrategies.erase(iter, sortedStrategies.end());
    }
    return sortedStrategies;
}

void MmeBundleSolver::setFinalShapeInOperands(pSlicedOperand operandA,
                                              pSlicedOperand operandB,
                                              pSlicedOperand output,
                                              pNode mmeNode)
{
    if (MMENodeFlattener::canFlattenMMENode(mmeNode))
    {
        SLC_DEBUG("Node {} should be flattened by this solver", mmeNode->getNodeName());
        HB_ASSERT(operandA->originalTensor == mmeNode->getInput(0), "operand mismatch");
        HB_ASSERT(operandB->originalTensor == mmeNode->getInput(1), "operand mismatch");
        HB_ASSERT(output->originalTensor   == mmeNode->getOutput(0), "operand mismatch");
        operandA->finalShape = MMENodeFlattener::getFlattenShape(operandA->originalTensor);
        operandB->finalShape = MMENodeFlattener::getFlattenShape(operandB->originalTensor);
        output->finalShape = MMENodeFlattener::getFlattenShape(output->originalTensor);

        //We initialize chunk dimensions as the final shape. They will be Sliced later
        operandA->chunkDimensions = operandA->finalShape;
        operandB->chunkDimensions = operandB->finalShape;
        output->chunkDimensions = output->finalShape;
    }
}

unsigned MmeBundleSolver::getNextSpatialSlicingDim(const DimVector& dims, const unsigned currentDim) const
{
    // dims are ordered by the dim controller from inner to outer, so slicing order is from last to first.
    // iterate the list backwards to find the next inner spatial dim to slice on
    auto dimIt = std::find(dims.rbegin(), dims.rend(), currentDim);
    HB_ASSERT(dimIt != dims.rend(), "Trying to get a spatial dimension of the operand, which doesn't exist");
    dimIt++;
    HB_ASSERT(dimIt != dims.rend(), "Trying to get the next spatial dimension of the operand, which doesn't exist");
    unsigned nextDim = *dimIt;
    // assert next dim is indeed spatial. it's the caller responsibility to call correctly
    HB_ASSERT((nextDim == DIM_H || nextDim == DIM_W || nextDim == DIM_D_FOR_5D_TENSOR),
              "Next dim is not spatial. invalid call");
    return nextDim;
}

bool TrivialSolver::effectiveForBundle()
{
    uint64_t inputTensorsSize = 0;
    // check that all inputs can fit in SRAM capacity
    for (const pTensor& t : m_mmeNode->getInputs())
    {
        if (t)
        {
            inputTensorsSize += t->getDenseSizeInBytes();
        }
    }
    return SlicingBrain::knobs.maxSRAMCapInBytes >= inputTensorsSize;
}

void TrivialSolver::createAllStrategies()
{
    SLC_DEBUG("Trivial SRAM usage: applying on bundle {}", getBundle()->getName());

    pMmeSlicingStrategy s = MmeSlicingStrategy::createStrategyForMMENode(m_halReader, m_mmeNode);
    setFinalShapeInOperands(s->getMmeSlicingData().bundleTensors[0],
                            s->getMmeSlicingData().bundleTensors[1],
                            s->getMmeSlicingData().masterOperand,
                            m_mmeNode);

    // Trivial solution - all inputs resides in SRAM, output in HBM.
    s->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setOutputIsInSRAM(false);
    s->setDoubleBuffer(false);

    addStrategy(s);
}

bool TPCScalarPipeSolver::effectiveForBundle()
{
    pNode node = getFirstTPCNodeFromBundle(*getBundle());
    TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
    HB_ASSERT_PTR(tpcNode);
    // The TPC node has no scalar inputs
    auto deviceId = deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType());
    if (tpcNode->getScalarPipeInputsSize(deviceId) == 0)
    {
        return false;
    }
    std::vector<bool> scalarPipeStatus = tpcNode->getInputsScalarPipeStatus(deviceId);
    if (scalarPipeStatus.size() != tpcNode->getNumInputs())
    {
        return false;
    }

    unsigned tensorIdx = 0;
    unsigned tensorsSize = 0;
    for (const pTensor& t : tpcNode->getInputs())
    {
        if (t && (scalarPipeStatus[tensorIdx] || t->inSram()))
        {
            tensorsSize += t->getDenseSizeInBytes();
        }
        tensorIdx++;
    }
    for (const pTensor& t : tpcNode->getOutputs())
    {
        if (t && t->inSram())
        {
            tensorsSize += t->getDenseSizeInBytes();
        }
    }

    return tensorsSize > 0 && SlicingBrain::knobs.maxSRAMCapInBytes >= tensorsSize;
}

void TPCScalarPipeSolver::createAllStrategies()
{
    SLC_DEBUG("TPC Scalar Pipe SRAM usage: applying on bundle {}", getBundle()->getName());
    pNode node = getFirstTPCNodeFromBundle(*getBundle());
    SlicingStrategyPtr strategy = SlicingStrategy::createStrategy(m_halReader, node);

    TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
    HB_ASSERT_PTR(tpcNode);
    auto devId = deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType());

    std::vector<bool> scalarPipeStatus = tpcNode->getInputsScalarPipeStatus(devId);
    HB_ASSERT(scalarPipeStatus.size() == tpcNode->getNumInputs(), "inputs number mismatch");

    KernelInstantiationWrapper tpcInstanceWrapper;
    if (!tpcNode->getInfoInstance(tpcInstanceWrapper, devId, true))
    {
        SLC_ERR("Failed to get instance information for TPC node: {}", tpcNode->getNodeName());
    }
    // Set scalar pipe inputs in SRAM
    for (unsigned idx = 0; idx < tpcNode->getNumInputs(); idx++)
    {
        strategy->setInputIsInSRAM(idx, tpcNode->getInput(idx)->inSram() || scalarPipeStatus[idx]);
    }
    for (unsigned idx = 1; idx < tpcNode->getNumOutputs(); idx++)
    {
        strategy->setInputIsInSRAM(idx + tpcNode->getNumInputs() - 1, tpcNode->getOutput(idx)->inSram());
    }

    strategy->setOutputIsInSRAM(tpcNode->getOutput(0)->inSram());
    strategy->setDoubleBuffer(false);

    auto& slicingData = strategy->getSlicingData();
    if (slicingData.masterOperand.get()->originalTensor->getDim() == 1)
    {
        slicingData.traversalPattern = SlicedOperandTraversalPattern::LEFT_TO_RIGHT_1D;
    }

    addStrategy(strategy);
}

class StrategyNormalizer : public StrategyVisitor
{
public:
    void setBasicNormalization(const SlicingStrategy::Metrics& met)
    {
        normalizedMet.HBMBandwidth = (SlicingBrain::knobs.hbmAvailableBWGBps - static_cast<float>(met.HBMBandwidth)) / SlicingBrain::knobs.hbmAvailableBWGBps;
        normalizedMet.SRAMCapacity = met.SRAMCapacity/static_cast<float>(SlicingBrain::knobs.maxSRAMCapInBytes);
        normalizedMet.DoubleBuffered = met.isDoubleBuffered ? 1 : 0;
        normalizedMet.valid = met.valid;
    }

    virtual void visit(SlicingStrategy& strategy) override
    {
        const MmeSlicingStrategy::Metrics& met = strategy.calculateMetrics();
        setBasicNormalization(met);
    }

    virtual void visit(MmeSlicingStrategy& strategy) override
    {
        const MmeSlicingStrategy::Metrics& met = strategy.calculateMetrics();
        const float maxSBReuse = (float)SlicedOperandUtils::getNarrowFullAxisSize(strategy) /
                                 (float)strategy.getMMENarrowGeometryInElements();
        setBasicNormalization(met);
        normalizedMet.MMEUtilization = (met.MMEUtilization < 0.02f ) ? 0.02f : met.MMEUtilization;
        normalizedMet.SBReuse = met.SBReuse / maxSBReuse;
        normalizedMet.walkingPattern = (strategy.getSlicingData().getWalkingDir() == StrategySlicingData::WalkingDir::LeftToRight) ?
                                       1 : 0;
        normalizedMet.valid = met.valid;
    }

    StrategyComparator::NormalizedMetrics normalizedMet;
};

StrategyComparator::NormalizedMetrics StrategyComparator::normalizeMetrics(const SlicingStrategyPtr& strategy) const
{
    StrategyNormalizer normalizer;
    strategy->accept(normalizer);

    return normalizer.normalizedMet;
}

// Given normalized metrics (each between 0-1), returns a weighted sum of the metrics between 0-1.
float StrategyComparator::getScore(const StrategyComparator::NormalizedMetrics& met) const
{
    static constexpr float
            w_util    = 32.,
            w_bw      = 16.,
            w_db      = 8.,
            w_cap     = 4.,
            w_walkPat = 2.,
            w_sb      = 1.,
            w_total   = w_util + w_db + w_bw + w_walkPat + w_cap + w_sb;
    float score = (w_util    * met.MMEUtilization +
                   w_bw      * met.HBMBandwidth   +
                   w_cap     * met.SRAMCapacity   +
                   w_walkPat * met.walkingPattern +
                   w_db      * met.DoubleBuffered +
                   w_sb      * met.SBReuse        ) / w_total;
    return score;
}

bool StrategyComparator::operator()(const SlicingStrategyPtr& a, const SlicingStrategyPtr& b) const
{
    NormalizedMetrics normMetA = normalizeMetrics(a);
    NormalizedMetrics normMetB = normalizeMetrics(b);
    HB_ASSERT(normMetA.valid && normMetB.valid, "Strategy metrics should be calculated");
    bool ret = (getScore(normMetA) <= getScore(normMetB));
    return ret;
}

bool DMATransposeSolver::effectiveForBundle()
{
    if (GCFG_ENABLE_DMA_TRANSPOSE_SOLVER.value() == false)
    {
        return false;
    }
    HB_ASSERT(getBundle()->getNodes().size() == 1, "DMATransposeSolver expected 1 node in bundle");
    NodePtr                           node             = getBundle()->getNodes().front();
    std::shared_ptr<DMATransposeNode> dmaTransposeNode = std::dynamic_pointer_cast<DMATransposeNode>(node);
    HB_ASSERT_PTR(dmaTransposeNode);
    if (dmaTransposeNode->isFullyUtilized())
    {
        SLC_DEBUG("Dma Transpose Node {} is fully utilized, DMATranspose solver is irrelevant",
                  dmaTransposeNode->getNodeName());
        return false;
    }
    // We use this optimization for simple 2D matrix transpose.
    if (dmaTransposeNode->getInput(0)->getDim() != 2)
    {
        SLC_DEBUG("Dma Transpose Node {} is not 2D, DMATranspose solver is irrelevant",
                  dmaTransposeNode->getNodeName());
        return false;
    }
    // Default values for Gaudi1 were found based on experimentation
    if (dmaTransposeNode->getInput(0)->getSizeInBytes(1) < GCFG_DMA_TRANSPOSE_SOLVER_MAX_SCD_SIZE.value() &&
        dmaTransposeNode->getInput(0)->getSizeInBytes(0) > GCFG_DMA_TRANSPOSE_SOLVER_MIN_FCD_SIZE.value())
    {
        return true;
    }
    return false;
}
void DMATransposeSolver::createAllStrategies()
{
    HB_ASSERT(getBundle()->getNodes().size() == 1, "DMATransposeSolver expected 1 node in bundle");
    NodePtr                           node             = getBundle()->getNodes().front();
    std::shared_ptr<DMATransposeNode> dmaTransposeNode = std::dynamic_pointer_cast<DMATransposeNode>(node);
    HB_ASSERT_PTR(dmaTransposeNode);
    SlicingStrategyPtr strategy     = SlicingStrategy::createStrategy(m_halReader, node);
    unsigned           slices       = 1;
    unsigned           sramCapacity = SlicingBrain::knobs.maxSRAMCapInBytes;
    unsigned           tensorSize   = node->getInput(0)->getTotalSizeInBytes();

    slices = tensorSize / sramCapacity + (tensorSize % sramCapacity != 0);

    auto& bundleSlicingData = strategy->getSlicingData();
    auto& inputOperand      = bundleSlicingData.bundleTensors.front();
    auto& outputOperand     = bundleSlicingData.masterOperand;
    inputOperand->chunkDimensions[0] /= slices;
    outputOperand->chunkDimensions[1] /= slices;
    bundleSlicingData.setOperandSliceBackwardMapping(
        bundleSlicingData.masterOperand,
        AccessPatternSliceMapper::createBwdMapping(node, {inputOperand}, {outputOperand}));
    strategy->setOutputIsInSRAM(true);
    strategy->setDoubleBuffer(false);
    addStrategy(strategy);
}
