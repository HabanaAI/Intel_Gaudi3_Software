#include "metrics_calculator.h"

#include "hal_reader/gaudi1/hal_reader.h"
#include "mme_shared_input.h"
#include "slicing_brain.h"
#include "slicing_utils.h"
#include "utils.h"

SlicingStrategy::Metrics& MmeBundleMetricsCalculator::calculate()
{
    return calculate(false);
}

SlicingStrategy::Metrics& MmeBundleMetricsCalculator::calculate(bool recalcSramCapOnly)
{
    auto& slicingData = m_mmeStrategy->getMmeSlicingData();
    std::vector<pSlicedOperand> operands({slicingData.getNarrow(),
                                          slicingData.getWide(),
                                          slicingData.masterOperand});

    m_metrics->SRAMCapacity = calculateSRAMCapacity(operands);
    if (!recalcSramCapOnly)
    {
        m_metrics->HBMBandwidth   = calculateHBMBandwidth();
        m_metrics->MMEUtilization = calculateMMEUtilization();
        fixMMEUtilIfHBMOverflow();
        m_metrics->SBReuse        = calculateSBReuse();
        m_metrics->valid          = true;
    }
    return *m_metrics;
}

SlicingStrategy::Metrics& MmeBundleMetricsCalculator::recalculateSramCapacityOnly()
{
    return calculate(true);
}

uint64_t MmeBundleMetricsCalculator::calculateSRAMCapacity(const std::vector<pSlicedOperand>& operands)
{
    uint64_t totalSize = MetricsCalculator::calculateSRAMCapacity(operands);
    auto& slicingData = m_mmeStrategy->getMmeSlicingData();

    for (const pBundleExpansion& candidate : slicingData.getRoleCandidates())
    {
        if (candidate && candidate->nodeToStitch && !slicingData.isCandidateInBundle(candidate))
        {
            totalSize += candidate->additionalSRAMCapacity;
            for (pBundleExpansion& dependantCandidate: candidate->dependentCandidates)
            {
                totalSize += dependantCandidate->additionalSRAMCapacity;
            }
        }
    }
    return totalSize;
}

double MmeBundleMetricsCalculator::calculateHBMBandwidth() const
{
    double mmeProcessingTime = getMMEProcessingTime();
    double tpcProcessingTime = getTPCProcessingTime();
    double pipeliningFactor = (tpcProcessingTime > 0.0 && mmeProcessingTime > 0.0) ?
                              SlicingBrain::knobs.aggProcessingTimePipeliningFactor : 1.0;
    double aggregatedTime = std::max(mmeProcessingTime, tpcProcessingTime) * pipeliningFactor;

    uint64_t mmeTraffic = getMMEHBMTraffic();
    uint64_t tpcTraffic = getTPCHBMTraffic();

    return ((mmeTraffic + tpcTraffic) / aggregatedTime);
}

void MmeBundleMetricsCalculator::fixMMEUtilIfHBMOverflow()
{
    if (m_metrics->HBMBandwidth > SlicingBrain::knobs.hbmAvailableBWGBps)
    {
        float ratio = SlicingBrain::knobs.hbmAvailableBWGBps / m_metrics->HBMBandwidth;
        m_metrics->MMEUtilization *= ratio;
        m_metrics->HBMBandwidth = SlicingBrain::knobs.hbmAvailableBWGBps;
    }
}


unsigned MmeBundleMetricsCalculator::calculateSBReuse()
{
    pSlicedOperand& narrowOperand = m_mmeStrategy->getMmeSlicingData().getNarrow();
    unsigned narrowAxisSize = SlicedOperandUtils::getNarrowFullAxisSize(*m_mmeStrategy);
    unsigned narrowActivations = std::ceil((float)narrowAxisSize / m_mmeStrategy->getMMENarrowGeometryInElements());
    if (narrowActivations > 1)
    {
        const DimVector& narrowSlicingDims   = m_mmeStrategy->getMmeSlicingData().getNarrowNonCommonSlicingDims();
        const SizeArray& narrowSize = narrowOperand->chunkDimensions;
        unsigned narrowAxisSliceSize = multiplyElements(narrowSize.data() + narrowSlicingDims.front(),
                                                        narrowSize.data() + narrowSlicingDims.back() + 1);
        return (narrowAxisSliceSize / m_mmeStrategy->getMMENarrowGeometryInElements());
    }
    return 0;
}

// MME utilization indicates how much is our slicing effective, we want it to be as close as possible to 1 -
// which means each MME activation uses all its multipliers.
// MME util = SizeToCalculate / (MME Size * num of MME activations)
float MmeBundleMetricsCalculator::calculateMMEUtilization()
{
    float MMEUtilization = 0;
    bool isBF16 = m_mmeStrategy->getMmeSlicingData().getWide()->originalTensor->getElementType()==syn_type_bf16;

    // calculate the denominator - how much is the MME active
    unsigned MMETotalSize   = m_halReader.getMmeVectorSize() * m_halReader.getMmeVectorSize() / (isBF16 ? 1 : 4);
    float mmeActivations = getNumOfMMETetrises();

    // calculate the numerator - how much we need to calculate.
    uint64_t narrowAxisSize = SlicedOperandUtils::getNarrowFullAxisSize(*m_mmeStrategy);
    uint64_t wideAxisSize   = SlicedOperandUtils::getWideFullAxisSize(*m_mmeStrategy);
    uint64_t sizeToProcess  = narrowAxisSize * wideAxisSize;

    // in case there is a shared input candidate - need to update the sizeToProccess and the mmeActivations.
    bool validForRole;
    pBundleExpansion candidate = SharedMMEInputCandidateHandler::getCandidateFromStrategy(m_mmeStrategy, validForRole);
    if (candidate && candidate->nodeToStitch && validForRole)
    {
        addCandidateUtilization(candidate, sizeToProcess, mmeActivations);
    }

    if (mmeActivations != 0)
    {
        MMEUtilization = static_cast<float>(sizeToProcess) / (MMETotalSize * mmeActivations);
    }
    return MMEUtilization;
}

void MmeBundleMetricsCalculator::addCandidateUtilization(const pBundleExpansion& candidate, uint64_t& sizeToProcess, float& mmeActivations)
{
    const pSlicedOperand& slaveOutput = candidate->slaveOperands.getOutput();
    uint64_t slaveSizeToProcess = slaveOutput->originalTensor->getDenseSizeInElements();
    // on dedw QRS are calculated in parallel by the MME stack - they dont need to be added to the sizeToProcess
    if (candidate->nodeToStitch->getNodeType() == Node::TYPE_DEDW && slaveOutput->originalTensor->getDim() > 2)
    {
        slaveSizeToProcess /= m_mmeStrategy->getMmeSlicingData().getQRSSize();
    }

    if (slaveSizeToProcess != 1)
    {
        sizeToProcess += slaveSizeToProcess;
    }

    mmeActivations += getNumOfMMETetrisesOfCandidate(candidate);
}

double MmeBundleMetricsCalculator::getMMEProcessingTime() const
{
    MmeSlicingStrategy::MmeSlicingData& slicingData = m_mmeStrategy->getMmeSlicingData();
    uint64_t minRollupCycles = m_mmeStrategy->getMMEWideGeometryInElements();
    uint64_t cyclesPerTetris = slicingData.getCommonDimSize() * slicingData.getQRSSize();
    double mmeTime = getNumOfMMETetrises() * std::max(cyclesPerTetris, minRollupCycles) / SlicingBrain::knobs.freqGHz;
    mmeTime += getSharedOperandMMEProcessingTime();

    return mmeTime;
}

// MME Processing time is calculated as num of tetrises * cycles per tetris / MME Frequency
// cycles per tetris = qrsSize * common-dim size
double MmeBundleMetricsCalculator::getSharedOperandMMEProcessingTime() const
{
    bool isValidForRole;
    const pBundleExpansion& candidate = SharedMMEInputCandidateHandler::getCandidateFromStrategy(m_mmeStrategy, isValidForRole);
    if (!candidate || !candidate->nodeToStitch) return 0.0;
    MmeDimController controller(candidate->nodeToStitch);
    uint64_t qrsSize = multiplyElements(controller.qrsSizes().begin(), controller.qrsSizes().end());
    uint64_t cdSize = 1;
    const SizeArray& sizes = candidate->slaveOperands.getInput()->finalShape;
    for (auto& dim : controller.commonDimOperandA())
    {
        cdSize *= sizes[dim];
    }
    uint64_t minRollupCycles = m_mmeStrategy->getMMEWideGeometryInElements();
    uint64_t cyclesPerTetris = std::max(minRollupCycles, qrsSize * cdSize);
    return getNumOfMMETetrisesOfCandidate(candidate) * cyclesPerTetris / SlicingBrain::knobs.freqGHz;
}

double MmeBundleMetricsCalculator::getTPCProcessingTime() const
{
    // Approximation for TPC processing time in bundling case, which include simple element-wise kernels.
    // The processing time is considered to be the time it takes to read and write the TPC operands to HBM.

    return getTPCHBMTraffic() / SlicingBrain::knobs.hbmAvailableBWGBps;
}

uint64_t MmeBundleMetricsCalculator::getMMEHBMTraffic() const
{
    return getMMEWideHBMTraffic() +
           getMMENarrowHBMTraffic() +
           getMMEOutputHBMTraffic() +
           getMMESharedOperandHbmTraffic();
}

uint64_t MmeBundleMetricsCalculator::getMMEWideHBMTraffic() const
{
    MmeSlicingStrategy::MmeSlicingData& slicingData = m_mmeStrategy->getMmeSlicingData();
    const pSlicedOperand  & wide      = slicingData.getWide();
    const pBundleExpansion& candidate = slicingData.getRoleCandidates()[BundleExpansion::WideInputProducer];
    if (candidate && candidate->nodeToStitch)
    {
        return 0;
    }
    return wide->originalTensor->getDenseSizeInBytes();
}

uint64_t MmeBundleMetricsCalculator::getMMENarrowHBMTraffic() const
{
    MmeSlicingStrategy::MmeSlicingData& slicingData = m_mmeStrategy->getMmeSlicingData();
    const pSlicedOperand  & narrow    = slicingData.getNarrow();
    const pBundleExpansion& candidate = slicingData.getRoleCandidates()[BundleExpansion::NarrowInputProducer];
    if (candidate && candidate->nodeToStitch)
    {
        return 0;
    }

    uint64_t traffic = narrow->originalTensor->getDenseSizeInBytes();
    if (!SlicedOperandUtils::isTriviallySliced(narrow))
    {
        // This is an approximation since the edge slices are brought only ~half the times in snake walking pattern.
        // It can be made more precise in the future if necessary
        // For now assume that for each wide slice in the non-CD, we need to bring the entire narrow operand from HBM.
        for (unsigned slicingDim : slicingData.getWideNonCommonSlicingDims())
        {
            traffic *= SlicedOperandUtils::nofSlices(slicingData.getWide(), slicingDim);
        }
    }
    return traffic;
}

uint64_t MmeBundleMetricsCalculator::getMMEOutputHBMTraffic() const
{
    MmeSlicingStrategy::MmeSlicingData& slicingData = m_mmeStrategy->getMmeSlicingData();
    const pBundleExpansion& candidate = slicingData.getRoleCandidates()[BundleExpansion::OutputConsumer];
    if (candidate && candidate->nodeToStitch && candidate->stitchedOperand)
    {
        return 0;
    }
    return slicingData.masterOperand->originalTensor->getDenseSizeInBytes();
}

uint64_t MmeBundleMetricsCalculator::getMMESharedOperandHbmTraffic() const
{
    uint64_t         sharedMMETraffic = 0ull;
    bool             validForRole;
    pBundleExpansion candidate = SharedMMEInputCandidateHandler::getCandidateFromStrategy(m_mmeStrategy, validForRole);

    if (!candidate || !candidate->nodeToStitch) return sharedMMETraffic;

    // The output must be written to HBM.
    for (const auto& operandTupple : candidate->slaveOperands)
    {
        // Unstitched operands are read/written from/to HBM
        sharedMMETraffic += operandTupple.second->originalTensor->getDenseSizeInBytes();
    }
    if (!validForRole)
    {
        // Invalid candidate means that the "stitchedOperand" is not really stitched, i.e it's read from HBM
        sharedMMETraffic += candidate->stitchedOperand->originalTensor->getDenseSizeInBytes();
    }

    return sharedMMETraffic;
}

// Calculate the number of activations (tetrises) on the master node output
// This is an approximation as the last slices are smaller in size
// # tetrises = activations for single slice * num of slices.
double MmeBundleMetricsCalculator::getNumOfMMETetrises() const
{
    MmeSlicingStrategy::MmeSlicingData& slicingData = m_mmeStrategy->getMmeSlicingData();
    return calculateNumOfMMETetrises(slicingData.masterOperand,
                                     slicingData.getWideOutputSlicingDims(),
                                     slicingData.getNarrowOutputSlicingDims());
}

double MmeBundleMetricsCalculator::getNumOfMMETetrisesOfCandidate(const pBundleExpansion& candidate) const
{
    // sanity check
    if (candidate->role != BundleExpansion::SharedInputConsumer) return 0.0;

    MmeSlicingStrategy::MmeSlicingData& slicingData = m_mmeStrategy->getMmeSlicingData();
    // figure out which operand is wide and which is the narrow.
    bool isSharedOperandWide = (slicingData.getWide() == candidate->stitchedOperand);
    const pSlicedOperand& wideOperand = isSharedOperandWide ? candidate->stitchedOperand : candidate->slaveOperands.getInput();
    const pSlicedOperand& slaveOutput = candidate->slaveOperands.getOutput();
    // get the relevant output slicing dims.
    MmeDimController controller(candidate->nodeToStitch);
    unsigned wideInputIdx = candidate->nodeToStitch->getInputIndexOfTensor(wideOperand->originalTensor);
    const DimVector& wideSlicingDims   = (wideInputIdx == 0) ? controller.heightOutput() : controller.widthOutput();
    const DimVector& narrowSlicingDims = (wideInputIdx == 0) ? controller.widthOutput() : controller.heightOutput();

    return calculateNumOfMMETetrises(slaveOutput,
                                     wideSlicingDims,
                                     narrowSlicingDims);
}

// Calculate the number of activations (tetrises) in an MME operation
// This is an approximation as the last slices can be smaller in size than the other slices.
// # tetrises = activations for single slice * num of slices.
double MmeBundleMetricsCalculator::calculateNumOfMMETetrises(const pSlicedOperand& outputOperand,
                                                             const DimVector&      wideOutputSlicingDims,
                                                             const DimVector&      narrowOutputSlicingDims) const
{
    // calculate activations per slice = ceil(size of each slice / mme geometry)
    // we need to calculate this separately for the wide and narrow operands.
    uint64_t wideOperandAxisSize = 1;
    uint64_t narrowOperandAxisSize = 1;
    for (uint32_t dim : wideOutputSlicingDims)
    {
        wideOperandAxisSize *= outputOperand->chunkDimensions[dim];
    }
    for (uint32_t dim : narrowOutputSlicingDims)
    {
        narrowOperandAxisSize *= outputOperand->chunkDimensions[dim];
    }

    double wideActivations = std::ceil((double)wideOperandAxisSize / m_mmeStrategy->getMMEWideGeometryInElements());
    double narrowActivations = std::ceil((double)narrowOperandAxisSize / m_mmeStrategy->getMMENarrowGeometryInElements());
    double activationsPerSlice = wideActivations * narrowActivations;

    return activationsPerSlice * static_cast<double>(SlicedOperandUtils::nofSlices(outputOperand));
}


uint64_t MmeBundleMetricsCalculator::getTPCHBMTraffic() const
{
    // Count all TPC candidates non-stitched operands

    uint64_t totalTraffic = 0ull;

    MmeSlicingStrategy::MmeSlicingData& slicingData = m_mmeStrategy->getMmeSlicingData();
    std::list<pBundleExpansion> allCandidates = slicingData.getInvalidCandidates();
    allCandidates.insert(allCandidates.end(),
                         slicingData.getRoleCandidates().begin(),
                         slicingData.getRoleCandidates().end());

    for (const pBundleExpansion& candidate : allCandidates)
    {
        if (candidate && candidate->nodeToStitch && candidate->role != BundleExpansion::SharedInputConsumer)
        {
            for (const auto& tensor : candidate->nodeToStitch->getOperands())
            {
                pTensor stitchedAfterReshape = tensor;
                if (candidate->reshapeNode && candidate->reshapeNode->getOutput(0) == tensor)
                {
                    stitchedAfterReshape = candidate->reshapeNode->getInput(0);
                }
                else if (candidate->reshapeNode && candidate->reshapeNode->getInput(0) == tensor)
                {
                    stitchedAfterReshape = candidate->reshapeNode->getOutput(0);
                }
                if (stitchedAfterReshape != candidate->stitchedOperand->originalTensor)
                {
                    // Approximation - for now assume strided do not count to HBM traffic more than non strided
                    totalTraffic += stitchedAfterReshape->getDenseSizeInBytes();
                }
            }
        }
    }

    return totalTraffic;
}

MetricsCalculator::MetricsCalculator(const HalReader&          halReader,
                                     SlicingStrategy*          strategy,
                                     SlicingStrategy::Metrics* metrics)
: m_halReader(halReader), m_strategy(strategy), m_metrics(metrics)
{}

uint64_t MetricsCalculator::calculateSRAMCapacity(const std::vector<pSlicedOperand>& operands)
{
    uint64_t totalSize = 0;
    // sum all input sizes
    for (const auto& operand : operands)
    {
        if (operand->resideInSRAM)
        {
            uint32_t numBuffers = SlicedOperandUtils::isTriviallySliced(operand) ? 1 : operand->numOfBuffers;
            totalSize += SlicedOperandUtils::getSliceSizeInBytes(operand) * numBuffers;
        }
    }

    return totalSize;
}

SlicingStrategy::Metrics& MetricsCalculator::calculate()
{
    m_metrics->SRAMCapacity   = calculateSRAMCapacity(m_strategy->getSlicingData().getSlicedOperands());
    m_metrics->HBMBandwidth   = 0;
    m_metrics->MMEUtilization = 0;
    m_metrics->SBReuse        = 0;
    m_metrics->valid          = true;
    return *m_metrics;
}

MmeSlicingStrategy::Metrics& MetricsCalculator::recalculateSramCapacityOnly()
{
    m_metrics->SRAMCapacity = calculateSRAMCapacity(m_strategy->getSlicingData().getSlicedOperands());
    return *m_metrics;
}
