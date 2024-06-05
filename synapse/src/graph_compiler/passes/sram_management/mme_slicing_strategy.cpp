#include "mme_slicing_strategy.h"

#include "hal_reader/gaudi1/hal_reader.h"
#include "metrics_calculator.h"
#include "mme_shared_input.h"
#include "slicing_brain.h"
#include "slicing_utils.h"

#include <cmath>
#include <utils.h>

class MmeSlicingStrategyImpl : public MmeSlicingStrategy
{
public:
    MmeSlicingStrategyImpl(const HalReader& halReader, TensorVector& inputs, const pTensor& output)
    : MmeSlicingStrategy(halReader, StrategySlicingDataPtr(new MmeSlicingData(halReader, inputs, output)))
    {}

    virtual ~MmeSlicingStrategyImpl() = default;
    MmeSlicingStrategyImpl(const MmeSlicingStrategyImpl& other, bool resetAlignment);
    SlicingStrategyPtr clone(bool resetAlignment) override;
    virtual Metrics& calculateMetrics() override;
    virtual MmeSlicingData& getMmeSlicingData() override { return static_cast<MmeSlicingData&>(getSlicingData());}
    virtual const MmeSlicingData& getMmeSlicingData() const override { return static_cast<const MmeSlicingData&>(getSlicingData());}
    virtual void printLog(int logLevel, const synapse::LogManager::LogType& logName) const override;
    virtual unsigned getMMENarrowGeometryInElements() const override;
    virtual unsigned getMMEWideGeometryInElements() const override;
    virtual unsigned alignToMMEWide(unsigned size, bool floor) override;
    virtual unsigned alignToMMENarrow(unsigned size, bool floor) override;

private:
    MmeSlicingStrategy::Metrics& calculateMetrics(bool recalcSramCapOnly);
    virtual MmeSlicingStrategy::Metrics& recalculateSramCapacity() override;
    unsigned getMMEGeometryInElements(const pSlicedOperand& slicedOperand) const;
};

pMmeSlicingStrategy MmeSlicingStrategy::createStrategyForMMENode(const HalReader& halReader, const pNode& mmeNode)
{
    const auto& shapeTensor = mmeNode->getInput(TENSOR_SHAPE_DEDX);
    TensorVector inputs = {mmeNode->getInput(TENSOR_IFM), mmeNode->getInput(TENSOR_WEIGHT)};
    if(shapeTensor)
    {
        inputs.push_back(shapeTensor);
    }
    pMmeSlicingStrategy s(new MmeSlicingStrategyImpl(halReader, inputs, mmeNode->getOutput(TENSOR_OFM)));
    MmeSlicingData& data = s->getMmeSlicingData();
    const auto& shapeBundleTensor = shapeTensor ? data.bundleTensors[2] : nullptr;
    data.setOutputSliceBackwardMapping(
        MMESliceMapper::mapOutputToInputs(mmeNode, data.bundleTensors[0], data.bundleTensors[1],
                                          data.masterOperand, shapeBundleTensor));
    data.setDimControllerNode(mmeNode);
    return s;
}

MmeSlicingStrategy& MmeSlicingStrategy::setOutputTraversalPattern(DimVector traversalPattern)
{
    getMmeSlicingData().traversalPattern = std::move(traversalPattern);
    return *this;
}

MmeSlicingStrategy& MmeSlicingStrategy::setGeometry(MmeGeometry geometry)
{
    getMmeSlicingData().MMEGeometryUsed = geometry;
    return *this;
}

// the goal of this function is to create a "unique" string from a slicing data, so that during the "solving" stage,
// when trying to merge the new strategies created by the graph size optimization solver, we will not add new strategies
// that in fact represent the same slicing as an exiting one.
std::string MmeSlicingStrategy::getSlicingDataString(bool exactMatch) const
{
    std::stringstream ss;

    for (auto dim : getMmeSlicingData().traversalPattern)
    {
        ss<<dim<<"_";
    }
    if (exactMatch)
    {
        ss << "|" << getMmeSlicingData().MMEGeometryUsed << ":";
    }
    for (auto op : getMmeSlicingData().getSlicedOperands())
    {
        ss<<op->toString()<<";";
    }

    return ss.str();
}

bool MmeSlicingStrategy::isValidForCacheLineAlignment(const pSlicedOperand& operand)
{
    if (recalculateSramCapacity().SRAMCapacity > SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        return false;
    }

    for (const auto& candidate : getMmeSlicingData().getRoleCandidates())
    {
        if (candidate == nullptr) continue;
        if (candidate->stitchedOperand == operand && candidate->reshapeNode != nullptr)
        {
            if (candidate->reshapeNode->getNodeType() == Node::TYPE_PHYSICAL_RESHAPE ||
                dynamic_cast<ReshapeNode*>(candidate->reshapeNode.get())->isRehsapeOnFcd())
            {
                // if we align the operand (and add strides to it) we might get a situation where a alias tensor is
                // already strided. in that case a memcopy will be added and we rather not align.
                return false;
            }
        }
    }
    return true;
}

void MmeSlicingStrategy::tryAlignOperandToCacheLine(pSlicedOperand& operand, bool adjustSramCapacityForSlave)
{
    if (operand->resideInSRAM)
    {
        operand->alignWithCacheLine = true;
        if (adjustSramCapacityForSlave)
        {
            getMmeSlicingData().adjustSRAMCapacity(
                getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer]);
        }

        if (!isValidForCacheLineAlignment(operand))
        {
            operand->alignWithCacheLine = false;  // alignment reverted
            if (adjustSramCapacityForSlave)
            {
                getMmeSlicingData().adjustSRAMCapacity(
                    getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer]);
            }
            recalculateSramCapacity();
        }
    }
}

void MmeSlicingStrategy::tryAlignToCacheLine()
{
    if (GCFG_SRAM_SLICER_ALIGN_TO_CACHE_LINE.value())
    {
        // Align master operands
        for (auto& operand : getSlicingData().bundleTensors)
        {
            tryAlignOperandToCacheLine(operand, false);
        }

        // Align slave non-shared operand
        if (getMmeSlicingData().hasRole(BundleExpansion::SharedInputConsumer))
        {
            auto nonSharedOperand =
                getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer]->slaveOperands.getInput();
            tryAlignOperandToCacheLine(nonSharedOperand, true);
        }
    }
}

void MmeSlicingStrategy::MmeSlicingData::setDimControllerNode(const pNode& mmeNode)
{
    m_dimController = std::make_shared<MmeDimController>(mmeNode);
}

pSlicedOperand& MmeSlicingStrategy::MmeSlicingData::getWide()
{
    return const_cast<pSlicedOperand&>(
        const_cast<const MmeSlicingData*>(this)->getWide());
}

const pSlicedOperand & MmeSlicingStrategy::MmeSlicingData::getWide() const
{
    auto it = std::find(m_dimController->widthOutput().begin(),
                        m_dimController->widthOutput().end(),
                        traversalPattern.front());

    return (it != m_dimController->widthOutput().end()) ? bundleTensors[0] : bundleTensors [1];
}

const DimVector& MmeSlicingStrategy::MmeSlicingData::getWideNonCommonSlicingDims() const
{
    auto it = std::find(m_dimController->widthOutput().begin(),
                        m_dimController->widthOutput().end(),
                        traversalPattern.front());

    return (it != m_dimController->widthOutput().end()) ?
            m_dimController->nonCommonDimOperandA() : m_dimController->nonCommonDimOperandB();
}

const DimVector& MmeSlicingStrategy::MmeSlicingData::getWideOutputSlicingDims() const
{
   auto it = std::find(m_dimController->widthOutput().begin(),
                        m_dimController->widthOutput().end(),
                        traversalPattern.front());

    return (it != m_dimController->widthOutput().end()) ?
            m_dimController->heightOutput() : m_dimController->widthOutput();
}

const DimVector& MmeSlicingStrategy::MmeSlicingData::getWideCommonSlicingDims() const
{
    auto it = std::find(m_dimController->widthOutput().begin(),
                        m_dimController->widthOutput().end(),
                        traversalPattern.front());

    return (it != m_dimController->widthOutput().end()) ?
           m_dimController->commonDimOperandA() : m_dimController->commonDimOperandB();
}

pSlicedOperand& MmeSlicingStrategy::MmeSlicingData::getNarrow()
{
    return const_cast<pSlicedOperand&>(
        const_cast<const MmeSlicingData*>(this)->getNarrow());
}

const pSlicedOperand& MmeSlicingStrategy::MmeSlicingData::getNarrow() const
{
    auto it = std::find(m_dimController->heightOutput().begin(),
                        m_dimController->heightOutput().end(),
                        traversalPattern.front());

    return (it != m_dimController->heightOutput().end()) ? bundleTensors[0] : bundleTensors[1];
}

const DimVector& MmeSlicingStrategy::MmeSlicingData::getNarrowNonCommonSlicingDims() const
{
    auto it = std::find(m_dimController->heightOutput().begin(),
                        m_dimController->heightOutput().end(),
                        traversalPattern.front());

    return (it != m_dimController->heightOutput().end()) ?
            m_dimController->nonCommonDimOperandA() : m_dimController->nonCommonDimOperandB();
}

const DimVector& MmeSlicingStrategy::MmeSlicingData::getNarrowOutputSlicingDims() const
{
    auto it = std::find(m_dimController->heightOutput().begin(),
                        m_dimController->heightOutput().end(),
                        traversalPattern.front());

    return (it != m_dimController->heightOutput().end()) ?
            m_dimController->heightOutput() : m_dimController->widthOutput();
}

MmeSlicingStrategy::MmeSlicingData::MmeSlicingData(const HalReader&    halReader,
                                                   const TensorVector& inputTensors,
                                                   const pTensor&      origOutput)
: StrategySlicingData(inputTensors, origOutput), m_halReader(halReader)
{
}

MmeSlicingStrategy::MmeSlicingData::MmeSlicingData(const MmeSlicingStrategy::MmeSlicingData& other)
: StrategySlicingData(other),
  MMEGeometryUsed(other.MMEGeometryUsed),
  m_halReader(other.m_halReader),
  m_dimController(other.m_dimController)
{
    for (const auto& candidate : other.m_roleValidCandidates)
    {
        if (candidate)
        {
            addValidCandidate(candidate);
        }
    }
    for (const auto& candidate : other.m_invalidCandidates)
    {
        if (candidate)
        {
            addInvalidCandidate(candidate);
        }
    }
}

StrategySlicingDataPtr MmeSlicingStrategy::MmeSlicingData::clone() const
{
    return StrategySlicingDataPtr(new MmeSlicingData(*this));
}

// note: it doesn't compare all members but only those of them who are relevant before bundle expansion stage
bool MmeSlicingStrategy::MmeSlicingData::compareInitialSlicing(const StrategySlicingData& other, bool exactMatch) const
{
    if (exactMatch)
    {
        if (MMEGeometryUsed != static_cast<const MmeSlicingData*>(&other)->MMEGeometryUsed) return false;
    }

    return  StrategySlicingData::compareInitialSlicing(other);
}

SlicedOperandTraversalPattern MmeSlicingStrategy::MmeSlicingData::getOutputSlices() const
{
    unsigned commonDimSlices = 1;
    if (m_dimController != nullptr)
    {
        commonDimSlices = SlicedOperandUtils::nofSlices(getWide(), getWideCommonSlicingDims());
    }
    SlicedOperandTraversalPattern masterTraversalPattern(masterOperand, traversalPattern,
                                                         isSnakeWalkingPatternEnabled(),
                                                         commonDimSlices);
    for (auto& slaveTraversalPattern : m_slaveTraversalPatterns)
    {
        masterTraversalPattern.addSlave(slaveTraversalPattern);
        if (m_sharedMMEInputStitchedToNarrow && slaveTraversalPattern.numCommonDimSlices() > 1)
        {
            // master's narrow sliced operand is shared with a slave that has partial slicing.
            // This means that the slave inputSliceChange signal will not be aligned to master and
            // eliminate the benefit of sharing the input slices.
            // Switch the master to signal for every narrow slice in order to align them.
            masterTraversalPattern.setInputSliceChangeDim(traversalPattern.front());
        }
    }
    return masterTraversalPattern;
}

StrategySlicingData::WalkingDir MmeSlicingStrategy::MmeSlicingData::getWalkingDir() const
{
    if (!m_dimController) {
        // TPC bundle strategies will not have MME dim controller
        return WalkingDir::LeftToRight;
    }
    auto it = std::find(m_dimController->widthOutput().begin(),
                        m_dimController->widthOutput().end(),
                        traversalPattern.front());
    return (it != m_dimController->widthOutput().end()) ? WalkingDir::LeftToRight : WalkingDir::TopToBottom;
}

uint64_t MmeSlicingStrategy::MmeSlicingData::getCommonDimSize() const
{
    uint64_t cdSize = 1;
    const DimVector& commonDims = getWideCommonSlicingDims();
    for (uint32_t cd : commonDims)
    {
        cdSize *= getWide()->finalShape[cd];
    }
    return cdSize;
}

uint64_t MmeSlicingStrategy::MmeSlicingData::getQRSSize() const
{
    return multiplyElements(m_dimController->qrsSizes().begin(), m_dimController->qrsSizes().end());
}

NodeSet MmeSlicingStrategy::MmeSlicingData::getStrategyNodes(const pBundle& bundle) const
{
    NodeSet bundleNodes {bundle->getNodes().begin(), bundle->getNodes().end()};
    for (const auto& candidate : getRoleCandidates())
    {
        if (candidate && candidate->nodeToStitch)
        {
            bundleNodes.insert(candidate->nodeToStitch);
            if (candidate->reshapeNode)
            {
                bundleNodes.insert(candidate->reshapeNode);
            }
        }
    }
    return bundleNodes;
}

NodeSet MmeSlicingStrategy::MmeSlicingData::getStrategyProducers() const
{
    NodeSet producers;
    for (const auto& candidate : getRoleCandidates())
    {
        if (candidate && candidate->nodeToStitch && BundleExpansion::isProducer(candidate->role))
        {
            producers.insert(candidate->nodeToStitch);
        }
    }
    return producers;
}

unsigned MmeSlicingStrategyImpl::getMMENarrowGeometryInElements() const
{
    return getMMEGeometryInElements(getMmeSlicingData().getNarrow());
}

unsigned MmeSlicingStrategyImpl::getMMEWideGeometryInElements() const
{
    return getMMEGeometryInElements(getMmeSlicingData().getWide());
}

unsigned MmeSlicingStrategyImpl::getMMEGeometryInElements(const pSlicedOperand& slicedOperand) const
{
    HB_ASSERT(slicedOperand != getMmeSlicingData().masterOperand, "sliced operand isn't master operand");
    unsigned MMEGeometryInElements = 0;
    switch (getMmeSlicingData().MMEGeometryUsed)
    {
    case gaudi_geometry_1wx4h:
        MMEGeometryInElements = (slicedOperand == getMmeSlicingData().bundleTensors[0]) ? 4 : 1;
        break;
    case gaudi_geometry_4wx1h:
        MMEGeometryInElements = (slicedOperand == getMmeSlicingData().bundleTensors[0]) ? 1 : 4;
        break;
    case gaudi_geometry_2wx2h:
        MMEGeometryInElements = 2;
        break;
    }
    return MMEGeometryInElements * m_halReader.getMmeVectorSize() /
           m_slicingData->masterOperand->originalTensor->getElementSizeInBytes();
}

MmeSlicingStrategy::Metrics& MmeSlicingStrategyImpl::calculateMetrics()
{
    return calculateMetrics(false);
}

MmeSlicingStrategy::Metrics& MmeSlicingStrategyImpl::recalculateSramCapacity()
{
    return calculateMetrics(true);
}

MmeSlicingStrategy::Metrics& MmeSlicingStrategyImpl::calculateMetrics(bool recalcSramCapOnly)
{
    // numOfOperandBuffers has an effect on the rest of the metrics so refresh it in case the operands changed
    // since last time the double buffering was set.
    // in some cases we can't update the num of buffers since the operands sizes might exceed the max sram capacity
    // it is the solver responsibilty to allow or forbid changing the number of operand buffers
    if (allowUpdateNumOfBuffers())
    {
        getMmeSlicingData().updateNumOfOperandBuffers(m_metrics.isDoubleBuffered);
    }

    MmeBundleMetricsCalculator calculator(m_halReader, this, &m_metrics);

    if  (recalcSramCapOnly)
    {
        return calculator.recalculateSramCapacityOnly();
    }
    else
    {
        return calculator.calculate();
    }
}

void MmeSlicingStrategyImpl::printLog(int logLevel, const synapse::LogManager::LogType& logName) const
{
    if (!log_level_at_least(logName, logLevel)) return;

    const MmeSlicingData& data = getMmeSlicingData();
    const std::string& walkingDir = (data.getWalkingDir()==StrategySlicingData::WalkingDir::LeftToRight) ? "Left-to-Right" : "Top-to-Bottom";
    std::string geometry = geometry2String(data.MMEGeometryUsed);

    SYN_LOG(logName, logLevel, "Slicing Strategy - {} , {}, graph size optimized: {}", walkingDir, geometry, m_graphSizeOptimized ? "true" : "false");
    for (int i=0; i < getMmeSlicingData().bundleTensors.size(); i++)
    {
        const pSlicedOperand& input = getMmeSlicingData().bundleTensors[i];
        const pTensor& origTensor = input->originalTensor;
        SYN_LOG(logName, logLevel, "Original Input [{}] {} : {}, Sliced : {}, Num of slices: {}, Buffers: {}, inSram: {}, alignedToCL:{}",
                i, origTensor->getName(),
                toString(input->finalShape.begin(), input->finalShape.begin()+input->originalTensor->getDim(), 'x'),
                  toString(input->chunkDimensions.data(), input->chunkDimensions.data() + origTensor->getDim(), 'x'),
                  SlicedOperandUtils::nofSlices(input), input->numOfBuffers, input->resideInSRAM, input->alignWithCacheLine);
    }
    const pSlicedOperand& output = getMmeSlicingData().masterOperand;
    SYN_LOG(logName, logLevel, "Original Output {} : {}, Sliced : {}, Num of slices: {}, Buffers: {}, inSram: {}, alignedToCL:{}",
            output->originalTensor->getName(),
            toString(output->finalShape.begin(), output->finalShape.begin()+output->originalTensor->getDim(), 'x'),
              toString(output->chunkDimensions.data(), output->chunkDimensions.data() + output->originalTensor->getDim(), 'x'),
              SlicedOperandUtils::nofSlices(output), output->numOfBuffers, output->resideInSRAM, output->alignWithCacheLine);

    SYN_LOG(logName,
            logLevel,
            "Metrics : SRAM Capacity : {}[MB], MME Utilization : {}, HBM Bandwidth: {}[GB/s], SB Reuse : {}, "
            "Double-Buffer : {}",
            bToMb(getMetrics().SRAMCapacity),
            getMetrics().MMEUtilization,
            getMetrics().HBMBandwidth,
            getMetrics().SBReuse,
            getMetrics().isDoubleBuffered);

    for (const auto& expansionCandidate : getMmeSlicingData().getRoleCandidates())
    {
        if (expansionCandidate!= nullptr)
        {
            SYN_LOG(logName, logLevel, "candidate Valid for role {} Node {}", expansionCandidate->role, expansionCandidate->nodeToStitch->getNodeName());
        }
    }
    for (const auto& expansionCandidate : getMmeSlicingData().getInvalidCandidates())
    {
        if (expansionCandidate!= nullptr)
        {
            SYN_LOG(logName, logLevel, "candidate Invalid for role {} Node {}", expansionCandidate->role, expansionCandidate->nodeToStitch->getNodeName());
        }
    }


}

unsigned MmeSlicingStrategyImpl::alignToMMEWide(unsigned size, bool floor)
{
    unsigned mmeWideGeometry = getMMEWideGeometryInElements();

    if (floor)
    {
        return std::floor((float)size / mmeWideGeometry) * mmeWideGeometry;
    }
    else
    {
        return std::ceil((float)size / mmeWideGeometry) * mmeWideGeometry;
    }
}

unsigned MmeSlicingStrategyImpl::alignToMMENarrow(unsigned size, bool floor)
{
    unsigned mmeNarrowGeometry = getMMENarrowGeometryInElements();

    if (floor)
    {
        return (unsigned)std::floor((float)size / mmeNarrowGeometry) * mmeNarrowGeometry;
    }
    else
    {
        return (unsigned)std::ceil((float)size / mmeNarrowGeometry) * mmeNarrowGeometry;
    }
}

MmeSlicingStrategyImpl::MmeSlicingStrategyImpl(const MmeSlicingStrategyImpl& other, bool resetAlignment)
: MmeSlicingStrategy(other, resetAlignment)
{
}

SlicingStrategyPtr MmeSlicingStrategyImpl::clone(bool resetAlignment)
{
    return SlicingStrategyPtr(new MmeSlicingStrategyImpl(*this, resetAlignment));
}

void MmeSlicingStrategy::MmeSlicingData::addValidCandidate(const pBundleExpansion& candidate, bool needToAdjust)
{
    pBundleExpansion adjustedCandidate     = needToAdjust ? getAdjustedCandidate(candidate) : candidate;
    m_roleValidCandidates[candidate->role] = adjustedCandidate;
    m_candidateInBundle[adjustedCandidate] = false;
}

void MmeSlicingStrategy::MmeSlicingData::addInvalidCandidate(const pBundleExpansion& candidate, bool needToAdjust)
{
    pBundleExpansion adjustedCandidate = needToAdjust ? getAdjustedCandidate(candidate) : candidate;
    m_invalidCandidates.push_back(adjustedCandidate);
}

MmeSlicingStrategy::MmeSlicingData::RoleCandidatesArray& MmeSlicingStrategy::MmeSlicingData::getRoleCandidates()
{
    return m_roleValidCandidates;
}

const MmeSlicingStrategy::MmeSlicingData::RoleCandidatesArray& MmeSlicingStrategy::MmeSlicingData::getRoleCandidates() const
{
    return m_roleValidCandidates;
}

const std::list<pBundleExpansion>& MmeSlicingStrategy::MmeSlicingData::getInvalidCandidates() const
{
    return m_invalidCandidates;
}

pBundleExpansion MmeSlicingStrategy::MmeSlicingData::getAdjustedCandidate(const pBundleExpansion& origCandidate)
{
    pBundleExpansion adjustedCandidate = std::make_shared<BundleExpansion>(*origCandidate);
    pSlicedOperand   operand;

    for (pSlicedOperand& currOperand : getSlicedOperands())
    {
        if (currOperand->originalTensor == origCandidate->stitchedOperand->originalTensor)
        {
            operand = currOperand;
        }
    }

    if (origCandidate->role == BundleExpansion::Role::SlaveInputProducer ||
        origCandidate->role == BundleExpansion::Role::SlaveOutputConsumer)
    {
        /* Slaves operand are not saved in bundleTensors yet.
         * We should look for the slave candidate. */
        operand = findStitchedOperandForCandidatesDependsOnSlave(origCandidate);
        HB_ASSERT(operand, "SharedInputConsumer is not in strategy");
    }

    if (operand)
    {
        adjustedCandidate->stitchedOperand = operand;
        adjustSRAMCapacity(adjustedCandidate);
        return adjustedCandidate;
    }

    HB_ASSERT(false, "Candidate stitched operand is not in the strategy sliced operands");
    return nullptr;
}

static bool isTiled(const pSlicedOperand& slicedOp)
{
    // If the operand is sliced without overlaps or padding, we can say that it is tiled (i.e the slices act as tiles).
    return !(slicedOp->hasOverlap() || slicedOp->hasOffset());
}

// Returns true if the operand includes slices that are bigger than the calculated SRAM capacity
static bool isExtraSized(const pSlicedOperand& slicedOp)
{
    // If the operand has bigger spatial edge slices
    return slicedOp->hasExtraSizedSlices();
}

bool MmeSlicingStrategy::MmeSlicingData::blockExpansionForRole(BundleExpansion::Role role)
{
    // If the strategy includes an operand with overlap or uneven slicing -
    // block expansion with any node that is related to that operand until the feature is implemented.

    // TODO - test if we can stitch operands with offset after (hasOffset() checks for offsetBefore or offsetAfter).
    bool wideInputIsNotTiled      = !isTiled(getWide());
    bool narrowInputIsNotTiled    = !isTiled(getNarrow());
    bool masterOutputIsNotTile    = !isTiled(masterOperand);

    // Operand can be extrasized if it is an output operand of a dedx operation.
    // It's because the dedx output buffer might include unitialized pixels, for pixels that do not affect the computation in fwd.
    // These leftover pixels are the extra size of the tensor, and this is why we check it only on output.
    // See [SW-112633] for a reference
    bool masterOutputIsExtraSized = isExtraSized(masterOperand);

    if(role == BundleExpansion::OutputConsumer)
    {
        return masterOutputIsNotTile || masterOutputIsExtraSized;
    }
    else if(role == BundleExpansion::WideInputProducer)
    {
        return wideInputIsNotTiled;
    }
    else if(role == BundleExpansion::NarrowInputProducer)
    {
        return narrowInputIsNotTiled;
    }
    else
    {
        // Currently slave MME is not allowed if one of the inputs has overlap.
        // TODO SW-25560:
        // Allow SharedInputConsumer in the calling function, if the candidate that was found is stiched to an operand without overlap
        return (wideInputIsNotTiled || narrowInputIsNotTiled);
    }
}

pSlicedOperand MmeSlicingStrategy::MmeSlicingData::findStitchedOperandForCandidatesDependsOnSlave(const pBundleExpansion& origCandidate)
{
    bool supportedCandidate = origCandidate->role == BundleExpansion::Role::SlaveInputProducer ||
                              origCandidate->role == BundleExpansion::Role::SlaveOutputConsumer;
    bool sharedInputConsumerFound = false;

    if (!supportedCandidate)
    {
        HB_ASSERT(supportedCandidate, "Function called with unsupported candidate");
    }

    for (const pBundleExpansion& candidate : getRoleCandidates())
    {
        if (candidate && candidate->role == BundleExpansion::SharedInputConsumer && !candidate->slaveOperands.empty())
        {
            sharedInputConsumerFound = true;

            if (candidate->stitchedOperand->originalTensor ==  origCandidate->stitchedOperand->originalTensor)
            {
                return candidate->stitchedOperand;
            }

            for (auto& operandTupple: candidate->slaveOperands)
            {
                auto operand = operandTupple.second;
                if (operand->originalTensor == origCandidate->stitchedOperand->originalTensor)
                {
                    return operand;
                }
            }
        }
    }

    for (const pBundleExpansion& candidate : getInvalidCandidates())
    {
        if (candidate && candidate->role == BundleExpansion::SharedInputConsumer && !candidate->slaveOperands.empty())
        {
            sharedInputConsumerFound = true;

            if (candidate->stitchedOperand->originalTensor ==  origCandidate->stitchedOperand->originalTensor)
            {
                return candidate->stitchedOperand;
            }

            for (auto& operandTupple: candidate->slaveOperands)
            {
                auto operand = operandTupple.second;
                if (operand->originalTensor == origCandidate->stitchedOperand->originalTensor)
                {
                    return operand;
                }
            }
        }
    }

    HB_ASSERT(sharedInputConsumerFound,
              "Slave*InputProducer and SlaveOutputConsumer depends on SharedInputConsumer but it is not found!");

    return nullptr;
}

void MmeSlicingStrategy::MmeSlicingData::adjustSRAMCapacity(pBundleExpansion& candidate)
{
    switch (candidate->role)
    {
    case BundleExpansion::WideInputProducer:
    case BundleExpansion::NarrowInputProducer:
    case BundleExpansion::SlaveInputProducer:
        break; // TPC producer does not require additional SRAM
    case BundleExpansion::SlaveOutputConsumer:
        {
            pSlicedOperand slaveOperand = findSlaveOutput();
            HB_ASSERT(slaveOperand, "adjustSRAMCapacity: Slave operand not found!");
            if (slaveOperand->resideInSRAM == false)
            {
                /* TPC consumer requires that the MME output will be in SRAM */
                candidate->additionalSRAMCapacity =  computeSRAMCapacity(slaveOperand);
            }
        }
        break;
    case BundleExpansion::OutputConsumer:
        if (masterOperand->resideInSRAM == false)
        {
            /* TPC consumer requires that the MME output will be in SRAM */
            candidate->additionalSRAMCapacity = computeSRAMCapacity(masterOperand);
        }
        break;
    case BundleExpansion::SharedInputConsumer:
    {
        candidate->additionalSRAMCapacity =
            SharedMMEInputCandidateHandler::getCandidateAdditionalCapacity(candidate);
        break;
    }
    default:
        HB_ASSERT(false, "Unexpected role for candidate");
    }
}

pSlicedOperand MmeSlicingStrategy::MmeSlicingData::findSlaveOutput()
{
    pSlicedOperand slaveOperand;

    for (const pBundleExpansion& candidate : getRoleCandidates())
    {
        if (candidate && candidate->role == BundleExpansion::SharedInputConsumer && !candidate->slaveOperands.empty())
        {
            pTensor slaveOutput = candidate->nodeToStitch->getOutputs().front();

            for (auto& operandTupple: candidate->slaveOperands)
            {
                auto operand = operandTupple.second;
                if (operand->originalTensor == slaveOutput)
                {
                    return operand;
                }
            }
        }
    }

    for (const pBundleExpansion& candidate : getInvalidCandidates())
    {
        if (candidate && candidate->role == BundleExpansion::SharedInputConsumer && !candidate->slaveOperands.empty())
        {
            pTensor slaveOutput = candidate->nodeToStitch->getOutputs().front();

            for (auto& operandTupple: candidate->slaveOperands)
            {
                auto operand = operandTupple.second;
                if (operand->originalTensor == slaveOutput)
                {
                    return operand;
                }
            }
        }
    }

    return nullptr;
}

uint32_t MmeSlicingStrategy::MmeSlicingData::computeSRAMCapacity(const pSlicedOperand& operand)
{
    uint32_t numOfBuffers = SlicedOperandUtils::isTriviallySliced(operand) ? 1 : 2;
    return SlicedOperandUtils::getSliceSizeInBytes(operand) * numOfBuffers;
}

void MmeSlicingStrategy::MmeSlicingData::addSlaveTraversalPattern(const pBundleExpansion& candidate)
{
    // in case of batch gemm shared input the slave traversal pattern is exacly like the master traversal pattern
    if (candidate->bundleNode->isBatchGemm() && candidate->nodeToStitch->isBatchGemm())
    {
        m_slaveTraversalPatterns.push_back(SlicedOperandTraversalPattern(candidate->slaveOperands.getOutput(),
                                                                         traversalPattern,
                                                                         isSnakeWalkingPatternEnabled(),
                                                                         getOutputSlices().numCommonDimSlices()));
    }
    else
    {
        m_slaveTraversalPatterns.push_back(getSlaveTraversalPattern(candidate->nodeToStitch,
                                                                    candidate->stitchedOperand->originalTensor,
                                                                    candidate->slaveOperands.getOutput()));
    }

    m_sharedMMEInputStitchedToNarrow = candidate->stitchedOperand->originalTensor == getNarrow()->originalTensor;
}

// Given a shared input MME slave candidate, we want to create a traversal pattern on the output operand.
// we need to understand the traversal pattern of the master, and apply it to the slave -
// 1) check if shared input is wide\narrow
// 2) identify the shared\non-shared operands by index of the slave node
// 3) get the output dimensions in order - narrow, wide.
SlicedOperandTraversalPattern MmeSlicingStrategy::MmeSlicingData::getSlaveTraversalPattern(const pNode& slaveNode,
                                                                                     const pTensor& sharedInput,
                                                                                     const pSlicedOperand& slaveOutputOperand)
{
    DimVector       slaveTraversalPattern;
    unsigned wideInputIdx, narrowInputIdx;
    MmeDimController dimController(slaveNode);
    // find shared\non-shared  operand index on slave node.
    unsigned sharedOperandInputIndex = slaveNode->getInputIndexOfTensor(sharedInput);
    unsigned nonSharedOperandInputIndex = sharedOperandInputIndex ^ 1;
    // find if shared operand is narrow\wide in the master strategy
    bool isSharedOperandWide = (getWide()->originalTensor == sharedInput);
    unsigned numSlices = SlicedOperandUtils::nofSlices(getWide(), getWideCommonSlicingDims());

    wideInputIdx = (isSharedOperandWide || (numSlices > 1)) ? sharedOperandInputIndex : nonSharedOperandInputIndex;
    narrowInputIdx = (isSharedOperandWide || (numSlices > 1)) ? nonSharedOperandInputIndex : sharedOperandInputIndex;

    // transform input indexes to proper traversal pattern (dim list)
    // first narrow and then wide.
    for (auto& index : {narrowInputIdx, wideInputIdx})
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

    if (slaveTraversalPattern.empty()) slaveTraversalPattern.push_back(0);

    // check common dim slices -
    auto& dims = (sharedOperandInputIndex == 0) ? dimController.commonDimOperandA() :
                 dimController.commonDimOperandB();
    unsigned totalNumSlices = SlicedOperandUtils::nofSlices(isSharedOperandWide ? getWide() : getNarrow(), dims);

    return SlicedOperandTraversalPattern(slaveOutputOperand, slaveTraversalPattern,
                                         isSnakeWalkingPatternEnabled(), totalNumSlices);
}

void MmeSlicingStrategy::MmeSlicingData::setCandidateAsBundled(pBundleExpansion& candidate)
{
    auto search = m_candidateInBundle.find(candidate);

    if (search != m_candidateInBundle.end())
    {
        search->second = true;
    }
    else
    {
        HB_ASSERT(false, "candidateAddedToBundle: candidate not found!");
    }
}

bool MmeSlicingStrategy::MmeSlicingData::isCandidateInBundle(const pBundleExpansion& candidate)
{
    auto search = m_candidateInBundle.find(candidate);

    if (search != m_candidateInBundle.end())
    {
        return search->second;
    }
    HB_ASSERT(false, "isCandidateInBundle: candidate not found!");
    return false;
}

bool MmeSlicingStrategy::MmeSlicingData::hasRole(BundleExpansion::Role role) const
{
    const auto& expansions = getRoleCandidates();
    return (expansions[role] != nullptr) && expansions[role]->nodeToStitch;
}

pSlicedOperand MmeSlicingStrategy::MmeSlicingData::getSlaveOutputOperand() const
{
    if (hasRole(BundleExpansion::SharedInputConsumer))
    {
        return getRoleCandidates()[BundleExpansion::SharedInputConsumer]->slaveOperands.getOutput();
    }
    return nullptr;
}

bool MmeSlicingStrategy::MmeSlicingData::isNodeStitched(const pNode& node) const
{
    for (const auto& candidate : getRoleCandidates())
    {
        if (candidate && (candidate->nodeToStitch == node))
        {
            return true;
        }
    }
    return false;
}