#include "slicing_brain.h"
#include "slicing_utils.h"

#include "strategy_slicing_data.h"
#include <algorithm>

StrategySlicingData::StrategySlicingData(const TensorVector& inputTensors,
                                         const pTensor& outputTensor)
: masterOperand(std::make_shared<SlicedOperand>(outputTensor))
{
    for (auto& input : inputTensors)
    {
        if (!input)
        {
            continue;
        }
        bundleTensors.push_back(std::make_shared<SlicedOperand>(input));
    }
}

template<class CONT>
void createContainerWitReplacment(const CONT& originalCont, CONT& outCont, const std::map<typename CONT::value_type, typename CONT::value_type>& replacements)
{
    for (const auto& val : originalCont)
    {
        auto replaceIter = replacements.find(val);
        HB_ASSERT(replaceIter != replacements.end(), "Expect replacements for all container values");
        outCont.push_back(replaceIter->second);
    }
}

StrategySlicingData::StrategySlicingData(const StrategySlicingData& other)
: enableGraphSizeOptimization(other.enableGraphSizeOptimization),
  traversalPattern(other.traversalPattern),
  m_snakeWalkingPatternEnabled(other.isSnakeWalkingPatternEnabled())
{
    std::map<pSlicedOperand, pSlicedOperand> oldToNewOperand;
    for (const auto& tensorSliceData : other.bundleTensors)
    {
        bundleTensors.push_back(std::make_shared<SlicedOperand>(*tensorSliceData));
        oldToNewOperand[tensorSliceData] = bundleTensors.back();
    }
    masterOperand = std::make_shared<SlicedOperand>(*other.masterOperand);
    oldToNewOperand[other.masterOperand] = masterOperand;

    // Copy bwd mapping
    for (const auto& mappingAndOperand : other.m_operandBackwardMappings)
    {
        pBackwardSliceMapping mapping = mappingAndOperand.second;
        std::vector<pSlicedOperand> inOperands;
        createContainerWitReplacment(mapping->getInOperands(), inOperands, oldToNewOperand);
        pSlicedOperand outOperand = oldToNewOperand[mapping->getOutOperand()];
        setOperandSliceBackwardMapping(outOperand, mapping->clone(inOperands, outOperand));
    }

    // Copy fwd mapping
    for (const auto& mappingsAndOperand : other.m_operandForwardMappings)
    {
        for (const auto& singleMapping : mappingsAndOperand.second)
        {
            pForwardSliceMapping      mapping = singleMapping;
            std::list<pSlicedOperand> inOperands;
            std::list<pSlicedOperand> outOperands;
            createContainerWitReplacment(mapping->getInputs(), inOperands, oldToNewOperand);
            createContainerWitReplacment(mapping->getOutputs(), outOperands, oldToNewOperand);
            pSlicedOperand mappedOperand = oldToNewOperand[mappingsAndOperand.first];
            addOperandSliceForwardMapping(mappedOperand, mapping->clone(inOperands, outOperands));
        }
    }
}

StrategySlicingDataPtr StrategySlicingData::clone() const
{
    return StrategySlicingDataPtr(new StrategySlicingData(*this));
}

bool StrategySlicingData::compareInitialSlicing(const StrategySlicingData& other, bool exactMatch) const
{
    if (traversalPattern != other.traversalPattern) return false;
    if (*masterOperand != *other.masterOperand) return false;
    if (bundleTensors.size() != bundleTensors.size()) return false;
    for (auto iter1 = bundleTensors.begin(), iter2 = other.bundleTensors.begin();
         iter1 != bundleTensors.end() && iter2 != other.bundleTensors.begin();
         ++iter1, ++iter2)
    {
        if (**iter1 != **iter2) return false;
    }
    return true;
}

StrategySlicingData::~StrategySlicingData()
{
}

std::vector<pSlicedOperand> StrategySlicingData::getSlicedOperands() const
{
    std::vector<pSlicedOperand> allOperands(bundleTensors.begin(), bundleTensors.end());
    allOperands.push_back(masterOperand);
    return allOperands;
}

pSlicedOperand StrategySlicingData::getSlicedOperand(const TensorPtr& tensor) const
{
    pSlicedOperand ret(nullptr);
    for (const auto& op : getSlicedOperands())
    {
        if (op->originalTensor == tensor)
        {
            if (ret != nullptr)
            {
                LOG_WARN(SRAM_SLICE, "Found more than 1 sliced operand for tensor {}", tensor->getName());
            }
            ret = op;
        }
    }
    return ret;
}

StrategySlicingData::slicedOperandAndDimList StrategySlicingData::getSlicedOperandsAndDims() const
{
    slicedOperandAndDimList lst;
    for (const pSlicedOperand& slicedOp : getSlicedOperands())
    {
        for (unsigned dim = 0; dim < slicedOp->originalTensor->getDim(); ++dim)
        {
            if (SlicedOperandUtils::isSlicedOnDimension(slicedOp, dim))
            {
                lst.push_back({slicedOp, dim});
            }
        }
    }
    return lst;
}

SliceReferenceList StrategySlicingData::getInputsForSlice(const SliceRefCommonDimIdxPair& slice) const
{
    auto iter = m_operandBackwardMappings.find(slice.first->operand);
    if (iter != m_operandBackwardMappings.end())
    {
        pBackwardSliceMapping mapper = iter->second;
        if (mapper)
        {
            return mapper->getInputs(slice);
        }
    }

    return {};
}

// default call to getInputsForSlice with common dim coordinate of 0
SliceReferenceList StrategySlicingData::getInputsForSlice(const pSliceReference& slice) const
{
    SliceRefCommonDimIdxPair slicePair = {slice, 0};
    return getInputsForSlice(slicePair);
}

SliceReferenceList StrategySlicingData::getOutputsForSlice(const pSliceReference& outputSlice) const
{
    auto iter = m_operandBackwardMappings.find(outputSlice->operand);
    if (iter != m_operandBackwardMappings.end())
    {
        pBackwardSliceMapping mapper = iter->second;
        if (mapper)
        {
            return mapper->getOutputs(outputSlice);
        }
    }

    return {};
}

SlicedOperandTraversalPattern StrategySlicingData::getOutputSlices() const
{
    SlicedOperandTraversalPattern masterTraversalPattern(masterOperand,
                                                         traversalPattern,
                                                         isSnakeWalkingPatternEnabled(),
                                                         numCommonDimSlices);

    for (auto& slaveTraversalPattern : m_slaveTraversalPatterns)
    {
        masterTraversalPattern.addSlave(slaveTraversalPattern);
    }

    return masterTraversalPattern;
}

std::list<std::pair<SliceReferenceList, SliceReferenceList>> StrategySlicingData::getFwdMappedSlices(
    const pSliceReference& slice) const
{
    auto mapping = m_operandForwardMappings.find(slice->operand);
    if (mapping != m_operandForwardMappings.end())
    {
        std::list<std::pair<SliceReferenceList, SliceReferenceList>> ret;
        for (auto& fwdMapping : mapping->second)
        {
            std::list<std::pair<SliceReferenceList, SliceReferenceList>> singleMapping =
                fwdMapping->getInputsAndOutputs(slice);
            ret.insert(ret.end(), singleMapping.begin(), singleMapping.end());
        }
        return ret;
    }
    return {};
}

std::pair<std::list<pSlicedOperand>, std::list<pSlicedOperand>>
StrategySlicingData::getFwdMappedSlicedOperands(const pSlicedOperand& slicedOperand) const
{
    auto mapping = m_operandForwardMappings.find(slicedOperand);
    if (mapping != m_operandForwardMappings.end())
    {
        // Currently used only for BN - assert there's only 1 mapping.
        // If this assert happens - need to handle multiple mappings
        HB_ASSERT(mapping->second.size() == 1, "Called only by BN, which is expected to have a single FWD mapping");
        return {mapping->second[0]->getInputs(), mapping->second[0]->getOutputs()};
    }
    return {};
}
std::pair<std::list<pSlicedOperand>, std::list<pSlicedOperand>>
StrategySlicingData::getBwdMappedSlicedOperands(const pSlicedOperand& slicedOperand) const
{
    auto mapping = m_operandBackwardMappings.find(slicedOperand);
    if (mapping != m_operandBackwardMappings.end())
    {
        auto                      inputs = std::move(mapping->second->getInOperands());
        auto                      output = std::move(mapping->second->getOutOperand());
        std::list<pSlicedOperand> inputsList;
        std::list<pSlicedOperand> outputsList = {output};
        inputsList.assign(inputs.begin(), inputs.end());
        return {inputsList, outputsList};
    }
    return {};
}

StrategySlicingData::WalkingDir StrategySlicingData::getWalkingDir() const
{
    return StrategySlicingData::WalkingDir::LeftToRight;
}

NodeSet StrategySlicingData::getStrategyNodes(const pBundle& bundle) const
{
    NodeSet bundleNodes {bundle->getNodes().begin(), bundle->getNodes().end()};
    return bundleNodes;
}

std::vector<pSlicedOperand> StrategySlicingData::addNodeOperandsToStrategy(const TensorVector&        nodeOperands,
                                                                           const StrategySlicingData& nodeSlicingData,
                                                                           const pSlicedOperand&      stitchedOperand)
{
    std::vector<pSlicedOperand> slicedOperands {};
    for (const auto& nodeOperand : nodeOperands)
    {
        const auto& slicedOperand = nodeSlicingData.getSlicedOperand(nodeOperand);
        if (!slicedOperand) continue;  // Aux etc.

        if (!stitchedOperand || (slicedOperand->originalTensor != stitchedOperand->originalTensor))
        {
            bundleTensors.push_back(slicedOperand);
            slicedOperands.push_back(slicedOperand);
        }
        else
        {
            const auto& strategyOperands = getSlicedOperands();
            HB_ASSERT(std::find(strategyOperands.begin(), strategyOperands.end(), stitchedOperand) !=
                          strategyOperands.end(),
                      "The stitched operand doesn't exist in the strategy");
            slicedOperands.push_back(stitchedOperand);
            // The stitched operand is already in the bundle slicing data.
        }
    }
    return slicedOperands;
}

void StrategySlicingData::setOutputSliceBackwardMapping(pBackwardSliceMapping mapping)
{
    m_operandBackwardMappings[masterOperand] = mapping;
}

void StrategySlicingData::setOperandSliceBackwardMapping(pSlicedOperand operand, pBackwardSliceMapping mapping)
{
    SLC_TRACE("Mapping tensor {} slices to inputs.", operand->originalTensor->getName());
    m_operandBackwardMappings[operand] = mapping;
}

void StrategySlicingData::setOperandSliceForwardMapping(pSlicedOperand operand, pForwardSliceMapping mapping)
{
    SLC_TRACE("Mapping tensor {} slices to a dependant operation.", operand->originalTensor->getName());
    m_operandForwardMappings[operand] = {mapping};
}

void StrategySlicingData::addOperandSliceForwardMapping(pSlicedOperand operand, pForwardSliceMapping mapping)
{
    SLC_TRACE("Mapping tensor {} slices to a dependant operation.", operand->originalTensor->getName());
    m_operandForwardMappings[operand].push_back(mapping);
}

void StrategySlicingData::updateNumOfOperandBuffers(bool doubleBuffer)
{
    for (pSlicedOperand &op : getSlicedOperands())
    {
        if (op->resideInSRAM && !SlicedOperandUtils::isTriviallySliced(op))
        {
            op->numOfBuffers = doubleBuffer ? 2 : 1;
        }
        else
        {
            // Non SRAM and trivially slices operand have a single buffer.
            op->numOfBuffers = 1;
        }
    }
}

std::list<SlicedOperandTraversalPattern>& StrategySlicingData::getSlavesPatterns()
{
    return m_slaveTraversalPatterns;
}

void StrategySlicingData::addSlaveTraversalPattern(const pSlicedOperand& operand)
{
    m_slaveTraversalPatterns.push_back(SlicedOperandTraversalPattern(operand,
                                                                     {DIM_C},
                                                                     isSnakeWalkingPatternEnabled(),
                                                                     1));
}

bool StrategySlicingData::isSnakeWalkingPatternEnabled() const
{
    return SlicingBrain::knobs.snakeWalkingTraversal && m_snakeWalkingPatternEnabled;
}

void StrategySlicingData::setSnakeWalkingPattern(bool enable)
{
    m_snakeWalkingPatternEnabled = enable;
}
