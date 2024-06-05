#include "bundle.h"
#include <habana_global_conf.h>
#include "node.h"
#include "utils.h"

pSlicedOperand SlaveOperands::getInput()
{
    if(m_operands.find(NonSharedInput) != m_operands.end())
    {
        return m_operands[NonSharedInput];
    }
    return nullptr;
}

void SlaveOperands::setInput(pSlicedOperand operand)
{
    m_operands[NonSharedInput] = operand;
}

pSlicedOperand SlaveOperands::getOutput()
{
    if(m_operands.find(Output) != m_operands.end())
    {
        return m_operands[Output];
    }
    return nullptr;
}

void SlaveOperands::setOutput(pSlicedOperand operand)
{
    m_operands[Output] = operand;
}

pSlicedOperand SlaveOperands::getShapeOperand()
{
    if(m_operands.find(Shape) != m_operands.end())
    {
        return m_operands[Shape];
    }
    return nullptr;
}

void SlaveOperands::setShapeOperand(pSlicedOperand operand)
{
    m_operands[Shape] = operand;
}

void SlaveOperands::copy(const SlaveOperands& other)
{
        for (auto otherSlaveOp : other.m_operands)
        {
            pSlicedOperand slaveOp = std::make_shared<SlicedOperand>(*(otherSlaveOp.second));
            m_operands[otherSlaveOp.first] = slaveOp;
        }
}

bool SlaveOperands::empty()
{
    return m_operands.empty();
}

std::map<SlaveOperands::OpernadType, pSlicedOperand>::iterator SlaveOperands::begin() noexcept
{
    return m_operands.begin();
}

std::map<SlaveOperands::OpernadType, pSlicedOperand>::const_iterator SlaveOperands::begin() const noexcept
{
    return m_operands.begin();
}

std::map<SlaveOperands::OpernadType, pSlicedOperand>::iterator SlaveOperands::end() noexcept
{
    return m_operands.end();
}

std::map<SlaveOperands::OpernadType, pSlicedOperand>::const_iterator SlaveOperands::end() const noexcept
{
    return m_operands.end();
}

std::vector<pSlicedOperand> SlaveOperands::getSlaveOperands()
{
    std::vector<pSlicedOperand> operandsList;
    for(auto operandTupple : m_operands)
    {
        operandsList.push_back(operandTupple.second);
    }

    return operandsList;
}


std::atomic<uint32_t> Bundle::s_nextBundleIndex {0};
uint32_t              Bundle::getNextBundleIndex()
{
    return s_nextBundleIndex.fetch_add(1, std::memory_order_relaxed);
}

void Bundle::addNode(pNode node)
{
    m_nodes.push_back(node);
}

void Bundle::removeNode(pNode node)
{
    auto it = find(m_nodes.begin(), m_nodes.end(), node);
    m_nodes.erase(it);
}

const NodeVector& Bundle::getNodes() const
{
    return m_nodes;
}

std::string Bundle::getName() const
{
    return fmt::format("Bundle_{}", index());
}

bool Bundle::Solution::SlicedOperand::operator== (const SlicedOperand& other) const
{
    return (originalTensor     == other.originalTensor  &&
            chunkDimensions    == other.chunkDimensions &&
            resideInSRAM       == other.resideInSRAM    &&
            numOfBuffers       == other.numOfBuffers    &&
            finalElementType   == other.finalElementType &&
            alignWithCacheLine == other.alignWithCacheLine);
}

bool Bundle::Solution::SlicedOperand::operator!= (const SlicedOperand& other) const
{
    return !operator==(other);
}

std::string Bundle::Solution::SlicedOperand::toString()
{
    return fmt::format("{}|{}|{:d}|{}|{:d}|{}",
                       originalTensor->getId(),
                       numOfBuffers,
                       resideInSRAM,
                       finalElementType,
                       alignWithCacheLine,
                       fmt::join(chunkDimensions.begin(), chunkDimensions.end(), "_"));
}

void Bundle::Solution::SlicedOperand::resetSlicingData()
{
    chunkDimensions  = finalShape;
    finalElementType = originalTensor->getElementType();
    overlapElementsCount.fill(0);
    offsetBefore.fill(0);
    offsetAfter.fill(0);
    minValidSliceSize.fill(1);
    countPaddingOnlySlice = true;
    requiresTensorView    = false;
    resideInSRAM          = false;
    numOfBuffers          = 1;
    alignWithCacheLine    = false;
    postSlicingHandler    = nullptr;
    sharedChainMultiBuf   = false;
}

void Bundle::Solution::SlicedOperand::copyShapeData(const SlicedOperand& other)
{
    chunkDimensions       = other.chunkDimensions;
    overlapElementsCount  = other.overlapElementsCount;
    offsetBefore          = other.offsetBefore;
    offsetAfter           = other.offsetAfter;
    extraLeftoverAfter    = other.extraLeftoverAfter;
    countPaddingOnlySlice = other.countPaddingOnlySlice;
    requiresTensorView    = other.requiresTensorView;
    minValidSliceSize     = other.minValidSliceSize;
}

bool Bundle::Solution::SlicedOperand::hasOverlap() const
{
    for (unsigned dim = 0; dim < originalTensor->getDim(); dim++)
    {
        if (overlapElementsCount[dim] != 0)
        {
            return true;
        }
    }
    return false;
}

bool Bundle::Solution::SlicedOperand::hasExtraSizedSlices() const
{
    for (unsigned dim = 0; dim < originalTensor->getDim(); dim++)
    {
        if (extraLeftoverAfter[dim] > 0)
        {
            return true;
        }
    }
    return false;
}

bool Bundle::Solution::SlicedOperand::hasOffset() const
{
    for (unsigned dim = 0; dim < originalTensor->getDim(); dim++)
    {
        if ((offsetBefore[dim] != 0) || (offsetAfter[dim] != 0))
        {
            return true;
        }
    }
    return false;
}

bool Bundle::Solution::SlicedOperand::isFirstSliceSmaller() const
{
    bool hasOffsetBefore = false;
    for (unsigned dim = 0; dim < originalTensor->getDim(); dim++)
    {
        if (offsetBefore[dim] > 0)
        {
            hasOffsetBefore = true;
        }
    }
    // When we slice on spatial and the convolution node has padding before -
    // the offset is reduced from the first slice.
    return requiresTensorView && hasOffsetBefore;
}

bool Bundle::Solution::SlicedOperand::SliceOperandComp::operator()(const pSlicedOperand& lhs,
                                                                   const pSlicedOperand& rhs) const
{
    if (lhs->originalTensor->getId() != rhs->originalTensor->getId())
    {
        return lhs->originalTensor->getId() < rhs->originalTensor->getId();
    }

    if (lhs->resideInSRAM != rhs->resideInSRAM)
    {
        return lhs->resideInSRAM < rhs->resideInSRAM;
    }

    if (lhs->chunkDimensions != rhs->chunkDimensions)
    {
        return lhs->chunkDimensions < rhs->chunkDimensions;
    }

    if (lhs->alignWithCacheLine != rhs->alignWithCacheLine)
    {
        return lhs->alignWithCacheLine < rhs->alignWithCacheLine;
    }

    return lhs->numOfBuffers < rhs->numOfBuffers;
};

size_t Bundle::Solution::Operation::SliceReference::Hasher::operator()(const pSliceReference& s) const
{
    return std::hash<std::string>()(
        fmt::format("{}:{}", fmt::join(s->coordinates.begin(), s->coordinates.end(), "_"), s->operand->toString()));
}

bool Bundle::Solution::Operation::SliceReference::IsEqual::operator()(const pSliceReference& obj1,
                                                                      const pSliceReference& obj2) const
{
    if (*obj1->operand != *obj2->operand) return false;
    if (obj1->coordinates != obj2->coordinates) return false;
    return true;
}

bool BundleExpansion::isExpansionEnabledForRole(BundleExpansion::Role role)
{
    switch (role)
    {
    case BundleExpansion::WideInputProducer:
    case BundleExpansion::NarrowInputProducer:
        return true;
    case BundleExpansion::OutputConsumer:
        return GCFG_SRAM_SLICER_MME_TPC_EXPANSION_ENABLED.value();
    case BundleExpansion::SharedInputConsumer:
        return GCFG_SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED.value();
    case BundleExpansion::SlaveInputProducer:
        return GCFG_SRAM_SLICER_SHARED_MME_INPUT_PRODUCER_EXPANSION_ENABLED.value();
    case BundleExpansion::SlaveOutputConsumer:
        return GCFG_SRAM_SLICER_SHARED_MME_INPUT_CONSUMER_EXPANSION_ENABLED.value();
    default:
        HB_ASSERT(false, "Unexpected role");
        return false;
    }
}

// For logging
std::string BundleExpansion::role2String(const BundleExpansion::Role& role)
{
    switch (role)
    {
    case BundleExpansion::SharedInputConsumer:   return "SharedInputConsumer";
    case BundleExpansion::WideInputProducer:     return "WideInputProducer";
    case BundleExpansion::NarrowInputProducer:   return "NarrowInputProducer";
    case BundleExpansion::OutputConsumer:        return "OutputConsumer";
    case BundleExpansion::SlaveOutputConsumer:   return "SlaveOutputConsumer";
    case BundleExpansion::SlaveInputProducer:    return "SlaveInputProducer";
    default:
        HB_ASSERT(false, "Convert an unexpected role to string");
        return "UnknownRole";
    }
}

bool BundleExpansion::isDependentRole(BundleExpansion::Role role)
{

    return (role == BundleExpansion::SlaveOutputConsumer || /* Depends on  SharedInputConsumer*/
            role == BundleExpansion::SlaveInputProducer);   /* Depends on  SharedInputConsumer*/
}

bool BundleExpansion::isProducer(BundleExpansion::Role role)
{
    switch (role)
    {
        case BundleExpansion::WideInputProducer:
        case BundleExpansion::NarrowInputProducer:
        case BundleExpansion::SlaveInputProducer:
            return true;
        case BundleExpansion::OutputConsumer:
        case BundleExpansion::SharedInputConsumer:
        case BundleExpansion::SlaveOutputConsumer:
            return false;
        default:
            HB_ASSERT(false, "Unexpected bundle expansion role");
    }
    return true;
}

BundleExpansion::Role BundleExpansion::masterToSlaveEquivalentRole(BundleExpansion::Role role)
{
    switch(role)
    {
     case BundleExpansion::NarrowInputProducer:
     case BundleExpansion::WideInputProducer:
         return BundleExpansion::SlaveInputProducer;
     case BundleExpansion::OutputConsumer:
         return BundleExpansion::SlaveOutputConsumer;
     default:
         HB_ASSERT(false, "masterToSlaveEquivalentRole: unexpected role!");
         return BundleExpansion::NumOfRoles;
    }
}
