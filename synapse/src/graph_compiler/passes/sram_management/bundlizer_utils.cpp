#include "bundlizer_utils.h"


BundleExpansionBreakerInterface::~BundleExpansionBreakerInterface() = default;
BundleChainExpansionCheckerInterface::~BundleChainExpansionCheckerInterface() = default;

bool GranularityCoverBundleExpansionBreaker::shouldBreak() const
{
    const TensorTile& granularity = m_accessPattern->getTensorGranularity(m_connectingTensor);
    for (auto dim : m_slicingDims)
    {
        if (granularity.geometry[dim] >= m_connectingTensor->getSizeInElements(dim))
        {
            LOG_DEBUG(SRAM_SLICE,
                      "{} - dim {} granularity covers the full tensor dim size",
                      m_candidateNode->getNodeName(),
                      dim);
            return true;
        }
    }
    return false;
}

bool GranularityMultipleExpansionBreaker::shouldBreak() const
{
    const TensorTile& granularity = m_accessPattern->getTensorGranularity(m_connectingTensor);
    for (auto dim : m_slicingDims)
    {
        if (m_chunkDimensions[dim] % granularity.geometry[dim] != 0)
        {
            SLC_DEBUG("{} - slice size ({}) is not a multiple of the granularity ({}) in the sliced dim ({})",
                      m_connectingTensor->getName(),
                      m_chunkDimensions[dim],
                      granularity.geometry[dim],
                      dim);
            return true;
        }
    }
    return false;
}

bool OverlapExpansionBreaker::shouldBreak() const
{
    const IntersectionTile& overlap = m_accessPattern->getTensorOverlap(m_connectingTensor);
    for (auto dim : m_slicingDims)
    {
        if (overlap.geometry[dim] != 0)
        {
            LOG_DEBUG(SRAM_SLICE,
                      "{} - selected dim slices contain overlap on dim {}, so they cannot be sliced",
                      m_candidateNode->getNodeName(),
                      dim);
            return true;
        }
    }
    return false;
}

bool OffsetExpansionBreaker::shouldBreak() const
{
    const TensorTile& granularity = m_accessPattern->getTensorGranularity(m_connectingTensor);
    for (auto dim : m_slicingDims)
    {
        if (granularity.offset[dim] != 0)
        {
            SLC_DEBUG("{} - sliced dim ({}) contains offset ({})",
                      m_connectingTensor->getName(),
                      dim,
                      granularity.offset[dim]);
            return true;
        }
    }
    return false;
}

bool NonStichedOffsetOverlapExpansionBreaker::shouldBreak() const
{
    for (const auto& nonStitchedOperand : m_candidateNode->getOperands())
    {
        if (nonStitchedOperand && (nonStitchedOperand != m_connectingTensor))  // Skip the stitched operand
        {
            for (auto dim : m_slicingDims)
            {
                const auto& matchingSlicingDims =
                    m_accessPattern->getTensorMatchingSlicedDims(nonStitchedOperand, m_connectingTensor, dim);
                const TensorTile& nonStitchedOperandGranularity =
                    m_accessPattern->getTensorGranularity(nonStitchedOperand);
                const IntersectionTile& nonStitchedOperandOverlap =
                    m_accessPattern->getTensorOverlap(nonStitchedOperand);
                for (const auto& projectedDim : matchingSlicingDims)
                {
                    if ((nonStitchedOperandOverlap.geometry[projectedDim] != 0) ||
                        (nonStitchedOperandGranularity.offset[projectedDim] != 0))
                    {
                        SLC_DEBUG(
                            "Non stitched operand {} - projected sliced dim ({}) contains overlap ({}) or offset ({})",
                            nonStitchedOperand->getName(),
                            projectedDim,
                            nonStitchedOperandOverlap.geometry[projectedDim],
                            nonStitchedOperandGranularity.offset[projectedDim]);
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool BaseBundleChainExpansionChecker::isChainBreaker() const
{
    const auto expansionBreakers = getExpansionBreakers();
    if (std::any_of(expansionBreakers.cbegin(), expansionBreakers.cend(), [](const auto& expansionBreaker) {
            return expansionBreaker->shouldBreak();
        }))
    {
        // Chain break needed
        return true;
    }
    else
    {
        // No chain break needed
        return false;
    }
}
