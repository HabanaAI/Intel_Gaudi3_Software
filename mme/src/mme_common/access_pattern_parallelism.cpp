#include "access_pattern_parallelism.h"
#include "access_pattern_utils.h"
#include "utils/logger.h"

namespace MmeCommon::AccessPatternDetails
{
void AccessPatternParallelismAdjuster::setParallelized(Dim idxSpcDim, size_t parallelismLevel)
{
    storeParallelism(idxSpcDim, parallelismLevel);
    auto scale = calcScale(idxSpcDim, parallelismLevel);
    resizeIndexSpace(idxSpcDim, parallelismLevel);
    rescaleOperandsAP(idxSpcDim, scale);
    addAuxAP(idxSpcDim, parallelismLevel);
}

void AccessPatternParallelismAdjuster::storeParallelism(Dim idxSpcDim, size_t parallelismLevel)
{
    MME_ASSERT(m_parallelismLevels.find(idxSpcDim) == m_parallelismLevels.end(),
               fmt::format("Index space dimension {} is already parallelized", idxSpcDim));
    m_parallelismLevels[idxSpcDim] = parallelismLevel;
}

uint64_t AccessPatternParallelismAdjuster::calcScale(Dim idxSpcDim, size_t parallelismLevel) const
{
    return div_round_up(m_accessPattern.indexSpace.at(idxSpcDim), parallelismLevel);
}

void AccessPatternParallelismAdjuster::resizeIndexSpace(Dim idxSpcDim, size_t parallelismLevel)
{
    m_accessPattern.indexSpace.at(idxSpcDim) = parallelismLevel;
}

void AccessPatternParallelismAdjuster::rescaleOperandsAP(Dim idxSpcDim, size_t scale)
{
    for (auto& operandAP : m_accessPattern.operandAccessPatterns)
    {
        rescaleOperandAP(operandAP.first, idxSpcDim, scale);
    }
}

void AccessPatternParallelismAdjuster::rescaleOperandAP(OperandRole role, size_t idxSpcDim, size_t scale)
{
    for (auto& dimAP : m_accessPattern.operandAccessPatterns.at(role).dimsAccessPattern)
    {
        if (dimAP.indexSpaceDim == idxSpcDim)
        {
            dimAP.size *= scale;
            dimAP.stride *= scale;
        }
    }
}

void AccessPatternParallelismAdjuster::addAuxAP(Dim idxSpcDim, size_t parallelismLevel)
{
    if (m_accessPattern.operandAccessPatterns.find(OperandRole::SCRATCH_PAD) ==
        m_accessPattern.operandAccessPatterns.end())
    {
        // ScratchPad access pattern doesn't exist
        m_accessPattern.operandAccessPatterns[OperandRole::SCRATCH_PAD] =
            Utils::accessPatternForOperandC(m_accessPattern);
    }
    m_accessPattern.operandAccessPatterns[OperandRole::SCRATCH_PAD].dimsAccessPattern.push_back(
        Utils::create1To1DimAccessPattern(idxSpcDim));

    m_accessPattern.operandAccessPatterns[OperandRole::CONST].dimsAccessPattern.push_back(
        Utils::create1To1DimAccessPattern(idxSpcDim));
}
}  // namespace MmeCommon::AccessPatternDetails