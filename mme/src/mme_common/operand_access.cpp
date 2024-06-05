#include "operand_access.h"
#include "access_pattern_utils.h"

namespace MmeCommon
{
uint64_t OperandAccess::granularityByTensorDim(EMmeInternalOperand operand, Dim dim) const
{
    return tensorDimGranularity(operandAccessPattern(operand), dim);
}

uint64_t OperandAccess::granularityByTensorDim(EMmeOperand operand, Dim dim) const
{
    return tensorDimGranularity(operandAccessPattern(operand), dim);
}

std::vector<OperandAccess::Dim> OperandAccess::mappedTensorDims(EMmeOperand operand, Dim idxSpcDim) const
{
    const auto& tap = operandAccessPattern(operand);
    std::vector<Dim> mappedDims;
    for (Dim tensorDim = 0; tensorDim < tap.dimsAccessPattern.size(); tensorDim++)
    {
        if (tap.dimsAccessPattern.at(tensorDim).indexSpaceDim == idxSpcDim)
        {
            mappedDims.push_back(tensorDim);
        }
    }
    return mappedDims;
}

const AccessPattern::TensorAccessPattern& OperandAccess::operandAccessPattern(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        case e_mme_op_a:
            return AccessPatternDetails::Utils::accessPatternForOperandA(m_accessPattern);
        case e_mme_op_b:
            return AccessPatternDetails::Utils::accessPatternForOperandB(m_accessPattern);
        default:
            return AccessPatternDetails::Utils::accessPatternForOperandC(m_accessPattern);
    }
}

const AccessPattern::TensorAccessPattern& OperandAccess::operandAccessPattern(EMmeOperand operand) const
{
    switch (operand)
    {
        case e_mme_op_x:
            return m_accessPattern.operandAccessPatterns.at(OperandRole::X);
        case e_mme_op_w:
            return m_accessPattern.operandAccessPatterns.at(OperandRole::W);
        default:
            return m_accessPattern.operandAccessPatterns.at(OperandRole::Y);
    }
}

uint64_t OperandAccess::tensorDimGranularity(const AccessPattern::TensorAccessPattern& tap, Dim tensorDim) const
{
    const auto& dimAP = tap.dimsAccessPattern.at(tensorDim);
    Dim nodeDim = dimAP.indexSpaceDim;
    return m_nodeGranularity.at(nodeDim) * dimAP.size;
}
}  // namespace MmeCommon