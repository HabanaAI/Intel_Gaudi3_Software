#pragma once

#include "mme_access_pattern.h"
#include "index_space_dimensions.h"

namespace MmeCommon::AccessPatternDetails
{
struct Utils
{
    using DimAP = AccessPattern::TensorAccessPattern::DimAccessPattern;

    static DimAP create1To1DimAccessPattern(Dim indexSpaceDim)
    {
        DimAP dimAP {};
        dimAP.indexSpaceDim = indexSpaceDim;
        dimAP.offset = 0;
        dimAP.size = 1;
        dimAP.stride = 1;
        return dimAP;
    }

    static DimAP createAllReqDimAccessPattern(uint64_t tensorDimSize, Dim indexSpaceDim)
    {
        DimAP dimAP {};
        dimAP.indexSpaceDim = indexSpaceDim;
        dimAP.size = tensorDimSize;
        dimAP.offset = 0;
        dimAP.stride = 0;
        return dimAP;
    }

    static const AccessPattern::TensorAccessPattern& accessPatternForOperandA(const AccessPattern& accessPattern)
    {
        MME_ASSERT(accessPattern.roleA != OperandRole::INVALID, "Input-A access pattern not defined");
        return accessPattern.operandAccessPatterns.at(accessPattern.roleA);
    }

    static const AccessPattern::TensorAccessPattern& accessPatternForOperandB(const AccessPattern& accessPattern)
    {
        MME_ASSERT(accessPattern.roleB != OperandRole::INVALID, "Input-B access pattern not defined");
        return accessPattern.operandAccessPatterns.at(accessPattern.roleB);
    }

    static const AccessPattern::TensorAccessPattern& accessPatternForOperandC(const AccessPattern& accessPattern)
    {
        MME_ASSERT(accessPattern.roleC != OperandRole::INVALID, "Output access pattern not defined");
        return accessPattern.operandAccessPatterns.at(accessPattern.roleC);
    }

    static const AccessPattern::TensorAccessPattern& accessPatternForOperand(EMmeInternalOperand operand,
                                                                             const AccessPattern& accessPattern)
    {
        switch (operand)
        {
            case e_mme_op_a:
                return accessPatternForOperandA(accessPattern);
            case e_mme_op_b:
                return accessPatternForOperandB(accessPattern);
            case e_mme_op_c:
                return accessPatternForOperandC(accessPattern);
            default:
                MME_ASSERT(false, "Unsupported operand");
                break;
        }
        // dummy
        return accessPatternForOperandA(accessPattern);
    }
};
}  // namespace MmeCommon::AccessPatternDetails