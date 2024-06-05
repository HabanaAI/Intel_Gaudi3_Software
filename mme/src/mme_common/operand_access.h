#ifndef MME__OPERAND_ACCESS_H
#define MME__OPERAND_ACCESS_H

#include "include/mme_access_pattern.h"

namespace MmeCommon
{
class OperandAccess
{
public:
    using Dim = size_t;

    template<typename GranularityContainer>
    OperandAccess(AccessPattern accessPattern, const GranularityContainer& nodeGranularity)
    : m_accessPattern(std::move(accessPattern)), m_nodeGranularity(nodeGranularity.begin(), nodeGranularity.end())
    {
    }

    // Returns the granularity of the operand at the given dimension
    uint64_t granularityByTensorDim(EMmeInternalOperand operand, Dim dim) const;
    uint64_t granularityByTensorDim(EMmeOperand operand, Dim dim) const;

    // Returns the operand dimensions that are mapped to the given index space dimension
    std::vector<Dim> mappedTensorDims(EMmeOperand operand, Dim idxSpcDim) const;

private:
    using Granularity = AccessPattern::IndexSpaceVector;

    const AccessPattern m_accessPattern;
    const Granularity m_nodeGranularity;

    const AccessPattern::TensorAccessPattern& operandAccessPattern(EMmeInternalOperand operand) const;
    const AccessPattern::TensorAccessPattern& operandAccessPattern(EMmeOperand operand) const;
    uint64_t tensorDimGranularity(const AccessPattern::TensorAccessPattern& tap, Dim tensorDim) const;
};
}  // namespace MmeCommon

#endif //MME__OPERAND_ACCESS_H
