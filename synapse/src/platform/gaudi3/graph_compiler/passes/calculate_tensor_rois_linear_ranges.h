#pragma once

#include "defs.h"
#include "passes/calculate_tensor_roi_linear_ranges.h"

namespace gaudi3
{
class CalculateTensorROIsLinearRanges : public ::CalculateTensorROIsLinearRanges
{
public:
    virtual void  calculateMmeLinearRanges(HabanaGraph& g, const NodePtr& node) const override;

protected:
    MmeCommon::EMmeOpType getMmeNodeOpType(const MmeNode&(mmeNode)) const override;
};

}  // namespace gaudi3