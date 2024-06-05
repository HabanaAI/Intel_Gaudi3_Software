#pragma once

#include "passes/calculate_tensor_roi_linear_ranges.h"

namespace gaudi
{

class MmeDescGeneratorWorkItem;

class CalculateTensorROIsLinearRanges : public ::CalculateTensorROIsLinearRanges
{
public:
    bool apply(HabanaGraph& g) const override;

    virtual void calculateMmeLinearRanges(HabanaGraph& g, const pNode& node) const override;

private:

    // For MME the linear ranges is calculated using
    // the descriptor generator
    // For not invoke it twice the descriptors is saved to graph.
    mutable std::list<MmeDescGeneratorWorkItem*> m_generators;
};

}
