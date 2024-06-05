#pragma once

#include "hal_reader/hal_reader.h"
#include "mme_dim_controller.h"

// The purpose of this class is to give solvers a range in which to search for the optimal solution (and replace
// the slicing brain knob that rigidly limit all the solutions with the same range).
// The solvers optimize for the execution of a specific bundle at a specific point in time,
// and limiting them helps with the bigger picture and the execution of multiple bundles,
// bundle interleaving, bundle expansion, etc.
class ReuseLimitCalculator
{
public:
    explicit ReuseLimitCalculator(const HalReader& halReader, const pNode& node);

    // Limit is used for both operands since it is not known which will be the wide and which the narrow at the time
    // of setting the limit.
    // Different limits for different operands can be added in the future.
    virtual uint64_t getLimit() const;

    // Ratio between ideal processing time (100% MME utilization) and worst case traffic time.
    // height and width are the aggregated spatial dimensions size of the output slice or tensor.
    virtual double getPrTrRatio(uint64_t height, uint64_t width) const;

protected:
    pNode m_node;
    MmeDimController m_dimCtrl;
    const uint64_t c_mmeVecSizeElement;

    static uint64_t aggregateDimSizes(const pTensor& operand, const DimVector& dims);
    inline uint64_t minimalReuseLimit() const;
    double getProcessingTime(uint64_t height, uint64_t width, uint64_t cdSize) const;
    double getTrafficTime(uint64_t height, uint64_t width, uint64_t cdSize) const;
};
