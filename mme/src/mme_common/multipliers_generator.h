#ifndef MME__MULTIPLIERS_GENERATOR_H
#define MME__MULTIPLIERS_GENERATOR_H

#include "include/mme_aspects.h"
#include "operand_access.h"

namespace MmeCommon::Brain::SolutionMultipliers
{
using Dim = size_t;

/*
 * This module is responsible to populate the solution multipliers based on:
 * 1. The base granularity of each index space dimension
 * 2. Previous solution multipliers
 * 3. The desired size for each aspect.
 */
class MultipliersGenerator
{
public:
    MultipliersGenerator(const MmeLayerParams& params,
                         const AccessPattern& accessPattern,
                         const MultiplierArray& granularity,
                         const MultiplierArray& previousMultipliers,
                         AspectFactoryPtr&& aspectFactory);

    // Inflate the previous multipliers until they reach (but not exceed) the desired size in the selected aspect
    void inflateUpTo(AspectName aspectName, uint64_t desiredSize, EMmeOperand operand);
    // Inflate the previous multipliers until they reach (with minimal overflow) the desired size in the selected aspect
    void inflateAtLeastTo(AspectName aspectName, uint64_t desiredSize, EMmeOperand operand);
    // Set the multipliers of the aspect to represent no slicing (full dims sizes)
    void setMaxMultiplier(AspectName aspectName);
    // Reset the multipliers of the aspect to the previous solution multipliers
    void resetToPrevious(AspectName aspectName);
    // Set the multipliers to a given initial solution
    void setMultipliers(MultiplierArray initialSolution);

    const MultiplierArray& getSolution() const;
    uint64_t aspectCurrentSize(AspectName aspectName, EMmeOperand operand) const;
    uint64_t aspectFullSize(AspectName aspectName, EMmeOperand operand) const;

private:
    const MmeLayerParams& m_params;
    const MultiplierArray m_maxMultipliers;

    const OperandAccess m_operandAccess;
    const MultiplierArray m_previousMultipliers;

    const AspectFactoryPtr m_aspectFactory;

    MultiplierArray m_solution;

    //
    // Aspect sepcific details calculations
    //

    IndexSpaceAspect idxSpaceAspect(AspectName aspectName) const;
    OperandAspect operandAspect(AspectName, EMmeOperand operand) const;

    //
    // Index space dim calculations
    //

    static MultiplierArray calculateMaxMultipliers(const AccessPattern& accessPattern,
                                                   const MultiplierArray& granularity);
    // Inflate the current solution by the factor in the given index space dim.
    void inflate(Dim idxSpcDim, uint64_t factor);
};

// The multiplier generator can be configured to work on any aspect type:
// - Logical semantic representation of the problem description (channles, spatials, etc.)
// - HW aspects - the height of the EUs or the geometry
// - etc.
// This namespace contains the implementation needed to use it to generate according to the aspects of the MME
// configuration aspects - geometry, concurrency, etc.
namespace MmeAspects
{
class MultipliersGenerator : public SolutionMultipliers::MultipliersGenerator
{
public:
    MultipliersGenerator(const MmeLayerParams& params,
                         const AccessPattern& accessPattern,
                         const MultiplierArray& granularity,
                         const MultiplierArray& previousMultipliers);

private:
};

}  // namespace MmeAspects
}  // namespace MmeCommon::Brain::SolutionMultipliers

#endif //MME__MULTIPLIERS_GENERATOR_H
