#pragma once

#include "pattern_solvers.h"

/*
 * This solver's goal is to create additional strategies on top of the original ones.
 * The additional strategies should have bigger slices and therefore, using them should lead to a smaller graph,
 * which is also faster to compile
 */
class GraphSizeOptimizationSolver : public MmeBundleSolver
{
public:
    using slicedOperandAndDim = MmeSlicingStrategy::MmeSlicingData::slicedOperandAndDim;
    using slicedOperandAndDimList = MmeSlicingStrategy::MmeSlicingData::slicedOperandAndDimList;
    using slicedOperandAndDimListList = std::list<slicedOperandAndDimList>;

    GraphSizeOptimizationSolver(const HalReader&          halReader,
                                const pBundle&            b,
                                const SlicingStrategyPtr& initialStrategy);
    virtual ~GraphSizeOptimizationSolver() = default;

    bool effectiveForBundle() override { return true; }
    virtual void createAllStrategies() override;

private:
    void findSlicedOperandAndDims();
    void createStrategiesWithBiggerSlices(const slicedOperandAndDimList& sameProportionSlicedOpAndDims);

    SlicingStrategyPtr m_initialStrategy = nullptr;
    //each sublist contains a group of operand+dim that should be increased together
    slicedOperandAndDimListList m_slicedOperandAndDims;
};
