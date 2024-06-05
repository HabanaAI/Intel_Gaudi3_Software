#pragma once

#include "pattern_solvers.h"

/* This solver is designed to work on nodes with large common dim.
 * It will calculate the slicing needed on the common dimension and then slice on the non-common dimension.
 * slice size is decided by knob - minCDSizeForPartials
 * Will consider 2D (slice to knob exactly) or 4D tensors (slice on batch dim to reach as close as possible to knob)
 * After common dim is sliced - try to slice the non-common dim to fit in SRAM.
 * All inputs + output should reside in SRAM .
 */
class CommonDimSlicingSolver : public MmeBundleSolver
{
public:
    explicit CommonDimSlicingSolver(const HalReader& halReader, const pBundle& b) : MmeBundleSolver(halReader, b) {}

    bool effectiveForBundle() override;
    void createAllStrategies() override;

private:
    uint32_t
    getCommonDimSliceChunkSize(const pNode& mmeNode, const pSlicedOperand& slicedOperandA, bool isSlicedOnBatch) const;
    // After common dim is sliced - try to slice on the non-common dim.
    void applyNonCommonSlicing(const pMmeSlicingStrategy& strategy);
    // Slice non-common dim to create smaller shapes by alternately reduce each input tensor size in its non-common dimension
    static void doNonCommonDimSlicing(pMmeSlicingStrategy& strategy);
    // After batch dim is sliced - try to slice on the spatial dims. Returns the sliced dim, if a solution was found.
    Settable<unsigned> applySpatialSlicing(const pMmeSlicingStrategy& strategy, unsigned spatialSliceAxis);
    // Returns true if spatial slicing can be applied and false otherwise
    bool canApplySpatialSlicing(const pMmeSlicingStrategy& strategy);
    // Try to add more strategies with bigger slices on the inner most sliced spatial dim, leaving the sliced outer
    // dimensions unchanged. This will prevent situations that the spatial dims are sliced and the batch is not.
    void createStrategiesWithBiggerSpatialSlices(const Settable<unsigned>& spatialSlicedDim,
                                                 const SlicingStrategyPtr& initialStrategy);
    // Get the slicing dims of operandA,operandB
    std::pair<unsigned, unsigned> getSlicingDims(const pSlicedOperand& opA, const pSlicedOperand& opB) const;
    // Returns true if this solver supports the bundle MME node type
    bool isSupportedNodeType();
};
