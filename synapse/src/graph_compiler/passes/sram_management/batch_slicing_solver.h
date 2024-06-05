#pragma once

#include "pattern_solvers.h"
#include "spatial_slicing_solver.h"
#include "mme_geometry.h"
#include "conv_base_node.h"

/*
 * Slicing 4D tensors on the non-common dimension. Slicing strategies will be returned
 * in a priority Q s.t. the highest priority given to the optimal solution.
 */
class NonCommon4DSolver : public NonCDSolver
{
public:
    explicit NonCommon4DSolver(const HalReader& halReader, const pBundle& b) : NonCDSolver(halReader, b) {}
    NonCommon4DSolver(const HalReader& halReader, const pBundle& b, const pMmeSlicingStrategy& initialStrategy);
    virtual ~NonCommon4DSolver() = default;
    bool effectiveForBundle() override;
    virtual void createAllStrategies() override;

private:
    Settable<unsigned> getNarrowSlicingDim(const pMmeSlicingStrategy& strategy) const override;
    Settable<unsigned> getWideSlicingDim(const pMmeSlicingStrategy& strategy) const override;

    // by default walking pattern assumes walking on a single height dimension (B)
    DimVector getWalkingPattern(unsigned numHeightSlicingDims = 1);

    void createStrategiesForGeometry(DimVector& walkingPattern, const MmeGeometry& geometry);

    bool isSpatialSlicingSupported(unsigned dim);

    bool computeSlicingForBatch(unsigned int batch, pMmeSlicingStrategy &strategy);
    bool     computeSlicingForSpatial(TSize                outputSliceHeight,
                                      pMmeSlicingStrategy& strategy,
                                      unsigned             numSpatialDimsToSlice,
                                      bool&                validSlicing);
    bool computeSlicingForNarrow(pMmeSlicingStrategy &strategy);
    bool     sliceSpatialDim(unsigned        wideSlicingDim,
                             TSize           outputSliceSize,
                             pSlicedOperand& wideOperand,
                             pSlicedOperand& outputOperand,
                             OffsetArray&    slicePadBefore,
                             OffsetArray&    slicePadAfter);
    void     getOutputSpatialDimSlicingBoundsAndStep(TSize     origDimSize,
                                                     unsigned  dim,
                                                     TSize&    startSize,
                                                     int&      step,
                                                     TSize&    limitSize);
    TSize getInputSliceSize(const SizeArray& outputSliceSizes, unsigned numDims, unsigned slicedDim, TSize maxInputSize);

    static TSize calculateDimsSize(const SizeArray& sizes, const DimVector& dims);
};
