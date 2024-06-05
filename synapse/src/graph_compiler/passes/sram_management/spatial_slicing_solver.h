#pragma once

#include "pattern_solvers.h"
#include "mme_geometry.h"

class NonCDSolver : public MmeBundleSolver
{
public:
    NonCDSolver(const HalReader& halReader, const pBundle& b);
    NonCDSolver(const HalReader& halReader, const pBundle& b, const pMmeSlicingStrategy& initialStrategy);
    ~NonCDSolver() override = default;

protected:
    pMmeSlicingStrategy getInitialStrategy(DimVector& traversalPattern, MmeGeometry geometry);

    virtual Settable<unsigned> getNarrowSlicingDim(const pMmeSlicingStrategy& strategy) const = 0;
    virtual Settable<unsigned> getWideSlicingDim(const pMmeSlicingStrategy& strategy) const = 0;
    virtual unsigned getNarrowOutputSlicingDim(pMmeSlicingStrategy& strategy);
    virtual unsigned getWideOutputSlicingDim(pMmeSlicingStrategy& strategy);

    pMmeSlicingStrategy m_initialStrategy = nullptr;
    const uint64_t      c_sliceSizeFactor;
};

/*
 * Solver responsible for slicing the tensors on the non-common dimension
 * It will return the slicing strategy ordered in the strategy queue such that
 * the optimal strategy considering only the MME node is first.
 * This solver is effective for 2D tensor only.
 */
class NonCD2DSolver : public NonCDSolver
{
public:
    NonCD2DSolver(const HalReader& halReader, const pBundle& b) : NonCDSolver(halReader, b) {}
    NonCD2DSolver(const HalReader& halReader, const pBundle& b, const pMmeSlicingStrategy& initialStrategy);
    virtual ~NonCD2DSolver() = default;
    bool effectiveForBundle() override;
    virtual void createAllStrategies() override;
private:
    DimVector           getOutputSlicingDimList();
    void setNarrowSlice(pMmeSlicingStrategy& strategy);
    void findWideSlicing(pMmeSlicingStrategy& strategy);
    pMmeSlicingStrategy findStrategyWithSmallerWideSlicing(const pMmeSlicingStrategy& strategy); // return null if failed to find.
    void findNarrowSlicing(pMmeSlicingStrategy& strategy);

    Settable<unsigned> getNarrowSlicingDim(const pMmeSlicingStrategy& strategy) const override;
    Settable<unsigned> getWideSlicingDim(const pMmeSlicingStrategy& strategy) const override;

    bool shouldFlatten() const;
};
