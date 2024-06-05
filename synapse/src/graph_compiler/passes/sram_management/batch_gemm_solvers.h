#pragma once

#include "pattern_solvers.h"

class BatchGemmSolver : public MmeBundleSolver
{
public:
    BatchGemmSolver(const HalReader& halReader, const pBundle& b) : MmeBundleSolver(halReader, b) {}
    ~BatchGemmSolver() override = default;
    bool effectiveForBundle() override;
    void createAllStrategies() override;
};

class BatchTinyGemmSolver : public MmeBundleSolver
{
public:
    BatchTinyGemmSolver(const HalReader& halReader, const pBundle& b) : MmeBundleSolver(halReader, b) {}
    ~BatchTinyGemmSolver() override = default;
    bool effectiveForBundle() override;
    void createAllStrategies() override;

private:
    bool calculateChunkSize(const pMmeSlicingStrategy&           strategy,
                            const std::array<pSlicedOperand, 3>& operands,
                            const unsigned                       dim,
                            const unsigned                       maxChunkSize);
    void fillChunksWithMaxPossibleSize(const pMmeSlicingStrategy& strategy, const unsigned maxBatchSize);
    void createStrategySlices(const pMmeSlicingStrategy& strategy);
};
