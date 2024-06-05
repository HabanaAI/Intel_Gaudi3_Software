#pragma once

#include "pattern_solvers.h"

class TpcBundleSolver : public Solver
{
public:
    TpcBundleSolver(const HalReader& halReader, const pBundle& bundle);
    virtual ~TpcBundleSolver() = default;
    bool effectiveForBundle() override;
    virtual void createAllStrategies() override;

    // Return pair of dimension and divider
    static Settable<std::pair<uint32_t, uint32_t>> getDimAndChunk(const TensorPtr&           tensor,
                                                                  const StrategySlicingData& mmeSlicingData);

private:
    // Add operands to strategy data
    // Return the added operands
    std::list<pSlicedOperand> addTensorsOperands(const TensorVector& tensors, StrategySlicingData& slicingData);

    NodeList getFinalNode();
    bool     checkIfDuplicatesExist(const std::vector<pSlicedOperand>& bundleTensors) const;
    void     removeDuplicatesIfExist(std::vector<pSlicedOperand>& bundleTensors) const;

    std::map<TensorPtr, pSlicedOperand> m_tensorToOperand;
};
