#pragma once

#include "mme_slicing_strategy.h"

// Base class only calculate sram capacity of operands
class MetricsCalculator
{
public:
    MetricsCalculator(const HalReader& halReader, SlicingStrategy* strategy, SlicingStrategy::Metrics* metrics);
    virtual ~MetricsCalculator() = default;
    virtual SlicingStrategy::Metrics& calculate();
    virtual SlicingStrategy::Metrics& recalculateSramCapacityOnly();

protected:
    virtual uint64_t calculateSRAMCapacity(const std::vector<pSlicedOperand>& operands);

    const HalReader&          m_halReader;
    SlicingStrategy* m_strategy;
    SlicingStrategy::Metrics* m_metrics;
};

class MmeBundleMetricsCalculator : public MetricsCalculator
{
public:
    MmeBundleMetricsCalculator(const HalReader&             halReader,
                               MmeSlicingStrategy*          strategy,
                               MmeSlicingStrategy::Metrics* metrics)
    : MetricsCalculator(halReader, strategy, metrics), m_mmeStrategy(strategy)
    {}
    ~MmeBundleMetricsCalculator() = default;
    virtual SlicingStrategy::Metrics& calculate() override;
    virtual SlicingStrategy::Metrics& recalculateSramCapacityOnly() override;
protected:
    SlicingStrategy::Metrics& calculate(bool recalcSramCapOnly);
    virtual uint64_t calculateSRAMCapacity(const std::vector<pSlicedOperand>& operands) override;
    unsigned calculateSBReuse();
    float calculateMMEUtilization();
    void addCandidateUtilization(const pBundleExpansion& candidate, uint64_t& sizeToProcess, float& mmeActivations);
    double calculateHBMBandwidth() const;
    void fixMMEUtilIfHBMOverflow();

    // Processing time in nsec
    double getMMEProcessingTime() const;
    double getTPCProcessingTime() const;
    double getSharedOperandMMEProcessingTime() const;

    // no. of MME tetrises in the sliced master output (assuming it's MME)
    double getNumOfMMETetrises() const;
    double getNumOfMMETetrisesOfCandidate(const pBundleExpansion& candidate) const;
    double calculateNumOfMMETetrises(const pSlicedOperand& outputOperand,
                                     const DimVector&      wideOutputSlicingDims,
                                     const DimVector&      narrowOutputSlicingDims) const;

    // no. of bytes read/written from/to HBM by TPC nodes (bundled and candidates)
    uint64_t getTPCHBMTraffic() const;

    // no. of bytes read/written from/to HBM by MME nodes (bundled and candidates)
    uint64_t getMMEHBMTraffic() const;
    uint64_t getMMEWideHBMTraffic() const;
    uint64_t getMMENarrowHBMTraffic() const;
    uint64_t getMMEOutputHBMTraffic() const;
    uint64_t getMMESharedOperandHbmTraffic() const;

    MmeSlicingStrategy* m_mmeStrategy;
};
