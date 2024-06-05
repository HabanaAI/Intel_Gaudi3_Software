#pragma once

#include "graph_compiler/passes/sram_management/strategy_operations_accumulator.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"
#include "cost_model.h"

namespace gaudi
{

// This structure holds the accumulated cost of all the operations of a single engine (MME/TPC/DMA).
// For later aggregation of the different engine costs to a single strategy cost, there may be a need
// to differentiate the first/last operation(s) from the total. For example, in the case of MME compute
// bound execution with TPC producer and TPC consumer, the bundle execution time will be composed of:
//  1. the MME total compute time
//  2. the time to produce the first inputs (TPC in parallel to DMA)
//  3. the time of the last MME slice processing by the TPC consumer
// For 2 and 3, the TPC/DMA prefix and suffix are required.
struct AccumulatedCost
{
    explicit AccumulatedCost(CostModel::Cost::Engine engine) : prefix(engine), suffix(engine), total(engine) {}
    virtual ~AccumulatedCost() = default;

    // Prefix is the cost of the first operation.
    CostModel::Cost prefix;
    // Suffix is the cost of the last operation.
    CostModel::Cost suffix;
    // Total cost of all the operations including the prefix and suffix
    CostModel::Cost total;


    AccumulatedCost& operator=(const AccumulatedCost&) = default;

    std::string toString() const
    {
        return fmt::format("Prefix[{}]  Suffix[{}]  Total[{}]", prefix.toString(), suffix.toString(), total.toString());
    }
};

// Evaluator of the cost of executing a bundle's computation according to the given strategy.
class StrategyCostModel
{
public:
    using CostEngine = CostModel::Cost::Engine;

    // Constructor
    StrategyCostModel(const HabanaGraph&         graph,
                      const pBundle&             bundle,
                      const pMmeSlicingStrategy& strategy,
                      const AllBrains&           slicingBrains)
    : m_graph(graph), m_bundle(bundle), m_strategy(strategy), m_slicingBrains(slicingBrains)
    {
    }

    static void reset();

    // Execution
    void model(CostModel& mmeCostModel,
               CostModel& tpcCostModel,
               CostModel& fetchCostModel,
               CostModel& evictCostModel);

    // Results queries
    const AccumulatedCost& getMmeCost() const;
    const AccumulatedCost& getTpcCost() const;

    // Fetching (copying from HBM to SRAM) and evicting (SRAM to HBM) don't have to always be implemented by the
    // same engine. The accumulator gives the different costs per engine.
    const AccumulatedCost& getFetchCost(CostEngine engine) const;
    const AccumulatedCost& getEvictionCost(CostEngine engine) const;

    const CostModel::Cost& getAggregatedCost() const;
    const CostModel::Cost& getInvalidCandidateCost(const pBundleExpansion& candidate) const;
    static const std::map<NodeSet, gaudi::CostModel::Cost>& getInvalidCandidatesCache();

    enum class BundleExecutionType
    {
        MMEComputeBound,
        TPCComputeBound,
    };

private:
    const HabanaGraph&         m_graph;
    const pBundle&             m_bundle;
    const pMmeSlicingStrategy& m_strategy;
    const AllBrains&           m_slicingBrains;

    // The invalid candidates cost is cached since many strategies (in the same bundle)
    // will need the same values (the same node/nodes might be invalid in multiple strategies).
    // This cache will be restarted in init().
    thread_local static std::map<NodeSet, gaudi::CostModel::Cost> s_invalidCandidatesCache;

    AccumulatedCost m_mmeAccumulatedCost{CostEngine::MME};
    AccumulatedCost m_tpcAccumulatedCost{CostEngine::TPC};
    AccumulatedCost m_tpcEvictAccumulatedCost{CostEngine::TPC};
    AccumulatedCost m_dmaFetchAccumulatedCost{CostEngine::DMA};
    AccumulatedCost m_dmaEvictAccumulatedCost{CostEngine::DMA};

    CostModel::Cost m_aggregatedCost{CostEngine::AGGREGATION};

    std::unordered_map<pNode, CostModel::Cost> m_invalidCandidatesCosts;

    bool m_modeled = false;     // A state that keeps whether the model() was invoked yet or not.

    void accumulateIndividualEngineCosts(CostModel& mmeCostModel,
                                         CostModel& tpcCostModel,
                                         CostModel& fetchCostModel,
                                         CostModel& evictCostModel);

    BundleExecutionType aggregateCosts();
    BundleExecutionType evaluateExecutionType();

    void aggregateHbmTraffic();
    void aggregateMmeComputeBoundTime();
    void aggregateTpcComputeBoundTime();
    void fixTimeForBwBoundExecution();
    void aggregateSlicingOverheads();
    void aggregateInvalidCandidatesCost();

    void addPrefixForMmeComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData);
    void addSuffixForMmeComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData);
    void addPrefixForMmeWithoutTpcProducer();

    void addSuffixForTpcComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData);
    void addPrefixForTpcComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData);

    void modelInvalidCandidates(CostModel& mmeCostModel,
                                CostModel& tpcCostModel,
                                CostModel& fetchCostModel,
                                CostModel& evictCostModel);
    void modelTpcCandidate(const pBundleExpansion& candidate, CostModel& costModel);
    void modelSlaveMmeCandidates(const pBundleExpansion&            slaveMme,
                                 const std::list<pBundleExpansion>& slaveTpcCandidates,
                                 CostModel&                         mmeCostModel,
                                 CostModel&                         tpcCostModel,
                                 CostModel&                         fetchCostModel,
                                 CostModel&                         evictCostModel);

    static uint64_t dataMovementTimeNano(uint64_t trafficBytes);
    uint64_t aggregateTpcDmaParallelTimeNano(const CostModel::Cost& tpcCost, const CostModel::Cost& dmaCost) const;
    bool isEvicted(const pSlicedOperand& slicedOperand) const;
    void            printStrategyCostLog(const BundleExecutionType& executionType) const;
    static bool isEvictedByTpc(const pSlicedOperand& slicedOperand);
    static bool     isOperandInSram(const pSlicedOperand& slicedOperand);
};

} // namespace gaudi