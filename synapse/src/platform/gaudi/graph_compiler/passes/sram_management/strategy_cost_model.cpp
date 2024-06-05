#include "strategy_cost_model.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"
#include "sram_management/pattern_solvers.h"

namespace gaudi
{

using CostEngine = CostModel::Cost::Engine;

// In C++11, enum class can't be used as unordered_map key without custom hash functor
struct CostEngineHasher
{
    size_t operator()(CostEngine e) const { return static_cast<size_t>(e); }
};

// Each cost model may return cost from different engines. This accumulator sums them by engine,
// keeping a prefix and suffix (the first and last operation) for each engine.
class CostModelPerEngineAccumulator
{
public:
    explicit CostModelPerEngineAccumulator(CostModel& costModel,
                                           unsigned   numOfMMEPrefixOps = 1,
                                           unsigned   numOfTPCPrefixOps = 1,
                                           unsigned   numOfDMAPrefixOps = 1)
    : m_costModel(costModel)
    {
        m_enginesPrefixCount[CostEngine::MME] = numOfMMEPrefixOps;
        m_enginesPrefixCount[CostEngine::TPC] = numOfTPCPrefixOps;
        m_enginesPrefixCount[CostEngine::DMA] = numOfDMAPrefixOps;
    }

    // Accumulate the cost of the next operation.
    // Set aside interesting individual costs like the first/last operation(s).
    void accumulate(const pNode& node,
                    const SliceReferenceList& inputs,
                    const SliceReferenceList& outputs)
    {
        CostModel::Cost operationCost = m_costModel.calcCost(node, inputs, outputs);
        addCost(operationCost);
    }

    void addCost(const CostModel::Cost& operationCost)
    {
        AccumulatedCost& accumulatedCost = getEngineAccumulatedCost(operationCost.engine);

        accumulatedCost.total += operationCost;
        if (operationCost.timeNano > 0)  // Count first/last "real" operation - with time > 0
        {
            accumulatedCost.suffix = operationCost;

            if ((m_enginesPrefixCount.count(operationCost.engine) > 0) &&
                (m_enginesPrefixCount[operationCost.engine] > 0))
            {
                accumulatedCost.prefix += operationCost;
                m_enginesPrefixCount[operationCost.engine]--;
            }
        }
    }

    const AccumulatedCost& operator[](const CostEngine engine)
    {
        return getEngineAccumulatedCost(engine);
    }

private:

    AccumulatedCost& getEngineAccumulatedCost(CostEngine engine)
    {
        auto iter = m_enginesCost.find(engine);
        if (iter == m_enginesCost.end())
        {
            iter = m_enginesCost.emplace(engine, AccumulatedCost(engine)).first;
        }
        return iter->second;
    }

    std::unordered_map<CostEngine, AccumulatedCost, CostEngineHasher> m_enginesCost;
    std::unordered_map<CostEngine, unsigned, CostEngineHasher>        m_enginesPrefixCount;
    CostModel& m_costModel;
};

// OperationHandler object to aggregate the cost of each individual operation.
// Implement the interface for the HandleEachStrategyOperation functor that
// should feed it the operations in the bundle execution order
class StrategyCostAccumulator : public OperationHandler
{
public:
    StrategyCostAccumulator(CostModelPerEngineAccumulator& mmeCostAccumulator,
                            CostModelPerEngineAccumulator& tpcCostAccumulator,
                            CostModelPerEngineAccumulator& fetchCostAccumulator,
                            CostModelPerEngineAccumulator& evictCostAccumulator)
    : m_mmeCostAccumulator(mmeCostAccumulator)
    , m_tpcCostAccumulator(tpcCostAccumulator)
    , m_fetchCostAccumulator(fetchCostAccumulator)
    , m_evictCostAccumulator(evictCostAccumulator)
    {}

    void handleOperation(const pNode& node,
                         const SliceReferenceList& inputs,
                         const SliceReferenceList& outputs) override
    {
        if (HabanaGraph::runsOnMME(node))
        {
            m_mmeCostAccumulator.accumulate(node, inputs, outputs);
        }
        else if (HabanaGraph::runsOnTPC(node))
        {
            m_tpcCostAccumulator.accumulate(node, inputs, outputs);
        }
        m_fetchCostAccumulator.accumulate(node, inputs, outputs);
        m_evictCostAccumulator.accumulate(node, inputs, outputs);
    }

private:
    CostModelPerEngineAccumulator& m_mmeCostAccumulator;
    CostModelPerEngineAccumulator& m_tpcCostAccumulator;
    CostModelPerEngineAccumulator& m_fetchCostAccumulator;
    CostModelPerEngineAccumulator& m_evictCostAccumulator;
};

thread_local std::map<NodeSet, gaudi::CostModel::Cost> StrategyCostModel::s_invalidCandidatesCache = {};

const AccumulatedCost& StrategyCostModel::getMmeCost() const
{
    HB_ASSERT(m_modeled, "Trying to get cost before invoking model()");
    return m_mmeAccumulatedCost;
}

const AccumulatedCost& StrategyCostModel::getTpcCost() const
{
    HB_ASSERT(m_modeled, "Trying to get cost before invoking model()");
    return m_tpcAccumulatedCost;
}

const AccumulatedCost& StrategyCostModel::getFetchCost(CostEngine engine) const
{
    HB_ASSERT(m_modeled, "Trying to get cost before invoking model()");
    HB_ASSERT(engine == CostEngine::DMA, "Currently, fetch operation can only be performed by DMA");
    return m_dmaFetchAccumulatedCost;
}

const AccumulatedCost& StrategyCostModel::getEvictionCost(CostEngine engine) const
{
    HB_ASSERT(m_modeled, "Trying to get cost before invoking model()");
    switch (engine)
    {
    case CostEngine::DMA:
        return m_dmaEvictAccumulatedCost;
    case CostEngine::TPC:
        return m_tpcEvictAccumulatedCost;
    default:
        HB_ASSERT(false, "Currently eviction can only be performed by DMA or TPC");
    }
    HB_ASSERT(false, "Currently eviction can only be performed by DMA or TPC");
    return *(new AccumulatedCost(engine)); // this line is not reachable, needed for compilation
}

const CostModel::Cost& StrategyCostModel::getAggregatedCost() const
{
    HB_ASSERT(m_modeled, "Trying to get cost before invoking model()");
    return m_aggregatedCost;
}

const CostModel::Cost& StrategyCostModel::getInvalidCandidateCost(const pBundleExpansion& candidate) const
{
    HB_ASSERT(m_modeled, "Trying to get cost before invoking model()");
    return m_invalidCandidatesCosts.at(candidate->nodeToStitch);
}

const std::map<NodeSet, gaudi::CostModel::Cost>& StrategyCostModel::getInvalidCandidatesCache()
{
    return s_invalidCandidatesCache;
}

void StrategyCostModel::reset()
{
    s_invalidCandidatesCache.clear();
}

void StrategyCostModel::model(CostModel& mmeCostModel,
                              CostModel& tpcCostModel,
                              CostModel& fetchCostModel,
                              CostModel& evictCostModel)
{
    HB_ASSERT(!m_modeled, "Duplicate invokation of model() for the same strategy cost model");
    accumulateIndividualEngineCosts(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);
    modelInvalidCandidates(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);
    const BundleExecutionType executionType = aggregateCosts();
    m_modeled = true;
    printStrategyCostLog(executionType);
}

void StrategyCostModel::printStrategyCostLog(const BundleExecutionType& executionType) const
{
    HB_ASSERT(m_modeled, "Trying to print strategy cost log before invoking model()");
    if (LOG_LEVEL_AT_LEAST_TRACE(COST_MODEL))
    {
        LOG_TRACE(COST_MODEL, "Cost-Model for Bundle {} Strategy:", m_bundle->index());
        m_strategy->printLog(0, synapse::LogManager::LogType::COST_MODEL);
        LOG_TRACE(COST_MODEL, "MME Cost        - {}", m_mmeAccumulatedCost.toString());
        LOG_TRACE(COST_MODEL, "TPC Cost        - {}", m_tpcAccumulatedCost.toString());
        LOG_TRACE(COST_MODEL, "TPC Evict Cost  - {}", m_tpcEvictAccumulatedCost.toString());
        LOG_TRACE(COST_MODEL, "DMA Fetch Cost  - {}", m_dmaFetchAccumulatedCost.toString());
        LOG_TRACE(COST_MODEL, "DMA Evict Cost  - {}", m_dmaEvictAccumulatedCost.toString());
        for (const auto& invalidCandidateCost : m_invalidCandidatesCosts)
        {
            LOG_TRACE(COST_MODEL,
                      "Invalid candidate node {} Cost - {}",
                      invalidCandidateCost.first->getNodeName(),
                      invalidCandidateCost.second.toString());
        }
        LOG_TRACE(COST_MODEL,
                  "Bundle Execution Type: {}, Aggregated Cost - {}",
                  (unsigned)executionType,
                  getAggregatedCost().toString());
    }
}

void StrategyCostModel::accumulateIndividualEngineCosts(CostModel& mmeCostModel,
                                                        CostModel& tpcCostModel,
                                                        CostModel& fetchCostModel,
                                                        CostModel& evictCostModel)
{
    unsigned numOfTpcPrefixOps = 1;
    if (m_strategy->getMmeSlicingData().hasRole(BundleExpansion::WideInputProducer) &&
        m_strategy->getMmeSlicingData().hasRole(BundleExpansion::NarrowInputProducer))
    {
        // In case we have 2 TPC producers we need to add 2 prefix operations.
        numOfTpcPrefixOps = 2;
    }

    CostModelPerEngineAccumulator mmeCostAccumulator(mmeCostModel);
    CostModelPerEngineAccumulator tpcCostAccumulator(tpcCostModel, 1, numOfTpcPrefixOps, 1);
    CostModelPerEngineAccumulator fetchCostAccumulator(fetchCostModel);
    CostModelPerEngineAccumulator evictCostAccumulator(evictCostModel);

    const auto&                 strategyNodes = m_strategy->getSlicingData().getStrategyNodes(m_bundle);
    HandleEachStrategyOperation handleEachStrategyOperation(m_graph,
                                                            m_bundle->index(),
                                                            NodeVector(strategyNodes.begin(), strategyNodes.end()),
                                                            m_strategy);
    StrategyCostAccumulator strategyCostAccumulator(mmeCostAccumulator,
                                                    tpcCostAccumulator,
                                                    fetchCostAccumulator,
                                                    evictCostAccumulator);
    handleEachStrategyOperation(strategyCostAccumulator);
    m_mmeAccumulatedCost = mmeCostAccumulator[CostEngine::MME];
    m_tpcAccumulatedCost = tpcCostAccumulator[CostEngine::TPC];
    m_tpcEvictAccumulatedCost = evictCostAccumulator[CostEngine::TPC];
    m_dmaFetchAccumulatedCost = fetchCostAccumulator[CostEngine::DMA];
    m_dmaEvictAccumulatedCost = evictCostAccumulator[CostEngine::DMA];
}

void StrategyCostModel::modelInvalidCandidates(CostModel& mmeCostModel,
                                               CostModel& tpcCostModel,
                                               CostModel& fetchCostModel,
                                               CostModel& evictCostModel)
{
    std::unordered_set<pBundleExpansion>        tpcMasterInvalidCandidates;
    std::unordered_set<pBundleExpansion>        tpcSlaveInvalidCandidates;
    std::unordered_map<pNode, pBundleExpansion> mmeSlaveInvalidCandidates;  // slave mme node -> expansion candidate

    // Map from slave MME candidate to its slave TPC expansions
    std::unordered_map<pBundleExpansion, std::list<pBundleExpansion>> slaveExpansions;

    for (const pBundleExpansion& invalidCandidate : m_strategy->getMmeSlicingData().getInvalidCandidates())
    {
        HB_ASSERT_PTR(invalidCandidate->nodeToStitch);
        switch (invalidCandidate->role)
        {
        case BundleExpansion::Role::WideInputProducer:
        case BundleExpansion::Role::NarrowInputProducer:
        case BundleExpansion::Role::OutputConsumer:
            tpcMasterInvalidCandidates.insert(invalidCandidate);
            break;
        case BundleExpansion::Role::SharedInputConsumer:
            mmeSlaveInvalidCandidates[invalidCandidate->nodeToStitch] = invalidCandidate;
            slaveExpansions[invalidCandidate]                         = {};
            break;
        case BundleExpansion::Role::SlaveInputProducer:
        case BundleExpansion::Role::SlaveOutputConsumer:
            tpcSlaveInvalidCandidates.insert(invalidCandidate);
            break;
        default:
            HB_ASSERT(false, "Unhandled invalid candidate role");
        }
    }

    // Free TPC candidates (not stitched to a slave MME node)
    std::unordered_set<pBundleExpansion> freeTpcExpansions(tpcMasterInvalidCandidates.begin(),
                                                           tpcMasterInvalidCandidates.end());

    for (const auto& tpcSlaveInvalidCandidate : tpcSlaveInvalidCandidates)
    {
        if (mmeSlaveInvalidCandidates.count(tpcSlaveInvalidCandidate->bundleNode) > 0)
        {
            // The TPC slave node should be stitched to a slave MME node
            slaveExpansions.at(mmeSlaveInvalidCandidates.at(tpcSlaveInvalidCandidate->bundleNode))
                .push_back(tpcSlaveInvalidCandidate);
        }
        else
        {
            // The TPC slave node should not be stitched to a slave MME node (the slave MME is a valid candidate)
            freeTpcExpansions.insert(tpcSlaveInvalidCandidate);
        }
    }

    for (const pBundleExpansion& invalidCandidate : freeTpcExpansions)
    {
        modelTpcCandidate(invalidCandidate, tpcCostModel);
    }
    for (const auto& slaveExpansion : slaveExpansions)
    {
        modelSlaveMmeCandidates(slaveExpansion.first,
                                slaveExpansion.second,
                                mmeCostModel,
                                tpcCostModel,
                                fetchCostModel,
                                evictCostModel);
    }
}

void StrategyCostModel::modelSlaveMmeCandidates(const pBundleExpansion&            slaveMme,
                                                const std::list<pBundleExpansion>& slaveTpcCandidates,
                                                CostModel&                         mmeCostModel,
                                                CostModel&                         tpcCostModel,
                                                CostModel&                         fetchCostModel,
                                                CostModel&                         evictCostModel)
{
    // The cost of the slave TPC candidates will be added to the slave MME candidate
    // and saved in the slaveMme entry in the m_invalidCandidatesCosts map.
    if (m_invalidCandidatesCosts.count(slaveMme->nodeToStitch) > 0)
    {
        // Already handled and inserted to m_invalidCandidatesCosts.
        return;
    }

    HB_ASSERT(slaveMme->winningStrategyForSlaveBundle,
              "Cost model for bundle {}: Missing winning strategy for slave bundle",
              m_bundle->index());
    const auto& cost = slaveMme->winningStrategyForSlaveBundle->getCost();
    HB_ASSERT(cost.is_set(),
              "Cost model for bundle {}: Missing cost for slave bundle winning strategy",
              m_bundle->index());

    m_invalidCandidatesCosts.insert({slaveMme->nodeToStitch, cost.value()});
}

void StrategyCostModel::modelTpcCandidate(const pBundleExpansion& candidate, CostModel& costModel)
{
    if (m_invalidCandidatesCosts.count(candidate->nodeToStitch) > 0)
    {
        // Already handled and inserted to m_invalidCandidatesCosts.
        return;
    }

    // Check if the invalid node already exists in the invalid candidates cache
    NodeSet invalidNodes = {candidate->nodeToStitch};
    if (s_invalidCandidatesCache.count(invalidNodes) > 0)
    {
        m_invalidCandidatesCosts.insert({candidate->nodeToStitch, s_invalidCandidatesCache.at(invalidNodes)});
        return;
    }

    // TODO: Modeling through slice reference here is a waste of time. Need to implement another interface in the cost
    //       models that receive just a simple node and calculate its cost.
    SliceReferenceList inputs;
    for (const pTensor& input : candidate->nodeToStitch->getInputs())
    {
        pSlicedOperand slicedInput = std::make_shared<Bundle::Solution::SlicedOperand>(input);
        pSliceReference inputSlice = std::make_shared<SliceReference>(slicedInput);
        inputs.push_back(inputSlice);
    }

    SliceReferenceList outputs;
    for (const pTensor& output : candidate->nodeToStitch->getOutputs())
    {
        pSlicedOperand slicedOutput = std::make_shared<Bundle::Solution::SlicedOperand>(output);
        pSliceReference outputSlice = std::make_shared<SliceReference>(slicedOutput);
        outputs.push_back(outputSlice);
    }

    const auto& cost = costModel.calcCost(candidate->nodeToStitch, inputs, outputs);
    m_invalidCandidatesCosts.insert({candidate->nodeToStitch, cost});

    s_invalidCandidatesCache.insert({invalidNodes, cost});
}

// Calculate the overall cost of executing the bundle with the given strategy
StrategyCostModel::BundleExecutionType StrategyCostModel::aggregateCosts()
{
    aggregateHbmTraffic();
    BundleExecutionType executionType = evaluateExecutionType();
    switch (executionType)
    {
    case BundleExecutionType::MMEComputeBound:
        aggregateMmeComputeBoundTime();
        break;
    case BundleExecutionType::TPCComputeBound:
        aggregateTpcComputeBoundTime();
        break;
    default:
        HB_ASSERT(false, "Unhandled bundle execution type");
    }
    fixTimeForBwBoundExecution();
    aggregateSlicingOverheads();
    aggregateInvalidCandidatesCost();
    return executionType;
}

void StrategyCostModel::aggregateHbmTraffic()
{
    m_aggregatedCost.hbmTrafficBytes = m_mmeAccumulatedCost.total.hbmTrafficBytes +
                                       m_tpcAccumulatedCost.total.hbmTrafficBytes +
                                       m_tpcEvictAccumulatedCost.total.hbmTrafficBytes +
                                       m_dmaFetchAccumulatedCost.total.hbmTrafficBytes +
                                       m_dmaEvictAccumulatedCost.total.hbmTrafficBytes;
}

bool StrategyCostModel::isOperandInSram(const pSlicedOperand& slicedOperand)
{
    return slicedOperand->resideInSRAM || slicedOperand->originalTensor->inSram();
}

// Compute bound MME execution time is the sum of the MME compute with the time to fetch/produce the first inputs and
// when there is a consumer, the time it takes to evict the last output.
void StrategyCostModel::aggregateMmeComputeBoundTime()
{
    m_aggregatedCost.timeNano = m_mmeAccumulatedCost.total.timeNano;

    const auto& slicingData = m_strategy->getMmeSlicingData();

    addPrefixForMmeComputeBoundTime(slicingData);

    // If needed add suffix
    const auto& slaveOperand = slicingData.getSlaveOutputOperand();
    if (isOperandInSram(slicingData.masterOperand) || (slaveOperand && isOperandInSram(slaveOperand)))
    {
        addSuffixForMmeComputeBoundTime(slicingData);
    }
}

void StrategyCostModel::addPrefixForMmeComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData)
{
    if (slicingData.hasRole(BundleExpansion::WideInputProducer) ||
        slicingData.hasRole(BundleExpansion::NarrowInputProducer) ||
        slicingData.hasRole(BundleExpansion::SlaveInputProducer))  // At least one TPC producer
    {
        if (m_dmaFetchAccumulatedCost.prefix.timeNano > 0)  // TPC + DMA fetch
        {
            // TPC producer is present. The prefix is an aggregation of the first TPC operation with the first DMA
            m_aggregatedCost.timeNano +=
                aggregateTpcDmaParallelTimeNano(m_tpcAccumulatedCost.prefix, m_dmaFetchAccumulatedCost.prefix);
        }
        else  // TPC only, no DMA fetch
        {
            m_aggregatedCost.timeNano += m_tpcAccumulatedCost.prefix.timeNano;
        }
    }
    else  // No TPC producers
    {
        addPrefixForMmeWithoutTpcProducer();
    }
}

void StrategyCostModel::addPrefixForMmeWithoutTpcProducer()
{
    m_aggregatedCost.timeNano += m_dmaFetchAccumulatedCost.prefix.timeNano;
}

void StrategyCostModel::addSuffixForMmeComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData)
{
    const pSlicedOperand& masterOutput = m_strategy->getMmeSlicingData().masterOperand;
    const pSlicedOperand& slaveOutput  = m_strategy->getMmeSlicingData().getSlaveOutputOperand();
    if (slicingData.hasRole(BundleExpansion::OutputConsumer) ||
        slicingData.hasRole(BundleExpansion::SlaveOutputConsumer))
    {
        // TPC consumer is present, the suffix is an aggregation of the last TPC consume and eviction (if needed)
        if (isEvicted(masterOutput) || (slaveOutput && isEvicted(slaveOutput)))
        {
            m_aggregatedCost.timeNano += aggregateTpcDmaParallelTimeNano(m_tpcAccumulatedCost.suffix,
                                                                         m_dmaEvictAccumulatedCost.suffix);
        }
        else
        {
            m_aggregatedCost.timeNano += m_tpcAccumulatedCost.suffix.timeNano;
        }
    }
    else
    {
        if (isEvictedByTpc(masterOutput) || (slaveOutput && isEvictedByTpc(slaveOutput)))
        {
            m_aggregatedCost.timeNano += m_tpcEvictAccumulatedCost.suffix.timeNano;
        }
        else
        {
            m_aggregatedCost.timeNano += m_dmaEvictAccumulatedCost.suffix.timeNano;
        }
    }
}

bool StrategyCostModel::isEvicted(const pSlicedOperand& slicedOperand) const
{
    if (!slicedOperand->resideInSRAM  && !slicedOperand->originalTensor->inSram())
    {
        // Non-SRAM operands do not need eviction
        return false;
    }

    if (slicedOperand->originalTensor->isUserManagedDram())
    {
        // Persistent operands always need eviction
        return true;
    }

    const NodeSet& bundleNodes = m_strategy->getSlicingData().getStrategyNodes(m_bundle);

    for (const pNode& consumer : m_graph.getTensorConsumers(slicedOperand->originalTensor))
    {
        if (bundleNodes.count(consumer) == 0)
        {
            // Any consumer that is not in the strategy nodes means an eviction is needed.
            return true;
        }
    }

    return false;
}

bool StrategyCostModel::isEvictedByTpc(const pSlicedOperand& slicedOperand)
{
    return slicedOperand->finalElementType != slicedOperand->originalTensor->getElementType();
}

// When TPC and DMA are operating in parallel, the cost models do not take into account that the
// HBM BW is shared between them. So the aggregated time of their parallel execution may be longer
// than the maximal time of one of them.
uint64_t StrategyCostModel::aggregateTpcDmaParallelTimeNano(const CostModel::Cost& tpcCost,
                                                            const CostModel::Cost& dmaCost) const
{
    uint64_t traffic = tpcCost.hbmTrafficBytes + dmaCost.hbmTrafficBytes;
    uint64_t dataMovementTime = dataMovementTimeNano(traffic);
    return std::max(dataMovementTime, std::max(tpcCost.timeNano, dmaCost.timeNano));
}

void StrategyCostModel::aggregateTpcComputeBoundTime()
{
    m_aggregatedCost.timeNano = m_tpcAccumulatedCost.total.timeNano;

    const auto& slicingData = m_strategy->getMmeSlicingData();

    addPrefixForTpcComputeBoundTime(slicingData);
    addSuffixForTpcComputeBoundTime(slicingData);
}

void StrategyCostModel::addSuffixForTpcComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData)
{
    // TODO: Support more cases: The TPC producer output may be evicted, the MME may write to SRAM and this may
    //       require eviction as well.
    if (!slicingData.hasRole(BundleExpansion::OutputConsumer) &&
        !slicingData.hasRole(BundleExpansion::SlaveOutputConsumer))
    {
        // Producer only - the mme is the last node at the bundle, so the suffix is taken from it.
        m_aggregatedCost.timeNano += m_mmeAccumulatedCost.suffix.timeNano;
    }
}

void StrategyCostModel::addPrefixForTpcComputeBoundTime(const MmeSlicingStrategy::MmeSlicingData& slicingData)
{
    if (!slicingData.hasRole(BundleExpansion::WideInputProducer) &&
        !slicingData.hasRole(BundleExpansion::NarrowInputProducer))
    {
        // Consumer only - need to add prefix - the first MME operation and the DMA required to start it.
        m_aggregatedCost.timeNano += m_mmeAccumulatedCost.prefix.timeNano;
        addPrefixForMmeWithoutTpcProducer();
    }
}

void StrategyCostModel::fixTimeForBwBoundExecution()
{
    // Update the aggregated cost for strategies that are BW bound.
    m_aggregatedCost.timeNano =
        std::max(m_aggregatedCost.timeNano, dataMovementTimeNano(m_aggregatedCost.hbmTrafficBytes));
}

void StrategyCostModel::aggregateSlicingOverheads()
{
    unsigned numBuffers =
        (!SlicedOperandUtils::isTriviallySliced(*m_strategy) && m_strategy->sramSlicedOperandsDoubleBuffered()) ? 2 : 1;

    // TODO: SW-48303 - improve overhead aggregation
    unsigned numOfSlicesFactor = 0;
    for (const auto& operand : m_strategy->getSlicingData().getSlicedOperands())
    {
        numOfSlicesFactor = std::max(numOfSlicesFactor, SlicedOperandUtils::nofSlices(operand));
    }

    // Add overhead per slice - consists of read/write latency, sync overhead.
    m_aggregatedCost.timeNano +=
        ((numOfSlicesFactor / (double)numBuffers * GCFG_SRAM_SLICER_COST_MODEL_OVERHEAD_PER_SLICE.value()) /
         SlicingBrain::knobs.freqGHz);
}

void StrategyCostModel::aggregateInvalidCandidatesCost()
{
    for (const auto& invalidCandidate : m_invalidCandidatesCosts)
    {
        // Make sure the node is not stitched as valid candidate in other place.
        if (!m_strategy->getMmeSlicingData().isNodeStitched(invalidCandidate.first))
        {
            m_aggregatedCost.hbmTrafficBytes += invalidCandidate.second.hbmTrafficBytes;
            m_aggregatedCost.timeNano += invalidCandidate.second.timeNano;
        }
    }
}

// Classify the strategy proposed solution to a specific aggregation class, according to the dominant time consumer
// of the bundle execution with the given strategy.
StrategyCostModel::BundleExecutionType StrategyCostModel::evaluateExecutionType()
{
    if (m_tpcAccumulatedCost.total.timeNano > m_mmeAccumulatedCost.total.timeNano)
    {
        return BundleExecutionType::TPCComputeBound;
    }
    return BundleExecutionType::MMEComputeBound;
}

// Net time to move the data from/to/within the HBM.
uint64_t StrategyCostModel::dataMovementTimeNano(uint64_t trafficBytes)
{
    // Currently not taking alignment or striding issues into consideration.
    return trafficBytes / SlicingBrain::knobs.hbmAvailableBWGBps;
}

} // namespace gaudi
