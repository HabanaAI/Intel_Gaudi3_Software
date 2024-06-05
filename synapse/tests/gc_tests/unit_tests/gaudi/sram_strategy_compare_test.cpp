#include "passes/sram_management/bundle_expander.h"
#include "passes/sram_management/spatial_slicing_solver.h"
#include "passes/sram_management/sram_management.h"
#include "sram_management_fe_test.h"
#include <passes/sram_management/strategy_cost_model.h>
#include <passes/sram_management/engine_cost_model.h>
#include "platform/gaudi/graph_compiler/passes.h"
#include "tpc_slicing_test_infra.h"

namespace gaudi
{

class SRAMStrategyCompareTest : public SRAMManagementTest
{
    using BaseClass = SRAMManagementTest;
public:
    void SetUp() override
    {
        SRAMManagementTest::SetUp();
        setGlobalConfForTest(GCFG_SRAM_SLICER_COST_MODEL_OVERHEAD_PER_SLICE, "0");
        setGlobalConfForTest(GCFG_SRAM_SLICER_COST_MODEL_ENABLED, "true");
        StrategyCostModel::reset();  // Clear the invalid candidates cache
    }

    static const uint64_t mmeOpTime;
    static const uint64_t tpcOpTime;
    static const uint64_t fetchOpTime;
    static const uint64_t evictOpTime;
    static const uint64_t mmeTraffic;
    static const uint64_t tpcTraffic;
    static const uint64_t fetchTraffic;
    static const uint64_t evictTraffic;

    using CostEngine = CostModel::Cost::Engine;

    // Test controllable cost model. Returns a pre-determined cost and count calls to calcCost.
    class MockCostModel : public gaudi::CostModel
    {
    public:
        MockCostModel(Cost::Engine engine, uint64_t time, uint64_t traffic)
        : m_cost(engine)
        {
            m_cost.hbmTrafficBytes = traffic;
            m_cost.timeNano = time;
        }
        Cost calcCost(const pNode& node,
                      const SliceReferenceList& inputs,
                      const SliceReferenceList& outputs) const override
        {
            numOfCalls++;
            return m_cost;
        }
        mutable unsigned numOfCalls = 0;
        Cost m_cost;
    };

    MockCostModel mmeCostModel{CostEngine::MME, mmeOpTime, mmeTraffic};
    MockCostModel tpcCostModel{CostEngine::TPC, tpcOpTime, tpcTraffic};
    MockCostModel fetchCostModel{CostEngine::DMA, fetchOpTime, fetchTraffic};
    MockCostModel evictCostModel{CostEngine::DMA, evictOpTime, evictTraffic};

    pTensor                     m_inputA;
    pTensor                     m_inputB;
    pTensor                     m_output;
    std::shared_ptr<Bundlizer>  m_bundlizer;
    pBundle                     m_bundle;
    pMmeSlicingStrategy            m_strategy;

    static constexpr unsigned expNumSlices = 4;

    void createSlicedGemmBundle(bool persistentOutput = false)
    {
        // Given a single MME bundle which is sliced to 4 operations:
        constexpr TSize
                w = 1024,
                k = 256,
                c = 128;

        m_inputA = createTensor({c, w}, syn_type_bf16, false);
        m_inputB = createTensor({k, c}, syn_type_bf16, false);
        if (persistentOutput)
        {
            m_output = createTensor({k, w}, syn_type_bf16);
        }
        else
        {
            TSize shape[] = {k, w};
            m_output = std::make_shared<Tensor>(ARRAY_SIZE(shape), shape, syn_type_bf16);
        }

        synGEMMParams gemmParams{};
        pNode gemm = NodeFactory::createNode({m_inputA, m_inputB}, {m_output}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
        GraphEditor::addNode(getGraph(), gemm);

        m_bundlizer = std::make_shared<Bundlizer>(getGraph());
        m_bundle = m_bundlizer->getMMEBundles().front();
        ASSERT_EQ(m_bundle->getNodes().size(), 1);

        m_strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), gemm);
        auto& sd = m_strategy->getMmeSlicingData();
        sd.bundleTensors[0]->chunkDimensions[DIM_W] = sd.masterOperand->chunkDimensions[DIM_W] = w / expNumSlices;
    }

    pBundleExpansion
    addSlaveMME(const pTensor& sharedInput, const pTensor& nonSharedInput, bool stitchedToStrategy, bool isValid = true)
    {
        pTensor slaveOutput         = m_output->clone(false, false);
        pTensor slaveNonSharedInput = nonSharedInput->clone(false, false);

        synGEMMParams gemmParams {};
        pNode         slaveMme = NodeFactory::createNode({sharedInput, slaveNonSharedInput},
                                                 {slaveOutput},
                                                 &gemmParams,
                                                 NodeFactory::gemmNodeTypeName,
                                                 "GEMM_slave");
        GraphEditor::addNode(getGraph(), slaveMme);

        if (stitchedToStrategy)
        {
            MMESlaveBrain                  slaveBrain {getGraph()};
            SharedMMEInputCandidateHandler handler;
            std::list<pBundleExpansion>    candidates = handler.findSharedMMEInputCandidate(m_strategy, getGraph());
            EXPECT_FALSE(candidates.empty());
            auto& candidate = candidates.front();
            EXPECT_EQ(candidate->nodeToStitch, slaveMme);
            auto adjustedCandidate = slaveBrain.adjustCandidateToStrategy(candidate, m_strategy);
            if (isValid)
            {
                m_strategy->getMmeSlicingData().addValidCandidate(adjustedCandidate, false);
                slaveBrain.addSharedOperandMme(adjustedCandidate, m_strategy);
            }
            else
            {
                m_strategy->getMmeSlicingData().addInvalidCandidate(adjustedCandidate);
            }
            return adjustedCandidate;
        }
        return nullptr;
    }

    void addSlaveTpcProducer(bool stitchedToStrategy)
    {
        const pBundleExpansion& mmeSlaveCandidate =
            m_strategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::Role::SharedInputConsumer];
        ASSERT_NE(mmeSlaveCandidate, nullptr);
        const pTensor& nonSharedInput = mmeSlaveCandidate->slaveOperands.getInput()->originalTensor;

        pTensor producerInput = nonSharedInput->clone(false, false);
        // Create a sliceable TPC producer node (slicing granularity on each dim is 1)
        pNode producer = TPCCustomIndexSpaceNode::createSliceableNode(producerInput, nonSharedInput);
        GraphEditor::addNode(getGraph(), producer);

        ASSERT_TRUE(loadTpcKernels(getGraph()));

        if (stitchedToStrategy)
        {
            pBundleExpansion candidate = m_bundlizer->findSlaveTpcProducerExpansionCandidate(m_strategy);
            ASSERT_NE(candidate, nullptr);
            ASSERT_EQ(candidate->nodeToStitch, producer);

            TPCSlaveBrain slaveBrain(getGraph());
            m_strategy->getMmeSlicingData().addValidCandidate(candidate);
            slaveBrain.addProducerToStrategy(candidate, m_strategy);
        }
    }

    void addSlaveTpcConsumer(bool stitchedToStrategy)
    {
        const pBundleExpansion& mmeSlaveCandidate =
            m_strategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::Role::SharedInputConsumer];
        ASSERT_NE(mmeSlaveCandidate, nullptr);
        const pTensor& slaveOutput = mmeSlaveCandidate->slaveOperands.getOutput()->originalTensor;

        pTensor consumerOutput = slaveOutput->clone(false, false);
        // Create a sliceable TPC consumer node (slicing granularity on each dim is 1)
        pNode consumer = TPCCustomIndexSpaceNode::createSliceableNode(slaveOutput, consumerOutput);
        GraphEditor::addNode(getGraph(), consumer);

        ASSERT_TRUE(loadTpcKernels(getGraph()));

        if (stitchedToStrategy)
        {
            pBundleExpansion candidate =
                m_bundlizer->findTpcConsumerExpansionCandidate(m_strategy,
                                                               ExpansionCandidatesSet(),
                                                               BundleExpansion::SlaveOutputConsumer);
            ASSERT_NE(candidate, nullptr);
            ASSERT_EQ(candidate->nodeToStitch, consumer);

            TPCSlaveBrain slaveBrain(getGraph());
            m_strategy->getMmeSlicingData().addValidCandidate(candidate);
            slaveBrain.addConsumerToStrategy(candidate, m_strategy);
        }
    }

    void addTpcProducer(const pTensor& produced, bool stitchedToStrategy, bool isWide = true)
    {
        pTensor producerInput = produced->clone(false, false);

        // Create a sliceable TPC producer node (slicing granularity on each dim is 1)
        pNode producer = TPCCustomIndexSpaceNode::createSliceableNode(producerInput, produced);
        GraphEditor::addNode(getGraph(), producer);

        ASSERT_TRUE(loadTpcKernels(getGraph()));

        if (stitchedToStrategy)
        {
            pBundleExpansion candidate;
            if (isWide)
            {
                candidate = m_bundlizer->findWideTpcProducerExpansionCandidate(m_strategy);
            }
            else
            {
                candidate = m_bundlizer->findNarrowTpcProducerExpansionCandidate(m_strategy);
            }

            ASSERT_NE(candidate, nullptr);
            ASSERT_EQ(candidate->nodeToStitch, producer);

            TPCSlaveBrain slaveBrain(getGraph());
            m_strategy->getMmeSlicingData().addValidCandidate(candidate);
            slaveBrain.addProducerToStrategy(candidate, m_strategy);
        }
    }

    pBundleExpansion addInvalidTpcProducer(const pTensor& produced, bool isWide = true)
    {
        pTensor producerInput = produced->clone(false, false);
        // Create a sliceable TPC producer node (slicing granularity on each dim is 1)
        pNode producer = TPCCustomIndexSpaceNode::createSliceableNode(producerInput, produced);
        GraphEditor::addNode(getGraph(), producer);

        EXPECT_TRUE(loadTpcKernels(getGraph()));

        pBundleExpansion candidate;
        if (isWide)
        {
            candidate = m_bundlizer->findWideTpcProducerExpansionCandidate(m_strategy);
        }
        else
        {
            candidate = m_bundlizer->findNarrowTpcProducerExpansionCandidate(m_strategy);
        }
        if (candidate) m_strategy->getMmeSlicingData().addInvalidCandidate(candidate);
        return candidate;
    }

    void addTpcConsumer(const pTensor& consumed, bool stitchedToStrategy)
    {
        pTensor consumerOutput = consumed->clone(false, false);
        // Create a sliceable TPC consumer node (slicing granularity on each dim is 1)
        pNode consumer = TPCCustomIndexSpaceNode::createSliceableNode(consumed, consumerOutput);
        GraphEditor::addNode(getGraph(), consumer);

        ASSERT_TRUE(loadTpcKernels(getGraph()));

        if (stitchedToStrategy)
        {
            pBundleExpansion candidate = m_bundlizer->findTpcConsumerExpansionCandidate(m_strategy);
            ASSERT_NE(candidate, nullptr);
            ASSERT_EQ(candidate->nodeToStitch, consumer);

            TPCSlaveBrain slaveBrain(getGraph());
            m_strategy->getMmeSlicingData().addValidCandidate(candidate);
            slaveBrain.addConsumerToStrategy(candidate, m_strategy);
        }
    }

    pBundleExpansion addInvalidTpcConsumer(const pTensor& consumed)
    {
        pTensor consumerOutput = consumed->clone(false, false);
        // Create a sliceable TPC consumer node (slicing granularity on each dim is 1)
        pNode consumer = TPCCustomIndexSpaceNode::createSliceableNode(consumed, consumerOutput);
        GraphEditor::addNode(getGraph(), consumer);

        EXPECT_TRUE(loadTpcKernels(getGraph()));

        pBundleExpansion candidate = m_bundlizer->findTpcConsumerExpansionCandidate(m_strategy);
        if(candidate) m_strategy->getMmeSlicingData().addInvalidCandidate(candidate);
        return candidate;
    }

    static void validateMockCost(MockCostModel& costModel, const AccumulatedCost& accCost,
            unsigned expOpCount, uint64_t opTime, uint64_t opTraffoc)
    {
        ASSERT_EQ(costModel.numOfCalls, expOpCount);

        ASSERT_EQ(accCost.total.timeNano, expOpCount * opTime);
        ASSERT_EQ(accCost.total.hbmTrafficBytes, expOpCount * opTraffoc);

        // If no operations are expected, in particular, there will not be a prefix or suffix
        uint64_t expPrefixSuffixTime = expOpCount > 0 ? opTime : 0;
        ASSERT_EQ(accCost.prefix.timeNano, expPrefixSuffixTime);
        ASSERT_EQ(accCost.suffix.timeNano, expPrefixSuffixTime);

    }

    void validateStrategyMockCost(const StrategyCostModel& strategyCost,
                                  unsigned expectedMmeOpCount, unsigned expectedTpcOpCount)
    {
        validateStrategyMockCost(strategyCost, expectedMmeOpCount, expectedTpcOpCount,
                                 expectedMmeOpCount + expectedTpcOpCount,
                                 expectedMmeOpCount + expectedTpcOpCount);
    }

    void validateStrategyMockCost(const StrategyCostModel& strategyCost,
                                  unsigned expectedMmeOpCount,
                                  unsigned expectedTpcOpCount,
                                  unsigned expectedFetchOpCount,
                                  unsigned expectedEvictOpCount)
    {
        validateMockCost(mmeCostModel, strategyCost.getMmeCost(), expectedMmeOpCount, mmeOpTime, mmeTraffic);
        validateMockCost(tpcCostModel, strategyCost.getTpcCost(), expectedTpcOpCount, tpcOpTime, tpcTraffic);
        validateMockCost(fetchCostModel,
                         strategyCost.getFetchCost(CostEngine::DMA),
                         expectedFetchOpCount,
                         fetchOpTime,
                         fetchTraffic);
        validateMockCost(evictCostModel,
                         strategyCost.getEvictionCost(CostEngine::DMA),
                         expectedEvictOpCount,
                         evictOpTime,
                         evictTraffic);
    }

    void assertZeroAccumulatedCost(const AccumulatedCost& accCost)
    {
        assertZeroCost(accCost.prefix);
        assertZeroCost(accCost.suffix);
        assertZeroCost(accCost.total);
    }

    void assertZeroCost(const CostModel::Cost& cost)
    {
        ASSERT_EQ(0ull, cost.hbmTrafficBytes);
        ASSERT_EQ(0ull, cost.timeNano);
    }

};

// Out of line so they can be used in std::max (which require a reference to a physical variable, not compile time const)
// Primary numbers to try to avoid cases where a bug is hiding because, for example,  mme op time id 2*tpc op time and
// counted an mme operation instead of 2 tpc.
const uint64_t SRAMStrategyCompareTest::mmeOpTime       = 11;
const uint64_t SRAMStrategyCompareTest::tpcOpTime       =  7;
const uint64_t SRAMStrategyCompareTest::fetchOpTime     =  5;
const uint64_t SRAMStrategyCompareTest::evictOpTime     = 27;
const uint64_t SRAMStrategyCompareTest::mmeTraffic      = 17;
const uint64_t SRAMStrategyCompareTest::tpcTraffic      = 13;
const uint64_t SRAMStrategyCompareTest::fetchTraffic    = 23;
const uint64_t SRAMStrategyCompareTest::evictTraffic    = 29;


TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_should_throw_exception_for_query_before_model)
{
    // Given
    createSlicedGemmBundle();

    // Check exceptions:
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    EXPECT_HB_ASSERT(strategyCost.getMmeCost());
    EXPECT_HB_ASSERT(strategyCost.getTpcCost());
    EXPECT_HB_ASSERT(strategyCost.getFetchCost(CostEngine::DMA));
    EXPECT_HB_ASSERT(strategyCost.getEvictionCost(CostEngine::DMA));
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_should_accumulate_mme_operations)
{
    // Given
    createSlicedGemmBundle();

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Then
    validateStrategyMockCost(strategyCost, expNumSlices, 0); // No TPC expected
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_should_accumulate_slave_mme_operations)
{
    // Given
    createSlicedGemmBundle();
    addSlaveMME(m_inputA, m_inputB, true);

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Then
    validateStrategyMockCost(strategyCost, 2 * expNumSlices, 0);  // No TPC expected
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_should_accumulate_tpc_operations)
{
    // Given
    createSlicedGemmBundle();
    addTpcConsumer(m_output, true);

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Then
    validateStrategyMockCost(strategyCost, expNumSlices, expNumSlices);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_model_tpc_invalid_candidates)
{
    // Given
    createSlicedGemmBundle();
    pBundleExpansion consumerCandidate = addInvalidTpcConsumer(m_output);
    ASSERT_NE(consumerCandidate, nullptr);
    ASSERT_NE(consumerCandidate->nodeToStitch, nullptr);
    pBundleExpansion producerCandidate = addInvalidTpcProducer(m_inputA);
    ASSERT_NE(producerCandidate, nullptr);
    ASSERT_NE(producerCandidate->nodeToStitch, nullptr);

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Validate that the invalid candidates are added to the cache
    ASSERT_EQ(StrategyCostModel::getInvalidCandidatesCache().size(), 2);

    // Then validate that the TPC cost model is called only twice (once for each invalid candidate)
    ASSERT_EQ(tpcCostModel.numOfCalls, 2);
    ASSERT_EQ(strategyCost.getInvalidCandidateCost(consumerCandidate).timeNano, tpcOpTime);
    ASSERT_EQ(strategyCost.getInvalidCandidateCost(producerCandidate).timeNano, tpcOpTime);

    // Validate that the invalid candidates cache is restarted after reset
    StrategyCostModel::reset();
    ASSERT_TRUE(StrategyCostModel::getInvalidCandidatesCache().empty());
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_accumulate_tpc_operations_when_it_substitutes_dma)
{
    // Data movement may be done by TPC even if the bundle solution does not describe it.
    // For example, when the MME output type is FP32 due to partial slicing while the original
    // data type is BF16, a cast TPC node will be used instead of an evicting DMA.

    // Given
    createSlicedGemmBundle();

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcCostModel, fetchCostModel, tpcCostModel);  // Use TPC mock as evictor

    // Then
    // Eviction by TPC is accumulated in the eviction cost and is not counted in the scheduled TPC cost.
    validateMockCost(tpcCostModel, strategyCost.getEvictionCost(CostEngine::TPC), expNumSlices, tpcOpTime, tpcTraffic);
    assertZeroAccumulatedCost(strategyCost.getTpcCost());
}

// The cost model should aggregate the costs of MME, TPC and DMA to a single cost of the bundle execution
// using the supplied strategy. Aggregation method is described in the design doc
// https://habanalabs.sharepoint.com/:w:/g/EXR852YeHURLoGis_eQ0a1QBmjGA0ZIENmu1zZ0FpeU-dA?e=lE7Lt9

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__mme_compute_bound_no_tpc)
{
    // Given
    createSlicedGemmBundle();

    // And given mme cost model s.t the MME compute would dominate the time estimation
    constexpr uint64_t mmeBigOpTime = 997; // Keep it prime
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeBigOpTime, mmeTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Then, assert that the aggregated time is the time to compute the MME nodes + the prefix DMA time to fetch
    // the first wide and narrow slices.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, expNumSlices * mmeBigOpTime + fetchOpTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_should_aggregate_costs_with_overhead_per_slice)
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_COST_MODEL_OVERHEAD_PER_SLICE, "100");

    // Given
    createSlicedGemmBundle();
    m_strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);

    // And given mme cost model s.t the MME compute would dominate the time estimation
    constexpr uint64_t mmeBigOpTime = 997;  // Keep it prime
    MockCostModel      mmeHeavyCostModel {CostEngine::MME, mmeBigOpTime, mmeTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Then, assert that the aggregated time is the time to compute the MME nodes + the prefix DMA time to fetch
    // the first wide and narrow slices + slicing overheads.
    uint64_t               slicingOverheads = 100 * expNumSlices / SlicingBrain::knobs.freqGHz;
    const CostModel::Cost& aggregatedCost   = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, expNumSlices * mmeBigOpTime + fetchOpTime + slicingOverheads);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__mme_compute_bound_tpc_producer)
{
    // Given
    createSlicedGemmBundle();
    addTpcProducer(m_inputA, true);

    // And given mme cost model s.t the MME compute would dominate the time estimation
    constexpr uint64_t mmeBigOpTime = 997; // Keep it prime
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeBigOpTime, mmeTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Compute the expected prefix time. The prefix is composed of one TPC produced slice and one DMA fetched.
    // Both engines (TPC/DMA) can work in parallel, so the time is either the max between them, or the traffic time.
    uint64_t prefixTrafficTime = (tpcTraffic + fetchTraffic) / SlicingBrain::knobs.hbmAvailableBWGBps;
    uint64_t maxOpTime = std::max(tpcOpTime, fetchOpTime);
    uint64_t expPrefixTime = std::max(maxOpTime, prefixTrafficTime);

    // Then, assert that the aggregated time is the time to compute the MME nodes + prefix time.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, expNumSlices * mmeBigOpTime + expPrefixTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__mme_compute_bound_tpc_consumer)
{
    // Given
    createSlicedGemmBundle();
    addTpcConsumer(m_output, true);

    // And given mme cost model s.t the MME compute would dominate the time estimation
    constexpr uint64_t mmeBigOpTime = 997; // Keep it prime
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeBigOpTime, mmeTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    uint64_t prefixTime = fetchOpTime; // No producer - DMA as prefix
    uint64_t suffixTime = tpcOpTime; // TPC consumer - bundle has suffix time

    // Then, assert that the aggregated time is the time to compute the MME nodes + prefix time.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, prefixTime + expNumSlices * mmeBigOpTime + suffixTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__mme_compute_bound_output_in_sram)
{
    // Given
    createSlicedGemmBundle();
    m_strategy->setOutputIsInSRAM(true);

    // And given mme cost model s.t the MME compute would dominate the time estimation
    constexpr uint64_t mmeBigOpTime = 997; // Keep it prime
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeBigOpTime, mmeTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    uint64_t prefixTime = fetchOpTime; // No producer - DMA as prefix
    uint64_t suffixTime = evictOpTime; // output in SRAM - bundle has suffix time

    // Then, assert that the aggregated time is the time to compute the MME nodes + prefix time.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, prefixTime + expNumSlices * mmeBigOpTime + suffixTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__mme_compute_bound_consumer_and_evicted)
{
    // Given
    createSlicedGemmBundle();
    addTpcConsumer(m_output, true);
    addTpcConsumer(m_output, false); // Adding another consumer that is not in the bundle means that the output needs to be evicted.

    // given mme cost model s.t the MME compute would dominate the time estimation
    constexpr uint64_t mmeBigOpTime = 997; // Keep it prime
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeBigOpTime, mmeTraffic};

    // and given dma cost model s.t. the suffix time would be aggregated to it.
    constexpr uint64_t dmaBigOpTime = 199;
    MockCostModel dmaHeavyCostModel{CostEngine::DMA, dmaBigOpTime, fetchTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, dmaHeavyCostModel, dmaHeavyCostModel);

    uint64_t prefixTime = dmaBigOpTime; // No producer - DMA as prefix
    // Compute the expected suffix time. The suffix is composed of one TPC consumed slice and one DMA eviction.
    // Both engines (TPC/DMA) can work in parallel, so the time is either the max between them, or the traffic time.
    uint64_t suffixTrafficTime = (tpcTraffic + evictTraffic) / SlicingBrain::knobs.hbmAvailableBWGBps;
    uint64_t maxOpTime = std::max(tpcOpTime, dmaBigOpTime);
    uint64_t expSuffixTime = std::max(maxOpTime, suffixTrafficTime);

    // Then, assert that the aggregated time is the time to compute the MME nodes + prefix time.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, prefixTime + expNumSlices * mmeBigOpTime + expSuffixTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__mme_compute_bound_tpc_eviction)
{
    // Given
    createSlicedGemmBundle();
    m_strategy->setOutputIsInSRAM(true);
    m_strategy->getMmeSlicingData().masterOperand->finalElementType = syn_type_float;

    // given mme cost model s.t the MME compute would dominate the time estimation
    constexpr uint64_t mmeBigOpTime = 997; // Keep it prime
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeBigOpTime, mmeTraffic};
    constexpr uint64_t tpcEvictTime = 199;
    MockCostModel tpcHeavyEvictCostModel{CostEngine::TPC, tpcEvictTime, tpcTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    // Use the TPC eviction mock cost for eviction modeling
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, fetchCostModel, tpcHeavyEvictCostModel);

    // Then
    uint64_t expPrefixTime = fetchOpTime;
    uint64_t expSuffixTime = tpcEvictTime;
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, expPrefixTime + expNumSlices * mmeBigOpTime + expSuffixTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__hbm_bw_bound_no_tpc)
{
    // Given
    createSlicedGemmBundle();

    // And given mme cost model s.t the traffic would dominate the time estimation
    constexpr uint64_t bigMmeTraffic = 1009 * 1009; // "Almost" prime, still low chance of collision
    constexpr uint64_t bigDmaTraffic = 1013 * 1013; // "Almost" prime, still low chance of collision
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeOpTime, bigMmeTraffic};
    MockCostModel dmaHeavyCostModel{CostEngine::DMA, fetchOpTime, bigDmaTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcCostModel, dmaHeavyCostModel, fetchCostModel);

    // Then, assert that the aggregated time is the time to carry out all the data movement.
    uint64_t totalExpMMETraffic = expNumSlices * bigMmeTraffic;
    uint64_t totalExpDmaTraffic = expNumSlices * bigDmaTraffic;
    uint64_t expAggTime = (totalExpMMETraffic + totalExpDmaTraffic) / SlicingBrain::knobs.hbmAvailableBWGBps;
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, expAggTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__hbm_bw_bound_tpc_consumer_producer)
{
    // Given
    createSlicedGemmBundle();
    addTpcProducer(m_inputA, true);
    addTpcConsumer(m_output, true);

    // And given mme cost model s.t the traffic would dominate the time estimation
    constexpr uint64_t bigMmeTraffic = 1009 * 1009; // "Almost" prime, still low chance of collision
    constexpr uint64_t bigTpcTraffic = 1019 * 1019; // "Almost" prime, still low chance of collision
    constexpr uint64_t bigFetchTraffic = 1013 * 1013; // "Almost" prime, still low chance of collision
    constexpr uint64_t bigEvictTraffic = 1019 * 1019; // "Almost" prime, still low chance of collision
    MockCostModel mmeHeavyCostModel{CostEngine::MME, mmeOpTime, bigMmeTraffic};
    MockCostModel tpcHeavyCostModel{CostEngine::TPC, tpcOpTime, bigTpcTraffic};
    MockCostModel fetchHeavyCostModel{CostEngine::DMA, fetchOpTime, bigFetchTraffic};
    MockCostModel evictHeavyCostModel{CostEngine::DMA, evictOpTime, bigEvictTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeHeavyCostModel, tpcHeavyCostModel, fetchHeavyCostModel, evictHeavyCostModel);

    // Do some test specific calculations:
    unsigned expMmeOperations = expNumSlices;
    unsigned expTpcOperations = 2 * expNumSlices;  // Producer and consumer for each slice
    // DMA cost is accumulated for all MME/TPC operations.
    // Expecting the non-mock dma cost model to only count the operations that actually involve data movement.
    unsigned expDmaOperations = expMmeOperations + expTpcOperations;

    uint64_t totalExpMmeTraffic = expMmeOperations * bigMmeTraffic;
    uint64_t totalExpTpcTraffic = expTpcOperations * bigTpcTraffic;
    uint64_t totalExpFetchTraffic = expDmaOperations * bigFetchTraffic;
    uint64_t totalExpEvictTraffic = expDmaOperations * bigEvictTraffic;
    uint64_t expAggTime = (totalExpMmeTraffic + totalExpTpcTraffic + totalExpFetchTraffic + totalExpEvictTraffic) /
                          SlicingBrain::knobs.hbmAvailableBWGBps;

    // Then, assert that the aggregated time is the expected aggregated time to carry out all the data movement.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, expAggTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__tpc_producer_bound)
{
    // Given
    createSlicedGemmBundle();
    addTpcProducer(m_inputA, true);

    // And given TPC cost model s.t TPC time will dominate the cost
    constexpr uint64_t tpcBigOpTime = 997; // Keep it prime
    MockCostModel tpcHeavyCostModel{CostEngine::TPC, tpcBigOpTime, tpcTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcHeavyCostModel, fetchCostModel, evictCostModel);

    uint64_t suffixTime = mmeOpTime; // only TPC producer - there is another MME after TPC is done

    // Then, assert that the aggregated time is the time to compute the TPC nodes + suffix time.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, expNumSlices * tpcBigOpTime + suffixTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__tpc_consumer_bound)
{
    // Given
    createSlicedGemmBundle();
    addTpcConsumer(m_output, true);

    // And given TPC cost model s.t TPC time will dominate the cost
    constexpr uint64_t tpcBigOpTime = 997; // Keep it prime
    MockCostModel tpcHeavyCostModel{CostEngine::TPC,tpcBigOpTime, tpcTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcHeavyCostModel, fetchCostModel, evictCostModel);

    uint64_t prefixTime = fetchOpTime + mmeOpTime; // only TPC consumer - it can't start before MME's first slice,
                                                 // which in turn can't start before DMA fetches its inputs.

    // Then, assert that the aggregated time is the time to compute the TPC nodes + prefix time.
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(aggregatedCost.timeNano, prefixTime + expNumSlices * tpcBigOpTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs__tpc_producer_consumer_bound)
{
    // Given
    createSlicedGemmBundle();
    addTpcProducer(m_inputA, true);
    addTpcConsumer(m_output, true);

    // And given TPC cost model s.t TPC time will dominate the cost
    constexpr uint64_t tpcBigOpTime = 997; // Keep it prime
    MockCostModel tpcHeavyCostModel{CostEngine::TPC,tpcBigOpTime, tpcTraffic};

    // When
    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcHeavyCostModel, fetchCostModel, evictCostModel);

    // Then, assert that the aggregated time is the time to compute the TPC nodes (producer and consumer).
    const CostModel::Cost& aggregatedCost = strategyCost.getAggregatedCost();
    // Each slice is both produced and consumed by TPC, so the total num of TPC ops = 2 * expSlices
    ASSERT_EQ(aggregatedCost.timeNano, 2 * expNumSlices * tpcBigOpTime);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_model_should_aggregate_costs_dedx_dedw_with_slave_tpc_evict)
{
    const unsigned b = 4, h = 1, w = 2048, k = 256, c = 128, r = 1, s = 1;

    std::vector<TSize> dySizes  = {k, w, h, b};
    std::vector<TSize> xSizes   = {c, w, h, b};
    std::vector<TSize> wghSizes = {k, c, s, r};

    pTensor dy  = createTensor(dySizes, syn_type_bf16);
    pTensor wgh = createTensor(wghSizes, syn_type_bf16);
    pTensor dw  = createTensor(wghSizes, syn_type_bf16);
    pTensor x   = createTensor(xSizes, syn_type_bf16);
    pTensor dx  = createTensor(xSizes, syn_type_bf16);

    // Start with dedx bundle
    synConvolutionParams params {};
    pNode dedx = NodeFactory::createNode({dy, wgh}, {dx}, &params, NodeFactory::deDxNodeTypeName, "dedx");
    GraphEditor::addNode(getGraph(), dedx);
    std::shared_ptr<Bundlizer> bundlizer = std::make_shared<Bundlizer>(getGraph());
    pBundle                    bundle    = bundlizer->getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);
    ASSERT_EQ(bundle->getNodes().front(), dedx);
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), dedx);
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
    auto& sd                                    = strategy->getMmeSlicingData();
    sd.bundleTensors[0]->chunkDimensions[DIM_B] = sd.masterOperand->chunkDimensions[DIM_B] = b / 2;

    // Add dedw to the bundle
    pNode dedw = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");
    GraphEditor::addNode(getGraph(), dedw);
    MMESlaveBrain                  slaveBrain {getGraph()};
    SharedMMEInputCandidateHandler handler;
    std::list<pBundleExpansion>    candidates = handler.findSharedMMEInputCandidate(strategy, getGraph());
    ASSERT_FALSE(candidates.empty());
    auto& candidate = candidates.front();
    ASSERT_EQ(candidate->nodeToStitch, dedw);
    auto adjustedCandidate = slaveBrain.adjustCandidateToStrategy(candidate, strategy);
    strategy->getMmeSlicingData().addValidCandidate(adjustedCandidate, false);
    slaveBrain.addSharedOperandMme(adjustedCandidate, strategy);
    bundlizer->addCandidateToBundle(bundle, adjustedCandidate);

    MMECostModel         mmeCM(*getGraph().getHALReader());
    TPCCostModel         tpcCM(*getGraph().getHALReader());
    DmaFetchCostModel    fetchCM(getGraph());
    DmaEvictionCostModel evictCM(getGraph(), bundle, strategy);

    StrategyCostModel strategyCost(getGraph(), bundle, strategy, getSlicingBrains());
    strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

    const auto& fetchCost      = strategyCost.getFetchCost(CostModel::Cost::Engine::DMA);
    const auto& mmeCost        = strategyCost.getMmeCost();
    const auto& tpcCost        = strategyCost.getTpcCost();
    const auto& dmaEvictCost   = strategyCost.getEvictionCost(CostModel::Cost::Engine::DMA);
    const auto& tpcEvictCost   = strategyCost.getEvictionCost(CostModel::Cost::Engine::TPC);
    const auto& aggregatedCost = strategyCost.getAggregatedCost();
    ASSERT_EQ(tpcCost.total.hbmTrafficBytes, 0);       // No TPC
    ASSERT_EQ(dmaEvictCost.total.hbmTrafficBytes, 0);  // No DMA eviction
    ASSERT_EQ(tpcEvictCost.total.hbmTrafficBytes,
              strategy->getMmeSlicingData()
                  .getSlaveOutputOperand()
                  ->originalTensor->getTotalSizeInBytes());  // The slave output needs TPC cast
    ASSERT_EQ(aggregatedCost.timeNano,
              mmeCost.total.timeNano + fetchCost.prefix.timeNano + tpcEvictCost.suffix.timeNano);
}

class SRAMStrategyCostModelPerEngineAccumulatorTest
: public SRAMStrategyCompareTest
, public testing::WithParamInterface<std::tuple<bool, bool, bool, bool, bool, bool, bool>>
// wide producer, narrow producer, consumer, should evict, slave mme, slave tpc producer, slave tpc consumer
{
public:
    SRAMStrategyCostModelPerEngineAccumulatorTest()
    : m_wideProducer(std::get<0>(GetParam())),
      m_narrowProducer(std::get<1>(GetParam())),
      m_consumer(std::get<2>(GetParam())),
      m_shouldEvict(std::get<3>(GetParam())),
      m_slaveMme(std::get<4>(GetParam())),
      m_slaveTpcProducer(std::get<5>(GetParam())),
      m_slaveTpcConsumer(std::get<6>(GetParam()))
    {
        createSlicedGemmBundle(m_shouldEvict);
        m_strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
        if (m_shouldEvict)
        {
            m_strategy->setOutputIsInSRAM(true);
        }
        if (m_wideProducer)
        {
            addTpcProducer(m_inputA, true, true);
        }
        if (m_narrowProducer)
        {
            addTpcProducer(m_inputB, true, false);
        }
        if (m_consumer)
        {
            addTpcConsumer(m_output, true);
        }
        if (m_slaveMme)
        {
            addSlaveMME(m_inputA, m_inputB, true);
            if (m_slaveTpcProducer)
            {
                addSlaveTpcProducer(true);
            }
            if (m_slaveTpcConsumer)
            {
                addSlaveTpcConsumer(true);
            }
        }

        m_wideSliceSize   = SlicedOperandUtils::getSliceSizeInBytes(m_strategy->getMmeSlicingData().getWide());
        m_narrowSliceSize = SlicedOperandUtils::getSliceSizeInBytes(m_strategy->getMmeSlicingData().getNarrow());
        m_outSliceSize    = SlicedOperandUtils::getSliceSizeInBytes(m_strategy->getMmeSlicingData().masterOperand);
        m_wideSize        = m_strategy->getMmeSlicingData().getWide()->originalTensor->getTotalSizeInBytes();
        m_narrowSize      = m_strategy->getMmeSlicingData().getNarrow()->originalTensor->getTotalSizeInBytes();
        m_outSize         = m_strategy->getMmeSlicingData().masterOperand->originalTensor->getTotalSizeInBytes();
    }

    void validateCost()
    {
        if ((m_slaveTpcProducer || m_slaveTpcConsumer) && !m_slaveMme)
        {
            return;  // Invalid case
        }

        MMECostModel         mmeCM(*getGraph().getHALReader());
        TPCCostModel         tpcCM(*getGraph().getHALReader());
        DmaFetchCostModel    fetchCM(getGraph());
        DmaEvictionCostModel evictCM(getGraph(), m_bundle, m_strategy);

        StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
        strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

        const auto& fetchCost    = strategyCost.getFetchCost(CostModel::Cost::Engine::DMA);
        const auto& mmeCost      = strategyCost.getMmeCost();
        const auto& tpcCost      = strategyCost.getTpcCost();
        const auto& dmaEvictCost = strategyCost.getEvictionCost(CostModel::Cost::Engine::DMA);
        const auto& tpcEvictCost = strategyCost.getEvictionCost(CostModel::Cost::Engine::TPC);

        const auto& aggregatedCost = strategyCost.getAggregatedCost();

        // Validate DMA fetch traffic cost
        uint64_t expectedDmaTotalTraffic  = 0;
        uint64_t expectedDmaPrefixTraffic = 0;
        uint64_t expectedDmaSuffixTraffic = 0;
        if (!m_wideProducer && !m_narrowProducer)
        {
            expectedDmaPrefixTraffic = m_wideSliceSize + m_narrowSliceSize;
            expectedDmaTotalTraffic  = m_wideSize + m_narrowSize;
            expectedDmaSuffixTraffic = m_wideSliceSize;
        }
        else if (m_wideProducer && m_narrowProducer)
        {
            if (!m_slaveMme || m_slaveTpcProducer)
            {
                assertZeroAccumulatedCost(fetchCost);
            }
        }
        else if (m_wideProducer)
        {
            expectedDmaPrefixTraffic = m_narrowSliceSize;
            expectedDmaTotalTraffic  = m_narrowSize;
            expectedDmaSuffixTraffic = m_narrowSliceSize;
        }
        else if (m_narrowProducer)
        {
            expectedDmaPrefixTraffic = m_wideSliceSize;
            expectedDmaTotalTraffic  = m_wideSize;
            expectedDmaSuffixTraffic = m_wideSliceSize;
        }
        if (m_slaveMme && !m_slaveTpcProducer)
        {
            expectedDmaPrefixTraffic = (expectedDmaPrefixTraffic == 0) ? m_narrowSliceSize : expectedDmaPrefixTraffic;
            expectedDmaTotalTraffic += m_narrowSize;
            expectedDmaSuffixTraffic = m_narrowSliceSize;
        }
        ASSERT_EQ(fetchCost.prefix.hbmTrafficBytes, expectedDmaPrefixTraffic);
        ASSERT_EQ(fetchCost.total.hbmTrafficBytes, expectedDmaTotalTraffic);
        ASSERT_EQ(fetchCost.suffix.hbmTrafficBytes, expectedDmaSuffixTraffic);

        // Validate MME traffic cost
        uint64_t expectedMmeTotalTraffic  = 0;
        uint64_t expectedMmeSuffixTraffic = 0;
        if (m_shouldEvict || m_consumer)
        {
            ASSERT_EQ(mmeCost.prefix.hbmTrafficBytes, 0);
        }
        else
        {
            ASSERT_EQ(mmeCost.prefix.hbmTrafficBytes, m_outSliceSize);
            expectedMmeTotalTraffic  = m_outSize;
            expectedMmeSuffixTraffic = m_outSliceSize;
        }
        if (m_slaveMme)
        {
            if (m_slaveTpcConsumer)
            {
                expectedMmeSuffixTraffic = 0;
            }
            else
            {
                expectedMmeTotalTraffic += m_outSize;
                expectedMmeSuffixTraffic = m_outSliceSize;
            }
        }
        ASSERT_EQ(mmeCost.total.hbmTrafficBytes, expectedMmeTotalTraffic);
        ASSERT_EQ(mmeCost.suffix.hbmTrafficBytes, expectedMmeSuffixTraffic);

        // Validate TPC traffic cost
        uint64_t expectedTpcPrefixTraffic = 0;
        uint64_t expectedTpcTotalTraffic  = 0;
        uint64_t expectedTpcSuffixTraffic = 0;
        if (!m_wideProducer && !m_narrowProducer && !m_consumer && !m_slaveTpcProducer &&
            !m_slaveTpcConsumer)  // No TPC
        {
            assertZeroAccumulatedCost(tpcCost);
        }
        else if (m_wideProducer && m_narrowProducer)  // TPC wide + narrow producers
        {
            expectedTpcPrefixTraffic = m_wideSliceSize + m_narrowSliceSize;
            expectedTpcTotalTraffic  = m_wideSize + m_narrowSize;
            expectedTpcSuffixTraffic = m_wideSliceSize;
        }
        else if (m_wideProducer)  // TPC wide producer
        {
            expectedTpcPrefixTraffic = m_wideSliceSize;
            expectedTpcTotalTraffic  = m_wideSize;
            expectedTpcSuffixTraffic = m_wideSliceSize;
        }
        else if (m_narrowProducer)  // TPC narrow producer
        {
            expectedTpcPrefixTraffic = m_narrowSliceSize;
            expectedTpcTotalTraffic  = m_narrowSize;
            expectedTpcSuffixTraffic = m_narrowSliceSize;
        }
        if (m_slaveTpcProducer)  // The non-shared operand is the narrow operand
        {
            expectedTpcPrefixTraffic = (expectedTpcPrefixTraffic == 0) ? m_narrowSliceSize : expectedTpcPrefixTraffic;
            expectedTpcTotalTraffic += m_narrowSize;
            expectedTpcSuffixTraffic = m_narrowSliceSize;
        }
        // Update expected total/suffix time in case we have TPC consumer
        if (m_consumer)
        {
            expectedTpcPrefixTraffic = (expectedTpcPrefixTraffic == 0) ? m_outSliceSize : expectedTpcPrefixTraffic;
            expectedTpcTotalTraffic += m_outSize;
            expectedTpcSuffixTraffic = m_outSliceSize;
        }
        if (m_slaveTpcConsumer)
        {
            expectedTpcPrefixTraffic = (expectedTpcPrefixTraffic == 0) ? m_outSliceSize : expectedTpcPrefixTraffic;
            expectedTpcTotalTraffic += m_outSize;
            expectedTpcSuffixTraffic = m_outSliceSize;
        }

        ASSERT_EQ(tpcCost.prefix.hbmTrafficBytes, expectedTpcPrefixTraffic);
        ASSERT_EQ(tpcCost.total.hbmTrafficBytes, expectedTpcTotalTraffic);
        ASSERT_EQ(tpcCost.suffix.hbmTrafficBytes, expectedTpcSuffixTraffic);

        // Validate DMA/TPC evict traffic cost
        if (m_shouldEvict)
        {
            ASSERT_EQ(dmaEvictCost.prefix.hbmTrafficBytes, m_outSliceSize);
            ASSERT_EQ(dmaEvictCost.total.hbmTrafficBytes, m_outSize);
            ASSERT_EQ(dmaEvictCost.suffix.hbmTrafficBytes, m_outSliceSize);
        }
        else
        {
            assertZeroAccumulatedCost(dmaEvictCost);
        }
        assertZeroAccumulatedCost(tpcEvictCost);

        // Validate aggregated traffic cost
        uint64_t expectedAggregatedTrafficCost = m_wideSize + m_narrowSize + m_outSize;
        if (m_shouldEvict && m_consumer)
        {
            expectedAggregatedTrafficCost += m_outSize;
        }
        if (m_slaveMme)
        {
            expectedAggregatedTrafficCost += m_narrowSize + m_outSize;  // The wide operand is shared
        }
        ASSERT_EQ(aggregatedCost.hbmTrafficBytes, expectedAggregatedTrafficCost);

        // Validate aggregated time cost
        uint64_t expectedAggregatedTimeCost = 0;
        uint64_t totalDataMovementTime      = aggregatedCost.hbmTrafficBytes / SlicingBrain::knobs.hbmAvailableBWGBps;
        expectedAggregatedTimeCost += mmeCost.total.timeNano;
        if (!m_wideProducer && !m_narrowProducer && (!m_slaveMme || !m_slaveTpcProducer))  // DMA fetch, no TPC
        {
            expectedAggregatedTimeCost += fetchCost.prefix.timeNano;
        }
        else if (m_wideProducer && m_narrowProducer && (!m_slaveMme || m_slaveTpcProducer))  // TPC fetch, no DMA
        {
            expectedAggregatedTimeCost += tpcCost.prefix.timeNano;
        }
        else  // TPC + DMA fetch
        {
            uint64_t traffic          = tpcCost.prefix.hbmTrafficBytes + fetchCost.prefix.hbmTrafficBytes;
            uint64_t dataMovementTime = traffic / SlicingBrain::knobs.hbmAvailableBWGBps;
            expectedAggregatedTimeCost +=
                std::max(dataMovementTime, std::max(tpcCost.prefix.timeNano, fetchCost.prefix.timeNano));
        }
        if (m_shouldEvict && (m_consumer || m_slaveTpcConsumer))  // DMA + TPC eviction
        {
            uint64_t traffic          = dmaEvictCost.suffix.hbmTrafficBytes + tpcCost.suffix.hbmTrafficBytes;
            uint64_t dataMovementTime = traffic / SlicingBrain::knobs.hbmAvailableBWGBps;
            expectedAggregatedTimeCost +=
                std::max(dataMovementTime, std::max(dmaEvictCost.suffix.timeNano, tpcCost.suffix.timeNano));
        }
        else if (m_shouldEvict)  // DMA eviction
        {
            expectedAggregatedTimeCost += dmaEvictCost.suffix.timeNano;
        }
        else if (m_consumer || m_slaveTpcConsumer)  // TPC eviction
        {
            expectedAggregatedTimeCost += tpcCost.suffix.timeNano;
        }
        // Update time for HBM BW bound strategies
        expectedAggregatedTimeCost = std::max(expectedAggregatedTimeCost, totalDataMovementTime);
        ASSERT_EQ(aggregatedCost.timeNano, expectedAggregatedTimeCost);
    }

protected:
    const bool m_wideProducer;
    const bool m_narrowProducer;
    const bool m_consumer;
    const bool m_shouldEvict;
    const bool m_slaveMme;
    const bool m_slaveTpcProducer;
    const bool m_slaveTpcConsumer;

    unsigned m_wideSliceSize;
    unsigned m_narrowSliceSize;
    unsigned m_outSliceSize;
    unsigned m_wideSize;
    unsigned m_narrowSize;
    unsigned m_outSize;
};

TEST_P(SRAMStrategyCostModelPerEngineAccumulatorTest, validate_cost)
{
    validateCost();
}

INSTANTIATE_TEST_SUITE_P(cost_model_per_engine_accumulation,
                         SRAMStrategyCostModelPerEngineAccumulatorTest,
                         ::testing::Combine(::testing::ValuesIn({false, true}),    // wide producer
                                            ::testing::ValuesIn({false, true}),    // narrow producer
                                            ::testing::ValuesIn({false, true}),    // consumer
                                            ::testing::ValuesIn({false, true}),    // should evict
                                            ::testing::ValuesIn({false, true}),    // slave mme
                                            ::testing::ValuesIn({false, true}),    // slave tpc producer
                                            ::testing::ValuesIn({false, true})));  // slave tpc consumer

class SRAMStrategyCompareInvalidCandidatesTest
: public SRAMStrategyCompareTest
, public testing::WithParamInterface<std::tuple<bool, bool, bool>>
// invalid wide producer, invalid narrow producer, invalid consumer
{
public:
    SRAMStrategyCompareInvalidCandidatesTest()
    : m_invalidWideProducer(std::get<0>(GetParam())),
      m_invalidNarrowProducer(std::get<1>(GetParam())),
      m_invalidConsumer(std::get<2>(GetParam()))
    {
    }

protected:
    const bool m_invalidWideProducer;
    const bool m_invalidNarrowProducer;
    const bool m_invalidConsumer;
};

TEST_P(SRAMStrategyCompareInvalidCandidatesTest, validate_invalid_candidates_cost)
{
    createSlicedGemmBundle();
    StrategyCostModel strategyCostWithoutInvalidCandidates(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCostWithoutInvalidCandidates.model(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    auto numOfCachedInvalidCandidates = StrategyCostModel::getInvalidCandidatesCache().size();
    auto numOfNewInvalidCandidates    = 0;

    pBundleExpansion wideProducerCandidate;
    if (m_invalidWideProducer)
    {
        wideProducerCandidate = addInvalidTpcProducer(m_inputA, true);
        ASSERT_NE(wideProducerCandidate, nullptr);
        numOfNewInvalidCandidates++;
    }
    pBundleExpansion narrowProducerCandidate;
    if (m_invalidNarrowProducer)
    {
        narrowProducerCandidate = addInvalidTpcProducer(m_inputB, false);
        ASSERT_NE(narrowProducerCandidate, nullptr);
        numOfNewInvalidCandidates++;
    }
    pBundleExpansion consumerCandidate;
    if (m_invalidConsumer)
    {
        consumerCandidate = addInvalidTpcConsumer(m_output);
        ASSERT_NE(consumerCandidate, nullptr);
        numOfNewInvalidCandidates++;
    }

    StrategyCostModel strategyCost(getGraph(), m_bundle, m_strategy, getSlicingBrains());
    strategyCost.model(mmeCostModel, tpcCostModel, fetchCostModel, evictCostModel);

    // Validate that the new invalid candidates are added to the cache
    ASSERT_EQ(StrategyCostModel::getInvalidCandidatesCache().size(),
              numOfCachedInvalidCandidates + numOfNewInvalidCandidates);

    uint64_t expectedAggregatedCost = strategyCostWithoutInvalidCandidates.getAggregatedCost().timeNano;
    if (m_invalidWideProducer)
    {
        uint64_t invalidCost = strategyCost.getInvalidCandidateCost(wideProducerCandidate).timeNano;
        ASSERT_GT(invalidCost, 0);
        expectedAggregatedCost += invalidCost;
    }
    if (m_invalidNarrowProducer)
    {
        uint64_t invalidCost = strategyCost.getInvalidCandidateCost(narrowProducerCandidate).timeNano;
        ASSERT_GT(invalidCost, 0);
        expectedAggregatedCost += invalidCost;
    }
    if (m_invalidConsumer)
    {
        uint64_t invalidCost = strategyCost.getInvalidCandidateCost(consumerCandidate).timeNano;
        ASSERT_GT(invalidCost, 0);
        expectedAggregatedCost += invalidCost;
    }

    ASSERT_EQ(strategyCost.getAggregatedCost().timeNano, expectedAggregatedCost);
}

INSTANTIATE_TEST_SUITE_P(strategy_cost_model_should_aggregate_invalid_candidates,
                         SRAMStrategyCompareInvalidCandidatesTest,
                         ::testing::Combine(::testing::ValuesIn({false, true}),  // wide producer
                                            ::testing::ValuesIn({false, true}),  // narrow producer
                                            ::testing::ValuesIn({false, true})   // consumer
                                            ));

TEST_F(SRAMStrategyCompareTest, all_invalid_candidates)
{
    // Create a graph with 2 MMEs sharing an input, with a TPC producer
    // to each MME input and a TPC consumer to each mme output.

    std::vector<TSize>    sizes          = {128, 128};
    pTensor               shared         = createTensor(sizes, syn_type_bf16);
    pTensor               gemm0Wgh       = createTensor(sizes, syn_type_bf16);
    pTensor               gemm0Out       = createTensor(sizes, syn_type_bf16);
    pTensor               gemm1Wgh       = createTensor(sizes, syn_type_bf16);
    pTensor               gemm1Out       = createTensor(sizes, syn_type_bf16);
    pTensor               reluInShared   = createTensor(sizes, syn_type_bf16);
    pTensor               reluInGemm0Wgh = createTensor(sizes, syn_type_bf16);
    pTensor               reluInGemm1Wgh = createTensor(sizes, syn_type_bf16);
    pTensor               reluOutGemm0   = createTensor(sizes, syn_type_bf16);
    pTensor               reluOutGemm1   = createTensor(sizes, syn_type_bf16);

    synGEMMParams gemm0Params {};
    synGEMMParams gemm1Params {};

    pNode gemm0 =
        NodeFactory::createNode({shared, gemm0Wgh}, {gemm0Out}, &gemm0Params, NodeFactory::gemmNodeTypeName, "gemm0");
    GraphEditor::addNode(getGraph(), gemm0);
    pNode gemm1 =
        NodeFactory::createNode({shared, gemm1Wgh}, {gemm1Out}, &gemm1Params, NodeFactory::gemmNodeTypeName, "gemm1");
    GraphEditor::addNode(getGraph(), gemm1);

    unsigned i = 0;
    for (std::pair<pTensor, pTensor> reluOperands :
         std::vector<std::pair<pTensor, pTensor>> {{reluInShared, shared},
                                                   {reluInGemm0Wgh, gemm0Wgh},
                                                   {reluInGemm1Wgh, gemm1Wgh},
                                                   {gemm0Out, reluOutGemm0},
                                                   {gemm1Out, reluOutGemm1}})
    {
        std::string name {"relu"};
        pNode       relu = NodeFactory::createNode({reluOperands.first},
                                             {reluOperands.second},
                                             nullptr,
                                             "relu_fwd_bf16",
                                             name + std::to_string(i++));
        GraphEditor::addNode(getGraph(), relu);
    }

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    // Input + output size, all the tensors are the same.
    const auto expectedTpcTraffic = reluInShared->getTotalSizeInBytes() * 2;

    auto expectedSlaveBundleTraffic = 0;
    auto expectedSlaveBundleTime    = 0;

    // Create bundles and initial strategies
    AllBrains          allBrains {getGraph()};
    SRAMSlicingManager sramManager {getGraph()};
    sramManager.generateInitialBundles();
    sramManager.generateInitialStrategies();
    auto bundlesSolvingData = sramManager.getBundlesSolvingData();
    ASSERT_EQ(bundlesSolvingData.size(), 2);

    BundleExpander bundleExpander {getGraph(), allBrains, sramManager.getBundlizer(), bundlesSolvingData};

    auto iter = bundlesSolvingData.begin();
    // First expand the first bundle (both are symmetric) without slave expansions to find the cost
    // of 1 gemm + 2 producers + 1 consumer.
    setGlobalConfForTest(GCFG_SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED, "false");
    const pBundle& slaveBundle          = iter->first;
    const auto&    winningSlaveStrategy = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(bundleExpander.generateExpandedStrategies(slaveBundle),
                            slaveBundle,
                            getGraph(),
                            allBrains));
    const auto& slaveBundleCost = winningSlaveStrategy->getCost();
    ASSERT_TRUE(slaveBundleCost.is_set());
    expectedSlaveBundleTraffic = slaveBundleCost.value().hbmTrafficBytes;
    expectedSlaveBundleTime    = slaveBundleCost.value().timeNano;

    // Expand the second bundle and check that the invalid candidates cost is as expected.
    setGlobalConfForTest(GCFG_SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED, "true");
    const pBundle& masterBundle = (++iter)->first;
    bool           allInvalidCandidatesStrategyFound = false;
    for (const auto& s : bundleExpander.generateExpandedStrategies(masterBundle))
    {
        pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);

        // Valid candidates are not stitched to the strategy and therefor the strategy cannot be modeled.
        if (std::any_of(strategy->getMmeSlicingData().getRoleCandidates().begin(),
                        strategy->getMmeSlicingData().getRoleCandidates().end(),
                        [](const pBundleExpansion& exp) { return exp != nullptr; }))
        {
            continue;  // only interested in strategy with all candidates invalid
        }
        allInvalidCandidatesStrategyFound = true;

        MMECostModel         mmeCM(*getGraph().getHALReader());
        TPCCostModel         tpcCM(*getGraph().getHALReader());
        DmaFetchCostModel    fetchCM(getGraph());
        DmaEvictionCostModel evictCM(getGraph(), masterBundle, strategy);

        StrategyCostModel strategyCost(getGraph(), masterBundle, strategy, getSlicingBrains());
        strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

        const auto& invalidCandidates  = strategy->getMmeSlicingData().getInvalidCandidates();
        bool        hasInvalidSlaveMme = false;
        for (const auto& invalidCandidate : invalidCandidates)
        {
            if (invalidCandidate->role == BundleExpansion::SharedInputConsumer)
            {
                hasInvalidSlaveMme = true;
            }
        }

        for (const auto& invalidCandidate : invalidCandidates)
        {
            if (hasInvalidSlaveMme && ((invalidCandidate->role == BundleExpansion::SlaveInputProducer) ||
                                       (invalidCandidate->role == BundleExpansion::SlaveOutputConsumer)))
            {
                // The slave TPC candidates are saved together with the slave MME
                continue;
            }
            const auto& invalidCost = strategyCost.getInvalidCandidateCost(invalidCandidate);
            switch (invalidCandidate->role)
            {
                case BundleExpansion::WideInputProducer:
                case BundleExpansion::NarrowInputProducer:
                case BundleExpansion::OutputConsumer:
                case BundleExpansion::SlaveOutputConsumer:
                case BundleExpansion::SlaveInputProducer:
                    ASSERT_EQ(invalidCost.hbmTrafficBytes, expectedTpcTraffic);
                    break;
                case BundleExpansion::SharedInputConsumer:
                    // If the slave MME is invalid - the slave producers and conumer must to be invalid as well
                    // The expected time should be the time of the slave bundle (1 gemm + 2 producers + 1 consumer).
                    ASSERT_EQ(invalidCost.timeNano, expectedSlaveBundleTime);
                    ASSERT_EQ(invalidCost.hbmTrafficBytes, expectedSlaveBundleTraffic);
                    break;
                default:
                    HB_ASSERT(false, "Unexpected role for candidate");
            }
        }
    }
    EXPECT_TRUE(allInvalidCandidatesStrategyFound);
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_with_mme_flattening)
{
    pTensor              x = createTensor({64, 56, 56, 64}, syn_type_bf16, true);
    pTensor              w = createTensor({1, 1, 64, 64}, syn_type_bf16, true);
    pTensor              y = createTensor({64, 56, 56, 64}, syn_type_bf16, false);
    synConvolutionParams params {};
    pNode conv = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    GraphEditor::addNode(getGraph(), conv);
    pTensor consumerOutput = createTensor({64, 56, 56, 64}, syn_type_bf16, true);
    pNode   consumer       = NodeFactory::createNode({y}, {consumerOutput}, nullptr, NOP_KERNEL_NAME, "consumer");
    GraphEditor::addNode(getGraph(), consumer);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    std::shared_ptr<Bundlizer> bundlizer = std::make_shared<Bundlizer>(getGraph());
    pBundle                    bundle    = bundlizer->getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);

    NonCD2DSolver solver(*getGraph().getHALReader(), bundle);
    solver.createAllStrategies();

    for (const auto& strategy : solver.getStrategies())
    {
        pMmeSlicingStrategy mmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(strategy);
        // Make sure the output is flattened
        ASSERT_NE(mmeStrategy->getMmeSlicingData().masterOperand->finalShape,
                  mmeStrategy->getMmeSlicingData().masterOperand->originalTensor->getAllSizesInElements());

        pBundleExpansion candidate = bundlizer->findTpcConsumerExpansionCandidate(mmeStrategy);
        if (!candidate || !candidate->nodeToStitch)
        {
            continue;
        }

        TPCSlaveBrain slaveBrain(getGraph());
        mmeStrategy->getMmeSlicingData().addValidCandidate(candidate);
        ASSERT_TRUE(slaveBrain.addConsumerToStrategy(candidate, mmeStrategy));

        MMECostModel         mmeCM(*getGraph().getHALReader());
        TPCCostModel         tpcCM(*getGraph().getHALReader());
        DmaFetchCostModel    fetchCM(getGraph());
        DmaEvictionCostModel evictCM(getGraph(), bundle, mmeStrategy);

        StrategyCostModel strategyCost(getGraph(), bundle, mmeStrategy, getSlicingBrains());
        strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

        ASSERT_EQ(consumerOutput->getTotalSizeInBytes(), strategyCost.getTpcCost().total.hbmTrafficBytes);
        ASSERT_EQ(0, strategyCost.getMmeCost().total.hbmTrafficBytes);
        ASSERT_EQ(x->getTotalSizeInBytes() + w->getTotalSizeInBytes(),
                  strategyCost.getFetchCost(CostModel::Cost::Engine::DMA).total.hbmTrafficBytes);
        ASSERT_EQ(0, strategyCost.getEvictionCost(CostModel::Cost::Engine::DMA).total.hbmTrafficBytes);
        ASSERT_EQ(0, strategyCost.getEvictionCost(CostModel::Cost::Engine::TPC).total.hbmTrafficBytes);
    }
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_with_wide_operand_reshape)
{
    const TSize w = 2024;
    const TSize k = 319;
    const TSize c = 125;

    pTensor inAProducer = createTensor({k, w / 4, 1, 4}, syn_type_bf16, true);
    pTensor inAReshape  = createTensor({k, w / 4, 1, 4}, syn_type_bf16, false);
    pTensor inA         = createTensor({k, w}, syn_type_bf16, false);
    pTensor inB         = createTensor({c, k}, syn_type_bf16, true);
    pTensor out         = createTensor({c, w}, syn_type_bf16, true);

    synGEMMParams gemmParams {};
    pNode         gemm = NodeFactory::createNode({inA, inB}, {out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    GraphEditor::addNode(getGraph(), gemm);
    pNode reshapeA =
        NodeFactory::createNode({inAReshape}, {inA}, nullptr, NodeFactory::reshapeNodeTypeName, "reshapeA");
    GraphEditor::addNode(getGraph(), reshapeA);
    pNode reluA = NodeFactory::createNode({inAProducer}, {inAReshape}, nullptr, "relu_fwd_bf16", "reluA");
    GraphEditor::addNode(getGraph(), reluA);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    std::shared_ptr<Bundlizer> bundlizer = std::make_shared<Bundlizer>(getGraph());
    pBundle                    bundle    = bundlizer->getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);

    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), gemm);
    auto&               sd       = strategy->getMmeSlicingData();
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
    sd.bundleTensors[0]->chunkDimensions[DIM_W] = sd.masterOperand->chunkDimensions[DIM_W] =
        (w / 2);  // pick a number that is a multiple of the granularity

    pMmeSlicingStrategy mmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(strategy);
    TPCSlaveBrain       slaveBrain(getGraph());
    ReshapeSlicingBrain reshapeBrain(getGraph());
    pBundleExpansion    wideProducerCandidate = bundlizer->findWideTpcProducerExpansionCandidate(mmeStrategy);
    ASSERT_NE(wideProducerCandidate, nullptr);
    ASSERT_NE(wideProducerCandidate->nodeToStitch, nullptr);
    ASSERT_NE(wideProducerCandidate->reshapeNode, nullptr);
    mmeStrategy->getMmeSlicingData().addValidCandidate(wideProducerCandidate);
    ASSERT_TRUE(reshapeBrain.addProducerToStrategy(wideProducerCandidate, mmeStrategy));
    ASSERT_TRUE(slaveBrain.addProducerToStrategy(wideProducerCandidate, mmeStrategy));

    MMECostModel         mmeCM(*getGraph().getHALReader());
    TPCCostModel         tpcCM(*getGraph().getHALReader());
    DmaFetchCostModel    fetchCM(getGraph());
    DmaEvictionCostModel evictCM(getGraph(), bundle, mmeStrategy);

    StrategyCostModel strategyCost(getGraph(), bundle, mmeStrategy, getSlicingBrains());
    strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

    ASSERT_EQ(strategyCost.getTpcCost().total.hbmTrafficBytes, inAProducer->getTotalSizeInBytes());
    ASSERT_EQ(strategyCost.getMmeCost().total.hbmTrafficBytes, out->getTotalSizeInBytes());
    ASSERT_EQ(strategyCost.getFetchCost(CostModel::Cost::Engine::DMA).total.hbmTrafficBytes,
              inB->getTotalSizeInBytes());
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::DMA));
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::TPC));
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_with_narrow_operand_reshape)
{
    const TSize w = 123;
    const TSize k = 319;
    const TSize c = 5640;

    pTensor inA         = createTensor({k, w}, syn_type_bf16, true);
    pTensor inBProducer = createTensor({k, 1, 4, c / 4}, syn_type_bf16, true);
    pTensor inBReshape  = createTensor({k, 1, 4, c / 4}, syn_type_bf16, false);
    pTensor inB         = createTensor({k, c}, syn_type_bf16, false);
    pTensor out         = createTensor({c, w}, syn_type_bf16, true);

    synGEMMParams gemmParams {};
    gemmParams.transpose_b = true;
    pNode         gemm = NodeFactory::createNode({inA, inB}, {out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    GraphEditor::addNode(getGraph(), gemm);
    pNode reshapeB =
        NodeFactory::createNode({inBReshape}, {inB}, nullptr, NodeFactory::reshapeNodeTypeName, "reshapeB");
    GraphEditor::addNode(getGraph(), reshapeB);
    pNode reluB = NodeFactory::createNode({inBProducer}, {inBReshape}, nullptr, "relu_fwd_bf16", "reluB");
    GraphEditor::addNode(getGraph(), reluB);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    std::shared_ptr<Bundlizer> bundlizer = std::make_shared<Bundlizer>(getGraph());
    pBundle                    bundle    = bundlizer->getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);

    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), gemm);
    auto&               sd       = strategy->getMmeSlicingData();
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
    // Operand b is transposed (slicing on w instead of c)
    sd.bundleTensors[1]->chunkDimensions[DIM_W] = sd.masterOperand->chunkDimensions[DIM_C] = 1000;

    pMmeSlicingStrategy mmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(strategy);
    TPCSlaveBrain       slaveBrain(getGraph());
    ReshapeSlicingBrain reshapeBrain(getGraph());
    pBundleExpansion    narrowProducerCandidate = bundlizer->findNarrowTpcProducerExpansionCandidate(mmeStrategy);
    ASSERT_NE(narrowProducerCandidate, nullptr);
    ASSERT_NE(narrowProducerCandidate->nodeToStitch, nullptr);
    ASSERT_NE(narrowProducerCandidate->reshapeNode, nullptr);
    mmeStrategy->getMmeSlicingData().addValidCandidate(narrowProducerCandidate);
    ASSERT_TRUE(reshapeBrain.addProducerToStrategy(narrowProducerCandidate, mmeStrategy));
    ASSERT_TRUE(slaveBrain.addProducerToStrategy(narrowProducerCandidate, mmeStrategy));

    MMECostModel         mmeCM(*getGraph().getHALReader());
    TPCCostModel         tpcCM(*getGraph().getHALReader());
    DmaFetchCostModel    fetchCM(getGraph());
    DmaEvictionCostModel evictCM(getGraph(), bundle, mmeStrategy);

    StrategyCostModel strategyCost(getGraph(), bundle, mmeStrategy, getSlicingBrains());
    strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

    ASSERT_EQ(strategyCost.getTpcCost().total.hbmTrafficBytes, inBProducer->getTotalSizeInBytes());
    ASSERT_EQ(strategyCost.getMmeCost().total.hbmTrafficBytes, out->getTotalSizeInBytes());
    ASSERT_EQ(strategyCost.getFetchCost(CostModel::Cost::Engine::DMA).total.hbmTrafficBytes,
              inA->getTotalSizeInBytes());
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::DMA));
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::TPC));
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_with_output_reshape)
{
    const TSize w = 3030;
    const TSize k = 319;
    const TSize c = 220;

    pTensor inA         = createTensor({k, w}, syn_type_bf16, true);
    pTensor inB         = createTensor({c, k}, syn_type_bf16, true);
    pTensor out         = createTensor({c, w}, syn_type_bf16, false);
    pTensor outReshape  = createTensor({c / 2, 2, 2, w / 2}, syn_type_bf16, false);
    pTensor outConsumer = createTensor({c / 2, 2, 2, w / 2}, syn_type_bf16, true);

    synGEMMParams gemmParams {};
    pNode         gemm = NodeFactory::createNode({inA, inB}, {out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    GraphEditor::addNode(getGraph(), gemm);
    pNode reshapeOut =
        NodeFactory::createNode({out}, {outReshape}, nullptr, NodeFactory::reshapeNodeTypeName, "reshapeOut");
    GraphEditor::addNode(getGraph(), reshapeOut);
    pNode reluOut = NodeFactory::createNode({outReshape}, {outConsumer}, nullptr, "relu_fwd_bf16", "reluOut");
    GraphEditor::addNode(getGraph(), reluOut);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    std::shared_ptr<Bundlizer> bundlizer = std::make_shared<Bundlizer>(getGraph());
    pBundle                    bundle    = bundlizer->getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);

    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), gemm);
    auto&               sd       = strategy->getMmeSlicingData();
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
    sd.bundleTensors[0]->chunkDimensions[DIM_W] = sd.masterOperand->chunkDimensions[DIM_W] = 500;

    pMmeSlicingStrategy mmeStrategy = std::static_pointer_cast<MmeSlicingStrategy>(strategy);
    TPCSlaveBrain       slaveBrain(getGraph());
    ReshapeSlicingBrain reshapeBrain(getGraph());
    pBundleExpansion    consumerCandidate = bundlizer->findTpcConsumerExpansionCandidate(mmeStrategy);
    ASSERT_NE(consumerCandidate, nullptr);
    ASSERT_NE(consumerCandidate->nodeToStitch, nullptr);
    ASSERT_NE(consumerCandidate->reshapeNode, nullptr);
    mmeStrategy->getMmeSlicingData().addValidCandidate(consumerCandidate);
    ASSERT_TRUE(reshapeBrain.addConsumerToStrategy(consumerCandidate, mmeStrategy));
    ASSERT_TRUE(slaveBrain.addConsumerToStrategy(consumerCandidate, mmeStrategy));

    MMECostModel         mmeCM(*getGraph().getHALReader());
    TPCCostModel         tpcCM(*getGraph().getHALReader());
    DmaFetchCostModel    fetchCM(getGraph());
    DmaEvictionCostModel evictCM(getGraph(), bundle, mmeStrategy);

    StrategyCostModel strategyCost(getGraph(), bundle, mmeStrategy, getSlicingBrains());
    strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

    ASSERT_EQ(strategyCost.getTpcCost().total.hbmTrafficBytes, outConsumer->getTotalSizeInBytes());
    ASSERT_EQ(strategyCost.getMmeCost().total.hbmTrafficBytes, 0);
    ASSERT_EQ(strategyCost.getFetchCost(CostModel::Cost::Engine::DMA).total.hbmTrafficBytes,
              inA->getTotalSizeInBytes() + inB->getTotalSizeInBytes());
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::DMA));
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::TPC));
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_with_unaligned_slices_both_operands_sliced)
{
    const TSize    w                 = 1024;
    const TSize    k                 = 128;
    const TSize    c                 = 3190;
    const unsigned numOfWideSlices   = 2;
    const unsigned numOfNarrowSlices = 10;
    pTensor        inAProducer       = createTensor({k, w}, syn_type_bf16, true);
    pTensor        inA               = createTensor({k, w}, syn_type_bf16, false);
    pTensor        inBProducer       = createTensor({c, k}, syn_type_bf16, true);
    pTensor        inB               = createTensor({c, k}, syn_type_bf16, false);
    pTensor        out               = createTensor({c, w}, syn_type_bf16, false);
    pTensor        outConsumer       = createTensor({c, w}, syn_type_bf16, true);

    synGEMMParams gemmParams {};
    pNode         gemm = NodeFactory::createNode({inA, inB}, {out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    GraphEditor::addNode(getGraph(), gemm);

    // Create a sliceable TPC producer for operand A (slicing granularity on each dim is 1)
    pNode producerA = TPCCustomIndexSpaceNode::createSliceableNode(inAProducer, inA);
    GraphEditor::addNode(getGraph(), producerA);

    // Create a sliceable TPC producer for operand B (slicing granularity on each dim is 1)
    pNode producerB = TPCCustomIndexSpaceNode::createSliceableNode(inBProducer, inB);
    GraphEditor::addNode(getGraph(), producerB);

    // Create a sliceable TPC consumer for output operand (slicing granularity on each dim is 1)
    pNode tpcOut = TPCCustomIndexSpaceNode::createSliceableNode(out, outConsumer);
    GraphEditor::addNode(getGraph(), tpcOut);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer(getGraph());
    pBundle   bundle = bundlizer.getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);

    // Create a strategy - the wide operand is sliced to 2 slices, the narrow operand is sliced to 10 slices
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), gemm);
    auto&               sd       = strategy->getMmeSlicingData();
    sd.bundleTensors[0]->chunkDimensions[DIM_W] = sd.masterOperand->chunkDimensions[DIM_W] = w / numOfWideSlices;
    sd.bundleTensors[1]->chunkDimensions[DIM_C] = sd.masterOperand->chunkDimensions[DIM_C] = c / numOfNarrowSlices;
    sd.bundleTensors[0]->alignWithCacheLine                                                = true;
    sd.bundleTensors[1]->alignWithCacheLine                                                = true;
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setDoubleBuffer(true);

    TPCSlaveBrain slaveBrain(getGraph());

    // Add wide TPC producer
    pBundleExpansion wideProducerCandidate = bundlizer.findWideTpcProducerExpansionCandidate(strategy);
    ASSERT_NE(wideProducerCandidate, nullptr);
    ASSERT_NE(wideProducerCandidate->nodeToStitch, nullptr);
    strategy->getMmeSlicingData().addValidCandidate(wideProducerCandidate);
    ASSERT_TRUE(slaveBrain.addProducerToStrategy(wideProducerCandidate, strategy));

    // Add narrow TPC producer
    pBundleExpansion narrowProducerCandidate = bundlizer.findNarrowTpcProducerExpansionCandidate(strategy);
    ASSERT_NE(narrowProducerCandidate, nullptr);
    ASSERT_NE(narrowProducerCandidate->nodeToStitch, nullptr);
    strategy->getMmeSlicingData().addValidCandidate(narrowProducerCandidate);
    ASSERT_TRUE(slaveBrain.addProducerToStrategy(narrowProducerCandidate, strategy));

    // Add TPC consumer
    pBundleExpansion consumerCandidate = bundlizer.findTpcConsumerExpansionCandidate(strategy);
    ASSERT_NE(consumerCandidate, nullptr);
    ASSERT_NE(consumerCandidate->nodeToStitch, nullptr);
    strategy->getMmeSlicingData().addValidCandidate(consumerCandidate);
    ASSERT_TRUE(slaveBrain.addConsumerToStrategy(consumerCandidate, strategy));

    // Estimate the strategy using the cost model
    MMECostModel         mmeCM(*getGraph().getHALReader());
    TPCCostModel         tpcCM(*getGraph().getHALReader());
    DmaFetchCostModel    fetchCM(getGraph());
    DmaEvictionCostModel evictCM(getGraph(), bundle, strategy);

    StrategyCostModel strategyCost(getGraph(), bundle, strategy, getSlicingBrains());
    strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

    const auto narrowSliceSize = SlicedOperandUtils::getSliceSizeInBytes(strategy->getMmeSlicingData().getNarrow());

    ASSERT_EQ(strategyCost.getTpcCost().total.hbmTrafficBytes,
              inAProducer->getTotalSizeInBytes() + inBProducer->getTotalSizeInBytes() +
                  outConsumer->getTotalSizeInBytes());
    ASSERT_EQ(strategyCost.getMmeCost().total.hbmTrafficBytes, 0);
    // The first row is fetched by TPC.
    // The second row is fetched by DMA, since we have double buffer the the last two narrow slices can be reused.
    ASSERT_EQ(strategyCost.getFetchCost(CostModel::Cost::Engine::DMA).total.hbmTrafficBytes,
              narrowSliceSize * (numOfNarrowSlices - 2));
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::DMA));
    assertZeroAccumulatedCost(strategyCost.getEvictionCost(CostModel::Cost::Engine::TPC));
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_slave_mme_flatten)
{
    std::vector<TSize>    sharedInSizes = {819, 64, 1, 1};
    pTensor               sharedIn      = createTensor(sharedInSizes, syn_type_single);
    std::vector<TSize>    convInSizes   = {64, 64, 64, 8};
    pTensor               convIn        = createTensor(convInSizes, syn_type_single);
    std::vector<TSize>    convOutSizes  = {819, 64, 64, 8};
    pTensor               convOut       = createTensor(convOutSizes, syn_type_single);
    std::vector<TSize>    dedxInSizes   = {819, 8, 8, 8};
    pTensor               dedxIn        = createTensor(dedxInSizes, syn_type_single);
    std::vector<TSize>    dedxOutSizes  = {64, 8, 8, 8};
    pTensor               dedxOut       = createTensor(dedxOutSizes, syn_type_single);

    synConvolutionParams convParams;
    pNode conv = NodeFactory::createNode({convIn, sharedIn}, {convOut}, &convParams, "spatial_convolution", "CONV");
    GraphEditor::addNode(getGraph(), conv);

    synConvolutionParams dedxParams;
    pNode                dedx = NodeFactory::createNode({dedxIn, sharedIn}, {dedxOut}, &dedxParams, "dedx", "DEDX");
    GraphEditor::addNode(getGraph(), dedx);

    AllBrains          allBrains {getGraph()};
    SRAMSlicingManager sramManager {getGraph()};
    sramManager.generateInitialBundles();
    sramManager.generateInitialStrategies();
    auto bundlesSolvingData = sramManager.getBundlesSolvingData();
    ASSERT_EQ(bundlesSolvingData.size(), 2);

    BundleExpander bundleExpander {getGraph(), allBrains, sramManager.getBundlizer(), bundlesSolvingData};
    for (const auto& bundleSolvingData : sramManager.getBundlesSolvingData())
    {
        ASSERT_EQ(bundleSolvingData.first->getNodes().size(), 1);

        // The DEDX node should be the master MME, the CONV node should be stitched as slave MME.
        if (bundleSolvingData.first->getNodes()[0]->getNodeName() != "DEDX")
        {
            continue;
        }

        for (const auto& s : bundleExpander.generateExpandedStrategies(bundleSolvingData.first))
        {
            pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);

            for (auto& candidate : strategy->getMmeSlicingData().getRoleCandidates())
            {
                if (candidate && candidate->nodeToStitch)
                {
                    ASSERT_EQ(candidate->role, BundleExpansion::SharedInputConsumer);
                    allBrains.m_mmeSlaveBrain.addSharedOperandMme(candidate, strategy);
                }
            }

            MMECostModel         mmeCM(*getGraph().getHALReader());
            TPCCostModel         tpcCM(*getGraph().getHALReader());
            DmaFetchCostModel    fetchCM(getGraph());
            DmaEvictionCostModel evictCM(getGraph(), bundleSolvingData.first, strategy);

            StrategyCostModel strategyCost(getGraph(), bundleSolvingData.first, strategy, allBrains);
            strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);

            const uint64_t masterOutSize =
                strategy->getMmeSlicingData().masterOperand->originalTensor->getTotalSizeInBytes();
            uint64_t   slaveOutSize    = 0;
            const auto slaveOutOperand = strategy->getMmeSlicingData().getSlaveOutputOperand();
            if (slaveOutOperand)
            {
                slaveOutSize = slaveOutOperand->originalTensor->getTotalSizeInBytes();
            }

            ASSERT_EQ(strategyCost.getMmeCost().total.hbmTrafficBytes, masterOutSize + slaveOutSize);
        }
    }
}

TEST_F(SRAMStrategyCompareTest, strategy_cost_accumulator_unaligned_operands)
{
    const TSize    w   = 511;
    const TSize    k   = 2056;
    const TSize    c   = 171;
    pTensor        inA = createTensor({k, w}, syn_type_bf16, true);
    pTensor        inB = createTensor({c, k}, syn_type_bf16, true);
    pTensor        out = createTensor({c, w}, syn_type_bf16, true);

    synGEMMParams gemmParams {};
    pNode         gemm = NodeFactory::createNode({inA, inB}, {out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    GraphEditor::addNode(getGraph(), gemm);

    Bundlizer bundlizer(getGraph());
    pBundle   bundle = bundlizer.getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);

    // A and B are aligned.
    pMmeSlicingStrategy strategyBothAligned =
        MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), gemm);
    strategyBothAligned->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setDoubleBuffer(true);
    strategyBothAligned->getMmeSlicingData().bundleTensors[0]->alignWithCacheLine = true;
    strategyBothAligned->getMmeSlicingData().bundleTensors[1]->alignWithCacheLine = true;
    MMECostModel         mmeCM(*getGraph().getHALReader());
    TPCCostModel         tpcCM(*getGraph().getHALReader());
    DmaFetchCostModel    fetchCM(getGraph());
    DmaEvictionCostModel evictCM(getGraph(), bundle, strategyBothAligned);
    StrategyCostModel    strategyCostBothAligned(getGraph(), bundle, strategyBothAligned, getSlicingBrains());
    strategyCostBothAligned.model(mmeCM, tpcCM, fetchCM, evictCM);
    const auto bothAlignedMmeTime = strategyCostBothAligned.getMmeCost().total.timeNano;

    // A and B are unaligned.
    pMmeSlicingStrategy strategyBothUnaligned =
        std::static_pointer_cast<MmeSlicingStrategy>(strategyBothAligned->clone(false));
    strategyBothUnaligned->getMmeSlicingData().bundleTensors[0]->alignWithCacheLine = false;
    strategyBothUnaligned->getMmeSlicingData().bundleTensors[1]->alignWithCacheLine = false;
    StrategyCostModel strategyCostBothUnaligned(getGraph(), bundle, strategyBothUnaligned, getSlicingBrains());
    strategyCostBothUnaligned.model(mmeCM, tpcCM, fetchCM, evictCM);
    const auto bothUnalignedMmeTime = strategyCostBothUnaligned.getMmeCost().total.timeNano;

    // A is unaligned, B is aligned.
    pMmeSlicingStrategy strategyAUnaligned =
        std::static_pointer_cast<MmeSlicingStrategy>(strategyBothAligned->clone(false));
    strategyAUnaligned->getMmeSlicingData().bundleTensors[0]->alignWithCacheLine = false;
    strategyAUnaligned->getMmeSlicingData().bundleTensors[1]->alignWithCacheLine = true;
    StrategyCostModel strategyCostAUnaligned(getGraph(), bundle, strategyAUnaligned, getSlicingBrains());
    strategyCostAUnaligned.model(mmeCM, tpcCM, fetchCM, evictCM);
    const auto aUnalignedMmeTime = strategyCostAUnaligned.getMmeCost().total.timeNano;

    // A is aligned, B is unaligned.
    pMmeSlicingStrategy strategyBUnaligned =
        std::static_pointer_cast<MmeSlicingStrategy>(strategyBothAligned->clone(false));
    strategyBUnaligned->getMmeSlicingData().bundleTensors[0]->alignWithCacheLine = true;
    strategyBUnaligned->getMmeSlicingData().bundleTensors[1]->alignWithCacheLine = false;
    StrategyCostModel strategyCostBUnaligned(getGraph(), bundle, strategyBUnaligned, getSlicingBrains());
    strategyCostBUnaligned.model(mmeCM, tpcCM, fetchCM, evictCM);
    const auto bUnalignedMmeTime = strategyCostBUnaligned.getMmeCost().total.timeNano;

    // Output size is 511x171, the MME selects 256x64 geometry so we have 2x3 activations.
    // The MME walks right and then down (skf) so A operand is reused.
    // The SB reuse factor will be 3 (171/64).
    const float sbReuseFactor = 3;

    // Unaligned input reads cost 2X BW.
    // 2 operands are unaligned – 2X slower.
    // 1 operand is unaligned and is reused in SB – ((1+SBReuseFactore)/SBReuseFactor)X slower.
    // 1 operand is unaligned and the other operand is reused in SB – 2X slower.
    EXPECT_NEAR(bothUnalignedMmeTime, bothAlignedMmeTime * 2, 1);
    EXPECT_NEAR(aUnalignedMmeTime, bothAlignedMmeTime * ((sbReuseFactor + 1) / sbReuseFactor), 1);
    EXPECT_NEAR(bUnalignedMmeTime, bothAlignedMmeTime * 2, 1);
}

class SRAMStrategyCompareBatchGemmTest
: public SRAMStrategyCompareTest
, public testing::WithParamInterface<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, bool>>
// w, k, c, b1, b2, 2X mode
{
public:
    SRAMStrategyCompareBatchGemmTest()
    : m_w(std::get<0>(GetParam())),
      m_k(std::get<1>(GetParam())),
      m_c(std::get<2>(GetParam())),
      m_b1(std::get<3>(GetParam())),
      m_b2(std::get<4>(GetParam())),
      m_is2xMode(std::get<5>(GetParam()))
    {
    }

protected:
    const TSize m_w;
    const TSize m_k;
    const TSize m_c;
    const TSize m_b1;
    const TSize m_b2;
    const bool  m_is2xMode;
};

TEST_P(SRAMStrategyCompareBatchGemmTest, validate_batch_gemm_cost)
{
    pTensor gemmIn1 = createTensor({m_k, m_w}, syn_type_single, true);
    pTensor gemmIn2 = createTensor({m_c, m_k}, syn_type_single, true);
    pTensor gemmOut = createTensor({m_c, m_w}, syn_type_single, true);

    pTensor bgemmIn1 = createTensor({m_k, m_w, m_b1, m_b2}, syn_type_single, true);
    pTensor bgemmIn2 = createTensor({m_c, m_k, m_b1, m_b2}, syn_type_single, true);
    pTensor bgemmOut = createTensor({m_c, m_w, m_b1, m_b2}, syn_type_single, true);

    synGEMMParams gemmParams {};

    pNode gemm =
        NodeFactory::createNode({gemmIn1, gemmIn2}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    GraphEditor::addNode(getGraph(), gemm);

    pNode bgemm = NodeFactory::createNode({bgemmIn1, bgemmIn2},
                                          {bgemmOut},
                                          &gemmParams,
                                          NodeFactory::batchGemmNodeTypeName,
                                          "BGEMM");
    GraphEditor::addNode(getGraph(), bgemm);

    Bundlizer   bundlizer(getGraph());
    const auto& bundles = bundlizer.getMMEBundles();
    ASSERT_EQ(bundles.size(), 2);

    uint64_t singleGemmMmeTime = 0;
    uint64_t batchGemmMmeTime  = 0;
    for (const auto& bundle : bundles)
    {
        bool                isBatchGemmBundle = bundle->getNodes().front()->isBatchGemm();
        pMmeSlicingStrategy strategy =
            MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), isBatchGemmBundle ? bgemm : gemm);
        strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setDoubleBuffer(true);
        MMECostModel         mmeCM(*getGraph().getHALReader());
        TPCCostModel         tpcCM(*getGraph().getHALReader());
        DmaFetchCostModel    fetchCM(getGraph());
        DmaEvictionCostModel evictCM(getGraph(), bundle, strategy);
        StrategyCostModel    strategyCost(getGraph(), bundle, strategy, getSlicingBrains());
        strategyCost.model(mmeCM, tpcCM, fetchCM, evictCM);
        if (isBatchGemmBundle)
        {
            batchGemmMmeTime = strategyCost.getMmeCost().total.timeNano;
        }
        else
        {
            singleGemmMmeTime = strategyCost.getMmeCost().total.timeNano;
        }
    }
    ASSERT_TRUE(singleGemmMmeTime > 0);
    ASSERT_TRUE(batchGemmMmeTime > 0);

    if (m_is2xMode)
    {
        // When 2wx2h geometry is used and each GEMM is small enough to fit into half of the MME EUs -
        // 2 GEMMs are executed concurrently.
        ASSERT_NEAR((float)batchGemmMmeTime / (std::ceil(m_b1 / 2.f) * m_b2), singleGemmMmeTime, 1);
    }
    else
    {
        ASSERT_NEAR((float)batchGemmMmeTime / (m_b1 * m_b2), singleGemmMmeTime, 1);
    }
}

INSTANTIATE_TEST_SUITE_P(strategy_cost_accumulator_batch_gemm,
                         SRAMStrategyCompareBatchGemmTest,
                         ::testing::Values(
                             // w, k, c, b1, b2, 2X mode
                             std::make_tuple(256, 512, 256, 15, 8, false),
                             std::make_tuple(512, 95, 943, 1, 83, false),
                             std::make_tuple(32, 512, 32, 100, 5, true),
                             std::make_tuple(16, 443, 16, 17, 10, true)));

}  // namespace gaudi