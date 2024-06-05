#include "sram_management_fe_test.h"
#include <passes/sram_management/strategy_cost_model.h>
#include <passes/sram_management/engine_cost_model.h>
#include "graph_compiler/passes/sram_management/solution_generator.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "tpc_slicing_test_infra.h"

namespace gaudi
{
bool sliceGraphToSRAMCapacity(HabanaGraph& g);

class EngineCostModelTest : public SRAMManagementTest
{
public:
    using CostEngine = CostModel::Cost::Engine;

    void SetUp() override
    {
        SRAMManagementTest::SetUp();
        StrategyCostModel::reset();  // Clear the invalid candidates cache
    }

    // Mock cost model
    class MockCostModel : public gaudi::CostModel
    {
    public:
        MockCostModel(Cost::Engine engine): m_cost(engine)
        {
        }

        Cost calcCost(const pNode& node,
                      const SliceReferenceList& inputs,
                      const SliceReferenceList& outputs) const override
        {
            return m_cost;
        }

    private:
        Cost m_cost;
    };

    MockCostModel mmeMockCostModel{CostEngine::MME};
    MockCostModel tpcMockCostModel{CostEngine::TPC};
    MockCostModel fetchMockCostModel{CostEngine::DMA};
    MockCostModel evictMockCostModel{CostEngine::DMA};

};

TEST_F(EngineCostModelTest, mme_cost_model_estimate_hbm_traffic)
{
    // Given a single MME bundle which is sliced to 13 operations:
    constexpr TSize
            w = 1024,
            k = 256,
            c = 128;

    constexpr unsigned numSlices = 13; // a number that will create an "edge" (last slice that's smaller than the other)

    for (const auto& elementType : {syn_type_bf16, syn_type_float})
    {
        pTensor a = createTensor({c, w}, elementType);
        pTensor b = createTensor({k, c}, elementType);
        pTensor o = createTensor({k, w}, elementType);
        synGEMMParams gemmParams{};
        pNode gemm = NodeFactory::createNode({a, b}, {o}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
        GraphEditor::addNode(getGraph(), gemm);

        Bundlizer bundlizer(getGraph());
        pBundle bundle = bundlizer.getMMEBundles().front();
        ASSERT_EQ(bundle->getNodes().size(), 1);

        pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), gemm);
        auto &sd = strategy->getMmeSlicingData();
        sd.bundleTensors[0]->chunkDimensions[DIM_W] = sd.masterOperand->chunkDimensions[DIM_W] = w / numSlices;
        sd.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_K] = sd.masterOperand->chunkDimensions[WEIGHT_DIM_K] = k / numSlices;
        // set both inputs to be in sram
        strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
        MMECostModel      mmeCM(*getGraph().getHALReader());
        StrategyCostModel strategyCostModel(getGraph(), bundle, strategy, getSlicingBrains());
        strategyCostModel.model(mmeCM, tpcMockCostModel, fetchMockCostModel, evictMockCostModel);
        const AccumulatedCost& accMMECost = strategyCostModel.getMmeCost();

        // expect the total hbm traffic to match the output size (because the inputs are in sram , the hbm traffic
        // responsible for bringing them to the sram is calculated in the dma cost model)
        uint64_t expectedHbmTraffic = o->getDenseSizeInBytes();
        ASSERT_EQ(accMMECost.total.hbmTrafficBytes, expectedHbmTraffic);
    }
}

TEST_F(EngineCostModelTest, tpc_cost_model_estimate_hbm_traffic)
{
    // Given a single MME bundle which is sliced to 13 operations:
    constexpr TSize
            b = 16,
            h = 256,
            w = 256,
            c = 32,
            r = 1,
            s = 1,
            k = 512;

    constexpr unsigned numSlices = 13; // a number that will create an "edge" (last slice that's smaller than the other)

    for (const auto& elementType : {syn_type_bf16, syn_type_float})
    {
        pTensor in          = createTensor({c, w, h, b}, elementType);
        pTensor tpcOutMmeIn = createTensor({c, w, h, b}, elementType);

        // Create a sliceable TPC producer node (slicing granularity on each dim is 1)
        pNode tpcNode = TPCCustomIndexSpaceNode::createSliceableNode(in, tpcOutMmeIn);
        GraphEditor::addNode(getGraph(), tpcNode);

        // create an mme node just so we'll have a normal "mme based" bundle
        pTensor wgh          = createTensor({k, c, s, r}, elementType);
        pTensor mmeOut       = createTensor({k, w, h, b}, elementType);
        synConvolutionParams convParams{};
        pNode                conv = NodeFactory::createNode({tpcOutMmeIn, wgh},
                                             {mmeOut},
                                             &convParams,
                                             NodeFactory::convolutionNodeTypeName,
                                             "conv");
        GraphEditor::addNode(getGraph(), conv);

        ASSERT_TRUE(loadTpcKernels(getGraph()));

        Bundlizer bundlizer(getGraph());
        pBundle bundle = bundlizer.getMMEBundles().front();
        ASSERT_EQ(bundle->getNodes().size(), 1);

        pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), conv);
        auto &sd = strategy->getMmeSlicingData();
        sd.bundleTensors[0]->chunkDimensions[DIM_W] = sd.masterOperand->chunkDimensions[DIM_W] = w / numSlices;
        sd.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_K] = sd.masterOperand->chunkDimensions[WEIGHT_DIM_K] = k / numSlices;
        // set both inputs in sram
        strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);

        // add tpc to bundle and strategy
        pBundleExpansion candidate = bundlizer.findWideTpcProducerExpansionCandidate(strategy);
        ASSERT_NE(candidate, nullptr);
        ASSERT_EQ(candidate->nodeToStitch, tpcNode);
        TPCSlaveBrain slaveBrain(getGraph());
        sd.addValidCandidate(candidate);
        slaveBrain.addProducerToStrategy(candidate, strategy);

        TPCCostModel      tpcCM(*getGraph().getHALReader());
        StrategyCostModel strategyCostModel(getGraph(), bundle, strategy, getSlicingBrains());
        strategyCostModel.model(mmeMockCostModel, tpcCM, fetchMockCostModel, evictMockCostModel);
        const AccumulatedCost& accTPCCost = strategyCostModel.getTpcCost();

        // expect the total TPC hbm traffic to match the input size (because the output is in sram and is consumed
        // directly by the mme
        uint64_t expectedHbmTraffic = in->getDenseSizeInBytes();
        ASSERT_EQ(accTPCCost.total.hbmTrafficBytes, expectedHbmTraffic);
    }
}


class DmaCostModelTestMme : public EngineCostModelTest, public testing::WithParamInterface<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(EngineCostModelTest, DmaCostModelTestMme,
                        ::testing::Combine(
                                ::testing::Range(1, 10), // numSlicesA
                                ::testing::Range(1, 10)  // numSlicesB
                        ));

TEST_P(DmaCostModelTestMme, estimate_hbm_traffic)
{
    int numSlicesA = std::get<0>(GetParam());
    int numSlicesB = std::get<1>(GetParam());

    // Given a single MME bundle which is sliced to 13 operations:
    constexpr TSize
            w = 1024,
            k = 2048,
            c = 2000;

    synDataType elementType = syn_type_float;
    pTensor a = createTensor({c, w}, elementType);
    pTensor b = createTensor({k, c}, elementType);
    pTensor o = createTensor({k, w}, elementType);
    synGEMMParams gemmParams{};
    pNode gemm = NodeFactory::createNode({a, b}, {o}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");
    GaudiGraph graph;
    GraphEditor::addNode(graph, gemm);

    Bundlizer bundlizer(graph);
    pBundle bundle = bundlizer.getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1);

    // create a specific strategy
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*graph.getHALReader(), gemm);
    auto &sd = strategy->getMmeSlicingData();
    sd.bundleTensors[0]->chunkDimensions[DIM_W] =
    sd.masterOperand->chunkDimensions[DIM_W] = w / numSlicesA;
    sd.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_K] =
    sd.masterOperand->chunkDimensions[WEIGHT_DIM_K] = k / numSlicesB;
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setDoubleBuffer(true);

    // estimate the strategy using the cost model:
    DmaFetchCostModel fetchCM(graph);
    StrategyCostModel strategyCostModel(graph, bundle, strategy, getSlicingBrains());
    strategyCostModel.model(mmeMockCostModel, tpcMockCostModel, fetchCM, evictMockCostModel);
    const AccumulatedCost& dmaFetchCost = strategyCostModel.getFetchCost(CostEngine::DMA);

    // create the actual solution and let the slicer create the final graph from this strategy:
    SolutionGenerator generator(graph, bundle, strategy);
    ASSERT_TRUE(generator.fillSolution());
    BundleSlicer::sliceBundle(*bundle, graph);
    // calculate hbm traffic from the actual memcpy (=dma) nodes in the final graph
    uint64_t actualDmaFetchHbmTrafficBytes = 0;
    for (pNode n : graph.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MEMCOPY)
        {
            const auto &in = n->getInput(0), out = n->getOutput(0);
            if (!in->inSram() && out->inSram())
            {
                ASSERT_EQ(in->getDenseSizeInBytes(), out->getDenseSizeInBytes());
                actualDmaFetchHbmTrafficBytes += in->getDenseSizeInBytes();
            }
        }
    }
    // the cost model result should match the reality:
    ASSERT_EQ(dmaFetchCost.total.hbmTrafficBytes, actualDmaFetchHbmTrafficBytes);
}


class DmaCostModelTestTpcProducer : public EngineCostModelTest, public testing::WithParamInterface<std::tuple<int, int, bool>> {};

INSTANTIATE_TEST_SUITE_P(EngineCostModelTest, DmaCostModelTestTpcProducer,
                        ::testing::Combine(
                                ::testing::Values(1, 2, 3, 4, 8, 20), // numSlicesA
                                ::testing::Values(1, 2, 3, 4, 8, 20), // numSlicesB
                                ::testing::Bool() // stitchedTensorIsPersistent
                        ));

TEST_P(DmaCostModelTestTpcProducer, estimate_hbm_traffic)
{
    int numSlicesA = std::get<0>(GetParam());
    int numSlicesB = std::get<1>(GetParam());
    bool stitchedTensorIsPersistent = std::get<2>(GetParam());

    // Given a single MME bundle which is sliced to 13 operations:
    constexpr TSize
            w = 1024,
            k = 2048,
            c = 2000;

    synDataType elementType = syn_type_float;

    pTensor a = createTensor({c, w}, elementType, stitchedTensorIsPersistent);
    pTensor b = createTensor({k, c}, elementType);
    pTensor o = createTensor({k, w}, elementType);
    synGEMMParams gemmParams{};
    pNode gemm = NodeFactory::createNode({a, b}, {o}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");

    pTensor in = createTensor({c, w}, elementType);
    // Create a sliceable TPC producer node (slicing granularity on each dim is 1)
    pNode tpcNode = TPCCustomIndexSpaceNode::createSliceableNode(in, a);

    GaudiGraph graph;
    GraphEditor::addNode(graph, tpcNode);
    GraphEditor::addNode(graph, gemm);

    ASSERT_TRUE(loadTpcKernels(graph));

    Bundlizer bundlizer(graph);
    pBundle bundle = bundlizer.getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1); // only mme node at this stage

    // create a specific strategy
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*graph.getHALReader(), gemm);
    auto &sd = strategy->getMmeSlicingData();
    sd.bundleTensors[0]->chunkDimensions[DIM_W] =
    sd.masterOperand->chunkDimensions[DIM_W] = w / numSlicesA;
    sd.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_K] =
    sd.masterOperand->chunkDimensions[WEIGHT_DIM_K] = k / numSlicesB;
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setDoubleBuffer(true);

    // make sure it's possible to add the tpc producer to this bundle (in other words, that tpc stitching is
    // allowed in this scenario), and add the tpc node:
    pBundleExpansion expCnd = bundlizer.findWideTpcProducerExpansionCandidate(strategy);
    ASSERT_EQ(expCnd->nodeToStitch, tpcNode);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, a);
    TPCSlaveBrain tpcBrain{graph};
    ASSERT_TRUE(tpcBrain.addProducerToStrategy(expCnd, strategy));
    bundle->addNode(tpcNode);

    // estimate the strategy using the cost models:
    DmaFetchCostModel    fetchCM(graph);
    DmaEvictionCostModel evictCM(graph, bundle, strategy);
    StrategyCostModel    strategyCostModel(graph, bundle, strategy, getSlicingBrains());
    strategyCostModel.model(mmeMockCostModel, tpcMockCostModel, fetchCM, evictCM);
    const AccumulatedCost& dmaFetchCost = strategyCostModel.getFetchCost(CostEngine::DMA);
    const AccumulatedCost& dmaEvictionCost = strategyCostModel.getEvictionCost(CostEngine::DMA);

    // create the actual solution and let the slicer create the final graph from this strategy:
    SolutionGenerator generator(graph, bundle, strategy);
    ASSERT_TRUE(generator.fillSolution());
    BundleSlicer::sliceBundle(*bundle, graph);
    // calculate hbm traffic from the actual memcpy (=dma) nodes in the final graph
    uint64_t actualDmaFetchHbmTrafficBytes = 0, actualDmaEvictionHbmTrafficBytes = 0;
    for (pNode n : graph.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MEMCOPY)
        {
            const auto &in = n->getInput(0), out = n->getOutput(0);
            if ((in->inSram() && !out->inSram()) ||
                (!in->inSram() && out->inSram()))
            {
                ASSERT_EQ(in->getDenseSizeInBytes(), out->getDenseSizeInBytes());
                actualDmaFetchHbmTrafficBytes    += out->inSram() ? in->getDenseSizeInBytes() : 0;
                actualDmaEvictionHbmTrafficBytes += out->inSram() ? 0 : in->getDenseSizeInBytes();
            }
        }
    }

    // the cost models result should match the reality:
    ASSERT_EQ(dmaFetchCost.total.hbmTrafficBytes, actualDmaFetchHbmTrafficBytes);
    ASSERT_EQ(dmaEvictionCost.total.hbmTrafficBytes, actualDmaEvictionHbmTrafficBytes);
}

class DmaCostModelTestTpcConsumer : public EngineCostModelTest, public testing::WithParamInterface<std::tuple<int, int, bool>> {};

INSTANTIATE_TEST_SUITE_P(EngineCostModelTest, DmaCostModelTestTpcConsumer,
                        ::testing::Combine(
                                ::testing::Values(1, 2, 3, 4, 8, 10), // numSlicesA
                                ::testing::Values(1, 2, 3, 4, 8, 10), // numSlicesB
                                ::testing::Bool() // stitchedTensorIsPersistent
                        ));

TEST_P(DmaCostModelTestTpcConsumer, estimate_hbm_traffic)
{
    int numSlicesA = std::get<0>(GetParam());
    int numSlicesB = std::get<1>(GetParam());
    bool stitchedTensorIsPersistent = std::get<2>(GetParam());

    // Given a single MME bundle which is sliced to 13 operations:
    constexpr TSize
            w = 1024,
            k = 2048,
            c = 2000;

    synDataType elementType = syn_type_bf16;

    pTensor a = createTensor({c, w}, elementType);
    pTensor b = createTensor({k, c}, elementType);
    pTensor mmeOut = createTensor({k, w}, elementType, stitchedTensorIsPersistent);
    synGEMMParams gemmParams{};
    pNode gemm = NodeFactory::createNode({a, b}, {mmeOut}, &gemmParams, NodeFactory::gemmNodeTypeName,
                                         "GEMM");

    pTensor tpcOut = createTensor({k, w}, elementType);
    // Create a sliceable TPC consumer node (slicing granularity on each dim is 1)
    pNode tpcNode = TPCCustomIndexSpaceNode::createSliceableNode(mmeOut, tpcOut);

    GaudiGraph graph;
    GraphEditor::addNode(graph, gemm);
    GraphEditor::addNode(graph, tpcNode);

    ASSERT_TRUE(loadTpcKernels(graph));

    Bundlizer bundlizer(graph);
    pBundle bundle = bundlizer.getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1); // only mme node at this stage

    // create a specific strategy
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*graph.getHALReader(), gemm);
    auto &sd = strategy->getMmeSlicingData();
    sd.bundleTensors[0]->chunkDimensions[DIM_W] =
    sd.masterOperand->chunkDimensions[DIM_W] = w / numSlicesA;
    sd.bundleTensors[1]->chunkDimensions[WEIGHT_DIM_K] =
    sd.masterOperand->chunkDimensions[WEIGHT_DIM_K] = k / numSlicesB;
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setDoubleBuffer(true);

    // make sure it's possible to add the tpc producer to this bundle (in other words, that tpc stitching is
    // allowed in this scenario), and add the tpc node:
    pBundleExpansion expCnd = bundlizer.findTpcConsumerExpansionCandidate(strategy);
    ASSERT_EQ(expCnd->nodeToStitch, tpcNode);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, mmeOut);
    TPCSlaveBrain tpcBrain{graph};
    ASSERT_TRUE(tpcBrain.addConsumerToStrategy(expCnd, strategy));
    bundle->addNode(tpcNode);

    // estimate the strategy using the cost models:
    DmaFetchCostModel    fetchCM(graph);
    DmaEvictionCostModel evictCM(graph, bundle, strategy);
    StrategyCostModel    strategyCostModel(graph, bundle, strategy, getSlicingBrains());
    strategyCostModel.model(mmeMockCostModel, tpcMockCostModel, fetchCM, evictCM);
    const AccumulatedCost& dmaFetchCost = strategyCostModel.getFetchCost(CostEngine::DMA);
    const AccumulatedCost& dmaEvictionCost = strategyCostModel.getEvictionCost(CostEngine::DMA);

    // create the actual solution and let the slicer create the final graph from this strategy:
    SolutionGenerator generator(graph, bundle, strategy);
    ASSERT_TRUE(generator.fillSolution());
    BundleSlicer::sliceBundle(*bundle, graph);
    // calculate hbm traffic from the actual memcpy (=dma) nodes in the final graph
    uint64_t actualDmaFetchHbmTrafficBytes = 0, actualDmaEvictionHbmTrafficBytes = 0;
    for (pNode n : graph.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MEMCOPY)
        {
            const auto &in = n->getInput(0), out = n->getOutput(0);
            if ((in->inSram() && !out->inSram()) ||
                (!in->inSram() && out->inSram()))
            {
                ASSERT_EQ(in->getDenseSizeInBytes(), out->getDenseSizeInBytes());
                actualDmaFetchHbmTrafficBytes    += out->inSram() ? in->getDenseSizeInBytes() : 0;
                actualDmaEvictionHbmTrafficBytes += out->inSram() ? 0 : in->getDenseSizeInBytes();
            }
        }
    }
    // the cost models result should match the reality:
    ASSERT_EQ(dmaFetchCost.total.hbmTrafficBytes, actualDmaFetchHbmTrafficBytes);
    ASSERT_EQ(dmaEvictionCost.total.hbmTrafficBytes, actualDmaEvictionHbmTrafficBytes);
}


class DmaCostModelTestPartials : public EngineCostModelTest, public testing::WithParamInterface<std::tuple<int, int, int, bool, synDataType>> {};

INSTANTIATE_TEST_SUITE_P(EngineCostModelTest, DmaCostModelTestPartials,
                        ::testing::Combine(
                                ::testing::Values(2, 4), // numSlicesCD
                                ::testing::Values(2, 4), // numSlicesDy
                                ::testing::Values(2, 4), // numSlicesX
                                ::testing::Bool(), // outputIsPersistent
                                ::testing::ValuesIn({syn_type_bf16, syn_type_float}) // elementType
                        ));

TEST_P(DmaCostModelTestPartials, estimate_hbm_traffic)
{
    int  numSlicesCD        = std::get<0>(GetParam());
    int  numSlicesDy        = std::get<1>(GetParam());
    int  numSlicesX         = std::get<2>(GetParam());
    bool outputIsPersistent = std::get<3>(GetParam());
    synDataType elementType = std::get<4>(GetParam());

    const TSize b = 10, hX = 140, wX = 140;
    const TSize c = 192, k = 256;
    synConvolutionParams params;
    params.dW = 2, params.dH = 2;

    std::vector<TSize> xSizes  = {c, wX, hX, b};
    std::vector<TSize> dwSizes = {k, c, params.kW, params.kH};
    std::vector<TSize> dySizes = {k,
                                  convOutputDimSize(wX, params.kW, params.dW, params.padL + params.padR, params.dilW),
                                  convOutputDimSize(hX, params.kH, params.dH, params.padT + params.padB, params.dilH),
                                  b}; // {k,wX/2,hX/2,b}

    pTensor dy = createTensor(dySizes, elementType);
    pTensor x = createTensor(xSizes, elementType);
    pTensor dw = createTensor(dwSizes, elementType, outputIsPersistent);
    pNode dedw = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");

    GaudiGraph graph;
    GraphEditor::addNode(graph, dedw);

    Bundlizer bundlizer(graph);
    pBundle bundle = bundlizer.getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1); // only mme node at this stage

    // create a specific strategy
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*graph.getHALReader(), dedw);
    auto &sd = strategy->getMmeSlicingData();
    sd.bundleTensors[0]->chunkDimensions[DIM_B] =
    sd.bundleTensors[1]->chunkDimensions[DIM_B] = b / numSlicesCD;
    sd.bundleTensors[0]->chunkDimensions[WEIGHT_DIM_K] =
    sd.masterOperand->chunkDimensions[WEIGHT_DIM_K] = k / numSlicesDy;
    sd.bundleTensors[1]->chunkDimensions[DIM_C] =
    sd.masterOperand->chunkDimensions[WEIGHT_DIM_C] = c / numSlicesX;
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setOutputIsInSRAM(
            true).setDoubleBuffer(true);
    // simulate the common solver - set output to float (to support correct accumulation of partials)
    sd.masterOperand->finalElementType = syn_type_float;

    // estimate the strategy using the cost models:
    DmaFetchCostModel    fetchCM(graph);
    DmaEvictionCostModel evictCM(graph, bundle, strategy);
    StrategyCostModel    strategyCostModel(graph, bundle, strategy, getSlicingBrains());
    strategyCostModel.model(mmeMockCostModel, tpcMockCostModel, fetchCM, evictCM);
    const AccumulatedCost& dmaFetchCost = strategyCostModel.getFetchCost(CostEngine::DMA);
    const AccumulatedCost& dmaEvictionCost = strategyCostModel.getEvictionCost(CostEngine::DMA);
    const AccumulatedCost& tpcEvictionCost = strategyCostModel.getEvictionCost(CostEngine::TPC);

    // create the actual solution and let the slicer create the final graph from this strategy:
    SolutionGenerator generator(graph, bundle, strategy);
    ASSERT_TRUE(generator.fillSolution());
    BundleSlicer::sliceBundle(*bundle, graph);
    // calculate hbm traffic from the actual memcpy (=dma) nodes in the final graph
    uint64_t actualDmaFetchHbmTrafficBytes = 0, actualDmaEvictionHbmTrafficBytes = 0, actualTpcEvictionHbmTrafficBytes = 0;
    for (pNode n : graph.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MEMCOPY)
        {
            const auto &in = n->getInput(0), out = n->getOutput(0);
            if ((in->inSram() && out->inDram()) || (in->inDram() && out->inSram()))
            {
                if (in->getElementType() == out->getElementType())
                {
                    ASSERT_EQ(in->getDenseSizeInBytes(), out->getDenseSizeInBytes());
                    actualDmaFetchHbmTrafficBytes += out->inSram() ? in->getDenseSizeInBytes() : 0;
                    actualDmaEvictionHbmTrafficBytes += out->inSram() ? 0 : in->getDenseSizeInBytes();
                }
                else
                {
                    // if data types of input and output are different, this copy will not become a dma
                    // node  eventually, but a tpc cast node
                    actualTpcEvictionHbmTrafficBytes += out->getDenseSizeInBytes();
                }
            }
        }
    }
    // the cost models result should match the reality:
    ASSERT_EQ(dmaFetchCost.total.hbmTrafficBytes, actualDmaFetchHbmTrafficBytes);
    ASSERT_EQ(dmaEvictionCost.total.hbmTrafficBytes, actualDmaEvictionHbmTrafficBytes);
    ASSERT_EQ(tpcEvictionCost.total.hbmTrafficBytes, actualTpcEvictionHbmTrafficBytes);
}


class DmaCostModelTestPartialsWithTpcProducer : public EngineCostModelTest, public testing::WithParamInterface<std::tuple<int, int, int, bool, synDataType, bool>> {};

INSTANTIATE_TEST_SUITE_P(EngineCostModelTest, DmaCostModelTestPartialsWithTpcProducer,
                        ::testing::Combine(
                                ::testing::Values(3, 4), // numSlicesCD
                                ::testing::Values(2, 4), // numSlicesDy
                                ::testing::Values(3, 5), // numSlicesX
                                ::testing::Bool(), // stitchedTensorIsPersistent
                                ::testing::ValuesIn({syn_type_bf16, syn_type_float}), // elementType
                                ::testing::Bool() // stitchedTo1stInput
                        ));

TEST_P(DmaCostModelTestPartialsWithTpcProducer, estimate_hbm_traffic)
{
    int  numSlicesCD                = std::get<0>(GetParam());
    int  numSlicesDy                = std::get<1>(GetParam());
    int  numSlicesX                 = std::get<2>(GetParam());
    bool stitchedTensorIsPersistent = std::get<3>(GetParam());
    synDataType elementType         = std::get<4>(GetParam());
    bool stitchedTo1stInput         = std::get<5>(GetParam());

    const TSize b = 10, hX = 140, wX = 140;
    const TSize c = 192, k = 256;
    synConvolutionParams params;
    params.dW = 2, params.dH = 2;

    std::vector<TSize> xSizes  = {c, wX, hX, b};
    std::vector<TSize> dwSizes = {k, c, params.kW, params.kH};
    std::vector<TSize> dySizes = {k,
                                  convOutputDimSize(wX, params.kW, params.dW, params.padL + params.padR, params.dilW),
                                  convOutputDimSize(hX, params.kH, params.dH, params.padT + params.padB, params.dilH),
                                  b}; // {k,wX/2,hX/2,b}

    pTensor dy = createTensor(dySizes, elementType);
    pTensor x = createTensor(xSizes, elementType);
    pTensor dw = createTensor(dwSizes, elementType, stitchedTensorIsPersistent);
    pNode dedw = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName,
                                         "dedw");

    pTensor tpcIn = createTensor(stitchedTo1stInput ? dySizes : xSizes, elementType);
    // Create a sliceable TPC producer node (slicing granularity on each dim is 1)
    pNode tpcNode = TPCCustomIndexSpaceNode::createSliceableNode(tpcIn, stitchedTo1stInput ? dy : x);

    GaudiGraph graph;
    GraphEditor::addNode(graph, tpcNode);
    GraphEditor::addNode(graph, dedw);

    ASSERT_TRUE(loadTpcKernels(graph));

    Bundlizer bundlizer(graph);
    pBundle bundle = bundlizer.getMMEBundles().front();
    ASSERT_EQ(bundle->getNodes().size(), 1); // only mme node at this stage

    // create a specific strategy
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*graph.getHALReader(), dedw);
    auto &sd = strategy->getMmeSlicingData();
    sd.bundleTensors[0]->chunkDimensions[DIM_B] =
    sd.bundleTensors[1]->chunkDimensions[DIM_B] = b / numSlicesCD;
    sd.bundleTensors[0]->chunkDimensions[WEIGHT_DIM_K] =
    sd.masterOperand->chunkDimensions[WEIGHT_DIM_K] = k / numSlicesDy;
    sd.bundleTensors[1]->chunkDimensions[DIM_C] =
    sd.masterOperand->chunkDimensions[WEIGHT_DIM_C] = c / numSlicesX;
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).setOutputIsInSRAM(true).setDoubleBuffer(true);
    // simulate the common solver - set output to float (to support correct accumulation of partials)
    sd.masterOperand->finalElementType = syn_type_float;

    // make sure it's possible to add the tpc producer to this bundle (in other words, that tpc stitching is
    // allowed in this scenario), and add the tpc node:
    pBundleExpansion expCnd;
    if (stitchedTo1stInput)
    {
        expCnd = bundlizer.findNarrowTpcProducerExpansionCandidate(strategy);
    }
    else
    {
        expCnd = bundlizer.findWideTpcProducerExpansionCandidate(strategy);
    }
    ASSERT_EQ(expCnd->nodeToStitch, tpcNode);
    ASSERT_EQ(expCnd->stitchedOperand->originalTensor, stitchedTo1stInput? dy : x);
    TPCSlaveBrain tpcBrain{graph};
    ASSERT_TRUE(tpcBrain.addProducerToStrategy(expCnd, strategy));
    bundle->addNode(tpcNode);

    // estimate the strategy using the cost models:
    DmaFetchCostModel    fetchCM(graph);
    DmaEvictionCostModel evictCM(graph, bundle, strategy);
    StrategyCostModel    strategyCostModel(graph, bundle, strategy, getSlicingBrains());
    strategyCostModel.model(mmeMockCostModel, tpcMockCostModel, fetchCM, evictCM);
    const AccumulatedCost& dmaFetchCost = strategyCostModel.getFetchCost(CostEngine::DMA);
    const AccumulatedCost& dmaEvictionCost = strategyCostModel.getEvictionCost(CostEngine::DMA);
    const AccumulatedCost& tpcEvictionCost = strategyCostModel.getEvictionCost(CostEngine::TPC);

    // create the actual solution and let the slicer create the final graph from this strategy:
    SolutionGenerator generator(graph, bundle, strategy);
    ASSERT_TRUE(generator.fillSolution());
    BundleSlicer::sliceBundle(*bundle, graph);
    // calculate hbm traffic from the actual memcpy (=dma) nodes in the final graph
    uint64_t actualDmaFetchHbmTrafficBytes = 0, actualDmaEvictionHbmTrafficBytes = 0, actualTpcEvictionHbmTrafficBytes = 0;
    for (pNode n : graph.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MEMCOPY)
        {
            const auto &in = n->getInput(0), out = n->getOutput(0);
            if ((in->inSram() && out->inDram()) || (in->inDram() && out->inSram()))
            {
                if (in->getElementType() == out->getElementType())
                {
                    ASSERT_EQ(in->getDenseSizeInBytes(), out->getDenseSizeInBytes());
                    actualDmaFetchHbmTrafficBytes += out->inSram() ? in->getDenseSizeInBytes() : 0;
                    actualDmaEvictionHbmTrafficBytes += out->inSram() ? 0 : in->getDenseSizeInBytes();
                }
                else
                {
                    // if data types of input and output are different, this copy will not become a dma
                    // node  eventually, but a tpc cast node
                    actualTpcEvictionHbmTrafficBytes += out->getDenseSizeInBytes();
                }
            }
        }
    }
    // the cost models result should match the reality:
    ASSERT_EQ(dmaFetchCost.total.hbmTrafficBytes, actualDmaFetchHbmTrafficBytes);
    ASSERT_EQ(dmaEvictionCost.total.hbmTrafficBytes, actualDmaEvictionHbmTrafficBytes);
    ASSERT_EQ(tpcEvictionCost.total.hbmTrafficBytes, actualTpcEvictionHbmTrafficBytes);
}

} // namespace gaudi
