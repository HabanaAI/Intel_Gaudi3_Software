#include "gc_gaudi_test_infra.h"
#include "graph_compiler/passes/complex_guid_extractor.h"
#include "syn_singleton.hpp"

/* tests using dummy complex guid library */
class SynGaudiComplexGUIDExtractorDummyTest : public SynGaudiTestInfra
{
public:
    void afterSynInitialize() override
    {
        synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");  // must set it first so we can configure the below
        synConfigurationGet("COMPLEX_GUID_EXTRACTOR_MODE", &m_prevMode, 1);
        synConfigurationSet("COMPLEX_GUID_EXTRACTOR_MODE", "2");
        SynGaudiTestInfra::afterSynInitialize();
    }
    void SetUpTest() override
    {
        int ret = system("cd $SYNAPSE_ROOT/tests/dummyComplexGuid/ && cmake . && make");
        LOG_DEBUG(SYN_TEST, "dummy lib compile return val - {}", ret);
        SynGaudiTestInfra::SetUpTest();
    }
    void TearDownTest() override
    {
        int ret = system("rm $BUILD_ROOT_LATEST/libdummyComplexGuid.so");
        LOG_DEBUG(SYN_TEST, "dummy lib remove return val - {}", ret);
        GCFG_COMPLEX_GUID_EXTRACTOR_MODE.setValue(m_prevMode);
        SynGaudiTestInfra::TearDownTest();
    }

    char m_prevMode = '0';
};

TEST_F_GC(SynGaudiComplexGUIDExtractorDummyTest, DISABLED_test_complex_guid_extractor_trivial_extraction)
{
    /*
     * Test Case - Complex GUID node extracted to simple node
     * Complex Guid Node "complex_add_f32(x,y)", decomposed to add(x,neg(neg(y))
     */
    const unsigned C           = 3;
    const unsigned W           = 3;
    const unsigned H           = 3;
    const unsigned N           = 1;
    size_t         elementsNum = C * W * H * N;

    unsigned sizes[] = {C, W, H, N};

    unsigned x   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes);
    unsigned y   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes);
    unsigned out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes);

    addNodeToGraph("complex_add_f32", {x, y}, {out});
    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(1, graph->getExeSortedNodes().size());
    compileAndRun();
    ASSERT_EQ(3, graph->getExeSortedNodes().size());

    std::vector<float> inputX(elementsNum);
    inputX.assign((float*)m_hostBuffers[x], (float*)m_hostBuffers[x] + elementsNum);
    std::vector<float> inputY(elementsNum);
    inputY.assign((float*)m_hostBuffers[y], (float*)m_hostBuffers[y] + elementsNum);
    std::vector<float> outBuffer(elementsNum);
    outBuffer.assign((float*)m_hostBuffers[out], (float*)m_hostBuffers[out] + elementsNum);

    for (unsigned i = 0; i < elementsNum; i++)
    {
        float ref = inputX[i] + inputY[i];
        ASSERT_EQ(ref, outBuffer[i]);
    }
}

TEST_F_GC(SynGaudiComplexGUIDExtractorDummyTest, DISABLED_test_complex_guid_extractor_nested_extraction)
{
    /*
     * Test Case - Complex GUID node extracted to another complex GUID node
     * Complex Guid Node "nested_complex_add_f32(x,y)", decomposed to complex_add(neg(neg(x), y)
     */
    const unsigned C           = 3;
    const unsigned W           = 3;
    const unsigned H           = 3;
    const unsigned N           = 1;
    size_t         elementsNum = C * W * H * N;

    unsigned sizes[] = {C, W, H, N};

    unsigned x   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes);
    unsigned y   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes);
    unsigned out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes);

    addNodeToGraph("nested_complex_add_f32", {x, y}, {out});
    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(1, graph->getExeSortedNodes().size());
    compileAndRun();
    ASSERT_EQ(5, graph->getExeSortedNodes().size());  // add(neg(neg(x), neg(neg(y)))

    std::vector<float> inputX(elementsNum);
    inputX.assign((float*)m_hostBuffers[x], (float*)m_hostBuffers[x] + elementsNum);
    std::vector<float> inputY(elementsNum);
    inputY.assign((float*)m_hostBuffers[y], (float*)m_hostBuffers[y] + elementsNum);
    std::vector<float> outBuffer(elementsNum);
    outBuffer.assign((float*)m_hostBuffers[out], (float*)m_hostBuffers[out] + elementsNum);

    for (unsigned i = 0; i < elementsNum; i++)
    {
        float ref = inputX[i] + inputY[i];
        ASSERT_EQ(ref, outBuffer[i]);
    }
}

TEST_F_GC(SynGaudiComplexGUIDExtractorDummyTest, DISABLED_test_complex_guid_extractor_internal_ctrl_dep)
{
    /*
     * Test Case - Complex GUID node extracted to simple nodes, with internal ctrl dep created by complex GUID lib
     * Complex Guid Node "complex_add_f32(x,y)", decomposed to add(x,neg(neg(y)), add is blocked by neg
     */
    const unsigned C           = 3;
    const unsigned W           = 3;
    const unsigned H           = 3;
    const unsigned N           = 1;
    size_t         elementsNum = C * W * H * N;

    unsigned sizes[] = {C, W, H, N};

    unsigned x   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes);
    unsigned y   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes);
    unsigned out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes);

    addNodeToGraph("ctrl_dep_complex_add_f32", {x, y}, {out});
    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(1, graph->getExeSortedNodes().size());
    auto complexAddNode = graph->getExeSortedNodes().front();
    ASSERT_EQ(0, graph->getBlockingNodes(complexAddNode).size());
    compileAndRun();
    ASSERT_EQ(3, graph->getExeSortedNodes().size());
    auto simpleAddNode = graph->getExeSortedNodes().back();
    ASSERT_NE(simpleAddNode, nullptr);
    ASSERT_EQ(1, graph->getBlockingNodes(simpleAddNode).size());  // verify ctrl dep was created

    std::vector<float> inputX(elementsNum);
    inputX.assign((float*)m_hostBuffers[x], (float*)m_hostBuffers[x] + elementsNum);
    std::vector<float> inputY(elementsNum);
    inputY.assign((float*)m_hostBuffers[y], (float*)m_hostBuffers[y] + elementsNum);
    std::vector<float> outBuffer(elementsNum);
    outBuffer.assign((float*)m_hostBuffers[out], (float*)m_hostBuffers[out] + elementsNum);

    for (unsigned i = 0; i < elementsNum; i++)
    {
        float ref = inputX[i] + inputY[i];
        ASSERT_EQ(ref, outBuffer[i]);
    }
}

TEST_F_GC(SynGaudiComplexGUIDExtractorDummyTest, DISABLED_test_complex_guid_extrenal_ctrl_dep_blocked)
{
    /*
     * Test Case User Graph: Conv -> [Complex Guid Node]
     * Complex Guid Node "complex abs(x)",  decomposed to abs(abs(x))
     * Complex Guid Node is blocked by Conv
     */
    unsigned convInputSizes[]  = {3, 4, 4, 1};
    unsigned weightsSizes[]    = {3, 3, 2, 2};
    unsigned convOutputSizes[] = {3, 3, 3, 1};

    synConvolutionParams params;

    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 2;
    params.kW   = 2;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    auto convInput   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, convInputSizes);
    auto convWeights = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, weightsSizes);
    auto convOutput  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convOutputSizes);

    auto complexAbsOutput = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convOutputSizes);

    synNodeId convNodeId;
    synNodeId complexAbsNodeId;
    addNodeToGraph("spatial_convolution",
                   {convInput, convWeights},
                   {convOutput},
                   (void*)&params,
                   sizeof(synConvolutionParams),
                   "conv0",
                   0,
                   &convNodeId);
    addNodeToGraph("complex_abs_f32", {convOutput}, {complexAbsOutput}, nullptr, 0, "complexAbs", 0, &complexAbsNodeId);

    auto graphHandle = m_graphs.front().graphHandle;
    synNodeDependencySet(graphHandle, &convNodeId, &complexAbsNodeId, 1, 1);
    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(graphHandle);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(2, graph->getExeSortedNodes().size());
    ASSERT_EQ(1, graph->getBlockedNodes(graph->getNodeByID(convNodeId)).size());
    compileAndRun();
    ASSERT_EQ(
        8,
        graph->getExeSortedNodes().size());  // Tpc weight packing node, reduction, static reshape, Conv, 2 abs, 2 DMA
    // get the conv id (it was changed during compilation)
    auto mmeNodes = graph->getSortedMMENodes();
    ASSERT_EQ(mmeNodes.size(), 1);
    auto conv = mmeNodes.front();
    ASSERT_EQ(2, graph->getBlockedNodes(conv).size());  // verify ctrl dep was created
}

TEST_F_GC(SynGaudiComplexGUIDExtractorDummyTest, DISABLED_test_complex_guid_extrenal_ctrl_dep_blocking)
{
    /*
     * Test Case User Graph: [Complex Guid Node] -> Conv
     * Complex Guid Node "complex abs(x)",  decomposed to abs(abs(x))
     * Complex Guid Node is blocking Conv
     */
    unsigned convInputSizes[]  = {3, 4, 4, 1};
    unsigned weightsSizes[]    = {3, 3, 2, 2};
    unsigned convOutputSizes[] = {3, 3, 3, 1};

    synConvolutionParams params;

    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 2;
    params.kW   = 2;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    auto complexAbsInput  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, convInputSizes);
    auto complexAbsOutput = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convInputSizes);

    auto convWeights = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, weightsSizes);
    auto convOutput  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convOutputSizes);

    synNodeId convNodeId;
    synNodeId complexAbsNodeId;
    addNodeToGraph("spatial_convolution",
                   {complexAbsOutput, convWeights},
                   {convOutput},
                   (void*)&params,
                   sizeof(synConvolutionParams),
                   "conv0",
                   0,
                   &convNodeId);
    addNodeToGraph("complex_abs_f32",
                   {complexAbsInput},
                   {complexAbsOutput},
                   nullptr,
                   0,
                   "complexAbs",
                   0,
                   &complexAbsNodeId);

    auto graphHandle = m_graphs.front().graphHandle;
    synNodeDependencySet(graphHandle, &complexAbsNodeId, &convNodeId, 1, 1);
    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(graphHandle);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(2, graph->getExeSortedNodes().size());
    ASSERT_EQ(1, graph->getBlockingNodes(graph->getNodeByID(convNodeId)).size());
    compileAndRun();
    ASSERT_EQ(
        8,
        graph->getExeSortedNodes().size());  // Tpc weight packing node, reduction, static reshape, Conv, 2 abs, 2 DMA
    // get the conv id (it was changed during compilation)
    auto mmeNodes = graph->getSortedMMENodes();
    ASSERT_EQ(mmeNodes.size(), 1);
    auto conv = mmeNodes.front();
    ASSERT_EQ(2, graph->getBlockingNodes(conv).size());  // verify ctrl dep was created
}

/* tests using actual complex guid library */
class SynTrainingComplexGUIDExtractorTest : public SynTrainingTestInfra
{
public:
    void afterSynInitialize() override
    {
        synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");  // must set it first so we can configure the below
        synConfigurationGet("COMPLEX_GUID_EXTRACTOR_MODE", &m_prevMode, 1);
        synConfigurationSet("COMPLEX_GUID_EXTRACTOR_MODE", "1");
        SynTrainingTestInfra::afterSynInitialize();
    }

    void TearDownTest() override
    {
        GCFG_COMPLEX_GUID_EXTRACTOR_MODE.setValue(m_prevMode);
        SynTrainingTestInfra::TearDownTest();
    }

private:
    char m_prevMode = '0';
};

TEST_F_GC(SynTrainingComplexGUIDExtractorTest, test_complex_guid_idle)
{
    ASSERT_TRUE(ComplexGuidExtractorSharedObject::instance().isInitialized());
}
