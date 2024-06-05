#include "flash_attention_tests.h"
#include "compilation_hal_reader.h"
#include "graph_editor.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "synapse_api_types.h"
#include "types.h"
#include "gtest/gtest.h"
#include "tile.h"
#include <algorithm>

enum FaInputTensorIndex
{
    Q_FWD = 0,
    DO_BWD = 0,
    Q_BWD = 1,
    K_FWD = 1,
    K_BWD = 2,
    V_FWD = 2,
    V_BWD = 3,
    SEED_FWD = 4,
    STATS1_BWD = 5,
    STATS2_BWD = 6,
    SEED_BWD = 7
};

enum FaOutputTensorIndex
{
    O_FWD = 0,
    DQ_BWD = 0,
    DK_BWD = 1,
    STATS1_FWD = 1,
    DV_BWD = 2,
    STATS2_FWD = 2
};

static void setAsPersistent(TensorPtr& tensor, unsigned tensorsCount)
{
    static synMemoryDescriptor memDesc(true);
    tensor->setMemoryDescriptor(memDesc);
    tensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + tensorsCount);
}

bool FlashAttentionTestInfra::maxTensorsExceedDram()
{
    // Multiple with 2 because input and output of softmax live in the same time.
    return (m_N * m_N * m_B * m_H * sizeof(bfloat16)) * 2 >= m_graph->getHALReader()->getDRAMSizeInBytes();
}

bool FlashAttentionTest::shouldSkip()
{
    return !m_sliceInCguid && m_testBuilder.maxTensorsExceedDram();
}

void FlashAttentionTest::SetUp()
{
    GraphOptimizerTest::SetUp();
    setGlobalConfForTest(GCFG_ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE, "true");
    // The flags PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE, PT_HPU_SDPA_BATCH_NUMHEADS_SLICE can't be set using the setGlobalConfForTest because it is not
    // synapse configuration, so setting it directly using setenv.
    m_sdpaSlicingPrevCfg = std::getenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE");
    setenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE", m_sliceInCguid ? "1" : "0", true);
    m_sdpaReshapePrevCfg = std::getenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE");
    setenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE", m_reshapeSoftmax && m_sliceInCguid ? "1" : "0", true); // Use reshape optimizations for softmax
}

void FlashAttentionTest::TearDown()
{
    // Reset env var to its original value, or unset it if it wasn't previously set
    if (m_sdpaSlicingPrevCfg)
    {
        setenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE", m_sdpaSlicingPrevCfg, true);
    }
    else
    {
        unsetenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE");
    }
    if (m_sdpaReshapePrevCfg)
    {
        setenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE", m_sdpaReshapePrevCfg, true);
    }
    else
    {
        unsetenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE");
    }
    GraphOptimizerTest::TearDown();
}

void FlashAttentionParametrizedTest::SetUp()
{
    FlashAttentionTest::SetUp();
    if (shouldSkip())
    {
        GTEST_SKIP() << fmt::format("Test configuration: [D={}, N={}, H={}, B={}, TriangularMask={}, DropoutProb={}, "
                                    "SliceInCguid={}] is not valid, skipping.",
                                    std::get<0>(GetParam()),
                                    std::get<1>(GetParam()),
                                    std::get<2>(GetParam()),
                                    std::get<3>(GetParam()),
                                    std::get<4>(GetParam()),
                                    std::get<5>(GetParam()),
                                    std::get<6>(GetParam()));
    }
}

void FlashAttentionParametrizedTest::checkMemEfficientSchedule()
{
    const auto& nodes = m_testBuilder.m_graph->getExeSortedNodes();
    unsigned prevChainId = 0;
    for (const auto& node : nodes)
    {
        if (!node->getNodeAnnotation().flashAttentionInfo.has_value()) continue;
        if (node->getNodeAnnotation().flashAttentionInfo->chainInfo.has_value())
        {
            unsigned currChainId = node->getNodeAnnotation().flashAttentionInfo->chainInfo.value();
            if (currChainId == prevChainId) continue;
            else
            {
                ASSERT_GT(currChainId, prevChainId)
                    << "Chain " << currChainId << " was scheduled in the middle of chain " << prevChainId;
                prevChainId = currChainId;
            }
        }
    }
}

bool FlashAttentionTest::isSoftmax(const NodePtr& n)
{
    return n->getGUID().find("softmax") != std::string::npos;
}

ns_Sdpa::Params FlashAttentionTestInfra::setParams()
{
    ns_Sdpa::Params params = {0};
    params.scale           = 1.0 / sqrt(m_D);
    params.is_causal       = true;
    params.dropout.ratio   = m_dropoutProb;
    return params;
}

TensorPtr FlashAttentionTestInfra::createStatsTensor(synDataType dtype, bool setPersistent)
{
    auto t = std::make_shared<Tensor>(m_statsTensorSizes.size(), m_statsTensorSizes.data(), dtype);
    if (setPersistent) setAsPersistent(t, m_tensorCount++);
    return t;
}

TensorPtr FlashAttentionTestInfra::createTensor(const std::string& name)
{
    auto t = std::make_shared<Tensor>(m_tensorSizes.size(), m_tensorSizes.data(), syn_type_bf16);
    t->setName(name);
    setAsPersistent(t, m_tensorCount++);
    return t;
}

NodePtr FlashAttentionTestInfra::createSdpaFwd()
{
    ns_Sdpa::Params params  = setParams();
    TensorVector    inputs  = {createTensor("Q"), createTensor("K"), createTensor("V")};
    TensorVector    outputs = {createTensor("O"),
                               createStatsTensor(syn_type_bf16, true),
                               createStatsTensor(syn_type_single, true)};

    if (params.dropout.ratio > 0.0)
    {
        const std::vector<TSize> dropoutMaskSizes = {1};
        TensorPtr t1 = std::make_shared<Tensor>(dropoutMaskSizes.size(), dropoutMaskSizes.data(), syn_type_int32);
        t1->setName("dropout_seed_in");
        setAsPersistent(t1, m_tensorCount++);
        inputs.push_back(nullptr);  // AttentionMask input - currently unsupported
        inputs.push_back(t1);
        TensorPtr t2 = std::make_shared<Tensor>(dropoutMaskSizes.size(), dropoutMaskSizes.data(), syn_type_int32);
        t2->setName("dropout_seed_out");
        outputs.push_back(t2);
    }

    NodePtr sdpaFwd = NodeFactory::createNode(inputs,
                                        outputs,
                                        &params,
                                        sizeof(params),
                                        "sdpa_recomp_fwd_bf16",
                                        "flash_attn_fwd_" + std::to_string(m_nodeIdx++));
    sdpaFwd->setDeterministic(true);
    GraphEditor::addNode(*m_graph, sdpaFwd);
    return sdpaFwd;
}

NodePtr FlashAttentionTestInfra::createSdpa(bool isFwd)
{
    NodePtr sdpa = isFwd ? createSdpaFwd() : createSdpaBwd();
    sdpa->setDeterministic(true);
    GraphEditor::addNode(*m_graph, sdpa);
    return sdpa;
};

NodePtr FlashAttentionTestInfra::createSdpaBwd()
{
    ns_Sdpa::Params params  = setParams();
    TensorVector    inputs  = {createTensor("dO"),
                               createTensor("Q"),
                               createTensor("K"),
                               createTensor("V"),
                               nullptr,  // AttentionMask input - currently unsupported
                               createStatsTensor(syn_type_bf16, true),
                               createStatsTensor(syn_type_single, true)};
    TensorVector    outputs = {createTensor("dQ"), createTensor("dK"), createTensor("dV")};

    if (params.dropout.ratio > 0.0)
    {
        const std::vector<TSize> dropoutMaskSizes = {1};
        TensorPtr t1 = std::make_shared<Tensor>(dropoutMaskSizes.size(), dropoutMaskSizes.data(), syn_type_int32);
        t1->setName("dropout_seed_in");
        setAsPersistent(t1, m_tensorCount++);
        inputs.push_back(t1);
    }

    return NodeFactory::createNode(inputs,
                                        outputs,
                                        &params,
                                        sizeof(params),
                                        "sdpa_recomp_bwd_bf16",
                                        "flash_attn_bwd_" + std::to_string(m_nodeIdx++));

}

NodePtr FlashAttentionTest::createSdpa(bool isFwd)
{
    return m_testBuilder.createSdpa(isFwd);
}

static bool isSlicedDim(const TensorPtr& big, const TensorPtr& small, unsigned dim)
{
    EXPECT_TRUE(small->getTensorAnnotation().origBigTensor == big)
        << "Undefined behavior, expecting tensors to have big-small relationship";
    return small->getSizeInElements(dim) < big->getSizeInElements(dim);
}

static gc::access_pattern::Dim findLastNoneDegenerateDim(const TensorPtr& tensor)
{
    for (gc::access_pattern::Dim i = tensor->getDim() - 1; i > 0; i--)
    {
        if (tensor->getSizeInElements(i) > 1)
        {
            return i;
        }
    }
    return 0;
}

void FlashAttentionFwdPipeliningTest::checkSlicedSoftmax(const HabanaGraph& g)
{
    std::map<NodePtr, NodeSet> softmaxBigToSlicedodes;
    for (const NodePtr& node : g.getNodes())
    {
        if (node->getNodeAnnotation().bundleInfo.is_set() && isSoftmax(node))
        {
            ASSERT_TRUE(node->getNodeAnnotation().origBigNode) << "Expecting sliced node to have origBigNode";
            softmaxBigToSlicedodes[node->getNodeAnnotation().origBigNode].insert(node);
        }
    }
    ASSERT_FALSE(softmaxBigToSlicedodes.empty()) << "Expecting softmax node to be sliced";
    // Expect all the sliced softmax nodes to be sliced on outer dim, while dim 0 and 1 are unsliced (due to the
    // triangular mask)
    for (const auto& origBigAndSlicedNodes : softmaxBigToSlicedodes)
    {
        const auto& origBig = origBigAndSlicedNodes.first;
        const auto& slicedNodes = origBigAndSlicedNodes.second;
        ASSERT_TRUE(std::all_of(slicedNodes.begin(), slicedNodes.end(), [&origBig](const NodePtr& slicedNode) {
            return !isSlicedDim(origBig->getOutput(0), slicedNode->getOutput(0), 0) &&
                   !isSlicedDim(origBig->getOutput(0), slicedNode->getOutput(0), 1);
        }));
        ASSERT_TRUE(std::all_of(slicedNodes.begin(), slicedNodes.end(), [&origBig](const NodePtr& slicedNode) {
            return isSlicedDim(origBig->getOutput(0),
                               slicedNode->getOutput(0),
                               findLastNoneDegenerateDim(origBig->getOutput(0)));
        }));
        ASSERT_GT(slicedNodes.size(), 1);
    }
}

bool FlashAttentionTest::runTest()
{
    return m_testBuilder.m_graph->compile();
}

TEST_P(FlashAttentionBwdSchedulingTest, sdpa_bwd_schedule_test)
{
    if (!m_gcSlicing) setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, "0");
    createSdpa(false);
    ASSERT_TRUE(runTest());
    checkMemEfficientSchedule();
}

TEST_P(FlashAttentionFwdSchedulingTest, sdpa_fwd_schedule_test)
{
    if (!m_gcSlicing) setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, "0");
    createSdpa(true);
    ASSERT_TRUE(runTest());
    checkMemEfficientSchedule();
}

TEST_P(FlashAttentionFwdPipeliningTest, sdpa_fwd_pipelining_test)
{
    createSdpa(true);
    ASSERT_TRUE(runTest());
    checkSlicedSoftmax(*m_testBuilder.m_graph);
}

TEST_P(FlashAttentionFwdNegTest, sdpa_fwd_neg_test_single_node)
{
    setGlobalConfForTest(GCFG_ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE, "false");
    createSdpa(true);
    ASSERT_FALSE(runTest());
}

// Due to an unsolved issue in cguid's huge tensor handling,
// the tests: "without_cguid_slicing" aren't enabled on gaudi3.

INSTANTIATE_TEST_SUITE_P(,
                         FlashAttentionBwdSchedulingTest,
                         testing::Combine(testing::Values(128),                // D
                                          testing::Values(2048, 6144, 32768),  // N
                                          testing::Values(12),                 // H
                                          testing::Values(4),                  // B
                                          testing::Values(true),               // Triagular attention mask used
                                          testing::Values(0.0, 0.1),           // Dropout prob
                                          testing::Values(true),               // Slice B&H in cguid
                                          testing::Values(true, false),        // Reshape softmax
                                          testing::Values(true, false),        // Enable gc slicing
                                          testing::Values(synDeviceGaudi2, synDeviceGaudi3)  // Device type
                                          ),
                         FlashAttentionParametrizedTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(without_cguid_slicing,
                         FlashAttentionBwdSchedulingTest,
                         testing::Combine(testing::Values(128),                // D
                                          testing::Values(2048, 6144, 32768),  // N
                                          testing::Values(12),                 // H
                                          testing::Values(4),                  // B
                                          testing::Values(true),               // Triagular attention mask used
                                          testing::Values(0.0, 0.1),           // Dropout prob
                                          testing::Values(false),              // Slice B&H in cguid
                                          testing::Values(true, false),        // Reshape softmax
                                          testing::Values(true, false),        // Enable gc slicing
                                          testing::Values(synDeviceGaudi2)     // Device type
                                          ),
                         FlashAttentionParametrizedTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(,
                         FlashAttentionFwdSchedulingTest,
                         testing::Combine(testing::Values(128),                // D
                                          testing::Values(2048, 6144, 32768),  // N
                                          testing::Values(12),                 // H
                                          testing::Values(4),                  // B
                                          testing::Values(true),               // Triagular attention mask used
                                          testing::Values(0.0, 0.1),           // Dropout prob
                                          testing::Values(true),               // Slice B&H in cguid
                                          testing::Values(true, false),        // Reshape softmax
                                          testing::Values(true, false),        // Enable gc slicing
                                          testing::Values(synDeviceGaudi2, synDeviceGaudi3)  // Device type
                                          ),
                         FlashAttentionParametrizedTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(without_cguid_slicing,
                         FlashAttentionFwdSchedulingTest,
                         testing::Combine(testing::Values(128),                // D
                                          testing::Values(2048, 6144, 32768),  // N
                                          testing::Values(12),                 // H
                                          testing::Values(4),                  // B
                                          testing::Values(true),               // Triagular attention mask used
                                          testing::Values(0.0, 0.1),           // Dropout prob
                                          testing::Values(false),              // Slice B&H in cguid
                                          testing::Values(true, false),        // Reshape softmax
                                          testing::Values(true, false),        // Enable gc slicing
                                          testing::Values(synDeviceGaudi2)     // Device type
                                          ),
                         FlashAttentionParametrizedTest::PrintToStringParamName {});

// The following tests FlashAttentionFwdNegTest and FlashAttentionFwdPipeliningTest
// were built based on gaudi2 properties, therefore they are executed only with gaudi2 graph.
INSTANTIATE_TEST_SUITE_P(neg_test,
                         FlashAttentionFwdNegTest,
                         testing::Combine(testing::Values(128),             // D
                                          testing::Values(32768),           // N
                                          testing::Values(48),              // H
                                          testing::Values(1),               // B
                                          testing::Values(true),            // Triagular attention mask used
                                          testing::Values(0.0, 0.1),        // Dropout prob
                                          testing::Values(false),           // Slice B&H in cguid
                                          testing::Values(false),           // Reshape softmax
                                          testing::Values(true),            // Enable gc slicing
                                          testing::Values(synDeviceGaudi2)  // Device type
                                          ),
                         FlashAttentionParametrizedTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(,
                         FlashAttentionFwdPipeliningTest,
                         testing::Combine(testing::Values(128),             // D
                                          testing::Values(1024),            // N
                                          testing::Values(4),               // H
                                          testing::Values(4),               // B
                                          testing::Values(true),            // Triagular attention mask used
                                          testing::Values(0.1),             // Dropout prob
                                          testing::Values(false),           // Slice B&H in cguid
                                          testing::Values(true),            // Reshape softmax
                                          testing::Values(true),            // Enable gc slicing
                                          testing::Values(synDeviceGaudi2)  // Device type
                                          ),
                         FlashAttentionParametrizedTest::PrintToStringParamName {});

// This test checks that when there's a producer that is not sdpa, the full graph schedule is generated without cycles
// (scenario from SW-168783)
TEST_F(FlashAttentionTest, sdpa_with_producer)
{
    const unsigned          D = 128;
    const unsigned          N = 2048;
    const unsigned          H = 4;
    const unsigned          B = 12;
    FlashAttentionTestInfra testBuilder(D, N, H, B, true);

    const NodePtr& sdpaFwd     = testBuilder.createSdpa(true);
    const auto&    tensorSizes = sdpaFwd->getInput(Q_FWD)->getAllSizesInElements();

    TensorPtr castIn = std::make_shared<Tensor>(tensorSizes.size(), tensorSizes.data(), syn_type_single);
    NodePtr   cast =
        NodeFactory::createNode({castIn}, {sdpaFwd->getInput(Q_FWD)}, nullptr, 0, "cast_f32_to_bf16", "cast");
    GraphEditor::addNode(*testBuilder.m_graph, cast, false);

    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*testBuilder.m_graph));
    testBuilder.m_graph->invalidateExecutionSchedule();
    ASSERT_TRUE(scheduleFlashAttentionNodes(*testBuilder.m_graph));
    ASSERT_TRUE(testBuilder.m_graph->getExeSortedNodes().size() > 0);
}

NodePtr FlashAttentionFwdBwdTest::createSdpa()
{
    const NodePtr& sdpaFwd = m_testBuilder.createSdpa(true);
    const NodePtr& sdpaBwd = m_testBuilder.createSdpa(false);
    GraphEditor::editNode(*m_testBuilder.m_graph, sdpaBwd, [&]() {
        sdpaBwd->replaceInput(Q_BWD, sdpaFwd->getInput(Q_FWD));
        sdpaBwd->replaceInput(K_BWD, sdpaFwd->getInput(K_FWD));
        sdpaBwd->replaceInput(V_BWD, sdpaFwd->getInput(V_FWD));
        sdpaBwd->replaceInput(STATS1_BWD, sdpaFwd->getOutput(STATS1_FWD));
        sdpaBwd->replaceInput(STATS2_BWD, sdpaFwd->getOutput(STATS2_FWD));
        if (m_testBuilder.m_dropoutProb > 0.0) sdpaBwd->replaceInput(SEED_BWD, sdpaFwd->getInput(SEED_FWD));
    });
    return sdpaFwd;
}

TEST_P(FlashAttentionFwdBwdTest, DISABLED_sdpa_fwd_bwd)
{
    createSdpa();
    ASSERT_TRUE(runTest());
}

INSTANTIATE_TEST_SUITE_P(,
                         FlashAttentionFwdBwdTest,
                         testing::Combine(testing::Values(128),          // D
                                          testing::Values(2048, 6144),   // N
                                          testing::Values(2),            // B
                                          testing::Values(1),            // H
                                          testing::Values(true),         // Triagular attention mask used
                                          testing::Values(0.0, 0.1),     // Dropout prob
                                          testing::Values(true, false),  // Slice B&H in cguid
                                          testing::Values(true, false),  // Reshape softmax
                                          testing::Values(true),         // Enable gc slicing
                                          testing::Values(synDeviceGaudi2)
                                          ),
                         FlashAttentionParametrizedTest::PrintToStringParamName {});