#include "layered_brain_test_common.h"
#include "gaudi2_graph.h"
#include "graph_editor.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "synapse_common_types.h"
#include "graph_optimizer_test.h"
#include "bundler/bundle_candidate_finders.h"
#include "node_factory.h"

using namespace gc::layered_brain;
using namespace gc::layered_brain::bundler;

class BundleCandidateFinderTest : public LayeredBrainCommonTest<Gaudi2Graph>
{
protected:
    BundleCandidateFinderTest() : m_compHalReaderSetter(getCompilationHalReaderSetter(m_graph)) {}
    void validateFinderRoute(const CandidateFinderPtr& finder, std::list<NodePtr> expectedRoute)
    {
        for (auto state = finder->next(); state.has_value(); state = finder->next())
        {
            NodePtr candidate;
            std::tie(std::ignore, candidate) = state.value();
            ASSERT_TRUE(expectedRoute.front() == candidate);
            ASSERT_TRUE(!expectedRoute.empty());
            expectedRoute.pop_front();
        }
    }

private:
    static CompilationHalReaderSetter getCompilationHalReaderSetter(const HabanaGraph& graph)
    {
        CompilationHalReader::setHalReader(Gaudi2HalReader::instance());
        return CompilationHalReaderSetter(&graph);
    }
    CompilationHalReaderSetter m_compHalReaderSetter;
};

class DfsProducersFinderTest : public BundleCandidateFinderTest
{
protected:
    CandidateFinderPtr getFinder(const TensorPtr& t) { return std::make_shared<DfsProducersFinder>(t, m_graph); }
};

/*
┌────┐      ┌──────┐
│add ├─────►│bgemm │
└────┘      └──────┘
 */
TEST_F(DfsProducersFinderTest, add_batch_gemm)
{
    constexpr unsigned batch0   = 4;
    constexpr unsigned batch1   = 3;
    const auto         bgemmIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto         bgemmIn1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto         bgemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto         bgemm =
        NodeFactory::createNode({bgemmIn0, bgemmIn1}, {bgemmOut}, nullptr, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(bgemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm)) << "Failed adding node: " << bgemm->getNodeName();

    const auto addIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto addIn1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add    = NodeFactory::createNode({addIn0, addIn1}, {bgemmIn0}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add)) << "Failed adding node: " << add->getNodeName();
    gaudi2::loadTpcKernels(m_graph);
    auto input0Finder = getFinder(bgemm->getInput(0));
    validateFinderRoute(input0Finder, {add});
    auto input1Finder = getFinder(bgemm->getInput(1));
    validateFinderRoute(input1Finder, {});
}

/*
┌────┐
│add ├──────►┌─────┐
└────┘       │     │
             │bgemm│
┌────┐       │     │
│add ├──────►└─────┘
└────┘
 */
TEST_F(DfsProducersFinderTest, two_add_one_batch_gemm)
{
    constexpr unsigned batch0 = 4;
    constexpr unsigned batch1 = 3;

    const auto add0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0In1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0Out = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto add1In0 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto bgemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto bgemm =
        NodeFactory::createNode({add0Out, add1Out}, {bgemmOut}, nullptr, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(bgemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm)) << "Failed adding node: " << bgemm->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    // Expecting each finder to find exactly one candidate (add)
    auto input0Finder = getFinder(bgemm->getInput(0));
    validateFinderRoute(input0Finder, {add0});
    auto input1Finder = getFinder(bgemm->getInput(1));
    validateFinderRoute(input1Finder, {add1});
}

/*
┌──────┐      ┌─────┐      ┌────────┐       ┌────────┐         ┌──────┐
│ mult ├─────►│ add ├─────►│ softmax├──────►│ dropout├────────►│      │
└──────┘      └─────┘      └────────┘       └────────┘         │      │
                                                               │ batch│
                                                               │ gemm │
         ┌─────┐      ┌────────┐      ┌────────────┐           │      │
         │add  ├─────►│reshape ├─────►│transpose   ├──────────►│      │
         └─────┘      └────────┘      └────────────┘           └──────┘
 */
TEST_F(DfsProducersFinderTest, milestone_0_bundle)
{
    /*************
     * bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0 node
     * inputs:
     *     t1360_bert_encoder_layer_1_attention_self_value_MatMul_0[1024, 14336] (dtype=bf16)
     *     t1362_bert_encoder_layer_1_attention_self_value_BiasAdd[1024, 1] (dtype=bf16)
     * outputs:
     *     t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1360_bert_encoder_layer_1_attention_self_value_MatMul_0 tensor
    std::vector<TSize> t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_max_sizes = {1024, 14336};
    const auto         t1360_bert_encoder_layer_1_attention_self_value_MatMul_0 =
        createTensor(t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_max_sizes, syn_type_bf16, true);

    // create t1362_bert_encoder_layer_1_attention_self_value_BiasAdd tensor
    std::vector<TSize> t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes = {1024, 1};
    const auto         t1362_bert_encoder_layer_1_attention_self_value_BiasAdd =
        createTensor(t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes, syn_type_bf16, true);

    // create t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0 tensor
    std::vector<TSize> t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_max_sizes = {1024, 14336};
    const auto         t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0 =
        createTensor(t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_max_sizes, syn_type_bf16, true);
    const auto add0 =
        NodeFactory::createGenericTPCNode({t1360_bert_encoder_layer_1_attention_self_value_MatMul_0,
                                           t1362_bert_encoder_layer_1_attention_self_value_BiasAdd},
                                          {t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0},
                                          nullptr,
                                          0,
                                          "add_fwd_bf16",
                                          "bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0");
    EXPECT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    /*************
     * bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0 node
     * inputs:
     *     t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0[1024, 14336] (dtype=bf16)
     *     t1365_bert_encoder_layer_1_attention_self_Reshape_2[64, 16, 512, 28] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1364_bert_encoder_layer_1_attention_self_Reshape_2_0[64, 16, 512, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1365_bert_encoder_layer_1_attention_self_Reshape_2 tensor
    std::vector<TSize> t1365_bert_encoder_layer_1_attention_self_Reshape_2_max_sizes = {64, 16, 512, 28};

    const auto t1365_bert_encoder_layer_1_attention_self_Reshape_2 =
        createShapeTensor(t1365_bert_encoder_layer_1_attention_self_Reshape_2_max_sizes,
                          t1365_bert_encoder_layer_1_attention_self_Reshape_2_max_sizes,
                          syn_type_uint32,
                          false);
    // create t1364_bert_encoder_layer_1_attention_self_Reshape_2_0 tensor
    std::vector<TSize> t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_max_sizes = {64, 16, 512, 28};
    const auto         t1364_bert_encoder_layer_1_attention_self_Reshape_2_0 =
        createTensor(t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_max_sizes,
                     t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_max_sizes,
                     syn_type_bf16,
                     false);

    const auto reshape0 = NodeFactory::createNode({t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0,
                                                   t1365_bert_encoder_layer_1_attention_self_Reshape_2},
                                                  {t1364_bert_encoder_layer_1_attention_self_Reshape_2_0},
                                                  nullptr,
                                                  0,
                                                  "reshape",
                                                  "bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0");

    EXPECT_TRUE(GraphEditor::addNode(m_graph, reshape0)) << "Failed adding node: " << reshape0->getNodeName();
    /*************
     * bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0 node
     * inputs:
     *     t1364_bert_encoder_layer_1_attention_self_Reshape_2_0[64, 16, 512, 28] (dtype=bf16)
     * outputs:
     *     t1366_bert_encoder_layer_1_attention_self_transpose_2_0[64, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1366_bert_encoder_layer_1_attention_self_transpose_2_0 tensor
    std::vector<TSize> t1366_bert_encoder_layer_1_attention_self_transpose_2_0_max_sizes = {64, 512, 16, 28};
    const auto         t1366_bert_encoder_layer_1_attention_self_transpose_2_0 =
        createTensor(t1366_bert_encoder_layer_1_attention_self_transpose_2_0_max_sizes,
                     t1366_bert_encoder_layer_1_attention_self_transpose_2_0_max_sizes,
                     syn_type_bf16,
                     false);

    unsigned char bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_params[] = {
        0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    const auto transpose0 =
        NodeFactory::createNode({t1364_bert_encoder_layer_1_attention_self_Reshape_2_0},
                                {t1366_bert_encoder_layer_1_attention_self_transpose_2_0},
                                bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_params,
                                sizeof(bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_params),
                                "transpose",
                                "bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0");
    EXPECT_TRUE(GraphEditor::addNode(m_graph, transpose0)) << "Failed adding node: " << transpose0->getNodeName();
    /*************
     * bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0 node
     * inputs:
     *     t1359_bert_encoder_layer_1_attention_self_MatMul_0[512, 512, 16, 28] (dtype=bf16)
     *     t1368_bert_encoder_layer_1_attention_self_Mul[1, 1, 1, 1] (dtype=bf16)
     * outputs:
     *     t1367_bert_encoder_layer_1_attention_self_Mul_0[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1359_bert_encoder_layer_1_attention_self_MatMul_0 tensor

    std::vector<TSize> t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes = {512, 512, 16, 28};
    const auto         t1359_bert_encoder_layer_1_attention_self_MatMul_0 =
        createTensor(t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes,
                     t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes,
                     syn_type_bf16,
                     false);
    // create t1368_bert_encoder_layer_1_attention_self_Mul tensor
    std::vector<TSize> t1368_bert_encoder_layer_1_attention_self_Mul_max_sizes = {1, 1, 1, 1};
    const auto         t1368_bert_encoder_layer_1_attention_self_Mul =
        createTensor(t1368_bert_encoder_layer_1_attention_self_Mul_max_sizes,
                     t1368_bert_encoder_layer_1_attention_self_Mul_max_sizes,
                     syn_type_bf16,
                     true);

    // create t1367_bert_encoder_layer_1_attention_self_Mul_0 tensor
    std::vector<TSize> t1367_bert_encoder_layer_1_attention_self_Mul_0_max_sizes = {512, 512, 16, 28};
    const auto         t1367_bert_encoder_layer_1_attention_self_Mul_0 =
        createTensor(t1367_bert_encoder_layer_1_attention_self_Mul_0_max_sizes,
                     t1367_bert_encoder_layer_1_attention_self_Mul_0_max_sizes,
                     syn_type_bf16,
                     true);

    const auto mult0 = NodeFactory::createGenericTPCNode(
        {t1359_bert_encoder_layer_1_attention_self_MatMul_0, t1368_bert_encoder_layer_1_attention_self_Mul},
        {t1367_bert_encoder_layer_1_attention_self_Mul_0},
        nullptr,
        0,
        "mult_fwd_bf16",
        "bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0");
    EXPECT_TRUE(GraphEditor::addNode(m_graph, mult0)) << "Failed adding node: " << mult0->getNodeName();
    /*************
     * bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0 node
     * inputs:
     *     t1367_bert_encoder_layer_1_attention_self_Mul_0[512, 512, 16, 28] (dtype=bf16)
     *     t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0[512, 512, 1, 28] (dtype=bf16)
     * outputs:
     *     t1370_bert_encoder_layer_1_attention_self_add_0[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0 tensor
    std::vector<TSize> t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_max_sizes = {512,
                                                                                                            512,
                                                                                                            1,
                                                                                                            28};
    const auto         t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0 =
        createTensor(t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_max_sizes,
                     t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_max_sizes,
                     syn_type_bf16,
                     true);

    // create t1370_bert_encoder_layer_1_attention_self_add_0 tensor
    std::vector<TSize> t1370_bert_encoder_layer_1_attention_self_add_0_max_sizes = {512, 512, 16, 28};
    const auto         t1370_bert_encoder_layer_1_attention_self_add_0 =
        createTensor(t1370_bert_encoder_layer_1_attention_self_add_0_max_sizes,
                     t1370_bert_encoder_layer_1_attention_self_add_0_max_sizes,
                     syn_type_bf16,
                     false);

    const auto add1 =
        NodeFactory::createGenericTPCNode({t1367_bert_encoder_layer_1_attention_self_Mul_0,
                                           t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0},
                                          {t1370_bert_encoder_layer_1_attention_self_add_0},
                                          nullptr,
                                          0,
                                          "add_fwd_bf16",
                                          "bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0");
    EXPECT_TRUE(add1 != nullptr);
    EXPECT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();
    /*************
     * bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0 node
     * inputs:
     *     t1370_bert_encoder_layer_1_attention_self_add_0[512, 512, 16, 28] (dtype=bf16)
     * outputs:
     *     t1371_bert_encoder_layer_1_attention_self_Softmax_0[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1371_bert_encoder_layer_1_attention_self_Softmax_0 tensor
    std::vector<TSize> t1371_bert_encoder_layer_1_attention_self_Softmax_0_max_sizes = {512, 512, 16, 28};
    const auto         t1371_bert_encoder_layer_1_attention_self_Softmax_0 =
        createTensor(t1371_bert_encoder_layer_1_attention_self_Softmax_0_max_sizes,
                     t1371_bert_encoder_layer_1_attention_self_Softmax_0_max_sizes,
                     syn_type_bf16,
                     false);

    unsigned char bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_params[] = {0, 0, 0, 0};
    const auto    softmax = NodeFactory::createGenericTPCNode(
        {t1370_bert_encoder_layer_1_attention_self_add_0},
        {t1371_bert_encoder_layer_1_attention_self_Softmax_0},
        bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_params,
        sizeof(bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_params),
        "softmax_fwd_bf16",
        "bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0");
    EXPECT_TRUE(GraphEditor::addNode(m_graph, softmax)) << "Failed adding node: " << softmax->getNodeName();
    /*************
     * bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0
     *node inputs: t1371_bert_encoder_layer_1_attention_self_Softmax_0[512, 512, 16, 28] (dtype=bf16)
     *     t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0[1]
     *(dtype=int32) outputs: t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28]
     *(dtype=bf16) t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28] (dtype=int8) ctrl
     *inputs: ctrl outputs:
     *************/

    // create t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0
    // tensor
    std::vector<TSize>
        t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_max_sizes =
            {1};
    const auto t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0 =
        createTensor(
            t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_max_sizes,
            t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_max_sizes,
            syn_type_int32,
            true);
    // create t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 tensor
    std::vector<TSize> t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes = {512, 512, 16, 28};
    const auto         t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 =
        createTensor(t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes,
                     t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes,
                     syn_type_bf16,
                     true);

    // create t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 tensor
    std::vector<TSize> t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes = {512, 512, 16, 28};
    const auto         t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 =
        createTensor(t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes,
                     t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes,
                     syn_type_int8,
                     true);
    unsigned char
        bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_params
            []         = {205, 204, 204, 61, 0, 0, 0, 0};
    const auto dropout = NodeFactory::createGenericTPCNode(
        {t1371_bert_encoder_layer_1_attention_self_Softmax_0,
         t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0},
        {t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0,
         t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0},
        (void*)
            bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_params,
        sizeof(
            bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_params),
        "dropout_fwd_bf16",
        "bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_"
        "0");
    EXPECT_TRUE(GraphEditor::addNode(m_graph, dropout)) << "Failed adding node: " << dropout->getNodeName();

    /*************
     * bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0 node
     * inputs:
     *     t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28] (dtype=bf16)
     *     t1366_bert_encoder_layer_1_attention_self_transpose_2_0[64, 512, 16, 28] (dtype=bf16)
     * outputs:
     *     t1374_bert_encoder_layer_1_attention_self_MatMul_1_0[64, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1374_bert_encoder_layer_1_attention_self_MatMul_1_0 tensor
    std::vector<TSize> t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_max_sizes = {64, 512, 16, 28};
    const auto         t1374_bert_encoder_layer_1_attention_self_MatMul_1_0 =
        createTensor(t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_max_sizes,
                     t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_max_sizes,
                     syn_type_bf16,
                     true);
    unsigned char bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_params[] = {0, 0};
    const auto    bgemm =
        NodeFactory::createNode({t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0,
                                 t1366_bert_encoder_layer_1_attention_self_transpose_2_0},
                                {t1374_bert_encoder_layer_1_attention_self_MatMul_1_0},
                                (void*)bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_params,
                                sizeof(bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_params),
                                "batch_gemm",
                                "bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0");

    EXPECT_TRUE(GraphEditor::addNode(m_graph, bgemm)) << "Failed adding node: " << bgemm->getNodeName();
    gaudi2::loadTpcKernels(m_graph);
    auto input0Finder = getFinder(bgemm->getInput(0));
    validateFinderRoute(input0Finder, {dropout, softmax, add1, mult0});
    auto input1Finder = getFinder(bgemm->getInput(1));
    validateFinderRoute(input1Finder, {transpose0, reshape0, add0});
}