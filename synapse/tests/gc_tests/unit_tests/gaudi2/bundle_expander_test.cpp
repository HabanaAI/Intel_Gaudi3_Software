#include "bundle_plane_graph.h"
#include "layered_brain_test_common.h"
#include "gaudi2_graph.h"
#include "graph_editor.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "graph_visualization.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "supported_devices_macros.h"
#include "synapse_common_types.h"
#include "graph_optimizer_test.h"
#include "bundler/bundle_expanders.h"
#include "node_factory.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "layered_brain_test.h"

using namespace gc::layered_brain;
using namespace gc::layered_brain::bundler;

class BundleExpanderTest : public LayeredBrainCommonTest<Gaudi2Graph>
{
};

class OperandExpanderTest : public BundleExpanderTest
{
protected:
    BundleExpanderPtr getOperandExpander(const TensorPtr& operand, const BundlePtr& bundle, BundleExpander::Type type)
    {
        switch (type)
        {
            case BundleExpander::Type::TPC_PRODUCERS:
                return std::make_shared<TpcProducersExpander>(operand, bundle, m_graph);
            case BundleExpander::Type::FIRST_TPC_PRODUCER:
                return std::make_shared<FirstTpcProducerExpander>(operand, bundle, m_graph);
            case BundleExpander::Type::FIRST_TPC_CONSUMER:
                return std::make_shared<FirstTpcConsumerExpander>(operand, bundle, m_graph);
            default:
                return nullptr;
        }
    }
};

using ExpanderInfo                  = std::tuple<bool, unsigned, BundleExpander::Type>;
using ExpectedExpansion             = std::vector<unsigned>;
using OperandExpanderTestParams     = std::tuple<ExpanderInfo, ExpectedExpansion>;
static constexpr bool inputOperand  = true;
static constexpr bool outputOperand = !inputOperand;

class OperandExpanderMilestone0Test
: public OperandExpanderTest
, public ::testing::WithParamInterface<OperandExpanderTestParams>
{
protected:
    std::vector<NodePtr> makeMilestone0Graph();
};

class OperandExpanderGraph0
: public OperandExpanderTest
, public ::testing::WithParamInterface<std::tuple<ExpanderInfo, ExpectedExpansion>>
{
protected:
    std::vector<NodePtr> makeGraph0();
};

/*
┌──────┐      ┌─────┐      ┌────────┐       ┌────────┐         ┌──────┐
│ mult ├─────►│ add ├─────►│ softmax├──────►│ dropout├────────►│      │
└──────┘      └─────┘      └────────┘       └────────┘         │      │
                                                               │ batch│
                                                               │ gemm │
         ┌─────┐      ┌────────┐      ┌────────────┐           │      │
         │add  ├─────►│reshape ├─────►│transpose   ├──────────►│      │
         └─────┘      └────────┘      └────────────┘           └──────┘

 Return vector: [add, reshape, transpose, mult, add, softmax, dropout, batch gemm]
 */
std::vector<NodePtr> OperandExpanderMilestone0Test::makeMilestone0Graph()
{
    std::vector<NodePtr> createdNodes;
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
    createdNodes.push_back(add0);

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
    createdNodes.push_back(reshape0);
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
    createdNodes.push_back(transpose0);
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
    createdNodes.push_back(mult0);
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
    createdNodes.push_back(add1);
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
    createdNodes.push_back(softmax);
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
    createdNodes.push_back(dropout);

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
    createdNodes.push_back(bgemm);

    EXPECT_TRUE(GraphEditor::addNode(m_graph, bgemm)) << "Failed adding node: " << bgemm->getNodeName();
    gaudi2::loadTpcKernels(m_graph);
    return createdNodes;
}

/*
┌────────┐       ┌────────┐         ┌──────┐
│ softmax├──────►│ dropout├────────►│      │
└────────┘       └────────┘         │      │      ┌─────┐
                                    │ batch│─────►│add0 ├
                                    │ gemm │      └─────┘
           ┌────────────┐           │      │
           │ add1       ├──────────►│      │
           └────────────┘           └──────┘
Return vector: [bgemm, add0, dropout, softmax, add1]
 */
std::vector<NodePtr> OperandExpanderGraph0::makeGraph0()
{
    std::vector<NodePtr> createdNodes;

    constexpr unsigned batch0   = 4;
    constexpr unsigned batch1   = 3;
    const auto         bgemmIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto         bgemmIn1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto         bgemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);

    const auto bgemm =
        NodeFactory::createNode({bgemmIn0, bgemmIn1}, {bgemmOut}, nullptr, NodeFactory::batchGemmNodeTypeName, "bgemm");
    HB_ASSERT(bgemm != nullptr, "");
    HB_ASSERT(GraphEditor::addNode(m_graph, bgemm), "Failed adding node: {}", bgemm->getNodeName());
    createdNodes.push_back(bgemm);

    const auto add0In1 = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto add0Out = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto add0 = NodeFactory::createGenericTPCNode({bgemmOut, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    HB_ASSERT(add0 != nullptr, "");
    HB_ASSERT(GraphEditor::addNode(m_graph, add0), "Failed adding node: {}", add0->getNodeName());
    createdNodes.push_back(add0);

    const auto dropoutIn0      = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    char       dropoutParams[] = {205, 204, 204, 61, 0, 0, 0, 0};
    const auto dropout         = NodeFactory::createGenericTPCNode({dropoutIn0},
                                                           {bgemmIn0},
                                                           dropoutParams,
                                                           sizeof(dropoutParams),
                                                           "dropout_fwd_f32",
                                                           "dropout");
    HB_ASSERT(dropout != nullptr, "");
    HB_ASSERT(GraphEditor::addNode(m_graph, dropout), "Failed adding node: {}", dropout->getNodeName());
    createdNodes.push_back(dropout);

    const auto softmaxIn0      = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    char       softmaxParams[] = {0, 0, 0, 0};
    const auto softmax         = NodeFactory::createGenericTPCNode({softmaxIn0},
                                                           {dropoutIn0},
                                                           softmaxParams,
                                                           sizeof(softmaxParams),
                                                           "softmax_fwd_f32",
                                                           "softmax");
    HB_ASSERT(softmax != nullptr, "");
    HB_ASSERT(GraphEditor::addNode(m_graph, softmax), "Failed adding node: {}", softmax->getNodeName());
    createdNodes.push_back(softmax);

    const auto add1In0 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1 = NodeFactory::createGenericTPCNode({add1In0, add1In1}, {bgemmIn1}, nullptr, "add_fwd_f32", "add1");
    HB_ASSERT(add1 != nullptr, "");
    HB_ASSERT(GraphEditor::addNode(m_graph, add1), "Failed adding node: {}", add1->getNodeName());
    createdNodes.push_back(add1);

    gaudi2::loadTpcKernels(m_graph);
    return createdNodes;
}

TEST_P(OperandExpanderMilestone0Test, bemm_ms0)
{
    const auto createdNodes = makeMilestone0Graph();

    // sanity for expected graph
    ASSERT_EQ(createdNodes.size(), 8);
    const auto bgemm = createdNodes.back();
    ASSERT_EQ(bgemm->getNodeType(), Node::TYPE_BATCH_GEMM);

    // add bgemm seed to bundle
    auto bundle = Bundle::create(m_graph);
    bundle->add(bgemm);

    // acquire operand for expander
    const auto [expanderInfo, expectedExpansion] = GetParam();
    const auto [isInput, idx, type]              = expanderInfo;
    const auto& operands                         = isInput ? bgemm->getInputs() : bgemm->getOutputs();
    ASSERT_LT(idx, operands.size());
    const auto& operand = operands.at(idx);

    // create expander and trigger expansion
    const auto expander = getOperandExpander(operand, bundle, type);
    expander->expand();

    // validate all nodes expected to be in expanded bundle are in it
    const auto& bundleNodes = bundle->getNodes();
    for (const auto& idx : expectedExpansion)
    {
        EXPECT_LT(idx, createdNodes.size());
        const auto& node = createdNodes.at(idx);
        EXPECT_NE(bundleNodes.find(node), bundleNodes.end())
            << fmt::format("Expected node {}, type {}, createdNodes[{}] to be in the bundle",
                           node->getNodeName(),
                           node->getNodeTypeStr(),
                           idx);
    }
}

// Return vector: [add, reshape, transpose, mult, add, softmax, dropout, batch gemm]
INSTANTIATE_TEST_SUITE_P(
    ,
    OperandExpanderMilestone0Test,
    ::testing::Values(
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 0, BundleExpander::Type::FIRST_TPC_PRODUCER},
                                                         {6 /*dropout*/}),
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 0, BundleExpander::Type::TPC_PRODUCERS},
                                                         {6 /*dropout*/, 5 /*softmax*/}),
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 1, BundleExpander::Type::FIRST_TPC_PRODUCER},
                                                         {2 /*transpose*/, 1 /*reshape*/, 0 /*add*/}),
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 1, BundleExpander::Type::TPC_PRODUCERS}, {})));

TEST_P(OperandExpanderGraph0, bgemm_two_in0_2_tpc_producers_in1_1_tpc_producer_1_tpc_consumer)
{
    const auto createdNodes = makeGraph0();

    // sanity for expected graph
    ASSERT_EQ(createdNodes.size(), 5);
    const auto bgemm = createdNodes.front();
    ASSERT_EQ(bgemm->getNodeType(), Node::TYPE_BATCH_GEMM);

    // add bgemm seed to bundle
    auto bundle = Bundle::create(m_graph);
    bundle->add(bgemm);

    // acquire operand for expander
    const auto [expanderInfo, expectedExpansion] = GetParam();
    const auto [isInput, idx, type]              = expanderInfo;
    const auto& operands                         = isInput ? bgemm->getInputs() : bgemm->getOutputs();
    ASSERT_LT(idx, operands.size());
    const auto& operand = operands.at(idx);

    // create expander and trigger expansion
    const auto expander = getOperandExpander(operand, bundle, type);
    expander->expand();

    // validate all nodes expected to be in expanded bundle are in it
    const auto& bundleNodes = bundle->getNodes();
    for (const auto& idx : expectedExpansion)
    {
        EXPECT_LT(idx, createdNodes.size());
        const auto& node = createdNodes.at(idx);
        EXPECT_NE(bundleNodes.find(node), bundleNodes.end())
            << fmt::format("Expected node {}, type {}, createdNodes[{}] to be in the bundle",
                           node->getNodeName(),
                           node->getNodeTypeStr(),
                           idx);
    }
}

// Return vector: [bgemm, add0, dropout, softmax, add1]
INSTANTIATE_TEST_SUITE_P(
    ,
    OperandExpanderGraph0,
    ::testing::Values(
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 0, BundleExpander::Type::FIRST_TPC_PRODUCER},
                                                         {2 /*dropout*/}),
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 0, BundleExpander::Type::TPC_PRODUCERS},
                                                         {2 /*dropout*/, 3 /*softmax*/}),
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({outputOperand, 0, BundleExpander::Type::FIRST_TPC_CONSUMER},
                                                         {1 /*add0*/}),
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 1, BundleExpander::Type::FIRST_TPC_PRODUCER},
                                                         {4 /*add1*/}),
        std::make_tuple<ExpanderInfo, ExpectedExpansion>({inputOperand, 1, BundleExpander::Type::TPC_PRODUCERS}, {})));

class MMEOperandExpanderTest : public LayeredBrainTest
{
protected:
    void reshape(const TensorPtr& t, const std::vector<TSize>& shape) const { t->reshape(shape.size(), shape.data()); }

    // Replaces the 'idx'th node in the chain with a MME node
    void replaceWithMME(size_t idx, TSize w, TSize h, TSize cd)
    {
        NodePtr n = m_nodeChain.at(idx);
        ASSERT_EQ(n->getNumInputs(), 1);
        ASSERT_EQ(n->getNumOutputs(), 1);
        TensorPtr a   = n->getInput(0);
        TensorPtr b   = newTensor();
        TensorPtr out = n->getOutput(0);
        reshape(a, {cd, h});
        reshape(b, {w, cd});
        reshape(out, {w, h});
        synGEMMParams params;
        NodePtr       mme =
            NodeFactory::createNode({a, b}, {out}, &params, NodeFactory::gemmNodeTypeName, fmt::format("gemm{}", idx));
        m_nodeChain[idx]  = mme;
        ASSERT_TRUE(GraphEditor::replaceNodes(m_graph, {n}, {mme}) == REPLACE_NODE_SUCCESS);
    }

    // Replaces the 'idx'th node in the chain with a reshape node
    void replaceWithReshape(size_t idx)
    {
        NodePtr n = m_nodeChain.at(idx);
        ASSERT_EQ(n->getNumInputs(), 1);
        ASSERT_EQ(n->getNumOutputs(), 1);
        NodePtr reshape  = NodeFactory::createNode(n->getInputs(),
                                                  n->getOutputs(),
                                                  nullptr,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  fmt::format("reshape{}", idx));
        m_nodeChain[idx] = reshape;
        ASSERT_TRUE(GraphEditor::replaceNodes(m_graph, {n}, {reshape}) == REPLACE_NODE_SUCCESS);
    }
};

TEST_F(MMEOperandExpanderTest, test_mme_expander_tpc_chain)
{
    // TPC0 -> TPC1 -> TPC2
    createGraph(3);

    BPGraphContext bpgCtx(m_graph);
    auto           bundle = Bundle::create(m_graph);

    // Start expanding from TPC1 input
    const auto producerExpander =
        std::make_shared<IterativeFirstMmeProducerExpander>(m_nodeChain.at(1)->getInput(0), bundle, m_graph);
    ASSERT_FALSE(producerExpander->expand());

    // Start expanding from TPC1 output
    const auto consumerExpander =
        std::make_shared<IterativeFirstMmeConsumerExpander>(m_nodeChain.at(1)->getOutput(0), bundle, m_graph);
    ASSERT_FALSE(consumerExpander->expand());
}

TEST_F(MMEOperandExpanderTest, test_mme_expander_mme_tpc_mme_chain)
{
    // MME0 -> TPC1 -> MME2
    createGraph(3);
    replaceWithMME(0, 128, 128, 64);  // Larger output to comply with MME_OUTPUTS_LARGER_THAN_INPUTS rule
    replaceWithMME(2, 64, 128, 128);  // Larger input to comply with MME_INPUTS_LARGER_THAN_OUTPUTS rule

    BPGraphContext bpgCtx(m_graph);
    auto           bundle = Bundle::create(m_graph);

    // Start expanding from TPC1 input
    const auto producerExpander =
        std::make_shared<IterativeFirstMmeProducerExpander>(m_nodeChain.at(1)->getInput(0), bundle, m_graph);
    ASSERT_TRUE(producerExpander->expand());

    ASSERT_EQ(bundle->getNodes().size(), 1);  // Added node: MME0

    // Start expanding from TPC1 output
    const auto consumerExpander =
        std::make_shared<IterativeFirstMmeConsumerExpander>(m_nodeChain.at(1)->getOutput(0), bundle, m_graph);
    ASSERT_TRUE(consumerExpander->expand());

    ASSERT_EQ(bundle->getNodes().size(), 2);  // Added node: MME2
}

TEST_F(MMEOperandExpanderTest, test_mme_expander_mme_tpc_mme_chain_with_logicals)
{
    // MME0 -> Reshape1 -> TPC2 -> Reshape3 -> Reshape4 -> MME5
    createGraph(6);
    replaceWithReshape(1);
    replaceWithReshape(3);
    replaceWithReshape(4);
    replaceWithMME(0, 128, 128, 64);  // Larger output to comply with MME_OUTPUTS_LARGER_THAN_INPUTS rule
    replaceWithMME(5, 64, 128, 128);  // Larger input to comply with MME_INPUTS_LARGER_THAN_OUTPUTS rule

    BPGraphContext bpgCtx(m_graph);
    auto           bundle = Bundle::create(m_graph);

    // Start expanding from TPC2 input
    const auto producerExpander =
        std::make_shared<IterativeFirstMmeProducerExpander>(m_nodeChain.at(2)->getInput(0), bundle, m_graph);
    ASSERT_TRUE(producerExpander->expand());

    ASSERT_EQ(bundle->getNodes().size(), 2);  // Added nodes: MME0, Reshape1

    // Start expanding from TPC2 output
    const auto consumerExpander =
        std::make_shared<IterativeFirstMmeConsumerExpander>(m_nodeChain.at(2)->getOutput(0), bundle, m_graph);
    ASSERT_TRUE(consumerExpander->expand());

    ASSERT_EQ(bundle->getNodes().size(), 5);  // Added nodes: Reshape3, Reshape4, MME5
}
