
#include "layered_brain_test_common.h"
#include "node_factory.h"
#include "bundler/bundle_seed_collector_factory.h"
#include "gaudi2_graph.h"
#include "graph_editor.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "synapse_common_types.h"
#include "bundler/bundle_seed_collectors.h"
#include "gtest/gtest.h"

using namespace gc::layered_brain;
using namespace gc::layered_brain::bundler;

class BundleSeedCollectorTest : public LayeredBrainCommonTest<Gaudi2Graph>
{
};

TEST_F(BundleSeedCollectorTest, single_batch_gemm_single_bgemm_graph)
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

    const auto& collector = std::make_shared<SingleBatchGemmCollector>(m_graph);
    ASSERT_TRUE(collector != nullptr);
    const auto seeds = collector->collect();
    ASSERT_EQ(seeds.size(), 1);
}

TEST_F(BundleSeedCollectorTest, isolated_single_batch_gemm_single_gemm_graph)
{
    constexpr unsigned batch0  = 4;
    constexpr unsigned batch1  = 3;
    const auto         gemmIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto         gemmIn1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto         gemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto         gemm =
        NodeFactory::createNode({gemmIn0, gemmIn1}, {gemmOut}, nullptr, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(gemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, gemm)) << "Failed adding node: " << gemm->getNodeName();

    const auto& collector = std::make_shared<SingleBatchGemmCollector>(m_graph);
    ASSERT_TRUE(collector != nullptr);
    const auto seeds = collector->collect();
    // Expecting collector to skip gemm since collector is searching for isolated bgemm seeds
    ASSERT_EQ(seeds.size(), 0);
}

TEST_F(BundleSeedCollectorTest, isolated_single_batch_gemm_batch_gemm_same_input_operand)
{
    constexpr unsigned batch0   = 4;
    constexpr unsigned batch1   = 3;
    const auto         bgemmIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto         bgemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto         bgemm =
        NodeFactory::createNode({bgemmIn0, bgemmIn0}, {bgemmOut}, nullptr, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(bgemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm)) << "Failed adding node: " << bgemm->getNodeName();

    const auto addIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto addIn1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add    = NodeFactory::createNode({addIn0, addIn1}, {bgemmIn0}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add)) << "Failed adding node: " << add->getNodeName();

    gaudi2::loadTpcKernels(m_graph);

    const auto& collector = std::make_shared<SingleBatchGemmCollector>(m_graph);
    ASSERT_TRUE(collector != nullptr);
    const auto seeds = collector->collect();

    // Expecting collector to skip gemm since collector is searching for isolated bgemm seeds
    ASSERT_EQ(seeds.size(), 0);
}

TEST_F(BundleSeedCollectorTest, two_batch_gemms_with_shared_input)
{
    constexpr unsigned batch0        = 4;
    constexpr unsigned batch1        = 3;
    const auto         bgemmSharedIn = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);

    const auto bgemm0In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto bgemm0Out = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto bgemm0    = NodeFactory::createNode({bgemmSharedIn, bgemm0In1},
                                                {bgemm0Out},
                                                nullptr,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm_0");
    ASSERT_TRUE(bgemm0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm0)) << "Failed adding node: " << bgemm0->getNodeName();

    const auto bgemm1In0 = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    const auto bgemm1Out = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    const auto bgemm1    = NodeFactory::createNode({bgemm1In0, bgemmSharedIn},
                                                {bgemm1Out},
                                                nullptr,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm_1");
    ASSERT_TRUE(bgemm1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm1)) << "Failed adding node: " << bgemm1->getNodeName();

    const auto& collector = std::make_shared<SingleBatchGemmCollector>(m_graph);
    ASSERT_TRUE(collector != nullptr);
    const auto seeds = collector->collect();

    // if multi seed collection is enabled and hybrid mode is disabled, single gemm collectors collect isolated nodes
    // assuming multi-mme collectors had their try.
    // otherwise, single mme collectors search for isolated mme nodes and in our case there are 0
    const auto expectedNumSeeds = MmeNodeCollector::bundleOnlyIsolatedMmeNodes() ? 0 : 2;
    ASSERT_EQ(seeds.size(), expectedNumSeeds);
}

/**
                   +-------------+
                   |             |
       +----+   +------+     +---v---+   +---------+   +----+
       |add0+-->|bgemm0|     |bgemm1 +-->|transpose+-->|add1|
       +----+   +------+     +-------+   +---------+   +----+

 */
TEST_F(BundleSeedCollectorTest, two_batch_gemm_seeds)
{
    constexpr unsigned batch0 = 4;
    constexpr unsigned batch1 = 3;

    const auto bgemm0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto bgemm0In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto bgemm0Out = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto bgemm0    = NodeFactory::createNode({bgemm0In0, bgemm0In1},
                                                {bgemm0Out},
                                                nullptr,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm_0");
    ASSERT_TRUE(bgemm0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm0)) << "Failed adding node: " << bgemm0->getNodeName();

    const auto add0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0In1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {bgemm0In0}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto bgemm1In0 = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    const auto bgemm1In1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto bgemm1Out = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    const auto bgemm1    = NodeFactory::createNode({bgemm1In0, bgemm1In1},
                                                {bgemm1Out},
                                                nullptr,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm_1");
    ASSERT_TRUE(bgemm1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm1)) << "Failed adding node: " << bgemm1->getNodeName();

    const auto         transposeOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    synTransposeParams transposeParams {.permutation = {TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch},
                                        .tensorDim   = 4};

    const auto transpose = NodeFactory::createNode({bgemm1Out},
                                                   {transposeOut},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose_bgemm_1");
    ASSERT_TRUE(transpose != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, transpose)) << "Failed adding node: " << transpose->getNodeName();

    const auto add1Out = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto add1    = NodeFactory::createNode({bgemm0Out, transposeOut}, {add1Out}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const auto& collector = std::make_shared<SingleBatchGemmCollector>(m_graph);
    ASSERT_TRUE(collector != nullptr);
    const auto seeds = collector->collect();
    ASSERT_EQ(seeds.size(), 2);
}

TEST_F(BundleSeedCollectorTest, single_batch_gemm_two_batch_gemm_seeds_already_bundled)
{
    constexpr unsigned batch0 = 4;
    constexpr unsigned batch1 = 3;

    const auto bgemm0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto bgemm0In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto bgemm0Out = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    auto       bgemm0    = NodeFactory::createNode({bgemm0In0, bgemm0In1},
                                          {bgemm0Out},
                                          nullptr,
                                          NodeFactory::batchGemmNodeTypeName,
                                          "bgemm_0");
    ASSERT_TRUE(bgemm0 != nullptr);
    GraphEditor::addNode(m_graph, bgemm0);
    auto bundle0 = Bundle::create(m_graph);
    bundle0->add(bgemm0);

    const auto bgemm1In0 = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    const auto bgemm1In1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto bgemm1Out = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    auto       bgemm1    = NodeFactory::createNode({bgemm1In0, bgemm1In1},
                                          {bgemm1Out},
                                          nullptr,
                                          NodeFactory::batchGemmNodeTypeName,
                                          "bgemm_1");
    ASSERT_TRUE(bgemm1 != nullptr);
    GraphEditor::addNode(m_graph, bgemm1);
    auto bundle1 = Bundle::create(m_graph);
    bundle1->add(bgemm1);

    const auto         transposeOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    synTransposeParams transposeParams {.permutation = {TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch},
                                        .tensorDim   = 4};

    const auto transpose = NodeFactory::createNode({bgemm1Out},
                                                   {transposeOut},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose_bgemm_1");
    ASSERT_TRUE(transpose != nullptr);
    GraphEditor::addNode(m_graph, transpose);

    const auto addOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto add    = NodeFactory::createNode({bgemm0Out, transposeOut}, {addOut}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add != nullptr);
    GraphEditor::addNode(m_graph, add);
    gaudi2::loadTpcKernels(m_graph);

    const auto& collector = std::make_shared<SingleBatchGemmCollector>(m_graph);
    ASSERT_TRUE(collector != nullptr);
    const auto seeds = collector->collect();
    ASSERT_EQ(seeds.size(), 0);
}

TEST_F(BundleSeedCollectorTest, single_broadcast_batch_gemm)
{
    // Batch gemm
    synGEMMParams gemmParams {};
    TensorPtr     relu_out   = createTensor(SizeVector {64, 64, 16}, syn_type_bf16);
    TensorPtr     b          = createTensor(SizeVector {64, 64}, syn_type_bf16);
    TensorPtr     gemmOut    = createTensor(SizeVector {64, 64, 16}, syn_type_bf16);
    NodePtr       batch_gemm = NodeFactory::createNode({relu_out, b},
                                                 {gemmOut},
                                                 &gemmParams,
                                                 NodeFactory::batchGemmNodeTypeName,
                                                 "batch_gemm");
    ASSERT_TRUE(GraphEditor::addNode(m_graph, batch_gemm));

    // Relu producer
    TensorPtr relu_in = createTensor(SizeVector {64, 64, 16}, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({relu_in}, {relu_out}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(m_graph, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));
    ASSERT_TRUE(alignAsymmetricBgemm(m_graph));
    const auto& collector = std::make_shared<SingleBatchGemmCollector>(m_graph);
    ASSERT_TRUE(collector != nullptr);
    const auto seeds = collector->collect();
    ASSERT_EQ(seeds.size(), 1);
}

TEST_F(BundleSeedCollectorTest, attention_bundle_seed)
{
    setGlobalConfForTest(GCFG_ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS, "True");

    //                [t2]          [t4]                      [t7]
    //                 |             |                         |
    // [t0]->TPC1->[t1]->BGEMM1->[t3]->BGEMM2->[t5]->TPC2->[t6]->BGEMM3->[t8]

    const synDataType dt    = syn_type_single;

    TensorVector tensors;
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t0
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t1
    tensors.push_back(createTensor(SizeVector {64, 128, 128, 128}, dt));   // t2
    tensors.push_back(createTensor(SizeVector {64, 128, 128, 128}, dt));   // t3
    tensors.push_back(createTensor(SizeVector {128, 64, 128, 128}, dt));   // t4
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t5
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t6
    tensors.push_back(createTensor(SizeVector {64, 128, 128, 128}, dt));   // t7
    tensors.push_back(createTensor(SizeVector {64, 128, 128, 128}, dt));   // t8

    const auto tpc1 = NodeFactory::createNode({tensors.at(0)}, {tensors.at(1)}, nullptr, "relu_fwd_f32", "tpc1");
    ASSERT_TRUE(tpc1);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, tpc1));

    synGEMMParams params;
    const auto    bgemm1 = NodeFactory::createNode({tensors.at(1), tensors.at(2)},
                                                {tensors.at(3)},
                                                &params,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm1");
    ASSERT_TRUE(bgemm1);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm1));

    const auto bgemm2 = NodeFactory::createNode({tensors.at(3), tensors.at(4)},
                                                {tensors.at(5)},
                                                &params,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm2");
    ASSERT_TRUE(bgemm2);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm2));

    const auto tpc2 = NodeFactory::createNode({tensors.at(5)}, {tensors.at(6)}, nullptr, "relu_fwd_f32", "tpc2");
    ASSERT_TRUE(tpc2);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, tpc2));

    const auto bgemm3 = NodeFactory::createNode({tensors.at(6), tensors.at(7)},
                                                {tensors.at(8)},
                                                &params,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm3");
    ASSERT_TRUE(bgemm3);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm3));

    gaudi2::loadTpcKernels(m_graph);
    const auto& collector = std::make_shared<AttentionCollector>(m_graph);
    ASSERT_TRUE(collector);
    const auto& seeds = collector->collect();

    // Expected a single seed: BGEMM2 -> TPC2 - > BGEMM3
    ASSERT_EQ(seeds.size(), 1);
    const auto& seedNodes = seeds.front().first->getNodes();
    ASSERT_EQ(seedNodes.size(), 3);
    ASSERT_TRUE(seedNodes.find(bgemm2) != seedNodes.end());
    ASSERT_TRUE(seedNodes.find(tpc2) != seedNodes.end());
    ASSERT_TRUE(seedNodes.find(bgemm3) != seedNodes.end());
}

TEST_F(BundleSeedCollectorTest, attention_bundle_seed_with_logicals)
{
    setGlobalConfForTest(GCFG_ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS, "True");

    //    [t1]                                                          [t6]
    //     |                                                             |
    // [t0]->BGEMM1->[t2]->[Reshape1]->[t3]->TPC1->[t4]->[Reshape2]->[t5]->BGEMM2->[t7]

    const synDataType dt    = syn_type_single;

    TensorVector tensors;
    tensors.push_back(createTensor(SizeVector {64, 128, 128, 128}, dt));   // t0
    tensors.push_back(createTensor(SizeVector {128, 64, 128, 128}, dt));   // t1
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t2
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t3
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t4
    tensors.push_back(createTensor(SizeVector {128, 128, 128, 128}, dt));  // t5
    tensors.push_back(createTensor(SizeVector {64, 128, 128, 128}, dt));   // t6
    tensors.push_back(createTensor(SizeVector {64, 128, 128, 128}, dt));   // t7

    synGEMMParams params;
    const auto    bgemm1 = NodeFactory::createNode({tensors.at(0), tensors.at(1)},
                                                {tensors.at(2)},
                                                &params,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm1");
    ASSERT_TRUE(bgemm1);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm1));

    const auto reshape1 = NodeFactory::createNode({tensors.at(2)},
                                                  {tensors.at(3)},
                                                  nullptr,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  "reshape1");
    ASSERT_TRUE(reshape1);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape1));

    const auto tpc1 = NodeFactory::createNode({tensors.at(3)}, {tensors.at(4)}, nullptr, "relu_fwd_f32", "tpc1");
    ASSERT_TRUE(tpc1);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, tpc1));

    const auto reshape2 = NodeFactory::createNode({tensors.at(4)},
                                                  {tensors.at(5)},
                                                  nullptr,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  "reshape2");
    ASSERT_TRUE(reshape2);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape2));

    const auto bgemm2 = NodeFactory::createNode({tensors.at(5), tensors.at(6)},
                                                {tensors.at(7)},
                                                &params,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm2");
    ASSERT_TRUE(bgemm2);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm2));

    gaudi2::loadTpcKernels(m_graph);
    const auto& collector = std::make_shared<AttentionCollector>(m_graph);
    ASSERT_TRUE(collector);
    const auto& seeds = collector->collect();

    // Expected a single seed: BGEMM1 -> Reshape1 -> TPC1 -> Reshape2 -> BGEMM2
    ASSERT_EQ(seeds.size(), 1);
    const auto& seedNodes = seeds.front().first->getNodes();
    ASSERT_EQ(seedNodes.size(), 5);
    for (const auto& n : m_graph.getNodes())
    {
        ASSERT_TRUE(seedNodes.find(n) != seedNodes.end());
    }
}