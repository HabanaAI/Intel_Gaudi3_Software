#include "passes.h"
#include "passes/sram_management/bundle.h"
#include "passes/sram_management/bundle_slicer.h"
#include "passes/sram_management/sram_management.h"
#include "node_factory.h"
#include "graph_optimizer_test.h"
#include "gaudi_graph.h"
#include "compilation_hal_reader.h"
#include "hal_reader/hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "passes/sram_management/solution_generator.h"
#include "passes/sram_management/tpc_bundle_solver.h"

class TPCBundleTest : public GraphOptimizerTest
{
public:
    virtual void SetUp() override;
};

using namespace gaudi;

void TPCBundleTest::SetUp()
{
    GraphOptimizerTest::SetUp();
    CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
}

TEST_F(TPCBundleTest, tpc_bundle_solution_generator_test)
{
    GaudiGraph graph;
    SizeArray dataDims = {128, 128, 12, 1, 1};

    pTensor firstOut = pTensor(new Tensor(3, dataDims.data(), syn_type_float));
    pTensor secondOut = firstOut->clone();
    pTensor thirdOut = firstOut->clone();
    thirdOut->setMemoryDescriptor(synMemoryDescriptor(true));

    pNode add1 = NodeFactory::createNode({firstOut->clone(), firstOut->clone()}, {firstOut}, nullptr, 0, "add_fwd_f32", "first_add");
    pNode add2 = NodeFactory::createNode({firstOut, firstOut->clone()}, {secondOut}, nullptr, 0, "add_fwd_f32", "second_add");
    pNode add3 = NodeFactory::createNode({secondOut, firstOut->clone()}, {thirdOut}, nullptr, 0, "add_fwd_f32", "third_add");

    GraphEditor::addNode(graph, add1);
    GraphEditor::addNode(graph, add2);
    GraphEditor::addNode(graph, add3);

    loadTpcKernels(graph);

    pBundle bundle(new Bundle(BundleType::TPC));
    bundle->addNode(add1);
    bundle->addNode(add2);
    bundle->addNode(add3);

    TpcBundleSolver solver(*graph.getHALReader(), bundle);
    ASSERT_TRUE(solver.effectiveForBundle());
    solver.createAllStrategies();

    const auto& strategies = solver.getStrategies();
    ASSERT_EQ(strategies.size(), 1);
    auto& slicingData = strategies.front()->getSlicingData();
    slicingData.traversalPattern = {2};
    ASSERT_EQ(slicingData.bundleTensors.size(), 6);
    SizeArray sliceChunk = {128, 128, 6, 1, 1};
    slicingData.masterOperand->chunkDimensions = sliceChunk;
    for (auto operand : slicingData.bundleTensors)
    {
        operand->chunkDimensions = slicingData.masterOperand->chunkDimensions;
    }

    SolutionGenerator generator(graph, bundle, strategies.front());
    ASSERT_TRUE(generator.fillSolution());

    auto opIter = bundle->getSolution().operations.begin();
    ASSERT_EQ(bundle->getSolution().operations.size(), 6);
    ASSERT_EQ(opIter->originalNode, add1);
    ASSERT_EQ(opIter->outputs[0]->operand->chunkDimensions, sliceChunk);
    ASSERT_EQ((++opIter)->originalNode, add2);
    ASSERT_EQ((++opIter)->originalNode, add3);
    ASSERT_EQ((++opIter)->originalNode, add1);
    ASSERT_EQ((++opIter)->originalNode, add2);
    ASSERT_EQ((++opIter)->originalNode, add3);

    BundleSlicer::sliceBundle(*bundle, graph);
    const auto& nodes = graph.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 13);

    uint32_t nodeIdx = 0;
    for (const auto& node : nodes)
    {
        switch(nodeIdx)
        {
        case 0:
        case 1:
        case 3:
        case 5:
            EXPECT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_SPLIT) << "Wrong node at idx: " << nodeIdx;
            break;
        case 2:
        case 4:
        case 6:
        case 7:
        case 9:
        case 11:
            EXPECT_EQ(node->getNodeType(), Node::TYPE_USER) << "Wrong node at idx: " << nodeIdx;
            break;
        case 8:
        case 10:
        case 12:
            EXPECT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT) << "Wrong node at idx: " << nodeIdx;
            break;
        }
        ++nodeIdx;
    }
}

TEST_F(TPCBundleTest, tpc_bundle_with_reshape_full_management)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph graph;
    SizeArray dataDims = {128, 128, 1, 3, 1};
    SizeArray reshapeDims = {128, 128, 3, 1, 1};

    pTensor firstOut = pTensor(new Tensor(4, dataDims.data(), syn_type_float));
    pTensor secondOut = firstOut->clone();
    pTensor thirdOut = firstOut->clone();
    pTensor reshapeOut = pTensor(new Tensor(3, reshapeDims.data(), syn_type_float));
    pTensor fourthOut = reshapeOut->clone();
    pTensor reshape2Out = firstOut->clone();
    pTensor finalOut = reshape2Out->clone();
    finalOut->setMemoryDescriptor(synMemoryDescriptor(true));

    pNode batchGemm1 = NodeFactory::createNode({firstOut->clone(), firstOut->clone()}, {firstOut}, nullptr, 0, NodeFactory::batchGemmNodeTypeName, "first_batch_gemm");
    pNode add1 = NodeFactory::createNode({firstOut, firstOut->clone()}, {secondOut}, nullptr, 0, "add_fwd_f32", "first_add");
    pNode add2 = NodeFactory::createNode({secondOut, firstOut->clone()}, {thirdOut}, nullptr, 0, "add_fwd_f32", "second_add");
    pNode reshape = NodeFactory::createNode({thirdOut}, {reshapeOut}, nullptr, 0, NodeFactory::reshapeNodeTypeName, "reshape");
    pNode add3 = NodeFactory::createNode({reshapeOut, reshapeOut->clone()}, {fourthOut}, nullptr, 0, "add_fwd_f32", "third_add");
    pNode reshape2 = NodeFactory::createNode({fourthOut}, {reshape2Out}, nullptr, 0, NodeFactory::reshapeNodeTypeName, "second_reshape");
    pNode batchGemm2 = NodeFactory::createNode({reshape2Out, reshape2Out->clone()}, {finalOut}, nullptr, 0, NodeFactory::batchGemmNodeTypeName, "second_batch_gemm");

    GraphEditor::addNode(graph, batchGemm1);
    GraphEditor::addNode(graph, add1);
    GraphEditor::addNode(graph, add2);
    GraphEditor::addNode(graph, reshape);
    GraphEditor::addNode(graph, add3);
    GraphEditor::addNode(graph, reshape2);
    GraphEditor::addNode(graph, batchGemm2);

    loadTpcKernels(graph);

    graphVisualizationPre(graph);
    SRAMSlicingManager manager(graph);
    manager.sliceGraph();
    pBundle tpcBundle = manager.getBundlizer().findBundleByNode(add1);
    ASSERT_TRUE(tpcBundle != nullptr);

    graphVisualizationPost(graph);
    const auto& nodes = graph.getExeSortedNodes();
    if (GCFG_SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE.value())
    {
        //TODO: The switch below checking node types is entirely skipped here!
        // 1 + 2 ( first bgemm bundle) + 5 tpc bundle + 1 + 2 (second bgemm bundle)
        ASSERT_EQ(nodes.size(), 11);
    }
    else
    {
        // 12 * 2 (batch gemm bundles) + 21 tpc bundle
        ASSERT_EQ(nodes.size(), 45);
    }

    uint32_t nodeIdx = 0;
    for (const auto& node : nodes)
    {
        // Skip the first batch gemm  bundle - 3 * (2 memcpy + 1 gemm) + 2 split + 1 concat
        if (nodeIdx < 12) {++nodeIdx; continue;}

        switch(nodeIdx)
        {
        case 12:
        case 13:
        case 14:
        case 15:
            EXPECT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_SPLIT) << "Wrong node at idx: " << nodeIdx;
            break;
        case 16:
        case 17:
        case 19:
        case 21:
        case 22:
        case 24:
        case 26:
        case 27:
        case 29:
            EXPECT_EQ(node->getNodeType(), Node::TYPE_USER) << "Wrong node at idx: " << nodeIdx;
            break;
        case 18:
        case 20:
        case 23:
        case 25:
        case 28:
        case 30:
            EXPECT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_RESHAPE) << "Wrong node at idx: " << nodeIdx;
            break;
        case 31:
        case 32:
            EXPECT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_CONCAT) << "Wrong node at idx: " << nodeIdx;
            break;
        }
        ++nodeIdx;
        if (nodeIdx > 32) break;  // skip the third bundle
    }
}

TEST_F(TPCBundleTest, tpc_bundle_with_no_slice)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph graph;
    SizeArray dataDims = {64, 32, 1, 1, 1};
    SizeArray reshapeDims = {64, 1, 1, 32, 1};
    SizeArray gemmSecondIn = {64, 64};
    // Working with 1x4 geometry - no slice should be encountered for gemm nodes

    pTensor firstOut = pTensor(new Tensor(2, dataDims.data(), syn_type_float));
    pTensor operandB = pTensor(new Tensor(2, gemmSecondIn.data(), syn_type_float));
    pTensor secondOut = firstOut->clone();
    pTensor thirdOut = firstOut->clone();
    pTensor reshapeOut = pTensor(new Tensor(4, reshapeDims.data(), syn_type_float));
    pTensor fourthOut = reshapeOut->clone();
    pTensor reshape2Out = firstOut->clone();
    pTensor finalOut = reshape2Out->clone();
    finalOut->setMemoryDescriptor(synMemoryDescriptor(true));

    pNode gemm1 = NodeFactory::createNode({firstOut->clone(), operandB}, {firstOut}, nullptr, 0, NodeFactory::gemmNodeTypeName, "first_gemm");
    pNode add1 = NodeFactory::createNode({firstOut, firstOut->clone()}, {secondOut}, nullptr, 0, "add_fwd_f32", "first_add");
    pNode add2 = NodeFactory::createNode({secondOut, firstOut->clone()}, {thirdOut}, nullptr, 0, "add_fwd_f32", "second_add");
    pNode reshape = NodeFactory::createNode({thirdOut}, {reshapeOut}, nullptr, 0, NodeFactory::reshapeNodeTypeName, "reshape");
    pNode add3 = NodeFactory::createNode({reshapeOut, reshapeOut->clone()}, {fourthOut}, nullptr, 0, "add_fwd_f32", "third_add");
    pNode reshape2 = NodeFactory::createNode({fourthOut}, {reshape2Out}, nullptr, 0, NodeFactory::reshapeNodeTypeName, "second_reshape");
    pNode gemm2 = NodeFactory::createNode({reshape2Out, operandB}, {finalOut}, nullptr, 0, NodeFactory::gemmNodeTypeName, "second_gemm");

    GraphEditor::addNode(graph, gemm1);
    GraphEditor::addNode(graph, add1);
    GraphEditor::addNode(graph, add2);
    GraphEditor::addNode(graph, reshape);
    GraphEditor::addNode(graph, add3);
    GraphEditor::addNode(graph, reshape2);
    GraphEditor::addNode(graph, gemm2);

    loadTpcKernels(graph);

    graphVisualizationPre(graph);
    SRAMSlicingManager manager(graph);
    manager.sliceGraph();
    pBundle tpcBundle = manager.getBundlizer().findBundleByNode(add1);
    ASSERT_TRUE(tpcBundle != nullptr);

    graphVisualizationPost(graph);
    const auto& nodes = graph.getExeSortedNodes();
    // Expected no slicing - 4 memcpy to sram are added for gemm operation
    ASSERT_EQ(nodes.size(), 11);
}

TEST_F(TPCBundleTest, tpc_bundle_multiple_outputs)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");

    GaudiGraph graph;
    SizeArray  sizes = {2048, 2048};

    pTensor gemm1In1 = pTensor(new Tensor(2, sizes.data(), syn_type_float));
    pTensor gemm1In2 = pTensor(new Tensor(2, sizes.data(), syn_type_float));
    pTensor gemm1Out = pTensor(new Tensor(2, sizes.data(), syn_type_float));

    pTensor addIn  = pTensor(new Tensor(2, sizes.data(), syn_type_float));
    pTensor addOut = pTensor(new Tensor(2, sizes.data(), syn_type_float));

    pTensor reluOut = pTensor(new Tensor(2, sizes.data(), syn_type_float));

    pTensor gelu1Out1 = pTensor(new Tensor(2, sizes.data(), syn_type_float));
    pTensor gelu1Out2 = pTensor(new Tensor(2, sizes.data(), syn_type_float));

    pTensor gelu2Out1 = pTensor(new Tensor(2, sizes.data(), syn_type_float));
    pTensor gelu2Out2 = pTensor(new Tensor(2, sizes.data(), syn_type_float));

    pTensor castIn  = pTensor(new Tensor(2, sizes.data(), syn_type_bf16));
    pTensor castOut = pTensor(new Tensor(2, sizes.data(), syn_type_float));

    pTensor gemm2Out = pTensor(new Tensor(2, sizes.data(), syn_type_float));

    synGEMMParams gemmParams {};

    pNode gemm1 =
        NodeFactory::createNode({gemm1In1, gemm1In2}, {gemm1Out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM1");
    pNode add  = NodeFactory::createNode({gemm1Out, addIn}, {addOut}, nullptr, 0, "add_fwd_f32", "ADD");
    pNode relu = NodeFactory::createNode({addOut}, {reluOut}, nullptr, 0, "relu_fwd_f32", "RELU");

    pNode gelu1 = NodeFactory::createNode({reluOut}, {gelu1Out1, gelu1Out2}, nullptr, 0, "gelu_fwd_f32", "GELU1");
    pNode gelu2 = NodeFactory::createNode({reluOut}, {gelu2Out1, gelu2Out2}, nullptr, 0, "gelu_fwd_f32", "GELU2");

    pNode cast = NodeFactory::createNode({castIn}, {castOut}, nullptr, 0, "cast_bf16_to_f32", "CAST");
    pNode gemm2 =
        NodeFactory::createNode({gelu2Out2, castOut}, {gemm2Out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM2");

    ASSERT_TRUE(GraphEditor::addNode(graph, gemm1));
    ASSERT_TRUE(GraphEditor::addNode(graph, add));
    ASSERT_TRUE(GraphEditor::addNode(graph, relu));
    ASSERT_TRUE(GraphEditor::addNode(graph, gelu1));
    ASSERT_TRUE(GraphEditor::addNode(graph, gelu2));
    ASSERT_TRUE(GraphEditor::addNode(graph, cast));
    ASSERT_TRUE(GraphEditor::addNode(graph, gemm2));

    loadTpcKernels(graph);

    SRAMSlicingManager manager(graph);
    manager.sliceGraph();

    // Expecting one TPC bundle : Add->RELU->GELU2->CAST
    const pBundle& tpcBundle = manager.getBundlizer().findBundleByNode(add);
    ASSERT_TRUE(tpcBundle);
    const auto tpcBundleId = tpcBundle->index();
    ASSERT_TRUE(manager.getBundlizer().findBundleByNode(relu));
    ASSERT_TRUE(manager.getBundlizer().findBundleByNode(gelu2));
    ASSERT_TRUE(manager.getBundlizer().findBundleByNode(cast));
    ASSERT_EQ(manager.getBundlizer().findBundleByNode(relu)->index(), tpcBundleId);
    ASSERT_EQ(manager.getBundlizer().findBundleByNode(gelu2)->index(), tpcBundleId);
    ASSERT_EQ(manager.getBundlizer().findBundleByNode(cast)->index(), tpcBundleId);

    const pBundle& mmeConsumerBundle = manager.getBundlizer().findBundleByNode(gemm2);
    ASSERT_TRUE(mmeConsumerBundle);
    const auto& mmeSolution = mmeConsumerBundle->getSolution();

    unsigned expectedNumOfSlicesTpcBundleNodes   = 1;
    unsigned expectedNumOfSlicesParallelCastNode = 1;
    for (const auto& operand : mmeSolution.operands)
    {
        // Find number of slices of the connecting tensor between the TPC bundle and the MME consumer (gelu2Out2).
        if (operand->originalTensor == gelu2Out2)
        {
            expectedNumOfSlicesTpcBundleNodes = SlicedOperandUtils::nofSlices(operand);
        }

        // Find number of slices of the parallel node (castOut)
        if (operand->originalTensor == castOut)
        {
            expectedNumOfSlicesParallelCastNode = SlicedOperandUtils::nofSlices(operand);
        }
    }

    std::map<pTensor, unsigned> tensorToExpectedNumSlices = {{gemm1Out, expectedNumOfSlicesTpcBundleNodes},
                                                             {addIn, expectedNumOfSlicesTpcBundleNodes},
                                                             {addOut, expectedNumOfSlicesTpcBundleNodes},
                                                             {reluOut, expectedNumOfSlicesTpcBundleNodes},
                                                             {gelu2Out1, expectedNumOfSlicesTpcBundleNodes},
                                                             {gelu2Out2, expectedNumOfSlicesTpcBundleNodes},
                                                             {castIn, expectedNumOfSlicesParallelCastNode},
                                                             {castOut, expectedNumOfSlicesParallelCastNode}};

    const auto& tpcSolution = tpcBundle->getSolution();
    ASSERT_EQ(tpcSolution.operands.size(), tensorToExpectedNumSlices.size());
    for (const auto& operand : tpcSolution.operands)
    {
        ASSERT_TRUE(tensorToExpectedNumSlices.count(operand->originalTensor) > 0);
        ASSERT_EQ(SlicedOperandUtils::nofSlices(operand), tensorToExpectedNumSlices.at(operand->originalTensor));
    }
}

TEST_F(TPCBundleTest, avoid_add_operandB_producer_to_two_tpc_bundles)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_BUNDLES, "true");
    GaudiGraph graph;
    SizeArray  dataDims     = {64, 32, 1, 1, 1};
    SizeArray  reshapeDims  = {64, 1, 1, 32, 1};
    SizeArray  gemmSecondIn = {64, 64};
    // Working with 1x4 geometry - no slice should be encountered for gemm nodes

    TensorPtr operandB_shared_bf16 = pTensor(new Tensor(2, gemmSecondIn.data(), syn_type_bf16));
    TensorPtr operandB_shared_f32  = pTensor(new Tensor(2, gemmSecondIn.data(), syn_type_float));
    NodePtr   cast =
        NodeFactory::createNode({operandB_shared_bf16}, {operandB_shared_f32}, nullptr, 0, "cast_bf16_to_f32", "cast");
    GraphEditor::addNode(graph, cast);

    TensorPtr firstOut_1    = pTensor(new Tensor(2, dataDims.data(), syn_type_float));
    TensorPtr operandB_1    = pTensor(new Tensor(2, gemmSecondIn.data(), syn_type_float));
    TensorPtr secondOut_1   = firstOut_1->clone();
    TensorPtr thirdOut_1    = firstOut_1->clone();
    TensorPtr reshapeOut_1  = pTensor(new Tensor(4, reshapeDims.data(), syn_type_float));
    TensorPtr fourthOut_1   = reshapeOut_1->clone();
    TensorPtr reshape2Out_1 = firstOut_1->clone();
    TensorPtr finalOut_1    = reshape2Out_1->clone();
    finalOut_1->setMemoryDescriptor(synMemoryDescriptor(true));

    NodePtr gemm1_1    = NodeFactory::createNode({firstOut_1->clone(), operandB_1},
                                              {firstOut_1},
                                              nullptr,
                                              0,
                                              NodeFactory::gemmNodeTypeName,
                                              "first_gemm_1");
    NodePtr add1_1     = NodeFactory::createNode({firstOut_1, firstOut_1->clone()},
                                             {secondOut_1},
                                             nullptr,
                                             0,
                                             "add_fwd_f32",
                                             "first_add_1");
    NodePtr add2_1     = NodeFactory::createNode({secondOut_1, firstOut_1->clone()},
                                             {thirdOut_1},
                                             nullptr,
                                             0,
                                             "add_fwd_f32",
                                             "second_add_1");
    NodePtr reshape_1  = NodeFactory::createNode({thirdOut_1},
                                                {reshapeOut_1},
                                                nullptr,
                                                0,
                                                NodeFactory::reshapeNodeTypeName,
                                                "reshape_1");
    NodePtr add3_1     = NodeFactory::createNode({reshapeOut_1, reshapeOut_1->clone()},
                                             {fourthOut_1},
                                             nullptr,
                                             0,
                                             "add_fwd_f32",
                                             "third_add_1");
    NodePtr reshape2_1 = NodeFactory::createNode({fourthOut_1},
                                                 {reshape2Out_1},
                                                 nullptr,
                                                 0,
                                                 NodeFactory::reshapeNodeTypeName,
                                                 "second_reshape_1");
    NodePtr gemm2_1    = NodeFactory::createNode({reshape2Out_1, operandB_shared_f32},
                                              {finalOut_1},
                                              nullptr,
                                              0,
                                              NodeFactory::gemmNodeTypeName,
                                              "second_gemm_1");

    GraphEditor::addNode(graph, gemm1_1);
    GraphEditor::addNode(graph, add1_1);
    GraphEditor::addNode(graph, add2_1);
    GraphEditor::addNode(graph, reshape_1);
    GraphEditor::addNode(graph, add3_1);
    GraphEditor::addNode(graph, reshape2_1);
    GraphEditor::addNode(graph, gemm2_1);

    TensorPtr firstOut_2    = pTensor(new Tensor(2, dataDims.data(), syn_type_float));
    TensorPtr operandB_2    = pTensor(new Tensor(2, gemmSecondIn.data(), syn_type_float));
    TensorPtr secondOut_2   = firstOut_2->clone();
    TensorPtr thirdOut_2    = firstOut_2->clone();
    TensorPtr reshapeOut_2  = pTensor(new Tensor(4, reshapeDims.data(), syn_type_float));
    TensorPtr fourthOut_2   = reshapeOut_2->clone();
    TensorPtr reshape2Out_2 = firstOut_2->clone();
    TensorPtr finalOut_2    = reshape2Out_2->clone();
    finalOut_2->setMemoryDescriptor(synMemoryDescriptor(true));

    NodePtr gemm1_2    = NodeFactory::createNode({firstOut_2->clone(), operandB_2},
                                              {firstOut_2},
                                              nullptr,
                                              0,
                                              NodeFactory::gemmNodeTypeName,
                                              "first_gemm_2");
    NodePtr add1_2     = NodeFactory::createNode({firstOut_2, firstOut_2->clone()},
                                             {secondOut_2},
                                             nullptr,
                                             0,
                                             "add_fwd_f32",
                                             "first_add_2");
    NodePtr add2_2     = NodeFactory::createNode({secondOut_2, firstOut_2->clone()},
                                             {thirdOut_2},
                                             nullptr,
                                             0,
                                             "add_fwd_f32",
                                             "second_add_2");
    NodePtr reshape_2  = NodeFactory::createNode({thirdOut_2},
                                                {reshapeOut_2},
                                                nullptr,
                                                0,
                                                NodeFactory::reshapeNodeTypeName,
                                                "reshape_2");
    NodePtr add3_2     = NodeFactory::createNode({reshapeOut_2, reshapeOut_2->clone()},
                                             {fourthOut_2},
                                             nullptr,
                                             0,
                                             "add_fwd_f32",
                                             "third_add_2");
    NodePtr reshape2_2 = NodeFactory::createNode({fourthOut_2},
                                                 {reshape2Out_2},
                                                 nullptr,
                                                 0,
                                                 NodeFactory::reshapeNodeTypeName,
                                                 "second_reshape_2");
    NodePtr gemm2_2    = NodeFactory::createNode({reshape2Out_2, operandB_shared_f32},
                                              {finalOut_2},
                                              nullptr,
                                              0,
                                              NodeFactory::gemmNodeTypeName,
                                              "second_gemm_2");

    GraphEditor::addNode(graph, gemm1_2);
    GraphEditor::addNode(graph, add1_2);
    GraphEditor::addNode(graph, add2_2);
    GraphEditor::addNode(graph, reshape_2);
    GraphEditor::addNode(graph, add3_2);
    GraphEditor::addNode(graph, reshape2_2);
    GraphEditor::addNode(graph, gemm2_2);

    loadTpcKernels(graph);

    graphVisualizationPre(graph);
    SRAMSlicingManager manager(graph);
    manager.sliceGraph();
    graphVisualizationPost(graph);

    pBundle tpcBundle_1 = manager.getBundlizer().findBundleByNode(add1_1);
    ASSERT_TRUE(tpcBundle_1 != nullptr);
    pBundle tpcBundle_2 = manager.getBundlizer().findBundleByNode(add1_2);
    ASSERT_TRUE(tpcBundle_2 != nullptr);
    pBundle castBundle = manager.getBundlizer().findBundleByNode(cast);
    ASSERT_TRUE(castBundle != nullptr);
    ASSERT_TRUE(castBundle == tpcBundle_1 || castBundle == tpcBundle_2);
}