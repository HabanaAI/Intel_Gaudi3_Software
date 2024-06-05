#include "compilation_hal_reader.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "types.h"
#include <vector>

class SchedulerTest : public GraphOptimizerTest
{
};

TEST_F(SchedulerTest, optimize_scheduling_for_logical_nodes_with_same_real_output_tensor)
{
    // Given the following graph:
    //                                                  ┌──────────┐             ┌──────────┐
    //                                                  │          │             │          │
    //                                            ┌─────►  SLICE1  ├─────────────►  BGEMM1  │
    //                                            │     │          │             │          │
    //                                            │     └──────────┘             └──────────┘
    // ┌───────────┐           ┌───────────┐      │
    // │           │           │           │──────│
    // │   CAST    ├───────────►  RESHAPE  |
    // │           │           │           ├──────┤
    // └───────────┘           └───────────┘      │
    //                                            │      ┌─────────┐             ┌──────────┐
    //                                            │      │         │             │          │
    //                                            └──────► SLICE2  ├─────────────►  BGEMM2  │
    //                                                   │         │             │          │
    //                                                   └─────────┘             └──────────┘
    // The slice is on the batch dim:
    // SLICE1 contains batches 64-128
    // SLICE2 contains batches 0-64
    // Both slices outputs are aliases of the same real tensor.
    // Expect SLICE2 to be scheudled before SLICE1 since its offset in the real tensor is lower.
    // This will enable better pipelining between the CAST node and the BGEMM nodes
    // (BGEMM2 which consumes the lower batches will be scheduled first).
    // The order of node creation is important here since the scheduler fallback tie breaker
    // is the node ID (SLICE1 is created before SLICE2 but expected to be scheduled after).

    std::vector<TSize> castShape     = {250012, 3328};
    std::vector<TSize> sliceInShape  = {250012, 26, 128};
    std::vector<TSize> sliceOutShape = {250012, 26, 64};
    std::vector<TSize> bgemmIn1Shape = {1024, 250012};
    std::vector<TSize> bgemmOutShape = {1024, 26, 64};

    Gaudi2Graph graph;
    CompilationHalReaderSetter halSetter(&graph);  // For extract slice multi-node

    TensorPtr castIn  = std::make_shared<Tensor>(castShape.size(), castShape.data(), syn_type_float);
    TensorPtr castOut = std::make_shared<Tensor>(castShape.size(), castShape.data(), syn_type_bf16);
    NodePtr   cast    = NodeFactory::createNode({castIn}, {castOut}, nullptr, "cast_f32_to_bf16", "CAST");
    ASSERT_TRUE(GraphEditor::addNode(graph, cast));

    TensorPtr reshapeOut = std::make_shared<Tensor>(sliceInShape.size(), sliceInShape.data(), syn_type_bf16);
    NodePtr   reshape =
        NodeFactory::createNode({castOut}, {reshapeOut}, nullptr, NodeFactory::reshapeNodeTypeName, "RESHAPE");
    ASSERT_TRUE(GraphEditor::addNode(graph, reshape));

    synSliceParams slice1Params = {.axes   = {2, 0, 0, 0, 0},
                                   .starts = {64, 0, 0, 0, 0},
                                   .ends   = {128, 0, 0, 0, 0},
                                   .steps  = {1, 1, 1, 1, 1}};
    TensorPtr      slice1Out    = std::make_shared<Tensor>(sliceOutShape.size(), sliceOutShape.data(), syn_type_bf16);
    NodePtr        slice1 =
        NodeFactory::createNode({reshapeOut}, {slice1Out}, &slice1Params, NodeFactory::sliceNodeTypeName, "SLICE1");
    ASSERT_TRUE(GraphEditor::addNode(graph, slice1));

    synGEMMParams bgemm1Params {};
    TensorPtr     bgemm1In1 = std::make_shared<Tensor>(bgemmIn1Shape.size(), bgemmIn1Shape.data(), syn_type_bf16);
    TensorPtr     bgemm1Out = std::make_shared<Tensor>(bgemmOutShape.size(), bgemmOutShape.data(), syn_type_bf16);
    NodePtr       bgemm1    = NodeFactory::createNode({slice1Out, bgemm1In1},
                                             {bgemm1Out},
                                             &bgemm1Params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "BGEMM1");
    ASSERT_TRUE(GraphEditor::addNode(graph, bgemm1));

    synSliceParams slice2Params = {.axes   = {2, 0, 0, 0, 0},
                                   .starts = {0, 0, 0, 0, 0},
                                   .ends   = {64, 0, 0, 0, 0},
                                   .steps  = {1, 1, 1, 1, 1}};
    TensorPtr      slice2Out    = std::make_shared<Tensor>(sliceOutShape.size(), sliceOutShape.data(), syn_type_bf16);
    NodePtr        slice2 =
        NodeFactory::createNode({reshapeOut}, {slice2Out}, &slice2Params, NodeFactory::sliceNodeTypeName, "SLICE2");
    ASSERT_TRUE(GraphEditor::addNode(graph, slice2));

    synGEMMParams bgemm2Params {};
    TensorPtr     bgemm2In1 = std::make_shared<Tensor>(bgemmIn1Shape.size(), bgemmIn1Shape.data(), syn_type_bf16);
    TensorPtr     bgemm2Out = std::make_shared<Tensor>(bgemmOutShape.size(), bgemmOutShape.data(), syn_type_bf16);
    NodePtr       bgemm2    = NodeFactory::createNode({slice2Out, bgemm2In1},
                                             {bgemm2Out},
                                             &bgemm2Params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "BGEMM2");
    ASSERT_TRUE(GraphEditor::addNode(graph, bgemm2));

    ASSERT_TRUE(extractMultiNodes(graph));  // To extract slice nodes
    ASSERT_TRUE(extractDataMovementMultiNodes(graph));  // To extract slice nodes
    ASSERT_TRUE(handleLogicalOps(graph));

    // Verify the execution schedule is as expected: SLICE2 -> SLICE1
    std::vector<TensorPtr> expectedExecOrder = {slice2->getOutput(0), slice1->getOutput(0)};
    std::vector<TensorPtr> actualExecOrder;
    for (const auto& n : graph.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_SLICE)
        {
            actualExecOrder.push_back(n->getOutput(0));
        }
    }

    ASSERT_TRUE(expectedExecOrder == actualExecOrder);
}

TEST_F(SchedulerTest, optimize_scheduling_for_max_path)
{
    static constexpr unsigned DIM           = 1;
    TSize                     sizes[DIM]    = {10};
    TSize                     outSizes[DIM] = {sizes[0] * 2};

    Gaudi2Graph graph;
    TensorPtr   in1       = std::make_shared<Tensor>(DIM, sizes, syn_type_float);
    TensorPtr   in2       = std::make_shared<Tensor>(DIM, sizes, syn_type_float);
    TensorPtr   cp1Out    = std::make_shared<Tensor>(DIM, sizes, syn_type_float);
    TensorPtr   cp2Out    = std::make_shared<Tensor>(DIM, sizes, syn_type_float);
    TensorPtr   concatOut = std::make_shared<Tensor>(DIM, outSizes, syn_type_float);
    TensorPtr   out       = std::make_shared<Tensor>(DIM, outSizes, syn_type_float);

    // create cp2 before cp1 so it will get a lower ID
    NodePtr cp2 = NodeFactory::createNode({in2}, {cp2Out}, nullptr, NodeFactory::memcpyNodeTypeName, "cp2");
    ASSERT_TRUE(GraphEditor::addNode(graph, cp2));
    NodePtr cp1 = NodeFactory::createNode({in1}, {cp1Out}, nullptr, NodeFactory::memcpyNodeTypeName, "cp1");
    ASSERT_TRUE(GraphEditor::addNode(graph, cp1));
    NodePtr cat = NodeFactory::createNode({cp1Out, cp2Out},
                                          {concatOut},
                                          nullptr,
                                          NodeFactory::concatenateNodeLogicalInternalTypeName,
                                          "cat");
    ASSERT_TRUE(GraphEditor::addNode(graph, cat));
    NodePtr relu = NodeFactory::createNode({concatOut}, {out}, nullptr, "relu_fwd_f32", "relu");
    ASSERT_TRUE(GraphEditor::addNode(graph, relu));

    // Verify the execution schedule is as expected.
    handleLogicalOps(graph);
    NodeVector expectedExecOrder = {cp1, cp2, cat, relu};
    ASSERT_EQ(graph.getExeSortedNodes(), expectedExecOrder);
}