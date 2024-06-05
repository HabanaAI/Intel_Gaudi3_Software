#include "compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"

// fusing a wait node to other nodes should succeed
TEST_F(GraphOptimizerTest, fuse_wait_to_nodes)
{
    GaudiGraph graph;
    pTensor    gemmInputA  = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmInputB  = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmOutput1 = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmOutput2 = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmOutput3 = std::make_shared<Tensor>(syn_type_bf16);

    struct synWaitParams waitParams(23);
    pNode waitNode      = NodeFactory::createNode({}, {}, &waitParams, NodeFactory::waitNodeTypeName, "wait");

    pNode gemmNode1      = NodeFactory::createNode({gemmInputA, gemmInputB}, {gemmOutput1},
                                                   nullptr, NodeFactory::gemmNodeTypeName, "gemm_node");

    pNode gemmNode2      = NodeFactory::createNode({gemmInputA, gemmInputB}, {gemmOutput2},
                                                   nullptr, NodeFactory::gemmNodeTypeName, "gemm_node");

    pNode gemmNode3      = NodeFactory::createNode({gemmInputA, gemmInputB}, {gemmOutput3},
                                                   nullptr, NodeFactory::gemmNodeTypeName, "gemm_node");

    GraphEditor::addNode(graph, gemmNode1);
    GraphEditor::addNode(graph, gemmNode2);
    GraphEditor::addNode(graph, gemmNode3);
    GraphEditor::addNode(graph, waitNode);

    NodeSet blocking;
    NodeSet blocked;

    blocking.insert(waitNode);
    blocked.insert(gemmNode1);
    blocked.insert(gemmNode2);
    blocked.insert(gemmNode3);
    graph.addControlDependency(blocking, blocked);
    blocking.clear();
    blocked.clear();

    ASSERT_EQ(graph.getNumNodes(), 4);
    ASSERT_EQ(fuseWaits(graph), true);
    ASSERT_EQ(graph.getNumNodes(), 3);

    for (pNode n : graph.getExeSortedNodes())
    {
        ASSERT_EQ(n->getNodeAnnotation().waitCycles, 23);
    }
}

// fusing more than one wait node to a node should return a compilation error
TEST_F(GraphOptimizerTest, fuse_2_waits_to_node)
{
    GaudiGraph graph;
    pTensor    gemmInputA = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmInputB = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmOutput = std::make_shared<Tensor>(syn_type_bf16);

    struct synWaitParams waitParams(23);
    pNode waitNode1      = NodeFactory::createNode({}, {}, &waitParams, NodeFactory::waitNodeTypeName, "wait");
    pNode waitNode2      = NodeFactory::createNode({}, {}, &waitParams, NodeFactory::waitNodeTypeName, "wait");

    pNode gemmNode       = NodeFactory::createNode({gemmInputA, gemmInputB}, {gemmOutput},
                                                   nullptr, NodeFactory::gemmNodeTypeName, "gemm_node");

    GraphEditor::addNode(graph, gemmNode);
    GraphEditor::addNode(graph, waitNode1);
    GraphEditor::addNode(graph, waitNode2);

    NodeSet blocking;
    NodeSet blocked;

    blocking.insert(waitNode1);
    blocking.insert(waitNode2);
    blocked.insert(gemmNode);

    graph.addControlDependency(blocking, blocked);
    blocking.clear();
    blocked.clear();

    ASSERT_EQ(fuseWaits(graph), false);
}

// fusing a wait node to a logical node should return a compilation error
TEST_F(GraphOptimizerTest, DISABLED_fuse_wait_to_logical_node)
{
    GaudiGraph graph;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    const TSize sizes[]       = {n, w, h, batch};
    const TSize sizesConcat[] = {n, 2 * w, h, batch};

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_float));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_float));
    pTensor t3 = pTensor(new Tensor(4U, sizesConcat, syn_type_float));

    struct synWaitParams waitParams(23);
    pNode waitNode    = NodeFactory::createNode({}, {}, &waitParams, NodeFactory::waitNodeTypeName, "wait");
    synConcatenateParams concatParams;
    concatParams.axis = 1;
    pNode concatNode =
        NodeFactory::createNode({t1, t2}, {t3}, &concatParams, NodeFactory::concatenateNodeTypeName, "concat4");

    GraphEditor::addNode(graph, waitNode);
    GraphEditor::addNode(graph, concatNode);

    NodeSet blocking;
    NodeSet blocked;
    blocking.insert(waitNode);
    blocked.insert(concatNode);

    graph.addControlDependency(blocking, blocked);
    blocking.clear();
    blocked.clear();

    ASSERT_EQ(fuseWaits(graph), false);
}

static void calcOutputSize(synSliceParams params, unsigned dim, TSize* outputSize)
{
    for (unsigned i = 0; i < dim; i++)
    {
        unsigned axis = params.axes[i];
        outputSize[axis] = (params.ends[i] - params.starts[i]) / params.steps[i];
    }
}

TEST_F(GraphOptimizerTest, fuse_wait_to_slice_3d)
{
    synSliceParams sliceParams = {{0, 1, 2}, {0, 0, 0}, {4, 2, 2}, {2, 2, 1}};
    unsigned totalNumberOfNodes = 12;
    unsigned numberOfDim = 3;

    TSize inputSize[]  = {4, 2, 2};
    TSize outputSize[3];
    calcOutputSize(sliceParams, numberOfDim, outputSize);

    GaudiGraph graph;
    CompilationHalReaderSetter compHalReaderSetter(&graph);
    pTensor sliceInput  = std::make_shared<Tensor>(numberOfDim, inputSize, syn_type_single);
    pTensor sliceOutput = std::make_shared<Tensor>(numberOfDim, outputSize, syn_type_single);

    pNode sliceNode = NodeFactory::createNode({sliceInput}, {sliceOutput},
                                              &sliceParams, NodeFactory::sliceNodeTypeName, "slice_node");

    struct synWaitParams waitParams(23);
    pNode waitNode = NodeFactory::createNode({}, {}, &waitParams, NodeFactory::waitNodeTypeName, "wait");

    GraphEditor::addNode(graph, sliceNode);
    GraphEditor::addNode(graph, waitNode);

    NodeSet blocking;
    NodeSet blocked;
    blocking.insert(waitNode);
    blocked.insert(sliceNode);

    graph.addControlDependency(blocking, blocked);
    blocking.clear();
    blocked.clear();

    ASSERT_EQ(graph.getNumNodes(), 2);
    ASSERT_EQ(fuseWaits(graph), true);
    ASSERT_EQ(graph.getNumNodes(), 1);

    // make sure that after extracting the wait is propagated
    ASSERT_TRUE(extractMultiNodes(graph));
    ASSERT_TRUE(extractDataMovementMultiNodes(graph));
    ASSERT_EQ(graph.getNumNodes(), totalNumberOfNodes);
    for (pNode n : graph.getExeSortedNodes())
    {
        ASSERT_EQ(n->getNodeAnnotation().waitCycles, 23);
    }
}
