#include <memory>
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "sim_graph.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"

TEST_F(GraphTests, isomorphic_simple)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;

    //The graphs: a -> src -> b -> mid -> c -> sink -> d
    NodePtr src, mid, sink;
    NodePtr src2, mid2, sink2;

    TensorPtr a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    TensorPtr a2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    //Create the nodes in an ugly order
    mid  = NodeFactory::createDebugNode(b, c, "");
    sink = NodeFactory::createDebugNode(c, d, "");
    src  = NodeFactory::createDebugNode(a, b, "");

    src2  = NodeFactory::createDebugNode(a2, b2, "");
    mid2  = NodeFactory::createDebugNode(b2, c2, "");
    sink2 = NodeFactory::createDebugNode(c2, d2, "");

    SimGraph g, g2;
    GraphEditor::addNode(g, sink);
    GraphEditor::addNode(g, mid);
    GraphEditor::addNode(g, src);

    GraphEditor::addNode(g2, src2);
    GraphEditor::addNode(g2, sink2);
    GraphEditor::addNode(g2, mid2);

    ASSERT_EQ(g.isomorphicTo(g2), true);
    ASSERT_EQ(g2.isomorphicTo(g), true);
}

TEST_F(GraphTests, not_isomorphic_simple)
{
    const unsigned tensor_dim = 1;
    const TSize size       = 1;
    const TSize size2      = 2;

    //The graphs: a -> src -> b -> mid -> c -> sink -> d
    NodePtr src, mid, sink;
    NodePtr src2, mid2, sink2;

    TensorPtr a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    TensorPtr a2 = TensorPtr(new Tensor(tensor_dim, &size2, syn_type_single));
    TensorPtr b2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    //Create the nodes in an ugly order
    mid  = NodeFactory::createDebugNode(b, c, "");
    sink = NodeFactory::createDebugNode(c, d, "");
    src  = NodeFactory::createDebugNode(a, b, "");

    src2  = NodeFactory::createDebugNode(a2, b2, "");
    mid2  = NodeFactory::createDebugNode(b2, c2, "");
    sink2 = NodeFactory::createDebugNode(c2, d2, "");

    SimGraph g, g2;
    GraphEditor::addNode(g, sink);
    GraphEditor::addNode(g, mid);
    GraphEditor::addNode(g, src);

    GraphEditor::addNode(g2, src2);
    GraphEditor::addNode(g2, sink2);
    GraphEditor::addNode(g2, mid2);

    ASSERT_EQ(g.isomorphicTo(g2), false);
    ASSERT_EQ(g2.isomorphicTo(g), false);
}

TEST_F(GraphTests, isomorphic_complex)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;

    //The graphs: a -> src -> b, c -> sink -> d
    NodePtr src, sink;
    NodePtr src2, sink2;

    TensorPtr a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    TensorPtr a2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    sink  = NodeFactory::createDebugJoinNode(b, c, d, "");
    src   = NodeFactory::createDebugForkNode(a, b, c, "");

    src2  = NodeFactory::createDebugForkNode(a2, b2, c2, "");
    sink2 = NodeFactory::createDebugJoinNode(b2, c2, d2, "");

    SimGraph g, g2;
    GraphEditor::addNode(g, sink);
    GraphEditor::addNode(g, src);

    GraphEditor::addNode(g2, src2);
    GraphEditor::addNode(g2, sink2);

    ASSERT_EQ(g.isomorphicTo(g2), true);
    ASSERT_EQ(g2.isomorphicTo(g), true);
}

TEST_F(GraphTests, not_isomorphic_same_structure)
{
    const unsigned tensor_dim = 1;
    const TSize    size = 1;

    //The graphs: a -> src -> b -> sink -> c
    NodePtr src, sink;
    NodePtr src2, sink2;

    TensorPtr a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    TensorPtr a2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    //Create the nodes in an ugly order
    sink = NodeFactory::createDebugNode(b, c, "");
    src = NodeFactory::createDebugNode(a, b, "");

    src2 = NodeFactory::createDebugNode(a2, b2, "");
    sink2 = NodeFactory::createDebug2Node(b2, c2, "");

    SimGraph g, g2;
    GraphEditor::addNode(g, sink);
    GraphEditor::addNode(g, src);

    GraphEditor::addNode(g2, src2);
    GraphEditor::addNode(g2, sink2);

    ASSERT_EQ(g.isomorphicTo(g2), false);
    ASSERT_EQ(g2.isomorphicTo(g), false);
}

TEST_F(GraphTests, not_isomorphic_different_convolution_params)
{
    const unsigned kW = 5;
    const unsigned kH = 5;
    const unsigned dW = 1;
    const unsigned dH = 1;
    const unsigned nOFM = 1;
    const unsigned wOFM = 5;
    const unsigned hOFM = 5;
    const unsigned nIFM = 1;
    //o = ((i - k + 2 * pad) / stride) + 1
    const unsigned wIFM = ((wOFM - 1) * dW) + kW;
    const unsigned hIFM = ((hOFM - 1) * dH) + kH;

    synConvolutionParams params;
    params.dH = dH;
    params.dW = dW;
    params.kH = kH;
    params.kW = kW;
    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    const TSize i_sizes[] = { nIFM, wIFM, hIFM, 1 };
    const TSize o_sizes[] = { nOFM, wOFM, hOFM, 1 };
    const TSize w_sizes[] = { nOFM, nIFM, kW, kH };

    TensorPtr IFM = TensorPtr(new Tensor(4U, i_sizes, syn_type_single));
    TensorPtr OFM = TensorPtr(new Tensor(4U, o_sizes, syn_type_single));
    TensorPtr W   = TensorPtr(new Tensor(4U, w_sizes, syn_type_single));

    SimGraph g, g2;
    NodePtr  n = getConvNodeWithGoyaLayouts(IFM, W, nullptr, OFM, params, "");
    GraphEditor::addNode(g, n);

    params.kW = params.kH = 3;
    NodePtr n2 = getConvNodeWithGoyaLayouts(IFM, W, nullptr, OFM, params, "");
    GraphEditor::addNode(g2, n2);

    ASSERT_EQ(g.isomorphicTo(g2), false);
    ASSERT_EQ(g2.isomorphicTo(g), false);
}
