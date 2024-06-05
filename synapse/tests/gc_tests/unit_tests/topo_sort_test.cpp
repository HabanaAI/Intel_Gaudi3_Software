#include <memory>
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "sim_graph.h"
#include "graph_optimizer_test.h"

TEST_F(GraphTests, pattern_match)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;

    /* Create the following graph:
     *              {a}-->[n1]-->{b}-->[c1]-->{c}-
     *                                            \_
     *                                             _[a1]-->{h}-->[n4]-->{i}
     *                                            /
     * {d}-->[n2]-->{e}-->[n3]-->{f}-->[c2]-->{g}-
     *
     * Match the pattern:
     * []-->
     *      \_
     *       _[]
     *      /
     * []-->
     */
    NodePtr n1, n2, n3, c1, c2, a1, n4;

    TensorPtr a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr e = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr f = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr g = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr h = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    n1  = NodeFactory::createDebugNode(a, b, "");
    c1  = NodeFactory::createDebugNode(b, c, "");
    n2  = NodeFactory::createDebugNode(d, e, "");
    n3  = NodeFactory::createDebugNode(e, f, "");
    c2  = NodeFactory::createDebugNode(f, g, "");
    a1  = NodeFactory::createDebugJoinNode(c, g, h, "");
    n4  = NodeFactory::createDebugNode(h, i, "");

    SimGraph graph;
    GraphEditor::addNode(graph, n1);
    GraphEditor::addNode(graph, c1);
    GraphEditor::addNode(graph, n2);
    GraphEditor::addNode(graph, n3);
    GraphEditor::addNode(graph, c2);

    ASSERT_FALSE(graph.isAncestor(c2, c1));
    ASSERT_FALSE(graph.isAncestor(c1, c2));
    ASSERT_TRUE(graph.isAncestor(n1, c1));

    NodePtr pc1, pc2, pa1;

    TensorPtr pa = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr pb = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr pc = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr pd = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr pe = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    pc1  = NodeFactory::createDebugNode(pa, pb, "");
    pc2  = NodeFactory::createDebugNode(pc, pd, "");
    pa1  = NodeFactory::createDebugJoinNode(pb, pd, pe, "");

    SimGraph pattern;
    GraphEditor::addNode(pattern, pc1);
    GraphEditor::addNode(pattern, pc2);
    GraphEditor::addNode(pattern, pa1);

    // no match
    NodeSet matches = graph.matchPatternWithSingleOutputNode(&pattern,
            [](NodePtr a, NodePtr b) {return a->getNodeType() == b->getNodeType();});

    ASSERT_EQ(matches.size(), 0);

    GraphEditor::addNode(graph, a1);
    GraphEditor::addNode(graph, n4);

    ASSERT_TRUE(graph.isAncestor(n2, a1));
    ASSERT_FALSE(graph.isAncestor(a1, n2));

    // match
    matches = graph.matchPatternWithSingleOutputNode(&pattern,
            [](NodePtr a, NodePtr b) {return a->getNodeType() == b->getNodeType();});

    ASSERT_EQ(matches.size(), 1);

    auto it = matches.begin();
    ASSERT_EQ(a1->getId(), (*it)->getId());

}

TEST_F(GraphTests, topological_sort)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;

    //The graph: a -> src -> b -> mid -> c -> sink -> d
    NodePtr src, mid, sink;

    TensorPtr a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    //Create the nodes in an ugly order
    mid  = NodeFactory::createDebugNode(b, c, "");
    sink = NodeFactory::createDebugNode(c, d, "");
    src  = NodeFactory::createDebugNode(a, b, "");

    SimGraph g;
    GraphEditor::addNode(g, sink);
    GraphEditor::addNode(g, mid);
    GraphEditor::addNode(g, src);

    g.compile();
    NodeVector sortedNodes = g.getTopoSortedNodes();

    ASSERT_EQ(sortedNodes.size(), 3U);
    auto it = sortedNodes.begin();
    EXPECT_EQ(src, *it++);
    EXPECT_EQ(mid, *it++);
    EXPECT_EQ(sink, *it++);
}

TEST_F(GraphTests, node_removal)
{
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;

    //The graph: a -> src -> b -> mid -> c -> sink -> d
    NodePtr src, mid, sink;

    TensorPtr a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    src  = NodeFactory::createDebugNode(a, b, "");
    mid  = NodeFactory::createDebugNode(b, c, "");
    sink = NodeFactory::createDebugNode(c, d, "");

    SimGraph g;
    GraphEditor::addNode(g, src);
    GraphEditor::addNode(g, mid);
    GraphEditor::addNode(g, sink);

    GraphEditor::removeNode(g, mid, src);

    g.compile();
    NodeVector sortedNodes = g.getTopoSortedNodes();

    ASSERT_EQ(sortedNodes.size(), 2U);
    auto it = sortedNodes.begin();
    EXPECT_EQ(src, *it++);
    EXPECT_EQ(sink, *it++);

}
