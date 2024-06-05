#include <memory>
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "sim_graph.h"
#include "graph_optimizer_test.h"

struct Visitor
{
    void visit_func(NodePtr node)
    {
        ++node_count;
    }
    int node_count;
};

TEST_F(GraphTests, topological_visit)
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

    EXPECT_EQ(g.getTopoSortedNodes().size(), 3);
}
