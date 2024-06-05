#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "sim_graph.h"
#include "test_utils.h"
#include "infra/global_conf_manager.h"
#include "gc_tests/unit_tests/graph_optimizer_test.h"

class GraphFixture : public GraphOptimizerTest
{
protected:

    // The fixture class contains pre-made tensors and pre-made graph for general
    // use by the test bodies, with the purpose to shorten the code. The pre-made
    // graph looks like this, GRAPH FLOWS FROM LEFT TO RIGHT:
    /*
                         /---b------[n6]---------------
                        /                              \
               /--a---[n2]---b---[n5]                   e
              /                    \                     \
             /                      d                     \
            /                        \                     \
      -i--[n1]----a---[n3]--z-      [n7]---f---[n8]---g---[n9]---h---[n10]---o-
            \                        /
             \                      c
              \                    /
               \--a---[n4]---------

    */
    virtual void SetUp() override
    {
        GraphOptimizerTest::SetUp();

        // Create tensors for general use by the test bodies
        ii = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        oo = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        aa = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        bb = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        cc = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        dd = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        ee = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        ff = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        gg = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        hh = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        zz = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        // Create tensors for the pre-made graph
        i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        e = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        f = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        g = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        h = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        z = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        // Create pre-made graph containing 10 nodes for general use by test bodies
        n1  = NodeFactory::createDebugNode(i, a, "");
        n2  = NodeFactory::createDebugNode(a, b, "");
        n3  = NodeFactory::createDebugNode(a, z, "");
        n4  = NodeFactory::createDebugNode(a, c, "");
        n5  = NodeFactory::createDebugNode(b, d, "");
        n6  = NodeFactory::createDebugNode(b, e, "");
        n7  = NodeFactory::createDebugJoinNode(c, d, f, "");
        n8  = NodeFactory::createDebugNode(f, g, "");
        n9  = NodeFactory::createDebugJoinNode(g, e, h, "");
        n10 = NodeFactory::createDebugNode(h, o, "");
        GraphEditor::addNode(preMadeGraph, n1);
        GraphEditor::addNode(preMadeGraph, n2);
        GraphEditor::addNode(preMadeGraph, n3);
        GraphEditor::addNode(preMadeGraph, n4);
        GraphEditor::addNode(preMadeGraph, n5);
        GraphEditor::addNode(preMadeGraph, n6);
        GraphEditor::addNode(preMadeGraph, n7);
        GraphEditor::addNode(preMadeGraph, n8);
        GraphEditor::addNode(preMadeGraph, n9);
        GraphEditor::addNode(preMadeGraph, n10);
    }

    virtual void TearDown() override
    {
        i.reset();
        o.reset();
        a.reset();
        b.reset();
        c.reset();
        d.reset();
        e.reset();
        f.reset();
        g.reset();
        h.reset();
        z.reset();
        ii.reset();
        oo.reset();
        aa.reset();
        bb.reset();
        cc.reset();
        dd.reset();
        ee.reset();
        ff.reset();
        gg.reset();
        hh.reset();
        zz.reset();
        n1.reset();
        n2.reset();
        n3.reset();
        n4.reset();
        n5.reset();
        n6.reset();
        n7.reset();
        n8.reset();
        n9.reset();
        n10.reset();
        preMadeGraph.clear();

        GraphOptimizerTest::TearDown();
    }

    const unsigned tensor_dim = 1;
    const TSize    size       = 1;

    // Tensors for general use by the tests
    TensorPtr  i;
    TensorPtr  o;
    TensorPtr  a;
    TensorPtr  b;
    TensorPtr  c;
    TensorPtr  d;
    TensorPtr  e;
    TensorPtr  f;
    TensorPtr  g;
    TensorPtr  h;
    TensorPtr  z;
    TensorPtr  ii;
    TensorPtr  oo;
    TensorPtr  aa;
    TensorPtr  bb;
    TensorPtr  cc;
    TensorPtr  dd;
    TensorPtr  ee;
    TensorPtr  ff;
    TensorPtr  gg;
    TensorPtr  hh;
    TensorPtr  zz;
    NodePtr    n1;
    NodePtr    n2;
    NodePtr    n3;
    NodePtr    n4;
    NodePtr    n5;
    NodePtr    n6;
    NodePtr    n7;
    NodePtr    n8;
    NodePtr    n9;
    NodePtr    n10;

    SimGraph preMadeGraph; // pre-made graph of 10 nodes for general use by test bodies
};

TEST_F(GraphFixture, add_node)
{
    SimGraph graph;
    NodePtr  n1 = NodeFactory::createDebugNode(aa, bb, "");
    GraphEditor::addNode(graph, n1);
    EXPECT_FALSE(graph.isEmpty());
    EXPECT_EQ(graph.getNumNodes(), 1);
}

TEST_F(GraphFixture, remove_node_1)
{
    SimGraph graph;
    NodePtr  n1 = NodeFactory::createDebugNode(aa, bb, "");
    GraphEditor::addNode(graph, n1);
    GraphEditor::removeNode(graph, n1);
    EXPECT_TRUE(graph.isEmpty());
    EXPECT_EQ(graph.getNumNodes(), 0);
}

TEST_F(GraphFixture, remove_node_2)
{
    SimGraph graph;
    NodePtr  n1  = NodeFactory::createDebugNode(aa, bb, "");
    NodePtr  n2  = NodeFactory::createDebugNode(bb, cc, "");
    GraphEditor::addNode(graph, n1);
    GraphEditor::addNode(graph, n2);
    GraphEditor::removeNode(graph, n1);
    EXPECT_FALSE(graph.isEmpty());
    EXPECT_EQ(graph.getNumNodes(), 1);
    EXPECT_TRUE(*(graph.getNodes().begin()) == n2);
}

TEST_F(GraphFixture, add_remove_add)
{
    SimGraph graph;
    NodePtr  n1  = NodeFactory::createDebugNode(aa, bb, "");
    NodePtr  n2  = NodeFactory::createDebugNode(bb, cc, "");
    GraphEditor::addNode(graph, n1);
    GraphEditor::addNode(graph, n2);
    GraphEditor::removeNode(graph, n2);
    GraphEditor::addNode(graph, n2);
    EXPECT_EQ(graph.getNumNodes(), 2);
}

TEST_F(GraphFixture, remove_node_replace_producer)
{
    SimGraph graph;
    //The graph: a -> src -> b -> mid -> c -> sink -> d
    NodePtr src  = NodeFactory::createDebugNode(aa, bb, "");
    NodePtr mid  = NodeFactory::createDebugNode(bb, cc, "");
    NodePtr sink = NodeFactory::createDebugNode(cc, dd, "");
    GraphEditor::addNode(graph, src);
    GraphEditor::addNode(graph, mid);
    GraphEditor::addNode(graph, sink);
    GraphEditor::removeNode(graph, mid, src);  // remove mid and replace its output with src's output
    EXPECT_EQ(graph.getNumNodes(), 2);
    TensorPtr t = sink->getInputs()[0];
    EXPECT_TRUE(t == bb);
    EXPECT_TRUE(graph.validateConnections());
}

TEST_F(GraphFixture, remove_node_replace_producer_for_two_consumers)
{
    SimGraph graph;
    NodePtr  n1  = NodeFactory::createDebugNode(ii, aa, "");
    NodePtr  n2  = NodeFactory::createDebugNode(aa, bb, "");
    NodePtr  n3  = NodeFactory::createDebugNode(aa, zz, "");
    NodePtr  n4  = NodeFactory::createDebugNode(aa, cc, "");
    NodePtr  n5  = NodeFactory::createDebugNode(bb, dd, "");
    NodePtr  n6  = NodeFactory::createDebugNode(bb, ee, "");
    GraphEditor::addNode(graph, n1);
    GraphEditor::addNode(graph, n2);
    GraphEditor::addNode(graph, n3);
    GraphEditor::addNode(graph, n4);
    GraphEditor::addNode(graph, n5);
    GraphEditor::addNode(graph, n6);
    EXPECT_EQ(graph.getNumNodes(), 6);

    // n2 outputs tensor b to both n5 and and n6. Remove n2 and put n1 as its replacement
    // producer and make sure n5 and n6 input is tensor a, which is n1 output.
    GraphEditor::removeNode(graph, n2, n1);
    EXPECT_EQ(graph.getNumNodes(), 5);
    TensorPtr t = n5->getInputs()[0];
    EXPECT_TRUE(t == aa);
    t = n6->getInputs()[0];
    EXPECT_TRUE(t == aa);
    EXPECT_TRUE(graph.validateConnections());
}

TEST_F(GraphFixture, add_10_nodes)
{
    EXPECT_EQ(preMadeGraph.getNumNodes(), 10);
    EXPECT_TRUE(preMadeGraph.validateConnections());
}

TEST_F(GraphFixture, acyclic_graph)
{
    SimGraph graph;
    //The graph: a -> first -> b -> second -> c
    NodePtr first  = NodeFactory::createDebugNode(aa, bb, "");
    NodePtr second = NodeFactory::createDebugNode(bb, cc, "");
    GraphEditor::addNode(graph, first);
    GraphEditor::addNode(graph, second);
    EXPECT_TRUE(graph.isAcyclicGraph());
}

TEST_F(GraphFixture, cyclic_graph)
{
    SimGraph graph;
    //The graph: a -> first -> b -> second -> a
    NodePtr first  = NodeFactory::createDebugNode(aa, bb, "");
    NodePtr second = NodeFactory::createDebugNode(bb, aa, "");
    GraphEditor::addNode(graph, first);
    GraphEditor::addNode(graph, second);
    EXPECT_FALSE(graph.isAcyclicGraph());
}

TEST_F(GraphFixture, is_ancestor)
{
    EXPECT_TRUE(preMadeGraph.isAncestor(n1, n10));
    EXPECT_FALSE(preMadeGraph.isAncestor(*n10, *n1)); // test also the signature that takes references
    EXPECT_FALSE(preMadeGraph.isAncestor(n4, n5));
}

TEST_F(GraphFixture, get_root_nodes_1)
{
    NodeList roots = preMadeGraph.getRootNodes();
    ASSERT_EQ(roots.size(), 1);
    EXPECT_TRUE(roots.front() == n1);
}

TEST_F(GraphFixture, get_root_nodes_2)
{
    SimGraph graph;
    NodePtr  n1 = NodeFactory::createDebugNode(ii, aa, "");
    NodePtr  n2 = NodeFactory::createDebugNode(aa, bb, "");
    NodePtr  n3 = NodeFactory::createDebugNode(aa, zz, "");
    NodePtr  n4 = NodeFactory::createDebugNode(hh, cc, "");
    NodePtr  n5 = NodeFactory::createDebugNode(bb, dd, "");
    NodePtr  n6 = NodeFactory::createDebugNode(bb, ee, "");
    NodePtr  n7 = NodeFactory::createDebugJoinNode(cc, dd, ff, "");
    GraphEditor::addNode(graph, n1);
    GraphEditor::addNode(graph, n2);
    GraphEditor::addNode(graph, n3);
    GraphEditor::addNode(graph, n4);
    GraphEditor::addNode(graph, n5);
    GraphEditor::addNode(graph, n6);
    GraphEditor::addNode(graph, n7);
    NodeList roots = graph.getRootNodes();
    EXPECT_EQ(roots.size(), 2);
    EXPECT_TRUE(std::find(roots.begin(), roots.end(), n1) != roots.end()); // n1 is a root
    EXPECT_TRUE(std::find(roots.begin(), roots.end(), n4) != roots.end()); // n4 is a root
}

TEST_F(GraphFixture, get_nodes)
{
    const NodeSet& allNodes = preMadeGraph.getNodes();
    EXPECT_EQ(preMadeGraph.getNumNodes(), 10);
    EXPECT_EQ(allNodes.size(), 10);
    EXPECT_TRUE(allNodes.find(n1)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n2)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n3)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n4)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n5)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n6)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n7)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n8)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n9)  != allNodes.end());
    EXPECT_TRUE(allNodes.find(n10) != allNodes.end());
}

TEST_F(GraphFixture, get_tensors)
{
    // Get all tensors in graph
    TensorSet allTensors = preMadeGraph.getTensors();
    EXPECT_EQ(allTensors.size(), 11);
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), i) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), o) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), a) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), b) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), c) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), d) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), e) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), f) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), g) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), h) != allTensors.end());
    EXPECT_TRUE(std::find(allTensors.begin(), allTensors.end(), z) != allTensors.end());

    // Get graph input tensors only
    std::list<TensorPtr> inTensors = preMadeGraph.getGraphInputs();
    EXPECT_EQ(inTensors.size(), 1);
    EXPECT_TRUE(std::find(inTensors.begin(), inTensors.end(), i) != inTensors.end());

    // Get graph output tensors only
    std::list<TensorPtr> outTensors = preMadeGraph.getGraphOutputs();
    EXPECT_EQ(outTensors.size(), 2);
    EXPECT_TRUE(std::find(outTensors.begin(), outTensors.end(), o) != outTensors.end());
    EXPECT_TRUE(std::find(outTensors.begin(), outTensors.end(), z) != outTensors.end());

    // Get graph intermediate tensors only
    std::list<TensorPtr> midTensors = preMadeGraph.getGraphIntermediates();
    EXPECT_EQ(midTensors.size(), 8);
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), a) != midTensors.end());
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), b) != midTensors.end());
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), c) != midTensors.end());
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), d) != midTensors.end());
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), e) != midTensors.end());
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), f) != midTensors.end());
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), g) != midTensors.end());
    EXPECT_TRUE(std::find(midTensors.begin(), midTensors.end(), h) != midTensors.end());
}

TEST_F(GraphFixture, get_tensor_producer_consumers)
{
    NodeList consumers = preMadeGraph.getTensorConsumers(a);
    EXPECT_EQ(consumers.size(), 3);
    EXPECT_TRUE(std::find(consumers.begin(), consumers.end(), n2) != consumers.end());
    EXPECT_TRUE(std::find(consumers.begin(), consumers.end(), n3) != consumers.end());
    EXPECT_TRUE(std::find(consumers.begin(), consumers.end(), n4) != consumers.end());
    NodePtr producer = preMadeGraph.getTensorProducer(f);
    EXPECT_TRUE(producer == n7);
}

TEST_F(GraphFixture, attach_nodes)
{
    // Connect n7 to n9 and disconnect n8 from n9
    preMadeGraph.attachNodes(n7, n9, 0, 0);
    EXPECT_TRUE(preMadeGraph.validateConnections());
    NodeList consumers = preMadeGraph.getTensorConsumers(f);
    EXPECT_EQ(consumers.size(), 2);
    EXPECT_TRUE(std::find(consumers.begin(), consumers.end(), n8) != consumers.end());
    EXPECT_TRUE(std::find(consumers.begin(), consumers.end(), n9) != consumers.end());
    consumers.clear();
    consumers = preMadeGraph.getTensorConsumers(g);
    EXPECT_EQ(consumers.size(), 0);
    // Tensor g becomes now a graph output since no one consumes it
    std::list<TensorPtr> outTensors = preMadeGraph.getGraphOutputs();
    EXPECT_EQ(outTensors.size(), 3);
    EXPECT_TRUE(std::find(outTensors.begin(), outTensors.end(), o) != outTensors.end());
    EXPECT_TRUE(std::find(outTensors.begin(), outTensors.end(), z) != outTensors.end());
    EXPECT_TRUE(std::find(outTensors.begin(), outTensors.end(), g) != outTensors.end());
}

TEST_F(GraphFixture, topological_sort)
{
#define SET_NODE_INDEX(node, node_index)    \
    for (unsigned i=0; i<10; i++)           \
    {                                       \
        if (topo[i] == (node))              \
        {                                   \
            (node_index) = i;               \
            break;                          \
        }                                   \
    }

    unsigned n1_index  = 0;
    unsigned n2_index  = 0;
    unsigned n3_index  = 0;
    unsigned n4_index  = 0;
    unsigned n5_index  = 0;
    unsigned n6_index  = 0;
    unsigned n7_index  = 0;
    unsigned n8_index  = 0;
    unsigned n9_index  = 0;
    unsigned n10_index = 0;

    NodeVector topo = preMadeGraph.getTopoSortedNodes();
    EXPECT_EQ(topo.size(), 10);

    SET_NODE_INDEX(n1, n1_index);
    SET_NODE_INDEX(n2, n2_index);
    SET_NODE_INDEX(n3, n3_index);
    SET_NODE_INDEX(n4, n4_index);
    SET_NODE_INDEX(n5, n5_index);
    SET_NODE_INDEX(n6, n6_index);
    SET_NODE_INDEX(n7, n7_index);
    SET_NODE_INDEX(n8, n8_index);
    SET_NODE_INDEX(n9, n9_index);
    SET_NODE_INDEX(n10, n10_index);

    EXPECT_TRUE(n10_index > n9_index);
    EXPECT_TRUE(n9_index  > n8_index);
    EXPECT_TRUE(n9_index  > n6_index);
    EXPECT_TRUE(n8_index  > n7_index);
    EXPECT_TRUE(n7_index  > n4_index);
    EXPECT_TRUE(n7_index  > n5_index);
    EXPECT_TRUE(n6_index  > n2_index);
    EXPECT_TRUE(n5_index  > n2_index);
    EXPECT_TRUE(n4_index  > n1_index);
    EXPECT_TRUE(n3_index  > n1_index);
    EXPECT_TRUE(n2_index  > n1_index);
    EXPECT_TRUE(n1_index  == 0);
}

TEST_F(GraphFixture, big_graph_topological_sort_performance)
{
#ifndef NDEBUG
    GTEST_SKIP();  // no point in measuring perf in debug mode
    return;
#endif

    const std::vector<std::tuple<int,int,int>> expectedResults = {
    //
    // After several benchmarks, we expect to get the following results
    // on VM machine (hopefully representing the worst-case scenario):
    //
    //            ----------------------------------------------------------
    //              Number     |    Maximum graph    |    Maximum
    //              of         |    population       |    Topological sort
    //              nodes      |    time (millisec)  |    time (millisec)
    //            ----------------------------------------------------------
std::make_tuple(    15000      ,    300              ,    30              ),
std::make_tuple(    50000      ,    1000             ,    100             ),
std::make_tuple(    100000     ,    2000             ,    200             ),
std::make_tuple(    500000     ,    15000            ,    1000            ),
std::make_tuple(    2000000    ,    40000            ,    4000            ),
std::make_tuple(    6000000    ,    104000           ,    11000           )
    //            ----------------------------------------------------------
    };

    unsigned      numNodesSelector        = 3; // 0=15K, 1=50K, 2=100K, 3=500K, 4=2000K, 5=6000K
    unsigned      NUM_NODES               = std::get<0>(expectedResults[numNodesSelector]);
    unsigned      MAX_POPULATION_TIME_MS  = std::get<1>(expectedResults[numNodesSelector]);
    unsigned      MAX_SORT_TIME_MS        = std::get<2>(expectedResults[numNodesSelector]);
    unsigned      NUM_ITERATIONS          = NUM_NODES / 10; // we add 10 nodes each iteration
    unsigned      tensor_dim              = 1;
    TSize         size                    = 1;
    unsigned      durationMs              = 0;
    SimGraph      graph;
    std::clock_t  c_start;
    std::clock_t  c_end;

    // define all variables up-front, everything is made to populate the graph in groups of 10 nodes each
    TensorPtr tin = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr a;
    TensorPtr b;
    TensorPtr c;
    TensorPtr d;
    TensorPtr e;
    TensorPtr f;
    TensorPtr g;
    TensorPtr h;
    TensorPtr z;
    TensorPtr tout;
    NodePtr   n1;
    NodePtr   n2;
    NodePtr   n3;
    NodePtr   n4;
    NodePtr   n5;
    NodePtr   n6;
    NodePtr   n7;
    NodePtr   n8;
    NodePtr   n9;
    NodePtr   n10;

    c_start = std::clock(); // start measuring graph population time

    // Populate the graph in groups of 10 nodes each
    for(unsigned i=0; i < NUM_ITERATIONS; ++i)
    {
        // Create tensors
        a = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        b = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        c = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        d = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        e = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        f = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        g = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        h = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        z = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
        tout = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

        // Create nodes
        n1  = NodeFactory::createDebugNode(tin, a, "");
        n2  = NodeFactory::createDebugNode(a, b, "");
        n3  = NodeFactory::createDebugNode(a, z, "");
        n4  = NodeFactory::createDebugNode(a, c, "");
        n5  = NodeFactory::createDebugNode(b, d, "");
        n6  = NodeFactory::createDebugNode(b, e, "");
        n7  = NodeFactory::createDebugJoinNode(c, d, f, "");
        n8  = NodeFactory::createDebugNode(f, g, "");
        n9  = NodeFactory::createDebugJoinNode(g, e, h, "");
        n10 = NodeFactory::createDebugNode(h, tout, "");

        // Add nodes to graph
        GraphEditor::addNode(graph, n1);
        GraphEditor::addNode(graph, n2);
        GraphEditor::addNode(graph, n3);
        GraphEditor::addNode(graph, n4);
        GraphEditor::addNode(graph, n5);
        GraphEditor::addNode(graph, n6);
        GraphEditor::addNode(graph, n7);
        GraphEditor::addNode(graph, n8);
        GraphEditor::addNode(graph, n9);
        GraphEditor::addNode(graph, n10);

        tin = tout;
    }

    c_end = std::clock(); // end of graph population time
    //std::cout << "Duration of graph population = " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms" << std::endl;
    durationMs = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    EXPECT_LE(durationMs, MAX_POPULATION_TIME_MS);

    // At this point we have populated graphs ready for benchmarking, lets measure topological sort.
    c_start = std::clock();
    NodeVector topo = graph.getTopoSortedNodes(); // do the work
    c_end = std::clock();
    //std::cout << "Duration of topological sort = " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms" << std::endl;

    EXPECT_EQ(topo.size(), NUM_NODES);
    durationMs = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    EXPECT_LE(durationMs, MAX_SORT_TIME_MS);
}

TEST_F(GraphFixture, copy_constructor)
{
    SimGraph graph1(preMadeGraph);
    SimGraph graph2(preMadeGraph);
    EXPECT_TRUE(graph1.validateConnections());
    EXPECT_TRUE(graph2.validateConnections());
    EXPECT_TRUE(graph1.isomorphicTo(graph2));
}

TEST_F(GraphFixture, assignment_operator)
{
     // make graph execution sort cache
    EXPECT_EQ(preMadeGraph.getExeSortedNodes().size(), preMadeGraph.getNumNodes());
    SimGraph graph1;
    SimGraph graph2;
    graph1 = preMadeGraph;
    graph2 = preMadeGraph;
    EXPECT_TRUE(graph1.validateConnections());
    EXPECT_TRUE(graph2.validateConnections());
    EXPECT_TRUE(graph1.isomorphicTo(graph2));

    // We cannot actually compare nodes and tensors from graph1 and graph2 since
    // they are all cloned and we didn't give them any distinct name or data; however,
    // we can expect to have only one root node (cloned from n1) which has one output
    // tensor (cloned from a) which has 3 consumers (cloned from n2, n3 & n4). So
    // lets validate this:
    NodeList roots = graph2.getRootNodes();
    ASSERT_EQ(roots.size(), 1);
    NodePtr      root = roots.front();
    TensorVector outputs = root->getOutputs();
    ASSERT_EQ(outputs.size(), 1);
    auto numConsumers = graph2.getNumberOfTensorConsumers(outputs[0]);
    EXPECT_EQ(numConsumers, 3);
}

TEST_F(GraphFixture, get_node_producers)
{
    NodeSet producers = preMadeGraph.getNodeProducers(n7);
    EXPECT_TRUE(std::find(producers.begin(), producers.end(), n5) != producers.end());
    EXPECT_TRUE(std::find(producers.begin(), producers.end(), n4) != producers.end());
    EXPECT_FALSE(std::find(producers.begin(), producers.end(), n2) != producers.end());
}

TEST_F(GraphFixture, get_node_consumers)
{
    NodeSet consumers = preMadeGraph.getNodeConsumers(n7);
    EXPECT_TRUE(std::find(consumers.begin(), consumers.end(), n8) != consumers.end());
    EXPECT_FALSE(std::find(consumers.begin(), consumers.end(), n9) != consumers.end());
}

TEST_F(GraphFixture, contains_node)
{
    EXPECT_TRUE(preMadeGraph.containsNode(n7));
    GraphEditor::removeNode(preMadeGraph, n7);
    EXPECT_FALSE(preMadeGraph.containsNode(n7));
}

TEST_F(GraphFixture, get_number_of_paths)
{
    EXPECT_EQ(3, preMadeGraph.getNumberOfPaths(n1, n10, Node::TENSOR_TYPE_DATA));
    EXPECT_EQ(0, preMadeGraph.getNumberOfPaths(n1, n10, Node::TENSOR_TYPE_CONTROL));
    EXPECT_EQ(3, preMadeGraph.getNumberOfPaths(n1, n10, Node::TENSOR_TYPE_ALL));

    EXPECT_EQ(2, preMadeGraph.getNumberOfPaths(n2, n9, Node::TENSOR_TYPE_ALL));
    EXPECT_EQ(0, preMadeGraph.getNumberOfPaths(n9, n2, Node::TENSOR_TYPE_ALL));
    GraphEditor::removeNode(preMadeGraph, n7);
    EXPECT_EQ(1, preMadeGraph.getNumberOfPaths(n1, n10, Node::TENSOR_TYPE_ALL));
}

TEST_F(GraphFixture, get_intersecting_nodes1)
{
    NodeSet intersectingNodes = preMadeGraph.getIntersectingNodes({n2, n4, n8});
    EXPECT_EQ(intersectingNodes.size(), 5);
    EXPECT_TRUE(intersectingNodes.find(n2) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n4) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n8) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n5) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n7) != intersectingNodes.end());
}

TEST_F(GraphFixture, get_intersecting_nodes2)
{
    NodeSet intersectingNodes = preMadeGraph.getIntersectingNodes({n1, n4});
    EXPECT_EQ(intersectingNodes.size(), 2);
    EXPECT_TRUE(intersectingNodes.find(n1) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n4) != intersectingNodes.end());
}

TEST_F(GraphFixture, get_intersecting_nodes3)
{
    NodeSet intersectingNodes = preMadeGraph.getIntersectingNodes({n10, n2});
    EXPECT_EQ(intersectingNodes.size(), 7);
    EXPECT_TRUE(intersectingNodes.find(n2) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n5) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n6) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n7) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n8) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n9) != intersectingNodes.end());
    EXPECT_TRUE(intersectingNodes.find(n10) != intersectingNodes.end());
}
