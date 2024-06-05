#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "sim_graph.h"

class CSETest : public GraphOptimizerTest
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_CSE_OPTIMIZATION, "true");
    }

public:
    SimGraph               m_g;
    static constexpr TSize M_SIZES[] = {1, 2, 3, 4};
    TensorPtr              m_t5Tag;

    void buildGraph()
    {
        TensorPtr t1  = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t2  = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t3  = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t4  = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t5  = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t6  = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t7  = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t2b = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t3b = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t4b = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t5b = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));
        TensorPtr t6b = TensorPtr(new Tensor(4U, M_SIZES, syn_type_single));

        NodePtr a = NodeFactory::createDebugNode(t1, t2, "a");
        a->setGUID("guid_a");
        NodePtr b = NodeFactory::createDebugNode(t2, t3, "b");
        b->setGUID("guid_b");
        NodePtr c = NodeFactory::createDebugNode(t3, t4, "c");
        c->setGUID("guid_c");
        NodePtr d = NodeFactory::createDebugNode(t3, t5, "d");
        d->setGUID("guid_d");
        NodePtr e = NodeFactory::createDebugJoinNode(t4, t5, t6, "e");
        e->setGUID("guid_e");
        NodePtr f = NodeFactory::createDebugJoinNode(t6, t6b, t7, "f");
        f->setGUID("guid_f");
        NodePtr a_tag = NodeFactory::createDebugNode(t1, t2b, "a_tag");
        a_tag->setGUID("guid_a");
        NodePtr b_tag = NodeFactory::createDebugNode(t2b, t3b, "b_tag");
        b_tag->setGUID("guid_b");
        NodePtr c_tag = NodeFactory::createDebugNode(t3b, t4b, "c_tag");
        c_tag->setGUID("guid_c");
        NodePtr d_tag = NodeFactory::createDebugNode(t3b, t5b, "d_tag");
        d_tag->setGUID("guid_d");
        NodePtr e_tag = NodeFactory::createDebugJoinNode(t4b, t5b, t6b, "e_tag");
        e_tag->setGUID("guid_e");

        ASSERT_TRUE(GraphEditor::addNode(m_g, a));
        ASSERT_TRUE(GraphEditor::addNode(m_g, b));
        ASSERT_TRUE(GraphEditor::addNode(m_g, c));
        ASSERT_TRUE(GraphEditor::addNode(m_g, d));
        ASSERT_TRUE(GraphEditor::addNode(m_g, e));
        ASSERT_TRUE(GraphEditor::addNode(m_g, f));
        ASSERT_TRUE(GraphEditor::addNode(m_g, a_tag));
        ASSERT_TRUE(GraphEditor::addNode(m_g, b_tag));
        ASSERT_TRUE(GraphEditor::addNode(m_g, c_tag));
        ASSERT_TRUE(GraphEditor::addNode(m_g, d_tag));
        ASSERT_TRUE(GraphEditor::addNode(m_g, e_tag));

        m_t5Tag = t5b;
    }
};

/*

                                         +----+   T4
                                      +->| C  |--------+
                                      |  +----+        |
                                      |                |
                                      |                v
               +----+   T2    +----+  |T3             +----+   T6
               | A  |-------->| B  |--+               | E  |---------+
               +----+         +----+  |               +----+         |
                ^                     |                ^             |
                |                     |  +----+   T5   |             v
                |                     +->| D  |--------+          +----+ T7
       T1       |                        +----+                   | F  |--->
+---------------+                                                 +----+
                |                         +----+    T4'              ^
                |                         | C' |--------->           |
                |                         +----+         |           |
                |                           ^            |           |
                v                           |            v           |
                +----+   T2'   +----+  T3'  |          +----+   T6'  |
                | A' |-------->| B' |-------+          | E' |--------+
                +----+         +----+       |          +----+
                                            v            ^
                                          +----+   T5'   |
                                          | D' |---------+
                                          +----+
*/
TEST_F(CSETest, test1)
{
    buildGraph();
    commonSubExpressionElimination(m_g);
    ASSERT_EQ(m_g.getNumNodes(), 6);
    graphVisualizationPost(m_g);
}

TEST_F(CSETest, test2)
{
    buildGraph();
    synMemoryDescriptor desc(/* isPersistent */ true);
    m_t5Tag->setMemoryDescriptor(desc);
    commonSubExpressionElimination(m_g);
    ASSERT_EQ(m_g.getNumNodes(), 7);
    graphVisualizationPost(m_g);
}