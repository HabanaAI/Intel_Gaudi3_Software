#include "pair_grads.h"

#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "scoped_configuration_change.h"
#include "types.h"
#include "node_factory.h"
#include "synapse_common_types.hpp"

class PairGradsTest : public GraphOptimizerTest
{
protected:
    Gaudi2Graph m_graph;

    TensorPtr createTensor(const std::vector<TSize> shape)
    {
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
    }

    NodePtr addBGemm(const TensorPtr& a, const TensorPtr& b, const TensorPtr& c)
    {
        synGEMMParams params {};
        NodePtr       bg = NodeFactory::createNode({a, b}, {c}, &params, NodeFactory::batchGemmNodeTypeName, "bgemm");
        HB_ASSERT(GraphEditor::addNode(m_graph, bg), "failed to add batch gemm");
        return bg;
    }

    NodePtr addGemm(const TensorPtr& a, const TensorPtr& b, const TensorPtr& c)
    {
        synGEMMParams params {};
        NodePtr       gemm = NodeFactory::createNode({a, b}, {c}, &params, NodeFactory::gemmNodeTypeName, "gemm");
        HB_ASSERT(GraphEditor::addNode(m_graph, gemm), "failed to add gemm");
        return gemm;
    }

    NodePtr addReshape(const TensorPtr& in, const TensorPtr& out)
    {
        NodePtr reshape = NodeFactory::createNode({in}, {out}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
        HB_ASSERT(GraphEditor::addNode(m_graph, reshape), "failed to add reshape");
        return reshape;
    }

    bool pairGrads() { return GradAReshapedGradBPairTransformer(m_graph).optimizeGradPairs(); }
};

TEST_F(PairGradsTest, grad_a_and_reshaped_grad_b_should_be_paired)
{
    // Given graph:
    //                 [wT]
    //                  |
    //                  v
    // [in (3/4D)]->BGMMgradA->[dA]
    //    |
    //    +-->reshape->[flatIn (2D)]
    //                     |
    //                     v
    //        [flatAT]->GMMgradB->[dB]

    TensorPtr in        = createTensor({2, 3, 40, 50});
    TensorPtr wT        = createTensor({50, 60});
    TensorPtr dA        = createTensor({2, 3, 40, 60});
    NodePtr   BGMMgradA = addBGemm(in, wT, dA);

    TensorPtr flatIn   = createTensor({2 * 3 * 40, 50});
    TensorPtr flatAT   = createTensor({60, 2 * 3 * 40});
    TensorPtr dB       = createTensor({60, 50});
    NodePtr   GMMgradB = addGemm(flatAT, flatIn, dB);

    NodePtr reshape = addReshape(in, flatIn);

    // when paird
    bool res = pairGrads();
    ASSERT_TRUE(res);

    // expected:
    //                                  [wT]
    // [in (3/4D)]                       |
    //    |                              v
    //    +-->reshape->[flatIn (2D)]->GMMgradA->[flatDA]->reshape->[dA]
    //                     |
    //                     v
    //        [flatAT]->GMMgradB->[dB]
    //
    // Checking from wT tensor to make sure gradA is computed by a new GEMM node reading flatIn and having its output
    // reshaped back to be dA
    ASSERT_EQ(1, m_graph.getNumberOfTensorConsumers(wT));
    NodePtr GMMgradA = *m_graph.getTensorConsumers(wT).begin();
    EXPECT_NE(BGMMgradA, GMMgradA);
    EXPECT_EQ(Node::TYPE_GEMM, GMMgradA->getNodeType());
    EXPECT_EQ(flatIn, GMMgradA->getInput(0));
    TensorPtr flatDA = GMMgradA->getOutput(0);
    ASSERT_EQ(1, m_graph.getNumberOfTensorConsumers(flatDA));
    NodePtr unflatten = *m_graph.getTensorConsumers(flatDA).begin();
    EXPECT_EQ(dA, unflatten->getOutput(0));
    EXPECT_EQ(Node::TYPE_INTERNAL_RESHAPE, unflatten->getNodeType());

    // Verify GMMgradB hasn't been hurt
    ASSERT_NE(nullptr, m_graph.getNodeByID(GMMgradB->getId())) << "Unexpected disappearance of GMMgradB from the graph";
    EXPECT_EQ(flatAT, GMMgradB->getInput(0));
    EXPECT_EQ(flatIn, GMMgradB->getInput(1));
    EXPECT_EQ(dB, GMMgradB->getOutput(0));
}