#include "gaudi_graph.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "habana_pass.h"

static const unsigned GEMM_DIM = 2;

class GemmTransposeFuse : public GraphOptimizerTest
{
};

TEST_F(GemmTransposeFuse, fuse_transpose_input_a)
{
    GaudiGraph graph;
    pTensor    transposeInput  = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    transposeOutput = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmInputB      = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmOutput      = std::make_shared<Tensor>(syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, GEMM_DIM};
    pNode transposeNode = NodeFactory::createNode({transposeInput}, {transposeOutput},
                                                  &transposeParams, NodeFactory::transposeNodeTypeName, "transpose_node");

    pNode gemmNode      = NodeFactory::createNode({transposeOutput, gemmInputB}, {gemmOutput},
                                                  nullptr, NodeFactory::gemmNodeTypeName, "gemm_node");

    GraphEditor::addNode(graph, transposeNode);
    GraphEditor::addNode(graph, gemmNode);
    fuseTransposeMme(graph);
    //Number of nodes should be one after fuse
    ASSERT_EQ(graph.getNumNodes(), 1);
    //Inputs should be 2 for gemm
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInputs().size(), 2);
    //First input should be transpose original input
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInput(0).get(), transposeInput.get());
}


TEST_F(GemmTransposeFuse, fuse_transpose_input_b)
{
    GaudiGraph graph;
    pTensor    transposeInput  = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    transposeOutput = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmInputA      = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    gemmOutput      = std::make_shared<Tensor>(syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, GEMM_DIM};
    pNode transposeNode = NodeFactory::createNode({transposeInput}, {transposeOutput},
                                                  &transposeParams, NodeFactory::transposeNodeTypeName, "transpose_node");

    pNode gemmNode      = NodeFactory::createNode({gemmInputA, transposeOutput}, {gemmOutput},
                                                  nullptr, NodeFactory::gemmNodeTypeName, "gemm_node");

    GraphEditor::addNode(graph, transposeNode);
    GraphEditor::addNode(graph, gemmNode);
    fuseTransposeMme(graph);
    //Number of nodes should be one after fuse
    ASSERT_EQ(graph.getNumNodes(), 1);
    //Inputs should be 2 for gemm
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInputs().size(), 2);
    //Second input should be transpose original input
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInput(1).get(), transposeInput.get());
}
