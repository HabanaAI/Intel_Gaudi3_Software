#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "habana_pass.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "types.h"

#define DIM 4

class lowerDedxTest : public GraphOptimizerTest
{
};

TEST_F(lowerDedxTest, lower_dedx_test)
{
    setGlobalConfForTest(GCFG_ENABLE_LOWER_DEDX, "true");

    Gaudi2Graph graph;
    CompilationHalReader::setHalReader(Gaudi2HalReader::instance());

    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 3;
    constexpr uint32_t k          = 2;

    TSize xSize[DIM] = {8, 7, 7, 4};
    TSize wSize[DIM] = {k, xSize[0], convParams.kW, convParams.kH};
    TSize ySize[DIM] = {
        k,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3]};

    pTensor dedy = std::make_shared<Tensor>(DIM, ySize, syn_type_bf16);
    pTensor w    = std::make_shared<Tensor>(DIM, wSize, syn_type_bf16);
    pTensor dedx = std::make_shared<Tensor>(DIM, xSize, syn_type_bf16);

    synMemoryDescriptor memDesc(true);
    dedy->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    dedy->setMemoryDescriptor(memDesc);
    w->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    w->setMemoryDescriptor(memDesc);
    dedx->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    dedx->setMemoryDescriptor(memDesc);

    pNode dedxNode = NodeFactory::createNode({dedy, w}, {dedx}, &convParams, NodeFactory::deDxNodeTypeName, "dedx");

    GraphEditor::addNode(graph, dedxNode);
    lowerDedx(graph);
    // Number of nodes should be >= 2 after lowerDedx pass
    ASSERT_GE(graph.getNumNodes(), 2);
    const NodeVector& nodes = graph.getExeSortedNodes();
    ASSERT_TRUE(nodes.back()->getNodeType() == Node::TYPE_TRANSPOSED_DEDX);
}