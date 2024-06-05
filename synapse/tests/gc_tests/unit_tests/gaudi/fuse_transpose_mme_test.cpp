#include "gaudi_graph.h"
#include "scoped_configuration_change.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "passes/sram_management/flatten_mme.h"
#include "habana_pass.h"
static const unsigned CONV_DIM = 4;

class mmeTransposeFuse : public GraphOptimizerTest
{
protected:
    void createGraph(unsigned    numOfNodes,
                     unsigned    numOfNodesAfterFlatten,
                     unsigned    gemmLocation,
                     bool        addFlatten,
                     bool        addNonFlattenReshape,
                     TSize       sizes[],
                     bool        transposeIndexA,
                     bool        transposeIndexB,
                     const char* mmeName)
    {
        GaudiGraph graph;

        pTensor transposeInputA  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
        pTensor transposeOutputA = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
        pTensor transposeInputB  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
        pTensor transposeOutputB = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
        pTensor gemmInputA       = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
        pTensor gemmInputB       = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
        pTensor gemmOutput       = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

        synTransposeParams transposeParams = {{TPD_Width, TPD_Height, TPD_4Dim_Batch, TPD_Channel}, CONV_DIM};

        if (transposeIndexA)
        {
            pNode transposeNodeA = NodeFactory::createNode({transposeInputA},
                                                           {transposeOutputA},
                                                           &transposeParams,
                                                           NodeFactory::transposeNodeTypeName,
                                                           "transpose_node_a");
            gemmInputA           = transposeOutputA;
            ASSERT_TRUE(GraphEditor::addNode(graph, transposeNodeA)) << "couldnt add transposeA node to graph";
        }
        if (transposeIndexB)
        {
            pNode transposeNodeB = NodeFactory::createNode({transposeInputB},
                                                           {transposeOutputB},
                                                           &transposeParams,
                                                           NodeFactory::transposeNodeTypeName,
                                                           "transpose_node_b");
            gemmInputB           = transposeOutputB;
            ASSERT_TRUE(GraphEditor::addNode(graph, transposeNodeB)) << "couldnt add transposeB node to graph";
        }
        if (addNonFlattenReshape)
        {
            pTensor   gemmInput     = transposeIndexA ? gemmInputA : gemmInputB;
            pTensor   reshapeInput  = gemmInput;
            SizeArray reshapeSizes  = {sizes[1], sizes[0], sizes[2], sizes[3]};
            pTensor   reshapeOutput = std::make_shared<Tensor>(4U, reshapeSizes.data(), syn_type_bf16);

            pNode nonFlatReshape = NodeFactory::createNode({reshapeInput},
                                                           {reshapeOutput},
                                                           nullptr,
                                                           NodeFactory::reshapeNodeTypeName,
                                                           "reshape");
            gemmInput            = reshapeOutput;
            if (transposeIndexA)
            {
                gemmInputA = gemmInput;
            }
            else
            {
                gemmInputB = gemmInput;
            }
            ASSERT_TRUE(GraphEditor::addNode(graph, nonFlatReshape)) << "couldnt add reshape node to graph";
        }
        synConvolutionParams convParams = synConvolutionParams();
        pNode                convNode =
            NodeFactory::createNode({gemmInputA, gemmInputB}, {gemmOutput}, &convParams, mmeName, "mme_node");

        ASSERT_TRUE(GraphEditor::addNode(graph, convNode)) << "couldnt add conv node to graph";
        if (addFlatten)
        {
            MMENodeFlattener flattener(graph);
            flattener.execute();
            ASSERT_EQ(graph.getNumNodes(), numOfNodesAfterFlatten);
        }
        fuseTransposeMme(graph);
        // Number of nodes should be one less after fuse
        ASSERT_EQ(graph.getNumNodes(), numOfNodes);

        unsigned index         = 0;
        bool     gemmNodeFound = false;
        const NodeVector& nodes         = graph.getExeSortedNodes();
        for (const pNode& node : nodes)
        {
            if (index == gemmLocation)
            {
                ASSERT_TRUE(node->getNodeType() == Node::TYPE_GEMM || node->getNodeType() == Node::TYPE_CONVOLUTION);
                gemmNodeFound = true;
                break;
            }
            index++;
        }
        ASSERT_TRUE(gemmNodeFound);
    }
};

TEST_F(mmeTransposeFuse, DISABLED_fuse_transpose_conv_input_a_with_flatten_pass)
{
    // add 3 reshape using flatten_mme, and one gemm
    TSize sizes[] = {5, 1, 1, 5};
    createGraph(4, 5, 2, true, false, sizes, true, false, NodeFactory::convolutionNodeTypeName);
}
TEST_F(mmeTransposeFuse, fuse_transpose_conv_input_a_with_nonflatten_reshape)
{
    // transpose->reshape->gemm we expect not to fuse the transpose in this case.
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(3, 3, 2, false, true, sizes, true, false, NodeFactory::convolutionNodeTypeName);
}

TEST_F(mmeTransposeFuse, fuse_transpose_conv_input_a)
{
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(1, 5, 0, false, false, sizes, true, false, NodeFactory::convolutionNodeTypeName);
}

TEST_F(mmeTransposeFuse, DISABLED_fuse_transpose_conv_input_b_with_flatten_pass)
{
    // add 3 reshape using flatten_mme, and one gemm
    TSize sizes[] = {5, 1, 1, 5};
    createGraph(4, 5, 2, true, false, sizes, false, true, NodeFactory::convolutionNodeTypeName);
}
TEST_F(mmeTransposeFuse, fuse_transpose_conv_input_b_with_nonflatten_reshape)
{
    // transpose->reshape->gemm we expect not to fuse the transpose in this case.
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(3, 3, 2, false, true, sizes, false, true, NodeFactory::convolutionNodeTypeName);
}
TEST_F(mmeTransposeFuse, fuse_transpose_conv_input_b)
{
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(1, 5, 0, false, false, sizes, false, true, NodeFactory::convolutionNodeTypeName);
}
TEST_F(mmeTransposeFuse, DISABLED_fuse_transpose_dedw_input_a_with_flatten_pass)
{
    // add 3 reshape using flatten_mme, and one gemm
    TSize sizes[] = {5, 1, 1, 5};
    createGraph(4, 5, 2, true, false, sizes, true, false, NodeFactory::deDwNodeTypeName);
}

TEST_F(mmeTransposeFuse, fuse_transpose_dedw_input_a)
{
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(1, 5, 0, false, false, sizes, true, false, NodeFactory::deDwNodeTypeName);
}

TEST_F(mmeTransposeFuse, DISABLED_fuse_transpose_dedw_input_b_with_flatten_pass)
{
    // add 3 reshape using flatten_mme, and one gemm
    TSize sizes[] = {5, 1, 1, 5};
    createGraph(4, 5, 2, true, false, sizes, false, true, NodeFactory::deDwNodeTypeName);
}

TEST_F(mmeTransposeFuse, fuse_transpose_dedw_input_b)
{
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(1, 5, 0, false, false, sizes, false, true, NodeFactory::deDwNodeTypeName);
}
TEST_F(mmeTransposeFuse, DISABLED_fuse_transpose_dedx_input_a_with_flatten_pass)
{
    // add 3 reshape using flatten_mme, and one gemm
    TSize sizes[] = {5, 1, 1, 5};
    createGraph(4, 5, 2, true, false, sizes, true, false, NodeFactory::deDxNodeTypeName);
}

TEST_F(mmeTransposeFuse, fuse_transpose_dedx_input_a)
{
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(1, 5, 0, false, false, sizes, true, false, NodeFactory::deDxNodeTypeName);
}

TEST_F(mmeTransposeFuse, DISABLED_fuse_transpose_dedx_input_b_with_flatten_pass)
{
    // add 3 reshape using flatten_mme, and one gemm
    TSize sizes[] = {5, 1, 1, 5};
    createGraph(4, 5, 2, true, false, sizes, false, true, NodeFactory::deDxNodeTypeName);
}

TEST_F(mmeTransposeFuse, fuse_transpose_dedx_input_b)
{
    TSize sizes[] = {1, 1, 1, 1};
    createGraph(1, 5, 0, false, false, sizes, false, true, NodeFactory::deDxNodeTypeName);
}

TEST_F(mmeTransposeFuse, DISABLED_fuse_transpose_reshape_conv_input)
{
    GaudiGraph graph;

    const TSize sizes[]     = { 1, 1, 1, 1 };

    pTensor transposeInput  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor reshapeOutput   = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputB      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

    synTransposeParams transposeParams = {{ TPD_Channel, TPD_Width, TPD_Height, TPD_4Dim_Batch}, CONV_DIM};
    pNode transposeNode = NodeFactory::createNode({transposeInput}, {transposeOutput},
                                                  &transposeParams, NodeFactory::transposeNodeTypeName, "transpose_node");

    pNode reshapeNode = NodeFactory::createNode({transposeOutput}, {reshapeOutput},
                                                nullptr, NodeFactory::reshapeNodeTypeName, "reshape_node");

    synConvolutionParams convParams = synConvolutionParams();
    pNode convNode      = NodeFactory::createNode({reshapeOutput, gemmInputB}, {gemmOutput},
                                                  &convParams, NodeFactory::convolutionNodeTypeName, "conv_node");

    GraphEditor::addNode(graph, transposeNode);
    GraphEditor::addNode(graph, reshapeNode);
    GraphEditor::addNode(graph, convNode);
    fuseTransposeMme(graph);
    //Number of nodes should be two after fuse
    ASSERT_EQ(graph.getNumNodes(), 2);
    //First input should be transpose original input
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInput(0).get(), transposeInput.get());
}

TEST_F(mmeTransposeFuse, fuse_transpose_conv_transpose_dedw)
{
    GaudiGraph graph;

    const TSize sizes[]     = { 1, 1, 1, 1 };

    pTensor transposeInput1  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput1 = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput2 = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputB       = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput       = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputB2      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput2      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Height, TPD_4Dim_Batch, TPD_Channel}, CONV_DIM};
    pNode transposeNode1 = NodeFactory::createNode({transposeInput1}, {transposeOutput1},
                                                  &transposeParams, NodeFactory::transposeNodeTypeName, "transpose_node_1");

    synConvolutionParams convParams = synConvolutionParams();
    pNode convNode      = NodeFactory::createNode({transposeOutput1, gemmInputB}, {gemmOutput},
                                                  &convParams, NodeFactory::convolutionNodeTypeName, "conv_node");

    pNode transposeNode2 = NodeFactory::createNode({gemmOutput}, {transposeOutput2},
                                                   &transposeParams, NodeFactory::transposeNodeTypeName, "transpose_node_2");

    pNode dedwNode      = NodeFactory::createNode({transposeOutput2, gemmInputB2}, {gemmOutput2},
                                                  &convParams, NodeFactory::deDwNodeTypeName, "dedw_node");

    GraphEditor::addNode(graph, transposeNode1);
    GraphEditor::addNode(graph, convNode);
    GraphEditor::addNode(graph, transposeNode2);
    GraphEditor::addNode(graph, dedwNode);
    fuseTransposeMme(graph);
    //Number of nodes should be two after fuse
    ASSERT_EQ(graph.getNumNodes(), 2);
    //First input should be transpose original input
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInput(0).get(), transposeInput1.get());
}

TEST_F(mmeTransposeFuse, fuse_transpose_conv_input_multiple_consumers)
{
    GaudiGraph graph;

    const TSize sizes[] = {1, 1, 1, 1};

    pTensor transposeInput  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor reshapeOutput   = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputB      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Height, TPD_4Dim_Batch, TPD_Channel}, CONV_DIM};
    pNode              transposeNode   = NodeFactory::createNode({transposeInput},
                                                  {transposeOutput},
                                                  &transposeParams,
                                                  NodeFactory::transposeNodeTypeName,
                                                  "transpose_node");

    pNode reshapeNode = NodeFactory::createNode({transposeOutput},
                                                {reshapeOutput},
                                                nullptr,
                                                NodeFactory::reshapeNodeTypeName,
                                                "reshape_node");

    synConvolutionParams convParams = synConvolutionParams();
    pNode                convNode   = NodeFactory::createNode({transposeOutput, gemmInputB},
                                             {gemmOutput},
                                             &convParams,
                                             NodeFactory::convolutionNodeTypeName,
                                             "conv_node");

    GraphEditor::addNode(graph, transposeNode);
    GraphEditor::addNode(graph, reshapeNode);
    GraphEditor::addNode(graph, convNode);
    fuseTransposeMme(graph);
    // Number of nodes should not change
    ASSERT_EQ(graph.getNumNodes(), 3);

    unsigned transposeLocation = 0;
    unsigned gemmLocation      = 2;
    unsigned index             = 0;
    const NodeVector& nodes             = graph.getExeSortedNodes();
    for (const pNode& node : nodes)
    {
        if (index == transposeLocation)
        {
            ASSERT_EQ(node->getNodeType(), Node::TYPE_INTERNAL_TRANSPOSE);
            ASSERT_EQ(node->getInput(0).get(), transposeInput.get());
        }
        else if (index == gemmLocation)
        {
            ASSERT_EQ(node->getNodeType(), Node::TYPE_GEMM);
            ASSERT_EQ(node->getInput(0).get(), transposeInput.get());
        }

        index++;
    }
}

TEST_F(mmeTransposeFuse, fuse_identity_transpose_inputA)
{
    GaudiGraph graph;
    setGlobalConfForTest(GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME, "true");

    const TSize sizes[] = {1, 1, 1, 1};

    pTensor transposeInput  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor identityOutput  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputB      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, CONV_DIM};
    pNode              transposeNode   = NodeFactory::createNode({transposeInput},
                                                  {transposeOutput},
                                                  &transposeParams,
                                                  NodeFactory::transposeNodeTypeName,
                                                  "transpose_node");

    pNode identityNode = NodeFactory::createNode({transposeOutput},
                                                 {identityOutput},
                                                 nullptr,
                                                 NodeFactory::identityNodeTypeName,
                                                 "identity_node");

    synGEMMParams params;
    pNode         gemmNode = NodeFactory::createNode({identityOutput, gemmInputB},
                                             {gemmOutput},
                                             &params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "gemm_node");

    GraphEditor::addNode(graph, transposeNode);
    GraphEditor::addNode(graph, identityNode);
    GraphEditor::addNode(graph, gemmNode);
    fuseTransposeMme(graph);

    ASSERT_EQ(graph.getNumNodes(), 1);
}

TEST_F(mmeTransposeFuse, fuse_identity_transpose_inputB)
{
    GaudiGraph graph;
    setGlobalConfForTest(GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME, "true");

    const TSize sizes[] = {1, 1, 1, 1};

    pTensor transposeInput  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor identityOutput  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputA      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, CONV_DIM};
    pNode              transposeNode   = NodeFactory::createNode({transposeInput},
                                                  {transposeOutput},
                                                  &transposeParams,
                                                  NodeFactory::transposeNodeTypeName,
                                                  "transpose_node");

    pNode identityNode = NodeFactory::createNode({transposeOutput},
                                                 {identityOutput},
                                                 nullptr,
                                                 NodeFactory::identityNodeTypeName,
                                                 "identity_node");

    synGEMMParams params;
    pNode         gemmNode = NodeFactory::createNode({gemmInputA, identityOutput},
                                             {gemmOutput},
                                             &params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "gemm_node");

    GraphEditor::addNode(graph, transposeNode);
    GraphEditor::addNode(graph, identityNode);
    GraphEditor::addNode(graph, gemmNode);
    fuseTransposeMme(graph);

    ASSERT_EQ(graph.getNumNodes(), 1);
}

TEST_F(mmeTransposeFuse, fuse_identity_transpose_output)
{
    GaudiGraph graph;
    setGlobalConfForTest(GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME, "true");

    const TSize sizes[] = {1, 1, 1, 1};

    pTensor transposeOutput = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor identityOutput  = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputA      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputB      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput      = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

    synGEMMParams params;
    pNode         gemmNode = NodeFactory::createNode({gemmInputA, gemmInputB},
                                             {gemmOutput},
                                             &params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "gemm_node");

    pNode identityNode = NodeFactory::createNode({gemmOutput},
                                                 {identityOutput},
                                                 nullptr,
                                                 NodeFactory::identityNodeTypeName,
                                                 "identity_node");

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, CONV_DIM};
    pNode              transposeNode   = NodeFactory::createNode({identityOutput},
                                                  {transposeOutput},
                                                  &transposeParams,
                                                  NodeFactory::transposeNodeTypeName,
                                                  "transpose_node");

    GraphEditor::addNode(graph, gemmNode);
    GraphEditor::addNode(graph, identityNode);
    GraphEditor::addNode(graph, transposeNode);
    fuseTransposeMme(graph);

    ASSERT_EQ(graph.getNumNodes(), 1);
}

TEST_F(mmeTransposeFuse, fuse_transpose_input_output)
{
    GaudiGraph graph;

    const TSize sizes[] = {1, 1, 1, 1};

    pTensor transposeInput   = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput1 = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmInputA       = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor gemmOutput       = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);
    pTensor transposeOutput2 = std::make_shared<Tensor>(4U, sizes, syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, CONV_DIM};
    pNode              transposeNode1  = NodeFactory::createNode({transposeInput},
                                                   {transposeOutput1},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose_node");
    synGEMMParams      params;
    pNode              gemmNode = NodeFactory::createNode({gemmInputA, transposeOutput1},
                                             {gemmOutput},
                                             &params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "batch_gemm_node");

    pNode transposeNode2 = NodeFactory::createNode({gemmOutput},
                                                   {transposeOutput2},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose_node");

    GraphEditor::addNode(graph, transposeNode1);
    GraphEditor::addNode(graph, gemmNode);
    GraphEditor::addNode(graph, transposeNode2);

    fuseTransposeMme(graph);

    ASSERT_EQ(graph.getNumNodes(), 1);
}

TEST_F(mmeTransposeFuse, fuse_transpose_input_gemm)
{
    GaudiGraph graph;

    const TSize sizes[] = {1, 1};

    pTensor transposeInput   = std::make_shared<Tensor>(2U, sizes, syn_type_bf16);
    pTensor transposeOutput1 = std::make_shared<Tensor>(2U, sizes, syn_type_bf16);
    pTensor gemmInputA       = std::make_shared<Tensor>(2U, sizes, syn_type_bf16);
    pTensor gemmOutput       = std::make_shared<Tensor>(2U, sizes, syn_type_bf16);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2U};
    pNode              transposeNode1  = NodeFactory::createNode({transposeInput},
                                                   {transposeOutput1},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose_node");
    synGEMMParams      params;
    pNode              gemmNode = NodeFactory::createNode({gemmInputA, transposeOutput1},
                                             {gemmOutput},
                                             &params,
                                             NodeFactory::gemmNodeTypeName,
                                             "gemm_node");

    GraphEditor::addNode(graph, gemmNode);
    GraphEditor::addNode(graph, transposeNode1);
    fuseTransposeMme(graph);

    ASSERT_EQ(graph.getNumNodes(), 1);
}

typedef enum
{
    INPUT_A = 0b001,
    INPUT_B = 0b010,
    OUTPUT  = 0b100,
} TransposePort;

class mmeTransposeFuseParam
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<unsigned, bool>>
{
public:
    mmeTransposeFuseParam() {}
    static TSize    sizes[5];
    static unsigned dim;

    pTensor
         createTransposeInput(GaudiGraph& graph, bool identity, std::string transpose_name, std::string identity_name);
    void createTransposeOutput(GaudiGraph& graph, bool identity, pTensor input);
};

TSize    mmeTransposeFuseParam::sizes[5] = {1, 1, 1, 1, 1};
unsigned mmeTransposeFuseParam::dim      = 5;

pTensor mmeTransposeFuseParam::createTransposeInput(GaudiGraph& graph,
                                                    bool        identity,
                                                    std::string transpose_name,
                                                    std::string identity_name)
{
    pTensor            input           = nullptr;
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth, TPD_Batch}, dim};

    pTensor transposeInput  = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);
    pTensor transposeOutput = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);
    pNode   transposeNode   = NodeFactory::createNode({transposeInput},
                                                  {transposeOutput},
                                                  &transposeParams,
                                                  NodeFactory::transposeNodeTypeName,
                                                  transpose_name);
    GraphEditor::addNode(graph, transposeNode);
    input = transposeOutput;

    if (identity)
    {
        pTensor identityOutput = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);
        pNode   identityNode   = NodeFactory::createNode({transposeOutput},
                                                     {identityOutput},
                                                     nullptr,
                                                     NodeFactory::identityNodeTypeName,
                                                     identity_name);
        GraphEditor::addNode(graph, identityNode);
        input = identityOutput;
    }

    return input;
}

void mmeTransposeFuseParam::createTransposeOutput(GaudiGraph& graph, bool identity, pTensor input)
{
    pTensor transposeInput = input;

    if (identity)
    {
        pTensor identityOutput = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);
        pNode   identityNode   = NodeFactory::createNode({input},
                                                     {identityOutput},
                                                     nullptr,
                                                     NodeFactory::identityNodeTypeName,
                                                     "identity_out");
        GraphEditor::addNode(graph, identityNode);
        transposeInput = identityOutput;
    }

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth, TPD_Batch}, dim};

    pTensor transposeOutput  = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);
    pNode   transposeNodeOut = NodeFactory::createNode({transposeInput},
                                                     {transposeOutput},
                                                     &transposeParams,
                                                     NodeFactory::transposeNodeTypeName,
                                                     "transpose_out");
    GraphEditor::addNode(graph, transposeNodeOut);
}

INSTANTIATE_TEST_SUITE_P(,
                         mmeTransposeFuseParam,
                         ::testing::Combine(::testing::Values<unsigned>(INPUT_A,
                                                                        INPUT_B,
                                                                        OUTPUT,
                                                                        INPUT_A | INPUT_B,
                                                                        INPUT_A | OUTPUT,
                                                                        INPUT_B | OUTPUT,
                                                                        INPUT_A | INPUT_B | OUTPUT),
                                            ::testing::Values<bool>(false, true)));

TEST_P(mmeTransposeFuseParam, fuse_transpose_mme)
{
    GaudiGraph graph;
    setGlobalConfForTest(GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME, "true");

    auto     params          = GetParam();
    unsigned transposePort   = std::get<0>(params);
    bool     addIdentityNode = std::get<1>(params);

    pTensor inputA          = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);
    pTensor inputB          = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);
    pTensor batchGemmOutput = std::make_shared<Tensor>(dim, sizes, syn_type_bf16);

    if (transposePort & INPUT_A)
    {
        inputA = createTransposeInput(graph, addIdentityNode, "transpose_node_A", "identity_node_A");
    }

    if (transposePort & INPUT_B)
    {
        inputB = createTransposeInput(graph, addIdentityNode, "transpose_node_B", "identity_node_B");
    }

    synGEMMParams gemmParams;
    pNode         batchGemmNode = NodeFactory::createNode({inputA, inputB},
                                                  {batchGemmOutput},
                                                  &gemmParams,
                                                  NodeFactory::batchGemmNodeTypeName,
                                                  "batch_gemm_node");
    ASSERT_TRUE(GraphEditor::addNode(graph, batchGemmNode)) << "node was not added to graph";

    if (transposePort & OUTPUT)
    {
        createTransposeOutput(graph, addIdentityNode, batchGemmOutput);
    }

    fuseTransposeMme(graph);

    ASSERT_EQ(graph.getNumNodes(), 1);
}