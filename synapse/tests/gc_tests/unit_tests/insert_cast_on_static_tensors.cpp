#include "data_type_utils.h"
#include "graph_factory.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "sim_graph.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "types.h"

class Gaudi2CastInsertOnStaticTensorTest : public GraphOptimizerTest
{
protected:
    virtual void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_CONSTANT_FOLDING, "true");
    }
    virtual void TearDown() override
    {
        GraphOptimizerTest::TearDown();
    }
};

NodePtr createConvNode(synDataType elementType, synDataType bufferType)
{
    const TSize H = 5;
    const TSize W = 5;
    const TSize C = 2;
    const TSize N = 1;
    const TSize K = 2;
    const TSize R = 2;
    const TSize S = 2;

    const std::vector<TSize> inputSizes = {C, W, H, N};

    const TSize weightsStride  = 1;
    const TSize weightsPadding = 1;

    const std::vector<TSize> weightsSizes = {K, C, S, R};

    const TSize outW = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC = K;
    //                                     C    W    H    N
    const std::vector<TSize> outSizes = {outC, outW, outH, N};

    synConvolutionParams params;
    params.dH   = weightsStride;
    params.dW   = weightsStride;
    params.kH   = S;
    params.kW   = R;
    params.padT = weightsPadding;
    params.padB = weightsPadding;
    params.padL = weightsPadding;

    params.padR = weightsPadding;
    params.dilH = 1;
    params.dilW = 1;

    auto weightsNumElem = std::accumulate(weightsSizes.begin(), weightsSizes.end(), 1U, std::multiplies<TSize>());
    char* weightBuffer = reinterpret_cast<char*>(allocateBufferForSynType(bufferType, weightsNumElem));
    TensorPtr weightInput = TensorPtr(new Tensor(weightsSizes.size(), weightsSizes.data(), elementType));
    weightInput->setTensorBuffer(weightBuffer, weightsNumElem * dataTypeSizeInBytes(bufferType), bufferType);
    weightInput->setAsWeights();
    weightInput->setAsStaticParam(true);

    TensorPtr    convIn  = TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_bf16));
    TensorPtr    convOut = TensorPtr(new Tensor(4U, outSizes.data(), syn_type_bf16));
    TensorVector inputs  = {convIn, weightInput};
    TensorVector outputs = {convOut};

    return NodeFactory::createNode(inputs, outputs, &params, NodeFactory::convolutionNodeTypeName, "conv_node");
}

TEST_F(Gaudi2CastInsertOnStaticTensorTest, static_tensor_type_not_match)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    synDataType bufferType = syn_type_float;
    NodePtr convNode = createConvNode(/*elementType=*/syn_type_bf16, bufferType);
    ASSERT_TRUE(GraphEditor::addNode(g, convNode));

    ASSERT_EQ(g.getNumNodes(), 1);

    // verify cast node inserted in graph
    ASSERT_TRUE(staticTensorsCastInsert(g));
    ASSERT_EQ(g.getNumNodes(), 2);

    TensorPtr castOutput = convNode->getInput(TENSOR_WEIGHT);
    NodePtr castNode = g.getTensorProducer(castOutput);
    ASSERT_NE(castNode, nullptr);
    ASSERT_EQ(castNode->getGUID(), "cast_f32_to_bf16");
    ASSERT_EQ(castNode->getInput(0)->getElementType(), bufferType);
    auto castAsTPCNode = std::dynamic_pointer_cast<TPCNode>(castNode);
    ASSERT_TRUE(castAsTPCNode != nullptr);
    ASSERT_NE(castAsTPCNode->getParamsSize(), sizeof(ns_CastKernel::ParamsV3));


    // verify cast node was removed from graph
    ASSERT_TRUE(eliminateNodesWithStaticInputs(g));
    ASSERT_EQ(g.getNumNodes(), 1);
    // verify conv node is still valid
    ASSERT_TRUE(convNode->validateNode());
}

TEST_F(Gaudi2CastInsertOnStaticTensorTest, static_tensor_type_match)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    NodePtr convNode = createConvNode(/*elemType=*/syn_type_bf16, /*bufferType=*/syn_type_bf16);
    ASSERT_TRUE(GraphEditor::addNode(g, convNode));

    ASSERT_EQ(g.getNumNodes(), 1);

    // verify cast node was not inserted in graph
    ASSERT_TRUE(staticTensorsCastInsert(g));
    ASSERT_EQ(g.getNumNodes(), 1);

    // verify graph hasn't changed
    ASSERT_TRUE(eliminateNodesWithStaticInputs(g));
    ASSERT_EQ(g.getNumNodes(), 1);

    // verify conv node is still valid
    ASSERT_TRUE(convNode->validateNode());
}

TEST_F(Gaudi2CastInsertOnStaticTensorTest, multi_consumers_static_tensor_type_not_match)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    synDataType bufferType  = syn_type_float;
    synDataType elementType = syn_type_bf16;

    NodePtr convNode        = createConvNode(elementType, bufferType);
    TensorPtr weightTensor  = convNode->getInput(TENSOR_WEIGHT);
    TensorPtr addInput      = weightTensor->clone(false, false);
    TensorPtr addOutput     = weightTensor->clone(false, false);
    NodePtr addNode         = NodeFactory::createNode({addInput, weightTensor}, {addOutput}, nullptr,
                                                      "add_fwd_bf16", "add_node");

    ASSERT_TRUE(GraphEditor::addNode(g, convNode));
    ASSERT_TRUE(GraphEditor::addNode(g, addNode));

    ASSERT_EQ(g.getNumNodes(), 2);
    ASSERT_EQ(g.getTensorConsumers(weightTensor).size(), 2);

    // verify cast node inserted in graph
    ASSERT_TRUE(staticTensorsCastInsert(g));
    ASSERT_EQ(g.getNumNodes(), 3);

    TensorPtr castOutput = convNode->getInput(TENSOR_WEIGHT);
    NodePtr castNode = g.getTensorProducer(castOutput);
    TensorPtr castInput = castNode->getInput(0);
    ASSERT_NE(castNode, nullptr);
    ASSERT_EQ(castNode->getGUID(), "cast_f32_to_bf16");
    ASSERT_EQ(castInput->getElementType(), bufferType);
    ASSERT_EQ(addNode->getInput(1), castOutput);
    ASSERT_EQ(g.getTensorConsumers(weightTensor).size(), 1);
    ASSERT_EQ(weightTensor, castInput);
    ASSERT_NE(weightTensor, castOutput);

    // verify cast node was removed from graph
    ASSERT_TRUE(eliminateNodesWithStaticInputs(g));
    ASSERT_EQ(g.getNumNodes(), 2);
    // verify conv node is still valid
    ASSERT_TRUE(convNode->validateNode());
}


