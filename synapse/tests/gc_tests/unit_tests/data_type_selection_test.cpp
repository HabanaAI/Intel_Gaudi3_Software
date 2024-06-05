#include <gtest/gtest.h>
#include "tensor.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "platform/gaudi2/graph_compiler/passes.h"

using namespace std;
using namespace gc;

class DataTypeSelectionTest : public GraphOptimizerTest
{
protected:
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(true);
    }
};

TEST_F(DataTypeSelectionTest, default_user_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize H = 5;
    const TSize W = 5;
    const TSize C = 2;
    const TSize N = 1;
    const TSize K = 2;
    const TSize R = 2;
    const TSize S = 2;

    const std::vector<TSize> inputSizes = {C, W, H, N};
    const std::vector<TSize> weightsSizes = {K, C, S, R};

    const TSize weightsStride  = 1;
    const TSize weightsPadding = 1;
    const TSize outW = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC = K;
    //                                        C    W    H    N
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

    std::vector<int8_t > weightsData = {2, 2, 1, 1, 0, 1, 2, 0, 0, 0, 1, 2, 1, 0, 1, 0};

    pTensor convIn     = pTensor(new Tensor(4U, inputSizes.data(), syn_type_single));
    pTensor convWeight = pTensor(new Tensor(4U, weightsSizes.data(), syn_type_single, reinterpret_cast<char*>(weightsData.data())));
    pTensor convOut    = pTensor(new Tensor(4U, outSizes.data(), syn_type_single));

    convWeight->setAsWeights();
    convWeight->setAsStaticParam(true);
    convWeight->setAsDataTypeMatchData();

    NodePtr node = NodeFactory::createNode({convIn, convWeight},
                                           {convOut},
                                           &params,
                                           "spatial_convolution",
                                           "conv_f32_node");

    ASSERT_EQ(node->getGUID(), "spatial_convolution");
    ASSERT_EQ(node->getNodePrecision(), syn_type_na) << "default user node precision should be syn_type_na";
}

TEST_F(DataTypeSelectionTest, f32_user_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize H = 5;
    const TSize W = 5;
    const TSize C = 2;
    const TSize N = 1;
    const TSize K = 2;
    const TSize R = 2;
    const TSize S = 2;

    const std::vector<TSize> inputSizes = {C, W, H, N};
    const std::vector<TSize> weightsSizes = {K, C, S, R};

    const TSize weightsStride  = 1;
    const TSize weightsPadding = 1;
    const TSize outW = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC = K;
    //                                        C    W    H    N
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

    std::vector<int8_t > weightsData = {2, 2, 1, 1, 0, 1, 2, 0, 0, 0, 1, 2, 1, 0, 1, 0};

    pTensor convIn     = pTensor(new Tensor(4U, inputSizes.data(), syn_type_single));
    pTensor convWeight = pTensor(new Tensor(4U, weightsSizes.data(), syn_type_single, reinterpret_cast<char*>(weightsData.data())));
    pTensor convOut    = pTensor(new Tensor(4U, outSizes.data(), syn_type_single));

    convWeight->setAsWeights();
    convWeight->setAsStaticParam(true);
    convWeight->setAsDataTypeMatchData();

    NodePtr node = NodeFactory::createNode({convIn, convWeight},
                                           {convOut},
                                           &params,
                                           "spatial_convolution_f32",
                                           "conv_f32_node");

    ASSERT_EQ(node->getGUID(), "spatial_convolution");
    ASSERT_EQ(node->getNodePrecision(), syn_type_single);
}

TEST_F(DataTypeSelectionTest, tpc_node_missing_user_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const std::vector<TSize> inSizes  = {1, 2, 2, 1};
    const std::vector<TSize> outSizes = {1, 2, 2, 1};

    pTensor inA = pTensor(new Tensor(4U, inSizes.data(), syn_type_fixed));
    pTensor inB = pTensor(new Tensor(4U, inSizes.data(), syn_type_fixed));
    pTensor out = pTensor(new Tensor(4U, outSizes.data(), syn_type_fixed));

    NodePtr node = NodeFactory::createNode({inA, inB}, {out}, nullptr, "logsoftmax", "logsoftmax_node");

    // TPCNode::createNode no longer adds any suffix as the default value for tpc nodes
    // while the node precision is syn_type_na
    ASSERT_EQ(node->getGUID(), "logsoftmax");
    ASSERT_EQ(node->getNodePrecision(), syn_type_na);
}

TEST_F(DataTypeSelectionTest, tpc_i8_user_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const std::vector<TSize> inSizes  = {1, 2, 2, 1};
    const std::vector<TSize> outSizes = {1, 2, 2, 1};

    pTensor inA = pTensor(new Tensor(4U, inSizes.data(), syn_type_fixed));
    pTensor inB = pTensor(new Tensor(4U, inSizes.data(), syn_type_fixed));
    pTensor out = pTensor(new Tensor(4U, outSizes.data(), syn_type_fixed));

    NodePtr node = NodeFactory::createNode({inA, inB},
                                           {out},
                                           nullptr,
                                           "add_i8",
                                           "add_i8_node");

    ASSERT_EQ(node->getGUID(), "add_i8");
    ASSERT_EQ(node->getNodePrecision(), syn_type_fixed);
}

TEST_F(DataTypeSelectionTest, user_node_type_precision)
{
    Gaudi2Graph  g;
    g.setInferenceMode(true);
    synDataType precision = syn_type_na;

    // User node type precision
    ASSERT_FALSE(g.getUserNodeTypePrecision("user_guid", precision));
    g.setUserNodeTypePrecision("user_guid", syn_type_int16);
    ASSERT_TRUE(g.getUserNodeTypePrecision("user_guid", precision));
    ASSERT_EQ(precision, syn_type_int16);
    ASSERT_FALSE(g.getUserNodeTypePrecision("add", precision));

    // Node type min precision
    ASSERT_EQ(g.getNodeTypeMinPrecision("add"), syn_type_bf16);
    ASSERT_EQ(g.getNodeTypeMinPrecision("notExistGuid"), syn_type_bf16);
}

TEST_F(DataTypeSelectionTest, crop_mirror_ignore_user_data_type_test)
{
    /*
     * Testing correct tensor data type selection resnet50 effort, using GCFG_IGNORE_USER_DATA_TYPE WA.
     * crop_mirror node is from the media pipe before resnet50.
     */
    GCFG_NUM_OF_LAYERS_TO_RAISE.setValue(0);
    GCFG_IGNORE_USER_DATA_TYPE.setValue(true);
    Gaudi2Graph  g;
    g.setInferenceMode(true);
    const std::vector<TSize> inDecoderSizes  = {1, 3, 256, 256};
    const std::vector<TSize> inMediaSizes    = {3, 1, 1};
    const std::vector<TSize> outSizes        = {1, 3, 224, 224};

    synMemoryDescriptor desc(true);
    pTensor inA = pTensor(new Tensor(4U, inDecoderSizes.data(), syn_type_uint8));
    inA->setMemoryDescriptor(desc);
    inA->setDynamicRange({0.5, 1, true});
    pTensor inB = pTensor(new Tensor(3U, inMediaSizes.data(), syn_type_float));
    inB->setAsStaticParam(true);
    pTensor inC = pTensor(new Tensor(3U, inMediaSizes.data(), syn_type_float));
    inC->setAsStaticParam(true);
    pTensor out = pTensor(new Tensor(4U, outSizes.data(), syn_type_float));

    NodePtr node = NodeFactory::createNode({inA, inB, inC},
                                           {out},
                                           nullptr,
                                           "crop_mirror_norm_u8",
                                           "cropMirror");
    ASSERT_TRUE(GraphEditor::addNode(g, node));
    ASSERT_TRUE(gaudi2::setGraphNodesPrecision(g));
    ASSERT_TRUE(setGraphTensorsDataType(g));
    ASSERT_TRUE(g.getNodeByID(node->getId()) != nullptr);
    auto inputs = g.getNodeByID(node->getId())->getInputs();
    ASSERT_EQ(inputs[0]->getElementType(), syn_type_uint8);
    ASSERT_EQ(inputs[1]->getElementType(), syn_type_float);
    ASSERT_EQ(inputs[2]->getElementType(), syn_type_float);
}

TEST_F(DataTypeSelectionTest, node_min_precision_logical_mme_tpc_nodes_1)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize    inSize[]   = {1, 2, 2, 1};
    const TSize    wSize[]    = {1, 1, 1, 1};
    const TSize    outSize[]  = {1, 1, 2, 1};

    pTensor ifm1 = pTensor(new Tensor(4U, inSize,  syn_type_na));
    pTensor ifm2 = pTensor(new Tensor(4U, wSize,   syn_type_na));
    pTensor ifm3 = pTensor(new Tensor(4U, wSize,   syn_type_na));
    pTensor ifm4 = pTensor(new Tensor(4U, wSize,   syn_type_na));
    pTensor ofm1 = pTensor(new Tensor(4U, outSize, syn_type_na));
    pTensor ofm2 = pTensor(new Tensor(4U, outSize, syn_type_na));
    pTensor ofm3 = pTensor(new Tensor(4U, outSize, syn_type_na));
    pTensor ofm4 = pTensor(new Tensor(4U, outSize, syn_type_na));

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ifm3->setName("ifm3");
    ifm4->setName("ifm4");
    ofm1->setName("ofm1");
    ofm2->setName("ofm2");
    ofm3->setName("ofm3");
    ofm4->setName("output");
    synConvolution3DParamsV2 params;

    pNode add1  = NodeFactory::createNode({ifm1, ifm2}, {ofm1}, nullptr, "add", "addNode1");
    pNode split = NodeFactory::createNode({ofm1}, {ofm2}, nullptr, "split", "splitNode");
    pNode conv  = NodeFactory::createNode({ofm2, ifm3}, {ofm3}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    pNode add2  = NodeFactory::createNode({ofm3, ifm4}, {ofm4}, nullptr, "add", "addNode2");

    ASSERT_TRUE(GraphEditor::addNode(g, add1))     << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, split))    << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, conv))     << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add2))     << "failed to add node to graph";
    ASSERT_TRUE(gaudi2::setGraphNodesPrecision(g)) << "failed to run setGraphTensorsDataType";
    ASSERT_TRUE(setGraphTensorsDataType(g))        << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 4) << "Expected 4 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm3->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ifm4->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm1->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ofm2->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ofm3->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm4->getElementType(), syn_type_bf16);
}

TEST_F(DataTypeSelectionTest, node_min_precision_logical_mme_tpc_nodes_2)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize    inSize[]   = {1, 2, 2, 1};
    const TSize    wSize[]    = {1, 1, 1, 1};
    const TSize    outSize[]  = {1, 1, 2, 1};

    pTensor ifm1 = pTensor(new Tensor(4U, inSize,  syn_type_na));
    pTensor ifm2 = pTensor(new Tensor(4U, wSize,   syn_type_na));
    pTensor ifm3 = pTensor(new Tensor(4U, wSize,   syn_type_na));
    pTensor ifm4 = pTensor(new Tensor(4U, wSize,   syn_type_na));
    pTensor ofm1 = pTensor(new Tensor(4U, outSize, syn_type_na));
    pTensor ofm2 = pTensor(new Tensor(4U, outSize, syn_type_na));
    pTensor ofm3 = pTensor(new Tensor(4U, outSize, syn_type_na));
    pTensor ofm4 = pTensor(new Tensor(4U, outSize, syn_type_na));

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ifm3->setName("ifm3");
    ifm4->setName("ifm4");
    ofm1->setName("ofm1");
    ofm2->setName("ofm2");
    ofm3->setName("ofm3");
    ofm4->setName("output");
    synConvolution3DParamsV2 params;

    pNode add   = NodeFactory::createNode({ifm1, ifm2}, {ofm1}, nullptr, "add", "addNode");
    pNode split = NodeFactory::createNode({ofm1}, {ofm2}, nullptr, "split", "splitNode");
    pNode conv1 = NodeFactory::createNode({ofm2, ifm3}, {ofm3}, &params, NodeFactory::convolutionNodeTypeName, "conv1");
    pNode conv2 = NodeFactory::createNode({ofm3, ifm4}, {ofm4}, &params, NodeFactory::convolutionNodeTypeName, "conv2");

    ASSERT_TRUE(GraphEditor::addNode(g, add))      << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, split))    << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, conv1))    << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, conv2))    << "failed to add node to graph";
    ASSERT_TRUE(gaudi2::setGraphNodesPrecision(g)) << "failed to run setGraphTensorsDataType";
    ASSERT_TRUE(setGraphTensorsDataType(g))        << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 4) << "Expected 4 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm3->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ifm4->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ofm1->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ofm2->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ofm3->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm4->getElementType(), syn_type_bf16);
}