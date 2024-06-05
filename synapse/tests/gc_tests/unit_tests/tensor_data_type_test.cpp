#include "tensor.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "graph_optimizer_test.h"

class TensorDataTypeTest : public GraphOptimizerTest
{
protected:
    virtual void SetUp()
    {
        GraphOptimizerTest::SetUp();
        GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(true);
    }
};

TEST_F(TensorDataTypeTest, user_node_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ofm  = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ofm->setName("output");

    pNode add = NodeFactory::createNode({ifm1, ifm2}, {ofm}, nullptr, "add_i16", "addNode");

    ASSERT_TRUE(GraphEditor::addNode(g, add)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_int16);
    ASSERT_EQ(ifm2->getElementType(), syn_type_int16);
    ASSERT_EQ(ofm->getElementType(), syn_type_int16);
}

TEST_F(TensorDataTypeTest, unselected_type_validation)
{
    GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(false);
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ofm  = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ofm->setName("output");

    pNode add = NodeFactory::createNode({ifm1, ifm2}, {ofm}, nullptr, "add_i16", "addNode");

    ASSERT_FALSE(GraphEditor::addNode(g, add))
        << "should prevent from type to not be selected when data type selection is off";
}

TEST_F(TensorDataTypeTest, node_min_precision_tpc_node)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ofm  = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ofm->setName("output");

    pNode add = NodeFactory::createNode({ifm1, ifm2}, {ofm}, nullptr, "add", "addNode");

    ASSERT_TRUE(GraphEditor::addNode(g, add)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm->getElementType(), syn_type_bf16);
}

TEST_F(TensorDataTypeTest, node_min_precision_mme_node)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize    inSize[]   = {1, 2, 2, 1};
    const TSize    wSize[]    = {1, 1, 1, 1};
    const TSize    outSize[]  = {1, 1, 2, 1};

    pTensor ifm1 = pTensor(new Tensor(4U, inSize,  syn_type_na));
    pTensor ifm2 = pTensor(new Tensor(4U, wSize,   syn_type_na));
    pTensor ofm  = pTensor(new Tensor(4U, outSize, syn_type_na));

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ofm->setName("output");
    synConvolution3DParamsV2 params;

    pNode conv = NodeFactory::createNode({ifm1, ifm2}, {ofm}, &params, NodeFactory::convolutionNodeTypeName, "convNode");

    ASSERT_TRUE(GraphEditor::addNode(g, conv)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ifm2->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ofm->getElementType(), syn_type_bf16);
}

TEST_F(TensorDataTypeTest, node_min_precision_mme_and_tpc_nodes)
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

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ifm2->setName("ifm3");
    ofm1->setName("IFM");
    ofm2->setName("ofm2");
    ofm3->setName("output");
    synConvolution3DParamsV2 params;

    pNode add1 = NodeFactory::createNode({ifm1, ifm2}, {ofm1}, nullptr, "add", "addNode1");
    pNode conv = NodeFactory::createNode({ofm1, ifm3}, {ofm2}, &params, NodeFactory::convolutionNodeTypeName, "convNode");
    pNode add2 = NodeFactory::createNode({ofm2, ifm4}, {ofm3}, nullptr, "add", "addNode2");

    ASSERT_TRUE(GraphEditor::addNode(g, add1)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, conv)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add2)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 3) << "Expected 3 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm3->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ifm4->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm1->getElementType(), syn_type_fp8_143);
    ASSERT_EQ(ofm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm3->getElementType(), syn_type_bf16);
}

TEST_F(TensorDataTypeTest, user_data_type)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize sizes[] = {1, 2, 2, 1};
    char        data[]  = {1, 2, 3, 4};

    pTensor ifm1 = pTensor(new Tensor(4U, sizes, syn_type_int8, reinterpret_cast<char*>(data)));
    pTensor ifm2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(data)));
    pTensor ofm  = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(data)));

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ofm->setName("output");

    pNode add = NodeFactory::createNode({ifm1, ifm2}, {ofm}, nullptr, "add", "addNode");
    GraphEditor::addNode(g, add);

    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_int8);
    ASSERT_EQ(ifm2->getElementType(), syn_type_float);
    ASSERT_EQ(ofm->getElementType(), syn_type_int16);
}

TEST_F(TensorDataTypeTest, user_node_precision_vs_user_data_type)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ofm  = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ofm->setName("output");

    ifm1->setElementType(syn_type_int8);
    ofm->setElementType(syn_type_int32);

    pNode add = NodeFactory::createNode({ifm1, ifm2}, {ofm}, nullptr, "add_i16", "addNode");

    ASSERT_TRUE(GraphEditor::addNode(g, add)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm1->getElementType(), syn_type_int8);
    ASSERT_EQ(ifm2->getElementType(), syn_type_int16);
    ASSERT_EQ(ofm->getElementType(), syn_type_int32);
}

TEST_F(TensorDataTypeTest, input_consumers_max_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ifm3 = std::make_shared<Tensor>();
    auto ofm1 = std::make_shared<Tensor>();
    auto ofm2 = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ifm3->setName("ifm3");
    ofm1->setName("output1");
    ofm2->setName("output2");

    pNode addi8  = NodeFactory::createNode({ifm1, ifm2}, {ofm1}, nullptr, "add", "addNodeI8");
    pNode addi16 = NodeFactory::createNode({ifm2, ifm3}, {ofm2}, nullptr, "add_i16", "addNodeI16");

    ASSERT_TRUE(GraphEditor::addNode(g, addi8)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, addi16)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 2) << "Expected 2 nodes";

    ASSERT_EQ(ifm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm3->getElementType(), syn_type_int16);
    ASSERT_EQ(ofm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm2->getElementType(), syn_type_int16);
}

TEST_F(TensorDataTypeTest, input_consumers_max_precision_with_user_data_type)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ifm3 = std::make_shared<Tensor>();
    auto ofm1 = std::make_shared<Tensor>();
    auto ofm2 = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ifm3->setName("ifm3");
    ofm1->setName("output1");
    ofm2->setName("output2");
    ofm2->setElementType(syn_type_float);

    pNode addi8  = NodeFactory::createNode({ifm1, ifm2}, {ofm1}, nullptr, "add", "addNodeI8");
    pNode addi16 = NodeFactory::createNode({ifm2, ifm3}, {ofm2}, nullptr, "add_i16", "addNodeI16");

    ASSERT_TRUE(GraphEditor::addNode(g, addi8)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, addi16)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 2) << "Expected 2 nodes";

    ASSERT_EQ(ifm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm3->getElementType(), syn_type_int16);
    ASSERT_EQ(ofm1->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm2->getElementType(), syn_type_float);
}

TEST_F(TensorDataTypeTest, output_consumers_max_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ifm3 = std::make_shared<Tensor>();
    auto ofm1 = std::make_shared<Tensor>();
    auto ofm2 = std::make_shared<Tensor>();
    auto ofm3 = std::make_shared<Tensor>();
    auto ofm4 = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ifm3->setName("ifm3");
    ofm1->setName("output1");
    ofm2->setName("output2");
    ofm3->setName("output3");
    ofm4->setName("output4");

    pNode add1 = NodeFactory::createNode({ifm1, ifm2}, {ofm1}, nullptr, "add", "addNodeI8_1");
    pNode add2 = NodeFactory::createNode({ifm1, ofm1}, {ofm2}, nullptr, "add_i32", "addNodeI32");
    pNode add3 = NodeFactory::createNode({ifm3, ofm1}, {ofm3}, nullptr, "add_i16", "addNodeI16");
    pNode add4 = NodeFactory::createNode({ofm2, ofm3}, {ofm4}, nullptr, "add_i8", "addNodeI8_2");

    ASSERT_TRUE(GraphEditor::addNode(g, add1)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add2)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add3)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add4)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 4) << "Expected 4 nodes";

    ASSERT_EQ(ifm1->getElementType(), syn_type_int32);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm3->getElementType(), syn_type_int16);
    ASSERT_EQ(ofm1->getElementType(), syn_type_int32);
    ASSERT_EQ(ofm2->getElementType(), syn_type_int8);
    ASSERT_EQ(ofm3->getElementType(), syn_type_int8);
    ASSERT_EQ(ofm4->getElementType(), syn_type_int8);
}

TEST_F(TensorDataTypeTest, output_consumers_max_precision_with_user_data_type)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm1 = std::make_shared<Tensor>();
    auto ifm2 = std::make_shared<Tensor>();
    auto ifm3 = std::make_shared<Tensor>();
    auto ofm1 = std::make_shared<Tensor>();
    auto ofm2 = std::make_shared<Tensor>();
    auto ofm3 = std::make_shared<Tensor>();
    auto ofm4 = std::make_shared<Tensor>();

    ifm1->setName("ifm1");
    ifm2->setName("ifm2");
    ifm3->setName("ifm3");
    ofm1->setName("output1");
    ofm2->setName("output2");
    ofm3->setName("output3");
    ofm4->setName("output4");
    ofm3->setElementType(syn_type_float);
    ofm4->setElementType(syn_type_int16);

    pNode add1 = NodeFactory::createNode({ifm1, ifm2}, {ofm1}, nullptr, "add", "addNodeI8_1");
    pNode add2 = NodeFactory::createNode({ifm1, ofm1}, {ofm2}, nullptr, "add_i32", "addNodeI32");
    pNode add3 = NodeFactory::createNode({ifm3, ofm1}, {ofm3}, nullptr, "add_i16", "addNodeI16");
    pNode add4 = NodeFactory::createNode({ofm2, ofm3}, {ofm4}, nullptr, "add_i8", "addNodeI8_2");

    ASSERT_TRUE(GraphEditor::addNode(g, add1)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add2)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add3)) << "failed to add node to graph";
    ASSERT_TRUE(GraphEditor::addNode(g, add4)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 4) << "Expected 4 nodes";

    ASSERT_EQ(ifm1->getElementType(), syn_type_int32);
    ASSERT_EQ(ifm2->getElementType(), syn_type_bf16);
    ASSERT_EQ(ifm3->getElementType(), syn_type_int16);
    ASSERT_EQ(ofm1->getElementType(), syn_type_int32);
    ASSERT_EQ(ofm2->getElementType(), syn_type_int8);
    ASSERT_EQ(ofm3->getElementType(), syn_type_float);
    ASSERT_EQ(ofm4->getElementType(), syn_type_int16);
}

TEST_F(TensorDataTypeTest, argmax_node_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm = std::make_shared<Tensor>();
    auto ofm = std::make_shared<Tensor>();

    ifm->setName("ifm1");
    ofm->setName("output");

    pNode node = NodeFactory::createNode({ifm}, {ofm}, nullptr, "argmax", "argmaxNode");

    ASSERT_TRUE(GraphEditor::addNode(g, node)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm->getElementType(), syn_type_int32);
}

TEST_F(TensorDataTypeTest, argmin_node_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm = std::make_shared<Tensor>();
    auto ofm = std::make_shared<Tensor>();

    ifm->setName("ifm1");
    ofm->setName("output");

    pNode node = NodeFactory::createNode({ifm}, {ofm}, nullptr, "argmin", "argminNode");

    ASSERT_TRUE(GraphEditor::addNode(g, node)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm->getElementType(), syn_type_int32);
}

TEST_F(TensorDataTypeTest, isnan_node_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm = std::make_shared<Tensor>();
    auto ofm = std::make_shared<Tensor>();

    ifm->setName("ifm1");
    ofm->setName("output");

    pNode node = NodeFactory::createNode({ifm}, {ofm}, nullptr, "isnan", "isnanNode");

    ASSERT_TRUE(GraphEditor::addNode(g, node)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm->getElementType(), syn_type_int8);
}

TEST_F(TensorDataTypeTest, isinf_node_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm = std::make_shared<Tensor>();
    auto ofm = std::make_shared<Tensor>();

    ifm->setName("ifm1");
    ofm->setName("output");

    pNode node = NodeFactory::createNode({ifm}, {ofm}, nullptr, "isinf", "isinfNode");

    ASSERT_TRUE(GraphEditor::addNode(g, node)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm->getElementType(), syn_type_bf16);
    ASSERT_EQ(ofm->getElementType(), syn_type_int8);
}

// Test disabled because topk node was moved to CGUID responsibility, where there is no precision in the guid
TEST_F(TensorDataTypeTest, DISABLED_topk_node_precision)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    auto ifm        = std::make_shared<Tensor>();
    auto ofm        = std::make_shared<Tensor>();
    auto ofmIndices = std::make_shared<Tensor>();

    ifm->setName("ifm1");
    ofm->setName("output");
    ofm->setName("indices");

    synBeamParams params;
    params.bsw  = 2;
    params.axis = 1;

    pNode node = NodeFactory::createNode({ifm}, {ofm, ofmIndices}, &params, "topk_i32", "topkNode");

    ASSERT_TRUE(GraphEditor::addNode(g, node)) << "failed to add node to graph";
    ASSERT_TRUE(setGraphTensorsDataType(g)) << "failed to run setGraphTensorsDataType";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expected 1 node";

    ASSERT_EQ(ifm->getElementType(), syn_type_int32);
    ASSERT_EQ(ofm->getElementType(), syn_type_int32);
    ASSERT_EQ(ofmIndices->getElementType(), syn_type_int16);
}
