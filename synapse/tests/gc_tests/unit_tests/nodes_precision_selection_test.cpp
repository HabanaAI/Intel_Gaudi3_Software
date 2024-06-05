#include <memory>
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"
#include "platform/gaudi2/graph_compiler/passes/set_nodes_precision.cpp"

class NodesPrecisionSelectionTest : public GraphOptimizerTest
{
protected:
    virtual void SetUp()
    {
        GraphOptimizerTest::SetUp();
        GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(true);
    }
};

TEST_F(NodesPrecisionSelectionTest, max_data_types)
{
    // valid dtypes
    ASSERT_EQ(getHighestGUIDDataType({syn_type_bf16, syn_type_int16}), syn_type_bf16);
    ASSERT_EQ(getHighestGUIDDataType({syn_type_bf16, syn_type_uint32}), syn_type_uint32);

    // mix of valid and invalid dtypes
    ASSERT_EQ(getHighestGUIDDataType({syn_type_na, syn_type_uint32}), syn_type_uint32);

    // mix of valid and invalid dtypes
    ASSERT_EQ(getHighestGUIDDataType({syn_type_fp8_143, syn_type_na}), syn_type_fp8_143);

    // only invalid dtypes - returns syn_type_na
    ASSERT_EQ(getHighestGUIDDataType({syn_type_max, syn_type_na}), syn_type_na);
}

TEST_F(NodesPrecisionSelectionTest, node_user_precision)
{
    Gaudi2Graph g;
    bool      ret;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 1;
    const TSize    sizes[]    = {1, 2, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_fixed));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_fixed));
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_fixed));

    pNode add = NodeFactory::createNode({t1, t2}, {t3}, nullptr, "add_i16", "add");
    GraphEditor::addNode(g, add);

    ret = gaudi2::setGraphNodesPrecision(g);
    ASSERT_TRUE(ret) << "failed to run nodes precision selection pass";
    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected a single node";

    ASSERT_EQ(add->getNodePrecision(), syn_type_int16);
    ASSERT_EQ(add->getGUID(), "add_i16");
}

TEST_F(NodesPrecisionSelectionTest, node_type_user_precision)
{
    Gaudi2Graph g;
    bool      ret;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 1;
    const TSize    sizes[]    = {1, 2, 2, 1};
    const char*    addGuid    = "add";

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_fixed));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_fixed));
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_fixed));

    pNode add = NodeFactory::createNode({t1, t2}, {t3}, nullptr, addGuid, "add");
    GraphEditor::addNode(g, add);

    synDataType userNodeTypePrecision = syn_type_int16;
    g.setUserNodeTypePrecision(addGuid, userNodeTypePrecision);

    ret = gaudi2::setGraphNodesPrecision(g);
    ASSERT_TRUE(ret) << "failed to run nodes precision selection pass";
    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected a single node";

    ASSERT_EQ(add->getNodePrecision(), userNodeTypePrecision);
    ASSERT_EQ(add->getGUID(), "add_i16");
}

TEST_F(NodesPrecisionSelectionTest, successors_and_predecessor_functions_simple_graph)
{
    /*
     * [t1] -> (tanh1) -> [t2] -> (tanh2) -> [t3]
     */

    Gaudi2Graph                   g;
    Gaudi2NodesPrecisionSelection nodesPrecisionRunner;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 2;
    const TSize    sizes[]    = {1, 2, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_fixed));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_fixed));
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_fixed));

    pNode tanh1 = NodeFactory::createNode({t1}, {t2}, nullptr, "tanh", "tanh1");
    pNode tanh2 = NodeFactory::createNode({t2}, {t3}, nullptr, "tanh", "tanh2");
    GraphEditor::addNode(g, tanh1);
    GraphEditor::addNode(g, tanh2);

    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected {} nodes" << numOfNodes;

    synDataType tanhMinPrecision = g.getNodeTypeMinPrecision("tanh");  // syn_type_int16

    synDataType tensorDataType = nodesPrecisionRunner.getTensorDataTypeFromConsumers(g, t2);
    ASSERT_EQ(tensorDataType, tanhMinPrecision);

    synDataType nodePrecision = nodesPrecisionRunner.getNodePrecisionFromSuccessors(g, tanh1);
    ASSERT_EQ(nodePrecision, tanhMinPrecision);

    nodePrecision = nodesPrecisionRunner.getNodePrecisionFromPredecessor(g, tanh2);
    ASSERT_EQ(nodePrecision, tanhMinPrecision);
}

TEST_F(NodesPrecisionSelectionTest, successors_and_predecessor_functions_complex_graph)
{
    /*
     * [t1] -> (split) -> [t2] -> (tanh) -> [t4]
     *                         -> (add)   -> [t5]
     *                 -> [t3] ->
     */

    Gaudi2Graph                   g;
    Gaudi2NodesPrecisionSelection nodesPrecisionRunner;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 3;
    const unsigned splitDim   = 1;
    const TSize    inSize[]   = {1, 2, 2, 1};
    const TSize    outSize[]  = {1, 1, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, inSize, syn_type_fixed));
    pTensor t2 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t3 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t4 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t5 = pTensor(new Tensor(4U, inSize, syn_type_fixed));

    pNode split = NodeFactory::createNode({t1}, {t2, t3}, &splitDim, "split", "split");
    pNode tanh  = NodeFactory::createNode({t2}, {t4}, nullptr, "tanh", "tanh");
    pNode add   = NodeFactory::createNode({t2, t3}, {t5}, nullptr, "add", "add");
    GraphEditor::addNode(g, split);
    GraphEditor::addNode(g, tanh);
    GraphEditor::addNode(g, add);

    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected {} nodes" << numOfNodes;

    synDataType tanhMinPrecision = g.getNodeTypeMinPrecision("tanh");  // syn_type_int16
    synDataType addMinPrecision  = g.getNodeTypeMinPrecision("add");   // syn_type_int8

    synDataType tensorDataType = nodesPrecisionRunner.getTensorDataTypeFromConsumers(g, t2);
    ASSERT_EQ(tensorDataType, tanhMinPrecision);

    tensorDataType = nodesPrecisionRunner.getTensorDataTypeFromConsumers(g, t3);
    ASSERT_EQ(tensorDataType, addMinPrecision);

    synDataType nodePrecision = nodesPrecisionRunner.getNodePrecisionFromSuccessors(g, split);
    ASSERT_EQ(nodePrecision, tanhMinPrecision);

    nodePrecision = nodesPrecisionRunner.getNodePrecisionFromPredecessor(g, add);
    // split min precision is syn_type_na, so it take the profile precision
    ASSERT_EQ(nodePrecision, nodesPrecisionRunner.getProfilePrecision());
}

TEST_F(NodesPrecisionSelectionTest, cast_node_no_user_node_precision)
{
    Gaudi2Graph g;
    bool      ret;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 1;
    const TSize    sizes[]    = {1, 2, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_int16));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_int32));

    pNode cast = NodeFactory::createNode({t1}, {t2}, nullptr, "cast_to_i32", "cast");
    GraphEditor::addNode(g, cast);

    ret = gaudi2::setGraphNodesPrecision(g);
    ASSERT_TRUE(ret) << "failed to run nodes precision selection pass";
    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected a single node";

    // take the profile precision cause no predecessor nodes
    ASSERT_EQ(cast->getNodePrecision(), syn_type_fp8_143);
    ASSERT_EQ(cast->getGUID(), "cast_hf8_to_i32");
}

TEST_F(NodesPrecisionSelectionTest, cast_node_with_user_node_precision)
{
    Gaudi2Graph g;
    bool      ret;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 1;
    const TSize    sizes[]    = {1, 2, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_int16));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_int32));

    pNode cast = NodeFactory::createNode({t1}, {t2}, nullptr, "cast_i16_to_i32", "cast");
    GraphEditor::addNode(g, cast);

    ret = gaudi2::setGraphNodesPrecision(g);
    ASSERT_TRUE(ret) << "failed to run nodes precision selection pass";
    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected a single node";

    ASSERT_EQ(cast->getNodePrecision(), syn_type_int16);
    ASSERT_EQ(cast->getGUID(), "cast_i16_to_i32");
}

TEST_F(NodesPrecisionSelectionTest, no_layers_to_raise)
{
    /*                    [w]  ->
     * [t1] -> (split) -> [t2] -> (conv) -> [t4]
     *                         -> (add)   -> [t5]
     *                 -> [t3] ->
     */

    Gaudi2Graph g;
    bool      ret;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 3;
    const unsigned splitDim   = 1;
    const TSize    inSize[]   = {1, 2, 2, 1};
    const TSize    wSize[]    = {1, 1, 1, 1};
    const TSize    outSize[]  = {1, 1, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, inSize,  syn_type_fixed));
    pTensor w  = pTensor(new Tensor(4U, wSize,   syn_type_fixed));
    pTensor t2 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t3 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t4 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t5 = pTensor(new Tensor(4U, inSize,  syn_type_fixed));

    synConvolutionParamsV2 params;

    pNode split = NodeFactory::createNode({t1}, {t2, t3}, &splitDim, "split", "split");
    pNode conv  = NodeFactory::createNode({t2, w}, {t4}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    pNode add   = NodeFactory::createNode({t2, t3}, {t5}, nullptr, "add_f8", "add");
    GraphEditor::addNode(g, split);
    GraphEditor::addNode(g, conv);
    GraphEditor::addNode(g, add);

    GCFG_NUM_OF_LAYERS_TO_RAISE.setValue(0);
    GCFG_PROFILE_PRECISION.setValue("hf8");

    ret = gaudi2::setGraphNodesPrecision(g);
    ASSERT_TRUE(ret) << "failed to run nodes precision selection pass";
    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected {} nodes" << numOfNodes;

    synDataType convPrecision = g.getNodeTypeMinPrecision(NodeFactory::convolutionNodeTypeName);

    // Split precision (f8) the highest between conv precision (hf8) and add precision (f8)
    ASSERT_EQ(split->getNodePrecision(), syn_type_fp8_152);
    ASSERT_EQ(split->getGUID(), "split");
    ASSERT_EQ(conv->getNodePrecision(), convPrecision);

    ASSERT_EQ(add->getNodePrecision(), syn_type_fp8_152);
    ASSERT_EQ(add->getGUID(), "add_f8");
}

TEST_F(NodesPrecisionSelectionTest, multi_node_types)
{
    /*                                     [w]  ->
     * [t1] -> (add) -> [t3] -> (split) -> [t4] -> (conv) -> [t5]
     * [t2] ->                          -------------------> [t6]
     */

    Gaudi2Graph g;
    bool      ret;
    g.setInferenceMode(true);

    const unsigned numOfNodes = 3;
    const unsigned splitDim   = 1;
    const TSize    inSize[]   = {1, 2, 2, 1};
    const TSize    wSize[]    = {1, 1, 1, 1};
    const TSize    outSize[]  = {1, 1, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, inSize,  syn_type_fixed));
    pTensor t2 = pTensor(new Tensor(4U, inSize,  syn_type_fixed));
    pTensor t3 = pTensor(new Tensor(4U, inSize,  syn_type_fixed));
    pTensor t4 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t5 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor t6 = pTensor(new Tensor(4U, outSize, syn_type_fixed));
    pTensor w  = pTensor(new Tensor(4U, wSize,   syn_type_fixed));

    synConvolutionParamsV2 params;

    pNode add   = NodeFactory::createNode({t1, t2}, {t3}, nullptr, "add", "add");
    pNode split = NodeFactory::createNode({t3}, {t4, t5}, &splitDim, "split", "split");
    pNode conv  = NodeFactory::createNode({t4, w}, {t6}, &params, NodeFactory::convolutionNodeTypeName, "conv");

    GraphEditor::addNode(g, add);
    GraphEditor::addNode(g, split);
    GraphEditor::addNode(g, conv);

    GCFG_NUM_OF_LAYERS_TO_RAISE.setValue(0);
    GCFG_PROFILE_PRECISION.setValue("hf8");

    ret = gaudi2::setGraphNodesPrecision(g);
    ASSERT_TRUE(ret) << "failed to run nodes precision selection pass";
    ASSERT_EQ(g.getNodes().size(), numOfNodes) << "Expected {} nodes" << numOfNodes;

    synDataType addPrecision  = g.getNodeTypeMinPrecision("add");
    synDataType convPrecision = g.getNodeTypeMinPrecision(NodeFactory::convolutionNodeTypeName);

    ASSERT_EQ(add->getNodePrecision(), addPrecision);
    ASSERT_EQ(add->getGUID(), "add_bf16");

    // Split logical node precision uses it consumer (conv) precision
    ASSERT_EQ(split->getNodePrecision(), convPrecision);
    ASSERT_EQ(split->getGUID(), "split");
    ASSERT_EQ(conv->getNodePrecision(), convPrecision);
}
