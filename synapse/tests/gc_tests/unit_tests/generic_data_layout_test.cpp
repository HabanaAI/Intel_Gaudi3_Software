#include "node.h"
#include "node_factory.h"
#include <perf_lib_layer_params.h>
#include "transpose_node.h"
#include "generic_graph_test.h"
#include "transposed_shape_node.h"
#include "transpose_dont_care_nodes.h"
#include "test_utils.h"

class GenericDataLayoutTest : public GenericGraphTest
{
protected:
    void wrap_single_add(bool enableRestrictedLayoutsMode = false);
    void wrap_single_add_scalar(bool enableRestrictedLayoutsMode = false);
};

TEST_P(GenericDataLayoutTest, wrap_squeeze_dynamic_shapes)
{
    /*
     * Similar to the squeeze node case in wrap_reshape_with_transposes test, except that the reshape node has a
     * shape tensor.
     * The test verify that the transpose that's left on the shape tensor has the appropriate permutation.
     */
    const std::string resizeGUIDString = getGUIDByDevice(GetParam(), "resize");
    const char*       resizeGUID       = resizeGUIDString.c_str();

    const TSize c              = 3;
    const TSize w              = 20;
    const TSize h              = 10;
    const TSize batch          = 1;

    float input[c * w * h * batch];

    const TSize inSizes[] = {c, w, h, batch};
    TensorPtr   T1        = std::make_shared<Tensor>(4U, inSizes, syn_type_single, reinterpret_cast<char*>(input));
    T1->setName("T1");
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    const TSize resizeSizes[] = {w, h, c, batch};
    TensorPtr   T2            = std::make_shared<Tensor>(4U, resizeSizes, syn_type_single);
    T2->setName("T2");

    synTransposeParams transposeParams;
    transposeParams.tensorDim = T1->getDim();
    // perm = (1, 2, 0, 3) = (CWHN)->(WHCN)
    transposeParams.permutation[0] = TPD_Width;
    transposeParams.permutation[1] = TPD_Height;
    transposeParams.permutation[2] = TPD_Channel;
    transposeParams.permutation[3] = TPD_4Dim_Batch;
    NodePtr trans0                 = NodeFactory::createNode({T1}, {T2}, &transposeParams, "transpose", "transpose_0");
    GraphEditor::addNode(*m_graph, trans0);

    TensorPtr T3 = std::make_shared<Tensor>(4U, resizeSizes, syn_type_single);
    const TSize resizeMinSizes[] = {w, 3, c, batch};
    T3->reshape(4U, resizeSizes, nullptr, resizeMinSizes);
    T3->setName("T3");

    ns_ResizeKernel::Params resizeParams;
    resizeParams.mode           = RESIZE_INTER_LINEAR;
    resizeParams.scaleDim1      = 1;
    resizeParams.scaleDim2      = 1;
    resizeParams.scaleDim3      = 1;
    resizeParams.useScales      = true;
    resizeParams.nearestMode    = ROUND_DEFAULT;
    resizeParams.coordTransMode = ASYMMETRIC_MODE;
    resizeParams.excludeOutside = false;
    // define layouts as they appear in ONNX
    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr resize        = NodeFactory::createNode({T2}, {T3}, &resizeParams, resizeGUID, "resize", layouts);
    GraphEditor::addNode(*m_graph, resize);

    TSize squeezedSizes[] = {w, h, c};
    TSize transSizes[]    = {c, w, h};
    TensorPtr T4 = std::make_shared<Tensor>(3U, squeezedSizes, syn_type_single);
    const TSize squeezedMinSizes[] = {w, 3, c};
    T4->reshape(3U, squeezedSizes, nullptr, squeezedMinSizes);
    T4->setName("T4");

    TensorPtr T3_shapeTensor = std::make_shared<Tensor>(3U, squeezedSizes, syn_type_single);
    T3_shapeTensor->setTensorType(SHAPE_TENSOR);
    T3_shapeTensor->setName("T3_shapeTensor");

    NodePtr reshape = NodeFactory::createNode({T3, T3_shapeTensor}, {T4}, nullptr, "reshape", "reshape");
    GraphEditor::addNode(*m_graph, reshape);

    TensorPtr T5 = std::make_shared<Tensor>(3U, transSizes, syn_type_single);
    T5->setName("T5");
    setTensorAsPersistent(T5, 1);  // graph output must be declared as persistent

    synTransposeParams transposeParams1;
    transposeParams1.tensorDim = T4->getDim();
    // perm = (2, 0, 1) = (WHC)->(CWH)
    transposeParams1.permutation[0] = TPD_Height;
    transposeParams1.permutation[1] = TPD_Channel;
    transposeParams1.permutation[2] = TPD_Width;
    transposeParams1.permutation[3] = TPD_4Dim_Batch;
    transposeParams1.permutation[4] = TPD_Batch;
    NodePtr trans1 = NodeFactory::createNode({T4}, {T5}, &transposeParams1, "transpose", "transpose_1");
    GraphEditor::addNode(*m_graph, trans1);

    ASSERT_TRUE((*m_graph).compile()) << "Failed to compile graph";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 0) << transposeNodeCount << " unexpected transpose nodes found in graph";

    for (const auto& node : (*m_graph).getNodeProducers(reshape))
    {
        TransposedShapeNode* transposedShapeNode = dynamic_cast<TransposedShapeNode*>(node.get());
        if (transposedShapeNode)
        {
            gc::Permutation testedPerm({2, 0, 1});  // (WHC)->(CWH)
            ASSERT_TRUE(gc::Permutation(transposedShapeNode->permutation()) == testedPerm)
                << "Wrong permutation in transpose node " << transposedShapeNode->getNodeName();
        }
    }
}

TEST_P(GenericDataLayoutTest, reshape_not_squeeze)
{
    /* this reshape was previously mistakenly treated as reshape by the "don't care" pass, while it isn't one and
     * later failed on assertion because of it. (and other such cases were found in pytorch)
     * this test makes sure that such reshapes where the output has more dimensions after the "matching" to the input
     * is done don't fail the "don't care" pass. */

    TSize inSize[] = {1, 1, 2};
    TSize outSize[] = {2, 1};

    TensorPtr inTensor(new Tensor(3, inSize, syn_type_bf16));
    TensorPtr outTensor(new Tensor(2, outSize, syn_type_bf16));

    NodePtr reshape = NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::reshapeNodeTypeName, "n1");
    ASSERT_TRUE(reshape != nullptr);

    GraphEditor::addNode(*m_graph, reshape);

    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
}

TEST_P(GenericDataLayoutTest, wrap_single_relu)
{
    /*
     * original graph is:
     * [NCHW tensor with NHWC memory] -> relu
     *
     * the transpose_dont_care_nodes pass (which is where we are testing changes) should wrap the relu with
     * appropriate transposes:
     * [NCHW tensor with NHWC memory] -> transpose ((NCHW)->(NHWC)) -> relu -> transpose ((NHWC)->(NCHW))
     */

    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T1 = std::make_shared<Tensor>(syn_type_single);
    T1->setName("T1");
    // perm = (2, 0, 1, 3) = (WHCN)->(CWHN)
    DimVector        vect {2, 0, 1, 3};
    gc::Permutation  t1Permutation(vect);
    T1->setPermutation(t1Permutation);
    T1->reshape(4U, inSizes, nullptr);
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T2");

    NodePtr relu1 = NodeFactory::createNode({T1}, {T2}, nullptr, reluGUID, "relu_1");
    GraphEditor::addNode(*m_graph, relu1);

    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    NodeSet nodeProducers = (*m_graph).getNodeProducers(relu1);
    NodeSet nodeConsumers = (*m_graph).getNodeConsumers(relu1);
    ASSERT_TRUE(!nodeProducers.empty()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(!nodeConsumers.empty()) << "Missing a transpose node after node " << relu1->getNodeName();
    NodePtr transpose1 = *(nodeProducers.begin());
    NodePtr transpose2 = *(nodeConsumers.begin());
    ASSERT_TRUE(transpose1->isTranspose()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(transpose2->isTranspose()) << "Missing a transpose node after node " << relu1->getNodeName();

    TransposeNode* transposeNode1 = dynamic_cast<TransposeNode*>(transpose1.get());
    TransposeNode* transposeNode2 = dynamic_cast<TransposeNode*>(transpose2.get());
    ASSERT_TRUE(transposeNode1 && gc::Permutation(transposeNode1->permutation()) == t1Permutation)
        << "Wrong permutation in transpose node " << transposeNode1->getNodeName();
    ASSERT_TRUE(transposeNode2 &&
                gc::Permutation(transposeNode2->permutation()) == t1Permutation.getInversePermutation())
        << "Wrong permutation in transpose node " << transposeNode2->getNodeName();
}

TEST_P(GenericDataLayoutTest, wrap_multiple_relu)
{
    /*
     * original graph is:
     * [NCHW tensor with NHWC memory] -> relu -> relu -> relu
     *
     * the transpose_dont_care_nodes pass (which is where we are testing changes) should wrap the relu nodes with
     * appropriate transposes:
     * [NCHW tensor with NHWC memory] -> transpose ((NCHW)->(NHWC)) -> relu -> transpose ((NHWC)->(NCHW))
     *     -> transpose ((NCHW)->(NHWC)) -> relu -> transpose ((NHWC)->(NCHW)) -> transpose ((NCHW)->(NHWC))
     *     -> relu -> transpose ((NHWC)->(NCHW))
     *
     * eventually, the remove_contiguous_transposes pass should get rid of the transposes between the relu nodes, since
     * two pairs of identity transposes were created
     */

    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T1 = std::make_shared<Tensor>(syn_type_single);
    T1->setName("T1");
    // perm = (2, 0, 1, 3) = (WHCN)->(CWHN)
    DimVector        vect {2, 0, 1, 3};
    gc::Permutation  t1Permutation(vect);
    T1->setPermutation(t1Permutation);
    T1->reshape(4U, inSizes, nullptr);
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T2");

    NodePtr relu1 = NodeFactory::createNode({T1}, {T2}, nullptr, reluGUID, "relu_1");
    GraphEditor::addNode(*m_graph, relu1);

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T3->setName("T3");

    NodePtr relu2 = NodeFactory::createNode({T2}, {T3}, nullptr, reluGUID, "relu_2");
    GraphEditor::addNode(*m_graph, relu2);

    TensorPtr T4 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T4->setName("T4");

    NodePtr relu3 = NodeFactory::createNode({T3}, {T4}, nullptr, reluGUID, "relu_3");
    GraphEditor::addNode(*m_graph, relu3);

    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 2) << transposeNodeCount << " unexpected transpose nodes found in graph";

    NodeSet nodeProducers = (*m_graph).getNodeProducers(relu1);
    NodeSet nodeConsumers = (*m_graph).getNodeConsumers(relu3);
    ASSERT_TRUE(!nodeProducers.empty()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(!nodeConsumers.empty()) << "Missing a transpose node after node " << relu3->getNodeName();
    NodePtr transpose1 = *(nodeProducers.begin());
    NodePtr transpose2 = *(nodeConsumers.begin());
    ASSERT_TRUE(transpose1->isTranspose()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(transpose2->isTranspose()) << "Missing a transpose node after node " << relu3->getNodeName();

    TransposeNode* transposeNode1 = dynamic_cast<TransposeNode*>(transpose1.get());
    TransposeNode* transposeNode2 = dynamic_cast<TransposeNode*>(transpose2.get());
    ASSERT_TRUE(transposeNode1 && gc::Permutation(transposeNode1->permutation()) == t1Permutation)
        << "Wrong permutation in transpose node " << transposeNode1->getNodeName();
    ASSERT_TRUE(transposeNode2 &&
                gc::Permutation(transposeNode2->permutation()) == t1Permutation.getInversePermutation())
        << "Wrong permutation in transpose node " << transposeNode2->getNodeName();
}

void GenericDataLayoutTest::wrap_single_add(bool enableRestrictedLayoutsMode)
{
    if (enableRestrictedLayoutsMode)
    {
        GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE.setValue(true);
    }

    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T1 = std::make_shared<Tensor>(syn_type_single);
    T1->setName("T1");
    // perm = (2, 0, 1, 3) = (WHCN)->(CWHN)
    DimVector        vect {2, 0, 1, 3};
    gc::Permutation  t1Permutation(vect);
    T1->setPermutation(t1Permutation);
    T1->reshape(4U, inSizes, nullptr);
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T2");
    setTensorAsPersistent(T2);  // graph input must be declared as persistent

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T3->setName("T3");

    NodePtr add = NodeFactory::createNode({T1, T2}, {T3}, nullptr, addGUID, "add");
    GraphEditor::addNode(*m_graph, add);

    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 3) << transposeNodeCount << " unexpected transpose nodes found in graph";
}

TEST_P(GenericDataLayoutTest, wrap_single_add)
{
    wrap_single_add();
}

TEST_P(GenericDataLayoutTest, wrap_single_add_feature_on)
{
    wrap_single_add(true);
}

TEST_P(GenericDataLayoutTest, wrap_single_add_with_identity_perm)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T1 = std::make_shared<Tensor>(syn_type_single);
    T1->setName("T1");
    // perm = (2, 0, 1, 3) = (WHCN)->(CWHN)
    DimVector        vect {2, 0, 1, 3};
    gc::Permutation  t1Permutation(vect);
    T1->setPermutation(t1Permutation);
    T1->reshape(4U, inSizes, nullptr);
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T2 = std::make_shared<Tensor>(syn_type_single);
    T2->setName("T2");
    // Identity permutation
    gc::Permutation t2Permutation(4);
    T2->setPermutation(t2Permutation);
    T2->reshape(4U, inSizes, nullptr);
    setTensorAsPersistent(T2);  // graph input must be declared as persistent

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T3->setName("T3");

    NodePtr add = NodeFactory::createNode({T1, T2}, {T3}, nullptr, addGUID, "add");
    GraphEditor::addNode(*m_graph, add);

    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 3) << transposeNodeCount << " unexpected transpose nodes found in graph";
}

void GenericDataLayoutTest::wrap_single_add_scalar(bool enableRestrictedLayoutsMode)
{
    if (enableRestrictedLayoutsMode)
    {
        GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE.setValue(true);
    }

    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T1 = std::make_shared<Tensor>(syn_type_single);
    T1->setName("T1");
    // perm = (2, 0, 1, 3) = (WHCN)->(CWHN)
    DimVector        vect {2, 0, 1, 3};
    gc::Permutation  t1Permutation(vect);
    T1->setPermutation(t1Permutation);
    T1->reshape(4U, inSizes, nullptr);
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    const TSize inSizes2[] = {1};
    TensorPtr   T2         = std::make_shared<Tensor>(1U, inSizes2, syn_type_single);
    T2->setName("T2");
    setTensorAsPersistent(T2);  // graph input must be declared as persistent

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T3->setName("T3");

    NodePtr add = NodeFactory::createNode({T1, T2}, {T3}, nullptr, addGUID, "add");
    if (enableRestrictedLayoutsMode)
    {
        add->getNodeIOManager().setInputRestrictedAtIndex(1);
    }
    GraphEditor::addNode(*m_graph, add);

    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 2) << transposeNodeCount << " unexpected transpose nodes found in graph";
}

TEST_P(GenericDataLayoutTest, wrap_single_add_scalar)
{
    wrap_single_add_scalar();
}

TEST_P(GenericDataLayoutTest, wrap_single_add_scalar_feature_on)
{
    wrap_single_add_scalar(true);
}

TEST_P(GenericDataLayoutTest, wrap_single_add_broadcasted)
{
    GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE.setValue(true);

    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T1 = std::make_shared<Tensor>(syn_type_single);
    T1->setName("T1");
    // perm = (2, 0, 1, 3) = (WHCN)->(CWHN)
    DimVector        vect {2, 0, 1, 3};
    gc::Permutation  t1Permutation(vect);
    T1->setPermutation(t1Permutation);
    T1->reshape(4U, inSizes, nullptr);
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    const TSize inSizes2[] = {w, h};
    TensorPtr T2 = std::make_shared<Tensor>(2U, inSizes2, syn_type_single);
    T2->setName("T2");
    setTensorAsPersistent(T2);  // graph input must be declared as persistent

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T3->setName("T3");

    NodePtr add = NodeFactory::createNode({T1, T2}, {T3}, nullptr, addGUID, "add");
    GraphEditor::addNode(*m_graph, add);

    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    unsigned int reshapeNodeCount   = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node == nullptr) continue;
        if (node->isTranspose())
        {
            ++transposeNodeCount;
        }
        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            ++reshapeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 3) << transposeNodeCount << " unexpected transpose nodes found in graph";
    ASSERT_EQ(reshapeNodeCount, 1) << reshapeNodeCount << " unexpected reshape nodes found in graph";
}

TEST_P(GenericDataLayoutTest, wrap_simple_graph)
{
    /*
     * original graph is:
     * [WHCN tensor] -> relu -> resize -> relu
     *
     * the resize is defined with conflicting layouts, so during data layout adjustment, transposes should be inserted
     * before and after the resize:
     * [WHCN tensor] -> relu -> transpose ((WHCN)->(CWHN)) -> resize -> transpose ((CWHN)->(WHCN)) -> relu
     *
     * the transpose_dont_care_nodes pass (which is where we are testing changes) should wrap the relu nodes with
     * appropriate transposes:
     * [WHCN tensor] -> transpose ((WHCN)->(CWHN)) -> relu -> transpose ((CWHN)->(WHCN)) -> transpose ((WHCN)->(CWHN))
     *     -> resize -> transpose ((CWHN)->(WHCN)) -> transpose ((WHCN)->(CWHN)) -> relu -> transpose ((CWHN)->(WHCN))
     *
     * eventually, the remove_contiguous_transposes pass should get rid of the transposes between the relu nodes, since
     * two pairs of identity transposes were created
     */

    const std::string reluGUIDString   = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID         = reluGUIDString.c_str();
    const std::string resizeGUIDString = getGUIDByDevice(GetParam(), "resize");
    const char*       resizeGUID       = resizeGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    TensorPtr T1 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T1");

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T2");

    NodePtr relu1 = NodeFactory::createNode({T1}, {T2}, nullptr, reluGUID, "relu_1");
    GraphEditor::addNode(*m_graph, relu1);

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T3->setName("T3");

    ns_ResizeKernel::Params resizeParams;
    resizeParams.mode           = RESIZE_INTER_LINEAR;
    resizeParams.scaleDim1      = 1;
    resizeParams.scaleDim2      = 1;
    resizeParams.scaleDim3      = 1;
    resizeParams.useScales      = true;
    resizeParams.nearestMode    = ROUND_DEFAULT;
    resizeParams.coordTransMode = ASYMMETRIC_MODE;
    resizeParams.excludeOutside = false;
    // define layouts as they appear in ONNX
    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr resize        = NodeFactory::createNode({T2}, {T3}, &resizeParams, resizeGUID, "resize", layouts);
    GraphEditor::addNode(*m_graph, resize);

    TensorPtr T4 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T4->setName("T4");

    NodePtr relu2 = NodeFactory::createNode({T3}, {T4}, nullptr, reluGUID, "relu_2");
    GraphEditor::addNode(*m_graph, relu2);

    ASSERT_TRUE(setSupportedLayouts(*m_graph)) << "failed to run setSupportedLayouts";
    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 2) << transposeNodeCount << " unexpected transpose nodes found in graph";

    NodeSet nodeProducers = (*m_graph).getNodeProducers(relu1);
    NodeSet nodeConsumers = (*m_graph).getNodeConsumers(relu2);
    ASSERT_TRUE(!nodeProducers.empty()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(!nodeConsumers.empty()) << "Missing a transpose node after node " << relu2->getNodeName();
    NodePtr transpose1 = *(nodeProducers.begin());
    NodePtr transpose2 = *(nodeConsumers.begin());
    ASSERT_TRUE(transpose1->isTranspose()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(transpose2->isTranspose()) << "Missing a transpose node after node " << relu2->getNodeName();

    gc::Permutation testedPerm({2, 0, 1, 3});  // (WHCN)->(CWHN)
    TransposeNode*  transposeNode1 = dynamic_cast<TransposeNode*>(transpose1.get());
    TransposeNode*  transposeNode2 = dynamic_cast<TransposeNode*>(transpose2.get());
    ASSERT_TRUE(transposeNode1 && gc::Permutation(transposeNode1->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode1->getNodeName();
    ASSERT_TRUE(transposeNode2 && gc::Permutation(transposeNode2->permutation()) == testedPerm.getInversePermutation())
        << "Wrong permutation in transpose node " << transposeNode2->getNodeName();
}

TEST_P(GenericDataLayoutTest, wrap_complex_graph)
{
    /*
     * notes:
     * - node marked with a "*" is conflicted
     * - t is a transpose, and t' is its inverse transpose
     *
     * original graph is:
     *      relu    resize*   relu    relu
     *         \ _____ |       | _____ /
     *                add     conv*
     *                  \     /
     *                   conv*
     *
     * post graph is:
     *        t        t       t        t
     *        |        |       |        |
     *      relu    resize*   relu    relu
     *         \ _____ |       | _____ /
     *                add     conv*
     *                  \     /
     *                   conv*
     *                     |
     *                     t'
     */

    const std::string reluGUIDString   = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID         = reluGUIDString.c_str();
    const std::string resizeGUIDString = getGUIDByDevice(GetParam(), "resize");
    const char*       resizeGUID       = resizeGUIDString.c_str();
    const std::string addGUIDString    = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID          = addGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    TensorPtr T1 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T1");
    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T2");
    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T3");
    TensorPtr T4 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T3");
    TensorPtr T5 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T5");
    TensorPtr T6 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T6");
    TensorPtr T7 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T7");
    TensorPtr T8 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T8");
    TensorPtr T9 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T9");
    TensorPtr T10 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T10");
    TensorPtr T11 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T11");

    NodePtr relu1 = NodeFactory::createNode({T1}, {T5}, nullptr, reluGUID, "relu_1");
    GraphEditor::addNode(*m_graph, relu1);

    ns_ResizeKernel::Params resizeParams;
    resizeParams.mode           = RESIZE_INTER_LINEAR;
    resizeParams.scaleDim1      = 1;
    resizeParams.scaleDim2      = 1;
    resizeParams.scaleDim3      = 1;
    resizeParams.useScales      = true;
    resizeParams.nearestMode    = ROUND_DEFAULT;
    resizeParams.coordTransMode = ASYMMETRIC_MODE;
    resizeParams.excludeOutside = false;
    // define layouts as they appear in ONNX
    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr resize        = NodeFactory::createNode({T2}, {T6}, &resizeParams, resizeGUID, "resize", layouts);
    GraphEditor::addNode(*m_graph, resize);

    NodePtr relu2 = NodeFactory::createNode({T3}, {T7}, nullptr, reluGUID, "relu_2");
    GraphEditor::addNode(*m_graph, relu2);

    NodePtr relu3 = NodeFactory::createNode({T4}, {T8}, nullptr, reluGUID, "relu_3");
    GraphEditor::addNode(*m_graph, relu3);

    NodePtr add = NodeFactory::createNode({T5, T6}, {T9}, nullptr, addGUID, "add");
    GraphEditor::addNode(*m_graph, add);

    synConvolutionParams convParams;
    convParams.dH   = 1;
    convParams.dW   = 1;
    convParams.kH   = 1;
    convParams.kW   = 1;
    convParams.padT = 0;
    convParams.padB = 0;
    convParams.padL = 0;
    convParams.padR = 0;
    convParams.dilH = 1;
    convParams.dilW = 1;

    layouts.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("CSKR")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr conv1 = NodeFactory::createNode({T7, T8}, {T10}, &convParams, "spatial_convolution", "conv1", layouts);
    GraphEditor::addNode(*m_graph, conv1);

    NodePtr conv2 = NodeFactory::createNode({T9, T10}, {T11}, &convParams, "spatial_convolution", "conv2", layouts);
    GraphEditor::addNode(*m_graph, conv2);

    ASSERT_TRUE(setSupportedLayouts(*m_graph)) << "failed to run setSupportedLayouts";
    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 5) << transposeNodeCount << " unexpected transpose nodes found in graph";

    NodeSet relu1Producers  = (*m_graph).getNodeProducers(relu1);
    NodeSet resizeProducers = (*m_graph).getNodeProducers(resize);
    NodeSet relu2Producers  = (*m_graph).getNodeProducers(relu2);
    NodeSet relu3Producers  = (*m_graph).getNodeProducers(relu3);
    NodeSet conv2Consumers  = (*m_graph).getNodeConsumers(conv2);
    ASSERT_TRUE(!relu1Producers.empty()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(!resizeProducers.empty()) << "Missing a transpose node before node " << resize->getNodeName();
    ASSERT_TRUE(!relu2Producers.empty()) << "Missing a transpose node before node " << relu2->getNodeName();
    ASSERT_TRUE(!relu3Producers.empty()) << "Missing a transpose node before node " << relu3->getNodeName();
    ASSERT_TRUE(!conv2Consumers.empty()) << "Missing a transpose node after node " << conv2->getNodeName();
    NodePtr transpose1 = *(relu1Producers.begin());
    NodePtr transpose2 = *(resizeProducers.begin());
    NodePtr transpose3 = *(relu2Producers.begin());
    NodePtr transpose4 = *(relu3Producers.begin());
    NodePtr transpose5 = *(conv2Consumers.begin());
    ASSERT_TRUE(transpose1->isTranspose()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(transpose2->isTranspose()) << "Missing a transpose node before node " << resize->getNodeName();
    ASSERT_TRUE(transpose3->isTranspose()) << "Missing a transpose node before node " << relu2->getNodeName();
    ASSERT_TRUE(transpose4->isTranspose()) << "Missing a transpose node before node " << relu3->getNodeName();
    ASSERT_TRUE(transpose5->isTranspose()) << "Missing a transpose node after node " << conv2->getNodeName();

    gc::Permutation testedPerm({2, 0, 1, 3});  // (WHCN)->(CWHN)
    TransposeNode*  transposeNode1 = dynamic_cast<TransposeNode*>(transpose1.get());
    TransposeNode*  transposeNode2 = dynamic_cast<TransposeNode*>(transpose2.get());
    TransposeNode*  transposeNode3 = dynamic_cast<TransposeNode*>(transpose3.get());
    TransposeNode*  transposeNode4 = dynamic_cast<TransposeNode*>(transpose4.get());
    TransposeNode*  transposeNode5 = dynamic_cast<TransposeNode*>(transpose5.get());
    ASSERT_TRUE(transposeNode1 && gc::Permutation(transposeNode1->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode1->getNodeName();
    ASSERT_TRUE(transposeNode2 && gc::Permutation(transposeNode2->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode2->getNodeName();
    ASSERT_TRUE(transposeNode3 && gc::Permutation(transposeNode3->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode3->getNodeName();
    ASSERT_TRUE(transposeNode4 && gc::Permutation(transposeNode4->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode4->getNodeName();
    ASSERT_TRUE(transposeNode5 && gc::Permutation(transposeNode5->permutation()) == testedPerm.getInversePermutation())
        << "Wrong permutation in transpose node " << transposeNode5->getNodeName();
}

TEST_P(GenericDataLayoutTest, wrap_complex_graph_bfs)
{
    /*
     * notes:
     * - node marked with a "*" is conflicted
     * - t is a transpose, and t' is its inverse transpose
     *
     * original graph is:
     *      relu    resize*   relu    relu
     *         \ _____ |       | _____ |
     *                add     conv*   relu
     *                  \     /
     *                   conv*
     *
     * post graph is:
     *        t        t       t        t
     *        |        |       |        |
     *      relu    resize*   relu    relu
     *         \ _____ |       | ______ |
     *                add     conv*    relu
     *                  \     /         |
     *                   conv*          t'
     *                     |
     *                     t'
     */

    setGlobalConfForTest(GCFG_TRANSPOSE_DONT_CARE_USE_BFS, "true");

    const std::string reluGUIDString   = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID         = reluGUIDString.c_str();
    const std::string resizeGUIDString = getGUIDByDevice(GetParam(), "resize");
    const char*       resizeGUID       = resizeGUIDString.c_str();
    const std::string addGUIDString    = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID          = addGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    TensorPtr T1 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T1");
    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T2");
    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T3");
    TensorPtr T4 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T3");
    TensorPtr T5 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T5");
    TensorPtr T6 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T6");
    TensorPtr T7 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T7");
    TensorPtr T8 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T8");
    TensorPtr T9 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T9");
    TensorPtr T10 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T10");
    TensorPtr T11 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T11");
    TensorPtr T12 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T12");

    NodePtr relu1 = NodeFactory::createNode({T1}, {T5}, nullptr, reluGUID, "relu_1");
    GraphEditor::addNode(*m_graph, relu1);

    ns_ResizeKernel::Params resizeParams;
    resizeParams.mode           = RESIZE_INTER_LINEAR;
    resizeParams.scaleDim1      = 1;
    resizeParams.scaleDim2      = 1;
    resizeParams.scaleDim3      = 1;
    resizeParams.useScales      = true;
    resizeParams.nearestMode    = ROUND_DEFAULT;
    resizeParams.coordTransMode = ASYMMETRIC_MODE;
    resizeParams.excludeOutside = false;
    // define layouts as they appear in ONNX
    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr resize        = NodeFactory::createNode({T2}, {T6}, &resizeParams, resizeGUID, "resize", layouts);
    GraphEditor::addNode(*m_graph, resize);

    NodePtr relu2 = NodeFactory::createNode({T3}, {T7}, nullptr, reluGUID, "relu_2");
    GraphEditor::addNode(*m_graph, relu2);

    NodePtr relu3 = NodeFactory::createNode({T4}, {T8}, nullptr, reluGUID, "relu_3");
    GraphEditor::addNode(*m_graph, relu3);

    NodePtr add = NodeFactory::createNode({T5, T6}, {T9}, nullptr, addGUID, "add");
    GraphEditor::addNode(*m_graph, add);

    synConvolutionParams convParams;
    convParams.dH   = 1;
    convParams.dW   = 1;
    convParams.kH   = 1;
    convParams.kW   = 1;
    convParams.padT = 0;
    convParams.padB = 0;
    convParams.padL = 0;
    convParams.padR = 0;
    convParams.dilH = 1;
    convParams.dilW = 1;

    layouts.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("CSKR")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr conv1 = NodeFactory::createNode({T7, T8}, {T10}, &convParams, "spatial_convolution", "conv1", layouts);
    GraphEditor::addNode(*m_graph, conv1);

    NodePtr relu4 = NodeFactory::createNode({T8}, {T11}, nullptr, reluGUID, "relu_4");
    GraphEditor::addNode(*m_graph, relu4);

    NodePtr conv2 = NodeFactory::createNode({T9, T10}, {T12}, &convParams, "spatial_convolution", "conv2", layouts);
    GraphEditor::addNode(*m_graph, conv2);

    ASSERT_TRUE(setSupportedLayouts(*m_graph)) << "failed to run setSupportedLayouts";
    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 6) << transposeNodeCount << " unexpected transpose nodes found in graph";

    NodeSet relu1Producers  = (*m_graph).getNodeProducers(relu1);
    NodeSet resizeProducers = (*m_graph).getNodeProducers(resize);
    NodeSet relu2Producers  = (*m_graph).getNodeProducers(relu2);
    NodeSet relu3Producers  = (*m_graph).getNodeProducers(relu3);
    NodeSet conv2Consumers  = (*m_graph).getNodeConsumers(conv2);
    NodeSet relu4Consumers  = (*m_graph).getNodeConsumers(relu4);
    ASSERT_TRUE(!relu1Producers.empty()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(!resizeProducers.empty()) << "Missing a transpose node before node " << resize->getNodeName();
    ASSERT_TRUE(!relu2Producers.empty()) << "Missing a transpose node before node " << relu2->getNodeName();
    ASSERT_TRUE(!relu3Producers.empty()) << "Missing a transpose node before node " << relu3->getNodeName();
    ASSERT_TRUE(!conv2Consumers.empty()) << "Missing a transpose node after node " << conv2->getNodeName();
    ASSERT_TRUE(!relu4Consumers.empty()) << "Missing a transpose node after node " << relu4->getNodeName();
    NodePtr transpose1 = *(relu1Producers.begin());
    NodePtr transpose2 = *(resizeProducers.begin());
    NodePtr transpose3 = *(relu2Producers.begin());
    NodePtr transpose4 = *(relu3Producers.begin());
    NodePtr transpose5 = *(conv2Consumers.begin());
    NodePtr transpose6 = *(relu4Consumers.begin());
    ASSERT_TRUE(transpose1->isTranspose()) << "Missing a transpose node before node " << relu1->getNodeName();
    ASSERT_TRUE(transpose2->isTranspose()) << "Missing a transpose node before node " << resize->getNodeName();
    ASSERT_TRUE(transpose3->isTranspose()) << "Missing a transpose node before node " << relu2->getNodeName();
    ASSERT_TRUE(transpose4->isTranspose()) << "Missing a transpose node before node " << relu3->getNodeName();
    ASSERT_TRUE(transpose5->isTranspose()) << "Missing a transpose node after node " << conv2->getNodeName();
    ASSERT_TRUE(transpose6->isTranspose()) << "Missing a transpose node after node " << relu4->getNodeName();

    gc::Permutation testedPerm({2, 0, 1, 3});  // (WHCN)->(CWHN)
    TransposeNode*  transposeNode1 = dynamic_cast<TransposeNode*>(transpose1.get());
    TransposeNode*  transposeNode2 = dynamic_cast<TransposeNode*>(transpose2.get());
    TransposeNode*  transposeNode3 = dynamic_cast<TransposeNode*>(transpose3.get());
    TransposeNode*  transposeNode4 = dynamic_cast<TransposeNode*>(transpose4.get());
    TransposeNode*  transposeNode5 = dynamic_cast<TransposeNode*>(transpose5.get());
    TransposeNode*  transposeNode6 = dynamic_cast<TransposeNode*>(transpose6.get());
    ASSERT_TRUE(transposeNode1 && gc::Permutation(transposeNode1->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode1->getNodeName();
    ASSERT_TRUE(transposeNode2 && gc::Permutation(transposeNode2->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode2->getNodeName();
    ASSERT_TRUE(transposeNode3 && gc::Permutation(transposeNode3->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode3->getNodeName();
    ASSERT_TRUE(transposeNode4 && gc::Permutation(transposeNode4->permutation()) == testedPerm)
        << "Wrong permutation in transpose node " << transposeNode4->getNodeName();
    ASSERT_TRUE(transposeNode5 && gc::Permutation(transposeNode5->permutation()) == testedPerm.getInversePermutation())
        << "Wrong permutation in transpose node " << transposeNode5->getNodeName();
    ASSERT_TRUE(transposeNode6 && gc::Permutation(transposeNode6->permutation()) == testedPerm.getInversePermutation())
        << "Wrong permutation in transpose node " << transposeNode6->getNodeName();
}

TEST_P(GenericDataLayoutTest, wrap_graph_with_concat_dim0)
{
    /*
     * notes:
     * - node marked with a "*" is conflicted
     * - t is a transpose, and t' is its inverse transpose
     *
     *                relu     conv*
     *                  \     /
     *                  concat
     *
     * post graph is:
     *                 t       t  t
     *                 |       | /
     *                relu    conv*
     *                  \     /
     *                  concat
     *                     |
     *                     t'
     */

    const std::string reluGUIDString   = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID         = reluGUIDString.c_str();

    const TSize c          = 3;
    const TSize w          = 20;
    const TSize h          = 10;
    const TSize batch      = 1;
    const TSize inSizes[]  = {w, h, c, batch};
    const TSize outSizes[] = {2*w, h, c, batch}; // output of concat

    TensorPtr T1 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T1");
    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T2");
    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T3");
    TensorPtr T4 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T2->setName("T3");
    TensorPtr T5 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T5");
    TensorPtr T6 = std::make_shared<Tensor>(4U, outSizes, syn_type_single);
    T2->setName("T6");

    NodePtr relu1 = NodeFactory::createNode({T1}, {T2}, nullptr, reluGUID, "relu_1");
    GraphEditor::addNode(*m_graph, relu1);

    synConvolutionParams convParams;
    convParams.dH   = 1;
    convParams.dW   = 1;
    convParams.kH   = 1;
    convParams.kW   = 1;
    convParams.padT = 0;
    convParams.padB = 0;
    convParams.padL = 0;
    convParams.padR = 0;
    convParams.dilH = 1;
    convParams.dilW = 1;

    // define layouts as they appear in ONNX
    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("CSKR")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr conv1 = NodeFactory::createNode({T3, T4}, {T5}, &convParams, "spatial_convolution", "conv1", layouts);
    GraphEditor::addNode(*m_graph, conv1);

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    NodePtr concat1 = NodeFactory::createNode({T2, T5}, {T6}, &concatParams, "concat", "concat1");
    GraphEditor::addNode(*m_graph, concat1);

    ASSERT_TRUE(setSupportedLayouts(*m_graph)) << "failed to run setSupportedLayouts";
    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";
    ASSERT_TRUE(removeContiguousTransposes(*m_graph)) << "failed to run removeContiguousTransposes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    unsigned expectedTransposes = 4;
    ASSERT_EQ(transposeNodeCount, expectedTransposes) << "Unexpected number of transpose nodes found in graph: "
                                                      << transposeNodeCount << ". Should be " << expectedTransposes;
}

TEST_P(GenericDataLayoutTest, prevent_propagate_to_larger_data)
{
    /*
     * original graph is:
     * [WHCN tensor] -> cast_f32_to_bf16 -> resize
     *
     * the resize is defined with conflicting layouts, so during data layout adjustment, transposes should be inserted
     * before and after the resize:
     * [WHCN tensor] -> cast_f32_to_bf16 -> transpose ((WHCN)->(CWHN)) -> resize -> transpose ((CWHN)->(WHCN))
     *
     * the transpose_dont_care_nodes pass (which is where we are testing changes) shouldn't wrap the cast node cause
     * of preventPropagateToLargerData and the graph should remain as it is.
     */

    const std::string resizeGUIDString = getGUIDByDevice(GetParam(), "resize", "bf16");
    const char*       resizeGUID       = resizeGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    TensorPtr T1 = std::make_shared<Tensor>(4U, inSizes, syn_type_single);
    T1->setName("T1");

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, syn_type_bf16);
    T2->setName("T2");

    NodePtr cast = NodeFactory::createNode({T1}, {T2}, nullptr, "cast_f32_to_bf16", "cast");
    GraphEditor::addNode(*m_graph, cast);

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, syn_type_bf16);
    T3->setName("T3");

    ns_ResizeKernel::Params resizeParams;
    resizeParams.mode           = RESIZE_INTER_LINEAR;
    resizeParams.scaleDim1      = 1;
    resizeParams.scaleDim2      = 1;
    resizeParams.scaleDim3      = 1;
    resizeParams.useScales      = true;
    resizeParams.nearestMode    = ROUND_DEFAULT;
    resizeParams.coordTransMode = ASYMMETRIC_MODE;
    resizeParams.excludeOutside = false;
    // define layouts as they appear in ONNX
    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr resize        = NodeFactory::createNode({T2}, {T3}, &resizeParams, resizeGUID, "resize", layouts);
    GraphEditor::addNode(*m_graph, resize);

    ASSERT_TRUE(setSupportedLayouts(*m_graph)) << "failed to run setSupportedLayouts";
    ASSERT_TRUE(adjustDataLayout(*m_graph)) << "failed to run adjustDataLayout";
    ASSERT_TRUE(transposeDontCareNodes(*m_graph)) << "failed to run transposeDontCareNodes";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 2) << transposeNodeCount << " unexpected transpose nodes found in graph";

    auto node = (*m_graph).getExeSortedNodes().begin();
    ASSERT_TRUE((*node)->isCast()) << "First node is expected to be the case node";
    std::advance(node, 1);
    ASSERT_TRUE((*node)->isTranspose()) << "Second node is expected to be a transpose node";
}

INSTANTIATE_TEST_SUITE_P(,
                         GenericDataLayoutTest,
                         ::testing::Values(synDeviceGaudi, synDeviceGaudi2),
                         GenericGraphTest::GetName());

class GenericDataLayoutReshapeTest
: public GraphOptimizerTest
, public testing::WithParamInterface<::testing::tuple<synDeviceType, int /* reshaped dimension */>>
{
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        synDeviceType deviceType = std::get<0>(GetParam());
        m_graph                  = GraphFactory::createGraph(deviceType, CompilationMode::Graph);
    }

    void TearDown() override
    {
        m_graph.reset();
        GraphOptimizerTest::TearDown();
    }

protected:
    std::unique_ptr<HabanaGraph> m_graph;

public:
    struct GetName
    {
        std::string operator()(const ::testing::TestParamInfo<::testing::tuple<synDeviceType, int>>& info) const
        {
            ::std::stringstream ss;
            ss << "_" << deviceTypeToString(::testing::get<0>(info.param)) << "_"
               << "reshaped_dim_" << ::testing::get<1>(info.param);
            return ss.str();
        }
    };
};

TEST_P(GenericDataLayoutReshapeTest, wrap_reshape_with_transposes)
{
    /*
     * Below is the flow for the squeeze node case (reshaped dimension is 3)
     *
     * original graph is:
     * transpose ((CWHN)->(WHCN)) -> resize_f32 -> squeeze -> transpose ((WHC)->(CWH))
     *
     * the resize is defined with conflicting layouts, so during data layout adjustment, transposes should be inserted
     * before and after the resize:
     * transpose ((CWHN)->(WHCN)) -> transpose ((WHCN)->(CWHN)) -> resize -> transpose ((CWHN)->(WHCN))
     *    -> squeeze -> transpose ((WHC)->(CWH))
     *
     * then, the transpose_dont_care_nodes pass (which is where we are testing changes) should wrap the squeeze with
     * appropriate transposes:
     * transpose ((CWHN)->(WHCN)) -> transpose ((WHCN)->(CWHN)) -> resize -> transpose ((CWHN)->(WHCN))
     *    -> transpose ((WHCN)->(CWHN)) -> squeeze -> transpose ((CWH)->(WHC)) -> transpose ((WHC)->(CWH))
     *
     * eventually, the remove_contiguous_transposes pass should get rid of all the transposes in the graph, since
     * three pairs of identity transposes were created
     */
    const std::string resizeGUIDString = getGUIDByDevice(std::get<0>(GetParam()), "resize");
    const char*       resizeGUID       = resizeGUIDString.c_str();

    const TSize c              = 3;
    const TSize w              = 20;
    const TSize h              = 10;
    const TSize batch          = 1;
    const TSize maxReshapedDim = 5;

    unsigned reshapedDim = std::get<1>(GetParam());
    HB_ASSERT(reshapedDim >= 3 && reshapedDim <= maxReshapedDim, "illegal reshaped dim");

    float input[c * w * h * batch];

    const TSize inSizes[] = {c, w, h, batch};
    TensorPtr   T1        = std::make_shared<Tensor>(4U, inSizes, syn_type_single, reinterpret_cast<char*>(input));
    T1->setName("T1");
    setTensorAsPersistent(T1);  // graph input must be declared as persistent

    const TSize resizeSizes[] = {w, h, c, batch};
    TensorPtr   T2            = std::make_shared<Tensor>(4U, resizeSizes, syn_type_single);
    T2->setName("T2");

    synTransposeParams transposeParams;
    transposeParams.tensorDim = T1->getDim();
    // perm = (1, 2, 0, 3) = (CWHN)->(WHCN)
    transposeParams.permutation[0] = TPD_Width;
    transposeParams.permutation[1] = TPD_Height;
    transposeParams.permutation[2] = TPD_Channel;
    transposeParams.permutation[3] = TPD_4Dim_Batch;
    NodePtr trans0                 = NodeFactory::createNode({T1}, {T2}, &transposeParams, "transpose", "transpose_0");
    GraphEditor::addNode(*m_graph, trans0);

    TensorPtr T3 = std::make_shared<Tensor>(4U, resizeSizes, syn_type_single);
    T3->setName("T3");

    ns_ResizeKernel::Params resizeParams;
    resizeParams.mode           = RESIZE_INTER_LINEAR;
    resizeParams.scaleDim1      = 1;
    resizeParams.scaleDim2      = 1;
    resizeParams.scaleDim3      = 1;
    resizeParams.useScales      = true;
    resizeParams.nearestMode    = ROUND_DEFAULT;
    resizeParams.coordTransMode = ASYMMETRIC_MODE;
    resizeParams.excludeOutside = false;
    // define layouts as they appear in ONNX
    Node::NodeProperties layouts;
    layouts.inputLayouts  = {gc::Layout("WHCN")};
    layouts.outputLayouts = {gc::Layout("WHCN")};
    NodePtr resize        = NodeFactory::createNode({T2}, {T3}, &resizeParams, resizeGUID, "resize", layouts);
    GraphEditor::addNode(*m_graph, resize);

    TSize squeezedSizes[] = {w, h, c, batch, 1};
    TSize transSizes[]    = {c, w, h, batch, 1};
    for (unsigned i = reshapedDim; i < maxReshapedDim; i++)
    {
        squeezedSizes[reshapedDim - 1] *= squeezedSizes[i];
        transSizes[reshapedDim - 1] *= transSizes[i];
    }
    TensorPtr T4 = std::make_shared<Tensor>(reshapedDim, squeezedSizes, syn_type_single);
    T4->setName("T4");

    NodePtr reshape = NodeFactory::createNode({T3}, {T4}, nullptr, "reshape", "reshape");
    GraphEditor::addNode(*m_graph, reshape);

    TensorPtr T5 = std::make_shared<Tensor>(reshapedDim, transSizes, syn_type_single);
    T5->setName("T5");
    setTensorAsPersistent(T5, 1);  // graph output must be declared as persistent

    synTransposeParams transposeParams1;
    transposeParams1.tensorDim = T4->getDim();
    // perm = (2, 0, 1) = (WHC)->(CWH)
    transposeParams1.permutation[0] = TPD_Height;
    transposeParams1.permutation[1] = TPD_Channel;
    transposeParams1.permutation[2] = TPD_Width;
    transposeParams1.permutation[3] = TPD_4Dim_Batch;
    transposeParams1.permutation[4] = TPD_Batch;
    NodePtr trans1 = NodeFactory::createNode({T4}, {T5}, &transposeParams1, "transpose", "transpose_1");
    GraphEditor::addNode(*m_graph, trans1);

    ASSERT_TRUE((*m_graph).compile()) << "Failed to compile graph";

    unsigned int transposeNodeCount = 0;
    for (const auto& node : (*m_graph).getExeSortedNodes())
    {
        if (node != nullptr && node->isTranspose())
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 0) << transposeNodeCount << " unexpected transpose nodes found in graph";
}

INSTANTIATE_TEST_SUITE_P(,
                         GenericDataLayoutReshapeTest,
                         ::testing::Values(::testing::make_tuple(synDeviceGaudi, 3),
                                           ::testing::make_tuple(synDeviceGaudi, 4),
                                           ::testing::make_tuple(synDeviceGaudi2, 3),
                                           ::testing::make_tuple(synDeviceGaudi2, 4)
                                           /*, 5  expand dims - add tests as part of [SW-88064] */),
                         GenericDataLayoutReshapeTest::GetName());
