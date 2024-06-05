#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "tpc_node.h"
#include "habana_global_conf.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "graph_compiler/passes/optimize_tpc_kernels.h"

// the only goal of this class is to be able to call the protected function instantiate() of the TPC node
class TestableTPCNode : public TPCNode
{
    static constexpr const SizeArray s_c_default_permutation = {0, 1, 2, 3, 4};  // = no permutation
public:
    TestableTPCNode(const TensorVector&              inputs,
                    const TensorVector&              outputs,
                    const std::string&               name,
                    UserParams                       params,
                    std::string_view                 GUID,
                    SizeArray                        permutation = s_c_default_permutation,
                    tpc_lib_api::TensorOperationType operation   = tpc_lib_api::TENSOR_OP_TRANSPOSE)
    : TPCNode(inputs, outputs, name, params), m_operation(operation), m_permutation(permutation)
    {
        setGUID(GUID);
    }

    virtual ~TestableTPCNode() {}

    tpc_lib_api::GlueCodeReturn instantiate(KernelInstantiationWrapper& instance)
    {
        instance.initParams(*this, tpc_lib_api::DEVICE_ID_GAUDI);
        return TPCNode::instantiate(instance);
    }

    tpc_lib_api::GlueCodeReturn
    getSuggestedTensorManipulation(tpc_lib_api::TensorManipulationSuggestion* suggestion) override
    {
        tpc_lib_api::GlueCodeReturn originalRes = TPCNode::getSuggestedTensorManipulation(suggestion);
        if (m_permutation != s_c_default_permutation)
        {
            if (m_operation == tpc_lib_api::TENSOR_OP_TRANSPOSE)
            {
                // manipulate suggestion (because currently the tpc glue code doesn't return transpose suggestion)
                suggestion->inputTensors[0].opType  = tpc_lib_api::TENSOR_OP_TRANSPOSE;
                suggestion->outputTensors[0].opType = tpc_lib_api::TENSOR_OP_TRANSPOSE;
                for (int i = 0; i < MAX_DIMENSIONS_NUM; ++i)
                {
                    suggestion->inputTensors[0].permutation[i] = static_cast<TransposePermutationDim>(m_permutation[i]);
                    suggestion->outputTensors[0].permutation[i] =
                        static_cast<TransposePermutationDim>(m_permutation[i]);
                }
            }
            else if (m_operation == tpc_lib_api::TENSOR_OP_RESHAPE)
            {
                suggestion->inputTensors[0].opType  = tpc_lib_api::TENSOR_OP_RESHAPE;
                suggestion->outputTensors[0].opType = tpc_lib_api::TENSOR_OP_RESHAPE;
                for (int i = 0; i < MAX_DIMENSIONS_NUM; ++i)
                {
                    suggestion->inputTensors[0].permutation[i] = static_cast<TransposePermutationDim>(m_permutation[i]);
                    suggestion->inputTensors[0].maxNewShape[i] = static_cast<TransposePermutationDim>(m_permutation[i]);
                    suggestion->outputTensors[0].permutation[i] =
                        static_cast<TransposePermutationDim>(m_permutation[i]);
                    suggestion->outputTensors[0].maxNewShape[i] =
                        static_cast<TransposePermutationDim>(m_permutation[i]);
                }
                return tpc_lib_api::GLUE_SUCCESS;
            }
            else
            {
                return tpc_lib_api::GLUE_FAILED;
            }
        }
        return originalRes;
    }

protected:
    tpc_lib_api::TensorOperationType m_operation;
    SizeArray                        m_permutation;
};

constexpr const SizeArray TestableTPCNode::s_c_default_permutation;

class OptimizeTPCKernelsTest : public GraphOptimizerTest
{
    void SetUp()
    {
        setGlobalConfForTest(GCFG_COMPLEX_GUID_EXTRACTOR_MODE, "0");
        setGlobalConfForTest(GCFG_ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION, "0");
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
        GraphOptimizerTest::SetUp();
    }

    virtual void TearDown() { GraphOptimizerTest::TearDown(); }
};

TEST_F(OptimizeTPCKernelsTest, inverse_suggestion_permutation_transpose_test)
{
    GaudiGraph g;

    const TSize d0      = 49;
    const TSize d1      = 49;
    const TSize d2      = 3;
    const TSize d3      = 12800;
    TSize       sizes[] = {d0, d1, d2, d3};

    unsigned int dim = ARRAY_SIZE(sizes);

    SizeArray permutation = {3, 0, 1, 2, 4};  // this mimics transpose suggestion which it's inversed permutation is not
                                              // identical to the permutation itself

    pTensor                          tSoftmaxIn(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor                          tSoftmaxOut(new Tensor(dim, sizes, syn_type_float));
    const TensorVector&              inputs {tSoftmaxIn};
    const TensorVector&              outputs {tSoftmaxOut};
    unsigned char                    params[] = {0, 0, 0, 0};
    std::shared_ptr<TestableTPCNode> nodeSoftmaxFwd =
        std::make_shared<TestableTPCNode>(inputs,
                                          outputs,
                                          "softmax_fwd_f32",
                                          params,
                                          "softmax_fwd_f32",
                                          permutation,
                                          tpc_lib_api::TENSOR_OP_TRANSPOSE);
    GraphEditor::addNode(g, nodeSoftmaxFwd);
    synMemoryDescriptor softamx_fwd_memDesc(true);
    tSoftmaxIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tSoftmaxIn->setMemoryDescriptor(softamx_fwd_memDesc);

    tSoftmaxOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tSoftmaxOut->setMemoryDescriptor(softamx_fwd_memDesc);

    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";

    const auto& exeSortedNodes = g.getExeSortedNodes();
    EXPECT_GE(exeSortedNodes.size(), 3)
        << "Expected the graph to consist of at least 3 nodes: at least one transpose nodes should be added both "
           "before and after the softmax node. But found only "
        << exeSortedNodes.size();

    auto isDmaTranspose = [](NodePtr node) { return node->getGUID() == NodeFactory::transposeDmaNodeTypeName; };

    unsigned numDmaTranspose = std::count_if(exeSortedNodes.begin(), exeSortedNodes.end(), isDmaTranspose);

    EXPECT_GE(numDmaTranspose, 2) << "Expected at least 2 dma transposes, but found only " << numDmaTranspose;
}

// This tests is similar to SynGaudiTestInfra.relu_forward_and_backward, but its goal is to test the
// OptimizeTPCKernels pass in Gaudi.
TEST_F(OptimizeTPCKernelsTest, DISABLED_relu_forward_and_backward)
{
    GaudiGraph   g;
    TSize        sizes[] = {20, 16, 16, 16};
    unsigned int dim     = ARRAY_SIZE(sizes);

    pTensor                          tFwdIn(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor                          tFwdOut(new Tensor(dim, sizes, syn_type_float));
    const TensorVector&              inputs {tFwdIn};
    const TensorVector&              outputs {tFwdOut};
    std::shared_ptr<TestableTPCNode> nodeReluFwd =
        std::make_shared<TestableTPCNode>(inputs, outputs, "node_relu_fwd", nullptr, "relu_fwd_f32");
    GraphEditor::addNode(g, nodeReluFwd);

    pTensor tBwdIn(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor tBwdOut(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, true));
    pNode   nodeReluBwd =
        NodeFactory::createGenericTPCNode({tBwdIn, tFwdOut}, {tBwdOut}, nullptr, "relu_bwd_f32", "node_relu_bwd");
    GraphEditor::addNode(g, nodeReluBwd);

    // Set graph's input tensor as persistent
    synMemoryDescriptor relu_fwd_memDesc(true);
    tFwdIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tFwdIn->setMemoryDescriptor(relu_fwd_memDesc);
    synMemoryDescriptor relu_bwd_memDesc(true);
    tBwdIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tBwdIn->setMemoryDescriptor(relu_bwd_memDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor transpose_memDesc(true);
    tBwdOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    tBwdOut->setMemoryDescriptor(transpose_memDesc);

    // VALIDATIONS:
    ASSERT_NE(nodeReluFwd, nullptr);
    KernelInstantiationWrapper instance;
    auto                       res = nodeReluFwd->instantiate(instance);
    ASSERT_EQ(res, tpc_lib_api::GLUE_SUCCESS) << "failed to instantiate tpc node";
    tpc_lib_api::TensorManipulationSuggestion suggestion;
    res = KernelDB::instance().GetSuggestedTensorManipulation(&instance.getGlueParams(),
                                                              &suggestion,
                                                              nodeReluFwd->getGUIDAndHash());
    ASSERT_EQ(res, tpc_lib_api::GLUE_SUCCESS) << "failed to get glue code suggestion";

    // we expected the suggestion to suggest "reshape"
    ASSERT_EQ(suggestion.inputTensors[0].opType, tpc_lib_api::TENSOR_OP_RESHAPE);
    ASSERT_GT(suggestion.inputTensors[0].maxNewShape[0], sizes[0]);  // we expect the suggestion to make C bigger
    ASSERT_LT(suggestion.inputTensors[0].maxNewShape[1], sizes[1]);  // in exchange for making W smaller
    // same for suggestion outputs
    ASSERT_EQ(suggestion.outputTensors[0].opType, tpc_lib_api::TENSOR_OP_RESHAPE);
    ASSERT_GT(suggestion.outputTensors[0].maxNewShape[0], sizes[0]);
    ASSERT_LT(suggestion.outputTensors[0].maxNewShape[1], sizes[1]);

    // Compile graph, then validate the RELU forward node inputs/outputs are changed according to the suggestion,
    // and that the relevant reshape/packing nodes are matching this change
    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";
    const NodeVector& nodes = g.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        ASSERT_TRUE(node->getNodeType() == Node::TYPE_USER || node->getNodeType() == Node::TYPE_STATIC_RESHAPE)
            << "Found an unexpected node in graph";

        ASSERT_NE(node->getInput(0), nullptr);
        ASSERT_NE(node->getOutput(0), nullptr);
        SizeArray inTensorSizes  = node->getInput(0)->getAllSizesInElements();
        SizeArray outTensorSizes = node->getOutput(0)->getAllSizesInElements();
        // all tensors in this graph should have the same dimensions
        ASSERT_EQ(node->getInput(0)->getDim(), dim);
        ASSERT_EQ(node->getOutput(0)->getDim(), dim);

        if (node->getNodeType() == Node::Node::TYPE_USER)
        {
            std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
            ASSERT_NE(tpcNode, nullptr);
            if (tpcNode->getGUID() == "relu_fwd_f32")  // this is the node we got the suggestion for
            {                                          // but now it should have a new shape
                ASSERT_EQ(0,
                          memcmp(inTensorSizes.data(),
                                 suggestion.inputTensors[0].maxNewShape,
                                 dim * sizeof(inTensorSizes[0])));
                ASSERT_EQ(0,
                          memcmp(outTensorSizes.data(),
                                 suggestion.outputTensors[0].maxNewShape,
                                 dim * sizeof(outTensorSizes[0])));
            }
        }
        else if (node->getNodeType() == Node::TYPE_STATIC_RESHAPE)
        {
            if (node->getNodeName().find("_in") != std::string::npos)
            {
                ASSERT_EQ(0, memcmp(inTensorSizes.data(), sizes, dim * sizeof(inTensorSizes[0])));
                ASSERT_EQ(0,
                          memcmp(outTensorSizes.data(),
                                 suggestion.inputTensors[0].maxNewShape,
                                 dim * sizeof(outTensorSizes[0])));
            }
            else  // pack out
            {
                ASSERT_EQ(0,
                          memcmp(inTensorSizes.data(),
                                 suggestion.outputTensors[0].maxNewShape,
                                 dim * sizeof(inTensorSizes[0])));
                ASSERT_EQ(0, memcmp(outTensorSizes.data(), sizes, dim * sizeof(outTensorSizes[0])));
            }
        }
    }
}

TEST_F(OptimizeTPCKernelsTest, softmax_transpose)
{
    GaudiGraph   g;
    TSize        sizes[]                   = {128, 128 * 2, 12, 32};
    unsigned int dim                       = ARRAY_SIZE(sizes);
    TSize        expectedTransposedSizes[] = {128 * 2, 128, 12, 32};
    // we need to mimic the transpose suggestion (the tpc currently doesn't support it):
    SizeArray permutation = {1, 0, 2, 3, 4};  // this mimics physical transpose (because the FCD is changed)

    pTensor                          tSoftmaxIn(new Tensor(dim, sizes, syn_type_float, nullptr, nullptr, false, true));
    pTensor                          tSoftmaxOut(new Tensor(dim, sizes, syn_type_float));
    const TensorVector&              inputs {tSoftmaxIn};
    const TensorVector&              outputs {tSoftmaxOut};
    unsigned char                    params[] = {0, 0, 0, 0};
    std::shared_ptr<TestableTPCNode> nodeSoftmaxFwd =
        std::make_shared<TestableTPCNode>(inputs, outputs, "softmax_fwd_f32", params, "softmax_fwd_f32", permutation);
    GraphEditor::addNode(g, nodeSoftmaxFwd);
    synMemoryDescriptor softamx_fwd_memDesc(true);
    tSoftmaxIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tSoftmaxIn->setMemoryDescriptor(softamx_fwd_memDesc);

    tSoftmaxOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tSoftmaxOut->setMemoryDescriptor(softamx_fwd_memDesc);

    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";
    const NodeVector& nodes = g.getExeSortedNodes();
    // since the softmax suggestion is to transpose input and output, expect the following node order:
    // dma->softmax->dma
    std::vector<Node::eNodeType> expectedNodeTypes   = {Node::TYPE_DMA, Node::TYPE_USER, Node::TYPE_DMA};
    auto                         expectedNodeTypeIdx = 0;
    ASSERT_EQ(nodes.size(), expectedNodeTypes.size());
    for (const NodePtr& node : nodes)
    {
        ASSERT_EQ(node->getNodeType(), expectedNodeTypes[expectedNodeTypeIdx]) << expectedNodeTypeIdx;
        expectedNodeTypeIdx++;
        if (node->getNodeType() == Node::TYPE_USER)
        {
            std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
            ASSERT_NE(tpcNode, nullptr);
            ASSERT_EQ(tpcNode->getGUID(), "softmax_fwd_f32");
            // check that now softmax inputs and outputs are transposed
            ASSERT_EQ(node->getInput(0)->getDim(), dim);
            ASSERT_EQ(node->getOutput(0)->getDim(), dim);
            SizeArray inTensorSizes  = node->getInput(0)->getAllSizesInElements();
            SizeArray outTensorSizes = node->getOutput(0)->getAllSizesInElements();
            ASSERT_EQ(0, memcmp(inTensorSizes.data(), expectedTransposedSizes, dim * sizeof(inTensorSizes[0])));
            ASSERT_EQ(0, memcmp(outTensorSizes.data(), expectedTransposedSizes, dim * sizeof(outTensorSizes[0])));
        }
    }
}

TEST_F(OptimizeTPCKernelsTest, relu_static)
{
    GaudiGraph g;
    TSize      sizes[]         = {64, 256, 4, 8};
    TSize      dim             = ARRAY_SIZE(sizes);
    TSize      expectedSizes[] = {128, 128, 4, 8};

    // we need to mimic the transpose suggestion (the tpc currently doesn't support it):
    SizeArray permutation = {expectedSizes[0], expectedSizes[1], expectedSizes[2], expectedSizes[3], 1};

    pTensor                          reluIn(new Tensor(dim, sizes, syn_type_float));
    pTensor                          reluOut(new Tensor(dim, sizes, syn_type_float));
    const TensorVector&              inputs {reluIn};
    const TensorVector&              outputs {reluOut};
    std::shared_ptr<TestableTPCNode> reluNode = std::make_shared<TestableTPCNode>(inputs,
                                                                                  outputs,
                                                                                  "Relu",
                                                                                  nullptr,
                                                                                  "relu_fwd_f32",
                                                                                  permutation,
                                                                                  tpc_lib_api::TENSOR_OP_RESHAPE);
    GraphEditor::addNode(g, reluNode);

    // Set graph's input tensors as persistent
    synMemoryDescriptor reluInMemDesc(true);
    reluIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    reluIn->setMemoryDescriptor(reluInMemDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor reluOutMemDesc(true);
    reluOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    reluOut->setMemoryDescriptor(reluOutMemDesc);

    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";
    const NodeVector& nodes = g.getExeSortedNodes();

    // expected packing -> relu -> unpack
    ASSERT_EQ(nodes.size(), 3);

    ASSERT_EQ(nodes[0]->getNodeType(), Node::TYPE_STATIC_RESHAPE);

    ASSERT_EQ(nodes[1]->getNodeType(), Node::TYPE_USER);
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(nodes[1]);
    ASSERT_NE(tpcNode, nullptr);
    ASSERT_TRUE(tpcNode->getGUID() == "relu_fwd_f32");

    ASSERT_EQ(tpcNode->getInput(0)->getDim(), dim);
    ASSERT_EQ(tpcNode->getOutput(0)->getDim(), dim);
    SizeArray inTensorSizes  = tpcNode->getInput(0)->getAllSizesInElements();
    SizeArray outTensorSizes = tpcNode->getOutput(0)->getAllSizesInElements();
    ASSERT_EQ(0, memcmp(inTensorSizes.data(), expectedSizes, dim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outTensorSizes.data(), expectedSizes, dim * sizeof(TSize)));
    SizeArray inMinTensorSizes  = tpcNode->getInput(0)->getAllMinimalSizesInElements();
    SizeArray outMinTensorSizes = tpcNode->getOutput(0)->getAllMinimalSizesInElements();
    ASSERT_EQ(0, memcmp(inMinTensorSizes.data(), expectedSizes, dim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outMinTensorSizes.data(), expectedSizes, dim * sizeof(TSize)));

    ASSERT_EQ(nodes[2]->getNodeType(), Node::TYPE_STATIC_RESHAPE);
}

TEST_F(OptimizeTPCKernelsTest, relu_dynamic_batch_same_dims)
{
    GaudiGraph   g;
    TSize        sizes[]            = {64, 256, 4, 8};
    TSize        minSizes[]         = {64, 256, 4, 2};
    unsigned int dim                = ARRAY_SIZE(sizes);
    TSize        expectedMaxSizes[] = {128, 128, 4, 8};
    TSize        expectedMinSizes[] = {128, 128, 4, 2};

    // we need to mimic the transpose suggestion (the tpc currently doesn't support it):
    SizeArray permutation = {expectedMaxSizes[0], expectedMaxSizes[1],
                             expectedMaxSizes[2], expectedMaxSizes[3], 1};

    TensorPtr                        reluIn  = std::make_shared<Tensor>(dim, sizes, syn_type_float, minSizes);
    TensorPtr                        reluOut = std::make_shared<Tensor>(dim, sizes, syn_type_float, minSizes);
    const TensorVector&              inputs {reluIn};
    const TensorVector&              outputs {reluOut};
    std::shared_ptr<TestableTPCNode> reluNode = std::make_shared<TestableTPCNode>(inputs,
                                                                                  outputs,
                                                                                  "Relu",
                                                                                  nullptr,
                                                                                  "relu_fwd_f32",
                                                                                  permutation,
                                                                                  tpc_lib_api::TENSOR_OP_RESHAPE);
    GraphEditor::addNode(g, reluNode);

    // Set graph's input tensors as persistent
    synMemoryDescriptor reluInMemDesc(true);
    reluIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    reluIn->setMemoryDescriptor(reluInMemDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor reluOutMemDesc(true);
    reluOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    reluOut->setMemoryDescriptor(reluOutMemDesc);

    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";
    const NodeVector& nodes = g.getExeSortedNodes();

    // expected deserialize -> packing -> relu -> unpack -> serialize
    ASSERT_EQ(nodes.size(), 5);
    ASSERT_EQ(nodes[0]->getNodeType(), Node::TYPE_DMA);
    ASSERT_EQ(nodes[1]->getNodeType(), Node::TYPE_STATIC_RESHAPE);

    ASSERT_EQ(nodes[2]->getNodeType(), Node::TYPE_USER);
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(nodes[2]);
    ASSERT_NE(tpcNode, nullptr);
    ASSERT_TRUE(tpcNode->getGUID() == "relu_fwd_f32");

    ASSERT_EQ(tpcNode->getInput(0)->getDim(), dim);
    ASSERT_EQ(tpcNode->getOutput(0)->getDim(), dim);
    SizeArray inTensorSizes  = tpcNode->getInput(0)->getAllSizesInElements();
    SizeArray outTensorSizes = tpcNode->getOutput(0)->getAllSizesInElements();
    ASSERT_EQ(0, memcmp(inTensorSizes.data(), expectedMaxSizes, dim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outTensorSizes.data(), expectedMaxSizes, dim * sizeof(TSize)));
    SizeArray inMinTensorSizes  = tpcNode->getInput(0)->getAllMinimalSizesInElements();
    SizeArray outMinTensorSizes = tpcNode->getOutput(0)->getAllMinimalSizesInElements();
    ASSERT_EQ(0, memcmp(inMinTensorSizes.data(), expectedMinSizes, dim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outMinTensorSizes.data(), expectedMinSizes, dim * sizeof(TSize)));

    ASSERT_EQ(nodes[3]->getNodeType(), Node::TYPE_STATIC_RESHAPE);
    ASSERT_EQ(nodes[4]->getNodeType(), Node::TYPE_DMA);
}

TEST_F(OptimizeTPCKernelsTest, relu_dynamic_batch_less_dims)
{
    GaudiGraph   g;
    TSize        sizes[]            = {64, 256, 4, 8};
    TSize        minSizes[]         = {64, 256, 4, 2};
    TSize        expectedMaxSizes[] = {256, 256, 1, 8};
    TSize        expectedMinSizes[] = {256, 256, 1, 2};
    unsigned int inDim              = ARRAY_SIZE(sizes);
    unsigned int outDim             = ARRAY_SIZE(expectedMaxSizes);

    // we need to mimic the transpose suggestion (the tpc currently doesn't support it):
    SizeArray permutation = {expectedMaxSizes[0], expectedMaxSizes[1],
                             expectedMaxSizes[2], expectedMaxSizes[3], 1};

    TensorPtr                        reluIn  = std::make_shared<Tensor>(inDim, sizes, syn_type_float, minSizes);
    TensorPtr                        reluOut = std::make_shared<Tensor>(inDim, sizes, syn_type_float, minSizes);
    const TensorVector&              inputs {reluIn};
    const TensorVector&              outputs {reluOut};
    std::shared_ptr<TestableTPCNode> reluNode = std::make_shared<TestableTPCNode>(inputs,
                                                                                  outputs,
                                                                                  "Relu",
                                                                                  nullptr,
                                                                                  "relu_fwd_f32",
                                                                                  permutation,
                                                                                  tpc_lib_api::TENSOR_OP_RESHAPE);
    GraphEditor::addNode(g, reluNode);

    // Set graph's input tensors as persistent
    synMemoryDescriptor reluInMemDesc(true);
    reluIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    reluIn->setMemoryDescriptor(reluInMemDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor reluOutMemDesc(true);
    reluOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    reluOut->setMemoryDescriptor(reluOutMemDesc);

    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";
    const NodeVector& nodes = g.getExeSortedNodes();

    // expected deserialize -> packing -> relu -> unpack -> serialize
    ASSERT_EQ(nodes.size(), 5);

    ASSERT_EQ(nodes[0]->getNodeType(), Node::TYPE_DMA);
    ASSERT_EQ(nodes[1]->getNodeType(), Node::TYPE_STATIC_RESHAPE);

    ASSERT_EQ(nodes[2]->getNodeType(), Node::TYPE_USER);
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(nodes[2]);
    ASSERT_NE(tpcNode, nullptr);
    ASSERT_TRUE(tpcNode->getGUID() == "relu_fwd_f32");

    ASSERT_EQ(tpcNode->getInput(0)->getDim(), outDim);
    ASSERT_EQ(tpcNode->getOutput(0)->getDim(), outDim);
    SizeArray inTensorSizes  = tpcNode->getInput(0)->getAllSizesInElements();
    SizeArray outTensorSizes = tpcNode->getOutput(0)->getAllSizesInElements();
    ASSERT_EQ(0, memcmp(inTensorSizes.data(), expectedMaxSizes, outDim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outTensorSizes.data(), expectedMaxSizes, outDim * sizeof(TSize)));
    SizeArray inMinTensorSizes  = tpcNode->getInput(0)->getAllMinimalSizesInElements();
    SizeArray outMinTensorSizes = tpcNode->getOutput(0)->getAllMinimalSizesInElements();
    ASSERT_EQ(0, memcmp(inMinTensorSizes.data(), expectedMinSizes, outDim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outMinTensorSizes.data(), expectedMinSizes, outDim * sizeof(TSize)));

    ASSERT_EQ(nodes[3]->getNodeType(), Node::TYPE_STATIC_RESHAPE);
    ASSERT_EQ(nodes[4]->getNodeType(), Node::TYPE_DMA);
}

TEST_F(OptimizeTPCKernelsTest, relu_multiple_dynamic_dims)
{
    GaudiGraph   g;
    TSize        sizes[]            = {64, 64, 4, 8};
    TSize        minSizes[]         = {64, 64, 1, 2};
    TSize        expectedMaxSizes[] = {4096, 1, 4, 8};
    TSize        expectedMinSizes[] = {4096, 1, 1, 2};
    unsigned int inDim              = ARRAY_SIZE(sizes);
    unsigned int outDim             = ARRAY_SIZE(expectedMaxSizes);

    // we need to mimic the transpose suggestion (the tpc currently doesn't support it):
    SizeArray permutation = {expectedMaxSizes[0], expectedMaxSizes[1],
                             expectedMaxSizes[2], expectedMaxSizes[3], 1};

    TensorPtr                        reluIn  = std::make_shared<Tensor>(inDim, sizes, syn_type_float, minSizes);
    TensorPtr                        reluOut = std::make_shared<Tensor>(inDim, sizes, syn_type_float, minSizes);
    const TensorVector&              inputs {reluIn};
    const TensorVector&              outputs {reluOut};
    std::shared_ptr<TestableTPCNode> reluNode = std::make_shared<TestableTPCNode>(inputs,
                                                                                  outputs,
                                                                                  "Relu",
                                                                                  nullptr,
                                                                                  "relu_fwd_f32",
                                                                                  permutation,
                                                                                  tpc_lib_api::TENSOR_OP_RESHAPE);
    GraphEditor::addNode(g, reluNode);

    // Set graph's input tensors as persistent
    synMemoryDescriptor reluInMemDesc(true);
    reluIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    reluIn->setMemoryDescriptor(reluInMemDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor reluOutMemDesc(true);
    reluOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    reluOut->setMemoryDescriptor(reluOutMemDesc);

    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";
    const NodeVector& nodes = g.getExeSortedNodes();

    // expected deserialize -> packing -> relu -> unpack -> serialize
    ASSERT_EQ(nodes.size(), 5);

    ASSERT_EQ(nodes[0]->getNodeType(), Node::TYPE_DMA);
    ASSERT_EQ(nodes[1]->getNodeType(), Node::TYPE_STATIC_RESHAPE);

    ASSERT_EQ(nodes[2]->getNodeType(), Node::TYPE_USER);
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(nodes[2]);
    ASSERT_NE(tpcNode, nullptr);
    ASSERT_TRUE(tpcNode->getGUID() == "relu_fwd_f32");

    ASSERT_EQ(tpcNode->getInput(0)->getDim(), outDim);
    ASSERT_EQ(tpcNode->getOutput(0)->getDim(), outDim);
    SizeArray inTensorSizes  = tpcNode->getInput(0)->getAllSizesInElements();
    SizeArray outTensorSizes = tpcNode->getOutput(0)->getAllSizesInElements();
    ASSERT_EQ(0, memcmp(inTensorSizes.data(), expectedMaxSizes, outDim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outTensorSizes.data(), expectedMaxSizes, outDim * sizeof(TSize)));
    SizeArray inMinTensorSizes  = tpcNode->getInput(0)->getAllMinimalSizesInElements();
    SizeArray outMinTensorSizes = tpcNode->getOutput(0)->getAllMinimalSizesInElements();
    ASSERT_EQ(0, memcmp(inMinTensorSizes.data(), expectedMinSizes, outDim * sizeof(TSize)));
    ASSERT_EQ(0, memcmp(outMinTensorSizes.data(), expectedMinSizes, outDim * sizeof(TSize)));

    ASSERT_EQ(nodes[3]->getNodeType(), Node::TYPE_STATIC_RESHAPE);
    ASSERT_EQ(nodes[4]->getNodeType(), Node::TYPE_DMA);
}

TEST_F(OptimizeTPCKernelsTest, mult_reshape_tile_suggestion)
{
    /*************
     * Verifying Tile suggestion for slow mult: [64,300,300,3] X [1,1,1,3]
     * Should eventually become: [64,300,15,60] X [1,1,1,60]
     * Current suggestion is [64,300,5,180] X [1,1,1,180]
     *************/

    GaudiGraph   g;
    TSize        inputA_sizes[] = {3, 300, 300, 64};
    TSize        inputB_sizes[] = {3, 1, 1, 1};
    unsigned int inDimA         = ARRAY_SIZE(inputA_sizes);
    unsigned int inDimB         = ARRAY_SIZE(inputB_sizes);

    TensorPtr multInA(new Tensor(inDimA, inputA_sizes, syn_type_float));
    TensorPtr multInB(new Tensor(inDimB, inputB_sizes, syn_type_float));
    TensorPtr multOut(new Tensor(inDimA, inputA_sizes, syn_type_float));

    const TensorVector& inputs {multInA, multInB};
    const TensorVector& outputs {multOut};
    // In case we would like to use optimizeTpcKernels test utils of instantiation.
    // currently features of TestableTPCNode are not in use as suggestion only appears upon graph compilation.
    std::shared_ptr<TestableTPCNode> multNode =
        std::make_shared<TestableTPCNode>(inputs, outputs, "Mult", nullptr, "mult_fwd_f32");
    GraphEditor::addNode(g, multNode);

    // Set graph's input tensors as persistent
    synMemoryDescriptor multInAMemDesc(true);
    multInA->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    multInA->setMemoryDescriptor(multInAMemDesc);
    synMemoryDescriptor multInBMemDesc(true);
    multInB->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    multInB->setMemoryDescriptor(multInBMemDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor multOutMemDesc(true);
    multOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    multOut->setMemoryDescriptor(multOutMemDesc);

    ASSERT_TRUE(g.compile()) << "optimizeTpcKernels failed";
    const NodeVector& nodes = g.getExeSortedNodes();

    // after applying suggestion, expected (Reshape, Tile) -> Mult -> Reshape
    ASSERT_EQ(nodes.size(), 4);

    ASSERT_EQ(nodes[0]->getNodeType(), Node::TYPE_STATIC_RESHAPE);
    ASSERT_NE(nodes[0]->getNodeName().find("reshape"), std::string::npos);

    ASSERT_EQ(nodes[1]->getNodeType(), Node::TYPE_USER);
    ASSERT_NE(nodes[1]->getNodeName().find("tile"), std::string::npos);

    ASSERT_EQ(nodes[2]->getNodeType(), Node::TYPE_USER);
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(nodes[2]);
    ASSERT_NE(tpcNode, nullptr);
    ASSERT_TRUE(tpcNode->getGUID() == "mult_fwd_f32");

    ASSERT_EQ(nodes[3]->getNodeType(), Node::TYPE_STATIC_RESHAPE);
    ASSERT_NE(nodes[3]->getNodeName().find("reshape"), std::string::npos);
}

TEST_F(OptimizeTPCKernelsTest, ignore_suggested_manipulation_for_rmw_tensors)
{
    GaudiGraph g;
    TSize      sizes[]     = {64, 256, 4, 8};
    SizeArray  permutation = {128, 128, 4, 8, 1};

    pTensor             reluIn(new Tensor(4, sizes, syn_type_float));
    synMemoryDescriptor reluInMemDesc(true);
    reluIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    reluIn->setMemoryDescriptor(reluInMemDesc);

    pTensor reluOut(new Tensor(4, sizes, syn_type_float));
    reluOut->setTensorInSram();
    reluOut->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(
        g.getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS));
    reluOut->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    TensorVector                     inputs {reluIn};
    TensorVector                     outputs {reluOut};
    std::shared_ptr<TestableTPCNode> reluNode = std::make_shared<TestableTPCNode>(inputs,
                                                                                  outputs,
                                                                                  "Relu",
                                                                                  nullptr,
                                                                                  "relu_fwd_f32",
                                                                                  permutation,
                                                                                  tpc_lib_api::TENSOR_OP_RESHAPE);
    GraphEditor::addNode(g, reluNode);
    ASSERT_TRUE(g.compile());
    const NodeVector& nodes = g.getExeSortedNodes();

    // Expecting a single node, no other nodes should be added to the graph - the suggested manipulation should be
    // ignored (reluOut is RMW).
    ASSERT_EQ(nodes.size(), 1);
}

TEST_F(OptimizeTPCKernelsTest, test_infer_min)
{
    {
        TensorPtr t(new Tensor(TensorShape(3, {56, 56, 16}, SizeArray({56, 56, 2})), syn_type_single));
        TSize      newMax[tpc_lib_api::MAX_TENSOR_DIM] = {196, 16, 16, 1, 1};
        NSizeArray expected                            = {196, 16, 2, 1, 1};
        NSizeArray out = GraphModeSuggestedManipulationHandler::getReshapeMinTensorShape(t, newMax, 3);
        ASSERT_EQ(out, expected);
    }
    {
        TensorPtr t(new Tensor(TensorShape(5, {1, 256, 1, 1, 2}, SizeArray({1, 256, 1, 0, 2})), syn_type_single));
        TSize      newMax[tpc_lib_api::MAX_TENSOR_DIM] = {64, 4, 1, 2, 1};
        NSizeArray expected                            = {64, 4, 0, 2, 1};
        NSizeArray out = GraphModeSuggestedManipulationHandler::getReshapeMinTensorShape(t, newMax, 3);
        ASSERT_EQ(out, expected);
    }
    // Flatten up to and including first dynamic dim
    {
        TensorPtr t(new Tensor(TensorShape(4, {2, 3, 128, 128}, SizeArray({2, 3, 3, 2})), syn_type_single));
        TSize      newMax[tpc_lib_api::MAX_TENSOR_DIM] = {2, 384, 1, 128, 1};
        NSizeArray expected                            = {2, 9, 1, 2, 1};
        NSizeArray out = GraphModeSuggestedManipulationHandler::getReshapeMinTensorShape(t, newMax, 4);
        ASSERT_EQ(out, expected);
    }
    {
        TensorPtr t(new Tensor(TensorShape(3, {2, 401002, 28}, SizeArray({2, 200500, 4})), syn_type_single));
        TSize      newMax[tpc_lib_api::MAX_TENSOR_DIM] = {802004, 1, 28, 1, 1};
        NSizeArray expected                            = {401000, 1, 4, 1, 1};
        NSizeArray out = GraphModeSuggestedManipulationHandler::getReshapeMinTensorShape(t, newMax, 3);
        ASSERT_EQ(out, expected);
    }
    {
        TensorPtr t(new Tensor(TensorShape(4, {3, 2304, 1536, 4}, SizeArray({3, 1152, 768, 4})), syn_type_single));
        TSize      newMax[tpc_lib_api::MAX_TENSOR_DIM] = {6912, 1, 1536, 4, 1};
        NSizeArray expected                            = {3456, 1, 768, 4, 1};
        NSizeArray out = GraphModeSuggestedManipulationHandler::getReshapeMinTensorShape(t, newMax, 4);
        ASSERT_EQ(out, expected);
    }
}