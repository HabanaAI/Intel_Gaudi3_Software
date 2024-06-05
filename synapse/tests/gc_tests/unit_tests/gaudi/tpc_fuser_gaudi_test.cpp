#include "gaudi_graph.h"
#include "graph_compiler/passes/tpc_fuser.h"
#include "tpc_fuser_test.h"
#include "scoped_configuration_change.h"
#include "perf_lib_layer_params.h"

class GaudiFuserGraph : public GaudiGraph
{
protected:
    virtual bool validateNode(const NodePtr& node) const override { return true; }
    friend class GraphEditor;
};

class GaudiTPCFuserTest : public TPCFuserTest
{
};

// Make sure the fusing is working on nodes that with dynamic shapes nodes
// TODO SW-48470 - MLIR fuser don't support fusing of fused node yet
TEST_F(GaudiTPCFuserTest, DISABLED_DynamicShapes_FuseAllDynamicNodes)
{
    GaudiGraph g;

    const TSize n        = 1;
    const TSize w        = 3;
    const TSize h        = 3;
    const TSize batchMax = 2;
    const TSize batchMin = 1;

    const TSize maxSizes[] = {n, w, h, batchMax};
    const TSize minSizes[] = {n, w, h, batchMin};

    synMemoryDescriptor persistentMemoryDesc(true);

    auto IN1 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    IN1->setName("in1");
    IN1->setDramOffset(0x7000);
    IN1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    auto IN2 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    IN2->setName("in2");
    IN2->setDramOffset(0x8000);
    IN2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    IN2->setMemoryDescriptor(persistentMemoryDesc);
    auto OUT1 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    OUT1->setName("out1");
    auto OUT2 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    OUT2->setName("out2");
    auto OUT3 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    OUT3->setName("out3");
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    std::unordered_set<pNode> cluster;
    pNode add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    pNode add2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, add2);

    auto nodeCountBeforeTPCFuser = g.getNumNodes();
    bool retVal                  = tpcFuser(g);
    ASSERT_EQ(retVal, true);
    ASSERT_EQ(nodeCountBeforeTPCFuser - 1, g.getNumNodes());

    pNode memcpy1 = NodeFactory::createNode({OUT2}, {OUT3}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "memcpy1");
    GraphEditor::addNode(g, memcpy1);
    nodeCountBeforeTPCFuser++;

    retVal = tpcFuser(g);
    ASSERT_EQ(retVal, true);
    ASSERT_EQ(nodeCountBeforeTPCFuser - 2, g.getNumNodes());
}

// TODO SW-48470 - MLIR fuser don't support fusing of fused node yet
TEST_F(GaudiTPCFuserTest, DISABLED_DynamicShapes_FuseStaticDynamicStatic)
{
    GaudiGraph g;

    const TSize n        = 256;
    const TSize w        = 3;
    const TSize h        = 3;
    const TSize batchMax = 2;
    const TSize batchMin = 1;

    const TSize maxSizes[] = {n, w, h, batchMax};
    const TSize minSizes[] = {n, w, h, batchMin};

    synMemoryDescriptor persistentMemoryDesc(true);

    auto IN1 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    IN1->setName("in1");
    IN1->setDramOffset(0x7000);
    IN1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    auto IN2 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    IN2->setName("in2");
    IN2->setDramOffset(0x8000);
    IN2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    IN2->setMemoryDescriptor(persistentMemoryDesc);
    auto IN3 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    IN3->setMemoryDescriptor(persistentMemoryDesc);
    auto OUT1 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    OUT1->setName("out1");
    auto OUT2 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    OUT2->setName("out2");
    auto OUT3 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    OUT3->setName("out3");
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    std::unordered_set<pNode> cluster;
    pNode add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    pNode add2 = NodeFactory::createGenericTPCNode({IN3, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, add2);

    auto nodeCountBeforeTPCFuser = g.getNumNodes();
    bool retVal                  = tpcFuser(g);
    ASSERT_EQ(retVal, true);
    ASSERT_EQ(nodeCountBeforeTPCFuser - 1, g.getNumNodes());

    pNode memcpy1 = NodeFactory::createNode({OUT2}, {OUT3}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "memcpy1");
    GraphEditor::addNode(g, memcpy1);
    nodeCountBeforeTPCFuser++;

    retVal = tpcFuser(g);
    ASSERT_EQ(retVal, true);
    ASSERT_EQ(nodeCountBeforeTPCFuser - 2, g.getNumNodes());
}

// TODO SW-48470 - MLIR fuser don't support fusing of fused node yet
TEST_F(GaudiTPCFuserTest, DISABLED_DynamicShapes_StaticStaticDynamic)
{
    GaudiGraph g;

    const TSize n        = 256;
    const TSize w        = 3;
    const TSize h        = 3;
    const TSize batchMax = 2;
    const TSize batchMin = 1;

    const TSize maxSizes[] = {n, w, h, batchMax};
    const TSize minSizes[] = {n, w, h, batchMin};

    synMemoryDescriptor persistentMemoryDesc(true);

    auto IN1 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    IN1->setName("in1");
    IN1->setDramOffset(0x7000);
    IN1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    auto IN2 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    IN2->setName("in2");
    IN2->setDramOffset(0x8000);
    IN2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    IN2->setMemoryDescriptor(persistentMemoryDesc);
    auto OUT1 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    OUT1->setName("out1");
    auto OUT2 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float);
    OUT2->setName("out2");
    auto OUT3 = std::make_shared<Tensor>(4U, maxSizes, syn_type_float, minSizes);
    OUT3->setName("out3");
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    std::unordered_set<pNode> cluster;
    pNode add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    pNode add2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, add2);

    bool retVal = tpcFuser(g);
    ASSERT_EQ(retVal, true);
    NodeVector nodes     = g.getExeSortedNodes();
    pNode      fusedNode = nodes.front();
    ASSERT_EQ(fusedNode->getInputs().size(), 2);
    ASSERT_EQ(fusedNode->getOutputs().size(), 1);
    // check that the all input tensor IDs are in the fused kernel, order does not matter.
    for (const auto& id : {IN1->getId(), IN2->getId()})
    {
        ASSERT_TRUE(id == fusedNode->getInput(0)->getId() || id == fusedNode->getInput(1)->getId());
    }
    ASSERT_EQ(fusedNode->getOutput(0)->getId(), OUT2->getId());

    pNode memcpy1 = NodeFactory::createNode({OUT2}, {OUT3}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "memcpy1");
    GraphEditor::addNode(g, memcpy1);

    auto nodeCountBeforeTPCFuser = g.getNumNodes();
    nodes                        = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 2);

    retVal = tpcFuser(g);
    ASSERT_EQ(retVal, true);
    ASSERT_EQ(nodeCountBeforeTPCFuser - 1, 1);
}

static gcapi::FuserRetVal_t
fuserGraphFuncOptGraphMissingPersistent(const FuserGraphTypeV4* graphIn, FuserGraphTypeV4* graphOut, bool debug)
{
    for (const auto& node : graphIn->nodes)
    {
        graphOut->nodes.push_back(node);
    }

    graphOut->nodes.erase(graphOut->nodes.begin() + 1);
    graphOut->nodes.front()->outputEdges.front().tensor = graphIn->nodes.back()->outputEdges.front().tensor;

    graphOut->deviceId = graphIn->deviceId;
    graphOut->kernelType = graphIn->kernelType;

    return gcapi::FUSER_SUCCESS;
}

static gcapi::FuserRetVal_t
fuserGraphFuncOptGraphAddReshape(const FuserGraphTypeV4* graphIn, FuserGraphTypeV4* graphOut, bool debug)
{
    // Return fused node with 2 reshapes - tpc fuser only inserts reshapes next to fusions

    unsigned uniqueTensorId = 10;

    auto nodeIn  = graphIn->nodes.front();  // the first relu node
    auto nodeOut = graphIn->nodes.back();   // the second relu node

    auto fusedNode = std::make_shared<FuserNodeTypeV4>();

    auto reshapeNode1 = std::make_shared<FuserNodeTypeV4>();

    char reshapeGuid[tpc_lib_api::MAX_NODE_NAME] = "reshape";

    auto firstReshapeEdgeInput    = std::make_shared<FuserEdgeTypeV4>();
    auto firstReshapeEdgeOutput   = std::make_shared<FuserEdgeTypeV4>();
    auto firstReshapeInputTensor  = std::make_shared<FuserTensorTypeV4>();
    auto firstReshapeOutputTensor = std::make_shared<FuserTensorTypeV4>();

    firstReshapeInputTensor                   = nodeIn->inputEdges[0].tensor;
    firstReshapeInputTensor->uniqueIdentifier = nodeIn->inputEdges[0].tensor->uniqueIdentifier;
    firstReshapeEdgeInput->tensor             = nodeIn->inputEdges[0].tensor;
    firstReshapeEdgeInput->targetNode         = std::make_shared<FuserNodeTypeV4>();

    firstReshapeOutputTensor->uniqueIdentifier = uniqueTensorId;
    uniqueTensorId++;
    firstReshapeOutputTensor->dataType     = firstReshapeInputTensor->dataType;
    firstReshapeOutputTensor->geometry     = firstReshapeInputTensor->geometry;
    firstReshapeOutputTensor->section.type = gcapi::SECTION_WORKSPACE;

    unsigned size0 = firstReshapeInputTensor->geometry.sizes[0];
    unsigned size1 = firstReshapeInputTensor->geometry.sizes[1];
    unsigned size2 = firstReshapeInputTensor->geometry.sizes[2];
    unsigned size3 = firstReshapeInputTensor->geometry.sizes[3];

    firstReshapeOutputTensor->geometry.sizes[0] = size0 * size1;
    firstReshapeOutputTensor->geometry.sizes[1] = size2 * size3;
    firstReshapeOutputTensor->geometry.sizes[2] = 1;
    firstReshapeOutputTensor->geometry.sizes[3] = 1;
    std::copy(firstReshapeOutputTensor->geometry.maxSizes,
              firstReshapeOutputTensor->geometry.maxSizes + HABANA_DIM_MAX,
              firstReshapeOutputTensor->geometry.minSizes);

    firstReshapeEdgeOutput->tensor     = firstReshapeOutputTensor;
    firstReshapeEdgeOutput->targetNode = fusedNode;

    strncpy(reshapeNode1->guid, reshapeGuid, tpc_lib_api::MAX_NODE_NAME);
    strncpy(reshapeNode1->nodeName, "reshape1", tpc_lib_api::MAX_NODE_NAME);
    reshapeNode1->inputEdges.push_back(*firstReshapeEdgeInput);
    reshapeNode1->outputEdges.push_back(*firstReshapeEdgeOutput);
    reshapeNode1->nodeParams       = nullptr;
    reshapeNode1->paramsSize       = 0;
    reshapeNode1->uniqueIdentifier = uniqueTensorId;
    uniqueTensorId++;

    // Finish creating the first reshape node reshapeNode1

    auto reshapeNode2             = std::make_shared<FuserNodeTypeV4>();
    auto secondReshapeEdgeInput   = std::make_shared<FuserEdgeTypeV4>();
    auto secondReshapeEdgeOutput  = std::make_shared<FuserEdgeTypeV4>();
    auto secondReshapeInputTensor = std::make_shared<FuserTensorTypeV4>();

    secondReshapeInputTensor->uniqueIdentifier = ++uniqueTensorId;
    secondReshapeInputTensor->dataType         = firstReshapeInputTensor->dataType;
    secondReshapeInputTensor->geometry         = firstReshapeOutputTensor->geometry;
    secondReshapeInputTensor->section.type     = gcapi::SECTION_WORKSPACE;

    secondReshapeEdgeInput->tensor     = secondReshapeInputTensor;
    secondReshapeEdgeInput->targetNode = fusedNode;

    secondReshapeEdgeOutput->tensor     = nodeOut->outputEdges[0].tensor;
    secondReshapeEdgeOutput->targetNode = std::make_shared<FuserNodeTypeV4>();
    ;

    strncpy(reshapeNode2->guid, reshapeGuid, tpc_lib_api::MAX_NODE_NAME);
    strncpy(reshapeNode2->nodeName, "reshape2", tpc_lib_api::MAX_NODE_NAME);
    reshapeNode2->inputEdges.push_back(*secondReshapeEdgeInput);
    reshapeNode2->outputEdges.push_back(*secondReshapeEdgeOutput);
    reshapeNode2->nodeParams       = nullptr;
    reshapeNode2->paramsSize       = 0;
    reshapeNode2->uniqueIdentifier = uniqueTensorId;

    // Create the fused node
    auto fusedEdgeInput    = std::make_shared<FuserEdgeTypeV4>();
    auto fusedEdgeOutput   = std::make_shared<FuserEdgeTypeV4>();
    auto fusedInputTensor  = std::make_shared<FuserTensorTypeV4>();
    auto fusedOutputTensor = std::make_shared<FuserTensorTypeV4>();

    fusedInputTensor           = firstReshapeOutputTensor;
    fusedInputTensor->geometry = firstReshapeOutputTensor->geometry;

    fusedEdgeInput->tensor     = fusedInputTensor;
    fusedEdgeInput->targetNode = reshapeNode1;

    fusedOutputTensor           = secondReshapeInputTensor;
    fusedOutputTensor->geometry = secondReshapeInputTensor->geometry;
    fusedEdgeOutput->tensor     = fusedOutputTensor;
    fusedEdgeOutput->targetNode = reshapeNode2;

    strncpy(fusedNode->guid, "fused_kernel_0_f32", tpc_lib_api::MAX_NODE_NAME);
    strncpy(fusedNode->nodeName, "fused_node", tpc_lib_api::MAX_NODE_NAME);
    fusedNode->inputEdges.push_back(*fusedEdgeInput);
    fusedNode->outputEdges.push_back(*fusedEdgeOutput);
    fusedNode->nodeParams       = nodeIn->nodeParams;
    fusedNode->paramsSize       = nodeIn->paramsSize;
    fusedNode->uniqueIdentifier = std::max(nodeIn->uniqueIdentifier, nodeOut->uniqueIdentifier) + 1;
    fusedNode->fusedIdentifiers = {nodeIn->uniqueIdentifier, nodeOut->uniqueIdentifier};
    fusedNode->newIdentifiers   = {reshapeNode1->uniqueIdentifier, reshapeNode2->uniqueIdentifier};

    graphOut->nodes.push_back(reshapeNode1);
    graphOut->nodes.push_back(fusedNode);
    graphOut->nodes.push_back(reshapeNode2);

    graphOut->deviceId = graphIn->deviceId;
    graphOut->kernelType = graphIn->kernelType;

    return gcapi::FUSER_SUCCESS;
}

TEST_F(GaudiTPCFuserTest, relu_forward_with_reshapes)
{
    GaudiFuserGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, "relu_fwd_f32", "relu1");
    GraphEditor::addNode(g, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    OUT1->setName("out2");

    pNode reluNode2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, "relu_fwd_f32", "relu2");
    GraphEditor::addNode(g, reluNode2);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();
    std::unordered_set<pNode> clusterNodes = clusterConstructor.popNextCluster();

    std::shared_ptr<GCTPCFuserWrapper> fuser(
        new GCTPCFuserWrapper(clusterNodes, g, &fuserGraphFuncOptGraphAddReshape, &fuserGraphGetEmptyPreGraph));

    fuser->optimizeCluster(g);
    ASSERT_EQ(replaceOptimizedCluster(g, fuser), optimizedGraphSuccess) << "Expected optimized graph";

    ASSERT_EQ(g.getExeSortedNodes().size(), 3) << "Expected optimized graph to have 3 nodes";

    /* Check for 2 reshapes and 1 TPC node*/
    unsigned reshapeCount = 0;
    for (auto node : g.getExeSortedNodes())
    {
        bool isReshapeNode = (dynamic_cast<ReshapeNode*>(node.get()) != nullptr);
        if (isReshapeNode)
        {
            reshapeCount++;
        }
        else
        {
            ASSERT_EQ(g.runsOnTPC(node), true) << "Expecting TPC node";
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
            ASSERT_EQ(tpcNode->getGUID(), "fused_kernel_0_f32") << "Unexpected GUID, expected fused_kernel_0_f32 node";
        }
    }

    ASSERT_EQ(reshapeCount, 2) << "Expected 2 reshapes nodes";
}

// Testing TPCFuser class
// Test case: Returned fusedGraph does not contain all external tensor.
//            two nodes where intermediate tensor is marked as persistent, is missing from fused graph
// Expecting no fusion.
//
TEST_F(GaudiTPCFuserTest, tpcFuserInvalidFusedGraphMissingExtTensors)
{
    GaudiFuserGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    synMemoryDescriptor memDesc(true);
    OUT1->setMemoryDescriptor(memDesc);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pNode sub1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "xor_i16", "sub1");
    GraphEditor::addNode(g, sub1);

    pNode sub2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "and_i16", "add2");
    GraphEditor::addNode(g, sub2);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();
    std::unordered_set<pNode> clusterNodes = clusterConstructor.popNextCluster();

    std::shared_ptr<GCTPCFuserWrapper> fuser(
        new GCTPCFuserWrapper(clusterNodes, g, &fuserGraphFuncOptGraphMissingPersistent, &fuserGraphGetEmptyPreGraph));

    fuser->optimizeCluster(g);
    ASSERT_EQ(replaceOptimizedCluster(g, fuser), optimizedGraphInvalidFusedGraph) << "Expected invalid optimized graph";

    /* Verifying that optimized graph has 2 nodes as original graph */

    ASSERT_EQ(g.getExeSortedNodes().size(), 2) << "Expected optimized graph to have 2 nodes";

    /* Verifiting that optimized graph has the different GUIDs from the ondes in the original graph
     * and as should be according to fuserGraphFuncOptGraphMissingPersistent function*/
    for (auto node : g.getExeSortedNodes())
    {
        ASSERT_EQ(g.runsOnTPC(node), true) << "Expecting TPC node";
        TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        ASSERT_NE(tpcNode->getGUID(), "add_i16") << "Unexpected GUID, fusion should not take place";
    }
}

pNode initNormMomentsNode()
{
    const TSize c = 128;
    const TSize w = 8;
    const TSize h = 4;
    const TSize b = 2;
    char        data[c * w * h * b];
    const TSize inSizes[] = {c, w, h, b, 1};

    const TSize wOut = 1;
    const TSize hOut = 1;
    char        outData[c * wOut * hOut * b];
    const TSize outSizes[] = {c, wOut, hOut, b, 1};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor in = pTensor(new Tensor(5U, inSizes, syn_type_float, reinterpret_cast<char*>(data)));
    in->setName("in");
    in->setMemoryDescriptor(persistentMemoryDesc);
    in->setDramOffset(0x1000);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    pTensor outVariance = pTensor(new Tensor(5U, outSizes, syn_type_float, reinterpret_cast<char*>(outData)));
    outVariance->setName("variance");
    outVariance->setMemoryDescriptor(persistentMemoryDesc);
    outVariance->setDramOffset(0x10000);
    outVariance->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    pTensor outMean = pTensor(new Tensor(5U, outSizes, syn_type_float, reinterpret_cast<char*>(outData)));
    outMean->setName("mean");
    outMean->setMemoryDescriptor(persistentMemoryDesc);
    outMean->setDramOffset(0x100000);
    outMean->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);

    ns_NormMomentsKernel::Params params;
    params.NormAxisBmp = 6;  // it's a bitmap

    const char guid[] = "norm_moments_fwd_f32";
    return NodeFactory::createGenericTPCNode({in}, {outMean, outVariance}, &params, guid, "nm");
}

// Test norm_moments flow
// Test is adapted from /trees/npu-stack/tpc_fuser/mlir/pytenettests/ComplexGuidTests/Normalization/NormMomentsTest.cpp
TEST_F(GaudiTPCFuserTest, norm_moments_flow)
{
    GaudiGraph g;
    NodePtr    normMomentsNode = initNormMomentsNode();
    ASSERT_TRUE(GraphEditor::addNode(g, normMomentsNode));

    ASSERT_EQ(g.getExeSortedNodes().size(), 1);
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(g));
    unsigned extractedNodesNumber = g.getExeSortedNodes().size();
    ASSERT_TRUE(extractedNodesNumber > 1);
    ASSERT_TRUE(tpcFuser(g));
    unsigned nodesAfterFuserNumber = g.getExeSortedNodes().size();

    // norm_moments solution in fuser and synapse is WIP.
    // Therefore at this stage we can only expect fusion of simple elementwise ops, not reduction ops.
    // So the expected node number after fusion can be changed, we only need it to be less than before the fusion.
    ASSERT_TRUE(nodesAfterFuserNumber < extractedNodesNumber);
}

// Test norm_moments clustering - verify all extracted nodes are in same cluster.
TEST_F(GaudiTPCFuserTest, norm_moments_clustering)
{
    GaudiGraph g;
    NodePtr    normMomentsNode = initNormMomentsNode();
    ASSERT_TRUE(GraphEditor::addNode(g, normMomentsNode));
    auto        nodeId   = normMomentsNode->getId();
    const auto& nodeGuid = normMomentsNode->getGUID();
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(g));
    ASSERT_TRUE(tpcFuser(g));
    int countCGNodes = 0;
    // verify that the originalComplexGuidId field has either the default value or the normMomentsNode id.
    for (const NodePtr& node : g.getNodes())
    {
        if (node->getNodeAnnotation().originalComplexGuidId == nodeId)
        {
            ASSERT_EQ(node->getNodeAnnotation().originalComplexGuid, nodeGuid);
            countCGNodes++;
        }
        else
        {
            ASSERT_EQ(node->getNodeAnnotation().originalComplexGuidId, ~uint64_t(0))
                << "Expected original complex guid node id";
            ASSERT_EQ(strcmp(node->getNodeAnnotation().originalComplexGuid, ""), 0) << "Expected original guid";
        }
    }
    ASSERT_GT(countCGNodes, 0);
}

// Test case: TPC cluster with inner control deps
TEST_F(GaudiTPCFuserTest, DISABLED_tpcFuserInnerControlDeps)
{
    GaudiGraph g;

    const TSize n     = 2048;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch * sizeof(float)];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr t0 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t0->setName("t0");
    t0->setDramOffset(0x7000);
    t0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    t0->setMemoryDescriptor(persistentMemoryDesc);
    TensorPtr t1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t1->setName("t1");
    t1->setDramOffset(0x9000);
    t1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    t1->setMemoryDescriptor(persistentMemoryDesc);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t2->setName("t2");
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t3->setName("t3");
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t4->setName("t4");
    TensorPtr t5 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t5->setName("t5");
    t5->setDramOffset(0x7000);
    t5->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    t5->setMemoryDescriptor(persistentMemoryDesc);

    NodePtr add1 = NodeFactory::createGenericTPCNode({t0, t1}, {t2}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    NodePtr relu = NodeFactory::createGenericTPCNode({t2}, {t3}, nullptr, "relu_fwd_f32", "relu");
    GraphEditor::addNode(g, relu);

    NodePtr add2 = NodeFactory::createGenericTPCNode({t0, t3}, {t4}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, add2);

    NodePtr add3 = NodeFactory::createGenericTPCNode({t0, t4}, {t5}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, add3);

    g.addControlDependency({relu}, {add3});

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    /* Verifying that optimized graph has these nodes: {fused_kernel}->{add3}
     * and only one control edge from Add to one of the fused nodes (that contained add3)
     */

    auto    t0Consumers = g.getTensorConsumers(t0);
    NodePtr t5Producer  = g.getTensorProducer(t5);
    ASSERT_GE(t5Producer->getNumInputs(Node::TENSOR_TYPE_CONTROL), 1);
    ASSERT_EQ(t5Producer->getNumOutputs(Node::TENSOR_TYPE_CONTROL), 0);
    for (const NodePtr& blocking : t0Consumers)
    {
        ASSERT_EQ(blocking->getNumInputs(Node::TENSOR_TYPE_CONTROL), 0);
        ASSERT_GE(blocking->getNumOutputs(Node::TENSOR_TYPE_CONTROL), 1);
    }

    unsigned fusedNodesCount = 0;
    unsigned addNodesCount   = 0;
    unsigned otherNodesCount = 0;

    for (auto node : g.getExeSortedNodes())
    {
        if (g.runsOnTPC(node) == true)
        {
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);

            if (tpcNode->getGUID().find("fused_kernel") != std::string::npos)
            {
                fusedNodesCount++;
            }
            else if (tpcNode->getGUID().find("add") != std::string::npos)
            {
                addNodesCount++;
            }
            else
            {
                otherNodesCount++;
            }
        }
    }
    ASSERT_EQ(fusedNodesCount, 1) << "Expected 1 fused TPC node";
    ASSERT_EQ(addNodesCount, 1) << "Expected 1 non-fused TPC nodes with guid add";
    ASSERT_EQ(otherNodesCount, 0) << "Expected 0 non-fused TPC nodes besides add and fused node";
}