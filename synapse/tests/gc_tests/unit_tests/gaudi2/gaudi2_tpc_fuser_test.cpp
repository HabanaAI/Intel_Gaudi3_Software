#include <gtest/gtest.h>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "graph_compiler/passes/tpc_fuser.h"
#include "test_utils.h"
#include "perf_lib_layer_params.h"
#include "tpc_fuser_test.h"

class Gaudi2TPCFuserTest : public TPCFuserTest
{
};

// Test case: TPC cluster with control deps
TEST_F(Gaudi2TPCFuserTest, DISABLED_tpcFuserClusterWithControlDeps)
{
    Gaudi2Graph g;

    const TSize n     = 256;
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
    TensorPtr t6 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t6->setName("t6");
    t6->setDramOffset(0x9000);
    t6->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    t6->setMemoryDescriptor(persistentMemoryDesc);

    NodePtr add1 = NodeFactory::createGenericTPCNode({t0, t1}, {t2}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    NodePtr add2 = NodeFactory::createGenericTPCNode({t0, t2}, {t3}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, add2);

    ns_CumSumKernel::Params cumsumParams = {0, 0, 0};
    NodePtr cumsum3 = NodeFactory::createGenericTPCNode({t3}, {t4}, &cumsumParams, "cumsum_fwd_f32", "cumsum3");
    GraphEditor::addNode(g, cumsum3);

    NodePtr add4 = NodeFactory::createGenericTPCNode({t0, t4}, {t5}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, add4);

    NodePtr add5 = NodeFactory::createGenericTPCNode({t0, t5}, {t6}, nullptr, "add_fwd_f32", "add5");
    GraphEditor::addNode(g, add5);

    g.addControlDependency({cumsum3}, {add5});

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    /* Verifying that optimized graph has these nodes: {fused_kernel}->{cumsum}->{add}->{add}
     * and only one control edge from cumsum to add5
     */

    unsigned fusedNodesCount    = 0;
    unsigned cumsumNodesCount   = 0;
    unsigned otherTpcNodesCount = 0;

    auto    t1Consumers = g.getTensorConsumers(t1);
    NodePtr t6Producer  = g.getTensorProducer(t6);
    ASSERT_GE(t6Producer->getNumInputs(Node::TENSOR_TYPE_CONTROL), 1);
    ASSERT_EQ(t6Producer->getNumOutputs(Node::TENSOR_TYPE_CONTROL), 0);
    for (const NodePtr& blocking : t1Consumers)
    {
        ASSERT_EQ(blocking->getNumInputs(Node::TENSOR_TYPE_CONTROL), 0);
        ASSERT_GE(blocking->getNumOutputs(Node::TENSOR_TYPE_CONTROL), 1);
    }

    for (auto node : g.getExeSortedNodes())
    {
        if (g.runsOnTPC(node) == true)
        {
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);

            if (tpcNode->getGUID().find("fused_kernel") != std::string::npos)
            {
                fusedNodesCount++;
            }
            else if (tpcNode->getGUID().find("cumsum") != std::string::npos)
            {
                cumsumNodesCount++;
            }
            else if (tpcNode->getGUID().find("add_fwd_f32") != std::string::npos)
            {
                otherTpcNodesCount++;
            }
        }
    }
    ASSERT_EQ(fusedNodesCount, 1) << "Expected 1 fused TPC nodes";
    ASSERT_EQ(cumsumNodesCount, 1) << "Expected 1 non-fused TPC nodes with guid cumsum";
    ASSERT_EQ(otherTpcNodesCount, 2) << "Expected 2 non-fused TPC nodes besides cumsum";
}

static pNode initNormMomentsNode()
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
TEST_F(Gaudi2TPCFuserTest, norm_moments_flow)
{
    Gaudi2Graph g;
    NodePtr     normMomentsNode = initNormMomentsNode();
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
TEST_F(Gaudi2TPCFuserTest, norm_moments_clustering)
{
    Gaudi2Graph g;
    NodePtr     normMomentsNode = initNormMomentsNode();
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