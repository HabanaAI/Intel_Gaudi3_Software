#include "graph_compiler/passes/tpc_fuser.h"
#include "tpc_fuser_test.h"
#include "scoped_configuration_change.h"
#include "graph_factory.h"
#include "generic_graph_test.h"
#include <memory>
#include "perf_lib_layer_params.h"

class GenericTPCFuserTest
: public TPCFuserTest
, public testing::WithParamInterface<synDeviceType>
{
    void SetUp() override
    {
        TPCFuserTest::SetUp();
        synDeviceType deviceType = GetParam();
        m_graph                  = GraphFactory::createGraph(deviceType, CompilationMode::Graph);
    }

    void TearDown() override
    {
        TPCFuserTest::TearDown();
    }

protected:
    std::unique_ptr<HabanaGraph> m_graph;
};

// Testing TPCFuser class
// Test case: Graph with two tpc-nodes-clusters
TEST_P(GenericTPCFuserTest, tpcFuserBasicPassTwoClusters)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 1;
    const TSize h     = 1;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[]        = {n, w, h, batch};

    uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2;
    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x7000);
    IN1->setMemorySectionID(memSecId++);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x8000);
    IN2->setMemorySectionID(memSecId++);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(memSecId++);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(memSecId++);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    IN5->setDramOffset(0xB000);
    IN5->setMemorySectionID(memSecId++);
    IN5->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(memSecId++);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    pTensor OUT6 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT6->setName("out6");
    pTensor OUT7 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT7->setName("out7");
    OUT7->setDramOffset(0xD000);
    OUT7->setMemorySectionID(memSecId++);
    OUT7->setMemoryDescriptor(persistentMemoryDesc);

    std::unordered_set<pNode> cluster;
    pNode                     xor1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, xor1);

    pNode xor2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, xor2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, addGUID, "add3");
    GraphEditor::addNode(*m_graph, or3);

    pNode xor4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, addGUID, "add4");
    GraphEditor::addNode(*m_graph, xor4);

    pNode flatten = NodeFactory::createNode({OUT4}, {OUT5}, nullptr, NodeFactory::DebugNodeTypeName, "other");
    GraphEditor::addNode(*m_graph, flatten);

    pNode or5 = NodeFactory::createGenericTPCNode({IN5, OUT5}, {OUT6}, nullptr, addGUID, "add5");
    GraphEditor::addNode(*m_graph, or5);

    pNode or6 = NodeFactory::createGenericTPCNode({IN5, OUT6}, {OUT7}, nullptr, addGUID, "add6");
    GraphEditor::addNode(*m_graph, or6);

    bool retVal = (*m_graph).compile();
    ASSERT_EQ(retVal, true);

    const NodeVector& execOrder = (*m_graph).getExeSortedNodes();
    ASSERT_EQ(execOrder.size(), 3);
    ASSERT_EQ(execOrder[0]->getInputs().size(), 4);
    ASSERT_EQ(execOrder[0]->getOutputs().size(), 2);
    ASSERT_EQ(execOrder[1]->getInputs().size(), 1);
    ASSERT_EQ(execOrder[1]->getOutputs().size(), 1);
    ASSERT_EQ(execOrder[2]->getInputs().size(), 2);
    ASSERT_EQ(execOrder[2]->getOutputs().size(), 1);
}

// Testing maxAvailableTPCs
// Test case: Graph with one tpc node
TEST_P(GenericTPCFuserTest, maxAvailableTpcTest)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    uint64_t                  memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2;
    ScopedConfigurationChange MaxAvailableTpcMode("TPC_ENGINES_ENABLED_MASK", "0x00003F");

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x7000);
    IN1->setMemorySectionID(memSecId++);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    OUT1->setDramOffset(0x8000);
    OUT1->setMemorySectionID(memSecId++);
    OUT1->setMemoryDescriptor(persistentMemoryDesc);

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu");
    GraphEditor::addNode(*m_graph, reluNode);

    bool retVal = (*m_graph).compile();
    ASSERT_EQ(retVal, true);

    for (auto node : (*m_graph).getExeSortedNodes())
    {
        if (node->isLogicalOperation())
        {
            continue;
        }
        else
        {
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
            ASSERT_NE(tpcNode, nullptr);
            ASSERT_EQ(tpcNode->getGUID(), reluGUID) << "Unexpected GUID, expected" << reluGUID << "node";
            ASSERT_EQ(tpcNode->getSucceededGlueParams().maxAvailableTpc, 6);
        }
    }

    m_graph = GraphFactory::createGraph(GetParam(), CompilationMode::Graph);

    ScopedConfigurationChange MaxAvailableTpcMode2("TPC_ENGINES_ENABLED_MASK", "0x000033");
    synMemoryDescriptor       persistentMemoryDescG2(true);

    pTensor IN1G2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1G2->setName("in1g2");
    IN1G2->setDramOffset(0x9000);
    IN1G2->setMemorySectionID(memSecId++);
    IN1G2->setMemoryDescriptor(persistentMemoryDescG2);

    pTensor OUT1G2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1G2->setName("out1g2");
    OUT1G2->setDramOffset(0x10000);
    OUT1G2->setMemorySectionID(memSecId++);
    OUT1G2->setMemoryDescriptor(persistentMemoryDescG2);

    pNode reluNodeG2 = NodeFactory::createGenericTPCNode({IN1G2}, {OUT1G2}, nullptr, reluGUID, "relu");
    GraphEditor::addNode(*m_graph, reluNodeG2);

    bool retVal2 = (*m_graph).compile();
    ASSERT_EQ(retVal2, true);
    for (auto node : (*m_graph).getExeSortedNodes())
    {
        if (node->isLogicalOperation())
        {
            continue;
        }
        else
        {
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
            ASSERT_NE(tpcNode, nullptr);
            ASSERT_EQ(tpcNode->getGUID(), reluGUID) << "Unexpected GUID, expected" << reluGUID << "node";
            ASSERT_EQ(tpcNode->getSucceededGlueParams().maxAvailableTpc, 4);
        }
    }
}

// inside a cluster, a TPC node's output is being consume by a following TPC node at both of its ports.
// TENSOR1 ---- TPC1 ---TENSOR3-----TPC2
//            /                 \   /
// TENSOR2 --/                   ---
// TPC Fuser is expected to continue clustering
TEST_P(GenericTPCFuserTest, tpcFuserTensorIsConsumerTwiceByTheSameNode)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    std::unordered_set<pNode> cluster;
    pNode                     add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, add1);

    pNode add2 = NodeFactory::createGenericTPCNode({IN3, OUT1}, {OUT2}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, add2);

    pNode add3 = NodeFactory::createGenericTPCNode({OUT2, OUT2}, {OUT3}, nullptr, addGUID, "add3");
    GraphEditor::addNode(*m_graph, add3);

    pNode add4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, addGUID, "add4");
    GraphEditor::addNode(*m_graph, add4);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned                  numCluster  = 0;
    std::unordered_set<pNode> currCluster = clusterConstructor.popNextCluster();

    // expecting only one cluster that will include all nodes above
    while (currCluster.size() != 0)
    {
        ASSERT_NE(numCluster, 1) << "Invalid number of clusters";
        ASSERT_EQ(currCluster.size(), 4) << "Invalid number of nodes in cluster";
        numCluster++;
        currCluster = clusterConstructor.popNextCluster();
    }
}

TEST_P(GenericTPCFuserTest, tpcFuserConstant)
{
    // Graph contain constant node without inputs
    // Expected not to optimize the graph since cluster size isn't larger than 1

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    std::unordered_set<pNode> cluster;
    pNode                     or1 = NodeFactory::createGenericTPCNode({}, {OUT1}, nullptr, "constant_f32", "const");
    GraphEditor::addNode(*m_graph, or1);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 clusters
    ASSERT_EQ(numCluster, 1);

    cluster = clusterConstructor.popNextCluster();

    std::shared_ptr<GCTPCFuserWrapper> clusterTPCFuser =
        std::make_shared<GCTPCFuserWrapper>(cluster,
                                            *m_graph,
                                            fuserGraphFuncOptReturnSameGraph,
                                            fuserGraphGetEmptyPreGraph);

    // Expected to not optimizing the graph since cluster size isn't larger than 1
    ASSERT_EQ(clusterTPCFuser->optimizeCluster(*m_graph), false);
}

TEST_P(GenericTPCFuserTest, tpcFuserConstantAdd)
{
    // Graph contain constant node without inputs and add node
    // Expected 1 cluster after tpc fuser

    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    ns_ConstantKernel::Params constParams;
    constParams.constant.f = 0.1;

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    std::unordered_set<pNode> cluster;
    pNode                     or1 = NodeFactory::createGenericTPCNode({}, {OUT1}, &constParams, "constant_f32", "const");
    GraphEditor::addNode(*m_graph, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "add");
    GraphEditor::addNode(*m_graph, or2);

    auto     nodeCountBeforeTPCFuser = (*m_graph).getNumNodes();

    bool retVal = tpcFuser(*m_graph);
    ASSERT_EQ(retVal, true);

    // Expected to fuse the constant and add
    ASSERT_EQ(nodeCountBeforeTPCFuser - 1, (*m_graph).getNumNodes());
}

TEST_P(GenericTPCFuserTest, tpcFuserConstantAddNoFusing)
{
    // Graph contain constant node without inputs and add node
    // In order to check return optimized graph fro mTPC Fuser
    // there is an hack that return the same cluster that contains
    // at first node without inputs
    // Expected 2 clusters after tpc fuser

    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    std::unordered_set<pNode> cluster;
    pNode                     or1 = NodeFactory::createGenericTPCNode({}, {OUT1}, nullptr, "constant_f32", "const");
    GraphEditor::addNode(*m_graph, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "add");
    GraphEditor::addNode(*m_graph, or2);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 clusters
    ASSERT_EQ(numCluster, 1);

    cluster = clusterConstructor.popNextCluster();

    std::shared_ptr<GCTPCFuserWrapper> clusterTPCFuser =
        std::make_shared<GCTPCFuserWrapper>(cluster,
                                            *m_graph,
                                            fuserGraphFuncOptReturnSameGraph,
                                            fuserGraphGetEmptyPreGraph);

    clusterTPCFuser->optimizeCluster(*m_graph);

    // Verfiy there is return ERROR on no inputs of the first node
    ASSERT_EQ(replaceOptimizedCluster(*m_graph, clusterTPCFuser), optimizedGraphSuccess) << "Expected graph success";

    /* Verifying that optimized graph has 2 nodes as original graph */

    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 2) << "Expected optimized graph to have 2 nodes";
}

// Multi consumers testing
// Test the following graph:
// tpc1-->tpc2
//     -->tpc3
// tpc1 has multi consumers, make sure all in the same cluster
TEST_P(GenericTPCFuserTest, oneMultiConsumer)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pNode relu2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    pNode relu3 = NodeFactory::createGenericTPCNode({OUT1}, {OUT3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

// Multi consumers testing
// Test the following graph:
// tpc1-->tpc2-->tpc5
//     -->tpc3-->tpc4
// tpc1 has multi consumers, make sure all in the same cluster
TEST_P(GenericTPCFuserTest, oneMultiConsumerLargerCluster)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");

    pNode relu2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    pNode relu5 = NodeFactory::createGenericTPCNode({OUT2}, {OUT5}, nullptr, reluGUID, "relu5");
    GraphEditor::addNode(*m_graph, relu5);

    pNode relu3 = NodeFactory::createGenericTPCNode({OUT1}, {OUT3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    pNode relu4 = NodeFactory::createGenericTPCNode({OUT3}, {OUT4}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu4);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

// Multi consumers testing
// 2 multi consumers
// expecting 1 cluster
TEST_P(GenericTPCFuserTest, twoMultiConsumers)
{
    GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.setValue(2);

    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");

    pNode relu2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    pNode relu3 = NodeFactory::createGenericTPCNode({OUT1}, {OUT3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    pNode relu4 = NodeFactory::createGenericTPCNode({OUT2}, {OUT4}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);

    pNode relu5 = NodeFactory::createGenericTPCNode({OUT2}, {OUT5}, nullptr, reluGUID, "relu5");
    GraphEditor::addNode(*m_graph, relu5);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 clusters
    ASSERT_EQ(numCluster, 1);

    GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.setValue(pre_GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER);
}

// Multi consumers testing
// 3 multi consumers while allowing only 2
// expecting 2 clusters
TEST_P(GenericTPCFuserTest, threeMultiConsumers)
{
    GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.setValue(2);

    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    pTensor OUT5 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");

    pTensor OUT6 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT6->setName("out6");

    pTensor OUT7 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT7->setName("out7");

    pNode relu2 = NodeFactory::createGenericTPCNode({OUT1}, {OUT2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    pNode relu3 = NodeFactory::createGenericTPCNode({OUT1}, {OUT3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    pNode relu4 = NodeFactory::createGenericTPCNode({OUT2}, {OUT4}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);

    pNode relu5 = NodeFactory::createGenericTPCNode({OUT2}, {OUT5}, nullptr, reluGUID, "relu5");
    GraphEditor::addNode(*m_graph, relu5);

    pNode relu6 = NodeFactory::createGenericTPCNode({OUT3}, {OUT6}, nullptr, reluGUID, "relu6");
    GraphEditor::addNode(*m_graph, relu6);

    pNode relu7 = NodeFactory::createGenericTPCNode({OUT3}, {OUT7}, nullptr, reluGUID, "relu7");
    GraphEditor::addNode(*m_graph, relu7);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);

    GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER.setValue(pre_GCFG_NUM_MAX_MULTI_CONSUMERS_IN_CLUSTER);
}

// Multi consumers testing
// 2 multi consumers while allowing only 2
// with 2 multi consumers there is a cycle , expecting 2 clusters
TEST_P(GenericTPCFuserTest, twoMultiConsumerPossiableCycle)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pTensor OUT_flat = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT_flat->setName("OUT_flat");

    pNode relu3 = NodeFactory::createGenericTPCNode({OUT1}, {OUT3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    pNode relu2 = NodeFactory::createGenericTPCNode({OUT3, OUT_flat}, {OUT2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    synFlattenParams flattenParams = {0};
    pNode            flatten =
        NodeFactory::createNode({OUT3}, {OUT_flat}, &flattenParams, NodeFactory::flattenNodeTypeName, "flatten");
    GraphEditor::addNode(*m_graph, flatten);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);
}

// Multi consumers testing
// Test the following graph:
// tpc1-->tpc3----------->tpc2
//     -->tpc3-->nonTPC-->tpc2
// tpc3 has multi consumers
// the graph contain cycle if all tpc nodes will be in the same cluster
// 2 cluster expected
TEST_P(GenericTPCFuserTest, multiConsumerPossiableCycle)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pTensor OUT_flat = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT_flat->setName("OUT_flat");

    pNode relu3 = NodeFactory::createGenericTPCNode({OUT1}, {OUT3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    pNode relu2 = NodeFactory::createGenericTPCNode({OUT3, OUT_flat}, {OUT2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    synFlattenParams flattenParams = {0};
    pNode            flatten =
        NodeFactory::createNode({OUT3}, {OUT_flat}, &flattenParams, NodeFactory::flattenNodeTypeName, "flatten");
    GraphEditor::addNode(*m_graph, flatten);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);
}

// Multi consumers testing
// Test the following graph:
// tpc1-->tpc3----------->tpc2
//     -->tpc3-->tpc4-->tpc2
// tpc3 has multi consumers
// the graph contain inner cycle if all tpc nodes will be in the same cluster
// 1 cluster expected due to suppoerted inner cycle
TEST_P(GenericTPCFuserTest, multiConsumerInnerCycle)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");

    pNode reluNode = NodeFactory::createGenericTPCNode({IN1}, {OUT1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, reluNode);

    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("OUT4");

    pNode relu3 = NodeFactory::createGenericTPCNode({OUT1}, {OUT3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    pNode relu2 = NodeFactory::createGenericTPCNode({OUT3, OUT4}, {OUT2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    pNode relu4 = NodeFactory::createNode({OUT3}, {OUT4}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

TEST_P(GenericTPCFuserTest, load_shared_object_test)
{
    gcapi::pfnFuseGraphV4 entry = TPCFuserSharedObject::instance().getFuseGraphFuncPtr();

    ASSERT_NE(entry, nullptr) << "failed loading shared object";
}

// Testing TPCClusterConstructor class
// Test the following graph:
// tpc1-->tpc2-->tpc3-->MME-->tpc4-->tpc5-->tpc6
TEST_P(GenericTPCFuserTest, simpleTest)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu", "bf16");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];
    char r1[n * w * h * batch];
    char r2[n * w * h * batch];
    char r3[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(in1)));
    IN->setName("in1");
    TensorPtr R1 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r1)));
    R1->setName("R1");
    TensorPtr R2 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r2)));
    R2->setName("R2");
    TensorPtr R3 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R3->setName("R3");
    TensorPtr R4 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R4");
    TensorPtr R5 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R5");
    TensorPtr R6 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R6");
    TensorPtr R7 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R7");
    TensorPtr R8 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R8");
    TensorPtr R9 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R9");

    NodePtr relu1 = NodeFactory::createGenericTPCNode({IN}, {R1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, relu1);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({R1}, {R2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({R2}, {R3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    NodePtr reshape1 = NodeFactory::createNode({R3}, {R4}, nullptr, "reshape", "re");
    GraphEditor::addNode(*m_graph, reshape1);

    NodePtr relu4 = NodeFactory::createGenericTPCNode({R4}, {R5}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);

    NodePtr relu5 = NodeFactory::createGenericTPCNode({R5}, {R6}, nullptr, reluGUID, "relu5");
    GraphEditor::addNode(*m_graph, relu5);

    NodePtr relu6 = NodeFactory::createGenericTPCNode({R6}, {R7}, nullptr, reluGUID, "relu6");
    GraphEditor::addNode(*m_graph, relu6);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu2->getId()));
    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu3->getId()));

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu4->getId()));

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu5->getId()));
    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu6->getId()));
}

// Testing TPCClusterConstructor class
// Test the following graph:
// tpc1-->tpc2-->tpc3-->MME-->tpc4-->tpc5-->tpc6
//                                   tpc5-->R8
TEST_P(GenericTPCFuserTest, nodeWithTwoOutputs)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu", "bf16");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];
    char r1[n * w * h * batch];
    char r2[n * w * h * batch];
    char r3[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(in1)));
    IN->setName("in1");
    TensorPtr R1 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r1)));
    R1->setName("R1");
    TensorPtr R2 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r2)));
    R2->setName("R2");
    TensorPtr R3 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R3->setName("R3");
    TensorPtr R4 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R4");
    TensorPtr R5 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R5");
    TensorPtr R6 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R6");
    TensorPtr R7 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R7");
    TensorPtr R8 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R8");
    TensorPtr R9 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R9");

    NodePtr relu1 = NodeFactory::createGenericTPCNode({IN}, {R1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, relu1);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({R1}, {R2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({R2}, {R3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    NodePtr reshape1 = NodeFactory::createNode({R3}, {R4}, nullptr, "reshape", "re");
    GraphEditor::addNode(*m_graph, reshape1);

    NodePtr relu4 = NodeFactory::createGenericTPCNode({R4}, {R5}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);

    NodePtr relu5 = NodeFactory::createGenericTPCNode({R5}, {R6, R8}, nullptr, reluGUID, "relu5");
    GraphEditor::addNode(*m_graph, relu5);

    NodePtr relu6 = NodeFactory::createGenericTPCNode({R6}, {R7}, nullptr, reluGUID, "relu6");
    GraphEditor::addNode(*m_graph, relu6);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu2->getId()));
    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu3->getId()));

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu4->getId()), clusterConstructor.getNodeCluster(relu5->getId()));

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu4->getId()), clusterConstructor.getNodeCluster(relu6->getId()));
}

// Testing TPCClusterConstructor class
// Test the following graph:
// tpc1-->tpc2-->tpc3-->MME-->tpc4-->tpc5-->tpc6
//               tpc3-------->tpc4
// If tpc3 and tpc4 will be clustered together a cycle will be formed
TEST_P(GenericTPCFuserTest, possibleCycle)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu", "bf16");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];
    char r1[n * w * h * batch];
    char r2[n * w * h * batch];
    char r3[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(in1)));
    IN->setName("in1");
    TensorPtr R1 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r1)));
    R1->setName("R1");
    TensorPtr R2 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r2)));
    R2->setName("R2");
    TensorPtr R3 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R3->setName("R3");
    TensorPtr R4 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R4");
    TensorPtr R5 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R5");
    TensorPtr R6 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R6");
    TensorPtr R7 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R7");
    TensorPtr R8 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R8");
    TensorPtr R9 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R9");

    NodePtr relu1 = NodeFactory::createGenericTPCNode({IN}, {R1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, relu1);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({R1}, {R2}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({R2}, {R3}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    synFlattenParams flattenParams = {0};
    pNode flatten = NodeFactory::createNode({R3}, {R4}, &flattenParams, NodeFactory::flattenNodeTypeName, "flatten");
    GraphEditor::addNode(*m_graph, flatten);

    NodePtr relu4 = NodeFactory::createGenericTPCNode({R3, R4}, {R5}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);

    NodePtr relu5 = NodeFactory::createGenericTPCNode({R5}, {R6}, nullptr, reluGUID, "relu5");
    GraphEditor::addNode(*m_graph, relu5);

    NodePtr relu6 = NodeFactory::createGenericTPCNode({R6}, {R7}, nullptr, reluGUID, "relu6");
    GraphEditor::addNode(*m_graph, relu6);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu2->getId()));
    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu3->getId()));

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu4->getId()), clusterConstructor.getNodeCluster(relu5->getId()));
    ASSERT_EQ(clusterConstructor.getNodeCluster(relu4->getId()), clusterConstructor.getNodeCluster(relu6->getId()));

    ASSERT_NE(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu6->getId()));
}

// Testing TPCClusterConstructor class
// Test the following graph:
// tpc1-->MME1-->tpc2-->MME2-->tpc3-->tpc4
//               tpc2--------->tpc3
TEST_P(GenericTPCFuserTest, possibleCycle2)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu", "bf16");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];
    char r1[n * w * h * batch];
    char r2[n * w * h * batch];
    char r3[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(in1)));
    IN->setName("in1");
    TensorPtr R1 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r1)));
    R1->setName("R1");
    TensorPtr R2 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r2)));
    R2->setName("R2");
    TensorPtr R3 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R3->setName("R3");
    TensorPtr R4 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R4");
    TensorPtr R5 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R5");
    TensorPtr R6 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R6");
    TensorPtr R7 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R7");
    TensorPtr R8 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R8");
    TensorPtr R9 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R9");

    NodePtr relu1 = NodeFactory::createGenericTPCNode({IN}, {R1}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, relu1);

    synFlattenParams flattenParams = {0};
    pNode flatten = NodeFactory::createNode({R1}, {R2}, &flattenParams, NodeFactory::flattenNodeTypeName, "flatten1");
    GraphEditor::addNode(*m_graph, flatten);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({R2}, {R3}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    pNode flatten2 = NodeFactory::createNode({R3}, {R4}, &flattenParams, NodeFactory::flattenNodeTypeName, "flatten2");
    GraphEditor::addNode(*m_graph, flatten2);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({R3, R4}, {R5}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    NodePtr relu4 = NodeFactory::createGenericTPCNode({R5}, {R6}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    ASSERT_NE(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu2->getId()));
    ASSERT_NE(clusterConstructor.getNodeCluster(relu2->getId()), clusterConstructor.getNodeCluster(relu3->getId()));
    ASSERT_NE(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu4->getId()));

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu3->getId()), clusterConstructor.getNodeCluster(relu4->getId()));
}

// Testing GCTPCFuserWrapper class
// Test the following graph:
// --(14-inputs)-->tpc1-->tpc2---(3inputs)-->tpc3-->
// Test that the TPCClusterConstructor created cluster that doesn't care about previous ext tensors limitation
TEST_P(GenericTPCFuserTest, externalInputs)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu", "bf16");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];
    char r1[n * w * h * batch];
    char r2[n * w * h * batch];
    char r3[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(in1)));
    IN->setName("in1");
    TensorPtr R1 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r1)));
    R1->setName("R1");
    TensorPtr R2 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r2)));
    R2->setName("R2");
    TensorPtr R3 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R3->setName("R3");
    TensorPtr R4 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R4");
    TensorPtr R5 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R5");
    TensorPtr R6 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R6");
    TensorPtr R7 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R7");
    TensorPtr R8 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R8");
    TensorPtr R9 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R9");
    TensorPtr R10 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R10");
    TensorPtr R11 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R11");
    TensorPtr R12 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R12");
    TensorPtr R13 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R13");
    TensorPtr R14 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R14");
    TensorPtr R15 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R15");
    TensorPtr R16 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R16");
    TensorPtr R17 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R17");
    TensorPtr R18 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16, reinterpret_cast<char*>(r3)));
    R4->setName("R18");

    NodePtr relu1 = NodeFactory::createGenericTPCNode({IN, R1, R2, R3, R4, R5, R6}, {R14}, nullptr, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, relu1);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({R14}, {R15}, nullptr, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({R15, R16, R17}, {R18}, nullptr, reluGUID, "relu3");
    GraphEditor::addNode(*m_graph, relu3);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu2->getId()));
    ASSERT_EQ(clusterConstructor.getNodeCluster(relu1->getId()), clusterConstructor.getNodeCluster(relu3->getId()));
}

// Testing GCTPCFuserWrapper class
// Test the following graph:
// In1-->XOR--Out1-->and --Out2-->
// In2-->----->
TEST_P(GenericTPCFuserTest, tpcFuserBasicFusion)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    TensorPtr IN2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    TensorPtr OUT1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    TensorPtr OUT2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out2");

    NodePtr add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, add1);

    NodePtr add2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, add2);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    std::unordered_set<NodePtr> cluster;
    cluster.insert(add1);
    cluster.insert(add2);

    gcapi::pfnFuseGraphV4            fuserFunc       = TPCFuserSharedObject::instance().getFuseGraphFuncPtr();
    gcapi::pfnGetFusedNodePreGraphV4 getPreGraphFunc = TPCFuserSharedObject::instance().getPreGraphFuncPtr();
    GCTPCFuserWrapper                fuser(cluster, *m_graph, fuserFunc, getPreGraphFunc);
    fuser.optimizeCluster(*m_graph);

    const FuserGraphTypeV4& fg = fuser.getOptimizedFuserGraph();

    for (auto node : fg.nodes)
    {
        for (auto edge : node->inputEdges)
        {
            auto targetNode = edge.targetNode.lock();
            LOG_DEBUG(GC,
                      "---{}---->{}",
                      edge.tensor->uniqueIdentifier,
                      (targetNode.get()) ? targetNode.get()->nodeName : "null");
        }

        for (auto edge : node->outputEdges)
        {
            auto targetNode = edge.targetNode.lock();
            LOG_DEBUG(GC,
                      "---{}---->{}",
                      edge.tensor->uniqueIdentifier,
                      (targetNode.get()) ? targetNode.get()->nodeName : "null");
        }
    }

    ASSERT_EQ(fg.nodes.size(), 1);
    ASSERT_EQ(fg.nodes.at(0)->inputEdges.size(), 2);
    ASSERT_EQ(fg.nodes.at(0)->outputEdges.size(), 1);
}

// Testing GCTPCFuserWrapper class
// Test the following graph:
// In1-->XOR--Out1-->and --Out2-->
// In2-->----->
TEST_P(GenericTPCFuserTest, tpcFuserFusion)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();
    const std::string addGUIDString  = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID        = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    TensorPtr IN2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    TensorPtr IN3 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    TensorPtr IN4 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    TensorPtr IN5 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    TensorPtr OUT1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    TensorPtr OUT2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    TensorPtr OUT3 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    TensorPtr OUT4 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    std::unordered_set<NodePtr> cluster;
    NodePtr                     add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, add1);
    cluster.insert(add1);

    NodePtr add2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, add2);
    cluster.insert(add2);

    NodePtr add3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, addGUID, "add3");
    GraphEditor::addNode(*m_graph, add3);
    cluster.insert(add3);

    NodePtr relu4 = NodeFactory::createGenericTPCNode({OUT3}, {OUT4}, nullptr, reluGUID, "relu4");
    GraphEditor::addNode(*m_graph, relu4);
    cluster.insert(relu4);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    gcapi::pfnFuseGraphV4            fuserFunc       = TPCFuserSharedObject::instance().getFuseGraphFuncPtr();
    gcapi::pfnGetFusedNodePreGraphV4 getPreGraphFunc = TPCFuserSharedObject::instance().getPreGraphFuncPtr();
    GCTPCFuserWrapper                fuser(cluster, *m_graph, fuserFunc, getPreGraphFunc);
    fuser.optimizeCluster(*m_graph);

    const FuserGraphTypeV4& fg = fuser.getOptimizedFuserGraph();

    for (auto t : fuser.getInternalTensors())
    {
        LOG_DEBUG(GC, "Internal tensor: {}", t.second->getName());
    }
    for (auto t : fuser.getExternalTensors())
    {
        LOG_DEBUG(GC, "External tensor: {}", t.second->getName());
    }

    ASSERT_EQ(fg.nodes.size(), 1);
    ASSERT_EQ(fg.nodes.at(0)->inputEdges.size(), 3);
    ASSERT_EQ(fg.nodes.at(0)->outputEdges.size(), 1);
    ASSERT_EQ(fuser.getInternalTensors().size(), 3);
    ASSERT_EQ(fuser.getExternalTensors().size(), 4);

    ASSERT_NE(fuser.getExternalTensors().find(IN1->getId()), fuser.getExternalTensors().end());
    ASSERT_NE(fuser.getExternalTensors().find(IN2->getId()), fuser.getExternalTensors().end());
    ASSERT_NE(fuser.getExternalTensors().find(IN3->getId()), fuser.getExternalTensors().end());
    ASSERT_NE(fuser.getExternalTensors().find(OUT4->getId()), fuser.getExternalTensors().end());

    ASSERT_NE(fuser.getInternalTensors().find(OUT1->getId()), fuser.getExternalTensors().end());
    ASSERT_NE(fuser.getInternalTensors().find(OUT2->getId()), fuser.getExternalTensors().end());
    ASSERT_NE(fuser.getInternalTensors().find(OUT3->getId()), fuser.getExternalTensors().end());
}

// Testing GCTPCFuserWrapper class
// Test case: Successive tpc nodes should be fused into a single nodes
//            Tests tensors classification
TEST_P(GenericTPCFuserTest, tpcFuserFusionSuccXors)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    TensorPtr IN2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    TensorPtr IN3 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    TensorPtr IN4 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    TensorPtr IN5 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    TensorPtr OUT1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    TensorPtr OUT2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    TensorPtr OUT3 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    TensorPtr OUT4 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    std::unordered_set<NodePtr> cluster;
    NodePtr                     add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, add1);
    cluster.insert(add1);

    NodePtr add2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, add2);
    cluster.insert(add2);

    NodePtr add3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, addGUID, "add3");
    GraphEditor::addNode(*m_graph, add3);
    cluster.insert(add3);

    NodePtr add4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, addGUID, "add4");
    GraphEditor::addNode(*m_graph, add4);
    cluster.insert(add4);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();

    gcapi::pfnFuseGraphV4            fuserFunc       = TPCFuserSharedObject::instance().getFuseGraphFuncPtr();
    gcapi::pfnGetFusedNodePreGraphV4 getPreGraphFunc = TPCFuserSharedObject::instance().getPreGraphFuncPtr();
    GCTPCFuserWrapper                fuser(cluster, *m_graph, fuserFunc, getPreGraphFunc);
    fuser.optimizeCluster(*m_graph);

    for (auto t : fuser.getInternalTensors())
    {
        LOG_DEBUG(GC, "Internal tensor: {}", t.second->getName());
    }
    for (auto t : fuser.getExternalTensors())
    {
        LOG_DEBUG(GC, "External tensor: {}", t.second->getName());
    }

    ASSERT_EQ(fuser.getOptimizedFuserGraph().nodes.size(), 1);
    ASSERT_EQ(fuser.getInternalTensors().size(), 3);
    ASSERT_EQ(fuser.getExternalTensors().size(), 5);
}

// Testing TPCFuser class
// Test case: A graph with tpc nodes only, a single cluster
TEST_P(GenericTPCFuserTest, tpcFuserBasicPassTest)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    TensorPtr IN2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    TensorPtr IN3 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    TensorPtr IN4 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    TensorPtr IN5 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    TensorPtr OUT1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    TensorPtr OUT2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");
    TensorPtr OUT3 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    TensorPtr OUT4 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    std::unordered_set<NodePtr> cluster;
    NodePtr                     add1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, add1);
    cluster.insert(add1);

    NodePtr add2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, add2);
    cluster.insert(add2);

    NodePtr add3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, addGUID, "add3");
    GraphEditor::addNode(*m_graph, add3);
    cluster.insert(add3);

    NodePtr add4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, addGUID, "add4");
    GraphEditor::addNode(*m_graph, add4);
    cluster.insert(add4);

    tpcFuser(*m_graph);

    for (auto node : (*m_graph).getExeSortedNodes())
    {
        LOG_TRACE(GC, "tpcFuserBasicPassTest: nodeName {}", node->getNodeName());

        for (auto inTensor : node->getInputs())
        {
            LOG_TRACE(GC, "tpcFuserBasicPassTest: node input {}", inTensor->getName());
        }
        for (auto outTensor : node->getOutputs())
        {
            LOG_TRACE(GC, "tpcFuserBasicPassTest: node output {}", outTensor->getName());
        }
    }

    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    ASSERT_EQ((*m_graph).getExeSortedNodes().front()->getOutputs().size(), 1);
    ASSERT_EQ((*m_graph).getExeSortedNodes().front()->getInputs().size(), 4);
}

// Testing TPCFuser class
// Test case: Graph with two tpc-nodes-clusters
TEST_P(GenericTPCFuserTest, tpcFuserMultiNodeOptimizedCluster)
{
    const std::string addGUIDString    = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID          = addGUIDString.c_str();
    const std::string cumsumGUIDString = getGUIDByDevice(GetParam(), "cumsum");
    const char*       cumsumGUID       = cumsumGUIDString.c_str();

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
    t6->setDramOffset(0x11000);
    t6->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    t6->setMemoryDescriptor(persistentMemoryDesc);

    NodePtr add1 = NodeFactory::createGenericTPCNode({t0, t1}, {t2}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, add1);

    NodePtr add2 = NodeFactory::createGenericTPCNode({t0, t2}, {t3}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, add2);

    ns_CumSumKernel::Params cumsumParams = {0, 0, 0};
    NodePtr cumsum3 = NodeFactory::createGenericTPCNode({t3}, {t4}, &cumsumParams, cumsumGUID, "cumsum3");
    GraphEditor::addNode(*m_graph, cumsum3);

    NodePtr add4 = NodeFactory::createGenericTPCNode({t0, t4}, {t5}, nullptr, addGUID, "add4");
    GraphEditor::addNode(*m_graph, add4);

    NodePtr abs5 = NodeFactory::createGenericTPCNode({t0, t5}, {t6}, nullptr, addGUID, "abs5");
    GraphEditor::addNode(*m_graph, abs5);

    bool ret = (*m_graph).compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    /* Verifying that optimized graph has these nodes: {fused_kernel}->{cumsum}->{fused_kernel} */

    unsigned fusedNodesCount  = 0;
    unsigned cumsumNodesCount = 0;

    for (auto node : (*m_graph).getExeSortedNodes())
    {
        if ((*m_graph).runsOnTPC(node) == true)
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
        }
    }
    ASSERT_EQ(fusedNodesCount, 2) << "Expected 2 fused TPC nodes";
    ASSERT_EQ(cumsumNodesCount, 1) << "Expected 1 non-fused TPC nodes (with guid cumsum)";
}

// Testing TPCFuser class
// Test case: Graph with two tpc-nodes-clusters
// the shared object function will be replaced by a local function that returned the same cluster as the optimized graph
TEST_P(GenericTPCFuserTest, tpcFuserOptClusterHasTheSameNode)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    TensorPtr IN1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    TensorPtr IN2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    TensorPtr OUT1 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    TensorPtr OUT2 = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out2");

    NodePtr node1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "node1");
    GraphEditor::addNode(*m_graph, node1);

    NodePtr node2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, addGUID, "node2");
    GraphEditor::addNode(*m_graph, node2);

    TPCClusterConstructor clusterConstructor(*m_graph);
    clusterConstructor.computeClusters();
    std::unordered_set<NodePtr> clusterNodes = clusterConstructor.popNextCluster();

    std::shared_ptr<GCTPCFuserWrapper> fuser(new GCTPCFuserWrapper(clusterNodes,
                                                                   *m_graph,
                                                                   &fuserGraphFuncOptimizedGraphUnchanged,
                                                                   &fuserGraphGetEmptyPreGraph));

    fuser->optimizeCluster(*m_graph);

    replaceOptimizedCluster(*m_graph, fuser);

    /* Verifying that optimized graph has 2 nodes as original graph */

    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 2) << "Expected optimized graph to have 2 nodes";

    /* Verifiting that optimized graph has the different GUIDs from the ondes in the original graph
     * and as should be according to fuserGraphFuncOptimizedGraphUnchanged function*/
    NodeVector nodes = (*m_graph).getExeSortedNodes();
    ASSERT_EQ(nodes[0], node1) << "Expecting node1 to point the same node object";
    ASSERT_EQ(nodes[0]->getId(), node1->getId()) << "Expecting node1 to keep its ID";
    ASSERT_EQ(nodes[1], node2) << "Expecting node2 to point the same node object";
    ASSERT_EQ(nodes[1]->getId(), node2->getId()) << "Expecting node2 to keep its ID";
}

// Testing GCTPCFuserWrapper class
// Test the following graph:
// In1-->XOR--Out1-->and --Out2-->
// Expected result :
// In1-->FusedNode--Out2-->
TEST_P(GenericTPCFuserTest, tpcFuserAndXorParamTest)
{
    const std::string addGUIDString = getGUIDByDevice(GetParam(), "add");
    const char*       addGUID       = addGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];
    char in2[n * w * h * batch];
    char in3[n * w * h * batch];
    char out1[n * w * h * batch];
    char out2[n * w * h * batch];

    fillWithRandom<char>(in1, n * w * h * batch);
    fillWithRandom<char>(in2, n * w * h * batch);
    fillWithRandom<char>(in3, n * w * h * batch);

    memset(out1, 0, sizeof(char) * n * w * h * batch);
    memset(out2, 0, sizeof(char) * n * w * h * batch);

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x7000);
    IN1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    IN1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in2)));
    IN2->setName("in2");
    IN2->setDramOffset(0x9000);
    IN2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    IN2->setMemoryDescriptor(persistentMemoryDesc);
    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in3)));
    IN3->setName("in3");
    IN3->setDramOffset(0x11000);
    IN3->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    IN3->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(out1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(out2)));
    OUT2->setName("out2");
    OUT2->setDramOffset(0x13000);
    OUT2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    OUT2->setMemoryDescriptor(persistentMemoryDesc);

    pNode node1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, addGUID, "add1");
    GraphEditor::addNode(*m_graph, node1);

    pNode node2 = NodeFactory::createGenericTPCNode({OUT1, IN3}, {OUT2}, nullptr, addGUID, "add2");
    GraphEditor::addNode(*m_graph, node2);

    bool ret = (*m_graph).compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned tpcNodesCount = 0;

    for (auto node : (*m_graph).getExeSortedNodes())
    {
        if ((*m_graph).runsOnTPC(node) == true)
        {
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);

            unsigned paramsSize = tpcNode->getParamsSize();
            ASSERT_EQ(paramsSize, 0) << "tpcFuserParamsTest: ERROR! expected no params";

            tpcNodesCount++;
        }
    }

    ASSERT_EQ(tpcNodesCount, 1) << "Expecting 1 fused TPC node";
}

// Testing GCTPCFuserWrapper class
// Test the following graph:
// In1-->relu--Out1-->relu --Out2-->
// Expected result :
// In1-->FusedNode--Out2-->
TEST_P(GenericTPCFuserTest, tpcFuserReluParamsTest)
{
    const std::string reluGUIDString = getGUIDByDevice(GetParam(), "relu");
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch * sizeof(float)];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t1->setName("t1");
    t1->setDramOffset(0x7000);
    t1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    t1->setMemoryDescriptor(persistentMemoryDesc);
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t2->setName("t2");
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    t3->setName("t3");
    t3->setDramOffset(0x8000);
    t3->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    t3->setMemoryDescriptor(persistentMemoryDesc);

    ns_ReluKernel::Params reluParams = {0};
    reluParams.threshold.f           = 2.0;
    pNode relu1                      = NodeFactory::createGenericTPCNode({t1}, {t2}, &reluParams, reluGUID, "relu1");
    GraphEditor::addNode(*m_graph, relu1);

    ns_ReluKernel::Params reluParams2 = {0};
    reluParams2.threshold.f           = 1.0;
    pNode relu2                       = NodeFactory::createGenericTPCNode({t2}, {t3}, &reluParams2, reluGUID, "relu2");
    GraphEditor::addNode(*m_graph, relu2);

    bool ret = (*m_graph).compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned tpcNodesCount = 0;

    for (auto node : (*m_graph).getExeSortedNodes())
    {
        if ((*m_graph).runsOnTPC(node) == true)
        {
            TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);

            unsigned    paramsSize       = tpcNode->getParamsSize();
            std::string tpcFuserLibName  = GCFG_TPC_FUSER_LIB_NAME.value();
            bool        isExpectedParams = tpcFuserLibName != std::string("libTPCFuser.so");
            if (isExpectedParams)
            {
                ASSERT_NE(paramsSize, 0) << "tpcFuserParamsTest: ERROR! paramsSize is 0!";
            }
            else
            {
                ASSERT_EQ(paramsSize, 0) << "tpcFuserParamsTest: ERROR! paramsSize is not 0!";
            }
            tpcNodesCount++;
        }
    }

    ASSERT_EQ(tpcNodesCount, 1) << "Expecting 1 fused TPC node";
}

INSTANTIATE_TEST_SUITE_P(,
                         GenericTPCFuserTest,
                         ::testing::Values(synDeviceGaudi, synDeviceGaudi2),
                         GenericGraphTest::GetName());

gcapi::FuserRetVal_t
fuserGraphFuncOptReturnSameGraph(const FuserGraphTypeV4* graphIn, FuserGraphTypeV4* graphOut, bool debug)
{
    for (const auto& node : graphIn->nodes)
    {
        graphOut->nodes.push_back(node);
    }

    graphOut->deviceId   = graphIn->deviceId;
    graphOut->kernelType = graphIn->kernelType;

    return gcapi::FUSER_SUCCESS;
}

gcapi::GlueCodeReturn_t fuserGraphGetEmptyPreGraph(const FuserNodeTypeV4* nodeIn, FuserGraphTypeV4** graphOut)
{
    return gcapi::GLUE_SUCCESS;
}

gcapi::FuserRetVal_t
fuserGraphFuncOptimizedGraphUnchanged(const FuserGraphTypeV4* graphIn, FuserGraphTypeV4* graphOut, bool debug)
{
    for (const auto& node : graphIn->nodes)
    {
        graphOut->nodes.push_back(node);
    }

    graphOut->deviceId   = graphIn->deviceId;
    graphOut->kernelType = graphIn->kernelType;
    return gcapi::FUSER_SUCCESS;
}
