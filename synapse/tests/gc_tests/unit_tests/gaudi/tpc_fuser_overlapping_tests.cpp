#include "gaudi_graph.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "graph_compiler/passes/tpc_fuser.h"
#include "perf_lib_layer_params.h"

class GaudiTPCFuserOverlappingTest : public GraphOptimizerTest
{
};

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserPersistentTensorsOneClusterGaudi)
{
    // 4 different TPC nodes with persistent tensors
    // No persistent tensors overlap
    // Expected 1 cluster

    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(6);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(8);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(9);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserPersistentTensorsTwoClustersGaudi)
{
    // 6 different TPC nodes with persistent tensors
    // 2 persistent tensors overlap but have reshape node between
    // Expected 2 cluster

    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[]        = {n, w, h, batch};
    const TSize reshapeSizes[] = {n, w, h * batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(6);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(8);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(9);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN5 = pTensor(new Tensor(3U, reshapeSizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    IN5->setDramOffset(0xB000);
    IN5->setMemorySectionID(10);
    IN5->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    pTensor OUT5 = pTensor(new Tensor(3U, reshapeSizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    pTensor OUT6 = pTensor(new Tensor(3U, reshapeSizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT6->setName("out6");
    OUT6->setDramOffset(0xC000);
    OUT6->setMemorySectionID(11);
    OUT6->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT7 = pTensor(new Tensor(3U, reshapeSizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT7->setName("out7");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    pNode reshape1 = NodeFactory::createNode({OUT4}, {OUT5}, nullptr, "reshape", "reshape1");
    GraphEditor::addNode(g, reshape1);

    pNode or5 = NodeFactory::createGenericTPCNode({IN5, OUT5}, {OUT6}, nullptr, "add_fwd_f32", "add5");
    GraphEditor::addNode(g, or5);

    pNode or6 = NodeFactory::createGenericTPCNode({IN5, OUT6}, {OUT7}, nullptr, "add_fwd_f32", "add6");
    GraphEditor::addNode(g, or6);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);
}

// Test case: Graph with two tpc-nodes-clusters
TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserSameAddInOut1Cluster)
{
    // 2 different TPC nodes having 2 persistent tensors with
    // the same section ID and address range (1 input and other output)
    // Expected 1 cluster
    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(6);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(8);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0x4000);
    IN4->setMemorySectionID(7);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT3->setDramOffset(0x1000);
    OUT3->setMemorySectionID(6);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserSameAddInputsTwoClusters)
{
    // 2 different TPC nodes with having 2 persistent inputs tensors with
    // the same section ID and address range
    // Expected 2 clusters
    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[]        = {n, w, h, batch};
    const TSize reshapeSizes[] = {n, w, h * batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(6);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(8);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0x1000);
    IN4->setMemorySectionID(6);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN5 = pTensor(new Tensor(3U, reshapeSizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    IN5->setDramOffset(0xB000);
    IN5->setMemorySectionID(10);
    IN5->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    // Expected 2 clusters duo to 2 inputs duo to the same address
    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserSameTwoOutputsTwoClusters)
{
    // 2 different TPC nodes having 2 persistent outputs tensors with
    // the same section ID and address range
    // Expected 2 clusters
    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(6);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(8);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0x4000);
    IN4->setMemorySectionID(7);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    OUT4->setDramOffset(0xC000);
    OUT4->setMemorySectionID(11);
    OUT4->setMemoryDescriptor(persistentMemoryDesc);

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or4 = NodeFactory::createGenericTPCNode({IN4, OUT3}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    // Expected 2 clusters duo to 2 outputs duo to the same address
    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserUnionTwoClustersJoinOneNode)
{
    // Union of 2 tpc nodes clusters
    // Expected 1 cluster
    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(8);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(6);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(9);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    IN5->setDramOffset(0xB000);
    IN5->setMemorySectionID(10);
    IN5->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    pTensor OUT5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or4 = NodeFactory::createGenericTPCNode({IN4, IN5}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    pNode or5 = NodeFactory::createGenericTPCNode({OUT3, OUT4}, {OUT5}, nullptr, "add_fwd_f32", "add5");
    GraphEditor::addNode(g, or5);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserUnionTwoClustersWithDuplicateInputTensor)
{
    // Union of 2 tpc nodes clusters, tensor IN2 is duplicate
    // Expected 1 cluster
    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(8);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(6);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(9);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    pTensor OUT5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or4 = NodeFactory::createGenericTPCNode({IN4, IN2}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    pNode or5 = NodeFactory::createGenericTPCNode({OUT3, OUT4}, {OUT5}, nullptr, "add_fwd_f32", "add5");
    GraphEditor::addNode(g, or5);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserUnionTwoClustersWithDuplicateInout)
{
    // Union of 2 tpc nodes clusters
    // Expected 1 cluster
    GaudiGraph g;

    const TSize n     = 256;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(8);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(6);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(9);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN5->setName("in5");
    IN5->setDramOffset(0xB000);
    IN5->setMemorySectionID(10);
    IN5->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN6 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    IN6->setName("in6");
    IN6->setDramOffset(0x3000);
    IN6->setMemorySectionID(17);
    IN6->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT4->setName("out4");
    pTensor OUT5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT5->setName("out5");
    pTensor OUT6 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT6->setName("out6");
    OUT6->setDramOffset(0xC000);
    OUT6->setMemorySectionID(14);
    OUT6->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT7 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
    OUT7->setName("out7");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or5 = NodeFactory::createGenericTPCNode({IN4, IN5}, {OUT5}, nullptr, "add_fwd_f32", "add5");
    GraphEditor::addNode(g, or5);

    pNode or6 = NodeFactory::createGenericTPCNode({OUT5, IN6}, {OUT6}, nullptr, "add_fwd_f32", "add6");
    GraphEditor::addNode(g, or6);

    pNode or4 = NodeFactory::createGenericTPCNode({OUT3, OUT6}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 1 cluster
    ASSERT_EQ(numCluster, 1);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserTwoClustersTwoInputsSameAddr)
{
    // Union of 2 tpc nodes clusters with identical 2 inputs persistent tensors
    // Expected 2 clusters
    GaudiGraph g;

    const TSize n = 256;
    const TSize w = 3;
    const TSize h = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(8);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(6);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(9);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN5->setName("in5");
    IN5->setDramOffset(0x1000);
    IN5->setMemorySectionID(8);
    IN5->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN6 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN6->setName("in6");
    IN6->setDramOffset(0x3000);
    IN6->setMemorySectionID(17);
    IN6->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT2->setName("out2");

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT4->setName("out4");
    pTensor OUT5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT5->setName("out5");
    pTensor OUT6 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT6->setName("out6");
    OUT6->setDramOffset(0xC000);
    OUT6->setMemorySectionID(14);
    OUT6->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT7 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT7->setName("out7");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or5 = NodeFactory::createGenericTPCNode({IN4, IN5}, {OUT5}, nullptr, "add_fwd_f32", "add5");
    GraphEditor::addNode(g, or5);

    pNode or6 = NodeFactory::createGenericTPCNode({OUT5, IN6}, {OUT6}, nullptr, "add_fwd_f32", "add6");
    GraphEditor::addNode(g, or6);

    pNode or4 = NodeFactory::createGenericTPCNode({OUT3, OUT6}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);
}

TEST_F(GaudiTPCFuserOverlappingTest, tpcFuserTwoClustersTwoOutputsSameAddr)
{
    // Union of 2 tpc nodes clusters with identical 2 outputs persistent tensors
    // Expected 2 clusters
    GaudiGraph g;

    const TSize n = 256;
    const TSize w = 3;
    const TSize h = 3;
    const TSize batch = 1;

    char in1[n * w * h * batch];

    const TSize sizes[] = {n, w, h, batch};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor IN1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN1->setName("in1");
    IN1->setDramOffset(0x1000);
    IN1->setMemorySectionID(8);
    IN1->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN2->setName("in2");
    IN2->setDramOffset(0x20000);
    IN2->setMemorySectionID(5);
    IN2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN3->setName("in3");
    IN3->setDramOffset(0x9000);
    IN3->setMemorySectionID(6);
    IN3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN4->setName("in4");
    IN4->setDramOffset(0xA000);
    IN4->setMemorySectionID(9);
    IN4->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN5->setName("in5");
    IN5->setDramOffset(0x9000);
    IN5->setMemorySectionID(10);
    IN5->setMemoryDescriptor(persistentMemoryDesc);

    pTensor IN6 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    IN6->setName("in6");
    IN6->setDramOffset(0x3000);
    IN6->setMemorySectionID(17);
    IN6->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT1 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT1->setName("out1");
    pTensor OUT2 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT2->setName("out2");
    OUT2->setDramOffset(0x3000);
    OUT2->setMemorySectionID(12);
    OUT2->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT3 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT3->setName("out3");
    OUT3->setDramOffset(0xC000);
    OUT3->setMemorySectionID(11);
    OUT3->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT4 = pTensor(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT4->setName("out4");
    pTensor OUT5 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT5->setName("out5");
    OUT5->setDramOffset(0x3000);
    OUT5->setMemorySectionID(12);
    OUT5->setMemoryDescriptor(persistentMemoryDesc);
    pTensor OUT6 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT6->setName("out6");
    OUT6->setDramOffset(0xC000);
    OUT6->setMemorySectionID(14);
    OUT6->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OUT7 = pTensor(new Tensor(3U, sizes, syn_type_float, reinterpret_cast<char *>(in1)));
    OUT7->setName("out7");

    std::unordered_set<pNode> cluster;
    pNode or1 = NodeFactory::createGenericTPCNode({IN1, IN2}, {OUT1}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, or1);

    pNode or2 = NodeFactory::createGenericTPCNode({IN2, OUT1}, {OUT2}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, or2);

    pNode or3 = NodeFactory::createGenericTPCNode({IN3, OUT2}, {OUT3}, nullptr, "add_fwd_f32", "add3");
    GraphEditor::addNode(g, or3);

    pNode or5 = NodeFactory::createGenericTPCNode({IN4, IN5}, {OUT5}, nullptr, "add_fwd_f32", "add5");
    GraphEditor::addNode(g, or5);

    pNode or6 = NodeFactory::createGenericTPCNode({OUT5, IN6}, {OUT6}, nullptr, "add_fwd_f32", "add6");
    GraphEditor::addNode(g, or6);

    pNode or4 = NodeFactory::createGenericTPCNode({OUT3, OUT6}, {OUT4}, nullptr, "add_fwd_f32", "add4");
    GraphEditor::addNode(g, or4);

    TPCClusterConstructor clusterConstructor(g);
    clusterConstructor.computeClusters();

    unsigned numCluster = clusterConstructor.getNumOfClusters();

    // Expected 2 clusters
    ASSERT_EQ(numCluster, 2);
}

class TPCFuserComplexGuidClusteringPersistentOverlapTest
: public GaudiTPCFuserOverlappingTest
, public testing::WithParamInterface<bool>
{
};

TEST_P(TPCFuserComplexGuidClusteringPersistentOverlapTest, DISABLED_norm_moments_clustering_persistent_overlap)
{
    bool           isPartialOverlap = GetParam();
    GaudiGraph     g;
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

    pTensor In = pTensor(new Tensor(5U, inSizes, syn_type_float, reinterpret_cast<char*>(data)));

    synMemoryDescriptor persistentMemoryDesc(true);
    In->setName("in");
    In->setMemoryDescriptor(persistentMemoryDesc);
    In->setDramOffset(0x1000);
    In->setMemorySectionID(8);
    pTensor outVariance = pTensor(new Tensor(5U, outSizes, syn_type_float, reinterpret_cast<char*>(outData)));
    outVariance->setName("variance");
    outVariance->setMemoryDescriptor(persistentMemoryDesc);
    unsigned outVarianceSectionOffset = 0x10000;
    unsigned outVarianceSectionId     = 9;
    outVariance->setDramOffset(outVarianceSectionOffset);
    outVariance->setMemorySectionID(outVarianceSectionId);
    pTensor outMean = pTensor(new Tensor(5U, outSizes, syn_type_float, reinterpret_cast<char*>(outData)));
    outMean->setName("mean");
    outMean->setMemoryDescriptor(persistentMemoryDesc);
    unsigned outMeanOffset = isPartialOverlap ? outVarianceSectionOffset + 10 : outVarianceSectionOffset;
    outMean->setDramOffset(outMeanOffset);              // outputs  overlap
    outMean->setMemorySectionID(outVarianceSectionId);  // same section id as variance

    ns_NormMomentsKernel::Params params;
    params.NormAxisBmp = 6;

    const char guid[]          = "norm_moments_fwd_f32";
    NodePtr    normMomentsNode = NodeFactory::createGenericTPCNode({In}, {outMean, outVariance}, &params, guid, "nm");

    ASSERT_TRUE(GraphEditor::addNode(g, normMomentsNode));
    ASSERT_EQ((g).getExeSortedNodes().size(), 1);
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(g));
    ASSERT_TRUE((g).getExeSortedNodes().size() > 1);
    auto nodeId = normMomentsNode->getId();

    try
    {
        tpcFuser((g));
        FAIL() << "Expected exception after persistent overlap check";
    }
    catch (std::exception& exception)
    {
        ASSERT_TRUE(std::string(exception.what())
                        .find(fmt::format("Cluster from complex guid Id {} has overlapping persistent tensors",
                                          nodeId)) != std::string::npos)
            << "Expected specific persistent overlap exception";
    }
}

/*
 * test param - isPartialOverlap.
 * Both test cases should throw an exception because of persistent tensor overlap.
 * When true  - test case is a persistent tensor which is partialy overlaped by another tensor (not allowed).
 * When false -  test case is a persistent tensor which is fully overlaped by another persistent tensor,
 *               therefore we have 2 persistent tensors which are fully overlapped, (only one is allowed)
 */
INSTANTIATE_TEST_SUITE_P(, TPCFuserComplexGuidClusteringPersistentOverlapTest, ::testing::Values(true, false));
