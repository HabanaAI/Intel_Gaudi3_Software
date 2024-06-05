#include "gaudi_graph.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"

class BatchNormMemcopyFuse : public GraphOptimizerTest
{
    void SetUp() { GraphOptimizerTest::SetUp(); setGlobalConfForTest(GCFG_ENABLE_BATCH_NORM_MEMCPY_FUSION, "true"); }
};

/*
                              ^
                              |
                 +-------+    |
       +-------->+memcopy+----+
       |         +-------+
+---------------+            +---------------+
|InputFeatureMap+----------->+BatchNormStage1|
+---------------+            +---------------+
*/

TEST_F(BatchNormMemcopyFuse, fuse_ifm_memcopy_fwd)
{
    GaudiGraph graph;
    pTensor    inputFeatureMapTensor                              = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    meanTensor                                         = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    copiedFeatureMapTensor                             = std::make_shared<Tensor>(syn_type_bf16);
    copiedFeatureMapTensor->getTensorAnnotation().memory.location = TENSOR_IN_DRAM;
    pTensor outputSigmas                                          = std::make_shared<Tensor>(syn_type_bf16);

    pNode batchNormNode = NodeFactory::createNode({inputFeatureMapTensor, meanTensor}, {outputSigmas},
            nullptr, "batch_norm_stage1_fwd_bf16", "batchNormNode");

    pNode memCopyNode = NodeFactory::createNode({inputFeatureMapTensor}, {copiedFeatureMapTensor},
                                                  nullptr, NodeFactory::memcpyNodeTypeName, "memCopyNode");

    GraphEditor::addNode(graph, batchNormNode);
    GraphEditor::addNode(graph, memCopyNode);
    fuseBatchNormMemCpy(graph);
    //MemCopy and BN should be fused
    ASSERT_EQ(graph.getNumNodes(), 1);
    //Inputs should remain untouched
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInputs().size(), 2);
    //A second output should be added
    ASSERT_EQ(graph.getExeSortedNodes().front()->getOutputs().size(), 2);
    //It should be equal to the output of memcpy
    ASSERT_EQ(graph.getExeSortedNodes().front()->getOutput(1).get(), copiedFeatureMapTensor.get());
}
/*
                                     +-------+
                                     |Memcopy|
                                     +---+---+
                                         ^
+---------+   +----------------+         |
|Batchnorm+-->+OutputFeatureMap+---------+
+---------+   +----------------+         |
                                         v
                                     +---+----+
                                     |NextNode|
                                     +--------+

*/
TEST_F(BatchNormMemcopyFuse, fuse_ofm_memcopy_fwd)
{
    GaudiGraph graph;
    pTensor    inputFeatureMapTensor = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    meanTensor            = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    sigmasTensor          = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    betaGammaTensor       = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    RunningMeanVarTensor  = std::make_shared<Tensor>(syn_type_bf16);

    pTensor outputFeatureMapTensor = std::make_shared<Tensor>(syn_type_bf16);
    pTensor runningMeanVarTensor   = std::make_shared<Tensor>(syn_type_bf16);
    pTensor IstdTensor             = std::make_shared<Tensor>(syn_type_bf16);

    pTensor copiedOutputFeatureMapTensor                                = std::make_shared<Tensor>(syn_type_bf16);
    copiedOutputFeatureMapTensor->getTensorAnnotation().memory.location = TENSOR_IN_DRAM;

    pNode batchNormNode = NodeFactory::createNode({inputFeatureMapTensor, meanTensor,
                                                   sigmasTensor, betaGammaTensor, RunningMeanVarTensor},
                                                   {outputFeatureMapTensor, runningMeanVarTensor, IstdTensor},
                                                   nullptr, "batch_norm_stage2_fwd_bf16", "batchNormNode");

    pNode memCopyNode = NodeFactory::createNode({outputFeatureMapTensor}, {copiedOutputFeatureMapTensor},
                                                nullptr, NodeFactory::memcpyNodeTypeName, "memCopyNode");

    GraphEditor::addNode(graph, batchNormNode);
    GraphEditor::addNode(graph, memCopyNode);
    fuseBatchNormMemCpy(graph);
    //MemCopy and BN should be fused
    ASSERT_EQ(graph.getNumNodes(), 1);
    //Inputs should remain untouched
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInputs().size(), 5);
    //A new output should be added
    ASSERT_EQ(graph.getExeSortedNodes().front()->getOutputs().size(), 4);
    //It should be equal to the output of memcpy
    ASSERT_EQ(graph.getExeSortedNodes().front()->getOutput(1).get(), copiedOutputFeatureMapTensor.get());
}

/*
                                     +-------+
                                     |Memcopy|
                                     +---+---+
                                         ^
+---------+   +----------------+         |
|Batchnorm+-->+     gradOut    +---------+
+---------+   +----------------+         |
                                         v
                                     +---+----+
                                     |NextNode|
                                     +--------+

*/
TEST_F(BatchNormMemcopyFuse, fuse_grad_memcopy_bwd)
{
    GaudiGraph graph;
    pTensor    inputFeatureMapTensor = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    meanAndIstdTensor     = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    sumDotpTensor         = std::make_shared<Tensor>(syn_type_bf16);
    pTensor    betaGammaTensor       = std::make_shared<Tensor>(syn_type_bf16);

    pTensor gradOutTensor       = std::make_shared<Tensor>(syn_type_bf16);
    pTensor gradBetaGammaTensor = std::make_shared<Tensor>(syn_type_bf16);

    pTensor copiedGradOutTensor                                = std::make_shared<Tensor>(syn_type_bf16);
    copiedGradOutTensor->getTensorAnnotation().memory.location = TENSOR_IN_DRAM;

    pNode batchNormNode = NodeFactory::createNode({inputFeatureMapTensor, meanAndIstdTensor,
                                                   sumDotpTensor, betaGammaTensor, betaGammaTensor},
                                                  {gradOutTensor, gradBetaGammaTensor},
                                                  nullptr, "batch_norm_stage1_bwd_bf16", "batchNormNode");

    pNode memCopyNode = NodeFactory::createNode({gradOutTensor}, {copiedGradOutTensor},
                                                nullptr, NodeFactory::memcpyNodeTypeName, "memCopyNode");

    GraphEditor::addNode(graph, batchNormNode);
    GraphEditor::addNode(graph, memCopyNode);
    fuseBatchNormMemCpy(graph);
    //MemCopy and BN should be fused
    ASSERT_EQ(graph.getNumNodes(), 1);
    //Inputs should remain untouched
    ASSERT_EQ(graph.getExeSortedNodes().front()->getInputs().size(), 5);
    //A new output should be added
    ASSERT_EQ(graph.getExeSortedNodes().front()->getOutputs().size(), 3);
    //It should be equal to the output of memcpy
    ASSERT_EQ(graph.getExeSortedNodes().front()->getOutput(1).get(), copiedGradOutTensor.get());
}
