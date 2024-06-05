#include "gaudi_graph.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "tpc_kernel_names.h"
#include "perf_lib_layer_params.h"


class GaudiSplitBatchNormTest : public GraphOptimizerTest
{

};

void split_bn_test_fwd(synDataType dtype, bool isTraining)
{
    GaudiGraph g;
    unsigned   channel = 3, height = 32, width = 32, batch = 8;

    SizeArray           ifmSizes    = {channel, height, width, batch};
    SizeArray           oneDimSizes = {channel};
    synMemoryDescriptor persistentMemoryDesc(true);

    auto kernelGuidStr = fmt::format("batch_norm{}{}", dir2Str(FWD), type2Str(dtype));

    pTensor IFM = pTensor(new Tensor(4U, ifmSizes.data(), dtype));
    IFM->setName("IFM");
    IFM->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    IFM->setMemoryDescriptor(persistentMemoryDesc);
    pTensor betaIn = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    betaIn->setName("BetaIn");
    betaIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    betaIn->setMemoryDescriptor(persistentMemoryDesc);
    pTensor gammaIn = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    gammaIn->setName("gammaIn");
    gammaIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    gammaIn->setMemoryDescriptor(persistentMemoryDesc);
    pTensor runningMeanIn = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    runningMeanIn->setName("runningMeanIn");
    runningMeanIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    runningMeanIn->setMemoryDescriptor(persistentMemoryDesc);
    pTensor runningVarIn = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    runningVarIn->setName("runningVarIn");
    runningVarIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    runningVarIn->setMemoryDescriptor(persistentMemoryDesc);

    pTensor OFM = pTensor(new Tensor(4U, ifmSizes.data(), dtype));
    OFM->setName("OFM");
    OFM->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);
    OFM->setMemoryDescriptor(persistentMemoryDesc);
    pTensor savedMeanOut = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    savedMeanOut->setName("savedMeanOut");
    savedMeanOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 7);
    savedMeanOut->setMemoryDescriptor(persistentMemoryDesc);
    pTensor iStdOut = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    iStdOut->setName("iStdOut");
    iStdOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 8);
    iStdOut->setMemoryDescriptor(persistentMemoryDesc);

    ns_BatchNormKernel::ParamsV2 bnParams;
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon     = 1e-05;
    bnParams.isTraining  = isTraining;

    pNode bnNode = NodeFactory::createNode({IFM, betaIn, gammaIn, runningMeanIn, runningVarIn},
                                           {OFM, savedMeanOut, iStdOut},
                                           &bnParams,
                                           kernelGuidStr.c_str(),
                                           "bn_node");
    GraphEditor::addNode(g, bnNode);

    //bool retVal = gaudi::splitBatchNorm(g);

    bool retVal = g.compile();
    ASSERT_EQ(retVal, true) << "Failed compilation";

    // Checking for st1 and st2 existence
    bool     st1Exist = false;
    bool     st2Exist = false;

    for (const NodePtr& node : g.getExeSortedNodes())
    {
        std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        if (tpcNode == nullptr)
            continue;

        if (tpcNode->getGUID().find("batch_norm_stage1") != std::string::npos)
        {
            st1Exist = true;
        }

        if (tpcNode->getGUID().find("batch_norm_stage2") != std::string::npos)
        {
            st2Exist = true;
        }

    }

    if (isTraining)
    {
        ASSERT_EQ(st1Exist, true) << "missing stage1 kernel in post graph";
    }
    else
    {
        ASSERT_EQ(st1Exist, false) << "unexpected stage1 kernel for inference in post graph";
    }
    ASSERT_EQ(st2Exist, true) << "missing stage2 kernel in post graph";
}

TEST_F(GaudiSplitBatchNormTest, split_bn_fwd_bf16_training)
{
    split_bn_test_fwd(syn_type_bf16, true);
}

TEST_F(GaudiSplitBatchNormTest, split_bn_fwd_f32_training)
{
    split_bn_test_fwd(syn_type_float, true);
}

TEST_F(GaudiSplitBatchNormTest, split_bn_fwd_bf16_no_training)
{
    split_bn_test_fwd(syn_type_bf16, false);
}

TEST_F(GaudiSplitBatchNormTest, split_bn_fwd_f32_no_training)
{
    split_bn_test_fwd(syn_type_float, false);
}

void split_bn_test_bwd(synDataType dtype, bool isTraining)
{
    GaudiGraph g;
    unsigned   channel = 3, height = 32, width = 32, batch = 8;

    SizeArray           ifmSizes    = {channel, height, width, batch};
    SizeArray           oneDimSizes = {channel};
    synMemoryDescriptor persistentMemoryDesc(true);

    auto kernelGuidStr = fmt::format("batch_norm{}{}", dir2Str(BWD), type2Str(dtype));

    pTensor IFM = pTensor(new Tensor(4U, ifmSizes.data(), dtype));
    IFM->setName("IFM");
    IFM->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    IFM->setMemoryDescriptor(persistentMemoryDesc);

    pTensor gradIn = pTensor(new Tensor(4U, ifmSizes.data(), dtype));
    gradIn->setName("gradIn");
    gradIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    gradIn->setMemoryDescriptor(persistentMemoryDesc);

    pTensor runningMeanIn = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    runningMeanIn->setName("runningMeanIn");
    runningMeanIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    runningMeanIn->setMemoryDescriptor(persistentMemoryDesc);

    pTensor runningIstdIn = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    runningIstdIn->setName("runningIstdIn");
    runningIstdIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    runningIstdIn->setMemoryDescriptor(persistentMemoryDesc);

    pTensor gammaIn = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    gammaIn->setName("gammaIn");
    gammaIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    gammaIn->setMemoryDescriptor(persistentMemoryDesc);

    pTensor gradOut = pTensor(new Tensor(4U, ifmSizes.data(), dtype));
    gradOut->setName("gradOut");
    gradOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);
    gradOut->setMemoryDescriptor(persistentMemoryDesc);

    pTensor gradBeta = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    gradBeta->setName("gradBeta");
    gradBeta->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 7);
    gradBeta->setMemoryDescriptor(persistentMemoryDesc);

    pTensor gradGamma = pTensor(new Tensor(1U, oneDimSizes.data(), syn_type_float));
    gradGamma->setName("gradGamma");
    gradGamma->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 8);
    gradGamma->setMemoryDescriptor(persistentMemoryDesc);

    ns_BatchNormKernel::ParamsV2 bnParams;
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon     = 1e-05;
    bnParams.isTraining  = isTraining;

    pNode bnNode = NodeFactory::createNode({IFM, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                                           {gradOut, gradBeta, gradGamma},
                                           &bnParams,
                                           kernelGuidStr.c_str(),
                                           "bn_node");
    GraphEditor::addNode(g, bnNode);

    bool retVal = g.compile();
    ASSERT_EQ(retVal, true) << "Failed compilation";

    // Checking for st1 and st2 existence
    const NodeVector& nodes       = g.getExeSortedNodes();
    bool     st1Exist = false;
    bool     st2Exist = false;
    bool     st1Training = false;
    bool     st2Training = false;

    for (const NodePtr& node : nodes)
    {
        std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        if (tpcNode == nullptr)
            continue;

        if (tpcNode->getGUID().find("batch_norm_stage1") != std::string::npos)
        {
            st1Exist = true;
            ns_BatchNormStage1Kernel::ParamsV2* params = static_cast<ns_BatchNormStage1Kernel::ParamsV2*>(tpcNode->getParams());
            st1Training = params->isTraining;
        }

        if (tpcNode->getGUID().find("batch_norm_stage2") != std::string::npos)
        {
            st2Exist = true;
            ns_BatchNormStage2Kernel::ParamsV2* params = static_cast<ns_BatchNormStage2Kernel::ParamsV2*>(tpcNode->getParams());
            st2Training = params->isTraining;
        }

    }

    ASSERT_EQ(st1Exist, true) << "missing stage1 kernel in post graph";
    ASSERT_EQ(st2Exist, true) << "missing stage2 kernel in post graph";
    ASSERT_EQ(st1Training, isTraining) << "stage1 kernel training param isn't correct in post graph";
    ASSERT_EQ(st2Training, isTraining) << "stage2 kernel training param isn't correct in post graph";
}

TEST_F(GaudiSplitBatchNormTest, split_bn_bwd_bf16_training)
{
    split_bn_test_bwd(syn_type_bf16, true);
}

TEST_F(GaudiSplitBatchNormTest, split_bn_bwd_f32_training)
{
    split_bn_test_bwd(syn_type_float, true);
}

TEST_F(GaudiSplitBatchNormTest, split_bn_bwd_bf16_no_training)
{
    split_bn_test_bwd(syn_type_bf16, false);
}

TEST_F(GaudiSplitBatchNormTest, split_bn_bwd_f32_no_training)
{
    split_bn_test_bwd(syn_type_float, false);
}
