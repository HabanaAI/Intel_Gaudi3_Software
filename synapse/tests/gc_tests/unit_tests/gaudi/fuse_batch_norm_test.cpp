#include <memory>
#include <gtest/gtest.h>
#include <syn_logging.h>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "habana_pass.h"
#include "types.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "graph_compiler/passes/fuse_batchnorm.cpp"
#include "fuse_batchnorm_patterns.h"

class FuseBatchNormTest : public GraphOptimizerTest
{
protected:
    using BN2Dir = BnStage2PatternFuser::BN2Dir;

    enum BN2Type
    {
        BN2_BF16,
        BN2_F32
    };

    static std::string_view type2Str(BN2Type type)
    {
        switch (type)
        {
            case BN2_BF16:
                return "bf16";
            case BN2_F32:
                return "f32";
            default:
                HB_ASSERT(false, "passed undefined type enum");
                return "";
        }
    }

    pNode createBN2Node(BN2Dir direction, BN2Type type)
    {
        std::string_view dirStr  = BnStage2PatternFuser::dir2Str(direction);
        std::string_view typeStr = type2Str(type);

        pTensor batchNormIfm           = std::make_shared<Tensor>(syn_type_bf16);
        pTensor k                      = std::make_shared<Tensor>(syn_type_bf16);
        pTensor tensorRunningMeanIn    = std::make_shared<Tensor>(syn_type_bf16);
        pTensor sigmaAndSigmaSquaredIn = std::make_shared<Tensor>(syn_type_bf16);
        pTensor meanIn                 = std::make_shared<Tensor>(syn_type_bf16);
        pTensor batchNormOfm           = std::make_shared<Tensor>(syn_type_bf16);
        pTensor runningMeanAndVarOut   = std::make_shared<Tensor>(syn_type_bf16);
        pTensor meanAndStdOut          = std::make_shared<Tensor>(syn_type_bf16);
        pTensor gradIn0                = std::make_shared<Tensor>(syn_type_bf16);
        pTensor sumDotP                = std::make_shared<Tensor>(syn_type_bf16);

        ns_BatchNormStage2Kernel::Params params;
        TensorVector                     inputs, outputs;

        if (direction == BnStage2PatternFuser::BnStage2PatternFuser::BN2_FWD)
        {
            inputs  = {batchNormIfm, k, tensorRunningMeanIn, sigmaAndSigmaSquaredIn, meanIn};
            outputs = {batchNormOfm};  // Removed {runningMeanAndVarOut, meanAndStdOut}
            // to insure the the graph will have a single output tensor
        }
        else
        {
            inputs  = {batchNormIfm, meanIn, batchNormOfm, gradIn0};
            outputs = {sumDotP};
        }

        pNode stage2Node = NodeFactory::createNode(inputs,
                                                   outputs,
                                                   nullptr,
                                                   fmt::format("batch_norm_stage2_{}_{}", dirStr, typeStr),
                                                   "BN2_pattern");

        /* For tests */
        std::dynamic_pointer_cast<TPCNode>(stage2Node)
            ->storeParamsInBuffer(&params, sizeof(ns_BatchNormStage2Kernel::Params));

        return stage2Node;
    }

    bool createPatternBatchNormStage2AddReluFwd(Graph* g, BN2Type type, int order)
    {
        /* BN2-->Add-->Relu */
        bool   status;
        BN2Dir direction = BnStage2PatternFuser::BnStage2PatternFuser::BN2_FWD;

        /* bn2 F16 */
        pNode stage2Node = createBN2Node(direction, type);

        /* add */
        pTensor     addOFM      = std::make_shared<Tensor>(syn_type_bf16);
        pTensor     addInput    = std::make_shared<Tensor>(syn_type_bf16);
        std::string addNodeGuid = fmt::format("add_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));

        TensorVector addInputs0 = {stage2Node->getOutput(0), addInput};
        TensorVector addInputs1 = {addInput, stage2Node->getOutput(0)};
        TensorVector addInputs  = order == 0 ? addInputs0 : addInputs1;
        pNode        addNode    = NodeFactory::createGenericTPCNode(addInputs, {addOFM}, nullptr, addNodeGuid, "");
        /* relu */
        pTensor     reluOFM      = std::make_shared<Tensor>(syn_type_bf16);
        std::string reluNodeGuid = fmt::format("relu_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));

        pNode reluNode = NodeFactory::createGenericTPCNode({addOFM}, {reluOFM}, nullptr, reluNodeGuid, "");
        status         = g->addNode(stage2Node);
        status         = status && g->addNode(addNode);
        status         = status && g->addNode(reluNode);

        return status;
    }

    bool createPatternBatchNormStage2AddReluBwd(Graph* g, BN2Type type)
    {
        /* relu->add->bn2 */
        bool   status;
        BN2Dir direction = BnStage2PatternFuser::BnStage2PatternFuser::BN2_BWD;

        /* bn2 F16 */
        pNode stage2Node = createBN2Node(direction, type);

        /* relu */
        pTensor     reluGradIn   = std::make_shared<Tensor>(syn_type_bf16);
        pTensor     reluGradOut  = std::make_shared<Tensor>(syn_type_bf16);
        std::string reluNodeGuid = fmt::format("relu_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));

        pNode reluNode = NodeFactory::createGenericTPCNode({reluGradIn}, {reluGradOut}, nullptr, reluNodeGuid, "");

        /* add */
        std::string addNodeGuid = fmt::format("add_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));
        pTensor     addGradOut  = std::make_shared<Tensor>(syn_type_bf16);

        pNode addNode =
            NodeFactory::createGenericTPCNode({reluGradOut}, {stage2Node->getInput(3)}, nullptr, addNodeGuid, "");

        status = g->addNode(stage2Node);
        status = status && g->addNode(addNode);
        status = status && g->addNode(reluNode);

        return status;
    }

    bool createPatternBatchNormStage2ReluFwd(Graph* g, BN2Type type)
    {
        bool   status;
        BN2Dir direction = BnStage2PatternFuser::BnStage2PatternFuser::BN2_FWD;

        /* bn2 */
        pNode stage2Node = createBN2Node(direction, type);

        /* relu */
        pTensor reluOFM = std::make_shared<Tensor>(syn_type_bf16);

        const std::string reluGuid =
            fmt::format("relu_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));
        pNode reluNode =
            NodeFactory::createGenericTPCNode({stage2Node->getOutput(0)}, {reluOFM}, nullptr, reluGuid, "");

        status = g->addNode(stage2Node);
        status = status && g->addNode(reluNode);

        return status;
    }

    bool createPatternBatchNormStage2ReluBwd(Graph* g, BN2Type type)
    {
        bool   status;
        BN2Dir direction = BnStage2PatternFuser::BN2_BWD;

        /* bn2 */
        pNode stage2Node = createBN2Node(direction, type);

        /* relu */
        pTensor           reluGradIn = std::make_shared<Tensor>(syn_type_bf16);
        const std::string reluGuid =
            fmt::format("relu_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));
        pNode reluNode =
            NodeFactory::createGenericTPCNode({reluGradIn}, {stage2Node->getInput(3)}, nullptr, reluGuid, "");

        status = g->addNode(stage2Node);
        status = status && g->addNode(reluNode);

        return status;
    }

void correctAddOutputsBwd(GaudiGraph& g, bool isPersistent = false)
{
    for (auto node : g.getExeSortedNodes())
    {
        TPCNodePtr addNode = std::dynamic_pointer_cast<TPCNode>(node);
        if (addNode != nullptr && addNode->getGUIDWithoutDtype() == "add_bwd")
        {
            // add bwd node has 2 output grads, adding the second one
            GraphEditor::removeNode(g, addNode);
            auto secOutput = std::make_shared<Tensor>(syn_type_bf16);
            if (isPersistent)
            {
                synMemoryDescriptor memDesc(true);
                secOutput->setMemoryDescriptor(memDesc);
                secOutput->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
                secOutput->setDramOffset(0x1000);

                addNode->getOutput(0)->setMemoryDescriptor(memDesc);
                addNode->getOutput(0)->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
                addNode->getOutput(0)->setDramOffset(0x2000);
            }
            addNode->addOutput(secOutput);
            GraphEditor::addNode(g, addNode);
            break;
        }
    }
}

void basicPatternAddReluTest(std::string fusedNodeGuidPrefix,
                             BN2Dir      direction,
                             BN2Type     type,
                             int         order = 0,
                             bool        persistentAddOutput = false)
{
    GaudiGraph graph;

    if(fusedNodeGuidPrefix == "batch_norm_stage2_add_relu_")
    {
        if (direction == BnStage2PatternFuser::BnStage2PatternFuser::BN2_FWD)
        {
            createPatternBatchNormStage2AddReluFwd(&graph, type, 0);
        }
        else
        {
            createPatternBatchNormStage2AddReluBwd(&graph, type);
            correctAddOutputsBwd(graph, persistentAddOutput);

            fusedNodeGuidPrefix = "batch_norm_stage2_relu_";
        }
    }
    else if(fusedNodeGuidPrefix == "batch_norm_stage2_relu_")
    {
        if (direction == BnStage2PatternFuser::BnStage2PatternFuser::BN2_FWD)
        {
            createPatternBatchNormStage2ReluFwd(&graph, type);
        }
        else
        {
            createPatternBatchNormStage2ReluBwd(&graph, type);
        }
    }
    else
    {
        FAIL();
    }
    pNode   rootNode  = graph.getRootNodes().front();
    pTensor rootInput = rootNode->getInputs()[0];
    pTensor logInput  = std::make_shared<Tensor>(syn_type_bf16);

    std::string logNodeGuid = fmt::format("log_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));
    pNode       logNode     = NodeFactory::createGenericTPCNode({logInput}, {rootInput}, nullptr, logNodeGuid, "");
    GraphEditor::addNode(graph, logNode);

    graphVisualizationPre(graph);
    fuseBatchNorm(graph);
    removeRedundantLogicalNodes(graph);
    graphVisualizationPost(graph);

    std::string fusedNodeGuid =
        fmt::format("{}{}_{}", fusedNodeGuidPrefix, BnStage2PatternFuser::dir2Str(direction), type2Str(type));

    if (persistentAddOutput)
    {
        // Check an optimization of fuse_batchnorm.
        // The optimization: if the output of add is persistent (the one that is not an input to batch norm)
        // then replace it by its non-persistent copy and plant an identity node between them.
        ASSERT_EQ(graph.getExeSortedNodes().size(), 3);
        auto nodesIterator = graph.getExeSortedNodes().begin();
        ASSERT_EQ((nodesIterator++)->get()->getGUID(), logNodeGuid);
        ASSERT_EQ((nodesIterator++)->get()->getGUID(), fusedNodeGuid);
        ASSERT_EQ((nodesIterator++)->get()->getGUID(), "identity");
    }
    else
    {
        ASSERT_EQ(graph.getExeSortedNodes().size(), 2);
        ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(graph.getExeSortedNodes().front())->getGUID(), logNodeGuid);
        ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(graph.getExeSortedNodes().back())->getGUID(), fusedNodeGuid);
    }
}

bool creatPatternTwoBn2AddRelu(GaudiGraph& g, BN2Dir direction, BN2Type type, int order = 0)
{
    if (direction == BnStage2PatternFuser::BnStage2PatternFuser::BN2_FWD)
    {
        createPatternBatchNormStage2AddReluFwd(&g, type, order);
    }
    else
    {
        createPatternBatchNormStage2AddReluBwd(&g, type);
    }

    for (auto node : g.getExeSortedNodes())
    {
        TPCNodePtr addNode = std::static_pointer_cast<TPCNode>(node);
        if (addNode->getGUIDWithoutDtype() != fmt::format("add_{}", BnStage2PatternFuser::dir2Str(direction)))
        {
            continue;
        }

        pNode bn2Node = createBN2Node(direction, type);
        if (direction == BnStage2PatternFuser::BnStage2PatternFuser::BN2_FWD)
        {
            bn2Node->replaceOutput(0, addNode->getInput(1));
        }
        else
        {
            GraphEditor::removeNode(g, addNode);
            addNode->addOutput(std::make_shared<Tensor>(syn_type_bf16));
            GraphEditor::addNode(g, addNode);
            bn2Node->replaceInput(3, addNode->getOutput(1));
        }
        GraphEditor::addNode(g, bn2Node);

        break;
    }

    return true;
}

void twoBnPatternAddReluTest(BN2Dir direction, BN2Type type)
{
    GaudiGraph graph;
    creatPatternTwoBn2AddRelu(graph, direction, type);

    pNode   rootNode  = graph.getRootNodes().front();
    pTensor rootInput = rootNode->getInputs()[0];

    const std::string bn2Guid =
        fmt::format("batch_norm_stage2_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));
    graphVisualizationPre(graph);
    fuseBatchNorm(graph);
    removeRedundantLogicalNodes(graph);
    graphVisualizationPost(graph);

    const std::string fusedNodeGuid =
        direction == BnStage2PatternFuser::BnStage2PatternFuser::BN2_BWD
            ? fmt::format("batch_norm_stage2_relu_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type))
            : fmt::format("batch_norm_stage2_add_relu_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));

    ASSERT_EQ(graph.getExeSortedNodes().size(), 2);
    if (direction == BnStage2PatternFuser::BN2_BWD)
    {
        ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(graph.getExeSortedNodes().front())->getGUID(), fusedNodeGuid);
        ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(graph.getExeSortedNodes().back())->getGUID(), bn2Guid);
    }
    else
    {
        ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(graph.getExeSortedNodes().front())->getGUID(), bn2Guid);
        ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(graph.getExeSortedNodes().back())->getGUID(), fusedNodeGuid);
    }
}

void bnPatternAddReluAddTestBwd(BN2Type type, bool add)
{
    BN2Dir     direction = BnStage2PatternFuser::BN2_BWD;
    GaudiGraph graph;
    if (add)
    {
        createPatternBatchNormStage2AddReluBwd(&graph, type);
        correctAddOutputsBwd(graph);
    }
    else
    {
        createPatternBatchNormStage2ReluBwd(&graph, type);
    }

    pNode   rootNode  = graph.getRootNodes().front();
    pTensor rootInput = rootNode->getInput(0);
    TensorVector addInputs = {std::make_shared<Tensor>(syn_type_bf16), std::make_shared<Tensor>(syn_type_bf16)};
    std::string  addGuid   = fmt::format("add_fwd_{}", type2Str(type));
    pNode        addNode   = NodeFactory::createGenericTPCNode(addInputs, {rootInput}, nullptr, addGuid, "");
    GraphEditor::addNode(graph, addNode);

    graphVisualizationPre(graph);
    fuseBatchNorm(graph);
    removeRedundantLogicalNodes(graph);
    graphVisualizationPost(graph);

    std::string fusedNodeGuid =
        fmt::format("batch_norm_stage2_add_relu_{}_{}", BnStage2PatternFuser::dir2Str(direction), type2Str(type));

    ASSERT_EQ(graph.getExeSortedNodes().size(), 1);
    ASSERT_EQ(std::dynamic_pointer_cast<TPCNode>(graph.getExeSortedNodes().front())->getGUID(), fusedNodeGuid);
}
};

TEST_F(FuseBatchNormTest, twoBN2TestAddReluFwdBf16)
{
    twoBnPatternAddReluTest(BnStage2PatternFuser::BN2_FWD, BN2_BF16);
}

TEST_F(FuseBatchNormTest, twoBN2TestAddReluBwdBf16)
{
    twoBnPatternAddReluTest(BnStage2PatternFuser::BN2_BWD, BN2_BF16);
}

TEST_F(FuseBatchNormTest, twoBN2TestAddReluFwdF32)
{
    twoBnPatternAddReluTest(BnStage2PatternFuser::BN2_FWD, BN2_F32);
}

TEST_F(FuseBatchNormTest, twoBN2TestAddReluBwdF32)
{
    twoBnPatternAddReluTest(BnStage2PatternFuser::BN2_BWD, BN2_F32);
}

TEST_F(FuseBatchNormTest, bn2AddReluAddBwdF32)
{
    bnPatternAddReluAddTestBwd(BN2_F32, true);
}

TEST_F(FuseBatchNormTest, bn2AddReluAddBwdBf16)
{
    bnPatternAddReluAddTestBwd(BN2_BF16, true);
}

TEST_F(FuseBatchNormTest, bn2AddReluBwdF32)
{
    bnPatternAddReluAddTestBwd(BN2_F32, false);
}

TEST_F(FuseBatchNormTest, bn2AddReluBwdBf16)
{
    bnPatternAddReluAddTestBwd(BN2_BF16, false);
}

TEST_F(FuseBatchNormTest, simpleTestAddReluBwdBf16)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_BWD, BN2_BF16);
}

TEST_F(FuseBatchNormTest, persistentAddOutputTestAddReluBwdBf16)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_BWD, BN2_BF16, 0, true);
}

TEST_F(FuseBatchNormTest, simpleTestAddReluFwdBf16)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_FWD, BN2_BF16);
}

TEST_F(FuseBatchNormTest, simpleTestAddReluBwdF32)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_BWD, BN2_F32);
}

TEST_F(FuseBatchNormTest, simpleTestAddReluFwdF32)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_FWD, BN2_F32);
}

TEST_F(FuseBatchNormTest, simpleTestAdd1ReluBwdBf16)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_BWD, BN2_BF16, 1);
}

TEST_F(FuseBatchNormTest, simpleTestAdd1ReluFwdBf16)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_FWD, BN2_BF16, 1);
}

TEST_F(FuseBatchNormTest, simpleTestAdd1ReluBwdF32)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_BWD, BN2_F32, 1);
}

TEST_F(FuseBatchNormTest, simpleTestAdd1ReluFwdF32)
{
    basicPatternAddReluTest("batch_norm_stage2_add_relu_", BnStage2PatternFuser::BN2_FWD, BN2_F32, 1);
}

TEST_F(FuseBatchNormTest, simpleTestReluBwdBF16)
{
    basicPatternAddReluTest("batch_norm_stage2_relu_", BnStage2PatternFuser::BN2_BWD, BN2_BF16);
}

TEST_F(FuseBatchNormTest, simpleTestReluFwdBF16)
{
    basicPatternAddReluTest("batch_norm_stage2_relu_", BnStage2PatternFuser::BN2_FWD, BN2_BF16);
}

TEST_F(FuseBatchNormTest, simpleTestReluBwdF32)
{
    basicPatternAddReluTest("batch_norm_stage2_relu_", BnStage2PatternFuser::BN2_BWD, BN2_F32);
}

TEST_F(FuseBatchNormTest, simpleTestReluFwdF32)
{
    basicPatternAddReluTest("batch_norm_stage2_relu_", BnStage2PatternFuser::BN2_FWD, BN2_F32);
}
