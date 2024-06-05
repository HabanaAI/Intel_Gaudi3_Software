#include "fuse_broadcast_bgemm_test.h"
#include "gaudi2_graph.h"
#include "graph.h"
#include "graph_editor.h"
#include "habana_global_conf.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "log_manager.h"
#include "node_utils.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "types.h"
#include "define_synapse_common.hpp"
#include "node_factory.h"
#include "fuse_broadcast.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <memory>

void FuseBroadcastAndBGEMMParametrizedTest::SetUp()
{
    GraphOptimizerTest::SetUp();
    setGlobalConfForTest(GCFG_ENABLE_FUSE_BROADCAST_BGEMM, "true");
}

void FuseBroadcastAndBGEMMParametrizedTest::TearDown()
{
    GraphOptimizerTest::TearDown();
}

void FuseBroadcastAndBGEMM::setAsPersistent(TensorPtr& tensor, unsigned tensorsCount)
{
    static synMemoryDescriptor memDesc(true);
    tensor->setMemoryDescriptor(memDesc);
    tensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + tensorsCount);
}

TensorPtr FuseBroadcastAndBGEMM::createTensor(const std::string& name,
                                              const SizeArray&   tensorSize,
                                              const synDataType  dataType,
                                              bool               isPersistent)
{
    auto t = std::make_shared<Tensor>(tensorSize.size(), tensorSize.data(), dataType);
    t->setName(name);
    if (isPersistent) setAsPersistent(t, m_tensorCount++);
    return t;
}

NodePtr FuseBroadcastAndBGEMM::createBroadcast(const SizeArray&  inputSize,
                                               const SizeArray&  outputSize,
                                               const synDataType dataType)
{
    const auto& input  = createTensor("broadcastInput", inputSize, dataType, true);
    const auto& output = createTensor("broadcastOutput", outputSize, dataType);
    return NodeFactory::createNode({input}, {output}, nullptr, 0, "broadcast", "broadcast");
}

NodePtr FuseBroadcastAndBGEMM::createCast(const TensorPtr& castInput)
{
    static std::string oneCast[]              = {"cast_bf16_to_f32"};
    static synDataType oneCastOutputDtype[]   = {syn_type_single};
    static std::string twoCasts[]             = {"cast_bf16_to_f16", "cast_f16_to_f32"};
    static synDataType twoCastsOutputsDtype[] = {syn_type_fp16, syn_type_single};

    const std::string* castGuidArray;
    const synDataType* castTypesArray;
    if (m_numOfTotalCastsNodes == 2)
    {
        castGuidArray  = twoCasts;
        castTypesArray = twoCastsOutputsDtype;
    }
    else if (m_numOfTotalCastsNodes == 1)
    {
        castGuidArray  = oneCast;
        castTypesArray = oneCastOutputDtype;
    }
    else
    {
        HB_ASSERT(0, "Expecting <= 2 casts in the tests");
    }

    const auto& output = createTensor(fmt::format("cast{}_output", m_castNodesCounter).c_str(),
                                      castInput->getAllSizesInElements(),
                                      castTypesArray[m_castNodesCounter]);

    const auto& cast = NodeFactory::createNode({castInput},
                                               {output},
                                               nullptr,
                                               0,
                                               castGuidArray[m_castNodesCounter],
                                               fmt::format("cast{}", m_castNodesCounter).c_str());
    m_castNodesCounter++;
    return cast;
}

NodePtr FuseBroadcastAndBGEMM::createReshape(const TensorPtr& reshapeInput)
{
    const auto& output = createTensor("reshape_output", m_reshapeOutSizes, syn_type_single);
    return NodeFactory::createNode({reshapeInput}, {output}, nullptr, 0, "reshape", "reshape");
}

void FuseBroadcastAndBGEMM::buildGraphWithBroadcastFusedPattern(HabanaGraph& g)
{
    const auto bCastType = m_numOfTotalCastsNodes > 0 ? syn_type_bf16 : syn_type_single;
    m_broadcast          = createBroadcast(m_broadcast0InSize, m_broadcastOutSize, bCastType);
    EXPECT_TRUE(GraphEditor::addNode(g, m_broadcast));

    auto bgemmFirstInput = m_broadcast->getOutput(0);
    if (m_numOfTotalCastsNodes > 0)
    {
        auto newCastInput = bgemmFirstInput;
        for (int i = 0; i < m_numOfTotalCastsNodes; ++i)
        {
            m_casts.push_back(createCast(newCastInput));
            EXPECT_TRUE(GraphEditor::addNode(g, m_casts.back()));
            newCastInput = m_casts.back()->getOutput(0);
        }
        bgemmFirstInput = newCastInput;
    }

    if (m_reshapeOutSizes != SizeArray({0}))
    {
        m_reshape = createReshape(bgemmFirstInput);
        EXPECT_TRUE(GraphEditor::addNode(g, m_reshape));
        bgemmFirstInput = m_reshape->getOutput(0);
    }

    const auto& bgemmOtherInput = createTensor("bgemmOtherInput",
                                                 bgemmFirstInput->getAllSizesInElements(),
                                                 bgemmFirstInput->getElementType(),
                                                 true);
    const auto& bgemmOutput =
        createTensor("bgemmOutput", bgemmFirstInput->getAllSizesInElements(), bgemmFirstInput->getElementType(), true);
    m_batchGemm = NodeFactory::createNode({bgemmFirstInput, bgemmOtherInput},
                                          {bgemmOutput},
                                          nullptr,
                                          0,
                                          "batch_gemm",
                                          "batch_gemm");
    EXPECT_TRUE(GraphEditor::addNode(g, m_batchGemm));
}

bool FuseBroadcastAndBGEMM::checkThatInGraph(const HabanaGraph& g, const NodePtr& n)
{
    return g.getNodes().find(n) != g.getNodes().cend();
}

unsigned FuseBroadcastAndBGEMM::countLogicalReshapes(const HabanaGraph& g)
{
    return std::count_if(g.getNodes().cbegin(), g.getNodes().cend(), [](const NodePtr& n) {
        return isLogicalReshape(n);
    });
}

static bool isConsumerOf(const HabanaGraph& g, const NodePtr& n1, const NodePtr& n2)
{
    const auto& consumers = g.getNodeConsumers(n2);
    return consumers.find(n1) != consumers.end();
}

static bool isProducerOf(const HabanaGraph& g, const NodePtr& n1, const NodePtr& n2)
{
    const auto& producers = g.getNodeProducers(n2);
    return producers.find(n1) != producers.end();
}

TEST_P(FuseBroadcastAndBGEMMParametrizedTest, broadcast_is_fused_to_bgemm)
{
    Gaudi2Graph g2;
    buildGraphWithBroadcastFusedPattern(g2);

    auto       nodesSizeBeforeFuse         = g2.getNodes().size();
    const auto bgemmOutputSizesBeforeFuse  = m_batchGemm->getOutput(0)->getAllSizesInElements();
    const auto bgemmOtherOpSizesBeforeFuse = m_batchGemm->getInput(1)->getAllSizesInElements();
    SizeArray  origReshapeInputSizes       = {0};
    if (m_reshape)
    {
        origReshapeInputSizes = m_reshape->getInput(0)->getAllSizesInElements();
    }

    ASSERT_TRUE(fuseBroadcast(g2)) << "Expecting that the pass finishes its job successfully";

    // If there was a reshape in the producer chain before the fuse it should have been replaced by two other reshapes.
    // And since the broadcast node should have been removed, there should be the same number of nodes right now.
    auto        nodesSizeAfterFuse = m_reshape ? nodesSizeBeforeFuse : nodesSizeBeforeFuse - 1;
    const auto& allNodes           = g2.getNodes();
    ASSERT_TRUE(allNodes.size() == nodesSizeAfterFuse)
        << fmt::format("Expecting this amount of nodes after fuse: {}", nodesSizeAfterFuse);
    if (m_reshape)
    {
        ASSERT_TRUE(!checkThatInGraph(g2, m_reshape) && countLogicalReshapes(g2) == 2)
            << "Expecting that the original reshape was replaced with two other reshapes";
    }
    for (const auto& n : allNodes)
    {
        if (n->getNodeType() == Node::TYPE_BROADCAST)
        {
            FAIL() << "Broadcast wasn't fused to batch gemm as expected";
        }
        if (n->getNodeType() == Node::TYPE_BATCH_GEMM)
        {
            // This assert is important because m_batchGemm is used in the reshapes check
            ASSERT_EQ(m_batchGemm, n) << "Expecting that it is the same node";

            const auto& bgemm = std::dynamic_pointer_cast<BatchGemmNode>(n);
            ASSERT_TRUE(bgemm && bgemm->validateNodeLayout()) << "Expecting that bgemm layout validation succeeds";
            ASSERT_FALSE(bgemm->isSymmetricLayout())
                << "Expecting that bgemm is asymmetric since the broadcast was fused to it";
            if (m_reshape)
            {
                // If there was a reshape, the bgemm output has been changed in the fusion pass,
                // so the new output sizes should be as the reshape input sizes (=m_broadcastOutSize).
                ASSERT_EQ(bgemm->getOutput(0)->getAllSizesInElements(), m_broadcastOutSize)
                    << "Expecting that the bgemm output sizes are like the original reshape output sizes";
            }
        }
        if (m_reshape)
        {
            // Original reshape should have been replaced with two other reshapes (it is explained in the pass header):
            // 1. A reshape to the other operand 2. A reshape to the bgemm output.
            if (isLogicalReshape(n))
            {
                if (isConsumerOf(g2, m_batchGemm, n))
                {
                    // The reshape of the other operand (should be an inverse to the original reshape)
                    // reshape -> bgemm
                    ASSERT_EQ(n->getInput(0)->getAllSizesInElements(), bgemmOtherOpSizesBeforeFuse)
                        << "Expecting that this reshape input is the other operand of the original bgemm";
                    ASSERT_EQ(n->getOutput(0)->getAllSizesInElements(), origReshapeInputSizes)
                        << "Expecting that this reshape is an inverse to the original reshape";
                }
                else if (isProducerOf(g2, m_batchGemm, n))
                {
                    // The reshape to the original bgemm output
                    // bgemm -> reshape
                    ASSERT_EQ(n->getInput(0)->getAllSizesInElements(), origReshapeInputSizes)
                        << "Expecting that bgemm output sizes were changed";
                    ASSERT_EQ(n->getOutput(0)->getAllSizesInElements(), bgemmOutputSizesBeforeFuse)
                        << "Expecting that the sizes of the output of the new reshape identical to the output of the "
                           "original bgemm";
                }
                else
                {
                    FAIL() << "Not expected reshape";
                }
            }
        }
    }
}

/*
    The following parameters create these kind of patterns:
    broadcast one/multiple dims -> non broadcasted bgemm
    broadcast one/multiple dims -> reshape -> non broadcasted bgemm
    broadcast one/multiple dims -> cast -> non broadcasted bgemm
    broadcast one/multiple dims -> cast -> reshape -> non broadcasted bgemm
    broadcast one/multiple dims -> cast -> cast -> non broadcasted bgemm
    broadcast one/multiple dims -> cast -> cast -> reshape -> non broadcasted bgemm
*/

INSTANTIATE_TEST_SUITE_P(,
                         FuseBroadcastAndBGEMMParametrizedTest,
                         testing::Combine(testing::Values(SizeArray({16, 16, 1, 8, 12})),
                                          testing::Values(SizeArray({16, 16, 4, 8, 12})),
                                          testing::Values(0, 1, 2),
                                          testing::Values(SizeArray({0}) /*no reshape*/,
                                                          SizeArray({16, 16, 32, 1, 12}),
                                                          SizeArray({16, 16, 8, 2, 24}))));

INSTANTIATE_TEST_SUITE_P(
    broadcastOnManyDims,
    FuseBroadcastAndBGEMMParametrizedTest,
    testing::Combine(testing::Values(SizeArray({16, 16, 1, 1, 1})),
                     testing::Values(SizeArray({16, 16, 4, 8, 12})),
                     testing::Values(1),
                     testing::Values(SizeArray({16, 16, 32, 1, 12}))));

void FuseBroadcastAndBGEMMTest::SetUp()
{
    GraphOptimizerTest::SetUp();
    setGlobalConfForTest(GCFG_ENABLE_FUSE_BROADCAST_BGEMM, "true");
}

void FuseBroadcastAndBGEMMTest::TearDown()
{
    GraphOptimizerTest::TearDown();
}

TEST_F(FuseBroadcastAndBGEMMTest, broadcast_is_fused_when_bgemm_other_op_is_broadcasted_on_other_dims)
{
    Gaudi2Graph           g2;
    FuseBroadcastAndBGEMM fusePattern({16, 16, 1, 8, 12}, {16, 16, 4, 8, 12}, 1, {0});
    fusePattern.buildGraphWithBroadcastFusedPattern(g2);

    // Change other operand
    auto& otherOperand = fusePattern.getBgemm()->getInput(1);
    auto  newSizes     = otherOperand->getAllSizesInElements();  // {16, 16, 4, 8, 12}
    newSizes[4]        = 1;
    otherOperand->reshape(otherOperand->getDim(), newSizes.data());  // {16, 16, 4, 8, 1}

    auto nodesAmountBeforeFuse = g2.getNodes().size();

    // Graph before pass:
    // {16, 16, 1, 8, 12} - > broadcast -> {16, 16, 4, 8, 12} -> cast -> {16, 16, 4, 8, 12} -> bgemm -> {16, 16, 4, 8,12}
    //                                                                   {16, 16, 4, 8, 1} ->
    ASSERT_TRUE(fuseBroadcast(g2));
    // Expected graph after pass:
    //                                    {16, 16, 1, 8, 12} -> cast -> {16, 16, 1, 8, 12} -> bgemm -> {16, 16, 4, 8, 12}
    //                                                                  {16, 16, 4, 8, 1} ->

    ASSERT_EQ(g2.getNodes().size(), nodesAmountBeforeFuse - 1) << "Expecting that broadcast was fused";
    for (const auto& node : g2.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_BROADCAST)
        {
            FAIL() << " Expecting that broadcast was fused";
        }
        if (node->getNodeType() == Node::TYPE_BATCH_GEMM)
        {
            const auto& bgemm = std::dynamic_pointer_cast<BatchGemmNode>(node);
            ASSERT_TRUE(bgemm && bgemm->validateNodeLayout()) << "Expecting that bgemm layout validation succeed";
            ASSERT_FALSE(bgemm->isSymmetricLayout())
                << "Expecting that bgemm is asymmetric since the broadcast fused to it";
        }
    }
}

bool FuseBroadcastAndBGEMMNegTest::validateBroadcastWasntFused(const HabanaGraph& g, unsigned expectedNodesAmount)
{
    return g.getNodes().size() == expectedNodesAmount &&
           std::any_of(g.getNodes().begin(), g.getNodes().end(), [](const NodePtr& node) {
               return node->getNodeType() == Node::TYPE_BROADCAST;
           });
}

TEST_F(FuseBroadcastAndBGEMMNegTest, broadcast_is_not_fused_when_bgemm_other_op_is_broadcasted_on_the_same_dims)
{
    Gaudi2Graph           g2;
    FuseBroadcastAndBGEMM fusePattern({16, 16, 1, 8, 12}, {16, 16, 4, 8, 12}, 1, {0});
    fusePattern.buildGraphWithBroadcastFusedPattern(g2);

    // Change other operand
    auto& otherOperand = fusePattern.getBgemm()->getInput(1);
    auto  newSizes     = otherOperand->getAllSizesInElements();  // {16, 16, 4, 8, 12}
    newSizes[2]        = 1;
    otherOperand->reshape(otherOperand->getDim(), newSizes.data());  // {16, 16, 1, 8, 12}

    auto nodesAmountBeforeFuse = g2.getNodes().size();

    // Graph before pass:
    // {16, 16, 1, 8, 12} - > broadcast -> {16, 16, 4, 8, 12} -> cast -> {16, 16, 4, 8, 12} -> bgemm -> {16, 16, 4, 8,12}
    //                                                                   {16, 16, 1, 8, 12} ->
    ASSERT_TRUE(fuseBroadcast(g2));
    // Graph after pass should be the same.

    ASSERT_TRUE(validateBroadcastWasntFused(g2, nodesAmountBeforeFuse))
        << "Expecting that the broadcast is not fused if bgemm contains broadcast on the same dims";
}

TEST_F(FuseBroadcastAndBGEMMNegTest, broadcast_is_not_fused_when_there_is_a_reshape_and_bgemm_is_asymmteric)
{
    Gaudi2Graph           g2;
    FuseBroadcastAndBGEMM fusePattern({16, 16, 1, 8, 12}, {16, 16, 4, 8, 12}, 1, {16, 16, 32, 1, 12});
    fusePattern.buildGraphWithBroadcastFusedPattern(g2);

    // Change other operand
    auto& otherOperand = fusePattern.getBgemm()->getInput(1);
    auto  newSizes     = otherOperand->getAllSizesInElements();  // {16, 16, 32, 1, 12}
    newSizes[4]        = 1;
    otherOperand->reshape(otherOperand->getDim(), newSizes.data());  // {16, 16, 32, 1, 1}
    auto nodesAmountBeforeFuse = g2.getNodes().size();

    // Graph before pass:
    // {16, 16, 1, 8, 12} - > broadcast -> {16, 16, 4, 8, 12} -> cast -> {16, 16, 4, 8, 12} -> reshape -> {16, 16, 32, 1, 12} -> bgemm -> {16, 16, 32, 1, 12}
    //                                                                                                     {16, 16, 32, 1, 1} ->
    ASSERT_TRUE(fuseBroadcast(g2));
    // Graph after pass should be the same.

    ASSERT_TRUE(validateBroadcastWasntFused(g2, nodesAmountBeforeFuse))
        << "Expecting that the broadcast is not fused when the chain contains reshape and bgemm is asymmetric";
}

TEST_F(FuseBroadcastAndBGEMMNegTest, broadcast_is_not_fused_when_reshape_is_on_spatial_dims)
{
    Gaudi2Graph           g2;
    FuseBroadcastAndBGEMM fusePattern({16, 16, 1, 8, 12}, {16, 16, 4, 8, 12}, 1, {32, 32, 1, 8, 12});
    fusePattern.buildGraphWithBroadcastFusedPattern(g2);
    auto nodesAmountBeforeFuse = g2.getNodes().size();

    // Graph before pass:
    // {16, 16, 1, 8, 12} - > broadcast -> {16, 16, 4, 8, 12} -> cast -> {16, 16, 4, 8, 12} -> reshape -> {32, 32, 1, 8, 12} -> bgemm -> {32, 32, 1, 8, 12}
    //                                                                                                    {32, 32, 1, 8, 12} ->
    ASSERT_TRUE(fuseBroadcast(g2));
    // Graph after pass should be the same.

    ASSERT_TRUE(validateBroadcastWasntFused(g2, nodesAmountBeforeFuse))
        << "Expecting that the broadcast is not fused when reshape is on the spatial dims";
}

TEST_F(FuseBroadcastAndBGEMMNegTest,
       broadcast_is_not_fused_when_there_are_more_than_one_reshape)
{
    Gaudi2Graph           g2;
    FuseBroadcastAndBGEMM fusePattern({16, 16, 1, 8, 12}, {16, 16, 4, 8, 12}, 0, {16, 16, 32, 1, 12});
    fusePattern.buildGraphWithBroadcastFusedPattern(g2);
    // broadcast -> reshape -> bgemm

    for (auto& node : g2.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_BROADCAST)
        {
            // Broadcast has one consumer which is a reshape.
            // we will insert another reshape between them.
            const auto& broadcastOutput = node->getOutput(0);
            auto        reshapedOutput    = broadcastOutput->clone(false, false, false);
            SizeArray   reshapedSizes     = {16, 16, 2, 16, 12};
            reshapedOutput->reshape(reshapedOutput->getDim(), reshapedSizes.data());
            const auto newReshape = NodeFactory::createInternalNode({broadcastOutput},
                                                                    {reshapedOutput},
                                                                    nullptr,
                                                                    NodeFactory::reshapeNodeTypeName,
                                                                    fmt::format("{}/reshape", node->getNodeName()));
            ASSERT_TRUE(GraphEditor::addNode(g2, newReshape)) << "Expecting to add new reshape successfully";
            for (auto& consumer : g2.getNodeConsumers(node))
            {
                if (consumer == newReshape) continue;
                ASSERT_TRUE(consumer->getInput(0) == broadcastOutput && consumer->getInputs().size() == 1)
                    << "Expecting that the broadcast output is the only input of the consumer";
                GraphEditor::editNode(g2, consumer, [&](const NodePtr& c) { c->replaceInput(0, reshapedOutput); });
            }
            break;
        }
    }
    auto nodesAmountBeforeFuse = g2.getNodes().size();

    // Graph before pass:
    // {16, 16, 1, 8, 12} - > broadcast -> {16, 16, 4, 8, 12} -> reshape -> {16, 16, 2, 16, 12} -> reshape -> {16, 16, 32, 1, 12} -> bgemm -> {16, 16, 32, 1, 12}
    //                                                                                                        {16, 16, 32, 1, 12} ->
    ASSERT_TRUE(fuseBroadcast(g2));
    // Graph after pass should be the same.

    ASSERT_TRUE(validateBroadcastWasntFused(g2, nodesAmountBeforeFuse))
        << "Expecting that the broadcast is not fused when there are more than one reshape";
}

TEST_F(FuseBroadcastAndBGEMMNegTest, only_one_broadcast_chain_is_optimized)
{
    Gaudi2Graph           g2;
    FuseBroadcastAndBGEMM fusePattern({16, 16, 1, 8, 12}, {16, 16, 4, 8, 12}, 1, {16, 16, 2, 16, 12});
    fusePattern.buildGraphWithBroadcastFusedPattern(g2);
    // broadcast -> cast -> reshape -> bgemm

    for (auto& node : g2.getNodes())
    {
        if (std::dynamic_pointer_cast<BatchGemmNode>(node) != nullptr)
        {
            // Add a broadcast before other operand
            auto otherOperand       = node->getInput(1);
            auto otherNotPersistent = otherOperand->clone(false, false, false);
            GraphEditor::editNode(g2, node, [&otherNotPersistent](const NodePtr& n) {
                n->replaceInput(1, otherNotPersistent);
            });

            auto sizes = otherOperand->getAllNSizesInElements();
            sizes[4]   = 1;
            otherOperand->reshape(otherOperand->getDim(), sizes.data());
            const NodePtr& broadcastOther =
                NodeFactory::createNode({otherOperand}, {otherNotPersistent}, nullptr, 0, "broadcast", "broadcast");
            ASSERT_TRUE(GraphEditor::addNode(g2, broadcastOther))
                << fmt::format("Failed to add {}", broadcastOther->getNodeName());
            break;
        }
    }
    auto nodesAmountBeforeFuse = g2.getNodes().size();
    // Graph before pass:
    // {16, 16, 1, 8, 12} - > broadcast -> {16, 16, 4, 8, 12} -> reshape -> {16, 16, 2, 16, 12} -> bgemm -> {16, 16, 2, 16, 12}
    //                                    {16, 16, 2, 16, 1} -> broadcast -> {16, 16, 2, 16, 12} ->
    ASSERT_TRUE(fuseBroadcast(g2));
    // Graph after pass:
    //                                                                       {16, 16, 1, 8, 12} -> bgemm -> {16, 16, 4, 8, 12} -> reshape -> {16, 16, 2, 16, 12}
    //  {16, 16, 2, 16, 1} -> broadcast -> {16, 16, 2, 16, 12} -> reshape -> {16, 16, 4, 8, 12} ->

    ASSERT_EQ(g2.getNodes().size(), nodesAmountBeforeFuse) << "Expecting there was no change in nodes number";
    ASSERT_TRUE(std::count_if(g2.getNodes().begin(),
                              g2.getNodes().end(),
                              [](const NodePtr& node) { return node->getNodeType() == Node::TYPE_BROADCAST; }) == 1)
        << "Expecting that the broadcast is not fused when reshape is on the spatial dims";
    for (const auto& node : g2.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_BATCH_GEMM)
        {
            const auto& bgemm = std::dynamic_pointer_cast<BatchGemmNode>(node);
            ASSERT_TRUE(bgemm && bgemm->validateNodeLayout()) << "Expecting that bgemm layout validation succeed";
            ASSERT_FALSE(bgemm->isSymmetricLayout())
                << "Expecting that bgemm is asymmetric since the broadcast fused to it";
        }
    }
}