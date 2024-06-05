#include <gtest/gtest.h>
#include <iostream>
#include "graph_optimizer_test.h"
#include "habana_graph.h"
#include "gaudi2_code_generator.h"
#include "mme_common/mme_common_enum.h"
#include "include/mme_common/recurring_misalignment_opt.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"

class MMEOptimizationTest : public GraphOptimizerTest
{
public:
    Gaudi2Graph buildTest(const char*          guid,
                          synConvolutionParams convParams,
                          const TSize          aSize[4],
                          const TSize          bSize[4],
                          const TSize          cSize[4],
                          synDataType          inputDataType,
                          synDataType          outputDataType);
    Gaudi2Graph createDedwTest(unsigned    K,
                               unsigned    C,
                               unsigned    F0,
                               unsigned    F1,
                               synDataType inputDataType,
                               synDataType outputDataType);
    void        createForwardTestForRecurringMisalignmentOpt(TSize    xSizes[4],
                                                             TSize    wSizes[4],
                                                             TSize    ySizes[4],
                                                             unsigned strides[3],
                                                             unsigned padding[3],
                                                             unsigned expectedNumSubProblems,
                                                             unsigned expectedCutPoints[]);
    void        createDedwWithConcurrency(unsigned    K,
                                          unsigned    C,
                                          unsigned    F0,
                                          unsigned    F1,
                                          unsigned    expectedCdConcurrencyLevel,
                                          unsigned    expectedBatchConcurrencyLevel,
                                          bool        expectedAsymmetricPortMode = false,
                                          synDataType inputDataType              = syn_type_bf16,
                                          synDataType outputDataType             = syn_type_bf16);
    void addNodesAndCompile(HabanaGraph& g, const NodePtr& addANode, const NodePtr addBNode, const NodePtr mmeNode);
};

//=======================================
void MMEOptimizationTest::addNodesAndCompile(HabanaGraph&   g,
                                             const NodePtr& addANode,
                                             const NodePtr  addBNode,
                                             const NodePtr  mmeNode)
{
    {
        EXPECT_NO_THROW({
            ASSERT_TRUE(GraphEditor::addNode(g, addANode));
            ASSERT_TRUE(GraphEditor::addNode(g, addBNode));
            ASSERT_TRUE(GraphEditor::addNode(g, mmeNode));
            ASSERT_TRUE(g.compile());
        });
    }
}

Gaudi2Graph MMEOptimizationTest::buildTest(const char*          guid,
                                           synConvolutionParams convParams,
                                           const TSize          aSize[4],
                                           const TSize          bSize[4],
                                           const TSize          cSize[4],
                                           synDataType          inputDataType,
                                           synDataType          outputDataType)
{
    Gaudi2Graph g;

    std::string addNodeName;
    switch (inputDataType)
    {
        case syn_type_bf16:
            addNodeName = "add_fwd_bf16";
            break;
        case syn_type_single:
            addNodeName = "add_fwd_f32";
            break;
        case syn_type_fp8_152:
        case syn_type_fp8_143:
            addNodeName = "add_fwd_f8";
            break;
        default:
            HB_ASSERT(0, "Unsupported data type");
    }

    // Add two ADD nodes with a and b as their outputs just to make a and b non-persistent
    TensorPtr  inA0     = TensorPtr(new Tensor(4U, aSize, inputDataType));
    TensorPtr  inA1     = TensorPtr(new Tensor(4U, aSize, inputDataType));
    TensorPtr  a        = TensorPtr(new Tensor(4U, aSize, inputDataType));
    const auto addANode = NodeFactory::createNode({inA0, inA1}, {a}, nullptr, addNodeName.c_str(), "add_a");

    TensorPtr  inB0     = TensorPtr(new Tensor(4U, bSize, inputDataType));
    TensorPtr  inB1     = TensorPtr(new Tensor(4U, bSize, inputDataType));
    TensorPtr  b        = TensorPtr(new Tensor(4U, bSize, inputDataType));
    const auto addBNode = NodeFactory::createNode({inB0, inB1}, {b}, nullptr, addNodeName.c_str(), "add_b");

    TensorPtr c = TensorPtr(new Tensor(4U, cSize, outputDataType));

    synMemoryDescriptor memDesc(true);  // persistent
                                        //    synMemoryDescriptor memDesc1(false);  // non persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    inA0->setDramOffset(0x20000);
    inA1->setDramOffset(0x40000);
    inB0->setDramOffset(0x60000);
    inB1->setDramOffset(0x80000);
    c->setDramOffset(0xB0000);

    c->setMemoryDescriptor(memDesc);
    inA0->setMemoryDescriptor(memDesc);
    inA1->setMemoryDescriptor(memDesc);
    inB0->setMemoryDescriptor(memDesc);
    inB1->setMemoryDescriptor(memDesc);

    NodePtr node = NodeFactory::createNode({a, b}, {c}, &convParams, guid, "mme_node");

    addNodesAndCompile(g, addANode, addBNode, node);
    return g;
}

//============================================================================================
//================= dedw with concurrency ====================================================
Gaudi2Graph MMEOptimizationTest::createDedwTest(unsigned    K,
                                                unsigned    C,
                                                unsigned    F0,
                                                unsigned    F1,
                                                synDataType inputDataType,
                                                synDataType outputDataType)
{
    unsigned ifmSizes[2]   = {20, 30};
    unsigned filterSize[2] = {F0, F1};
    unsigned strides[2]    = {1, 1};
    unsigned dilation[2]   = {1, 1};
    unsigned padding[2]    = {1, 1};

    TSize xSize[4] = {C, ifmSizes[0], ifmSizes[1], 1};
    TSize wSize[4] = {K, C, F0, F1};
    // Calc y sizes
    unsigned y[2];
    for (int i = 0; i < 2; i++)
    {
        y[i] = convOutputDimSize(xSize[i + 1], filterSize[i], strides[i], 2 * padding[i], dilation[i]);
    }
    TSize ySize[4] = {K, y[0], y[1], 1};

    synConvolutionParams convParams;
    convParams.kW   = F0;  // filter size
    convParams.kH   = F1;
    convParams.dW   = 1;  // stride
    convParams.dH   = 1;
    convParams.padL = padding[0];  // padding
    convParams.padR = padding[0];
    convParams.padT = padding[1];
    convParams.padB = padding[1];

    Gaudi2Graph g =
        buildTest(NodeFactory::deDwNodeTypeName, convParams, ySize, xSize, wSize, inputDataType, outputDataType);
    return g;
}

// ========== Concurrency testing ==============================
void MMEOptimizationTest::createDedwWithConcurrency(unsigned    K,
                                                    unsigned    C,
                                                    unsigned    F0,
                                                    unsigned    F1,
                                                    unsigned    expectedCdConcurrencyLevel,
                                                    unsigned    expectedBatchConcurrencyLevel,
                                                    bool        expectedAsymmetricPortMode,
                                                    synDataType inputDataType,
                                                    synDataType outputDataType)
{
    Gaudi2Graph g = createDedwTest(K, C, F0, F1, inputDataType, outputDataType);

    // Checks
    bool castFound = false;
    for (const NodePtr& n : g.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_DEDW)
        {
            gaudi2::MmeDescriptorGenerator& descGen = downcaster<Gaudi2CodeGenerator>(g.getCodeGenerator().get())->getMmeNodeDescriptorGenerator(n);
            auto                            params  = descGen.getParams();

            if (expectedCdConcurrencyLevel > 1)
            {
                EXPECT_TRUE(params.strategy.cdConcurrencyEn == MmeCommon::TurnedOn);
            }
            if (expectedBatchConcurrencyLevel > 1)
            {
                EXPECT_TRUE(params.strategy.batchConcurrencyEn == MmeCommon::TurnedOn);
            }

            const unsigned cdConcurrencyLevel    = MmeCommon::MmeBrain::getGeometryCdConcurrency(MmeCommon::e_mme_Gaudi2, params);
            const unsigned batchConcurrencyLevel = MmeCommon::MmeBrain::getEffectiveBatchConcurrency(MmeCommon::e_mme_Gaudi2, params);
            const bool     asymmetricPortMode    = MmeCommon::MmeBrain::isAsymPortConfigMode(MmeCommon::e_mme_Gaudi2, params);

            EXPECT_TRUE(cdConcurrencyLevel == expectedCdConcurrencyLevel)
                << "Mismatch! Actual cd concurrency " << std::to_string(cdConcurrencyLevel) << ", expected "
                << std::to_string(expectedCdConcurrencyLevel);
            EXPECT_TRUE(batchConcurrencyLevel == expectedBatchConcurrencyLevel)
                << "Mismatch! Actual batch concurrency " << std::to_string(batchConcurrencyLevel) << ", expected "
                << std::to_string(expectedBatchConcurrencyLevel);
            EXPECT_TRUE(asymmetricPortMode == expectedAsymmetricPortMode)
                << "Mismatch! Actual asymmetric port mode is " << std::to_string(asymmetricPortMode) << ", expected "
                << std::to_string(expectedAsymmetricPortMode);
        }
        if (n->isCast()) castFound = true;
    }
    if (expectedCdConcurrencyLevel > 1 && (outputDataType == syn_type_fp8_143 || outputDataType == syn_type_fp8_152))
    {
        EXPECT_TRUE(castFound);
    }
}

TEST_F(MMEOptimizationTest, gaudi2_concurrency_fullCore)
{
    createDedwWithConcurrency(128, 128, 1, 1, 4, 1);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_fullCore_fp8_143)
{
    createDedwWithConcurrency(128, 128, 1, 1, 4, 1, false, syn_type_fp8_143, syn_type_fp8_143);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_fullCore_fp8_152)
{
    createDedwWithConcurrency(128, 128, 1, 1, 4, 1, false, syn_type_fp8_152, syn_type_fp8_152);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_fullGeo)
{
    // cd con and batch con are 2x, batch wins
    createDedwWithConcurrency(256, 256, 1, 2, 1, 2);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_smallSpatial)
{
    createDedwWithConcurrency(22, 13, 3, 3, 8, 1);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_batchWinsF2)
{
    // cd wins because F2 = 4 and batch con is fully utilized
    createDedwWithConcurrency(62, 42, 3, 4, 1, 4);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_partialSizes)
{
    // hybrid wins. Batch cannot win because F2 = 2. Cd and Hybrid reach 4x, so Hybrid wins.
    createDedwWithConcurrency(62, 42, 3, 2, 2, 2);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_batchConCannotUtilizeAllFilters)
{
    // cd concurrency wins because batch concurrency cannot utilize all filters
    createDedwWithConcurrency(64, 60, 3, 3, 4, 1);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_batchConUtilizeAllFilters)
{
    // batch concurrency wins because F1=4 and therefore it utilizes all filters
    createDedwWithConcurrency(64, 60, 3, 4, 1, 4);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_asymmetricPortMode_fullB0)
{
    // Hybrid utilizes the 4xh geometry with assymetric port mode to reach 6x
    createDedwWithConcurrency(64, 32, 3, 3, 2, 3, true);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_asymmetricPortMode_partialB0)
{
    // Hybrid utilizes the 4xh geometry with assymetric port mode to reach 6x
    createDedwWithConcurrency(32, 32, 3, 3, 2, 3, true);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_asymmetricPortMode_smallC)
{
    // Hybrid utilizes the 4xh geometry with assymetric port mode to reach 6x
    createDedwWithConcurrency(6, 32, 3, 3, 2, 3, true);
};

TEST_F(MMEOptimizationTest, gaudi2_concurrency_asymmetricPortMode_F0equal4)
{
    // Hybrid utilizes the 4xh geometry with assymetric port mode to reach 6x
    createDedwWithConcurrency(64, 32, 4, 3, 2, 3, true);
};

// This test is disabled because it will prevent the merge of Step7 gerrit
TEST_F(MMEOptimizationTest, DISABLED_gaudi2_concurrency_asymmetricPortMode_FullSize)
{
    createDedwWithConcurrency(64, 128, 3, 3, 2, 1, true);
};

// This test is disabled because it will prevent the merge of Step7 gerrit
TEST_F(MMEOptimizationTest, DISABLED_gaudi2_concurrency_asymmetricPortMode_PartialSize)
{
    createDedwWithConcurrency(60, 120, 3, 3, 2, 1, true);
};

// This test is disabled because it will prevent the merge of Step7 gerrit
TEST_F(MMEOptimizationTest, DISABLED_gaudi2_concurrency_asymmetricPortMode_PartialSizeVariousFilterSizes)
{
    // Output size after lowering is 60x360. This size enables utilizing 4xh geo with cd con of 2x,
    // with no penalty due to asymmetric port mode
    createDedwWithConcurrency(60, 90, 4, 3, 2, 1, true);
};

//======================================================================================
//================= Recurring Misalignment Optimization ================================
void MMEOptimizationTest::createForwardTestForRecurringMisalignmentOpt(TSize    xSizes[4],
                                                                       TSize    wSizes[4],
                                                                       TSize    ySizes[4],
                                                                       unsigned strides[3],
                                                                       unsigned padding[3],
                                                                       unsigned expectedNumSubProblems,
                                                                       unsigned expectedCutPoints[])
{
    synConvolutionParams convParams;
    convParams.kW   = wSizes[2];  // filter size
    convParams.kH   = wSizes[3];
    convParams.dW   = strides[0];  // stride
    convParams.dH   = strides[1];
    convParams.padL = padding[0];  // padding
    convParams.padR = padding[0];
    convParams.padT = padding[1];
    convParams.padB = padding[1];

    Gaudi2Graph g = buildTest(NodeFactory::convolutionNodeTypeName,
                              convParams,
                              xSizes,
                              wSizes,
                              ySizes,
                              syn_type_bf16,
                              syn_type_bf16);

    // Checks
    bool nodeFound = false;
    for (const NodePtr& n : g.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_CONVOLUTION)
        {
            nodeFound                                               = true;
            gaudi2::MmeDescriptorGenerator&          descGen        = downcaster<Gaudi2CodeGenerator>(g.getCodeGenerator().get())->getMmeNodeDescriptorGenerator(n);
            const MmeCommon::ConvSubProblemContainer subProblems    = descGen.getSubProblems();
            unsigned                                 numSubProblems = subProblems.size();
            EXPECT_EQ(numSubProblems, expectedNumSubProblems)
                << "Wrong number of sub-problems. Actual " << std::to_string(numSubProblems) << ", expected "
                << std::to_string(expectedNumSubProblems);
            for (int i = 0; i < numSubProblems; i++)
            {
                // Use this when multi sub-problems is supported
                auto     subProblemParams = subProblems[i].params;
                unsigned cdCutPoint       = MmeCommon::RecurringMisalignmentOptimization::getCutPointPerSubProblem(
                    subProblemParams,
                    descGen.getGeoAttr(),
                    MmeCommon::e_mme_Gaudi2);
                EXPECT_EQ(cdCutPoint, expectedCutPoints[i])
                    << "Wrong cd cut point in sub-problem " << std::to_string(i) << ": actual "
                    << std::to_string(cdCutPoint) << ", expected " << std::to_string(expectedCutPoints[i]);
            }
        }
    }
    // Make sure the test does not pass just because no node was verified against expectation
    EXPECT_EQ(nodeFound, true) << "Error! Check found no node";
}

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_singleCutPoint_noSpatialSteps)
{
    TSize    xSizes[4]              = {16, 40, 6, 1};
    TSize    wSizes[4]              = {256, 16, 8, 1};
    TSize    ySizes[4]              = {256, 9, 4, 1};
    unsigned strides[3]             = {4, 2, 1};
    unsigned padding[3]             = {1, 1, 0};
    unsigned expectedNumSubProblems = 1;
    unsigned expectedCutPoints[1]   = {16};

    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_singleCutPoint_noPartials)
{
    TSize    xSizes[4]              = {32, 60, 40, 1};
    TSize    wSizes[4]              = {10, 32, 4, 1};
    TSize    ySizes[4]              = {10, 15, 20, 1};
    unsigned strides[3]             = {4, 2, 1};
    unsigned padding[3]             = {1, 0, 0};
    unsigned expectedNumSubProblems = 1;
    unsigned expectedCutPoints[1]   = {32};

    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_singleCutPoint_bothPartials)
{
    TSize    xSizes[4]              = {48, 60, 4, 1};
    TSize    wSizes[4]              = {10, 48, 4, 2};
    TSize    ySizes[4]              = {10, 16, 2, 1};
    unsigned strides[3]             = {4, 2, 1};
    unsigned padding[3]             = {2, 0, 1};
    unsigned expectedNumSubProblems = 1;
    unsigned expectedCutPoints[1]   = {32};

    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_singleCutPoint_partialsInPrefix)
{
    TSize    xSizes[4]              = {48, 60, 40, 1};
    TSize    wSizes[4]              = {260, 48, 3, 3};
    TSize    ySizes[4]              = {260, 15, 20, 1};
    unsigned strides[3]             = {4, 2, 1};
    unsigned padding[3]             = {1, 1, 0};
    unsigned expectedNumSubProblems = 1;
    unsigned expectedCutPoints[1]   = {48};

    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_singleCutPoint_multipleFcdSpSteps)
{
    setGlobalConfForTest(GCFG_ENABLE_PIPELINE_MANAGEMENT, "false");
    setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, "0");

    TSize    xSizes[4]              = {80, 60, 20, 1};
    TSize    wSizes[4]              = {152, 80, 4, 3};
    TSize    ySizes[4]              = {152, 16, 10, 1};
    unsigned strides[3]             = {4, 2, 1};
    unsigned padding[3]             = {2, 1, 0};
    unsigned expectedNumSubProblems = 1;
    unsigned expectedCutPoints[1]   = {32};

    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_2cutPoints_firstCdCut_secondNode)
{
    TSize    xSizes[4]              = {32, 20, 1, 1};
    TSize    wSizes[4]              = {10, 32, 3, 3};
    TSize    ySizes[4]              = {10, 20, 1, 1};
    unsigned strides[3]             = {1, 1, 1};
    unsigned padding[3]             = {1, 1, 0};
    unsigned expectedNumSubProblems = 2;
    unsigned expectedCutPoints[2]   = {32, 0};

    setGlobalConfForTest(GCFG_ENABLE_CONV_PACKING_TRAINING, "false");
    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_2cutPointsWithPacking)
{
    TSize    xSizes[4]              = {16, 20, 2, 1};
    TSize    wSizes[4]              = {10, 16, 5, 5};
    TSize    ySizes[4]              = {10, 20, 2, 1};
    unsigned strides[3]             = {1, 1, 1};
    unsigned padding[3]             = {2, 2, 0};
    unsigned expectedNumSubProblems = 2;
    unsigned expectedCutPoints[4]   = {32, 0};

    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_4cutPoints)
{
    TSize    xSizes[4]              = {16, 20, 2, 1};
    TSize    wSizes[4]              = {10, 16, 5, 5};
    TSize    ySizes[4]              = {10, 20, 2, 1};
    unsigned strides[3]             = {1, 1, 1};
    unsigned padding[3]             = {2, 2, 0};
    unsigned expectedNumSubProblems = 4;
    unsigned expectedCutPoints[4]   = {32, 16, 0, 48};

    setGlobalConfForTest(GCFG_ENABLE_CONV_PACKING_TRAINING, "false");
    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};

TEST_F(MMEOptimizationTest, gaudi2_recurringMisalignment_8cutPoints)
{
    TSize    xSizes[4]              = {24, 17, 5, 1};
    TSize    wSizes[4]              = {9, 24, 3, 3};
    TSize    ySizes[4]              = {9, 19, 7, 1};
    unsigned strides[3]             = {1, 1, 1};
    unsigned padding[3]             = {2, 2, 0};
    unsigned expectedNumSubProblems = 8;
    unsigned expectedCutPoints[8]   = {48, 24, 0, 40, 16, 56, 32, 8};

    createForwardTestForRecurringMisalignmentOpt(xSizes,
                                                 wSizes,
                                                 ySizes,
                                                 strides,
                                                 padding,
                                                 expectedNumSubProblems,
                                                 expectedCutPoints);
};
