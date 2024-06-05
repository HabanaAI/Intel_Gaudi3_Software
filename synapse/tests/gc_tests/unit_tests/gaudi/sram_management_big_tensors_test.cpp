#include "access_pattern_generator.h"
#include "pattern_solvers.h"
#include "sram_management.h"
#include "sram_management_fe_test.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "tpc_slicing_test_infra.h"
#include "slicing_utils.h"
#include "bundle_expander.h"

namespace gaudi
{
class SRAMManagementBigTensorsTest : public SRAMManagementTest
{
protected:
    void SetUp() override
    {
        SRAMManagementTest::SetUp();

        setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "false");
        setGlobalConfForTest(GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING, "false");
        setGlobalConfForTest(GCFG_IGNORE_INDEX_SPACE_FOR_SLICING, "false");
    }
};

class ReshapeCustomAllRequiredIndexSpaceNode : public ReshapeNode
{
public:
    ReshapeCustomAllRequiredIndexSpaceNode(const TensorVector& inputs,
                                           const TensorVector& outputs,
                                           const std::string&  name)
    : ReshapeNode(inputs, outputs, name)
    {
    }

    static NodePtr create(const TensorVector& inputs, const TensorVector& outputs, const std::string& name)
    {
        return NodePtr(new ReshapeCustomAllRequiredIndexSpaceNode(inputs, outputs, name));
    }

protected:
    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override
    {
        Dim inputOuterDim  = findLastNoneDegenerateDim(getInput(0)->getAllNSizesInElements());
        Dim outputOuterDim = findLastNoneDegenerateDim(getOutput(0)->getAllNSizesInElements());

        NodeTile::Size numOfGranules = 1;  // 'All required' access pattern

        auto nodeGeometry = {numOfGranules};

        auto ap = std::make_shared<NodeAccessPattern>(nodeGeometry.begin(), nodeGeometry.end());

        ap->addTensorAccessPattern(
            getInput(0),
            TensorAccessPatternPtr {new ReshapeTensorAccessPattern(getInput(0), inputOuterDim, numOfGranules)});

        ap->addTensorAccessPattern(
            getOutput(0),
            TensorAccessPatternPtr {new ReshapeTensorAccessPattern(getOutput(0), outputOuterDim, numOfGranules)});

        return ap;
    }

private:
    static Dim findLastNoneDegenerateDim(const NSizeArray& dimSizes)
    {
        Dim curDim = dimSizes.size() - 1;
        while (curDim > 0)
        {
            if (dimSizes.at(curDim) > 1)
            {
                break;
            }
            curDim--;
        }
        return curDim;
    }
};

class SRAMManagementBigTensorsStitchReshapedTpcTest
: public SRAMManagementBigTensorsTest
, public testing::WithParamInterface<std::tuple<bool, bool, bool>>
{
public:
    SRAMManagementBigTensorsStitchReshapedTpcTest()
    : m_isSliceableTpc(std::get<0>(GetParam())),
      m_hasInputOverlap(std::get<1>(GetParam())),
      m_isSliceableReshape(std::get<2>(GetParam()))
    {
    }

    void runSingleTest()
    {
        unsigned batchSize = 256;
        // Create convolution node
        synConvolutionParams  convParams {};
        std::vector<TSize> aSizes   = {512, 28, 28, batchSize};
        std::vector<TSize> bSizes   = {256, 512, 1, 1};
        std::vector<TSize> outSizes = {256, 28, 28, batchSize};
        TensorPtr             a        = createTensor(aSizes, syn_type_float, true);
        TensorPtr             b        = createTensor(bSizes, syn_type_float, true);
        TensorPtr             convOut  = createTensor(outSizes, syn_type_float, false);
        NodePtr               conv =
            NodeFactory::createNode({a, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
        ASSERT_TRUE(GraphEditor::addNode(getGraph(), conv));

        // Create reshape node
        std::vector<TSize>    reshapedSizes = {256 * 28 * 28, batchSize};
        TensorPtr             reshapeOut    = createTensor(reshapedSizes, syn_type_float, false);
        NodePtr               reshape       = m_isSliceableReshape
                                                  ? NodeFactory::createNode({convOut}, {reshapeOut}, nullptr, "reshape", "reshape")
                                                  : ReshapeCustomAllRequiredIndexSpaceNode::create({convOut}, {reshapeOut}, "reshape");
        ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshape));

        // Create TPC consumer node
        TPCCustomIndexSpaceNode::Params nodeParams {};
        nodeParams.dims.emplace_back(reshapedSizes[0],
                                     m_isSliceableTpc ? 1 : reshapedSizes[0],
                                     m_hasInputOverlap ? 2 : 0);
        nodeParams.dims.emplace_back(reshapedSizes[1],
                                     m_isSliceableTpc ? 1 : reshapedSizes[1],
                                     m_hasInputOverlap ? 2 : 0);
        nodeParams.transpose = false;
        NodePtr tpc          = TPCCustomIndexSpaceNode::create(nodeParams, reshapeOut);
        ASSERT_TRUE(GraphEditor::addNode(getGraph(), tpc));

        ASSERT_TRUE(gaudi::loadTpcKernels(getGraph()));

        Bundlizer   bundlizer(getGraph());
        const auto& mmeBundles = bundlizer.getMMEBundles();
        ASSERT_EQ(mmeBundles.size(), 1);

        // Slice convolution on batch dim
        unsigned            numOfSlices = 64;
        pMmeSlicingStrategy strategy =
            MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(),
                                                         mmeBundles.front()->getNodes().front());
        strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
        strategy->getMmeSlicingData().bundleTensors[0]->chunkDimensions[DIM_B] = batchSize / numOfSlices;
        strategy->getMmeSlicingData().masterOperand->chunkDimensions[DIM_B]    = batchSize / numOfSlices;

        pBundleExpansion expCnd = bundlizer.findTpcConsumerExpansionCandidate(strategy);

        if (m_isSliceableTpc && !m_hasInputOverlap && m_isSliceableReshape)
        {
            ASSERT_TRUE(expCnd->nodeToStitch);
            ASSERT_TRUE(expCnd->reshapeNode);
        }
        else
        {
            ASSERT_FALSE(expCnd->nodeToStitch);
        }
    }

protected:
    bool m_isSliceableTpc;
    bool m_hasInputOverlap;
    bool m_isSliceableReshape;
};

TEST_P(SRAMManagementBigTensorsStitchReshapedTpcTest, test_reshaped_tpc_stitching)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(test_reshaped_tpc_stitching_full,
                         SRAMManagementBigTensorsStitchReshapedTpcTest,
                         ::testing::Combine(::testing::ValuesIn({true, false}),  // isSliceableTpc
                                            ::testing::ValuesIn({true, false}),  // hasInputOverlap
                                            ::testing::ValuesIn({true, false})   // isSliceableReshape
                                            ));

TEST_F(SRAMManagementBigTensorsTest, stitch_tpc_producer_with_reshape_to_mme_node)
{
    // Relu producer
    std::vector<TSize>    aReshapedSizes = {512 * 28, 28, 256};
    TensorPtr             aIn            = createTensor(aReshapedSizes, syn_type_bf16, true);
    TensorPtr             a              = createTensor(aReshapedSizes, syn_type_bf16, false);
    NodePtr               reluA = NodeFactory::createGenericTPCNode({aIn}, {a}, nullptr, "relu_fwd_bf16", "reluA");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reluA));

    // Reshape
    std::vector<TSize>    aSizes    = {512, 28, 28, 256};
    TensorPtr             reshapedA = createTensor(aSizes, syn_type_bf16, false);
    NodePtr               reshapeA  = NodeFactory::createNode({a}, {reshapedA}, nullptr, "reshape", "reshapeA");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshapeA));

    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize>    bSizes   = {256, 512, 1, 1};
    TensorPtr             b        = createTensor(bSizes, syn_type_bf16, true);
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16, true);
    NodePtr               conv =
        NodeFactory::createNode({reshapedA, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), conv));

    ASSERT_TRUE(loadTpcKernels(getGraph()));
    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));

    // Make sure all the nodes in the same bundle
    const NodeSet& nodes = getGraph().getNodes();
    ASSERT_FALSE(nodes.empty());
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    unsigned expectedBundleIdx = (*nodes.begin())->getNodeAnnotation().bundleInfo->bundleIndex;
    for (const auto& n : nodes)
    {
        ASSERT_EQ(n->getNodeAnnotation().bundleInfo->bundleIndex, expectedBundleIdx);
    }

    for (const auto& n : nodes)
    {
        if (getGraph().runsOnMME(n))  // Conv
        {
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getInput(1)->inSram());
            ASSERT_FALSE(n->getOutput(0)->inSram());
        }
        else if ((getGraph().runsOnTPC(n)))  // Relu
        {
            ASSERT_FALSE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getOutput(0)->inSram());
        }
        else if (n->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)  // Reshape
        {
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getOutput(0)->inSram());
        }
    }
}

TEST_F(SRAMManagementBigTensorsTest, stitch_tpc_consumer_with_reshape_to_mme_node)
{
    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize> aSizes   = {512, 28, 28, 256};
    std::vector<TSize> bSizes   = {256, 512, 1, 1};
    std::vector<TSize> outSizes = {256, 28, 28, 256};
    TensorPtr          a        = createTensor(aSizes, syn_type_bf16, true);
    TensorPtr          b        = createTensor(bSizes, syn_type_bf16, true);
    TensorPtr          convOut  = createTensor(outSizes, syn_type_bf16, false);
    NodePtr            conv =
        NodeFactory::createNode({a, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), conv));

    // Reshape
    std::vector<TSize>    reshapedSizes = {256 * 28, 28, 256};
    TensorPtr             reshapeOut    = createTensor(reshapedSizes, syn_type_bf16, false);
    NodePtr               reshape = NodeFactory::createNode({convOut}, {reshapeOut}, nullptr, "reshape", "reshape");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshape));

    // Relu consumer
    TensorPtr reluOut = createTensor(reshapedSizes, syn_type_bf16, true);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({reshapeOut}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), relu));

    ASSERT_TRUE(gaudi::loadTpcKernels(getGraph()));
    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));

    // Make sure all the nodes in the same bundle
    const NodeSet& nodes = getGraph().getNodes();
    ASSERT_FALSE(nodes.empty());
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    unsigned expectedBundleIdx = (*nodes.begin())->getNodeAnnotation().bundleInfo->bundleIndex;
    for (const auto& n : nodes)
    {
        ASSERT_EQ(n->getNodeAnnotation().bundleInfo->bundleIndex, expectedBundleIdx);
    }

    for (const auto& n : nodes)
    {
        if (getGraph().runsOnMME(n))  // Conv
        {
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getInput(1)->inSram());
            ASSERT_TRUE(n->getOutput(0)->inSram());
        }
        else if ((getGraph().runsOnTPC(n)))  // Relu
        {
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_FALSE(n->getOutput(0)->inSram());
        }
        else if (n->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)  // Reshape
        {
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getOutput(0)->inSram());
        }
    }
}

TEST_F(SRAMManagementBigTensorsTest, stitch_tpc_producers_and_consumer_with_reshape_to_mme_node)
{
    // Relu producer for first input
    std::vector<TSize>    aReshapedSizes = {512 * 28, 28, 256};
    TensorPtr             aIn            = createTensor(aReshapedSizes, syn_type_bf16, true);
    TensorPtr             a              = createTensor(aReshapedSizes, syn_type_bf16, false);
    NodePtr               reluA = NodeFactory::createGenericTPCNode({aIn}, {a}, nullptr, "relu_fwd_bf16", "reluA");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reluA));

    // Reshape for first input
    std::vector<TSize>    aSizes    = {512, 28, 28, 256};
    TensorPtr             reshapedA = createTensor(aSizes, syn_type_bf16, false);
    NodePtr               reshapeA  = NodeFactory::createNode({a}, {reshapedA}, nullptr, "reshape", "reshapeA");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshapeA));

    // Relu producer for second input
    std::vector<TSize>    bReshapedSizes = {256, 4, 128, 1};
    TensorPtr             bIn            = createTensor(bReshapedSizes, syn_type_bf16, true);
    TensorPtr             b              = createTensor(bReshapedSizes, syn_type_bf16, false);
    NodePtr               reluB = NodeFactory::createGenericTPCNode({bIn}, {b}, nullptr, "relu_fwd_bf16", "reluB");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reluB));

    // Reshape for second input
    std::vector<TSize>    bSizes    = {256, 4 * 128, 1, 1};
    TensorPtr             reshapedB = createTensor(bSizes, syn_type_bf16, false);
    NodePtr               reshapeB  = NodeFactory::createNode({b}, {reshapedB}, nullptr, "reshape", "reshapeB");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshapeB));

    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16, false);
    NodePtr               conv     = NodeFactory::createNode({reshapedA, reshapedB},
                                           {convOut},
                                           &convParams,
                                           NodeFactory::convolutionNodeTypeName,
                                           "conv");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), conv));

    // Reshape for output
    std::vector<TSize>    outReshapedSizes = {256 * 28, 28, 256};
    TensorPtr             reshapedOut      = createTensor(outReshapedSizes, syn_type_bf16, false);
    NodePtr reshapeOut = NodeFactory::createNode({convOut}, {reshapedOut}, nullptr, "reshape", "reshapeOut");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshapeOut));

    // Relu consumer
    TensorPtr reluOut = createTensor(outReshapedSizes, syn_type_bf16, true);
    NodePtr   reluConsumer =
        NodeFactory::createGenericTPCNode({reshapedOut}, {reluOut}, nullptr, "relu_fwd_bf16", "reluConsumer");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reluConsumer));

    ASSERT_TRUE(gaudi::loadTpcKernels(getGraph()));
    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));

    // Make sure all the nodes in the same bundle
    const NodeSet& nodes = getGraph().getNodes();
    ASSERT_FALSE(nodes.empty());
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    unsigned expectedBundleIdx = (*nodes.begin())->getNodeAnnotation().bundleInfo->bundleIndex;
    for (const auto& n : nodes)
    {
        ASSERT_EQ(n->getNodeAnnotation().bundleInfo->bundleIndex, expectedBundleIdx);
    }

    for (const auto& n : nodes)
    {
        if (getGraph().runsOnMME(n))  // Conv
        {
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getInput(1)->inSram());
            ASSERT_TRUE(n->getOutput(0)->inSram());
        }
        else if ((getGraph().runsOnTPC(n)))  // Relu
        {
            const auto& realConsumers = getGraph().getNodeRealConsumers(n, Node::TENSOR_TYPE_DATA);
            bool        isProducer    = std::any_of(realConsumers.begin(), realConsumers.end(), [&](const NodePtr& n) {
                return getGraph().runsOnMME(n);
            });
            if (isProducer)
            {
                ASSERT_FALSE(n->getInput(0)->inSram());
                ASSERT_TRUE(n->getOutput(0)->inSram());
            }
            else
            {
                ASSERT_TRUE(n->getInput(0)->inSram());
                ASSERT_FALSE(n->getOutput(0)->inSram());
            }
        }
        else if (n->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)  // Reshape
        {
            ASSERT_TRUE(n->getInput(0)->inSram());
            ASSERT_TRUE(n->getOutput(0)->inSram());
        }
    }
}

TEST_F(SRAMManagementBigTensorsTest, dont_align_to_cl_when_stitch_through_reshape_on_fcd)
{
    // Relu producer
    std::vector<TSize>    aReshapedSizes = {511 * 28, 28, 256};
    TensorPtr             aIn            = createTensor(aReshapedSizes, syn_type_bf16, true);
    TensorPtr             a              = createTensor(aReshapedSizes, syn_type_bf16, false);
    NodePtr               reluA = NodeFactory::createGenericTPCNode({aIn}, {a}, nullptr, "relu_fwd_bf16", "reluA");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reluA));

    // Reshape
    std::vector<TSize>    aSizes    = {511, 28, 28, 256};
    TensorPtr             reshapedA = createTensor(aSizes, syn_type_bf16, false);
    NodePtr               reshapeA  = NodeFactory::createNode({a}, {reshapedA}, nullptr, "reshape", "reshapeA");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), reshapeA));

    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize>    bSizes   = {256, 511, 1, 1};
    TensorPtr             b        = createTensor(bSizes, syn_type_bf16, true);
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16, true);
    NodePtr               conv =
        NodeFactory::createNode({reshapedA, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), conv));

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    // Create bundles and initial strategies
    AllBrains          allBrains(getGraph());
    SRAMSlicingManager sramManager(getGraph());
    sramManager.generateInitialBundles();
    sramManager.generateInitialStrategies();
    auto bundlesSolvingData = sramManager.getBundlesSolvingData();

    // Expand the bundle with the TPC producer
    BundleExpander bundleExpander(getGraph(), allBrains, sramManager.getBundlizer(), bundlesSolvingData);
    ASSERT_EQ(bundlesSolvingData.size(), 1);
    const pBundle& bundle = bundlesSolvingData.begin()->first;

    for (const auto& s : bundleExpander.generateExpandedStrategies(bundle))
    {
        pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
        const auto& producer = strategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::WideInputProducer];
        const auto& slicedReshapedA = strategy->getMmeSlicingData().getSlicedOperand(reshapedA);
        ASSERT_NE(slicedReshapedA, nullptr);
        if (producer && producer->nodeToStitch)
        {
            ASSERT_NE(producer->reshapeNode, nullptr);
            // CL alignment should be blocked when stitching producer through reshape on FCD
            ASSERT_FALSE(slicedReshapedA->alignWithCacheLine);
        }
        else
        {
            // No producer, CL alignment is allowed
            ASSERT_TRUE(slicedReshapedA->alignWithCacheLine);
        }
    }
}

TEST_F(SRAMManagementBigTensorsTest, tpc_slice_should_get_dynamic_info_from_tpc_node)
{
    // Create a dynamic TPC node
    std::vector<TSize>    minSizes = {1000, 1000};
    std::vector<TSize>    maxSizes = {2000, 2000};
    TensorPtr             in       = createTensor(maxSizes, syn_type_bf16, true, minSizes);
    TensorPtr             out      = createTensor(maxSizes, syn_type_bf16, true, minSizes);
    NodePtr               relu     = NodeFactory::createGenericTPCNode({in}, {out}, nullptr, "relu_fwd_bf16", "RELU");
    ASSERT_TRUE(GraphEditor::addNode(getGraph(), relu));

    ASSERT_TRUE(getGraph().compile());

    auto tpcNode = std::dynamic_pointer_cast<TPCNode>(relu);
    ASSERT_TRUE(tpcNode);
    auto sifVersion = tpcNode->getShapeInferenceFunctionVersion();
    const auto& dynamicShapeProjectionTensors = tpcNode->getDynamicShapeProjectionsTensors();

    auto tpcSlice = tpcNode->getSlice();
    ASSERT_TRUE(tpcSlice);

    ASSERT_EQ(tpcSlice->getShapeInferenceFunctionVersion(), sifVersion);
    ASSERT_EQ(tpcSlice->getDynamicShapeProjectionsTensors().size(), dynamicShapeProjectionTensors.size());
}

TEST_F(SRAMManagementBigTensorsTest, tpc_nodes_should_not_be_stitched_to_flattened_operands)
{
    std::vector<TSize>    shape = {128, 128, 128, 128};
    pTensor               in    = createTensor(shape, syn_type_bf16);
    pTensor               out   = createTensor(shape, syn_type_bf16);

    // Create a sliceable TPC node (slicing granularity on each dim is 1)
    pNode tpcNode = TPCCustomIndexSpaceNode::createSliceableNode(in, out);
    GraphEditor::addNode(getGraph(), tpcNode);

    ASSERT_TRUE(loadTpcKernels(getGraph()));

    Bundlizer bundlizer {getGraph()};

    // Make sure that TPC nodes can't be stitched to flattened operands
    SizeArray flattenShape = {128, 128 * 128 * 128, 1, 1, 1};

    pSlicedOperand triviallySlicedOperand(new Solution::SlicedOperand(out));
    EXPECT_TRUE(bundlizer.isNodeEligibleForStitching(tpcNode, triviallySlicedOperand, nullptr));
    // Flatten the operand
    triviallySlicedOperand->finalShape      = flattenShape;
    triviallySlicedOperand->chunkDimensions = triviallySlicedOperand->finalShape;
    EXPECT_FALSE(bundlizer.isNodeEligibleForStitching(tpcNode, triviallySlicedOperand, nullptr));

    pSlicedOperand slicedOperand(new Solution::SlicedOperand(out));
    slicedOperand->chunkDimensions[0] = 64;
    EXPECT_TRUE(bundlizer.isNodeEligibleForStitching(tpcNode, slicedOperand, nullptr));
    // Flatten the operand and slice FCD
    slicedOperand->finalShape         = flattenShape;
    slicedOperand->chunkDimensions    = slicedOperand->finalShape;
    slicedOperand->chunkDimensions[0] = 64;
    EXPECT_FALSE(bundlizer.isNodeEligibleForStitching(tpcNode, slicedOperand, nullptr));
}

}  // namespace gaudi