#include "compilation_hal_reader.h"
#include "gaudi3_graph.h"
#include "perf_lib_layer_params.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "slicing_brain.h"
#include "synapse_common_types.hpp"
#include "types.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "slicer/slicer.h"

using namespace gc::layered_brain;

class SlicerTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<bool>  // iterative mode
{
protected:
    SlicerTest() : m_halSetter(&m_graph) {}

    TensorPtr
    createTensor(const std::vector<TSize>& shape, bool isPersistent = false, synDataType dataType = syn_type_bf16) const
    {
        TensorPtr tensor = std::make_shared<Tensor>(shape.size(), shape.data(), dataType);
        if (isPersistent)
        {
            synMemoryDescriptor memDesc(true);
            tensor->setMemoryDescriptor(memDesc);
        }
        return tensor;
    }

    TensorPtr createShapeTensor(const std::vector<TSize>& shape) const
    {
        return std::make_shared<Tensor>(shape.size(),
                                        shape.data(),
                                        syn_type_uint32,
                                        nullptr,
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        nullptr,
                                        SHAPE_TENSOR);
    }

    void addNodeToGraph(const NodePtr& node)
    {
        ASSERT_TRUE(node);
        ASSERT_TRUE(GraphEditor::addNode(m_graph, node));
        m_origNodes.insert(node);
    }

    void sliceGraph(const NodeVector& bundleNodes)
    {
        setGlobalConfForTest(GCFG_ENABLE_LB_SAMPLE_MODE, "True");

        ASSERT_TRUE(gaudi3::loadTpcKernels(m_graph));  // Required to intialize nodes access-pattern
        ASSERT_TRUE(validateNodesLayout(m_graph));

        Slicer slicer(m_graph, BUNDLE_IDX, bundleNodes);
        if (m_iterativeMode)
        {
            const auto& strategies = slicer.getStrategies();
            ASSERT_FALSE(strategies.empty());

            // Create a sliced graph from each one of the strategies before inflation
            for (const auto& strategy : strategies)
            {
                auto fullSlicedGraph = slicer.sliceBundleByStrategy(strategy);
                validateSlicedGraph(bundleNodes, fullSlicedGraph);

                auto partialSlicedGraph = slicer.sliceBundleByStrategy(strategy, /*dryRun*/ true);
                validateSlicedGraph(bundleNodes, partialSlicedGraph, /*dryRun*/ true);
            }

            // Select one strategy to inflate and create a sliced graph after each inflation step
            auto                       strategyToInflate   = strategies.front();
            static constexpr std::array<InflationType, 4> inflationPriorities = {InflationType::INFLATE_FOR_UTILIZATION,
                                                                                 InflationType::INFLATE_FOR_BW,
                                                                                 InflationType::INFLATE_FOR_PERFORATION,
                                                                                 InflationType::INFLATE_FOR_NUM_SLICES};
            HB_ASSERT_PTR(strategyToInflate->getMmeSolution());
            HB_ASSERT(strategyToInflate->getMmeSolution()->QORs.size() >= 1,
                      "Expected at least one MME in each strategy");
            for (auto inflationType : inflationPriorities)
            {
                NodePtr nodeToInflate = (inflationType == InflationType::INFLATE_FOR_NUM_SLICES)
                                            ? nullptr
                                            : strategyToInflate->getMmeSolution()->QORs.begin()->first;
                while (slicer.inflateStrategy(inflationType, strategyToInflate, nodeToInflate))
                {
                    auto slicedGraph = slicer.sliceBundleByStrategy(strategyToInflate);
                    validateSlicedGraph(bundleNodes, slicedGraph);
                }
            }
        }
        else
        {
            auto slicedGraph = slicer.getSlicedBundle();
            validateSlicedGraph(bundleNodes, slicedGraph);
        }
    }

    void
    validateSlicedGraph(const NodeVector& bundleNodes, const HabanaGraphPtr& slicedGraph, bool dryRun = false) const
    {
        // Validate brain data is updated as expected
        auto brainData = slicedGraph->getLayeredBrainData();
        ASSERT_TRUE(brainData);
        ASSERT_TRUE(brainData->m_bundleData.find(BUNDLE_IDX) != brainData->m_bundleData.end());
        const BundleData& bundleData = brainData->m_bundleData.at(BUNDLE_IDX);
        ASSERT_TRUE(bundleData.getBundleViews());
        ASSERT_TRUE(bundleData.getFinalStrategy());
        for (BundleViewId bvdId = 0; bvdId < bundleData.getBundleViews()->getNumOfBundleViews(); bvdId++)
        {
            const auto& bundleView = bundleData.getBundleViews()->getBundleView(bvdId);
            const auto& strategy   = bundleData.getFinalStrategy();
            const auto& multiplier = strategy->getBVDMultiplier(bvdId);
            uint64_t    numSlices =
                multiplier.isSliced() ? div_round_up(bundleView.resolution, multiplier.getMultiplier()) : 1;
            if (dryRun && numSlices > strategy->getPipelineDepth())
            {
                numSlices = strategy->getPipelineDepth();
            }
            ASSERT_EQ(bundleData.getNumOfSlicesPerBVD(bvdId), numSlices);
        }

        ASSERT_TRUE(slicedGraph->isConnectedGraph());
        const auto& slicedGraphNodes = slicedGraph->getNodes();
        for (const auto& slicedNode : slicedGraphNodes)
        {
            if (slicedNode->getNodeAnnotation().origBigNode)
            {
                ASSERT_TRUE(std::find(bundleNodes.begin(),
                                      bundleNodes.end(),
                                      slicedNode->getNodeAnnotation().origBigNode) != bundleNodes.end());
                ASSERT_TRUE(slicedNode->getNodeAnnotation().bundleInfo.is_set());
                ASSERT_EQ(slicedNode->getNodeAnnotation().bundleInfo->bundleIndex, BUNDLE_IDX);
            }
        }

        // Original graph should not be changed
        const auto& origGraphNodes = m_graph.getNodes();
        ASSERT_EQ(origGraphNodes.size(), m_origNodes.size());
        for (const auto& origNode : m_origNodes)
        {
            ASSERT_TRUE(std::find(origGraphNodes.begin(), origGraphNodes.end(), origNode) != origGraphNodes.end());
        }
    }

    void validateCDSlicing(const NodeVector& bundleNodes, const NodePtr& mmeNode, bool castRequired)
    {
        Slicer      slicer(m_graph, BUNDLE_IDX, bundleNodes);
        const auto& strategies = slicer.getStrategies();

        bool cdSlicingStrategyFound = false;
        for (const auto& strategy : strategies)
        {
            const auto& cdDims = strategy->getMMECommonDims(mmeNode);
            ASSERT_EQ(cdDims.size(), 1);
            if (strategy->getBVDMultiplier(cdDims.front()).isSliced())
            {
                cdSlicingStrategyFound = true;
                ASSERT_EQ(strategy->getNodeQORs(mmeNode)->solutionRequirements.requiresCast, castRequired);
            }
        }
        ASSERT_TRUE(cdSlicingStrategyFound);
    }

    Gaudi3Graph                m_graph;
    CompilationHalReaderSetter m_halSetter;
    NodeSet                    m_origNodes;
    static constexpr BundleIdx BUNDLE_IDX = 5;
    const bool                 m_iterativeMode = GetParam();
};

TEST_P(SlicerTest, slice_bundle_from_bert)
{
    // Producers chain for operand A : mult -> add -> softmax -> dropout
    NodePtr mult =
        NodeFactory::createGenericTPCNode({createTensor({512, 512, 16, 28}, true), createTensor({1, 1, 1, 1}, true)},
                                          {createTensor({512, 512, 16, 28})},
                                          nullptr,
                                          "mult_fwd_bf16",
                                          "MULT");
    addNodeToGraph(mult);
    NodePtr addA = NodeFactory::createGenericTPCNode({mult->getOutput(0), createTensor({512, 512, 1, 28}, true)},
                                                     {createTensor({512, 512, 16, 28})},
                                                     nullptr,
                                                     "add_fwd_bf16",
                                                     "ADD_A");
    addNodeToGraph(addA);
    ns_Softmax::Params softmaxParams;
    softmaxParams.dim = 0;
    NodePtr softmax   = NodeFactory::createGenericTPCNode({addA->getOutput(0)},
                                                        {createTensor({512, 512, 16, 28})},
                                                        &softmaxParams,
                                                        "softmax_fwd_bf16",
                                                        "SOFTMAX");
    addNodeToGraph(softmax);
    ns_DropoutKernel::Params dropoutParams;
    dropoutParams.ratio = 0;
    dropoutParams.seed  = 0;
    NodePtr dropout     = NodeFactory::createGenericTPCNode(
        {softmax->getOutput(0), createTensor({1}, true, syn_type_int32)},
        {createTensor({512, 512, 16, 28}, true), createTensor({512, 512, 16, 28}, true, syn_type_int8)},
        &dropoutParams,
        "dropout_fwd_bf16",
        "DROPOUT");
    addNodeToGraph(dropout);

    // Producers chain for operand B : add -> reshape -> transpose
    NodePtr addB = NodeFactory::createGenericTPCNode({createTensor({1024, 14336}, true), createTensor({1024, 1}, true)},
                                                     {createTensor({1024, 14336})},
                                                     nullptr,
                                                     "add_fwd_bf16",
                                                     "ADD_B");
    addNodeToGraph(addB);
    NodePtr reshape = NodeFactory::createNode({addB->getOutput(0), createShapeTensor({64, 16, 512, 28})},
                                              {createTensor({64, 16, 512, 28})},
                                              nullptr,
                                              NodeFactory::reshapeNodeTypeName,
                                              "RESHAPE");
    addNodeToGraph(reshape);
    synTransposeParams transposeParams = {{TPD_Channel, TPD_Height, TPD_Width, TPD_4Dim_Batch}, 4};
    NodePtr            transpose       = NodeFactory::createNode({reshape->getOutput(0)},
                                                {createTensor({{64, 512, 16, 28}})},
                                                &transposeParams,
                                                NodeFactory::transposeNodeTypeName,
                                                "TRANSPOSE");
    addNodeToGraph(transpose);

    // BGEMM with TPC producers for both operands
    synGEMMParams bgemmParams(false, false);
    NodePtr       bgemm = NodeFactory::createNode({dropout->getOutput(0), transpose->getOutput(0)},
                                            {createTensor({64, 512, 16, 28}, true)},
                                            &bgemmParams,
                                            NodeFactory::batchGemmNodeTypeName,
                                            "BGEMM");
    addNodeToGraph(bgemm);

    sliceGraph({mult, addA, softmax, dropout, addB, reshape, transpose, bgemm});
}

TEST_P(SlicerTest, slice_bundle_with_broadcast_bgemm)
{
    // Broadcast BGEMM with TPC producers and consumer
    synGEMMParams bgemmParams(false, false);
    NodePtr       bgemm = NodeFactory::createNode({createTensor({30528, 512, 28}), createTensor({1024, 30528, 1})},
                                            {createTensor({1024, 512, 28}, true)},
                                            &bgemmParams,
                                            NodeFactory::batchGemmNodeTypeName,
                                            "BGEMM");
    addNodeToGraph(bgemm);

    NodePtr reluA = NodeFactory::createGenericTPCNode({createTensor({30528, 512, 28}, true)},
                                                      {bgemm->getInput(0)},
                                                      nullptr,
                                                      "relu_fwd_bf16",
                                                      "RELU_A");
    addNodeToGraph(reluA);

    NodePtr reluB = NodeFactory::createGenericTPCNode({createTensor({1024, 30528, 1}, true)},
                                                      {bgemm->getInput(1)},
                                                      nullptr,
                                                      "relu_fwd_bf16",
                                                      "RELU_B");
    addNodeToGraph(reluB);

    NodePtr reluOut = NodeFactory::createGenericTPCNode({bgemm->getOutput(0)},
                                                        {createTensor({1024, 512, 28}, true)},
                                                        nullptr,
                                                        "relu_fwd_bf16",
                                                        "RELU_OUT");
    addNodeToGraph(reluOut);

    sliceGraph({reluOut, bgemm, reluA, reluB});
}

TEST_P(SlicerTest, slice_bundle_with_broadcast_bgemm_and_broadcast_tpc)
{
    // Broadcast BGEMM with broadcast TPC producers and consumer
    synGEMMParams bgemmParams(false, false);
    NodePtr       bgemm = NodeFactory::createNode({createTensor({30528, 512, 28}), createTensor({1024, 30528, 1})},
                                            {createTensor({1024, 512, 28}, true)},
                                            &bgemmParams,
                                            NodeFactory::batchGemmNodeTypeName,
                                            "BGEMM");
    addNodeToGraph(bgemm);

    NodePtr addA =
        NodeFactory::createGenericTPCNode({createTensor({30528, 512, 28}, true), createTensor({30528, 1}, true)},
                                          {bgemm->getInput(0)},
                                          nullptr,
                                          "add_fwd_bf16",
                                          "ADD_A");
    addNodeToGraph(addA);

    NodePtr addB = NodeFactory::createGenericTPCNode({createTensor({1024}, true), createTensor({1024, 30528, 1}, true)},
                                                     {bgemm->getInput(1)},
                                                     nullptr,
                                                     "add_fwd_bf16",
                                                     "ADD_B");
    addNodeToGraph(addB);

    NodePtr addOut = NodeFactory::createGenericTPCNode({bgemm->getOutput(0), createTensor({1024, 1, 1}, true)},
                                                       {createTensor({1024, 512, 28}, true)},
                                                       nullptr,
                                                       "add_fwd_bf16",
                                                       "ADD_OUT");
    addNodeToGraph(addOut);

    sliceGraph({addOut, addA, addB, bgemm});
}

TEST_P(SlicerTest, slice_bundle_with_shared_input_bgemms)
{
    // Create a graph with TPC producer and 2 BGEMMs consumers, one is broadcasted and the other is not.

    NodePtr relu = NodeFactory::createGenericTPCNode({createTensor({30528, 512, 28}, true)},
                                                     {createTensor({30528, 512, 28})},
                                                     nullptr,
                                                     "relu_fwd_bf16",
                                                     "RELU");
    addNodeToGraph(relu);

    synGEMMParams bgemm1Params(false, false);
    NodePtr       bgemm1 = NodeFactory::createNode({relu->getOutput(0), createTensor({1024, 30528, 1}, true)},
                                             {createTensor({1024, 512, 28}, true)},
                                             &bgemm1Params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "BGEMM1");
    addNodeToGraph(bgemm1);

    synGEMMParams bgemm2Params(false, false);
    NodePtr       bgemm2 = NodeFactory::createNode({relu->getOutput(0), createTensor({1024, 30528, 28}, true)},
                                             {createTensor({1024, 512, 28}, true)},
                                             &bgemm2Params,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "BGEMM2");
    addNodeToGraph(bgemm2);

    sliceGraph({bgemm1, bgemm2, relu});
}

TEST_P(SlicerTest, slice_bundle_with_shared_input_gemms)
{
    // Create a graph with 2 GEMMs sharing operand A.

    synGEMMParams gemmParams(false, false);
    NodePtr       gemm1 = NodeFactory::createNode({createTensor({2048, 2048}, true), createTensor({2048, 2048}, true)},
                                            {createTensor({2048, 2048}, true)},
                                            &gemmParams,
                                            NodeFactory::gemmNodeTypeName,
                                            "GEMM1");
    addNodeToGraph(gemm1);
    gemm1->getInput(0)->setName("A", true);
    gemm1->getInput(1)->setName("B1", true);
    gemm1->getOutput(0)->setName("C1", true);

    NodePtr gemm2 = NodeFactory::createNode({gemm1->getInput(0), createTensor({2048, 2048}, true)},
                                            {createTensor({2048, 2048}, true)},
                                            &gemmParams,
                                            NodeFactory::gemmNodeTypeName,
                                            "GEMM2");
    addNodeToGraph(gemm2);
    gemm2->getInput(1)->setName("B2", true);
    gemm2->getOutput(0)->setName("C2", true);

    sliceGraph({gemm1, gemm2});
}

TEST_P(SlicerTest, slice_gemm_with_large_cd)
{
    synGEMMParams gemmParams(false, false);
    NodePtr       gemm = NodeFactory::createNode(
        {createTensor({16384, 512}, true, syn_type_single), createTensor({512, 16384}, true, syn_type_single)},
        {createTensor({512, 512}, true, syn_type_single)},
        &gemmParams,
        NodeFactory::gemmNodeTypeName,
        "GEMM");
    addNodeToGraph(gemm);

    validateCDSlicing({gemm}, gemm, false);
    sliceGraph({gemm});
}

TEST_P(SlicerTest, slice_bf16_gemm_with_large_cd)
{
    synGEMMParams gemmParams(false, false);
    NodePtr       gemm = NodeFactory::createNode(
        {createTensor({16384, 512}, true, syn_type_bf16), createTensor({512, 16384}, true, syn_type_bf16)},
        {createTensor({512, 512}, true, syn_type_bf16)},
        &gemmParams,
        NodeFactory::gemmNodeTypeName,
        "GEMM");
    addNodeToGraph(gemm);

    validateCDSlicing({gemm}, gemm, true);
    sliceGraph({gemm});
}

TEST_P(SlicerTest, inflate_for_utilization_single_node)
{
    synConvolutionParams convParams;
    NodePtr              conv = NodeFactory::createNode(
        {createTensor({64, 56, 56, 64}, true, syn_type_bf16), createTensor({64, 64, 1, 1}, true, syn_type_bf16)},
        {createTensor({64, 56, 56, 64}, true, syn_type_bf16)},
        &convParams,
        NodeFactory::convolutionNodeTypeName,
        "CONV");
    addNodeToGraph(conv);

    Slicer slicer(m_graph, BUNDLE_IDX, {conv});
    bool   validStrategyFound = false;
    for (const auto& strategy : slicer.getStrategies())
    {
        auto oldUtilization = strategy->getNodeQORs(conv)->perfAttr.mmeUtilization;
        bool res = slicer.inflateStrategy(gc::layered_brain::InflationType::INFLATE_FOR_UTILIZATION, strategy, conv);
        auto newUtilization = strategy->getNodeQORs(conv)->perfAttr.mmeUtilization;
        if (!strategy->getMMEInflateForUtilizationBVDs(conv).empty())
        {
            ASSERT_TRUE(res);
            ASSERT_GT(newUtilization, oldUtilization);
            validStrategyFound = true;
        }
        else
        {
            ASSERT_FALSE(res);
            ASSERT_EQ(newUtilization, oldUtilization);
        }
    }
    ASSERT_TRUE(validStrategyFound);

    sliceGraph({conv});
}

TEST_P(SlicerTest, inflate_for_utilization_2_nodes_weights_sharing)
{
    synConvolutionParams convParams;
    NodePtr              conv1 = NodeFactory::createNode(
        {createTensor({64, 56, 56, 64}, true, syn_type_bf16), createTensor({64, 64, 1, 1}, true, syn_type_bf16)},
        {createTensor({64, 56, 56, 64}, true, syn_type_bf16)},
        &convParams,
        NodeFactory::convolutionNodeTypeName,
        "CONV1");
    addNodeToGraph(conv1);
    NodePtr conv2 = NodeFactory::createNode({createTensor({64, 56, 56, 64}, true, syn_type_bf16), conv1->getInput(1)},
                                            {createTensor({64, 56, 56, 64}, true, syn_type_bf16)},
                                            &convParams,
                                            NodeFactory::convolutionNodeTypeName,
                                            "CONV2");
    addNodeToGraph(conv2);

    Slicer slicer(m_graph, BUNDLE_IDX, {conv1, conv2});
    bool   validStrategyFound = false;
    for (const auto& strategy : slicer.getStrategies())
    {
        const auto& inflateBVDsForConv1 = strategy->getMMEInflateForUtilizationBVDs(conv1);
        const auto& inflateBVDsForConv2 = strategy->getMMEInflateForUtilizationBVDs(conv2);
        if (!inflateBVDsForConv1.empty() && !inflateBVDsForConv2.empty())
        {
            for (const auto& inflateBVD : inflateBVDsForConv1) // Expect different IFU dims for each conv
            {
                ASSERT_TRUE(std::find(inflateBVDsForConv2.begin(), inflateBVDsForConv2.end(), inflateBVD) ==
                            inflateBVDsForConv2.end());
            }

            auto oldUtilizationConv1 = strategy->getNodeQORs(conv1)->perfAttr.mmeUtilization;
            auto oldUtilizationConv2 = strategy->getNodeQORs(conv2)->perfAttr.mmeUtilization;
            bool res =
                slicer.inflateStrategy(gc::layered_brain::InflationType::INFLATE_FOR_UTILIZATION, strategy, conv1);
            auto newUtilizationConv1 = strategy->getNodeQORs(conv1)->perfAttr.mmeUtilization;
            ASSERT_TRUE(res);
            ASSERT_GT(newUtilizationConv1, oldUtilizationConv1);
            ASSERT_EQ(strategy->getNodeQORs(conv2)->perfAttr.mmeUtilization, oldUtilizationConv2);

            res = slicer.inflateStrategy(gc::layered_brain::InflationType::INFLATE_FOR_UTILIZATION, strategy, conv2);
            auto newUtilizationConv2 = strategy->getNodeQORs(conv2)->perfAttr.mmeUtilization;
            ASSERT_TRUE(res);
            ASSERT_GT(newUtilizationConv2, oldUtilizationConv2);
            ASSERT_EQ(strategy->getNodeQORs(conv1)->perfAttr.mmeUtilization, newUtilizationConv1);

            validStrategyFound = true;
        }
    }
    ASSERT_TRUE(validStrategyFound);

    sliceGraph({conv1, conv2});
}

TEST_P(SlicerTest, DISABLED_inflate_for_utilization_2_nodes_input_sharing) // TODO: enable once SW-160644 is resolved
{
    synConvolutionParams convParams;
    NodePtr              conv1 = NodeFactory::createNode(
        {createTensor({64, 56, 56, 64}, true, syn_type_bf16), createTensor({64, 64, 1, 1}, true, syn_type_bf16)},
        {createTensor({64, 56, 56, 64}, true, syn_type_bf16)},
        &convParams,
        NodeFactory::convolutionNodeTypeName,
        "CONV1");
    addNodeToGraph(conv1);
    NodePtr conv2 = NodeFactory::createNode({conv1->getInput(0), createTensor({64, 64, 1, 1}, true, syn_type_bf16)},
                                            {createTensor({64, 56, 56, 64}, true, syn_type_bf16)},
                                            &convParams,
                                            NodeFactory::convolutionNodeTypeName,
                                            "CONV2");
    addNodeToGraph(conv2);

    Slicer slicer(m_graph, BUNDLE_IDX, {conv1, conv2});
    bool   validStrategyFound = false;
    for (const auto& strategy : slicer.getStrategies())
    {
        if (!strategy->getMMEInflateForUtilizationBVDs(conv1).empty() &&
            !strategy->getMMEInflateForUtilizationBVDs(conv2).empty())
        {
            ASSERT_EQ(strategy->getMMEInflateForUtilizationBVDs(conv1),
                      strategy->getMMEInflateForUtilizationBVDs(conv2));
            auto oldUtilizationConv1 = strategy->getNodeQORs(conv1)->perfAttr.mmeUtilization;
            auto oldUtilizationConv2 = strategy->getNodeQORs(conv2)->perfAttr.mmeUtilization;
            bool res =
                slicer.inflateStrategy(gc::layered_brain::InflationType::INFLATE_FOR_UTILIZATION, strategy, conv1);
            ASSERT_TRUE(res);
            ASSERT_GT(strategy->getNodeQORs(conv1)->perfAttr.mmeUtilization, oldUtilizationConv1);
            ASSERT_GT(strategy->getNodeQORs(conv2)->perfAttr.mmeUtilization, oldUtilizationConv2);

            validStrategyFound = true;
        }
    }
    ASSERT_TRUE(validStrategyFound);

    sliceGraph({conv1, conv2});
}

TEST_P(SlicerTest, DISABLED_inflate_for_utilization_2_nodes_with_conflict) // TODO: enable once SW-160644 is resolved
{
    // Create the following sequence: CONV1 -> reshape -> CONV2
    synConvolutionParams convParams;
    NodePtr              conv1 = NodeFactory::createNode(
        {createTensor({80, 10, 10, 64}, true, syn_type_bf16), createTensor({80, 80, 1, 1}, true, syn_type_bf16)},
        {createTensor({80, 10, 10, 64}, false, syn_type_bf16)},
        &convParams,
        NodeFactory::convolutionNodeTypeName,
        "CONV1");
    addNodeToGraph(conv1);
    NodePtr reshape = NodeFactory::createNode({conv1->getOutput(0)},
                                              {createTensor({20, 20, 20, 64}, false, syn_type_bf16)},
                                              nullptr,
                                              NodeFactory::reshapeNodeTypeName,
                                              "RESHAPE");
    addNodeToGraph(reshape);
    NodePtr conv2 = NodeFactory::createNode({reshape->getOutput(0), createTensor({20, 20, 1, 1}, true, syn_type_bf16)},
                                            {createTensor({20, 20, 20, 64}, true, syn_type_bf16)},
                                            &convParams,
                                            NodeFactory::convolutionNodeTypeName,
                                            "CONV2");
    addNodeToGraph(conv2);

    Slicer slicer(m_graph, BUNDLE_IDX, {conv1, conv2, reshape});
    bool   validStrategyFound = false;
    for (const auto& strategy : slicer.getStrategies())
    {
        // conv1 is solved before conv2 (has more consumers in the bundle) -
        // since the 2 nodes are conflicting, conv1 should get priority for inflation.
        ASSERT_TRUE(strategy->getMMEInflateForUtilizationBVDs(conv2).empty());
        if (!strategy->getMMEInflateForUtilizationBVDs(conv1).empty())
        {
            auto oldUtilizationConv1 = strategy->getNodeQORs(conv1)->perfAttr.mmeUtilization;
            auto oldUtilizationConv2 = strategy->getNodeQORs(conv2)->perfAttr.mmeUtilization;
            bool res =
                slicer.inflateStrategy(gc::layered_brain::InflationType::INFLATE_FOR_UTILIZATION, strategy, conv1);
            ASSERT_TRUE(res);
            ASSERT_GT(strategy->getNodeQORs(conv1)->perfAttr.mmeUtilization, oldUtilizationConv1);
            ASSERT_NE(strategy->getNodeQORs(conv2)->perfAttr.mmeUtilization, oldUtilizationConv2);
            validStrategyFound = true;
        }
    }
    ASSERT_TRUE(validStrategyFound);

    sliceGraph({conv1, conv2, reshape});
}

INSTANTIATE_TEST_SUITE_P(slicer_test, SlicerTest, ::testing::Values(false, true));  // fwd progress / iterative mode
