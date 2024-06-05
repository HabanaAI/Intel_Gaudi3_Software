#include "bundle_view.h"
#include "slicer/bundle_views_collector.h"
#include "gaudi3_graph.h"
#include "operation_slice.h"
#include "scoped_configuration_change.h"
#include "slicer/sliced_node_generator.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tile.h"
#include "types.h"
#include "tpc_slicing_test_infra.h"
#include "compilation_hal_reader.h"
#include <optional>

using namespace gc::layered_brain;

class SlicedNodeGeneratorTest : public GraphOptimizerTest
{
public:
    using TileSizePerBVD   = std::vector<TSize>;
    using TileOffsetPerBVD = std::vector<TOffset>;

    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        m_enablePerforation = std::make_unique<ScopedConfigurationChange>("ENABLE_LAYERED_BRAIN_PERFORATION", "true");
    }

protected:
    TensorPtr createTensor(const std::vector<TSize>& shape)
    {
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
    }

    TensorPtr createTensor() { return createTensor(m_tensorSizes); }

    NodePtr createMMENode()
    {
        synGEMMParams params(false, false);
        NodePtr       node = NodeFactory::createNode({createTensor(), createTensor()},
                                               {createTensor()},
                                               &params,
                                               NodeFactory::gemmNodeTypeName,
                                               "MME");
        EXPECT_TRUE(node);
        return node;
    }

    NodePtr createGEMMWithLargeCD()
    {
        synGEMMParams params(false, false);
        NodePtr       node = NodeFactory::createNode({createTensor({16384, 512}), createTensor({512, 16384})},
                                               {createTensor({512, 512})},
                                               &params,
                                               NodeFactory::gemmNodeTypeName,
                                               "MME");
        EXPECT_TRUE(node);
        return node;
    }

    NodePtr createTPCNode()
    {
        NodePtr node = TPCCustomIndexSpaceNode::createSliceableNode({createTensor()}, {createTensor()});
        EXPECT_TRUE(node);
        return node;
    }

    void createBundleViews(const NodePtr& node, TSize granularity)
    {
        BundleViewsCollector bvdCollector({node});
        TileSizePerTensor    granularityPerTensor;
        TileSizePerNode      granularityPerNode;
        HB_ASSERT_PTR(node->getNodeAccessPattern());
        granularityPerNode[node] =
            NodeTile::Geometry(node->getNodeAccessPattern()->getNodeResolution().size(), granularity);
        for (const auto& t : node->getOperands())
        {
            granularityPerTensor[t] = TensorTile::Geometry(t->getDim(), granularity);
        }
        m_bundleViews = bvdCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);
    }

    NodeTile translateSizeAndOffsetPerBVDToNodeTile(const NodePtr&          origNode,
                                                    const TileSizePerBVD&   size,
                                                    const TileOffsetPerBVD& offset) const
    {
        const auto&        fullTile = origNode->getNodeAccessPattern()->getNodeResolution();
        NodeTile           sliceTile(fullTile);

        HB_ASSERT(size.size() == m_bundleViews->getNumOfBundleViews(),
                  "size rank should be {}",
                  m_bundleViews->getNumOfBundleViews());
        HB_ASSERT(offset.size() == m_bundleViews->getNumOfBundleViews(),
                  "offset rank should be {}",
                  m_bundleViews->getNumOfBundleViews());

        // Translate from bundle-view dims to node dims.
        for (Dim nodeDim = 0; nodeDim < fullTile.size(); nodeDim++)
        {
            if (m_bundleViews->isNodeDimMappedToBVD(origNode, nodeDim))
            {
                BundleViewId bvd             = m_bundleViews->getBVDForNodeDim(origNode, nodeDim);
                sliceTile.geometry[nodeDim]  = size.at(bvd);
                sliceTile.offset[nodeDim]    = offset.at(bvd);
            }
        }

        return sliceTile;
    }

    void validateSlicedNodeAnnotations(const NodePtr&          slicedNode,
                                       const NodePtr&          origNode,
                                       BundleIdx               bundleIdx,
                                       const TileSizePerBVD&   expectedSize,
                                       const TileOffsetPerBVD& expectedOffset) const
    {
        ASSERT_TRUE(slicedNode);
        ASSERT_TRUE(slicedNode->getNodeAnnotation().bundleInfo.is_set());
        ASSERT_EQ(slicedNode->getNodeAnnotation().bundleInfo->bundleIndex, bundleIdx);
        ASSERT_TRUE(slicedNode->getNodeName().find(origNode->getNodeName()) != std::string::npos);
        ASSERT_EQ(slicedNode->getNodeAnnotation().origBigNode, origNode);
        ASSERT_TRUE(slicedNode->getNodeAnnotation().sliceROI.has_value());

        const auto& expectedSliceTile = translateSizeAndOffsetPerBVDToNodeTile(origNode, expectedSize, expectedOffset);

        ASSERT_EQ(slicedNode->getNodeAnnotation().sliceROI->geometry, expectedSliceTile.geometry);
        ASSERT_EQ(slicedNode->getNodeAnnotation().sliceROI->offset, expectedSliceTile.offset);
    }

    const std::vector<TSize>                   m_tensorSizes = {128, 128};
    Gaudi3Graph                                m_graph;
    CompilationHalReaderSetter                 m_halReaderSetter {&m_graph};
    BundleViewContainerPtr                     m_bundleViews;
    std::unique_ptr<ScopedConfigurationChange> m_enablePerforation;
};

TEST_F(SlicedNodeGeneratorTest, generate_mme_slices_no_perforation)
{
    const BundleIdx bundleIdx = 10;

    // Create a GEMM node with dim size = 128, granularity = 2, 3 BVDs
    const TSize granularity = 2;
    NodePtr     origNode    = createMMENode();
    createBundleViews(origNode, granularity);
    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 3);
    auto        mmeSolution = std::make_shared<MmeSolution>();
    StrategyPtr strategy    = std::make_shared<Strategy>(mmeSolution);
    strategy->setBVDMultiplier(0, BVDMultiplier());      // BVD 0 has 1 slice: 128
    strategy->setBVDMultiplier(1, BVDMultiplier(20UL));  // BVD 1 has 4 slices: 40, 40, 40, 8
    strategy->setBVDMultiplier(2, BVDMultiplier(32UL));  // BVD 2 has 2 slices: 64, 64
    PerforationPerNode perforation;
    perforation[origNode] = std::nullopt;
    strategy->setPerforationData(perforation);
    SlicedNodeGenerator slicedNodeGenerator(bundleIdx, m_bundleViews, strategy);

    BVDCoord coord1      = {0, 1, 0};
    NodePtr  slicedNode1 = slicedNodeGenerator.getSlicedNode(origNode, coord1);
    validateSlicedNodeAnnotations(slicedNode1, origNode, bundleIdx, {128, 40, 64}, {0, 40, 0});
    ASSERT_TRUE(slicedNode1->getNodeAnnotation().m_dcoreROIs.empty());  // No perforation is expected

    BVDCoord coord2      = {0, 3, 1};  // Last slice is smaller
    NodePtr  slicedNode2 = slicedNodeGenerator.getSlicedNode(origNode, coord2);
    validateSlicedNodeAnnotations(slicedNode2, origNode, bundleIdx, {128, 8, 64}, {0, 120, 64});
    ASSERT_TRUE(slicedNode2->getNodeAnnotation().m_dcoreROIs.empty());  // No perforation is expected

    ASSERT_NE(slicedNode1->getNodeName(), slicedNode2->getNodeName());
}

TEST_F(SlicedNodeGeneratorTest, generate_tpc_slices_no_perforation)
{
    const BundleIdx bundleIdx = 15;

    // Create a TPC node with dim size = 128, granularity = 4, 2 BVDs (1:1 mapping between tensor-dims and node-dims)
    const TSize granularity = 4;
    NodePtr     origNode    = createTPCNode();
    createBundleViews(origNode, granularity);
    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 2);
    StrategyPtr strategy = std::make_shared<Strategy>();
    strategy->setBVDMultiplier(0, BVDMultiplier(8UL));  // BVD 0 has 4 slices: 32, 32, 32, 32
    strategy->setBVDMultiplier(1, BVDMultiplier());     // BVD 1 has 1 slice: 128
    PerforationPerNode perforation;
    perforation[origNode] = std::nullopt;
    strategy->setPerforationData(perforation);
    SlicedNodeGenerator slicedNodeGenerator(bundleIdx, m_bundleViews, strategy);

    BVDCoord coord      = {1, 0};
    NodePtr  slicedNode = slicedNodeGenerator.getSlicedNode(origNode, coord);
    validateSlicedNodeAnnotations(slicedNode, origNode, bundleIdx, {32, 128}, {32, 0});
    ASSERT_TRUE(slicedNode->getNodeAnnotation().m_dcoreROIs.empty());  // No perforation is expected

    TensorPtr origTensor   = origNode->getInput(0);
    TensorPtr slicedTensor = origTensor->clone();
    slicedNode->replaceInput(0, slicedTensor);
    OffsetArray slicedTensorOffset;
    slicedTensorOffset.fill(0);
    slicedTensorOffset[0] = 4;
    slicedTensorOffset[1] = 2;
    slicedNodeGenerator.updateTensorSliceOffset(slicedNode, origTensor, slicedTensor, slicedTensorOffset);
    auto operationSlice = std::dynamic_pointer_cast<OperationSlice>(slicedNode);
    ASSERT_TRUE(operationSlice);
    ASSERT_EQ(operationSlice->getOriginalTensor(slicedTensor), origTensor);
    ASSERT_EQ(operationSlice->getTensorSliceOffsetInDim(slicedTensor, 0), slicedTensorOffset[0]);
    ASSERT_EQ(operationSlice->getTensorSliceOffsetInDim(slicedTensor, 1), slicedTensorOffset[1]);
}

TEST_F(SlicedNodeGeneratorTest, generate_tpc_slices_with_perforation)
{
    const BundleIdx bundleIdx = 25;

    // Create a TPC node with dim size = 128, granularity = 4, 2 BVDs (1:1 mapping between tensor-dims and node-dims)
    const TSize granularity = 4;
    NodePtr     origNode    = createTPCNode();
    createBundleViews(origNode, granularity);
    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 2);  // 1:1 mapping between tensor dims and node dims
    StrategyPtr strategy = std::make_shared<Strategy>();
    strategy->setBVDMultiplier(0, BVDMultiplier(8UL));  // BVD 0 has 4 slices: 32, 32, 32, 32
    strategy->setBVDMultiplier(1, BVDMultiplier());     // BVD 1 has 1 slice: 128
    PerforationPerNode perforation;
    perforation[origNode] = 0;  // Perforate on BVD 0
    strategy->setPerforationData(perforation);
    SlicedNodeGenerator slicedNodeGenerator(bundleIdx, m_bundleViews, strategy);

    BVDCoord coord      = {2, 0};
    NodePtr  slicedNode = slicedNodeGenerator.getSlicedNode(origNode, coord);
    validateSlicedNodeAnnotations(slicedNode, origNode, bundleIdx, {32, 128}, {64, 0});

    const auto& slicedROISize   = slicedNode->getNodeAnnotation().sliceROI->geometry;
    const auto& slicedROIOffset = slicedNode->getNodeAnnotation().sliceROI->offset;
    ASSERT_EQ(m_bundleViews->getNodeDimsInBVD(0, origNode).size(), 1);
    ASSERT_EQ(m_bundleViews->getNodeDimsInBVD(1, origNode).size(), 1);
    unsigned nodeDimBVD0 = m_bundleViews->getNodeDimsInBVD(0, origNode).front();
    unsigned nodeDimBVD1 = m_bundleViews->getNodeDimsInBVD(1, origNode).front();

    const unsigned expectedNumDcores = 4;
    ASSERT_EQ(slicedNode->getNodeAnnotation().m_dcoreROIs.size(), expectedNumDcores);
    for (auto dcore = 0; dcore < expectedNumDcores; dcore++)
    {
        const auto& dcoreROI = slicedNode->getNodeAnnotation().m_dcoreROIs.at(dcore);
        // BVD 0 is perforated
        ASSERT_EQ(dcoreROI.size[nodeDimBVD0], slicedROISize.at(nodeDimBVD0) / expectedNumDcores);
        ASSERT_EQ(dcoreROI.baseOffset[nodeDimBVD0],
                  slicedROIOffset.at(nodeDimBVD0) + (dcore * dcoreROI.size[nodeDimBVD0]));

        // BVD 1 is not perforated
        ASSERT_EQ(dcoreROI.size[nodeDimBVD1], slicedROISize.at(nodeDimBVD1));
        ASSERT_EQ(dcoreROI.baseOffset[nodeDimBVD1], slicedROIOffset.at(nodeDimBVD1));
    }
}

TEST_F(SlicedNodeGeneratorTest, generate_mme_slices_with_cd_perforation)
{
    setGlobalConfForTest(GCFG_ENABLE_CD_PARALLEL, "true");
    const BundleIdx bundleIdx = 25;

    const TSize granularity = 1;
    NodePtr     gemm        = createGEMMWithLargeCD();
    createBundleViews(gemm, granularity);
    auto        mmeSolution = std::make_shared<MmeSolution>();
    StrategyPtr strategy    = std::make_shared<Strategy>(mmeSolution);
    for (auto bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        strategy->setBVDMultiplier(bvd, BVDMultiplier());  // All BVDs are unsliced
    }
    BundleViewId       perforationBVD = m_bundleViews->getBVDForTensorDim(gemm->getInput(0), 0);  // Perforate on CD
    PerforationPerNode perforation;
    perforation[gemm] = perforationBVD;
    strategy->setPerforationData(perforation);

    auto mmeNode = std::dynamic_pointer_cast<MmeNode>(gemm);
    ASSERT_TRUE(mmeNode);
    strategy->getMmeSolution()->QORs[gemm]                                         = std::make_shared<SolutionParams>();
    strategy->getMmeSolution()->QORs[gemm]->solutionRequirements.perforationDimVec = {perforationBVD};
    auto mmeParams                                  = mmeNode->getMmeBrainIfc()->getRecommendedMmeLayerParams();
    mmeParams.strategy.cdConcurrencyEn              = MmeCommon::BoolWithUndef::TurnedOff;
    mmeParams.strategy.isDeterministic              = false;
    strategy->getMmeSolution()->brainSolution[gemm] = std::make_shared<MmeCommon::MmeBrainSolution>();
    strategy->getMmeSolution()->brainSolution[gemm]->strategy = mmeParams.strategy;

    SlicedNodeGenerator slicedNodeGenerator(bundleIdx, m_bundleViews, strategy);

    BVDCoord coord      = {0, 0, 0};
    NodePtr  slicedNode = slicedNodeGenerator.getSlicedNode(gemm, coord);
    slicedNodeGenerator.addAuxTensors(gemm, slicedNode);

    ASSERT_FALSE(slicedNode->getNodeAnnotation().m_dcoreROIs.empty());
    ASSERT_TRUE(slicedNode->getNodeAnnotation().perforationDim.has_value());

    ASSERT_GT(slicedNode->getNumInputs(), gemm->getNumInputs());
    ASSERT_TRUE(slicedNode->getInput(TENSOR_AUX_CD_SCRATCHPAD));
    ASSERT_TRUE(slicedNode->getInput(TENSOR_AUX_CD_SCRATCHPAD)->isAuxTensor());
    ASSERT_TRUE(slicedNode->getInput(TENSOR_AUX_CD_REDUCTION));
    ASSERT_TRUE(slicedNode->getInput(TENSOR_AUX_CD_REDUCTION)->isAuxTensor());
}