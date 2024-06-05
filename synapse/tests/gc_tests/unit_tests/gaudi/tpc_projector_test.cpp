#include "compilation_hal_reader.h"
#include "graph_optimizer_test.h"
#include "gaudi_graph.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "sram_management/pipeline_management/node_projector.h"
#include "platform/gaudi/graph_compiler/passes.h"

class TPCProjectorTest : public GraphOptimizerTest
{
protected:
    GaudiGraph                 m_graph;
    CompilationHalReaderSetter m_halSetter;

    TPCProjectorTest() : m_halSetter(&m_graph) {}

    void validateProjection(const NodePtr& node, const SlicingStrategyPtr& nodeStrategy, unsigned slicedBatchSize) const
    {
        const auto& slicedOperands = nodeStrategy->getSlicingData().getSlicedOperands();
        ASSERT_EQ(node->getOperands().size(), slicedOperands.size());

        // Verify that the node operands are sliced correctly on batch dim, other dims should remain unsliced.
        for (const auto& slicedOperand : slicedOperands)
        {
            ASSERT_TRUE(slicedOperand);
            for (auto i = 0; i < slicedOperand->originalTensor->getDim(); i++)
            {
                if (i != DIM_B)
                {
                    // Dimension is not sliced
                    EXPECT_EQ(slicedOperand->chunkDimensions[i], slicedOperand->originalTensor->getSizeInElements(i));
                }
                else
                {
                    // Batch dimension is sliced
                    EXPECT_EQ(slicedOperand->chunkDimensions[i], slicedBatchSize);
                }
            }
        }
    }
};

TEST_F(TPCProjectorTest, project_maxpool_with_overlap)
{
    synConvolutionParams        convParams(3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
    const std::vector<TSize>    dySizes = {256, 21, 13, 4};
    const std::vector<TSize>    wSizes  = {256, 256, 3, 3};
    const std::vector<TSize>    dxSizes = {256, 21, 13, 4};
    TensorPtr                   dy      = TensorPtr(new Tensor(dySizes.size(), dySizes.data(), syn_type_bf16));
    TensorPtr                   w       = TensorPtr(new Tensor(wSizes.size(), wSizes.data(), syn_type_bf16));
    TensorPtr                   dx      = TensorPtr(new Tensor(dxSizes.size(), dxSizes.data(), syn_type_bf16));
    pNode dedx = NodeFactory::createNode({dy, w}, {dx}, &convParams, NodeFactory::deDxNodeTypeName, "dedx");
    ASSERT_TRUE(GraphEditor::addNode(m_graph, dedx));

    ns_SpatialReduction::Params maxpoolParams;
    maxpoolParams.pad_w_begin                   = 0;
    maxpoolParams.pad_w_end                     = 0;
    maxpoolParams.pad_h_begin                   = 0;
    maxpoolParams.pad_h_end                     = 0;
    maxpoolParams.kernel_w                      = 1;
    maxpoolParams.kernel_h                      = 1;
    maxpoolParams.stride_w                      = 2;
    maxpoolParams.stride_h                      = 2;
    maxpoolParams.dilation_w                    = 1;
    maxpoolParams.dilation_h                    = 1;
    maxpoolParams.pooling_convention            = POOLING_CONVENTION_VALID;
    const std::vector<TSize>    poolingOutSizes = {256, 42, 25, 4};
    TensorPtr                   poolingIn       = TensorPtr(new Tensor(dxSizes.size(), dxSizes.data(), syn_type_int16));
    TensorPtr poolingOut = TensorPtr(new Tensor(poolingOutSizes.size(), poolingOutSizes.data(), syn_type_bf16));
    pNode     maxpool    = NodeFactory::createGenericTPCNode({dx, poolingIn},
                                                      {poolingOut},
                                                      &maxpoolParams,
                                                      "maxpool_2d_bwd_bf16",
                                                      "maxpool");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, maxpool));
    ASSERT_TRUE(gaudi::loadTpcKernels(m_graph));

    auto                 bundleStrategy = SlicingStrategy::createStrategy(*m_graph.getHALReader(), dedx);
    StrategySlicingData& slicingData    = bundleStrategy->getSlicingData();
    pSlicedOperand       slicedDy       = slicingData.getSlicedOperand(dy);
    ASSERT_NE(slicedDy, nullptr);
    pSlicedOperand slicedDx = slicingData.getSlicedOperand(dx);
    ASSERT_NE(slicedDx, nullptr);
    // Slice on batch dim
    unsigned slicedBatchSize         = 3;
    slicedDy->chunkDimensions[DIM_B] = slicedBatchSize;
    slicedDx->chunkDimensions[DIM_B] = slicedBatchSize;

    AccessPatternNodeSolutionProjector projector(maxpool);
    const auto&                        maxpoolStrategy = projector.getNodeStrategy(bundleStrategy, dx);

    validateProjection(maxpool, maxpoolStrategy, slicedBatchSize);
}

TEST_F(TPCProjectorTest, project_maxpool_with_overlap_and_negative_offset)
{
    synConvolutionParams        convParams(1, 1, 1, 1, 0, 0, 0, 0, 1, 1);
    const std::vector<TSize>    dySizes = {128, 7, 7, 256};
    const std::vector<TSize>    wSizes  = {128, 832, 1, 1};
    const std::vector<TSize>    dxSizes = {832, 7, 7, 256};
    TensorPtr                   dy      = TensorPtr(new Tensor(dySizes.size(), dySizes.data(), syn_type_bf16));
    TensorPtr                   w       = TensorPtr(new Tensor(wSizes.size(), wSizes.data(), syn_type_bf16));
    TensorPtr                   dx      = TensorPtr(new Tensor(dxSizes.size(), dxSizes.data(), syn_type_bf16));
    pNode dedx = NodeFactory::createNode({dy, w}, {dx}, &convParams, NodeFactory::deDxNodeTypeName, "dedx");
    ASSERT_TRUE(GraphEditor::addNode(m_graph, dedx));

    ns_SpatialReduction::Params maxpoolParams;
    maxpoolParams.pad_w_begin                   = 1;
    maxpoolParams.pad_w_end                     = 1;
    maxpoolParams.pad_h_begin                   = 1;
    maxpoolParams.pad_h_end                     = 1;
    maxpoolParams.kernel_w                      = 3;
    maxpoolParams.kernel_h                      = 3;
    maxpoolParams.stride_w                      = 1;
    maxpoolParams.stride_h                      = 1;
    maxpoolParams.dilation_w                    = 1;
    maxpoolParams.dilation_h                    = 1;
    maxpoolParams.pooling_convention            = POOLING_CONVENTION_FULL;
    const std::vector<TSize>    poolingOutSizes = {832, 7, 7, 256};
    TensorPtr                   poolingIn       = TensorPtr(new Tensor(dxSizes.size(), dxSizes.data(), syn_type_int16));
    TensorPtr poolingOut = TensorPtr(new Tensor(poolingOutSizes.size(), poolingOutSizes.data(), syn_type_bf16));
    pNode     maxpool    = NodeFactory::createGenericTPCNode({dx, poolingIn},
                                                      {poolingOut},
                                                      &maxpoolParams,
                                                      "maxpool_2d_bwd_bf16",
                                                      "maxpool");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, maxpool));
    ASSERT_TRUE(gaudi::loadTpcKernels(m_graph));

    auto                 bundleStrategy = SlicingStrategy::createStrategy(*m_graph.getHALReader(), dedx);
    StrategySlicingData& slicingData    = bundleStrategy->getSlicingData();
    pSlicedOperand       slicedDy       = slicingData.getSlicedOperand(dy);
    ASSERT_NE(slicedDy, nullptr);
    pSlicedOperand slicedDx = slicingData.getSlicedOperand(dx);
    ASSERT_NE(slicedDx, nullptr);
    // Slice on batch dim
    unsigned slicedBatchSize         = 51;
    slicedDy->chunkDimensions[DIM_B] = slicedBatchSize;
    slicedDx->chunkDimensions[DIM_B] = slicedBatchSize;

    AccessPatternNodeSolutionProjector projector(maxpool);
    const auto&                        maxpoolStrategy = projector.getNodeStrategy(bundleStrategy, dx);

    validateProjection(maxpool, maxpoolStrategy, slicedBatchSize);
}