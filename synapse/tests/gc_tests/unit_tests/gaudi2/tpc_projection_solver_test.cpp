#include "compilation_hal_reader.h"
#include "graph_optimizer_test.h"
#include "gaudi2_graph.h"
#include "node_factory.h"
#include "passes/sram_management/pipeline_management/node_projector.h"
#include "passes/sram_management/pipeline_management/node_solver.h"
#include "sram_management/pipeline_management/node_projector.h"
#include "sram_management/slicing_strategy.h"
#include "tpc_slicing_test_infra.h"
#include "utils.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "sram_management/slicing_utils.h"

class TPCNodeProjectorTest : public GraphOptimizerTest,
public ::testing::WithParamInterface<
    std::tuple< unsigned,   // FCD size
                unsigned,   // FCD granularity
                int,        // FCD input overlap
                unsigned,   // SCD size
                unsigned,   // SCD granularity
                int,        // SCD input overlap
                bool        // output transposed compared to input
>>
{
    using BaseClass = GraphOptimizerTest;

protected:
    Gaudi2Graph                m_graph;
    CompilationHalReaderSetter m_halSetter;
    NodePtr                    m_tpcNode;
    NodePtr                    m_gemm;

    unsigned fcdSize;
    unsigned fcdGranularity;
    int      fcdInputOverlap;
    unsigned scdSize;
    unsigned scdGranularity;
    int      scdInputOverlap;
    bool     transpose;

    TPCNodeProjectorTest() : m_halSetter(&m_graph)
    {
        std::tie(fcdSize, fcdGranularity, fcdInputOverlap, scdSize, scdGranularity, scdInputOverlap, transpose) =
            GetParam();
    }
    virtual ~TPCNodeProjectorTest() = default;

    virtual void test()
    {
        addTPCNode();
        addGEMMConsumer();

        const AccessPatternNodeSolutionProjector projector(m_tpcNode);

        SlicingStrategyPtr bundleStrategy = createBundleStrategy();

        for (const auto& fcd : DimSliceRange(fcdSize, fcdGranularity))
        {
            if (fcd.offset) continue;  // This test does not need to check any offset != 0
            for (const auto& scd : DimSliceRange(scdSize, scdGranularity))
            {
                if (scd.offset) continue;  // This test does not need to check any offset != 0

                sliceGemmInput(bundleStrategy, fcd.sliceSize, scd.sliceSize);
                ASSERT_FALSE(HasFailure());

                const TensorPtr& gemmIn       = m_tpcNode->getOutput(0);
                auto             nodeStrategy = projector.getNodeStrategy(bundleStrategy, gemmIn);
                ASSERT_NE(nullptr, nodeStrategy);

                validateProjection(bundleStrategy, nodeStrategy);
            }
        }
    }

    virtual void addTPCNode()
    {
        TPCCustomIndexSpaceNode::Params nodeParams;
        nodeParams.dims.emplace_back(fcdSize, fcdGranularity, fcdInputOverlap);  // dims[0]
        nodeParams.dims.emplace_back(scdSize, scdGranularity, scdInputOverlap);  // dims[1]
        nodeParams.transpose = transpose;
        m_tpcNode            = TPCCustomIndexSpaceNode::create(nodeParams);
        GraphEditor::addNode(m_graph, m_tpcNode);
    }

    void addGEMMConsumer()
    {
        const TensorPtr& gemmIn = m_tpcNode->getOutput(0);

        TSize     wghSizes[] = {512, gemmIn->getSizeInElements(0)};
        TensorPtr wgh        = std::make_shared<Tensor>(ARRAY_SIZE(wghSizes), wghSizes, syn_type_float);

        TSize     ofmSizes[] = {wgh->getSizeInElements(0), gemmIn->getSizeInElements(1)};
        TensorPtr ofm        = std::make_shared<Tensor>(ARRAY_SIZE(ofmSizes), ofmSizes, syn_type_float);

        synGEMMParams params {};
        m_gemm = NodeFactory::createNode({gemmIn, wgh}, {ofm}, &params, NodeFactory::gemmNodeTypeName, "gemm");

        GraphEditor::addNode(m_graph, m_gemm);
    }

    SlicingStrategyPtr createBundleStrategy()
    {
        auto strategy = SlicingStrategy::createStrategy(*m_graph.getHALReader(), m_gemm);
        return strategy;
    }

    void sliceGemmInput(SlicingStrategyPtr& bundleStrategy, unsigned fcdSliceSize, unsigned scdSliceSize)
    {
        const TensorPtr&     connectingTensor = m_tpcNode->getOutput(0);
        StrategySlicingData& slicingData      = bundleStrategy->getSlicingData();
        pSlicedOperand       slicedGemmInput  = slicingData.getSlicedOperand(connectingTensor);
        ASSERT_NE(nullptr, slicedGemmInput) << "Sliced TPC output wasn't found";
        slicedGemmInput->chunkDimensions[transpose ? 1 : 0] = fcdSliceSize;
        slicedGemmInput->chunkDimensions[transpose ? 0 : 1] = scdSliceSize;

        LOG_DEBUG(GO_TEST,
                  "ChunkSize: [{}] (Out of full size: [{}x{}])",
                  toString(slicedGemmInput->chunkDimensions.begin(),
                           std::next(slicedGemmInput->chunkDimensions.begin(), 2),
                           'x'),
                  transpose ? scdSize : fcdSize,
                  transpose ? fcdSize : scdSize);
    }

    void validateProjection(const SlicingStrategyPtr& bundleStrategy, const SlicingStrategyPtr& nodeStrategy) const
    {
        validateTPCOutputProjection(bundleStrategy, nodeStrategy);
        validateTPCInputProjection(bundleStrategy, nodeStrategy);
    }

    virtual void validateTPCOutputProjection(const SlicingStrategyPtr& bundleStrategy,
                                             const SlicingStrategyPtr& nodeStrategy) const
    {
        const TensorPtr&      connectingTensor = m_tpcNode->getOutput(0);
        const pSlicedOperand& slicedGemmInput  = bundleStrategy->getSlicingData().getSlicedOperand(connectingTensor);

        const auto& slicedOutput = nodeStrategy->getSlicingData().getSlicedOperand(connectingTensor);
        ASSERT_NE(nullptr, slicedOutput);
        EXPECT_EQ(slicedOutput->chunkDimensions, slicedGemmInput->chunkDimensions);
    }

    virtual void validateTPCInputProjection(const SlicingStrategyPtr& bundleStrategy,
                                            const SlicingStrategyPtr& nodeStrategy) const
    {
        const auto& slicedInput = nodeStrategy->getSlicingData().getSlicedOperand(m_tpcNode->getInput(0));
        ASSERT_NE(nullptr, slicedInput);
        const auto& slicedOutput = nodeStrategy->getSlicingData().getSlicedOperand(m_tpcNode->getOutput(0));
        ASSERT_NE(nullptr, slicedOutput);
        EXPECT_EQ(slicedInput->chunkDimensions[0], slicedOutput->chunkDimensions[transpose ? 1 : 0]);
        EXPECT_EQ(slicedInput->chunkDimensions[1], slicedOutput->chunkDimensions[transpose ? 0 : 1]);
    }
};

TEST_P(TPCNodeProjectorTest, tps_should_set_input_slice_size)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(operandSliceTest,
                         TPCNodeProjectorTest,
                         ::testing::Combine(::testing::Range(224u, 257u, 16u),  // FCD size
                                            ::testing::Range(24u, 41u, 8u),     // FCD granularity
                                            ::testing::Values(0),               // FCD overlap
                                            ::testing::Range(50u, 101u, 25u),   // SCD size
                                            ::testing::Range(21u, 36u, 7u),     // SCD granularity
                                            ::testing::Values(0),               // SCD overlap
                                            ::testing::Values(true, false)));   // output transposed compared to input

class TPCNodeProjectorOverlapTest : public TPCNodeProjectorTest
{
    using BaseClass = TPCNodeProjectorTest;

protected:
    void validateTPCInputProjection(const SlicingStrategyPtr& bundleStrategy,
                                    const SlicingStrategyPtr& nodeStrategy) const override
    {
        const auto& slicedInput = nodeStrategy->getSlicingData().getSlicedOperand(m_tpcNode->getInput(0));
        ASSERT_NE(nullptr, slicedInput);
        const auto& slicedOutput = nodeStrategy->getSlicingData().getSlicedOperand(m_tpcNode->getOutput(0));
        ASSERT_NE(nullptr, slicedOutput);
        EXPECT_EQ(SlicedOperandUtils::nofSlices(slicedOutput), SlicedOperandUtils::nofSlices(slicedInput));

        validateTPCInputProjectionDimSize(slicedInput, slicedOutput, 0, fcdGranularity, fcdInputOverlap);
        validateTPCInputProjectionDimSize(slicedInput, slicedOutput, 1, scdGranularity, scdInputOverlap);
    }

    void validateTPCInputProjectionDimSize(const pSlicedOperand& slicedInput,
                                           const pSlicedOperand& slicedOutput,
                                           Dim                   dim,
                                           unsigned              granularity,
                                           int                   overlap) const
    {
        if (slicedOutput->chunkDimensions[dim] < slicedOutput->finalShape[dim])
        {  // Dimension is sliced
            unsigned granulesPerSlice = div_round_up(slicedOutput->chunkDimensions[dim], granularity);
            unsigned expChunkSize     = granulesPerSlice * granularity + overlap;
            EXPECT_EQ(slicedInput->chunkDimensions[dim], expChunkSize);
        }
        else
        {  // Dimension is not sliced
            EXPECT_EQ(slicedInput->chunkDimensions[dim], slicedOutput->finalShape[dim]);
        }
    }
};

INSTANTIATE_TEST_SUITE_P(operandSliceTest,
                         TPCNodeProjectorOverlapTest,
                         testing::Combine(testing::Values(256u),     // FCD size
                                          testing::Values(128u),     // FCD granularity
                                          testing::Values(-64, 16),  // FCD overlap
                                          testing::Values(64u),      // SCD size
                                          testing::Values(8u),       // SCD granularity
                                          testing::Values(-2, 3),    // SCD overlap
                                          testing::Values(false)));  // output transposed compared to input

TEST_P(TPCNodeProjectorOverlapTest, overlapping_input_dimensions_should_be_sliced_with_overlap)
{
    test();
}