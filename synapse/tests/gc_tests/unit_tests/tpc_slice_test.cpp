#include "einsum_node.h"
#include "graph_optimizer_test.h"
#include "operation_slice.h"
#include "synapse_common_types.h"
#include "types.h"
#include "utils.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <memory>

#include "tpc_slicing_test_infra.h"

using TestParamTuple = std::tuple<unsigned,  // dim0 size
                                  unsigned,  // dim0 granularity
                                  unsigned,  // dim1 size
                                  unsigned,  // dim1 granularity
                                  bool>;     // transpose

// Test params tuple extractor
class TestParams
{
public:
    struct DimParams
    {
        unsigned size;
        unsigned granularity;
    };
    DimParams dim0;
    DimParams dim1;
    bool      transpose;

    TestParams() = delete;

    TestParams(const TestParamTuple& paramsTuple)
    : dim0 {std::get<0>(paramsTuple), std::get<1>(paramsTuple)},
      dim1 {std::get<2>(paramsTuple), std::get<3>(paramsTuple)},
      transpose {std::get<4>(paramsTuple)}
    {
    }
};

class TPCSliceTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<TestParamTuple>
{
protected:
    TestParams m_params;

    TPCSliceTest() : m_params(GetParam()) {}

    void SetUp() { GraphOptimizerTest::SetUp(); }

    NodePtr generateTPCNode()
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        nodeParams.dims.emplace_back(m_params.dim0.size, m_params.dim0.granularity);
        nodeParams.dims.emplace_back(m_params.dim1.size, m_params.dim1.granularity);
        nodeParams.transpose = m_params.transpose;
        return TPCCustomIndexSpaceNode::create(nodeParams);
    }

    NodePtr generateSlice(const NodePtr&                 tpcNode,
                          const DimSliceRange::DimSlice& dim0Slice,
                          const DimSliceRange::DimSlice& dim1Slice) const
    {
        auto tpcSlice         = tpcNode->getSlice();
        auto asOperationSlice = std::dynamic_pointer_cast<OperationSlice>(tpcSlice);
        assert(nullptr != tpcSlice);

        TSize inputSizes[]  = {dim0Slice.sliceSize, dim1Slice.sliceSize};
        TSize outputSizes[] = {dim1Slice.sliceSize, dim0Slice.sliceSize};

        auto inputSlice  = TensorPtr(new Tensor(2, inputSizes, syn_type_float));
        auto outputSlice = TensorPtr(new Tensor(2, m_params.transpose ? outputSizes : inputSizes, syn_type_float));

        OperationSlice::OffsetArray inputOffsets  = {dim0Slice.offset, dim1Slice.offset};
        OperationSlice::OffsetArray outputOffsets = {dim1Slice.offset, dim0Slice.offset};

        tpcSlice->replaceInput(0, inputSlice);
        asOperationSlice->addTensorSliceOffset(inputSlice, tpcNode->getInput(0), inputOffsets);

        tpcSlice->replaceOutput(0, outputSlice);
        asOperationSlice->addTensorSliceOffset(outputSlice,
                                               tpcNode->getOutput(0),
                                               m_params.transpose ? outputOffsets : inputOffsets);

        return tpcSlice;
    }

    void testSlice(const NodePtr&                 tpcNode,
                   const DimSliceRange::DimSlice& dim0Slice,
                   const DimSliceRange::DimSlice& dim1Slice)
    {
        // Given a TPC node, with input and output shape and granularity according to the test params, and slicing
        // details for each dimension (size and offset), this test checks the generation of the slice node ROI and
        // input/output ROIs.
        // If the transpose param is true, the output dimensions are transposed compared to the input dimensions and the
        // access pattern is permuted relative to the input access pattern (tensorDim[i].indexSpaceDim == 1-i). In this
        // case we expect the output slices to have transposed shape compared to the input slice and also, the offset of
        // the output slice will be permuted compared to the input slice offset meaning:
        // outputSlice.offset[dim] == inputSlice.offset[1-dim]

        LOG_TRACE(GO_TEST,
                  "Testing slice with size {}x{}, and offset {}x{}",
                  dim0Slice.sliceSize,
                  dim1Slice.sliceSize,
                  dim0Slice.offset,
                  dim1Slice.offset);
        auto tpcSlice = generateSlice(tpcNode, dim0Slice, dim1Slice);

        // Test full node ROI
        auto nodeROI = tpcSlice->generateRoi();
        EXPECT_EQ(nodeROI.baseOffset[0], dim0Slice.offset / m_params.dim0.granularity);
        EXPECT_EQ(nodeROI.baseOffset[1], dim1Slice.offset / m_params.dim1.granularity);
        EXPECT_EQ(nodeROI.size[0], div_round_up(dim0Slice.sliceSize, m_params.dim0.granularity));
        EXPECT_EQ(nodeROI.size[1], div_round_up(dim1Slice.sliceSize, m_params.dim1.granularity));

        // Test operands ROI for full node ROI
        auto expDim0SliceSize = dim0Slice.sliceSize;
        auto expDim0Offset    = 0;
        auto expDim1SliceSize = dim1Slice.sliceSize;
        auto expDim1Offset    = 0;

        auto inputROI = tpcSlice->getInputROI(nodeROI, 0);
        ASSERT_TRUE(inputROI.is_set());
        EXPECT_EQ(inputROI->size[0], expDim0SliceSize);
        EXPECT_EQ(inputROI->size[1], expDim1SliceSize);
        EXPECT_EQ(inputROI->baseOffset[0], expDim0Offset);
        EXPECT_EQ(inputROI->baseOffset[1], expDim1Offset);

        auto outputROI = tpcSlice->getOutputROI(nodeROI, 0);
        ASSERT_TRUE(outputROI.is_set());
        EXPECT_EQ(outputROI->baseOffset[0], expDim0Offset);
        EXPECT_EQ(outputROI->baseOffset[1], expDim1Offset);
        EXPECT_EQ(outputROI->size[0], m_params.transpose ? expDim1SliceSize : expDim0SliceSize);
        EXPECT_EQ(outputROI->size[1], m_params.transpose ? expDim0SliceSize : expDim1SliceSize);

        // Test operands ROI for part of the full node ROI - if possible (if the slice size <= granularity, it
        // covers a single index space window, so there is no way to produce a smaller ROI for it)
        if (nodeROI.size[0] > 1)
        {
            nodeROI.size[0]--;
            nodeROI.baseOffset[0]++;
            expDim0SliceSize -= m_params.dim0.granularity;
            expDim0Offset += m_params.dim0.granularity;
        }
        if (nodeROI.size[1] > 1)
        {
            nodeROI.size[1]--;
            nodeROI.baseOffset[1]++;
            expDim1SliceSize -= m_params.dim1.granularity;
            expDim1Offset += m_params.dim1.granularity;
        }
        auto newInputROI = tpcSlice->getInputROI(nodeROI, 0);
        ASSERT_TRUE(newInputROI.is_set());
        EXPECT_EQ(newInputROI->size[0], expDim0SliceSize);
        EXPECT_EQ(newInputROI->size[1], expDim1SliceSize);
        EXPECT_EQ(newInputROI->baseOffset[0], expDim0Offset);
        EXPECT_EQ(newInputROI->baseOffset[1], expDim1Offset);

        auto newOutputROI = tpcSlice->getOutputROI(nodeROI, 0);
        ASSERT_TRUE(newOutputROI.is_set());
        EXPECT_EQ(newOutputROI->baseOffset[0], m_params.transpose ? expDim1Offset : expDim0Offset);
        EXPECT_EQ(newOutputROI->baseOffset[1], m_params.transpose ? expDim0Offset : expDim1Offset);
        EXPECT_EQ(newOutputROI->size[0], m_params.transpose ? expDim1SliceSize : expDim0SliceSize);
        EXPECT_EQ(newOutputROI->size[1], m_params.transpose ? expDim0SliceSize : expDim1SliceSize);
    }
};

TEST_P(TPCSliceTest, test_tpc_slicing_infra)
{
    auto tpcNode = generateTPCNode();

    LOG_TRACE(GO_TEST,
              "Testing TPC slicing of node with sizes {}x{} and granularity {}x{} ({}transposed)",
              m_params.dim0.size,
              m_params.dim1.size,
              m_params.dim0.granularity,
              m_params.dim1.granularity,
              m_params.transpose ? "" : "not ");

    // These 2 loops generate all possible slices sizes and offset in the 2 dimensions tested, and check their ROI
    // generation
    for (const auto& dim1Slice : DimSliceRange(m_params.dim1.size, m_params.dim1.granularity))
    {
        for (const auto& dim0Slice : DimSliceRange(m_params.dim0.size, m_params.dim0.granularity))
        {
            testSlice(tpcNode, dim0Slice, dim1Slice);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(operand_slice,
                         TPCSliceTest,
                         testing::Combine(testing::Values(20, 40, 64),  // dim0 size
                                          testing::Values(8, 10, 64),   // dim0 granularity
                                          testing::Values(15, 40, 63),  // dim1 size
                                          testing::Values(5, 16, 32),   // dim1 granularity
                                          testing::Values(false, true)  // transpose
                                          ));
