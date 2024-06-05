#include "bundle_view.h"
#include "slicer/sliced_tensor_generator.h"
#include "graph_optimizer_test.h"
#include "strategy.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"

using namespace gc::layered_brain;

class SlicedTensorGeneratorTest : public GraphOptimizerTest
{
protected:
    void createNodeAndTensor(int offset = 0)
    {
        // Create a node with 1:1 mapping between node dims and tensor dims.
        ASSERT_EQ(m_origTensorSizes.size(), m_tensorGranularity.size());
        TPCCustomIndexSpaceNode::Params nodeParams;
        for (auto i = 0; i < m_origTensorSizes.size(); i++)
        {
            nodeParams.dims.emplace_back(m_origTensorSizes.at(i), m_tensorGranularity.at(i), 0, offset);
        }
        nodeParams.transpose = false;
        m_origNode           = TPCCustomIndexSpaceNode::create(nodeParams);
        ASSERT_TRUE(m_origNode);
        m_origTensor = m_origNode->getInput(0);
        ASSERT_TRUE(m_origTensor);
    }

    void validateSlicedTensor(const TensorPtr&   slicedTensor,
                              const OffsetArray& slicedTensorOffset,
                              const SizeArray&   expectedSize,
                              const OffsetArray& expectedOffset)
    {
        ASSERT_TRUE(slicedTensor);
        ASSERT_NE(slicedTensor, m_origTensor);
        ASSERT_EQ(slicedTensor->getTensorAnnotation().origBigTensor, m_origTensor);
        ASSERT_TRUE(slicedTensor->getName().find(m_origTensor->getName()) != std::string::npos);
        ASSERT_EQ(slicedTensor->getAllSizesInElements(), expectedSize);
        ASSERT_EQ(slicedTensorOffset, expectedOffset);
    }

    const BundleIdx          m_bundleIdx         = 6;
    const std::vector<TSize> m_origTensorSizes   = {512, 128, 256, 24};
    TensorTile::Geometry     m_tensorGranularity = {1, 5, 4, 6};
    TensorPtr                m_origTensor;
    NodePtr                  m_origNode;
};

TEST_F(SlicedTensorGeneratorTest, generate_sliced_tensor_from_unsliced_node)
{
    createNodeAndTensor();

    SlicedTensorGenerator slicedTensorGenerator(m_bundleIdx);

    NodeTile nodeTile(m_origNode->getNodeAccessPattern()->getNodeResolution());
    const auto& [slicedTensor, offset] =
        slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, {0, 0, 0, 0});

    OffsetArray expectedOffset {};
    expectedOffset.fill(0);
    validateSlicedTensor(slicedTensor, offset, m_origTensor->getAllSizesInElements(), expectedOffset);

    // Existing tensors should be reused.
    const auto& [newSlicedTensor, newOffset] =
        slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, {0, 0, 0, 0});
    EXPECT_EQ(newSlicedTensor, slicedTensor);
    EXPECT_EQ(newOffset, offset);
}

TEST_F(SlicedTensorGeneratorTest, generate_sliced_tensor_single_sliced_dim_even_slicing)
{
    createNodeAndTensor();

    SlicedTensorGenerator slicedTensorGenerator(m_bundleIdx);

    Dim   slicedDim          = 2;  // Same dim for node and tensor (1:1 mapping)
    auto  numGranulesInSlice = 8;
    TSize slicedDimSize      = m_tensorGranularity[slicedDim] * numGranulesInSlice;
    ASSERT_TRUE(m_origTensorSizes[slicedDim] % slicedDimSize == 0);
    unsigned numSlices = div_round_up(m_origTensorSizes[slicedDim], slicedDimSize);

    NodeTile nodeTile(m_origNode->getNodeAccessPattern()->getNodeResolution());
    nodeTile.geometry.at(slicedDim) = numGranulesInSlice;

    SizeArray expectedSliceSizes  = m_origTensor->getAllSizesInElements();
    expectedSliceSizes[slicedDim] = slicedDimSize;

    for (auto i = 0; i < numSlices; i++)
    {
        BVDCoord coord {0, 0, 0, 0};
        coord[slicedDim]                   = i;
        const auto& [slicedTensor, offset] =
            slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);

        OffsetArray expectedOffset {};
        expectedOffset.fill(0);
        expectedOffset[slicedDim] = i * slicedDimSize;
        validateSlicedTensor(slicedTensor, offset, expectedSliceSizes, expectedOffset);

        // Existing tensors should be reused.
        const auto& [newSlicedTensor, newOffset] =
            slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);
        EXPECT_EQ(newSlicedTensor, slicedTensor);
        EXPECT_EQ(newOffset, offset);

        nodeTile.offset.at(slicedDim) += nodeTile.geometry.at(slicedDim);
    }
}

TEST_F(SlicedTensorGeneratorTest, generate_sliced_tensor_single_sliced_dim_last_slice_smaller)
{
    createNodeAndTensor();

    SlicedTensorGenerator slicedTensorGenerator(m_bundleIdx);

    Dim   slicedDim          = 2;  // Same dim for node and tensor (1:1 mapping)
    auto  numGranulesInSlice = 10;
    TSize slicedDimSize      = m_tensorGranularity[slicedDim] * numGranulesInSlice;
    ASSERT_TRUE(m_origTensorSizes[slicedDim] % slicedDimSize != 0);
    unsigned numSlices                 = div_round_up(m_origTensorSizes[slicedDim], slicedDimSize);
    TSize    slicedDimSizeForLastSlice = m_origTensorSizes[slicedDim] - ((numSlices - 1) * slicedDimSize);
    ASSERT_NE(slicedDimSize, slicedDimSizeForLastSlice);

    NodeTile nodeTile(m_origNode->getNodeAccessPattern()->getNodeResolution());
    nodeTile.geometry.at(slicedDim) = numGranulesInSlice;

    for (auto i = 0; i < numSlices; i++)
    {
        BVDCoord coord {0, 0, 0, 0};
        coord[slicedDim]                   = i;
        const auto& [slicedTensor, offset] =
            slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);

        SizeArray expectedSliceSizes  = m_origTensor->getAllSizesInElements();
        expectedSliceSizes[slicedDim] = (i == numSlices - 1) ? slicedDimSizeForLastSlice : slicedDimSize;
        OffsetArray expectedOffset {};
        expectedOffset.fill(0);
        expectedOffset[slicedDim] = i * slicedDimSize;
        validateSlicedTensor(slicedTensor, offset, expectedSliceSizes, expectedOffset);

        // Existing tensors should be reused.
        const auto& [newSlicedTensor, newOffset] =
            slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);
        EXPECT_EQ(newSlicedTensor, slicedTensor);
        EXPECT_EQ(newOffset, offset);

        nodeTile.offset.at(slicedDim) += nodeTile.geometry.at(slicedDim);
    }
}

TEST_F(SlicedTensorGeneratorTest, generate_sliced_tensor_multiple_sliced_dims)
{
    createNodeAndTensor();

    SlicedTensorGenerator slicedTensorGenerator(m_bundleIdx);

    Dim  slicedDim0             = 0;  // Same dim for node and tensor (1:1 mapping)
    Dim  slicedDim3             = 3;  // Same dim for node and tensor (1:1 mapping)
    auto numGranulesInSliceDim0 = 10;
    auto numGranulesInSliceDim3 = 1;

    TSize slicedDim0Size = m_tensorGranularity[slicedDim0] * numGranulesInSliceDim0;
    TSize slicedDim3Size = m_tensorGranularity[slicedDim3] * numGranulesInSliceDim3;
    ASSERT_TRUE(m_origTensorSizes[slicedDim0] % slicedDim0Size != 0);
    ASSERT_TRUE(m_origTensorSizes[slicedDim3] % slicedDim3Size == 0);
    unsigned numSlicesDim0              = div_round_up(m_origTensorSizes[slicedDim0], slicedDim0Size);
    unsigned numSlicesDim3              = div_round_up(m_origTensorSizes[slicedDim3], slicedDim3Size);
    TSize    slicedDim0SizeForLastSlice = m_origTensorSizes[slicedDim0] - ((numSlicesDim0 - 1) * slicedDim0Size);
    ASSERT_NE(slicedDim0Size, slicedDim0SizeForLastSlice);

    NodeTile nodeTile(m_origNode->getNodeAccessPattern()->getNodeResolution());
    nodeTile.geometry.at(slicedDim0) = numGranulesInSliceDim0;
    nodeTile.geometry.at(slicedDim3) = numGranulesInSliceDim3;

    for (auto i = 0; i < numSlicesDim0; i++)
    {
        for (auto j = 0; j < numSlicesDim3; j++)
        {
            BVDCoord coord {0, 0, 0, 0};
            coord[slicedDim0]                  = i;
            coord[slicedDim3]                  = j;
            const auto& [slicedTensor, offset] =
                slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);

            SizeArray expectedSliceSizes   = m_origTensor->getAllSizesInElements();
            expectedSliceSizes[slicedDim0] = (i == numSlicesDim0 - 1) ? slicedDim0SizeForLastSlice : slicedDim0Size;
            expectedSliceSizes[slicedDim3] = slicedDim3Size;
            OffsetArray expectedOffset {};
            expectedOffset.fill(0);
            expectedOffset[slicedDim0] = i * slicedDim0Size;
            expectedOffset[slicedDim3] = j * slicedDim3Size;
            validateSlicedTensor(slicedTensor, offset, expectedSliceSizes, expectedOffset);

            // Existing tensors should be reused.
            const auto& [newSlicedTensor, newOffset] =
                slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);
            EXPECT_EQ(newSlicedTensor, slicedTensor);
            EXPECT_EQ(newOffset, offset);

            nodeTile.offset.at(slicedDim3) += nodeTile.geometry.at(slicedDim3);
        }
        nodeTile.offset.at(slicedDim3) = 0;
        nodeTile.offset.at(slicedDim0) += nodeTile.geometry.at(slicedDim0);
    }
}

TEST_F(SlicedTensorGeneratorTest, generate_sliced_tensor_with_negative_offset)
{
    int inputOffset = -2;
    createNodeAndTensor(inputOffset);

    SlicedTensorGenerator slicedTensorGenerator(m_bundleIdx);

    Dim   slicedDim          = 2;  // Same dim for node and tensor (1:1 mapping)
    auto  numGranulesInSlice = 10;
    TSize slicedDimSize      = m_tensorGranularity[slicedDim] * numGranulesInSlice;
    ASSERT_TRUE(m_origTensorSizes[slicedDim] % slicedDimSize != 0);
    unsigned numSlices                  = div_round_up(m_origTensorSizes[slicedDim], slicedDimSize);
    TSize    slicedDimSizeForFirstSlice = slicedDimSize + inputOffset;
    TSize    slicedDimSizeForLastSlice =
        m_origTensorSizes[slicedDim] - ((numSlices - 2) * slicedDimSize) - slicedDimSizeForFirstSlice;

    NodeTile nodeTile(m_origNode->getNodeAccessPattern()->getNodeResolution());
    nodeTile.geometry.at(slicedDim) = numGranulesInSlice;

    for (auto i = 0; i < numSlices; i++)
    {
        BVDCoord coord {0, 0, 0, 0};
        coord[slicedDim] = i;
        const auto& [slicedTensor, offset] =
            slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);

        SizeArray expectedSliceSizes = m_origTensor->getAllSizesInElements();
        expectedSliceSizes[slicedDim] =
            (i == 0) ? slicedDimSizeForFirstSlice : ((i == numSlices - 1) ? slicedDimSizeForLastSlice : slicedDimSize);
        OffsetArray expectedOffset {};
        expectedOffset.fill(0);
        expectedOffset[slicedDim] = (i == 0) ? 0 : ((int)(i * slicedDimSize) + inputOffset);
        validateSlicedTensor(slicedTensor, offset, expectedSliceSizes, expectedOffset);

        // Existing tensors should be reused.
        const auto& [newSlicedTensor, newOffset] =
            slicedTensorGenerator.getSlicedTensor(m_origNode, nodeTile, m_origTensor, coord);
        EXPECT_EQ(newSlicedTensor, slicedTensor);
        EXPECT_EQ(newOffset, offset);

        nodeTile.offset.at(slicedDim) += nodeTile.geometry.at(slicedDim);
    }
}