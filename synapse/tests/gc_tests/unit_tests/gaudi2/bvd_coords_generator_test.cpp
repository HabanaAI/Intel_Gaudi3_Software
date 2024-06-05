#include "slicer/bvd_coords_generator.h"
#include "graph_optimizer_test.h"
#include "strategy.h"
#include "synapse_common_types.h"
#include "node_factory.h"
#include "types.h"

using namespace gc::layered_brain;

class TPCNodeForTest : public TPCNode
{
public:
    static NodePtr create(TSize dimSize, unsigned numIndexSpaceDims)
    {
        return NodePtr(
            new TPCNodeForTest({createTensor(4, dimSize), createTensor(2, dimSize), createTensor(1, dimSize)},
                               {createTensor(3, dimSize)},
                               dimSize,
                               numIndexSpaceDims));
    }

protected:
    TPCNodeForTest(const TensorVector& inputs, const TensorVector& outputs, TSize dimSize, unsigned numIndexSpaceDims)
    : TPCNode(inputs, outputs, "TPC")
    {
        setGUID(NOP_KERNEL_NAME);
        m_instanceWrapper.updateGlueCodeParamsAndTensorAccessPatternPointers(*this);
        auto& instance = m_instanceWrapper.getInstance();
        // Init index-space geometry for BVD resolution calculation
        // (tensors access-patterns are not needed).
        instance.indexSpaceRank = numIndexSpaceDims;
        for (auto i = 0; i < instance.indexSpaceRank; i++)
        {
            instance.indexSpaceGeometry[i] = dimSize;
        }
        m_instanceWrapper.setInstantiated(true);
    }
    static TensorPtr createTensor(unsigned numDims, TSize dimSize)
    {
        SizeVector sizes(numDims, dimSize);
        return std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
    }
};

class BVDCoordsGeneratorTest : public GraphOptimizerTest
{
protected:
    void createNode()
    {
        m_node = TPCNodeForTest::create(m_dimSize, m_numBVDs);
        ASSERT_TRUE(m_node);
    }

    void createBVDs()
    {
        m_bundleViews = std::make_shared<BundleViewContainer>(m_numBVDs);

        // Map node dims to BVDs - 1:1 mapping between node dims and BVDs.
        for (Dim i = 0; i < m_numBVDs; i++)
        {
            m_bundleViews->mapNodeDimToBVD(m_node, i, i, m_dimGranularity);
        }

        // Map input 0 to BVDs - 1:1 mapping between tensor dims and BVDs (tensor is 4D).
        TensorPtr input0 = m_node->getInput(0);
        m_bundleViews->mapTensorDimToBVD(input0, 0, 0, m_dimGranularity);
        m_bundleViews->mapTensorDimToBVD(input0, 1, 1, m_dimGranularity);
        m_bundleViews->mapTensorDimToBVD(input0, 2, 2, m_dimGranularity);
        m_bundleViews->mapTensorDimToBVD(input0, 3, 3, m_dimGranularity);

        // Map input 1 to BVDs - tensor dim 0 -> BVD 1, tensor dim 1 -> BVD 0, BVDs 2, 3 are not mapped (tensor is 2D).
        TensorPtr input1 = m_node->getInput(1);
        m_bundleViews->mapTensorDimToBVD(input1, 0, 1, m_dimGranularity);
        m_bundleViews->mapTensorDimToBVD(input1, 1, 0, m_dimGranularity);

        // Map input 2 to BVDs - tensor dim 0 is mapped to BVD 3, BVDs 0, 1, 2 are not mapped (tensor is 1D).
        TensorPtr input2 = m_node->getInput(2);
        m_bundleViews->mapTensorDimToBVD(input2, 0, 3, m_dimGranularity);

        // Map output to BVDs - tensor dim 0 -> BVD 0, tensor dim 1 -> BVD 2, tensor dim 2 -> BVD 3, BVD 1 is not mapped
        // (tensor is 3D).
        TensorPtr output = m_node->getOutput(0);
        m_bundleViews->mapTensorDimToBVD(output, 0, 0, m_dimGranularity);
        m_bundleViews->mapTensorDimToBVD(output, 1, 2, m_dimGranularity);
        m_bundleViews->mapTensorDimToBVD(output, 2, 3, m_dimGranularity);

        for (auto i = 0; i < m_numBVDs; i++)
        {
            ASSERT_EQ(m_node->getNodeAccessPattern()->getNodeResolution()[i], m_dimSize);
        }

        for (BundleViewId bvd = 0; bvd < m_numBVDs; bvd++)
        {
            ASSERT_EQ(m_bundleViews->getBundleView(bvd).resolution, div_round_up(m_dimSize, m_dimGranularity));
        }
    }

    StrategyPtr createUnslicedStrategy()
    {
        StrategyPtr slicingStrategy = std::make_shared<Strategy>();
        for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
        {
            slicingStrategy->setBVDMultiplier(bvd, BVDMultiplier());
        }
        return slicingStrategy;
    }

    const TSize            m_dimSize        = 512;
    uint64_t               m_dimGranularity = 2;
    unsigned               m_numBVDs        = 4;
    NodePtr                m_node;
    BundleViewContainerPtr m_bundleViews;
};

TEST_F(BVDCoordsGeneratorTest, generate_bvd_coords_for_unsliced_strategy)
{
    createNode();
    createBVDs();
    StrategyPtr strategy = createUnslicedStrategy();

    BVDCoordsGenerator bvdCoordsGenerator(m_bundleViews, strategy);
    const auto&        numOfSlicesPerBVD = bvdCoordsGenerator.getNumOfSlicesPerBVD();
    ASSERT_EQ(numOfSlicesPerBVD.size(), m_bundleViews->getNumOfBundleViews());
    ASSERT_TRUE(areAllElementsEqual(numOfSlicesPerBVD.begin(), numOfSlicesPerBVD.end(), 1));
    const auto& coords = bvdCoordsGenerator.getBVDCoordsForNode(m_node);

    // Expecting a single slice
    ASSERT_EQ(coords.size(), 1);
    BVDCoord exepctedCoord = {0, 0, 0, 0};
    ASSERT_EQ(*coords.begin(), exepctedCoord);
    for (const auto& tensor : m_node->getOperands())
    {
        BVDCoord tensorBVDCoord = bvdCoordsGenerator.projectBVDCoordOnTensor(tensor, *coords.begin());
        ASSERT_EQ(tensorBVDCoord, exepctedCoord);
    }
}

TEST_F(BVDCoordsGeneratorTest, generate_bvd_coords_for_sliced_strategy)
{
    createNode();
    createBVDs();

    StrategyPtr  strategy  = createUnslicedStrategy();
    BundleViewId slicedBVD = 1;
    strategy->setBVDMultiplier(slicedBVD, BVDMultiplier(m_bundleViews->getBundleView(slicedBVD).resolution / 4UL));
    unsigned numSlices = div_round_up(m_bundleViews->getBundleView(slicedBVD).resolution,
                                      strategy->getBVDMultiplier(slicedBVD).getMultiplier());

    BVDCoordsGenerator bvdCoordsGenerator(m_bundleViews, strategy);
    const auto&        numOfSlicesPerBVD = bvdCoordsGenerator.getNumOfSlicesPerBVD();
    ASSERT_EQ(numOfSlicesPerBVD.size(), m_bundleViews->getNumOfBundleViews());
    for (BundleViewId bvdId = 0; bvdId < m_bundleViews->getNumOfBundleViews(); bvdId++)
    {
        ASSERT_EQ(numOfSlicesPerBVD[bvdId], (bvdId == slicedBVD) ? numSlices : 1);
    }

    const auto& coords = bvdCoordsGenerator.getBVDCoordsForNode(m_node);
    ASSERT_EQ(coords.size(), numSlices);

    for (auto i = 0; i < numSlices; i++)
    {
        BVDCoord expectedNodeBVDCoord = {0, i, 0, 0};  // BVD 1 is sliced, 1:1 mapping between node dims to BVDs.
        ASSERT_TRUE(coords.find(expectedNodeBVDCoord) != coords.end());

        BVDCoord in0BVDCoord = bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getInput(0), expectedNodeBVDCoord);
        BVDCoord expectedIn0BVDCoord = {0, i, 0, 0};  // BVD 1 is sliced, dim 1 is mapped to BVD 1.
        ASSERT_EQ(in0BVDCoord, expectedIn0BVDCoord);

        BVDCoord in1BVDCoord = bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getInput(1), expectedNodeBVDCoord);
        BVDCoord expectedIn1BVDCoord = {0, i, 0, 0};  // BVD 1 is sliced, dim 0 is mapped to BVD 1.
        ASSERT_EQ(in1BVDCoord, expectedIn1BVDCoord);

        BVDCoord in2BVDCoord = bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getInput(2), expectedNodeBVDCoord);
        BVDCoord expectedIn2BVDCoord = {0, 0, 0, 0};  // BVD 1 is sliced, none of the dimensions is mapped to BVD 1.
        ASSERT_EQ(in2BVDCoord, expectedIn2BVDCoord);

        BVDCoord outBVDCoord = bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getOutput(0), expectedNodeBVDCoord);
        BVDCoord expectedOutBVDCoord = {0, 0, 0, 0};  // BVD 1 is sliced, none of the dimensions is mapped to BVD 1.
        ASSERT_EQ(outBVDCoord, expectedOutBVDCoord);
    }
}

TEST_F(BVDCoordsGeneratorTest, generate_bvd_coords_for_strategy_sliced_on_multiple_bvds)
{
    createNode();
    createBVDs();

    StrategyPtr strategy = createUnslicedStrategy();
    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 4);
    strategy->setBVDMultiplier(
        0,
        BVDMultiplier(m_bundleViews->getBundleView(0).resolution / 4UL));  // BVD 0 sliced to 4 slices
    strategy->setBVDMultiplier(
        1,
        BVDMultiplier(m_bundleViews->getBundleView(1).resolution / 8UL));  // BVD 1 sliced to 8 slices
    // BVD 2 remains unsliced.
    strategy->setBVDMultiplier(
        3,
        BVDMultiplier(m_bundleViews->getBundleView(3).resolution / 2UL));  // BVD 3 sliced to 2 slices

    BVDCoordsGenerator bvdCoordsGenerator(m_bundleViews, strategy);
    const auto&        numSlicesPerBVD = bvdCoordsGenerator.getNumOfSlicesPerBVD();
    ASSERT_EQ(numSlicesPerBVD.size(), m_bundleViews->getNumOfBundleViews());
    uint64_t totalNumSlices = 1;
    for (BundleViewId bvdId = 0; bvdId < m_bundleViews->getNumOfBundleViews(); bvdId++)
    {
        const auto& multiplier = strategy->getBVDMultiplier(bvdId);
        if (multiplier.isSliced())  // BVD is sliced
        {
            ASSERT_EQ(numSlicesPerBVD[bvdId],
                      div_round_up(m_bundleViews->getBundleView(bvdId).resolution, multiplier.getMultiplier()));
        }
        else  // BVD is not sliced
        {
            ASSERT_EQ(numSlicesPerBVD[bvdId], 1);
        }
        totalNumSlices *= numSlicesPerBVD[bvdId];
    }

    const auto& coords = bvdCoordsGenerator.getBVDCoordsForNode(m_node);
    ASSERT_EQ(coords.size(), totalNumSlices);

    for (auto bvd0Coord = 0; bvd0Coord < numSlicesPerBVD[0]; bvd0Coord++)
    {
        for (auto bvd1Coord = 0; bvd1Coord < numSlicesPerBVD[1]; bvd1Coord++)
        {
            for (auto bvd2Coord = 0; bvd2Coord < numSlicesPerBVD[2]; bvd2Coord++)
            {
                for (auto bvd3Coord = 0; bvd3Coord < numSlicesPerBVD[3]; bvd3Coord++)
                {
                    BVDCoord expectedNodeBVDCoord = {bvd0Coord,
                                                     bvd1Coord,
                                                     bvd2Coord,
                                                     bvd3Coord};  // 1:1 mapping between node dims to BVDs.
                    ASSERT_TRUE(coords.find(expectedNodeBVDCoord) != coords.end());

                    // Input 0 - 1:1 mapping between tensor dims and BVDs (tensor is 4D).
                    BVDCoord in0BVDCoord =
                        bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getInput(0), expectedNodeBVDCoord);
                    BVDCoord expectedIn0BVDCoord = {bvd0Coord, bvd1Coord, bvd2Coord, bvd3Coord};
                    ASSERT_EQ(in0BVDCoord, expectedIn0BVDCoord);

                    // Input 1 - tensor dim 0 -> BVD 1, tensor dim 1 -> BVD 0, BVDs 2, 3 are not mapped (tensor is 2D).
                    BVDCoord in1BVDCoord =
                        bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getInput(1), expectedNodeBVDCoord);
                    BVDCoord expectedIn1BVDCoord = {bvd0Coord, bvd1Coord, 0, 0};
                    ASSERT_EQ(in1BVDCoord, expectedIn1BVDCoord);

                    // Input 2 - tensor dim 0 is mapped to BVD 3, BVDs 0, 1, 2 are not mapped (tensor is 1D).
                    BVDCoord in2BVDCoord =
                        bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getInput(2), expectedNodeBVDCoord);
                    BVDCoord expectedIn2BVDCoord = {0, 0, 0, bvd3Coord};
                    ASSERT_EQ(in2BVDCoord, expectedIn2BVDCoord);

                    // Output 0 - tensor dim 0 -> BVD 0, tensor dim 1 -> BVD 2, tensor dim 2 -> BVD 3, BVD 1 is not
                    // mapped (tensor is 3D).
                    BVDCoord outBVDCoord =
                        bvdCoordsGenerator.projectBVDCoordOnTensor(m_node->getOutput(0), expectedNodeBVDCoord);
                    BVDCoord expectedOutBVDCoord = {bvd0Coord, 0, bvd2Coord, bvd3Coord};
                    ASSERT_EQ(outBVDCoord, expectedOutBVDCoord);
                }
            }
        }
    }
}

TEST_F(BVDCoordsGeneratorTest, generate_bvd_coords_sample_mode)
{
    setGlobalConfForTest(GCFG_ENABLE_LB_SAMPLE_MODE, "True");

    createNode();
    createBVDs();

    StrategyPtr strategy = createUnslicedStrategy();
    ASSERT_EQ(m_bundleViews->getNumOfBundleViews(), 4);
    strategy->setBVDMultiplier(
        0,
        BVDMultiplier(m_bundleViews->getBundleView(0).resolution / 4UL));  // BVD 0 sliced to 4 slices
    strategy->setBVDMultiplier(
        1,
        BVDMultiplier(m_bundleViews->getBundleView(1).resolution / 8UL));  // BVD 1 sliced to 8 slices
    // BVD 2 remains unsliced.
    strategy->setBVDMultiplier(
        3,
        BVDMultiplier(m_bundleViews->getBundleView(3).resolution / 2UL));  // BVD 3 sliced to 2 slices

    const unsigned pipelineDepth = 2;
    strategy->setPipelineDepth(pipelineDepth);

    BVDCoordsGenerator bvdCoordsGenerator(m_bundleViews, strategy, /*dryRun*/ true);
    const auto&        numSlicesPerBVD = bvdCoordsGenerator.getNumOfSlicesPerBVD();
    const auto&        coords          = bvdCoordsGenerator.getBVDCoordsForNode(m_node);

    // Num slices in dry run = min(total_num_slices, pipeline-depth)
    NumSlicesPerBVD expectedNumSlicesPerfBVD = {pipelineDepth, pipelineDepth, 1, pipelineDepth};
    ASSERT_EQ(numSlicesPerBVD, expectedNumSlicesPerfBVD);

    ASSERT_EQ(coords.size(), pipelineDepth * pipelineDepth * 1 * pipelineDepth);

    for (auto bvd0Coord = 0; bvd0Coord < expectedNumSlicesPerfBVD.at(0); bvd0Coord++)
    {
        for (auto bvd1Coord = 0; bvd1Coord < expectedNumSlicesPerfBVD.at(1); bvd1Coord++)
        {
            for (auto bvd2Coord = 0; bvd2Coord < expectedNumSlicesPerfBVD.at(2); bvd2Coord++)
            {
                for (auto bvd3Coord = 0; bvd3Coord < expectedNumSlicesPerfBVD.at(3); bvd3Coord++)
                {
                    BVDCoord expectedNodeBVDCoord = {bvd0Coord, bvd1Coord, bvd2Coord, bvd3Coord};
                    ASSERT_TRUE(coords.find(expectedNodeBVDCoord) != coords.end());
                }
            }
        }
    }
}