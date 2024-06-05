#include "bundle_view.h"
#include "compilation_hal_reader.h"
#include "gaudi3_graph.h"
#include "slicer/strategy_inflator.h"
#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"
#include "types.h"

using namespace gc::layered_brain;

class StrategyInflatorTest : public GraphOptimizerTest
{
protected:
    NodePtr createNode(TSize dimSize)
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        for (auto i = 0; i < m_numBVDs; i++)
        {
            nodeParams.dims.emplace_back(dimSize, 1);
        }
        nodeParams.transpose = false;
        NodePtr node         = TPCCustomIndexSpaceNode::create(nodeParams);
        EXPECT_TRUE(node);
        return node;
    }

    void createBVDs(uint64_t bvdResolution)
    {
        m_bundleViews = std::make_shared<BundleViewContainer>(m_numBVDs);
        m_node        = createNode(bvdResolution);
        // Map node dims to BVDs - 1:1 mapping between node dims and BVDs.
        for (Dim i = 0; i < m_numBVDs; i++)
        {
            m_bundleViews->mapNodeDimToBVD(m_node, i, i, 1);
        }

        // Map tensors to BVDs
        m_bundleViews->mapTensorDimToBVD(m_node->getInput(0), 0, 3, 1);
        m_bundleViews->mapTensorDimToBVD(m_node->getInput(0), 1, 2, 1);
        m_bundleViews->mapTensorDimToBVD(m_node->getInput(0), 2, 1, 1);
        m_bundleViews->mapTensorDimToBVD(m_node->getInput(0), 3, 0, 1);

        m_bundleViews->mapTensorDimToBVD(m_node->getOutput(0), 0, 1, 1);
        m_bundleViews->mapTensorDimToBVD(m_node->getOutput(0), 1, 2, 1);
        m_bundleViews->mapTensorDimToBVD(m_node->getOutput(0), 2, 3, 1);
        m_bundleViews->mapTensorDimToBVD(m_node->getOutput(0), 3, 0, 1);

        // Create another tensor and map it to BVD 3 - will be used to test BVDs inflation order.
        const std::vector<TSize> sizes = {bvdResolution};
        TensorPtr                t     = std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float);
        m_bundleViews->mapTensorDimToBVD(t, 0, 3, 1);

        for (BundleViewId bvd = 0; bvd < m_numBVDs; bvd++)
        {
            ASSERT_EQ(m_bundleViews->getBundleView(bvd).resolution, bvdResolution);
        }
    }

    StrategyPtr createUnslicedStrategy()
    {
        auto mmeSolution            = std::make_shared<MmeSolution>();
        mmeSolution->QORs[m_node]   = std::make_shared<SolutionParams>();
        StrategyPtr slicingStrategy = std::make_shared<Strategy>(mmeSolution);
        for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
        {
            slicingStrategy->setBVDMultiplier(bvd, BVDMultiplier());
        }
        return slicingStrategy;
    }

    void validateStrategy(const StrategyPtr& strategy, const std::vector<std::optional<uint64_t>> expectedMultipliers)
    {
        ASSERT_EQ(expectedMultipliers.size(), m_numBVDs);
        for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
        {
            if (expectedMultipliers.at(bvd).has_value())
            {
                ASSERT_TRUE(strategy->getBVDMultiplier(bvd).isSliced());
                ASSERT_EQ(strategy->getBVDMultiplier(bvd).getMultiplier(), expectedMultipliers.at(bvd).value());
            }
            else
            {
                ASSERT_FALSE(strategy->getBVDMultiplier(bvd).isSliced());
            }
        }
    }

    const std::vector<InflationType> inflationTypes = {InflationType::INFLATE_FOR_UTILIZATION,
                                                       InflationType::INFLATE_FOR_BW,
                                                       InflationType::INFLATE_FOR_PERFORATION,
                                                       InflationType::INFLATE_FOR_NUM_SLICES};
    const unsigned                   m_numBVDs      = 4;
    BundleViewContainerPtr           m_bundleViews;
    NodePtr                          m_node;
    Gaudi3Graph                      m_graph;
    CompilationHalReaderSetter       m_halReaderSetter {&m_graph};
};

TEST_F(StrategyInflatorTest, no_inflation_when_all_bvds_unsliced)
{
    const uint64_t bvdResolution = 5;
    createBVDs(bvdResolution);
    StrategyPtr strategy = createUnslicedStrategy();
    PerforationPerNode perforationPerNode;
    perforationPerNode[m_node] = 0;
    strategy->setPerforationData(perforationPerNode);

    StrategyInflator inflator(m_bundleViews);
    for (auto inflationType : inflationTypes)
    {
        // Expect no inflation, same multipliers.
        ASSERT_FALSE(
            inflator.inflateOneStep(inflationType,
                                    strategy,
                                    (inflationType == InflationType::INFLATE_FOR_NUM_SLICES) ? nullptr : m_node));
        validateStrategy(strategy, {std::nullopt, std::nullopt, std::nullopt, std::nullopt});
    }
}

TEST_F(StrategyInflatorTest, no_inflation_when_all_bvds_trivially_sliced)
{
    const uint64_t bvdResolution = 5;
    createBVDs(bvdResolution);
    StrategyPtr                          strategy = createUnslicedStrategy();
    std::vector<std::optional<uint64_t>> expectedMultipliers;
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        strategy->setBVDMultiplier(bvd, BVDMultiplier(m_bundleViews->getBundleView(bvd).resolution));
        expectedMultipliers.push_back(m_bundleViews->getBundleView(bvd).resolution);
    }
    PerforationPerNode perforationPerNode;
    perforationPerNode[m_node] = 0;
    strategy->setPerforationData(perforationPerNode);

    StrategyInflator inflator(m_bundleViews);
    for (auto inflationType : inflationTypes)
    {
        // Expect no inflation, same multipliers.
        ASSERT_FALSE(
            inflator.inflateOneStep(inflationType,
                                    strategy,
                                    (inflationType == InflationType::INFLATE_FOR_NUM_SLICES) ? nullptr : m_node));
        validateStrategy(strategy, expectedMultipliers);
    }
}

TEST_F(StrategyInflatorTest, inflate_according_to_inflation_type)
{
    const uint64_t bvdResolution = 3;
    createBVDs(bvdResolution);
    StrategyPtr    strategy   = createUnslicedStrategy();
    const uint64_t multiplier = 1;
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        strategy->setBVDMultiplier(bvd, BVDMultiplier(multiplier));
    }
    strategy->setBVDMultiplier(3, BVDMultiplier()); // BVD 3 is unsliced
    ASSERT_EQ(m_numBVDs, 4);
    strategy->getMmeSolution()->QORs[m_node]->solutionRequirements.bwInflationDim = 1;

    StrategyInflator inflator(m_bundleViews);

    // 1) Inflate for utilization (no candidates):
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_UTILIZATION, strategy, m_node));

    // 2) Inflate for BW (BVD 1):
    // BVD 1 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_BW, strategy, m_node));
    validateStrategy(strategy, {multiplier, multiplier * 2, multiplier, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_BW, strategy, m_node));
    validateStrategy(strategy, {multiplier, std::nullopt, multiplier, std::nullopt});
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_BW, strategy, m_node));

    // 3) Inflate for number of slices - based on distance from FCD : BVD 2 (tensor dim 1) -> BVD 0 (tensor dim 3):
    // BVD 2 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, std::nullopt, multiplier * 2, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, std::nullopt, std::nullopt, std::nullopt});
    // BVD 0 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier * 2, std::nullopt, std::nullopt, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {std::nullopt, std::nullopt, std::nullopt, std::nullopt});

    // Expect no inflation - all BVDs are unsliced
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
}

TEST_F(StrategyInflatorTest, inflate_according_to_distance_from_fcd_with_tie_breaker)
{
    const uint64_t bvdResolution = 3;
    createBVDs(bvdResolution);
    StrategyPtr    strategy   = createUnslicedStrategy();
    const uint64_t multiplier = 1;
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        strategy->setBVDMultiplier(bvd, BVDMultiplier(multiplier));
    }
    ASSERT_EQ(m_numBVDs, 4);

    StrategyInflator inflator(m_bundleViews);

    // BVDs should be inflated based on their distance from FCD - start with BVDs that are mapped to FCD dimensions on
    // the bundle tensors. If different tensors in the bundle have conflicting order then the dimension with highest
    // number of tensors in the bundle should be selected.
    // Mapped tensor dims:
    // BVD 0 : (input, 3), (output, 3)
    // BVD 1 : (input, 2), (output, 0)
    // BVD 2 : (input, 1), (output, 1)
    // BVD 3 : (input, 0), (output, 2), (t, 0)
    // -> BVD 3 should be inflated before BVD1 because two FCD tensor dims are mapped to it.

    // BVD 3 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, multiplier, multiplier, multiplier * 2});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, multiplier, multiplier, std::nullopt});

    // BVD 1 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, multiplier * 2, multiplier, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, std::nullopt, multiplier, std::nullopt});

    // BVD 2 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, std::nullopt, multiplier * 2, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier, std::nullopt, std::nullopt, std::nullopt});

    // BVD 0 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {multiplier * 2, std::nullopt, std::nullopt, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {std::nullopt, std::nullopt, std::nullopt, std::nullopt});

    // Expect no inflation - all BVDs are unsliced
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
}

TEST_F(StrategyInflatorTest, common_dim_should_not_be_inflated_all_the_way)
{
    const uint64_t bvdResolution = 12;
    createBVDs(bvdResolution);
    StrategyPtr    strategy            = createUnslicedStrategy();
    const uint64_t multiplier          = 2;
    strategy->setBVDMultiplier(1, BVDMultiplier(multiplier));
    strategy->getMmeSolution()->QORs[m_node]->solutionRequirements.cdDims.push_back(1);
    ASSERT_EQ(m_numBVDs, 4);

    StrategyInflator inflator(m_bundleViews);

    // BVD 1 should be inflated
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {std::nullopt, multiplier * 2, std::nullopt, std::nullopt});

    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {std::nullopt, multiplier * 3, std::nullopt, std::nullopt});

    // Expect no inflation - BVD 1 is a sliced common dim - max multiplier should be <=
    // half BVD resolution
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    ASSERT_TRUE(strategy->getBVDMultiplier(1).isSliced());
}

TEST_F(StrategyInflatorTest, perforated_dim_should_be_inflated_to_multiplication_of_num_dcores)
{
    const uint64_t bvdResolution = 48;
    createBVDs(bvdResolution);
    StrategyPtr strategy = createUnslicedStrategy();
    ASSERT_EQ(m_numBVDs, 4);
    ASSERT_EQ(CompilationHalReader::getHalReader()->getNumDcores(), 4);
    const BundleViewId perforatedBVD = 1;
    strategy->setBVDMultiplier(perforatedBVD, BVDMultiplier(3UL));
    PerforationPerNode perforationPerNode;
    perforationPerNode[m_node] = perforatedBVD;
    strategy->setPerforationData(perforationPerNode);

    StrategyInflator inflator(m_bundleViews);

    // BVD 1 should be inflated to multiplication of 4, one step each time
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_PERFORATION, strategy, m_node));
    validateStrategy(strategy, {std::nullopt, 6UL, std::nullopt, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_PERFORATION, strategy, m_node));
    validateStrategy(strategy, {std::nullopt, 9UL, std::nullopt, std::nullopt});
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_PERFORATION, strategy, m_node));
    validateStrategy(strategy, {std::nullopt, 12UL, std::nullopt, std::nullopt});

    // Expect no inflation - BVD 1 multiplier is devided by 4
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_PERFORATION, strategy, m_node));
    validateStrategy(strategy, {std::nullopt, 12UL, std::nullopt, std::nullopt});

    // Inflations to reduce num slices should use the new multiplier - 12.
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {std::nullopt, 24UL, std::nullopt, std::nullopt});
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 2);

    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    validateStrategy(strategy, {std::nullopt, std::nullopt, std::nullopt, std::nullopt});
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 1);

    // Expect no inflation - all BVDs are unsliced
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
}

TEST_F(StrategyInflatorTest, inflate_for_num_slices_should_reduce_num_of_slices_in_each_step)
{
    const uint64_t bvdResolution = 10;
    createBVDs(bvdResolution);
    ASSERT_EQ(m_numBVDs, 4);
    StrategyPtr    strategy   = createUnslicedStrategy();
    const uint64_t multiplier = 1;
    strategy->setBVDMultiplier(1, BVDMultiplier(multiplier));
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 10);

    StrategyInflator inflator(m_bundleViews);

    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 5);
    validateStrategy(strategy, {std::nullopt, multiplier * 2, std::nullopt, std::nullopt});

    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 4);
    validateStrategy(strategy, {std::nullopt, multiplier * 3, std::nullopt, std::nullopt});

    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 3);
    validateStrategy(strategy, {std::nullopt, multiplier * 4, std::nullopt, std::nullopt});

    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 2);
    validateStrategy(strategy, {std::nullopt, multiplier * 5, std::nullopt, std::nullopt});

    // Skip multipliers 6,7,8,9 -> same num of slices as previous multiplier (5)
    ASSERT_TRUE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
    ASSERT_EQ(strategy->getNumOfSlicesForBVD(1, m_bundleViews), 1);
    validateStrategy(strategy, {std::nullopt, std::nullopt, std::nullopt, std::nullopt});

    // Expect no inflation - all BVDs are unsliced
    ASSERT_FALSE(inflator.inflateOneStep(InflationType::INFLATE_FOR_NUM_SLICES, strategy));
}
