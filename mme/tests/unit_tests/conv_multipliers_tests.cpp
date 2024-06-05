#include <memory>

#include "index_space_dimensions.h"
#include "multipliers_generator.h"
#include "mme_unit_test.h"
#include "mme_params_factory.h"
#include "include/mme_common/mme_brain.h"
#include "operand_access.h"
#include "include/mme_access_pattern.h"

using namespace MmeCommon;
using namespace MmeCommon::Brain;

using ConvAspect = ConvolutionAspects::Name;
using ISDim = AccessPatternDetails::Conv::IndexSpaceDim;

struct MmeUTConvMultipliersTest
: public MMEUnitTest
, public ::testing::WithParamInterface<std::tuple<ChipType,  // chip
                                                  EMmeOpType,  // Operation
                                                  unsigned  // No. of spatial dimensions
                                                  >>
{
    constexpr static size_t C = 256;
    constexpr static size_t K = 512;
    constexpr static size_t W = 24;
    constexpr static size_t H = 24;
    constexpr static size_t D = 12;
    constexpr static size_t B = 32;

    ChipType m_chip;
    EMmeOpType m_opType;
    unsigned m_spatialDimsNr;

    std::unique_ptr<MmeLayerParams> m_params;
    AccessPattern m_accessPattern;
    MultiplierArray m_granularity;
    MultiplierArray m_previousMultipliers;

    std::unique_ptr<SolutionMultipliers::MultipliersGenerator> m_multipliersGenerator;

    MmeUTConvMultipliersTest()
    {
        std::tie(m_chip, m_opType, m_spatialDimsNr) = GetParam();
        initLayerParams();
        m_accessPattern = AccessPatternFactory::createFrom(m_params.get());
        m_granularity.resize(m_accessPattern.indexSpace.size(), 1);
        m_previousMultipliers.resize(m_accessPattern.indexSpace.size(), 1);
    }

    void initLayerParams()
    {
        m_params = std::make_unique<MmeLayerParams>(getMmeLayerParams(m_chip));

        m_params->opType = m_opType;

        m_params->x.sizes = {C, W, H, D, 1};
        m_params->y.sizes = {K, W, H, D, 1};
        m_params->w.sizes = {K, C, 1, 1, 1};

        m_params->conv.spatialDimsNr = m_spatialDimsNr;
        m_params->x.sizes[m_spatialDimsNr + 1] = B;
        m_params->y.sizes[m_spatialDimsNr + 1] = B;
        for (size_t dim = m_spatialDimsNr + 2; dim < m_params->x.sizes.size(); dim++)
        {
            m_params->x.sizes[dim] = 1;
            m_params->y.sizes[dim] = 1;
        }
    }

    void initMultiplierGenerator()
    {
        m_multipliersGenerator = std::make_unique<SolutionMultipliers::MultipliersGenerator>(
            *m_params,
            m_accessPattern,
            m_granularity,
            m_previousMultipliers,
            AspectFactoryPtr(new ConvolutionAspects::Factory(m_params.get())));
    }

    SolutionMultipliers::MultipliersGenerator& getMultipliersGenerator() const
    {
        MME_ASSERT(m_multipliersGenerator.get() != nullptr, "Multipliers generator not initialized");
        return *m_multipliersGenerator;
    }

    MultiplierArray getInflateUpToSolution(ConvAspect aspect, uint64_t desiredSize, EMmeOperand operand) const
    {
        auto& mgn = getMultipliersGenerator();
        mgn.inflateUpTo(aspect, desiredSize, operand);
        return mgn.getSolution();
    }

    MultiplierArray getInflateAtLeastToSolution(ConvAspect aspect, uint64_t desiredSize, EMmeOperand operand) const
    {
        auto& mgn = getMultipliersGenerator();
        mgn.inflateAtLeastTo(aspect, desiredSize, operand);
        return mgn.getSolution();
    }

    MultiplierArray getInflateToMaxSolution(ConvAspect aspect) const
    {
        auto& mgn = getMultipliersGenerator();
        mgn.setMaxMultiplier(aspect);
        return mgn.getSolution();
    }

    void validateSpatialMultipliers(const MultiplierArray& expMultipliers, const MultiplierArray& solution) const
    {
        using namespace AccessPatternDetails::Conv;
        EXPECT_EQ(expMultipliers.at(ISDim::DIM_WIDTH), solution.at(ISDim::DIM_WIDTH));
        if (m_spatialDimsNr > 1)
        {
            EXPECT_EQ(expMultipliers.at(ISDim::DIM_HEIGHT), solution.at(ISDim::DIM_HEIGHT));
        }
        if (m_spatialDimsNr > 2)
        {
            EXPECT_EQ(expMultipliers.at(ISDim::DIM_DEPTH), solution.at(ISDim::DIM_DEPTH));
        }
        EXPECT_EQ(expMultipliers.at(ISDim::DIM_BATCH), solution.at(ISDim::DIM_BATCH));
    }
};

TEST_P(MmeUTConvMultipliersTest, conv_multipliers_should_set_ifm_channels_multiplier_for_full_size)
{
    // in-channels previous solution is 4 granules of 3 index elements each
    m_granularity.at(ISDim::DIM_IN_CHANNELS) = 3;
    m_previousMultipliers.at(ISDim::DIM_IN_CHANNELS) = 4;
    initMultiplierGenerator();

    // When inflating the in-channels aspect up to the full dim size
    auto solution = getInflateUpToSolution(ConvAspect::IN_CHANNELS, C, e_mme_op_x);

    // Expect the multiplier to cover the entire dimension with granules
    auto expMultiplier = div_round_up(C, 3);
    EXPECT_EQ(expMultiplier, solution.at(ISDim::DIM_IN_CHANNELS));

    // When inflating the in-channels aspect at least to the full dim size
    solution = getInflateAtLeastToSolution(ConvAspect::IN_CHANNELS, C, e_mme_op_x);

    // Expects the same multiplier since full size is always rounded up
    EXPECT_EQ(expMultiplier, solution.at(ISDim::DIM_IN_CHANNELS));

    // When setting full size
    solution = getInflateToMaxSolution(ConvAspect::IN_CHANNELS);

    // Still expecting the same multiplier
    EXPECT_EQ(expMultiplier, solution.at(ISDim::DIM_IN_CHANNELS));
}

TEST_P(MmeUTConvMultipliersTest, conv_multipliers_should_set_ifm_channels_multiplier_for_part_size)
{
    // in-channels previous solution is 5 granules of 2 index elements each
    m_granularity.at(ISDim::DIM_IN_CHANNELS) = 2;
    m_previousMultipliers.at(ISDim::DIM_IN_CHANNELS) = 5;
    initMultiplierGenerator();

    // When inflating the in-channels aspect to up to 64 elements
    auto solution = getInflateUpToSolution(ConvAspect::IN_CHANNELS, C / 4, e_mme_op_x);

    // Expect the multiplier to cover 64 elements of the tensor with granules of size 2.
    // So the desired multiplier is 32, but since the previous multiplier is 5 and the inflator
    // is bound to multiples of that, the actual multiple will be 30.
    EXPECT_EQ(round_down_to_multiple(C / 8, 5), solution.at(ISDim::DIM_IN_CHANNELS));

    // When inflating the in-channels aspect to at least 64 elements
    solution = getInflateAtLeastToSolution(ConvAspect::IN_CHANNELS, C / 4, e_mme_op_x);

    // Now, the approximation to 32 with multiples of 5 would be from above, so the expected value is 35
    EXPECT_EQ(round_to_multiple(C / 8, 5), solution.at(ISDim::DIM_IN_CHANNELS));
}

TEST_P(MmeUTConvMultipliersTest, conv_multipliers_should_set_ofm_channels_multiplier_for_full_size)
{
    // in-channels previous solution is 4 granules of 3 index elements each
    m_granularity.at(ISDim::DIM_OUT_CHANNELS) = 6;
    m_previousMultipliers.at(ISDim::DIM_OUT_CHANNELS) = 7;
    initMultiplierGenerator();

    // When inflating the in-channels aspect up to the full dim size
    auto solution = getInflateUpToSolution(ConvAspect::OUT_CHANNELS, K, e_mme_op_y);

    // Expect the multiplier to cover the entire dimension with granules
    auto expMultiplier = div_round_up(K, 6);
    EXPECT_EQ(expMultiplier, solution.at(ISDim::DIM_OUT_CHANNELS));

    // When inflating the in-channels aspect at least to the full dim size
    solution = getInflateAtLeastToSolution(ConvAspect::OUT_CHANNELS, K, e_mme_op_y);

    // Expects the same multiplier since full size is always rounded up
    EXPECT_EQ(expMultiplier, solution.at(ISDim::DIM_OUT_CHANNELS));

    // When setting full size
    solution = getInflateToMaxSolution(ConvAspect::OUT_CHANNELS);

    // Still expecting the same multiplier
    EXPECT_EQ(expMultiplier, solution.at(ISDim::DIM_OUT_CHANNELS));
}

TEST_P(MmeUTConvMultipliersTest, conv_multipliers_should_set_ofm_channels_multiplier_for_part_size)
{
    // in-channels previous solution is 5 granules of 2 index elements each
    m_granularity.at(ISDim::DIM_OUT_CHANNELS) = 2;
    m_previousMultipliers.at(ISDim::DIM_OUT_CHANNELS) = 9;
    initMultiplierGenerator();

    // When inflating the in-channels aspect to up to 64 elements
    auto solution = getInflateUpToSolution(ConvAspect::OUT_CHANNELS, K / 2, e_mme_op_y);

    // Expect the multiplier to cover 256 elements of the tensor with granules of size 1.
    // So the desired multiplier is 256, but since the previous multiplier is 9 and the inflator
    // is bound to multiples of that, the actual multiple will be rounding down of 128 to multiples of 9 (14*9=126).
    EXPECT_EQ(round_down_to_multiple(K / 4, 9), solution.at(ISDim::DIM_OUT_CHANNELS));

    // When inflating the in-channels aspect to at least 64 elements
    solution = getInflateAtLeastToSolution(ConvAspect::OUT_CHANNELS, K / 2, e_mme_op_y);

    // Now, the approximation would be from above, so the expected value is round up of 256 to multiples of 9 (15*9=135)
    EXPECT_EQ(round_to_multiple(K / 4, 9), solution.at(ISDim::DIM_OUT_CHANNELS));
}

TEST_P(MmeUTConvMultipliersTest, conv_multipliers_should_set_spatial_multipliers_for_full_size)
{
    // Set some garbage as spatial granularity and previous solution
    m_granularity.at(ISDim::DIM_WIDTH) = 1;
    m_granularity.at(ISDim::DIM_HEIGHT) = 2;
    m_granularity.at(ISDim::DIM_DEPTH) = 3;
    m_granularity.at(ISDim::DIM_BATCH) = 4;
    m_previousMultipliers.at(ISDim::DIM_WIDTH) = 1;
    m_previousMultipliers.at(ISDim::DIM_HEIGHT) = 2;
    m_previousMultipliers.at(ISDim::DIM_DEPTH) = 3;
    m_previousMultipliers.at(ISDim::DIM_BATCH) = 4;
    initMultiplierGenerator();

    // Expect max multipliers on all spatial dimensions
    MultiplierArray expMultipliers(m_previousMultipliers.size());
    for (auto dim : {ISDim::DIM_WIDTH, ISDim::DIM_HEIGHT, ISDim::DIM_DEPTH, ISDim::DIM_BATCH})
    {
        expMultipliers[dim] = div_round_up(m_accessPattern.indexSpace.at(dim), m_granularity.at(dim));
    }

    auto operand = m_params->getExternalOperand(e_mme_op_c);

    auto solution = getInflateUpToSolution(ConvAspect::SPATIAL, B * D * H * W, operand);
    validateSpatialMultipliers(expMultipliers, solution);

    solution = getInflateAtLeastToSolution(ConvAspect::SPATIAL, B * D * H * W, operand);
    validateSpatialMultipliers(expMultipliers, solution);

    solution = getInflateToMaxSolution(ConvAspect::SPATIAL);
    validateSpatialMultipliers(expMultipliers, solution);
}

TEST_P(MmeUTConvMultipliersTest, conv_multipliers_should_set_spatial_multipliers_for_part_size_w)
{
    constexpr uint64_t granuleSize = 2;
    m_granularity.at(ISDim::DIM_WIDTH) = granuleSize;
    m_previousMultipliers.at(ISDim::DIM_WIDTH) = 3;
    initMultiplierGenerator();

    MultiplierArray expMultipliers(m_previousMultipliers.size(), 1);

    // When inflating up to
    auto solution = getInflateUpToSolution(ConvAspect::SPATIAL, 8 * granuleSize, e_mme_op_y);
    // The solution mutliplier should be divisible by 3 (the previous multiplier) and approximate the desired no. of
    // granules from below - so 6 granules (9 is the next multiple of 3 and is too high)
    expMultipliers[ISDim::DIM_WIDTH] = 6;
    validateSpatialMultipliers(expMultipliers, solution);

    solution = getInflateAtLeastToSolution(ConvAspect::SPATIAL, 8 * granuleSize, e_mme_op_y);
    // With the same calculation as before, now we expect 9 granules
    expMultipliers[ISDim::DIM_WIDTH] = 9;
    validateSpatialMultipliers(expMultipliers, solution);
}

TEST_P(MmeUTConvMultipliersTest, conv_multipliers_should_set_spatial_multipliers_for_part_size_b)
{
    constexpr uint64_t granuleSize = 3;
    m_granularity.at(ISDim::DIM_BATCH) = granuleSize;
    m_previousMultipliers.at(ISDim::DIM_BATCH) = 2;
    initMultiplierGenerator();

    // Expect max multipliers on all spatial dimensions (except the batch)
    MultiplierArray expMultipliers(m_previousMultipliers.size());
    for (auto dim : {ISDim::DIM_WIDTH, ISDim::DIM_HEIGHT, ISDim::DIM_DEPTH})
    {
        expMultipliers[dim] = div_round_up(m_accessPattern.indexSpace.at(dim), m_granularity.at(dim));
    }

    uint64_t sampleSize = W;
    if (m_spatialDimsNr > 1) sampleSize *= H;
    if (m_spatialDimsNr > 2) sampleSize *= D;

    // The spatial aspect here will be inflated to cover multiples 3 full samples
    uint64_t aspectGranuleSize = sampleSize * granuleSize;

    // The solution mutliplier should be divisible by 2 (the previous multiplier) and approximate the desired no. of
    // aspect granules from below - so 4 granules (6 is the next multiple of 2 and is too high)
    auto solution = getInflateUpToSolution(ConvAspect::SPATIAL, 5 * aspectGranuleSize, e_mme_op_y);
    // The solution is expected to be 12 samples
    expMultipliers[ISDim::DIM_BATCH] = 4;
    validateSpatialMultipliers(expMultipliers, solution);

    solution = getInflateAtLeastToSolution(ConvAspect::SPATIAL, 5 * aspectGranuleSize, e_mme_op_y);
    // With the same calculation as before, now we expect 6 granules
    expMultipliers[ISDim::DIM_BATCH] = 6;
    validateSpatialMultipliers(expMultipliers, solution);
}

INSTANTIATE_TEST_CASE_P(mme_brain,
                        MmeUTConvMultipliersTest,
                        ::testing::Combine(::testing::Values(e_mme_Gaudi, e_mme_Gaudi2, e_mme_Gaudi3),
                                           ::testing::Values(e_mme_fwd, e_mme_dedx, e_mme_transposed_dedx, e_mme_dedw),
                                           ::testing::Range(1u, 4u)  // No. of spatial dims (end exclusive)
                                           ));
