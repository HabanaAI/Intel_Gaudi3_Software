#include "mme_access_pattern.h"
#include "index_space_dimensions.h"
#include "mme_unit_test.h"

#include "mme_params_factory.h"

using namespace MmeCommon;
using namespace MmeCommon::AccessPatternDetails;

struct MMEAccessPatternTest : public MMEUnitTest
{
    static void
    validate1To1DimAccessPattern(const MmeCommon::AccessPattern::TensorAccessPattern::DimAccessPattern& dimAP,
                                 Dim expIndexSpaceMapping)
    {
        EXPECT_EQ(expIndexSpaceMapping, dimAP.indexSpaceDim);
        EXPECT_EQ(0, dimAP.offset);
        EXPECT_EQ(1, dimAP.stride);
        EXPECT_EQ(1, dimAP.size);
    }

    static void
    validateAllReqDimAccessPattern(const MmeCommon::AccessPattern::TensorAccessPattern::DimAccessPattern& dimAP,
                                   uint64_t tensorDimSize,
                                   Dim expIndexSpaceMapping)
    {
        EXPECT_EQ(expIndexSpaceMapping, dimAP.indexSpaceDim);
        EXPECT_EQ(0, dimAP.offset);
        EXPECT_EQ(0, dimAP.stride);
        EXPECT_EQ(tensorDimSize, dimAP.size);
    }

    static LayerSemantics baseLayerSemantics(const MmeLayerParams& layerParams, size_t rank = MME_MAX_TENSOR_DIMS)
    {
        using TensorShape = LayerSemantics::TensorProperties::TensorShape;
        LayerSemantics lsm;
        lsm.op = layerParams.opType;
        lsm.convParams = layerParams.conv;
        lsm.operandShapes[OperandRole::X] = LayerSemantics::TensorProperties {
            TensorShape(layerParams.x.sizes.begin(), layerParams.x.sizes.begin() + rank)};
        lsm.operandShapes[OperandRole::W] = LayerSemantics::TensorProperties {
            TensorShape(layerParams.w.sizes.begin(), layerParams.w.sizes.begin() + rank)};
        lsm.operandShapes[OperandRole::Y] = LayerSemantics::TensorProperties {
            TensorShape(layerParams.y.sizes.begin(), layerParams.y.sizes.begin() + rank)};
        return lsm;
    }
};

struct MmeUTGemmAccessPatternTest
: public MMEAccessPatternTest
, public ::testing::WithParamInterface<std::tuple<ChipType,
                                                  bool,  // transpose A
                                                  bool,  // transpose B
                                                  bool,  // Use semantic query API
                                                  unsigned  // Num dims
                                                  >>
{
    constexpr static uint64_t FCD_SIZE = 256;
    constexpr static uint64_t SPT_SIZE = 1024;
    constexpr static uint64_t BT_SIZE = 32;

    ChipType m_chip;
    bool m_atpos;
    bool m_btpos;
    bool m_semanticQuery;
    unsigned m_numDims;

    MmeUTGemmAccessPatternTest() { std::tie(m_chip, m_atpos, m_btpos, m_semanticQuery, m_numDims) = GetParam(); }

    AccessPattern getAccessPattern(const MmeLayerParams& layerParams) const
    {
        if (m_semanticQuery)
        {
            auto query = baseLayerSemantics(layerParams, m_numDims);
            return AccessPatternFactory::createFrom(&query);
        }
        return AccessPatternFactory::createFrom(&layerParams);
    }

    MmeLayerParams gemmLayerParams() const
    {
        auto params = getMmeLayerParams(m_chip);
        params.x.sizes = {FCD_SIZE, SPT_SIZE, BT_SIZE, BT_SIZE - 1, BT_SIZE + 1};
        params.w.sizes = {FCD_SIZE, FCD_SIZE, BT_SIZE, BT_SIZE - 1, BT_SIZE + 1};
        params.y.sizes = {FCD_SIZE, SPT_SIZE, BT_SIZE, BT_SIZE - 1, BT_SIZE + 1};

        switch ((static_cast<unsigned>(m_atpos) << 1) + static_cast<unsigned>(m_btpos))
        {
            case 0:
                params.opType = EMmeOpType::e_mme_ab;
                break;
            case 1:
                params.opType = EMmeOpType::e_mme_abt;
                std::swap(params.w.sizes[0], params.w.sizes[1]);
                break;
            case 2:
                params.opType = EMmeOpType::e_mme_atb;
                std::swap(params.x.sizes[0], params.x.sizes[1]);
                break;
            case 3:
                params.opType = EMmeOpType::e_mme_atbt;
                std::swap(params.w.sizes[0], params.w.sizes[1]);
                std::swap(params.x.sizes[0], params.x.sizes[1]);
                break;
        }

        return params;
    }

    void validateIndexSpace(const AccessPattern::IndexSpaceVector& isv) const
    {
        using namespace Gemm;
        EXPECT_EQ(FCD_SIZE, isv.at(DIM_OUT_FCD));
        EXPECT_EQ(FCD_SIZE, isv.at(DIM_OPERANDS_COMMON));
        EXPECT_EQ(SPT_SIZE, isv.at(DIM_OUT_HEIGHT));
        if (m_semanticQuery)
        {
            EXPECT_EQ(m_numDims > 2 ? BT_SIZE : 1, isv.at(DIM_BATCH_0));
            EXPECT_EQ(m_numDims > 3 ? BT_SIZE - 1 : 1, isv.at(DIM_BATCH_1));
            EXPECT_EQ(m_numDims > 4 ? BT_SIZE + 1 : 1, isv.at(DIM_BATCH_2));
        }
        else
        {
            EXPECT_EQ(BT_SIZE, isv.at(DIM_BATCH_0));
            EXPECT_EQ(BT_SIZE - 1, isv.at(DIM_BATCH_1));
            EXPECT_EQ(BT_SIZE + 1, isv.at(DIM_BATCH_2));
        }
    }

    void validateOperandAccessPattern(const AccessPattern& accessPattern,
                                      OperandRole role,
                                      const std::vector<Dim>& expMapping) const
    {
        ASSERT_NE(accessPattern.operandAccessPatterns.find(role), accessPattern.operandAccessPatterns.end());
        validateOperandAccessPattern(accessPattern.operandAccessPatterns.at(role), expMapping);
    }

    void validateOperandAccessPattern(const MmeCommon::AccessPattern::TensorAccessPattern& tap,
                                      const std::vector<Dim>& expMapping,
                                      std::optional<size_t> expNumDims = std::nullopt) const
    {
        constexpr static size_t EXP_NUM_DIMS_FOR_LAYER_PARAMS = 5;
        if (!expNumDims)
        {
            expNumDims = m_semanticQuery ? m_numDims : EXP_NUM_DIMS_FOR_LAYER_PARAMS;
        }

        ASSERT_EQ(*expNumDims, tap.dimsAccessPattern.size());
        for (Dim dim = 0; dim < *expNumDims; dim++)
        {
            validate1To1DimAccessPattern(tap.dimsAccessPattern.at(dim), expMapping[dim]);
            EXPECT_FALSE(HasFailure()) << "Failure at dim " << dim;
        }
    }
};

TEST_P(MmeUTGemmAccessPatternTest, ap_factory_should_returen_1_to_1_ap_for_bgemm)
{
    using namespace Gemm;

    auto layerParams = gemmLayerParams();
    auto accessPattern = getAccessPattern(layerParams);

    validateIndexSpace(accessPattern.indexSpace);

    const std::vector<Dim> OP_A_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_atpos ? DIM_OUT_HEIGHT : DIM_OPERANDS_COMMON,
        m_atpos ? DIM_OPERANDS_COMMON : DIM_OUT_HEIGHT,
        DIM_BATCH_0,
        DIM_BATCH_1,
        DIM_BATCH_2,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::X, OP_A_EXP_INDEX_SPACE_DIM_MAPPING);

    const std::vector<Dim> OP_B_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_btpos ? DIM_OPERANDS_COMMON : DIM_OUT_FCD,
        m_btpos ? DIM_OUT_FCD : DIM_OPERANDS_COMMON,
        DIM_BATCH_0,
        DIM_BATCH_1,
        DIM_BATCH_2,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::W, OP_B_EXP_INDEX_SPACE_DIM_MAPPING);

    const std::vector<Dim> OP_C_EXP_INDEX_SPACE_DIM_MAPPING = {
        DIM_OUT_FCD,
        DIM_OUT_HEIGHT,
        DIM_BATCH_0,
        DIM_BATCH_1,
        DIM_BATCH_2,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::Y, OP_C_EXP_INDEX_SPACE_DIM_MAPPING);
}

TEST_P(MmeUTGemmAccessPatternTest, ap_factory_should_returen_bcast_ap_for_bcast_w)
{
    using namespace Gemm;

    auto layerParams = gemmLayerParams();
    layerParams.w.sizes[2] = layerParams.w.sizes[3] = 1;

    auto accessPattern = getAccessPattern(layerParams);

    validateIndexSpace(accessPattern.indexSpace);

    const std::vector<Dim> OP_B_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_btpos ? DIM_OPERANDS_COMMON : DIM_OUT_FCD,
        m_btpos ? DIM_OUT_FCD : DIM_OPERANDS_COMMON,
        DIM_BCAST,
        DIM_BCAST,
        DIM_BATCH_2,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::W, OP_B_EXP_INDEX_SPACE_DIM_MAPPING);
}

TEST_P(MmeUTGemmAccessPatternTest, ap_factory_should_returen_bcast_ap_for_bcast_x)
{
    using namespace Gemm;

    auto layerParams = gemmLayerParams();
    layerParams.x.sizes[3] = layerParams.x.sizes[4] = 1;

    auto accessPattern = getAccessPattern(layerParams);

    validateIndexSpace(accessPattern.indexSpace);

    const std::vector<Dim> OP_A_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_atpos ? DIM_OUT_HEIGHT : DIM_OPERANDS_COMMON,
        m_atpos ? DIM_OPERANDS_COMMON : DIM_OUT_HEIGHT,
        DIM_BATCH_0,
        DIM_BCAST,
        DIM_BCAST,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::X, OP_A_EXP_INDEX_SPACE_DIM_MAPPING);
}

TEST_P(MmeUTGemmAccessPatternTest, ap_factory_should_returen_bcast_ap_for_mixed_bcast)
{
    using namespace Gemm;

    auto layerParams = gemmLayerParams();
    layerParams.w.sizes[2] = layerParams.w.sizes[4] = layerParams.x.sizes[3] = 1;

    auto accessPattern = getAccessPattern(layerParams);

    validateIndexSpace(accessPattern.indexSpace);

    const std::vector<Dim> OP_A_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_atpos ? DIM_OUT_HEIGHT : DIM_OPERANDS_COMMON,
        m_atpos ? DIM_OPERANDS_COMMON : DIM_OUT_HEIGHT,
        DIM_BATCH_0,
        DIM_BCAST,
        DIM_BATCH_2,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::X, OP_A_EXP_INDEX_SPACE_DIM_MAPPING);

    const std::vector<Dim> OP_B_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_btpos ? DIM_OPERANDS_COMMON : DIM_OUT_FCD,
        m_btpos ? DIM_OUT_FCD : DIM_OPERANDS_COMMON,
        DIM_BCAST,
        DIM_BATCH_1,
        DIM_BCAST,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::W, OP_B_EXP_INDEX_SPACE_DIM_MAPPING);
}

INSTANTIATE_TEST_CASE_P(
    _,
    MmeUTGemmAccessPatternTest,
    ::testing::Combine(::testing::Values(ChipType::e_mme_Gaudi, ChipType::e_mme_Gaudi2, ChipType::e_mme_Gaudi3),
                       ::testing::Bool(),  // transpose-A
                       ::testing::Bool(),  // transpose-B
                       ::testing::Bool(),  // use semantic query
                       ::testing::Range(2u, 6u)  // num dims (end exclusive)
                       ));

// For testing features that are only available with semantic query
class MmeUTGemmSemanticAccessPatternTest : public MmeUTGemmAccessPatternTest
{
};

// Test the semantic query mode for tensors with different ranks
TEST_P(MmeUTGemmSemanticAccessPatternTest, ap_factory_should_return_correct_ap_for_different_ranks)
{
    using namespace Gemm;

    {
        auto layerParams = gemmLayerParams();
        // Get layer params with full rank
        auto semanticLayer = baseLayerSemantics(layerParams);
        // Reduce the rank of W
        semanticLayer.operandShapes.at(OperandRole::W).shape.resize(m_numDims);

        auto accessPattern = AccessPatternFactory::createFrom(&semanticLayer);

        const std::vector<Dim> OP_B_EXP_INDEX_SPACE_DIM_MAPPING = {
            m_btpos ? DIM_OPERANDS_COMMON : DIM_OUT_FCD,
            m_btpos ? DIM_OUT_FCD : DIM_OPERANDS_COMMON,
            DIM_BATCH_0,
            DIM_BATCH_1,
            DIM_BATCH_2,
        };
        validateOperandAccessPattern(accessPattern, OperandRole::W, OP_B_EXP_INDEX_SPACE_DIM_MAPPING);
    }

    {
        auto layerParams = gemmLayerParams();
        // Get layer params with full rank
        auto semanticLayer = baseLayerSemantics(layerParams, layerParams.x.sizes.size());
        // Reduce the rank of X
        semanticLayer.operandShapes.at(OperandRole::X).shape.resize(m_numDims);

        auto accessPattern = AccessPatternFactory::createFrom(&semanticLayer);

        const std::vector<Dim> OP_A_EXP_INDEX_SPACE_DIM_MAPPING = {
            m_atpos ? DIM_OUT_HEIGHT : DIM_OPERANDS_COMMON,
            m_atpos ? DIM_OPERANDS_COMMON : DIM_OUT_HEIGHT,
            DIM_BATCH_0,
            DIM_BATCH_1,
            DIM_BATCH_2,
        };
        validateOperandAccessPattern(accessPattern, OperandRole::X, OP_A_EXP_INDEX_SPACE_DIM_MAPPING);
    }
}

TEST_P(MmeUTGemmSemanticAccessPatternTest, ap_factory_should_return_correct_ap_for_bias)
{
    auto params = gemmLayerParams();
    auto semantic = baseLayerSemantics(params);
    semantic.operandShapes[OperandRole::BIAS] = {LayerSemantics::TensorProperties::TensorShape {params.y.sizes[0]}};

    auto accessPattern = AccessPatternFactory::createFrom(&semantic);

    const std::vector<Dim> OP_BIAS_EXP_INDEX_SPACE_DIM_MAPPING = {Gemm::DIM_OUT_FCD};

    validateOperandAccessPattern(accessPattern.operandAccessPatterns.at(OperandRole::BIAS),
                                 OP_BIAS_EXP_INDEX_SPACE_DIM_MAPPING,
                                 1);
}

TEST_P(MmeUTGemmSemanticAccessPatternTest, ap_factory_should_return_correct_ap_for_masks)
{
    using namespace Gemm;

    auto params = gemmLayerParams();
    auto semantic = baseLayerSemantics(params, m_numDims);
    semantic.operandShapes[OperandRole::MASK_A] = semantic.operandShapes.at(OperandRole::X);
    semantic.operandShapes[OperandRole::MASK_A].shape[0] = 5;
    if (m_atpos)
        std::swap(semantic.operandShapes[OperandRole::MASK_A].shape[0],
                  semantic.operandShapes[OperandRole::MASK_A].shape[1]);
    semantic.operandShapes[OperandRole::MASK_B] = semantic.operandShapes.at(OperandRole::W);
    semantic.operandShapes[OperandRole::MASK_B].shape[1] = 5;
    if (m_btpos)
        std::swap(semantic.operandShapes[OperandRole::MASK_B].shape[0],
                  semantic.operandShapes[OperandRole::MASK_B].shape[1]);

    if (m_numDims > 2)
    {
        semantic.operandShapes[OperandRole::MASK_A].shape[2] = 1;
        semantic.operandShapes[OperandRole::MASK_B].shape[2] = 1;
    }

    auto accessPattern = AccessPatternFactory::createFrom(&semantic);

    EXPECT_EQ(accessPattern.indexSpace.at(Gemm::DIM_MASKS_COMMON), 5);
    EXPECT_EQ(accessPattern.indexSpace.at(Gemm::DIM_NUM_IDENTICAL_MASKS), 1);

    const std::vector<Dim> MASK_A_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_atpos ? DIM_OUT_HEIGHT : DIM_MASKS_COMMON,
        m_atpos ? DIM_MASKS_COMMON : DIM_OUT_HEIGHT,
        DIM_BCAST,
        DIM_BATCH_1,
        DIM_BATCH_2,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::MASK_A, MASK_A_EXP_INDEX_SPACE_DIM_MAPPING);

    const std::vector<Dim> MASK_B_EXP_INDEX_SPACE_DIM_MAPPING = {
        m_btpos ? DIM_MASKS_COMMON : DIM_OUT_FCD,
        m_btpos ? DIM_OUT_FCD : DIM_MASKS_COMMON,
        DIM_BCAST,
        DIM_BATCH_1,
        DIM_BATCH_2,
    };
    validateOperandAccessPattern(accessPattern, OperandRole::MASK_B, MASK_B_EXP_INDEX_SPACE_DIM_MAPPING);
}

INSTANTIATE_TEST_CASE_P(
    _,
    MmeUTGemmSemanticAccessPatternTest,
    ::testing::Combine(::testing::Values(ChipType::e_mme_Gaudi, ChipType::e_mme_Gaudi2, ChipType::e_mme_Gaudi3),
                       ::testing::Bool(),  // transpose-A
                       ::testing::Bool(),  // transpose-B
                       ::testing::Values(true),  // use semantic query
                       ::testing::Range(2u, 6u)  // num dims (end exclusive)
                       ));

using SpatialDimPropertyBase = uint32_t;
enum SpatialDimProperty : SpatialDimPropertyBase
{
    FILTER = 1 << 0,
    PAD_BEFORE = 1 << 1,
    PAD_AFTER = 1 << 2,
    STRIDE = 1 << 3,

    // DILATION is not tested by itself since when there is filter>1 the dimension is not strict even without
    // dilation > 1 and when the filter==1, diletion has no effect.

    MAX = 1 << 4
};

class MmeUTConvAccessPatternTest
: public MMEAccessPatternTest
, public ::testing::WithParamInterface<std::tuple<ChipType,
                                                  EMmeOpType,
                                                  SpatialDimPropertyBase,  // Non strict dim properties
                                                  unsigned,  // Nr. of spatial dimensions
                                                  bool  // use semantic query
                                                  >>
{
public:
    MmeUTConvAccessPatternTest()
    {
        std::tie(m_chip, m_op, m_nonStrictProperties, m_spatialDimNr, m_semanticQuery) = GetParam();
    }

    ChipType m_chip;
    EMmeOpType m_op;
    SpatialDimPropertyBase m_nonStrictProperties;
    unsigned m_nonStrictSpatialDim = 0;
    unsigned m_spatialDimNr;
    bool m_semanticQuery;

    constexpr static uint64_t CD_SIZE = 256;
    constexpr static uint64_t K_SIZE = 512;
    constexpr static uint64_t SP_SIZE = 16;
    constexpr static uint64_t BT_SIZE = 64;
    constexpr static uint64_t FILTER_SIZE = 3;

    void testAccessPattern(Dim nonStrictSpatialDim)
    {
        m_nonStrictSpatialDim = nonStrictSpatialDim;
        auto params = convLayerParams();
        auto accessPattern = getAccessPattern(params);

        validateIndexSpace(accessPattern.indexSpace);

        validateFeatureMap(getXAP(accessPattern), Conv::DIM_IN_CHANNELS, params.x.sizes);
        validateFeatureMap(getYAP(accessPattern), Conv::DIM_OUT_CHANNELS, params.y.sizes);

        validateWgh(getWAP(accessPattern), params.w.sizes);
    }

    AccessPattern getAccessPattern(const MmeLayerParams& params) const
    {
        if (m_semanticQuery)
        {
            auto semQ = baseLayerSemantics(params, m_spatialDimNr + 2);
            return AccessPatternFactory::createFrom(&semQ);
        }
        return AccessPatternFactory::createFrom(&params);
    }

    MmeLayerParams convLayerParams() const
    {
        auto params = getMmeLayerParams(m_chip);
        params.opType = m_op;

        params.x.sizes = {CD_SIZE, 1, 1, 1, 1};
        params.w.sizes = {K_SIZE, CD_SIZE, 1, 1, 1};
        params.y.sizes = {K_SIZE, 1, 1, 1, 1};

        for (size_t dim = 1; dim < batchDim(); dim++)
        {
            params.x.sizes[dim] = SP_SIZE;
            params.y.sizes[dim] = SP_SIZE;
        }
        params.x.sizes[batchDim()] = BT_SIZE;
        params.y.sizes[batchDim()] = BT_SIZE;

        if (m_op == MmeCommon::e_mme_transposed_dedx)
        {
            std::swap(params.w.sizes[0], params.w.sizes[1]);
        }

        params.conv.dilation.fill(1);
        params.conv.padding.fill(0);
        params.conv.stride.fill(1);
        params.conv.spatialDimsNr = m_spatialDimNr;

        if (m_nonStrictProperties & SpatialDimProperty::FILTER)
        {
            // Skip C,K
            params.w.sizes.at(m_nonStrictSpatialDim + 2) = 3;
        }
        if (m_nonStrictProperties & SpatialDimProperty::PAD_BEFORE)
        {
            // negativity is not important, as long as it's != 0
            params.conv.padding.at(m_nonStrictSpatialDim) = -1;
        }
        if (m_nonStrictProperties & SpatialDimProperty::PAD_AFTER)
        {
            // Skip K, add elements to the output as if they're generated by padding "after"
            params.y.sizes.at(m_nonStrictSpatialDim + 1) += 5;
        }
        if (m_nonStrictProperties & SpatialDimProperty::STRIDE)
        {
            // will cause padding after too (due to y size)
            params.conv.stride.at(m_nonStrictSpatialDim) = 2;
        }

        return params;
    }

    Dim batchDim() const { return m_spatialDimNr + 1; }  // Channels + spatials

    void validateIndexSpace(const AccessPattern::IndexSpaceVector& isv) const
    {
        using namespace Conv;
        EXPECT_EQ(CD_SIZE, isv.at(DIM_IN_CHANNELS));
        EXPECT_EQ(K_SIZE, isv.at(DIM_OUT_CHANNELS));

        EXPECT_EQ(isStrict(0) ? SP_SIZE : 1, isv.at(DIM_WIDTH));
        EXPECT_EQ(m_spatialDimNr > 1 && isStrict(1) ? SP_SIZE : 1, isv.at(DIM_HEIGHT));
        EXPECT_EQ(m_spatialDimNr > 2 && isStrict(2) ? SP_SIZE : 1, isv.at(DIM_DEPTH));

        EXPECT_EQ(1, isv.at(DIM_FILTER_S));
        EXPECT_EQ(1, isv.at(DIM_FILTER_R));
        EXPECT_EQ(1, isv.at(DIM_FILTER_Q));

        EXPECT_EQ(BT_SIZE, isv.at(DIM_BATCH));
    }

    bool isStrict(Dim spatialDim) const { return m_nonStrictProperties == 0 || m_nonStrictSpatialDim != spatialDim; }

    static const AccessPattern::TensorAccessPattern& getXAP(const AccessPattern& ap)
    {
        return ap.operandAccessPatterns.at(OperandRole::X);
    }

    static const AccessPattern::TensorAccessPattern& getYAP(const AccessPattern& ap)
    {
        return ap.operandAccessPatterns.at(OperandRole::Y);
    }

    static const AccessPattern::TensorAccessPattern& getWAP(const AccessPattern& ap)
    {
        return ap.operandAccessPatterns.at(OperandRole::W);
    }

    void validateFeatureMap(const AccessPattern::TensorAccessPattern& tap,
                            Dim channelsIdxSpcDim,
                            const SizeArray& tensorSizes) const
    {
        validate1To1DimAccessPattern(tap.dimsAccessPattern.at(0), channelsIdxSpcDim);

        Dim spatialDim = 0;
        for (auto convDim : {Conv::DIM_WIDTH, Conv::DIM_HEIGHT, Conv::DIM_DEPTH})
        {
            if (spatialDim >= m_spatialDimNr) break;
            const Dim tensorDim = spatialDim + 1;
            if (isStrict(spatialDim))
            {
                validate1To1DimAccessPattern(tap.dimsAccessPattern.at(tensorDim), convDim);
            }
            else
            {
                validateAllReqDimAccessPattern(tap.dimsAccessPattern.at(tensorDim), tensorSizes[tensorDim], convDim);
            }
            spatialDim++;
        }

        validate1To1DimAccessPattern(tap.dimsAccessPattern.at(batchDim()), Conv::DIM_BATCH);
    }

    void validateWgh(const AccessPattern::TensorAccessPattern& wap, const SizeArray& wghSizes) const
    {
        validate1To1DimAccessPattern(wap.dimsAccessPattern.at(0),
                                     m_op == MmeCommon::e_mme_transposed_dedx ? Conv::DIM_IN_CHANNELS
                                                                              : Conv::DIM_OUT_CHANNELS);
        validate1To1DimAccessPattern(wap.dimsAccessPattern.at(1),
                                     m_op == MmeCommon::e_mme_transposed_dedx ? Conv::DIM_OUT_CHANNELS
                                                                              : Conv::DIM_IN_CHANNELS);

        Dim filterDim = 0;
        for (auto convDim : {Conv::DIM_FILTER_S, Conv::DIM_FILTER_R, Conv::DIM_FILTER_Q})
        {
            if (filterDim >= m_spatialDimNr) break;
            const Dim wghDim = filterDim + 2;
            validateAllReqDimAccessPattern(wap.dimsAccessPattern.at(wghDim), wghSizes[wghDim], convDim);
            filterDim++;
        }
    }
};

TEST_P(MmeUTConvAccessPatternTest, ap_factory_should_create_1_to_1_for_strict_all_req_otherwise)
{
    for (Dim nonStrictSpatialDim = 0; nonStrictSpatialDim < m_spatialDimNr; nonStrictSpatialDim++)
    {
        testAccessPattern(nonStrictSpatialDim);
    }
}

INSTANTIATE_TEST_CASE_P(
    _,
    MmeUTConvAccessPatternTest,
    ::testing::Combine(::testing::Values(e_mme_Gaudi, e_mme_Gaudi2, e_mme_Gaudi3),
                       ::testing::Values(e_mme_fwd, e_mme_dedx, e_mme_dedw, e_mme_transposed_dedx),
                       ::testing::Range(static_cast<SpatialDimPropertyBase>(0),
                                        static_cast<SpatialDimPropertyBase>(SpatialDimProperty::MAX)),
                       ::testing::Range(1u, 4u),  // Spatial dims number (Range is end-exclusive)
                       ::testing::Bool()  // Semantic query
                       ));

// Suite to check features available only through the semantic API
class MmeUTConvSemanticAccessPatternTest : public MmeUTConvAccessPatternTest
{
public:
    static void compareDimAccessPatterns(const AccessPattern::TensorAccessPattern::DimAccessPattern& exp,
                                         const AccessPattern::TensorAccessPattern::DimAccessPattern& actual)
    {
        EXPECT_EQ(exp.indexSpaceDim, actual.indexSpaceDim);
        EXPECT_EQ(exp.size, actual.size);
        EXPECT_EQ(exp.stride, actual.stride);
        EXPECT_EQ(exp.offset, actual.offset);
    }

    OperandRole outputRole() const
    {
        return m_op == MmeCommon::e_mme_fwd    ? OperandRole::Y
               : m_op == MmeCommon::e_mme_dedw ? OperandRole::W
                                               : OperandRole::X;
    }
};

TEST_P(MmeUTConvSemanticAccessPatternTest, ap_factory_should_support_output_copy)
{
    auto params = convLayerParams();
    auto sem = baseLayerSemantics(params, m_spatialDimNr + 2);
    sem.operandShapes[OperandRole::OUTPUT_COPY] = sem.operandShapes.at(outputRole());

    auto accessPattern = AccessPatternFactory::createFrom(&sem);

    ASSERT_NE(accessPattern.operandAccessPatterns.end(),
              accessPattern.operandAccessPatterns.find(OperandRole::OUTPUT_COPY));

    const auto& outputAP = accessPattern.operandAccessPatterns.at(outputRole());
    const auto& outputCopyAP = accessPattern.operandAccessPatterns.at(OperandRole::OUTPUT_COPY);

    EXPECT_EQ(outputAP.dimsAccessPattern.size(), outputCopyAP.dimsAccessPattern.size());
    for (Dim dim = 0; dim < outputAP.dimsAccessPattern.size(); dim++)
    {
        compareDimAccessPatterns(outputAP.dimsAccessPattern.at(dim), outputCopyAP.dimsAccessPattern.at(dim));
    }
}

TEST_P(MmeUTConvSemanticAccessPatternTest, ap_factory_should_support_fwd_with_bias)
{
    if (m_op != MmeCommon::e_mme_fwd) return;

    auto params = convLayerParams();
    auto sem = baseLayerSemantics(params, m_spatialDimNr + 2);
    sem.operandShapes[OperandRole::BIAS] = sem.operandShapes.at(OperandRole::Y);
    sem.operandShapes[OperandRole::BIAS].shape.resize(1);

    auto accessPattern = AccessPatternFactory::createFrom(&sem);

    ASSERT_NE(accessPattern.operandAccessPatterns.end(), accessPattern.operandAccessPatterns.find(OperandRole::BIAS));

    const auto& outputAP = accessPattern.operandAccessPatterns.at(OperandRole::Y);
    const auto& biasAP = accessPattern.operandAccessPatterns.at(OperandRole::BIAS);

    EXPECT_EQ(1, biasAP.dimsAccessPattern.size());
    compareDimAccessPatterns(outputAP.dimsAccessPattern.front(), biasAP.dimsAccessPattern.front());
}

TEST_P(MmeUTConvSemanticAccessPatternTest, ap_factory_should_support_dedx_with_shape)
{
    if (m_op != MmeCommon::e_mme_dedx && m_op != MmeCommon::e_mme_transposed_dedx) return;

    auto params = convLayerParams();
    auto sem = baseLayerSemantics(params, m_spatialDimNr + 2);
    sem.operandShapes[OperandRole::SHAPE] = sem.operandShapes.at(OperandRole::X);

    auto accessPattern = AccessPatternFactory::createFrom(&sem);

    ASSERT_NE(accessPattern.operandAccessPatterns.end(), accessPattern.operandAccessPatterns.find(OperandRole::SHAPE));

    const auto& outputAP = accessPattern.operandAccessPatterns.at(OperandRole::X);
    const auto& shapeAP = accessPattern.operandAccessPatterns.at(OperandRole::SHAPE);

    EXPECT_EQ(outputAP.dimsAccessPattern.size(), shapeAP.dimsAccessPattern.size());
    for (Dim dim = 0; dim < outputAP.dimsAccessPattern.size(); dim++)
    {
        compareDimAccessPatterns(outputAP.dimsAccessPattern.at(dim), shapeAP.dimsAccessPattern.at(dim));
    }
}

TEST_P(MmeUTConvSemanticAccessPatternTest, ap_factory_should_resize_idx_spc_when_applying_parallelism)
{
    constexpr size_t BT_PARALLEL_LEVEL = 2;
    constexpr size_t W_PARALLEL_LEVEL = 4;

    auto params = convLayerParams();
    auto sem = baseLayerSemantics(params, m_spatialDimNr + 2);

    auto accessPattern = AccessPatternFactory::createFrom(&sem);
    AccessPatternFactory::applyParallelism(&accessPattern, Conv::DIM_WIDTH, W_PARALLEL_LEVEL);
    AccessPatternFactory::applyParallelism(&accessPattern, Conv::DIM_BATCH, BT_PARALLEL_LEVEL);

    EXPECT_EQ(BT_PARALLEL_LEVEL, accessPattern.indexSpace.at(Conv::DIM_BATCH));
    EXPECT_EQ(W_PARALLEL_LEVEL, accessPattern.indexSpace.at(Conv::DIM_WIDTH));
    const auto& xAP = accessPattern.operandAccessPatterns.at(OperandRole::X).dimsAccessPattern;
    EXPECT_EQ(BT_SIZE / BT_PARALLEL_LEVEL, xAP.at(batchDim()).size);
    EXPECT_EQ(BT_SIZE / BT_PARALLEL_LEVEL, xAP.at(batchDim()).stride);
    EXPECT_EQ(SP_SIZE / W_PARALLEL_LEVEL, xAP.at(1).size);
    EXPECT_EQ(SP_SIZE / W_PARALLEL_LEVEL, xAP.at(1).stride);
    const auto& yAP = accessPattern.operandAccessPatterns.at(OperandRole::Y).dimsAccessPattern;
    EXPECT_EQ(BT_SIZE / BT_PARALLEL_LEVEL, yAP.at(batchDim()).size);
    EXPECT_EQ(BT_SIZE / BT_PARALLEL_LEVEL, yAP.at(batchDim()).stride);
    EXPECT_EQ(SP_SIZE / W_PARALLEL_LEVEL, yAP.at(1).size);
    EXPECT_EQ(SP_SIZE / W_PARALLEL_LEVEL, yAP.at(1).stride);
}

TEST_P(MmeUTConvSemanticAccessPatternTest, ap_factory_should_add_aux_access_when_applying_parallelism)
{
    constexpr size_t BT_PARALLEL_LEVEL = 2;
    constexpr size_t W_PARALLEL_LEVEL = 4;

    auto params = convLayerParams();
    auto sem = baseLayerSemantics(params, m_spatialDimNr + 2);

    auto accessPattern = AccessPatternFactory::createFrom(&sem);
    AccessPatternFactory::applyParallelism(&accessPattern, Conv::DIM_WIDTH, W_PARALLEL_LEVEL);
    AccessPatternFactory::applyParallelism(&accessPattern, Conv::DIM_BATCH, BT_PARALLEL_LEVEL);

    ASSERT_NE(accessPattern.operandAccessPatterns.end(),
              accessPattern.operandAccessPatterns.find(OperandRole::SCRATCH_PAD));
    ASSERT_NE(accessPattern.operandAccessPatterns.end(), accessPattern.operandAccessPatterns.find(OperandRole::CONST));

    const auto& outputAP = accessPattern.operandAccessPatterns.at(outputRole());
    const auto& scratchPadAP = accessPattern.operandAccessPatterns.at(OperandRole::SCRATCH_PAD);
    const auto& constAP = accessPattern.operandAccessPatterns.at(OperandRole::CONST);

    Dim dim = 0;
    for (; dim < outputAP.dimsAccessPattern.size(); dim++)
    {
        compareDimAccessPatterns(outputAP.dimsAccessPattern.at(dim), scratchPadAP.dimsAccessPattern.at(dim));
    }
    validate1To1DimAccessPattern(scratchPadAP.dimsAccessPattern.at(dim), Conv::DIM_WIDTH);
    validate1To1DimAccessPattern(constAP.dimsAccessPattern.at(0), Conv::DIM_WIDTH);
    validate1To1DimAccessPattern(scratchPadAP.dimsAccessPattern.at(dim + 1), Conv::DIM_BATCH);
    validate1To1DimAccessPattern(constAP.dimsAccessPattern.at(1), Conv::DIM_BATCH);
}

INSTANTIATE_TEST_CASE_P(
    _,
    MmeUTConvSemanticAccessPatternTest,
    ::testing::Combine(::testing::Values(e_mme_Gaudi, e_mme_Gaudi2, e_mme_Gaudi3),
                       ::testing::Values(e_mme_fwd, e_mme_dedx, e_mme_dedw, e_mme_transposed_dedx),
                       ::testing::Values(static_cast<SpatialDimPropertyBase>(0)),  // No need for non-strict
                       ::testing::Range(1u, 4u),  // Spatial dims number (Range is end-exclusive)
                       ::testing::Values(true)  // Semantic query
                       ));
