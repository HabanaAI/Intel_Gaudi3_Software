#include "mme_access_pattern.h"
#include "mme_aspects.h"
#include "mme_params_factory.h"
#include "mme_unit_test.h"
#include "index_space_dimensions.h"

using namespace MmeCommon;
using namespace MmeCommon::AccessPatternDetails;

class PhysicalAspectsTest : public MMEUnitTest
{
public:
    using PhAspect = PhysicalAspects::Name;

    template<typename Container>
    static void validateAspect(const Aspect& aspect, const Container& expectedDims)
    {
        ASSERT_EQ(expectedDims.size(), aspect.size());
        auto aspectDimIter = aspect.begin();
        auto expectedDimIter = expectedDims.begin();
        for (; aspectDimIter != aspect.end(); ++aspectDimIter, ++expectedDimIter)
        {
            EXPECT_EQ(*expectedDimIter, *aspectDimIter);
        }
    }

    static void validateAspect(const Aspect& aspect, const std::initializer_list<Dim>& expectedDims)
    {
        validateAspect(aspect, std::vector(expectedDims));
    }
};

class MmeUTGemmPhysicalAspectsTest
: public PhysicalAspectsTest
, public ::testing::WithParamInterface<std::tuple<ChipType,
                                                  bool,  // transpose A
                                                  bool,  // transpose B
                                                  bool,  // Broadcast B
                                                  unsigned  // Num dims
                                                  >>
{
public:
    constexpr static uint64_t FCD_SIZE = 256;
    constexpr static uint64_t SPT_SIZE = 1024;
    constexpr static uint64_t BT_SIZE = 32;

    ChipType m_chip;
    bool m_atpos;
    bool m_btpos;
    bool m_broadcastB;
    unsigned m_numDims;

    MmeUTGemmPhysicalAspectsTest() { std::tie(m_chip, m_atpos, m_btpos, m_broadcastB, m_numDims) = GetParam(); }

    MmeLayerParams gemmLayerParams() const
    {
        auto params = getMmeLayerParams(m_chip);
        params.x.sizes = {FCD_SIZE, SPT_SIZE, BT_SIZE, BT_SIZE - 1, BT_SIZE + 1};
        params.y.sizes = {FCD_SIZE, SPT_SIZE, BT_SIZE, BT_SIZE - 1, BT_SIZE + 1};
        params.w.sizes = {FCD_SIZE, FCD_SIZE, BT_SIZE, BT_SIZE - 1, BT_SIZE + 1};
        if (m_broadcastB)
        {
            params.w.sizes[GEMM_DIM_B1] = 1;
        }

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

    IndexSpaceAspect getIndexSpaceAspect(PhAspect aspect) const
    {
        auto params = gemmLayerParams();
        PhysicalAspects::Factory factory(&params);
        return factory.create(AspectName(aspect));
    }

    OperandAspect getOperandAspect(PhAspect aspect, EMmeOperand operand) const
    {
        auto params = gemmLayerParams();
        PhysicalAspects::Factory factory(&params);
        return factory.create(AspectName(aspect), operand);
    }
};

TEST_P(MmeUTGemmPhysicalAspectsTest, physical_aspects_generator_should_provide_output_fcd_as_bgemm_width)
{
    auto widthIdxSpcAspect = getIndexSpaceAspect(PhAspect::OUTPUT_WIDTH);
    validateAspect(widthIdxSpcAspect, {Gemm::DIM_OUT_FCD});

    auto widthOperandCAspect = getOperandAspect(PhAspect::OUTPUT_WIDTH, EMmeOperand::e_mme_op_y);
    validateAspect(widthOperandCAspect, {DIM_K});

    auto widthOperandBAspect = getOperandAspect(PhAspect::OUTPUT_WIDTH, EMmeOperand::e_mme_op_w);
    validateAspect(widthOperandBAspect, {m_btpos ? WEIGHT_DIM_C : DIM_K});

    auto widthOperandAAspect = getOperandAspect(PhAspect::OUTPUT_WIDTH, EMmeOperand::e_mme_op_x);
    validateAspect(widthOperandAAspect, {});
}

TEST_P(MmeUTGemmPhysicalAspectsTest, physical_aspects_generator_should_provide_input_cd_as_bgemm_common)
{
    auto cdIdxSpcAspect = getIndexSpaceAspect(PhAspect::INPUTS_COMMON);
    validateAspect(cdIdxSpcAspect, {Gemm::DIM_OPERANDS_COMMON});

    auto cdOperandCAspect = getOperandAspect(PhAspect::INPUTS_COMMON, EMmeOperand::e_mme_op_y);
    validateAspect(cdOperandCAspect, {});

    auto cdOperandBAspect = getOperandAspect(PhAspect::INPUTS_COMMON, EMmeOperand::e_mme_op_w);
    validateAspect(cdOperandBAspect, {m_btpos ? DIM_K : WEIGHT_DIM_C});

    auto cdOperandAAspect = getOperandAspect(PhAspect::INPUTS_COMMON, EMmeOperand::e_mme_op_x);
    validateAspect(cdOperandAAspect, {m_atpos ? DIM_W : DIM_C});
}

TEST_P(MmeUTGemmPhysicalAspectsTest, physical_aspects_generator_should_provide_height_dims_for_bgemm)
{
    const auto params = gemmLayerParams();
    auto heightIdxSpcAspect = getIndexSpaceAspect(PhAspect::OUTPUT_HEIGHT);
    auto heightOperandCAspect = getOperandAspect(PhAspect::OUTPUT_HEIGHT, EMmeOperand::e_mme_op_y);
    auto heightOperandAAspect = getOperandAspect(PhAspect::OUTPUT_HEIGHT, EMmeOperand::e_mme_op_x);

    const auto opAHeightDim = m_atpos ? GEMM_DIM_W : GEMM_DIM_H;
    if (params.canFlatten())
    {
        validateAspect(heightIdxSpcAspect, {Gemm::DIM_OUT_HEIGHT, Gemm::DIM_BATCH_0});
        validateAspect(heightOperandCAspect, {GEMM_DIM_H, GEMM_DIM_B1});
        validateAspect(heightOperandAAspect, {opAHeightDim, GEMM_DIM_B1});
    }
    else
    {
        validateAspect(heightIdxSpcAspect, {Gemm::DIM_OUT_HEIGHT});
        validateAspect(heightOperandCAspect, {GEMM_DIM_H});
        validateAspect(heightOperandAAspect, {opAHeightDim});
    }

    auto heightOperandBAspect = getOperandAspect(PhAspect::OUTPUT_HEIGHT, EMmeOperand::e_mme_op_w);
    validateAspect(heightOperandBAspect, {});
}

TEST_P(MmeUTGemmPhysicalAspectsTest, physical_aspects_generator_should_provide_batch_dims_for_bgemm)
{
    const auto params = gemmLayerParams();

    auto batchIdxSpcAspect = getIndexSpaceAspect(PhAspect::GROUPS);

    std::vector<size_t> indexSpaceBatchDims = {Gemm::DIM_BATCH_1, Gemm::DIM_BATCH_2};
    if (!params.canFlatten()) indexSpaceBatchDims.insert(indexSpaceBatchDims.begin(), Gemm::DIM_BATCH_0);
    validateAspect(batchIdxSpcAspect, indexSpaceBatchDims);

    auto batchOperandCAspect = getOperandAspect(PhAspect::GROUPS, EMmeOperand::e_mme_op_y);
    auto batchOperandAAspect = getOperandAspect(PhAspect::GROUPS, EMmeOperand::e_mme_op_x);
    auto batchOperandBAspect = getOperandAspect(PhAspect::GROUPS, EMmeOperand::e_mme_op_w);

    std::vector<size_t> tensorBatchDims = {GEMM_DIM_B2, GEMM_DIM_B3};
    if (!params.canFlatten()) tensorBatchDims.insert(tensorBatchDims.begin(), GEMM_DIM_B1);
    validateAspect(batchOperandCAspect, tensorBatchDims);
    validateAspect(batchOperandAAspect, tensorBatchDims);

    std::vector<size_t> wghBatchDims = {GEMM_DIM_B2, GEMM_DIM_B3};
    if (!m_broadcastB)
    {
        wghBatchDims.insert(wghBatchDims.begin(), GEMM_DIM_B1);  // operand B broadcast is not dependent on flattening
    }
    validateAspect(batchOperandBAspect, wghBatchDims);
}

INSTANTIATE_TEST_CASE_P(
    _,
    MmeUTGemmPhysicalAspectsTest,
    ::testing::Combine(::testing::Values(ChipType::e_mme_Gaudi, ChipType::e_mme_Gaudi2, ChipType::e_mme_Gaudi3),
                       ::testing::Bool(),  // transpose-A
                       ::testing::Bool(),  // transpose-B
                       ::testing::Bool(),  // broadcast operand B
                       ::testing::Range(2u, 6u)  // num dims (end exclusive)
                       ));

class MmeUTConvPhysicalAspectsTest
: public PhysicalAspectsTest
, public ::testing::WithParamInterface<std::tuple<ChipType,
                                                  EMmeOpType,
                                                  unsigned  // Nr. of spatial dimensions
                                                  >>
{
public:
    using ISDim = AccessPatternDetails::Conv::IndexSpaceDim;

    MmeUTConvPhysicalAspectsTest() { std::tie(m_chip, m_op, m_spatialDimNr) = GetParam(); }

    ChipType m_chip;
    EMmeOpType m_op;
    unsigned m_spatialDimNr;

    constexpr static uint64_t CD_SIZE = 256;
    constexpr static uint64_t K_SIZE = 512;
    constexpr static uint64_t SP_SIZE = 16;
    constexpr static uint64_t BT_SIZE = 64;
    constexpr static uint64_t FILTER_SIZE = 3;

    Dim batchDim() const { return m_spatialDimNr + 1; }  // Channels + spatials

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

        return params;
    }

    IndexSpaceAspect getIndexSpaceAspect(PhAspect aspect) const
    {
        auto params = convLayerParams();
        PhysicalAspects::Factory factory(&params);
        return factory.create(AspectName(aspect));
    }

    OperandAspect getOperandAspect(PhAspect aspect, EMmeOperand operand) const
    {
        auto params = convLayerParams();
        PhysicalAspects::Factory factory(&params);
        return factory.create(AspectName(aspect), operand);
    }

    std::vector<size_t> convSpatialDims() const
    {
        std::vector<size_t> dims(m_spatialDimNr + 1);
        for (size_t i = 0; i < m_spatialDimNr; i++)
        {
            dims[i] = DIM_W + i;
        }
        dims[m_spatialDimNr] = batchDim();
        return dims;
    }
};

TEST_P(MmeUTConvPhysicalAspectsTest, physical_aspects_generator_should_provide_conv_output_width)
{
    const auto params = convLayerParams();

    PhysicalAspects::Factory factory(&params);

    auto widthAspect_idxSpc = factory.create(AspectName(PhAspect::OUTPUT_WIDTH));
    // expected index space aspect content:
    auto expDim_idxSpc = params.isDedxOperation() ? ISDim::DIM_IN_CHANNELS : ISDim::DIM_OUT_CHANNELS;
    validateAspect(widthAspect_idxSpc, {expDim_idxSpc});

    auto cOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_c);
    auto widthAspect_opC = factory.create(AspectName(PhAspect::OUTPUT_WIDTH), cOperand);
    // expected operand C aspect content:
    auto expDim_opC = params.isDedxOperation() ? DIM_C : DIM_K;
    validateAspect(widthAspect_opC, {expDim_opC});

    auto bOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_b);
    auto widthAspect_opB = factory.create(AspectName(PhAspect::OUTPUT_WIDTH), bOperand);
    // expected operand B aspect content:
    auto expDim_opB = (params.isDedxOperation() && m_op != e_mme_transposed_dedx) ? WEIGHT_DIM_C : DIM_K;
    validateAspect(widthAspect_opB, {expDim_opB});

    auto aOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_a);
    auto widthAspect_opA = factory.create(AspectName(PhAspect::OUTPUT_WIDTH), aOperand);
    // expected operand A aspect empty
    validateAspect(widthAspect_opA, {});
}

TEST_P(MmeUTConvPhysicalAspectsTest, physical_aspects_generator_should_provide_conv_common_dims)
{
    const auto params = convLayerParams();

    PhysicalAspects::Factory factory(&params);

    auto commonAspect_idxSpc = factory.create(AspectName(PhAspect::INPUTS_COMMON));
    const auto expIdxSpcDim =
        params.isDedwOperation()
            ? std::vector<Dim> {ISDim::DIM_WIDTH, ISDim::DIM_HEIGHT, ISDim::DIM_DEPTH, ISDim::DIM_BATCH}
        : params.isDedxOperation() ? std::vector<Dim> {ISDim::DIM_OUT_CHANNELS}
                                   : std::vector<Dim> {ISDim::DIM_IN_CHANNELS};
    validateAspect(commonAspect_idxSpc, expIdxSpcDim);

    auto cOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_c);
    auto commonAspect_opC = factory.create(AspectName(PhAspect::INPUTS_COMMON), cOperand);
    validateAspect(commonAspect_opC, {});

    auto bOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_b);
    auto commonAspect_opB = factory.create(AspectName(PhAspect::INPUTS_COMMON), bOperand);
    if (params.isDedwOperation())
    {
        validateAspect(commonAspect_opB, convSpatialDims());
    }
    else
    {
        const auto opBCommonDim = (params.isDedxOperation() && m_op != e_mme_transposed_dedx) ? DIM_K : WEIGHT_DIM_C;
        validateAspect(commonAspect_opB, {opBCommonDim});
    }

    auto aOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_a);
    auto commonAspect_opA = factory.create(AspectName(PhAspect::INPUTS_COMMON), aOperand);
    if (params.isDedwOperation())
    {
        validateAspect(commonAspect_opA, convSpatialDims());
    }
    else
    {
        const auto opACommonDim = params.isDedxOperation() ? DIM_K : DIM_C;
        validateAspect(commonAspect_opA, {opACommonDim});
    }
}

TEST_P(MmeUTConvPhysicalAspectsTest, physical_aspects_generator_should_provide_conv_height_dims)
{
    const auto params = convLayerParams();

    PhysicalAspects::Factory factory(&params);

    auto heightAspect_idxSpc = factory.create(AspectName(PhAspect::OUTPUT_HEIGHT));
    if (params.isDedwOperation())
    {
        validateAspect(heightAspect_idxSpc, {ISDim::DIM_IN_CHANNELS});
    }
    else
    {
        validateAspect(heightAspect_idxSpc, {ISDim::DIM_WIDTH, ISDim::DIM_HEIGHT, ISDim::DIM_DEPTH, ISDim::DIM_BATCH});
    }

    auto cOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_c);
    auto heightAspect_opC = factory.create(AspectName(PhAspect::OUTPUT_HEIGHT), cOperand);
    if (params.isDedwOperation())
    {
        validateAspect(heightAspect_opC, {WEIGHT_DIM_C});
    }
    else
    {
        validateAspect(heightAspect_opC, convSpatialDims());
    }

    auto bOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_b);
    auto heightAspect_opB = factory.create(AspectName(PhAspect::OUTPUT_HEIGHT), bOperand);
    validateAspect(heightAspect_opB, {});

    auto aOperand = params.getExternalOperand(EMmeInternalOperand::e_mme_op_a);
    auto heightAspect_opA = factory.create(AspectName(PhAspect::OUTPUT_HEIGHT), aOperand);
    if (params.isDedwOperation())
    {
        validateAspect(heightAspect_opA, {DIM_C});
    }
    else
    {
        validateAspect(heightAspect_opA, convSpatialDims());
    }
}

INSTANTIATE_TEST_CASE_P(_,
                        MmeUTConvPhysicalAspectsTest,
                        ::testing::Combine(::testing::Values(e_mme_Gaudi, e_mme_Gaudi2, e_mme_Gaudi3),
                                           ::testing::Values(e_mme_fwd, e_mme_dedx, e_mme_dedw, e_mme_transposed_dedx),
                                           ::testing::Range(1u, 4u)  // Spatial dims number (Range is end-exclusive)
                                           ));
