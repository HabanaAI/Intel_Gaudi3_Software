#include "mme_brain_test.h"
#include "include/mme_common/mme_common_enum.h"
#include "src/mme_common/mme_geo_factory.h"
#include "index_space_dimensions.h"
#include <gtest/gtest.h>

static constexpr size_t K_INDEX = 0;
static constexpr size_t C_INDEX = 1;
static constexpr size_t W_INDEX = 2;
static constexpr size_t H_INDEX = 3;
static constexpr size_t BATCH_INDEX = 4;
static constexpr size_t OP_INDEX = 5;
static constexpr size_t TYPE_INDEX = 6;
static constexpr size_t GEOMETRY_INDEX = 7;
static constexpr size_t PATTERN_INDEX = 8;
static constexpr size_t FETCH_A_INDEX = 9;
static constexpr size_t FETCH_B_INDEX = 10;

const unsigned dH = 1;
const unsigned dW = 1;
const unsigned kH = 3;
const unsigned kW = 3;
const unsigned padH = 0;
const unsigned padW = 0;

#define EPSILON 0.001

using namespace Gaudi2;
using namespace MmeCommon::AccessPatternDetails;

void MmeUTBrainTest::ConvGeomtryTest(size_t nOFM,
                                   size_t nIFM,
                                   size_t wOFM,
                                   size_t hOFM,
                                   size_t batch,
                                   EMmeOpType operation,
                                   EMmeDataType dataType,
                                   EMmeGeometry expectedGeometry,
                                   EMmePattern expectedPattern,
                                   unsigned fetchA,
                                   unsigned fetchB)
{
    // o = ((i - k + 2 * pad) / stride) + 1
    const unsigned wIFM = ((wOFM - 1) * dW) + kW - (2 * padW);
    const unsigned hIFM = ((hOFM - 1) * dH) + kH - (2 * padH);

    auto params = MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2);
    params.opType = operation;
    params.x.sizes = {nIFM, wIFM, hIFM, 1, batch};
    params.w.sizes = {nOFM, nIFM, kW, kH, 1};
    params.y.sizes = {nOFM, wOFM, hOFM, 1, batch};
    params.x.elementType = params.w.elementType = params.y.elementType = dataType;

    //  turn on optimizations
    params.strategy.sbReuse = true;
    params.strategy.unrollEn = true;
    params.strategy.batchConcurrencyEn = MmeCommon::TurnedOn;

    //  set geometry and pattern to be set by brain
    params.strategy.geometry = e_mme_geometry_nr;
    params.strategy.pattern = e_mme_patterns_nr;
    MmeCommon::MmeBrain brain(MmeCommon::e_mme_Gaudi2);
    brain.getRecommendedStrategy(params);

    ASSERT_EQ(params.strategy.geometry, expectedGeometry);
    ASSERT_EQ(params.strategy.pattern, expectedPattern);

    //  just call the perf calculator for now, will add tests later. remove comment once merge into MME repo
    PerfAttr perfAttr;
    brain.getPerfAttr(params, perfAttr);

    ASSERT_EQ(perfAttr.fetchNrA, fetchA);
    ASSERT_EQ(perfAttr.fetchNrB, fetchB);
}

class MmeUTBrainParametricTest
: public MmeUTBrainTest
, public testing::WithParamInterface<std::tuple<size_t,
                                                size_t,
                                                size_t,
                                                size_t,
                                                size_t,
                                                EMmeOpType,
                                                EMmeDataType,
                                                EMmeGeometry,
                                                EMmePattern,
                                                unsigned,
                                                unsigned>>
{
};

TEST_P(MmeUTBrainParametricTest, DISABLED_convGeometryTests)
{
    ConvGeomtryTest(std::get<K_INDEX>(GetParam()),
                    std::get<C_INDEX>(GetParam()),
                    std::get<W_INDEX>(GetParam()),
                    std::get<H_INDEX>(GetParam()),
                    std::get<BATCH_INDEX>(GetParam()),
                    std::get<OP_INDEX>(GetParam()),
                    std::get<TYPE_INDEX>(GetParam()),
                    std::get<GEOMETRY_INDEX>(GetParam()),
                    std::get<PATTERN_INDEX>(GetParam()),
                    std::get<FETCH_A_INDEX>(GetParam()),
                    std::get<FETCH_B_INDEX>(GetParam()));
}

INSTANTIATE_TEST_CASE_P(
    convGeometryTests,
    MmeUTBrainParametricTest,
    ::testing::Values(
        // Full Geometry Test
        // 128 x 1024 - full 4xH
        std::make_tuple(128, 64, 32, 32, 1, e_mme_fwd, e_type_bf16, e_mme_geometry_4xh, e_mme_z_reduction_skf, 1, 1),

        // 256 x 512 - full 2xH
        std::make_tuple(256, 64, 32, 16, 1, e_mme_fwd, e_type_bf16, e_mme_geometry_2xh, e_mme_z_reduction_skf, 1, 1),

        // 512 x 256 - full 2xW
        std::make_tuple(512, 64, 16, 16, 1, e_mme_fwd, e_type_bf16, e_mme_geometry_2xw, e_mme_z_reduction_skf, 1, 1),

        // 1024 x 128 - full 4xW
        std::make_tuple(1024, 64, 8, 16, 1, e_mme_fwd, e_type_bf16, e_mme_geometry_4xw, e_mme_z_reduction_ksf, 1, 1),

        // 2048 x 2048 - large geometry, no clear preference, currently set to 2xH
        std::make_tuple(2048, 64, 64, 32, 1, e_mme_fwd, e_type_bf16, e_mme_geometry_2xh, e_mme_z_reduction_skf, 1, 4),

        // 2112 x 2048 - large geometry, last FCD very small should prefer 4xH
        std::make_tuple(2112, 64, 64, 32, 1, e_mme_fwd, e_type_bf16, e_mme_geometry_4xh, e_mme_z_reduction_skf, 1, 2),

        // 2048 x 2148 - large geometry, last spatial very small should prefer 4xW
        std::make_tuple(2048, 64, 64, 33, 1, e_mme_fwd, e_type_bf16, e_mme_geometry_4xw, e_mme_z_reduction_ksf, 2, 1),

        // 2048 x 2048 - large geometry with large CD, partial reuse.
        std::make_tuple(2048,
                        2048,
                        64,
                        32,
                        1,
                        e_mme_fwd,
                        e_type_bf16,
                        e_mme_geometry_2xh,
                        e_mme_z_reduction_skf,
                        2,
                        4)));

TEST_F(MmeUTBrainTest, bert_large_test)
{
    MmeBrain brain(ChipType::e_mme_Gaudi2);
    auto params = MmeCommon::MmeBrain::getDefaultParams(ChipType::e_mme_Gaudi2);
    params.x.sizes = {4096, 1536, 1, 1, 1};
    params.x.elementType = MmeCommon::e_type_bf16;
    params.x.strides = {1, 4096, 6291456, 6291456, 6291456};
    params.w.sizes = {1024, 4096, 1, 1, 1};
    params.w.elementType = MmeCommon::e_type_bf16;
    params.w.strides = {1, 1024, 4194304, 4194304, 4194304};
    params.y.sizes = {1024, 1536, 1, 1, 1};
    params.y.elementType = MmeCommon::e_type_bf16;
    params.y.strides = {1, 1024, 9437184, 9437184, 9437184};
    params.strategy.pattern = e_mme_z_reduction_skf;  // walk right
    params.strategy.geometry = MmeCommon::e_mme_geometry_4xw;

    PerfAttr perfAttr;
    brain.getPerfAttr(params, perfAttr);
    ASSERT_EQ(perfAttr.mmeUtilization, (float) 12 / 13);
}

TEST_F(MmeUTBrainTest, narrow_geo_utilizaiton)
{
    MmeBrain brain(ChipType::e_mme_Gaudi2);
    auto params = MmeCommon::MmeBrain::getDefaultParams(ChipType::e_mme_Gaudi2);
    params.opType = MmeCommon::e_mme_ab;
    params.x.sizes = {256, 1024, 1, 1, 1};
    params.x.elementType = MmeCommon::e_type_bf16;
    params.x.strides = {1, 4096, 6291456, 6291456, 6291456};
    params.w.sizes = {128, 256, 1, 1, 1};
    params.w.elementType = MmeCommon::e_type_bf16;
    params.w.strides = {1, 1024, 4194304, 4194304, 4194304};
    params.y.sizes = {128, 1024, 1, 1, 1};
    params.y.elementType = MmeCommon::e_type_bf16;
    params.strategy.pattern = e_mme_sp_reduction_fkc;  // walk down

    PerfAttr perfAttr;

    //  symmetric geometry - half the FCD not utilized
    params.strategy.geometry = MmeCommon::e_mme_geometry_4xh;
    brain.getPerfAttr(params, perfAttr);
    ASSERT_EQ(perfAttr.mmeUtilization, 0.5);

    //  narrow geometry - EU fully utilized but overall 50% utilization due to port constraints
    params.strategy.geometry = MmeCommon::e_mme_geometry_4xh;
    brain.getPerfAttr(params, perfAttr);
    ASSERT_EQ(perfAttr.mmeUtilization, 0.5);
}

static void setTensorView(MmeTensorView& view,
                          const std::vector<unsigned>& shape,
                          const std::vector<unsigned>& strides,
                          EMmeDataType dataType)
{
    view.elementType = dataType;
    for (int i = 0; i < shape.size(); i++)
    {
        view.sizes[i] = shape[i];
        if (strides.size() == 0)
        {
            view.strides[i] = (i == 0 ? 1 : view.strides[i - 1] * view.sizes[i - 1]);
        }
        else
        {
            view.strides[i] = strides[i];
        }
    }
}

void MmeUTBrainTest::checkFlattening(const std::vector<unsigned>& aShape,
                                   const std::vector<unsigned>& bShape,
                                   const std::vector<unsigned>& cShape,
                                   const std::vector<unsigned>& aStrides,
                                   const std::vector<unsigned>& bStrides,
                                   const std::vector<unsigned>& cStrides,
                                   EMmeDataType dataType,
                                   MmeCommon::EMmeGeometry expectedGeometry,
                                   unsigned expectedFlatteningFactor)
{
    MmeBrain brain(ChipType::e_mme_Gaudi2);
    auto params = MmeCommon::MmeBrain::getDefaultParams(ChipType::e_mme_Gaudi2);
    params.opType = MmeCommon::e_mme_ab;
    params.strategy.geometry = e_mme_geometry_nr;
    params.strategy.pattern = e_mme_patterns_nr;
    params.strategy.flattenEn = true;
    setTensorView(params.x, aShape, aStrides, dataType);
    setTensorView(params.w, bShape, bStrides, dataType);
    setTensorView(params.y, cShape, cStrides, dataType);

    brain.getRecommendedStrategy(params);

    ASSERT_EQ(params.strategy.geometry, expectedGeometry);
    ASSERT_EQ(brain.getFlatteningFactor(), expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_bf16_single_divisor_4xw)
{
    // Single divisor that improves the mme utilization
    const std::vector<unsigned> aShape = {5, 4, 2};
    const std::vector<unsigned> bShape = {1024, 5, 1};
    const std::vector<unsigned> cShape = {1024, 4, 2};
    const unsigned expectedFlatteningFactor = 2;
    const MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_4xw;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    {},
                    {},
                    {},
                    MmeCommon::e_type_bf16,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_bf16_single_divisor_2xw)
{
    // Single divisor that improves the mme utilization
    const std::vector<unsigned> aShape = {5, 32, 40};
    const std::vector<unsigned> bShape = {1024, 5, 1};
    const std::vector<unsigned> cShape = {1024, 32, 40};
    const unsigned expectedFlatteningFactor = 8;
    const MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_2xw;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    {},
                    {},
                    {},
                    MmeCommon::e_type_bf16,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_bf16_multiple_divisors_unique_utilizations)
{
    // Multiple divisors, each of different utilization. Choose the best one
    const std::vector<unsigned> aShape = {5, 5, 231};
    const std::vector<unsigned> bShape = {1024, 5, 1};
    const std::vector<unsigned> cShape = {1024, 5, 231};
    const unsigned expectedFlatteningFactor = 231;
    const MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_2xw;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    {},
                    {},
                    {},
                    MmeCommon::e_type_bf16,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_bf16_multiple_divisors_non_unique_utilizations)
{
    // Multiple divisors, some of the same utilization. Choose the minimal divisor.
    const std::vector<unsigned> aShape = {5, 192, 24};
    const std::vector<unsigned> bShape = {240, 5, 1};
    const std::vector<unsigned> cShape = {240, 192, 24};
    const unsigned expectedFlatteningFactor = 8;
    const MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_2xh;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    {},
                    {},
                    {},
                    MmeCommon::e_type_bf16,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, DISABLED_flattening_multiple_divisors_full_utilization)
{
    // Multiple divisors, full flattening gives best utilization
    const std::vector<unsigned> aShape = {5, 128, 48};
    const std::vector<unsigned> bShape = {128, 5, 1};
    const std::vector<unsigned> cShape = {128, 128, 48};
    const unsigned expectedFlatteningFactor = 48;
    const MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_4xh;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    {},
                    {},
                    {},
                    MmeCommon::e_type_bf16,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_bf16_optimal_spatial_size)
{
    // The spatial size is already optimal such that flattening of 1 is already optimal
    const std::vector<unsigned> aShape = {5, 1024, 20};
    const std::vector<unsigned> bShape = {256, 5, 1};
    const std::vector<unsigned> cShape = {256, 1024, 20};
    const unsigned expectedFlatteningFactor = 1;
    const MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_2xh;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    {},
                    {},
                    {},
                    MmeCommon::e_type_bf16,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_strided_on_dims_other_than_2)
{
    // Checks that strides on all dims other than dim 2 do not cancel flattening
    const std::vector<unsigned> aShape = {5, 5, 300};
    const std::vector<unsigned> bShape = {1024, 5, 1};
    const std::vector<unsigned> cShape = {1024, 5, 300};
    const std::vector<unsigned> aStrides = {1, 8, 8 * 5, 8 * 8 * 280, 8 * 16 * 280};
    const std::vector<unsigned> bStrides = {1, 1024, 1024 * 5, 1024 * 10, 1024 * 20};
    const std::vector<unsigned> cStrides = {1, 1040, 1040 * 5, 1040 * 8 * 280, 1040 * 8 * 280};
    MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_2xh;
    unsigned expectedFlatteningFactor = 50;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    aStrides,
                    bStrides,
                    cStrides,
                    MmeCommon::e_type_bf16,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_strided_a_on_dim2_cancel_flattening)
{
    // Here flattening should not occur because the first batch dim of a is strided
    const std::vector<unsigned> aShape = {5, 192, 20, 1, 1};
    const std::vector<unsigned> bShape = {510, 5, 1, 1, 1};
    const std::vector<unsigned> cShape = {510, 192, 20, 1, 1};
    const std::vector<unsigned> aStrides = {1, 10, 15 * 192, 15 * 192 * 20, 15 * 192 * 20};
    const std::vector<unsigned> bStrides = {1, 540, 540 * 5, 540 * 10, 540 * 10};
    const std::vector<unsigned> cStrides = {1, 550, 550 * 192, 550 * 192 * 80, 550 * 192 * 80};
    MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_2xh;
    unsigned expectedFlatteningFactor = 1;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    aStrides,
                    bStrides,
                    cStrides,
                    MmeCommon::e_type_fp32,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

TEST_F(MmeUTBrainTest, flattening_strided_c_on_dim2_cancel_flattening)
{
    // Here flattening should not occur because the first batch dim of c is strided
    const std::vector<unsigned> aShape = {5, 192, 20, 1, 1};
    const std::vector<unsigned> bShape = {520, 5, 1, 1, 1};
    const std::vector<unsigned> cShape = {520, 192, 20, 1, 1};
    const std::vector<unsigned> aStrides = {1, 10, 10 * 192, 10 * 192 * 20, 10 * 192 * 20};
    const std::vector<unsigned> bStrides = {1, 540, 540 * 5, 540 * 10, 540 * 10};
    const std::vector<unsigned> cStrides = {1, 550, 1100 * 192, 1100 * 192 * 80, 1100 * 192 * 80};
    MmeCommon::EMmeGeometry expectedGeometry = MmeCommon::e_mme_geometry_2xh;
    unsigned expectedFlatteningFactor = 1;

    checkFlattening(aShape,
                    bShape,
                    cShape,
                    aStrides,
                    bStrides,
                    cStrides,
                    MmeCommon::e_type_fp32,
                    expectedGeometry,
                    expectedFlatteningFactor);
}

MmeLayerParams
MmeUTBrainTest::setParams(std::vector<unsigned>& nodeDims, EMmeOpType operation, EMmeDataType dataType, bool broadcastB)
{
    auto params = MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi3);
    params.opType = operation;
    MME_ASSERT(params.isGemmOperation() || params.isConvOperation(), "invalid op type");
    if (params.isGemmOperation())
    {
        unsigned cd = nodeDims[0];
        unsigned width = nodeDims[1];
        unsigned height = nodeDims[2];
        unsigned batch = nodeDims[3];
        if (operation == MmeCommon::e_mme_ab || operation == MmeCommon::e_mme_abt)
            params.x.sizes = {cd, height, batch, 1, 1};
        else
            params.x.sizes = {height, cd, batch, 1, 1};
        if (operation == MmeCommon::e_mme_ab || operation == MmeCommon::e_mme_atb)
            params.w.sizes = {width, cd, broadcastB ? 1 : batch, 1, 1};
        else
            params.w.sizes = {cd, width, broadcastB ? 1 : batch, 1, 1};
        params.y.sizes = {width, height, batch, 1, 1};
        params.x.elementType = params.w.elementType = params.y.elementType = dataType;
    }
    else
    {
        unsigned dimC = nodeDims[Conv::DIM_IN_CHANNELS];
        unsigned dimW = nodeDims[Conv::DIM_WIDTH];
        unsigned dimH = nodeDims[Conv::DIM_HEIGHT];
        unsigned dimD = nodeDims[Conv::DIM_DEPTH];
        unsigned dimB = nodeDims[Conv::DIM_BATCH];
        unsigned dimK = nodeDims[Conv::DIM_OUT_CHANNELS];
        unsigned dimQ = nodeDims[Conv::DIM_FILTER_Q];
        unsigned dimR = nodeDims[Conv::DIM_FILTER_R];
        unsigned dimS = nodeDims[Conv::DIM_FILTER_S];
        params.x.sizes = {dimC, dimW, dimH, dimD, dimB};
        params.y.sizes = {dimK, dimW, dimH, dimD, dimB};
        params.w.sizes = {dimK, dimC, dimS, dimR, dimQ};
    }

    for (int dim = 1; dim < MAX_DIMENSION; dim++)
    {
        params.x.strides[dim] = params.x.strides[dim - 1] * params.x.sizes[dim - 1];
        params.w.strides[dim] = params.w.strides[dim - 1] * params.w.sizes[dim - 1];
        params.y.strides[dim] = params.y.strides[dim - 1] * params.y.sizes[dim - 1];
    }
    //  turn on optimizations
    params.strategy.sbReuse = true;
    params.strategy.flattenEn = true;
    params.strategy.batchConcurrencyEn = MmeCommon::TurnedOn;

    //  set geometry and pattern to be set by brain
    params.strategy.geometry = e_mme_geometry_nr;
    params.strategy.pattern = e_mme_patterns_nr;

    return params;
}

void compareSize(unsigned origSize, unsigned geometrySize, unsigned granularity, unsigned solution)
{
    if (granularity > geometrySize)
    {
        // granularity is larger than the geometry itself, work on the minimum granule of 1
        ASSERT_EQ(solution, 1);
    }
    else if (origSize < geometrySize)
    {
        // need the whole dimension, div round up with the granularity.
        // need to round up in case the size is not divisible by the granularity to make sure we also take the last
        // elements
        ASSERT_EQ(solution, div_round_up(origSize, granularity));
    }
    else
    {
        // need part of the dimensions, div round down
        // need ot round down to make sure we dont exceed geometry size
        ASSERT_EQ(solution, div_round_down(geometrySize, granularity));
    }
}

void MmeUTBrainTest::BrainSolutionTest(MmeSolutionParams testParams)
{
    unsigned cd, width, height, batch;
    auto params = setParams(testParams.nodeDims, testParams.opType, testParams.dataType, testParams.broadcastB);
    bool isGemmOp = params.isGemmOperation();
    bool isDedw = testParams.opType == e_mme_dedw;
    bool isDedx = testParams.opType == e_mme_dedx;
    MmeBrainKnobs knobs;
    knobs.operationModes = {false, false, true /*enable concurrency opt*/};
    MmeCommon::MmeBrain brain(MmeCommon::e_mme_Gaudi3);
    brain.setBrainKnobs(knobs);
    MultiplierArray previousMultiplier, granularity;
    if (isGemmOp)
    {
        previousMultiplier.resize(Gemm::MAX_INDEX_SPACE_DIM, 1);
        granularity.resize(Gemm::MAX_INDEX_SPACE_DIM, 1);

        cd = testParams.nodeDims[0];
        width = testParams.nodeDims[1];
        height = testParams.nodeDims[2];
        batch = testParams.nodeDims[3];

        granularity.at(Gemm::DIM_OPERANDS_COMMON) = testParams.granularity[0];
        granularity.at(Gemm::DIM_OUT_FCD) = testParams.granularity[1];
        granularity.at(Gemm::DIM_OUT_HEIGHT) = testParams.granularity[2];
        granularity.at(Gemm::DIM_BATCH_0) = testParams.granularity[3];
        granularity.at(Gemm::DIM_BATCH_1) = 1;
    }
    else if (isDedw)
    {
        previousMultiplier.resize(Conv::MAX_INDEX_SPACE_DIM, 1);
        granularity.resize(Conv::MAX_INDEX_SPACE_DIM, 1);
        width = testParams.nodeDims[Conv::DIM_OUT_CHANNELS];
        height = testParams.nodeDims[Conv::DIM_IN_CHANNELS];
        batch = -1;  // no non-spatial batch dim in conv operations
        granularity[Conv::DIM_FILTER_R] = testParams.nodeDims[Conv::DIM_FILTER_R];
        granularity[Conv::DIM_FILTER_S] = testParams.nodeDims[Conv::DIM_FILTER_S];
        granularity[Conv::DIM_FILTER_Q] = testParams.nodeDims[Conv::DIM_FILTER_Q];
        granularity[Conv::DIM_WIDTH] = testParams.nodeDims[Conv::DIM_WIDTH];
        granularity[Conv::DIM_HEIGHT] = testParams.nodeDims[Conv::DIM_HEIGHT];
        granularity[Conv::DIM_DEPTH] = testParams.nodeDims[Conv::DIM_DEPTH];
        // only dimensions that support actual granularity
        granularity[Conv::DIM_BATCH] = testParams.granularity[0];
        granularity[Conv::DIM_OUT_CHANNELS] = testParams.granularity[1];
        granularity[Conv::DIM_IN_CHANNELS] = testParams.granularity[2];
    }
    else
    {
        MME_ASSERT(params.isFwdOrDedx(), "unexpected opType");
        previousMultiplier.resize(Conv::MAX_INDEX_SPACE_DIM, 1);
        granularity.resize(Conv::MAX_INDEX_SPACE_DIM, 1);
        int cdIndex = isDedx ? Conv::DIM_OUT_CHANNELS : Conv::DIM_IN_CHANNELS;
        int fcdIndex = isDedx ? Conv::DIM_IN_CHANNELS : Conv::DIM_OUT_CHANNELS;
        cd = testParams.nodeDims[cdIndex];
        width = testParams.nodeDims[fcdIndex];
        height = testParams.nodeDims[Conv::DIM_BATCH];
        batch = -1;  // no non-spatial batch dim in conv operations
        granularity[Conv::DIM_FILTER_R] = testParams.nodeDims[Conv::DIM_FILTER_R];
        granularity[Conv::DIM_FILTER_S] = testParams.nodeDims[Conv::DIM_FILTER_S];
        granularity[Conv::DIM_FILTER_Q] = testParams.nodeDims[Conv::DIM_FILTER_Q];
        granularity[Conv::DIM_WIDTH] = testParams.nodeDims[Conv::DIM_WIDTH];
        granularity[Conv::DIM_HEIGHT] = testParams.nodeDims[Conv::DIM_HEIGHT];
        granularity[Conv::DIM_DEPTH] = testParams.nodeDims[Conv::DIM_DEPTH];
        // only dimensions that support actual granularity
        if (isDedx)
        {
            granularity[Conv::DIM_OUT_CHANNELS] = testParams.granularity[0];
            granularity[Conv::DIM_IN_CHANNELS] = testParams.granularity[1];
        }
        else
        {
            granularity[Conv::DIM_IN_CHANNELS] = testParams.granularity[0];
            granularity[Conv::DIM_OUT_CHANNELS] = testParams.granularity[1];
        }
        granularity[Conv::DIM_BATCH] = testParams.granularity[2];
    }
    MmeBrainSolutionContainer solutions = brain.getMmeSolutions(params, granularity, previousMultiplier);
    int widthNodeDim = isGemmOp ? (int) Gemm::DIM_OUT_FCD : isDedx ? Conv::DIM_IN_CHANNELS : Conv::DIM_OUT_CHANNELS;
    for (auto& solution : solutions)
    {
        params.strategy = solution->strategy;
        auto geoAttr = MmeCommon::getGeoAttr(MmeCommon::e_mme_Gaudi3, params);

        unsigned widthSize = std::min(width, geoAttr->getGeometryWidth());
        unsigned heightSize = std::min(height, geoAttr->getGeometryHeight());

        compareSize(width,
                    geoAttr->getGeometryWidth(),
                    granularity[widthNodeDim],
                    solution->solutionDimMultipliers[widthNodeDim]);
        if (isGemmOp)
        {
            if (solution->requirements.cdSliced)
            {
                ASSERT_EQ(solution->solutionDimMultipliers[Gemm::DIM_OPERANDS_COMMON],
                          div_round_up(knobs.minCd, granularity[Gemm::DIM_OPERANDS_COMMON]));
            }
            else
            {
                ASSERT_EQ(solution->solutionDimMultipliers[Gemm::DIM_OPERANDS_COMMON],
                          div_round_up(params.getSingleGemmCD(), granularity[Gemm::DIM_OPERANDS_COMMON]));
            }

            compareSize(height,
                        geoAttr->getGeometryHeight(),
                        granularity[Gemm::DIM_OUT_HEIGHT],
                        solution->solutionDimMultipliers[Gemm::DIM_OUT_HEIGHT]);

            if (testParams.broadcastB && heightSize < geoAttr->getGeometryHeight())
            {
                unsigned geometryBatchGranules =
                    div_round_down(div_round_down(geoAttr->getGeometryHeight(), heightSize),
                                   granularity[Gemm::DIM_BATCH_0]);
                unsigned totalBatchGranules = div_round_up(batch, granularity[Gemm::DIM_BATCH_0]);
                unsigned batchGranules = std::max(std::min(geometryBatchGranules, totalBatchGranules), (unsigned) 1);
                ASSERT_EQ(solution->solutionDimMultipliers[Gemm::DIM_BATCH_0], batchGranules);
            }
            else
            {
                compareSize(batch,
                            geoAttr->getGeometryConcurrency(),
                            granularity[Gemm::DIM_BATCH_0],
                            solution->solutionDimMultipliers[Gemm::DIM_BATCH_0]);
            }
        }
        else if (isDedw)
        {
            compareSize(height,
                        geoAttr->getGeometryHeight(),
                        granularity[Conv::DIM_IN_CHANNELS],
                        solution->solutionDimMultipliers[Conv::DIM_IN_CHANNELS]);

            if (solution->requirements.cdSliced)
            {
                unsigned CDGranularity = multiplyElements(&granularity[Conv::DIM_DEPTH], &granularity[Conv::DIM_WIDTH+1]);
                unsigned actualMinCd = div_round_up(knobs.minCd, CDGranularity);
                ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_BATCH],
                          div_round_up(actualMinCd, granularity[Conv::DIM_BATCH]));
            }
            else
            {
                ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_BATCH],
                          div_round_up(params.y.sizes[DIM_B], granularity[Conv::DIM_BATCH]));
            }
        }
        else
        {
            unsigned basicSpatialSizeGranularity = multiplyElements(&granularity[Conv::DIM_DEPTH], &granularity[Conv::DIM_WIDTH]);
            unsigned effectiveGeometry = div_round_down(geoAttr->getGeometryHeight(), basicSpatialSizeGranularity);
            compareSize(height,
                        effectiveGeometry,
                        granularity[Conv::DIM_BATCH],
                        solution->solutionDimMultipliers[Conv::DIM_BATCH]);

            int cdNodeDim = isDedx ? Conv::DIM_OUT_CHANNELS : Conv::DIM_IN_CHANNELS;
            if (solution->requirements.cdSliced)
            {
                unsigned cdMultiples = params.getCDSize() / params.getOperand(e_mme_op_a).sizes[0];
                unsigned actualMinCd = div_round_up(knobs.minCd, cdMultiples);
                ASSERT_EQ(solution->solutionDimMultipliers[cdNodeDim],
                          div_round_up(actualMinCd, granularity[cdNodeDim]));
            }
            else
            {
                ASSERT_EQ(solution->solutionDimMultipliers[cdNodeDim],
                          div_round_up(params.getOperand(e_mme_op_a).sizes[0], granularity[cdNodeDim]));
            }
        }

        if (!isGemmOp)
        {
            // verify the all=required fields
            ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_FILTER_R], 1);
            ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_FILTER_S], 1);
            ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_FILTER_Q], 1);
            ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_WIDTH], 1);
            ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_HEIGHT], 1);
            ASSERT_EQ(solution->solutionDimMultipliers[Conv::DIM_DEPTH], 1);
        }
    }
}

class MmeUTBrainSolutionParametricTest
: public MmeUTBrainTest
, public testing::WithParamInterface<MmeSolutionParams>
{
};

TEST_P(MmeUTBrainSolutionParametricTest, brain_solution_param)
{
    BrainSolutionTest(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    BrainSolutionBgemmTests,
    MmeUTBrainSolutionParametricTest,
    ::testing::Values(
        // basic
        MmeSolutionParams {{128, 128, 128, 128}, {1, 1, 1, 1}},  // basic small
        MmeSolutionParams {{2048, 2048, 2048, 4}, {1, 1, 1, 1}},  // basic large
        MmeSolutionParams {{2048, 4096, 4096, 12}, {1, 2048, 2048, 1}},  // basic large, at least 2 geometries
        MmeSolutionParams {{1024, 256, 256, 16}, {2, 2, 2, 2}},  // basic granularity
        MmeSolutionParams {{128, 128, 128, 8},
                           {128, 128, 128, 4}},  // granularity edge case - equal to sizes, small batch
        MmeSolutionParams {{128, 128, 128, 8},
                           {128, 128, 128, 8}},  // granularity edge case - equal to sizes, full batch
        MmeSolutionParams {{128, 128, 128, 128}, {128, 128, 128, 128}},  // granularity edge case - equal to sizes
        MmeSolutionParams {{128, 128, 128, 128},
                           {128, 128, 128, 128},
                           true},  // granularity edge case - equal to sizes, broadcast
        MmeSolutionParams {{1024, 1024, 128, 64},
                           {1024, 1024, 128, 64},
                           true},  // granularity edge case - equal to sizes, large
        MmeSolutionParams {{4147, 4147, 4147, 16},
                           {3, 3, 3, 3}},  // sizes not divisible by granularity, larger than geoemtry
        MmeSolutionParams {{20, 31, 28, 8}, {3, 3, 3, 3}},  // sizes not divisible by granularity, smaller than geometry
        MmeSolutionParams {{20, 31, 28, 40},
                           {3, 3, 3, 3}},  // sizes not divisible by granularity, smaller than geometry, large batch
        MmeSolutionParams {{20, 31, 28, 40},
                           {3, 3, 3, 3},
                           true},  // sizes not divisible by granularity, smaller than geometry, large batch, broadcast
        MmeSolutionParams {{256, 768, 512, 13}, {1, 1, 1, 1}, true},  // basic broadcast test
        MmeSolutionParams {{256, 768, 3, 13}, {1, 1, 1, 1}, true},  // basic broadcast test, fits in geometry
        MmeSolutionParams {{256, 1024, 3, 28}, {256, 1024, 3, 4}, true},  // broadcast test, full granularity sizes
        MmeSolutionParams {{256, 768, 49, 128}, {256, 768, 49, 4}, true}
        // broadcast test, full granularity sizes, large batch
        ));

INSTANTIATE_TEST_CASE_P(BrainSolutionConvTests,
                        MmeUTBrainSolutionParametricTest,
                        ::testing::Values(
                            // basic
                            MmeSolutionParams {{4, 8, 4, 1, 256, 256, 1, 1, 1}, {1, 1, 1}, false, e_mme_fwd},
                            MmeSolutionParams {{100, 7, 7, 1, 20, 20, 1, 1, 1}, {1, 1, 1}, false, e_mme_fwd},
                            MmeSolutionParams {{100, 13, 13, 1, 30, 1000, 1, 1, 1}, {5, 5, 5}, false, e_mme_fwd},
                            MmeSolutionParams {{4, 8, 4, 1, 256, 256, 1, 1, 1}, {1, 1, 1}, false, e_mme_dedx},
                            MmeSolutionParams {{100, 7, 7, 1, 20, 20, 1, 1, 1}, {1, 1, 1}, false, e_mme_dedx},
                            MmeSolutionParams {{100, 13, 13, 1, 1000, 30, 1, 1, 1}, {5, 5, 5}, false, e_mme_dedx}));

INSTANTIATE_TEST_CASE_P(
    BrainSolutionCdSplitTests,
    MmeUTBrainSolutionParametricTest,
    ::testing::Values(
        // basic
        MmeSolutionParams {{4196, 1024, 1024, 16}, {1, 1, 1, 1}},  // bgemm simple
        MmeSolutionParams {{4196, 128, 128, 16}, {1, 1, 1, 1}},  // bgemm tiny
        MmeSolutionParams {{4, 8, 4, 1, 4196, 256, 1, 1, 1}, {1, 1, 1}, false, e_mme_fwd},  // fwd simple
        MmeSolutionParams {{4, 8, 4, 1, 1024, 256, 1, 1, 3},
                           {1, 1, 1},
                           false,
                           e_mme_fwd},  // fwd C not large enough, overall CD is
        MmeSolutionParams {{4, 8, 4, 1, 256, 1024, 1, 1, 3}, {1, 1, 1}, false, e_mme_dedx},
        // dedx K not large enough, overall CD is
        MmeSolutionParams {{50, 100, 20, 1, 1024, 1024, 1, 1, 1}, {1, 1, 1}, false, e_mme_dedw},  // dedw simple
        MmeSolutionParams {{50, 35, 37, 1, 1024, 1024, 1, 1, 1},
                           {3, 1, 1},
                           false,
                           e_mme_dedw},  // dedw complex CD, granularity
        MmeSolutionParams {{50, 35, 37, 1, 1024, 1024, 1, 1, 1}, {3, 1, 1}, false, e_mme_dedw},  // dedw complex CD
        MmeSolutionParams {{50, 100, 20, 1, 20, 20, 1, 1, 1}, {1, 1, 1}, false, e_mme_dedw}  // dedw tiny (concurrency)
        ));

void MmeUTBrainTest::BrainBwTest(MmeBwTestParams testParams)
{
    unsigned cd = testParams.nodeDims[0];
    unsigned width = testParams.nodeDims[1];
    unsigned height = testParams.nodeDims[2];
    unsigned batch = testParams.nodeDims[3];
    auto params = setParams(testParams.nodeDims, testParams.opType, MmeCommon::e_type_bf16, false);
    params.strategy.geometry = e_mme_geometry_2xh;
    params.strategy.pattern = MmeCommon::e_mme_sp_reduction_fck;
    MmeCommon::MmeBrain brain(MmeCommon::e_mme_Gaudi3);
    MultiplierArray previousMultiplier(Gemm::MAX_INDEX_SPACE_DIM, 1);
    MultiplierArray granularity(Gemm::MAX_INDEX_SPACE_DIM, 1);
    MmeBrainSolutionContainer solutions = brain.getMmeSolutions(params, granularity, previousMultiplier);

    for (auto& solution : solutions)
    {
        // for simplicity only check the no opt solution
        if (solution->solutionType != NO_OPT) continue;

        PerfAttr perfAttr;
        brain.setParamsToSolutionSize(params, solution->solutionDimMultipliers, granularity);
        if (solution->requirements.bwInflationDim.has_value())
        {
            MME_ASSERT(testParams.opType == e_mme_ab, "code below is only valid for AB");
            switch (solution->requirements.bwInflationDim.value())
            {
                case Gemm::DIM_OUT_FCD:
                    params.w.sizes[GEMM_DIM_W] = width;
                    params.y.sizes[GEMM_DIM_W] = width;
                    break;
                case Gemm::DIM_OUT_HEIGHT:
                    params.x.sizes[GEMM_DIM_H] = height;
                    params.y.sizes[GEMM_DIM_H] = height;
                    break;
                default:
                    MME_ASSERT(0, "invalid inflation dim");
            }
            brain.getPerfAttr(params, perfAttr);
        }
        else
        {
            perfAttr = solution->perfAttr;
        }
        ASSERT_EQ(perfAttr.memoryAttrA.accessBW, testParams.accessBwA);
        ASSERT_EQ(perfAttr.memoryAttrB.accessBW, testParams.accessBwB);
        ASSERT_EQ(perfAttr.memoryAttrA.accessesPerChip, testParams.fetchNrA);
        ASSERT_EQ(perfAttr.memoryAttrB.accessesPerChip, testParams.fetchNrB);
    }
}

class MmeUTBrainBwParametricTest
: public MmeUTBrainTest
, public testing::WithParamInterface<MmeBwTestParams>
{
};

TEST_P(MmeUTBrainBwParametricTest, brain_BW_param)
{
    BrainBwTest(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    BrainBwTests,
    MmeUTBrainBwParametricTest,
    ::testing::Values(
        // basic
        MmeBwTestParams {{128, 128, 128, 8}, MmeCommon::e_mme_ab, 200 * 32, 200 * 32, 1, 1},  // no movement
        MmeBwTestParams {{128, 2000, 2000, 8}, MmeCommon::e_mme_ab, 50 * 32, 200 * 32, 2, 4},  // 4 steps right, 1 step
                                                                                               // down
        MmeBwTestParams {{128, 4000, 2000, 8}, MmeCommon::e_mme_ab, 25 * 32, 200 * 32, 2, 4},  // 8 steps right, 1 step
                                                                                               // down
        MmeBwTestParams {{3280, 4000, 2000, 8}, MmeCommon::e_mme_ab, 50 * 32, 200 * 32, 4, 4}  // same but with partials
        ));

void MmeUTBrainTest::BrainUtilizaitonTest(MmeUtilzationTestParams testParams)
{
    unsigned cd = testParams.nodeDims[0];
    unsigned width = testParams.nodeDims[1];
    unsigned height = testParams.nodeDims[2];
    unsigned batch = testParams.nodeDims[3];
    auto params = setParams(testParams.nodeDims, e_mme_ab, MmeCommon::e_type_bf16, false);
    params.strategy.geometry = e_mme_geometry_2xh;
    params.strategy.pattern = MmeCommon::e_mme_sp_reduction_fck;
    MmeCommon::MmeBrain brain(MmeCommon::e_mme_Gaudi3);
    MultiplierArray previousMultiplier = {1, 1, 1, 1, 1, 1, 1, 1, 1}, granularity = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    granularity[Gemm::DIM_OPERANDS_COMMON] = testParams.granularity[0];
    granularity[Gemm::DIM_OUT_FCD] = testParams.granularity[1];
    granularity[Gemm::DIM_OUT_HEIGHT] = testParams.granularity[2];
    granularity[Gemm::DIM_BATCH_0] = testParams.granularity[3];

    MmeBrainSolutionContainer solutions = brain.getMmeSolutions(params, granularity, previousMultiplier);
    for (auto& solution : solutions)
    {
        // for simplicity only check the no opt solution
        if (solution->solutionType != NO_OPT) continue;

        ASSERT_LT(std::abs(testParams.maxUtilization - solution->perfAttr.maxUtilization), EPSILON);
        ASSERT_LT(std::abs(testParams.mmeUtilization - solution->perfAttr.mmeUtilization), EPSILON);
    }
}

class MmeUTBrainUtilizationParametricTest
: public MmeUTBrainTest
, public testing::WithParamInterface<MmeUtilzationTestParams>
{
};

TEST_P(MmeUTBrainUtilizationParametricTest, brain_utilization_param)
{
    BrainUtilizaitonTest(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    BrainSolutionUtilizationTest,
    MmeUTBrainUtilizationParametricTest,
    ::testing::Values(
        // basic
        MmeUtilzationTestParams {{256, 1024, 1024, 16}, {1, 1, 1, 1}, 1.0, 1.0},  // sanity
        MmeUtilzationTestParams {{256, 1024, 1024, 16},
                                 {1, 256, 256, 4},
                                 1.0,
                                 1.0},  // granularity multiple of geometry
        MmeUtilzationTestParams {{256, 1000, 1000, 16},
                                 {1, 256, 256, 4},
                                 (125 * 125) / (128 * 128.0),
                                 (125 * 125) / (128 * 128.0)},  // granularity multiple of geometry, not full sizes
        MmeUtilzationTestParams {{256, 800, 1000, 16},
                                 {1, 200, 200, 4},
                                 (125 * 25) / (128 * 32.0),
                                 (125 * 25) / (128 * 32.0)},  // sizes are multiple of granularity
        MmeUtilzationTestParams {
            {256, 1000, 1000, 16},
            {1, 200, 200, 4},
            (125 * 125) / (128 * 128.0),
            ((125 * 125) / (128 * 128.0)) * 2.0 /
                3},  // sizes are multiple of granularity, but under utilization due ot granularity causes extra step
        MmeUtilzationTestParams {{256, 1024, 1024, 16}, {1, 200, 1, 1}, 1.0, 2.0 / 3},  // width, 3 steps instead of 2
        MmeUtilzationTestParams {{256, 1024, 2048, 16}, {1, 1, 100, 1}, 1.0, 2.0 / 3},  // height, 3 steps instead of 2
        MmeUtilzationTestParams {{256, 256, 256, 16}, {1, 1, 1, 5}, 1.0, 17.0 / 32},
        MmeUtilzationTestParams {{256, 1024, 2048, 16},
                                 {1, 200, 100, 1},
                                 1.0,
                                 4.0 / 9},  // height and width, 9 steps instead of 4
        MmeUtilzationTestParams {{256, 1020, 2040, 16},
                                 {1, 200, 100, 1},
                                 (255 * 255) / (256 * 256.0),
                                 ((255 * 255) / (256 * 256.0)) * (4.0 / 9)}  // height and width, 9 steps instead of 4
        ));

void MmeUTBrainTest::BrainRedundantStrategiesTest(MmeSolutionParams testParams)
{
    auto params = setParams(testParams.nodeDims, testParams.opType, testParams.dataType, testParams.broadcastB);
    params.strategy.cdConcurrencyEn = MmeCommon::TurnedOn;
    params.strategy.batchConcurrencyEn = MmeCommon::TurnedOn;
    MmeCommon::MmeBrain brain(MmeCommon::e_mme_Gaudi3, {false, false, true /*enable concurrency opt solutions*/});
    MultiplierArray previousMultiplier(Gemm::MAX_INDEX_SPACE_DIM, 1);
    testParams.granularity.resize(Gemm::MAX_INDEX_SPACE_DIM, 1);
    MmeBrainSolutionContainer solutions = brain.getMmeSolutions(params, testParams.granularity, previousMultiplier);

    ASSERT_EQ(solutions.size(), testParams.solutionNr);
}

class MmeUTBrainSolutionNrParametricTest
: public MmeUTBrainTest
, public testing::WithParamInterface<MmeSolutionParams>
{
};

TEST_P(MmeUTBrainSolutionNrParametricTest, brain_solution_nr)
{
    BrainRedundantStrategiesTest(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    BrainSolutionNrTests,
    MmeUTBrainSolutionNrParametricTest,
    ::testing::Values(
        // basic
        MmeSolutionParams {{256, 256, 256, 8}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 1},  // single solution,
                                                                                                // dcore split
        MmeSolutionParams {{256, 512, 256, 8}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 2},  // two solution, dcore
                                                                                                // split
        MmeSolutionParams {{256, 512, 512, 8}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 2},  // two solutions, dcore
                                                                                                // split
        MmeSolutionParams {{256, 384, 384, 8}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 2},  // two solutions, dcore
                                                                                                // split
        MmeSolutionParams {{384, 384, 16, 8}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 2},  // 4 geometries, 2
                                                                                               // directions each
        MmeSolutionParams {{256, 256, 256, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 1},  // single solution, no
                                                                                                // batches
        MmeSolutionParams {{256, 512, 256, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 1},  // single solution, no
                                                                                                // batches
        MmeSolutionParams {{256, 256, 512, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 1},  // single solution, no
                                                                                                // batches
        MmeSolutionParams {{256, 512, 512, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 2},  // single solution, no
                                                                                                // batches
        MmeSolutionParams {{256, 256, 2048, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 1},  // single solution, no
                                                                                                 // batches
        MmeSolutionParams {{256, 2048, 256, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 1},  // single solution, no
                                                                                                 // batches
        MmeSolutionParams {{256, 2048, 2048, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 6},  // 4 geometry, only 2
                                                                                                  // can move freely
        MmeSolutionParams {{256, 1024, 5000, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 6},  // 4 geometry, only 2
                                                                                                  // can move freely
        MmeSolutionParams {{256, 4048, 4048, 1}, {1, 1, 1, 1}, false, e_mme_ab, e_type_bf16, 8},  // 4 geometries, 2
                                                                                                  // directions each
        MmeSolutionParams {{2560, 256, 256, 8},
                           {1, 1, 1, 1},
                           false,
                           e_mme_ab,
                           e_type_bf16,
                           2},  // 1 geometry, 2 solutions with cd split
        MmeSolutionParams {{2560, 256, 256, 8},
                           {2560, 1, 1, 1},
                           false,
                           e_mme_ab,
                           e_type_bf16,
                           1},  // 1 geometry, 1 solutions, no cd split
        MmeSolutionParams {{50, 100, 20, 5, 256, 256, 1, 1, 1},
                           {1, 1, 1, 1},
                           false,
                           e_mme_dedw,
                           e_type_bf16,
                           4},  // 4 solutions, 1 geometry, 4 concurrency optimizations
        MmeSolutionParams {{50, 100, 20, 5, 256, 256, 1, 1, 1},
                           {50, 100, 20, 1},
                           false,
                           e_mme_dedw,
                           e_type_bf16,
                           4},  // 4 solutions, 1 geometry, 4 concurrency optimizations
        MmeSolutionParams {{50, 100, 20, 5, 256, 256, 1, 1, 1}, {50, 100, 20, 5}, false, e_mme_dedw, e_type_bf16,
        2}  // 2 solutions, 1 geometry, 2 concurrency optimizations (cant perform batch concurrency)
        ));

void MmeUTBrainTest::BrainSolutionInflationTest(MmeInflationParams testParams)
{
    auto params = setParams(testParams.nodeDims, testParams.opType, testParams.dataType, testParams.broadcastB);
    // limit to a single solution to make testing simple
    params.strategy.geometry = e_mme_geometry_4xh;
    params.strategy.pattern = MmeCommon::e_mme_sp_reduction_fck;
    MmeCommon::MmeBrain brain(MmeCommon::e_mme_Gaudi3);
    MultiplierArray previousMultiplier(Gemm::MAX_INDEX_SPACE_DIM, 1);
    testParams.granularity.resize(Gemm::MAX_INDEX_SPACE_DIM, 1);

    MmeBrainSolutionContainer solutions = brain.getMmeSolutions(params, testParams.granularity, previousMultiplier);
    ASSERT_EQ(solutions.size(), 1);
    MmeBrainSolutionPtr solution = solutions.front();
    ASSERT_LT(solution->perfAttr.mmeUtilization, testParams.requiredPerf);  // make sure we have something to inflate
    MmeBrainSolutionPtr inflatedSolution = brain.inflateForUtilization(params,
                                                                       solution,
                                                                       testParams.granularity,
                                                                       PhysicalAspects::Name::OUTPUT_HEIGHT,
                                                                       testParams.requiredPerf);
    MME_ASSERT(inflatedSolution != nullptr, "failed to inflate");
    ASSERT_LT(testParams.requiredPerf, inflatedSolution->perfAttr.mmeUtilization);
}

class MmeUTBrainSolutionInflationParametricTest
: public MmeUTBrainTest
, public testing::WithParamInterface<MmeInflationParams>
{
};

TEST_P(MmeUTBrainSolutionInflationParametricTest, brain_solution_inflation)
{
    BrainSolutionInflationTest(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    BrainSolutionInflationTests,
    MmeUTBrainSolutionInflationParametricTest,
    ::testing::Values(MmeInflationParams {{256, 1, 21, 21, 256, 256, 1, 1, 1}, {1, 1, 1, 1}, 0.9, false, e_mme_fwd},
                      MmeInflationParams {{64, 1, 28, 112, 256, 256, 1, 1, 1}, {1, 1, 28, 112}, 0.9, false, e_mme_fwd},
                      MmeInflationParams {{256, 1, 17, 21, 256, 256, 1, 1, 1}, {1, 1, 1, 1}, 0.9, false, e_mme_dedx},
                      MmeInflationParams {{256, 1024, 768, 28}, {1, 1, 1, 1}, 0.8, true, MmeCommon::e_mme_ab},
                      MmeInflationParams {{256, 1024, 300, 39}, {1, 1, 1, 1}, 0.9, true, MmeCommon::e_mme_abt}));
