#include "../gaudi_tests/syn_gaudi_two_run_compare_test.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "../gaudi_tests/dynamic_shapes_types.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "test_types.hpp"
#include <vector>

class SynTrainingConvNonCdSlicingTest
: public SynTrainingTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<DimSizes, DimSizes, int, int, int, unsigned>>
// ifmHeight, ifmWidth, nIFM, nOFM, filter, batch
{
public:
    SynTrainingConvNonCdSlicingTest();
    // returns the output tensor index to validate correctness
    virtual unsigned addNode() = 0;
    void             runSingleTest();
    virtual bool     blockTestForConvParams();

protected:
    DimSizes m_xHeight;
    DimSizes m_xWidth;
    unsigned m_batch;
    unsigned m_nIFM;
    unsigned m_nOFM;
    unsigned m_filter;

    synConvolutionParams m_params;

    DimSizes m_yHeight;
    DimSizes m_yWidth;

    ShapeSizes m_xSizes;
    ShapeSizes m_wSizes;
    ShapeSizes m_ySizes;

    unsigned m_tensorXIdx;
    unsigned m_tensorWIdx;
    unsigned m_tensorYIdx;
};

SynTrainingConvNonCdSlicingTest::SynTrainingConvNonCdSlicingTest()
: m_xHeight(std::get<0>(GetParam())),
  m_xWidth(std::get<1>(GetParam())),
  m_batch(std::get<5>(GetParam())),
  m_nIFM(std::get<2>(GetParam())),
  m_nOFM(std::get<3>(GetParam())),
  m_filter(std::get<4>(GetParam())),
  m_params(m_filter, m_filter, 1, 1, 0, 0, 0, 0, 1, 1)
{
    setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});

    m_yHeight.min =
        convOutputDimSize(m_xHeight.min, m_params.kW, m_params.dW, m_params.padT + m_params.padB, m_params.dilH);
    m_yHeight.max =
        convOutputDimSize(m_xHeight.max, m_params.kH, m_params.dH, m_params.padT + m_params.padB, m_params.dilH);
    m_yHeight.actual =
        convOutputDimSize(m_xHeight.actual, m_params.kH, m_params.dH, m_params.padT + m_params.padB, m_params.dilH);

    m_yWidth.min =
        convOutputDimSize(m_xWidth.min, m_params.kW, m_params.dW, m_params.padL + m_params.padR, m_params.dilW);
    m_yWidth.max =
        convOutputDimSize(m_xWidth.max, m_params.kW, m_params.dW, m_params.padL + m_params.padR, m_params.dilW);
    m_yWidth.actual =
        convOutputDimSize(m_xWidth.actual, m_params.kW, m_params.dW, m_params.padL + m_params.padR, m_params.dilW);

    m_xSizes.min    = {m_nIFM, m_xWidth.min, m_xHeight.min, m_batch};
    m_xSizes.max    = {m_nIFM, m_xWidth.max, m_xHeight.max, m_batch};
    m_xSizes.actual = {m_nIFM, m_xWidth.actual, m_xHeight.actual, m_batch};
    m_wSizes.actual = {m_nOFM, m_nIFM, m_filter, m_filter};
    m_ySizes.min    = {m_nOFM, m_yWidth.min, m_yHeight.min, m_batch};
    m_ySizes.max    = {m_nOFM, m_yWidth.max, m_yHeight.max, m_batch};
    m_ySizes.actual = {m_nOFM, m_yWidth.actual, m_yHeight.actual, m_batch};
}

bool SynTrainingConvNonCdSlicingTest::blockTestForConvParams()
{
    return false;
}

void SynTrainingConvNonCdSlicingTest::runSingleTest()
{
    if (blockTestForConvParams())
    {
        return;
    }

    unsigned             tensorToValidateIdx = addNode();
    GlobalConfTestSetter convPacking("ENABLE_CONV_PACKING_TRAINING", "false");
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "70000");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    compareRunsResults({tensorToValidateIdx});
}

class SynTrainingConvFwdNonCdSlicingTest : public SynTrainingConvNonCdSlicingTest
{
public:
    SynTrainingConvFwdNonCdSlicingTest() : SynTrainingConvNonCdSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynTrainingConvFwdNonCdSlicingTest::addNode()
{
    unsigned graphIndex = 0;

    m_tensorXIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_xSizes.max.data(),
                                       m_xSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "X",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_xSizes.min.data());

    m_tensorWIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_wSizes.actual.data(),
                                       m_wSizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "W");

    m_tensorYIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_ySizes.max.data(),
                                       m_ySizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "Y",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_ySizes.min.data());

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {m_tensorXIdx, m_tensorWIdx},
                   {m_tensorYIdx},
                   &m_params,
                   sizeof(m_params),
                   "conv",
                   graphIndex);
    return m_tensorYIdx;
}

TEST_P_GC(SynTrainingConvFwdNonCdSlicingTest, single_conv_fwd_ASIC_CI)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingConvFwdNonCdSlicingTest,
                         // ifmHeight, ifmWidth, nIFM, nOFM, filter, batch
                         ::testing::Values(std::make_tuple(DimSizes(10), DimSizes(10), 10, 100, 3, 20),
                                           std::make_tuple(DimSizes(5), DimSizes(10), 10, 100, 3, 20),
                                           std::make_tuple(DimSizes(32), DimSizes(32), 3, 64, 7, 64),
                                           std::make_tuple(DimSizes(40), DimSizes(15), 10, 100, 3, 128),
                                           std::make_tuple(DimSizes(12), DimSizes(64), 10, 100, 5, 55)));

class SynTrainingConvDedxNonCdSlicingTest : public SynTrainingConvFwdNonCdSlicingTest
{
public:
    SynTrainingConvDedxNonCdSlicingTest() : SynTrainingConvFwdNonCdSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynTrainingConvDedxNonCdSlicingTest::addNode()
{
    m_tensorYIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_ySizes.actual.data(),
                                       m_ySizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "dY");

    m_tensorWIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_wSizes.actual.data(),
                                       m_wSizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "W");

    m_tensorXIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_xSizes.actual.data(),
                                       m_xSizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "dX");

    addNodeToGraph(NodeFactory::deDxNodeTypeName,
                   {m_tensorYIdx, m_tensorWIdx},
                   {m_tensorXIdx},
                   &m_params,
                   sizeof(m_params),
                   "dedx");
    return m_tensorXIdx;
}

TEST_P_GC(SynTrainingConvDedxNonCdSlicingTest, single_dedx_ASIC_CI)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingConvDedxNonCdSlicingTest,
                         // ifmHeight, ifmWidth, nIFM, nOFM, filter, batch
                         ::testing::Values(std::make_tuple(DimSizes(10), DimSizes(10), 10, 10, 3, 20),
                                           std::make_tuple(DimSizes(32), DimSizes(32), 4, 3, 7, 64),
                                           std::make_tuple(DimSizes(40), DimSizes(15), 10, 8, 3, 128),
                                           std::make_tuple(DimSizes(50), DimSizes(80), 10, 5, 5, 100)));
class SynTrainingConvDedwCdSlicingTest : public SynTrainingConvFwdNonCdSlicingTest
{
public:
    SynTrainingConvDedwCdSlicingTest() : SynTrainingConvFwdNonCdSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynTrainingConvDedwCdSlicingTest::addNode()
{
    m_tensorYIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_ySizes.max.data(),
                                       m_ySizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "dY",
                                       0,
                                       0,
                                       nullptr,
                                       m_ySizes.min.data());
    m_tensorXIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_xSizes.max.data(),
                                       m_xSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "X",
                                       0,
                                       0,
                                       nullptr,
                                       m_xSizes.min.data());
    m_tensorWIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_wSizes.actual.data(),
                                       m_wSizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "dW");

    addNodeToGraph(NodeFactory::deDwNodeTypeName,
                   {m_tensorYIdx, m_tensorXIdx},
                   {m_tensorWIdx},
                   &m_params,
                   sizeof(m_params),
                   "dedw");
    return m_tensorWIdx;
}

TEST_P_GC(SynTrainingConvDedwCdSlicingTest, single_dedw_ASIC_CI)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingConvDedwCdSlicingTest,
                         // ifmHeight, ifmWidth, nIFM, nOFM, filter, batch
                         ::testing::Values(std::make_tuple(DimSizes(30), DimSizes(20), 10, 20, 3, 30),
                                           std::make_tuple(DimSizes(10), DimSizes(10), 10, 26, 3, 20),
                                           std::make_tuple(DimSizes(32), DimSizes(32), 4, 3, 7, 64),
                                           std::make_tuple(DimSizes(40), DimSizes(15), 10, 8, 3, 128),
                                           std::make_tuple(DimSizes(20), DimSizes(80), 10, 5, 5, 100)));

class SynTrainingGemmSlicingTest
: public SynTrainingTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<DimSizes, DimSizes, DimSizes, DimSizes>>
// heightA, commonDim, widthB, batch
{
public:
    SynTrainingGemmSlicingTest();
    // returns the output tensor index to validate correctness
    virtual unsigned addNode() = 0;
    void             runSingleTest();

protected:
    DimSizes m_heightA;
    DimSizes m_commonDim;
    DimSizes m_widthB;
    DimSizes m_batch;

    ShapeSizes m_aSizes;
    ShapeSizes m_bSizes;
    ShapeSizes m_cSizes;

    unsigned m_tensorAIdx;
    unsigned m_tensorBIdx;
    unsigned m_tensorCIdx;
};

SynTrainingGemmSlicingTest::SynTrainingGemmSlicingTest()
: m_heightA(std::get<0>(GetParam())),
  m_commonDim(std::get<1>(GetParam())),
  m_widthB(std::get<2>(GetParam())),
  m_batch(std::get<3>(GetParam()))
{
    setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});

    m_aSizes.min    = {m_commonDim.min, m_heightA.min /*, m_batch.min*/};
    m_aSizes.max    = {m_commonDim.max, m_heightA.max /*, m_batch.max*/};
    m_aSizes.actual = {m_commonDim.actual, m_heightA.actual /*, m_batch.actual*/};

    m_bSizes.min    = {m_widthB.min, m_commonDim.min /*, m_batch.min*/};
    m_bSizes.max    = {m_widthB.max, m_commonDim.max /*, m_batch.max*/};
    m_bSizes.actual = {m_widthB.actual, m_commonDim.actual /*, m_batch.actual*/};

    m_cSizes.min    = {m_widthB.min, m_heightA.min /*, m_batch.min*/};
    m_cSizes.max    = {m_widthB.max, m_heightA.max /*, m_batch.max*/};
    m_cSizes.actual = {m_widthB.actual, m_heightA.actual /*, m_batch.actual*/};
    if (m_batch.max > 1)
    {
        m_aSizes.min.push_back(m_batch.min);
        m_aSizes.max.push_back(m_batch.max);
        m_aSizes.actual.push_back(m_batch.actual);

        m_bSizes.min.push_back(m_batch.min);
        m_bSizes.max.push_back(m_batch.max);
        m_bSizes.actual.push_back(m_batch.actual);

        m_cSizes.min.push_back(m_batch.min);
        m_cSizes.max.push_back(m_batch.max);
        m_cSizes.actual.push_back(m_batch.actual);
    }
}

void SynTrainingGemmSlicingTest::runSingleTest()
{
    unsigned tensorToValidateIdx = addNode();
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "18874368");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({tensorToValidateIdx});
}

class SynTrainingGemmFwdSlicingTest : public SynTrainingGemmSlicingTest
{
public:
    SynTrainingGemmFwdSlicingTest() : SynTrainingGemmSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynTrainingGemmFwdSlicingTest::addNode()
{
    unsigned      graphIndex = 0;
    synGEMMParams params;

    m_tensorAIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_aSizes.max.data(),
                                       m_aSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "A",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_aSizes.min.data());

    m_tensorBIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_bSizes.max.data(),
                                       m_bSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "B",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_bSizes.min.data());

    m_tensorCIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_cSizes.max.data(),
                                       m_cSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "C",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_cSizes.min.data());

    if (m_batch.max > 1)
    {
        addNodeToGraph(NodeFactory::batchGemmNodeTypeName,
                       {m_tensorAIdx, m_tensorBIdx},
                       {m_tensorCIdx},
                       &params,
                       sizeof(params),
                       "BGEMM",
                       graphIndex);
    }
    else
    {
        addNodeToGraph(NodeFactory::gemmNodeTypeName,
                       {m_tensorAIdx, m_tensorBIdx},
                       {m_tensorCIdx},
                       &params,
                       sizeof(params),
                       "GEMM",
                       graphIndex);
    }
    return m_tensorCIdx;
}

TEST_P_GC(SynTrainingGemmFwdSlicingTest, slicing_gemm)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    big_common_dim_gemm_single_ASIC_CI,
    SynTrainingGemmFwdSlicingTest,
    ::testing::Values(std::make_tuple(DimSizes(2048), DimSizes(186368), DimSizes(4), DimSizes(1)),
                      std::make_tuple(DimSizes(4), DimSizes(186368), DimSizes(2048), DimSizes(1)),
                      std::make_tuple(DimSizes(100), DimSizes(186368), DimSizes(2000), DimSizes(1))));

class SynTrainingGemmDedwSlicingTest : public SynTrainingGemmSlicingTest
{
public:
    SynTrainingGemmDedwSlicingTest() : SynTrainingGemmSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynTrainingGemmDedwSlicingTest::addNode()
{
    unsigned graphIndex = 0;

    m_tensorCIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_cSizes.max.data(),
                                       m_cSizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "dC",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_cSizes.min.data());

    m_tensorAIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_aSizes.max.data(),
                                       m_aSizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "A",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_aSizes.min.data());

    m_tensorBIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_bSizes.max.data(),
                                       m_bSizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "dB",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_bSizes.min.data());

    if (m_batch.max > 1)
    {
        addNodeToGraph(NodeFactory::batchGemmDeDwNodeTypeName,
                       {m_tensorCIdx, m_tensorAIdx},
                       {m_tensorBIdx},
                       nullptr,
                       0,
                       "bgemm_dedw",
                       graphIndex);
    }
    else
    {
        addNodeToGraph(NodeFactory::gemmDeDwNodeTypeName,
                       {m_tensorCIdx, m_tensorAIdx},
                       {m_tensorBIdx},
                       nullptr,
                       0,
                       "gemm_dedw",
                       graphIndex);
    }
    return m_tensorBIdx;
}

TEST_P_GC(SynTrainingGemmDedwSlicingTest, slicing_gemm_dedw, {synDeviceGaudi})
{
    runSingleTest();
}

// Note: This test disables SRAM mgmt, so batch gemm bwd ops are not split to single gemm bwd ops.
// This is not supported by gaudi MME, but the following sizes don't fail.
INSTANTIATE_TEST_SUITE_P(slicing_gemm_dedw_single_ASIC_CI,
                         SynTrainingGemmDedwSlicingTest,
                         ::testing::Values(std::make_tuple(DimSizes(16), DimSizes(100), DimSizes(80), DimSizes(3)),
                                           std::make_tuple(DimSizes(16), DimSizes(2047), DimSizes(1000), DimSizes(3)),
                                           std::make_tuple(DimSizes(16), DimSizes(2047), DimSizes(1000), DimSizes(1))));

class SynTrainingGemmDedxSlicingTest : public SynTrainingGemmSlicingTest
{
public:
    SynTrainingGemmDedxSlicingTest() : SynTrainingGemmSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynTrainingGemmDedxSlicingTest::addNode()
{
    unsigned graphIndex = 0;

    m_tensorCIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_cSizes.max.data(),
                                       m_cSizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "dC",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_cSizes.min.data());

    m_tensorBIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_bSizes.max.data(),
                                       m_bSizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "B",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_bSizes.min.data());

    m_tensorAIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_aSizes.max.data(),
                                       m_aSizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "dA",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_aSizes.min.data());

    if (m_batch.max > 1)
    {
        addNodeToGraph(NodeFactory::batchGemmDeDxNodeTypeName,
                       {m_tensorCIdx, m_tensorBIdx},
                       {m_tensorAIdx},
                       nullptr,
                       0,
                       "bgemm_dedx",
                       graphIndex);
    }
    else
    {
        addNodeToGraph(NodeFactory::gemmDeDxNodeTypeName,
                       {m_tensorCIdx, m_tensorBIdx},
                       {m_tensorAIdx},
                       nullptr,
                       0,
                       "gemm_dedx",
                       graphIndex);
    }
    return m_tensorAIdx;
}

TEST_P_GC(SynTrainingGemmDedxSlicingTest, slicing_gemm_dedx, {synDeviceGaudi})
{
    runSingleTest();
}

// Note: This test disables SRAM mgmt, so batch gemm bwd ops are not split to single gemm bwd ops.
// This is not supported by gaudi MME, but the following sizes don't fail.
INSTANTIATE_TEST_SUITE_P(slicing_gemm_dedx_single_ASIC_CI,
                         SynTrainingGemmDedxSlicingTest,
                         ::testing::Values(std::make_tuple(DimSizes(16), DimSizes(100), DimSizes(80), DimSizes(3)),
                                           std::make_tuple(DimSizes(16), DimSizes(2047), DimSizes(2000), DimSizes(3)),
                                           std::make_tuple(DimSizes(16), DimSizes(2047), DimSizes(1000), DimSizes(1))));

class SynTrainingGemmFwdBundleSlicingTest : public SynTrainingGemmSlicingTest
{
public:
    SynTrainingGemmFwdBundleSlicingTest() : SynTrainingGemmSlicingTest() {}
    unsigned addNode() override;

protected:
    virtual void addProducers(unsigned graphIndex);
};

unsigned SynTrainingGemmFwdBundleSlicingTest::addNode()
{
    unsigned graphIndex = 0;

    m_tensorAIdx = createTensor(INPUT_TENSOR,
                                MEM_INIT_NONE,
                                nullptr,
                                m_aSizes.max.data(),
                                m_aSizes.max.size(),
                                syn_type_bf16,
                                nullptr,
                                m_aSizes.min.data());

    m_tensorBIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_bSizes.max.data(),
                                       m_bSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "B",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_bSizes.min.data());

    m_tensorCIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_cSizes.max.data(),
                                       m_cSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "C",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_cSizes.min.data());

    if (m_batch.max > 1)
    {
        addNodeToGraph(NodeFactory::batchGemmNodeTypeName,
                       {m_tensorAIdx, m_tensorBIdx},
                       {m_tensorCIdx},
                       nullptr,
                       0,
                       "BGEMM",
                       graphIndex);
    }
    else
    {
        addNodeToGraph(NodeFactory::gemmNodeTypeName,
                       {m_tensorAIdx, m_tensorBIdx},
                       {m_tensorCIdx},
                       nullptr,
                       0,
                       "GEMM",
                       graphIndex);
    }

    addProducers(graphIndex);

    return m_tensorCIdx;
}

void SynTrainingGemmFwdBundleSlicingTest::addProducers(unsigned graphIndex)
{
    // Add relu producer
    unsigned reluA_in = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            m_aSizes.max.data(),
                                            m_aSizes.max.size(),
                                            syn_type_bf16,
                                            nullptr,
                                            "reluIn",
                                            graphIndex,
                                            0,
                                            nullptr,
                                            m_aSizes.min.data());

    addNodeToGraph("relu_fwd_bf16", {reluA_in}, {m_tensorAIdx}, nullptr, 0, "Relu", graphIndex);
}

TEST_P_GC(SynTrainingGemmFwdBundleSlicingTest, slicing_gemm_bundle)
{
    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    gemm_bundle_single_ASIC_CI,
    SynTrainingGemmFwdBundleSlicingTest,
    ::testing::Values(std::make_tuple(DimSizes(2048), DimSizes(2000), DimSizes(4000), DimSizes(1)),
                      std::make_tuple(DimSizes(1000), DimSizes(256), DimSizes(2048), DimSizes(1)),
                      std::make_tuple(DimSizes(8000), DimSizes(200), DimSizes(1000), DimSizes(1))));

// Test gemm which is too large to be sliced to fit SRAM
INSTANTIATE_TEST_SUITE_P(
    gemm_bundle_too_large_ASIC_CI,
    SynTrainingGemmFwdBundleSlicingTest,
    ::testing::Values(std::make_tuple(DimSizes(1024), DimSizes(20000), DimSizes(200), DimSizes(1))));

INSTANTIATE_TEST_SUITE_P(bgemm_bundle_single,
                         SynTrainingGemmFwdBundleSlicingTest,
                         ::testing::Values(std::make_tuple(DimSizes(384), DimSizes(64), DimSizes(384), DimSizes(24)),
                                           std::make_tuple(DimSizes(128), DimSizes(64), DimSizes(128), DimSizes(16)),
                                           std::make_tuple(DimSizes(128), DimSizes(64), DimSizes(128), DimSizes(5)),
                                           std::make_tuple(DimSizes(64), DimSizes(64), DimSizes(64), DimSizes(10))));
class SynTrainingGemmProducerChainBundlingTest : public SynTrainingGemmFwdBundleSlicingTest
{
public:
    SynTrainingGemmProducerChainBundlingTest() : SynTrainingGemmFwdBundleSlicingTest() {}

protected:
    void addProducers(unsigned graphIndex) override
    {
        ShapeSizes packedReluShape;
        packedReluShape.max.push_back(m_aSizes.max.front());
        packedReluShape.max.push_back(m_aSizes.max.back());
        packedReluShape.max.push_back(1);
        packedReluShape.max.push_back(1);
        packedReluShape.min.push_back(m_aSizes.min.front());
        packedReluShape.min.push_back(m_aSizes.min.back());
        packedReluShape.min.push_back(1);
        packedReluShape.min.push_back(1);

        ShapeSizes unpackedReluShape;
        unpackedReluShape.max.push_back(m_aSizes.max.front());
        unpackedReluShape.max.push_back(1);
        unpackedReluShape.max.push_back(1);
        unpackedReluShape.max.push_back(m_aSizes.max.back());
        unpackedReluShape.min.push_back(m_aSizes.min.front());
        unpackedReluShape.min.push_back(1);
        unpackedReluShape.min.push_back(1);
        unpackedReluShape.min.push_back(m_aSizes.min.back());

        // Add relu->unpack->reshape producer
        unsigned reluA_in = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                packedReluShape.max.data(),
                                                packedReluShape.max.size(),
                                                syn_type_bf16,
                                                nullptr,
                                                "reluIn",
                                                graphIndex,
                                                0,
                                                nullptr,
                                                packedReluShape.min.data());

        unsigned reluA_out = createTensor(INPUT_TENSOR,
                                          MEM_INIT_NONE,
                                          nullptr,
                                          packedReluShape.max.data(),
                                          packedReluShape.max.size(),
                                          syn_type_bf16,
                                          nullptr,
                                          packedReluShape.min.data());

        unsigned unpackA_out = createTensor(INPUT_TENSOR,
                                            MEM_INIT_NONE,
                                            nullptr,
                                            unpackedReluShape.max.data(),
                                            unpackedReluShape.max.size(),
                                            syn_type_bf16,
                                            nullptr,
                                            unpackedReluShape.min.data());

        addNodeToGraph("relu_fwd_bf16", {reluA_in}, {reluA_out}, nullptr, 0, "Relu", graphIndex);
        addNodeToGraph("static_reshape", {reluA_out}, {unpackA_out}, nullptr, 0, "StaticReshape", graphIndex);
        addNodeToGraph("reshape", {unpackA_out}, {m_tensorAIdx}, nullptr, 0, "Reshape", graphIndex);
    }
};

TEST_P_GC(SynTrainingGemmProducerChainBundlingTest, slicing_gemm_prod_chain_bundle)
{
    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(gemm_bundle_prod_chain_single,
                         SynTrainingGemmProducerChainBundlingTest,
                         ::testing::Values(std::make_tuple(DimSizes(2048), DimSizes(256), DimSizes(512), DimSizes(1)),
                                           std::make_tuple(DimSizes(2768), DimSizes(300), DimSizes(1024), DimSizes(1)),
                                           std::make_tuple(DimSizes(8000), DimSizes(128), DimSizes(140), DimSizes(1))));

class SynTrainingGemmSameProducerBundlingTest : public SynTrainingGemmFwdBundleSlicingTest
{
public:
    SynTrainingGemmSameProducerBundlingTest() : SynTrainingGemmFwdBundleSlicingTest() {}

protected:
    unsigned addNode() override
    {
        unsigned      graphIndex = 0;
        synGEMMParams params;

        m_tensorAIdx = createTensor(INPUT_TENSOR,
                                    MEM_INIT_NONE,
                                    nullptr,
                                    m_aSizes.max.data(),
                                    m_aSizes.max.size(),
                                    syn_type_bf16,
                                    nullptr,
                                    m_aSizes.min.data());

        m_tensorCIdx = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_cSizes.max.data(),
                                           m_cSizes.max.size(),
                                           syn_type_bf16,
                                           nullptr,
                                           "C",
                                           graphIndex,
                                           0,
                                           nullptr,
                                           m_cSizes.min.data());

        if (m_batch.max > 1)
        {
            addNodeToGraph(NodeFactory::batchGemmNodeTypeName,
                           {m_tensorAIdx, m_tensorAIdx},
                           {m_tensorCIdx},
                           &params,
                           sizeof(params),
                           "BGEMM",
                           graphIndex);
        }
        else
        {
            addNodeToGraph(NodeFactory::gemmNodeTypeName,
                           {m_tensorAIdx, m_tensorAIdx},
                           {m_tensorCIdx},
                           &params,
                           sizeof(params),
                           "GEMM",
                           graphIndex);
        }

        addProducers(graphIndex);

        return m_tensorCIdx;
    }
};

TEST_P_GC(SynTrainingGemmSameProducerBundlingTest, slicing_gemm_same_prod_bundle)
{
    // make sure the input sizes for both operands are equal, as same input is expected
    ASSERT_EQ(m_aSizes.max, m_bSizes.max);
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(gemm_bundle_same_prod,
                         SynTrainingGemmSameProducerBundlingTest,
                         ::testing::Values(std::make_tuple(DimSizes(256), DimSizes(256), DimSizes(256), DimSizes(1)),
                                           std::make_tuple(DimSizes(128), DimSizes(128), DimSizes(128), DimSizes(1))));

INSTANTIATE_TEST_SUITE_P(bgemm_bundle_same_prod,
                         SynTrainingGemmSameProducerBundlingTest,
                         ::testing::Values(std::make_tuple(DimSizes(256), DimSizes(256), DimSizes(256), DimSizes(8)),
                                           std::make_tuple(DimSizes(256), DimSizes(256), DimSizes(256), DimSizes(20)),
                                           std::make_tuple(DimSizes(128), DimSizes(128), DimSizes(128), DimSizes(12)),
                                           std::make_tuple(DimSizes(128), DimSizes(128), DimSizes(128), DimSizes(18))));

INSTANTIATE_TEST_SUITE_P(
    gemm_bgemm_bundle_same_prod_ASIC_CI,
    SynTrainingGemmSameProducerBundlingTest,
    ::testing::Values(std::make_tuple(DimSizes(2768), DimSizes(2768), DimSizes(2768), DimSizes(4)),
                      std::make_tuple(DimSizes(2768), DimSizes(2768), DimSizes(2768), DimSizes(1))));

class SynGaudi2MaskedBGemmBundlingTest : public SynTrainingGemmFwdBundleSlicingTest
{
public:
    SynGaudi2MaskedBGemmBundlingTest() : SynTrainingGemmFwdBundleSlicingTest()
    {
        // Add another batch dimension - the base class adds one and masked bgemm needs 2 batch dims
        m_aSizes.min.push_back(m_batch.min);
        m_aSizes.max.push_back(m_batch.max);
        m_aSizes.actual.push_back(m_batch.actual);

        m_bSizes.min.push_back(m_batch.min);
        m_bSizes.max.push_back(m_batch.max);
        m_bSizes.actual.push_back(m_batch.actual);

        m_cSizes.min.push_back(m_batch.min);
        m_cSizes.max.push_back(m_batch.max);
        m_cSizes.actual.push_back(m_batch.actual);

        m_maskASizes = m_aSizes;
        m_maskBSizes = m_bSizes;
        // set masks internal batch dim to 1
        m_maskASizes.min[DIM_GEMM_BATCH]    = 1;
        m_maskASizes.max[DIM_GEMM_BATCH]    = 1;
        m_maskASizes.actual[DIM_GEMM_BATCH] = 1;

        m_maskBSizes.min[DIM_GEMM_BATCH]    = 1;
        m_maskBSizes.max[DIM_GEMM_BATCH]    = 1;
        m_maskBSizes.actual[DIM_GEMM_BATCH] = 1;

        // set masks common dim to 13
        m_maskASizes.min[DIM_C]    = 13;
        m_maskASizes.max[DIM_C]    = 13;
        m_maskASizes.actual[DIM_C] = 13;

        m_maskBSizes.min[DIM_W]    = 13;
        m_maskBSizes.max[DIM_W]    = 13;
        m_maskBSizes.actual[DIM_W] = 13;
    }

protected:
    ShapeSizes m_maskASizes;
    ShapeSizes m_maskBSizes;
    unsigned   m_tensorMaskAIdx;
    unsigned   m_tensorMaskBIdx;

    unsigned addNode() override
    {
        unsigned graphIndex = 0;

        m_tensorAIdx = createTensor(INPUT_TENSOR,
                                    MEM_INIT_NONE,
                                    nullptr,
                                    m_aSizes.max.data(),
                                    m_aSizes.max.size(),
                                    syn_type_bf16,
                                    nullptr,
                                    m_aSizes.min.data());

        m_tensorBIdx = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           m_bSizes.max.data(),
                                           m_bSizes.max.size(),
                                           syn_type_bf16,
                                           nullptr,
                                           "B",
                                           graphIndex,
                                           0,
                                           nullptr,
                                           m_bSizes.min.data());

        m_tensorMaskAIdx = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               m_maskASizes.max.data(),
                                               m_maskASizes.max.size(),
                                               syn_type_bf16,
                                               nullptr,
                                               "MaskA",
                                               graphIndex,
                                               0,
                                               nullptr,
                                               m_maskASizes.min.data());

        m_tensorMaskBIdx = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               m_maskBSizes.max.data(),
                                               m_maskBSizes.max.size(),
                                               syn_type_bf16,
                                               nullptr,
                                               "MaskB",
                                               graphIndex,
                                               0,
                                               nullptr,
                                               m_maskBSizes.min.data());

        m_tensorCIdx = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_cSizes.max.data(),
                                           m_cSizes.max.size(),
                                           syn_type_bf16,
                                           nullptr,
                                           "C",
                                           graphIndex,
                                           0,
                                           nullptr,
                                           m_cSizes.min.data());

        addNodeToGraph(NodeFactory::maskedBatchGemmNodeTypeName,
                       {m_tensorAIdx, m_tensorBIdx, m_tensorMaskAIdx, m_tensorMaskBIdx},
                       {m_tensorCIdx},
                       nullptr,
                       0,
                       "MaskedBgemm",
                       graphIndex);

        addProducers(graphIndex);

        return m_tensorCIdx;
    }
};

TEST_P_GC(SynGaudi2MaskedBGemmBundlingTest, slicing_masked_bgemm, {synDeviceGaudi2})
{
    // masked bgemm expects 4D tensors
    ASSERT_EQ(m_aSizes.max.size(), 4);
    ASSERT_EQ(m_bSizes.max.size(), 4);
    ASSERT_EQ(m_cSizes.max.size(), 4);

    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(maksed_bgemm_bundle,
                         SynGaudi2MaskedBGemmBundlingTest,
                         ::testing::Values(std::make_tuple(DimSizes(256), DimSizes(256), DimSizes(256), DimSizes(8)),
                                           std::make_tuple(DimSizes(512), DimSizes(128), DimSizes(384), DimSizes(6))));

class MantaRayTestUtils : public SynTrainingTwoRunCompareTest
{
public:
    void                          runSingleTest();
    virtual std::vector<unsigned> addNode() = 0;

    // Create a chain of nodes leading to a given tensor
    unsigned createProducerChain(std::string        mask,
                                 unsigned           producedIdx,
                                 unsigned           graphIndex,
                                 const ShapeSizes&  shape,
                                 const std::string& templateName);

    virtual unsigned createLogicalNode(ShapeSizes&        shapeToProduce,
                                       unsigned           tensorIndexToConsume,
                                       unsigned           graphIndex,
                                       synDataType        dataType,
                                       const std::string& name);

    virtual unsigned createTPCNode(ShapeSizes&        shapeToProduce,
                                   unsigned           tensorIndexToConsume,
                                   unsigned           graphIndex,
                                   synDataType        dataType,
                                   const std::string& name);
};

unsigned MantaRayTestUtils::createLogicalNode(ShapeSizes&        shapeToProduce,
                                              unsigned int       tensorIndexToConsume,
                                              unsigned           graphIndex,
                                              synDataType        dataType,
                                              const std::string& name)
{
    // This tensor is intermediate but the testing framework asserts that it's not for some reason
    unsigned tensorIndex = createTensor(INPUT_TENSOR,
                                        MEM_INIT_NONE,
                                        nullptr,
                                        shapeToProduce.max.data(),
                                        shapeToProduce.max.size(),
                                        dataType,
                                        nullptr,
                                        shapeToProduce.min.data());

    static const std::string guid = NodeFactory::staticReshapeNodeTypeName;
    addNodeToGraph(guid.c_str(), {tensorIndexToConsume}, {tensorIndex}, nullptr, 0, name.c_str(), graphIndex);

    return tensorIndex;
}

unsigned MantaRayTestUtils::createTPCNode(ShapeSizes&        shapeToProduce,
                                          unsigned int       tensorIndexToConsume,
                                          unsigned int       graphIndex,
                                          synDataType        dataType,
                                          const std::string& name)
{
    // This tensor is intermediate but the testing framework asserts that it's not for some reason
    unsigned tensorIndex = createTensor(INPUT_TENSOR,
                                        MEM_INIT_NONE,
                                        nullptr,
                                        shapeToProduce.max.data(),
                                        shapeToProduce.max.size(),
                                        dataType,
                                        nullptr,
                                        shapeToProduce.min.data());

    // If you use the relu GUID from the node factory it doesn't work
    static const std::string guid = "relu_fwd_bf16";

    addNodeToGraph(guid.c_str(), {tensorIndexToConsume}, {tensorIndex}, nullptr, 0, name.c_str(), graphIndex);

    return tensorIndex;
}

unsigned MantaRayTestUtils::createProducerChain(std::string        mask,
                                                unsigned           producedIdx,
                                                unsigned           graphIndex,
                                                const ShapeSizes&  shape,
                                                const std::string& templateName)
{
    unsigned   prevProdOutput = producedIdx;
    ShapeSizes reshapedShape  = shape;
    unsigned   numDims        = reshapedShape.max.size();

    if (numDims <= 2)
    {
        // Add a dummy external dimension
        reshapedShape.max.push_back(1);
    }
    else
    {
        // smoosh dim 1 into dim 0
        reshapedShape.max[1] *= reshapedShape.max[0];
        reshapedShape.max.erase(reshapedShape.max.begin());
    }
    reshapedShape.min = reshapedShape.max;

    std::string nodeName;
    size_t      lastLogicalIndex = mask.find_last_not_of("pP");
    if (lastLogicalIndex == std::string::npos) lastLogicalIndex = mask.length() + 1;

    ShapeSizes shapes[]     = {shape, reshapedShape};
    unsigned   lastShapeIdx = 0;
    for (unsigned prodIdx = 0; prodIdx < mask.length(); ++prodIdx)
    {
        bool isPhysicalProducer = mask[prodIdx] == 'P' || mask[prodIdx] == 'p';
        nodeName                = templateName + " producer #" + std::to_string(prodIdx);
        nodeName += isPhysicalProducer ? std::string(" (Physical)") : std::string(" (Logical)");

        ShapeSizes outputShape = shapes[lastShapeIdx];
        if (!isPhysicalProducer)
        {
            if (prodIdx == lastLogicalIndex)
            {
                // Last logical needs to return to the original shape so GEMMs can read it
                outputShape = shape;
            }
            else
            {
                // Toggle between 1 and 0
                lastShapeIdx = 1 - lastShapeIdx;
                outputShape  = shapes[lastShapeIdx];
            }
        }

        prevProdOutput = isPhysicalProducer
                             ? createTPCNode(outputShape, prevProdOutput, graphIndex, syn_type_bf16, nodeName)
                             : createLogicalNode(outputShape, prevProdOutput, graphIndex, syn_type_bf16, nodeName);
    }
    return prevProdOutput;
}

void MantaRayTestUtils::runSingleTest()
{
    std::vector<unsigned> tensorToValidateIdx = addNode();
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "10485760");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    // We want to force multiple TPCs on producer chains, fuser messes with those
    addConfigurationToRun(FIRST_RUN, "RUN_TPC_FUSER", "0");
    addConfigurationToRun(SECOND_RUN, "RUN_TPC_FUSER", "0");

    compareRunsResults({tensorToValidateIdx});
}

class SynTrainingMantaRaySlicingTest
: public MantaRayTestUtils
, public testing::WithParamInterface<
      std::tuple<DimSizes, DimSizes, DimSizes, DimSizes, unsigned, std::string, std::string>>

{
public:
    SynTrainingMantaRaySlicingTest()
    : m_heightA(std::get<0>(GetParam())),
      m_commonDim(std::get<1>(GetParam())),
      m_widthB(std::get<2>(GetParam())),
      m_batch(std::get<3>(GetParam())),
      m_numGemms(std::get<4>(GetParam())),
      m_sharedOperandMask(std::get<5>(GetParam())),
      m_unsharedOperandMask(std::get<6>(GetParam()))
    {
        setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});

        m_aSizes.min    = {m_commonDim.min, m_heightA.min /*, m_batch.min*/};
        m_aSizes.max    = {m_commonDim.max, m_heightA.max /*, m_batch.max*/};
        m_aSizes.actual = {m_commonDim.actual, m_heightA.actual /*, m_batch.actual*/};

        m_bSizes.min    = {m_widthB.min, m_commonDim.min /*, m_batch.min*/};
        m_bSizes.max    = {m_widthB.max, m_commonDim.max /*, m_batch.max*/};
        m_bSizes.actual = {m_widthB.actual, m_commonDim.actual /*, m_batch.actual*/};

        m_cSizes.min    = {m_widthB.min, m_heightA.min /*, m_batch.min*/};
        m_cSizes.max    = {m_widthB.max, m_heightA.max /*, m_batch.max*/};
        m_cSizes.actual = {m_widthB.actual, m_heightA.actual /*, m_batch.actual*/};
        if (m_batch.max > 1)
        {
            m_aSizes.min.push_back(m_batch.min);
            m_aSizes.max.push_back(m_batch.max);
            m_aSizes.actual.push_back(m_batch.actual);

            m_bSizes.min.push_back(m_batch.min);
            m_bSizes.max.push_back(m_batch.max);
            m_bSizes.actual.push_back(m_batch.actual);

            m_cSizes.min.push_back(m_batch.min);
            m_cSizes.max.push_back(m_batch.max);
            m_cSizes.actual.push_back(m_batch.actual);
        }
    }

    virtual std::vector<unsigned> addNode();

protected:
    DimSizes m_heightA;
    DimSizes m_commonDim;
    DimSizes m_widthB;
    DimSizes m_batch;

    ShapeSizes m_aSizes;
    ShapeSizes m_bSizes;
    ShapeSizes m_cSizes;

    unsigned    m_numGemms;
    std::string m_sharedOperandMask;
    std::string m_unsharedOperandMask;
};

std::vector<unsigned> SynTrainingMantaRaySlicingTest::addNode()
{
    unsigned      graphIndex = 0;
    synGEMMParams params;

    unsigned              tensorAIdx, tensorBIdx;
    std::vector<unsigned> tensorCIdx;

    std::string tensorAName = "A";
    std::string tensorBName = "B";
    std::string tensorCName = "C";
    std::string nodeName    = "GEMM";

    tensorAIdx = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     m_aSizes.max.data(),
                                     m_aSizes.max.size(),
                                     syn_type_bf16,
                                     nullptr,
                                     "A",
                                     graphIndex,
                                     0,
                                     nullptr,
                                     m_aSizes.min.data());

    // Add producer chain
    tensorAIdx = createProducerChain(m_sharedOperandMask, tensorAIdx, graphIndex, m_aSizes, tensorAName);

    // Add GEMMs with shared input operand
    for (unsigned gemmIdx = 0; gemmIdx < m_numGemms; ++gemmIdx)
    {
        tensorBName = "B #" + std::to_string(gemmIdx);
        tensorCName = "C #" + std::to_string(gemmIdx);
        nodeName    = "GEMM #" + std::to_string(gemmIdx);

        tensorBIdx = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         m_bSizes.max.data(),
                                         m_bSizes.max.size(),
                                         syn_type_bf16,
                                         nullptr,
                                         tensorBName.c_str(),
                                         graphIndex,
                                         0,
                                         nullptr,
                                         m_bSizes.min.data());

        tensorBIdx = createProducerChain(m_unsharedOperandMask, tensorBIdx, graphIndex, m_bSizes, tensorBName);

        unsigned tmp = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_cSizes.max.data(),
                                           m_cSizes.max.size(),
                                           syn_type_bf16,
                                           nullptr,
                                           tensorCName.c_str(),
                                           graphIndex,
                                           0,
                                           nullptr,
                                           m_cSizes.min.data());

        tensorCIdx.push_back(tmp);

        std::string guid = (m_batch.max > 1) ? NodeFactory::batchGemmNodeTypeName : NodeFactory::gemmNodeTypeName;

        addNodeToGraph(guid.c_str(),
                       {tensorAIdx, tensorBIdx},
                       {tmp},
                       &params,
                       sizeof(params),
                       nodeName.c_str(),
                       graphIndex);
    }

    return tensorCIdx;
}

TEST_P_GC(SynTrainingMantaRaySlicingTest, slicing_gemm)
{
    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    runSingleTest();
}

// Params:
// A height, common dimension, B width, batch size, #GEMMs, producer chain description (master, non-master)

// List of tests:
// Trivial, everything fits easily
// Must-slice: everything fits but you have to slice A and its producers
// Various operands can't fit

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    slicing_gemm_ASIC_CI,
    SynTrainingMantaRaySlicingTest,
    ::testing::Values(
        // Trivial: everything will fit
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 2, "", ""),        // Two GEMMs, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 2, "P", ""),       // Two GEMMs, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 2, "PLL", ""),     // Two GEMMs, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 2, "PLL", "PLL"),  // Two GEMMs, producer chains
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 3, "", ""),        // Three GEMMs, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 3, "P", ""),    // Three GEMMs, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 3, "PLL", ""),  // Three GEMMs, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(1), 3, "PLL", "PLL"),  // Three GEMMs, producer chains

        // Must-slice: everything will fit if you slice
        std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "", ""),     // Two GEMMs, no producers
        std::make_tuple(DimSizes(2048), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "P", ""),    // Two GEMMs, one producer
        std::make_tuple(DimSizes(2048), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "PLL", ""),  // Two GEMMs, producer chain
        std::make_tuple(DimSizes(2048), DimSizes(1024), DimSizes(256), DimSizes(1), 2, "PLL", "PLL"),  // Two GEMMs, producer chains
        std::make_tuple(DimSizes(3072), DimSizes(2048), DimSizes(256), DimSizes(1), 3, "", ""),     // Three GEMMs, no producers
        std::make_tuple(DimSizes(1536), DimSizes(1024), DimSizes(256), DimSizes(1), 3, "P", ""),    // Three GEMMs, one producer
        std::make_tuple(DimSizes(1536), DimSizes(1024), DimSizes(256), DimSizes(1), 3, "PLL", ""),  // Three GEMMs, producer chain
        std::make_tuple(DimSizes(1536), DimSizes(512), DimSizes(256), DimSizes(1), 3, "PLL", "PLL"),  // Three GEMMs, producer chains

        // Can't fit unshared operands
        std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(1024), DimSizes(1), 2, "PLL", "PLL"),  // Two GEMMs, producer chains
        std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(512), DimSizes(1), 3, "PLL", "PLL"),  // Three GEMMs, producer chains
        std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(1024), DimSizes(1), 3, "PLL", "PLL"),  // Three GEMMs, producer chains

        // Can't fit all of master producer chain
        std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(1024), DimSizes(1), 2, "PPPPPPLLLL", ""),  // Two GEMMs, producer chain
        std::make_tuple(DimSizes(1024), DimSizes(4096), DimSizes(1024), DimSizes(1), 2, "PPLL", "PPLL"),  // Two GEMMs, producer chains
        std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(128), DimSizes(1), 2, "PLL", "PLL"),  // Two GEMMs, producer chains

        // Interesting chains
        std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "LPL", ""),  // Two GEMMs, first producer is logical
        std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "", "P"),  // Two GEMMs, only unshared is produced
        std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "", "LPL"),  // Two GEMMs, first producer is logical
        std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "LPL", "LPL"),  // Two GEMMs, first producer is logical
        std::make_tuple(DimSizes(1024), DimSizes(1024), DimSizes(256), DimSizes(1), 2, "PLPPLLLLLPPLPLL", "LLPPPLPLLPLLLPLPL"),  // Two GEMMs, crazy chain
        std::make_tuple(DimSizes(1024), DimSizes(1024), DimSizes(256), DimSizes(1), 1, "PLPPLLLLLPPLPLL", "LLPPPLPLLPLLLPLPL"),  // One GEMM, crazy chain
        std::make_tuple(DimSizes(1024), DimSizes(1024), DimSizes(256), DimSizes(1), 3, "PLPPLLLLLPPLPLL", "LLPPPLPLLPLLLPLPL"),  // Three GEMMs, crazy chain
        std::make_tuple(DimSizes(1024), DimSizes(1024), DimSizes(256), DimSizes(1), 1, "PLPPLLLLLPPLPLL", ""),  // One GEMM, crazy chain
        std::make_tuple(DimSizes(1024), DimSizes(1024), DimSizes(256), DimSizes(1), 2, "PLPPLLLLLPPLPLL", ""),  // Two GEMMs, crazy chain

        // Batch Gemms
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 1, "", ""),     // One BGEMM, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 1, "P", ""),    // One BGEMM, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 1, "PLL", ""),  // One BGEMM, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 1, "PLL", "PLL"),  // One BGEMM, producer chains
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 2, "", ""),     // Two BGEMMs, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 2, "P", ""),    // Two BGEMMs, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 2, "PLL", ""),  // Two BGEMMs, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 2, "PLL", "PLL"),  // Two BGEMMs, producer chains
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 3, "", ""),     // Three BGEMMs, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 3, "P", ""),    // Three BGEMMs, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 3, "PLL", ""),  // Three BGEMMs, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), DimSizes(16), 3, "PLL", "PLL")  // Three BGEMMs, producer chains

        ));
// clang-format on

// Params: Height (BHW), common dim (K), channels (C)
//         Num of dedx, num of dedw, producer chain descriptions
class SynTrainingMantaRayBWDSlicingTest
: public MantaRayTestUtils
, public testing::WithParamInterface<
      std::tuple<DimSizes, DimSizes, DimSizes, unsigned, unsigned, std::string, std::string>>
{
public:
    SynTrainingMantaRayBWDSlicingTest()
    : m_H(std::get<0>(GetParam())),
      m_K(std::get<1>(GetParam())),
      m_C(std::get<2>(GetParam())),
      m_dedx(std::get<3>(GetParam())),
      m_dedw(std::get<4>(GetParam())),
      m_sharedOperandMask(std::get<5>(GetParam())),
      m_unsharedOperandMask(std::get<6>(GetParam()))
    {
        m_gradOutSizes.min    = {m_K.min, m_H.min};
        m_gradOutSizes.max    = {m_K.max, m_H.max};
        m_gradOutSizes.actual = {m_K.actual, m_H.actual};

        m_gradWeightsSizes.min    = {m_K.min, m_C.min};
        m_gradWeightsSizes.max    = {m_K.max, m_C.max};
        m_gradWeightsSizes.actual = {m_K.actual, m_C.actual};

        m_gradInSizes.min    = {m_C.min, m_H.min};
        m_gradInSizes.max    = {m_C.max, m_H.max};
        m_gradInSizes.actual = {m_C.actual, m_H.actual};
    }

    std::vector<unsigned> addNode();

protected:
    DimSizes m_H;
    DimSizes m_K;
    DimSizes m_C;

    ShapeSizes m_gradOutSizes;
    ShapeSizes m_gradWeightsSizes;
    ShapeSizes m_gradInSizes;

    unsigned    m_dedx;
    unsigned    m_dedw;
    std::string m_sharedOperandMask;
    std::string m_unsharedOperandMask;
};

std::vector<unsigned> SynTrainingMantaRayBWDSlicingTest::addNode()
{
    unsigned      graphIndex = 0;
    synGEMMParams params;

    unsigned              gradOutIdx, inIdx, wIdx;
    std::vector<unsigned> outputIndices;

    std::string gradOutName = "GradOut";
    std::string wName       = "W";
    std::string inName      = "A";
    std::string outName     = "GradIn";
    std::string nodeName    = "GEMM";

    gradOutIdx = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     m_gradOutSizes.max.data(),
                                     m_gradOutSizes.max.size(),
                                     syn_type_bf16,
                                     nullptr,
                                     gradOutName.c_str(),
                                     graphIndex,
                                     0,
                                     nullptr,
                                     m_gradOutSizes.min.data());

    // Add producer chain
    gradOutIdx = createProducerChain(m_sharedOperandMask, gradOutIdx, graphIndex, m_gradOutSizes, gradOutName);

    // Add dedx nodes sharing the gradOut
    params.transpose_a = false;
    params.transpose_b = true;
    for (unsigned gemmIdx = 0; gemmIdx < m_dedx; ++gemmIdx)
    {
        wName    = "W #" + std::to_string(gemmIdx);
        outName  = "GradIn #" + std::to_string(gemmIdx);
        nodeName = "dE/dX #" + std::to_string(gemmIdx);

        wIdx = createPersistTensor(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   m_gradWeightsSizes.max.data(),
                                   m_gradWeightsSizes.max.size(),
                                   syn_type_bf16,
                                   nullptr,
                                   wName.c_str(),
                                   graphIndex,
                                   0,
                                   nullptr,
                                   m_gradWeightsSizes.min.data());

        wIdx = createProducerChain(m_unsharedOperandMask, wIdx, graphIndex, m_gradWeightsSizes, wName);

        unsigned tmp = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_gradInSizes.max.data(),
                                           m_gradInSizes.max.size(),
                                           syn_type_bf16,
                                           nullptr,
                                           outName.c_str(),
                                           graphIndex,
                                           0,
                                           nullptr,
                                           m_gradInSizes.min.data());

        outputIndices.push_back(tmp);

        std::string guid = NodeFactory::gemmNodeTypeName;

        addNodeToGraph(guid.c_str(), {gradOutIdx, wIdx}, {tmp}, &params, sizeof(params), nodeName.c_str(), graphIndex);
    }

    // Add dedw nodes sharing the gradOut
    params.transpose_a = true;
    params.transpose_b = false;
    for (unsigned gemmIdx = 0; gemmIdx < m_dedw; ++gemmIdx)
    {
        inName   = "Input #" + std::to_string(gemmIdx);
        outName  = "GradWeights #" + std::to_string(gemmIdx);
        nodeName = "dE/dW #" + std::to_string(gemmIdx);

        inIdx = createPersistTensor(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    m_gradInSizes.max.data(),
                                    m_gradInSizes.max.size(),
                                    syn_type_bf16,
                                    nullptr,
                                    inName.c_str(),
                                    graphIndex,
                                    0,
                                    nullptr,
                                    m_gradInSizes.min.data());

        inIdx = createProducerChain(m_unsharedOperandMask, inIdx, graphIndex, m_gradInSizes, inName);

        unsigned tmp = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_gradWeightsSizes.max.data(),
                                           m_gradWeightsSizes.max.size(),
                                           syn_type_bf16,
                                           nullptr,
                                           outName.c_str(),
                                           graphIndex,
                                           0,
                                           nullptr,
                                           m_gradWeightsSizes.min.data());

        outputIndices.push_back(tmp);

        std::string guid = NodeFactory::gemmNodeTypeName;

        addNodeToGraph(guid.c_str(), {inIdx, gradOutIdx}, {tmp}, &params, sizeof(params), nodeName.c_str(), graphIndex);
    }

    return outputIndices;
}

TEST_P_GC(SynTrainingMantaRayBWDSlicingTest, slicing_dedxdedw)
{
    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    runSingleTest();
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    slicing_dedxdedw_ASIC_CI,
    SynTrainingMantaRayBWDSlicingTest,
    ::testing::Values(
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 1, "", ""),        // Two GEMMs, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 1, "P", ""),       // Two GEMMs, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 1, "PLL", ""),     // Two GEMMs, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 1, "PLL", "PLL"),  // Two GEMMs, producer chains
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 2, 1, "", ""),        // Three GEMMs, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 2, 1, "P", ""),       // Three GEMMs, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 2, 1, "PLL", ""),  // Three GEMMs, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 2, 1, "PLL", "PLL"),  // Three GEMMs, producer chains
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 2, "", ""),        // Three GEMMs, no producers
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 2, "P", ""),       // Three GEMMs, one producer
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 2, "PLL", ""),  // Three GEMMs, producer chain
        std::make_tuple(DimSizes(256), DimSizes(1024), DimSizes(256), 1, 2, "PLL", "PLL"),  // Three GEMMs, producer chains

        // Must-slice: everything will fit if you slice
        std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), 1, 1, "", ""),     // Two GEMMs, no producers
        std::make_tuple(DimSizes(2048), DimSizes(2048), DimSizes(256), 1, 1, "P", ""),    // Two GEMMs, one producer
        std::make_tuple(DimSizes(2048), DimSizes(2048), DimSizes(256), 1, 1, "PLL", ""),  // Two GEMMs, producer chain
        std::make_tuple(DimSizes(2048), DimSizes(1024), DimSizes(256), 1, 1, "PLL", "PLL"),  // Two GEMMs, producer chains
        std::make_tuple(DimSizes(3072), DimSizes(2048), DimSizes(256), 2, 1, "", ""),    // Three GEMMs, no producers
        std::make_tuple(DimSizes(1536), DimSizes(1024), DimSizes(256), 2, 1, "P", ""),   // Three GEMMs, one producer
        std::make_tuple(DimSizes(1536), DimSizes(512), DimSizes(256), 2, 1, "PLL", ""),  // Three GEMMs, producer chain
        std::make_tuple(DimSizes(1536), DimSizes(512), DimSizes(256), 2, 1, "PLL", "PLL"),  // Three GEMMs, producer chains
        std::make_tuple(DimSizes(3072), DimSizes(2048), DimSizes(256), 1, 2, "", ""),       // Three GEMMs, no producers
        std::make_tuple(DimSizes(1536), DimSizes(1024), DimSizes(256), 1, 2, "P", ""),      // Three GEMMs, one producer
        std::make_tuple(DimSizes(1536), DimSizes(512), DimSizes(256), 1, 2, "PLL", ""),  // Three GEMMs, producer chain
        std::make_tuple(DimSizes(1536), DimSizes(512), DimSizes(256), 1, 2, "PLL", "PLL")  // Three GEMMs, producer chains
        /*
        //Can't fit unshared operands
        std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(1024), DimSizes(1), 2, "PLL", "PLL"), //Two GEMMs,
        producer chains std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(512), DimSizes(1), 3, "PLL", "PLL"),
        //Three GEMMs, producer chains std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(1024), DimSizes(1), 3,
        "PLL", "PLL"), //Three GEMMs, producer chains

        //Can't fit all of master producer chain
        std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(1024), DimSizes(1), 2, "PPPPPPLLLL", ""), //Two GEMMs,
        producer chain std::make_tuple(DimSizes(1024), DimSizes(4096), DimSizes(1024), DimSizes(1), 2, "PPLL", "PPLL"),
        //Two GEMMs, producer chains std::make_tuple(DimSizes(2048), DimSizes(4096), DimSizes(128), DimSizes(1), 2,
        "PLL", "PLL"), //Two GEMMs, producer chains

        //Interesting chains
        std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "LPL",""), //Two GEMMs, first
        producer is logical std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256), DimSizes(1), 2, "","P"),
        //Two GEMMs, only unshared is produced std::make_tuple(DimSizes(4096), DimSizes(2048), DimSizes(256),
        DimSizes(1), 2, "","LPL"), //Two GEMMs, first producer is logical std::make_tuple(DimSizes(4096),
        DimSizes(2048), DimSizes(256), DimSizes(1), 2, "LPL","LPL"), //Two GEMMs, first producer is logical
        std::make_tuple(DimSizes(1024), DimSizes(1024), DimSizes(256), DimSizes(1), 2,
        "PLPPLLLLLPPLPLL","LLPPPLPLLPLLLPLPL") //Two GEMMs, crazy chain
        */
        ));
// clang-format on

struct BatchGemmParams
{
    unsigned              in0Height;
    unsigned              in1Width;
    unsigned              commonDim;
    std::vector<unsigned> in0Batch;
    std::vector<unsigned> in1Batch;
};

class SynTrainingBroadcastBgemm
: public SynTrainingTwoRunCompareTest
, public testing::WithParamInterface<BatchGemmParams>
{
public:
    void broadcastBgemm();
    SynTrainingBroadcastBgemm() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); };

private:
    std::vector<unsigned> createShape(unsigned width, unsigned height, const std::vector<unsigned> batchDims);
};

std::vector<unsigned>
SynTrainingBroadcastBgemm::createShape(unsigned width, unsigned height, const std::vector<unsigned> batchDims)
{
    std::vector<unsigned> shape;
    shape.reserve(MAX_DIMENSIONS_NUM);
    shape.push_back(width);
    shape.push_back(height);

    shape.insert(shape.end(), batchDims.begin(), batchDims.end());
    return shape;
}

void SynTrainingBroadcastBgemm::broadcastBgemm()
{
    TestSizes bgemmOutputShape({1, 1, 1, 1, 1});

    auto        in0Height = GetParam().in0Height;
    auto        in1Width  = GetParam().in1Width;
    auto        commonDim = GetParam().commonDim;
    const auto& in0Batch  = GetParam().in0Batch;
    const auto& in1Batch  = GetParam().in1Batch;

    std::vector<unsigned> in0Shape, in1Shape;
    in0Shape     = createShape(commonDim, in0Height, in0Batch);
    auto in0Rank = in0Shape.size();

    in1Shape     = createShape(in1Width, commonDim, in1Batch);
    auto in1Rank = in1Shape.size();

    unsigned reluIn = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_POSITIVE,
                                          nullptr,
                                          (unsigned*)in0Shape.data(),
                                          in0Rank,
                                          syn_type_bf16);

    auto bgemmInput0 = createTensor(OUTPUT_TENSOR,
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    (unsigned*)in0Shape.data(),
                                    in0Rank,
                                    syn_type_bf16);

    addNodeToGraph("relu_fwd_bf16", {reluIn}, {bgemmInput0}, nullptr, 0, "relu");

    unsigned bgemmInput1 = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_RANDOM_POSITIVE,
                                               nullptr,
                                               (unsigned*)in1Shape.data(),
                                               in1Rank,
                                               syn_type_bf16);

    bgemmOutputShape[0] = in1Width;
    bgemmOutputShape[1] = in0Height;

    auto outputRank = std::max(in0Rank, in1Rank);
    for (int dim = DIM_GEMM_BATCH; dim < outputRank; dim++)
    {
        if (dim < in0Rank && dim < in1Rank)
        {
            bgemmOutputShape[dim] = std::max(in0Batch[dim - DIM_GEMM_BATCH], in1Batch[dim - DIM_GEMM_BATCH]);
        }
        else if (dim < in0Rank)
        {
            bgemmOutputShape[dim] = in0Batch[dim - DIM_GEMM_BATCH];
        }
        else  // dim < in1Rank
        {
            bgemmOutputShape[dim] = in1Batch[dim - DIM_GEMM_BATCH];
        }
    }

    unsigned bgemmOut = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_RANDOM_POSITIVE,
                                            nullptr,
                                            (unsigned*)bgemmOutputShape.data(),
                                            outputRank,
                                            syn_type_bf16);

    addNodeToGraph("batch_gemm", {bgemmInput0, bgemmInput1}, {bgemmOut}, nullptr, 0, "batch_gemm_asymm_bcast");

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "70000");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    compareRunsResults({bgemmOut});
}

TEST_P_GC(SynTrainingBroadcastBgemm, AsymmetricBroadcastBgemm)
{
    broadcastBgemm();
}

// Params in0Height, in1Width, common-dim, in0-batch (vector), in1-batch (vecd sp   ctor)
INSTANTIATE_TEST_SUITE_P(asymmetric_broadcast_bgemm_ASIC_CI,
                         SynTrainingBroadcastBgemm,
                         ::testing::Values(BatchGemmParams {128, 512, 512, {32, 16}, {}},
                                           BatchGemmParams {128, 512, 512, {16}, {}},
                                           BatchGemmParams {128, 512, 512, {}, {16}}));

INSTANTIATE_TEST_SUITE_P(asymmetric_broadcast_bgemm_sanity,
                         SynTrainingBroadcastBgemm,
                         ::testing::Values(BatchGemmParams {64, 11, 127, {31}, {}},
                                           BatchGemmParams {64, 11, 127, {}, {31}},
                                           BatchGemmParams {64, 11, 127, {31, 7}, {31}},
                                           BatchGemmParams {64, 11, 127, {31}, {31, 7}},
                                           BatchGemmParams {64, 11, 127, {31, 7}, {1}},
                                           BatchGemmParams {64, 11, 127, {1}, {31, 7}}));

INSTANTIATE_TEST_SUITE_P(asymmetric_broadcast_bgemm_multiple_batch_dims,
                         SynTrainingBroadcastBgemm,
                         ::testing::Values(BatchGemmParams {64, 11, 127, {8, 4}, {}},
                                           BatchGemmParams {64, 11, 127, {8, 4, 2}, {}}));

// JIRA SW-85205
INSTANTIATE_TEST_SUITE_P(DISABLED_asymmetric_broadcast_bgemm_3batch_dims_ASIC_CI,
                         SynTrainingBroadcastBgemm,
                         ::testing::Values(BatchGemmParams {64, 16, 128, {64, 16, 8}, {}},
                                           BatchGemmParams {64, 16, 128, {64, 16}, {}}));

TEST_F_GC(SynTrainingTestInfra, shared_mme_input_doesnt_fit_sram, {synDeviceGaudi2, synDeviceGaudi3})
{
    // Graph #0

    /*************
     * g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_dedx3d_n542_0 node
     * inputs:
     *     g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad[32, 128,
     *128, 128, 2] (dtype=bf16) g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0[32, 32, 3, 3, 3]
     *(dtype=bf16) g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2[32, 128, 128, 128, 2] (dtype=uint32)
     *(shape tensor) outputs: g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0[32, 128, 128, 128, 2]
     *(dtype=bf16) ctrl inputs: g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reshape_n538_control_edge_3936[]
     *(dtype=invalid) g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reshape_n539_control_edge_3937[] (dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_split_50_control_edge_3938[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reshape_51_control_edge_3939[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reduce_sum_stage1_52_control_edge_3940[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reshape_53_control_edge_3941[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reduce_sum_stage2_54_control_edge_3942[]
     *(dtype=invalid) ctrl outputs:
     *     g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_dedx3d_n542_control_edge_4162[] (dtype=invalid)
     *************/

    // create g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad tensor
    unsigned
        g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad_max_sizes
            [] = {32, 128, 128, 128, 2};
    unsigned
        g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad_min_sizes
            [] = {32, 128, 128, 128, 2};
    unsigned g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad",
            MEM_INIT_NONE,
            nullptr,
            g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad_max_sizes,
            5,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0 tensor
    unsigned g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0_max_sizes[] = {32, 32, 3, 3, 3};
    unsigned g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0_min_sizes[] = {32, 32, 3, 3, 3};
    unsigned g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0",
                      MEM_INIT_NONE,
                      nullptr,
                      g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2 tensor
    unsigned g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_max_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_min_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2 =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2",
                      MEM_INIT_NONE,
                      nullptr,
                      g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_max_sizes,
                      5,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0 tensor
    unsigned g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0_max_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0_min_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0",
                      MEM_INIT_NONE,
                      nullptr,
                      g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_dedx3d_n542_0_id;
    unsigned char g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_dedx3d_n542_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx3d",
                   {g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad,
                    g_0_t1374_conv3d_21_conv3d_readvariableop_fp32_to_bf16_cast_53_0,
                    g_0_t1646_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2},
                   {g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0},
                   (void*)g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_dedx3d_n542_0_params,
                   128,
                   "g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_dedx3d_n542_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_dedx3d_n542_0_id);

    /*************
     * g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_0 node
     * inputs:
     *     g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad[32, 128,
     *128, 128, 2] (dtype=bf16) g_0_t1366_leakyrelu_20_0[32, 128, 128, 128, 2] (dtype=bf16) outputs:
     *     g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0[32, 32, 3, 3, 3] (dtype=bf16)
     * ctrl inputs:
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reshape_n538_control_edge_3936[] (dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reshape_n539_control_edge_3937[] (dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_split_50_control_edge_3938[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reshape_51_control_edge_3939[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reduce_sum_stage1_52_control_edge_3940[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reshape_53_control_edge_3941[]
     *(dtype=invalid)
     *     g_0_gradients_conv3d_21_BiasAdd_grad_BiasAddGrad_reduce_sum_fwd_bf16_n540_complex_reduce_sum_stage2_54_control_edge_3942[]
     *(dtype=invalid) ctrl outputs:
     *     g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_control_edge_3943[] (dtype=invalid)
     *************/

    // create g_0_t1366_leakyrelu_20_0 tensor
    unsigned g_0_t1366_leakyrelu_20_0_max_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1366_leakyrelu_20_0_min_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1366_leakyrelu_20_0             = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_t1366_leakyrelu_20_0",
                                                      MEM_INIT_NONE,
                                                      nullptr,
                                                      g_0_t1366_leakyrelu_20_0_max_sizes,
                                                      5,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_t1366_leakyrelu_20_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0 tensor
    unsigned g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0_max_sizes[] = {32, 32, 3, 3, 3};
    unsigned g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0_min_sizes[] = {32, 32, 3, 3, 3};
    unsigned g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0",
                      MEM_INIT_NONE,
                      nullptr,
                      g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_0_id;
    unsigned char g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw3d",
                   {g_0_t1629_gradients_habana_instance_normalization_21_HabanaInstanceNorm_grad_HabanaInstanceNormGrad,
                    g_0_t1366_leakyrelu_20_0},
                   {g_0_t1647_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_0},
                   (void*)g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_0_params,
                   128,
                   "g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_0_id);

    /*************
     * g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0 node
     * inputs:
     *     g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0[32, 128, 128, 128, 2] (dtype=bf16)
     *     g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0[32, 128, 128, 128, 2] (dtype=bf16)
     * outputs:
     *     g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0[32, 128, 128, 128, 2] (dtype=bf16)
     * ctrl inputs:
     *     g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_control_edge_3943[] (dtype=invalid)
     * ctrl outputs:
     *************/

    // create g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0 tensor
    unsigned g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0_max_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0_min_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0",
                      MEM_INIT_NONE,
                      nullptr,
                      g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0 tensor
    unsigned g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0_max_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0_min_sizes[] = {32, 128, 128, 128, 2};
    unsigned g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0",
                      MEM_INIT_NONE,
                      nullptr,
                      g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0_id;
    unsigned char g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0_params[] =
        {0, 0, 0, 64, 225, 122, 132, 63};
    addNodeToGraph("leakyrelu_bwd_bf16",
                   {g_0_t1645_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropInputV2_0,
                    g_0_t1375_habana_instance_normalization_20_habanainstancenorm_0},
                   {g_0_t1650_gradients_LeakyRelu_20_grad_LeakyReluGrad_0},
                   (void*)g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0_params,
                   8,
                   "g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0_id);

    synNodeId blocking_g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0[] = {
        g_0_gradients_conv3d_21_Conv3D_grad_Conv3DBackpropFilterV2_dedw3d_n543_0_id};
    setNodeDependency(blocking_g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0,
                      &g_0_gradients_LeakyRelu_20_grad_LeakyReluGrad_leakyrelu_bwd_bf16_n545_0_id,
                      1,
                      1);

    compileTopology("shared_mme_input_doesnt_fit_sram", 0);
}

TEST_F_GC(SynTrainingTwoRunCompareTest,
          float8_batch_norm_with_negative_access_pattern_ASIC_CI,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    // Graph #0 - Taken from resnet in fp8 (bundle #52)

    // create conv_ifm tensor
    unsigned conv_ifm_max_sizes[] = {3, 224, 224, 256};
    unsigned conv_ifm_min_sizes[] = {3, 224, 224, 256};
    unsigned conv_ifm             = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "conv_ifm",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      conv_ifm_max_sizes,
                                      4,
                                      syn_type_fp8_152,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv_ifm_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    /*************
     * conv_node node
     * inputs:
     *     conv_ifm[3, 224, 224, 256]
     *(dtype=float8)
     *     conv_w[64,
     *3, 7, 7] (dtype=float8) outputs: conv_ofm[64, 112, 112, 256]
     *(dtype=float8) ctrl inputs: ctrl outputs:
     *************/

    // create
    // conv_w
    // tensor
    unsigned conv_w_max_sizes[] = {64, 3, 7, 7};
    unsigned conv_w_min_sizes[] = {64, 3, 7, 7};
    unsigned conv_w             = createTensors(
        1,
        INPUT_TENSOR,
        true,
        "g_0_t792_while_body__1_while_resnet50_conv1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_371_bf16_to_fp8_152_"
        "cast_155_0",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        conv_w_max_sizes,
        4,
        syn_type_fp8_152,
        nullptr,
        0,
        0,
        nullptr,
        false,
        conv_w_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    // create conv_ofm tensor
    unsigned      conv_ofm_max_sizes[] = {64, 112, 112, 256};
    unsigned      conv_ofm_min_sizes[] = {64, 112, 112, 256};
    unsigned      conv_ofm             = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "conv_ofm",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      conv_ofm_max_sizes,
                                      4,
                                      syn_type_fp8_152,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv_ofm_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId     conv_node_id;
    unsigned char conv_node_params[] = {7, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0,  0,  3,   0,   0, 0,
                                        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 11, 62, 1,   0,   0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  255, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {conv_ifm, conv_w},
                   {conv_ofm},
                   (void*)conv_node_params,
                   72,
                   "conv_node",
                   0 /*graphIndex*/,
                   &conv_node_id);

    /*************
     * bn_st1_node node
     * inputs:
     *     conv_ofm[64, 112, 112, 256] (dtype=float8)
     *     bn_st1_in2[64] (dtype=float32)
     * outputs:
     *     bn_st1_out[64, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create bn_st1_in2 tensor
    unsigned bn_st1_in2_max_sizes[] = {64};
    unsigned bn_st1_in2_min_sizes[] = {64};
    unsigned bn_st1_in2             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "bn_st1_in2",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        bn_st1_in2_max_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        bn_st1_in2_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create bn_st1_out tensor
    unsigned      bn_st1_out_max_sizes[] = {64, 3};
    unsigned      bn_st1_out_min_sizes[] = {64, 3};
    unsigned      bn_st1_out             = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true,
                                        "bn_st1_out",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        bn_st1_out_max_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        bn_st1_out_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     bn_st1_node_id;
    unsigned char bn_st1_node_params[] = {0, 0, 0, 0, 0, 0, 49, 0, 1, 127, 0, 0};
    addNodeToGraph("batch_norm_stage1_fwd_f8",
                   {conv_ofm, bn_st1_in2},
                   {bn_st1_out},
                   (void*)bn_st1_node_params,
                   12,
                   "bn_st1_node",
                   0 /*graphIndex*/,
                   &bn_st1_node_id);

    /*************
     * bn_st2_node node
     * inputs:
     *     conv_ofm[64, 112, 112, 256] (dtype=float8)
     *     bn_st1_in2[64] (dtype=float32)
     *     bn_st1_out[64, 3] (dtype=float32)
     *     bn_st2_in4[64, 2] (dtype=float32)
     *     bn_st2_in5[64, 2] (dtype=float32)
     * outputs:
     *     bn_st2_out0[64, 112, 112, 256] (dtype=float8)
     *     bn_st2_out1[64, 2] (dtype=float32)
     *     bn_st2_out2[64, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create bn_st2_in4 tensor
    unsigned bn_st2_in4_max_sizes[] = {64, 2};
    unsigned bn_st2_in4_min_sizes[] = {64, 2};
    unsigned bn_st2_in4             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "bn_st2_in4",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        bn_st2_in4_max_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        bn_st2_in4_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create bn_st2_in5 tensor
    unsigned bn_st2_in5_max_sizes[] = {64, 2};
    unsigned bn_st2_in5_min_sizes[] = {64, 2};
    unsigned bn_st2_in5             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "bn_st2_in5",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        bn_st2_in5_max_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        bn_st2_in5_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create bn_st2_out0 tensor
    unsigned bn_st2_out0_max_sizes[] = {64, 112, 112, 256};
    unsigned bn_st2_out0_min_sizes[] = {64, 112, 112, 256};
    unsigned bn_st2_out0             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "bn_st2_out0",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         bn_st2_out0_max_sizes,
                                         4,
                                         syn_type_fp8_152,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         bn_st2_out0_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create bn_st2_out1 tensor
    unsigned bn_st2_out1_max_sizes[] = {64, 2};
    unsigned bn_st2_out1_min_sizes[] = {64, 2};
    unsigned bn_st2_out1             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "bn_st2_out1",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         bn_st2_out1_max_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         bn_st2_out1_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create bn_st2_out2 tensor
    unsigned      bn_st2_out2_max_sizes[] = {64, 2};
    unsigned      bn_st2_out2_min_sizes[] = {64, 2};
    unsigned      bn_st2_out2             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "bn_st2_out2",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         bn_st2_out2_max_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         bn_st2_out2_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     bn_st2_node_id;
    unsigned char bn_st2_node_params[] = {205, 204, 204, 61, 0, 0, 0, 0, 0, 0, 49, 0, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph("batch_norm_stage2_relu_fwd_f8",
                   {conv_ofm, bn_st1_in2, bn_st1_out, bn_st2_in4, bn_st2_in5},
                   {bn_st2_out0, bn_st2_out1, bn_st2_out2},
                   (void*)bn_st2_node_params,
                   20,
                   "bn_st2_node",
                   0 /*graphIndex*/,
                   &bn_st2_node_id);
    // compileAndRun();

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "-1");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({bn_st2_out2, bn_st1_out, conv_ofm, bn_st2_out0, bn_st2_out1});
}

TEST_F_GC(SynTrainingTwoRunCompareTest,
          mme_with_consumer_output_doesnt_fit_sram_ASIC_CI,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams params;
    params.kW = 3;
    params.kH = 3;
    params.dW = 2;
    params.dH = 2;

    unsigned    batch       = 2;
    unsigned    inChannels  = 400;
    unsigned    outChannels = 100;
    TestSizeVec xSizes      = {inChannels, 256, 256, batch};
    unsigned    yWidth = convOutputDimSize(xSizes[DIM_W], params.kW, params.dW, params.padL + params.padR, params.dilW);
    unsigned yHeight   = convOutputDimSize(xSizes[DIM_H], params.kH, params.dH, params.padT + params.padB, params.dilH);
    TestSizeVec ySizes = {outChannels, yWidth, yHeight, batch};
    TestSizeVec wSizes = {outChannels, inChannels, params.kW, params.kH};

    auto x = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes.data(), xSizes.size(), syn_type_bf16);
    auto y = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes.data(), ySizes.size(), syn_type_bf16);
    auto w = createPersistTensor(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 wSizes.data(),
                                 wSizes.size(),
                                 syn_type_bf16,
                                 nullptr,
                                 "W");

    addNodeToGraph(NodeFactory::deDxNodeTypeName, {y, w}, {x}, &params, sizeof(params), "dedx");

    // Add relu producer
    unsigned reluProdIn = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              ySizes.data(),
                                              ySizes.size(),
                                              syn_type_bf16,
                                              nullptr,
                                              "reluIn");

    addNodeToGraph("relu_fwd_bf16", {reluProdIn}, {y}, nullptr, 0, "ReluProducer");

    // Add relu consumer
    unsigned reluConsOut = createPersistTensor(OUTPUT_TENSOR,
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               xSizes.data(),
                                               xSizes.size(),
                                               syn_type_bf16,
                                               nullptr,
                                               "reluOut");

    addNodeToGraph("relu_fwd_bf16", {x}, {reluConsOut}, nullptr, 0, "ReluConsumer");

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({reluConsOut});
}

TEST_F_GC(SynTrainingTwoRunCompareTest,
          bgemm_consumer_chain_with_single_tpc_ASIC_CI,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    // Graph #0

    /*************
     * g_0_3_attention_batch_gemm_bf16_285_0 node
     * inputs:
     *     g_0_tensor_38_id_2641_3_attention_aten__transpose[128, 2048, 12] (dtype=bf16)
     *     g_0_tensor_34_id_2645_3_attention_aten__transpose[2048, 128, 12] (dtype=bf16)
     * outputs:
     *     g_0_tensor_39_id_2647_3_attention_aten__bmm[2048, 2048, 12] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_38_id_2641_3_attention_aten__transpose tensor
    unsigned g_0_tensor_38_id_2641_3_attention_aten__transpose_max_sizes[] = {128, 2048, 12};
    unsigned g_0_tensor_38_id_2641_3_attention_aten__transpose_min_sizes[] = {128, 2048, 12};
    unsigned g_0_tensor_38_id_2641_3_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_38_id_2641_3_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_38_id_2641_3_attention_aten__transpose_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_38_id_2641_3_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_34_id_2645_3_attention_aten__transpose tensor
    unsigned g_0_tensor_34_id_2645_3_attention_aten__transpose_max_sizes[] = {2048, 128, 12};
    unsigned g_0_tensor_34_id_2645_3_attention_aten__transpose_min_sizes[] = {2048, 128, 12};
    unsigned g_0_tensor_34_id_2645_3_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_34_id_2645_3_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_34_id_2645_3_attention_aten__transpose_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_34_id_2645_3_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_39_id_2647_3_attention_aten__bmm tensor
    unsigned g_0_tensor_39_id_2647_3_attention_aten__bmm_max_sizes[] = {2048, 2048, 12};
    unsigned g_0_tensor_39_id_2647_3_attention_aten__bmm_min_sizes[] = {2048, 2048, 12};
    unsigned g_0_tensor_39_id_2647_3_attention_aten__bmm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_39_id_2647_3_attention_aten__bmm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_39_id_2647_3_attention_aten__bmm_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_39_id_2647_3_attention_aten__bmm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_3_attention_batch_gemm_bf16_285_0_id;
    unsigned char g_0_3_attention_batch_gemm_bf16_285_0_params[] = {0, 0};
    addNodeToGraph(
        "batch_gemm",
        {g_0_tensor_38_id_2641_3_attention_aten__transpose, g_0_tensor_34_id_2645_3_attention_aten__transpose},
        {g_0_tensor_39_id_2647_3_attention_aten__bmm},
        (void*)g_0_3_attention_batch_gemm_bf16_285_0_params,
        2,
        "g_0_3_attention_batch_gemm_bf16_285_0",
        0 /*graphIndex*/,
        &g_0_3_attention_batch_gemm_bf16_285_0_id);

    /*************
     * g_0_3_attention_constant_f32_286_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_41[1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_41 tensor
    unsigned      g_0_tensor_41_max_sizes[] = {1};
    unsigned      g_0_tensor_41_min_sizes[] = {1};
    unsigned      g_0_tensor_41             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_41",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_41_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_41_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_3_attention_constant_f32_286_0_id;
    unsigned char g_0_3_attention_constant_f32_286_0_params[] = {243, 4, 181, 61};
    addNodeToGraph("constant_f32",
                   {},
                   {g_0_tensor_41},
                   (void*)g_0_3_attention_constant_f32_286_0_params,
                   4,
                   "g_0_3_attention_constant_f32_286_0",
                   0 /*graphIndex*/,
                   &g_0_3_attention_constant_f32_286_0_id);

    /*************
     * g_0_3_attention_cast_f32_to_bf16_287_0 node
     * inputs:
     *     g_0_tensor_41[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_42[1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_42 tensor
    unsigned      g_0_tensor_42_max_sizes[] = {1};
    unsigned      g_0_tensor_42_min_sizes[] = {1};
    unsigned      g_0_tensor_42             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_42",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_42_max_sizes,
                                           1,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_42_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_3_attention_cast_f32_to_bf16_287_0_id;
    unsigned char g_0_3_attention_cast_f32_to_bf16_287_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_41},
                   {g_0_tensor_42},
                   (void*)g_0_3_attention_cast_f32_to_bf16_287_0_params,
                   4,
                   "g_0_3_attention_cast_f32_to_bf16_287_0",
                   0 /*graphIndex*/,
                   &g_0_3_attention_cast_f32_to_bf16_287_0_id);

    /*************
     * g_0_3_attention_mult_fwd_bf16_288_0 node
     * inputs:
     *     g_0_tensor_39_id_2647_3_attention_aten__bmm[2048, 2048, 12] (dtype=bf16)
     *     g_0_tensor_42[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_40_id_2649_3_attention_aten__mul[2048, 2048, 12] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_40_id_2649_3_attention_aten__mul tensor
    unsigned g_0_tensor_40_id_2649_3_attention_aten__mul_max_sizes[] = {2048, 2048, 12};
    unsigned g_0_tensor_40_id_2649_3_attention_aten__mul_min_sizes[] = {2048, 2048, 12};
    unsigned g_0_tensor_40_id_2649_3_attention_aten__mul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_40_id_2649_3_attention_aten__mul",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_tensor_40_id_2649_3_attention_aten__mul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_40_id_2649_3_attention_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_attention_mult_fwd_bf16_288_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_39_id_2647_3_attention_aten__bmm, g_0_tensor_42},
                   {g_0_tensor_40_id_2649_3_attention_aten__mul},
                   nullptr,
                   0,
                   "g_0_3_attention_mult_fwd_bf16_288_0",
                   0 /*graphIndex*/,
                   &g_0_3_attention_mult_fwd_bf16_288_0_id);

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_40_id_2649_3_attention_aten__mul});
}

TEST_F_GC(SynTrainingTwoRunCompareTest,
          bgemm_with_producers_and_consumer_chain_ASIC_CI,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    // bundle taken from gpt3-fp8, mme consumer have granularity > 1

    // Graph #0

    // create g_0_tensor_1082 tensor
    unsigned g_0_tensor_1082_max_sizes[] = {3072, 6, 688};
    unsigned g_0_tensor_1082_min_sizes[] = {3072, 6, 688};
    unsigned g_0_tensor_1082             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1082",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1082_max_sizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1082_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1084 tensor
    unsigned g_0_tensor_1084_max_sizes[] = {3072, 6, 688};
    unsigned g_0_tensor_1084_min_sizes[] = {3072, 6, 688};
    unsigned g_0_tensor_1084             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_1084",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_1084_max_sizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1084_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1085 tensor
    unsigned      g_0_tensor_1085_max_sizes[] = {3072, 6, 688};
    unsigned      g_0_tensor_1085_min_sizes[] = {3072, 6, 688};
    unsigned      g_0_tensor_1085             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1085",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_1085_max_sizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1085_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_model_orig_model_orig_model_encoder_5_gelu_fwd_f32_639_0_id;
    unsigned char g_0_module_model_model_orig_model_orig_model_encoder_5_gelu_fwd_f32_639_0_params[] = {1};
    addNodeToGraph("gelu_fwd_f32",
                   {g_0_tensor_1082},
                   {g_0_tensor_1084, g_0_tensor_1085},
                   (void*)g_0_module_model_model_orig_model_orig_model_encoder_5_gelu_fwd_f32_639_0_params,
                   1,
                   "g_0_module_model_model_orig_model_orig_model_encoder_5_gelu_fwd_f32_639_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_model_orig_model_orig_model_encoder_5_gelu_fwd_f32_639_0_id);

    /*************
     * g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_640_0 node
     * inputs:
     *     g_0_tensor_1084[3072, 6, 688] (dtype=float32)
     * outputs:
     *     g_0_tensor_1087[3072, 6, 688] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1087 tensor
    unsigned      g_0_tensor_1087_max_sizes[] = {3072, 6, 688};
    unsigned      g_0_tensor_1087_min_sizes[] = {3072, 6, 688};
    unsigned      g_0_tensor_1087             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1087",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_1087_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1087_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_640_0_id;
    unsigned char g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_640_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_1084},
                   {g_0_tensor_1087},
                   (void*)g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_640_0_params,
                   4,
                   "g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_640_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_640_0_id);

    /*************
     * g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_641_0 node
     * inputs:
     *     g_0_tensor_1088__placeholder_0[3072, 768] (dtype=float32)
     * outputs:
     *     g_0_tensor_1090[3072, 768] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1088__placeholder_0 tensor
    unsigned g_0_tensor_1088__placeholder_0_max_sizes[] = {3072, 768};
    unsigned g_0_tensor_1088__placeholder_0_min_sizes[] = {3072, 768};
    unsigned g_0_tensor_1088__placeholder_0             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1088__placeholder_0",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1088__placeholder_0_max_sizes,
                                                            2,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1088__placeholder_0_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1090 tensor
    unsigned      g_0_tensor_1090_max_sizes[] = {3072, 768};
    unsigned      g_0_tensor_1090_min_sizes[] = {3072, 768};
    unsigned      g_0_tensor_1090             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1090",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_1090_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1090_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_641_0_id;
    unsigned char g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_641_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_1088__placeholder_0},
                   {g_0_tensor_1090},
                   (void*)g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_641_0_params,
                   4,
                   "g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_641_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_641_0_id);

    /*************
     * g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_642_0 node
     * inputs:
     *     g_0_tensor_1091__placeholder_0[768] (dtype=float32)
     * outputs:
     *     g_0_tensor_1093[768] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1091__placeholder_0 tensor
    unsigned g_0_tensor_1091__placeholder_0_max_sizes[] = {768};
    unsigned g_0_tensor_1091__placeholder_0_min_sizes[] = {768};
    unsigned g_0_tensor_1091__placeholder_0             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1091__placeholder_0",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1091__placeholder_0_max_sizes,
                                                            1,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1091__placeholder_0_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1093 tensor
    unsigned      g_0_tensor_1093_max_sizes[] = {768};
    unsigned      g_0_tensor_1093_min_sizes[] = {768};
    unsigned      g_0_tensor_1093             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1093",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_1093_max_sizes,
                                             1,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1093_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_642_0_id;
    unsigned char g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_642_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_1091__placeholder_0},
                   {g_0_tensor_1093},
                   (void*)g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_642_0_params,
                   4,
                   "g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_642_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_model_orig_model_orig_model_encoder_5_cast_f32_to_bf16_642_0_id);

    /*************
     * g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_reshape_643_0 node
     * inputs:
     *     g_0_tensor_1087[3072, 6, 688] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view[3072, 6, 688]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view tensor
    unsigned g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view_max_sizes[] = {
        3072,
        6,
        688};
    unsigned g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view_min_sizes[] = {
        3072,
        6,
        688};
    unsigned g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view",
        MEM_INIT_ALL_ZERO,
        nullptr,
        g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view_max_sizes,
        3,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_reshape_643_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_1087},
                   {g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view},
                   nullptr,
                   0,
                   "g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_reshape_643_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_reshape_643_0_id);

    /*************
     * g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_batch_gemm_644_0 node
     * inputs:
     *     g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view[3072, 6, 688]
     *(dtype=bf16) g_0_tensor_1090[3072, 768] (dtype=bf16) g_0_tensor_1093[768] (dtype=bf16) outputs:
     *     g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear[768, 6, 688]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear tensor
    unsigned g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear_max_sizes[] = {
        768,
        6,
        688};
    unsigned g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear_min_sizes[] = {
        768,
        6,
        688};
    unsigned g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_batch_gemm_644_0_id;
    unsigned char g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_batch_gemm_644_0_params[] = {0, 1};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_1094_id_6878_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__view,
                    g_0_tensor_1090,
                    g_0_tensor_1093},
                   {g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear},
                   (void*)g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_batch_gemm_644_0_params,
                   2,
                   "g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_batch_gemm_644_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_model_orig_model_orig_model_encoder_5_fc2_batch_gemm_644_0_id);

    /*************
     * g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_dropout_fwd_bf16_645_0 node
     * inputs:
     *     g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear[768, 6, 688]
     *(dtype=bf16) g_0_tensor_1096__placeholder_1[1] (dtype=int32) outputs:
     *     g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout[768,
     *6, 688] (dtype=bf16)
     *     g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout[768,
     *6, 688] (dtype=int8) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1096__placeholder_1 tensor
    unsigned g_0_tensor_1096__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_1096__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_1096__placeholder_1             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1096__placeholder_1",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1096__placeholder_1_max_sizes,
                                                            1,
                                                            syn_type_int32,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1096__placeholder_1_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout
    // tensor
    unsigned
        g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_max_sizes
            [] = {768, 6, 688};
    unsigned
        g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_min_sizes
            [] = {768, 6, 688};
    unsigned g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout
    // tensor
    unsigned
        g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_max_sizes
            [] = {768, 6, 688};
    unsigned
        g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_min_sizes
            [] = {768, 6, 688};
    unsigned g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_max_sizes,
            3,
            syn_type_int8,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_dropout_fwd_bf16_645_0_id;
    unsigned char g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_dropout_fwd_bf16_645_0_params[] =
        {205, 204, 204, 61, 138, 127, 0, 0};
    addNodeToGraph(
        "dropout_fwd_bf16",
        {g_0_tensor_1095_id_6888_module_model_model_orig_model_orig_model_encoder_5_fc2_aten__linear,
         g_0_tensor_1096__placeholder_1},
        {g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout,
         g_0_tensor_1098_id_6895_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout},
        (void*)g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_dropout_fwd_bf16_645_0_params,
        8,
        "g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_dropout_fwd_bf16_645_0",
        0 /*graphIndex*/,
        &g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_dropout_fwd_bf16_645_0_id);
    setNodeDeterminstic(g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_dropout_fwd_bf16_645_0_id);

    /*************
     * g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_cast_bf16_to_f32_646_0 node
     * inputs:
     *     g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout[768,
     *6, 688] (dtype=bf16) outputs: g_0_tensor_1100[768, 6, 688] (dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1100 tensor
    unsigned      g_0_tensor_1100_max_sizes[] = {768, 6, 688};
    unsigned      g_0_tensor_1100_min_sizes[] = {768, 6, 688};
    unsigned      g_0_tensor_1100             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1100",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_1100_max_sizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1100_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_cast_bf16_to_f32_646_0_id;
    unsigned char g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_cast_bf16_to_f32_646_0_params[] = {0,
                                                                                                                     0,
                                                                                                                     0,
                                                                                                                     0};
    addNodeToGraph(
        "cast_bf16_to_f32",
        {g_0_tensor_1097_id_6893_module_model_model_orig_model_orig_model_encoder_5_dropout3_hpu___fused_dropout},
        {g_0_tensor_1100},
        (void*)g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_cast_bf16_to_f32_646_0_params,
        4,
        "g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_cast_bf16_to_f32_646_0",
        0 /*graphIndex*/,
        &g_0_module_model_model_orig_model_orig_model_encoder_5_dropout3_cast_bf16_to_f32_646_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_BUNDLES_WITH_CONSUMERS_AND_PRODUCERS", "true");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    compareRunsResults({g_0_tensor_1100});
}