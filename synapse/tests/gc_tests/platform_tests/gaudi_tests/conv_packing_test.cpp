#include "gc_autogen_test.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "dynamic_shapes_types.h"
#include "syn_singleton.hpp"

class SynTrainingConvPackingTest
: public SynTrainingTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<int /* kernel size */,
                                                int /* stride */,
                                                int /* ifmC */,
                                                int /* ifm spatial size*/,
                                                int /* ifmB */,
                                                int /* ofmK */,
                                                bool /* 3dConv*/,
                                                synDataType /* data type */,
                                                bool /* isConst */>>
{
public:
    SynTrainingConvPackingTest();
    // returns the output tensor index to validate correctness
    virtual unsigned addNode() = 0;
    void             runSingleTest();
    virtual bool     blockTestForConvParams();

protected:
    unsigned m_filter;
    unsigned m_stride;
    unsigned m_nIFM;
    unsigned m_batch;
    unsigned m_nOFM;
    unsigned m_dilation;
    int      m_padBefore;
    int      m_padAfter;
    unsigned m_xWidth;  // the width can't be dynamic
    DimSizes m_xHeight;
    DimSizes m_xDepth;
    bool     m_3dConv;
    bool     m_const;

    synConvolutionParams m_params;

    unsigned m_yWidth;  // the width can't be dynamic
    DimSizes m_yHeight;
    DimSizes m_yDepth;

    ShapeSizes m_xSizes;
    ShapeSizes m_wSizes;
    ShapeSizes m_ySizes;

    unsigned m_tensorXIdx;
    unsigned m_tensorWIdx;
    unsigned m_tensorYIdx;

    synDataType m_type;
};

SynTrainingConvPackingTest::SynTrainingConvPackingTest()
: m_filter(std::get<0>(GetParam())),
  m_stride(std::get<1>(GetParam())),
  m_nIFM(std::get<2>(GetParam())),
  m_batch(std::get<4>(GetParam())),
  m_nOFM(std::get<5>(GetParam())),
  m_dilation(1),
  m_padBefore(0),
  m_padAfter(0),
  m_xWidth(std::get<3>(GetParam())),
  m_xHeight(m_xWidth),  // TODO SW-47038 - currenlty not testing dynamic
  m_xDepth(m_xWidth),
  m_3dConv(std::get<6>(GetParam())),
  m_const(std::get<8>(GetParam())),
  m_params(m_filter,
           m_filter,
           m_stride,
           m_stride,
           m_padBefore,
           m_padAfter,
           m_padBefore,
           m_padAfter,
           m_dilation,
           m_dilation),
  m_type(std::get<7>(GetParam()))
{
    m_yHeight.min    = convOutputDimSize(m_xHeight.min, m_filter, m_stride, m_padBefore + m_padAfter, m_dilation);
    m_yHeight.max    = convOutputDimSize(m_xHeight.max, m_filter, m_stride, m_padBefore + m_padAfter, m_dilation);
    m_yHeight.actual = convOutputDimSize(m_xHeight.actual, m_filter, m_stride, m_padBefore + m_padAfter, m_dilation);

    m_yWidth = convOutputDimSize(m_xWidth, m_filter, m_stride, m_padBefore + m_padAfter, m_dilation);
    if (m_3dConv)
    {
        m_yDepth.min    = convOutputDimSize(m_xDepth.min, m_filter, m_stride, m_padBefore + m_padAfter, m_dilation);
        m_yDepth.max    = convOutputDimSize(m_xDepth.max, m_filter, m_stride, m_padBefore + m_padAfter, m_dilation);
        m_yDepth.actual = convOutputDimSize(m_xDepth.actual, m_filter, m_stride, m_padBefore + m_padAfter, m_dilation);

        m_xSizes.min    = {m_nIFM, m_xWidth, m_xHeight.min, m_xDepth.min, m_batch};
        m_xSizes.max    = {m_nIFM, m_xWidth, m_xHeight.max, m_xDepth.max, m_batch};
        m_xSizes.actual = {m_nIFM, m_xWidth, m_xHeight.actual, m_xDepth.actual, m_batch};
        m_wSizes.actual = {m_nOFM, m_nIFM, m_filter, m_filter, m_filter};
        m_ySizes.min    = {m_nOFM, m_yWidth, m_yHeight.min, m_yDepth.min, m_batch};
        m_ySizes.max    = {m_nOFM, m_yWidth, m_yHeight.max, m_yDepth.max, m_batch};
        m_ySizes.actual = {m_nOFM, m_yWidth, m_yHeight.actual, m_yDepth.actual, m_batch};
    }
    else
    {
        m_xSizes.min    = {m_nIFM, m_xWidth, m_xHeight.min, m_batch};
        m_xSizes.max    = {m_nIFM, m_xWidth, m_xHeight.max, m_batch};
        m_xSizes.actual = {m_nIFM, m_xWidth, m_xHeight.actual, m_batch};
        m_wSizes.actual = {m_nOFM, m_nIFM, m_filter, m_filter};
        m_ySizes.min    = {m_nOFM, m_yWidth, m_yHeight.min, m_batch};
        m_ySizes.max    = {m_nOFM, m_yWidth, m_yHeight.max, m_batch};
        m_ySizes.actual = {m_nOFM, m_yWidth, m_yHeight.actual, m_batch};
    }

    setTestPackage(TEST_PACKAGE_CONV_PACKING);
}

bool SynTrainingConvPackingTest::blockTestForConvParams()
{
    return m_3dConv && ((m_xWidth > 64 || m_nIFM > 64) && m_batch > 2);
}

static bool isPackingNode(NodePtr node)
{
    bool                     isPacking = false;
    std::shared_ptr<TPCNode> tpcNode   = std::dynamic_pointer_cast<TPCNode>(node);
    if (tpcNode)
    {
        std::string_view guidName = tpcNode->getGUIDWithoutDtype();
        isPacking                 = guidName == "conv_weight_packing_fwd";
    }
    return isPacking;
}

void SynTrainingConvPackingTest::runSingleTest()
{
    GCFG_ENABLE_WEIGHT_PACKING_CONSTANT_FOLDING.setValue(true);
    if (blockTestForConvParams())
    {
        return;
    }
    // std::cout << "IFM spatial = " << m_xWidth << "; in channels = " << m_nIFM << "; out channels = " << m_nOFM << ";
    // batch = " << m_batch << ";" << std::endl; std::cout << "filter = " << m_filter << "; stride = " << m_stride << ";
    // dilation = " << m_dilation  << std::endl;

    unsigned tensorToValidateIdx = addNode();

    if (m_const)
    {
        pushGlobalConf("ENABLE_CONSTANT_FOLDING", "true");
    }

    addConfigurationToRun(FIRST_RUN, "ENABLE_CONV_PACKING_TRAINING", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_CONV_PACKING_TRAINING", "false");
    compareRunsResults({tensorToValidateIdx});
    if (m_const)
    {
        auto&        graphData = getGraph(FIRST_RUN);
        HabanaGraph* g         = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);
        for (const NodePtr& n : g->getNodes())
        {
            ASSERT_FALSE(isPackingNode(n)) << "test expect to fold conv packing TPC node";
        }
    }
}

class SynTrainingConvFwdPackingTest : public SynTrainingConvPackingTest
{
public:
    SynTrainingConvFwdPackingTest() : SynTrainingConvPackingTest() {}
    unsigned addNode() override;
};

class SynTrainingConvFwdPackingTestLong : public SynTrainingConvFwdPackingTest
{
public:
    SynTrainingConvFwdPackingTestLong() : SynTrainingConvFwdPackingTest() {}
};

unsigned SynTrainingConvFwdPackingTest::addNode()
{
    unsigned graphIndex = 0;

    m_tensorXIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_xSizes.max.data(),
                                       m_xSizes.max.size(),
                                       m_type,
                                       nullptr,
                                       "X",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_xSizes.min.data());
    if (m_const)
    {
        setGraphInferenceMode();
        m_tensorWIdx = createConstPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                m_wSizes.actual.data(),
                                                m_wSizes.actual.size(),
                                                m_type,
                                                nullptr,
                                                "W",
                                                0,
                                                0,
                                                nullptr);
    }
    else
    {
        m_tensorWIdx = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           m_wSizes.actual.data(),
                                           m_wSizes.actual.size(),
                                           m_type,
                                           nullptr,
                                           "W");
    }

    m_tensorYIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_ySizes.max.data(),
                                       m_ySizes.max.size(),
                                       m_type,
                                       nullptr,
                                       "Y",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_ySizes.min.data());
    if (m_3dConv)
    {
        synConvolution3DParams conv3dParams(m_params);
        conv3dParams.kernel[CONV_KERNEL_DEPTH] = m_filter;
        conv3dParams.stride[CONV_STRIDE_DEPTH] = m_stride;
        conv3dParams.dilation[CONV_DIL_DEPTH]  = m_dilation;
        conv3dParams.padding[CONV_PAD_FRONT]   = m_padBefore;
        conv3dParams.padding[CONV_PAD_BACK]    = m_padAfter;
        addNodeToGraph(NodeFactory::convolution3DNodeTypeName,
                       {m_tensorXIdx, m_tensorWIdx},
                       {m_tensorYIdx},
                       &conv3dParams,
                       sizeof(conv3dParams),
                       "conv3d_fwd",
                       graphIndex);
    }
    else
    {
        addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                       {m_tensorXIdx, m_tensorWIdx},
                       {m_tensorYIdx},
                       &m_params,
                       sizeof(m_params),
                       "FwdConvolution");
    }
    return m_tensorYIdx;
}

TEST_P_GC(SynTrainingConvFwdPackingTest, pack_conv_fwd)
{
    runSingleTest();
}

// [SW-86940] Enable long packing test on Gaudi1
TEST_P_GC(SynTrainingConvFwdPackingTestLong, pack_conv_fwd_DAILY, {synDeviceGaudi2, synDeviceGaudi3})
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_f32_sanity,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_float, false),
                                           std::make_tuple(5, 2, 8, 11, 1, 7, false, syn_type_float, false),
                                           std::make_tuple(9, 3, 87, 24, 2, 14, true, syn_type_float, false),
                                           std::make_tuple(9, 5, 64, 64, 12, 8, false, syn_type_float, false)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_f32_ASIC_CI,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_float, false),
                                           std::make_tuple(4, 1, 3, 57, 8, 8, true, syn_type_float, false),
                                           std::make_tuple(4, 2, 128, 112, 5, 5, false, syn_type_float, false),
                                           std::make_tuple(4, 3, 64, 50, 3, 13, true, syn_type_float, false)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_bf16_sanity,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_bf16, false),
                                           std::make_tuple(5, 2, 8, 11, 1, 7, false, syn_type_bf16, false),
                                           std::make_tuple(9, 3, 87, 24, 2, 14, true, syn_type_bf16, false),
                                           std::make_tuple(9, 5, 64, 64, 12, 8, false, syn_type_bf16, false)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_bf16_ASIC_CI,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_bf16, false),
                                           std::make_tuple(4, 1, 3, 57, 8, 8, true, syn_type_bf16, false),
                                           std::make_tuple(4, 2, 128, 112, 5, 5, false, syn_type_bf16, false),
                                           std::make_tuple(4, 3, 64, 50, 3, 13, true, syn_type_bf16, false)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_const_f32_sanity,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_float, true),
                                           std::make_tuple(5, 2, 8, 11, 1, 7, false, syn_type_float, true),
                                           std::make_tuple(9, 3, 87, 24, 2, 14, true, syn_type_float, true),
                                           std::make_tuple(9, 5, 64, 64, 12, 8, false, syn_type_float, true)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_const_f32_ASIC_CI,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_float, true),
                                           std::make_tuple(4, 1, 3, 57, 8, 8, true, syn_type_float, true),
                                           std::make_tuple(4, 2, 128, 112, 5, 5, false, syn_type_float, true),
                                           std::make_tuple(4, 3, 64, 50, 3, 13, true, syn_type_float, true)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_const_bf16_sanity,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_bf16, true),
                                           std::make_tuple(5, 2, 8, 11, 1, 7, false, syn_type_bf16, true),
                                           std::make_tuple(9, 3, 87, 24, 2, 14, true, syn_type_bf16, true),
                                           std::make_tuple(9, 5, 64, 64, 12, 8, false, syn_type_bf16, true)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_const_bf16_ASIC_CI,
                         SynTrainingConvFwdPackingTest,
                         ::testing::Values(std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_bf16, true),
                                           std::make_tuple(4, 1, 3, 57, 8, 8, true, syn_type_bf16, true),
                                           std::make_tuple(4, 2, 128, 112, 5, 5, false, syn_type_bf16, true),
                                           std::make_tuple(4, 3, 64, 50, 3, 13, true, syn_type_bf16, true)));

INSTANTIATE_TEST_SUITE_P(conv_2D_fwd_packing_full_ASIC_CI,
                         SynTrainingConvFwdPackingTestLong,
                         ::testing::Combine(::testing::ValuesIn({2, 4}),            // kernel size
                                            ::testing::Range(1, 4),                 // stride
                                            ::testing::ValuesIn({3, 64, 87, 128}),  // ifmC
                                            ::testing::ValuesIn({14, 56}),          // ifm spatial size
                                            ::testing::ValuesIn({1, 2, 16}),        // ifmB
                                            ::testing::ValuesIn({1, 3, 16, 64}),    // ofmK
                                            ::testing::ValuesIn({false}),           // 3dConv
                                            ::testing::ValuesIn({syn_type_bf16}),
                                            ::testing::ValuesIn({false, true})));  // const

INSTANTIATE_TEST_SUITE_P(conv_3D_fwd_packing_full_ASIC_CI,
                         SynTrainingConvFwdPackingTestLong,
                         ::testing::Combine(::testing::ValuesIn({1, 3, 5}),   // kernel size
                                            ::testing::Range(1, 3),           // stride
                                            ::testing::ValuesIn({1, 5, 32}),  // ifmC
                                            ::testing::ValuesIn({7, 112}),    // ifm spatial size
                                            ::testing::ValuesIn({1, 2, 4}),   // ifmB
                                            ::testing::ValuesIn({1, 2, 32}),  // ofmK
                                            ::testing::ValuesIn({true}),      // 3dConv
                                            ::testing::ValuesIn({syn_type_float}),
                                            ::testing::ValuesIn({false, true})));  // const

class SynTrainingConvDedxPackingTest : public SynTrainingConvPackingTest
{
public:
    SynTrainingConvDedxPackingTest() : SynTrainingConvPackingTest() {}
    unsigned addNode() override;
    bool     blockTestForConvParams() override;
};

class SynTrainingConvDedxPackingTestLong : public SynTrainingConvDedxPackingTest

{
public:
    SynTrainingConvDedxPackingTestLong() : SynTrainingConvDedxPackingTest() {}
};

bool SynTrainingConvDedxPackingTest::blockTestForConvParams()
{
    // block also dedx with stride and dilation which have common divisor, due to MME stack bug - SW-23339
    return ((gcd(m_stride, m_dilation) != 1) || SynTrainingConvPackingTest::blockTestForConvParams());
}

unsigned SynTrainingConvDedxPackingTest::addNode()
{
    m_tensorYIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_ySizes.actual.data(),
                                       m_ySizes.actual.size(),
                                       m_type,
                                       nullptr,
                                       "dY");

    m_tensorWIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_wSizes.actual.data(),
                                       m_wSizes.actual.size(),
                                       m_type,
                                       nullptr,
                                       "W");

    m_tensorXIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_xSizes.actual.data(),
                                       m_xSizes.actual.size(),
                                       m_type,
                                       nullptr,
                                       "dX");

    if (m_3dConv)
    {
        synConvolution3DParams conv3dParams(m_params);
        conv3dParams.kernel[CONV_KERNEL_DEPTH] = m_filter;
        conv3dParams.stride[CONV_STRIDE_DEPTH] = m_stride;
        conv3dParams.dilation[CONV_DIL_DEPTH]  = m_dilation;
        conv3dParams.padding[CONV_PAD_FRONT]   = m_padBefore;
        conv3dParams.padding[CONV_PAD_BACK]    = m_padAfter;

        addNodeToGraph(NodeFactory::deDx3DNodeTypeName,
                       {m_tensorYIdx, m_tensorWIdx},
                       {m_tensorXIdx},
                       &conv3dParams,
                       sizeof(conv3dParams),
                       "dedx3d");
    }
    else
    {
        addNodeToGraph(NodeFactory::deDxNodeTypeName,
                       {m_tensorYIdx, m_tensorWIdx},
                       {m_tensorXIdx},
                       &m_params,
                       sizeof(m_params),
                       "dEdX");
    }
    return m_tensorXIdx;
}

INSTANTIATE_TEST_SUITE_P(conv_dedx_packing_f32_sanity,
                         SynTrainingConvDedxPackingTest,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_float, false),
                                           std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_float, false),
                                           std::make_tuple(5, 1, 8, 6, 1, 7, false, syn_type_float, false)));

INSTANTIATE_TEST_SUITE_P(conv_dedx_packing_bf16_sanity,
                         SynTrainingConvDedxPackingTest,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_bf16, false),
                                           std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_bf16, false),
                                           std::make_tuple(5, 1, 8, 6, 1, 7, false, syn_type_bf16, false),
                                           std::make_tuple(9, 1, 87, 24, 2, 14, true, syn_type_bf16, false)));

INSTANTIATE_TEST_SUITE_P(conv_dedx_packing_f32_ASIC_CI,
                         SynTrainingConvDedxPackingTest,
                         ::testing::Values(std::make_tuple(9, 1, 64, 64, 12, 8, false, syn_type_float, false),
                                           std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_float, false),
                                           std::make_tuple(4, 1, 3, 56, 8, 8, true, syn_type_float, false),
                                           std::make_tuple(4, 1, 128, 112, 5, 5, false, syn_type_float, false),
                                           std::make_tuple(4, 1, 64, 18, 3, 13, true, syn_type_float, false),
                                           std::make_tuple(9, 1, 87, 24, 2, 14, true, syn_type_float, false)));

INSTANTIATE_TEST_SUITE_P(conv_dedx_packing_bf16_ASIC_CI,
                         SynTrainingConvDedxPackingTest,
                         ::testing::Values(std::make_tuple(9, 1, 64, 64, 12, 8, false, syn_type_bf16, false),
                                           std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_bf16, false),
                                           std::make_tuple(4, 1, 3, 56, 8, 8, true, syn_type_bf16, false),
                                           std::make_tuple(4, 1, 128, 112, 5, 5, false, syn_type_bf16, false),
                                           std::make_tuple(4, 1, 64, 18, 3, 13, true, syn_type_bf16, false)));

INSTANTIATE_TEST_SUITE_P(conv_2D_dedx_packing_full_ASIC_CI,
                         SynTrainingConvDedxPackingTest,
                         ::testing::Combine(::testing::ValuesIn({1, 2, 5}),       // kernel size
                                            ::testing::ValuesIn({1}),             // stride
                                            ::testing::ValuesIn({1, 64, 128}),    // ifmC
                                            ::testing::ValuesIn({7, 14, 56}),     // ifm spatial size
                                            ::testing::ValuesIn({1, 2, 16}),      // ifmB
                                            ::testing::ValuesIn({1, 2, 16, 32}),  // ofmK
                                            ::testing::ValuesIn({false}),         // 3dConv
                                            ::testing::ValuesIn({syn_type_float}),
                                            ::testing::ValuesIn({false})));  // const

INSTANTIATE_TEST_SUITE_P(conv_3D_dedx_packing_full_ASIC_CI,
                         SynTrainingConvDedxPackingTestLong,
                         ::testing::Combine(::testing::ValuesIn({3, 5}),          // kernel size
                                            ::testing::ValuesIn({1}),             // stride
                                            ::testing::ValuesIn({1, 3, 32, 87}),  // ifmC
                                            ::testing::ValuesIn({28, 112}),       // ifm spatial size
                                            ::testing::ValuesIn({1, 2, 4}),       // ifmB
                                            ::testing::ValuesIn({1, 3, 16, 32}),  // ofmK
                                            ::testing::ValuesIn({true}),          // 3dConv
                                            ::testing::ValuesIn({syn_type_bf16}),
                                            ::testing::ValuesIn({false})));  // const

TEST_P_GC(SynTrainingConvDedxPackingTest, pack_conv_dedx)
{
    runSingleTest();
}

// [SW-86940] Enable long packing test on Gaudi1
TEST_P_GC(SynTrainingConvDedxPackingTestLong, pack_conv_dedx_DAILY, {synDeviceGaudi2})
{
    runSingleTest();
}

// Different class to fp8 because it is tested only in Gaudi2
class SynTrainingConvDedxPackingFloat8Test : public SynTrainingConvDedxPackingTest
{
public:
    SynTrainingConvDedxPackingFloat8Test() : SynTrainingConvDedxPackingTest() {}
};

INSTANTIATE_TEST_SUITE_P(conv_dedx_packing_f8_sanity,
                         SynTrainingConvDedxPackingFloat8Test,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_fp8_152, false),
                                           std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_fp8_152, false),
                                           std::make_tuple(5, 1, 8, 6, 1, 7, false, syn_type_fp8_152, false),
                                           std::make_tuple(9, 1, 87, 24, 2, 14, true, syn_type_fp8_152, false)));

INSTANTIATE_TEST_SUITE_P(conv_dedx_packing_f8_ASIC_CI,
                         SynTrainingConvDedxPackingFloat8Test,
                         ::testing::Values(std::make_tuple(9, 1, 64, 64, 12, 8, false, syn_type_fp8_152, false),
                                           std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_fp8_152, false),
                                           std::make_tuple(4, 1, 3, 56, 8, 8, true, syn_type_fp8_152, false),
                                           std::make_tuple(4, 1, 128, 112, 5, 5, false, syn_type_fp8_152, false),
                                           std::make_tuple(4, 1, 64, 18, 3, 13, true, syn_type_fp8_152, false)));

INSTANTIATE_TEST_SUITE_P(conv_2D_dedx_packing_full_ASIC_CI,
                         SynTrainingConvDedxPackingFloat8Test,
                         ::testing::Combine(::testing::ValuesIn({2, 5}),        // kernel size
                                            ::testing::ValuesIn({1}),           // stride
                                            ::testing::ValuesIn({1, 3, 128}),   // ifmC
                                            ::testing::ValuesIn({7, 56, 112}),  // ifm spatial size
                                            ::testing::ValuesIn({1, 5, 16}),    // ifmB
                                            ::testing::ValuesIn({1, 2, 32}),    // ofmK
                                            ::testing::ValuesIn({false}),       // 3dConv
                                            ::testing::ValuesIn({syn_type_fp8_152}),
                                            ::testing::ValuesIn({false})));  // const

INSTANTIATE_TEST_SUITE_P(conv_3D_dedx_packing_full_ASIC_CI,
                         SynTrainingConvDedxPackingFloat8Test,
                         ::testing::Combine(::testing::ValuesIn({1, 3}),       // kernel size
                                            ::testing::ValuesIn({1}),          // stride
                                            ::testing::ValuesIn({1, 64, 87}),  // ifmC
                                            ::testing::ValuesIn({14, 55}),     // ifm spatial size
                                            ::testing::ValuesIn({1, 2, 32}),   // ifmB
                                            ::testing::ValuesIn({1, 3, 16}),   // ofmK
                                            ::testing::ValuesIn({true}),       // 3dConv
                                            ::testing::ValuesIn({syn_type_fp8_152}),
                                            ::testing::ValuesIn({false})));  // const

TEST_P_GC(SynTrainingConvDedxPackingFloat8Test, pack_conv_dedx, {synDeviceGaudi2})
{
    runSingleTest();
}

// Different class to fp8 because it is tested only in Gaudi2
class SynTrainingConvFwdPackingFloat8Test : public SynTrainingConvFwdPackingTest
{
public:
    SynTrainingConvFwdPackingFloat8Test() : SynTrainingConvFwdPackingTest() {}
};

// TDDO: Test not stable in CI [SW-94810]
TEST_P_GC(SynTrainingConvFwdPackingFloat8Test, DISABLED_pack_conv_fwd, {synDeviceGaudi2})
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_f8_sanity,
                         SynTrainingConvFwdPackingFloat8Test,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_fp8_152, false),
                                           std::make_tuple(5, 2, 8, 11, 1, 7, false, syn_type_fp8_152, false),
                                           std::make_tuple(9, 3, 87, 24, 2, 14, true, syn_type_fp8_152, false),
                                           std::make_tuple(9, 5, 64, 64, 12, 8, false, syn_type_fp8_152, false)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_f8_ASIC_CI,
                         SynTrainingConvFwdPackingFloat8Test,
                         ::testing::Values(std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_fp8_152, false),
                                           std::make_tuple(4, 1, 3, 57, 8, 8, true, syn_type_fp8_152, false),
                                           std::make_tuple(4, 2, 128, 112, 5, 5, false, syn_type_fp8_152, false),
                                           std::make_tuple(4, 3, 64, 50, 3, 13, true, syn_type_fp8_152, false)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_const_f8_sanity,
                         SynTrainingConvFwdPackingFloat8Test,
                         ::testing::Values(std::make_tuple(3, 1, 3, 6, 1, 1, false, syn_type_fp8_152, true),
                                           std::make_tuple(5, 2, 8, 11, 1, 7, false, syn_type_fp8_152, true),
                                           std::make_tuple(9, 3, 87, 24, 2, 14, true, syn_type_fp8_152, true),
                                           std::make_tuple(9, 5, 64, 64, 12, 8, false, syn_type_fp8_152, true)));

INSTANTIATE_TEST_SUITE_P(conv_fwd_packing_const_f8_ASIC_CI,
                         SynTrainingConvFwdPackingFloat8Test,
                         ::testing::Values(std::make_tuple(9, 1, 21, 112, 4, 3, false, syn_type_fp8_152, true),
                                           std::make_tuple(4, 1, 3, 57, 8, 8, true, syn_type_fp8_152, true),
                                           std::make_tuple(4, 2, 128, 112, 5, 5, false, syn_type_fp8_152, true),
                                           std::make_tuple(4, 3, 64, 50, 3, 13, true, syn_type_fp8_152, true)));

INSTANTIATE_TEST_SUITE_P(conv_2D_fwd_packing_full_ASIC_CI,
                         SynTrainingConvFwdPackingFloat8Test,
                         ::testing::Combine(::testing::ValuesIn({2, 3}),           // kernel size
                                            ::testing::Range(1, 3),                // stride
                                            ::testing::ValuesIn({1, 3, 87, 128}),  // ifmC
                                            ::testing::ValuesIn({14, 56}),         // ifm spatial size
                                            ::testing::ValuesIn({1, 2, 16}),       // ifmB
                                            ::testing::ValuesIn({1, 3, 16}),       // ofmK
                                            ::testing::ValuesIn({false}),          // 3dConv
                                            ::testing::ValuesIn({syn_type_fp8_152}),
                                            ::testing::ValuesIn({false, true})));  // const

INSTANTIATE_TEST_SUITE_P(conv_3D_fwd_packing_full_ASIC_CI,
                         SynTrainingConvFwdPackingFloat8Test,
                         ::testing::Combine(::testing::ValuesIn({1, 3, 5}),   // kernel size
                                            ::testing::Range(1, 4),           // stride
                                            ::testing::ValuesIn({1, 5, 32}),  // ifmC
                                            ::testing::ValuesIn({7, 112}),    // ifm spatial size
                                            ::testing::ValuesIn({1, 5, 4}),   // ifmB
                                            ::testing::ValuesIn({2, 3, 32}),  // ofmK
                                            ::testing::ValuesIn({true}),      // 3dConv
                                            ::testing::ValuesIn({syn_type_fp8_152}),
                                            ::testing::ValuesIn({false, true})));  // const
