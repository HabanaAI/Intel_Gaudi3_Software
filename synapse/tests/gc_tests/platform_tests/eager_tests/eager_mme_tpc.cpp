#include "eager_tests_defs.h"
#include "data_type_utils.h"
#include "defs.h"
#include "node_annotation.h"
#include "syn_data_type_type_conversions.h"
#include "test_utils.h"
#include "types.h"
#include "node_factory.h"

#include "runtime/common/syn_singleton.hpp"

#include "gtest/gtest.h"

#include <memory>

using namespace eager_mode;

typedef enum
{
    OUT_0_IDX = 1,
    OUT_1_IDX = 0,
} mmeOutputsIdx;

class SynTrainingEagerTestsMmeTpc
: public SynTrainingEagerTests
, public testing::WithParamInterface<std::tuple<int /* stride */,
                                                int /* ifmC */,
                                                int /* ifm spatial size*/,
                                                int /* ifmB */,
                                                int /* ofmK */,
                                                ERepefenceOp /* op */>>
{
public:
    template<typename DType>
    void runMmeTest();
    void initTestParams();

private:
    synConvolutionParams                        m_convParams;
    std::array<unsigned, SYN_MAX_TENSOR_DIM> m_xSize, m_wSize, m_ySize;
    ERepefenceOp                                m_op;
};

void SynTrainingEagerTestsMmeTpc::initTestParams()
{
    m_convParams.kH         = 3;
    m_convParams.kW         = 3;
    m_convParams.dW         = std::get<0>(GetParam());
    m_convParams.dH         = std::get<0>(GetParam());
    unsigned ifmC           = std::get<1>(GetParam());
    unsigned ifmSpatialSize = std::get<2>(GetParam());
    unsigned batch          = std::get<3>(GetParam());
    unsigned ofmK           = std::get<4>(GetParam());
    m_op                    = std::get<5>(GetParam());

    m_xSize = {ifmC, ifmSpatialSize, ifmSpatialSize, batch, 1};
    m_wSize = {ofmK, ifmC, m_convParams.kW, m_convParams.kH, 1};
    m_ySize = {ofmK,
               convOutputDimSize(m_xSize[1],
                                 m_convParams.kW,
                                 m_convParams.dW,
                                 m_convParams.padL + m_convParams.padR,
                                 m_convParams.dilW),
               convOutputDimSize(m_xSize[2],
                                 m_convParams.kH,
                                 m_convParams.dH,
                                 m_convParams.padT + m_convParams.padB,
                                 m_convParams.dilH),
               batch,
               1};
}

template<typename DType>
void SynTrainingEagerTestsMmeTpc::runMmeTest()
{
    // Configuration relevant to running test in graph mode, they are not affect eager.
    // Disable all bundling algorithms, as the test does the bundling instead of the compiler.
    pushGlobalConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    pushGlobalConf("ENABLE_PIPELINE_MANAGEMENT", "false");
    // CDC causes memset to be added, this is not what the test intends to do
    pushGlobalConf("ENABLE_EAGER_MME_CONCURRENCY", "0");

    initTestParams();

    //
    // Create MME node
    //
    const synDataType dtype       = asSynType<DType>();
    const TensorUsage xUsage      = m_op == REFERENCE_OP_DEDX ? OUTPUT_TENSOR : INPUT_TENSOR;
    const MemInitType xInitSelect = xUsage == OUTPUT_TENSOR ? MEM_INIT_ALL_ZERO : MEM_INIT_RANDOM_WITH_NEGATIVE;
    const unsigned    xTensorIndex =
        createPersistTensor(xUsage, xInitSelect, nullptr, m_xSize.data(), m_xSize.size(), dtype);
    const TensorUsage wUsage      = m_op == REFERENCE_OP_DEDW ? OUTPUT_TENSOR : INPUT_TENSOR;
    const MemInitType wInitSelect = wUsage == OUTPUT_TENSOR ? MEM_INIT_ALL_ZERO : MEM_INIT_RANDOM_WITH_NEGATIVE;
    const unsigned    wTensorIndex =
        createPersistTensor(wUsage, wInitSelect, nullptr, m_wSize.data(), m_wSize.size(), dtype);
    const TensorUsage yUsage      = m_op == REFERENCE_OP_FWD ? OUTPUT_TENSOR : INPUT_TENSOR;
    const MemInitType yInitSelect = yUsage == OUTPUT_TENSOR ? MEM_INIT_ALL_ZERO : MEM_INIT_RANDOM_WITH_NEGATIVE;
    const unsigned    yTensorIndex =
        createPersistTensor(yUsage, yInitSelect, nullptr, m_ySize.data(), m_ySize.size(), dtype);

    std::string   guid           = "spatial_convolution";
    TensorIndices inputIndices   = {xTensorIndex, wTensorIndex};
    unsigned      mmeOutputIndex = yTensorIndex;
    auto          outDimSizes    = m_ySize;
    if (m_op == REFERENCE_OP_DEDW)
    {
        guid            = "dedw";
        inputIndices[0] = yTensorIndex;
        inputIndices[1] = xTensorIndex;
        mmeOutputIndex  = wTensorIndex;
        outDimSizes     = m_wSize;
    }
    else if (m_op == REFERENCE_OP_DEDX)
    {
        guid            = "dedx";
        inputIndices[0] = yTensorIndex;
        inputIndices[1] = wTensorIndex;
        mmeOutputIndex  = xTensorIndex;
        outDimSizes     = m_xSize;
    }

    addNodeToGraph(guid.c_str(), inputIndices, {mmeOutputIndex}, (void*)&m_convParams, sizeof(m_convParams));

    //
    // Create TPC node
    //
    unsigned tpcOutputIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outDimSizes.data(), outDimSizes.size(), dtype);
    const auto guidStr = fmt::format("neg_{}", getDtypeSuffixFromSynDataType(dtype));
    addNodeToGraph(guidStr.c_str(), {mmeOutputIndex}, {tpcOutputIndex});

    //
    // Run & validate
    //

    compileAndRun();

    // Check MME result
    synTensorDescriptor xDesc    = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc    = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc    = m_tensorDescs[yTensorIndex];
    DType*              xData    = castHostBuffer<DType>(xTensorIndex);
    DType*              wData    = castHostBuffer<DType>(wTensorIndex);
    DType*              yData    = castHostBuffer<DType>(yTensorIndex);
    DType*              mmeData  = castHostOutBuffer<DType>(mmeOutputIndex);
    CoordArray          wrongIdx = {0};
    bool                ret      = checkMmeOp(xDesc,
                          m_op == REFERENCE_OP_DEDX ? (char*)mmeData : (char*)xData,
                          wDesc,
                          m_op == REFERENCE_OP_DEDW ? (char*)mmeData : (char*)wData,
                          yDesc,
                          m_op == REFERENCE_OP_FWD ? (char*)mmeData : (char*)yData,
                          m_convParams,
                          m_op,
                          wrongIdx,
                          m_deviceType);
    ASSERT_EQ(ret, true) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',');

    // Check TPC result
    auto*    tpcData = castHostOutBuffer<DType>(tpcOutputIndex);
    unsigned length  = multiplyElements(outDimSizes.begin(), outDimSizes.end());
    for (int i = 0; i < length; i++)
    {
        ASSERT_FLOAT_EQ(mmeData[i], -tpcData[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    eager_mme_tpc_single,
    SynTrainingEagerTestsMmeTpc,
    ::testing::Combine(::testing::ValuesIn({1}),   // stride
                       ::testing::ValuesIn({16}),  // ifmC
                       ::testing::ValuesIn({64}),  // ifm spatial size
                       ::testing::ValuesIn({4}),   // ifmB
                       ::testing::ValuesIn({16}),  // ofmK
                       ::testing::ValuesIn({REFERENCE_OP_FWD, REFERENCE_OP_DEDX, REFERENCE_OP_DEDW})));

INSTANTIATE_TEST_SUITE_P(
    eager_mme_tpc_full_DAILY,
    SynTrainingEagerTestsMmeTpc,
    ::testing::Combine(::testing::ValuesIn({2, 3}),    // stride
                       ::testing::ValuesIn({8, 32}),   // ifmC
                       ::testing::ValuesIn({12, 45}),  // ifm spatial size
                       ::testing::ValuesIn({1, 6}),    // ifmB
                       ::testing::ValuesIn({8, 23}),   // ofmK
                       ::testing::ValuesIn({REFERENCE_OP_FWD, REFERENCE_OP_DEDX, REFERENCE_OP_DEDW})));

TEST_P_GC(SynTrainingEagerTestsMmeTpc, mme_tpc_f32)
{
    runMmeTest<float>();
}
