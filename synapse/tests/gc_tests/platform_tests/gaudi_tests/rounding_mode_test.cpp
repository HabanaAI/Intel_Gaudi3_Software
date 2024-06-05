#include "gc_gaudi_test_infra.h"
#include "synapse_api.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "graph_entries_container.hpp"
#include "gtest/gtest.h"

class SynGaudiRoundingModeTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<synRoundingMode, synDataType>>
{
public:
    SynGaudiRoundingModeTest() {};
    MmeCommon::RoundingMode synapseToMmeRoundingMode(synRoundingMode synapseValue)
    {
        switch (synapseValue)
        {
            case synRoundingMode::synRoundToNearest:
                return MmeCommon::RoundingMode::RoundToNearest;
            case synRoundingMode::synRoundToZero:
                return MmeCommon::RoundingMode::RoundToZero;
            case synRoundingMode::synRoundUp:
                return MmeCommon::RoundingMode::RoundUp;
            case synRoundingMode::synRoundDown:
                return MmeCommon::RoundingMode::RoundDown;
            case synRoundingMode::synStochasticRounding:
                return MmeCommon::RoundingMode::StochasticRounding;
            case synRoundingMode::synRoundAwayFromZero:
                return MmeCommon::RoundingMode::RoundAwayFromZero;
            case synRoundingMode::synStochasticRoundingAndNearest:
                return MmeCommon::RoundingMode::StochasticRoundingAndNearest;
            default:
            {
                HB_ASSERT(false, "Not a valid synRoundingMode!");
                return MmeCommon::RoundingMode::RoundToNearest;
            }
        }
    };
    void runSingleTest()
    {
        unsigned        inSizes[]    = {1024, 320};
        unsigned        ySizes[]     = {1024, 1024};
        synRoundingMode roundingMode = std::get<0>(GetParam());
        synDataType     dataType     = std::get<1>(GetParam());
        unsigned        inIndex =
            createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSizes, 2, dataType);
        unsigned yIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 2, dataType);

        synGEMMParams  gemmParams(true /* transpose_a */, false /* transpose_b */);
        synGraphHandle graphHandle = getGraph(0).graphHandle;

        synNodeId nodeId;

        addNodeToGraph("gemm", {inIndex, inIndex}, {yIndex}, &gemmParams, sizeof(gemmParams), "gemm", 0, &nodeId);

        synTensorDescriptor xDesc = m_tensorDescs[inIndex];
        synTensorDescriptor wDesc = m_tensorDescs[inIndex];
        synTensorDescriptor yDesc = m_tensorDescs[yIndex];
        void*               xData = m_hostBuffers[inIndex];
        void*               wData = m_hostBuffers[inIndex];
        void*               yData = m_hostBuffers[yIndex];

        CoordArray wrongIdx       = {0};
        float      expectedResult = 0;

        synNodeSetRoundingMode(graphHandle, nodeId, roundingMode);
        compileAndRun();
        bool ret = checkBatchGemmOp(xDesc,
                                    (char*)xData,
                                    wDesc,
                                    (char*)wData,
                                    yDesc,
                                    (char*)yData,
                                    REFERENCE_OP_ATB,
                                    wrongIdx,
                                    (float*)yData,
                                    m_deviceType,
                                    synapseToMmeRoundingMode(roundingMode));
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, yDesc.m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                             << " Got value: " << getIndexValue(sizes, wrongIdx, yDesc.m_dataType, yData)
                             << " Expected: " << expectedResult;
    }
};

class SynGaudi2and3RoundingModeTest : public SynGaudiRoundingModeTest
{
public:
    SynGaudi2and3RoundingModeTest() : SynGaudiRoundingModeTest() {};
    MmeCommon::RoundingMode synapseToMmeRoundingMode(synRoundingMode synapseValue)
    {
        switch (synapseValue)
        {
            case synRoundingMode::synRoundToNearest:
                return MmeCommon::RoundingMode::RoundToNearest;
            case synRoundingMode::synRoundToZero:
                return MmeCommon::RoundingMode::RoundToZero;
            case synRoundingMode::synRoundUp:
                return MmeCommon::RoundingMode::RoundUp;
            case synRoundingMode::synRoundDown:
                return MmeCommon::RoundingMode::RoundDown;
            case synRoundingMode::synStochasticRounding:
                return MmeCommon::RoundingMode::StochasticRounding;
            case synRoundingMode::synRoundAwayFromZero:
                return MmeCommon::RoundingMode::RoundAwayFromZero;
            case synRoundingMode::synStochasticRoundingAndNearest:
                return MmeCommon::RoundingMode::StochasticRoundingAndNearest;
            default:
            {
                HB_ASSERT(false, "Not a valid synRoundingMode!");
                return MmeCommon::RoundingMode::RoundToNearest;
            }
        }
    }
};
class SynGaudi1RoundingModeTest : public SynGaudiRoundingModeTest
{
public:
    SynGaudi1RoundingModeTest() : SynGaudiRoundingModeTest() {};
    MmeCommon::RoundingMode synapseToMmeRoundingMode(synRoundingMode synapseValue)
    {
        switch (synapseValue)
        {
            case synRoundingMode::synRoundToNearest:
                return MmeCommon::RoundingMode::RoundToNearest;
            case synRoundingMode::synRoundToZero:
                return MmeCommon::RoundingMode::RoundToZero;
            case synRoundingMode::synRoundUp:
                return MmeCommon::RoundingMode::RoundUp;
            case synRoundingMode::synRoundDown:
                return MmeCommon::RoundingMode::RoundDown;

            default:
            {
                HB_ASSERT(false, "Not a valid synRoundingMode!");
                return MmeCommon::RoundingMode::RoundToNearest;
            }
        }
    }
};

// TODO: [SW-99164] enable the test for synDeviceGaudi3, blocking issue: syn_type_fp16
TEST_P_GC(SynGaudi2and3RoundingModeTest, roundingMode_mme, {'+', synDeviceGaudi2, '-', synDeviceGaudi3})
{
    runSingleTest();
}

TEST_P_GC(SynGaudi1RoundingModeTest, roundingMode_mme, {synDeviceGaudi})
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SynGaudi2and3RoundingModeTest,
    ::testing::Combine(

        ::testing::ValuesIn(
            {synRoundToNearest, synRoundToZero, synRoundUp, synRoundDown, synStochasticRounding, synRoundAwayFromZero}),
        ::testing::ValuesIn({syn_type_single, syn_type_bf16, syn_type_fp16, syn_type_fp8_143, syn_type_fp8_152})));

INSTANTIATE_TEST_SUITE_P(StochasticRoundingAndNearest,
                         SynGaudi2and3RoundingModeTest,
                         ::testing::Combine(

                             ::testing::ValuesIn({synStochasticRoundingAndNearest}),
                             ::testing::ValuesIn({syn_type_fp8_143, syn_type_fp8_152})));

INSTANTIATE_TEST_SUITE_P(,
                         SynGaudi1RoundingModeTest,
                         ::testing::Combine(

                             ::testing::ValuesIn({synRoundToNearest, synRoundToZero, synRoundUp, synRoundDown}),
                             ::testing::ValuesIn({syn_type_single, syn_type_bf16})));

TEST_F_GC(SynGaudi2and3RoundingModeTest, roundingMode_api, {synDeviceGaudi2})
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, synDeviceGaudi2)) << "Failed to create graph";

    synTensor in0, in1, out;
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&in0, graphHandle, DATA_TENSOR, "in0")) << "Failed to create tensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&in1, graphHandle, DATA_TENSOR, "in1")) << "Failed to create tensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&out, graphHandle, DATA_TENSOR, "out0")) << "Failed to create tensor";

    synTensorGeometry geometry {};
    geometry.dims = 1;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(in0, &geometry, synGeometryDims)) << "Failed synTensorSetGeometry";
    ASSERT_EQ(synSuccess, synTensorSetGeometry(in1, &geometry, synGeometryDims)) << "Failed synTensorSetGeometry";
    ASSERT_EQ(synSuccess, synTensorSetGeometry(out, &geometry, synGeometryDims)) << "Failed synTensorSetGeometry";

    synTensorDeviceLayout deviceLayout {};
    deviceLayout.deviceDataType = syn_type_fp8_152;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(in0, &deviceLayout)) << "Failed synTensorSetDeviceLayout";
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(in1, &deviceLayout)) << "Failed synTensorSetDeviceLayout";
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(out, &deviceLayout)) << "Failed synTensorSetDeviceLayout";

    synGEMMParams gemmParams(false /* transpose_a */, false /* transpose_b */);

    synTensor in[2] = {in0, in1};
    synNodeId gemmNodeId;
    ASSERT_EQ(synSuccess,
              synNodeCreateWithId(graphHandle,
                                  in,
                                  &out,
                                  2,
                                  1,
                                  &gemmParams,
                                  sizeof(gemmParams),
                                  "gemm",
                                  "my_gemm",
                                  &gemmNodeId,
                                  nullptr,
                                  nullptr))
        << "Failed to create node with GUID "
        << "gemm";

    synRoundingMode pRoundingMode;
    pRoundingMode = synRoundingMode::synRoundDown;
    ASSERT_EQ(synSuccess, synNodeGetRoundingMode(graphHandle, gemmNodeId, &pRoundingMode))
        << "synNodeGetRoundingMode Failed";
    ASSERT_EQ(synRoundingMode::synRoundToNearest, pRoundingMode) << "wrong default value";  // default value
    ASSERT_EQ(synSuccess, synNodeSetRoundingMode(graphHandle, gemmNodeId, synRoundingMode::synRoundDown))
        << "synNodeSetRoundingMode Failed";
    ASSERT_EQ(synSuccess, synNodeGetRoundingMode(graphHandle, gemmNodeId, &pRoundingMode))
        << "Failed to set RoundingMode";
    ASSERT_EQ(synRoundingMode::synRoundDown, pRoundingMode);
    const auto rc = synGraphDestroy(graphHandle);
    ASSERT_EQ(rc, synSuccess);
}
