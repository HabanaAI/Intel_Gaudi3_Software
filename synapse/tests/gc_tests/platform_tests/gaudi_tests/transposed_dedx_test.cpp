#include "global_conf_test_setter.h"
#include "gc_gaudi_test_infra.h"
#include "habana_global_conf.h"
#include "stubs/port.h"
#include "synapse_common_types.h"

class SynTrainingTransposedDedx
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>
{
};

TEST_P_GC(SynTrainingTransposedDedx, transposed_dedx_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    GlobalConfTestSetter gConvVar("ENABLE_INTERNAL_NODES", "true");
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = std::get<0>(GetParam());
    uint32_t  k                   = std::get<1>(GetParam());
    uint32_t  channelsIn          = std::get<2>(GetParam());
    uint32_t  spatial             = std::get<3>(GetParam());
    uint32_t  batch               = std::get<4>(GetParam());
    TestSizes xSize               = {channelsIn, spatial, spatial, batch};
    TestSizes wTSize              = {xSize[0], k, convParams.kW, convParams.kH};
    TestSizes ySize               = {
        k,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3]};

    unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        ySize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_bf16);

    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     wTSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_bf16);

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,  // initializer
                                        xSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_bf16);

    addNodeToGraph("transposed_dedx", {dedy, w}, {dedx}, &convParams, sizeof(convParams));

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[dedy];
    synTensorDescriptor dedxDesc = m_tensorDescs[dedx];
    synTensorDescriptor wDesc    = m_tensorDescs[w];

    char* dedyData = (char*)m_hostBuffers[dedy];
    char* dedxData = (char*)m_hostBuffers[dedx];
    char* wData    = (char*)m_hostBuffers[w];

    CoordArray wrongIdx;
    float      expectedResult = 0;

    bool ret = checkMmeOp(dedxDesc,
                          dedxData,
                          wDesc,
                          wData,
                          dedyDesc,
                          dedyData,
                          convParams,
                          REFERENCE_OP_TRANSPOSED_DEDX,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[dedx].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true) << "Wrong value for Transposed Dedx op at index: "
                         << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
                         << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedx].m_dataType, m_hostBuffers[dedx])
                         << " Expected: " << expectedResult;
}

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingTransposedDedx,
                         ::testing::Values(std::make_tuple(3, 2, 8, 7, 4), std::make_tuple(1, 3, 64, 14, 5)));