#include "global_conf_test_setter.h"
#include "gc_gaudi_test_infra.h"

TEST_F_GC(SynGaudiTestInfra, dedx_dedw_test_ASIC_CI)
{
    GlobalConfTestSetter gSet("SRAM_SLICER_MAX_CAPACITY_BYTES", "5000000");

    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 3;
    constexpr uint32_t yChannels = 512;
    TestSizes xSize = {64, 7, 7, 256, 1};
    TestSizes wSize = {yChannels, xSize[0], convParams.kW, convParams.kH, 1};
    TestSizes ySize = {yChannels,
                       convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
                       convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
                       xSize[3], 1};

    unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        ySize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_bf16);

    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr, // initializer
                                     wSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_bf16);

    unsigned dedw = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        wSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_float);

    unsigned x = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr, // initializer
                                     xSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_bf16);

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        xSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_bf16);

    addNodeToGraph("dedx", {dedy, w}, {dedx}, &convParams, sizeof(convParams));
    addNodeToGraph("dedw", {dedy, x}, {dedw}, &convParams, sizeof(convParams));

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[dedy];
    synTensorDescriptor xDesc = m_tensorDescs[x];
    synTensorDescriptor dedxDesc = m_tensorDescs[dedx];
    synTensorDescriptor wDesc = m_tensorDescs[w];
    synTensorDescriptor dedwDesc = m_tensorDescs[dedw];

    char* dedyData = (char*)m_hostBuffers[dedy];
    char* xData    = (char*)m_hostBuffers[x];
    char* dedxData = (char*)m_hostBuffers[dedx];
    char* wData    = (char*)m_hostBuffers[w];
    char* dedwData = (char*)m_hostBuffers[dedw];

    CoordArray wrongIdx;
    float expectedResult = 0;

    bool ret = checkMmeOp(dedxDesc,
                          dedxData,
                          wDesc,
                          wData,
                          dedyDesc,
                          dedyData,
                          convParams,
                          REFERENCE_OP_DEDX,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[dedx].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true) << "Wrong value for DEDX op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                << " Got value: " << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedx].m_dataType, m_hostBuffers[dedx])
                << " Expected: " << expectedResult;

    ret = checkMmeOp(xDesc,
                     xData,
                     dedwDesc,
                     dedwData,
                     dedyDesc,
                     dedyData,
                     convParams,
                     REFERENCE_OP_DEDW,
                     wrongIdx,
                     m_deviceType,
                     &expectedResult);

    castNcopy(sizes, m_tensorDescs[dedw].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true) << "Wrong value for DEDX op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                << " Got value: " << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedw].m_dataType, m_hostBuffers[dedw])
                << " Expected: " << expectedResult;
}
