#include "global_conf_test_setter.h"
#include "gc_gaudi_test_infra.h"

class SynGaudiTestDedxDynamic : public SynGaudiTestInfra
{
public:
    SynGaudiTestDedxDynamic()
    {
        GlobalConfTestSetter gSet("SRAM_SLICER_MAX_CAPACITY_BYTES", "5000000");
        // does not run on Gaudi2 yet
        setSupportedDevices({synDeviceGaudi});
    }

    void runTest(unsigned input_size, unsigned min_input_size, unsigned act_input_size, unsigned kernel_size)
    {
        synConvolutionParamsV2 convParams;

        convParams.kH = convParams.kW = kernel_size;
        convParams.dH = convParams.dW = 2;
        convParams.paddingType        = PADDING_SAME;

        constexpr uint32_t yChannels = 512;
        TestSizes          xSize     = {64, input_size, input_size, 256, 1};
        TestSizes          xSizeMin  = {64, min_input_size, min_input_size, 256, 1};
        TestSizes          xSizeAct  = {64, act_input_size, act_input_size, 256, 1};

        TestSizes wSize = {yChannels, xSize[0], convParams.kW, convParams.kH, 1};

        TestSizes ySize = {yChannels,
                           convOutputDimSizeSamePadding(xSize[1], convParams.dW),
                           convOutputDimSizeSamePadding(xSize[2], convParams.dH),
                           xSize[3],
                           1};

        TestSizes ySizeMin = {yChannels,
                              convOutputDimSizeSamePadding(xSizeMin[1], convParams.dW),
                              convOutputDimSizeSamePadding(xSizeMin[2], convParams.dH),
                              xSize[3],
                              1};

        TestSizes ySizeAct = {yChannels,
                              convOutputDimSizeSamePadding(xSizeAct[1], convParams.dW),
                              convOutputDimSizeSamePadding(xSizeAct[2], convParams.dH),
                              xSize[3],
                              1};

        // compute padding according to max size
        auto padX       = std::max(0U, (ySize[1] - 1) * convParams.dW + convParams.kW - xSize[1]);
        auto padY       = std::max(0U, (ySize[2] - 1) * convParams.dH + convParams.kH - xSize[2]);
        convParams.padL = padX / 2;
        convParams.padT = padY / 2;
        convParams.padR = padX - convParams.padL;
        convParams.padB = padY - convParams.padT;

        unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,  // initializer
                                            ySize.data(),
                                            DEFAULT_SIZES,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            ySizeMin.data());

        unsigned w = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,  // initializer
                                         wSize.data(),
                                         DEFAULT_SIZES,
                                         syn_type_bf16);

        unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,  // initializer
                                            xSize.data(),
                                            DEFAULT_SIZES,
                                            syn_type_bf16,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            xSizeMin.data());

        unsigned dxShape = createShapeTensor(INPUT_TENSOR,
                                             xSize.data(),
                                             xSizeMin.data(),
                                             DEFAULT_SIZES,
                                             syn_type_single,
                                             "dx_shape",
                                             0);

        addNodeToGraph("dedx", {dedy, w, dxShape}, {dedx}, &convParams, sizeof(convParams));

        compileTopology();

        setActualSizes(dedy, ySizeAct.data());
        setActualSizes(dedx, xSizeAct.data());
        setActualSizes(dxShape, xSizeAct.data());

        runTopology();

        // create fake descriptors for checking
        unsigned dedyFake = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,  // initializer
                                                ySizeAct.data(),
                                                DEFAULT_SIZES,
                                                syn_type_bf16);

        unsigned dedxFake = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,  // initializer
                                                xSizeAct.data(),
                                                DEFAULT_SIZES,
                                                syn_type_bf16);
        // fix conv params for checking
        //
        auto padXAct    = std::max(0U, (ySizeAct[1] - 1) * convParams.dW + convParams.kW - xSizeAct[1]);
        auto padYAct    = std::max(0U, (ySizeAct[2] - 1) * convParams.dH + convParams.kH - xSizeAct[2]);
        convParams.padL = padXAct / 2;
        convParams.padT = padYAct / 2;
        convParams.padR = padXAct - convParams.padL;
        convParams.padB = padYAct - convParams.padT;

        synTensorDescriptor dedyDesc = m_tensorDescs[dedyFake];
        synTensorDescriptor dedxDesc = m_tensorDescs[dedxFake];
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
                              REFERENCE_OP_DEDX,
                              wrongIdx,
                              m_deviceType,
                              &expectedResult);

        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[dedx].m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true)
            << "Wrong value for DEDX op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
            << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedx].m_dataType, m_hostBuffers[dedx])
            << " Expected: " << expectedResult;
    }
};

TEST_F_GC(SynGaudiTestDedxDynamic, DISABLED_same_padding_dedx_test_even_odd_small_padd)
{
    runTest(8, 4, 7, 3);
}

TEST_F_GC(SynGaudiTestDedxDynamic, DISABLED_same_padding_dedx_test_odd_even_small_pad)
{
    runTest(7, 4, 6, 3);
}

TEST_F_GC(SynGaudiTestDedxDynamic, DISABLED_same_padding_dedx_test_even_odd_large_pad)
{
    runTest(10, 5, 7, 5);
}

TEST_F_GC(SynGaudiTestDedxDynamic, DISABLED_same_padding_dedx_test_odd_even_large_pad)
{
    runTest(9, 5, 6, 5);
}
