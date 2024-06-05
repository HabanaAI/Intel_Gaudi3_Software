#include <memory>
#include "gc_gaudi_test_infra.h"
#include "node_operations_test.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "node_factory.h"

SynTrainingNodeOperations::SynTrainingNodeOperations()
{
    setTestPackage(TEST_PACKAGE_NODE_OPERATIONS);
    setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
}

void SynTrainingNodeOperations::SetUpTest()
{
    SynTrainingTestInfra::SetUpTest();
    pushGlobalConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "4000000");
}

void SynTrainingNodeOperations::TearDownTest()
{
    SynTrainingTestInfra::TearDownTest();
}

void SynTrainingNodeOperations::runMmeTest(const TestSizes&            xSize,
                                           const TestSizes&            wSize,
                                           const TestSizes&            ySize,
                                           const synConvolutionParams& params,
                                           ERepefenceOp                op,
                                           synDataType                 dtype,
                                           synDataType                 outputDType,
                                           bool                        usePearsonCompare)
{
    TensorUsage xUsage = op == REFERENCE_OP_DEDX ? OUTPUT_TENSOR : INPUT_TENSOR;
    TensorUsage wUsage = op == REFERENCE_OP_DEDW ? OUTPUT_TENSOR : INPUT_TENSOR;
    TensorUsage yUsage = op == REFERENCE_OP_FWD ? OUTPUT_TENSOR : INPUT_TENSOR;

    synDataType outputDataType = (outputDType == syn_type_na ? dtype : outputDType);
    synDataType xDataType      = op == REFERENCE_OP_DEDX ? outputDataType : dtype;
    synDataType wDataType      = op == REFERENCE_OP_DEDW ? outputDataType : dtype;
    synDataType yDataType      = op == REFERENCE_OP_FWD ? outputDataType : dtype;

    unsigned      xTensorIndex  = createPersistTensor(xUsage,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                (unsigned*)xSize.data(),
                                                DEFAULT_SIZES /* dims */,
                                                xDataType,
                                                nullptr,
                                                "x");
    unsigned      wTensorIndex  = createPersistTensor(wUsage,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                (unsigned*)wSize.data(),
                                                DEFAULT_SIZES /* dims */,
                                                wDataType,
                                                nullptr,
                                                "w");
    unsigned      yTensorIndex  = createPersistTensor(yUsage,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                (unsigned*)ySize.data(),
                                                DEFAULT_SIZES /* dims */,
                                                yDataType,
                                                nullptr,
                                                "y");
    std::string guid = "spatial_convolution";
    TensorIndices inputIndices = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};
    if (op == REFERENCE_OP_DEDW)
    {
        guid = "dedw";
        inputIndices[0] = yTensorIndex;
        inputIndices[1] = xTensorIndex;
        outputIndices[0] = wTensorIndex;
    }
    else if (op == REFERENCE_OP_DEDX)
    {
        guid = "dedx";
        inputIndices[0] = yTensorIndex;
        inputIndices[1] = wTensorIndex;
        outputIndices[0] = xTensorIndex;
    }
    unsigned paramSize = sizeof(params);
    addNodeToGraph(guid.c_str(), inputIndices, outputIndices, (void*)&params, paramSize);
    compileAndRun();

    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];
    void*               xData = m_hostBuffers[xTensorIndex];
    void*               wData = m_hostBuffers[wTensorIndex];
    void*               yData = m_hostBuffers[yTensorIndex];

    CoordArray wrongIdx       = {0};
    float      expectedResult = 0;
    bool       ret            = checkMmeOp(xDesc,
                          (char*)xData,
                          wDesc,
                          (char*)wData,
                          yDesc,
                          (char*)yData,
                          params,
                          op,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult,
                          usePearsonCompare);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[outputIndices[0]].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                         << " Got value: "
                         << getIndexValue(sizes,
                                          wrongIdx,
                                          m_tensorDescs[outputIndices[0]].m_dataType,
                                          m_hostBuffers[outputIndices[0]])
                         << " Expected: " << expectedResult;
}

void SynTrainingNodeOperations::runMmeTest(const TestSizes&            xSize,
                                           const unsigned int          yChannels,
                                           const synConvolutionParams& convParams,
                                           ERepefenceOp                op,
                                           synDataType                 dataType,
                                           synDataType                 outputDataType,
                                           bool                        usePearsonCompare)
{
    TestSizes wSize = {yChannels, xSize[0], convParams.kW, convParams.kH, 1};
    TestSizes ySize = {yChannels,
                       convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
                       convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
                       xSize[3], 1};

    runMmeTest(xSize, wSize, ySize, convParams, op, dataType, outputDataType, usePearsonCompare);
}

TEST_F_GC(SynTrainingNodeOperations, simple_fwd_L2)
{
    synConvolutionParams    convParams;
    convParams.kH = 1;
    convParams.kW = 1;
    TestSizes xSize = {2048,1,1,64,1};
    runMmeTest(xSize, 1000, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingNodeOperations, fwd_align_opt1)
{
    synConvolutionParams convParams;
    convParams.kW   = 4;
    convParams.kH   = 3;
    convParams.padL = 1;  // required. produces an offset of 1 * 16
    convParams.padT = 1;
    convParams.dW   = 4;  // stride of 4 gives steps of 4*16*2 bytes, 1 full CL
    convParams.dH   = 2;
    TestSizes xSize = {16, 200, 120, 1, 1};  // single misalignment point in offset of 16*1 elements (32/128 bytes)
    runMmeTest(xSize, 580, convParams, REFERENCE_OP_FWD, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, fwd_align_opt2)
{
    synConvolutionParams convParams;
    convParams.kH   = 4;
    convParams.kW   = 7;
    convParams.padL = 2;  // required. produces an offset of 2 * 8
    convParams.padT = 3;
    convParams.dW   = 4;  // Stride of 4 gives steps of 4*8*4=128 bytes, 1 full CL
    convParams.dH   = 2;
    TestSizes xSize = {8, 120, 80, 1, 1};  // single misalignment point in offset of 8*2 elements (64/128 bytes)
    runMmeTest(xSize, 400, convParams, REFERENCE_OP_FWD);  // fp32: 4 bytes
}

TEST_F_GC(SynTrainingNodeOperations, fwd_align_opt3)
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;  // required. produces an offset of 1 * 48
    convParams.padT = 1;
    convParams.dW   = 4;  // Stride of 4 gives steps of 4*48*2=384 bytes which are 3 full CLs
    convParams.dH   = 2;
    TestSizes xSize = {48, 200, 100, 1, 1};  // single misalignment point in offset of 48*1 elements (96/128 bytes)
    runMmeTest(xSize, 300, convParams, REFERENCE_OP_FWD, syn_type_bf16);  // bf16: 2 bytes
}

TEST_F_GC(SynTrainingNodeOperations,
          CDConcurrency_exceedsSramWhenReductionInFp32_ASIC,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {1536, 49, 49, 1, 1};
    runMmeTest(xSize, 2048, convParams, REFERENCE_OP_DEDW, syn_type_bf16);
}

// In this dedw test the spatial dims after lowering are 32x96. Both cd and batch concurrency can
// provide 4x. Cd concurrency is of higher acceleration because the appropriate filter size is 3.
TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_unet3d_dedw3d_n5550_bundle_86_op_4_bf16, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 1;
    convParams.padB = 1;
    TestSizes xSize = {32, 128, 128, 1, 1};
    runMmeTest(xSize, 32, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

// In this dedw test the output spatial dims after lowering are 32x12. Geometry wise, both cd and batch concurrency
// can provide 8x. Cd wins because batch concurrency cannot utilize the 8x
TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_unet3d_dedw3d_n3188_bundle_66_op_13_fp16, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 1;
    convParams.padB = 1;
    TestSizes xSize = {4, 128, 128, 1, 1};
    runMmeTest(xSize, 32, convParams, REFERENCE_OP_DEDW, syn_type_fp16, syn_type_fp16);
}

TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_unet3d_dedw3d_n3388_bundle_68_op_1_bf16, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 1;
    convParams.padB = 1;
    TestSizes xSize = {32, 128, 128, 1, 1};
    runMmeTest(xSize, 64, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_baseTest_EvenCD_4x_bf16, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    TestSizes            xSize = {13, 10, 2, 5, 1};
    runMmeTest(xSize, 20, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_baseTest_OddCD_4x_fp16, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 1;
    convParams.padB = 1;
    TestSizes xSize = {33, 240, 6, 13, 1};
    runMmeTest(xSize, 22, convParams, REFERENCE_OP_DEDW, syn_type_fp16, syn_type_fp16);
}

TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_fullGeo_2x, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 2;
    convParams.padB = 1;
    TestSizes xSize = {64, 3, 3, 4, 1};
    runMmeTest(xSize, 256, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_WithReduction, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 1;
    convParams.padB = 1;
    TestSizes xSize = {128, 96, 96, 4, 1};
    runMmeTest(xSize, 128, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, CDConcurrency_WithReduction2, {synDeviceGaudi2, synDeviceGaudi3})
{

    unsigned k=32, c=32, b=4, w=64, h=64, pearsonCheck = false;
    auto flagStr = getenv("K_VAL");
    if (flagStr != nullptr)
    {
        k = std::atoi(flagStr);
    }
    flagStr = getenv("C_VAL");
    if (flagStr != nullptr)
    {
        c = std::atoi(flagStr);
    }
    flagStr = getenv("B_VAL");
    if (flagStr != nullptr)
    {
        b = std::atoi(flagStr);
    }
    flagStr = getenv("W_VAL");
    if (flagStr != nullptr)
    {
        w = std::atoi(flagStr);
    }
    flagStr = getenv("H_VAL");
    if (flagStr != nullptr)
    {
        h = std::atoi(flagStr);
    }
    flagStr = getenv("PEARSON_CHECK");
    if (flagStr != nullptr)
    {
        pearsonCheck = std::atoi(flagStr);
    }

    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 1;
    convParams.padB = 1;
    TestSizes xSize = {c, w, h, b, 1};
    runMmeTest(xSize, k, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16, pearsonCheck ? true : false);
}

TEST_F_GC(SynTrainingNodeOperations, dedw)
{
    synConvolutionParams    convParams;
    convParams.kH = 2;
    convParams.kW = 2;
    TestSizes xSize = {1,3,3,1,1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingNodeOperations, dedwFp8CdConcurrency4x_basic, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {3, 40, 10, 2, 1};
    runMmeTest(xSize, 64, convParams, REFERENCE_OP_DEDW, syn_type_fp8_152);
}

TEST_F_GC(SynTrainingNodeOperations, dedwFp8CdConcurrency4x_basic_fp32, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {3, 40, 10, 2, 1};
    runMmeTest(xSize, 64, convParams, REFERENCE_OP_DEDW, syn_type_fp8_152, syn_type_single);
}

TEST_F_GC(SynTrainingNodeOperations, dedwFp8CdConcurrency4x_3x3kernel, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    TestSizes xSize = {3, 20, 10, 2, 1};
    runMmeTest(xSize, 32, convParams, REFERENCE_OP_DEDW, syn_type_fp8_152, syn_type_single);
}

TEST_F_GC(SynTrainingNodeOperations, dedwFp8CdConcurrency4x_3x3kernel_strides_padding, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padT = 1;
    convParams.padB = 1;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.dW   = 2;
    convParams.dH   = 2;
    TestSizes xSize = {3, 20, 10, 2, 1};
    runMmeTest(xSize, 32, convParams, REFERENCE_OP_DEDW, syn_type_fp8_152);
}

TEST_F_GC(SynTrainingNodeOperations, dedwFp8CdConcurrency4x_large, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padT = 1;
    convParams.padB = 1;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.dW   = 2;
    convParams.dH   = 2;
    TestSizes xSize = {42, 20, 10, 2, 1};
    runMmeTest(xSize, 128, convParams, REFERENCE_OP_DEDW, syn_type_fp8_152);
}

// This test is expected not to activate the optimization because after owering output size is too larget
TEST_F_GC(SynTrainingNodeOperations, dedwFp8CdConcurrency4x_tooLarge, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padT = 1;
    convParams.padB = 1;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.dW   = 2;
    convParams.dH   = 2;
    TestSizes xSize = {128, 20, 10, 2, 1};
    runMmeTest(xSize, 128, convParams, REFERENCE_OP_DEDW, syn_type_fp8_152, syn_type_single);
}

TEST_F_GC(SynTrainingNodeOperations, dedwFp8CdConcurrency4x_resnet50_layer1_conv1, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH   = 7;
    convParams.kW   = 7;
    convParams.padT = 3;
    convParams.padB = 3;
    convParams.padL = 3;
    convParams.padR = 3;
    convParams.dW   = 2;
    convParams.dH   = 2;
    TestSizes xSize = {3, 224, 224, 1, 2};
    runMmeTest(xSize, 64, convParams, REFERENCE_OP_DEDW, syn_type_fp8_152);
}

TEST_F_GC(SynTrainingNodeOperations, dedwAsBgemm_fp32)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {30, 250, 25, 8, 1};
    runMmeTest(xSize, 25, convParams, REFERENCE_OP_DEDW, syn_type_single);
}

TEST_F_GC(SynTrainingNodeOperations, dedwAsBgemm_bf16)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {60, 250, 25, 8, 1};
    runMmeTest(xSize, 50, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, dedwAsBgemmWide_fp32)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {30, 250, 25, 8, 1};
    runMmeTest(xSize, 50, convParams, REFERENCE_OP_DEDW, syn_type_single, syn_type_single, true);
}

TEST_F_GC(SynTrainingNodeOperations, dedwAsBgemmWide_bf16)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {60, 250, 25, 8, 1};
    runMmeTest(xSize, 100, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, dedwAsBgemmFull_fp32)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {32, 250, 25, 8, 1};
    runMmeTest(xSize, 64, convParams, REFERENCE_OP_DEDW, syn_type_single, syn_type_single, true);
}

TEST_F_GC(SynTrainingNodeOperations, dedwAsBgemmFull_bf16)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {64, 250, 25, 8, 1};
    runMmeTest(xSize, 128, convParams, REFERENCE_OP_DEDW, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, dedx)
{
    synConvolutionParams    convParams;
    convParams.kH = 3;
    convParams.kW = 3;
    convParams.padT = 1;
    convParams.padB = 1;
    convParams.padL = 1;
    convParams.padR = 1;
    TestSizes xSize = {1,5,5,1,1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingNodeOperations, dedw_strided)
{
    synConvolutionParams    convParams;
    convParams.kH = 2;
    convParams.kW = 2;
    convParams.dW = 2;
    convParams.dH = 2;
    convParams.padT = 1;
    convParams.padB = 1;
    convParams.padL = 1;
    convParams.padR = 1;
    TestSizes xSize = {1,4,4,1,1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingStridedOperations, dedx_strided)
{
    synConvolutionParams    convParams;
    convParams.kH = 2;
    convParams.kW = 2;
    convParams.dW = 2;
    convParams.dH = 2;

    TestSizes xSize = {1,4,4,1,1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingNodeOperations, fwd_one_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 1;
    TestSizes xSize = {1, 1, 6, 1, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingNodeOperations, fwd_split_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 3;
    TestSizes xSize = {1, 6, 1, 4, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingNodeOperations, fwd_all_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {1, 1, 1, 4, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingNodeOperations, dedx_one_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 1;
    TestSizes xSize = {1, 1, 6, 1, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingNodeOperations, dedx_split_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 3;
    TestSizes xSize = {1, 6, 1, 4, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingNodeOperations, dedx_all_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {1, 1, 1, 4, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingNodeOperations, dedw_one_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 1;
    TestSizes xSize = {1, 1, 6, 1, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingNodeOperations, dedw_split_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 3;
    TestSizes xSize = {1, 6, 1, 4, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingNodeOperations, dedw_all_sp_trivial_dim)
{
    synConvolutionParams convParams;
    convParams.kH   = 1;
    convParams.kW   = 1;
    TestSizes xSize = {1, 1, 1, 4, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingNodeOperations, dedb)
{
    const unsigned dims = 4;
    unsigned ifmDimSizes[] = {2, 3, 1, 1};
    unsigned ofmDimSizes[] = {1, 3, 1, 1};

    const float ifmBuffer[] = {-1.0, -1.0,
                                1.0,  3.0,
                                4.0,  5.0};

    const float ofmRefBuffer[] = {-2.0, 4.0, 9.0};

    ns_Reduction::Params params;
    params.reductionDimension = 0; // sum and reduce first dimension size to 1

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmBuffer, ifmDimSizes, dims,
                                                syn_type_single);
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims,
                                                syn_type_single);

    TensorIndices inputIndices  = {xTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph("reduce_sum_fwd_f32", inputIndices, outputIndices, (void*)&params, sizeof(ns_Reduction::Params));
    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes); i++)
    {
        ASSERT_EQ(*pOutputBuffer, ofmRefBuffer[i]) << "Mismatch at index " << i
                                                             << " Result:" << *pOutputBuffer
                                                             << " Ref: " << ofmRefBuffer[i];
        pOutputBuffer++;
    }
}

void SynTrainingGemmTest::doGemmTest(std::array<unsigned, 2> aSize,
                                     std::array<unsigned, 2> bSize,
                                     bool                    transposeA,
                                     bool                    transposeB,
                                     synDataType             dtype)
{
    const unsigned dims          = 2;
    unsigned       aDimSizes[]   = {aSize[0], aSize[1], 1, 1};
    unsigned       bDimSizes[]   = {bSize[0], bSize[1], 1, 1};
    unsigned       outDimSizes[] = {transposeB ? bSize[1] : bSize[0], transposeA ? aSize[0] : aSize[1], 1, 1};

    unsigned aTensorIndex =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, aDimSizes, dims, dtype);
    unsigned bTensorIndex =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, bDimSizes, dims, dtype);
    unsigned outTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outDimSizes, dims, dtype);

    TensorIndices inputIndices  = {aTensorIndex, bTensorIndex};
    TensorIndices outputIndices = {outTensorIndex};

    synGEMMParams params;
    params.transpose_a = transposeA;
    params.transpose_b = transposeB;

    addNodeToGraph("gemm", inputIndices, outputIndices, (void*)&params, sizeof(synGEMMParams));
    compileAndRun();

    ERepefenceOp op;
    if (transposeA)
    {
        if (transposeB)
        {
            op = REFERENCE_OP_ATBT;
        }
        else
        {
            op = REFERENCE_OP_ATB;
        }
    }
    else
    {
        if (transposeB)
        {
            op = REFERENCE_OP_ABT;
        }
        else
        {
            op = REFERENCE_OP_AB;
        }
    }

    void* matAVal = m_hostBuffers[aTensorIndex];
    void* matBVal = m_hostBuffers[bTensorIndex];
    void* matCVal = m_hostBuffers[outTensorIndex];

    synTensorDescriptor aDesc   = m_tensorDescs[aTensorIndex];
    synTensorDescriptor bDesc   = m_tensorDescs[bTensorIndex];
    synTensorDescriptor outDesc = m_tensorDescs[outTensorIndex];

    CoordArray wrongIdx       = {0};
    float      expectedResult = 0;
    bool       ret            = checkBatchGemmOp(aDesc,
                                (char*)matAVal,
                                bDesc,
                                (char*)matBVal,
                                outDesc,
                                (char*)matCVal,
                                op,
                                wrongIdx,
                                &expectedResult,
                                m_deviceType);

    if (!ret)
    {
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[outputIndices[0]].m_sizes, SYN_MAX_TENSOR_DIM);
        EXPECT_FLOAT_EQ(getIndexValue(sizes,
                                      wrongIdx,
                                      m_tensorDescs[outputIndices[0]].m_dataType,
                                      m_hostBuffers[outputIndices[0]]),
                        expectedResult);
    }
}

TEST_F_GC(SynTrainingGemmTest, gemm_test_256_256)
{
    doGemmTest(std::array<unsigned,2>({128,256}),
               std::array<unsigned,2>({512,128}),
               false, false);
}

TEST_F_GC(SynTrainingGemmTest, gemm_test_256_256_fp16, {synDeviceGaudi2, synDeviceGaudi3})
{
    doGemmTest(std::array<unsigned, 2>({128, 256}), std::array<unsigned, 2>({512, 128}), false, false, syn_type_fp16);
}

TEST_F_GC(SynTrainingGemmTest, gemm_test_256_256_transpose_a)
{
    doGemmTest(std::array<unsigned,2>({129,255}),
               std::array<unsigned,2>({513,255}),
               true, false);
}

TEST_F_GC(SynTrainingGemmTest, gemm_test_256_256_transpose_b)
{
    doGemmTest(std::array<unsigned,2>({255,129}),
               std::array<unsigned,2>({255,513}),false, true);
}

TEST_F_GC(SynTrainingStridedOperations, dedx_with_stride_and_padding_symmetric)
{
    synConvolutionParams convParams;
    convParams.dH   = 2;
    convParams.dW   = 2;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.dilH = 1;
    convParams.dilW = 1;
    convParams.setPadT(1);
    convParams.setPadB(1);
    convParams.setPadL(1);
    convParams.setPadR(1);

    TestSizes xSize = {1, 3, 3, 1, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingStridedOperations, dedx_with_stride_and_padding_asymmetric)
{
    synConvolutionParams convParams;
    convParams.dH   = 2;
    convParams.dW   = 2;
    convParams.kH   = 3;
    convParams.kW   = 4;
    convParams.dilH = 1;
    convParams.dilW = 1;
    convParams.setPadT(1);
    convParams.setPadB(1);
    convParams.setPadL(1);
    convParams.setPadR(1);

    TestSizes xSize = {1, 7, 8, 1, 1};
    runMmeTest(xSize, 1, convParams, REFERENCE_OP_DEDX);
}

static void resetIrrelevantDims( synTensorDescriptor &desc )
{
    for (int i=desc.m_dims; i<SYN_MAX_TENSOR_DIM; i++ )
    {
        desc.m_sizes[i] = 1;
    }
}

void SynTrainingBatchGemmTest::doBatchGemmTest(const TestSizes& xSize,
                                               const TestSizes& wSize,
                                               const unsigned   xRank,
                                               const unsigned   wRank,
                                               ERepefenceOp     op,
                                               synDataType      inputDataType,
                                               synDataType      outputDataType,
                                               TestSizes*       optYSizes)
{
    std::shared_ptr<synGEMMParams> params = std::make_shared<synGEMMParams>();
    switch (op)
    {
        case REFERENCE_OP_AB:
        case REFERENCE_OP_FWD:
        case REFERENCE_OP_DEDX:
        case REFERENCE_OP_DEDW:
            break;
        case REFERENCE_OP_ABT:
            params->transpose_b = true;
            break;
        case REFERENCE_OP_ATB:
            params->transpose_a = true;
            break;
        case REFERENCE_OP_ATBT:
            params->transpose_a = true;
            params->transpose_b = true;
            break;
        default:
            break;
    }
    TestSizes ySize;
    if (!optYSizes)
    {
        ySize    = xSize;
        ySize[0] = params->transpose_b ? wSize[1] : wSize[0];
        ySize[1] = params->transpose_a ? xSize[0] : xSize[1];
        for (int dim = DIM_GEMM_BATCH; dim < MAX_DIMENSIONS_NUM; dim++)
        {
            if (ySize[dim] == 1) ySize[dim] = wSize[dim];
        }
    }
    else
    {
        ySize = *optYSizes;
    }
    TensorUsage xUsage = INPUT_TENSOR;
    TensorUsage wUsage = INPUT_TENSOR;
    TensorUsage yUsage = OUTPUT_TENSOR;
    switch (op)
    {
        case REFERENCE_OP_AB:
        case REFERENCE_OP_ABT:
        case REFERENCE_OP_ATB:
        case REFERENCE_OP_ATBT:
            xUsage = INPUT_TENSOR;
            wUsage = INPUT_TENSOR;
            yUsage = OUTPUT_TENSOR;
            break;
        case REFERENCE_OP_FWD:
            xUsage = INPUT_TENSOR;
            wUsage = INPUT_TENSOR;
            yUsage = OUTPUT_TENSOR;
            params = nullptr;
            break;
        case REFERENCE_OP_DEDX:
        case REFERENCE_OP_TRANSPOSED_DEDX:
            xUsage = OUTPUT_TENSOR;
            wUsage = INPUT_TENSOR;
            yUsage = INPUT_TENSOR;
            params = nullptr;
            break;
        case REFERENCE_OP_DEDW:
            xUsage = INPUT_TENSOR;
            wUsage = OUTPUT_TENSOR;
            yUsage = INPUT_TENSOR;
            params = nullptr;
            break;
    }

    unsigned xTensorIndex = createPersistTensor(xUsage,
                                                MEM_INIT_RANDOM_POSITIVE,
                                                nullptr,
                                                (unsigned*)xSize.data(),
                                                xRank,
                                                (REFERENCE_OP_DEDX == op ? outputDataType : inputDataType));
    unsigned wTensorIndex = createPersistTensor(wUsage,
                                                MEM_INIT_RANDOM_POSITIVE,
                                                nullptr,
                                                (unsigned*)wSize.data(),
                                                wRank,
                                                (REFERENCE_OP_DEDW == op ? outputDataType : inputDataType));
    unsigned yTensorIndex =
        createPersistTensor(yUsage,
                            MEM_INIT_RANDOM_POSITIVE,
                            nullptr,
                            (unsigned*)ySize.data(),
                            xRank,
                            (REFERENCE_OP_DEDX == op || REFERENCE_OP_DEDW == op ? inputDataType : outputDataType));
    std::string   guid          = "batch_gemm";
    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};
    if (op == REFERENCE_OP_DEDW)
    {
        guid += "_dedw";
        inputIndices[0]  = yTensorIndex;
        inputIndices[1]  = xTensorIndex;
        outputIndices[0] = wTensorIndex;
    }
    else if (op == REFERENCE_OP_DEDX)
    {
        guid += "_dedx";
        inputIndices[0]  = yTensorIndex;
        inputIndices[1]  = wTensorIndex;
        outputIndices[0] = xTensorIndex;
    }

    unsigned paramSize = params ? sizeof(*params) : 0;
    addNodeToGraph(guid.c_str(), inputIndices, outputIndices, (void*)params.get(), paramSize);

    compileAndRun();

    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];
    // todo AlonG: check with Omer for the better way to reset the irrelevant dims
    resetIrrelevantDims(xDesc);
    resetIrrelevantDims(wDesc);
    resetIrrelevantDims(yDesc);
    void* xData = m_hostBuffers[xTensorIndex];
    void* wData = m_hostBuffers[wTensorIndex];
    void* yData = m_hostBuffers[yTensorIndex];

    CoordArray wrongIdx       = {0};
    float      expectedResult = 0;
    bool       ret            = checkBatchGemmOp(xDesc,
                                (char*)xData,
                                wDesc,
                                (char*)wData,
                                yDesc,
                                (char*)yData,
                                op,
                                wrongIdx,
                                &expectedResult,
                                m_deviceType);

    if (!ret)
    {
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[outputIndices[0]].m_sizes, SYN_MAX_TENSOR_DIM);
        EXPECT_FLOAT_EQ(getIndexValue(sizes,
                                      wrongIdx,
                                      m_tensorDescs[outputIndices[0]].m_dataType,
                                      m_hostBuffers[outputIndices[0]]),
                        expectedResult);
    }
}

void SynTrainingBatchGemmTest::doBatchGemmTest(const TestSizes& xSize,
                                               const TestSizes& wSize,
                                               const unsigned   rank,
                                               ERepefenceOp     op,
                                               synDataType      inputDataType,
                                               synDataType      outputDataType,
                                               TestSizes*       optYSizes)
{
    doBatchGemmTest(xSize, wSize, rank, rank, op, inputDataType, outputDataType, optYSizes);
}

// ---------------- Flatenning tests ----------------
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_ab_bf16_basic)
{
    doBatchGemmTest(TestSizes({32, 96, 12, 1, 1}),
                    TestSizes({16, 32, 1, 1, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_ab_fp32_basic)
{
    doBatchGemmTest(TestSizes({32, 96, 12, 1, 1}),
                    TestSizes({16, 32, 1, 1, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_float,
                    syn_type_float);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_abt_bf16_basic)
{
    doBatchGemmTest(TestSizes({65, 26, 12, 1, 1}),
                    TestSizes({65, 16, 1, 1, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_abt_fp32_basic)
{
    doBatchGemmTest(TestSizes({512, 96, 4, 1, 1}),
                    TestSizes({512, 16, 1, 1, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_float,
                    syn_type_float);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_ab_bf16_full)
{
    doBatchGemmTest(TestSizes({32, 96, 12, 2, 3}),
                    TestSizes({16, 32, 1, 2, 3}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_ab_fp32_full)
{
    doBatchGemmTest(TestSizes({63, 6, 12, 1, 4}),
                    TestSizes({65, 63, 1, 1, 4}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_float,
                    syn_type_float);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_abt_bf16_full_double_broadcast)
{
    doBatchGemmTest(TestSizes({15, 26, 12, 3, 2}),
                    TestSizes({15, 16, 1, 3, 1}),
                    5,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_flatenning_abt_fp32_full_broadcast)
{
    doBatchGemmTest(TestSizes({512, 96, 4, 1, 2}),
                    TestSizes({512, 16, 1, 2, 1}),
                    5,
                    REFERENCE_OP_ABT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingNodeOperations, flattening_fwd_fp32)
{
    synConvolutionParams convParams;  // Default parameters are ok for conversion to gemm
    TestSizes            xSize = {16, 12, 36, 24, 1};
    runMmeTest(xSize, 10, convParams, REFERENCE_OP_FWD, syn_type_single);
}

TEST_F_GC(SynTrainingNodeOperations, flattening_fwd_bf16)
{
    synConvolutionParams convParams;
    TestSizes            xSize = {160, 112, 6, 4, 1};
    runMmeTest(xSize, 180, convParams, REFERENCE_OP_FWD, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, flattening_dedw_bf16)
{
    synConvolutionParams convParams;
    TestSizes            xSize = {256, 1, 24, 4, 1};
    runMmeTest(xSize, 320, convParams, REFERENCE_OP_DEDW, syn_type_bf16);
}

TEST_F_GC(SynTrainingNodeOperations, flattening_dedw_fp32)
{
    synConvolutionParams convParams;
    TestSizes            xSize = {200, 2, 1, 64, 1};
    runMmeTest(xSize, 32, convParams, REFERENCE_OP_DEDW, syn_type_single);
}

TEST_F_GC(SynTrainingNodeOperations, flattening_dedx_fp32)
{
    synConvolutionParams convParams;
    TestSizes            xSize = {216, 6, 1, 4, 1};
    runMmeTest(xSize, 225, convParams, REFERENCE_OP_DEDX, syn_type_single);
}

TEST_F_GC(SynTrainingNodeOperations, flattening_dedx_bf16)
{
    synConvolutionParams convParams;
    TestSizes            xSize = {225, 600, 20, 1, 1};
    runMmeTest(xSize, 6, convParams, REFERENCE_OP_DEDX, syn_type_bf16);
}

// ----------------------- AtBt tests -------------------------

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_fp32_1_asym)
{
    doBatchGemmTest(TestSizes({64, 31, 2, 4, 1}),
                    TestSizes({31, 80, 1, 1, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_bf16_1_asym)
{
    doBatchGemmTest(TestSizes({8, 12, 3, 1, 1}),
                    TestSizes({12, 6, 1, 1, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_fp32_2)
{
    doBatchGemmTest(TestSizes({137, 61, 5, 2, 1}),
                    TestSizes({61, 51, 5, 2, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_bf16_2_L2)
{
    doBatchGemmTest(TestSizes({127, 161, 3, 4, 1}),
                    TestSizes({161, 251, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_64x128_fp32_aligned)
{
    doBatchGemmTest(TestSizes({128, 128, 1, 3, 1}),
                    TestSizes({128, 64, 1, 3, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_128x128_bf_unaligned_L2)
{
    doBatchGemmTest(TestSizes({101, 128, 3, 2, 1}),
                    TestSizes({128, 101, 3, 2, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_fp32_dualLoops)
{
    doBatchGemmTest(TestSizes({260, 66, 1, 3, 1}),
                    TestSizes({66, 300, 1, 3, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_fp32_4w1h)
{
    doBatchGemmTest(TestSizes({5, 66, 1, 3, 1}),
                    TestSizes({66, 512, 1, 3, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_fp32_1w4h)
{
    doBatchGemmTest(TestSizes({300, 66, 1, 3, 1}),
                    TestSizes({66, 5, 1, 3, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_bf16_2xw)
{
    doBatchGemmTest(TestSizes({100, 66, 4, 3, 1}),
                    TestSizes({66, 30, 4, 3, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_fp32_2xh)
{
    doBatchGemmTest(TestSizes({60, 66, 4, 3, 1}),
                    TestSizes({66, 200, 4, 3, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_fp32_2xh_2)
{
    doBatchGemmTest(TestSizes({60, 66, 4, 3, 1}),
                    TestSizes({66, 120, 4, 3, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

// ----------------------- AtB tests -------------------------

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_bf16_1)
{
    doBatchGemmTest(TestSizes({127, 161, 3, 4, 1}),
                    TestSizes({51, 161, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_bf16_5_dims)
{
    doBatchGemmTest(TestSizes({127, 161, 3, 4, 2}),
                    TestSizes({51, 161, 3, 4, 2}),
                    5,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_fp32_1_L2)
{
    doBatchGemmTest(TestSizes({27, 161, 3, 4, 1}),
                    TestSizes({251, 161, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_bf16_2_L2)
{
    doBatchGemmTest(TestSizes({99, 281, 20, 7, 1}),
                    TestSizes({101, 281, 20, 7, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_fp32_2_L2)
{
    doBatchGemmTest(TestSizes({99, 281, 20, 7, 1}),
                    TestSizes({101, 281, 20, 7, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_bf16_3)
{
    doBatchGemmTest(TestSizes({63, 55, 3, 4, 1}),
                    TestSizes({129, 55, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_fp32_3)
{
    doBatchGemmTest(TestSizes({383, 55, 3, 4, 1}),
                    TestSizes({21, 55, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_bf16_4)
{
    doBatchGemmTest(TestSizes({47, 123, 3, 4, 1}),
                    TestSizes({27, 123, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_bf16_5_L2)
{
    doBatchGemmTest(TestSizes({64, 128, 12, 32, 1}),
                    TestSizes({64, 128, 12, 32, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_335x357_bf)
{
    doBatchGemmTest(TestSizes({335, 215, 2, 3, 1}),
                    TestSizes({357, 215, 2, 3, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_315x307_bf)
{
    doBatchGemmTest(TestSizes({315, 203, 2, 3, 1}),
                    TestSizes({307, 203, 2, 3, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_335x37_bf)
{
    doBatchGemmTest(TestSizes({335, 25, 2, 3, 1}),
                    TestSizes({37, 25, 2, 3, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_33x357_bf)
{
    doBatchGemmTest(TestSizes({33, 25, 2, 3, 1}),
                    TestSizes({357, 25, 2, 3, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_bf16);
}

// ----------------------- ABt tests -------------------------
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_bf16_1)
{
    doBatchGemmTest(TestSizes({61, 27, 3, 4, 1}),
                    TestSizes({61, 151, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_bf16_5_dims)
{
    doBatchGemmTest(TestSizes({61, 27, 3, 4, 2}),
                    TestSizes({61, 151, 3, 4, 2}),
                    5,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_fp32_1)
{
    doBatchGemmTest(TestSizes({61, 137, 3, 4, 1}),
                    TestSizes({61, 51, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_fp32_5_dims)
{
    doBatchGemmTest(TestSizes({61, 137, 3, 4, 2}),
                    TestSizes({61, 51, 3, 4, 2}),
                    5,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_bf16_2_L2)
{
    doBatchGemmTest(TestSizes({281, 99, 20, 7, 1}),
                    TestSizes({281, 101, 20, 7, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_fp32_2_L2)
{
    doBatchGemmTest(TestSizes({281, 99, 20, 7, 1}),
                    TestSizes({281, 101, 20, 7, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_bf16_3)
{
    doBatchGemmTest(TestSizes({55, 63, 3, 4, 1}),
                    TestSizes({55, 321, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_fp32_3)
{
    doBatchGemmTest(TestSizes({55, 383, 3, 4, 1}),
                    TestSizes({55, 21, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_bf16_4)
{
    doBatchGemmTest(TestSizes({123, 47, 3, 4, 1}),
                    TestSizes({123, 27, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_fp32_4)
{
    doBatchGemmTest(TestSizes({12, 247, 3, 4, 1}),
                    TestSizes({12, 27, 3, 4, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_bf16_5_L2)
{
    doBatchGemmTest(TestSizes({128, 64, 12, 32, 1}),
                    TestSizes({128, 64, 12, 32, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_fp32_5_L2)
{
    doBatchGemmTest(TestSizes({128, 64, 12, 32, 1}),
                    TestSizes({128, 64, 12, 32, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_335x357_bf)
{
    doBatchGemmTest(TestSizes({215, 335, 2, 3, 1}),
                    TestSizes({215, 357, 2, 3, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_315x37_bf)
{
    doBatchGemmTest(TestSizes({203, 315, 2, 3, 1}),
                    TestSizes({203, 37, 2, 3, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_bf16);
}

// ===== bfloat -> float32 tests =============
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_27x27_bf2fp32)
{
    doBatchGemmTest(TestSizes({16, 27, 1, 4, 1}),
                    TestSizes({229, 16, 1, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_27x27_bf2fp32_5_dims)
{
    doBatchGemmTest(TestSizes({16, 27, 1, 4, 2}),
                    TestSizes({229, 16, 1, 4, 2}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_128x128_bf2fp32_aligned)
{
    doBatchGemmTest(TestSizes({128, 128, 3, 2, 1}),
                    TestSizes({128, 128, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_128x64_bf2fp32_aligned)
{
    doBatchGemmTest(TestSizes({128, 64, 3, 2, 1}),
                    TestSizes({128, 128, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_128x64_bf2fp32_unaligned)
{
    doBatchGemmTest(TestSizes({101, 64, 3, 2, 1}),
                    TestSizes({128, 101, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_112x27_bf2fp32)
{
    doBatchGemmTest(TestSizes({32, 27, 2, 2, 1}),
                    TestSizes({152, 32, 2, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_54x112_bf2fp32)
{
    doBatchGemmTest(TestSizes({47, 112, 2, 3, 1}),
                    TestSizes({54, 47, 2, 3, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_float);
}

// ===== bfloat -> bfloat tests =============
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_27x27_bf_5_dims)
{
    doBatchGemmTest(TestSizes({32, 27, 1, 4, 1}),
                    TestSizes({27, 32, 2, 4, 2}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_220x190_bf)
{
    // This test should run in 4wx1h
    doBatchGemmTest(TestSizes({32, 190, 1, 4, 1}),
                    TestSizes({220, 32, 1, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_190x220_bf)
{
    // This test should run in 4wx1h
    doBatchGemmTest(TestSizes({32, 220, 1, 4, 1}),
                    TestSizes({190, 32, 1, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_112x27_bf)
{
    doBatchGemmTest(TestSizes({32, 27, 2, 2, 1}),
                    TestSizes({112, 32, 2, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_54x112_bf)
{
    doBatchGemmTest(TestSizes({47, 112, 2, 3, 1}),
                    TestSizes({54, 47, 2, 3, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_128x128_bf_unaligned)
{
    doBatchGemmTest(TestSizes({101, 128, 3, 2, 1}),
                    TestSizes({128, 101, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_128x128_bf_aligned)
{
    doBatchGemmTest(TestSizes({128, 128, 3, 2, 1}),
                    TestSizes({128, 128, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_128x64_bf_aligned)
{
    doBatchGemmTest(TestSizes({128, 64, 3, 2, 1}),
                    TestSizes({128, 128, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_128x64_bf_unaligned)
{
    doBatchGemmTest(TestSizes({101, 64, 3, 2, 1}),
                    TestSizes({128, 101, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_64x128_bf_aligned)
{
    doBatchGemmTest(TestSizes({128, 128, 3, 2, 1}),
                    TestSizes({64, 128, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_64x128_bf_unaligned)
{
    doBatchGemmTest(TestSizes({101, 128, 3, 2, 1}),
                    TestSizes({64, 101, 3, 2, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_ab_335x357_bf)
{
    doBatchGemmTest(TestSizes({215, 335, 2, 3, 1}),
                    TestSizes({357, 215, 2, 3, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_ab_335x357_bf_5_dims)
{
    doBatchGemmTest(TestSizes({215, 335, 2, 3, 2}),
                    TestSizes({357, 215, 2, 3, 2}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_ab_315x307_bf)
{
    doBatchGemmTest(TestSizes({203, 315, 2, 3, 1}),
                    TestSizes({307, 203, 2, 3, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

// ===== float or bfloat -> float tests =============
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_fwd_27x27_fp32)
{
    doBatchGemmTest(TestSizes({32, 27, 1, 4, 1}),
                    TestSizes({27, 32, 1, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_atb_27x27_bf2fp32)
{
    doBatchGemmTest(TestSizes({27, 32, 1, 4, 1}),
                    TestSizes({27, 32, 1, 4, 1}),
                    4,
                    REFERENCE_OP_ATB,
                    syn_type_bf16,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_abt_64x128_bf2fp32_aligned)
{
    doBatchGemmTest(TestSizes({128, 128, 3, 2, 1}),
                    TestSizes({128, 64, 3, 2, 1}),
                    4,
                    REFERENCE_OP_ABT,
                    syn_type_bf16,
                    syn_type_float);
}

// ===== 2x tests =============
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_2xw_ab_112x44_bf)
{
    doBatchGemmTest(TestSizes({60, 44, 6, 4, 1}),
                    TestSizes({212, 60, 6, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_2xw_ab_112x44_bf_5_dims)
{
    doBatchGemmTest(TestSizes({60, 44, 6, 4, 2}),
                    TestSizes({212, 60, 6, 4, 2}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_2xh_ab_44x112_bf)
{
    doBatchGemmTest(TestSizes({60, 260, 6, 4, 1}),
                    TestSizes({44, 60, 6, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_2xh_ab_64x128_bf)
{
    doBatchGemmTest(TestSizes({60, 128, 6, 4, 1}),
                    TestSizes({64, 60, 6, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_2xw_ab_128x64_bf)
{
    doBatchGemmTest(TestSizes({60, 64, 6, 4, 1}),
                    TestSizes({128, 60, 6, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_2xh_ab_64x260_bf)
{
    doBatchGemmTest(TestSizes({60, 260, 6, 4, 1}),
                    TestSizes({64, 60, 6, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}
TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_noSplit_2xw_ab_260x64_bf)
{
    doBatchGemmTest(TestSizes({60, 64, 6, 4, 1}),
                    TestSizes({260, 60, 6, 4, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_bf16,
                    syn_type_bf16);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_asym)
{
    doBatchGemmTest(TestSizes({32, 46, 128, 1, 1}), TestSizes({27, 32, 1, 1, 1}), 3, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_asym2)
{
    doBatchGemmTest(TestSizes({32, 46, 1, 4, 1}), TestSizes({27, 32, 1, 1, 1}), 4, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_asym_weights)
{
    doBatchGemmTest(TestSizes({32, 46, 1, 1, 1}),
                    TestSizes({27, 32, 128, 1, 1}),
                    3,
                    REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedx_asym)
{
    doBatchGemmTest(TestSizes({128, 15, 14, 1, 1}), TestSizes({48, 128, 1, 1, 1}), 3, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedx_asym_weights)
{
    TestSizes ySizes = TestSizes({48, 15, 1, 1, 1});
    doBatchGemmTest(TestSizes({128, 15, 2, 1, 1}),
                    TestSizes({48, 128, 2, 1, 1}),
                    4,
                    REFERENCE_OP_DEDX,
                    syn_type_float,
                    syn_type_float,
                    &ySizes);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test)
{
    doBatchGemmTest(TestSizes({32, 27, 128, 1, 1}), TestSizes({27, 32, 128, 1, 1}), 3, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_multi)
{
    doBatchGemmTest(TestSizes({32, 27, 3, 4, 1}), TestSizes({27, 32, 3, 4, 1}), 4, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedx)
{
    doBatchGemmTest(TestSizes({10, 15, 18, 1, 1}), TestSizes({8, 10, 18, 1, 1}), 3, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedw)
{
    doBatchGemmTest(TestSizes({128, 16, 14, 1, 1}), TestSizes({32, 128, 14, 1, 1}), 3, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedw_asym)
{
    doBatchGemmTest(TestSizes({128, 16, 1, 1, 1}),
                    TestSizes({32, 128, 14, 1, 1}),
                    3,
                    REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedw_asym_op_b)
{
    TestSizes ySizes = TestSizes({32, 16, 1, 1, 1});
    doBatchGemmTest(TestSizes({128, 16, 14, 1, 1}),
                    TestSizes({32, 128, 14, 1, 1}),
                    3,
                    REFERENCE_OP_DEDW,
                    syn_type_float,
                    syn_type_float,
                    &ySizes);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedw_partial_bcast)
{
    TestSizes ySizes = TestSizes({32, 16, 14, 1, 1});
    doBatchGemmTest(TestSizes({128, 16, 14, 12, 1}),
                    TestSizes({32, 128, 14, 12, 1}),
                    4,
                    REFERENCE_OP_DEDW,
                    syn_type_float,
                    syn_type_float,
                    &ySizes);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedw_partial_bcast2)
{
    TestSizes ySizes = TestSizes({32, 16, 1, 12, 4});
    doBatchGemmTest(TestSizes({128, 16, 14, 12, 1}),
                    TestSizes({32, 128, 14, 12, 4}),
                    5,
                    REFERENCE_OP_DEDW,
                    syn_type_float,
                    syn_type_float,
                    &ySizes);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedx_partial_bcast)
{
    TestSizes ySizes = TestSizes({48, 15, 2, 1, 1});
    doBatchGemmTest(TestSizes({128, 15, 2, 2, 1}),
                    TestSizes({48, 128, 1, 2, 1}),
                    4,
                    REFERENCE_OP_DEDX,
                    syn_type_float,
                    syn_type_float,
                    &ySizes);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_dedx_partial_bcast2)
{
    TestSizes ySizes = TestSizes({48, 15, 4, 1, 1});
    doBatchGemmTest(TestSizes({128, 15, 4, 2, 1}),
                    TestSizes({48, 128, 4, 2, 1}),
                    4,
                    REFERENCE_OP_DEDX,
                    syn_type_float,
                    syn_type_float,
                    &ySizes);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_partial_bcast)
{
    doBatchGemmTest(TestSizes({32, 46, 1, 4, 4}),
                    TestSizes({27, 32, 3, 1, 4}),
                    5,
                    REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_partial_bcast2)
{
    doBatchGemmTest(TestSizes({28, 54, 1, 14, 1}),
                    TestSizes({42, 28, 32, 1, 1}),
                    5,
                    REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_partial_bcast3)
{
    doBatchGemmTest(TestSizes({32, 64, 3, 4, 1}),
                    TestSizes({86, 32, 3, 1, 1}),
                    4,
                    REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batch_gemm_test_partial_bcast_weights)
{
    doBatchGemmTest(TestSizes({32, 46, 128, 1, 1}),
                    TestSizes({27, 32, 128, 15, 1}),
                    4,
                    REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_partial_bcast)
{
    doBatchGemmTest(TestSizes({64, 31, 2, 4, 1}),
                    TestSizes({31, 80, 2, 1, 1}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_partial_bcast2)
{
    doBatchGemmTest(TestSizes({64, 31, 2, 4, 1}),
                    TestSizes({31, 80, 1, 1, 4}),
                    5,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, batchGemm_atbt_partial_bcast3)
{
    doBatchGemmTest(TestSizes({64, 31, 1, 1, 1}),
                    TestSizes({31, 80, 1, 3, 4}),
                    4,
                    REFERENCE_OP_ATBT,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, ab_trivial_dim)
{
    doBatchGemmTest(TestSizes({16, 20, 1, 3, 1}),
                    TestSizes({24, 16, 1, 3, 1}),
                    4,
                    REFERENCE_OP_AB,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, ab_trivial_dim2)
{
    doBatchGemmTest(TestSizes({16, 20, 1, 1, 3}),
                    TestSizes({24, 16, 1, 1, 3}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_float,
                    syn_type_float);
}

TEST_F_GC(SynTrainingBatchGemmTest, ab_trivial_dim3)
{
    doBatchGemmTest(TestSizes({16, 20, 3, 1, 3}),
                    TestSizes({24, 16, 3, 1, 3}),
                    5,
                    REFERENCE_OP_AB,
                    syn_type_float,
                    syn_type_float);
}

struct MaskedBGemmParams
{
    unsigned              heightA;
    unsigned              commonDim;
    unsigned              widthB;
    std::vector<unsigned> batchDims;
    unsigned              numMaskVectors;
    bool                  transposeA;
    bool                  transposeB;
    synDataType           dataType;
};

class SynGaudiMaskedBatchGemmTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<MaskedBGemmParams>
{
protected:
    std::array<unsigned, 4> m_inputs;
    unsigned                m_output;

    static std::vector<unsigned> getOperandShape(std::vector<unsigned> shape, bool transposed)
    {
        if (transposed)
        {
            std::swap(shape[0], shape[1]);
        }
        if (!GetParam().batchDims.empty())
        {
            shape.insert(shape.end(), GetParam().batchDims.begin(), GetParam().batchDims.end());
        }
        return shape;
    }

    static std::vector<unsigned> getMaskShape(std::vector<unsigned> shape, bool transposed)
    {
        shape                 = getOperandShape(shape, transposed);
        shape[DIM_GEMM_BATCH] = 1;
        return shape;
    }

    void addNode()
    {
        synGEMMParams         params(GetParam().transposeA, GetParam().transposeB);
        std::vector<unsigned> shape = getOperandShape({GetParam().commonDim, GetParam().heightA}, params.transpose_a);
        m_inputs[0]                 = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          shape.data(),
                                          shape.size(),
                                          GetParam().dataType,
                                          nullptr,
                                          "A");

        shape       = getOperandShape({GetParam().widthB, GetParam().commonDim}, params.transpose_b);
        m_inputs[1] = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          shape.data(),
                                          shape.size(),
                                          GetParam().dataType,
                                          nullptr,
                                          "B");

        shape       = getMaskShape({GetParam().numMaskVectors, GetParam().heightA}, params.transpose_a);
        m_inputs[2] = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          shape.data(),
                                          shape.size(),
                                          GetParam().dataType,
                                          nullptr,
                                          "MaskA");

        shape       = getMaskShape({GetParam().widthB, GetParam().numMaskVectors}, params.transpose_b);
        m_inputs[3] = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          shape.data(),
                                          shape.size(),
                                          GetParam().dataType,
                                          nullptr,
                                          "MaskB");
        shape       = getOperandShape({GetParam().widthB, GetParam().heightA}, false);
        m_output    = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       shape.data(),
                                       shape.size(),
                                       GetParam().dataType,
                                       nullptr,
                                       "C");

        addNodeToGraph(NodeFactory::maskedBatchGemmNodeTypeName,
                       {m_inputs[0], m_inputs[1], m_inputs[2], m_inputs[3]},
                       {m_output},
                       &params,
                       sizeof(params),
                       "MaskedBGemm");
    }

    ERepefenceOp getOp()
    {
        ERepefenceOp op;
        if (GetParam().transposeA)
        {
            if (GetParam().transposeB)
            {
                op = REFERENCE_OP_ATBT;
            }
            else
            {
                op = REFERENCE_OP_ATB;
            }
        }
        else
        {
            if (GetParam().transposeB)
            {
                op = REFERENCE_OP_ABT;
            }
            else
            {
                op = REFERENCE_OP_AB;
            }
        }
        return op;
    }

    void validateAccuracy()
    {
        addNode();
        compileAndRun();

        synTensorDescriptor xDesc = m_tensorDescs[m_inputs[0]];
        synTensorDescriptor wDesc = m_tensorDescs[m_inputs[1]];
        synTensorDescriptor xMaskDesc = m_tensorDescs[m_inputs[2]];
        synTensorDescriptor wMaskDesc = m_tensorDescs[m_inputs[3]];
        synTensorDescriptor yDesc = m_tensorDescs[m_output];
        // todo AlonG: check with Omer for the better way to reset the irrelevant dims
        resetIrrelevantDims(xDesc);
        resetIrrelevantDims(wDesc);
        resetIrrelevantDims(yDesc);
        void* xData = m_hostBuffers[m_inputs[0]];
        void* wData = m_hostBuffers[m_inputs[1]];
        void* xMaskData = m_hostBuffers[m_inputs[2]];
        void* wMaskData = m_hostBuffers[m_inputs[3]];
        void* yData = m_hostBuffers[m_output];

        CoordArray wrongIdx       = {0};
        float      expectedResult = 0;
        bool       ret            = checkMaskedBatchGemmOp(xDesc,
                                          (char*)xData,
                                          wDesc,
                                          (char*)wData,
                                          xMaskDesc,
                                          (char*)xMaskData,
                                          wMaskDesc,
                                          (char*)wMaskData,
                                          yDesc,
                                          (char*)yData,
                                          getOp(),
                                          wrongIdx,
                                          &expectedResult,
                                          m_deviceType);

        if (!ret)
        {
            TSize sizes[SYN_MAX_TENSOR_DIM];
            castNcopy(sizes, m_tensorDescs[m_output].m_sizes, SYN_MAX_TENSOR_DIM);
            EXPECT_FLOAT_EQ(getIndexValue(sizes,
                                          wrongIdx,
                                          m_tensorDescs[m_output].m_dataType,
                                          m_hostBuffers[m_output]),
                            expectedResult);
        }
    }
};

TEST_P_GC(SynGaudiMaskedBatchGemmTest, masked_bgemm_accuracy, {synDeviceGaudi2})
{
    validateAccuracy();
}

INSTANTIATE_TEST_SUITE_P(masked_bgemm,
                         SynGaudiMaskedBatchGemmTest,
                         ::testing::Values(MaskedBGemmParams {64, 128, 24, {12, 24}, 13, false, false, syn_type_float},
                                           MaskedBGemmParams {12, 28, 100, {13, 13}, 16, false, true, syn_type_float},
                                           MaskedBGemmParams {123, 218, 16, {2, 16}, 10, true, true, syn_type_float},
                                           MaskedBGemmParams {50, 300, 70, {32, 2}, 11, true, false, syn_type_float},
                                           MaskedBGemmParams {2048, 128, 2048, {12, 1}, 6, true, false, syn_type_bf16},
                                           MaskedBGemmParams {64, 128, 24, {1, 24}, 13, false, false, syn_type_float}));