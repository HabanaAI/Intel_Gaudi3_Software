#include "syn_gaudi_two_run_compare_test.h"

class SynTrainingMmeSlicingTest : public SynTrainingTwoRunCompareTest

{
public:
    SynTrainingMmeSlicingTest();

protected:
    void runMmeTest(const TestSizes&            xSize,
                    const TestSizes&            wSize,
                    const TestSizes&            ySize,
                    const synConvolutionParams& params,
                    ERepefenceOp                op,
                    synDataType                 dtype          = syn_type_single,
                    synDataType                 outputDataType = syn_type_na);

    void runMmeTest(const TestSizes&            xSize,
                    const unsigned int          yChannels,
                    const synConvolutionParams& convParams,
                    ERepefenceOp                op,
                    synDataType                 dataType       = syn_type_single,
                    synDataType                 outputDataType = syn_type_na);
};

void SynTrainingMmeSlicingTest::runMmeTest(const TestSizes&            xSize,
                                           const TestSizes&            wSize,
                                           const TestSizes&            ySize,
                                           const synConvolutionParams& params,
                                           ERepefenceOp                op,
                                           synDataType                 dtype,
                                           synDataType                 outputDType)
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
    std::string   guid          = "spatial_convolution";
    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};
    if (op == REFERENCE_OP_DEDW)
    {
        guid             = "dedw";
        inputIndices[0]  = yTensorIndex;
        inputIndices[1]  = xTensorIndex;
        outputIndices[0] = wTensorIndex;
    }
    else if (op == REFERENCE_OP_DEDX)
    {
        guid             = "dedx";
        inputIndices[0]  = yTensorIndex;
        inputIndices[1]  = wTensorIndex;
        outputIndices[0] = xTensorIndex;
    }
    unsigned paramSize = sizeof(params);
    addNodeToGraph(guid.c_str(), inputIndices, outputIndices, (void*)&params, paramSize);
    compileAndRun();

    addConfigurationToRun(FIRST_RUN,
                          "SRAM_SLICER_MAX_CAPACITY_BYTES",
                          GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.getDefaultValuesStr());

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({outputIndices[0]});
}

void SynTrainingMmeSlicingTest::runMmeTest(const TestSizes&            xSize,
                                           const unsigned int          yChannels,
                                           const synConvolutionParams& convParams,
                                           ERepefenceOp                op,
                                           synDataType                 dataType,
                                           synDataType                 outputDataType)
{
    TestSizes wSize = {yChannels, xSize[0], convParams.kW, convParams.kH, 1};
    TestSizes ySize = {
        yChannels,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3],
        1};

    runMmeTest(xSize, wSize, ySize, convParams, op, dataType, outputDataType);
}

SynTrainingMmeSlicingTest::SynTrainingMmeSlicingTest()
{
    setSupportedDevices({'-', synDeviceGaudi3});
}

TEST_F_GC(SynTrainingMmeSlicingTest, fwd_fit_in_sram)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 7;
    convParams.dH = convParams.dW = 3;
    TestSizes xSize = {6, 30, 30, 100, 1};
    runMmeTest(xSize, 1000, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingMmeSlicingTest, fwd_spatial_slicing_1)
{
    synConvolutionParams convParams;
    TestSizes xSize = {7, 30, 30, 300, 1};
    runMmeTest(xSize, 500, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingMmeSlicingTest, fwd_spatial_slicing_2)
{
    synConvolutionParams convParams;
    TestSizes xSize = {5000, 29, 30, 1, 1};
    runMmeTest(xSize, 2, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingMmeSlicingTest, fwd_spatial_slicing_3)
{
    synConvolutionParams convParams;
    TestSizes xSize = {224, 2, 1, 1, 1};
    runMmeTest(xSize, 7232, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingMmeSlicingTest, fwd_batch_slicing_1)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 2;
    convParams.dH = convParams.dW = 1;
    TestSizes xSize = {7, 80, 50, 174, 1};
    runMmeTest(xSize, 100, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingMmeSlicingTest, fwd_batch_slicing_2_ASIC)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 5;
    convParams.dH = convParams.dW = 5;
    TestSizes xSize = {500, 17, 17, 20, 1};
    runMmeTest(xSize, 500, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingMmeSlicingTest, fwd_batch_slicing_3_ASIC)  // 43sec on simulator in release mode.
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 7;
    convParams.dH = convParams.dW = 2;
    TestSizes xSize = {100, 20, 20, 200, 1};
    runMmeTest(xSize, 500, convParams, REFERENCE_OP_FWD);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_fit_in_sram_ASIC)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 7;
    convParams.dH = convParams.dW = 3;
    TestSizes xSize = {1000, 30, 30, 100, 1};
    runMmeTest(xSize, 6, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_spatial_slicing_1)
{
    synConvolutionParams convParams;
    TestSizes xSize = {7, 30, 30, 70, 1};
    runMmeTest(xSize, 500, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_spatial_slicing_2)
{
    synConvolutionParams convParams;
    TestSizes xSize = {2, 29, 30, 1, 1};
    runMmeTest(xSize, 5000, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_spatial_slicing_3)
{
    synConvolutionParams convParams;
    TestSizes xSize = {224, 2, 1, 1, 1};
    runMmeTest(xSize, 7232, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_batch_slicing_1_L2, {synDeviceGaudi})
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 2;
    convParams.dH = convParams.dW = 1;
    TestSizes xSize = {100, 80, 50, 174, 1};
    runMmeTest(xSize, 7, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_batch_slicing_1_ASIC_CI, {synDeviceGaudi})
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 2;
    convParams.dH = convParams.dW = 1;
    TestSizes xSize               = {100, 80, 50, 17, 1};
    runMmeTest(xSize, 7, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_batch_slicing_2)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 5;
    convParams.dH = convParams.dW = 5;
    TestSizes xSize = {500, 17, 17, 20, 1};
    runMmeTest(xSize, 500, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_batch_slicing_3_L2, {synDeviceGaudi})
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 7;
    convParams.dH = convParams.dW = 2;
    TestSizes xSize = {500, 20, 20, 200, 1};
    runMmeTest(xSize, 100, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_batch_slicing_3_ASIC)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 7;
    convParams.dH = convParams.dW = 2;
    TestSizes xSize               = {500, 20, 20, 33, 1};
    runMmeTest(xSize, 100, convParams, REFERENCE_OP_DEDX);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_dw_fit_in_sram)
{
    synConvolutionParams convParams;
    TestSizes xSize = {20, 30, 30, 15, 1};
    runMmeTest(xSize, 10, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_dw_cd_slicing_1)
{
    synConvolutionParams convParams;
    TestSizes xSize = {3, 152, 152, 80, 1};
    runMmeTest(xSize, 5, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingMmeSlicingTest, bwd_dw_cd_slicing_2)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 3;
    convParams.dH = convParams.dW = 2;
    TestSizes xSize = {300, 10, 32, 10, 1};
    runMmeTest(xSize, 150, convParams, REFERENCE_OP_DEDW);
}

TEST_F_GC(SynTrainingMmeSlicingTest, DISABLED_bwd_dw_cd_slicing_3)
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 2;
    convParams.dH = convParams.dW = 1;
    TestSizes xSize = {30, 20, 20, 50, 1};
    runMmeTest(xSize, 100, convParams, REFERENCE_OP_DEDW);
}
