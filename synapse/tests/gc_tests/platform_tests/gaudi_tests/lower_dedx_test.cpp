#include "global_conf_test_setter.h"
#include "habana_global_conf.h"
#include "synapse_common_types.h"
#include "syn_gaudi_two_run_compare_test.h"

class SynTrainingLowerDedx : public SynTrainingTwoRunCompareTest
{
};

TEST_F_GC(SynTrainingLowerDedx, lower_dedx_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 3;
    constexpr uint32_t k          = 2;
    TestSizes          xSize      = {8, 7, 7, 4};
    TestSizes          wSize      = {k, xSize[0], convParams.kW, convParams.kH};
    TestSizes          ySize      = {
        k,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3]};

    unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        ySize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     wSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_single);

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,  // initializer
                                        xSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    addNodeToGraph("dedx", {dedy, w}, {dedx}, &convParams, sizeof(convParams));

    addConfigurationToRun(FIRST_RUN, "ENABLE_LOWER_DEDX", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LOWER_DEDX", "true");

    compareRunsResults({dedx});
}

TEST_F_GC(SynTrainingLowerDedx, lower_dedx_slicing_test, {synDeviceGaudi2, synDeviceGaudi3})
{
    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 3;
    constexpr uint32_t k          = 2;
    TestSizes          xSize      = {8, 64, 64, 14, 1};
    TestSizes          wSize      = {k, xSize[0], convParams.kW, convParams.kH};
    TestSizes          ySize      = {
        k,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3]};

    unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        ySize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     wSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_single);

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,  // initializer
                                        xSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    addNodeToGraph("dedx", {dedy, w}, {dedx}, &convParams, sizeof(convParams));

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({dedx});
}