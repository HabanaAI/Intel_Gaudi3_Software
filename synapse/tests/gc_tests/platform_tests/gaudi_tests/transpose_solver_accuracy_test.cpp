#include "syn_gaudi_two_run_compare_test.h"

TEST_F_GC(SynGaudiTwoRunCompareTest, transpose_solver_accuracy_test, {synDeviceGaudi})
{
    unsigned inSizes[]  = {1638400, 4};
    unsigned outSizes[] = {4, 1638400};
    addConfigurationToRun(FIRST_RUN, "DMA_TRANSPOSE_SOLVER_ENABLED", "true");
    addConfigurationToRun(SECOND_RUN, "DMA_TRANSPOSE_SOLVER_ENABLED", "false");
    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2};

    auto in  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, inSizes, 2, syn_type_float);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, outSizes, 2, syn_type_float);
    addNodeToGraph("transpose", {in}, {out}, &transposeParams, sizeof(transposeParams));
    compareRunsResults({out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, dynamic_transpose_2d_solver_ASIC_CI, {synDeviceGaudi})
{
    const unsigned maxFcd       = 22356736;
    const unsigned minFcd       = 4131136;
    const unsigned scd          = 2;
    unsigned       inMaxSizes[] = {maxFcd, scd};
    unsigned       inMinSizes[] = {minFcd, scd};

    unsigned outMaxSizes[] = {scd, maxFcd};
    unsigned outMinSizes[] = {scd, minFcd};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            2,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             2,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2};

    addNodeToGraph("transpose", {inTensor}, {outTensor}, &transposeParams, sizeof(transposeParams));

    const unsigned halfActual = (maxFcd + minFcd) / 2;

    unsigned inActualSizes[2]  = {halfActual, scd};
    unsigned outActualSizes[2] = {scd, halfActual};

    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, outActualSizes);

    addConfigurationToRun(FIRST_RUN, "ENABLE_DMA_TRANSPOSE_SOLVER", "true");

    addConfigurationToRun(SECOND_RUN, "ENABLE_DMA_TRANSPOSE_SOLVER", "false");

    compareRunsResults({outTensor});
}
