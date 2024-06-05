#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "scoped_configuration_change.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, concat_same_input_x3)
{
    static constexpr auto FCD_AXIS = 0;
    static constexpr auto SCD_AXIS = 1;

    const synConcatenateParams concatFcdParams = {.axis = FCD_AXIS};
    const synConcatenateParams concatScdParams = {.axis = SCD_AXIS};

    constexpr auto dims            = 2u;
    constexpr auto numAggregations = 3u;

    const unsigned concatInSizes[] = {3, 2};
    const float    concatInData[]  = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    const auto concatFcdIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_FROM_INITIALIZER,
                                                 concatInData,
                                                 const_cast<unsigned*>(concatInSizes),
                                                 dims,
                                                 syn_type_single,
                                                 nullptr,
                                                 "concatFcd_in");

    const unsigned concatFcdOutSizes[] = {concatInSizes[0] * numAggregations, concatInSizes[1]};

    const auto concatFcdOut = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  const_cast<unsigned*>(concatFcdOutSizes),
                                                  dims,
                                                  syn_type_single,
                                                  nullptr,
                                                  "concatFcd_out");

    const auto concatScdIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_FROM_INITIALIZER,
                                                 concatInData,
                                                 const_cast<unsigned*>(concatInSizes),
                                                 dims,
                                                 syn_type_single,
                                                 nullptr,
                                                 "concatScd_in");

    const unsigned concatScdOutSizes[] = {concatInSizes[0], concatInSizes[1] * numAggregations};

    const auto concatScdOut = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  const_cast<unsigned*>(concatScdOutSizes),
                                                  dims,
                                                  syn_type_single,
                                                  nullptr,
                                                  "concatScd_out");

    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   {concatFcdIn, concatFcdIn, concatFcdIn},
                   {concatFcdOut},
                   const_cast<synConcatenateParams*>(&concatFcdParams),
                   sizeof(concatFcdParams),
                   "CONCAT_FCD");

    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   {concatScdIn, concatScdIn, concatScdIn},
                   {concatScdOut},
                   const_cast<synConcatenateParams*>(&concatScdParams),
                   sizeof(concatScdParams),
                   "CONCAT_SCD");

    compileAndRun();

    // validate concat on SCD
    const auto* pBufConcatScdIn  = castHostBuffer<float>(concatScdIn);
    const auto* pBufConcatScdOut = castHostBuffer<float>(concatScdOut);

    for (unsigned j = 0; j < concatScdOutSizes[1]; ++j)
        for (unsigned k = 0; k < concatScdOutSizes[0]; ++k)
        {
            const auto concatOutIdx = k + (j * concatScdOutSizes[0]);
            const auto concatInIdx  = k + ((j % concatInSizes[1]) * concatInSizes[0]);

            ASSERT_EQ(pBufConcatScdOut[concatOutIdx], pBufConcatScdIn[concatInIdx]);
        }

    // validate concat on FCD
    const auto* pBufConcatFcdIn  = castHostBuffer<float>(concatFcdIn);
    const auto* pBufConcatFcdOut = castHostBuffer<float>(concatFcdOut);

    for (unsigned j = 0; j < concatFcdOutSizes[1]; ++j)
        for (unsigned k = 0; k < concatFcdOutSizes[0]; ++k)
        {
            const auto concatOutIdx = k + (j * concatFcdOutSizes[0]);
            const auto concatInIdx  = (k % concatInSizes[0]) + (j * concatInSizes[0]);

            ASSERT_EQ(pBufConcatFcdOut[concatOutIdx], pBufConcatFcdIn[concatInIdx]);
        }
}

TEST_F_GC(SynGaudiTwoRunCompareTest, concat_on_fcd, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    addConfigurationToRun(FIRST_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "false");
    addConfigurationToRun(SECOND_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "true");

    unsigned addIn1Sizes[] = {4, 128, 256, 4};
    unsigned addIn1        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "addIn1",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    addIn1Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addIn1Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned addIn2Sizes[] = {4, 128, 256, 4};
    unsigned addIn2        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "addIn2",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    addIn2Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addIn2Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned addOutSizes[] = {4, 128, 256, 4};
    unsigned addOut        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "addOut",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    addOutSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addOutSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_f32", {addIn1, addIn2}, {addOut}, nullptr, 0, "ADD");

    unsigned sigmoidIn1Sizes[] = {1, 128, 256, 4};
    unsigned sigmoidIn1        = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "sigmoidIn1",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        sigmoidIn1Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        sigmoidIn1Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned sigmoidIn2Sizes[] = {1, 128, 256, 4};
    unsigned sigmoidIn2        = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "sigmoidIn2",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        sigmoidIn2Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        sigmoidIn2Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned sigmoidOutSizes[] = {1, 128, 256, 4};
    unsigned sigmoidOut        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "sigmoidOut",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        sigmoidOutSizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        sigmoidOutSizes,
                                        synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("sigmoid_bwd_f32", {sigmoidIn1, sigmoidIn2}, {sigmoidOut}, nullptr, 0, "SIGMOID");

    unsigned softmaxIn1Sizes[] = {10, 128, 256, 4};
    unsigned softmaxIn1        = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "softmaxIn1",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        softmaxIn1Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        softmaxIn1Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned softmaxIn2Sizes[] = {10, 128, 256, 4};
    unsigned softmaxIn2        = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "softmaxIn2",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        softmaxIn2Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        softmaxIn2Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned softmaxOutSizes[] = {10, 128, 256, 4};
    unsigned softmaxOut        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "softmaxOut",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        softmaxOutSizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        softmaxOutSizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned char softmaxParams[] = {0, 0, 0, 0};
    addNodeToGraph("softmax_bwd_f32", {softmaxIn1, softmaxIn2}, {softmaxOut}, (void*)softmaxParams, 4, "SOFTMAX");

    unsigned whereIn1Sizes[] = {2, 128, 256, 4};
    unsigned whereIn1        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "whereIn1",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      whereIn1Sizes,
                                      4,
                                      syn_type_int8,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      whereIn1Sizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned whereIn2Sizes[] = {1};
    unsigned whereIn2        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "whereIn2",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      whereIn2Sizes,
                                      1,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      true,
                                      whereIn2Sizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned whereIn3Sizes[] = {2, 128, 256, 4};
    unsigned whereIn3        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "whereIn3",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      whereIn3Sizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      whereIn3Sizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned whereOutSizes[] = {2, 128, 256, 4};
    unsigned whereOut        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "whereOut",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      whereOutSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      whereOutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("where_fwd_f32", {whereIn1, whereIn2, whereIn3}, {whereOut}, nullptr, 0, "WHERE");

    unsigned concatIn5Sizes[] = {1, 128, 256, 4};
    unsigned concatIn5        = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "concatIn5",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       concatIn5Sizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatIn5Sizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned concatOutSizes[] = {18, 128, 256, 4};
    unsigned concatOut        = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "concatOut",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       concatOutSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    addNodeToGraph("concat",
                   {softmaxOut, sigmoidOut, addOut, whereOut, concatIn5},
                   {concatOut},
                   &concatParams,
                   sizeof(concatParams),
                   "CONCAT");

    compareRunsResults({concatOut});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, split_concat_on_fcd, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    addConfigurationToRun(FIRST_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "false");
    addConfigurationToRun(SECOND_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "true");

    unsigned convIn1Sizes[] = {8, 896, 256, 2};
    unsigned convIn1        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "convIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     convIn1Sizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     convIn1Sizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned convIn2Sizes[] = {18, 8, 1, 1};
    unsigned convIn2        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "convIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     convIn2Sizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     true,
                                     convIn2Sizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned convOutSizes[] = {18, 896, 256, 2};
    unsigned convOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "convOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     convOutSizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     convOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned char convParams[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 236, 192, 1,   0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   253, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {convIn1, convIn2}, {convOut}, (void*)convParams, 72, "CONV");

    unsigned split0Sizes[] = {10, 896, 256, 2};
    unsigned split0        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "split0",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    split0Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    split0Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned split1Sizes[] = {1, 896, 256, 2};
    unsigned split1        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "split1",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    split1Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    split1Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned split2Sizes[] = {4, 896, 256, 2};
    unsigned split2        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "split2",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    split2Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    split2Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned split3Sizes[] = {2, 896, 256, 2};
    unsigned split3        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "split3",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    split3Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    split3Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned split4Sizes[] = {1, 896, 256, 2};
    unsigned split4        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "split4",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    split4Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    split4Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    synSplitParams splitParams;
    splitParams.axis = 0;
    addNodeToGraph("split",
                   {convOut},
                   {split0, split1, split2, split3, split4},
                   &splitParams,
                   sizeof(splitParams),
                   "SPLIT");

    unsigned sinOutSizes[] = {10, 896, 256, 2};
    unsigned sinOut        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "sinOut",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    sinOutSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    sinOutSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("sin_fwd_f32", {split0}, {sinOut}, nullptr, 0, "SIN");

    unsigned sigmoidOutSizes[] = {1, 896, 256, 2};
    unsigned sigmoidOut        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "sigmoidOut",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        sigmoidOutSizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        sigmoidOutSizes,
                                        synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("sigmoid_fwd_f32", {split1}, {sigmoidOut}, nullptr, 0, "SIGMOID");

    unsigned negOutSizes[] = {4, 896, 256, 2};
    unsigned negOut        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "negOut",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    negOutSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    negOutSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("neg_fwd_f32", {split2}, {negOut}, nullptr, 0, "negOut");

    unsigned reluOutSizes[] = {2, 896, 256, 2};
    unsigned reluOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "reluOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     reluOutSizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     reluOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_f32", {split3}, {reluOut}, nullptr, 0, "RELU");

    unsigned addInSizes[] = {1, 896, 256, 2};
    unsigned addIn        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "addIn",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   addInSizes,
                                   4,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   addInSizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned addOutSizes[] = {1, 896, 256, 2};
    unsigned addOut        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "addOut",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    addOutSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addOutSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_f32", {addIn, split4}, {addOut}, nullptr, 0, "ADD");

    unsigned concatOutSizes[] = {18, 896, 256, 2};
    unsigned concatOut        = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "concatOut",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       concatOutSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    addNodeToGraph("concat",
                   {sinOut, sigmoidOut, negOut, reluOut, addOut},
                   {concatOut},
                   &concatParams,
                   sizeof(concatParams),
                   "CONCAT");

    unsigned neg2OutSizes[] = {1, 896, 256, 2};
    unsigned neg2Out        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "neg2Out",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     neg2OutSizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     neg2OutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("neg_fwd_f32", {addOut}, {neg2Out}, nullptr, 0, "NEG2");

    compareRunsResults({concatOut, neg2Out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, concat_same_inputs)
{
    addConfigurationToRun(FIRST_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "false");
    addConfigurationToRun(SECOND_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "true");

    unsigned concatInSizes[] = {6, 128, 256, 4};
    unsigned concatIn1       = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "concatIn1",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       concatInSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatInSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned concatIn2 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "concatIn2",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       concatInSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatInSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned concatOutSizes[] = {18, 128, 256, 4};
    unsigned concatOut        = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "concatOut",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       concatOutSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    addNodeToGraph("concat",
                   {concatIn1, concatIn2, concatIn1},
                   {concatOut},
                   &concatParams,
                   sizeof(concatParams),
                   "CONCAT");

    compareRunsResults({concatOut});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, split_3_dims)
{
    addConfigurationToRun(FIRST_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "false");
    addConfigurationToRun(SECOND_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "true");

    unsigned splitInSizes[] = {18, 128, 256};
    unsigned splitIn        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "splitIn",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     splitInSizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     splitInSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned splitOutSizes[] = {6, 128, 256};
    unsigned splitOut1       = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "splitOut1",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       splitOutSizes,
                                       3,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       splitOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned splitOut2 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "splitOut2",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       splitOutSizes,
                                       3,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       splitOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned splitOut3 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "splitOut3",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       splitOutSizes,
                                       3,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       splitOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    synConcatenateParams splitParams;
    splitParams.axis = 0;
    addNodeToGraph("split", {splitIn}, {splitOut1, splitOut2, splitOut3}, &splitParams, sizeof(splitParams), "SPLIT");

    compareRunsResults({splitOut1, splitOut2, splitOut3});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, split_with_const_input)
{
    addConfigurationToRun(FIRST_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "false");
    addConfigurationToRun(SECOND_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "true");

    ScopedConfigurationChange hbmSizeCfg("HBM_GLOBAL_MEM_SIZE_MEGAS", "256");
    ScopedConfigurationChange constTensorSizeCfg("MAX_CONST_TENSOR_SIZE_BYTES", "0x6400000");

    unsigned splitInSizes[] = {18, 128, 256};
    unsigned splitIn        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "splitIn",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     splitInSizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     true,  // const input
                                     splitInSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned splitOutSizes[] = {6, 128, 256};
    unsigned splitOut1       = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "splitOut1",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       splitOutSizes,
                                       3,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       splitOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned splitOut2 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "splitOut2",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       splitOutSizes,
                                       3,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       splitOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned splitOut3 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "splitOut3",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       splitOutSizes,
                                       3,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       splitOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    synConcatenateParams splitParams;
    splitParams.axis = 0;
    addNodeToGraph("split", {splitIn}, {splitOut1, splitOut2, splitOut3}, &splitParams, sizeof(splitParams), "SPLIT");

    compareRunsResults({splitOut1, splitOut2, splitOut3});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, concat_with_const_input)
{
    addConfigurationToRun(FIRST_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "false");
    addConfigurationToRun(SECOND_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "true");

    ScopedConfigurationChange hbmSizeCfg("HBM_GLOBAL_MEM_SIZE_MEGAS", "256");
    ScopedConfigurationChange constTensorSizeCfg("MAX_CONST_TENSOR_SIZE_BYTES", "0x6400000");

    unsigned concatInSizes[] = {6, 128, 256, 4};
    unsigned concatIn1       = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "concatIn1",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       concatInSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatInSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned concatIn2 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "concatIn2",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       concatInSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       true,  // const input
                                       concatInSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned concatOutSizes[] = {12, 128, 256, 4};
    unsigned concatOut        = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "concatOut",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       concatOutSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    addNodeToGraph("concat", {concatIn1, concatIn2}, {concatOut}, &concatParams, sizeof(concatParams), "CONCAT");

    compareRunsResults({concatOut});
}

class SynTrainingSplitConcatMultiNodes : public SynGaudiTwoRunCompareTest
{
public:
    void splitConcatMultiNodesTest(unsigned sobResetLimit);
};

void SynTrainingSplitConcatMultiNodes::splitConcatMultiNodesTest(unsigned sobResetLimit)
{
    addConfigurationToRun(FIRST_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "false");
    addConfigurationToRun(SECOND_RUN, "OPTIMIZE_SPLIT_CONCAT_ON_FCD", "true");

    addConfigurationToRun(FIRST_RUN, "ARC_SYNC_SCHEME_SIGNAL_LIMIT", std::to_string(sobResetLimit));
    addConfigurationToRun(SECOND_RUN, "ARC_SYNC_SCHEME_SIGNAL_LIMIT", std::to_string(sobResetLimit));

    unsigned split1InSizes[] = {32, 32, 16, 10};
    unsigned split1In        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "split1In",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      split1InSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      split1InSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned split1Out0Sizes[] = {32, 32, 4, 10};
    unsigned split1Out0        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split1Out0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split1Out0Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split1Out0Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned split1Out1Sizes[] = {32, 32, 4, 10};
    unsigned split1Out1        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split1Out1",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split1Out1Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split1Out1Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned split1Out2Sizes[] = {32, 32, 4, 10};
    unsigned split1Out2        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split1Out2",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split1Out2Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split1Out2Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned split1Out3Sizes[] = {32, 32, 4, 10};
    unsigned split1Out3        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split1Out3",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split1Out3Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split1Out3Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned char split1Params[] = {2, 0, 0, 0};
    addNodeToGraph("split",
                   {split1In},
                   {split1Out0, split1Out1, split1Out2, split1Out3},
                   (void*)split1Params,
                   4,
                   "SPLIT1");

    unsigned split2InSizes[] = {8, 4, 3, 3};
    unsigned split2In        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "split2In",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      split2InSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      split2InSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned split2Out0Sizes[] = {2, 4, 3, 3};
    unsigned split2Out0        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split2Out0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split2Out0Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split2Out0Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned split2Out1Sizes[] = {2, 4, 3, 3};
    unsigned split2Out1        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split2Out1",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split2Out1Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split2Out1Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned split2Out2Sizes[] = {2, 4, 3, 3};
    unsigned split2Out2        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split2Out2",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split2Out2Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split2Out2Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned split2Out3Sizes[] = {2, 4, 3, 3};
    unsigned split2Out3        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "split2Out3",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        split2Out3Sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        split2Out3Sizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned char split2Params[] = {0, 0, 0, 0};
    addNodeToGraph("split",
                   {split2In},
                   {split2Out0, split2Out1, split2Out2, split2Out3},
                   (void*)split2Params,
                   4,
                   "SPLIT2");

    unsigned transpose1OutSizes[] = {4, 32, 32, 10};
    unsigned transpose1Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose1Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose1OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose1OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose1Params[] = {2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {split1Out0}, {transpose1Out}, (void*)transpose1Params, 24, "TRANSPOSE1");

    unsigned conv1OutSizes[] = {2, 32, 32, 10};
    unsigned conv1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "conv1Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      conv1OutSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv1OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char conv1Params[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,   1,   0,   0, 0,
                                   1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 176, 110, 1,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   180, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {transpose1Out, split2Out0}, {conv1Out}, (void*)conv1Params, 72, "CONV1");

    unsigned transpose2OutSizes[] = {32, 32, 2, 10};
    unsigned transpose2Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose2Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose2OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose2OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose2Params[] = {1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {conv1Out}, {transpose2Out}, (void*)transpose2Params, 24, "TRANSPOSE2");

    unsigned transpose3OutSizes[] = {4, 32, 32, 10};
    unsigned transpose3Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose3Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose3OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose3OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose3Params[] = {2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {split1Out1}, {transpose3Out}, (void*)transpose3Params, 24, "TRANSPOSE3");

    unsigned conv2OutSizes[] = {2, 32, 32, 10};
    unsigned conv2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "conv2Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      conv2OutSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv2OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char conv2Params[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,   1,   0,   0, 0,
                                   1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 176, 110, 1,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   180, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {transpose3Out, split2Out1}, {conv2Out}, (void*)conv2Params, 72, "CONV2");

    unsigned transpose4OutSizes[] = {32, 32, 2, 10};
    unsigned transpose4Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose4Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose4OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose4OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose4Params[] = {1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {conv2Out}, {transpose4Out}, (void*)transpose4Params, 24, "TRANSPOSE4");

    unsigned transpose5OutSizes[] = {4, 32, 32, 10};
    unsigned transpose5Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose5Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose5OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose5OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose5Params[] = {2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {split1Out2}, {transpose5Out}, (void*)transpose5Params, 24, "TRANSPOSE5");

    unsigned conv3OutSizes[] = {2, 32, 32, 10};
    unsigned conv3Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "conv3Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      conv3OutSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv3OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char conv3Params[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,   1,   0,   0, 0,
                                   1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 176, 110, 1,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   180, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {transpose5Out, split2Out2}, {conv3Out}, (void*)conv3Params, 72, "CONV3");

    unsigned transpose6OutSizes[] = {32, 32, 2, 10};
    unsigned transpose6Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose6Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose6OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose6OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose6Params[] = {1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {conv3Out}, {transpose6Out}, (void*)transpose6Params, 24, "TRANSPOSE6");

    unsigned transpose7OutSizes[] = {4, 32, 32, 10};
    unsigned transpose7Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose7Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose7OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose7OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose7Params[] = {2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {split1Out3}, {transpose7Out}, (void*)transpose7Params, 24, "TRANSPOSE7");

    unsigned conv4OutSizes[] = {2, 32, 32, 10};
    unsigned conv4Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "conv4Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      conv4OutSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv4OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char conv4OutParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,   1,   0,   0, 0,
                                      1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 176, 110, 1,   0,   0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   180, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {transpose7Out, split2Out3}, {conv4Out}, (void*)conv4OutParams, 72, "CONV4");

    unsigned transpose8OutSizes[] = {32, 32, 2, 10};
    unsigned transpose8Out        = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "transpose8Out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           transpose8OutSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           transpose8OutSizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned char transpose8Params[] = {1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {conv4Out}, {transpose8Out}, (void*)transpose8Params, 24, "TRANSPOSE8");

    unsigned concatOutSizes[] = {32, 32, 8, 10};
    unsigned concatOut        = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "concatOut",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       concatOutSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concatOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned char concatParams[] = {2, 0, 0, 0};
    addNodeToGraph("concat",
                   {transpose2Out, transpose4Out, transpose6Out, transpose8Out},
                   {concatOut},
                   (void*)concatParams,
                   4,
                   "CONCAT");

    compareRunsResults({concatOut});
}

TEST_F_GC(SynTrainingSplitConcatMultiNodes, split_concat_multi_nodes)
{
    splitConcatMultiNodesTest(GCFG_ARC_SYNC_SCHEME_SIGNAL_LIMIT.value());
}

TEST_F_GC(SynTrainingSplitConcatMultiNodes, split_concat_multi_nodes_with_sob_reset, {synDeviceGaudi3})
{
    GlobalConfTestSetter conf("ENABLE_CONV_PACKING_TRAINING", "false");  // Disable packingMmeNodes pass
    splitConcatMultiNodesTest(7);
}

class SynTrainingSplitConcatNDimTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<TestNSizes /* concatenatedSize */,
                                                unsigned /* split dim */,
                                                bool /* is concat or split */,
                                                synDataType /* operands data type */>>
{
protected:
    void runTest();

    // get split dimension size for split input 'i'
    unsigned getSplitSize(unsigned i) const
    {
        unsigned splitDimSize = m_sizes[m_splitDim] / m_numSplits;
        return (i < m_numSplits - 1) ? splitDimSize : m_sizes[m_splitDim] - splitDimSize * i;
    }

    // get split input index for  0 <= elementIndex < m_sizes[splitDim]
    unsigned getSplitIndex(unsigned elementIndex)
    {
        unsigned splitDimSize = m_sizes[m_splitDim] / m_numSplits;
        return std::min(m_numSplits - 1, elementIndex / splitDimSize);
    }

    // get element index in split input 'splitIndex' for 0 <= elementIndex < m_sizes[splitDim]
    unsigned getSplitElementIndex(unsigned elementIndex, unsigned splitIndex)
    {
        unsigned splitDimSize = m_sizes[m_splitDim] / m_numSplits;
        return elementIndex - splitDimSize * splitIndex;
    }

    TestNSizes  m_sizes;
    unsigned    m_dim;
    unsigned    m_splitDim;
    unsigned    m_numSplits;
    bool        m_isConcat;
    synDataType m_dataType;
};

void SynTrainingSplitConcatNDimTest::runTest()
{
    TensorIndices splitTensors;

    unsigned concatTensor = createPersistTensor(m_isConcat ? OUTPUT_TENSOR : INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                m_sizes.data(),
                                                m_dim,
                                                m_dataType);

    TestNSizes splitSize = m_sizes;
    for (unsigned i = 0; i < m_numSplits; i++)
    {
        splitSize[m_splitDim] = getSplitSize(i);
        unsigned split        = createPersistTensor(m_isConcat ? INPUT_TENSOR : OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             splitSize.data(),
                                             m_dim,
                                             m_dataType);
        splitTensors.push_back(split);
    }

    if (m_isConcat)
    {
        addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                       splitTensors,
                       {concatTensor},
                       &m_splitDim,
                       sizeof(m_splitDim));
    }
    else
    {
        addNodeToGraph(NodeFactory::splitNodeTypeName, {concatTensor}, splitTensors, &m_splitDim, sizeof(m_splitDim));
    }

    compileAndRun();

    uint64_t outerSize = multiplyElements(m_sizes.begin(), m_sizes.begin() + m_splitDim);
    uint64_t innerSize = multiplyElements(m_sizes.begin() + m_splitDim + 1, m_sizes.end());

    for (uint64_t outerIdx = 0; outerIdx < outerSize; outerIdx++)
    {
        for (uint64_t splitDimIdx = 0; splitDimIdx < m_sizes[m_splitDim]; splitDimIdx++)
        {
            unsigned splitIndex   = getSplitIndex(splitDimIdx);  // index of split into tensor
            unsigned splitDimSize = getSplitSize(splitIndex);    // size of dimension in split into
            uint64_t splitElementIdx =
                getSplitElementIndex(splitDimIdx, splitIndex);  // current element index in split input
            for (uint64_t innerIdx = 0; innerIdx < innerSize; innerIdx++)
            {
                uint64_t concatenationOffset =
                    innerIdx + splitDimIdx * innerSize + outerIdx * innerSize * m_sizes[m_splitDim];
                uint64_t splitOffset = innerIdx + splitElementIdx * innerSize + outerIdx * innerSize * splitDimSize;

                float splitValue  = ((float*)m_hostBuffers[splitTensors[splitIndex]])[splitOffset];
                float concatValue = ((float*)m_hostBuffers[concatTensor])[concatenationOffset];

                ASSERT_EQ(splitValue, concatValue) << "Mismatch at split index " << splitIndex
                                                   << " Expected: " << splitValue << " Result: " << concatValue;
            }
        }
    }
}

TEST_P_GC(SynTrainingSplitConcatNDimTest, high_rank_split_concat_test, {synDeviceGaudi, synDeviceGaudi2})
{
    m_sizes     = std::get<0>(GetParam());
    m_splitDim  = std::get<1>(GetParam());
    m_isConcat  = std::get<2>(GetParam());
    m_dataType  = std::get<3>(GetParam());
    m_dim       = 6;
    m_numSplits = 3;

    runTest();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SynTrainingSplitConcatNDimTest,
    ::testing::Combine(::testing::ValuesIn({TestNSizes {3, 2, 128, 2, 2, 3}}),                  // sizes
                       ::testing::ValuesIn<unsigned>({unsigned(0), unsigned(2), unsigned(5)}),  // split dim
                       ::testing::ValuesIn({false, true}),                                      // is concat
                       ::testing::ValuesIn({syn_type_single, syn_type_int64})                   // data type
                       ));
