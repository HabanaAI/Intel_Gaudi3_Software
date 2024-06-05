#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "gc_gaudi_test_infra.h"
#include "syn_gaudi_two_run_compare_test.h"

class SynGaudiDynamicTpcSlicesPatchingTest
: public SynGaudiTwoRunCompareTest
, public testing::WithParamInterface<int>
{
public:
    SynGaudiDynamicTpcSlicesPatchingTest() : m_actualBatchSize(GetParam()) {}

    void runSingleTest()
    {
        // The test creates a single convolution node with bias tensor
        // that will be transformed to convolution -> add.
        // The convolution is sliced on batch dim, which is dynamic.

        unsigned aMaxSizes[] = {256, 14, 14, 62};
        unsigned aMinSizes[] = {256, 14, 14, 14};
        unsigned a           = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "A",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   aMaxSizes,
                                   4,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   aMinSizes,
                                   synTensorType::DATA_TENSOR)[0];

        unsigned bMaxSizes[] = {256, 256, 3, 3};
        unsigned bMinSizes[] = {256, 256, 3, 3};
        unsigned b           = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "B",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   bMaxSizes,
                                   4,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   bMinSizes,
                                   synTensorType::DATA_TENSOR)[0];

        unsigned biasMaxSizes[] = {256};
        unsigned biasMinSizes[] = {256};
        unsigned bias           = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "BIAS",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      biasMaxSizes,
                                      1,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      biasMinSizes,
                                      synTensorType::DATA_TENSOR)[0];

        unsigned outMaxSizes[] = {256, 14, 14, 62};
        unsigned outMinSizes[] = {256, 14, 14, 14};
        unsigned out           = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "OUT",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     outMaxSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     outMinSizes,
                                     synTensorType::DATA_TENSOR)[0];

        unsigned char convParams[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,   0,   0, 0, 1, 0, 0, 0, 1, 0,
                                      0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 223, 137, 1, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0};
        addNodeToGraph("spatial_convolution", {a, b, bias}, {out}, (void*)convParams, 104, "CONV");

        unsigned actualSizes[] = {256, 14, 14, m_actualBatchSize};
        setActualSizes(a, actualSizes);
        setActualSizes(out, actualSizes);

        // Big-small config:
        addConfigurationToRun(FIRST_RUN, "IGNORE_INDEX_SPACE_FOR_SLICING", "false");
        addConfigurationToRun(FIRST_RUN, "ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "false");
        addConfigurationToRun(FIRST_RUN, "ENABLE_SLICER_RESHAPE_ALIGNMENT", "false");

        // The reference is unsliced graph:
        addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
        addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

        compareRunsResults({out});
    }

protected:
    unsigned m_actualBatchSize;
};

TEST_P_GC(SynGaudiDynamicTpcSlicesPatchingTest, dynamic_tpc_slices_patching)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(dynamic_tpc_slices_patching_full_ASIC_CI,
                         SynGaudiDynamicTpcSlicesPatchingTest,
                         ::testing::Range(14, 63));  // Actual batch size