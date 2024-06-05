#include "gc_gaudi_test_infra.h"
#include "syn_gaudi_two_run_compare_test.h"

class LNBwdGemmTest : public SynGaudiTwoRunCompareTest
{
};

TEST_F_GC(LNBwdGemmTest, layer_norm_bwd_gemm)
{
    // Graph #0

    /*************
     * LN_Bwd node
     * inputs:
     *     LN_input_0[128, 1, 1, 4608] (dtype=float32)
     *     LN_input_1[128, 1, 1, 4608] (dtype=float32)
     *     LN_input_2[1, 1, 1, 4608] (dtype=float32)
     *     LN_input_3[1, 1, 1, 4608] (dtype=float32)
     *     LN_input_4[128] (dtype=float32)
     * outputs:
     *     LN_output_0[128, 1, 1, 4608] (dtype=float32)
     *     LN_output_1[128] (dtype=float32)
     *     LN_output_2[128] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create LN_input_0 tensor
    unsigned LN_input_0_max_sizes[] = {128, 1, 1, 4608};
    unsigned LN_input_0_min_sizes[] = {128, 1, 1, 4608};
    unsigned LN_input_0             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "LN_input_0",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        LN_input_0_max_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        LN_input_0_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create LN_input_1 tensor
    unsigned LN_input_1_max_sizes[] = {128, 1, 1, 4608};
    unsigned LN_input_1_min_sizes[] = {128, 1, 1, 4608};
    unsigned LN_input_1             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "LN_input_1",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        LN_input_1_max_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        LN_input_1_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create LN_input_2 tensor
    unsigned LN_input_2_max_sizes[] = {1, 1, 1, 4608};
    unsigned LN_input_2_min_sizes[] = {1, 1, 1, 4608};
    unsigned LN_input_2             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "LN_input_2",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        LN_input_2_max_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        LN_input_2_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create LN_input_3 tensor
    unsigned LN_input_3_max_sizes[] = {1, 1, 1, 4608};
    unsigned LN_input_3_min_sizes[] = {1, 1, 1, 4608};
    unsigned LN_input_3             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "LN_input_3",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        LN_input_3_max_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        LN_input_3_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create LN_input_4 tensor
    unsigned LN_input_4_max_sizes[] = {128};
    unsigned LN_input_4_min_sizes[] = {128};
    unsigned LN_input_4             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "LN_input_4",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        LN_input_4_max_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        LN_input_4_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create LN_output_0 tensor
    unsigned LN_output_0_max_sizes[] = {128, 1, 1, 4608};
    unsigned LN_output_0_min_sizes[] = {128, 1, 1, 4608};
    unsigned LN_output_0             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "LN_output_0",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         LN_output_0_max_sizes,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         LN_output_0_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create LN_output_1 tensor
    unsigned LN_output_1_max_sizes[] = {128};
    unsigned LN_output_1_min_sizes[] = {128};
    unsigned LN_output_1             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "LN_output_1",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         LN_output_1_max_sizes,
                                         1,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         LN_output_1_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create LN_output_2 tensor
    unsigned      LN_output_2_max_sizes[] = {128};
    unsigned      LN_output_2_min_sizes[] = {128};
    unsigned      LN_output_2             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "LN_output_2",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         LN_output_2_max_sizes,
                                         1,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         LN_output_2_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     LN_Bwd_id;
    unsigned char LN_Bwd_params[] = {1, 0, 0, 0, 111, 18, 131, 58};
    addNodeToGraph("layer_norm_bwd_f32",
                   {LN_input_0, LN_input_1, LN_input_2, LN_input_3, LN_input_4},
                   {LN_output_0, LN_output_1, LN_output_2},
                   (void*)LN_Bwd_params,
                   8,
                   "LN_Bwd",
                   0 /*graphIndex*/,
                   &LN_Bwd_id);

    /*************
     * Reshape node
     * inputs:
     *     LN_output_0[128, 1, 1, 4608] (dtype=float32)
     *     Reshape_ST[128, 4608] (dtype=uint32) (shape tensor)
     * outputs:
     *     Reshape_output[128, 4608] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create Reshape_ST tensor
    unsigned Reshape_ST_max_sizes[] = {128, 4608};
    unsigned Reshape_ST_min_sizes[] = {128, 4608};
    unsigned Reshape_ST             = createTensors(1,
                                        INPUT_TENSOR,
                                        false,
                                        "Reshape_ST",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        Reshape_ST_max_sizes,
                                        2,
                                        syn_type_uint32,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        Reshape_ST_min_sizes,
                                        synTensorType::SHAPE_TENSOR)[0];

    // create Reshape_output tensor
    unsigned  Reshape_output_max_sizes[] = {128, 4608};
    unsigned  Reshape_output_min_sizes[] = {128, 4608};
    unsigned  Reshape_output             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "Reshape_output",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            Reshape_output_max_sizes,
                                            2,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            Reshape_output_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId Reshape_id;
    addNodeToGraph("reshape",
                   {LN_output_0, Reshape_ST},
                   {Reshape_output},
                   nullptr,
                   0,
                   "Reshape",
                   0 /*graphIndex*/,
                   &Reshape_id);

    /*************
     * Gemm node
     * inputs:
     *     Reshape_output[128, 4608] (dtype=float32)
     *     Gemm_input_1[128, 512] (dtype=float32)
     * outputs:
     *     Gemm_output[512, 4608] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create Gemm_input_1 tensor
    unsigned Gemm_input_1_max_sizes[] = {128, 512};
    unsigned Gemm_input_1_min_sizes[] = {128, 512};
    unsigned Gemm_input_1             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "Gemm_input_1",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          Gemm_input_1_max_sizes,
                                          2,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          Gemm_input_1_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create Gemm_output tensor
    unsigned      Gemm_output_max_sizes[] = {512, 4608};
    unsigned      Gemm_output_min_sizes[] = {512, 4608};
    unsigned      Gemm_output             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "Gemm_output",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         Gemm_output_max_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         Gemm_output_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     Gemm_id;
    unsigned char Gemm_params[] = {0, 1};
    addNodeToGraph("gemm",
                   {Reshape_output, Gemm_input_1},
                   {Gemm_output},
                   (void*)Gemm_params,
                   2,
                   "Gemm",
                   0 /*graphIndex*/,
                   &Gemm_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_EXPERIMENTAL_FLAGS", "true");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "ENABLE_EXPERIMENTAL_FLAGS", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({LN_output_1, LN_output_2, Gemm_output});
}
