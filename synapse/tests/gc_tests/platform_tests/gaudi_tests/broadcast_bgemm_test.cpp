#include "syn_gaudi_two_run_compare_test.h"
TEST_F_GC(SynTrainingTwoRunCompareTest, broadcast_bgemm, {synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0_broadcast_first_run_0 node
     * inputs:
     *     g_0_broadcast_in[16, 16, 1, 8, 12] (dtype=float32)
     * outputs:
     *     g_0_broadcast_out[16, 16, 4, 8, 12] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_broadcast_in tensor
    unsigned g_0_broadcast_in_max_sizes[] = {16,16,1,1,12};
    unsigned g_0_broadcast_in_min_sizes[] = {16,16,1,1,12};
    unsigned g_0_broadcast_in = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_broadcast_in",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_broadcast_in_max_sizes,
                                          5,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_broadcast_in_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_broadcast_out tensor
    unsigned g_0_broadcast_out_max_sizes[] = {16,16,4,8,12};
    unsigned g_0_broadcast_out_min_sizes[] = {16,16,4,8,12};
    unsigned g_0_broadcast_out = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_broadcast_out",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_broadcast_out_max_sizes,
                                           5,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_broadcast_out_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_broadcast_first_run_0_id;
    addNodeToGraph("broadcast", {g_0_broadcast_in}, {g_0_broadcast_out}, nullptr, 0, "g_0_broadcast_first_run_0", 0 /*graphIndex*/, &g_0_broadcast_first_run_0_id);

    /*************
     * g_0_batch_gemm_first_run_0 node
     * inputs:
     *     g_0_broadcast_out[16, 16, 4, 8, 12] (dtype=float32)
     *     g_0_bgemm_other_input[16, 16, 4, 8, 12] (dtype=float32)
     * outputs:
     *     g_0_batch_gemm_out[16, 16, 4, 8, 12] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_bgemm_other_input tensor
    unsigned g_0_bgemm_other_input_max_sizes[] = {16,16,4,8,12};
    unsigned g_0_bgemm_other_input_min_sizes[] = {16,16,4,8,12};
    unsigned g_0_bgemm_other_input = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_bgemm_other_input",
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               g_0_bgemm_other_input_max_sizes,
                                               5,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_bgemm_other_input_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_batch_gemm_out tensor
    unsigned g_0_batch_gemm_out_max_sizes[] = {16,16,4,8,12};
    unsigned g_0_batch_gemm_out_min_sizes[] = {16,16,4,8,12};
    unsigned g_0_batch_gemm_out = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_batch_gemm_out",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_batch_gemm_out_max_sizes,
                                            5,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_batch_gemm_out_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_batch_gemm_first_run_0_id;
    unsigned char g_0_batch_gemm_first_run_0_params[] = {0,0};
    addNodeToGraph("batch_gemm", {g_0_broadcast_out, g_0_bgemm_other_input}, {g_0_batch_gemm_out}, (void*)g_0_batch_gemm_first_run_0_params, 2, "g_0_batch_gemm_first_run_0", 0 /*graphIndex*/, &g_0_batch_gemm_first_run_0_id);

    addConfigurationToRun(SECOND_RUN, "ENABLE_FUSE_BROADCAST_BGEMM", "false");
    addConfigurationToRun(SECOND_RUN, "MAKE_BROADCAST_PHYSICAL", "false");

    compareRunsResults({g_0_batch_gemm_out});
}
