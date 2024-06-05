#include "syn_gaudi_two_run_compare_test.h"

// logsoftmax_fwd supposed to be split only on the batch dimmension.
// currently, it is on the block list until the split will be done on the outer dimension. [SW-75665]
TEST_F_GC(SynGaudiTwoRunCompareTest, graph_data_logsoftmax_L2)
{
    unsigned g_0_tensor_101_id_953_1_max_sizes[] = {512, 2};
    unsigned g_0_tensor_101_id_953_1_min_sizes[] = {512, 2};
    unsigned g_0_tensor_101_id_953_1             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_tensor_101_id_953_1",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_tensor_101_id_953_1_max_sizes,
                                                     2,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_tensor_101_id_953_1_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_86_id_955_max_sizes[] = {1000, 512};
    unsigned g_0_tensor_86_id_955_min_sizes[] = {1000, 512};
    unsigned g_0_tensor_86_id_955             = createTensors(1,
                                                  INPUT_TENSOR,
                                                  true,
                                                  "g_0_tensor_86_id_955",
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  g_0_tensor_86_id_955_max_sizes,
                                                  2,
                                                  syn_type_single,
                                                  nullptr,
                                                  0,
                                                  0,
                                                  nullptr,
                                                  false,
                                                  g_0_tensor_86_id_955_min_sizes,
                                                  synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_tensor_103_gemm_max_sizes[] = {1000, 2};
    unsigned      g_0_tensor_103_gemm_min_sizes[] = {1000, 2};
    unsigned      g_0_tensor_103_gemm             = createTensors(1,
                                                 OUTPUT_TENSOR,
                                                 false,
                                                 "g_0_tensor_103_gemm",
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 g_0_tensor_103_gemm_max_sizes,
                                                 2,
                                                 syn_type_single,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_tensor_103_gemm_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_GEMM460_0_id;
    unsigned char g_0_GEMM460_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_tensor_101_id_953_1, g_0_tensor_86_id_955},
                   {g_0_tensor_103_gemm},
                   (void*)g_0_GEMM460_0_params,
                   2,
                   "g_0_GEMM460_0",
                   0 /*graphIndex*/,
                   &g_0_GEMM460_0_id);

    unsigned g_0_tensor_102_max_sizes[] = {1000};
    unsigned g_0_tensor_102_min_sizes[] = {1000};
    unsigned g_0_tensor_102             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_102",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_102_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_102_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_tensor_104_id_959_max_sizes[] = {1000, 2};
    unsigned  g_0_tensor_104_id_959_min_sizes[] = {1000, 2};
    unsigned  g_0_tensor_104_id_959             = createTensors(1,
                                                   OUTPUT_TENSOR,
                                                   true,
                                                   "g_0_tensor_104_id_959",
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   g_0_tensor_104_id_959_max_sizes,
                                                   2,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_tensor_104_id_959_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_TPC461_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_tensor_102, g_0_tensor_103_gemm},
                   {g_0_tensor_104_id_959},
                   nullptr,
                   0,
                   "g_0_TPC461_0",
                   0 /*graphIndex*/,
                   &g_0_TPC461_0_id);

    unsigned      g_0_tensor_105_id_963_max_sizes[] = {1000, 2};
    unsigned      g_0_tensor_105_id_963_min_sizes[] = {1000, 2};
    unsigned      g_0_tensor_105_id_963             = createTensors(1,
                                                   OUTPUT_TENSOR,
                                                   false,
                                                   "g_0_tensor_105_id_963",
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   g_0_tensor_105_id_963_max_sizes,
                                                   2,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_tensor_105_id_963_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_TPC462_0_id;
    unsigned char g_0_TPC462_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("logsoftmax_fwd_f32",
                   {g_0_tensor_104_id_959},
                   {g_0_tensor_105_id_963},
                   (void*)g_0_TPC462_0_params,
                   4,
                   "g_0_TPC462_0",
                   0 /*graphIndex*/,
                   &g_0_TPC462_0_id);

    unsigned g_0_tensor_106_max_sizes[] = {2};
    unsigned g_0_tensor_106_min_sizes[] = {2};
    unsigned g_0_tensor_106             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_106",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_106_max_sizes,
                                            1,
                                            syn_type_int32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_106_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_107_id_965 tensor
    unsigned      nll_output_max_sizes[] = {1};
    unsigned      nll_output_min_sizes[] = {1};
    unsigned      nll_output             = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true,
                                        "g_0_tensor_107_id_965",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        nll_output_max_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        nll_output_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_TPC463_0_id;
    unsigned char g_0_TPC463_0_params[] = {0, 0, 0, 0, 156, 255, 255, 255};
    addNodeToGraph("nll_loss_fwd_f32",
                   {g_0_tensor_105_id_963, g_0_tensor_106},
                   {nll_output},
                   (void*)g_0_TPC463_0_params,
                   8,
                   "g_0_TPC463_0",
                   0 /*graphIndex*/,
                   &g_0_TPC463_0_id);

    unsigned g_0_tensor_110_id_971_max_sizes[] = {1000, 2};
    unsigned g_0_tensor_110_id_971_min_sizes[] = {1000, 2};
    unsigned g_0_tensor_110_id_971             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_tensor_110_id_971",
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   g_0_tensor_110_id_971_max_sizes,
                                                   2,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_tensor_110_id_971_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_tensor_112_max_sizes[] = {1000, 2};
    unsigned      g_0_tensor_112_min_sizes[] = {1000, 2};
    unsigned      g_0_tensor_112             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_112",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_tensor_112_max_sizes,
                                            2,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_112_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_TPC466_0_id;
    unsigned char g_0_TPC466_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("logsoftmax_bwd_f32",
                   {g_0_tensor_105_id_963, g_0_tensor_110_id_971},
                   {g_0_tensor_112},
                   (void*)g_0_TPC466_0_params,
                   4,
                   "g_0_TPC466_0",
                   0 /*graphIndex*/,
                   &g_0_TPC466_0_id);

    unsigned      gemm_output_max_sizes[] = {512, 1000};
    unsigned      gemm_output_min_sizes[] = {512, 1000};
    unsigned      gemm_output             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "gemm_output",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         gemm_output_max_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         gemm_output_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_GEMM539_0_id;
    unsigned char g_0_GEMM539_0_params[] = {1, 0};
    addNodeToGraph("gemm",
                   {g_0_tensor_112, g_0_tensor_101_id_953_1},
                   {gemm_output},
                   (void*)g_0_GEMM539_0_params,
                   2,
                   "g_0_GEMM539_0",
                   0 /*graphIndex*/,
                   &g_0_GEMM539_0_id);

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "18874368");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({gemm_output, nll_output});
}
