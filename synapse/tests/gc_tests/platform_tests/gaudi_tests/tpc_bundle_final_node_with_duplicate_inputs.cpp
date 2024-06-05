#include "syn_gaudi_two_run_compare_test.h"

/*
 *    Graph #0:
 *    +-----+    +-----+    +-----+    +-----+    +-----+
 *    | MME +--->+ TPC1+--->+ TPC2+--->+ TPC3+--->+ MME |
 *    +-----+    +-----+    +-----+    +-----+    +-----+
 *
 *    TPC3 contains duplicate inputs.
 */
TEST_F_GC(SynGaudiTwoRunCompareTest, tpc_bundle_final_node_with_duplicate_inputs, {synDeviceGaudi})
{
    // Graph #0

    /*************
     * g_0_0_addmm_bf16_71_complex_gemm_0 node
     * inputs:
     *     tensor_53_id_1521_0_aten__view_1[4,2] (dtype=bf16)
     *     tensor_46_id_626_0_aten__view_1[4,4] (dtype=bf16)
     * outputs:
     *     g_0_bf16_1_830[4,2] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_53_id_1521_0_aten__view_1 tensor
    unsigned tensor_53_id_1521_0_aten_view_1_max_sizes[] = {4, 2};
    unsigned tensor_53_id_1521_0_aten_view_1_min_sizes[] = {4, 2};
    unsigned tensor_53_id_1521_0_aten_view_1             = createTensors(1,
                                                             INPUT_TENSOR,
                                                             true,
                                                             "tensor_53_id_1521_0_aten__view_1",
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             tensor_53_id_1521_0_aten_view_1_max_sizes,
                                                             2,
                                                             syn_type_bf16,
                                                             nullptr,
                                                             0,
                                                             0,
                                                             nullptr,
                                                             false,
                                                             tensor_53_id_1521_0_aten_view_1_min_sizes,
                                                             synTensorType::DATA_TENSOR)[0];
    // create tensor_46_id_626_0_aten__view_1 tensor
    unsigned tensor_46_id_626_0_aten_view_1_max_sizes[] = {4, 4};
    unsigned tensor_46_id_626_0_aten_view_1_min_sizes[] = {4, 4};
    unsigned tensor_46_id_626_0_aten_view_1             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "tensor_46_id_626_0_aten_view_1",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            tensor_46_id_626_0_aten_view_1_max_sizes,
                                                            2,
                                                            syn_type_bf16,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            tensor_46_id_626_0_aten_view_1_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    // create g_0_bf16_1_830 tensor
    unsigned g_0_bf16_1_830_max_sizes[] = {4, 2};
    unsigned g_0_bf16_1_830_min_sizes[] = {4, 2};
    unsigned g_0_bf16_1_830             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_bf16_1_830",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_bf16_1_830_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_bf16_1_830_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    synNodeId g_0_0_addmm_bf16_71_complex_gemm_0;
    addNodeToGraph("gemm",
                   {tensor_53_id_1521_0_aten_view_1, tensor_46_id_626_0_aten_view_1},
                   {g_0_bf16_1_830},
                   nullptr,
                   0,
                   "g_0_0_addmm_bf16_71_complex_gemm_0",
                   0 /*graphIndex*/,
                   &g_0_0_addmm_bf16_71_complex_gemm_0);

    /*************
     * g_0_0_addmm_bf16_71_complex_mult_bf16_2_complex_mult_bf16_0_0 node
     * inputs:
     *     g_0_bf16_1_830[4,2] (dtype=bf16)
     *     g_0_tensor_57[1] (dtype=bf16)
     * outputs:
     *     g_0_bf16_3_831[4,2] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_57 tensor
    unsigned g_0_tensor_57_max_sizes[] = {1};
    unsigned g_0_tensor_57_min_sizes[] = {1};
    unsigned g_0_tensor_57             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_57",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_57_max_sizes,
                                           1,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_57_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    // create g_0_bf16_3_831 tensor
    unsigned  g_0_bf16_3_831_max_sizes[] = {4, 2};
    unsigned  g_0_bf16_3_831_min_sizes[] = {4, 2};
    unsigned  g_0_bf16_3_831             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_bf16_3_831",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_bf16_3_831_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_bf16_3_831_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_0_addmm_bf16_71_complex_mult_bf16_2_complex_mult_bf16_0_0_id;
    addNodeToGraph("mult_bf16",
                   {g_0_bf16_1_830, g_0_tensor_57},
                   {g_0_bf16_3_831},
                   nullptr,
                   0,
                   "g_0_0_addmm_bf16_71_complex_mult_bf16_2_complex_mult_bf16_0_0",
                   0 /*graphIndex*/,
                   &g_0_0_addmm_bf16_71_complex_mult_bf16_2_complex_mult_bf16_0_0_id);

    /*************
     * g_0_0_addmm_bf16_71_complex_add_bf16_6_complex_add_bf16_0_0 node
     * inputs:
     *    g_0_bf16_3_831[4,2]
     *    g_0_bf16_5_832[4]
     * outputs:
     *     g_0_tensor_59[4, 2] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_bf16_5_832 tensor
    unsigned g_0_bf16_5_832_max_sizes[] = {4};
    unsigned g_0_bf16_5_832_min_sizes[] = {4};
    unsigned g_0_bf16_5_832             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_bf16_5_832",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_bf16_5_832_max_sizes,
                                            1,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_bf16_5_832_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    // create g_0_tensor_59 tensor
    unsigned  g_0_tensor_59_max_sizes[] = {4, 2};
    unsigned  g_0_tensor_59_min_sizes[] = {4, 2};
    unsigned  g_0_tensor_59             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_59",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_59_max_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_59_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_0_addmm_bf16_71_complex_add_bf16_6_complex_add_bf16_0_0_id;
    addNodeToGraph("add_bf16",
                   {g_0_bf16_3_831, g_0_bf16_5_832},
                   {g_0_tensor_59},
                   nullptr,
                   0,
                   "g_0_0_addmm_bf16_71_complex_add_bf16_6_complex_add_bf16_0_0",
                   0 /*graphIndex*/,
                   &g_0_0_addmm_bf16_71_complex_add_bf16_6_complex_add_bf16_0_0_id);

    /*************
     * g_0__add_fwd_bf16_81_complex_add_fwd_bf16_0_0 node
     * inputs:
     *     g_0_tensor_59[4, 2] (dtype=bf16)
     *     g_0_tensor_59[4, 2] (dtype=bf16)
     * outputs:
     *     g_0_tensor_70_id_1543_aten_add[4, 2] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_70_id_1543_aten_add tensor
    unsigned  g_0_tensor_70_id_1543_aten_add_max_sizes[] = {4, 2};
    unsigned  g_0_tensor_70_id_1543_aten_add_min_sizes[] = {4, 2};
    unsigned  g_0_tensor_70_id_1543_aten_add             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            false,
                                                            "g_0_tensor_70_id_1543_aten_add",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_70_id_1543_aten_add_max_sizes,
                                                            2,
                                                            syn_type_bf16,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_70_id_1543_aten_add_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_add_fwd_bf16_81_complex_add_fwd_bf16_0_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_tensor_59, g_0_tensor_59},
                   {g_0_tensor_70_id_1543_aten_add},
                   nullptr,
                   0,
                   "g_0__add_fwd_bf16_81_complex_add_fwd_bf16_0_0",
                   0 /*graphIndex*/,
                   &g_0_add_fwd_bf16_81_complex_add_fwd_bf16_0_0_id);

    /*************
     * g_0_0_addmm_bf16_94_complex_gemm_0
     * inputs:
     *     g_0_tensor_70_id_1543_aten_add[4, 2] (dtype=bf16)
     *     g_0_tensor_46_id_626_0_aten_view_1[4,4] (dtype=bf16)
     * outputs:
     *     g_0_bf16_1_839[4, 2] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_46_id_626_0_aten_view_1 tensor
    unsigned g_0_tensor_46_id_626_0_aten_view_1_max_sizes[] = {4, 4};
    unsigned g_0_tensor_46_id_626_0_aten_view_1_min_sizes[] = {4, 4};
    unsigned g_0_tensor_46_id_626_0_aten_view_1             = createTensors(1,
                                                                INPUT_TENSOR,
                                                                true,
                                                                "g_0_tensor_46_id_626_0_aten_view_1",
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                g_0_tensor_46_id_626_0_aten_view_1_max_sizes,
                                                                2,
                                                                syn_type_bf16,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_tensor_46_id_626_0_aten_view_1_min_sizes,
                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_bf16_1_839 tensor
    unsigned  g_0_bf16_1_839_max_sizes[] = {4, 2};
    unsigned  g_0_bf16_1_839_min_sizes[] = {4, 2};
    unsigned  g_0_bf16_1_839             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_bf16_1_839",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_bf16_1_839_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_bf16_1_839_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_0_addmm_bf16_94_complex_gemm_0_id;
    addNodeToGraph("gemm",
                   {g_0_tensor_70_id_1543_aten_add, g_0_tensor_46_id_626_0_aten_view_1},
                   {g_0_bf16_1_839},
                   nullptr,
                   0,
                   "g_0_0_addmm_bf16_94_complex_gemm_0",
                   0 /*graphIndex*/,
                   &g_0_0_addmm_bf16_94_complex_gemm_0_id);

    addConfigurationToRun(SECOND_RUN, "ENABLE_TPC_BUNDLES", "false");
    compareRunsResults({g_0_bf16_1_839});
}
