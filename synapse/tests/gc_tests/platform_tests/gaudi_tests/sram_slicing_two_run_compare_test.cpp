#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "global_conf_test_setter.h"
#include "habana_nodes.h"
#include "scoped_configuration_change.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tensor.h"

// Reproducer for SW-149744 - bundle from MNIST.
// The bundle includes DEDX+DEDW nodes with shared input producers chain: cast->maxpool->fused
// (the maxpool has AP with offsets).
TEST_F_GC(SynGaudiTwoRunCompareTest, shared_input_dedx_dedw_with_producers_with_non_strict_AP_ASIC_CI)
{
    // Graph #0

    /*************
     * g_0_cast_i32_to_u8_1532_0 node
     * inputs:
     *     g_0_tensor_20__placeholder_2[3, 3, 50, 64] (dtype=int32)
     * outputs:
     *     g_0_tensor_22[3, 3, 50, 64] (dtype=uint8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_20__placeholder_2 tensor
    unsigned g_0_tensor_20__placeholder_2_max_sizes[] = {3, 3, 50, 64};
    unsigned g_0_tensor_20__placeholder_2_min_sizes[] = {3, 3, 50, 64};
    unsigned g_0_tensor_20__placeholder_2             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_20__placeholder_2",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_20__placeholder_2_max_sizes,
                                                          4,
                                                          syn_type_int32,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_20__placeholder_2_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    setPermutationForTensor(g_0_tensor_20__placeholder_2, {2, 0, 1, 3});

    // create g_0_tensor_22 tensor
    unsigned      g_0_tensor_22_max_sizes[] = {3, 3, 50, 64};
    unsigned      g_0_tensor_22_min_sizes[] = {3, 3, 50, 64};
    unsigned      g_0_tensor_22             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_22",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_22_max_sizes,
                                           4,
                                           syn_type_uint8,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_22_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_cast_i32_to_u8_1532_0_id;
    unsigned char g_0_cast_i32_to_u8_1532_0_params[] = {4, 0, 0, 0};
    const char*   castInputLayouts[]                 = {"WHCN"};
    const char*   castOutputLayouts[]                = {"WHCN"};
    addNodeToGraph("cast_i32_to_u8",
                   {g_0_tensor_20__placeholder_2},
                   {g_0_tensor_22},
                   (void*)g_0_cast_i32_to_u8_1532_0_params,
                   4,
                   "g_0_cast_i32_to_u8_1532_0",
                   0 /*graphIndex*/,
                   &g_0_cast_i32_to_u8_1532_0_id,
                   castInputLayouts,
                   castOutputLayouts);

    /*************
     * g_0_maxpool_2d_bwd_f32_1533_0 node
     * inputs:
     *     g_0_tensor_16_view_3_1[3, 3, 50, 64] (dtype=float32)
     *     g_0_tensor_22[3, 3, 50, 64] (dtype=uint8)
     * outputs:
     *     g_0_tensor_23[7, 7, 50, 64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_16_view_3_1 tensor
    unsigned g_0_tensor_16_view_3_1_max_sizes[] = {3, 3, 50, 64};
    unsigned g_0_tensor_16_view_3_1_min_sizes[] = {3, 3, 50, 64};
    unsigned g_0_tensor_16_view_3_1             = createTensors(1,
                                                    INPUT_TENSOR,
                                                    true,
                                                    "g_0_tensor_16_view_3_1",
                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                    nullptr,
                                                    g_0_tensor_16_view_3_1_max_sizes,
                                                    4,
                                                    syn_type_single,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    g_0_tensor_16_view_3_1_min_sizes,
                                                    synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_23 tensor
    unsigned      g_0_tensor_23_max_sizes[] = {7, 7, 50, 64};
    unsigned      g_0_tensor_23_min_sizes[] = {7, 7, 50, 64};
    unsigned      g_0_tensor_23             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_23",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_23_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_23_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_maxpool_2d_bwd_f32_1533_0_id;
    unsigned char g_0_maxpool_2d_bwd_f32_1533_0_params[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                            0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 2, 0,
                                                            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    const char*   maxpoolInputLayouts[]                  = {"WHCN", "WHCN"};
    const char*   maxpoolOutputLayouts[]                 = {"WHCN"};
    addNodeToGraph("maxpool_2d_bwd_f32",
                   {g_0_tensor_16_view_3_1, g_0_tensor_22},
                   {g_0_tensor_23},
                   (void*)g_0_maxpool_2d_bwd_f32_1533_0_params,
                   44,
                   "g_0_maxpool_2d_bwd_f32_1533_0",
                   0 /*graphIndex*/,
                   &g_0_maxpool_2d_bwd_f32_1533_0_id,
                   maxpoolInputLayouts,
                   maxpoolOutputLayouts);

    /*************
     * g_0_relu_bwd_f32_1534_0 node
     * inputs:
     *     g_0_tensor_23[7, 7, 50, 64] (dtype=float32)
     *     g_0_tensor_19__placeholder_1[7, 7, 50, 64] (dtype=float32)
     * outputs:
     *     g_0_tensor_24_threshold_backward_1_1[7, 7, 50, 64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_19__placeholder_1 tensor
    unsigned g_0_tensor_19__placeholder_1_max_sizes[] = {7, 7, 50, 64};
    unsigned g_0_tensor_19__placeholder_1_min_sizes[] = {7, 7, 50, 64};
    unsigned g_0_tensor_19__placeholder_1             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_19__placeholder_1",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_19__placeholder_1_max_sizes,
                                                          4,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_19__placeholder_1_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_24_threshold_backward_1_1 tensor
    unsigned  g_0_tensor_24_threshold_backward_1_1_max_sizes[] = {7, 7, 50, 64};
    unsigned  g_0_tensor_24_threshold_backward_1_1_min_sizes[] = {7, 7, 50, 64};
    unsigned  g_0_tensor_24_threshold_backward_1_1             = createTensors(1,
                                                                  OUTPUT_TENSOR,
                                                                  false,
                                                                  "g_0_tensor_24_threshold_backward_1_1",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_tensor_24_threshold_backward_1_1_max_sizes,
                                                                  4,
                                                                  syn_type_single,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_tensor_24_threshold_backward_1_1_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_relu_bwd_f32_1534_0_id;
    addNodeToGraph("relu_bwd_f32",
                   {g_0_tensor_23, g_0_tensor_19__placeholder_1},
                   {g_0_tensor_24_threshold_backward_1_1},
                   nullptr,
                   0,
                   "g_0_relu_bwd_f32_1534_0",
                   0 /*graphIndex*/,
                   &g_0_relu_bwd_f32_1534_0_id);

    /*************
     * g_0_dedx_1535_0 node
     * inputs:
     *     g_0_tensor_24_threshold_backward_1_1[7, 7, 50, 64] (dtype=float32)
     *     g_0_tensor_26__placeholder_2[5, 5, 20, 50] (dtype=float32)
     * outputs:
     *     g_0_tensor_27_109[11, 11, 20, 64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_26__placeholder_2 tensor
    unsigned g_0_tensor_26__placeholder_2_max_sizes[] = {5, 5, 20, 50};
    unsigned g_0_tensor_26__placeholder_2_min_sizes[] = {5, 5, 20, 50};
    unsigned g_0_tensor_26__placeholder_2             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_26__placeholder_2",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_26__placeholder_2_max_sizes,
                                                          4,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_26__placeholder_2_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_27_109 tensor
    unsigned      g_0_tensor_27_109_max_sizes[] = {11, 11, 20, 64};
    unsigned      g_0_tensor_27_109_min_sizes[] = {11, 11, 20, 64};
    unsigned      g_0_tensor_27_109             = createTensors(1,
                                               OUTPUT_TENSOR,
                                               true,
                                               "g_0_tensor_27_109",
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               g_0_tensor_27_109_max_sizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_tensor_27_109_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_dedx_1535_0_id;
    unsigned char g_0_dedx_1535_0_params[] = {5, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const char*   dedxInputLayouts[]       = {"WHCN", "SRCK"};
    const char*   dedxOutputLayouts[]      = {"WHCN"};
    addNodeToGraph("dedx",
                   {g_0_tensor_24_threshold_backward_1_1, g_0_tensor_26__placeholder_2},
                   {g_0_tensor_27_109},
                   (void*)g_0_dedx_1535_0_params,
                   112,
                   "g_0_dedx_1535_0",
                   0 /*graphIndex*/,
                   &g_0_dedx_1535_0_id,
                   dedxInputLayouts,
                   dedxOutputLayouts);

    /*************
     * g_0_dedw_1536_0 node
     * inputs:
     *     g_0_tensor_24_threshold_backward_1_1[7, 7, 50, 64] (dtype=float32)
     *     g_0_tensor_25__placeholder_1[11, 11, 20, 64] (dtype=float32)
     * outputs:
     *     g_0_tensor_28_110[5, 5, 20, 50] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_25__placeholder_1 tensor
    unsigned g_0_tensor_25__placeholder_1_max_sizes[] = {11, 11, 20, 64};
    unsigned g_0_tensor_25__placeholder_1_min_sizes[] = {11, 11, 20, 64};
    unsigned g_0_tensor_25__placeholder_1             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_25__placeholder_1",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_25__placeholder_1_max_sizes,
                                                          4,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_25__placeholder_1_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_28_110 tensor
    unsigned      g_0_tensor_28_110_max_sizes[] = {5, 5, 20, 50};
    unsigned      g_0_tensor_28_110_min_sizes[] = {5, 5, 20, 50};
    unsigned      g_0_tensor_28_110             = createTensors(1,
                                               OUTPUT_TENSOR,
                                               true,
                                               "g_0_tensor_28_110",
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               g_0_tensor_28_110_max_sizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_tensor_28_110_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_dedw_1536_0_id;
    unsigned char g_0_dedw_1536_0_params[] = {5, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const char*   dedwInputLayouts[]       = {"WHCN", "WHCN"};
    const char*   dedwOutputLayouts[]      = {"SRCK"};
    addNodeToGraph("dedw",
                   {g_0_tensor_24_threshold_backward_1_1, g_0_tensor_25__placeholder_1},
                   {g_0_tensor_28_110},
                   (void*)g_0_dedw_1536_0_params,
                   112,
                   "g_0_dedw_1536_0",
                   0 /*graphIndex*/,
                   &g_0_dedw_1536_0_id,
                   dedwInputLayouts,
                   dedwOutputLayouts);

    /*************
     * g_0_reduce_sum_fwd_f32_1537_0 node
     * inputs:
     *     g_0_tensor_24_threshold_backward_1_1[7, 7, 50, 64] (dtype=float32)
     * outputs:
     *     g_0_tensor_30[7, 7, 50, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_30 tensor
    unsigned      g_0_tensor_30_max_sizes[] = {7, 7, 50, 1};
    unsigned      g_0_tensor_30_min_sizes[] = {7, 7, 50, 1};
    unsigned      g_0_tensor_30             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_30",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_30_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_30_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_reduce_sum_fwd_f32_1537_0_id;
    unsigned char g_0_reduce_sum_fwd_f32_1537_0_params[] = {3, 0, 0, 0};
    addNodeToGraph("reduce_sum_fwd_f32",
                   {g_0_tensor_24_threshold_backward_1_1},
                   {g_0_tensor_30},
                   (void*)g_0_reduce_sum_fwd_f32_1537_0_params,
                   4,
                   "g_0_reduce_sum_fwd_f32_1537_0",
                   0 /*graphIndex*/,
                   &g_0_reduce_sum_fwd_f32_1537_0_id);

    /*************
     * g_0_reduce_sum_fwd_f32_1538_0 node
     * inputs:
     *     g_0_tensor_30[7, 7, 50, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_31[7, 1, 50, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_31 tensor
    unsigned      g_0_tensor_31_max_sizes[] = {7, 1, 50, 1};
    unsigned      g_0_tensor_31_min_sizes[] = {7, 1, 50, 1};
    unsigned      g_0_tensor_31             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_31",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_31_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_31_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_reduce_sum_fwd_f32_1538_0_id;
    unsigned char g_0_reduce_sum_fwd_f32_1538_0_params[] = {1, 0, 0, 0};
    addNodeToGraph("reduce_sum_fwd_f32",
                   {g_0_tensor_30},
                   {g_0_tensor_31},
                   (void*)g_0_reduce_sum_fwd_f32_1538_0_params,
                   4,
                   "g_0_reduce_sum_fwd_f32_1538_0",
                   0 /*graphIndex*/,
                   &g_0_reduce_sum_fwd_f32_1538_0_id);

    /*************
     * g_0_reduce_sum_fwd_f32_1539_0 node
     * inputs:
     *     g_0_tensor_31[7, 1, 50, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_32[1, 1, 50, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_32 tensor
    unsigned      g_0_tensor_32_max_sizes[] = {1, 1, 50, 1};
    unsigned      g_0_tensor_32_min_sizes[] = {1, 1, 50, 1};
    unsigned      g_0_tensor_32             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_32",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_32_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_32_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_reduce_sum_fwd_f32_1539_0_id;
    unsigned char g_0_reduce_sum_fwd_f32_1539_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("reduce_sum_fwd_f32",
                   {g_0_tensor_31},
                   {g_0_tensor_32},
                   (void*)g_0_reduce_sum_fwd_f32_1539_0_params,
                   4,
                   "g_0_reduce_sum_fwd_f32_1539_0",
                   0 /*graphIndex*/,
                   &g_0_reduce_sum_fwd_f32_1539_0_id);

    // Force MantaRay bundlizer to enable bundling of multiple TPC producers in a chain.
    addConfigurationToRun(FIRST_RUN, "PIPELINE_MANAGEMENT_FORCE_BUNDLIZER", "2");
    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");

    compareRunsResults({g_0_tensor_27_109, g_0_tensor_28_110, g_0_tensor_32});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, bert_pt_fwd_bundle_unsliced_chain_ASIC_CI)
{
    unsigned g_0_tensor_357_max_sizes[] = {1024};
    unsigned g_0_tensor_357_min_sizes[] = {1024};
    unsigned g_0_tensor_357             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_357",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_357_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_357_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast_max_sizes[] = {1024};
    unsigned g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast_min_sizes[] = {1024};
    unsigned g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_3_attention_output_dense_cast_f32_to_bf16_226_0_id;
    unsigned char g_0_bert_encoder_3_attention_output_dense_cast_f32_to_bf16_226_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_357},
                   {g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast},
                   (void*)g_0_bert_encoder_3_attention_output_dense_cast_f32_to_bf16_226_0_params,
                   4,
                   "g_0_bert_encoder_3_attention_output_dense_cast_f32_to_bf16_226_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_attention_output_dense_cast_f32_to_bf16_226_0_id);

    unsigned g_0_tensor_402_id_1863_bert_encoder_3_attention_output_dense_aten__matmul_max_sizes[] = {1024, 512, 28};
    unsigned g_0_tensor_402_id_1863_bert_encoder_3_attention_output_dense_aten__matmul_min_sizes[] = {1024, 512, 28};
    unsigned g_0_tensor_402_id_1863_bert_encoder_3_attention_output_dense_aten__matmul =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_402_id_1863_bert_encoder_3_attention_output_dense_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_402_id_1863_bert_encoder_3_attention_output_dense_aten__matmul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_402_id_1863_bert_encoder_3_attention_output_dense_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add__max_sizes[] = {1024, 512, 28};
    unsigned g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add__min_sizes[] = {1024, 512, 28};
    unsigned g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add_ =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add_",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add__max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add__min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_add_fwd_bf16_290_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_tensor_402_id_1863_bert_encoder_3_attention_output_dense_aten__matmul,
                    g_0_tensor_358_id_1860_bert_encoder_3_attention_output_dense_hpu__cast},
                   {g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add_},
                   nullptr,
                   0,
                   "g_0_add_fwd_bf16_290_0",
                   0 /*graphIndex*/,
                   &g_0_add_fwd_bf16_290_0_id);

    unsigned g_0_tensor_404_max_sizes[] = {1};
    unsigned g_0_tensor_404_min_sizes[] = {1};
    unsigned g_0_tensor_404             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_404",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_404_max_sizes,
                                            1,
                                            syn_type_int32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_404_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_max_sizes[] = {1024,
                                                                                                               512,
                                                                                                               28};
    unsigned g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_min_sizes[] = {1024,
                                                                                                               512,
                                                                                                               28};
    unsigned g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_406_id_1871_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_max_sizes[] = {1024,
                                                                                                               512,
                                                                                                               28};
    unsigned g_0_tensor_406_id_1871_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_min_sizes[] = {1024,
                                                                                                               512,
                                                                                                               28};
    unsigned g_0_tensor_406_id_1871_bert_encoder_3_attention_output_dropout_hpu___fused_dropout =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_406_id_1871_bert_encoder_3_attention_output_dropout_hpu___fused_dropout",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_406_id_1871_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_max_sizes,
                      3,
                      syn_type_int8,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_406_id_1871_bert_encoder_3_attention_output_dropout_hpu___fused_dropout_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_3_attention_output_dropout_dropout_fwd_bf16_258_0_id;
    unsigned char g_0_bert_encoder_3_attention_output_dropout_dropout_fwd_bf16_258_0_params[] =
        {205, 204, 204, 61, 104, 127, 0, 0};
    addNodeToGraph("dropout_fwd_bf16",
                   {g_0_tensor_403_id_1863_bert_encoder_3_attention_output_dense_aten__add_, g_0_tensor_404},
                   {g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout,
                    g_0_tensor_406_id_1871_bert_encoder_3_attention_output_dropout_hpu___fused_dropout},
                   (void*)g_0_bert_encoder_3_attention_output_dropout_dropout_fwd_bf16_258_0_params,
                   8,
                   "g_0_bert_encoder_3_attention_output_dropout_dropout_fwd_bf16_258_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_attention_output_dropout_dropout_fwd_bf16_258_0_id);

    unsigned g_0_tensor_356_id_1774_bert_encoder_2_output_LayerNorm_aten__native_layer_norm_max_sizes[] = {1024,
                                                                                                           512,
                                                                                                           28};
    unsigned g_0_tensor_356_id_1774_bert_encoder_2_output_LayerNorm_aten__native_layer_norm_min_sizes[] = {1024,
                                                                                                           512,
                                                                                                           28};
    unsigned g_0_tensor_356_id_1774_bert_encoder_2_output_LayerNorm_aten__native_layer_norm =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_356_id_1774_bert_encoder_2_output_LayerNorm_aten__native_layer_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_356_id_1774_bert_encoder_2_output_LayerNorm_aten__native_layer_norm_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_356_id_1774_bert_encoder_2_output_LayerNorm_aten__native_layer_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add_max_sizes[] = {1024, 512, 28};
    unsigned g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add_min_sizes[] = {1024, 512, 28};
    unsigned g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_3_attention_output_add_fwd_bf16_259_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_tensor_405_id_1869_bert_encoder_3_attention_output_dropout_hpu___fused_dropout,
                    g_0_tensor_356_id_1774_bert_encoder_2_output_LayerNorm_aten__native_layer_norm},
                   {g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add},
                   nullptr,
                   0,
                   "g_0_bert_encoder_3_attention_output_add_fwd_bf16_259_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_attention_output_add_fwd_bf16_259_0_id);

    unsigned  g_0_tensor_410_max_sizes[] = {1024, 14336, 1, 1};
    unsigned  g_0_tensor_410_min_sizes[] = {1024, 14336, 1, 1};
    unsigned  g_0_tensor_410             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_410",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_410_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_410_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_3_attention_output_LayerNorm_reshape_260_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_407_id_1874_bert_encoder_3_attention_output_aten__add},
                   {g_0_tensor_410},
                   nullptr,
                   0,
                   "g_0_bert_encoder_3_attention_output_LayerNorm_reshape_260_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_attention_output_LayerNorm_reshape_260_0_id);

    unsigned g_0_tensor_408_max_sizes[] = {1024};
    unsigned g_0_tensor_408_min_sizes[] = {1024};
    unsigned g_0_tensor_408             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_408",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_408_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_408_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_tensor_412_max_sizes[] = {1024};
    unsigned  g_0_tensor_412_min_sizes[] = {1024};
    unsigned  g_0_tensor_412             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_412",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_412_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_412_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_3_attention_output_LayerNorm_reshape_262_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_408},
                   {g_0_tensor_412},
                   nullptr,
                   0,
                   "g_0_bert_encoder_3_attention_output_LayerNorm_reshape_262_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_attention_output_LayerNorm_reshape_262_0_id);

    unsigned g_0_tensor_411_max_sizes[] = {1024};
    unsigned g_0_tensor_411_min_sizes[] = {1024};
    unsigned g_0_tensor_411             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_411",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_411_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_411_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_413_max_sizes[] = {1024, 14336, 1, 1};
    unsigned g_0_tensor_413_min_sizes[] = {1024, 14336, 1, 1};
    unsigned g_0_tensor_413             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_413",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_413_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_413_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_414_id_1879_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_max_sizes[] =
        {1, 14336, 1, 1};
    unsigned g_0_tensor_414_id_1879_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_min_sizes[] =
        {1, 14336, 1, 1};
    unsigned g_0_tensor_414_id_1879_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_tensor_414_id_1879_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_414_id_1879_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_max_sizes,
        4,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_414_id_1879_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_415_id_1881_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_max_sizes[] =
        {1, 14336, 1, 1};
    unsigned g_0_tensor_415_id_1881_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_min_sizes[] =
        {1, 14336, 1, 1};
    unsigned g_0_tensor_415_id_1881_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_tensor_415_id_1881_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_415_id_1881_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_max_sizes,
        4,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_415_id_1881_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_3_attention_output_LayerNorm_layer_norm_fwd_bf16_263_0_id;
    unsigned char g_0_bert_encoder_3_attention_output_LayerNorm_layer_norm_fwd_bf16_263_0_params[] =
        {1, 54, 205, 106, 204, 188, 140, 43};
    addNodeToGraph("layer_norm_fwd_bf16",
                   {g_0_tensor_410, g_0_tensor_411, g_0_tensor_412},
                   {g_0_tensor_413,
                    g_0_tensor_414_id_1879_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm,
                    g_0_tensor_415_id_1881_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm},
                   (void*)g_0_bert_encoder_3_attention_output_LayerNorm_layer_norm_fwd_bf16_263_0_params,
                   8,
                   "g_0_bert_encoder_3_attention_output_LayerNorm_layer_norm_fwd_bf16_263_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_attention_output_LayerNorm_layer_norm_fwd_bf16_263_0_id);

    unsigned g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_max_sizes[] = {
        1024,
        512,
        28};
    unsigned g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_min_sizes[] = {
        1024,
        512,
        28};
    unsigned g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_max_sizes,
        3,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_3_attention_output_LayerNorm_reshape_264_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_413},
                   {g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm},
                   nullptr,
                   0,
                   "g_0_bert_encoder_3_attention_output_LayerNorm_reshape_264_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_attention_output_LayerNorm_reshape_264_0_id);

    unsigned g_0_tensor_424_max_sizes[] = {1024, 4096};
    unsigned g_0_tensor_424_min_sizes[] = {1024, 4096};
    unsigned g_0_tensor_424             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_424",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_424_max_sizes,
                                            2,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_424_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast_max_sizes[] = {1024, 4096};
    unsigned g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast_min_sizes[] = {1024, 4096};
    unsigned g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_3_intermediate_dense_act_cast_f32_to_bf16_269_0_id;
    unsigned char g_0_bert_encoder_3_intermediate_dense_act_cast_f32_to_bf16_269_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_424},
                   {g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast},
                   (void*)g_0_bert_encoder_3_intermediate_dense_act_cast_f32_to_bf16_269_0_params,
                   4,
                   "g_0_bert_encoder_3_intermediate_dense_act_cast_f32_to_bf16_269_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_intermediate_dense_act_cast_f32_to_bf16_269_0_id);

    unsigned g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t_max_sizes[] = {4096, 1024};
    unsigned g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t_min_sizes[] = {4096, 1024};
    unsigned g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_3_intermediate_dense_act_transpose_270_0_id;
    unsigned char g_0_bert_encoder_3_intermediate_dense_act_transpose_270_0_params[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_425_id_1885_bert_encoder_3_intermediate_dense_act_hpu__cast},
                   {g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t},
                   (void*)g_0_bert_encoder_3_intermediate_dense_act_transpose_270_0_params,
                   24,
                   "g_0_bert_encoder_3_intermediate_dense_act_transpose_270_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_intermediate_dense_act_transpose_270_0_id);

    unsigned g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul_max_sizes[] = {4096, 512, 28};
    unsigned g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul_min_sizes[] = {4096, 512, 28};
    unsigned g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_3_intermediate_dense_act_batch_gemm_271_0_id;
    unsigned char g_0_bert_encoder_3_intermediate_dense_act_batch_gemm_271_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_416_id_1877_bert_encoder_3_attention_output_LayerNorm_aten__native_layer_norm,
                    g_0_tensor_426_id_1889_bert_encoder_3_intermediate_dense_act_aten__t},
                   {g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul},
                   (void*)g_0_bert_encoder_3_intermediate_dense_act_batch_gemm_271_0_params,
                   2,
                   "g_0_bert_encoder_3_intermediate_dense_act_batch_gemm_271_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_3_intermediate_dense_act_batch_gemm_271_0_id);

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_427_id_1891_bert_encoder_3_intermediate_dense_act_aten__matmul});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, tpc_producer_with_input_overlap_and_offset_ASIC_CI)
{
    unsigned g_0_t8363_downstream_model_MetroNet_batch_normalization_1_FusedBatchNormV3_0_max_sizes[] = {64,
                                                                                                         480,
                                                                                                         240,
                                                                                                         8};
    unsigned g_0_t8363_downstream_model_MetroNet_batch_normalization_1_FusedBatchNormV3_0_min_sizes[] = {64,
                                                                                                         480,
                                                                                                         240,
                                                                                                         8};
    unsigned g_0_t8363_downstream_model_MetroNet_batch_normalization_1_FusedBatchNormV3_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t8363_downstream_model_MetroNet_batch_normalization_1_FusedBatchNormV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t8363_downstream_model_MetroNet_batch_normalization_1_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t8363_downstream_model_MetroNet_batch_normalization_1_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0_max_sizes[] = {64,
                                                                                                        960,
                                                                                                        480,
                                                                                                        8};
    unsigned g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0_min_sizes[] = {64,
                                                                                                        960,
                                                                                                        480,
                                                                                                        8};
    unsigned g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_resize_fwd_f32_n3460_0_id;
    unsigned char g_0_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_resize_fwd_f32_n3460_0_params[] =
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 3, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 192, 3, 0, 0, 224, 1, 0, 0, 8, 0, 0, 0};
    addNodeToGraph(
        "resize_fwd_f32",
        {g_0_t8363_downstream_model_MetroNet_batch_normalization_1_FusedBatchNormV3_0},
        {g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0},
        (void*)g_0_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_resize_fwd_f32_n3460_0_params,
        44,
        "g_0_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_resize_fwd_f32_n3460_0",
        0 /*graphIndex*/,
        &g_0_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_resize_fwd_f32_n3460_0_id);

    unsigned g_0_t3443_downstream_model_metronet_final_redu_conv2d_readvariableop_0_max_sizes[] = {12, 64, 1, 1};
    unsigned g_0_t3443_downstream_model_metronet_final_redu_conv2d_readvariableop_0_min_sizes[] = {12, 64, 1, 1};
    unsigned g_0_t3443_downstream_model_metronet_final_redu_conv2d_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3443_downstream_model_metronet_final_redu_conv2d_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3443_downstream_model_metronet_final_redu_conv2d_readvariableop_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3443_downstream_model_metronet_final_redu_conv2d_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0 tensor
    unsigned g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0_max_sizes[] = {12, 960, 480, 8};
    unsigned g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0_min_sizes[] = {12, 960, 480, 8};
    unsigned g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_downstream_model_MetroNet_final_redu_Conv2D_spatial_convolution_n3461_0_id;
    unsigned char g_0_downstream_model_MetroNet_final_redu_Conv2D_spatial_convolution_n3461_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t8382_downstream_model_MetroNet_up_sampling2d_2_resize_ResizeBilinear_0,
                    g_0_t3443_downstream_model_metronet_final_redu_conv2d_readvariableop_0},
                   {g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0},
                   (void*)g_0_downstream_model_MetroNet_final_redu_Conv2D_spatial_convolution_n3461_0_params,
                   104,
                   "g_0_downstream_model_MetroNet_final_redu_Conv2D_spatial_convolution_n3461_0",
                   0 /*graphIndex*/,
                   &g_0_downstream_model_MetroNet_final_redu_Conv2D_spatial_convolution_n3461_0_id);

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_t8383_downstream_model_MetroNet_final_redu_Conv2D_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, gemm_with_avg_pool_2d_bwd_f32)
{
    // Make sure bundle expansion is done correctly when we have: gemm -> reshape -> transpose -> avg_pool_2d_bwd_f32.
    // Compare a run with bundle expansion enabled to a run with bundle expansion disabled.

    /*************
     * GEMM28274 node
     * inputs: [tensor_1239[1000, 128](dtype=float32), tensor_1238[2048, 1000](dtype=float32)]
     * output: [tensor_1240[2048, 128](dtype=float32)]
     *************/

    // create tensor_1239 tensor
    unsigned tensor_1239_sizes[] = {1000, 128};
    unsigned tensor_1239         = createTensors(1,
                                         INPUT_TENSOR,
                                         true,  // isPersistent
                                         "tensor_1239",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,  // initializer
                                         tensor_1239_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false)[0];

    // create tensor_1238 tensor
    unsigned tensor_1238_sizes[] = {2048, 1000};
    unsigned tensor_1238         = createTensors(1,
                                         INPUT_TENSOR,
                                         true,  // isPersistent
                                         "tensor_1238",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,  // initializer
                                         tensor_1238_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false)[0];

    // create tensor_1240 tensor
    unsigned      tensor_1240_sizes[] = {2048, 128};
    unsigned      tensor_1240         = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,  // isPersistent
                                         "tensor_1240",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,  // initializer
                                         tensor_1240_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false)[0];
    unsigned char GEMM28274_params[]  = {0, 0};
    addNodeToGraph("gemm", {tensor_1239, tensor_1238}, {tensor_1240}, (void*)GEMM28274_params, 2, "GEMM28274");

    /*************
     * Reshape28275 node
     * inputs: [tensor_1240[2048, 128](dtype=float32)]
     * output: [tensor_1241[1, 1, 2048, 128](dtype=float32)]
     *************/

    // create tensor_1241 tensor
    unsigned tensor_1241_sizes[] = {1, 1, 2048, 128};
    unsigned tensor_1241         = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,  // isPersistent
                                         "tensor_1241",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,  // initializer
                                         tensor_1241_sizes,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false)[0];
    addNodeToGraph("reshape", {tensor_1240}, {tensor_1241}, nullptr, 0, "Reshape28275");

    /*************
     * Transpose28276 node
     * inputs: [tensor_1241[1, 1, 2048, 128](dtype=float32)]
     * output: [tensor_1242[2048, 1, 1, 128](dtype=float32)]
     *************/

    // create tensor_1242 tensor
    unsigned      tensor_1242_sizes[]     = {2048, 1, 1, 128};
    unsigned      tensor_1242             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,  // isPersistent
                                         "tensor_1242",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,  // initializer
                                         tensor_1242_sizes,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false)[0];
    unsigned char Transpose28276_params[] = {2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose", {tensor_1241}, {tensor_1242}, (void*)Transpose28276_params, 24, "Transpose28276");

    /*************
     * TPC28277 node
     * inputs: [tensor_1242[2048, 1, 1, 128](dtype=float32)]
     * output: [tensor_1243[2048, 7, 7, 128](dtype=float32)]
     *************/

    // create tensor_1243 tensor
    unsigned      tensor_1243_sizes[] = {2048, 7, 7, 128};
    unsigned      tensor_1243         = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,  // isPersistent
                                         "tensor_1243",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,  // initializer
                                         tensor_1243_sizes,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false)[0];
    unsigned char TPC28277_params[]   = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0,
                                       7, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    addNodeToGraph("avg_pool_2d_bwd_f32", {tensor_1242}, {tensor_1243}, (void*)TPC28277_params, 48, "TPC28277");

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", "true");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", "false");

    compareRunsResults({tensor_1243});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, tpc_bundle_with_parallel_tpcs_ASIC_CI)
{
    ScopedConfigurationChange fuserCfg("RUN_TPC_FUSER", "false");

    unsigned gemmIn1Sizes[] = {4096, 7168};
    unsigned gemmIn1        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemmIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemmIn1Sizes,
                                     2,
                                     syn_type_bf16)[0];

    unsigned gemmIn2Sizes[] = {1024, 4096};
    unsigned gemmIn2        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemmIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemmIn2Sizes,
                                     2,
                                     syn_type_bf16)[0];

    unsigned gemmOutSizes[] = {1024, 7168};
    unsigned gemmOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     gemmOutSizes,
                                     2,
                                     syn_type_bf16)[0];

    unsigned char gemmParams[] = {0, 0};
    addNodeToGraph("gemm", {gemmIn1, gemmIn2}, {gemmOut}, (void*)gemmParams, 2, "gemm");

    unsigned add1InSizes[] = {1024, 1};
    unsigned add1In        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "add1In",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    add1InSizes,
                                    2,
                                    syn_type_bf16)[0];

    unsigned add1OutSizes[] = {1024, 7168};
    unsigned add1Out        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "add1Out",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     add1OutSizes,
                                     2,
                                     syn_type_bf16)[0];

    addNodeToGraph("add_fwd_bf16", {gemmOut, add1In}, {add1Out}, nullptr, 0, "add1");

    unsigned relu1OutSizes[] = {1024, 7168};
    unsigned relu1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "relu1Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      relu1OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("relu_fwd_bf16", {add1Out}, {relu1Out}, nullptr, 0, "relu1");

    unsigned add2InSizes[] = {1024, 7168};
    unsigned add2In        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "add2In",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    add2InSizes,
                                    2,
                                    syn_type_bf16)[0];

    unsigned add2OutSizes[] = {1024, 7168};
    unsigned add2Out        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "add2Out",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     add2OutSizes,
                                     2,
                                     syn_type_bf16)[0];

    addNodeToGraph("add_fwd_bf16", {add2In, relu1Out}, {add2Out}, nullptr, 0, "add2");

    unsigned relu2OutSizes[] = {1024, 7168};
    unsigned relu2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "relu2Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      relu2OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("relu_fwd_bf16", {add2Out}, {relu2Out}, nullptr, 0, "relu2");

    unsigned cast1InSizes[] = {1024, 1024};
    unsigned cast1In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "cast1In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     cast1InSizes,
                                     2,
                                     syn_type_single)[0];

    unsigned cast1OutSizes[] = {1024, 1024};
    unsigned cast1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "cast1Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      cast1OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("cast_f32_to_bf16", {cast1In}, {cast1Out}, nullptr, 0, "cast1");

    unsigned gemm1OutSizes[] = {1024, 7168};
    unsigned gemm1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm1Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      gemm1OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    unsigned char gemm1Params[] = {0, 0};
    addNodeToGraph("gemm", {relu2Out, cast1Out}, {gemm1Out}, (void*)gemm1Params, 2, "gemm1");

    unsigned cast2InSizes[] = {1024, 1024};
    unsigned cast2In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "cast2In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     cast2InSizes,
                                     2,
                                     syn_type_single)[0];

    unsigned cast2OutSizes[] = {1024, 1024};
    unsigned cast2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "cast2Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      cast2OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("cast_f32_to_bf16", {cast2In}, {cast2Out}, nullptr, 0, "cast2");

    unsigned gemm2OutSizes[] = {1024, 7168};
    unsigned gemm2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm2Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      gemm2OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    unsigned char gemm2Params[] = {0, 0};
    addNodeToGraph("gemm", {relu2Out, cast2Out}, {gemm2Out}, (void*)gemm2Params, 2, "gemm2");

    unsigned cast3InSizes[] = {1024, 1024};
    unsigned cast3In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "cast3In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     cast3InSizes,
                                     2,
                                     syn_type_single)[0];

    unsigned cast3OutSizes[] = {1024, 1024};
    unsigned cast3Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "cast3Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      cast3OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("cast_f32_to_bf16", {cast3In}, {cast3Out}, nullptr, 0, "cast3");

    unsigned gemm3OutSizes[] = {1024, 7168};
    unsigned gemm3Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm3Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      gemm3OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    unsigned char gemm3Params[] = {0, 0};
    addNodeToGraph("gemm", {relu2Out, cast3Out}, {gemm3Out}, (void*)gemm3Params, 2, "gemm3");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_BUNDLES", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_TPC_BUNDLES", "true");

    compareRunsResults({gemm1Out, gemm2Out, gemm3Out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, dedw_dedx_with_2_consumers)
{
    /*************
     * n2047_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput node
     * inputs: [t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0[32, 7, 7,
     *128](dtype=bf16), t616_F_TO_BF_Cast_118_0[32, 128, 3, 3](dtype=bf16)] output:
     *[t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0[128, 7, 7,
     *128](dtype=bf16)]
     *************/

    // create t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0 tensor
    unsigned t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0_sizes[] = {32,
                                                                                                          7,
                                                                                                          7,
                                                                                                          128};
    unsigned t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,  // isPersistent
                      "t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,  // initializer
                      t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false)[0];

    // create t616_F_TO_BF_Cast_118_0 tensor
    unsigned t616_F_TO_BF_Cast_118_0_sizes[] = {32, 128, 3, 3};
    unsigned t616_F_TO_BF_Cast_118_0         = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,  // isPersistent
                                                     "t616_F_TO_BF_Cast_118_0",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,  // initializer
                                                     t616_F_TO_BF_Cast_118_0_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false)[0];

    // create t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0 tensor
    unsigned t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0_sizes[] =
        {128, 7, 7, 128};
    unsigned t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,  // isPersistent
            "t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0",
            MEM_INIT_ALL_ZERO,
            nullptr,  // initializer
            t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false)[0];
    unsigned char n2047_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_params[] =
        {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1,  0,   0,   0,   1,  0,  0, 0, 1, 0, 0, 0, 1,   0,   0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 49, 54,  1,   0,   0,  0,  0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  126, 188, 173, 77, 86, 0, 0, 1, 0, 0, 0, 255, 127, 0, 0};
    addNodeToGraph(
        "dedx",
        {t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0, t616_F_TO_BF_Cast_118_0},
        {t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0},
        (void*)n2047_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_params,
        88,
        "n2047_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput");

    /*************
     * n2048_BF_TO_F_Cast_460 node
     * inputs: [t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0[128, 7, 7,
     *128](dtype=bf16)] output: [t3152_BF_TO_F_Cast_460_0[128, 7, 7, 128](dtype=float32)]
     *************/

    // create t3152_BF_TO_F_Cast_460_0 tensor
    unsigned t3152_BF_TO_F_Cast_460_0_sizes[] = {128, 7, 7, 128};
    unsigned t3152_BF_TO_F_Cast_460_0         = createTensors(1,
                                                      OUTPUT_TENSOR,
                                                      true,  // isPersistent
                                                      "t3152_BF_TO_F_Cast_460_0",
                                                      MEM_INIT_ALL_ZERO,
                                                      nullptr,  // initializer
                                                      t3152_BF_TO_F_Cast_460_0_sizes,
                                                      4,
                                                      syn_type_single,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false)[0];
    addNodeToGraph("cast_bf16_to_f32",
                   {t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0},
                   {t3152_BF_TO_F_Cast_460_0},
                   nullptr,
                   0,
                   "n2048_BF_TO_F_Cast_460");

    /*************
     * n2045_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter node
     * inputs: [t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0[32, 7, 7,
     *128](dtype=bf16), t3093_conv5_block16_1_relu_Relu_0[128, 7, 7, 128](dtype=bf16)] output:
     *[t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0[32, 128, 3,
     *3](dtype=bf16)]
     *************/

    // create t3093_conv5_block16_1_relu_Relu_0 tensor
    unsigned t3093_conv5_block16_1_relu_Relu_0_sizes[] = {128, 7, 7, 128};
    unsigned t3093_conv5_block16_1_relu_Relu_0         = createTensors(1,
                                                               INPUT_TENSOR,
                                                               true,  // isPersistent
                                                               "t3093_conv5_block16_1_relu_Relu_0",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,  // initializer
                                                               t3093_conv5_block16_1_relu_Relu_0_sizes,
                                                               4,
                                                               syn_type_bf16,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false)[0];

    // create t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0 tensor
    unsigned t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0_sizes[] =
        {32, 128, 3, 3};
    unsigned t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,  // isPersistent
            "t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0",
            MEM_INIT_ALL_ZERO,
            nullptr,  // initializer
            t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false)[0];
    unsigned char
        n2045_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_params[] = {
            3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1,  0,  0,   0,   1,  0,  0, 0, 1, 0, 0, 0, 1,   0,   0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 49, 54, 1,   0,   0,  0,  0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  53, 188, 173, 77, 86, 0, 0, 1, 0, 0, 0, 255, 127, 0, 0};
    addNodeToGraph(
        "dedw",
        {t3148_training_SGD_gradients_gradients_conv5_block16_concat_concat_grad_Slice_1_0,
         t3093_conv5_block16_1_relu_Relu_0},
        {t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0},
        (void*)n2045_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_params,
        88,
        "n2045_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter");

    /*************
     * n2049_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad node
     * inputs: [t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0[128, 7, 7,
     *128](dtype=bf16), t3093_conv5_block16_1_relu_Relu_0[128, 7, 7, 128](dtype=bf16)] output:
     *[t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0[128, 7, 7, 128](dtype=bf16)]
     *************/

    // create t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0 tensor
    unsigned t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0_sizes[] = {128,
                                                                                                         7,
                                                                                                         7,
                                                                                                         128};
    unsigned t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,  // isPersistent
                      "t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,  // initializer
                      t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false)[0];
    addNodeToGraph("relu_bwd_bf16",
                   {t3151_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropInput_0,
                    t3093_conv5_block16_1_relu_Relu_0},
                   {t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0},
                   nullptr,
                   0,
                   "n2049_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad");

    /*************
     * n2054_training_SGD_gradients_gradients_conv5_block16_1_bn_FusedBatchNormV3_grad_FusedBatchNormGradV3 node
     * inputs: [t3075_conv5_block16_1_conv_Conv2D_0[128, 7, 7, 128](dtype=bf16),
     *t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0[128, 7, 7, 128](dtype=bf16),
     *t3077_conv5_block16_1_bn_FusedBatchNormV3_hfbn_v3_1[128](dtype=float32), t3160[128](dtype=float32),
     *t254_conv5_block16_1_bn_readvariableop_0[128](dtype=float32)] output: [t3154[128, 7, 7, 128](dtype=bf16),
     *t3156[128](dtype=float32), t3155[128](dtype=float32)]
     *************/

    // create t3075_conv5_block16_1_conv_Conv2D_0 tensor
    unsigned t3075_conv5_block16_1_conv_Conv2D_0_sizes[] = {128, 7, 7, 128};
    unsigned t3075_conv5_block16_1_conv_Conv2D_0         = createTensors(1,
                                                                 INPUT_TENSOR,
                                                                 true,  // isPersistent
                                                                 "t3075_conv5_block16_1_conv_Conv2D_0",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,  // initializer
                                                                 t3075_conv5_block16_1_conv_Conv2D_0_sizes,
                                                                 4,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false)[0];

    // create t3077_conv5_block16_1_bn_FusedBatchNormV3_hfbn_v3_1 tensor
    unsigned t3077_conv5_block16_1_bn_FusedBatchNormV3_hfbn_v3_1_sizes[] = {128};
    unsigned t3077_conv5_block16_1_bn_FusedBatchNormV3_hfbn_v3_1 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,  // isPersistent
                      "t3077_conv5_block16_1_bn_FusedBatchNormV3_hfbn_v3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,  // initializer
                      t3077_conv5_block16_1_bn_FusedBatchNormV3_hfbn_v3_1_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false)[0];

    // create t3160 tensor
    unsigned t3160_sizes[] = {128};
    unsigned t3160         = createTensors(1,
                                   INPUT_TENSOR,
                                   true,  // isPersistent
                                   "t3160",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,  // initializer
                                   t3160_sizes,
                                   1,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false)[0];

    // create t254_conv5_block16_1_bn_readvariableop_0 tensor
    unsigned t254_conv5_block16_1_bn_readvariableop_0_sizes[] = {128};
    unsigned t254_conv5_block16_1_bn_readvariableop_0         = createTensors(1,
                                                                      INPUT_TENSOR,
                                                                      true,  // isPersistent
                                                                      "t254_conv5_block16_1_bn_readvariableop_0",
                                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                      nullptr,  // initializer
                                                                      t254_conv5_block16_1_bn_readvariableop_0_sizes,
                                                                      1,
                                                                      syn_type_single,
                                                                      nullptr,
                                                                      0,
                                                                      0,
                                                                      nullptr,
                                                                      false)[0];

    // create t3154 tensor
    unsigned t3154_sizes[] = {128, 7, 7, 128};
    unsigned t3154         = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,  // isPersistent
                                   "t3154",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,  // initializer
                                   t3154_sizes,
                                   4,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false)[0];

    // create t3156 tensor
    unsigned t3156_sizes[] = {128};
    unsigned t3156         = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,  // isPersistent
                                   "t3156",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,  // initializer
                                   t3156_sizes,
                                   1,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false)[0];

    // create t3155 tensor
    unsigned t3155_sizes[] = {128};
    unsigned t3155         = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,  // isPersistent
                                   "t3155",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,  // initializer
                                   t3155_sizes,
                                   1,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false)[0];
    unsigned char
        n2054_training_SGD_gradients_gradients_conv5_block16_1_bn_FusedBatchNormV3_grad_FusedBatchNormGradV3_params[] =
            {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {t3075_conv5_block16_1_conv_Conv2D_0,
         t3153_training_SGD_gradients_gradients_conv5_block16_1_relu_Relu_grad_ReluGrad_0,
         t3077_conv5_block16_1_bn_FusedBatchNormV3_hfbn_v3_1,
         t3160,
         t254_conv5_block16_1_bn_readvariableop_0},
        {t3154, t3156, t3155},
        (void*)
            n2054_training_SGD_gradients_gradients_conv5_block16_1_bn_FusedBatchNormV3_grad_FusedBatchNormGradV3_params,
        12,
        "n2054_training_SGD_gradients_gradients_conv5_block16_1_bn_FusedBatchNormV3_grad_FusedBatchNormGradV3");

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", "true");

    compareRunsResults({t3154,
                        t3156,
                        t3155,
                        t3149_training_SGD_gradients_gradients_conv5_block16_2_conv_Conv2D_grad_Conv2DBackpropFilter_0,
                        t3152_BF_TO_F_Cast_460_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, snake_pattern_with_double_buffer_ASIC_CI)
{
    // This test verifies that the results are correct in snake walking pattern with double buffer scenario:

    // Slicing Strategy - Left-to-Right , 4Wx1H, graph size optimized: true
    // Original Input [0] tensor_a : 1024x2048, Sliced : 1024x1024, Num of slices: 2, Buffers: 2,
    // inSram: true, alignedToCL:true
    // Original Input [1] tensor_b : 4096x1024, Sliced : 1024x1024, Num of slices: 4, Buffers: 2,
    // inSram: true, alignedToCL:true
    // Original Output tensor_c : 4096x2048, Sliced : 1024x1024, Num of slices: 8, Buffers: 1,
    // inSram: false, alignedToCL:false

    unsigned k = 1024, m = 2048, n = 4096;
    unsigned tensorASizes[] = {k, m};
    unsigned tensorA        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,  // isPersistent
                                     "tensor_a",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     tensorASizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false)[0];

    unsigned tensorBSizes[] = {n, k};
    unsigned tensorB        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,  // isPersistent
                                     "tensor_b",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     tensorBSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false)[0];

    unsigned      tensorCSizes[] = {n, m};
    unsigned      tensorC        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,  // isPersistent
                                     "tensor_c",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,  // initializer
                                     tensorCSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false)[0];
    unsigned char gemmParams[]   = {0, 0};
    addNodeToGraph("gemm", {tensorA, tensorB}, {tensorC}, (void*)gemmParams, 2, "gemm");

    addConfigurationToRun(FIRST_RUN, "ENABLE_SRAM_MULTI_BUFFERING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_SRAM_MULTI_BUFFERING", "true");

    compareRunsResults({tensorC});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, test_cl_align_with_reshape_and_eviction_ASIC_CI)
{
    unsigned addSize[] = {80, 32, 32, 8};
    unsigned addIn1    = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "addIn1",
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    addSize,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addSize,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned addIn2 = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "addIn2",
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    addSize,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addSize,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned  addOut = createTensors(1,
                                    OUTPUT_TENSOR,
                                    true,
                                    "addOut",
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    addSize,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addSize,
                                    synTensorType::DATA_TENSOR)[0];
    synNodeId add1Id;
    addNodeToGraph("add_fwd_f32", {addIn1, addIn2}, {addOut}, nullptr, 0, "Add1", 0 /*graphIndex*/, &add1Id);

    unsigned convWeightsSize[] = {480, 80, 1, 1};
    unsigned convWeights       = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "convWeights",
                                         MEM_INIT_RANDOM_POSITIVE,
                                         nullptr,
                                         convWeightsSize,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         convWeightsSize,
                                         synTensorType::DATA_TENSOR)[0];

    unsigned      convOutSize[] = {480, 32, 32, 8};
    unsigned      convOut       = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "convOut",
                                     MEM_INIT_RANDOM_POSITIVE,
                                     nullptr,
                                     convOutSize,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     convOutSize,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId     convId;
    unsigned char convParams[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,   0,  0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 182, 243, 1,  0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   17, 127, 0, 0};

    addNodeToGraph("spatial_convolution",
                   {addOut, convWeights},
                   {convOut},
                   (void*)convParams,
                   72,
                   "Conv",
                   0 /*graphIndex*/,
                   &convId);

    unsigned  reshape2OutSizes[] = {30720, 128, 1, 1};
    unsigned  reshape2Out        = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "reshape2Out",
                                         MEM_INIT_RANDOM_POSITIVE,
                                         nullptr,
                                         reshape2OutSizes,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         reshape2OutSizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId reshape2Id;
    addNodeToGraph("reshape", {convOut}, {reshape2Out}, nullptr, 0, "Reshape2", 0 /*graphIndex*/, &reshape2Id);

    unsigned  reluSize[] = {30720, 128, 1, 1};
    unsigned  reluOut    = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "reluOut",
                                     MEM_INIT_RANDOM_POSITIVE,
                                     nullptr,
                                     reluSize,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     reluSize,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId reluId;
    addNodeToGraph("relu_fwd_f32", {reshape2Out}, {reluOut}, nullptr, 0, "Relu", 0 /*graphIndex*/, &reluId);

    unsigned conv2Weights = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "conv2Weights",
                                          MEM_INIT_RANDOM_POSITIVE,
                                          nullptr,
                                          convWeightsSize,
                                          4,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          convWeightsSize,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned  conv2Out = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "conv2Out",
                                      MEM_INIT_RANDOM_POSITIVE,
                                      nullptr,
                                      convOutSize,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      convOutSize,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId conv2Id;
    addNodeToGraph("spatial_convolution",
                   {addOut, conv2Weights},
                   {conv2Out},
                   (void*)convParams,
                   72,
                   "Conv2",
                   0 /*graphIndex*/,
                   &conv2Id);

    setNodeDependency(&reluId, &conv2Id, 1, 1);  // Force the second convolution to be in a separate bundle

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "true");

    compareRunsResults({addOut, convOut, reluOut, conv2Out, conv2Weights, convWeights});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, check_tensors_allocation_ASIC_CI)
{
    unsigned in2Sizes[] = {1024, 33510};
    unsigned in2        = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "in2",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 in2Sizes,
                                 2,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in2Sizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned in1Sizes[] = {33510, 4096};
    unsigned in1        = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "in1",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 in1Sizes,
                                 2,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in1Sizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned outSizes[] = {1024, 4096};
    unsigned out        = createTensors(1,
                                 OUTPUT_TENSOR,
                                 false,
                                 "out",
                                 MEM_INIT_ALL_ZERO,
                                 nullptr,
                                 outSizes,
                                 2,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 outSizes,
                                 synTensorType::DATA_TENSOR)[0];

    synNodeId     gemmId;
    unsigned char params[] = {0, 0};
    addNodeToGraph("gemm", {in1, in2}, {out}, (void*)params, 2, "GEMM", 0 /*graphIndex*/, &gemmId);

    unsigned reshapeInSizes[] = {1024, 256, 16};
    unsigned reshapeIn        = createTensors(1,
                                       INPUT_TENSOR,
                                       false,
                                       "reshapeIn",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       reshapeInSizes,
                                       3,
                                       syn_type_uint32,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       reshapeInSizes,
                                       synTensorType::SHAPE_TENSOR)[0];

    unsigned reshapeOutSizes[] = {1024, 256, 16};
    unsigned reshapeOut        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "reshapeOut",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        reshapeOutSizes,
                                        3,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        reshapeOutSizes,
                                        synTensorType::DATA_TENSOR)[0];

    synNodeId reshapeId;
    addNodeToGraph("reshape", {out, reshapeIn}, {reshapeOut}, nullptr, 0, "RESHAPE", 0 /*graphIndex*/, &reshapeId);

    unsigned castOutSizes[] = {1024, 256, 16};
    unsigned castOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "castOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     castOutSizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     castOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synNodeId castId;
    addNodeToGraph("cast_bf16_to_f32", {reshapeOut}, {castOut}, nullptr, 0, "CAST", 0 /*graphIndex*/, &castId);

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "true");

    compareRunsResults({castOut});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, 2_shared_inputs_ASIC_CI)
{
    unsigned tensorAsizes[] = {2048, 2048};
    unsigned tensorA        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,  // isPersistent
                                     "tensorA",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     tensorAsizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false)[0];

    unsigned tensorBSizes[] = {1024, 2048};
    unsigned tensorB        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,  // isPersistent
                                     "tensorB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     tensorBSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false)[0];

    unsigned tensorCSizes[] = {1024, 2048};
    unsigned tensorC        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,  // isPersistent
                                     "tensorC",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,  // initializer
                                     tensorCSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false)[0];

    unsigned tensorC2 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,  // isPersistent
                                      "tensorC2",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,  // initializer
                                      tensorCSizes,
                                      2,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false)[0];

    unsigned char gemmParams[] = {0, 0};
    addNodeToGraph("gemm", {tensorA, tensorB}, {tensorC}, (void*)gemmParams, 2, "gemm");
    addNodeToGraph("gemm", {tensorA, tensorB}, {tensorC2}, (void*)gemmParams, 2, "gemm2");

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "true");

    compareRunsResults({tensorC, tensorC2});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, dropout_fwd_from_sram_and_hbm)
{
    // The dropout fwd requests scalar-pipe for its second input (seed).
    // Make sure the results are identical regardless of the tensor placement (SRAM / HBM).
    unsigned inputSizes[] = {1024, 7168};
    unsigned input        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "input",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   inputSizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   inputSizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned seedSizes[] = {1};
    unsigned seed        = createTensors(1,
                                  INPUT_TENSOR,
                                  true,
                                  "seed",
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  seedSizes,
                                  1,
                                  syn_type_int32,
                                  nullptr,
                                  0,
                                  0,
                                  nullptr,
                                  false,
                                  seedSizes,
                                  synTensorType::DATA_TENSOR)[0];

    unsigned output1Sizes[] = {1024, 7168};
    unsigned output1        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "output1",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     output1Sizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     output1Sizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned      output2Sizes[] = {1024, 7168};
    unsigned      output2        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "output2",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     output2Sizes,
                                     2,
                                     syn_type_int8,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     output2Sizes,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId     id;
    unsigned char params[] = {205, 204, 204, 61, 0, 0, 0, 0};
    addNodeToGraph("dropout_fwd_bf16",
                   {input, seed},
                   {output1, output2},
                   (void*)params,
                   8,
                   "dropout",
                   0 /*graphIndex*/,
                   &id);

    addConfigurationToRun(FIRST_RUN, "MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT", "0");
    addConfigurationToRun(SECOND_RUN, "MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT", "128");

    compareRunsResults({output1, output2});
}

TEST_F_GC(SynGaudiTwoRunCompareTest,
          tpc_bundle_with_multiple_outputs_ASIC,
          {synDeviceGaudi})  // Disabled for Gaudi2 - long time on sim
{
    unsigned castInSizes[] = {128, 64, 3, 3};
    unsigned castIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "castIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    castInSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    castInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned castOutSizes[] = {128, 64, 3, 3};
    unsigned castOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "castOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     castOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     castOutSizes,
                                     synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("cast_f32_to_bf16", {castIn}, {castOut}, nullptr, 0, "CAST");

    unsigned conv1In1Sizes[] = {64, 570, 570, 8};
    unsigned conv1In1        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "conv1In1",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      conv1In1Sizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv1In1Sizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned conv1In2Sizes[] = {64, 64, 3, 3};
    unsigned conv1In2        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "conv1In2",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      conv1In2Sizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv1In2Sizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned conv1OutSizes[] = {64, 568, 568, 8};
    unsigned conv1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "conv1Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      conv1OutSizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv1OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char conv1Params[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,   0,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 35, 183, 1,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {conv1In1, conv1In2}, {conv1Out}, (void*)conv1Params, 72, "CONV1");

    unsigned addInSizes[] = {64, 1, 1, 1};
    unsigned addIn        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "addIn",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   addInSizes,
                                   4,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   addInSizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned addOutSizes[] = {64, 568, 568, 8};
    unsigned addOut        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "addOut",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    addOutSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addOutSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_bf16", {conv1Out, addIn}, {addOut}, nullptr, 0, "ADD");

    unsigned reluOutSizes[] = {64, 568, 568, 8};
    unsigned reluOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "reluOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     reluOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     reluOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {addOut}, {reluOut}, nullptr, 0, "RELU");

    unsigned maxPool1Out1Sizes[] = {64, 284, 284, 8};
    unsigned maxPool1Out1        = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "maxPool1Out1",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          maxPool1Out1Sizes,
                                          4,
                                          syn_type_int16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          maxPool1Out1Sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned maxPool1Out2Sizes[] = {64, 284, 284, 8};
    unsigned maxPool1Out2        = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "maxPool1Out2",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          maxPool1Out2Sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          maxPool1Out2Sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned char maxPool1Params[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
                                      0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("maxpool_2d_fwd_bf16",
                   {reluOut},
                   {maxPool1Out1, maxPool1Out2},
                   (void*)maxPool1Params,
                   44,
                   "MAX_POOL1");

    unsigned maxPool2Out1Sizes[] = {64, 284, 284, 8};
    unsigned maxPool2Out1        = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "maxPool2Out1",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          maxPool2Out1Sizes,
                                          4,
                                          syn_type_int16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          maxPool2Out1Sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned maxPool2Out2Sizes[] = {64, 284, 284, 8};
    unsigned maxPool2Out2        = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "maxPool2Out2",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          maxPool2Out2Sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          maxPool2Out2Sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned char maxPool2Params[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
                                      0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("maxpool_2d_fwd_bf16",
                   {reluOut},
                   {maxPool2Out1, maxPool2Out2},
                   (void*)maxPool2Params,
                   44,
                   "MAX_POOL2");

    unsigned conv2OutSizes[] = {128, 282, 282, 8};
    unsigned conv2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "conv2Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      conv2OutSizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv2OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char conv2Params[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,   0,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 35, 183, 1,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {maxPool2Out2, castOut}, {conv2Out}, (void*)conv2Params, 72, "CONV2");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_BUNDLES", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_TPC_BUNDLES", "true");

    compareRunsResults({maxPool1Out1, maxPool1Out2, maxPool2Out1, conv2Out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, batch_gemm_with_non_symmetrical_layout)
{
    unsigned sharedInSizes[] = {1024, 128, 64};
    unsigned sharedIn        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "sharedIn",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sharedInSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      sharedInSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned gemm1InSizes[] = {1024, 1024};
    unsigned gemm1In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemm1In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemm1InSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     gemm1InSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemm1OutSizes[] = {1024, 128, 64};
    unsigned gemm1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm1Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      gemm1OutSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      gemm1OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char gemm1Params[] = {0, 1};
    addNodeToGraph("batch_gemm", {sharedIn, gemm1In}, {gemm1Out}, (void*)gemm1Params, 2, "BGEMM1");

    unsigned gemm2InSizes[] = {1024, 128, 64};
    unsigned gemm2In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemm2In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemm2InSizes,
                                     3,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     gemm2InSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemm2OutSizes[] = {1024, 1024, 64};
    unsigned gemm2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm2Out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      gemm2OutSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      gemm2OutSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned char gemm2Params[] = {1, 0};
    addNodeToGraph("batch_gemm", {gemm2In, sharedIn}, {gemm2Out}, (void*)gemm2Params, 2, "BGEMM2");

    addConfigurationToRun(FIRST_RUN, "SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE", "true");

    compareRunsResults({gemm1Out, gemm2Out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, optimize_mme_node_as_bgemm_should_bundle_memset_ASIC_CI)
{
    unsigned bnIn1Sizes[] = {64, 56, 56, 256};
    unsigned bnIn1        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "bnIn1",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   bnIn1Sizes,
                                   4,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   bnIn1Sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned bnIn2Sizes[] = {64, 56, 56, 256};
    unsigned bnIn2        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "bnIn2",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   bnIn2Sizes,
                                   4,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   bnIn2Sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned bnIn3Sizes[] = {64};
    unsigned bnIn3        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "bnIn3",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   bnIn3Sizes,
                                   1,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   bnIn3Sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned bnIn4Sizes[] = {64};
    unsigned bnIn4        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "bnIn4",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   bnIn4Sizes,
                                   1,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   bnIn4Sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned bnIn5Sizes[] = {64};
    unsigned bnIn5        = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "bnIn5",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   bnIn5Sizes,
                                   1,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   bnIn5Sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned bnOut1Sizes[] = {64, 56, 56, 256};
    unsigned bnOut1        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "bnOut1",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    bnOut1Sizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    bnOut1Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned bnOut2Sizes[] = {64};
    unsigned bnOut2        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    true,
                                    "bnOut2",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    bnOut2Sizes,
                                    1,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    bnOut2Sizes,
                                    synTensorType::DATA_TENSOR)[0];
    unsigned bnOut3Sizes[] = {64};
    unsigned bnOut3        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    true,
                                    "bnOut3",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    bnOut3Sizes,
                                    1,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    bnOut3Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned char bnParams[] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph("batch_norm_bwd_bf16",
                   {bnIn1, bnIn2, bnIn3, bnIn4, bnIn5},
                   {bnOut1, bnOut2, bnOut3},
                   (void*)bnParams,
                   16,
                   "BN");

    unsigned dedwInSizes[] = {64, 56, 56, 256};
    unsigned dedwIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "dedwIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dedwInSizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedwInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned dedwOutSizes[] = {64, 64, 1, 1};
    unsigned dedwOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "dedwOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedwOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedwOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synConvolutionParams dedwParams(1, 1, 1, 1, 0, 0, 0, 0, 1, 1);
    addNodeToGraph("dedw", {bnOut1, dedwIn}, {dedwOut}, &dedwParams, sizeof(dedwParams), "DEDW");

    unsigned dedxIn1Sizes[] = {64, 64, 1, 1};
    unsigned dedxIn1        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "dedxIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     dedxIn1Sizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedxIn1Sizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned dedxIn2Sizes[] = {64, 56, 56, 256};
    unsigned dedxIn2        = createTensors(1,
                                     INPUT_TENSOR,
                                     false,
                                     "dedxIn2",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedxIn2Sizes,
                                     4,
                                     syn_type_uint32,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedxIn2Sizes,
                                     synTensorType::SHAPE_TENSOR)[0];

    unsigned dedxOutSizes[] = {64, 56, 56, 256};
    unsigned dedxOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "dedxOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     dedxOutSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedxOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synConvolutionParams dedxParams(1, 1, 1, 1, 0, 0, 0, 0, 1, 1);
    addNodeToGraph("dedx", {bnOut1, dedxIn1, dedxIn2}, {dedxOut}, &dedxParams, sizeof(dedxParams), "DEDX");

    compareRunsResults({dedwOut, dedxOut});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, test_cl_align_with_two_reshapes_and_eviction_ASIC_CI)
{
    unsigned addSizes[] = {163840, 4, 1, 1};
    unsigned addIn1     = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "addIn1",
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    addSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned addIn2 = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "addIn2",
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    addSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addSizes,
                                    synTensorType::DATA_TENSOR)[0];
    unsigned addOut = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "addOut",
                                    MEM_INIT_RANDOM_POSITIVE,
                                    nullptr,
                                    addSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addSizes,
                                    synTensorType::DATA_TENSOR)[0];

    synNodeId add1Id;
    addNodeToGraph("add_fwd_f32", {addIn1, addIn2}, {addOut}, nullptr, 0, "Add1", 0 /*graphIndex*/, &add1Id);

    unsigned firstReshapeSizes[] = {80, 32, 32, 8};
    unsigned reshapeOutput       = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "Reshape1Output",
                                           MEM_INIT_RANDOM_POSITIVE,
                                           nullptr,
                                           firstReshapeSizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           firstReshapeSizes,
                                           synTensorType::DATA_TENSOR)[0];

    synNodeId firatReshapeId;
    addNodeToGraph("reshape", {addOut}, {reshapeOutput}, nullptr, 0, "Reshape1", 0 /*graphIndex*/, &firatReshapeId);

    unsigned convWeightsSize[] = {480, 80, 1, 1};
    unsigned convWeights       = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "convWeights",
                                         MEM_INIT_RANDOM_POSITIVE,
                                         nullptr,
                                         convWeightsSize,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         convWeightsSize,
                                         synTensorType::DATA_TENSOR)[0];

    unsigned      convOutSize[] = {480, 32, 32, 8};
    unsigned      convOut       = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "convOut",
                                     MEM_INIT_RANDOM_POSITIVE,
                                     nullptr,
                                     convOutSize,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     convOutSize,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId     convId;
    unsigned char convParams[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,   0,  0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 182, 243, 1,  0,   0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   17, 127, 0, 0};

    addNodeToGraph("spatial_convolution",
                   {reshapeOutput, convWeights},
                   {convOut},
                   (void*)convParams,
                   72,
                   "Conv",
                   0 /*graphIndex*/,
                   &convId);

    unsigned  reshape2OutSizes[] = {30720, 128, 1, 1};
    unsigned  reshape2Out        = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "reshape2Out",
                                         MEM_INIT_RANDOM_POSITIVE,
                                         nullptr,
                                         reshape2OutSizes,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         reshape2OutSizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId reshape2Id;
    addNodeToGraph("reshape", {convOut}, {reshape2Out}, nullptr, 0, "Reshape2", 0 /*graphIndex*/, &reshape2Id);

    unsigned  reluSize[] = {30720, 128, 1, 1};
    unsigned  reluOut    = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "reluOut",
                                     MEM_INIT_RANDOM_POSITIVE,
                                     nullptr,
                                     reluSize,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     reluSize,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId reluId;
    addNodeToGraph("relu_fwd_f32", {reshape2Out}, {reluOut}, nullptr, 0, "Relu", 0 /*graphIndex*/, &reluId);

    unsigned conv2Weights = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "conv2Weights",
                                          MEM_INIT_RANDOM_POSITIVE,
                                          nullptr,
                                          convWeightsSize,
                                          4,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          convWeightsSize,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned  conv2Out = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "conv2Out",
                                      MEM_INIT_RANDOM_POSITIVE,
                                      nullptr,
                                      convOutSize,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      convOutSize,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId conv2Id;
    addNodeToGraph("spatial_convolution",
                   {reshapeOutput, conv2Weights},
                   {conv2Out},
                   (void*)convParams,
                   72,
                   "Conv2",
                   0 /*graphIndex*/,
                   &conv2Id);

    setNodeDependency(&reluId, &conv2Id, 1, 1);  // Force the second convolution to be in a separate bundle

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_COST_MODEL_ENABLED", "true");

    compareRunsResults({reshapeOutput, convOut, reluOut, conv2Out, conv2Weights, convWeights});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, conv_bn_with_packing_ASIC_CI)
{
    unsigned g_0_layer1_0_conv1_output_max_sizes[] = {64, 56, 56, 256};
    unsigned g_0_layer1_0_conv1_output_min_sizes[] = {64, 56, 56, 256};
    unsigned g_0_layer1_0_conv1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer1_0_conv1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer1_0_conv1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_0_conv1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_bias_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_bias_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_bias             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_layer1_0_bn1_bias",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   g_0_layer1_0_bn1_bias_max_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_layer1_0_bn1_bias_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_weight_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_weight_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer1_0_bn1_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer1_0_bn1_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_0_bn1_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_0_bn1_running_mean tensor
    unsigned g_0_layer1_0_bn1_running_mean_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_mean_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_mean             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_0_bn1_running_mean",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer1_0_bn1_running_mean_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_0_bn1_running_mean_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_0_bn1_running_var tensor
    unsigned g_0_layer1_0_bn1_running_var_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_var_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_var             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_layer1_0_bn1_running_var",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer1_0_bn1_running_var_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer1_0_bn1_running_var_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_layer1_0_bn1_output tensor
    unsigned g_0_layer1_0_bn1_output_max_sizes[] = {64, 56, 56, 256};
    unsigned g_0_layer1_0_bn1_output_min_sizes[] = {64, 56, 56, 256};
    unsigned g_0_layer1_0_bn1_output             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     false,
                                                     "g_0_layer1_0_bn1_output",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_layer1_0_bn1_output_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_0_bn1_output_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_saved_mean_max_sizes[] = {64, 1, 1, 1};
    unsigned g_0_layer1_0_bn1_saved_mean_min_sizes[] = {64, 1, 1, 1};
    unsigned g_0_layer1_0_bn1_saved_mean             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer1_0_bn1_saved_mean",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_layer1_0_bn1_saved_mean_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer1_0_bn1_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_layer1_0_bn1_saved_var_max_sizes[] = {64, 1, 1, 1};
    unsigned      g_0_layer1_0_bn1_saved_var_min_sizes[] = {64, 1, 1, 1};
    unsigned      g_0_layer1_0_bn1_saved_var             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer1_0_bn1_saved_var",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_layer1_0_bn1_saved_var_max_sizes,
                                                        4,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer1_0_bn1_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_0_bn1_0_id;
    unsigned char g_0_layer1_0_bn1_0_params[] = {172, 197, 39, 55, 205, 204, 204, 61, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_layer1_0_conv1_output,
                    g_0_layer1_0_bn1_bias,
                    g_0_layer1_0_bn1_weight,
                    g_0_layer1_0_bn1_running_mean,
                    g_0_layer1_0_bn1_running_var},
                   {g_0_layer1_0_bn1_output, g_0_layer1_0_bn1_saved_mean, g_0_layer1_0_bn1_saved_var},
                   (void*)g_0_layer1_0_bn1_0_params,
                   12,
                   "g_0_layer1_0_bn1_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_bn1_0_id);

    unsigned  g_0_layer1_0_relu1_output_max_sizes[] = {64, 56, 56, 256};
    unsigned  g_0_layer1_0_relu1_output_min_sizes[] = {64, 56, 56, 256};
    unsigned  g_0_layer1_0_relu1_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_layer1_0_relu1_output",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_0_relu1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_0_relu1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer1_0_relu1_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_layer1_0_bn1_output},
                   {g_0_layer1_0_relu1_output},
                   nullptr,
                   0,
                   "g_0_layer1_0_relu1_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_relu1_0_id);

    unsigned g_0_layer1_0_conv2_weight_max_sizes[] = {64, 64, 3, 3};
    unsigned g_0_layer1_0_conv2_weight_min_sizes[] = {64, 64, 3, 3};
    unsigned g_0_layer1_0_conv2_weight             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer1_0_conv2_weight",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer1_0_conv2_weight_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_0_conv2_weight_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_layer1_0_conv2_output_max_sizes[] = {64, 56, 56, 256};
    unsigned      g_0_layer1_0_conv2_output_min_sizes[] = {64, 56, 56, 256};
    unsigned      g_0_layer1_0_conv2_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_layer1_0_conv2_output",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_0_conv2_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_0_conv2_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_0_conv2_0_id;
    unsigned char g_0_layer1_0_conv2_0_params[] = {3, 0, 0, 0, 3, 0, 0,  0,   1, 0, 0, 0, 1, 0, 0,   0,   1, 0,
                                                   0, 0, 1, 0, 0, 0, 1,  0,   0, 0, 1, 0, 0, 0, 1,   0,   0, 0,
                                                   1, 0, 0, 0, 0, 0, 68, 131, 1, 0, 0, 0, 0, 0, 0,   0,   0, 0,
                                                   0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 1, 0, 0, 0, 254, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_layer1_0_relu1_output, g_0_layer1_0_conv2_weight},
                   {g_0_layer1_0_conv2_output},
                   (void*)g_0_layer1_0_conv2_0_params,
                   72,
                   "g_0_layer1_0_conv2_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_conv2_0_id);

    unsigned g_0_layer1_0_bn2_bias_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_bias_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_bias             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_layer1_0_bn2_bias",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   g_0_layer1_0_bn2_bias_max_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_layer1_0_bn2_bias_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn2_weight_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_weight_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer1_0_bn2_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer1_0_bn2_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_0_bn2_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn2_running_mean_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_running_mean_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_running_mean             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_0_bn2_running_mean",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer1_0_bn2_running_mean_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_0_bn2_running_mean_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn2_running_var_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_running_var_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn2_running_var             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_layer1_0_bn2_running_var",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer1_0_bn2_running_var_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer1_0_bn2_running_var_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn2_output_max_sizes[] = {64, 56, 56, 256};
    unsigned g_0_layer1_0_bn2_output_min_sizes[] = {64, 56, 56, 256};
    unsigned g_0_layer1_0_bn2_output             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     false,
                                                     "g_0_layer1_0_bn2_output",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_layer1_0_bn2_output_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_0_bn2_output_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn2_saved_mean_max_sizes[] = {64, 1, 1, 1};
    unsigned g_0_layer1_0_bn2_saved_mean_min_sizes[] = {64, 1, 1, 1};
    unsigned g_0_layer1_0_bn2_saved_mean             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer1_0_bn2_saved_mean",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_layer1_0_bn2_saved_mean_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer1_0_bn2_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_layer1_0_bn2_saved_var_max_sizes[] = {64, 1, 1, 1};
    unsigned      g_0_layer1_0_bn2_saved_var_min_sizes[] = {64, 1, 1, 1};
    unsigned      g_0_layer1_0_bn2_saved_var             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer1_0_bn2_saved_var",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_layer1_0_bn2_saved_var_max_sizes,
                                                        4,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer1_0_bn2_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_0_bn2_0_id;
    unsigned char g_0_layer1_0_bn2_0_params[] = {172, 197, 39, 55, 205, 204, 204, 61, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_layer1_0_conv2_output,
                    g_0_layer1_0_bn2_bias,
                    g_0_layer1_0_bn2_weight,
                    g_0_layer1_0_bn2_running_mean,
                    g_0_layer1_0_bn2_running_var},
                   {g_0_layer1_0_bn2_output, g_0_layer1_0_bn2_saved_mean, g_0_layer1_0_bn2_saved_var},
                   (void*)g_0_layer1_0_bn2_0_params,
                   12,
                   "g_0_layer1_0_bn2_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_bn2_0_id);

    unsigned  g_0_layer1_0_relu2_output_max_sizes[] = {64, 56, 56, 256};
    unsigned  g_0_layer1_0_relu2_output_min_sizes[] = {64, 56, 56, 256};
    unsigned  g_0_layer1_0_relu2_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       true,
                                                       "g_0_layer1_0_relu2_output",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_0_relu2_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_0_relu2_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer1_0_relu2_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_layer1_0_bn2_output},
                   {g_0_layer1_0_relu2_output},
                   nullptr,
                   0,
                   "g_0_layer1_0_relu2_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_relu2_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_layer1_0_relu2_output, g_0_layer1_0_bn2_saved_mean, g_0_layer1_0_bn2_saved_var});
}
TEST_F_GC(SynGaudiTwoRunCompareTest, shared_input_consumer_and_producer_ASIC_CI)
{
    // Graph #0

    /*************
     * g_0_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1396_0 node
     * inputs:
     *     g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0[64, 56, 56, 256] (dtype=bf16)
     *     g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0[64, 56, 56, 256] (dtype=bf16)
     *     g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3[64] (dtype=float32)
     *     g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3[64] (dtype=float32)
     *     g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0[64] (dtype=float32)
     * outputs:
     *     g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0[64, 56, 56, 256]
     *(dtype=bf16) g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2[64]
     *(dtype=float32) g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1[64]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0 tensor
    unsigned g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0_max_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0_min_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0 tensor
    unsigned g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0_max_sizes[] = {64,
                                                                                                        56,
                                                                                                        56,
                                                                                                        256};
    unsigned g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0_min_sizes[] = {64,
                                                                                                        56,
                                                                                                        56,
                                                                                                        256};
    unsigned g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3 tensor
    unsigned g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3_max_sizes[] = {64};
    unsigned g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3_min_sizes[] = {64};
    unsigned g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3 tensor
    unsigned g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_max_sizes[] = {64};
    unsigned g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_min_sizes[] = {64};
    unsigned g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0 tensor
    unsigned g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0 tensor
    unsigned g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0_max_sizes[] =
        {64, 56, 56, 256};
    unsigned g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0_min_sizes[] =
        {64, 56, 56, 256};
    unsigned g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0 = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    // create g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2 tensor
    unsigned g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2_max_sizes[] = {
        64};
    unsigned g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2_min_sizes[] = {
        64};
    unsigned g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    // create g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1 tensor
    unsigned g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1_max_sizes[] = {
        64};
    unsigned g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1_min_sizes[] = {
        64};
    unsigned g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1396_0_id;
    unsigned char
        g_0_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1396_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t1004_while_body_1_while_resnet50_res2b_branch2a_Conv2D_0,
         g_0_t3014_while_body_1_gradient_tape_while_resnet50_activation_4_ReluGrad_0,
         g_0_t1013_while_body_1_while_resnet50_bn2b_branch2a_FusedBatchNormV3_3,
         g_0_t3021_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3,
         g_0_t589_while_body_1_while_resnet50_bn2b_branch2a_readvariableop_0},
        {g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0,
         g_0_t3017_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_2,
         g_0_t3016_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_1},
        (void*)
            g_0_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1396_0_params,
        16,
        "g_0_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1396_0",
        0 /*graphIndex*/,
        &g_0_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1396_0_id);

    /*************
     * g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_dedx_n1397_0 node
     * inputs:
     *     g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0[64, 56, 56, 256]
     *(dtype=bf16) g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0[64, 256, 1, 1] (dtype=bf16)
     *     g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput[256, 56, 56,
     *256] (dtype=uint32) (shape tensor) outputs:
     *     g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0[256, 56, 56,
     *256] (dtype=bf16) ctrl inputs:
     *     g_0_while_body_1_while_resnet50_activation_3_Relu_relu_fwd_bf16_n327_control_edge_2204[] (dtype=invalid)
     * ctrl outputs:
     *************/

    // create g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0 tensor
    unsigned g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0_max_sizes[] = {64, 256, 1, 1};
    unsigned g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0_min_sizes[] = {64, 256, 1, 1};
    unsigned g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput tensor
    unsigned g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_max_sizes[] =
        {256, 56, 56, 256};
    unsigned g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_min_sizes[] =
        {256, 56, 56, 256};
    unsigned g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput =
        createTensors(
            1,
            INPUT_TENSOR,
            false,
            "g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_max_sizes,
            4,
            syn_type_uint32,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_min_sizes,
            synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0 tensor
    unsigned
        g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0_max_sizes[] =
            {256, 56, 56, 256};
    unsigned
        g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0_min_sizes[] =
            {256, 56, 56, 256};
    unsigned g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_dedx_n1397_0_id;
    unsigned char
        g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_dedx_n1397_0_params[] =
            {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,   0,   0,   0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 15, 207, 1,   0,   0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   254, 127, 0, 0};
    addNodeToGraph(
        "dedx",
        {g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0,
         g_0_t831_while_body_1_while_resnet50_res2b_branch2a_Conv2D_Cast_0,
         g_0_t3025_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput},
        {g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0},
        (void*)
            g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_dedx_n1397_0_params,
        72,
        "g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_dedx_n1397_0",
        0 /*graphIndex*/,
        &g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_dedx_n1397_0_id);

    /*************
     * g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_dedw_n1398_0 node
     * inputs:
     *     g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0[64, 56, 56, 256]
     *(dtype=bf16) g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0[256, 56, 56, 256] (dtype=bf16) outputs:
     *     g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0[64, 256, 1,
     *1] (dtype=bf16) ctrl inputs:
     *     g_0_while_body_1_while_resnet50_activation_3_Relu_relu_fwd_bf16_n327_control_edge_2204[] (dtype=invalid)
     * ctrl outputs:
     *************/

    // create g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0 tensor
    unsigned g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0_max_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0_min_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0 tensor
    unsigned
        g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0_max_sizes[] =
            {64, 256, 1, 1};
    unsigned
        g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0_min_sizes[] =
            {64, 256, 1, 1};
    unsigned g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_dedw_n1398_0_id;
    unsigned char
        g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_dedw_n1398_0_params[] =
            {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,   0,   0,   0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 15, 207, 1,   0,   0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   254, 127, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_t3015_while_body_1_gradient_tape_while_resnet50_bn2b_branch2a_FusedBatchNormGradV3_0,
         g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0},
        {g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0},
        (void*)
            g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_dedw_n1398_0_params,
        72,
        "g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_dedw_n1398_0",
        0 /*graphIndex*/,
        &g_0_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_dedw_n1398_0_id);

    /*************
     * g_0_while_body_1_while_AddN_14_add_fwd_bf16_n1413_0 node
     * inputs:
     *     g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0[256, 56, 56, 256] (dtype=bf16)
     *     g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0[256, 56, 56,
     *256] (dtype=bf16) outputs: g_0_t3053_while_body_1_while_AddN_14_0[256, 56, 56, 256] (dtype=bf16) ctrl inputs: ctrl
     *outputs:
     *************/

    // create g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0 tensor
    unsigned g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0_max_sizes[] = {256,
                                                                                                        56,
                                                                                                        56,
                                                                                                        256};
    unsigned g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0_min_sizes[] = {256,
                                                                                                        56,
                                                                                                        56,
                                                                                                        256};
    unsigned g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3053_while_body_1_while_AddN_14_0 tensor
    unsigned  g_0_t3053_while_body_1_while_AddN_14_0_max_sizes[] = {256, 56, 56, 256};
    unsigned  g_0_t3053_while_body_1_while_AddN_14_0_min_sizes[] = {256, 56, 56, 256};
    unsigned  g_0_t3053_while_body_1_while_AddN_14_0             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "g_0_t3053_while_body_1_while_AddN_14_0",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_t3053_while_body_1_while_AddN_14_0_max_sizes,
                                                                    4,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_t3053_while_body_1_while_AddN_14_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body_1_while_AddN_14_add_fwd_bf16_n1413_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_t2990_while_body_1_gradient_tape_while_resnet50_activation_6_ReluGrad_0,
                    g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0},
                   {g_0_t3053_while_body_1_while_AddN_14_0},
                   nullptr,
                   0,
                   "g_0_while_body_1_while_AddN_14_add_fwd_bf16_n1413_0",
                   0 /*graphIndex*/,
                   &g_0_while_body_1_while_AddN_14_add_fwd_bf16_n1413_0_id);

    /*************
     * g_0_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_relu_bwd_bf16_n1414_0 node
     * inputs:
     *     g_0_t3053_while_body_1_while_AddN_14_0[256, 56, 56, 256] (dtype=bf16)
     *     g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0[256, 56, 56, 256] (dtype=bf16)
     * outputs:
     *     g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0[256, 56, 56, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0 tensor
    unsigned g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0_max_sizes[] = {256,
                                                                                                        56,
                                                                                                        56,
                                                                                                        256};
    unsigned g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0_min_sizes[] = {256,
                                                                                                        56,
                                                                                                        56,
                                                                                                        256};
    unsigned g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_relu_bwd_bf16_n1414_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_t3053_while_body_1_while_AddN_14_0, g_0_t1003_while_body_1_while_resnet50_activation_3_Relu_0},
                   {g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0},
                   nullptr,
                   0,
                   "g_0_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_relu_bwd_bf16_n1414_0",
                   0 /*graphIndex*/,
                   &g_0_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_relu_bwd_bf16_n1414_0_id);

    /*************
     * g_0_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1419_0 node
     * inputs:
     *     g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0[256, 56, 56, 256] (dtype=bf16)
     *     g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0[256, 56, 56, 256] (dtype=bf16)
     *     g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3[256] (dtype=float32)
     *     g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3[256] (dtype=float32)
     *     g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0[256] (dtype=float32)
     * outputs:
     *     g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0[256, 56, 56, 256]
     *(dtype=bf16) g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2[256]
     *(dtype=float32) g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1[256]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0 tensor
    unsigned g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_max_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_min_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3 tensor
    unsigned g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_max_sizes[] = {256};
    unsigned g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_min_sizes[] = {256};
    unsigned g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3 tensor
    unsigned g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_max_sizes[] = {256};
    unsigned g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_min_sizes[] = {256};
    unsigned g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0 tensor
    unsigned g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0 tensor
    unsigned g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0_max_sizes[] =
        {256, 56, 56, 256};
    unsigned g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0_min_sizes[] =
        {256, 56, 56, 256};
    unsigned g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    // create g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2 tensor
    unsigned g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2_max_sizes[] = {
        256};
    unsigned g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2_min_sizes[] = {
        256};
    unsigned g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    // create g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1 tensor
    unsigned g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1_max_sizes[] = {
        256};
    unsigned g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1_min_sizes[] = {
        256};
    unsigned g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1419_0_id;
    unsigned char
        g_0_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1419_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0,
         g_0_t3054_while_body_1_gradient_tape_while_resnet50_activation_3_ReluGrad_0,
         g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3,
         g_0_t3061_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3,
         g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0},
        {g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0,
         g_0_t3057_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_2,
         g_0_t3056_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_1},
        (void*)
            g_0_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1419_0_params,
        16,
        "g_0_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1419_0",
        0 /*graphIndex*/,
        &g_0_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1419_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({
        g_0_t3055_while_body_1_gradient_tape_while_resnet50_bn2a_branch2c_FusedBatchNormGradV3_0,         // consumer
        g_0_t3024_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropInput_0,  // dedx
        g_0_t3026_while_body_1_gradient_tape_while_resnet50_res2b_branch2a_Conv2D_Conv2DBackpropFilter_0  // dedw
    });
}

TEST_F_GC(SynGaudiTwoRunCompareTest, conv_bn_no_packing_ASIC_CI)
{
    GlobalConfTestSetter disablePacking("ENABLE_CONV_PACKING_TRAINING", "false");

    unsigned g_0_t534_while_body_1_while_resnet50_res2a_branch2c_conv2d_readvariableop_0_max_sizes[] = {256, 64, 1, 1};
    unsigned g_0_t534_while_body_1_while_resnet50_res2a_branch2c_conv2d_readvariableop_0_min_sizes[] = {256, 64, 1, 1};
    unsigned g_0_t534_while_body_1_while_resnet50_res2a_branch2c_conv2d_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t534_while_body_1_while_resnet50_res2a_branch2c_conv2d_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t534_while_body_1_while_resnet50_res2a_branch2c_conv2d_readvariableop_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t534_while_body_1_while_resnet50_res2a_branch2c_conv2d_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0_max_sizes[] = {256, 64, 1, 1};
    unsigned g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0_min_sizes[] = {256, 64, 1, 1};
    unsigned g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_cast_f32_to_bf16_n243_0_id;
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_t534_while_body_1_while_resnet50_res2a_branch2c_conv2d_readvariableop_0},
                   {g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0},
                   nullptr,
                   0,
                   "g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_cast_f32_to_bf16_n243_0",
                   0 /*graphIndex*/,
                   &g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_cast_f32_to_bf16_n243_0_id);

    unsigned g_0_t961_while_body_1_while_resnet50_res2a_branch2b_Conv2D_0_max_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t961_while_body_1_while_resnet50_res2a_branch2b_Conv2D_0_min_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t961_while_body_1_while_resnet50_res2a_branch2b_Conv2D_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t961_while_body_1_while_resnet50_res2a_branch2b_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t961_while_body_1_while_resnet50_res2a_branch2b_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t961_while_body_1_while_resnet50_res2a_branch2b_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t582_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t582_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t582_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t582_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t582_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t582_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t581_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t581_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t581_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t581_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t581_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t581_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t583_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t583_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t583_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t583_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t583_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t583_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t584_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t584_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t584_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t584_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t584_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t584_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_max_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_min_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t970_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_max_sizes[] = {64};
    unsigned g_0_t970_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_min_sizes[] = {64};
    unsigned g_0_t970_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t970_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t970_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t970_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t972_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t972_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t972_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t972_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t972_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t972_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t963_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_max_sizes[] = {64};
    unsigned g_0_t963_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_min_sizes[] = {64};
    unsigned g_0_t963_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t963_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t963_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t963_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t969_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t969_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t969_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t969_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t969_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t969_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    synNodeId     g_0_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_id;
    unsigned char g_0_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_fwd_bf16",
        {g_0_t961_while_body_1_while_resnet50_res2a_branch2b_Conv2D_0,
         g_0_t582_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_1_0,
         g_0_t581_while_body_1_while_resnet50_bn2a_branch2b_readvariableop_0,
         g_0_t583_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0,
         g_0_t584_while_body_1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0},
        {g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0,
         g_0_t970_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3,
         g_0_t972_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3,
         g_0_t963_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1,
         g_0_t969_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3},
        (void*)g_0_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_params,
        16,
        "g_0_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0",
        0 /*graphIndex*/,
        &g_0_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_id);

    unsigned g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0_max_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0_min_sizes[] = {64, 56, 56, 256};
    unsigned g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body_1_while_resnet50_activation_2_Relu_relu_fwd_bf16_n316_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_t962_while_body_1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0},
                   {g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0},
                   nullptr,
                   0,
                   "g_0_while_body_1_while_resnet50_activation_2_Relu_relu_fwd_bf16_n316_0",
                   0 /*graphIndex*/,
                   &g_0_while_body_1_while_resnet50_activation_2_Relu_relu_fwd_bf16_n316_0_id);

    unsigned g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_max_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_min_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_spatial_convolution_n317_0_id;
    unsigned char g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_spatial_convolution_n317_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 44, 40, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  254, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t981_while_body_1_while_resnet50_activation_2_Relu_0,
                    g_0_t829_while_body_1_while_resnet50_res2a_branch2c_Conv2D_Cast_0},
                   {g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0},
                   (void*)g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_spatial_convolution_n317_0_params,
                   72,
                   "g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_spatial_convolution_n317_0",
                   0 /*graphIndex*/,
                   &g_0_while_body_1_while_resnet50_res2a_branch2c_Conv2D_spatial_convolution_n317_0_id);

    unsigned g_0_t586_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_1_0_max_sizes[] = {256};
    unsigned g_0_t586_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_1_0_min_sizes[] = {256};
    unsigned g_0_t586_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t586_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t586_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t586_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t587_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t587_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t587_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t587_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t587_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t587_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t588_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {256};
    unsigned g_0_t588_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {256};
    unsigned g_0_t588_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t588_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t588_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t588_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0_max_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0_min_sizes[] = {256, 56, 56, 256};
    unsigned g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_max_sizes[] = {256};
    unsigned g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_min_sizes[] = {256};
    unsigned g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_max_sizes[] = {256};
    unsigned g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_min_sizes[] = {256};
    unsigned g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1_max_sizes[] = {256};
    unsigned g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1_min_sizes[] = {256};
    unsigned g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_max_sizes[] = {256};
    unsigned g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_min_sizes[] = {256};
    unsigned g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    synNodeId     g_0_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n318_0_id;
    unsigned char g_0_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n318_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_fwd_bf16",
        {g_0_t982_while_body_1_while_resnet50_res2a_branch2c_Conv2D_0,
         g_0_t586_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_1_0,
         g_0_t585_while_body_1_while_resnet50_bn2a_branch2c_readvariableop_0,
         g_0_t587_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_0,
         g_0_t588_while_body_1_while_resnet50_bn2a_branch2c_fusedbatchnormv3_readvariableop_1_0},
        {g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0,
         g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3,
         g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3,
         g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1,
         g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3},
        (void*)g_0_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n318_0_params,
        16,
        "g_0_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n318_0",
        0 /*graphIndex*/,
        &g_0_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n318_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_t983_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_0,
                        g_0_t991_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_3,
                        g_0_t993_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3,
                        g_0_t984_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3_1,
                        g_0_t990_while_body_1_while_resnet50_bn2a_branch2c_FusedBatchNormV3});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, maxpool_conv_bn_ASIC_CI)
{
    unsigned g_0_worker_0_relu_output_max_sizes[] = {64, 112, 112, 64};
    unsigned g_0_worker_0_relu_output_min_sizes[] = {64, 112, 112, 64};
    unsigned g_0_worker_0_relu_output             = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_worker_0_relu_output",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_worker_0_relu_output_max_sizes,
                                                      4,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_worker_0_relu_output_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_worker_0_maxpoolmax_indices_max_sizes[] = {64, 56, 56, 64};
    unsigned g_0_worker_0_maxpoolmax_indices_min_sizes[] = {64, 56, 56, 64};
    unsigned g_0_worker_0_maxpoolmax_indices             = createTensors(1,
                                                             OUTPUT_TENSOR,
                                                             true,
                                                             "g_0_worker_0_maxpoolmax_indices",
                                                             MEM_INIT_ALL_ZERO,
                                                             nullptr,
                                                             g_0_worker_0_maxpoolmax_indices_max_sizes,
                                                             4,
                                                             syn_type_int16,
                                                             nullptr,
                                                             0,
                                                             0,
                                                             nullptr,
                                                             false,
                                                             g_0_worker_0_maxpoolmax_indices_min_sizes,
                                                             synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_worker_0_maxpool_output_max_sizes[] = {64, 56, 56, 64};
    unsigned      g_0_worker_0_maxpool_output_min_sizes[] = {64, 56, 56, 64};
    unsigned      g_0_worker_0_maxpool_output             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_worker_0_maxpool_output",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_worker_0_maxpool_output_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_worker_0_maxpool_output_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_worker_0_maxpool_0_id;
    unsigned char g_0_worker_0_maxpool_0_params[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                                                     0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("maxpool_2d_fwd_bf16",
                   {g_0_worker_0_relu_output},
                   {g_0_worker_0_maxpoolmax_indices, g_0_worker_0_maxpool_output},
                   (void*)g_0_worker_0_maxpool_0_params,
                   44,
                   "g_0_worker_0_maxpool_0",
                   0 /*graphIndex*/,
                   &g_0_worker_0_maxpool_0_id);

    unsigned g_0_layer1_0_conv1_weight_max_sizes[] = {64, 64, 1, 1};
    unsigned g_0_layer1_0_conv1_weight_min_sizes[] = {64, 64, 1, 1};
    unsigned g_0_layer1_0_conv1_weight             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer1_0_conv1_weight",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer1_0_conv1_weight_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_0_conv1_weight_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_layer1_0_conv1_output_max_sizes[] = {64, 56, 56, 64};
    unsigned      g_0_layer1_0_conv1_output_min_sizes[] = {64, 56, 56, 64};
    unsigned      g_0_layer1_0_conv1_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_layer1_0_conv1_output",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_layer1_0_conv1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer1_0_conv1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_0_conv1_0_id;
    unsigned char g_0_layer1_0_conv1_0_params[] = {1, 0, 0, 0, 1, 0, 0,  0, 1, 0, 0, 0, 1, 0, 0,   0,   0, 0,
                                                   0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 1,   0,   0, 0,
                                                   1, 0, 0, 0, 0, 0, 51, 7, 1, 0, 0, 0, 0, 0, 0,   0,   0, 0,
                                                   0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0, 252, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_worker_0_maxpool_output, g_0_layer1_0_conv1_weight},
                   {g_0_layer1_0_conv1_output},
                   (void*)g_0_layer1_0_conv1_0_params,
                   72,
                   "g_0_layer1_0_conv1_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_conv1_0_id);

    unsigned g_0_layer1_0_bn1_bias_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_bias_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_bias             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_layer1_0_bn1_bias",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   g_0_layer1_0_bn1_bias_max_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_layer1_0_bn1_bias_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_weight_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_weight_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer1_0_bn1_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer1_0_bn1_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_0_bn1_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_running_mean_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_mean_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_mean             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_0_bn1_running_mean",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer1_0_bn1_running_mean_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_0_bn1_running_mean_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_running_var_max_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_var_min_sizes[] = {64};
    unsigned g_0_layer1_0_bn1_running_var             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_layer1_0_bn1_running_var",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer1_0_bn1_running_var_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer1_0_bn1_running_var_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_output_max_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_0_bn1_output_min_sizes[] = {64, 56, 56, 64};
    unsigned g_0_layer1_0_bn1_output             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     true,
                                                     "g_0_layer1_0_bn1_output",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_layer1_0_bn1_output_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer1_0_bn1_output_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer1_0_bn1_saved_mean_max_sizes[] = {64, 1, 1, 1};
    unsigned g_0_layer1_0_bn1_saved_mean_min_sizes[] = {64, 1, 1, 1};
    unsigned g_0_layer1_0_bn1_saved_mean             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer1_0_bn1_saved_mean",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_layer1_0_bn1_saved_mean_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer1_0_bn1_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_layer1_0_bn1_saved_var_max_sizes[] = {64, 1, 1, 1};
    unsigned      g_0_layer1_0_bn1_saved_var_min_sizes[] = {64, 1, 1, 1};
    unsigned      g_0_layer1_0_bn1_saved_var             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer1_0_bn1_saved_var",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_layer1_0_bn1_saved_var_max_sizes,
                                                        4,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer1_0_bn1_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer1_0_bn1_0_id;
    unsigned char g_0_layer1_0_bn1_0_params[] = {172, 197, 39, 55, 205, 204, 204, 61, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_layer1_0_conv1_output,
                    g_0_layer1_0_bn1_bias,
                    g_0_layer1_0_bn1_weight,
                    g_0_layer1_0_bn1_running_mean,
                    g_0_layer1_0_bn1_running_var},
                   {g_0_layer1_0_bn1_output, g_0_layer1_0_bn1_saved_mean, g_0_layer1_0_bn1_saved_var},
                   (void*)g_0_layer1_0_bn1_0_params,
                   12,
                   "g_0_layer1_0_bn1_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_bn1_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_layer1_0_bn1_output,
                        g_0_layer1_0_bn1_saved_mean,
                        g_0_layer1_0_bn1_saved_var,
                        g_0_worker_0_maxpoolmax_indices});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, reshape_consumer_with_granularity_constraint)
{
    // Graph #0

    /*************
     * gemm_node node
     * inputs:
     *     gemm_in0[1024, 4000] (dtype=bf16)
     *     gemm_in1[364, 1024] (dtype=bf16)
     * outputs:
     *     gemm_out[364, 4000] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create gemm_in0 tensor
    unsigned gemm_in0_max_sizes[] = {1024, 4000};
    unsigned gemm_in0_min_sizes[] = {1024, 4000};
    unsigned gemm_in0             = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "gemm_in0",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      gemm_in0_max_sizes,
                                      2,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      gemm_in0_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create gemm_in1 tensor
    unsigned gemm_in1_max_sizes[] = {364, 1024};
    unsigned gemm_in1_min_sizes[] = {364, 1024};
    unsigned gemm_in1             = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "gemm_in1",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      gemm_in1_max_sizes,
                                      2,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      gemm_in1_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create gemm_out tensor
    unsigned      gemm_out_max_sizes[] = {364, 4000};
    unsigned      gemm_out_min_sizes[] = {364, 4000};
    unsigned      gemm_out             = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "gemm_out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      gemm_out_max_sizes,
                                      2,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      gemm_out_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId     gemm_node_id;
    unsigned char gemm_node_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {gemm_in0, gemm_in1},
                   {gemm_out},
                   (void*)gemm_node_params,
                   2,
                   "gemm_node",
                   0 /*graphIndex*/,
                   &gemm_node_id);

    /*************
     * reshape_node node
     * inputs:
     *     gemm_out[364, 4000] (dtype=bf16)
     *     reshape_shape_in[364, 1000, 4] (dtype=uint32) (shape tensor)
     * outputs:
     *     reshape_out[364, 1000, 4] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create reshape_shape_in tensor
    unsigned reshape_shape_in_max_sizes[] = {364, 1000, 4};
    unsigned reshape_shape_in_min_sizes[] = {364, 1000, 4};
    unsigned reshape_shape_in             = createTensors(1,
                                              INPUT_TENSOR,
                                              false,
                                              "reshape_shape_in",
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              reshape_shape_in_max_sizes,
                                              3,
                                              syn_type_uint32,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              false,
                                              reshape_shape_in_min_sizes,
                                              synTensorType::SHAPE_TENSOR)[0];

    // create reshape_out tensor
    unsigned  reshape_out_max_sizes[] = {364, 1000, 4};
    unsigned  reshape_out_min_sizes[] = {364, 1000, 4};
    unsigned  reshape_out             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "reshape_out",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         reshape_out_max_sizes,
                                         3,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         reshape_out_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId reshape_node_id;
    addNodeToGraph("reshape",
                   {gemm_out, reshape_shape_in},
                   {reshape_out},
                   nullptr,
                   0,
                   "reshape_node",
                   0 /*graphIndex*/,
                   &reshape_node_id);

    /*************
     * add_node node
     * inputs:
     *     reshape_out[364, 1000, 4] (dtype=bf16)
     *     add_in1[364, 1, 1] (dtype=bf16)
     * outputs:
     *     add_out[364, 1000, 4] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create add_in1 tensor
    unsigned add_in1_max_sizes[] = {364, 1, 1};
    unsigned add_in1_min_sizes[] = {364, 1, 1};
    unsigned add_in1             = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "add_in1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     add_in1_max_sizes,
                                     3,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     add_in1_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];

    // create add_out tensor
    unsigned  add_out_max_sizes[] = {364, 1000, 4};
    unsigned  add_out_min_sizes[] = {364, 1000, 4};
    unsigned  add_out             = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "add_out",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     add_out_max_sizes,
                                     3,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     add_out_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId add_node_id;
    addNodeToGraph("add_fwd_bf16",
                   {reshape_out, add_in1},
                   {add_out},
                   nullptr,
                   0,
                   "add_node",
                   0 /*graphIndex*/,
                   &add_node_id);

    /*************
     * cast_node node
     * inputs:
     *     add_out[364, 1000, 4] (dtype=bf16)
     * outputs:
     *     cast_out[364, 1000, 4] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create cast_out tensor
    unsigned  cast_out_max_sizes[] = {364, 1000, 4};
    unsigned  cast_out_min_sizes[] = {364, 1000, 4};
    unsigned  cast_out             = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "cast_out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      cast_out_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      cast_out_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId cast_node_id;
    addNodeToGraph("cast_bf16_to_f32", {add_out}, {cast_out}, nullptr, 0, "cast_node", 0 /*graphIndex*/, &cast_node_id);

    /*************
     * conv_node node
     * inputs:
     *     conv_x[3, 1344, 832, 4] (dtype=bf16)
     *     conv_w[64, 3, 7, 7] (dtype=bf16)
     * outputs:
     *     conv_y[64, 672, 416, 4] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create conv_x tensor
    unsigned conv_x_max_sizes[] = {3, 1344, 832, 4};
    unsigned conv_x_min_sizes[] = {3, 1344, 832, 4};
    unsigned conv_x             = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "conv_x",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    conv_x_max_sizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    conv_x_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];

    // create conv_w tensor
    unsigned conv_w_max_sizes[] = {64, 3, 7, 7};
    unsigned conv_w_min_sizes[] = {64, 3, 7, 7};
    unsigned conv_w             = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "conv_w",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    conv_w_max_sizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    conv_w_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];

    // create conv_y tensor
    unsigned      conv_y_max_sizes[] = {64, 672, 416, 4};
    unsigned      conv_y_min_sizes[] = {64, 672, 416, 4};
    unsigned      conv_y             = createTensors(1,
                                    OUTPUT_TENSOR,
                                    true,
                                    "conv_y",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    conv_y_max_sizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    conv_y_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     conv_node_id;
    unsigned char conv_node_params[] = {
        7, 0, 0, 0, 7, 0, 0, 0,   2,  0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 151, 89, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {conv_x, conv_w},
                   {conv_y},
                   (void*)conv_node_params,
                   104,
                   "conv_node",
                   0 /*graphIndex*/,
                   &conv_node_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({cast_out, conv_y});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, conv_bn_inference_ASIC_CI)
{
    unsigned g_0_t7322_resnet50_conv1_conv2d_readvariableop_0_max_sizes[] = {64, 3, 7, 7};
    unsigned g_0_t7322_resnet50_conv1_conv2d_readvariableop_0_min_sizes[] = {64, 3, 7, 7};
    unsigned g_0_t7322_resnet50_conv1_conv2d_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7322_resnet50_conv1_conv2d_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7322_resnet50_conv1_conv2d_readvariableop_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7322_resnet50_conv1_conv2d_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_t7596_resnet50_conv1_Conv2D_Cast_0_max_sizes[] = {64, 3, 7, 7};
    unsigned  g_0_t7596_resnet50_conv1_Conv2D_Cast_0_min_sizes[] = {64, 3, 7, 7};
    unsigned  g_0_t7596_resnet50_conv1_Conv2D_Cast_0             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "g_0_t7596_resnet50_conv1_Conv2D_Cast_0",
                                                                    MEM_INIT_ALL_ZERO,
                                                                    nullptr,
                                                                    g_0_t7596_resnet50_conv1_Conv2D_Cast_0_max_sizes,
                                                                    4,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_t7596_resnet50_conv1_Conv2D_Cast_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_resnet50_conv1_Conv2D_Cast_cast_f32_to_bf16_n3961_0_id;
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_t7322_resnet50_conv1_conv2d_readvariableop_0},
                   {g_0_t7596_resnet50_conv1_Conv2D_Cast_0},
                   nullptr,
                   0,
                   "g_0_resnet50_conv1_Conv2D_Cast_cast_f32_to_bf16_n3961_0",
                   0 /*graphIndex*/,
                   &g_0_resnet50_conv1_Conv2D_Cast_cast_f32_to_bf16_n3961_0_id);

    unsigned g_0_t7594_resnet50_conv1_pad_Cast_0_max_sizes[] = {3, 224, 224, 256};
    unsigned g_0_t7594_resnet50_conv1_pad_Cast_0_min_sizes[] = {3, 224, 224, 256};
    unsigned g_0_t7594_resnet50_conv1_pad_Cast_0             = createTensors(1,
                                                                 INPUT_TENSOR,
                                                                 true,
                                                                 "g_0_t7594_resnet50_conv1_pad_Cast_0",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_t7594_resnet50_conv1_pad_Cast_0_max_sizes,
                                                                 4,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_t7594_resnet50_conv1_pad_Cast_0_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_t7597_resnet50_conv1_Conv2D_0_max_sizes[] = {64, 112, 112, 256};
    unsigned      g_0_t7597_resnet50_conv1_Conv2D_0_min_sizes[] = {64, 112, 112, 256};
    unsigned      g_0_t7597_resnet50_conv1_Conv2D_0             = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               false,
                                                               "g_0_t7597_resnet50_conv1_Conv2D_0",
                                                               MEM_INIT_ALL_ZERO,
                                                               nullptr,
                                                               g_0_t7597_resnet50_conv1_Conv2D_0_max_sizes,
                                                               4,
                                                               syn_type_bf16,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_t7597_resnet50_conv1_Conv2D_0_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_resnet50_conv1_Conv2D_conv2d_with_padding_spatial_convolution_n3962_0_id;
    unsigned char g_0_resnet50_conv1_Conv2D_conv2d_with_padding_spatial_convolution_n3962_0_params[] = {
        7, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0,   0,   3,   0,   0, 0,
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 124, 114, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   254, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t7594_resnet50_conv1_pad_Cast_0, g_0_t7596_resnet50_conv1_Conv2D_Cast_0},
                   {g_0_t7597_resnet50_conv1_Conv2D_0},
                   (void*)g_0_resnet50_conv1_Conv2D_conv2d_with_padding_spatial_convolution_n3962_0_params,
                   72,
                   "g_0_resnet50_conv1_Conv2D_conv2d_with_padding_spatial_convolution_n3962_0",
                   0 /*graphIndex*/,
                   &g_0_resnet50_conv1_Conv2D_conv2d_with_padding_spatial_convolution_n3962_0_id);

    unsigned g_0_t7378_resnet50_bn_conv1_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t7378_resnet50_bn_conv1_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t7378_resnet50_bn_conv1_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7378_resnet50_bn_conv1_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7378_resnet50_bn_conv1_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7378_resnet50_bn_conv1_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7377_resnet50_bn_conv1_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t7377_resnet50_bn_conv1_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t7377_resnet50_bn_conv1_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7377_resnet50_bn_conv1_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7377_resnet50_bn_conv1_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7377_resnet50_bn_conv1_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7379_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t7379_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t7379_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7379_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7379_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7379_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7380_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t7380_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t7380_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7380_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7380_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7380_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0_max_sizes[] = {64, 112, 112, 256};
    unsigned g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0_min_sizes[] = {64, 112, 112, 256};
    unsigned g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7708_resnet50_bn_conv1_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t7708_resnet50_bn_conv1_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t7708_resnet50_bn_conv1_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t7708_resnet50_bn_conv1_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t7708_resnet50_bn_conv1_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7708_resnet50_bn_conv1_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7709_resnet50_bn_conv1_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t7709_resnet50_bn_conv1_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t7709_resnet50_bn_conv1_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t7709_resnet50_bn_conv1_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t7709_resnet50_bn_conv1_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7709_resnet50_bn_conv1_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7710_resnet50_bn_conv1_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t7710_resnet50_bn_conv1_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t7710_resnet50_bn_conv1_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t7710_resnet50_bn_conv1_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t7710_resnet50_bn_conv1_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7710_resnet50_bn_conv1_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7711_resnet50_bn_conv1_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t7711_resnet50_bn_conv1_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t7711_resnet50_bn_conv1_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t7711_resnet50_bn_conv1_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t7711_resnet50_bn_conv1_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7711_resnet50_bn_conv1_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_resnet50_bn_conv1_FusedBatchNormV3_batch_norm_fwd_bf16_n4017_0_id;
    unsigned char g_0_resnet50_bn_conv1_FusedBatchNormV3_batch_norm_fwd_bf16_n4017_0_params[] =
        {149, 191, 214, 51, 0, 0, 128, 63, 159, 240, 39, 55, 0, 0, 0, 0};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_t7597_resnet50_conv1_Conv2D_0,
                    g_0_t7378_resnet50_bn_conv1_readvariableop_1_0,
                    g_0_t7377_resnet50_bn_conv1_readvariableop_0,
                    g_0_t7379_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_0,
                    g_0_t7380_resnet50_bn_conv1_fusedbatchnormv3_readvariableop_1_0},
                   {g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0,
                    g_0_t7708_resnet50_bn_conv1_FusedBatchNormV3,
                    g_0_t7709_resnet50_bn_conv1_FusedBatchNormV3,
                    g_0_t7710_resnet50_bn_conv1_FusedBatchNormV3,
                    g_0_t7711_resnet50_bn_conv1_FusedBatchNormV3},
                   (void*)g_0_resnet50_bn_conv1_FusedBatchNormV3_batch_norm_fwd_bf16_n4017_0_params,
                   16,
                   "g_0_resnet50_bn_conv1_FusedBatchNormV3_batch_norm_fwd_bf16_n4017_0",
                   0 /*graphIndex*/,
                   &g_0_resnet50_bn_conv1_FusedBatchNormV3_batch_norm_fwd_bf16_n4017_0_id);

    unsigned  g_0_t7712_resnet50_activation_Relu_0_max_sizes[] = {64, 112, 112, 256};
    unsigned  g_0_t7712_resnet50_activation_Relu_0_min_sizes[] = {64, 112, 112, 256};
    unsigned  g_0_t7712_resnet50_activation_Relu_0             = createTensors(1,
                                                                  OUTPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t7712_resnet50_activation_Relu_0",
                                                                  MEM_INIT_ALL_ZERO,
                                                                  nullptr,
                                                                  g_0_t7712_resnet50_activation_Relu_0_max_sizes,
                                                                  4,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t7712_resnet50_activation_Relu_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_resnet50_activation_Relu_relu_fwd_bf16_n4018_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_t7706_resnet50_bn_conv1_FusedBatchNormV3_0},
                   {g_0_t7712_resnet50_activation_Relu_0},
                   nullptr,
                   0,
                   "g_0_resnet50_activation_Relu_relu_fwd_bf16_n4018_0",
                   0 /*graphIndex*/,
                   &g_0_resnet50_activation_Relu_relu_fwd_bf16_n4018_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_t7712_resnet50_activation_Relu_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, bn_eviction_with_cl_align_ASIC)
{
    // The graph includes a single bundle BN->[T1]->Conv->[T2]->BN and a consumer outside the bundle
    // for the intermediate tensor between the BN producer and the Conv node (T1).
    // T1 is aligned to CL to improve the performance of the Conv node.
    // In addition, the BN eviction fuser adds an additional output to the BN to avoid a DMA copy
    // to evict T1 from SRAM to HBM.
    // The test validates that the results are correct when the pipelined tensor in SRAM (T1)
    // is aligned to CL and the evicted output in HBM is not aligned.

    // Graph #0

    /*************
     * g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3273_0 node
     * inputs:
     *     g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0[288, 14, 14, 128] (dtype=bf16)
     *     g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0[288] (dtype=float32)
     *     g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0[288] (dtype=float32)
     *     g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0[288] (dtype=float32)
     *     g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0[288] (dtype=float32)
     * outputs:
     *     g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3[288, 14, 14, 128] (dtype=bf16)
     *     g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3[288] (dtype=float32)
     *     g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3[288] (dtype=float32)
     *     g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1[288] (dtype=float32)
     *     g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3[288] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *     g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3273_control_edge_11043[] (dtype=invalid)
     *************/

    // create g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0 tensor
    unsigned g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0_max_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0_min_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0 tensor
    unsigned g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0_max_sizes[] = {288};
    unsigned g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0_min_sizes[] = {288};
    unsigned g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0 tensor
    unsigned g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0_max_sizes[] = {288};
    unsigned g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0_min_sizes[] = {288};
    unsigned g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0_max_sizes[] = {288};
    unsigned g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0_min_sizes[] = {288};
    unsigned g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {288};
    unsigned g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {288};
    unsigned g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3 tensor
    unsigned g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3_max_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3_min_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3 tensor
    unsigned g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3_max_sizes[] = {288};
    unsigned g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3_min_sizes[] = {288};
    unsigned g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3 tensor
    unsigned g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3_max_sizes[] = {288};
    unsigned g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3_min_sizes[] = {288};
    unsigned g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1 tensor
    unsigned g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1_max_sizes[] = {288};
    unsigned g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1_min_sizes[] = {288};
    unsigned g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3 tensor
    unsigned g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3_max_sizes[] = {288};
    unsigned g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3_min_sizes[] = {288};
    unsigned g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3273_0_id;
    unsigned char g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3273_0_params[] =
        {149, 191, 214, 51, 10, 215, 35, 60, 164, 140, 56, 55, 1, 0, 0, 0};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_t6188_densenet_concatenate_18_concat_fp32_to_bf16_cast_407_0,
                    g_0_t1205_conv4_2_x1_bn_beta_regularizer_l2loss_readvariableop_0,
                    g_0_t1204_conv4_2_x1_bn_gamma_regularizer_l2loss_readvariableop_0,
                    g_0_t1528_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_0,
                    g_0_t1529_densenet_conv4_2_x1_bn_fusedbatchnormv3_readvariableop_1_0},
                   {g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3,
                    g_0_t6197_densenet_conv4_2_x1_bn_FusedBatchNormV3_3,
                    g_0_t6199_densenet_conv4_2_x1_bn_FusedBatchNormV3,
                    g_0_t6190_densenet_conv4_2_x1_bn_FusedBatchNormV3_1,
                    g_0_t6196_densenet_conv4_2_x1_bn_FusedBatchNormV3},
                   (void*)g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3273_0_params,
                   16,
                   "g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3273_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3273_0_id);

    /*************
     * g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_relu_fwd_bf16_n3272_0 node
     * inputs:
     *     g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3[288, 14, 14, 128] (dtype=bf16)
     * outputs:
     *     g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0[288, 14, 14, 128] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *     g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_relu_fwd_bf16_n3272_control_edge_11042[] (dtype=invalid)
     *************/

    // create g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0 tensor
    unsigned g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0_max_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0_min_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_relu_fwd_bf16_n3272_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_t6206_densenet_conv4_2_x1_bn_FusedBatchNormV3},
                   {g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0},
                   nullptr,
                   0,
                   "g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_relu_fwd_bf16_n3272_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_conv4_2_x1_bn_FusedBatchNormV3_relu_fwd_bf16_n3272_0_id);

    /*************
     * g_0_densenet_conv4_2_x1_Conv2D_spatial_convolution_n3281_0 node
     * inputs:
     *     g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0[288, 14, 14, 128] (dtype=bf16)
     *     g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0[128, 288, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_t6209_densenet_conv4_2_x1_Conv2D_0[128, 14, 14, 128] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0 tensor
    unsigned g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0_max_sizes[] = {128, 288, 1, 1};
    unsigned g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0_min_sizes[] = {128, 288, 1, 1};
    unsigned g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6209_densenet_conv4_2_x1_Conv2D_0 tensor
    unsigned      g_0_t6209_densenet_conv4_2_x1_Conv2D_0_max_sizes[] = {128, 14, 14, 128};
    unsigned      g_0_t6209_densenet_conv4_2_x1_Conv2D_0_min_sizes[] = {128, 14, 14, 128};
    unsigned      g_0_t6209_densenet_conv4_2_x1_Conv2D_0             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "g_0_t6209_densenet_conv4_2_x1_Conv2D_0",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_t6209_densenet_conv4_2_x1_Conv2D_0_max_sizes,
                                                                    4,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_t6209_densenet_conv4_2_x1_Conv2D_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_densenet_conv4_2_x1_Conv2D_spatial_convolution_n3281_0_id;
    unsigned char g_0_densenet_conv4_2_x1_Conv2D_spatial_convolution_n3281_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0,
                    g_0_t5009_densenet_conv4_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_408_0},
                   {g_0_t6209_densenet_conv4_2_x1_Conv2D_0},
                   (void*)g_0_densenet_conv4_2_x1_Conv2D_spatial_convolution_n3281_0_params,
                   112,
                   "g_0_densenet_conv4_2_x1_Conv2D_spatial_convolution_n3281_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_conv4_2_x1_Conv2D_spatial_convolution_n3281_0_id);

    /*************
     * g_0_densenet_conv4_2_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3283_0 node
     * inputs:
     *     g_0_t6209_densenet_conv4_2_x1_Conv2D_0[128, 14, 14, 128] (dtype=bf16)
     *     g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0[128] (dtype=float32)
     *     g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0[128] (dtype=float32)
     *     g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0[128] (dtype=float32)
     *     g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0[128] (dtype=float32)
     * outputs:
     *     g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3[128, 14, 14, 128] (dtype=bf16)
     *     g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3[128] (dtype=float32)
     *     g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3[128] (dtype=float32)
     *     g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1[128] (dtype=float32)
     *     g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3[128] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0 tensor
    unsigned g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0_max_sizes[] = {128};
    unsigned g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0_min_sizes[] = {128};
    unsigned g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0 tensor
    unsigned g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0_max_sizes[] = {128};
    unsigned g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0_min_sizes[] = {128};
    unsigned g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0_max_sizes[] = {128};
    unsigned g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0_min_sizes[] = {128};
    unsigned g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {128};
    unsigned g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {128};
    unsigned g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3 tensor
    unsigned g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3_max_sizes[] = {128, 14, 14, 128};
    unsigned g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3_min_sizes[] = {128, 14, 14, 128};
    unsigned g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3 tensor
    unsigned g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3_max_sizes[] = {128};
    unsigned g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3_min_sizes[] = {128};
    unsigned g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3 tensor
    unsigned g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3_max_sizes[] = {128};
    unsigned g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3_min_sizes[] = {128};
    unsigned g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1 tensor
    unsigned g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1_max_sizes[] = {128};
    unsigned g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1_min_sizes[] = {128};
    unsigned g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3 tensor
    unsigned g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3_max_sizes[] = {128};
    unsigned g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3_min_sizes[] = {128};
    unsigned g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_densenet_conv4_2_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3283_0_id;
    unsigned char g_0_densenet_conv4_2_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3283_0_params[] =
        {149, 191, 214, 51, 10, 215, 35, 60, 164, 140, 56, 55, 1, 0, 0, 0};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_t6209_densenet_conv4_2_x1_Conv2D_0,
                    g_0_t1208_conv4_2_x2_bn_beta_regularizer_l2loss_readvariableop_0,
                    g_0_t1207_conv4_2_x2_bn_gamma_regularizer_l2loss_readvariableop_0,
                    g_0_t1530_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_0,
                    g_0_t1531_densenet_conv4_2_x2_bn_fusedbatchnormv3_readvariableop_1_0},
                   {g_0_t6227_densenet_conv4_2_x2_bn_FusedBatchNormV3,
                    g_0_t6218_densenet_conv4_2_x2_bn_FusedBatchNormV3_3,
                    g_0_t6220_densenet_conv4_2_x2_bn_FusedBatchNormV3,
                    g_0_t6211_densenet_conv4_2_x2_bn_FusedBatchNormV3_1,
                    g_0_t6217_densenet_conv4_2_x2_bn_FusedBatchNormV3},
                   (void*)g_0_densenet_conv4_2_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3283_0_params,
                   16,
                   "g_0_densenet_conv4_2_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3283_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_conv4_2_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n3283_0_id);

    /*************
     * g_0_gradient_tape_densenet_relu4_2_x1_ReluGrad_relu_bwd_bf16_n6041_0 node
     * inputs:
     *     g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0[288, 14, 14, 128] (dtype=bf16)
     *     g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0[288, 14, 14, 128] (dtype=bf16)
     * outputs:
     *     g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0[288, 14, 14, 128] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0 tensor
    unsigned g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0_max_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0_min_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0 tensor
    unsigned g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0_max_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0_min_sizes[] = {288, 14, 14, 128};
    unsigned g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_tape_densenet_relu4_2_x1_ReluGrad_relu_bwd_bf16_n6041_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_t11139_gradient_tape_densenet_conv4_2_x1_Conv2D_Conv2DBackpropInput_0,
                    g_0_t6189_densenet_conv4_2_x1_bn_FusedBatchNormV3_0},
                   {g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0},
                   nullptr,
                   0,
                   "g_0_gradient_tape_densenet_relu4_2_x1_ReluGrad_relu_bwd_bf16_n6041_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_tape_densenet_relu4_2_x1_ReluGrad_relu_bwd_bf16_n6041_0_id);

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_t11141_gradient_tape_densenet_relu4_2_x1_ReluGrad_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, cycle_detection_ASIC)
{
    unsigned g_0_t1415_resnet34_conv2d_28_conv2d_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1415_resnet34_conv2d_28_conv2d_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1415_resnet34_conv2d_28_conv2d_0             = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_t1415_resnet34_conv2d_28_conv2d_0",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_t1415_resnet34_conv2d_28_conv2d_0_max_sizes,
                                                                   4,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_t1415_resnet34_conv2d_28_conv2d_0_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_max_sizes[] = {256};
    unsigned g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_min_sizes[] = {256};
    unsigned g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes[] = {
            256};
    unsigned
        g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes[] = {
            256};
    unsigned g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1368_l2loss_85_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t1368_l2loss_85_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t1368_l2loss_85_readvariableop_0             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t1368_l2loss_85_readvariableop_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t1368_l2loss_85_readvariableop_0_max_sizes,
                                                                  1,
                                                                  syn_type_single,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t1368_l2loss_85_readvariableop_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes[] =
            {256, 38, 38, 128};
    unsigned
        g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes[] =
            {256, 38, 38, 128};
    unsigned g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes[] = {
            256};
    unsigned
        g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes[] = {
            256};
    unsigned g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes[] = {
            256};
    unsigned
        g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes[] = {
            256};
    unsigned g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_id;
    unsigned char
        g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t1415_resnet34_conv2d_28_conv2d_0,
         g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0,
         g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1,
         g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3,
         g_0_t1368_l2loss_85_readvariableop_0},
        {g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2,
         g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1},
        (void*)
            g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_params,
        16,
        "g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_"
        "n906_0",
        0 /*graphIndex*/,
        &g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_id);

    unsigned g_0_t1416_cast_29_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t1416_cast_29_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t1416_cast_29_0             = createTensors(1,
                                                 INPUT_TENSOR,
                                                 true,
                                                 "g_0_t1416_cast_29_0",
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 g_0_t1416_cast_29_0_max_sizes,
                                                 4,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_t1416_cast_29_0_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2309_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2309_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2309_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t2309_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2309_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2309_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_dedx_n907_0_id;
    unsigned char g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_dedx_n907_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,  0,   1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 70, 109, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
                    g_0_t1416_cast_29_0,
                    g_0_t2309_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput},
                   {g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0},
                   (void*)g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_dedx_n907_0_params,
                   72,
                   "g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_dedx_n907_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_dedx_n907_0_id);

    unsigned g_0_t1417_resnet34_relu_25_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1417_resnet34_relu_25_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1417_resnet34_relu_25_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t1417_resnet34_relu_25_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t1417_resnet34_relu_25_0_max_sizes,
                                                          4,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t1417_resnet34_relu_25_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_dedw_n908_0_id;
    unsigned char g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_dedw_n908_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,  0,   1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 70, 109, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
                    g_0_t1417_resnet34_relu_25_0},
                   {g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0},
                   (void*)g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_dedw_n908_0_params,
                   72,
                   "g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_dedw_n908_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_dedw_n908_0_id);

    unsigned g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_resnet34_Relu_25_grad_ReluGrad_relu_bwd_bf16_n910_0_id;
    addNodeToGraph(
        "relu_bwd_bf16",
        {g_0_t2308_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropInput_0, g_0_t1417_resnet34_relu_25_0},
        {g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0},
        nullptr,
        0,
        "g_0_gradients_resnet34_Relu_25_grad_ReluGrad_relu_bwd_bf16_n910_0",
        0 /*graphIndex*/,
        &g_0_gradients_resnet34_Relu_25_grad_ReluGrad_relu_bwd_bf16_n910_0_id);

    unsigned g_0_t1418_resnet34_conv2d_27_conv2d_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1418_resnet34_conv2d_27_conv2d_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1418_resnet34_conv2d_27_conv2d_0             = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_t1418_resnet34_conv2d_27_conv2d_0",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_t1418_resnet34_conv2d_27_conv2d_0_max_sizes,
                                                                   4,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_t1418_resnet34_conv2d_27_conv2d_0_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1577_resnet34_batch_normalization_27_FusedBatchNormV3_1_max_sizes[] = {256};
    unsigned g_0_t1577_resnet34_batch_normalization_27_FusedBatchNormV3_1_min_sizes[] = {256};
    unsigned g_0_t1577_resnet34_batch_normalization_27_FusedBatchNormV3_1 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1577_resnet34_batch_normalization_27_FusedBatchNormV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1577_resnet34_batch_normalization_27_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1577_resnet34_batch_normalization_27_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2320_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes[] = {
            256};
    unsigned
        g_0_t2320_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes[] = {
            256};
    unsigned g_0_t2320_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t2320_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t2320_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2320_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1366_l2loss_82_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t1366_l2loss_82_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t1366_l2loss_82_readvariableop_0             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t1366_l2loss_82_readvariableop_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t1366_l2loss_82_readvariableop_0_max_sizes,
                                                                  1,
                                                                  syn_type_single,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t1366_l2loss_82_readvariableop_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes[] =
            {256, 38, 38, 128};
    unsigned
        g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes[] =
            {256, 38, 38, 128};
    unsigned g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2316_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes[] = {
            256};
    unsigned
        g_0_t2316_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes[] = {
            256};
    unsigned g_0_t2316_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_2 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t2316_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_2",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2316_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2316_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2315_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes[] = {
            256};
    unsigned
        g_0_t2315_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes[] = {
            256};
    unsigned g_0_t2315_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_1 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t2315_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_1",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2315_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2315_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n915_0_id;
    unsigned char
        g_0_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n915_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t1418_resnet34_conv2d_27_conv2d_0,
         g_0_t2313_gradients_resnet34_Relu_25_grad_ReluGrad_0,
         g_0_t1577_resnet34_batch_normalization_27_FusedBatchNormV3_1,
         g_0_t2320_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3,
         g_0_t1366_l2loss_82_readvariableop_0},
        {g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t2316_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_2,
         g_0_t2315_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_1},
        (void*)
            g_0_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n915_0_params,
        16,
        "g_0_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_"
        "n915_0",
        0 /*graphIndex*/,
        &g_0_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n915_0_id);

    unsigned g_0_t1419_cast_28_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t1419_cast_28_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t1419_cast_28_0             = createTensors(1,
                                                 INPUT_TENSOR,
                                                 true,
                                                 "g_0_t1419_cast_28_0",
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 g_0_t1419_cast_28_0_max_sizes,
                                                 4,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_t1419_cast_28_0_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2324_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2324_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2324_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t2324_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2324_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2324_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_dedx_n916_0_id;
    unsigned char g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_dedx_n916_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,  0,   1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 75, 170, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
                    g_0_t1419_cast_28_0,
                    g_0_t2324_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput},
                   {g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0},
                   (void*)g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_dedx_n916_0_params,
                   72,
                   "g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_dedx_n916_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_dedx_n916_0_id);

    unsigned g_0_t1420_resnet34_relu_24_0_max_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1420_resnet34_relu_24_0_min_sizes[] = {256, 38, 38, 128};
    unsigned g_0_t1420_resnet34_relu_24_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t1420_resnet34_relu_24_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t1420_resnet34_relu_24_0_max_sizes,
                                                          4,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t1420_resnet34_relu_24_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_dedw_n917_0_id;
    unsigned char g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_dedw_n917_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,  0,   1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 75, 170, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,   255, 127, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_t2314_gradients_resnet34_batch_normalization_27_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
                    g_0_t1420_resnet34_relu_24_0},
                   {g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0},
                   (void*)g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_dedw_n917_0_params,
                   72,
                   "g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_dedw_n917_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_dedw_n917_0_id);

    synNodeId blocking_g_0_gradients_resnet34_Relu_25_grad_ReluGrad_relu_bwd_bf16_n910_0[] = {
        g_0_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_dedw_n908_0_id};
    setNodeDependency(blocking_g_0_gradients_resnet34_Relu_25_grad_ReluGrad_relu_bwd_bf16_n910_0,
                      &g_0_gradients_resnet34_Relu_25_grad_ReluGrad_relu_bwd_bf16_n910_0_id,
                      1,
                      1);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_t2310_gradients_resnet34_conv2d_28_Conv2D_grad_Conv2DBackpropFilter_0,
                        g_0_t2325_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropFilter_0,
                        g_0_t2323_gradients_resnet34_conv2d_27_Conv2D_grad_Conv2DBackpropInput_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, conv_maxpool_cast_ASIC_CI)
{
    unsigned g_0_t931_l2loss_50_readvariableop_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t931_l2loss_50_readvariableop_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t931_l2loss_50_readvariableop_0             = createTensors(1,
                                                                 INPUT_TENSOR,
                                                                 true,
                                                                 "g_0_t931_l2loss_50_readvariableop_0",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_t931_l2loss_50_readvariableop_0_max_sizes,
                                                                 4,
                                                                 syn_type_single,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_t931_l2loss_50_readvariableop_0_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0 tensor
    unsigned g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_cast_f32_to_bf16_n954_0_id;
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_t931_l2loss_50_readvariableop_0},
                   {g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0},
                   nullptr,
                   0,
                   "g_0_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_cast_f32_to_bf16_n954_0",
                   0 /*graphIndex*/,
                   &g_0_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_cast_f32_to_bf16_n954_0_id);

    unsigned g_0_t2733_fpn_post_hoc_d5_BiasAdd_0_max_sizes[] = {256, 42, 26, 64};
    unsigned g_0_t2733_fpn_post_hoc_d5_BiasAdd_0_min_sizes[] = {256, 42, 26, 64};
    unsigned g_0_t2733_fpn_post_hoc_d5_BiasAdd_0             = createTensors(1,
                                                                 INPUT_TENSOR,
                                                                 true,
                                                                 "g_0_t2733_fpn_post_hoc_d5_BiasAdd_0",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_t2733_fpn_post_hoc_d5_BiasAdd_0_max_sizes,
                                                                 4,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_t2733_fpn_post_hoc_d5_BiasAdd_0_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2756_fpn_p6_MaxPool_max_sizes[] = {256, 21, 13, 64};
    unsigned g_0_t2756_fpn_p6_MaxPool_min_sizes[] = {256, 21, 13, 64};
    unsigned g_0_t2756_fpn_p6_MaxPool             = createTensors(1,
                                                      OUTPUT_TENSOR,
                                                      true,
                                                      "g_0_t2756_fpn_p6_MaxPool",
                                                      MEM_INIT_ALL_ZERO,
                                                      nullptr,
                                                      g_0_t2756_fpn_p6_MaxPool_max_sizes,
                                                      4,
                                                      syn_type_int16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_t2756_fpn_p6_MaxPool_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_t2755_fpn_p6_MaxPool_0_max_sizes[] = {256, 21, 13, 64};
    unsigned      g_0_t2755_fpn_p6_MaxPool_0_min_sizes[] = {256, 21, 13, 64};
    unsigned      g_0_t2755_fpn_p6_MaxPool_0             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        false,
                                                        "g_0_t2755_fpn_p6_MaxPool_0",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_t2755_fpn_p6_MaxPool_0_max_sizes,
                                                        4,
                                                        syn_type_bf16,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_t2755_fpn_p6_MaxPool_0_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_fpn_p6_MaxPool_maxpool_2d_fwd_bf16_n1214_0_id;
    unsigned char g_0_fpn_p6_MaxPool_maxpool_2d_fwd_bf16_n1214_0_params[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("maxpool_2d_fwd_bf16",
                   {g_0_t2733_fpn_post_hoc_d5_BiasAdd_0},
                   {g_0_t2756_fpn_p6_MaxPool, g_0_t2755_fpn_p6_MaxPool_0},
                   (void*)g_0_fpn_p6_MaxPool_maxpool_2d_fwd_bf16_n1214_0_params,
                   44,
                   "g_0_fpn_p6_MaxPool_maxpool_2d_fwd_bf16_n1214_0",
                   0 /*graphIndex*/,
                   &g_0_fpn_p6_MaxPool_maxpool_2d_fwd_bf16_n1214_0_id);

    unsigned      g_0_t2757_rpn_head_4_rpn_Conv2D_0_max_sizes[] = {256, 21, 13, 64};
    unsigned      g_0_t2757_rpn_head_4_rpn_Conv2D_0_min_sizes[] = {256, 21, 13, 64};
    unsigned      g_0_t2757_rpn_head_4_rpn_Conv2D_0             = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               true,
                                                               "g_0_t2757_rpn_head_4_rpn_Conv2D_0",
                                                               MEM_INIT_ALL_ZERO,
                                                               nullptr,
                                                               g_0_t2757_rpn_head_4_rpn_Conv2D_0_max_sizes,
                                                               4,
                                                               syn_type_bf16,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_t2757_rpn_head_4_rpn_Conv2D_0_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_rpn_head_4_rpn_Conv2D_spatial_convolution_n1215_0_id;
    unsigned char g_0_rpn_head_4_rpn_Conv2D_spatial_convolution_n1215_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,  1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 176, 68, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,  168, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t2755_fpn_p6_MaxPool_0, g_0_t2161_rpn_head_4_rpn_Conv2D_ReadVariableOp_fp32_to_bf16_cast_130_0},
                   {g_0_t2757_rpn_head_4_rpn_Conv2D_0},
                   (void*)g_0_rpn_head_4_rpn_Conv2D_spatial_convolution_n1215_0_params,
                   72,
                   "g_0_rpn_head_4_rpn_Conv2D_spatial_convolution_n1215_0",
                   0 /*graphIndex*/,
                   &g_0_rpn_head_4_rpn_Conv2D_spatial_convolution_n1215_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_t2757_rpn_head_4_rpn_Conv2D_0, g_0_t2756_fpn_p6_MaxPool});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, align_mme_inputs_to_cl_ASIC)
{
    unsigned g_0_t2072_gradients_Cast_74_grad_Cast_0_max_sizes[] = {324, 38, 38, 240};
    unsigned g_0_t2072_gradients_Cast_74_grad_Cast_0_min_sizes[] = {324, 38, 38, 240};
    unsigned g_0_t2072_gradients_Cast_74_grad_Cast_0             = createTensors(1,
                                                                     INPUT_TENSOR,
                                                                     true,
                                                                     "g_0_t2072_gradients_Cast_74_grad_Cast_0",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_t2072_gradients_Cast_74_grad_Cast_0_max_sizes,
                                                                     4,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_t2072_gradients_Cast_74_grad_Cast_0_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_t1677_resnet34_Relu_26_0 tensor
    unsigned g_0_t1677_resnet34_Relu_26_0_max_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t1677_resnet34_Relu_26_0_min_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t1677_resnet34_Relu_26_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t1677_resnet34_Relu_26_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t1677_resnet34_Relu_26_0_max_sizes,
                                                          4,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t1677_resnet34_Relu_26_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes[] = {324,
                                                                                                         256,
                                                                                                         3,
                                                                                                         3};
    unsigned g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes[] = {324,
                                                                                                         256,
                                                                                                         3,
                                                                                                         3};
    unsigned g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_dedw_n766_0_id;
    unsigned char g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_dedw_n766_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,   1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6, 120, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,   252, 127, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_t2072_gradients_Cast_74_grad_Cast_0, g_0_t1677_resnet34_Relu_26_0},
                   {g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0},
                   (void*)g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_dedw_n766_0_params,
                   72,
                   "g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_dedw_n766_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_dedw_n766_0_id);

    unsigned g_0_t1608_Cast_50_0_max_sizes[] = {324, 256, 3, 3};
    unsigned g_0_t1608_Cast_50_0_min_sizes[] = {324, 256, 3, 3};
    unsigned g_0_t1608_Cast_50_0             = createTensors(1,
                                                 INPUT_TENSOR,
                                                 true,
                                                 "g_0_t1608_Cast_50_0",
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 g_0_t1608_Cast_50_0_max_sizes,
                                                 4,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_t1608_Cast_50_0_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2082_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_max_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t2082_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_min_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t2082_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t2082_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2082_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2082_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0_max_sizes[] = {256,
                                                                                                        38,
                                                                                                        38,
                                                                                                        240};
    unsigned g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0_min_sizes[] = {256,
                                                                                                        38,
                                                                                                        38,
                                                                                                        240};
    unsigned g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_dedx_n767_0_id;
    unsigned char g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_dedx_n767_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,   1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6, 120, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,   252, 127, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_t2072_gradients_Cast_74_grad_Cast_0,
                    g_0_t1608_Cast_50_0,
                    g_0_t2082_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput},
                   {g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0},
                   (void*)g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_dedx_n767_0_params,
                   72,
                   "g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_dedx_n767_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_dedx_n767_0_id);

    unsigned g_0_t2297_gradients_AddN_49_max_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t2297_gradients_AddN_49_min_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t2297_gradients_AddN_49             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_t2297_gradients_AddN_49",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_t2297_gradients_AddN_49_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_t2297_gradients_AddN_49_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_t2296_gradients_AddN_49_0_max_sizes[] = {256, 38, 38, 240};
    unsigned  g_0_t2296_gradients_AddN_49_0_min_sizes[] = {256, 38, 38, 240};
    unsigned  g_0_t2296_gradients_AddN_49_0             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_t2296_gradients_AddN_49_0",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_t2296_gradients_AddN_49_0_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_t2296_gradients_AddN_49_0_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_AddN_49_add_fwd_bf16_n900_0_id;
    addNodeToGraph(
        "add_fwd_bf16",
        {g_0_t2081_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropInput_0, g_0_t2297_gradients_AddN_49},
        {g_0_t2296_gradients_AddN_49_0},
        nullptr,
        0,
        "g_0_gradients_AddN_49_add_fwd_bf16_n900_0",
        0 /*graphIndex*/,
        &g_0_gradients_AddN_49_add_fwd_bf16_n900_0_id);

    unsigned g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_max_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_min_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_resnet34_Relu_26_grad_ReluGrad_relu_bwd_bf16_n901_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_t2296_gradients_AddN_49_0, g_0_t1677_resnet34_Relu_26_0},
                   {g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0},
                   nullptr,
                   0,
                   "g_0_gradients_resnet34_Relu_26_grad_ReluGrad_relu_bwd_bf16_n901_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_resnet34_Relu_26_grad_ReluGrad_relu_bwd_bf16_n901_0_id);

    unsigned g_0_t1415_resnet34_conv2d_28_conv2d_0_max_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t1415_resnet34_conv2d_28_conv2d_0_min_sizes[] = {256, 38, 38, 240};
    unsigned g_0_t1415_resnet34_conv2d_28_conv2d_0             = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_t1415_resnet34_conv2d_28_conv2d_0",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_t1415_resnet34_conv2d_28_conv2d_0_max_sizes,
                                                                   4,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_t1415_resnet34_conv2d_28_conv2d_0_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_max_sizes[] = {256};
    unsigned g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_min_sizes[] = {256};
    unsigned g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes[] = {
            256};
    unsigned
        g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes[] = {
            256};
    unsigned g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1368_l2loss_85_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t1368_l2loss_85_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t1368_l2loss_85_readvariableop_0             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t1368_l2loss_85_readvariableop_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t1368_l2loss_85_readvariableop_0_max_sizes,
                                                                  1,
                                                                  syn_type_single,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t1368_l2loss_85_readvariableop_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes[] =
            {256, 38, 38, 240};
    unsigned
        g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes[] =
            {256, 38, 38, 240};
    unsigned g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes[] = {
            256};
    unsigned
        g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes[] = {
            256};
    unsigned g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes[] = {
            256};
    unsigned
        g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes[] = {
            256};
    unsigned g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_id;
    unsigned char
        g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t1415_resnet34_conv2d_28_conv2d_0,
         g_0_t2298_gradients_resnet34_Relu_26_grad_ReluGrad_0,
         g_0_t1580_resnet34_batch_normalization_28_FusedBatchNormV3_1,
         g_0_t2305_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3,
         g_0_t1368_l2loss_85_readvariableop_0},
        {g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2,
         g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1},
        (void*)
            g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_params,
        16,
        "g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_"
        "n906_0",
        0 /*graphIndex*/,
        &g_0_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n906_0_id);

    synNodeId blocking_g_0_gradients_AddN_49_add_fwd_bf16_n900_0[] = {
        g_0_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_dedw_n766_0_id};
    setNodeDependency(blocking_g_0_gradients_AddN_49_add_fwd_bf16_n900_0,
                      &g_0_gradients_AddN_49_add_fwd_bf16_n900_0_id,
                      1,
                      1);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults(
        {g_0_t2080_gradients_ssd_class_net_class_3_Conv2D_grad_Conv2DBackpropFilter_0,
         g_0_t2299_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t2301_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_2,
         g_0_t2300_gradients_resnet34_batch_normalization_28_FusedBatchNormV3_grad_FusedBatchNormGradV3_1});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, shared_mme_creates_cycle3)
{
    // Graph #0

    /*************
     * g_0_model_dense_MatMul_gemm_n3882_0 node
     * inputs:
     *     g_0_t6054_iteratorgetnext_0[10, 2] (dtype=float32)
     *     g_0_t6055_model_dense_matmul_readvariableop_0[10, 10] (dtype=float32)
     * outputs:
     *     g_0_t6059_model_dense_MatMul_0[10, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t6054_iteratorgetnext_0 tensor
    unsigned g_0_t6054_iteratorgetnext_0_max_sizes[] = {10, 2};
    unsigned g_0_t6054_iteratorgetnext_0_min_sizes[] = {10, 2};
    unsigned g_0_t6054_iteratorgetnext_0             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_t6054_iteratorgetnext_0",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_t6054_iteratorgetnext_0_max_sizes,
                                                         2,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_t6054_iteratorgetnext_0_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_t6055_model_dense_matmul_readvariableop_0 tensor
    unsigned g_0_t6055_model_dense_matmul_readvariableop_0_max_sizes[] = {10, 10};
    unsigned g_0_t6055_model_dense_matmul_readvariableop_0_min_sizes[] = {10, 10};
    unsigned g_0_t6055_model_dense_matmul_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6055_model_dense_matmul_readvariableop_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t6055_model_dense_matmul_readvariableop_0_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6055_model_dense_matmul_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6059_model_dense_MatMul_0 tensor
    unsigned      g_0_t6059_model_dense_MatMul_0_max_sizes[] = {10, 2};
    unsigned      g_0_t6059_model_dense_MatMul_0_min_sizes[] = {10, 2};
    unsigned      g_0_t6059_model_dense_MatMul_0             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            false,
                                                            "g_0_t6059_model_dense_MatMul_0",
                                                            MEM_INIT_ALL_ZERO,
                                                            nullptr,
                                                            g_0_t6059_model_dense_MatMul_0_max_sizes,
                                                            2,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_t6059_model_dense_MatMul_0_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_dense_MatMul_gemm_n3882_0_id;
    unsigned char g_0_model_dense_MatMul_gemm_n3882_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t6054_iteratorgetnext_0, g_0_t6055_model_dense_matmul_readvariableop_0},
                   {g_0_t6059_model_dense_MatMul_0},
                   (void*)g_0_model_dense_MatMul_gemm_n3882_0_params,
                   2,
                   "g_0_model_dense_MatMul_gemm_n3882_0",
                   0 /*graphIndex*/,
                   &g_0_model_dense_MatMul_gemm_n3882_0_id);

    /*************
     * g_0_model_dense_BiasAdd_add_fwd_f32_n3884_0 node
     * inputs:
     *     g_0_t6059_model_dense_MatMul_0[10, 2] (dtype=float32)
     *     g_0_t6061_model_dense_BiasAdd[10, 1] (dtype=float32)
     * outputs:
     *     g_0_t6060_model_dense_BiasAdd_0[10, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t6061_model_dense_BiasAdd tensor
    unsigned g_0_t6061_model_dense_BiasAdd_max_sizes[] = {10, 1};
    unsigned g_0_t6061_model_dense_BiasAdd_min_sizes[] = {10, 1};
    unsigned g_0_t6061_model_dense_BiasAdd             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_t6061_model_dense_BiasAdd",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_t6061_model_dense_BiasAdd_max_sizes,
                                                           2,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_t6061_model_dense_BiasAdd_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_t6060_model_dense_BiasAdd_0 tensor
    unsigned  g_0_t6060_model_dense_BiasAdd_0_max_sizes[] = {10, 2};
    unsigned  g_0_t6060_model_dense_BiasAdd_0_min_sizes[] = {10, 2};
    unsigned  g_0_t6060_model_dense_BiasAdd_0             = createTensors(1,
                                                             OUTPUT_TENSOR,
                                                             false,
                                                             "g_0_t6060_model_dense_BiasAdd_0",
                                                             MEM_INIT_ALL_ZERO,
                                                             nullptr,
                                                             g_0_t6060_model_dense_BiasAdd_0_max_sizes,
                                                             2,
                                                             syn_type_single,
                                                             nullptr,
                                                             0,
                                                             0,
                                                             nullptr,
                                                             false,
                                                             g_0_t6060_model_dense_BiasAdd_0_min_sizes,
                                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_dense_BiasAdd_add_fwd_f32_n3884_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_t6059_model_dense_MatMul_0, g_0_t6061_model_dense_BiasAdd},
                   {g_0_t6060_model_dense_BiasAdd_0},
                   nullptr,
                   0,
                   "g_0_model_dense_BiasAdd_add_fwd_f32_n3884_0",
                   0 /*graphIndex*/,
                   &g_0_model_dense_BiasAdd_add_fwd_f32_n3884_0_id);

    /*************
     * g_0_model_dense_1_MatMul_gemm_n3885_0 node
     * inputs:
     *     g_0_t6060_model_dense_BiasAdd_0[10, 2] (dtype=float32)
     *     g_0_t6056_model_dense_1_matmul_readvariableop_0[1, 10] (dtype=float32)
     * outputs:
     *     g_0_t6063_model_dense_1_MatMul_0[1, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t6056_model_dense_1_matmul_readvariableop_0 tensor
    unsigned g_0_t6056_model_dense_1_matmul_readvariableop_0_max_sizes[] = {1, 10};
    unsigned g_0_t6056_model_dense_1_matmul_readvariableop_0_min_sizes[] = {1, 10};
    unsigned g_0_t6056_model_dense_1_matmul_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6056_model_dense_1_matmul_readvariableop_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t6056_model_dense_1_matmul_readvariableop_0_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6056_model_dense_1_matmul_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6063_model_dense_1_MatMul_0 tensor
    unsigned      g_0_t6063_model_dense_1_MatMul_0_max_sizes[] = {1, 2};
    unsigned      g_0_t6063_model_dense_1_MatMul_0_min_sizes[] = {1, 2};
    unsigned      g_0_t6063_model_dense_1_MatMul_0             = createTensors(1,
                                                              OUTPUT_TENSOR,
                                                              true,
                                                              "g_0_t6063_model_dense_1_MatMul_0",
                                                              MEM_INIT_ALL_ZERO,
                                                              nullptr,
                                                              g_0_t6063_model_dense_1_MatMul_0_max_sizes,
                                                              2,
                                                              syn_type_single,
                                                              nullptr,
                                                              0,
                                                              0,
                                                              nullptr,
                                                              false,
                                                              g_0_t6063_model_dense_1_MatMul_0_min_sizes,
                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_dense_1_MatMul_gemm_n3885_0_id;
    unsigned char g_0_model_dense_1_MatMul_gemm_n3885_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t6060_model_dense_BiasAdd_0, g_0_t6056_model_dense_1_matmul_readvariableop_0},
                   {g_0_t6063_model_dense_1_MatMul_0},
                   (void*)g_0_model_dense_1_MatMul_gemm_n3885_0_params,
                   2,
                   "g_0_model_dense_1_MatMul_gemm_n3885_0",
                   0 /*graphIndex*/,
                   &g_0_model_dense_1_MatMul_gemm_n3885_0_id);

    /*************
     * g_0_gradient_tape_model_dense_1_MatMul_MatMul_gemm_n3893_0 node
     * inputs:
     *     g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0[1, 2] (dtype=float32)
     *     g_0_t6056_model_dense_1_matmul_readvariableop_0[1, 10] (dtype=float32)
     * outputs:
     *     g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0[10, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0 tensor
    unsigned g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0_max_sizes[] = {1, 2};
    unsigned g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0_min_sizes[] = {1, 2};
    unsigned g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      true,
                      g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0 tensor
    unsigned g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0_max_sizes[] = {10, 2};
    unsigned g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0_min_sizes[] = {10, 2};
    unsigned g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_model_dense_1_MatMul_MatMul_gemm_n3893_0_id;
    unsigned char g_0_gradient_tape_model_dense_1_MatMul_MatMul_gemm_n3893_0_params[] = {0, 1};
    addNodeToGraph(
        "gemm",
        {g_0_t6076_gradient_tape_model_tf_math_reduce_mean_truediv_0, g_0_t6056_model_dense_1_matmul_readvariableop_0},
        {g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0},
        (void*)g_0_gradient_tape_model_dense_1_MatMul_MatMul_gemm_n3893_0_params,
        2,
        "g_0_gradient_tape_model_dense_1_MatMul_MatMul_gemm_n3893_0",
        0 /*graphIndex*/,
        &g_0_gradient_tape_model_dense_1_MatMul_MatMul_gemm_n3893_0_id);

    /*************
     * g_0_gradient_tape_model_dense_MatMul_MatMul_gemm_n3894_0 node
     * inputs:
     *     g_0_t6054_iteratorgetnext_0[10, 2] (dtype=float32)
     *     g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0[10, 2] (dtype=float32)
     * outputs:
     *     g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0[10, 10] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0 tensor
    unsigned g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0_max_sizes[] = {10, 10};
    unsigned g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0_min_sizes[] = {10, 10};
    unsigned g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_model_dense_MatMul_MatMul_gemm_n3894_0_id;
    unsigned char g_0_gradient_tape_model_dense_MatMul_MatMul_gemm_n3894_0_params[] = {1, 0};
    addNodeToGraph("gemm",
                   {g_0_t6054_iteratorgetnext_0, g_0_t6077_gradient_tape_model_dense_1_MatMul_MatMul_0},
                   {g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0},
                   (void*)g_0_gradient_tape_model_dense_MatMul_MatMul_gemm_n3894_0_params,
                   2,
                   "g_0_gradient_tape_model_dense_MatMul_MatMul_gemm_n3894_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_tape_model_dense_MatMul_MatMul_gemm_n3894_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_t6078_gradient_tape_model_dense_MatMul_MatMul_0, g_0_t6063_model_dense_1_MatMul_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, shared_mme_creates_cycle)
{
    // Graph #0

    /*************
     * g_0_g_0_MatMul_batch_gemm_n16_0_0 node
     * inputs:
     *     n16_in0[2, 3, 3] (dtype=float32)
     *     n16_in1[4, 2, 3] (dtype=float32)
     * outputs:
     *     g_0_g_0_t31_MatMul_0[4, 3, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create n16_in0 tensor
    unsigned n16_in0_max_sizes[] = {2, 3, 3};
    unsigned n16_in0_min_sizes[] = {2, 3, 3};
    unsigned n16_in0             = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "n16_in0",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     n16_in0_max_sizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     n16_in0_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];

    // create n16_in1 tensor
    unsigned n16_in1_max_sizes[] = {4, 2, 3};
    unsigned n16_in1_min_sizes[] = {4, 2, 3};
    unsigned n16_in1             = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "n16_in1",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     n16_in1_max_sizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     n16_in1_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_g_0_t31_MatMul_0 tensor
    unsigned      g_0_g_0_t31_MatMul_0_max_sizes[] = {4, 3, 3};
    unsigned      g_0_g_0_t31_MatMul_0_min_sizes[] = {4, 3, 3};
    unsigned      g_0_g_0_t31_MatMul_0             = createTensors(1,
                                                  OUTPUT_TENSOR,
                                                  false,
                                                  "g_0_g_0_t31_MatMul_0",
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  g_0_g_0_t31_MatMul_0_max_sizes,
                                                  3,
                                                  syn_type_single,
                                                  nullptr,
                                                  0,
                                                  0,
                                                  nullptr,
                                                  false,
                                                  g_0_g_0_t31_MatMul_0_min_sizes,
                                                  synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_g_0_MatMul_batch_gemm_n16_0_0_id;
    unsigned char g_0_g_0_MatMul_batch_gemm_n16_0_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {n16_in0, n16_in1},
                   {g_0_g_0_t31_MatMul_0},
                   (void*)g_0_g_0_MatMul_batch_gemm_n16_0_0_params,
                   2,
                   "g_0_g_0_MatMul_batch_gemm_n16_0_0",
                   0 /*graphIndex*/,
                   &g_0_g_0_MatMul_batch_gemm_n16_0_0_id);

    /*************
     * g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_1_MatMul_batch_gemm_n22_0_0
     *node inputs: n16_in0[2, 3, 3] (dtype=float32) n22_in1[1, 2, 3] (dtype=float32) outputs: n22_out[1, 3, 3]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create n22_in1 tensor
    unsigned n22_in1_max_sizes[] = {1, 2, 3};
    unsigned n22_in1_min_sizes[] = {1, 2, 3};
    unsigned n22_in1             = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "n22_in1",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     n22_in1_max_sizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     n22_in1_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];

    // create n22_out tensor
    unsigned n22_out_max_sizes[] = {1, 3, 3};
    unsigned n22_out_min_sizes[] = {1, 3, 3};
    unsigned n22_out             = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "n22_out",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     n22_out_max_sizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     n22_out_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_1_MatMul_batch_gemm_n22_0_0_id;
    unsigned char
        g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_1_MatMul_batch_gemm_n22_0_0_params
            [] = {0, 0};
    addNodeToGraph(
        "batch_gemm",
        {n16_in0, n22_in1},
        {n22_out},
        (void*)
            g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_1_MatMul_batch_gemm_n22_0_0_params,
        2,
        "g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_1_MatMul_"
        "batch_gemm_n22_0_0",
        0 /*graphIndex*/,
        &g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_1_MatMul_batch_gemm_n22_0_0_id);

    /*************
     * g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_MatMul_batch_gemm_n21_0_0
     *node inputs: n16_in1[4, 2, 3] (dtype=float32) n21_in1[1, 4, 3] (dtype=float32) outputs: n22_in1[1, 2, 3]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create n21_in1 tensor
    unsigned n21_in1_max_sizes[] = {1, 4, 3};
    unsigned n21_in1_min_sizes[] = {1, 4, 3};
    unsigned n21_in1             = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "n21_in1",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     n21_in1_max_sizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     n21_in1_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_MatMul_batch_gemm_n21_0_0_id;
    unsigned char
        g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_MatMul_batch_gemm_n21_0_0_params
            [] = {0, 0};
    addNodeToGraph(
        "batch_gemm",
        {n16_in1, n21_in1},
        {n22_in1},
        (void*)
            g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_MatMul_batch_gemm_n21_0_0_params,
        2,
        "g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_MatMul_"
        "batch_gemm_n21_0_0",
        0 /*graphIndex*/,
        &g_0_g_0_LinearOperatorFullMatrix_o_LinearOperatorFullMatrix_matmul_LinearOperatorFullMatrix_matmul_MatMul_batch_gemm_n21_0_0_id);

    /*************
     * g_0_g_0_MatMul_1_batch_gemm_n23_0_0 node
     * inputs:
     *     g_0_g_0_t31_MatMul_0[4, 3, 3] (dtype=float32)
     *     n21_in1[1, 4, 3] (dtype=float32)
     * outputs:
     *     g_0_g_0_t40_MatMul_1_0[1, 3, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_g_0_t40_MatMul_1_0 tensor
    unsigned      g_0_g_0_t40_MatMul_1_0_max_sizes[] = {1, 3, 3};
    unsigned      g_0_g_0_t40_MatMul_1_0_min_sizes[] = {1, 3, 3};
    unsigned      g_0_g_0_t40_MatMul_1_0             = createTensors(1,
                                                    OUTPUT_TENSOR,
                                                    true,
                                                    "g_0_g_0_t40_MatMul_1_0",
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    g_0_g_0_t40_MatMul_1_0_max_sizes,
                                                    3,
                                                    syn_type_single,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    g_0_g_0_t40_MatMul_1_0_min_sizes,
                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_g_0_MatMul_1_batch_gemm_n23_0_0_id;
    unsigned char g_0_g_0_MatMul_1_batch_gemm_n23_0_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_g_0_t31_MatMul_0, n21_in1},
                   {g_0_g_0_t40_MatMul_1_0},
                   (void*)g_0_g_0_MatMul_1_batch_gemm_n23_0_0_params,
                   2,
                   "g_0_g_0_MatMul_1_batch_gemm_n23_0_0",
                   0 /*graphIndex*/,
                   &g_0_g_0_MatMul_1_batch_gemm_n23_0_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_g_0_t40_MatMul_1_0, n22_out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, shared_mme_creates_cycle2)
{
    // Graph #0

    /*************
     * gemm_n19 node
     * inputs:
     *     g_t39[2, 2] (dtype=float32)
     *     g_t40_[1, 2] (dtype=float32)
     * outputs:
     *     g_t42[1, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_t39 tensor
    unsigned g_t39_max_sizes[] = {2, 2};
    unsigned g_t39_min_sizes[] = {2, 2};
    unsigned g_t39             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "g_t39",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   g_t39_max_sizes,
                                   2,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_t39_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create g_t40_ tensor
    unsigned g_t40_max_sizes[] = {1, 2};
    unsigned g_t40_min_sizes[] = {1, 2};
    unsigned g_t40_            = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "g_t40_",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    g_t40_max_sizes,
                                    2,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    g_t40_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];

    // create g_t42 tensor
    unsigned      g_t42_max_sizes[] = {1, 2};
    unsigned      g_t42_min_sizes[] = {1, 2};
    unsigned      g_t42             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,
                                   "g_t42",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   g_t42_max_sizes,
                                   2,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_t42_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     gemm_n19_id;
    unsigned char gemm_n19_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_t39, g_t40_},
                   {g_t42},
                   (void*)gemm_n19_params,
                   2,
                   "gemm_n19",
                   0 /*graphIndex*/,
                   &gemm_n19_id);

    /*************
     * g_add_fwd_f32_n25 node
     * inputs:
     *     g_t51[1, 1] (dtype=float32)
     *     g_t47_mul[2, 1] (dtype=float32)
     * outputs:
     *     g_t50[2, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_t51 tensor
    unsigned g_t51_max_sizes[] = {1, 1};
    unsigned g_t51_min_sizes[] = {1, 1};
    unsigned g_t51             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "g_t51",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   g_t51_max_sizes,
                                   2,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_t51_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create g_t47_mul tensor
    unsigned g_t47_mul_max_sizes[] = {2, 1};
    unsigned g_t47_mul_min_sizes[] = {2, 1};
    unsigned g_t47_mul             = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "g_t47_mul",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       g_t47_mul_max_sizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_t47_mul_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_t50 tensor
    unsigned  g_t50_max_sizes[] = {2, 1};
    unsigned  g_t50_min_sizes[] = {2, 1};
    unsigned  g_t50             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   false,
                                   "g_t50",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   g_t50_max_sizes,
                                   2,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_t50_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_add_fwd_f32_n25_id;
    addNodeToGraph("add_fwd_f32",
                   {g_t51, g_t47_mul},
                   {g_t50},
                   nullptr,
                   0,
                   "g_add_fwd_f32_n25",
                   0 /*graphIndex*/,
                   &g_add_fwd_f32_n25_id);

    /*************
     * gemm_n26 node
     * inputs:
     *     g_t40_[1, 2] (dtype=float32)
     *     g_t50[2, 1] (dtype=float32)
     * outputs:
     *     g_t53_[2, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_t53_ tensor
    unsigned      g_t53_max_sizes[] = {2, 2};
    unsigned      g_t53_min_sizes[] = {2, 2};
    unsigned      g_t53_            = createTensors(1,
                                    OUTPUT_TENSOR,
                                    true,
                                    "g_t53_",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    g_t53_max_sizes,
                                    2,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    g_t53_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     gemm_n26_id;
    unsigned char gemm_n26_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_t40_, g_t50},
                   {g_t53_},
                   (void*)gemm_n26_params,
                   2,
                   "gemm_n26",
                   0 /*graphIndex*/,
                   &gemm_n26_id);

    /*************
     * gemm_n27 node
     * inputs:
     *     g_t39[2, 2] (dtype=float32)
     *     g_t53_[2, 2] (dtype=float32)
     * outputs:
     *     g_t54[2, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_t54 tensor
    unsigned      g_t54_max_sizes[] = {2, 2};
    unsigned      g_t54_min_sizes[] = {2, 2};
    unsigned      g_t54             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,
                                   "g_t54",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   g_t54_max_sizes,
                                   2,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_t54_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     gemm_n27_id;
    unsigned char gemm_n27_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_t39, g_t53_},
                   {g_t54},
                   (void*)gemm_n27_params,
                   2,
                   "gemm_n27",
                   0 /*graphIndex*/,
                   &gemm_n27_id);

    /*************
     * 1_gemm_n29 node
     * inputs:
     *     g_t42[1, 2] (dtype=float32)
     *     g_t50[2, 1] (dtype=float32)
     * outputs:
     *     g_t56_1[2, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_t56_1 tensor
    unsigned      g_t56_1_max_sizes[] = {2, 2};
    unsigned      g_t56_1_min_sizes[] = {2, 2};
    unsigned      g_t56_1             = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "g_t56_1",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     g_t56_1_max_sizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     g_t56_1_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId     gemm_n29_id;
    unsigned char gemm_n29_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_t42, g_t50},
                   {g_t56_1},
                   (void*)gemm_n29_params,
                   2,
                   "1_gemm_n29",
                   0 /*graphIndex*/,
                   &gemm_n29_id);

    /*************
     * g_gemm_n30 node
     * inputs:
     *     g_t40_[1, 2] (dtype=float32)
     *     g_t50[2, 1] (dtype=float32)
     * outputs:
     *     g_t57[2, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_t57 tensor
    unsigned      g_t57_max_sizes[] = {2, 2};
    unsigned      g_t57_min_sizes[] = {2, 2};
    unsigned      g_t57             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,
                                   "g_t57",
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   g_t57_max_sizes,
                                   2,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_t57_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     g_gemm_n30_id;
    unsigned char g_gemm_n30_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_t40_, g_t50},
                   {g_t57},
                   (void*)g_gemm_n30_params,
                   2,
                   "g_gemm_n30",
                   0 /*graphIndex*/,
                   &g_gemm_n30_id);

    /*************
     * g_1_gemm_n31 node
     * inputs:
     *     g_t39[2, 2] (dtype=float32)
     *     g_t57[2, 2] (dtype=float32)
     * outputs:
     *     g_t58_1[2, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_t58_1 tensor
    unsigned      g_t58_1_max_sizes[] = {2, 2};
    unsigned      g_t58_1_min_sizes[] = {2, 2};
    unsigned      g_t58_1             = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "g_t58_1",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     g_t58_1_max_sizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     g_t58_1_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId     g_1_gemm_n31_id;
    unsigned char g_1_gemm_n31_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_t39, g_t57},
                   {g_t58_1},
                   (void*)g_1_gemm_n31_params,
                   2,
                   "g_1_gemm_n31",
                   0 /*graphIndex*/,
                   &g_1_gemm_n31_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_t42, g_t53_, g_t54, g_t56_1, g_t57, g_t58_1});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, single_buffer_strategy_sram_alloc_ASIC_CI)
{
    unsigned addIn0Sizes[] = {96, 256, 256, 8};
    unsigned addIn0        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "addIn0",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    addIn0Sizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    addIn0Sizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned addIn1Sizes[] = {96, 256, 256, 8};
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

    unsigned addOutSizes[] = {96, 256, 256, 8};
    unsigned addOut        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "addOut",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
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

    addNodeToGraph("add_fwd_f32", {addIn0, addIn1}, {addOut}, nullptr, 0, "ADD");

    unsigned dedwInSizes[] = {16, 256, 256, 8};
    unsigned dedwIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "dedwIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dedwInSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedwInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned dedwOutSizes[] = {96, 16, 1, 1};
    unsigned dedwOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "dedwOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     dedwOutSizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedwOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned char dedwParams[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 242, 193, 1, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw", {addOut, dedwIn}, {dedwOut}, (void*)dedwParams, 104, "DEDW");

    unsigned dedxInSizes[] = {96, 16, 1, 1};
    unsigned dedxIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "dedxIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dedxInSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    dedxInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned dedxShapeSizes[] = {16, 256, 256, 8};
    unsigned dedxShape        = createTensors(1,
                                       INPUT_TENSOR,
                                       false,
                                       "dedxShape",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       dedxShapeSizes,
                                       4,
                                       syn_type_uint32,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       dedxShapeSizes,
                                       synTensorType::SHAPE_TENSOR)[0];

    unsigned      dedxOutSizes[] = {16, 256, 256, 8};
    unsigned      dedxOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "dedxOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     dedxOutSizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     dedxOutSizes,
                                     synTensorType::DATA_TENSOR)[0];
    unsigned char dedxParams[]   = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx", {addOut, dedxIn, dedxShape}, {dedxOut}, (void*)dedxParams, 104, "DEDX");

    unsigned multInSizes[] = {16, 256, 256, 8};
    unsigned multIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "multIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    multInSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    multInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned multOutSizes[] = {16, 256, 256, 8};
    unsigned multOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "multOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     multOutSizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     multOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("mult_fwd_f32", {multIn, dedxOut}, {multOut}, nullptr, 0, "MULT");

    unsigned reshapeShapeSizes[] = {16, 524288, 1, 1};
    unsigned reshapeShape        = createTensors(1,
                                          INPUT_TENSOR,
                                          false,
                                          "reshapeShape",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          reshapeShapeSizes,
                                          4,
                                          syn_type_uint32,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          reshapeShapeSizes,
                                          synTensorType::SHAPE_TENSOR)[0];

    unsigned reshapeOutSizes[] = {16, 524288, 1, 1};
    unsigned reshapeOut        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true,
                                        "reshapeOut",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        reshapeOutSizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        reshapeOutSizes,
                                        synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("reshape", {dedxOut, reshapeShape}, {reshapeOut}, nullptr, 0, "RESHAPE");

    unsigned memcpyOutSizes[] = {16, 524288, 1, 1};
    unsigned memcpyOut        = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "memcpyOut",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       memcpyOutSizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       memcpyOutSizes,
                                       synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("memcpy", {reshapeOut}, {memcpyOut}, nullptr, 0, "MEMCOPY");

    unsigned reshape2OutSizes[] = {524288, 16};
    unsigned reshape2Out        = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "reshape2Out",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         reshape2OutSizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         reshape2OutSizes,
                                         synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("reshape", {memcpyOut}, {reshape2Out}, nullptr, 0, "RESHAPE2");

    unsigned reluInSizes[] = {1024, 524288};
    unsigned reluIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "reluIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    reluInSizes,
                                    2,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    reluInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned gemmInSizes[] = {1024, 524288};
    unsigned gemmIn        = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false,
                                    "gemmIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    gemmInSizes,
                                    2,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    gemmInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_f32", {reluIn}, {gemmIn}, nullptr, 0, "RELU");

    unsigned gemmOutSizes[] = {1024, 16};
    unsigned gemmOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "gemmOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemmOutSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     gemmOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synGEMMParams gemmParams {};
    addNodeToGraph("gemm", {reshape2Out, gemmIn}, {gemmOut}, &gemmParams, sizeof(gemmParams), "GEMM");

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({multOut, reshapeOut, memcpyOut, gemmOut, dedwOut});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, correct_slicing_of_bgemm_without_any_producers_and_large_input_cannot_be_sliced)
{
    // Graph #0

    /*************
     * g_0_n840__hpu_matmul_backward_batch_gemm_0 node
     * inputs:
     *     g_0_tensor_2289_t8448__0_1[2, 3, 64] (dtype=bf16)
     *     g_0_tensor_2294[1024, 2] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2295[1024, 3, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2289_t8448__0_1 tensor
    unsigned g_0_tensor_2289_t8448__0_1_max_sizes[] = {2, 3, 64};
    unsigned g_0_tensor_2289_t8448__0_1_min_sizes[] = {2, 3, 64};
    unsigned g_0_tensor_2289_t8448__0_1             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_tensor_2289_t8448__0_1",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_tensor_2289_t8448__0_1_max_sizes,
                                                        3,
                                                        syn_type_bf16,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_tensor_2289_t8448__0_1_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2294 tensor
    unsigned g_0_tensor_2294_max_sizes[] = {1024, 2};
    unsigned g_0_tensor_2294_min_sizes[] = {1024, 2};
    unsigned g_0_tensor_2294             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_2294",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_2294_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2294_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2295 tensor
    unsigned      g_0_tensor_2295_max_sizes[] = {1024, 3, 64};
    unsigned      g_0_tensor_2295_min_sizes[] = {1024, 3, 64};
    unsigned      g_0_tensor_2295             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_2295",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_tensor_2295_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2295_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_n840__hpu_matmul_backward_batch_gemm_0_id;
    unsigned char g_0_n840__hpu_matmul_backward_batch_gemm_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_2289_t8448__0_1, g_0_tensor_2294},
                   {g_0_tensor_2295},
                   (void*)g_0_n840__hpu_matmul_backward_batch_gemm_0_params,
                   2,
                   "g_0_n840__hpu_matmul_backward_batch_gemm_0",
                   0 /*graphIndex*/,
                   &g_0_n840__hpu_matmul_backward_batch_gemm_0_id);

    // no slicing for 1st run (baseline)
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    // pipeline management/sram management enabled for 2nd run
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({g_0_tensor_2295});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, cast_both_inputs_bgemm_512x1x512)
{
    std::vector<unsigned> inputAShape = {1, 512, 16};
    std::vector<unsigned> inputBShape = {512, 1, 16};
    std::vector<unsigned> outputShape = {512, 512, 16};

    auto af32 = createPersistTensor(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    inputAShape.data(),
                                    inputAShape.size(),
                                    syn_type_float,
                                    nullptr,
                                    "a_f32");
    auto abf16 =
        createTensor(INPUT_TENSOR, MEM_INIT_NONE, nullptr, inputAShape.data(), inputAShape.size(), syn_type_bf16);
    auto bf32 = createPersistTensor(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    inputBShape.data(),
                                    inputBShape.size(),
                                    syn_type_float,
                                    nullptr,
                                    "b_f32");
    auto bbf16 =
        createTensor(INPUT_TENSOR, MEM_INIT_NONE, nullptr, inputBShape.data(), inputBShape.size(), syn_type_bf16);

    auto out = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_NONE,
                                   nullptr,
                                   outputShape.data(),
                                   outputShape.size(),
                                   syn_type_bf16,
                                   nullptr,
                                   "out");

    addNodeToGraph("cast_f32_to_bf16", {af32}, {abf16}, nullptr, 0, "cast_a");
    addNodeToGraph("cast_f32_to_bf16", {bf32}, {bbf16}, nullptr, 0, "cast_b");

    synGEMMParams params {};
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {abf16, bbf16}, {out}, &params, sizeof(params), "bgemm");

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, dedx3d_relu_b2_3x3_stride_1_padding_1_ASIC)
{
    std::vector<unsigned> fmShape  = {1536, 32, 2, 16, 2};
    std::vector<unsigned> wghShape = {1536, 1536, 3, 3, 3};

    synConvolution3DParams convParams {};
    convParams.kernel[CONV_KERNEL_WIDTH]  = 3;
    convParams.kernel[CONV_KERNEL_HEIGHT] = 3;
    convParams.kernel[CONV_KERNEL_DEPTH]  = 3;
    convParams.padding[CONV_PAD_LEFT]     = 1;
    convParams.padding[CONV_PAD_RIGHT]    = 1;
    convParams.padding[CONV_PAD_TOP]      = 1;
    convParams.padding[CONV_PAD_BOTTOM]   = 1;
    convParams.padding[CONV_PAD_FRONT]    = 1;
    convParams.padding[CONV_PAD_BACK]     = 1;

    auto dy  = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  fmShape.data(),
                                  fmShape.size(),
                                  syn_type_float,
                                  nullptr,
                                  "DY");
    auto wgh = createPersistTensor(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   wghShape.data(),
                                   wghShape.size(),
                                   syn_type_float,
                                   nullptr,
                                   "WGH");
    auto dx  = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_NONE,
                                  nullptr,
                                  fmShape.data(),
                                  fmShape.size(),
                                  syn_type_float,
                                  nullptr,
                                  "DX");

    addNodeToGraph(NodeFactory::deDx3DNodeTypeName, {dy, wgh}, {dx}, &convParams, sizeof(convParams), "dedx");

    auto reluIn   = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      fmShape.data(),
                                      fmShape.size(),
                                      syn_type_float,
                                      nullptr,
                                      "X");
    auto reluGrad = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_NONE,
                                        nullptr,
                                        fmShape.data(),
                                        fmShape.size(),
                                        syn_type_float,
                                        nullptr,
                                        "ReLU-DX");
    addNodeToGraph("relu_bwd_f32", {dx, reluIn}, {reluGrad}, nullptr, 0, "relu_bwd");

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({dx, reluGrad});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, shared_mme_with_duplicated_operand)
{
    // Graph #0

    /*************
     * mult_node node
     * inputs:
     *     mult_in0[3, 4, 3] (dtype=float32)
     *     mult_in1[1, 1, 1] (dtype=float32)
     * outputs:
     *     mult_out[3, 4, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create mult_in0 tensor
    unsigned mult_in0_max_sizes[] = {3, 4, 3};
    unsigned mult_in0_min_sizes[] = {3, 4, 3};
    unsigned mult_in0             = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "mult_in0",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      mult_in0_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      mult_in0_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create mult_in1 tensor
    unsigned mult_in1_max_sizes[] = {1, 1, 1};
    unsigned mult_in1_min_sizes[] = {1, 1, 1};
    unsigned mult_in1             = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "mult_in1",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      mult_in1_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      mult_in1_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create mult_out tensor
    unsigned  mult_out_max_sizes[] = {3, 4, 3};
    unsigned  mult_out_min_sizes[] = {3, 4, 3};
    unsigned  mult_out             = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "mult_out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      mult_out_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      mult_out_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId mult_node_id;
    addNodeToGraph("mult_fwd_f32",
                   {mult_in0, mult_in1},
                   {mult_out},
                   nullptr,
                   0,
                   "mult_node",
                   0 /*graphIndex*/,
                   &mult_node_id);

    /*************
     * add_node node
     * inputs:
     *     add_in0[1, 1, 1] (dtype=float32)
     *     mult_out[3, 4, 3] (dtype=float32)
     * outputs:
     *     add_out[3, 4, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create add_in0 tensor
    unsigned add_in0_max_sizes[] = {1, 1, 1};
    unsigned add_in0_min_sizes[] = {1, 1, 1};
    unsigned add_in0             = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "add_in0",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     add_in0_max_sizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     add_in0_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];

    // create add_out tensor
    unsigned  add_out_max_sizes[] = {3, 4, 3};
    unsigned  add_out_min_sizes[] = {3, 4, 3};
    unsigned  add_out             = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "add_out",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     add_out_max_sizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     add_out_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    synNodeId add_node_id;
    addNodeToGraph("add_fwd_f32",
                   {add_in0, mult_out},
                   {add_out},
                   nullptr,
                   0,
                   "add_node",
                   0 /*graphIndex*/,
                   &add_node_id);

    /*************
     * g_0_MatMul_batch_gemm_n12_0 node
     * inputs:
     *     add_out[3, 4, 3] (dtype=float32)
     *     add_out[3, 4, 3] (dtype=float32)
     * outputs:
     *     bgemm_n12_out[4, 4, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create bgemm_n12_out tensor
    unsigned      bgemm_n12_out_max_sizes[] = {4, 4, 3};
    unsigned      bgemm_n12_out_min_sizes[] = {4, 4, 3};
    unsigned      bgemm_n12_out             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "bgemm_n12_out",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           bgemm_n12_out_max_sizes,
                                           3,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_n12_out_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_MatMul_batch_gemm_n12_0_id;
    unsigned char g_0_MatMul_batch_gemm_n12_0_params[] = {0, 1};
    addNodeToGraph("batch_gemm",
                   {add_out, add_out},
                   {bgemm_n12_out},
                   (void*)g_0_MatMul_batch_gemm_n12_0_params,
                   2,
                   "g_0_MatMul_batch_gemm_n12_0",
                   0 /*graphIndex*/,
                   &g_0_MatMul_batch_gemm_n12_0_id);

    /*************
     * g_0_LinearOperatorLowRankUpdate_add_to_tensor_LinearOperatorLowRankUpdate_to_dense_LinearOperatorLowRankUpdate_matmul_MatMul_batch_gemm_n16_0
     *node inputs: add_out[3, 4, 3] (dtype=float32) bgemm_n16_in1[4, 4, 3] (dtype=float32) outputs: bgemm_n16_out[4, 3,
     *3] (dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create bgemm_n16_in1 tensor
    unsigned bgemm_n16_in1_max_sizes[] = {4, 4, 3};
    unsigned bgemm_n16_in1_min_sizes[] = {4, 4, 3};
    unsigned bgemm_n16_in1             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "bgemm_n16_in1",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           bgemm_n16_in1_max_sizes,
                                           3,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           true,
                                           bgemm_n16_in1_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create bgemm_n16_out tensor
    unsigned bgemm_n16_out_max_sizes[] = {4, 3, 3};
    unsigned bgemm_n16_out_min_sizes[] = {4, 3, 3};
    unsigned bgemm_n16_out             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "bgemm_n16_out",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           bgemm_n16_out_max_sizes,
                                           3,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_n16_out_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_LinearOperatorLowRankUpdate_add_to_tensor_LinearOperatorLowRankUpdate_to_dense_LinearOperatorLowRankUpdate_matmul_MatMul_batch_gemm_n16_0_id;
    unsigned char
        g_0_LinearOperatorLowRankUpdate_add_to_tensor_LinearOperatorLowRankUpdate_to_dense_LinearOperatorLowRankUpdate_matmul_MatMul_batch_gemm_n16_0_params
            [] = {1, 0};
    addNodeToGraph(
        "batch_gemm",
        {add_out, bgemm_n16_in1},
        {bgemm_n16_out},
        (void*)
            g_0_LinearOperatorLowRankUpdate_add_to_tensor_LinearOperatorLowRankUpdate_to_dense_LinearOperatorLowRankUpdate_matmul_MatMul_batch_gemm_n16_0_params,
        2,
        "g_0_LinearOperatorLowRankUpdate_add_to_tensor_LinearOperatorLowRankUpdate_to_dense_"
        "LinearOperatorLowRankUpdate_matmul_MatMul_batch_gemm_n16_0",
        0 /*graphIndex*/,
        &g_0_LinearOperatorLowRankUpdate_add_to_tensor_LinearOperatorLowRankUpdate_to_dense_LinearOperatorLowRankUpdate_matmul_MatMul_batch_gemm_n16_0_id);

    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    compareRunsResults({bgemm_n16_out, bgemm_n12_out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, dedw_3d_b1_3x3_stride_1_padding_1_ASIC_CI)
{
    std::vector<unsigned> inputsShape = {384, 64, 4, 32, 1};
    std::vector<unsigned> dedwShape   = {384, 384, 3, 3, 3};

    synConvolution3DParams convParams {};
    convParams.kernel[CONV_KERNEL_WIDTH]  = 3;
    convParams.kernel[CONV_KERNEL_HEIGHT] = 3;
    convParams.kernel[CONV_KERNEL_DEPTH]  = 3;
    convParams.padding[CONV_PAD_LEFT]     = 1;
    convParams.padding[CONV_PAD_RIGHT]    = 1;
    convParams.padding[CONV_PAD_TOP]      = 1;
    convParams.padding[CONV_PAD_BOTTOM]   = 1;
    convParams.padding[CONV_PAD_FRONT]    = 1;
    convParams.padding[CONV_PAD_BACK]     = 1;

    auto x  = createPersistTensor(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 inputsShape.data(),
                                 inputsShape.size(),
                                 syn_type_float,
                                 nullptr,
                                 "X");
    auto dy = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  inputsShape.data(),
                                  inputsShape.size(),
                                  syn_type_float,
                                  nullptr,
                                  "DY");
    auto dw = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_NONE,
                                  nullptr,
                                  dedwShape.data(),
                                  dedwShape.size(),
                                  syn_type_float,
                                  nullptr,
                                  "DW");

    addNodeToGraph(NodeFactory::deDw3DNodeTypeName, {dy, x}, {dw}, &convParams, sizeof(convParams));

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({dw});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, tpc_bundle_with_negative_address_slices_ASIC_CI)
{
    std::vector<unsigned> fmSizes = {1024, 32 * 1024};
    std::vector<unsigned> wSizes  = {1024, 1024};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      fmSizes.data(),
                                      fmSizes.size(),
                                      syn_type_float,
                                      nullptr,
                                      "IN");
    unsigned w  = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     wSizes.data(),
                                     wSizes.size(),
                                     syn_type_float,
                                     nullptr,
                                     "W");
    unsigned intermediate =
        createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, fmSizes.data(), fmSizes.size(), syn_type_float);

    synGEMMParams gemmParams {};
    addNodeToGraph("gemm", {in, w}, {intermediate}, &gemmParams, sizeof(gemmParams), "GEMM_IN");

    for (int i = 0; i < 10; i++)
    {
        unsigned nextIntermediate =
            createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, fmSizes.data(), fmSizes.size(), syn_type_float);
        const std::string name = "RELU" + std::to_string(i);
        addNodeToGraph("relu_fwd_f32", {intermediate}, {nextIntermediate}, nullptr, 0, name.c_str());
        intermediate = nextIntermediate;
    }

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_NONE,
                                       nullptr,
                                       fmSizes.data(),
                                       fmSizes.size(),
                                       syn_type_float,
                                       nullptr,
                                       "OUT");

    addNodeToGraph("gemm", {intermediate, w}, {out}, &gemmParams, sizeof(gemmParams), "GEMM_OUT");

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    addConfigurationToRun(SECOND_RUN, "RUN_TPC_FUSER", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_HBM_SLICES_ALLOCATION_OPTIMIZATION", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS", "false");

    compareRunsResults({out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, tpc_producers_chain_with_2_shared_tensors)
{
    // Graph #0

    /*************
     * g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_softmax_stage1_311_complex_softmax_stage1_299_0
     *node inputs: g_0_t1110_bert_encoder_layer_0_attention_self_add_0[512, 512, 16, 125] (dtype=bf16) outputs:
     *     g_0_complex_softmax_stg1_exp_out_629[512, 512, 16, 125] (dtype=bf16)
     *     g_0_complex_softmax_stg1_norm_out_630[512, 1, 16, 125] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/
    const unsigned batch = 8;  // originally was 125, but smaller value is enough to test the feature

    // create g_0_t1110_bert_encoder_layer_0_attention_self_add_0 tensor
    unsigned g_0_t1110_bert_encoder_layer_0_attention_self_add_0_max_sizes[] = {512, 512, 16, batch};
    unsigned g_0_t1110_bert_encoder_layer_0_attention_self_add_0_min_sizes[] = {512, 512, 16, batch};
    unsigned g_0_t1110_bert_encoder_layer_0_attention_self_add_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1110_bert_encoder_layer_0_attention_self_add_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1110_bert_encoder_layer_0_attention_self_add_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1110_bert_encoder_layer_0_attention_self_add_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_complex_softmax_stg1_exp_out_629 tensor
    unsigned g_0_complex_softmax_stg1_exp_out_629_max_sizes[] = {512, 512, 16, batch};
    unsigned g_0_complex_softmax_stg1_exp_out_629_min_sizes[] = {512, 512, 16, batch};
    unsigned g_0_complex_softmax_stg1_exp_out_629             = createTensors(1,
                                                                  OUTPUT_TENSOR,
                                                                  false,
                                                                  "g_0_complex_softmax_stg1_exp_out_629",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_complex_softmax_stg1_exp_out_629_max_sizes,
                                                                  4,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_complex_softmax_stg1_exp_out_629_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    // create g_0_complex_softmax_stg1_norm_out_630 tensor
    unsigned g_0_complex_softmax_stg1_norm_out_630_max_sizes[] = {512, 1, 16, batch};
    unsigned g_0_complex_softmax_stg1_norm_out_630_min_sizes[] = {512, 1, 16, batch};
    unsigned g_0_complex_softmax_stg1_norm_out_630             = createTensors(1,
                                                                   OUTPUT_TENSOR,
                                                                   false,
                                                                   "g_0_complex_softmax_stg1_norm_out_630",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_complex_softmax_stg1_norm_out_630_max_sizes,
                                                                   4,
                                                                   syn_type_single,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_complex_softmax_stg1_norm_out_630_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_softmax_stage1_311_complex_softmax_stage1_299_0_id;
    unsigned char
        g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_softmax_stage1_311_complex_softmax_stage1_299_0_params
            [] = {1, 0, 0, 0, 1, 0, 0, 0};
    addNodeToGraph(
        "softmax_stage1_fwd_bf16",
        {g_0_t1110_bert_encoder_layer_0_attention_self_add_0},
        {g_0_complex_softmax_stg1_exp_out_629, g_0_complex_softmax_stg1_norm_out_630},
        (void*)
            g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_softmax_stage1_311_complex_softmax_stage1_299_0_params,
        8,
        "g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_softmax_stage1_311_"
        "complex_softmax_stage1_299_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_softmax_stage1_311_complex_softmax_stage1_299_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_mult_fused_stage2_312_0 node
     * inputs:
     *     g_0_complex_softmax_stg1_exp_out_629[512, 512, 16, 125] (dtype=bf16)
     *     g_0_complex_softmax_stg1_norm_out_630[512, 1, 16, 125] (dtype=float32)
     * outputs:
     *     g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0[512, 512, 16, 125] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0 tensor
    unsigned g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0_max_sizes[] = {512, 512, 16, batch};
    unsigned g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0_min_sizes[] = {512, 512, 16, batch};
    unsigned g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_mult_fused_stage2_312_0_id;
    unsigned char
        g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_mult_fused_stage2_312_0_params
            [] = {1, 0, 0, 0};
    addNodeToGraph(
        "mult_swizzled_fwd_bf16",
        {g_0_complex_softmax_stg1_exp_out_629, g_0_complex_softmax_stg1_norm_out_630},
        {g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0},
        (void*)
            g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_mult_fused_stage2_312_0_params,
        4,
        "g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_mult_fused_stage2_312_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_attention_self_HabanaSoftmax_softmax_fwd_bf16_n360_complex_mult_fused_stage2_312_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_MatMul_1_batch_gemm_n361_0 node
     * inputs:
     *     g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0[512, 512, 16, 125] (dtype=bf16)
     *     g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0[64, 512, 16, 125] (dtype=bf16)
     * outputs:
     *     g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0[64, 512, 16, 125] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0 tensor
    unsigned g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0_max_sizes[] = {64, 512, 16, batch};
    unsigned g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0_min_sizes[] = {64, 512, 16, batch};
    unsigned g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0 tensor
    unsigned g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0_max_sizes[] = {64, 512, 16, batch};
    unsigned g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0_min_sizes[] = {64, 512, 16, batch};
    unsigned g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_layer_0_attention_self_MatMul_1_batch_gemm_n361_0_id;
    unsigned char g_0_bert_encoder_layer_0_attention_self_MatMul_1_batch_gemm_n361_0_params[] = {1, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_t1111_bert_encoder_layer_0_attention_self_HabanaSoftmax_0,
                    g_0_t1095_bert_encoder_layer_0_attention_self_transpose_2_0},
                   {g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0},
                   (void*)g_0_bert_encoder_layer_0_attention_self_MatMul_1_batch_gemm_n361_0_params,
                   2,
                   "g_0_bert_encoder_layer_0_attention_self_MatMul_1_batch_gemm_n361_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_attention_self_MatMul_1_batch_gemm_n361_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_EXPERIMENTAL_FLAGS", "true");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "ENABLE_EXPERIMENTAL_FLAGS", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");

    compareRunsResults({g_0_t1112_bert_encoder_layer_0_attention_self_MatMul_1_0});
}

// TODO SW-140050 - reenable the test using regular BN guid (as cud_bn has been removed)
TEST_F_GC(SynGaudiTwoRunCompareTest, DISABLED_bundle_consumer_with_2_same_input_tensors)
{
    // Graph #0

    /*************
     * g_0_gradient_features_18_0_dedw_0 node
     * inputs:
     *     g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward[1280, 7, 7, 256] (dtype=bf16)
     *     g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv[320, 7, 7, 256] (dtype=bf16)
     * outputs:
     *     g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable[1280, 320, 1, 1]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward tensor
    unsigned g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward_max_sizes[] = {1280, 7, 7, 256};
    unsigned g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward_min_sizes[] = {1280, 7, 7, 256};
    unsigned g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv tensor
    unsigned g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv_max_sizes[] = {320, 7, 7, 256};
    unsigned g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv_min_sizes[] = {320, 7, 7, 256};
    unsigned g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable tensor
    unsigned g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable_max_sizes[] = {1280,
                                                                                                              320,
                                                                                                              1,
                                                                                                              1};
    unsigned g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable_min_sizes[] = {1280,
                                                                                                              320,
                                                                                                              1,
                                                                                                              1};
    unsigned g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_features_18_0_dedw_0_id;
    unsigned char g_0_gradient_features_18_0_dedw_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward,
                    g_0_tensor_806_3950_features_17_conv_3_hpu_native_batch_norm_rmv},
                   {g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable},
                   (void*)g_0_gradient_features_18_0_dedw_0_params,
                   104,
                   "g_0_gradient_features_18_0_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_features_18_0_dedw_0_id);

    /*************
     * g_0_gradient_features_18_0_dedx_0 node
     * inputs:
     *     g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward[1280, 7, 7, 256] (dtype=bf16)
     *     g_0_tensor_814_3958_features_18_0_hpu_cast[1280, 320, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable[320, 7, 7, 256]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_814_3958_features_18_0_hpu_cast tensor
    unsigned g_0_tensor_814_3958_features_18_0_hpu_cast_max_sizes[] = {1280, 320, 1, 1};
    unsigned g_0_tensor_814_3958_features_18_0_hpu_cast_min_sizes[] = {1280, 320, 1, 1};
    unsigned g_0_tensor_814_3958_features_18_0_hpu_cast =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_814_3958_features_18_0_hpu_cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_814_3958_features_18_0_hpu_cast_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_814_3958_features_18_0_hpu_cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable tensor
    unsigned g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable_max_sizes[] = {320,
                                                                                                              7,
                                                                                                              7,
                                                                                                              256};
    unsigned g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable_min_sizes[] = {320,
                                                                                                              7,
                                                                                                              7,
                                                                                                              256};
    unsigned g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_features_18_0_dedx_0_id;
    unsigned char g_0_gradient_features_18_0_dedx_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,  1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 97, 110, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_tensor_944_4102_gradient_features_aten_native_batch_norm_backward,
                    g_0_tensor_814_3958_features_18_0_hpu_cast},
                   {g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable},
                   (void*)g_0_gradient_features_18_0_dedx_0_params,
                   104,
                   "g_0_gradient_features_18_0_dedx_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_features_18_0_dedx_0_id);

    /*************
     * g_0_gradient_features_17_cud_bn_bwd_ex_0 node
     * inputs:
     *     g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable[320, 7, 7, 256] (dtype=bf16)
     *     g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable[320, 7, 7, 256]
     *(dtype=bf16) g_0_tensor_801[320] (dtype=float32) g_0_tensor_951[320] (dtype=float32)
     *     g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv[320] (dtype=float32)
     *     g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv[320] (dtype=float32)
     * outputs:
     *     g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward[320, 7, 7, 256] (dtype=bf16)
     *     g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward[320] (dtype=float32)
     *     g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward[320] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable tensor
    unsigned g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable_max_sizes[] = {320, 7, 7, 256};
    unsigned g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable_min_sizes[] = {320, 7, 7, 256};
    unsigned g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_801 tensor
    unsigned g_0_tensor_801_max_sizes[] = {320};
    unsigned g_0_tensor_801_min_sizes[] = {320};
    unsigned g_0_tensor_801             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_801",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_801_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_801_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_951 tensor
    unsigned g_0_tensor_951_max_sizes[] = {320};
    unsigned g_0_tensor_951_min_sizes[] = {320};
    unsigned g_0_tensor_951             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_951",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_951_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_951_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv tensor
    unsigned g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv_max_sizes[] = {320};
    unsigned g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv_min_sizes[] = {320};
    unsigned g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv tensor
    unsigned g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv_max_sizes[] = {320};
    unsigned g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv_min_sizes[] = {320};
    unsigned g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward tensor
    unsigned g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward_max_sizes[] = {320, 7, 7, 256};
    unsigned g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward_min_sizes[] = {320, 7, 7, 256};
    unsigned g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward tensor
    unsigned g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward_max_sizes[] = {320};
    unsigned g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward_min_sizes[] = {320};
    unsigned g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward tensor
    unsigned g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward_max_sizes[] = {320};
    unsigned g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward_min_sizes[] = {320};
    unsigned g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_features_17_cud_bn_bwd_ex_0_id;
    unsigned char g_0_gradient_features_17_cud_bn_bwd_ex_0_params[] = {0, 0, 0, 0, 0, 0, 0, 0, 172, 197, 39, 55};
    addNodeToGraph("cud_bn_bwd_ex",
                   {g_0_tensor_800_3937_features_17_conv_2_aten_convolution_overrideable,
                    g_0_tensor_948_4113_gradient_features_18_0_aten_convolution_backward_overrideable,
                    g_0_tensor_801,
                    g_0_tensor_951,
                    g_0_tensor_809_3951_features_17_conv_3_hpu_native_batch_norm_rmv,
                    g_0_tensor_810_3952_features_17_conv_3_hpu_native_batch_norm_rmv},
                   {g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward,
                    g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward,
                    g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward},
                   (void*)g_0_gradient_features_17_cud_bn_bwd_ex_0_params,
                   12,
                   "g_0_gradient_features_17_cud_bn_bwd_ex_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_features_17_cud_bn_bwd_ex_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_947_4114_gradient_features_18_0_aten_convolution_backward_overrideable,
                        g_0_tensor_952_4125_gradient_features_17_aten_native_batch_norm_backward,
                        g_0_tensor_953_4126_gradient_features_17_aten_native_batch_norm_backward,
                        g_0_tensor_954_4127_gradient_features_17_aten_native_batch_norm_backward});
}

class SynGaudiSliceAsymmetricBgemmSpatially
: public SynGaudiTwoRunCompareTest
, public testing::WithParamInterface<unsigned>  // bgemmOutWidth
{
public:
    void runSingleTest()
    {
        // Graph #0

        unsigned bgemmOutWidth = GetParam();  // 250012;

        /*************
         * g_0__cast_f32_to_bf16_3374_0 node
         * inputs:
         *     g_0_tensor_14[1024, 250012] (dtype=float32)
         * outputs:
         *     g_0_tensor_4526_16102_hpu_cast[1024, 250012] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_tensor_14 tensor
        unsigned g_0_tensor_14_max_sizes[] = {1024, bgemmOutWidth};
        unsigned g_0_tensor_14_min_sizes[] = {1024, bgemmOutWidth};
        unsigned g_0_tensor_14             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_tensor_14",
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               g_0_tensor_14_max_sizes,
                                               2,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_tensor_14_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_4526_16102_hpu_cast tensor
        unsigned      g_0_tensor_4526_16102_hpu_cast_max_sizes[] = {1024, bgemmOutWidth};
        unsigned      g_0_tensor_4526_16102_hpu_cast_min_sizes[] = {1024, bgemmOutWidth};
        unsigned      g_0_tensor_4526_16102_hpu_cast             = createTensors(1,
                                                                OUTPUT_TENSOR,
                                                                true,
                                                                "g_0_tensor_4526_16102_hpu_cast",
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                g_0_tensor_4526_16102_hpu_cast_max_sizes,
                                                                2,
                                                                syn_type_bf16,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_tensor_4526_16102_hpu_cast_min_sizes,
                                                                synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0__cast_f32_to_bf16_3374_0_id;
        unsigned char g_0__cast_f32_to_bf16_3374_0_params[] = {0, 0, 0, 0};
        addNodeToGraph("cast_f32_to_bf16",
                       {g_0_tensor_14},
                       {g_0_tensor_4526_16102_hpu_cast},
                       (void*)g_0__cast_f32_to_bf16_3374_0_params,
                       4,
                       "g_0__cast_f32_to_bf16_3374_0",
                       0 /*graphIndex*/,
                       &g_0__cast_f32_to_bf16_3374_0_id);

        /*************
         * g_0__transpose_3375_0 node
         * inputs:
         *     g_0_tensor_4526_16102_hpu_cast[1024, 250012] (dtype=bf16)
         * outputs:
         *     g_0_tensor_4527_16105_aten_t[250012, 1024] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_tensor_4527_16105_aten_t tensor
        unsigned      g_0_tensor_4527_16105_aten_t_max_sizes[] = {bgemmOutWidth, 1024};
        unsigned      g_0_tensor_4527_16105_aten_t_min_sizes[] = {bgemmOutWidth, 1024};
        unsigned      g_0_tensor_4527_16105_aten_t             = createTensors(1,
                                                              OUTPUT_TENSOR,
                                                              false,
                                                              "g_0_tensor_4527_16105_aten_t",
                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                              nullptr,
                                                              g_0_tensor_4527_16105_aten_t_max_sizes,
                                                              2,
                                                              syn_type_bf16,
                                                              nullptr,
                                                              0,
                                                              0,
                                                              nullptr,
                                                              false,
                                                              g_0_tensor_4527_16105_aten_t_min_sizes,
                                                              synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0__transpose_3375_0_id;
        unsigned char g_0__transpose_3375_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
        addNodeToGraph("transpose",
                       {g_0_tensor_4526_16102_hpu_cast},
                       {g_0_tensor_4527_16105_aten_t},
                       (void*)g_0__transpose_3375_0_params,
                       24,
                       "g_0__transpose_3375_0",
                       0 /*graphIndex*/,
                       &g_0__transpose_3375_0_id);

        /*************
         * g_0__batch_gemm_3377_0 node
         * inputs:
         *     g_0_tensor_4528_16090_hpu_strided_view[1024, 13, 4] (dtype=bf16)
         *     g_0_tensor_4527_16105_aten_t[250012, 1024] (dtype=bf16)
         * outputs:
         *     g_0_tensor_4529_16108_aten_matmul[250012, 13, 4] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_tensor_4528_16090_hpu_strided_view tensor
        unsigned g_0_tensor_4528_16090_hpu_strided_view_max_sizes[] = {1024, 13, 4};
        unsigned g_0_tensor_4528_16090_hpu_strided_view_min_sizes[] = {1024, 13, 4};
        unsigned g_0_tensor_4528_16090_hpu_strided_view =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_4528_16090_hpu_strided_view",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_4528_16090_hpu_strided_view_max_sizes,
                          3,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_4528_16090_hpu_strided_view_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_4529_16108_aten_matmul tensor
        unsigned      g_0_tensor_4529_16108_aten_matmul_max_sizes[] = {bgemmOutWidth, 13, 4};
        unsigned      g_0_tensor_4529_16108_aten_matmul_min_sizes[] = {bgemmOutWidth, 13, 4};
        unsigned      g_0_tensor_4529_16108_aten_matmul             = createTensors(1,
                                                                   OUTPUT_TENSOR,
                                                                   true,
                                                                   "g_0_tensor_4529_16108_aten_matmul",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_tensor_4529_16108_aten_matmul_max_sizes,
                                                                   3,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_tensor_4529_16108_aten_matmul_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0__batch_gemm_3377_0_id;
        unsigned char g_0__batch_gemm_3377_0_params[] = {0, 0};
        addNodeToGraph("batch_gemm",
                       {g_0_tensor_4528_16090_hpu_strided_view, g_0_tensor_4527_16105_aten_t},
                       {g_0_tensor_4529_16108_aten_matmul},
                       (void*)g_0__batch_gemm_3377_0_params,
                       2,
                       "g_0__batch_gemm_3377_0",
                       0 /*graphIndex*/,
                       &g_0__batch_gemm_3377_0_id);

        addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
        addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
        addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
        addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
        // Disable TPC optimizations to make sure the cast is bundled with bgemm. Otherwise a reshape might block it.
        addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");

        compareRunsResults({g_0_tensor_4529_16108_aten_matmul});
    }
};

// TODO SW-97880 - Enable for gaudi1 once the test with param 250012 doesn't fail on comparison
TEST_P_GC(SynGaudiSliceAsymmetricBgemmSpatially, slice_asym_bgemm_spatially, {synDeviceGaudi2})
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(slice_bgemm_spatially_ASIC_CI,
                         SynGaudiSliceAsymmetricBgemmSpatially,
                         ::testing::Values(250012, 1024, 2000, 5006, 10012, 17180));

INSTANTIATE_TEST_SUITE_P(slice_bgemm_spatially_single,
                         SynGaudiSliceAsymmetricBgemmSpatially,
                         ::testing::Values(250012 / 4));

// TODO SW-97880 - Enable for gaudi1 once the test with param 250012 doesn't fail on comparison
TEST_F_GC(SynGaudiTwoRunCompareTest, copy_strided_input_to_sram_producer_not_bundled_ASIC_CI, {synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0__cast_f32_to_bf16_52055_0 node
     * inputs:
     *     g_0_tensor_337[1024, 250012] (dtype=float32)
     * outputs:
     *     g_0_tensor_5077_id_117409_hpu__cast[1024, 250012] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_337 tensor
    unsigned g_0_tensor_337_max_sizes[] = {1024, 250012};
    unsigned g_0_tensor_337_min_sizes[] = {1024, 250012};
    unsigned g_0_tensor_337             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_337",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_337_max_sizes,
                                            2,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_337_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_5077_id_117409_hpu__cast tensor
    unsigned      g_0_tensor_5077_id_117409_hpu__cast_max_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5077_id_117409_hpu__cast_min_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5077_id_117409_hpu__cast             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_tensor_5077_id_117409_hpu__cast",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_5077_id_117409_hpu__cast_max_sizes,
                                                                 2,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_5077_id_117409_hpu__cast_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__cast_f32_to_bf16_52055_0_id;
    unsigned char g_0__cast_f32_to_bf16_52055_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_337},
                   {g_0_tensor_5077_id_117409_hpu__cast},
                   (void*)g_0__cast_f32_to_bf16_52055_0_params,
                   4,
                   "g_0__cast_f32_to_bf16_52055_0",
                   0 /*graphIndex*/,
                   &g_0__cast_f32_to_bf16_52055_0_id);

    /*************
     * g_0__transpose_52056_0 node
     * inputs:
     *     g_0_tensor_5077_id_117409_hpu__cast[1024, 250012] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5078_id_117412_aten__t_1[250012, 1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5078_id_117412_aten__t_1 tensor
    unsigned      g_0_tensor_5078_id_117412_aten__t_1_max_sizes[] = {250012, 1024};
    unsigned      g_0_tensor_5078_id_117412_aten__t_1_min_sizes[] = {250012, 1024};
    unsigned      g_0_tensor_5078_id_117412_aten__t_1             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_tensor_5078_id_117412_aten__t_1",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_5078_id_117412_aten__t_1_max_sizes,
                                                                 2,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_5078_id_117412_aten__t_1_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__transpose_52056_0_id;
    unsigned char g_0__transpose_52056_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_5077_id_117409_hpu__cast},
                   {g_0_tensor_5078_id_117412_aten__t_1},
                   (void*)g_0__transpose_52056_0_params,
                   24,
                   "g_0__transpose_52056_0",
                   0 /*graphIndex*/,
                   &g_0__transpose_52056_0_id);

    /*************
     * g_0__cast_f32_to_bf16_52050_0 node
     * inputs:
     *     g_0_tensor_337[1024, 250012] (dtype=float32)
     * outputs:
     *     g_0_tensor_5072_id_117418_hpu__cast[1024, 250012] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5072_id_117418_hpu__cast tensor
    unsigned      g_0_tensor_5072_id_117418_hpu__cast_max_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5072_id_117418_hpu__cast_min_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5072_id_117418_hpu__cast             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_tensor_5072_id_117418_hpu__cast",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_5072_id_117418_hpu__cast_max_sizes,
                                                                 2,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_5072_id_117418_hpu__cast_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__cast_f32_to_bf16_52050_0_id;
    unsigned char g_0__cast_f32_to_bf16_52050_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_337},
                   {g_0_tensor_5072_id_117418_hpu__cast},
                   (void*)g_0__cast_f32_to_bf16_52050_0_params,
                   4,
                   "g_0__cast_f32_to_bf16_52050_0",
                   0 /*graphIndex*/,
                   &g_0__cast_f32_to_bf16_52050_0_id);

    /*************
     * g_0__transpose_52051_0 node
     * inputs:
     *     g_0_tensor_5072_id_117418_hpu__cast[1024, 250012] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5073_id_117421_aten__t_1[250012, 1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5073_id_117421_aten__t_1 tensor
    unsigned      g_0_tensor_5073_id_117421_aten__t_1_max_sizes[] = {250012, 1024};
    unsigned      g_0_tensor_5073_id_117421_aten__t_1_min_sizes[] = {250012, 1024};
    unsigned      g_0_tensor_5073_id_117421_aten__t_1             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_tensor_5073_id_117421_aten__t_1",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_5073_id_117421_aten__t_1_max_sizes,
                                                                 2,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_5073_id_117421_aten__t_1_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__transpose_52051_0_id;
    unsigned char g_0__transpose_52051_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_5072_id_117418_hpu__cast},
                   {g_0_tensor_5073_id_117421_aten__t_1},
                   (void*)g_0__transpose_52051_0_params,
                   24,
                   "g_0__transpose_52051_0",
                   0 /*graphIndex*/,
                   &g_0__transpose_52051_0_id);

    /*************
     * g_0__transpose_52052_0 node
     * inputs:
     *     g_0_tensor_5071_id_117392_aten__native_layer_norm[1024, 64, 78] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5074_id_117397_aten__transpose_1[1024, 78, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5071_id_117392_aten__native_layer_norm tensor
    unsigned g_0_tensor_5071_id_117392_aten__native_layer_norm_max_sizes[] = {1024, 64, 78};
    unsigned g_0_tensor_5071_id_117392_aten__native_layer_norm_min_sizes[] = {1024, 64, 78};
    unsigned g_0_tensor_5071_id_117392_aten__native_layer_norm =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_5071_id_117392_aten__native_layer_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_5071_id_117392_aten__native_layer_norm_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_5071_id_117392_aten__native_layer_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_5074_id_117397_aten__transpose_1 tensor
    unsigned g_0_tensor_5074_id_117397_aten__transpose_1_max_sizes[] = {1024, 78, 64};
    unsigned g_0_tensor_5074_id_117397_aten__transpose_1_min_sizes[] = {1024, 78, 64};
    unsigned g_0_tensor_5074_id_117397_aten__transpose_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_5074_id_117397_aten__transpose_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_5074_id_117397_aten__transpose_1_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_5074_id_117397_aten__transpose_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__transpose_52052_0_id;
    unsigned char g_0__transpose_52052_0_params[] = {0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_5071_id_117392_aten__native_layer_norm},
                   {g_0_tensor_5074_id_117397_aten__transpose_1},
                   (void*)g_0__transpose_52052_0_params,
                   24,
                   "g_0__transpose_52052_0",
                   0 /*graphIndex*/,
                   &g_0__transpose_52052_0_id);

    /*************
     * g_0__slice_52053_0 node
     * inputs:
     *     g_0_tensor_5074_id_117397_aten__transpose_1[1024, 78, 64] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5075_id_117406_aten__slice_1[1024, 26, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5075_id_117406_aten__slice_1 tensor
    unsigned      g_0_tensor_5075_id_117406_aten__slice_1_max_sizes[] = {1024, 26, 64};
    unsigned      g_0_tensor_5075_id_117406_aten__slice_1_min_sizes[] = {1024, 26, 64};
    unsigned      g_0_tensor_5075_id_117406_aten__slice_1             = createTensors(1,
                                                                     OUTPUT_TENSOR,
                                                                     false,
                                                                     "g_0_tensor_5075_id_117406_aten__slice_1",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_tensor_5075_id_117406_aten__slice_1_max_sizes,
                                                                     3,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_tensor_5075_id_117406_aten__slice_1_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__slice_52053_0_id;
    unsigned char g_0__slice_52053_0_params[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,  0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,  1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,  0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0};
    addNodeToGraph("slice",
                   {g_0_tensor_5074_id_117397_aten__transpose_1},
                   {g_0_tensor_5075_id_117406_aten__slice_1},
                   (void*)g_0__slice_52053_0_params,
                   400,
                   "g_0__slice_52053_0",
                   0 /*graphIndex*/,
                   &g_0__slice_52053_0_id);

    /*************
     * g_0__batch_gemm_52054_0 node
     * inputs:
     *     g_0_tensor_5075_id_117406_aten__slice_1[1024, 26, 64] (dtype=bf16)
     *     g_0_tensor_5073_id_117421_aten__t_1[250012, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5076_id_117424_aten__matmul[250012, 26, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5076_id_117424_aten__matmul tensor
    unsigned      g_0_tensor_5076_id_117424_aten__matmul_max_sizes[] = {250012, 26, 64};
    unsigned      g_0_tensor_5076_id_117424_aten__matmul_min_sizes[] = {250012, 26, 64};
    unsigned      g_0_tensor_5076_id_117424_aten__matmul             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    true,
                                                                    "g_0_tensor_5076_id_117424_aten__matmul",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_tensor_5076_id_117424_aten__matmul_max_sizes,
                                                                    3,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_tensor_5076_id_117424_aten__matmul_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__batch_gemm_52054_0_id;
    unsigned char g_0__batch_gemm_52054_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_5075_id_117406_aten__slice_1, g_0_tensor_5073_id_117421_aten__t_1},
                   {g_0_tensor_5076_id_117424_aten__matmul},
                   (void*)g_0__batch_gemm_52054_0_params,
                   2,
                   "g_0__batch_gemm_52054_0",
                   0 /*graphIndex*/,
                   &g_0__batch_gemm_52054_0_id);

    /*************
     * g_0__slice_52057_0 node
     * inputs:
     *     g_0_tensor_5074_id_117397_aten__transpose_1[1024, 78, 64] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5079_id_117403_aten__slice_1[1024, 26, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5079_id_117403_aten__slice_1 tensor
    unsigned      g_0_tensor_5079_id_117403_aten__slice_1_max_sizes[] = {1024, 26, 64};
    unsigned      g_0_tensor_5079_id_117403_aten__slice_1_min_sizes[] = {1024, 26, 64};
    unsigned      g_0_tensor_5079_id_117403_aten__slice_1             = createTensors(1,
                                                                     OUTPUT_TENSOR,
                                                                     false,
                                                                     "g_0_tensor_5079_id_117403_aten__slice_1",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_tensor_5079_id_117403_aten__slice_1_max_sizes,
                                                                     3,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_tensor_5079_id_117403_aten__slice_1_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__slice_52057_0_id;
    unsigned char g_0__slice_52057_0_params[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,  0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,  1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,  0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0};
    addNodeToGraph("slice",
                   {g_0_tensor_5074_id_117397_aten__transpose_1},
                   {g_0_tensor_5079_id_117403_aten__slice_1},
                   (void*)g_0__slice_52057_0_params,
                   400,
                   "g_0__slice_52057_0",
                   0 /*graphIndex*/,
                   &g_0__slice_52057_0_id);

    /*************
     * g_0__batch_gemm_52058_0 node
     * inputs:
     *     g_0_tensor_5079_id_117403_aten__slice_1[1024, 26, 64] (dtype=bf16)
     *     g_0_tensor_5078_id_117412_aten__t_1[250012, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5080_id_117415_aten__matmul[250012, 26, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5080_id_117415_aten__matmul tensor
    unsigned      g_0_tensor_5080_id_117415_aten__matmul_max_sizes[] = {250012, 26, 64};
    unsigned      g_0_tensor_5080_id_117415_aten__matmul_min_sizes[] = {250012, 26, 64};
    unsigned      g_0_tensor_5080_id_117415_aten__matmul             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    true,
                                                                    "g_0_tensor_5080_id_117415_aten__matmul",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_tensor_5080_id_117415_aten__matmul_max_sizes,
                                                                    3,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_tensor_5080_id_117415_aten__matmul_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__batch_gemm_52058_0_id;
    unsigned char g_0__batch_gemm_52058_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_5079_id_117403_aten__slice_1, g_0_tensor_5078_id_117412_aten__t_1},
                   {g_0_tensor_5080_id_117415_aten__matmul},
                   (void*)g_0__batch_gemm_52058_0_params,
                   2,
                   "g_0__batch_gemm_52058_0",
                   0 /*graphIndex*/,
                   &g_0__batch_gemm_52058_0_id);

    /*************
     * g_0__transpose_52107_0 node
     * inputs:
     *     g_0_tensor_5078_id_117412_aten__t_1[250012, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5145[1024, 250012] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5145 tensor
    unsigned      g_0_tensor_5145_max_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5145_min_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5145             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_5145",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_5145_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_5145_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__transpose_52107_0_id;
    unsigned char g_0__transpose_52107_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_5078_id_117412_aten__t_1},
                   {g_0_tensor_5145},
                   (void*)g_0__transpose_52107_0_params,
                   24,
                   "g_0__transpose_52107_0",
                   0 /*graphIndex*/,
                   &g_0__transpose_52107_0_id);

    /*************
     * g_0__transpose_52096_0 node
     * inputs:
     *     g_0_tensor_5073_id_117421_aten__t_1[250012, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5134[1024, 250012] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5134 tensor
    unsigned      g_0_tensor_5134_max_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5134_min_sizes[] = {1024, 250012};
    unsigned      g_0_tensor_5134             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_5134",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_5134_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_5134_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__transpose_52096_0_id;
    unsigned char g_0__transpose_52096_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_5073_id_117421_aten__t_1},
                   {g_0_tensor_5134},
                   (void*)g_0__transpose_52096_0_params,
                   24,
                   "g_0__transpose_52096_0",
                   0 /*graphIndex*/,
                   &g_0__transpose_52096_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_5080_id_117415_aten__matmul,
                        g_0_tensor_5076_id_117424_aten__matmul,
                        g_0_tensor_5145,
                        g_0_tensor_5134});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, conv_with_spatial_slicing_and_padding_first_slice_is_smaller_ASIC_CI)
{
    unsigned g_0_t12866_grid_backbone_grid_backbone_grid_unet_up_1_upsamp_concat_5_0_max_sizes[] = {1024, 64, 4, 32, 1};
    unsigned g_0_t12866_grid_backbone_grid_backbone_grid_unet_up_1_upsamp_concat_5_0_min_sizes[] = {1024, 64, 4, 32, 1};
    unsigned g_0_t12866_grid_backbone_grid_backbone_grid_unet_up_1_upsamp_concat_5_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t12866_grid_backbone_grid_backbone_grid_unet_up_1_upsamp_concat_5_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12866_grid_backbone_grid_backbone_grid_unet_up_1_upsamp_concat_5_0_max_sizes,
                      5,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12866_grid_backbone_grid_backbone_grid_unet_up_1_upsamp_concat_5_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1798_read_84_readvariableop_0_max_sizes[] = {256, 1024, 3, 3, 3};
    unsigned g_0_t1798_read_84_readvariableop_0_min_sizes[] = {256, 1024, 3, 3, 3};
    unsigned g_0_t1798_read_84_readvariableop_0             = createTensors(1,
                                                                INPUT_TENSOR,
                                                                true,
                                                                "g_0_t1798_read_84_readvariableop_0",
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                g_0_t1798_read_84_readvariableop_0_max_sizes,
                                                                5,
                                                                syn_type_single,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_t1798_read_84_readvariableop_0_min_sizes,
                                                                synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0_max_sizes[] = {256, 64, 4, 32, 1};
    unsigned g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0_min_sizes[] = {256, 64, 4, 32, 1};
    unsigned g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0_max_sizes,
                      5,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_spatial_convolution3d_n7996_0_id;
    unsigned char g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_spatial_convolution3d_n7996_0_params[] =
        {3, 0, 0,   0,   3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
         1, 0, 0,   0,   1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 250, 110, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "spatial_convolution3d",
        {g_0_t12866_grid_backbone_grid_backbone_grid_unet_up_1_upsamp_concat_5_0, g_0_t1798_read_84_readvariableop_0},
        {g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0},
        (void*)g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_spatial_convolution3d_n7996_0_params,
        128,
        "g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_spatial_convolution3d_n7996_0",
        0 /*graphIndex*/,
        &g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_spatial_convolution3d_n7996_0_id);

    unsigned g_0_t12869_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_max_sizes[] = {256, 1, 1, 1, 1};
    unsigned g_0_t12869_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_min_sizes[] = {256, 1, 1, 1, 1};
    unsigned g_0_t12869_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t12869_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12869_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_max_sizes,
                      5,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12869_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0_max_sizes[] = {256, 64, 4, 32, 1};
    unsigned g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0_min_sizes[] = {256, 64, 4, 32, 1};
    unsigned g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0_max_sizes,
                      5,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_add_fwd_f32_n7998_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_t12867_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Conv3D_1_0,
                    g_0_t12869_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1},
                   {g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0},
                   nullptr,
                   0,
                   "g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_add_fwd_f32_n7998_0",
                   0 /*graphIndex*/,
                   &g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_add_fwd_f32_n7998_0_id);

    unsigned g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0_max_sizes[] = {256, 64, 4, 32, 1};
    unsigned g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0_min_sizes[] = {256, 64, 4, 32, 1};
    unsigned g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0_max_sizes,
                      5,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_relu_fwd_f32_n7999_0_id;
    addNodeToGraph("relu_fwd_f32",
                   {g_0_t12868_grid_backbone_grid_backbone_grid_unet_up_1_conv1_BiasAdd_1_0},
                   {g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0},
                   nullptr,
                   0,
                   "g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_relu_fwd_f32_n7999_0",
                   0 /*graphIndex*/,
                   &g_0_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_relu_fwd_f32_n7999_0_id);

    unsigned g_0_t12739_grid_backbone_grid_backbone_grid_unet_up_1_conv_res_skip_BiasAdd_1_0_max_sizes[] = {256,
                                                                                                            64,
                                                                                                            4,
                                                                                                            32,
                                                                                                            1};
    unsigned g_0_t12739_grid_backbone_grid_backbone_grid_unet_up_1_conv_res_skip_BiasAdd_1_0_min_sizes[] = {256,
                                                                                                            64,
                                                                                                            4,
                                                                                                            32,
                                                                                                            1};
    unsigned g_0_t12739_grid_backbone_grid_backbone_grid_unet_up_1_conv_res_skip_BiasAdd_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t12739_grid_backbone_grid_backbone_grid_unet_up_1_conv_res_skip_BiasAdd_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12739_grid_backbone_grid_backbone_grid_unet_up_1_conv_res_skip_BiasAdd_1_0_max_sizes,
                      5,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12739_grid_backbone_grid_backbone_grid_unet_up_1_conv_res_skip_BiasAdd_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0_max_sizes[] = {256,
                                                                                                       64,
                                                                                                       4,
                                                                                                       32,
                                                                                                       1};
    unsigned g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0_min_sizes[] = {256,
                                                                                                       64,
                                                                                                       4,
                                                                                                       32,
                                                                                                       1};
    unsigned g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0_max_sizes,
                      5,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_add_fwd_f32_n8000_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_t12871_grid_backbone_grid_backbone_grid_unet_up_1_conv1_Relu_1_0,
                    g_0_t12739_grid_backbone_grid_backbone_grid_unet_up_1_conv_res_skip_BiasAdd_1_0},
                   {g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0},
                   nullptr,
                   0,
                   "g_0_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_add_fwd_f32_n8000_0",
                   0 /*graphIndex*/,
                   &g_0_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_add_fwd_f32_n8000_0_id);

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_t12872_grid_backbone_grid_backbone_grid_unet_up_1_add_res_skip_add_1_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, grad_a_and_reshaped_grad_b_pair_accuracy_check_ASIC_CI)
{
    TestSizeVec sharedSize         = {128, 64, 28};
    TestSizeVec reshapedSharedSize = {128, 64 * 28};
    TestSizeVec bSize              = {128, 128};

    auto shared = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sharedSize.data(),
                                      sharedSize.size(),
                                      syn_type_bf16,
                                      nullptr,
                                      "shared");

    auto b = createPersistTensor(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 bSize.data(),
                                 bSize.size(),
                                 syn_type_bf16,
                                 nullptr,
                                 "B");

    auto dA = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_NONE,
                                  nullptr,
                                  sharedSize.data(),
                                  sharedSize.size(),
                                  syn_type_bf16,
                                  nullptr,
                                  "dA");

    synGEMMParams gradAParams {false, true};
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {shared, b}, {dA}, &gradAParams, sizeof(gradAParams), "gradA");

    auto flatShared = createTensor(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   reshapedSharedSize.data(),
                                   reshapedSharedSize.size(),
                                   syn_type_bf16,
                                   nullptr);

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {shared}, {flatShared}, nullptr, 0, "reshape");

    auto flatA = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     reshapedSharedSize.data(),
                                     reshapedSharedSize.size(),
                                     syn_type_bf16,
                                     nullptr,
                                     "flat_A");

    auto dB = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  bSize.data(),
                                  bSize.size(),
                                  syn_type_bf16,
                                  nullptr,
                                  "dB");

    synGEMMParams gradBParams {true, false};
    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {flatA, flatShared},
                   {dB},
                   &gradBParams,
                   sizeof(gradBParams),
                   "gradB");

    addConfigurationToRun(FIRST_RUN, "ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING", "true");

    compareRunsResults({dA, dB});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, grad_a_and_reshaped_grad_b_pair_accuracy_check_tiny)
{
    TestSizeVec sharedSize         = {64, 4, 33};
    TestSizeVec reshapedSharedSize = {64, 4 * 33};
    TestSizeVec bSize              = {64, 64};

    auto shared = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sharedSize.data(),
                                      sharedSize.size(),
                                      syn_type_bf16,
                                      nullptr,
                                      "shared");

    auto b = createPersistTensor(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 bSize.data(),
                                 bSize.size(),
                                 syn_type_bf16,
                                 nullptr,
                                 "B");

    auto dA = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_NONE,
                                  nullptr,
                                  sharedSize.data(),
                                  sharedSize.size(),
                                  syn_type_bf16,
                                  nullptr,
                                  "dA");

    synGEMMParams gradAParams {false, true};
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {shared, b}, {dA}, &gradAParams, sizeof(gradAParams), "gradA");

    auto flatShared = createTensor(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   reshapedSharedSize.data(),
                                   reshapedSharedSize.size(),
                                   syn_type_bf16,
                                   nullptr);

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {shared}, {flatShared}, nullptr, 0, "reshape");

    auto flatA = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     reshapedSharedSize.data(),
                                     reshapedSharedSize.size(),
                                     syn_type_bf16,
                                     nullptr,
                                     "flat_A");

    auto dB = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  bSize.data(),
                                  bSize.size(),
                                  syn_type_bf16,
                                  nullptr,
                                  "dB");

    synGEMMParams gradBParams {true, false};
    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {flatA, flatShared},
                   {dB},
                   &gradBParams,
                   sizeof(gradBParams),
                   "gradB");

    addConfigurationToRun(FIRST_RUN, "ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING", "true");

    compareRunsResults({dA, dB});
}

TEST_F_GC(SynGaudiTwoRunCompareTest,
          manta_ray_bundle_with_logic_transpose_then_reshape_on_transposed_dims_leading_to_memcpy,
          {synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0_module_module_encoder_cast_f32_to_bf16_32453_0 node
     * inputs:
     *     g_0_tensor_32[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_34[1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_32 tensor
    unsigned g_0_tensor_32_max_sizes[] = {1};
    unsigned g_0_tensor_32_min_sizes[] = {1};
    unsigned g_0_tensor_32             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_32",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_32_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_32_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_34 tensor
    unsigned      g_0_tensor_34_max_sizes[] = {1};
    unsigned      g_0_tensor_34_min_sizes[] = {1};
    unsigned      g_0_tensor_34             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_34",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_34_max_sizes,
                                           1,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_34_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_module_encoder_cast_f32_to_bf16_32453_0_id;
    unsigned char g_0_module_module_encoder_cast_f32_to_bf16_32453_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_32},
                   {g_0_tensor_34},
                   (void*)g_0_module_module_encoder_cast_f32_to_bf16_32453_0_params,
                   4,
                   "g_0_module_module_encoder_cast_f32_to_bf16_32453_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_cast_f32_to_bf16_32453_0_id);

    /*************
     * g_0_module_module_encoder_mult_fwd_bf16_32454_0 node
     * inputs:
     *     g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding[1024, 20, 408] (dtype=bf16)
     *     g_0_tensor_34[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_33_id_110810_module_module_encoder_aten__mul[1024, 20, 408] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding tensor
    unsigned g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding_max_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding_min_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_33_id_110810_module_module_encoder_aten__mul tensor
    unsigned g_0_tensor_33_id_110810_module_module_encoder_aten__mul_max_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_33_id_110810_module_module_encoder_aten__mul_min_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_33_id_110810_module_module_encoder_aten__mul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_33_id_110810_module_module_encoder_aten__mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_33_id_110810_module_module_encoder_aten__mul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_33_id_110810_module_module_encoder_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_module_encoder_mult_fwd_bf16_32454_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_31_id_110808_module_module_encoder_embed_tokens_aten__embedding, g_0_tensor_34},
                   {g_0_tensor_33_id_110810_module_module_encoder_aten__mul},
                   nullptr,
                   0,
                   "g_0_module_module_encoder_mult_fwd_bf16_32454_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_mult_fwd_bf16_32454_0_id);

    /*************
     * g_0_module_module_encoder_add_fwd_bf16_32455_0 node
     * inputs:
     *     g_0_tensor_33_id_110810_module_module_encoder_aten__mul[1024, 20, 408] (dtype=bf16)
     *     g_0_tensor_27_id_110841_module_module_encoder_aten__view[1024, 20, 408] (dtype=bf16)
     * outputs:
     *     g_0_tensor_35_id_110847_module_module_encoder_aten__add[1024, 20, 408] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_27_id_110841_module_module_encoder_aten__view tensor
    unsigned g_0_tensor_27_id_110841_module_module_encoder_aten__view_max_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_27_id_110841_module_module_encoder_aten__view_min_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_27_id_110841_module_module_encoder_aten__view =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_27_id_110841_module_module_encoder_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_27_id_110841_module_module_encoder_aten__view_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_27_id_110841_module_module_encoder_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_35_id_110847_module_module_encoder_aten__add tensor
    unsigned g_0_tensor_35_id_110847_module_module_encoder_aten__add_max_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_35_id_110847_module_module_encoder_aten__add_min_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_35_id_110847_module_module_encoder_aten__add =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_35_id_110847_module_module_encoder_aten__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_35_id_110847_module_module_encoder_aten__add_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_35_id_110847_module_module_encoder_aten__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_module_encoder_add_fwd_bf16_32455_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_tensor_33_id_110810_module_module_encoder_aten__mul,
                    g_0_tensor_27_id_110841_module_module_encoder_aten__view},
                   {g_0_tensor_35_id_110847_module_module_encoder_aten__add},
                   nullptr,
                   0,
                   "g_0_module_module_encoder_add_fwd_bf16_32455_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_add_fwd_bf16_32455_0_id);

    /*************
     * g_0_module_module_encoder_dropout_module_dropout_fwd_bf16_32456_0 node
     * inputs:
     *     g_0_tensor_35_id_110847_module_module_encoder_aten__add[1024, 20, 408] (dtype=bf16)
     *     g_0_tensor_36[1] (dtype=int32)
     * outputs:
     *     g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout[1024, 20, 408] (dtype=bf16)
     *     g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout[1024, 20, 408] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_36 tensor
    unsigned g_0_tensor_36_max_sizes[] = {1};
    unsigned g_0_tensor_36_min_sizes[] = {1};
    unsigned g_0_tensor_36             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_36",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_36_max_sizes,
                                           1,
                                           syn_type_int32,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_36_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout tensor
    unsigned g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout_max_sizes[] = {1024,
                                                                                                             20,
                                                                                                             408};
    unsigned g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout_min_sizes[] = {1024,
                                                                                                             20,
                                                                                                             408};
    unsigned g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout tensor
    unsigned g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout_max_sizes[] = {1024,
                                                                                                             20,
                                                                                                             408};
    unsigned g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout_min_sizes[] = {1024,
                                                                                                             20,
                                                                                                             408};
    unsigned g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout_max_sizes,
                      3,
                      syn_type_int8,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_module_encoder_dropout_module_dropout_fwd_bf16_32456_0_id;
    unsigned char g_0_module_module_encoder_dropout_module_dropout_fwd_bf16_32456_0_params[] =
        {205, 204, 76, 62, 88, 127, 0, 0};
    addNodeToGraph("dropout_fwd_bf16",
                   {g_0_tensor_35_id_110847_module_module_encoder_aten__add, g_0_tensor_36},
                   {g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout,
                    g_0_tensor_38_id_110855_module_module_encoder_dropout_module_hpu___fused_dropout},
                   (void*)g_0_module_module_encoder_dropout_module_dropout_fwd_bf16_32456_0_params,
                   8,
                   "g_0_module_module_encoder_dropout_module_dropout_fwd_bf16_32456_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_dropout_module_dropout_fwd_bf16_32456_0_id);

    /*************
     * g_0_module_module_encoder_mult_fwd_bf16_32457_0 node
     * inputs:
     *     g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout[1024, 20, 408] (dtype=bf16)
     *     g_0_tensor_10[1, 20, 408] (dtype=bf16)
     * outputs:
     *     g_0_tensor_39_id_110867_module_module_encoder_aten__mul[1024, 20, 408] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_10 tensor
    unsigned g_0_tensor_10_max_sizes[] = {1, 20, 408};
    unsigned g_0_tensor_10_min_sizes[] = {1, 20, 408};
    unsigned g_0_tensor_10             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_10",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_10_max_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_10_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_39_id_110867_module_module_encoder_aten__mul tensor
    unsigned g_0_tensor_39_id_110867_module_module_encoder_aten__mul_max_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_39_id_110867_module_module_encoder_aten__mul_min_sizes[] = {1024, 20, 408};
    unsigned g_0_tensor_39_id_110867_module_module_encoder_aten__mul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_39_id_110867_module_module_encoder_aten__mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_39_id_110867_module_module_encoder_aten__mul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_39_id_110867_module_module_encoder_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_module_encoder_mult_fwd_bf16_32457_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_37_id_110853_module_module_encoder_dropout_module_hpu___fused_dropout, g_0_tensor_10},
                   {g_0_tensor_39_id_110867_module_module_encoder_aten__mul},
                   nullptr,
                   0,
                   "g_0_module_module_encoder_mult_fwd_bf16_32457_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_mult_fwd_bf16_32457_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_layer_norm_transpose_32458_0 node
     * inputs:
     *     g_0_tensor_39_id_110867_module_module_encoder_aten__mul[1024, 20, 408] (dtype=bf16)
     * outputs:
     *     g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose[1024, 408, 20]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose tensor
    unsigned g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose_max_sizes[] = {1024,
                                                                                                                 408,
                                                                                                                 20};
    unsigned g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose_min_sizes[] = {1024,
                                                                                                                 408,
                                                                                                                 20};
    unsigned g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_module_encoder_0_self_attn_layer_norm_transpose_32458_0_id;
    unsigned char g_0_module_module_encoder_0_self_attn_layer_norm_transpose_32458_0_params[] = {
        0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_39_id_110867_module_module_encoder_aten__mul},
                   {g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose},
                   (void*)g_0_module_module_encoder_0_self_attn_layer_norm_transpose_32458_0_params,
                   24,
                   "g_0_module_module_encoder_0_self_attn_layer_norm_transpose_32458_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_0_self_attn_layer_norm_transpose_32458_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32459_0 node
     * inputs:
     *     g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose[1024, 408, 20]
     *(dtype=bf16) outputs: g_0_tensor_43[1024, 8160, 1, 1] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_43 tensor
    unsigned  g_0_tensor_43_max_sizes[] = {1024, 8160, 1, 1};
    unsigned  g_0_tensor_43_min_sizes[] = {1024, 8160, 1, 1};
    unsigned  g_0_tensor_43             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_43",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_43_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_43_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32459_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_40_id_110869_module_module_encoder_0_self_attn_layer_norm_aten__transpose},
                   {g_0_tensor_43},
                   nullptr,
                   0,
                   "g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32459_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32459_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_32462_0 node
     * inputs:
     *     g_0_tensor_43[1024, 8160, 1, 1] (dtype=bf16)
     *     g_0_tensor_44[1024] (dtype=bf16)
     *     g_0_tensor_45[1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_46[1024, 8160, 1, 1] (dtype=bf16)
     *     g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm[1, 8160, 1, 1]
     *(dtype=bf16) g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm[1, 8160,
     *1, 1] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_44 tensor
    unsigned g_0_tensor_44_max_sizes[] = {1024};
    unsigned g_0_tensor_44_min_sizes[] = {1024};
    unsigned g_0_tensor_44             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_44",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_44_max_sizes,
                                           1,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_44_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_45 tensor
    unsigned g_0_tensor_45_max_sizes[] = {1024};
    unsigned g_0_tensor_45_min_sizes[] = {1024};
    unsigned g_0_tensor_45             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_45",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_45_max_sizes,
                                           1,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_45_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_46 tensor
    unsigned g_0_tensor_46_max_sizes[] = {1024, 8160, 1, 1};
    unsigned g_0_tensor_46_min_sizes[] = {1024, 8160, 1, 1};
    unsigned g_0_tensor_46             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_46",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_46_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_46_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm tensor
    unsigned g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes[] =
        {1, 8160, 1, 1};
    unsigned g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes[] =
        {1, 8160, 1, 1};
    unsigned g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm tensor
    unsigned g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes[] =
        {1, 8160, 1, 1};
    unsigned g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes[] =
        {1, 8160, 1, 1};
    unsigned g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_module_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_32462_0_id;
    unsigned char g_0_module_module_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_32462_0_params[] =
        {1, 8, 0, 248, 172, 197, 39, 55};
    addNodeToGraph("layer_norm_fwd_bf16",
                   {g_0_tensor_43, g_0_tensor_44, g_0_tensor_45},
                   {g_0_tensor_46,
                    g_0_tensor_47_id_110873_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm,
                    g_0_tensor_48_id_110875_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm},
                   (void*)g_0_module_module_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_32462_0_params,
                   8,
                   "g_0_module_module_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_32462_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_32462_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32463_0 node
     * inputs:
     *     g_0_tensor_46[1024, 8160, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm[1024, 408, 20]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm tensor
    unsigned g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes[] =
        {1024, 408, 20};
    unsigned g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes[] =
        {1024, 408, 20};
    unsigned g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32463_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_46},
                   {g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm},
                   nullptr,
                   0,
                   "g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32463_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_0_self_attn_layer_norm_reshape_32463_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_v_proj_batch_gemm_32465_0 node
     * inputs:
     *     g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm[1024, 408, 20]
     *(dtype=bf16) g_0_tensor_52[1024, 1024] (dtype=bf16) outputs: g_0_tensor_53[1024, 408, 20] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_tensor_52 tensor
    unsigned g_0_tensor_52_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_52_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_52             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_52",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_52_max_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_52_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_53 tensor
    unsigned      g_0_tensor_53_max_sizes[] = {1024, 408, 20};
    unsigned      g_0_tensor_53_min_sizes[] = {1024, 408, 20};
    unsigned      g_0_tensor_53             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_53",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_53_max_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_53_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_module_encoder_0_self_attn_v_proj_batch_gemm_32465_0_id;
    unsigned char g_0_module_module_encoder_0_self_attn_v_proj_batch_gemm_32465_0_params[] = {0, 0};
    addNodeToGraph(
        "batch_gemm",
        {g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm, g_0_tensor_52},
        {g_0_tensor_53},
        (void*)g_0_module_module_encoder_0_self_attn_v_proj_batch_gemm_32465_0_params,
        2,
        "g_0_module_module_encoder_0_self_attn_v_proj_batch_gemm_32465_0",
        0 /*graphIndex*/,
        &g_0_module_module_encoder_0_self_attn_v_proj_batch_gemm_32465_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_k_proj_batch_gemm_32470_0 node
     * inputs:
     *     g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm[1024, 408, 20]
     *(dtype=bf16) g_0_tensor_59[1024, 1024] (dtype=bf16) outputs: g_0_tensor_60[1024, 408, 20] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_tensor_59 tensor
    unsigned g_0_tensor_59_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_59_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_59             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
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

    // create g_0_tensor_60 tensor
    unsigned      g_0_tensor_60_max_sizes[] = {1024, 408, 20};
    unsigned      g_0_tensor_60_min_sizes[] = {1024, 408, 20};
    unsigned      g_0_tensor_60             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_60",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_60_max_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_60_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_module_encoder_0_self_attn_k_proj_batch_gemm_32470_0_id;
    unsigned char g_0_module_module_encoder_0_self_attn_k_proj_batch_gemm_32470_0_params[] = {0, 0};
    addNodeToGraph(
        "batch_gemm",
        {g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm, g_0_tensor_59},
        {g_0_tensor_60},
        (void*)g_0_module_module_encoder_0_self_attn_k_proj_batch_gemm_32470_0_params,
        2,
        "g_0_module_module_encoder_0_self_attn_k_proj_batch_gemm_32470_0",
        0 /*graphIndex*/,
        &g_0_module_module_encoder_0_self_attn_k_proj_batch_gemm_32470_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_q_proj_batch_gemm_32476_0 node
     * inputs:
     *     g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm[1024, 408, 20]
     *(dtype=bf16) g_0_tensor_67[1024, 1024] (dtype=bf16) outputs: g_0_tensor_68[1024, 408, 20] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_tensor_67 tensor
    unsigned g_0_tensor_67_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_67_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_67             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_67",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_67_max_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_67_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_68 tensor
    unsigned      g_0_tensor_68_max_sizes[] = {1024, 408, 20};
    unsigned      g_0_tensor_68_min_sizes[] = {1024, 408, 20};
    unsigned      g_0_tensor_68             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_68",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_68_max_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_68_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_module_encoder_0_self_attn_q_proj_batch_gemm_32476_0_id;
    unsigned char g_0_module_module_encoder_0_self_attn_q_proj_batch_gemm_32476_0_params[] = {0, 0};
    addNodeToGraph(
        "batch_gemm",
        {g_0_tensor_49_id_110871_module_module_encoder_0_self_attn_layer_norm_aten__native_layer_norm, g_0_tensor_67},
        {g_0_tensor_68},
        (void*)g_0_module_module_encoder_0_self_attn_q_proj_batch_gemm_32476_0_params,
        2,
        "g_0_module_module_encoder_0_self_attn_q_proj_batch_gemm_32476_0",
        0 /*graphIndex*/,
        &g_0_module_module_encoder_0_self_attn_q_proj_batch_gemm_32476_0_id);

    // TODO [SW-111621]: After the following jira is done, instead of compiling only the first run,
    // uncomment the code below to trigger data comparison between both runs in addition to compilation.
    compileTopology("", FIRST_RUN);

    // addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    // addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    // compareRunsResults({g_0_tensor_68,
    //                     g_0_tensor_60,
    //                     g_0_tensor_53,
    //                     g_0_tensor_39_id_110867_module_module_encoder_aten__mul});
}

// TODO SW-141004 does not work on Gaudi3
TEST_F_GC(SynGaudiTwoRunCompareTest, masked_bgemm_slice_on_spatial, {synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0_query_batch_gemm_0_0 node
     * inputs:
     *     g_0_tensor_0[1024, 512, 1] (dtype=float32)
     *     g_0_tensor_1[1024, 1024] (dtype=float32)
     *     g_0_tensor_2[1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_3_id_46_query_aten__linear[1024, 512, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_0 tensor
    unsigned g_0_tensor_0_max_sizes[] = {1024, 512, 1};
    unsigned g_0_tensor_0_min_sizes[] = {1024, 512, 1};
    unsigned g_0_tensor_0             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_0",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_0_max_sizes,
                                          3,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_0_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1 tensor
    unsigned g_0_tensor_1_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_1_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_1             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_1",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_1_max_sizes,
                                          2,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_1_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2 tensor
    unsigned g_0_tensor_2_max_sizes[] = {1024};
    unsigned g_0_tensor_2_min_sizes[] = {1024};
    unsigned g_0_tensor_2             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_2",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_2_max_sizes,
                                          1,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_2_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3_id_46_query_aten__linear tensor
    unsigned      g_0_tensor_3_id_46_query_aten__linear_max_sizes[] = {1024, 512, 1};
    unsigned      g_0_tensor_3_id_46_query_aten__linear_min_sizes[] = {1024, 512, 1};
    unsigned      g_0_tensor_3_id_46_query_aten__linear             = createTensors(1,
                                                                   OUTPUT_TENSOR,
                                                                   true,
                                                                   "g_0_tensor_3_id_46_query_aten__linear",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_tensor_3_id_46_query_aten__linear_max_sizes,
                                                                   3,
                                                                   syn_type_single,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_tensor_3_id_46_query_aten__linear_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_query_batch_gemm_0_0_id;
    unsigned char g_0_query_batch_gemm_0_0_params[] = {0, 1};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_0, g_0_tensor_1, g_0_tensor_2},
                   {g_0_tensor_3_id_46_query_aten__linear},
                   (void*)g_0_query_batch_gemm_0_0_params,
                   2,
                   "g_0_query_batch_gemm_0_0",
                   0 /*graphIndex*/,
                   &g_0_query_batch_gemm_0_0_id);

    /*************
     * g_0__reshape_5_0 node
     * inputs:
     *     g_0_tensor_3_id_46_query_aten__linear[1024, 512, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_14_id_58_aten__view[64, 16, 512, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_14_id_58_aten__view tensor
    unsigned  g_0_tensor_14_id_58_aten__view_max_sizes[] = {64, 16, 512, 1};
    unsigned  g_0_tensor_14_id_58_aten__view_min_sizes[] = {64, 16, 512, 1};
    unsigned  g_0_tensor_14_id_58_aten__view             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            false,
                                                            "g_0_tensor_14_id_58_aten__view",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_14_id_58_aten__view_max_sizes,
                                                            4,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_14_id_58_aten__view_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__reshape_5_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_3_id_46_query_aten__linear},
                   {g_0_tensor_14_id_58_aten__view},
                   nullptr,
                   0,
                   "g_0__reshape_5_0",
                   0 /*graphIndex*/,
                   &g_0__reshape_5_0_id);

    /*************
     * g_0__transpose_6_0 node
     * inputs:
     *     g_0_tensor_14_id_58_aten__view[64, 16, 512, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_15_id_60_aten__permute[64, 512, 16, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_15_id_60_aten__permute tensor
    unsigned      g_0_tensor_15_id_60_aten__permute_max_sizes[] = {64, 512, 16, 1};
    unsigned      g_0_tensor_15_id_60_aten__permute_min_sizes[] = {64, 512, 16, 1};
    unsigned      g_0_tensor_15_id_60_aten__permute             = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               false,
                                                               "g_0_tensor_15_id_60_aten__permute",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               g_0_tensor_15_id_60_aten__permute_max_sizes,
                                                               4,
                                                               syn_type_single,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_tensor_15_id_60_aten__permute_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__transpose_6_0_id;
    unsigned char g_0__transpose_6_0_params[] = {0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0,
                                                 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_14_id_58_aten__view},
                   {g_0_tensor_15_id_60_aten__permute},
                   (void*)g_0__transpose_6_0_params,
                   24,
                   "g_0__transpose_6_0",
                   0 /*graphIndex*/,
                   &g_0__transpose_6_0_id);

    /*************
     * g_0__masked_batch_gemm_7_0 node
     * inputs:
     *     g_0_tensor_15_id_60_aten__permute[64, 512, 16, 1] (dtype=float32)
     *     g_0_tensor_13_id_64_aten__permute[512, 64, 16, 1] (dtype=float32)
     *     g_0_tensor_16[13, 512, 1, 1] (dtype=float32)
     *     g_0_tensor_17[512, 13, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_18_id_76_hpu__masked_batch_gemm[512, 512, 16, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_13_id_64_aten__permute tensor
    unsigned g_0_tensor_13_id_64_aten__permute_max_sizes[] = {512, 64, 16, 1};
    unsigned g_0_tensor_13_id_64_aten__permute_min_sizes[] = {512, 64, 16, 1};
    unsigned g_0_tensor_13_id_64_aten__permute             = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               false,
                                                               "g_0_tensor_13_id_64_aten__permute",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               g_0_tensor_13_id_64_aten__permute_max_sizes,
                                                               4,
                                                               syn_type_single,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_tensor_13_id_64_aten__permute_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_16 tensor
    unsigned g_0_tensor_16_max_sizes[] = {13, 512, 1, 1};
    unsigned g_0_tensor_16_min_sizes[] = {13, 512, 1, 1};
    unsigned g_0_tensor_16             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_16",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_16_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_16_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_17 tensor
    unsigned g_0_tensor_17_max_sizes[] = {512, 13, 1, 1};
    unsigned g_0_tensor_17_min_sizes[] = {512, 13, 1, 1};
    unsigned g_0_tensor_17             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_17",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_17_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_17_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_18_id_76_hpu__masked_batch_gemm tensor
    unsigned g_0_tensor_18_id_76_hpu__masked_batch_gemm_max_sizes[] = {512, 512, 16, 1};
    unsigned g_0_tensor_18_id_76_hpu__masked_batch_gemm_min_sizes[] = {512, 512, 16, 1};
    unsigned g_0_tensor_18_id_76_hpu__masked_batch_gemm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_18_id_76_hpu__masked_batch_gemm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_18_id_76_hpu__masked_batch_gemm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_18_id_76_hpu__masked_batch_gemm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__masked_batch_gemm_7_0_id;
    unsigned char g_0__masked_batch_gemm_7_0_params[] = {0, 0};
    addNodeToGraph("masked_batch_gemm",
                   {g_0_tensor_15_id_60_aten__permute, g_0_tensor_13_id_64_aten__permute, g_0_tensor_16, g_0_tensor_17},
                   {g_0_tensor_18_id_76_hpu__masked_batch_gemm},
                   (void*)g_0__masked_batch_gemm_7_0_params,
                   2,
                   "g_0__masked_batch_gemm_7_0",
                   0 /*graphIndex*/,
                   &g_0__masked_batch_gemm_7_0_id);

    /*************
     * g_0__transpose_4_0 node
     * inputs:
     *     g_0_tensor_12_id_62_aten__view[64, 16, 512, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_13_id_64_aten__permute[512, 64, 16, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_12_id_62_aten__view tensor
    unsigned      g_0_tensor_12_id_62_aten__view_max_sizes[] = {64, 16, 512, 1};
    unsigned      g_0_tensor_12_id_62_aten__view_min_sizes[] = {64, 16, 512, 1};
    unsigned      g_0_tensor_12_id_62_aten__view             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            false,
                                                            "g_0_tensor_12_id_62_aten__view",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_12_id_62_aten__view_max_sizes,
                                                            4,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_12_id_62_aten__view_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__transpose_4_0_id;
    unsigned char g_0__transpose_4_0_params[] = {2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                                 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_12_id_62_aten__view},
                   {g_0_tensor_13_id_64_aten__permute},
                   (void*)g_0__transpose_4_0_params,
                   24,
                   "g_0__transpose_4_0",
                   0 /*graphIndex*/,
                   &g_0__transpose_4_0_id);

    /*************
     * g_0__reshape_3_0 node
     * inputs:
     *     g_0_tensor_7_id_51_key_aten__linear[1024, 512, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_12_id_62_aten__view[64, 16, 512, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_7_id_51_key_aten__linear tensor
    unsigned  g_0_tensor_7_id_51_key_aten__linear_max_sizes[] = {1024, 512, 1};
    unsigned  g_0_tensor_7_id_51_key_aten__linear_min_sizes[] = {1024, 512, 1};
    unsigned  g_0_tensor_7_id_51_key_aten__linear             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 true,
                                                                 "g_0_tensor_7_id_51_key_aten__linear",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_7_id_51_key_aten__linear_max_sizes,
                                                                 3,
                                                                 syn_type_single,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_7_id_51_key_aten__linear_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__reshape_3_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_7_id_51_key_aten__linear},
                   {g_0_tensor_12_id_62_aten__view},
                   nullptr,
                   0,
                   "g_0__reshape_3_0",
                   0 /*graphIndex*/,
                   &g_0__reshape_3_0_id);

    /*************
     * g_0_key_batch_gemm_1_0 node
     * inputs:
     *     g_0_tensor_4[1024, 512, 1] (dtype=float32)
     *     g_0_tensor_5[1024, 1024] (dtype=float32)
     *     g_0_tensor_6[1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_7_id_51_key_aten__linear[1024, 512, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_4 tensor
    unsigned g_0_tensor_4_max_sizes[] = {1024, 512, 1};
    unsigned g_0_tensor_4_min_sizes[] = {1024, 512, 1};
    unsigned g_0_tensor_4             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_4",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_4_max_sizes,
                                          3,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_4_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_5 tensor
    unsigned g_0_tensor_5_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_5_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_5             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_5",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_5_max_sizes,
                                          2,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_5_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_6 tensor
    unsigned      g_0_tensor_6_max_sizes[] = {1024};
    unsigned      g_0_tensor_6_min_sizes[] = {1024};
    unsigned      g_0_tensor_6             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_6",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_6_max_sizes,
                                          1,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_6_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_key_batch_gemm_1_0_id;
    unsigned char g_0_key_batch_gemm_1_0_params[] = {0, 1};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_4, g_0_tensor_5, g_0_tensor_6},
                   {g_0_tensor_7_id_51_key_aten__linear},
                   (void*)g_0_key_batch_gemm_1_0_params,
                   2,
                   "g_0_key_batch_gemm_1_0",
                   0 /*graphIndex*/,
                   &g_0_key_batch_gemm_1_0_id);

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_18_id_76_hpu__masked_batch_gemm});
}
TEST_F_GC(SynGaudiTwoRunCompareTest, maxpool_producer_with_shape_tensor_ASIC_CI, {synDeviceGaudi2})
{
    const char* dedwInputLayouts[]  = {"WHCN", "WHCN"};
    const char* dedwOutputLayouts[] = {"SRCK"};

    const char* dedxInputLayouts[]  = {"WHCN", "SRCK", "WHCN"};
    const char* dedxOutputLayouts[] = {"WHCN"};

    const char* maxpoolInputLayouts[]  = {"WHCN", "WHCN", "WHCN"};
    const char* maxpoolOutputLayouts[] = {"WHCN"};

    unsigned g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_max_sizes[] =
        {19, 13, 256, 2};
    unsigned g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_min_sizes[] =
        {18, 12, 256, 2};
    unsigned g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward = createTensors(
        1,
        INPUT_TENSOR,
        true,
        "g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_275_max_sizes[] = {19, 13, 256, 2};
    unsigned g_0_tensor_275_min_sizes[] = {18, 12, 256, 2};
    unsigned g_0_tensor_275             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_275",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_275_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_275_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes
            [] = {3, 3, 256, 256};
    unsigned
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes
            [] = {3, 3, 256, 256};
    unsigned g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_id;
    unsigned char g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0,   1,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 109, 46, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward, g_0_tensor_275},
        {g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable},
        (void*)g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_params,
        104,
        "g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0",
        0 /*graphIndex*/,
        &g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_id,
        dedwInputLayouts,
        dedwOutputLayouts);

    unsigned g_0_tensor_278_max_sizes[] = {19, 13, 256, 2};
    unsigned g_0_tensor_278_min_sizes[] = {18, 12, 256, 2};
    unsigned g_0_tensor_278             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_278",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_278_max_sizes,
                                            4,
                                            syn_type_uint32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_278_min_sizes,
                                            synTensorType::SHAPE_TENSOR)[0];

    unsigned
        g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes
            [] = {19, 13, 256, 2};
    unsigned
        g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes
            [] = {18, 12, 256, 2};
    unsigned g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_410_max_sizes[] = {19, 13, 256, 2};
    unsigned g_0_tensor_410_min_sizes[] = {18, 12, 256, 2};
    unsigned g_0_tensor_410             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_410",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_410_max_sizes,
                                            4,
                                            syn_type_int16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_410_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_411_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_411_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_411             = createTensors(1,
                                            INPUT_TENSOR,
                                            false,
                                            "g_0_tensor_411",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_411_max_sizes,
                                            4,
                                            syn_type_uint32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_411_min_sizes,
                                            synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_max_sizes[] =
        {38, 25, 256, 2};
    unsigned g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_min_sizes[] =
        {36, 24, 256, 2};
    unsigned g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_id;
    unsigned char g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_params[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "maxpool_2d_bwd_bf16",
        {g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
         g_0_tensor_410,
         g_0_tensor_411},
        {g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward},
        (void*)g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_params,
        44,
        "g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0",
        0 /*graphIndex*/,
        &g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_id,
        maxpoolInputLayouts,
        maxpoolOutputLayouts);

    unsigned g_0_tensor_416_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_416_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_416             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_416",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_416_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_416_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes[] =
        {3, 3, 256, 256};
    unsigned g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes[] =
        {3, 3, 256, 256};
    unsigned g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_backbone_fpn_output5_dedw_10101_0_id;
    unsigned char g_0_gradient_backbone_fpn_output5_dedw_10101_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0,  1,   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 39, 166, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward, g_0_tensor_416},
        {g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable},
        (void*)g_0_gradient_backbone_fpn_output5_dedw_10101_0_params,
        104,
        "g_0_gradient_backbone_fpn_output5_dedw_10101_0",
        0 /*graphIndex*/,
        &g_0_gradient_backbone_fpn_output5_dedw_10101_0_id,
        dedwInputLayouts,
        dedwOutputLayouts);

    unsigned g_0_tensor_417_max_sizes[] = {3, 3, 256, 256};
    unsigned g_0_tensor_417_min_sizes[] = {3, 3, 256, 256};
    unsigned g_0_tensor_417             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_417",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_417_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_417_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_419_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_419_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_419             = createTensors(1,
                                            INPUT_TENSOR,
                                            false,
                                            "g_0_tensor_419",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_419_max_sizes,
                                            4,
                                            syn_type_uint32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_419_min_sizes,
                                            synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes[] =
        {38, 25, 256, 2};
    unsigned g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes[] =
        {36, 24, 256, 2};
    unsigned g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_backbone_fpn_output5_dedx_10102_0_id;
    unsigned char g_0_gradient_backbone_fpn_output5_dedx_10102_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward,
                    g_0_tensor_417,
                    g_0_tensor_419},
                   {g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable},
                   (void*)g_0_gradient_backbone_fpn_output5_dedx_10102_0_params,
                   104,
                   "g_0_gradient_backbone_fpn_output5_dedx_10102_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_backbone_fpn_output5_dedx_10102_0_id,
                   dedxInputLayouts,
                   dedxOutputLayouts);

    setActualSizes(
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward,
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_max_sizes);
    setActualSizes(
        g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
        g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes);
    setActualSizes(g_0_tensor_275, g_0_tensor_275_max_sizes);
    setActualSizes(
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes);
    setActualSizes(g_0_tensor_410, g_0_tensor_410_max_sizes);
    setActualSizes(g_0_tensor_416, g_0_tensor_416_max_sizes);
    setActualSizes(g_0_tensor_417, g_0_tensor_417_max_sizes);
    setActualSizes(
        g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable,
        g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes);
    setActualSizes(g_0_tensor_411, g_0_tensor_411_max_sizes);
    setActualSizes(g_0_tensor_278, g_0_tensor_278_max_sizes);
    setActualSizes(g_0_tensor_419, g_0_tensor_419_max_sizes);

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "-1");
    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults(
        {g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
         g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable,
         g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable});
}
TEST_F_GC(SynGaudiTwoRunCompareTest, shared_mme_sliced_on_common_dim_with_non_shared_unsliceable_producer_ASIC_CI)
{
    // Graph #0

    /*************
     * g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n12_0 node
     * inputs:
     *     g_0_t489_readvariableop_42_0[768, 768] (dtype=float32)
     * outputs:
     *     g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0[768, 768] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t489_readvariableop_42_0 tensor
    unsigned g_0_t489_readvariableop_42_0_max_sizes[] = {768,768};
    unsigned g_0_t489_readvariableop_42_0_min_sizes[] = {768,768};
    unsigned g_0_t489_readvariableop_42_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_t489_readvariableop_42_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_t489_readvariableop_42_0_max_sizes,
                                                      2,
                                                      syn_type_single,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_t489_readvariableop_42_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0 tensor
    unsigned g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_max_sizes[] = {768,768};
    unsigned g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_min_sizes[] = {768,768};
    unsigned g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0 = createTensors(1,
                                                                                                                        OUTPUT_TENSOR,
                                                                                                                        false,
                                                                                                                        "g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0",
                                                                                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                                        nullptr,
                                                                                                                        g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_max_sizes,
                                                                                                                        2,
                                                                                                                        syn_type_bf16,
                                                                                                                        nullptr,
                                                                                                                        0,
                                                                                                                        0,
                                                                                                                        nullptr,
                                                                                                                        false,
                                                                                                                        g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_min_sizes,
                                                                                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n12_0_id;
    addNodeToGraph("cast_f32_to_bf16", {g_0_t489_readvariableop_42_0}, {g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0}, nullptr, 0, "g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n12_0", 0 /*graphIndex*/, &g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n12_0_id);

    /*************
     * g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n174_0 node
     * inputs:
     *     g_0_t951_bert_embeddings_Reshape_4_0[768, 128, 1] (dtype=bf16)
     *     g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0[768, 128, 32] (dtype=bf16)
     * outputs:
     *     g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0[768, 128, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t951_bert_embeddings_Reshape_4_0 tensor
    unsigned g_0_t951_bert_embeddings_Reshape_4_0_max_sizes[] = {768,128,1};
    unsigned g_0_t951_bert_embeddings_Reshape_4_0_min_sizes[] = {768,128,1};
    unsigned g_0_t951_bert_embeddings_Reshape_4_0 = createTensors(1,
                                                              INPUT_TENSOR,
                                                              true,
                                                              "g_0_t951_bert_embeddings_Reshape_4_0",
                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                              nullptr,
                                                              g_0_t951_bert_embeddings_Reshape_4_0_max_sizes,
                                                              3,
                                                              syn_type_bf16,
                                                              nullptr,
                                                              0,
                                                              0,
                                                              nullptr,
                                                              false,
                                                              g_0_t951_bert_embeddings_Reshape_4_0_min_sizes,
                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0 tensor
    unsigned g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_max_sizes[] = {768,128,32};
    unsigned g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_min_sizes[] = {768,128,32};
    unsigned g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0 = createTensors(1,
                                                                                                   INPUT_TENSOR,
                                                                                                   true,
                                                                                                   "g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0",
                                                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                   nullptr,
                                                                                                   g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_max_sizes,
                                                                                                   3,
                                                                                                   syn_type_bf16,
                                                                                                   nullptr,
                                                                                                   0,
                                                                                                   0,
                                                                                                   nullptr,
                                                                                                   false,
                                                                                                   g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_min_sizes,
                                                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0 tensor
    unsigned g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_max_sizes[] = {768,128,32};
    unsigned g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_min_sizes[] = {768,128,32};
    unsigned g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0 = createTensors(1,
                                                                                            OUTPUT_TENSOR,
                                                                                            false,
                                                                                            "g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0",
                                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                            nullptr,
                                                                                            g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_max_sizes,
                                                                                            3,
                                                                                            syn_type_bf16,
                                                                                            nullptr,
                                                                                            0,
                                                                                            0,
                                                                                            nullptr,
                                                                                            false,
                                                                                            g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_min_sizes,
                                                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n174_0_id;
    addNodeToGraph("add_fwd_bf16", {g_0_t951_bert_embeddings_Reshape_4_0, g_0_t948_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0}, {g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0}, nullptr, 0, "g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n174_0", 0 /*graphIndex*/, &g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n174_0_id);

    /*************
     * g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n175_0 node
     * inputs:
     *     g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0[768, 128, 32] (dtype=bf16)
     *     g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768,128,1,32};
    unsigned g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768,128,1,32};
    unsigned g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                            INPUT_TENSOR,
                                                                            true,
                                                                            "g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm",
                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                            nullptr,
                                                                            g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                            4,
                                                                            syn_type_uint32,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                            synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768,128,1,32};
    unsigned g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768,128,1,32};
    unsigned g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                            OUTPUT_TENSOR,
                                                                            false,
                                                                            "g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm",
                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                            nullptr,
                                                                            g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                            4,
                                                                            syn_type_bf16,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n175_0_id;
    addNodeToGraph("reshape", {g_0_t953_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0, g_0_t958_bert_embeddings_LayerNorm_HabanaLayerNorm}, {g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm}, nullptr, 0, "g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n175_0", 0 /*graphIndex*/, &g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n175_0_id);

    /*************
     * g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n179_0 node
     * inputs:
     *     g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     *     g_0_t495_readvariableop_20_0[768] (dtype=float32)
     *     g_0_t494_readvariableop_16_0[768] (dtype=float32)
     * outputs:
     *     g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     *     g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm[1, 128, 1, 32] (dtype=float32)
     *     g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm[1, 128, 1, 32] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t495_readvariableop_20_0 tensor
    unsigned g_0_t495_readvariableop_20_0_max_sizes[] = {768};
    unsigned g_0_t495_readvariableop_20_0_min_sizes[] = {768};
    unsigned g_0_t495_readvariableop_20_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_t495_readvariableop_20_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_t495_readvariableop_20_0_max_sizes,
                                                      1,
                                                      syn_type_single,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_t495_readvariableop_20_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t494_readvariableop_16_0 tensor
    unsigned g_0_t494_readvariableop_16_0_max_sizes[] = {768};
    unsigned g_0_t494_readvariableop_16_0_min_sizes[] = {768};
    unsigned g_0_t494_readvariableop_16_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_t494_readvariableop_16_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_t494_readvariableop_16_0_max_sizes,
                                                      1,
                                                      syn_type_single,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_t494_readvariableop_16_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768,128,1,32};
    unsigned g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768,128,1,32};
    unsigned g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                            OUTPUT_TENSOR,
                                                                            false,
                                                                            "g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm",
                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                            nullptr,
                                                                            g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                            4,
                                                                            syn_type_bf16,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {1,128,1,32};
    unsigned g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {1,128,1,32};
    unsigned g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                            OUTPUT_TENSOR,
                                                                            true,
                                                                            "g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm",
                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                            nullptr,
                                                                            g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                            4,
                                                                            syn_type_single,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {1,128,1,32};
    unsigned g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {1,128,1,32};
    unsigned g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                            OUTPUT_TENSOR,
                                                                            true,
                                                                            "g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm",
                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                            nullptr,
                                                                            g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                            4,
                                                                            syn_type_single,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n179_0_id;
    unsigned char g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n179_0_params[] = {1,0,0,0,111,18,131,58};
    addNodeToGraph("layer_norm_fwd_bf16", {g_0_t957_bert_embeddings_LayerNorm_HabanaLayerNorm, g_0_t495_readvariableop_20_0, g_0_t494_readvariableop_16_0}, {g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm, g_0_t961_bert_embeddings_LayerNorm_HabanaLayerNorm, g_0_t963_bert_embeddings_LayerNorm_HabanaLayerNorm}, (void*)g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n179_0_params, 8, "g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n179_0", 0 /*graphIndex*/, &g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n179_0_id);

    /*************
     * g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n176_0 node
     * inputs:
     *     g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     *     g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 32] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0[768, 128, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768,128,32};
    unsigned g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768,128,32};
    unsigned g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                            INPUT_TENSOR,
                                                                            true,
                                                                            "g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm",
                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                            nullptr,
                                                                            g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                            3,
                                                                            syn_type_uint32,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                            synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {768,128,32};
    unsigned g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {768,128,32};
    unsigned g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0 = createTensors(1,
                                                                              OUTPUT_TENSOR,
                                                                              false,
                                                                              "g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0",
                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                              nullptr,
                                                                              g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0_max_sizes,
                                                                              3,
                                                                              syn_type_bf16,
                                                                              nullptr,
                                                                              0,
                                                                              0,
                                                                              nullptr,
                                                                              false,
                                                                              g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0_min_sizes,
                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n176_0_id;
    addNodeToGraph("reshape", {g_0_t959_bert_embeddings_LayerNorm_HabanaLayerNorm, g_0_t960_bert_embeddings_LayerNorm_HabanaLayerNorm}, {g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0}, nullptr, 0, "g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n176_0", 0 /*graphIndex*/, &g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n176_0_id);

    /*************
     * g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n181_0 node
     * inputs:
     *     g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0[768, 128, 32] (dtype=bf16)
     *     g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0[1] (dtype=int32)
     * outputs:
     *     g_0_t967_bert_embeddings_dropout_Mul_1_0[768, 128, 32] (dtype=bf16)
     *     g_0_t968_bert_embeddings_dropout_Mul_1_0[768, 128, 32] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0 tensor
    unsigned g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_max_sizes[] = {1};
    unsigned g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_min_sizes[] = {1};
    unsigned g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0 = createTensors(1,
                                                                                                                                   INPUT_TENSOR,
                                                                                                                                   true,
                                                                                                                                   "g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0",
                                                                                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                                                   nullptr,
                                                                                                                                   g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_max_sizes,
                                                                                                                                   1,
                                                                                                                                   syn_type_int32,
                                                                                                                                   nullptr,
                                                                                                                                   0,
                                                                                                                                   0,
                                                                                                                                   nullptr,
                                                                                                                                   false,
                                                                                                                                   g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_min_sizes,
                                                                                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_t967_bert_embeddings_dropout_Mul_1_0 tensor
    unsigned g_0_t967_bert_embeddings_dropout_Mul_1_0_max_sizes[] = {768,128,32};
    unsigned g_0_t967_bert_embeddings_dropout_Mul_1_0_min_sizes[] = {768,128,32};
    unsigned g_0_t967_bert_embeddings_dropout_Mul_1_0 = createTensors(1,
                                                                  OUTPUT_TENSOR,
                                                                  false,
                                                                  "g_0_t967_bert_embeddings_dropout_Mul_1_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t967_bert_embeddings_dropout_Mul_1_0_max_sizes,
                                                                  3,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t967_bert_embeddings_dropout_Mul_1_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    // create g_0_t968_bert_embeddings_dropout_Mul_1_0 tensor
    unsigned g_0_t968_bert_embeddings_dropout_Mul_1_0_max_sizes[] = {768,128,32};
    unsigned g_0_t968_bert_embeddings_dropout_Mul_1_0_min_sizes[] = {768,128,32};
    unsigned g_0_t968_bert_embeddings_dropout_Mul_1_0 = createTensors(1,
                                                                  OUTPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t968_bert_embeddings_dropout_Mul_1_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t968_bert_embeddings_dropout_Mul_1_0_max_sizes,
                                                                  3,
                                                                  syn_type_int8,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t968_bert_embeddings_dropout_Mul_1_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n181_0_id;
    unsigned char g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n181_0_params[] = {205,204,204,61,0,0,0,0};
    addNodeToGraph("dropout_fwd_bf16", {g_0_t954_bert_embeddings_LayerNorm_HabanaLayerNorm_0, g_0_t965_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0}, {g_0_t967_bert_embeddings_dropout_Mul_1_0, g_0_t968_bert_embeddings_dropout_Mul_1_0}, (void*)g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n181_0_params, 8, "g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n181_0", 0 /*graphIndex*/, &g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n181_0_id);

    /*************
     * g_0_bert_encoder_Reshape_1_reshape_n182_0 node
     * inputs:
     *     g_0_t967_bert_embeddings_dropout_Mul_1_0[768, 128, 32] (dtype=bf16)
     *     g_0_t970_bert_encoder_Reshape_1[768, 4096] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t969_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t970_bert_encoder_Reshape_1 tensor
    unsigned g_0_t970_bert_encoder_Reshape_1_max_sizes[] = {768,4096};
    unsigned g_0_t970_bert_encoder_Reshape_1_min_sizes[] = {768,4096};
    unsigned g_0_t970_bert_encoder_Reshape_1 = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_t970_bert_encoder_Reshape_1",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_t970_bert_encoder_Reshape_1_max_sizes,
                                                         2,
                                                         syn_type_uint32,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_t970_bert_encoder_Reshape_1_min_sizes,
                                                         synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t969_bert_encoder_Reshape_1_0 tensor
    unsigned g_0_t969_bert_encoder_Reshape_1_0_max_sizes[] = {768,4096};
    unsigned g_0_t969_bert_encoder_Reshape_1_0_min_sizes[] = {768,4096};
    unsigned g_0_t969_bert_encoder_Reshape_1_0 = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_t969_bert_encoder_Reshape_1_0",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_t969_bert_encoder_Reshape_1_0_max_sizes,
                                                           2,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_t969_bert_encoder_Reshape_1_0_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_Reshape_1_reshape_n182_0_id;
    addNodeToGraph("reshape", {g_0_t967_bert_embeddings_dropout_Mul_1_0, g_0_t970_bert_encoder_Reshape_1}, {g_0_t969_bert_encoder_Reshape_1_0}, nullptr, 0, "g_0_bert_encoder_Reshape_1_reshape_n182_0", 0 /*graphIndex*/, &g_0_bert_encoder_Reshape_1_reshape_n182_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n194_0 node
     * inputs:
     *     g_0_t969_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     *     g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0[768, 768] (dtype=bf16)
     * outputs:
     *     g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0 tensor
    unsigned g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0_max_sizes[] = {768,4096};
    unsigned g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0_min_sizes[] = {768,4096};
    unsigned g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0 = createTensors(1,
                                                                                     OUTPUT_TENSOR,
                                                                                     true,
                                                                                     "g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0",
                                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                     nullptr,
                                                                                     g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0_max_sizes,
                                                                                     2,
                                                                                     syn_type_bf16,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0_min_sizes,
                                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n194_0_id;
    unsigned char g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n194_0_params[] = {0,0};
    addNodeToGraph("gemm", {g_0_t969_bert_encoder_Reshape_1_0, g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0}, {g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0}, (void*)g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n194_0_params, 2, "g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n194_0", 0 /*graphIndex*/, &g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n194_0_id);

    /*************
     * g_0_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_gemm_n3472_0 node
     * inputs:
     *     g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0[768, 4096] (dtype=bf16)
     *     g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0[768, 768] (dtype=bf16)
     * outputs:
     *     g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0 tensor
    unsigned g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0_max_sizes[] = {768,4096};
    unsigned g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0_min_sizes[] = {768,4096};
    unsigned g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0 = createTensors(1,
                                                                                                            INPUT_TENSOR,
                                                                                                            true,
                                                                                                            "g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0",
                                                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                            nullptr,
                                                                                                            g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0_max_sizes,
                                                                                                            2,
                                                                                                            syn_type_bf16,
                                                                                                            nullptr,
                                                                                                            0,
                                                                                                            0,
                                                                                                            nullptr,
                                                                                                            false,
                                                                                                            g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0_min_sizes,
                                                                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0 tensor
    unsigned g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0_max_sizes[] = {768,4096};
    unsigned g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0_min_sizes[] = {768,4096};
    unsigned g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0 = createTensors(1,
                                                                                                              OUTPUT_TENSOR,
                                                                                                              true,
                                                                                                              "g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0",
                                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                              nullptr,
                                                                                                              g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0_max_sizes,
                                                                                                              2,
                                                                                                              syn_type_bf16,
                                                                                                              nullptr,
                                                                                                              0,
                                                                                                              0,
                                                                                                              nullptr,
                                                                                                              false,
                                                                                                              g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0_min_sizes,
                                                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_gemm_n3472_0_id;
    unsigned char g_0_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_gemm_n3472_0_params[] = {0,1};
    addNodeToGraph("gemm", {g_0_t5858_gradients_1_bert_encoder_layer_0_attention_self_Reshape_2_grad_Reshape_0, g_0_t634_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0}, {g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0}, (void*)g_0_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_gemm_n3472_0_params, 2, "g_0_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_gemm_n3472_0", 0 /*graphIndex*/, &g_0_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_gemm_n3472_0_id);


    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_t986_bert_encoder_layer_0_attention_self_value_MatMul_0,g_0_t5860_gradients_1_bert_encoder_layer_0_attention_self_value_MatMul_grad_MatMul_0});
}

TEST_F_GC(SynGaudiTwoRunCompareTest,
          sliced_conv_with_tensor_view_and_external_concat_ASIC,
          {synDeviceGaudi, synDeviceGaudi2})
{
    const char* convInputLayouts[]  = {"WHDCN", "SRQCK"};
    const char* convOutputLayouts[] = {"WHDCN"};

    unsigned g_0_tensor_19_id_2945_input_block_conv2_0_aten__convolution_overrideable_max_sizes[] = {128,
                                                                                                     128,
                                                                                                     128,
                                                                                                     32,
                                                                                                     2};
    unsigned g_0_tensor_19_id_2945_input_block_conv2_0_aten__convolution_overrideable_min_sizes[] = {128,
                                                                                                     128,
                                                                                                     128,
                                                                                                     32,
                                                                                                     2};
    unsigned g_0_tensor_19_id_2945_input_block_conv2_0_aten__convolution_overrideable =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_19_id_2945_input_block_conv2_0_aten__convolution_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_19_id_2945_input_block_conv2_0_aten__convolution_overrideable_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_19_id_2945_input_block_conv2_0_aten__convolution_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_21_max_sizes[] = {32};
    unsigned g_0_tensor_21_min_sizes[] = {32};
    unsigned g_0_tensor_21             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_21",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_21_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_21_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_20_max_sizes[] = {32};
    unsigned g_0_tensor_20_min_sizes[] = {32};
    unsigned g_0_tensor_20             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_20",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_20_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_20_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm_max_sizes[] = {128, 128, 128, 32, 2};
    unsigned g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm_min_sizes[] = {128, 128, 128, 32, 2};
    unsigned g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_23_id_2949_input_block_conv2_1_hpu__instance_norm_max_sizes[] = {32, 2};
    unsigned g_0_tensor_23_id_2949_input_block_conv2_1_hpu__instance_norm_min_sizes[] = {32, 2};
    unsigned g_0_tensor_23_id_2949_input_block_conv2_1_hpu__instance_norm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_23_id_2949_input_block_conv2_1_hpu__instance_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_23_id_2949_input_block_conv2_1_hpu__instance_norm_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_23_id_2949_input_block_conv2_1_hpu__instance_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_24_id_2951_input_block_conv2_1_hpu__instance_norm_max_sizes[] = {32, 2};
    unsigned g_0_tensor_24_id_2951_input_block_conv2_1_hpu__instance_norm_min_sizes[] = {32, 2};
    unsigned g_0_tensor_24_id_2951_input_block_conv2_1_hpu__instance_norm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_24_id_2951_input_block_conv2_1_hpu__instance_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_24_id_2951_input_block_conv2_1_hpu__instance_norm_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_24_id_2951_input_block_conv2_1_hpu__instance_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    const char* inInputLayouts[]  = {"WHDCN", "", ""};
    const char* inOutputLayouts[] = {"WHDCN", "CN", "CN"};

    synNodeId     g_0_input_block_conv2_1_instance_norm_fwd_bf16_993_0_id;
    unsigned char g_0_input_block_conv2_1_instance_norm_fwd_bf16_993_0_params[] = {102, 102, 102, 63, 172, 197, 39, 55};
    addNodeToGraph(
        "instance_norm_fwd_bf16",
        {g_0_tensor_19_id_2945_input_block_conv2_0_aten__convolution_overrideable, g_0_tensor_21, g_0_tensor_20},
        {g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm,
         g_0_tensor_23_id_2949_input_block_conv2_1_hpu__instance_norm,
         g_0_tensor_24_id_2951_input_block_conv2_1_hpu__instance_norm},
        (void*)g_0_input_block_conv2_1_instance_norm_fwd_bf16_993_0_params,
        8,
        "g_0_input_block_conv2_1_instance_norm_fwd_bf16_993_0",
        0 /*graphIndex*/,
        &g_0_input_block_conv2_1_instance_norm_fwd_bf16_993_0_id,
        inInputLayouts,
        inOutputLayouts);

    unsigned g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu_max_sizes[] = {128, 128, 128, 32, 2};
    unsigned g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu_min_sizes[] = {128, 128, 128, 32, 2};
    unsigned g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_input_block_conv2_2_relu_fwd_bf16_994_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_tensor_22_id_2947_input_block_conv2_1_hpu__instance_norm},
                   {g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu},
                   nullptr,
                   0,
                   "g_0_input_block_conv2_2_relu_fwd_bf16_994_0",
                   0 /*graphIndex*/,
                   &g_0_input_block_conv2_2_relu_fwd_bf16_994_0_id);

    unsigned g_0_tensor_37_id_2957_0_conv1_0_hpu__cast_max_sizes[] = {3, 3, 3, 32, 64};
    unsigned g_0_tensor_37_id_2957_0_conv1_0_hpu__cast_min_sizes[] = {3, 3, 3, 32, 64};
    unsigned g_0_tensor_37_id_2957_0_conv1_0_hpu__cast =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_37_id_2957_0_conv1_0_hpu__cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_37_id_2957_0_conv1_0_hpu__cast_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_37_id_2957_0_conv1_0_hpu__cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable_max_sizes[] = {64, 64, 64, 64, 2};
    unsigned g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable_min_sizes[] = {64, 64, 64, 64, 2};
    unsigned g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_0_conv1_0_spatial_convolution3d_1001_0_id;
    unsigned char g_0_0_conv1_0_spatial_convolution3d_1001_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution3d",
                   {g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu, g_0_tensor_37_id_2957_0_conv1_0_hpu__cast},
                   {g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable},
                   (void*)g_0_0_conv1_0_spatial_convolution3d_1001_0_params,
                   128,
                   "g_0_0_conv1_0_spatial_convolution3d_1001_0",
                   0 /*graphIndex*/,
                   &g_0_0_conv1_0_spatial_convolution3d_1001_0_id,
                   convInputLayouts,
                   convOutputLayouts);

    unsigned g_0_tensor_226_id_3239_4_upsample_conv_0_aten__convolution_overrideable_max_sizes[] = {128,
                                                                                                    128,
                                                                                                    128,
                                                                                                    32,
                                                                                                    2};
    unsigned g_0_tensor_226_id_3239_4_upsample_conv_0_aten__convolution_overrideable_min_sizes[] = {128,
                                                                                                    128,
                                                                                                    128,
                                                                                                    32,
                                                                                                    2};
    unsigned g_0_tensor_226_id_3239_4_upsample_conv_0_aten__convolution_overrideable =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_226_id_3239_4_upsample_conv_0_aten__convolution_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_226_id_3239_4_upsample_conv_0_aten__convolution_overrideable_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_226_id_3239_4_upsample_conv_0_aten__convolution_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_tensor_227_id_3241_4_aten__cat_max_sizes[] = {128, 128, 128, 64, 2};
    unsigned      g_0_tensor_227_id_3241_4_aten__cat_min_sizes[] = {128, 128, 128, 64, 2};
    unsigned      g_0_tensor_227_id_3241_4_aten__cat             = createTensors(1,
                                                                OUTPUT_TENSOR,
                                                                false,
                                                                "g_0_tensor_227_id_3241_4_aten__cat",
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                g_0_tensor_227_id_3241_4_aten__cat_max_sizes,
                                                                5,
                                                                syn_type_bf16,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_tensor_227_id_3241_4_aten__cat_min_sizes,
                                                                synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_4_concat_1096_0_id;
    unsigned char g_0_4_concat_1096_0_params[] = {3, 0, 0, 0};
    addNodeToGraph("concat",
                   {g_0_tensor_226_id_3239_4_upsample_conv_0_aten__convolution_overrideable,
                    g_0_tensor_25_id_2953_input_block_conv2_2_aten__relu},
                   {g_0_tensor_227_id_3241_4_aten__cat},
                   (void*)g_0_4_concat_1096_0_params,
                   4,
                   "g_0_4_concat_1096_0",
                   0 /*graphIndex*/,
                   &g_0_4_concat_1096_0_id);

    unsigned g_0_tensor_5_id_3246_4_conv1_0_hpu__cast_max_sizes[] = {3, 3, 3, 64, 32};
    unsigned g_0_tensor_5_id_3246_4_conv1_0_hpu__cast_min_sizes[] = {3, 3, 3, 64, 32};
    unsigned g_0_tensor_5_id_3246_4_conv1_0_hpu__cast =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_5_id_3246_4_conv1_0_hpu__cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_5_id_3246_4_conv1_0_hpu__cast_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_5_id_3246_4_conv1_0_hpu__cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable_max_sizes[] = {128, 128, 128, 32, 2};
    unsigned g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable_min_sizes[] = {128, 128, 128, 32, 2};
    unsigned g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_4_conv1_0_spatial_convolution3d_1097_0_id;
    unsigned char g_0_4_conv1_0_spatial_convolution3d_1097_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution3d",
                   {g_0_tensor_227_id_3241_4_aten__cat, g_0_tensor_5_id_3246_4_conv1_0_hpu__cast},
                   {g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable},
                   (void*)g_0_4_conv1_0_spatial_convolution3d_1097_0_params,
                   128,
                   "g_0_4_conv1_0_spatial_convolution3d_1097_0",
                   0 /*graphIndex*/,
                   &g_0_4_conv1_0_spatial_convolution3d_1097_0_id,
                   convInputLayouts,
                   convOutputLayouts);

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_38_id_2958_0_conv1_0_aten__convolution_overrideable,
                        g_0_tensor_228_id_3247_4_conv1_0_aten__convolution_overrideable});
}
TEST_F_GC(SynGaudiTwoRunCompareTest,
          pipeline_management_vision_bundle_all_producer_chains_affect_slicing_granularity_ASIC_CI)
{
    // Graph #0

    /*************
     * g_0_0_0_0_attn2_reshape_3387_0 node
     * inputs:
     *     g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity[64, 9216, 5, 2] (dtype=bf16)
     * outputs:
     *     g_0_tensor_317_id_18830_0_0_0_attn2_aten__view[64, 9216, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity tensor
    unsigned g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity_max_sizes[] = {64, 9216, 5, 2};
    unsigned g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity_min_sizes[] = {64, 9216, 5, 2};
    unsigned g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_317_id_18830_0_0_0_attn2_aten__view tensor
    unsigned g_0_tensor_317_id_18830_0_0_0_attn2_aten__view_max_sizes[] = {64, 9216, 10};
    unsigned g_0_tensor_317_id_18830_0_0_0_attn2_aten__view_min_sizes[] = {64, 9216, 10};
    unsigned g_0_tensor_317_id_18830_0_0_0_attn2_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_317_id_18830_0_0_0_attn2_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_317_id_18830_0_0_0_attn2_aten__view_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_317_id_18830_0_0_0_attn2_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_0_0_0_attn2_reshape_3387_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_316_id_18828_0_0_0_attn2_hpu__identity},
                   {g_0_tensor_317_id_18830_0_0_0_attn2_aten__view},
                   nullptr,
                   0,
                   "g_0_0_0_0_attn2_reshape_3387_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_reshape_3387_0_id);

    /*************
     * g_0_0_0_0_attn2_cast_bf16_to_f32_3388_0 node
     * inputs:
     *     g_0_tensor_317_id_18830_0_0_0_attn2_aten__view[64, 9216, 10] (dtype=bf16)
     * outputs:
     *     g_0_tensor_319[64, 9216, 10] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_319 tensor
    unsigned      g_0_tensor_319_max_sizes[] = {64, 9216, 10};
    unsigned      g_0_tensor_319_min_sizes[] = {64, 9216, 10};
    unsigned      g_0_tensor_319             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_319",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_319_max_sizes,
                                            3,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_319_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_0_0_0_attn2_cast_bf16_to_f32_3388_0_id;
    unsigned char g_0_0_0_0_attn2_cast_bf16_to_f32_3388_0_params[] = {4, 0, 0, 0};
    addNodeToGraph("cast_bf16_to_f32",
                   {g_0_tensor_317_id_18830_0_0_0_attn2_aten__view},
                   {g_0_tensor_319},
                   (void*)g_0_0_0_0_attn2_cast_bf16_to_f32_3388_0_params,
                   4,
                   "g_0_0_0_0_attn2_cast_bf16_to_f32_3388_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_cast_bf16_to_f32_3388_0_id);

    /*************
     * g_0_0_0_0_attn2_reshape_3374_0 node
     * inputs:
     *     g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity[64, 77, 5, 2] (dtype=bf16)
     * outputs:
     *     g_0_tensor_297_id_18850_0_0_0_attn2_aten__view[64, 77, 10] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity tensor
    unsigned g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity_max_sizes[] = {64, 77, 5, 2};
    unsigned g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity_min_sizes[] = {64, 77, 5, 2};
    unsigned g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_297_id_18850_0_0_0_attn2_aten__view tensor
    unsigned g_0_tensor_297_id_18850_0_0_0_attn2_aten__view_max_sizes[] = {64, 77, 10};
    unsigned g_0_tensor_297_id_18850_0_0_0_attn2_aten__view_min_sizes[] = {64, 77, 10};
    unsigned g_0_tensor_297_id_18850_0_0_0_attn2_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_297_id_18850_0_0_0_attn2_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_297_id_18850_0_0_0_attn2_aten__view_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_297_id_18850_0_0_0_attn2_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_0_0_0_attn2_reshape_3374_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_296_id_18848_0_0_0_attn2_hpu__identity},
                   {g_0_tensor_297_id_18850_0_0_0_attn2_aten__view},
                   nullptr,
                   0,
                   "g_0_0_0_0_attn2_reshape_3374_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_reshape_3374_0_id);

    /*************
     * g_0_0_0_0_attn2_cast_bf16_to_f32_3375_0 node
     * inputs:
     *     g_0_tensor_297_id_18850_0_0_0_attn2_aten__view[64, 77, 10] (dtype=bf16)
     * outputs:
     *     g_0_tensor_299[64, 77, 10] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_299 tensor
    unsigned      g_0_tensor_299_max_sizes[] = {64, 77, 10};
    unsigned      g_0_tensor_299_min_sizes[] = {64, 77, 10};
    unsigned      g_0_tensor_299             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_299",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_299_max_sizes,
                                            3,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_299_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_0_0_0_attn2_cast_bf16_to_f32_3375_0_id;
    unsigned char g_0_0_0_0_attn2_cast_bf16_to_f32_3375_0_params[] = {4, 0, 0, 0};
    addNodeToGraph("cast_bf16_to_f32",
                   {g_0_tensor_297_id_18850_0_0_0_attn2_aten__view},
                   {g_0_tensor_299},
                   (void*)g_0_0_0_0_attn2_cast_bf16_to_f32_3375_0_params,
                   4,
                   "g_0_0_0_0_attn2_cast_bf16_to_f32_3375_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_cast_bf16_to_f32_3375_0_id);

    /*************
     * g_0_0_0_0_attn2_constant_f32_3390_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_322[1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_322 tensor
    unsigned      g_0_tensor_322_max_sizes[] = {1};
    unsigned      g_0_tensor_322_min_sizes[] = {1};
    unsigned      g_0_tensor_322             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_322",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_322_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_322_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_0_0_0_attn2_constant_f32_3390_0_id;
    unsigned char g_0_0_0_0_attn2_constant_f32_3390_0_params[] = {0, 0, 0, 62};
    addNodeToGraph("constant_f32",
                   {},
                   {g_0_tensor_322},
                   (void*)g_0_0_0_0_attn2_constant_f32_3390_0_params,
                   4,
                   "g_0_0_0_0_attn2_constant_f32_3390_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_constant_f32_3390_0_id);

    /*************
     * g_0_0_0_0_attn2_mult_fwd_f32_3391_0 node
     * inputs:
     *     g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm[77, 9216, 10] (dtype=float32)
     *     g_0_tensor_322[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul[77, 9216, 10] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm tensor
    unsigned g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm_max_sizes[] = {77, 9216, 10};
    unsigned g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm_min_sizes[] = {77, 9216, 10};
    unsigned g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm_max_sizes,
                      3,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul tensor
    unsigned g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul_max_sizes[] = {77, 9216, 10};
    unsigned g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul_min_sizes[] = {77, 9216, 10};
    unsigned g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul_max_sizes,
                      3,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_0_0_0_attn2_mult_fwd_f32_3391_0_id;
    addNodeToGraph("mult_fwd_f32",
                   {g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm, g_0_tensor_322},
                   {g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul},
                   nullptr,
                   0,
                   "g_0_0_0_0_attn2_mult_fwd_f32_3391_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_mult_fwd_f32_3391_0_id);

    /*************
     * g_0_0_0_0_attn2_transpose_3376_0 node
     * inputs:
     *     g_0_tensor_299[64, 77, 10] (dtype=float32)
     * outputs:
     *     g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose[77, 64, 10] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose tensor
    unsigned g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose_max_sizes[] = {77, 64, 10};
    unsigned g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose_min_sizes[] = {77, 64, 10};
    unsigned g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose_max_sizes,
                      3,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_0_0_0_attn2_transpose_3376_0_id;
    unsigned char g_0_0_0_0_attn2_transpose_3376_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
                                                               0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_299},
                   {g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose},
                   (void*)g_0_0_0_0_attn2_transpose_3376_0_params,
                   24,
                   "g_0_0_0_0_attn2_transpose_3376_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_transpose_3376_0_id);

    /*************
     * g_0_0_0_0_attn2_batch_gemm_f32_3389_0 node
     * inputs:
     *     g_0_tensor_319[64, 9216, 10] (dtype=float32)
     *     g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose[77, 64, 10] (dtype=float32)
     * outputs:
     *     g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm[77, 9216, 10] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/
    synNodeId     g_0_0_0_0_attn2_batch_gemm_f32_3389_0_id;
    unsigned char g_0_0_0_0_attn2_batch_gemm_f32_3389_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_319, g_0_tensor_300_id_18870_0_0_0_attn2_aten__transpose},
                   {g_0_tensor_320_id_18872_0_0_0_attn2_aten__bmm},
                   (void*)g_0_0_0_0_attn2_batch_gemm_f32_3389_0_params,
                   2,
                   "g_0_0_0_0_attn2_batch_gemm_f32_3389_0",
                   0 /*graphIndex*/,
                   &g_0_0_0_0_attn2_batch_gemm_f32_3389_0_id);

    // base is forced to vision bundling policy
    addConfigurationToRun(FIRST_RUN, "PIPELINE_MANAGEMENT_FORCE_BUNDLIZER", "1");

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_321_id_18874_0_0_0_attn2_aten__mul});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, slice_leftover_test_ASIC_CI)
{
    unsigned yTensorSizes[] = {128, 36, 34, 32, 4};
    unsigned yTensor        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, yTensorSizes, 5);

    unsigned wTensorSizes[] = {128, 64, 2, 2, 2};
    unsigned wTensor        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wTensorSizes, 5);

    unsigned xTensorSizes[] = {64, 64, 64, 64, 4};
    unsigned xTensor        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xTensorSizes, 5);

    synConvolution3DParams dedx3dParams {};
    dedx3dParams.kernel[CONV_KERNEL_WIDTH]  = 2;
    dedx3dParams.kernel[CONV_KERNEL_HEIGHT] = 2;
    dedx3dParams.kernel[CONV_KERNEL_DEPTH]  = 2;
    dedx3dParams.stride[CONV_DIL_WIDTH]     = 2;
    dedx3dParams.stride[CONV_DIL_HEIGHT]    = 2;
    dedx3dParams.stride[CONV_DIL_DEPTH]     = 2;
    dedx3dParams.padding[CONV_PAD_LEFT]     = 5;
    dedx3dParams.padding[CONV_PAD_RIGHT]    = 4;
    dedx3dParams.padding[CONV_PAD_TOP]      = 3;
    dedx3dParams.padding[CONV_PAD_BOTTOM]   = 2;
    dedx3dParams.padding[CONV_PAD_FRONT]    = 1;
    dedx3dParams.padding[CONV_PAD_BACK]     = 0;

    addNodeToGraph(NodeFactory::deDx3DNodeTypeName, {yTensor, wTensor}, {xTensor}, &dedx3dParams, sizeof(dedx3dParams));

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "-1");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({xTensor});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, spatial_slicing_of_dedx_with_shape_tensor_ASIC_CI)
{
    synConvolutionParams params {};
    params.kH   = 3;
    params.kW   = 3;
    params.padT = 1;
    params.padB = 1;
    params.padL = 1;
    params.padR = 1;

    const unsigned batchSize = 2;

    const unsigned xH = 128;
    const unsigned xW = 128;
    const unsigned xC = 256;
    const unsigned yH = convOutputDimSize(xH, params.kH, params.dH, params.padT + params.padB, params.dilH);
    const unsigned yW = convOutputDimSize(xW, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned yC = xC;

    std::vector<unsigned> xSize = {xC, xW, xH, batchSize};
    std::vector<unsigned> wSize = {xC, xC, params.kW, params.kH};
    std::vector<unsigned> ySize = {yC, yW, yH, batchSize};

    unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        ySize.data(),
                                        ySize.size(),
                                        syn_type_single,
                                        nullptr,
                                        "dy");

    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     wSize.data(),
                                     wSize.size(),
                                     syn_type_single,
                                     nullptr,
                                     "w");

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        xSize.data(),
                                        xSize.size(),
                                        syn_type_single,
                                        nullptr,
                                        "dx");

    unsigned dxShape =
        createShapeTensor(INPUT_TENSOR, xSize.data(), xSize.data(), xSize.size(), syn_type_single, "dx_shape");

    addNodeToGraph(NodeFactory::deDxNodeTypeName, {dedy, w, dxShape}, {dedx}, &params, sizeof(params));

    // Use smaller SRAM to force spatial slicing
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "10000000");

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({dedx});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, bgemm_with_broadcast_on_operand_A_ASIC_CI)
{
    const unsigned batchSize = 13, commonDim = 384, height = 196, width = 768;
    unsigned       aSizes[]   = {height, commonDim};
    unsigned       bSizes[]   = {commonDim, width, batchSize};
    unsigned       outSizes[] = {width, height, batchSize};

    unsigned bgemmInA = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      2,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned bgemmInB = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInB",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      bSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      bSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned bgemmOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "bgemmOut",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      outSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      outSizes,
                                      synTensorType::DATA_TENSOR)[0];

    synGEMMParams params(true, true);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({bgemmOut});
}

// Validates accuracy when dynamic bgemm is flattened
class BatchGemmFlattenDynamicTest : public SynGaudiTwoRunCompareTest
{
};

TEST_F_GC(BatchGemmFlattenDynamicTest, batch_gemm_flatten_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned in0Sizes[] = {1024, 128, 32};
    unsigned in0MinSizes[] = {1024, 128, 1};
    unsigned in0           = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "in0",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 in0Sizes,
                                 3,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in0MinSizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned in1Sizes[] = {1024, 1024};
    unsigned in1        = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "gemm1In",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 in1Sizes,
                                 2,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in1Sizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned outSizes[]    = {1024, 128, 32};
    unsigned outMinSizes[] = {1024, 128, 1};
    unsigned out           = createTensors(1,
                                 OUTPUT_TENSOR,
                                 true,
                                 "gemm1Out",
                                 MEM_INIT_ALL_ZERO,
                                 nullptr,
                                 outSizes,
                                 3,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 outMinSizes,
                                 synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("batch_gemm", {in0, in1}, {out}, nullptr, 0, "BGEMM1");

    unsigned actualSizes[] = {1024, 128, 16};
    setActualSizes(in0, actualSizes);
    setActualSizes(out, actualSizes);
    addConfigurationToRun(FIRST_RUN, "ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING", "true");

    compareRunsResults({out});
}

// Test for bug SW-147931 fix in Mantaray ignoreNonTransposeLogicalsRelaxed
TEST_F_GC(SynGaudiTwoRunCompareTest, transpose_with_persistent_input_sram_calc_ASIC)
{
    // Graph #0

    /*************
     * g_0_3_attention_query_key_value_strided_view_6284_0 node
     * inputs:
     *     g_0_tensor_31__placeholder_0[576] (dtype=bf16)
     * outputs:
     *     g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view[576] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_31__placeholder_0 tensor
    unsigned g_0_tensor_31__placeholder_0_max_sizes[] = {576};
    unsigned g_0_tensor_31__placeholder_0_min_sizes[] = {576};
    unsigned g_0_tensor_31__placeholder_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_tensor_31__placeholder_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_tensor_31__placeholder_0_max_sizes,
                                                      1,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_tensor_31__placeholder_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view tensor
    unsigned g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view_max_sizes[] = {576};
    unsigned g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view_min_sizes[] = {576};
    unsigned g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view = createTensors(1,
                                                                                             OUTPUT_TENSOR,
                                                                                             false,
                                                                                             "g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view",
                                                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                             nullptr,
                                                                                             g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view_max_sizes,
                                                                                             1,
                                                                                             syn_type_bf16,
                                                                                             nullptr,
                                                                                             0,
                                                                                             0,
                                                                                             nullptr,
                                                                                             false,
                                                                                             g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view_min_sizes,
                                                                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_attention_query_key_value_strided_view_6284_0_id;
    unsigned char g_0_3_attention_query_key_value_strided_view_6284_0_params[] = {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("strided_view", {g_0_tensor_31__placeholder_0}, {g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view}, (void*)g_0_3_attention_query_key_value_strided_view_6284_0_params, 208, "g_0_3_attention_query_key_value_strided_view_6284_0", 0 /*graphIndex*/, &g_0_3_attention_query_key_value_strided_view_6284_0_id);

    /*************
     * g_0_3_attention_query_key_value_strided_view_6285_0 node
     * inputs:
     *     g_0_tensor_33__placeholder_0[1536, 576] (dtype=bf16)
     * outputs:
     *     g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view[1536, 576] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_33__placeholder_0 tensor
    unsigned g_0_tensor_33__placeholder_0_max_sizes[] = {1536,576};
    unsigned g_0_tensor_33__placeholder_0_min_sizes[] = {1536,576};
    unsigned g_0_tensor_33__placeholder_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_tensor_33__placeholder_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_tensor_33__placeholder_0_max_sizes,
                                                      2,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_tensor_33__placeholder_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view tensor
    unsigned g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view_max_sizes[] = {1536,576};
    unsigned g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view_min_sizes[] = {1536,576};
    unsigned g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view = createTensors(1,
                                                                                             OUTPUT_TENSOR,
                                                                                             false,
                                                                                             "g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view",
                                                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                             nullptr,
                                                                                             g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view_max_sizes,
                                                                                             2,
                                                                                             syn_type_bf16,
                                                                                             nullptr,
                                                                                             0,
                                                                                             0,
                                                                                             nullptr,
                                                                                             false,
                                                                                             g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view_min_sizes,
                                                                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_attention_query_key_value_strided_view_6285_0_id;
    unsigned char g_0_3_attention_query_key_value_strided_view_6285_0_params[] = {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("strided_view", {g_0_tensor_33__placeholder_0}, {g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view}, (void*)g_0_3_attention_query_key_value_strided_view_6285_0_params, 208, "g_0_3_attention_query_key_value_strided_view_6285_0", 0 /*graphIndex*/, &g_0_3_attention_query_key_value_strided_view_6285_0_id);

    /*************
     * g_0_3_input_layernorm_cast_i32_to_bf16_6271_0 node
     * inputs:
     *     g_0_tensor_9__placeholder_0[1] (dtype=int32)
     * outputs:
     *     g_0_tensor_11[1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_9__placeholder_0 tensor
    unsigned g_0_tensor_9__placeholder_0_max_sizes[] = {1};
    unsigned g_0_tensor_9__placeholder_0_min_sizes[] = {1};
    unsigned g_0_tensor_9__placeholder_0 = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_tensor_9__placeholder_0",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_tensor_9__placeholder_0_max_sizes,
                                                     1,
                                                     syn_type_int32,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_tensor_9__placeholder_0_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_11 tensor
    unsigned g_0_tensor_11_max_sizes[] = {1};
    unsigned g_0_tensor_11_min_sizes[] = {1};
    unsigned g_0_tensor_11 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_11",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_11_max_sizes,
                                       1,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_11_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_cast_i32_to_bf16_6271_0_id;
    unsigned char g_0_3_input_layernorm_cast_i32_to_bf16_6271_0_params[] = {0,0,0,0};
    addNodeToGraph("cast_i32_to_bf16", {g_0_tensor_9__placeholder_0}, {g_0_tensor_11}, (void*)g_0_3_input_layernorm_cast_i32_to_bf16_6271_0_params, 4, "g_0_3_input_layernorm_cast_i32_to_bf16_6271_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_cast_i32_to_bf16_6271_0_id);

    /*************
     * g_0_3_input_layernorm_strided_view_6272_0 node
     * inputs:
     *     g_0_tensor_12__placeholder_0[1536] (dtype=bf16)
     * outputs:
     *     g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view[1536] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_12__placeholder_0 tensor
    unsigned g_0_tensor_12__placeholder_0_max_sizes[] = {1536};
    unsigned g_0_tensor_12__placeholder_0_min_sizes[] = {1536};
    unsigned g_0_tensor_12__placeholder_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_tensor_12__placeholder_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_tensor_12__placeholder_0_max_sizes,
                                                      1,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_tensor_12__placeholder_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view tensor
    unsigned g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view_max_sizes[] = {1536};
    unsigned g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view_min_sizes[] = {1536};
    unsigned g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view = createTensors(1,
                                                                                   OUTPUT_TENSOR,
                                                                                   false,
                                                                                   "g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view",
                                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                   nullptr,
                                                                                   g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view_max_sizes,
                                                                                   1,
                                                                                   syn_type_bf16,
                                                                                   nullptr,
                                                                                   0,
                                                                                   0,
                                                                                   nullptr,
                                                                                   false,
                                                                                   g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view_min_sizes,
                                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_strided_view_6272_0_id;
    unsigned char g_0_3_input_layernorm_strided_view_6272_0_params[] = {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("strided_view", {g_0_tensor_12__placeholder_0}, {g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view}, (void*)g_0_3_input_layernorm_strided_view_6272_0_params, 208, "g_0_3_input_layernorm_strided_view_6272_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_strided_view_6272_0_id);

    /*************
     * g_0_3_input_layernorm_add_fwd_bf16_6273_0 node
     * inputs:
     *     g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view[1536] (dtype=bf16)
     *     g_0_tensor_11[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_14_id_29361_3_input_layernorm_aten__add[1536] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_14_id_29361_3_input_layernorm_aten__add tensor
    unsigned g_0_tensor_14_id_29361_3_input_layernorm_aten__add_max_sizes[] = {1536};
    unsigned g_0_tensor_14_id_29361_3_input_layernorm_aten__add_min_sizes[] = {1536};
    unsigned g_0_tensor_14_id_29361_3_input_layernorm_aten__add = createTensors(1,
                                                                            OUTPUT_TENSOR,
                                                                            true,
                                                                            "g_0_tensor_14_id_29361_3_input_layernorm_aten__add",
                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                            nullptr,
                                                                            g_0_tensor_14_id_29361_3_input_layernorm_aten__add_max_sizes,
                                                                            1,
                                                                            syn_type_bf16,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_tensor_14_id_29361_3_input_layernorm_aten__add_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_add_fwd_bf16_6273_0_id;
    addNodeToGraph("add_fwd_bf16", {g_0_tensor_13_id_5190_3_input_layernorm_hpu__strided_view, g_0_tensor_11}, {g_0_tensor_14_id_29361_3_input_layernorm_aten__add}, nullptr, 0, "g_0_3_input_layernorm_add_fwd_bf16_6273_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_add_fwd_bf16_6273_0_id);

    /*************
     * g_0_3_input_layernorm_cast_bf16_to_f32_6275_0 node
     * inputs:
     *     g_0_tensor_14_id_29361_3_input_layernorm_aten__add[1536] (dtype=bf16)
     * outputs:
     *     g_0_tensor_20[1536] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_20 tensor
    unsigned g_0_tensor_20_max_sizes[] = {1536};
    unsigned g_0_tensor_20_min_sizes[] = {1536};
    unsigned g_0_tensor_20 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_20",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_20_max_sizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_20_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_cast_bf16_to_f32_6275_0_id;
    unsigned char g_0_3_input_layernorm_cast_bf16_to_f32_6275_0_params[] = {0,0,0,0};
    addNodeToGraph("cast_bf16_to_f32", {g_0_tensor_14_id_29361_3_input_layernorm_aten__add}, {g_0_tensor_20}, (void*)g_0_3_input_layernorm_cast_bf16_to_f32_6275_0_params, 4, "g_0_3_input_layernorm_cast_bf16_to_f32_6275_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_cast_bf16_to_f32_6275_0_id);

    /*************
     * g_0_3_input_layernorm_reshape_6276_0 node
     * inputs:
     *     g_0_tensor_20[1536] (dtype=float32)
     * outputs:
     *     g_0_tensor_21[1536] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_21 tensor
    unsigned g_0_tensor_21_max_sizes[] = {1536};
    unsigned g_0_tensor_21_min_sizes[] = {1536};
    unsigned g_0_tensor_21 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_21",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_21_max_sizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_21_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_reshape_6276_0_id;
    addNodeToGraph("reshape", {g_0_tensor_20}, {g_0_tensor_21}, nullptr, 0, "g_0_3_input_layernorm_reshape_6276_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_reshape_6276_0_id);

    /*************
     * g_0_3_input_layernorm_strided_view_6274_0 node
     * inputs:
     *     g_0_tensor_15__placeholder_0[1536] (dtype=bf16)
     * outputs:
     *     g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view[1536] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_15__placeholder_0 tensor
    unsigned g_0_tensor_15__placeholder_0_max_sizes[] = {1536};
    unsigned g_0_tensor_15__placeholder_0_min_sizes[] = {1536};
    unsigned g_0_tensor_15__placeholder_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_tensor_15__placeholder_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_tensor_15__placeholder_0_max_sizes,
                                                      1,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_tensor_15__placeholder_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view tensor
    unsigned g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view_max_sizes[] = {1536};
    unsigned g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view_min_sizes[] = {1536};
    unsigned g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view = createTensors(1,
                                                                                   OUTPUT_TENSOR,
                                                                                   false,
                                                                                   "g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view",
                                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                   nullptr,
                                                                                   g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view_max_sizes,
                                                                                   1,
                                                                                   syn_type_bf16,
                                                                                   nullptr,
                                                                                   0,
                                                                                   0,
                                                                                   nullptr,
                                                                                   false,
                                                                                   g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view_min_sizes,
                                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_strided_view_6274_0_id;
    unsigned char g_0_3_input_layernorm_strided_view_6274_0_params[] = {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("strided_view", {g_0_tensor_15__placeholder_0}, {g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view}, (void*)g_0_3_input_layernorm_strided_view_6274_0_params, 208, "g_0_3_input_layernorm_strided_view_6274_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_strided_view_6274_0_id);

    /*************
     * g_0_3_input_layernorm_cast_bf16_to_f32_6277_0 node
     * inputs:
     *     g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view[1536] (dtype=bf16)
     * outputs:
     *     g_0_tensor_22[1536] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_22 tensor
    unsigned g_0_tensor_22_max_sizes[] = {1536};
    unsigned g_0_tensor_22_min_sizes[] = {1536};
    unsigned g_0_tensor_22 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_22",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_22_max_sizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_22_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_cast_bf16_to_f32_6277_0_id;
    unsigned char g_0_3_input_layernorm_cast_bf16_to_f32_6277_0_params[] = {0,0,0,0};
    addNodeToGraph("cast_bf16_to_f32", {g_0_tensor_16_id_5234_3_input_layernorm_hpu__strided_view}, {g_0_tensor_22}, (void*)g_0_3_input_layernorm_cast_bf16_to_f32_6277_0_params, 4, "g_0_3_input_layernorm_cast_bf16_to_f32_6277_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_cast_bf16_to_f32_6277_0_id);

    /*************
     * g_0_3_input_layernorm_reshape_6278_0 node
     * inputs:
     *     g_0_tensor_22[1536] (dtype=float32)
     * outputs:
     *     g_0_tensor_23[1536] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_23 tensor
    unsigned g_0_tensor_23_max_sizes[] = {1536};
    unsigned g_0_tensor_23_min_sizes[] = {1536};
    unsigned g_0_tensor_23 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_23",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_23_max_sizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_23_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_reshape_6278_0_id;
    addNodeToGraph("reshape", {g_0_tensor_22}, {g_0_tensor_23}, nullptr, 0, "g_0_3_input_layernorm_reshape_6278_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_reshape_6278_0_id);

    /*************
     * g_0_embed_embedding_dropout_dropout_fwd_bf16_6268_0 node
     * inputs:
     *     g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view[1536, 2048, 8] (dtype=bf16)
     *     g_0_tensor_2__placeholder_1[1] (dtype=int32)
     * outputs:
     *     g_0_tensor_5[1536, 2048, 8] (dtype=bf16)
     *     g_0_tensor_6[1536, 2048, 8] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view tensor
    unsigned g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view_max_sizes[] = {1536,2048,8};
    unsigned g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view_min_sizes[] = {1536,2048,8};
    unsigned g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view = createTensors(1,
                                                                                  INPUT_TENSOR,
                                                                                  true,
                                                                                  "g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view",
                                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                  nullptr,
                                                                                  g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view_max_sizes,
                                                                                  3,
                                                                                  syn_type_bf16,
                                                                                  nullptr,
                                                                                  0,
                                                                                  0,
                                                                                  nullptr,
                                                                                  false,
                                                                                  g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view_min_sizes,
                                                                                  synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2__placeholder_1 tensor
    unsigned g_0_tensor_2__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_2__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_2__placeholder_1 = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_tensor_2__placeholder_1",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_tensor_2__placeholder_1_max_sizes,
                                                     1,
                                                     syn_type_int32,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_tensor_2__placeholder_1_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_5 tensor
    unsigned g_0_tensor_5_max_sizes[] = {1536,2048,8};
    unsigned g_0_tensor_5_min_sizes[] = {1536,2048,8};
    unsigned g_0_tensor_5 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "g_0_tensor_5",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      g_0_tensor_5_max_sizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_5_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_6 tensor
    unsigned g_0_tensor_6_max_sizes[] = {1536,2048,8};
    unsigned g_0_tensor_6_min_sizes[] = {1536,2048,8};
    unsigned g_0_tensor_6 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "g_0_tensor_6",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      g_0_tensor_6_max_sizes,
                                      3,
                                      syn_type_int8,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_6_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_embed_embedding_dropout_dropout_fwd_bf16_6268_0_id;
    unsigned char g_0_embed_embedding_dropout_dropout_fwd_bf16_6268_0_params[] = {205,204,204,61,0,0,0,0};
    addNodeToGraph("dropout_fwd_bf16", {g_0_tensor_1_id_29332_embed_embedding_dropout_aten__view, g_0_tensor_2__placeholder_1}, {g_0_tensor_5, g_0_tensor_6}, (void*)g_0_embed_embedding_dropout_dropout_fwd_bf16_6268_0_params, 8, "g_0_embed_embedding_dropout_dropout_fwd_bf16_6268_0", 0 /*graphIndex*/, &g_0_embed_embedding_dropout_dropout_fwd_bf16_6268_0_id);

    /*************
     * g_0__transpose_6269_0 node
     * inputs:
     *     g_0_tensor_5[1536, 2048, 8] (dtype=bf16)
     * outputs:
     *     g_0_tensor_7_id_29354_aten__transpose[1536, 8, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_7_id_29354_aten__transpose tensor
    unsigned g_0_tensor_7_id_29354_aten__transpose_max_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_7_id_29354_aten__transpose_min_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_7_id_29354_aten__transpose = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               false,
                                                               "g_0_tensor_7_id_29354_aten__transpose",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               g_0_tensor_7_id_29354_aten__transpose_max_sizes,
                                                               3,
                                                               syn_type_bf16,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_tensor_7_id_29354_aten__transpose_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__transpose_6269_0_id;
    unsigned char g_0__transpose_6269_0_params[] = {0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0};
    addNodeToGraph("transpose", {g_0_tensor_5}, {g_0_tensor_7_id_29354_aten__transpose}, (void*)g_0__transpose_6269_0_params, 24, "g_0__transpose_6269_0", 0 /*graphIndex*/, &g_0__transpose_6269_0_id);

    /*************
     * g_0__identity_6270_0 node
     * inputs:
     *     g_0_tensor_7_id_29354_aten__transpose[1536, 8, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_8_id_29356_hpu__identity[1536, 8, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_8_id_29356_hpu__identity tensor
    unsigned g_0_tensor_8_id_29356_hpu__identity_max_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_8_id_29356_hpu__identity_min_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_8_id_29356_hpu__identity = createTensors(1,
                                                             OUTPUT_TENSOR,
                                                             true,
                                                             "g_0_tensor_8_id_29356_hpu__identity",
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             g_0_tensor_8_id_29356_hpu__identity_max_sizes,
                                                             3,
                                                             syn_type_bf16,
                                                             nullptr,
                                                             0,
                                                             0,
                                                             nullptr,
                                                             false,
                                                             g_0_tensor_8_id_29356_hpu__identity_min_sizes,
                                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__identity_6270_0_id;
    addNodeToGraph("identity", {g_0_tensor_7_id_29354_aten__transpose}, {g_0_tensor_8_id_29356_hpu__identity}, nullptr, 0, "g_0__identity_6270_0", 0 /*graphIndex*/, &g_0__identity_6270_0_id);

    /*************
     * g_0_3_input_layernorm_reshape_6279_0 node
     * inputs:
     *     g_0_tensor_8_id_29356_hpu__identity[1536, 8, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_24[1536, 16384, 1, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_24 tensor
    unsigned g_0_tensor_24_max_sizes[] = {1536,16384,1,1};
    unsigned g_0_tensor_24_min_sizes[] = {1536,16384,1,1};
    unsigned g_0_tensor_24 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_24",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_24_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_24_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_reshape_6279_0_id;
    addNodeToGraph("reshape", {g_0_tensor_8_id_29356_hpu__identity}, {g_0_tensor_24}, nullptr, 0, "g_0_3_input_layernorm_reshape_6279_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_reshape_6279_0_id);

    /*************
     * g_0_3_input_layernorm_layer_norm_fwd_bf16_6280_0 node
     * inputs:
     *     g_0_tensor_24[1536, 16384, 1, 1] (dtype=bf16)
     *     g_0_tensor_23[1536] (dtype=float32)
     *     g_0_tensor_21[1536] (dtype=float32)
     * outputs:
     *     g_0_tensor_25[1536, 16384, 1, 1] (dtype=bf16)
     *     g_0_tensor_26[1, 16384, 1, 1] (dtype=float32)
     *     g_0_tensor_27[1, 16384, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_25 tensor
    unsigned g_0_tensor_25_max_sizes[] = {1536,16384,1,1};
    unsigned g_0_tensor_25_min_sizes[] = {1536,16384,1,1};
    unsigned g_0_tensor_25 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_25",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_25_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_25_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_26 tensor
    unsigned g_0_tensor_26_max_sizes[] = {1,16384,1,1};
    unsigned g_0_tensor_26_min_sizes[] = {1,16384,1,1};
    unsigned g_0_tensor_26 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_26",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_26_max_sizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_26_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_27 tensor
    unsigned g_0_tensor_27_max_sizes[] = {1,16384,1,1};
    unsigned g_0_tensor_27_min_sizes[] = {1,16384,1,1};
    unsigned g_0_tensor_27 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_27",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_27_max_sizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_27_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_layer_norm_fwd_bf16_6280_0_id;
    unsigned char g_0_3_input_layernorm_layer_norm_fwd_bf16_6280_0_params[] = {1,0,0,0,172,197,39,55};
    addNodeToGraph("layer_norm_fwd_bf16", {g_0_tensor_24, g_0_tensor_23, g_0_tensor_21}, {g_0_tensor_25, g_0_tensor_26, g_0_tensor_27}, (void*)g_0_3_input_layernorm_layer_norm_fwd_bf16_6280_0_params, 8, "g_0_3_input_layernorm_layer_norm_fwd_bf16_6280_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_layer_norm_fwd_bf16_6280_0_id);

    /*************
     * g_0_3_input_layernorm_reshape_6281_0 node
     * inputs:
     *     g_0_tensor_25[1536, 16384, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_28[1536, 8, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_28 tensor
    unsigned g_0_tensor_28_max_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_28_min_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_28 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_28",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_28_max_sizes,
                                       3,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_28_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_input_layernorm_reshape_6281_0_id;
    addNodeToGraph("reshape", {g_0_tensor_25}, {g_0_tensor_28}, nullptr, 0, "g_0_3_input_layernorm_reshape_6281_0", 0 /*graphIndex*/, &g_0_3_input_layernorm_reshape_6281_0_id);

    /*************
     * g_0_3_attention_query_key_value_reshape_6286_0 node
     * inputs:
     *     g_0_tensor_28[1536, 8, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view[1536, 8, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view tensor
    unsigned g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view_max_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view_min_sizes[] = {1536,8,2048};
    unsigned g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view = createTensors(1,
                                                                                       OUTPUT_TENSOR,
                                                                                       false,
                                                                                       "g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view",
                                                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                       nullptr,
                                                                                       g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view_max_sizes,
                                                                                       3,
                                                                                       syn_type_bf16,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_attention_query_key_value_reshape_6286_0_id;
    addNodeToGraph("reshape", {g_0_tensor_28}, {g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view}, nullptr, 0, "g_0_3_attention_query_key_value_reshape_6286_0", 0 /*graphIndex*/, &g_0_3_attention_query_key_value_reshape_6286_0_id);

    /*************
     * g_0_3_attention_query_key_value_batch_gemm_6287_0 node
     * inputs:
     *     g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view[1536, 8, 2048] (dtype=bf16)
     *     g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view[1536, 576] (dtype=bf16)
     *     g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view[576] (dtype=bf16)
     * outputs:
     *     g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear[576, 8, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear tensor
    unsigned g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear_max_sizes[] = {576,8,2048};
    unsigned g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear_min_sizes[] = {576,8,2048};
    unsigned g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear = createTensors(1,
                                                                                         OUTPUT_TENSOR,
                                                                                         true,
                                                                                         "g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear",
                                                                                         MEM_INIT_ALL_ZERO,
                                                                                         nullptr,
                                                                                         g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear_max_sizes,
                                                                                         3,
                                                                                         syn_type_bf16,
                                                                                         nullptr,
                                                                                         0,
                                                                                         0,
                                                                                         nullptr,
                                                                                         false,
                                                                                         g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear_min_sizes,
                                                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_3_attention_query_key_value_batch_gemm_6287_0_id;
    unsigned char g_0_3_attention_query_key_value_batch_gemm_6287_0_params[] = {0,1};
    addNodeToGraph("batch_gemm", {g_0_tensor_35_id_29378_3_attention_query_key_value_aten__view, g_0_tensor_34_id_5242_3_attention_query_key_value_hpu__strided_view, g_0_tensor_32_id_5319_3_attention_query_key_value_hpu__strided_view}, {g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear}, (void*)g_0_3_attention_query_key_value_batch_gemm_6287_0_params, 2, "g_0_3_attention_query_key_value_batch_gemm_6287_0", 0 /*graphIndex*/, &g_0_3_attention_query_key_value_batch_gemm_6287_0_id);

    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_36_id_29380_3_attention_query_key_value_aten__linear});
}

// The bundle is taken from a FWD layer in GPT3 FP8.
// It includes a GEMM node with producer to operand A and operand B.
// In G2 both chains are expected to be sliced in SRAM.
// Disabled for G1 (convert_to_fp8_bf16 + fp8_gemm_bf16 are not supported) and G3 (long time on sim).
TEST_F_GC(SynGaudiTwoRunCompareTest, gemm_with_2_producers_ASIC_CI, {synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0_10_post_attention_layernorm_layer_norm_fwd_bf16_6610_0 node
     * inputs:
     *     g_0_tensor_25[12288, 2048, 1, 1] (dtype=bf16)
     *     g_0_tensor_24[12288] (dtype=float32)
     *     g_0_tensor_22[12288] (dtype=float32)
     * outputs:
     *     g_0_tensor_26[12288, 2048, 1, 1] (dtype=bf16)
     *     g_0_tensor_27[1, 2048, 1, 1] (dtype=float32)
     *     g_0_tensor_28[1, 2048, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_25 tensor
    unsigned g_0_tensor_25_max_sizes[] = {12288, 2048, 1, 1};
    unsigned g_0_tensor_25_min_sizes[] = {12288, 2048, 1, 1};
    unsigned g_0_tensor_25             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_25",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_25_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_25_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_24 tensor
    unsigned g_0_tensor_24_max_sizes[] = {12288};
    unsigned g_0_tensor_24_min_sizes[] = {12288};
    unsigned g_0_tensor_24             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_24",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_24_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_24_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_22 tensor
    unsigned g_0_tensor_22_max_sizes[] = {12288};
    unsigned g_0_tensor_22_min_sizes[] = {12288};
    unsigned g_0_tensor_22             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_22",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_22_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_22_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_26 tensor
    unsigned g_0_tensor_26_max_sizes[] = {12288, 2048, 1, 1};
    unsigned g_0_tensor_26_min_sizes[] = {12288, 2048, 1, 1};
    unsigned g_0_tensor_26             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_26",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_26_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_26_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_27 tensor
    unsigned g_0_tensor_27_max_sizes[] = {1, 2048, 1, 1};
    unsigned g_0_tensor_27_min_sizes[] = {1, 2048, 1, 1};
    unsigned g_0_tensor_27             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_27",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_27_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_27_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_28 tensor
    unsigned      g_0_tensor_28_max_sizes[] = {1, 2048, 1, 1};
    unsigned      g_0_tensor_28_min_sizes[] = {1, 2048, 1, 1};
    unsigned      g_0_tensor_28             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_28",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_28_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_28_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_post_attention_layernorm_layer_norm_fwd_bf16_6610_0_id;
    unsigned char g_0_10_post_attention_layernorm_layer_norm_fwd_bf16_6610_0_params[] = {1, 0, 0, 0, 172, 197, 39, 55};
    addNodeToGraph("layer_norm_fwd_bf16",
                   {g_0_tensor_25, g_0_tensor_24, g_0_tensor_22},
                   {g_0_tensor_26, g_0_tensor_27, g_0_tensor_28},
                   (void*)g_0_10_post_attention_layernorm_layer_norm_fwd_bf16_6610_0_params,
                   8,
                   "g_0_10_post_attention_layernorm_layer_norm_fwd_bf16_6610_0",
                   0 /*graphIndex*/,
                   &g_0_10_post_attention_layernorm_layer_norm_fwd_bf16_6610_0_id);

    /*************
     * g_0_10_post_attention_layernorm_reshape_6611_0 node
     * inputs:
     *     g_0_tensor_26[12288, 2048, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_29[12288, 1, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_29 tensor
    unsigned  g_0_tensor_29_max_sizes[] = {12288, 1, 2048};
    unsigned  g_0_tensor_29_min_sizes[] = {12288, 1, 2048};
    unsigned  g_0_tensor_29             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_29",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_29_max_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_29_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_post_attention_layernorm_reshape_6611_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_26},
                   {g_0_tensor_29},
                   nullptr,
                   0,
                   "g_0_10_post_attention_layernorm_reshape_6611_0",
                   0 /*graphIndex*/,
                   &g_0_10_post_attention_layernorm_reshape_6611_0_id);

    /*************
     * g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6616_0 node
     * inputs:
     *     g_0_tensor_29[12288, 1, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view[12288, 1, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view tensor
    unsigned g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes[] = {12288,
                                                                                                          1,
                                                                                                          2048};
    unsigned g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes[] = {12288,
                                                                                                          1,
                                                                                                          2048};
    unsigned g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6616_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_29},
                   {g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view},
                   nullptr,
                   0,
                   "g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6616_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6616_0_id);

    /*************
     * g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6617_0 node
     * inputs:
     *     g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view[12288, 1, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view[12288, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view tensor
    unsigned g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes[] = {12288, 2048};
    unsigned g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes[] = {12288, 2048};
    unsigned g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6617_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_35_id_29646_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view},
                   {g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view},
                   nullptr,
                   0,
                   "g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6617_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6617_0_id);

    /*************
     * g_0_10_post_attention_layernorm_reshape_6612_0 node
     * inputs:
     *     g_0_tensor_27[1, 2048, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_30[1, 1, 2048] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_30 tensor
    unsigned  g_0_tensor_30_max_sizes[] = {1, 1, 2048};
    unsigned  g_0_tensor_30_min_sizes[] = {1, 1, 2048};
    unsigned  g_0_tensor_30             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_30",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_30_max_sizes,
                                           3,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_30_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_post_attention_layernorm_reshape_6612_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_27},
                   {g_0_tensor_30},
                   nullptr,
                   0,
                   "g_0_10_post_attention_layernorm_reshape_6612_0",
                   0 /*graphIndex*/,
                   &g_0_10_post_attention_layernorm_reshape_6612_0_id);

    /*************
     * g_0_10_post_attention_layernorm_reshape_6613_0 node
     * inputs:
     *     g_0_tensor_28[1, 2048, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_31[1, 1, 2048] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_31 tensor
    unsigned  g_0_tensor_31_max_sizes[] = {1, 1, 2048};
    unsigned  g_0_tensor_31_min_sizes[] = {1, 1, 2048};
    unsigned  g_0_tensor_31             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_31",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_31_max_sizes,
                                           3,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_31_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_post_attention_layernorm_reshape_6613_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_28},
                   {g_0_tensor_31},
                   nullptr,
                   0,
                   "g_0_10_post_attention_layernorm_reshape_6613_0",
                   0 /*graphIndex*/,
                   &g_0_10_post_attention_layernorm_reshape_6613_0_id);

    /*************
     * g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6618_0 node
     * inputs:
     *     g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view[12288, 2048] (dtype=bf16)
     *     g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_39[12288, 2048] (dtype=float8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_39 tensor
    unsigned      g_0_tensor_39_max_sizes[] = {12288, 2048};
    unsigned      g_0_tensor_39_min_sizes[] = {12288, 2048};
    unsigned      g_0_tensor_39             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_39",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_39_max_sizes,
                                           2,
                                           syn_type_fp8_152,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_39_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6618_0_id;
    unsigned char g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6618_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("convert_to_fp8_bf16",
                   {g_0_tensor_36_id_29648_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view,
                    g_0_tensor_34_id_29666_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity},
                   {g_0_tensor_39},
                   (void*)g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6618_0_params,
                   4,
                   "g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6618_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6618_0_id);

    /*************
     * g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6622_0 node
     * inputs:
     *     g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice[75497472] (dtype=bf16)
     * outputs:
     *     g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view[12288, 6144] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice tensor
    unsigned g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice_max_sizes[] = {75497472};
    unsigned g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice_min_sizes[] = {75497472};
    unsigned g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view tensor
    unsigned g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes[] = {12288, 6144};
    unsigned g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes[] = {12288, 6144};
    unsigned g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6622_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_42_id_12888_10_mlp_dense_h_to_4h_output_parallel_linear_aten__slice},
                   {g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view},
                   nullptr,
                   0,
                   "g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6622_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_h_to_4h_output_parallel_linear_reshape_6622_0_id);

    /*************
     * g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6623_0 node
     * inputs:
     *     g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view[12288, 6144] (dtype=bf16)
     *     g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_46[12288, 6144] (dtype=float8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_46 tensor
    unsigned      g_0_tensor_46_max_sizes[] = {12288, 6144};
    unsigned      g_0_tensor_46_min_sizes[] = {12288, 6144};
    unsigned      g_0_tensor_46             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_46",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_46_max_sizes,
                                           2,
                                           syn_type_fp8_152,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_46_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6623_0_id;
    unsigned char g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6623_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("convert_to_fp8_bf16",
                   {g_0_tensor_43_id_12889_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view,
                    g_0_tensor_41_id_29692_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity},
                   {g_0_tensor_46},
                   (void*)g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6623_0_params,
                   4,
                   "g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6623_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_h_to_4h_output_parallel_linear_convert_to_fp8_bf16_6623_0_id);

    /*************
     * g_0_10_mlp_dense_h_to_4h_output_parallel_linear_fp8_gemm_bf16_6636_0 node
     * inputs:
     *     g_0_tensor_39[12288, 2048] (dtype=float8)
     *     g_0_tensor_46[12288, 6144] (dtype=float8)
     *     g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     *     g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     *     g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view[6144] (dtype=bf16)
     * outputs:
     *     g_0_tensor_61[6144, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view tensor
    unsigned g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes[] = {6144};
    unsigned g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes[] = {6144};
    unsigned g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_61 tensor
    unsigned      g_0_tensor_61_max_sizes[] = {6144, 2048};
    unsigned      g_0_tensor_61_min_sizes[] = {6144, 2048};
    unsigned      g_0_tensor_61             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_61",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_61_max_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_61_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_mlp_dense_h_to_4h_output_parallel_linear_fp8_gemm_bf16_6636_0_id;
    unsigned char g_0_10_mlp_dense_h_to_4h_output_parallel_linear_fp8_gemm_bf16_6636_0_params[] = {0, 1};
    addNodeToGraph("fp8_gemm_bf16",
                   {g_0_tensor_39,
                    g_0_tensor_46,
                    g_0_tensor_59_id_29708_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity,
                    g_0_tensor_57_id_29702_10_mlp_dense_h_to_4h_output_parallel_linear_hpu__identity,
                    g_0_tensor_48_id_12891_10_mlp_dense_h_to_4h_output_parallel_linear_aten__view},
                   {g_0_tensor_61},
                   (void*)g_0_10_mlp_dense_h_to_4h_output_parallel_linear_fp8_gemm_bf16_6636_0_params,
                   2,
                   "g_0_10_mlp_dense_h_to_4h_output_parallel_linear_fp8_gemm_bf16_6636_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_h_to_4h_output_parallel_linear_fp8_gemm_bf16_6636_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_SLICING_BOTH_PRODUCER_CHAINS", "true");
    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "ENABLE_SLICING_BOTH_PRODUCER_CHAINS", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_61, g_0_tensor_30, g_0_tensor_31});
}

// The bundle is taken from a FWD layer in GPT3 FP8.
// It includes a GEMM node with producer chain to operand A (TPC->reshape) and a single producer to operand B.
// In G2 both chains are expected to be sliced in SRAM.
// Disabled for G1 (convert_to_fp8_bf16 + fp8_gemm_bf16 are not supported) and G3 (long time on sim).
TEST_F_GC(SynGaudiTwoRunCompareTest, gemm_with_2_producer_chains_including_reshape_ASIC_CI, {synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0_10_mlp_reshape_6638_0 node
     * inputs:
     *     g_0_tensor_61[6144, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_63_id_29718_10_mlp_aten__view[6144, 1, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_61 tensor
    unsigned g_0_tensor_61_max_sizes[] = {6144, 2048};
    unsigned g_0_tensor_61_min_sizes[] = {6144, 2048};
    unsigned g_0_tensor_61             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_61",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_61_max_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_61_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_63_id_29718_10_mlp_aten__view tensor
    unsigned g_0_tensor_63_id_29718_10_mlp_aten__view_max_sizes[] = {6144, 1, 2048};
    unsigned g_0_tensor_63_id_29718_10_mlp_aten__view_min_sizes[] = {6144, 1, 2048};
    unsigned g_0_tensor_63_id_29718_10_mlp_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_63_id_29718_10_mlp_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_63_id_29718_10_mlp_aten__view_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_63_id_29718_10_mlp_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_mlp_reshape_6638_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_61},
                   {g_0_tensor_63_id_29718_10_mlp_aten__view},
                   nullptr,
                   0,
                   "g_0_10_mlp_reshape_6638_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_reshape_6638_0_id);

    /*************
     * g_0_10_mlp_gelu_fwd_bf16_6639_0 node
     * inputs:
     *     g_0_tensor_63_id_29718_10_mlp_aten__view[6144, 1, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_65[6144, 1, 2048] (dtype=bf16)
     *     g_0_tensor_66[6144, 1, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_65 tensor
    unsigned g_0_tensor_65_max_sizes[] = {6144, 1, 2048};
    unsigned g_0_tensor_65_min_sizes[] = {6144, 1, 2048};
    unsigned g_0_tensor_65             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_65",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_65_max_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_65_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_66 tensor
    unsigned      g_0_tensor_66_max_sizes[] = {6144, 1, 2048};
    unsigned      g_0_tensor_66_min_sizes[] = {6144, 1, 2048};
    unsigned      g_0_tensor_66             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_66",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_66_max_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_66_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_mlp_gelu_fwd_bf16_6639_0_id;
    unsigned char g_0_10_mlp_gelu_fwd_bf16_6639_0_params[] = {1};
    addNodeToGraph("gelu_fwd_bf16",
                   {g_0_tensor_63_id_29718_10_mlp_aten__view},
                   {g_0_tensor_65, g_0_tensor_66},
                   (void*)g_0_10_mlp_gelu_fwd_bf16_6639_0_params,
                   1,
                   "g_0_10_mlp_gelu_fwd_bf16_6639_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_gelu_fwd_bf16_6639_0_id);

    /*************
     * g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6642_0 node
     * inputs:
     *     g_0_tensor_65[6144, 1, 2048] (dtype=bf16)
     * outputs:
     *     g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view[6144, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view tensor
    unsigned g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_max_sizes[] = {6144, 2048};
    unsigned g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_min_sizes[] = {6144, 2048};
    unsigned g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6642_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_65},
                   {g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view},
                   nullptr,
                   0,
                   "g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6642_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6642_0_id);

    /*************
     * g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6643_0 node
     * inputs:
     *     g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view[6144, 2048] (dtype=bf16)
     *     g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_73[6144, 2048] (dtype=float8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_73 tensor
    unsigned      g_0_tensor_73_max_sizes[] = {6144, 2048};
    unsigned      g_0_tensor_73_min_sizes[] = {6144, 2048};
    unsigned      g_0_tensor_73             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_73",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_73_max_sizes,
                                           2,
                                           syn_type_fp8_152,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_73_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6643_0_id;
    unsigned char g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6643_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("convert_to_fp8_bf16",
                   {g_0_tensor_70_id_29722_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view,
                    g_0_tensor_69_id_29740_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity},
                   {g_0_tensor_73},
                   (void*)g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6643_0_params,
                   4,
                   "g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6643_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6643_0_id);

    /*************
     * g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6647_0 node
     * inputs:
     *     g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice[75497472] (dtype=bf16)
     * outputs:
     *     g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view[6144, 12288] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice tensor
    unsigned g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice_max_sizes[] = {75497472};
    unsigned g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice_min_sizes[] = {75497472};
    unsigned g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view tensor
    unsigned g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_max_sizes[] = {6144, 12288};
    unsigned g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_min_sizes[] = {6144, 12288};
    unsigned g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6647_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_76_id_12892_10_mlp_dense_4h_to_h_output_parallel_linear_aten__slice},
                   {g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view},
                   nullptr,
                   0,
                   "g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6647_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_4h_to_h_output_parallel_linear_reshape_6647_0_id);

    /*************
     * g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6648_0 node
     * inputs:
     *     g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view[6144, 12288] (dtype=bf16)
     *     g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_80[6144, 12288] (dtype=float8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_80 tensor
    unsigned      g_0_tensor_80_max_sizes[] = {6144, 12288};
    unsigned      g_0_tensor_80_min_sizes[] = {6144, 12288};
    unsigned      g_0_tensor_80             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_80",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_80_max_sizes,
                                           2,
                                           syn_type_fp8_152,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_80_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6648_0_id;
    unsigned char g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6648_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("convert_to_fp8_bf16",
                   {g_0_tensor_77_id_12893_10_mlp_dense_4h_to_h_output_parallel_linear_aten__view,
                    g_0_tensor_75_id_29766_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity},
                   {g_0_tensor_80},
                   (void*)g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6648_0_params,
                   4,
                   "g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6648_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_4h_to_h_output_parallel_linear_convert_to_fp8_bf16_6648_0_id);

    /*************
     * g_0_10_mlp_dense_4h_to_h_output_parallel_linear_fp8_gemm_bf16_6659_0 node
     * inputs:
     *     g_0_tensor_73[6144, 2048] (dtype=float8)
     *     g_0_tensor_80[6144, 12288] (dtype=float8)
     *     g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     *     g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_93[12288, 2048] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity tensor
    unsigned g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes[] = {1};
    unsigned g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes[] = {1};
    unsigned g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_93 tensor
    unsigned      g_0_tensor_93_max_sizes[] = {12288, 2048};
    unsigned      g_0_tensor_93_min_sizes[] = {12288, 2048};
    unsigned      g_0_tensor_93             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_93",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_93_max_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_93_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_10_mlp_dense_4h_to_h_output_parallel_linear_fp8_gemm_bf16_6659_0_id;
    unsigned char g_0_10_mlp_dense_4h_to_h_output_parallel_linear_fp8_gemm_bf16_6659_0_params[] = {0, 1};
    addNodeToGraph("fp8_gemm_bf16",
                   {g_0_tensor_73,
                    g_0_tensor_80,
                    g_0_tensor_91_id_29782_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity,
                    g_0_tensor_89_id_29776_10_mlp_dense_4h_to_h_output_parallel_linear_hpu__identity},
                   {g_0_tensor_93},
                   (void*)g_0_10_mlp_dense_4h_to_h_output_parallel_linear_fp8_gemm_bf16_6659_0_params,
                   2,
                   "g_0_10_mlp_dense_4h_to_h_output_parallel_linear_fp8_gemm_bf16_6659_0",
                   0 /*graphIndex*/,
                   &g_0_10_mlp_dense_4h_to_h_output_parallel_linear_fp8_gemm_bf16_6659_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_SLICING_BOTH_PRODUCER_CHAINS", "true");
    // The reference is unsliced
    addConfigurationToRun(SECOND_RUN, "ENABLE_SLICING_BOTH_PRODUCER_CHAINS", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({g_0_tensor_93, g_0_tensor_66});
}