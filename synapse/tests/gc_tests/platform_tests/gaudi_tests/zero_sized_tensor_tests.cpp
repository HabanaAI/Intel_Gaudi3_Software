#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include <numeric>

class SynTrainingZeroSizedTensorFlow : public SynTrainingTestInfra
{
public:
    // Enable tests on gaudi3 https://jira.habana-labs.com/browse/SW-158934
    SynTrainingZeroSizedTensorFlow() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
};

// TODO populate with more test cases. (SW-64771)
/* single node with no input*/
TEST_F_GC(SynTrainingZeroSizedTensorFlow, random_normal_op_zst)
{
    // Graph #0

    /*************
     * g_0_TPC1_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_1[5, 5] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1 tensor
    unsigned      g_0_tensor_1_max_sizes[] = {2, 5, 0};
    unsigned      g_0_tensor_1_min_sizes[] = {2, 5, 0};
    unsigned      g_0_tensor_1             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "g_0_tensor_1",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          g_0_tensor_1_max_sizes,
                                          3,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_1_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_TPC1_0_id;
    unsigned char g_0_TPC1_0_params[] = {0, 0, 0, 0, 0, 0, 128, 63, 47, 107, 7, 49};
    addNodeToGraph("random_normal_fwd_f32",
                   {},
                   {g_0_tensor_1},
                   (void*)g_0_TPC1_0_params,
                   12,
                   "g_0_TPC1_0",
                   0 /*graphIndex*/,
                   &g_0_TPC1_0_id);

    compileTopology("random_normal_op_zst", 0);
    setActualSizes(g_0_tensor_1, {2, 5, 0});
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, gemm_zst_CD_zero, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Graph #0

    /*************
     * g_0_gemm_0 node
     * inputs:
     *     g_0_input1_dynamic[8, 0] min[2, 0] (dtype=bf16)
     *     g_0_input2_dynamic[8, 0] min[2, 0] (dtype=bf16)
     * outputs:
     *     g_0_output_dynamic[8, 8] min[2, 2] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input1_dynamic tensor
    unsigned g_0_input1_dynamic_max_sizes[] = {8, 0};
    unsigned g_0_input1_dynamic_min_sizes[] = {2, 0};
    unsigned g_0_input1_dynamic             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_input1_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_input1_dynamic_max_sizes,
                                                2,
                                                syn_type_bf16,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_input1_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_input2_dynamic tensor
    unsigned g_0_input2_dynamic_max_sizes[] = {8, 0};
    unsigned g_0_input2_dynamic_min_sizes[] = {2, 0};
    unsigned g_0_input2_dynamic             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_input2_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_input2_dynamic_max_sizes,
                                                2,
                                                syn_type_bf16,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_input2_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned      g_0_output_dynamic_max_sizes[] = {8, 8};
    unsigned      g_0_output_dynamic_min_sizes[] = {2, 2};
    unsigned      g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                2,
                                                syn_type_bf16,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gemm_0_id;
    unsigned char g_0_gemm_0_params[] = {1, 0};
    addNodeToGraph("gemm",
                   {g_0_input1_dynamic, g_0_input2_dynamic},
                   {g_0_output_dynamic},
                   (void*)g_0_gemm_0_params,
                   2,
                   "g_0_gemm_0",
                   0 /*graphIndex*/,
                   &g_0_gemm_0_id);

    compileTopology("gemm_zst_CD_zero", 0);
    setActualSizes(g_0_input1_dynamic, {2, 0});
    setActualSizes(g_0_input2_dynamic, {2, 0});
    setActualSizes(g_0_output_dynamic, {2, 2});
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, const_zst_test_fwd_f32)
{
    unsigned      g_0_tensor_0_max_sizes[] = {7, 7, 8, 0};
    unsigned      g_0_tensor_0_min_sizes[] = {7, 7, 8, 0};
    unsigned      g_0_tensor_0             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "g_0_tensor_0",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          g_0_tensor_0_max_sizes,
                                          4,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_0_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId     constant_f32_0_id;
    unsigned char constant_f32_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("constant_f32",
                   {},
                   {g_0_tensor_0},
                   (void*)constant_f32_0_params,
                   4,
                   "constant_f32_0",
                   0 /*graphIndex*/,
                   &constant_f32_0_id);

    compileTopology("const_zst_test_fwd_f32", 0);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, acos_zst_test_fwd_f32)
{
    // Graph #0

    /*************
     * g_0_acos_fwd_f32_0 node
     * inputs:
     *     g_0_input_dynamic[0] (dtype=float32)
     * outputs:
     *     g_0_output_dynamic[0] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input_dynamic tensor
    unsigned g_0_input_dynamic_max_sizes[] = {0};
    unsigned g_0_input_dynamic_min_sizes[] = {0};
    unsigned g_0_input_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_input_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_input_dynamic_max_sizes,
                                               1,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_input_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned  g_0_output_dynamic_max_sizes[] = {0};
    unsigned  g_0_output_dynamic_min_sizes[] = {0};
    unsigned  g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_acos_fwd_f32_0_id;
    addNodeToGraph("acos_fwd_f32",
                   {g_0_input_dynamic},
                   {g_0_output_dynamic},
                   nullptr,
                   0,
                   "g_0_acos_fwd_f32_0",
                   0 /*graphIndex*/,
                   &g_0_acos_fwd_f32_0_id);

    compileTopology("acos_zst_test_fwd_f32", 0);
    setActualSizes(g_0_input_dynamic, g_0_input_dynamic_max_sizes);
    setActualSizes(g_0_output_dynamic, g_0_output_dynamic_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, bn_zst_test_fwd_bf16, {synDeviceGaudi, synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0_batch_norm_fwd_bf16_0 node
     * inputs:
     *     g_0_input_dynamic[4, 229, 229, 0] (dtype=bf16)
     *     g_0_gamma_dynamic[4] (dtype=float32)
     *     g_0_beta_dynamic[4] (dtype=float32)
     *     g_0_running_mean_dynamic[4] (dtype=float32)
     *     g_0_running_variance_dynamic[4] (dtype=float32)
     * outputs:
     *     g_0_output_dynamic[4, 229, 229, 0] (dtype=bf16)
     *     g_0_output_mean_dynamic[4] (dtype=float32)
     *     g_0_output_istd_dynamic[4] (dtype=float32)
     *     g_0_output_run_mean_dynamic[4] (dtype=float32)
     *     g_0_output_run_var_dynamic[4] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input_dynamic tensor
    unsigned g_0_input_dynamic_max_sizes[] = {4, 229, 229, 0};
    unsigned g_0_input_dynamic_min_sizes[] = {4, 229, 229, 0};
    unsigned g_0_input_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_input_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_input_dynamic_max_sizes,
                                               4,
                                               syn_type_bf16,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_input_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_gamma_dynamic tensor
    unsigned g_0_gamma_dynamic_max_sizes[] = {4};
    unsigned g_0_gamma_dynamic_min_sizes[] = {4};
    unsigned g_0_gamma_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_gamma_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_gamma_dynamic_max_sizes,
                                               1,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_gamma_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_beta_dynamic tensor
    unsigned g_0_beta_dynamic_max_sizes[] = {4};
    unsigned g_0_beta_dynamic_min_sizes[] = {4};
    unsigned g_0_beta_dynamic             = createTensors(1,
                                              INPUT_TENSOR,
                                              true,
                                              "g_0_beta_dynamic",
                                              MEM_INIT_ALL_ZERO,
                                              nullptr,
                                              g_0_beta_dynamic_max_sizes,
                                              1,
                                              syn_type_single,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              false,
                                              g_0_beta_dynamic_min_sizes,
                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_running_mean_dynamic tensor
    unsigned g_0_running_mean_dynamic_max_sizes[] = {4};
    unsigned g_0_running_mean_dynamic_min_sizes[] = {4};
    unsigned g_0_running_mean_dynamic             = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_running_mean_dynamic",
                                                      MEM_INIT_ALL_ZERO,
                                                      nullptr,
                                                      g_0_running_mean_dynamic_max_sizes,
                                                      1,
                                                      syn_type_single,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_running_mean_dynamic_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_running_variance_dynamic tensor
    unsigned g_0_running_variance_dynamic_max_sizes[] = {4};
    unsigned g_0_running_variance_dynamic_min_sizes[] = {4};
    unsigned g_0_running_variance_dynamic             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_running_variance_dynamic",
                                                          MEM_INIT_ALL_ZERO,
                                                          nullptr,
                                                          g_0_running_variance_dynamic_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_running_variance_dynamic_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned g_0_output_dynamic_max_sizes[] = {4, 229, 229, 0};
    unsigned g_0_output_dynamic_min_sizes[] = {4, 229, 229, 0};
    unsigned g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                4,
                                                syn_type_bf16,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_output_mean_dynamic tensor
    unsigned g_0_output_mean_dynamic_max_sizes[] = {4};
    unsigned g_0_output_mean_dynamic_min_sizes[] = {4};
    unsigned g_0_output_mean_dynamic             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     true,
                                                     "g_0_output_mean_dynamic",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_output_mean_dynamic_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_output_mean_dynamic_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_output_istd_dynamic tensor
    unsigned g_0_output_istd_dynamic_max_sizes[] = {4};
    unsigned g_0_output_istd_dynamic_min_sizes[] = {4};
    unsigned g_0_output_istd_dynamic             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     true,
                                                     "g_0_output_istd_dynamic",
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     g_0_output_istd_dynamic_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_output_istd_dynamic_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_output_run_mean_dynamic tensor
    unsigned g_0_output_run_mean_dynamic_max_sizes[] = {4};
    unsigned g_0_output_run_mean_dynamic_min_sizes[] = {4};
    unsigned g_0_output_run_mean_dynamic             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_output_run_mean_dynamic",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_output_run_mean_dynamic_max_sizes,
                                                         1,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_output_run_mean_dynamic_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_output_run_var_dynamic tensor
    unsigned      g_0_output_run_var_dynamic_max_sizes[] = {4};
    unsigned      g_0_output_run_var_dynamic_min_sizes[] = {4};
    unsigned      g_0_output_run_var_dynamic             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_output_run_var_dynamic",
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        g_0_output_run_var_dynamic_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_output_run_var_dynamic_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_batch_norm_fwd_bf16_0_id;
    unsigned char g_0_batch_norm_fwd_bf16_0_params[] = {0, 0, 0, 0, 102, 102, 102, 63, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_input_dynamic,
                    g_0_gamma_dynamic,
                    g_0_beta_dynamic,
                    g_0_running_mean_dynamic,
                    g_0_running_variance_dynamic},
                   {g_0_output_dynamic,
                    g_0_output_mean_dynamic,
                    g_0_output_istd_dynamic,
                    g_0_output_run_mean_dynamic,
                    g_0_output_run_var_dynamic},
                   (void*)g_0_batch_norm_fwd_bf16_0_params,
                   12,
                   "g_0_batch_norm_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &g_0_batch_norm_fwd_bf16_0_id);

    compileTopology("bn_zst_test_fwd_bf16", 0);
    setActualSizes(g_0_input_dynamic, {4, 229, 229, 0});
    setActualSizes(g_0_output_dynamic, {4, 229, 229, 0});
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, concat_zst_test_fwd_u8)
{
    // Graph #0

    /*************
     * g_0_concat_0 node
     * inputs:
     *     g_0_input0[0] (dtype=uint8)
     *     g_0_input1[0] (dtype=uint8)
     *     g_0_input2[0] (dtype=uint8)
     *     g_0_input3[0] (dtype=uint8)
     * outputs:
     *     g_0_output[0] (dtype=uint8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input0 tensor
    unsigned g_0_input0_max_sizes[] = {0};
    unsigned g_0_input0_min_sizes[] = {0};
    unsigned g_0_input0             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "g_0_input0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_input0_max_sizes,
                                        1,
                                        syn_type_uint8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_input0_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_input1 tensor
    unsigned g_0_input1_max_sizes[] = {0};
    unsigned g_0_input1_min_sizes[] = {0};
    unsigned g_0_input1             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "g_0_input1",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_input1_max_sizes,
                                        1,
                                        syn_type_uint8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_input1_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_input2 tensor
    unsigned g_0_input2_max_sizes[] = {0};
    unsigned g_0_input2_min_sizes[] = {0};
    unsigned g_0_input2             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "g_0_input2",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_input2_max_sizes,
                                        1,
                                        syn_type_uint8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_input2_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_input3 tensor
    unsigned g_0_input3_max_sizes[] = {0};
    unsigned g_0_input3_min_sizes[] = {0};
    unsigned g_0_input3             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "g_0_input3",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_input3_max_sizes,
                                        1,
                                        syn_type_uint8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_input3_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_output tensor
    unsigned      g_0_output_max_sizes[] = {0};
    unsigned      g_0_output_min_sizes[] = {0};
    unsigned      g_0_output             = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true,
                                        "g_0_output",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_output_max_sizes,
                                        1,
                                        syn_type_uint8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_output_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_concat_0_id;
    unsigned char g_0_concat_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("concat",
                   {g_0_input0, g_0_input1, g_0_input2, g_0_input3},
                   {g_0_output},
                   (void*)g_0_concat_0_params,
                   4,
                   "g_0_concat_0",
                   0 /*graphIndex*/,
                   &g_0_concat_0_id);

    compileTopology("concat_zst_test_fwd_u8", 0);
    setActualSizes(g_0_input0, g_0_input0_max_sizes);
    setActualSizes(g_0_input1, g_0_input1_max_sizes);
    setActualSizes(g_0_input2, g_0_input2_max_sizes);
    setActualSizes(g_0_input3, g_0_input3_max_sizes);
    setActualSizes(g_0_output, g_0_output_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, gather_zst_test_fwd_i32)
{
    // Graph #0

    /*************
     * g_0_gather_fwd_i32_complex_gather_fwd_i32_2_0 node
     * inputs:
     *     g_0_input1_dynamic[2, 0] (dtype=int32)
     *     g_0_input2_dynamic[2, 0] (dtype=int32)
     * outputs:
     *     g_0_output_dynamic[2, 2, 0] (dtype=int32)
     * ctrl inputs:
     * ctrl outputs:
     *************/
    // handle a case where an intermediate tensor become a graph input tensor
    // create g_0_input1_dynamic tensor
    unsigned g_0_input1_dynamic_max_sizes[] = {2, 0};
    unsigned g_0_input1_dynamic_min_sizes[] = {2, 0};
    unsigned g_0_input1_dynamic             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_input1_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_input1_dynamic_max_sizes,
                                                2,
                                                syn_type_int32,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_input1_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_input2_dynamic tensor
    unsigned g_0_input2_dynamic_max_sizes[] = {2, 1};
    unsigned g_0_input2_dynamic_min_sizes[] = {2, 1};
    unsigned g_0_input2_dynamic             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_input2_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_input2_dynamic_max_sizes,
                                                2,
                                                syn_type_int32,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_input2_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned      g_0_output_dynamic_max_sizes[] = {2, 2, 1};
    unsigned      g_0_output_dynamic_min_sizes[] = {2, 2, 1};
    unsigned      g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                3,
                                                syn_type_int32,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gather_fwd_i32_complex_gather_fwd_i32_2_0_id;
    unsigned char g_0_gather_fwd_i32_complex_gather_fwd_i32_2_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("gather_fwd_i32",
                   {g_0_input1_dynamic, g_0_input2_dynamic},
                   {g_0_output_dynamic},
                   (void*)g_0_gather_fwd_i32_complex_gather_fwd_i32_2_0_params,
                   8,
                   "g_0_gather_fwd_i32_complex_gather_fwd_i32_2_0",
                   0 /*graphIndex*/,
                   &g_0_gather_fwd_i32_complex_gather_fwd_i32_2_0_id);

    compileTopology("gather_zst_test_fwd_i32", 0);
    setActualSizes(g_0_input1_dynamic, g_0_input1_dynamic_max_sizes);
    setActualSizes(g_0_input2_dynamic, g_0_input2_dynamic_max_sizes);
    setActualSizes(g_0_output_dynamic, g_0_output_dynamic_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, identity_zst_test_bf16)
{
    // Graph #0

    /*************
     * g_0_identity_0 node
     * inputs:
     *     g_0_input_dynamic[0] (dtype=bf16)
     * outputs:
     *     g_0_output_dynamic[0] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input_dynamic tensor
    unsigned g_0_input_dynamic_max_sizes[] = {0};
    unsigned g_0_input_dynamic_min_sizes[] = {0};
    unsigned g_0_input_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_input_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_input_dynamic_max_sizes,
                                               1,
                                               syn_type_bf16,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_input_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned  g_0_output_dynamic_max_sizes[] = {0};
    unsigned  g_0_output_dynamic_min_sizes[] = {0};
    unsigned  g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                1,
                                                syn_type_bf16,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_identity_0_id;
    addNodeToGraph("identity",
                   {g_0_input_dynamic},
                   {g_0_output_dynamic},
                   nullptr,
                   0,
                   "g_0_identity_0",
                   0 /*graphIndex*/,
                   &g_0_identity_0_id);

    compileTopology("identity_zst_test_bf16", 0);
    setActualSizes(g_0_input_dynamic, g_0_input_dynamic_max_sizes);
    setActualSizes(g_0_output_dynamic, g_0_output_dynamic_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, sigmoid_zst_test_fwd_f32)
{
    // Graph #0

    /*************
     * g_0_sigmoid_fwd_f32_0 node
     * inputs:
     *     g_0_input_dynamic[0] (dtype=float32)
     * outputs:
     *     g_0_output_dynamic[0] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input_dynamic tensor
    unsigned g_0_input_dynamic_max_sizes[] = {0};
    unsigned g_0_input_dynamic_min_sizes[] = {0};
    unsigned g_0_input_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_input_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_input_dynamic_max_sizes,
                                               1,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_input_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned  g_0_output_dynamic_max_sizes[] = {0};
    unsigned  g_0_output_dynamic_min_sizes[] = {0};
    unsigned  g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_sigmoid_fwd_f32_0_id;
    addNodeToGraph("sigmoid_fwd_f32",
                   {g_0_input_dynamic},
                   {g_0_output_dynamic},
                   nullptr,
                   0,
                   "g_0_sigmoid_fwd_f32_0",
                   0 /*graphIndex*/,
                   &g_0_sigmoid_fwd_f32_0_id);

    compileTopology("sigmoid_zst_test_fwd_f32", 0);
    setActualSizes(g_0_input_dynamic, g_0_input_dynamic_max_sizes);
    setActualSizes(g_0_output_dynamic, g_0_output_dynamic_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, sparsesegmentsum_zst_test_fwd_bf16)
{
    // Graph #0

    /*************
     * g_0_sparse_segment_sum_fwd_bf16_0 node
     * inputs:
     *     g_0_input1_dynamic[0, 3] (dtype=bf16)
     *     g_0_input2_dynamic[1] (dtype=int32)
     *     g_0_input3_dynamic[1] (dtype=int32)
     *     g_0_input_shape[0, 3] (dtype=bf16) (shape tensor)
     * outputs:
     *     g_0_output_dynamic[0, 3] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input1_dynamic tensor
    unsigned g_0_input1_dynamic_max_sizes[] = {0, 3};
    unsigned g_0_input1_dynamic_min_sizes[] = {0, 3};
    unsigned g_0_input1_dynamic             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_input1_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_input1_dynamic_max_sizes,
                                                2,
                                                syn_type_bf16,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_input1_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_input2_dynamic tensor
    unsigned g_0_input2_dynamic_max_sizes[] = {1};
    unsigned g_0_input2_dynamic_min_sizes[] = {1};
    unsigned g_0_input2_dynamic             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_input2_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_input2_dynamic_max_sizes,
                                                1,
                                                syn_type_int32,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_input2_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_input3_dynamic tensor
    unsigned g_0_input3_dynamic_max_sizes[] = {1};
    unsigned g_0_input3_dynamic_min_sizes[] = {1};
    unsigned g_0_input3_dynamic             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_input3_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_input3_dynamic_max_sizes,
                                                1,
                                                syn_type_int32,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_input3_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_input_shape tensor
    unsigned g_0_input_shape_max_sizes[] = {0, 3};
    unsigned g_0_input_shape_min_sizes[] = {0, 3};
    unsigned g_0_input_shape             = createTensors(1,
                                             INPUT_TENSOR,
                                             false,
                                             "g_0_input_shape",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_input_shape_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_input_shape_min_sizes,
                                             synTensorType::SHAPE_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned  g_0_output_dynamic_max_sizes[] = {0, 3};
    unsigned  g_0_output_dynamic_min_sizes[] = {0, 3};
    unsigned  g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                2,
                                                syn_type_bf16,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_sparse_segment_sum_fwd_bf16_0_id;
    addNodeToGraph("sparse_segment_sum_fwd_bf16",
                   {g_0_input1_dynamic, g_0_input2_dynamic, g_0_input3_dynamic, g_0_input_shape},
                   {g_0_output_dynamic},
                   nullptr,
                   0,
                   "g_0_sparse_segment_sum_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &g_0_sparse_segment_sum_fwd_bf16_0_id);

    compileTopology("sparsesegmentsum_zst_test_fwd_bf16", 0);
    setActualSizes(g_0_input_shape, g_0_input_shape_max_sizes);
    setActualSizes(g_0_input1_dynamic, g_0_input1_dynamic_max_sizes);
    setActualSizes(g_0_output_dynamic, g_0_output_dynamic_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, stridedsliceshape_grad_zst_test_bwd_f32)
{
    // Graph #0

    /*************
     * g_0_strided_slice_grad_0 node
     * inputs:
     *     g_0_sliced_dyn[0, 4, 4, 4, 4] (dtype=float32)
     *     g_0_shape_tensor[0, 4, 4, 4, 4] (dtype=float32) (shape tensor)
     * outputs:
     *     g_0_gradout_dyn[0, 4, 4, 4, 4] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_sliced_dyn tensor
    unsigned g_0_sliced_dyn_max_sizes[] = {0, 4, 4, 4, 4};
    unsigned g_0_sliced_dyn_min_sizes[] = {0, 4, 4, 4, 4};
    unsigned g_0_sliced_dyn             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_sliced_dyn",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_sliced_dyn_max_sizes,
                                            5,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_sliced_dyn_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_shape_tensor tensor
    unsigned g_0_shape_tensor_max_sizes[] = {0, 4, 4, 4, 4};
    unsigned g_0_shape_tensor_min_sizes[] = {0, 4, 4, 4, 4};
    unsigned g_0_shape_tensor             = createTensors(1,
                                              INPUT_TENSOR,
                                              false,
                                              "g_0_shape_tensor",
                                              MEM_INIT_ALL_ZERO,
                                              nullptr,
                                              g_0_shape_tensor_max_sizes,
                                              5,
                                              syn_type_single,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              false,
                                              g_0_shape_tensor_min_sizes,
                                              synTensorType::SHAPE_TENSOR)[0];

    // create g_0_gradout_dyn tensor
    unsigned      g_0_gradout_dyn_max_sizes[] = {0, 4, 4, 4, 4};
    unsigned      g_0_gradout_dyn_min_sizes[] = {0, 4, 4, 4, 4};
    unsigned      g_0_gradout_dyn             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_gradout_dyn",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             g_0_gradout_dyn_max_sizes,
                                             5,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_gradout_dyn_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_strided_slice_grad_0_id;
    unsigned char g_0_strided_slice_grad_0_params[] = {
        0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("strided_slice_grad",
                   {g_0_sliced_dyn, g_0_shape_tensor},
                   {g_0_gradout_dyn},
                   (void*)g_0_strided_slice_grad_0_params,
                   400,
                   "g_0_strided_slice_grad_0",
                   0 /*graphIndex*/,
                   &g_0_strided_slice_grad_0_id);

    compileTopology("stridedsliceshape_grad_zst_test_bwd_f32", 0);
    setActualSizes(g_0_shape_tensor, g_0_shape_tensor_max_sizes);
    setActualSizes(g_0_sliced_dyn, g_0_sliced_dyn_max_sizes);
    setActualSizes(g_0_gradout_dyn, g_0_gradout_dyn_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, tanh_zst_test_fwd_f32)
{
    // Graph #0

    /*************
     * g_0_tanh_fwd_f32_0 node
     * inputs:
     *     g_0_input_dynamic[0] (dtype=float32)
     * outputs:
     *     g_0_output_dynamic[0] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input_dynamic tensor
    unsigned g_0_input_dynamic_max_sizes[] = {0};
    unsigned g_0_input_dynamic_min_sizes[] = {0};
    unsigned g_0_input_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_input_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_input_dynamic_max_sizes,
                                               1,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_input_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned  g_0_output_dynamic_max_sizes[] = {0};
    unsigned  g_0_output_dynamic_min_sizes[] = {0};
    unsigned  g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_tanh_fwd_f32_0_id;
    addNodeToGraph("tanh_fwd_f32",
                   {g_0_input_dynamic},
                   {g_0_output_dynamic},
                   nullptr,
                   0,
                   "g_0_tanh_fwd_f32_0",
                   0 /*graphIndex*/,
                   &g_0_tanh_fwd_f32_0_id);

    compileTopology("tanh_zst_test_fwd_f32", 0);
    setActualSizes(g_0_input_dynamic, g_0_input_dynamic_max_sizes);
    setActualSizes(g_0_output_dynamic, g_0_output_dynamic_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, transpose_zst_test_int8, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Graph #0

    /*************
     * g_0_transpose_0 node
     * inputs:
     *     g_0_input_dynamic[4000, 0] min[500, 0] (dtype=int8)
     * outputs:
     *     g_0_output_dynamic[0, 4000] min[0, 500] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input_dynamic tensor
    unsigned g_0_input_dynamic_max_sizes[] = {4000, 0};
    unsigned g_0_input_dynamic_min_sizes[] = {500, 0};
    unsigned g_0_input_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_input_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_input_dynamic_max_sizes,
                                               2,
                                               syn_type_int8,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_input_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_output_dynamic tensor
    unsigned      g_0_output_dynamic_max_sizes[] = {0, 4000};
    unsigned      g_0_output_dynamic_min_sizes[] = {0, 500};
    unsigned      g_0_output_dynamic             = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "g_0_output_dynamic",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                g_0_output_dynamic_max_sizes,
                                                2,
                                                syn_type_int8,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_output_dynamic_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_transpose_0_id;
    unsigned char g_0_transpose_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_input_dynamic},
                   {g_0_output_dynamic},
                   (void*)g_0_transpose_0_params,
                   24,
                   "g_0_transpose_0",
                   0 /*graphIndex*/,
                   &g_0_transpose_0_id);

    compileTopology("transpose_zst_test_int8", 0);
    setActualSizes(g_0_input_dynamic, g_0_input_dynamic_max_sizes);
    setActualSizes(g_0_output_dynamic, g_0_output_dynamic_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, unique_zst_test_fwd_f32)
{
    // Graph #0

    /*************
     * g_0_unique_fwd_f32_0 node
     * inputs:
     *     g_0_input_dynamic[0] (dtype=float32)
     * outputs:
     *     g_0_output1_dynamic[0] (dtype=float32)
     *     g_0_output2_dynamic[5] (dtype=int32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_input_dynamic tensor
    unsigned g_0_input_dynamic_max_sizes[] = {0};
    unsigned g_0_input_dynamic_min_sizes[] = {0};
    unsigned g_0_input_dynamic             = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "g_0_input_dynamic",
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               g_0_input_dynamic_max_sizes,
                                               1,
                                               syn_type_single,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false,
                                               g_0_input_dynamic_min_sizes,
                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_output1_dynamic tensor
    unsigned g_0_output1_dynamic_max_sizes[] = {0};
    unsigned g_0_output1_dynamic_min_sizes[] = {0};
    unsigned g_0_output1_dynamic             = createTensors(1,
                                                 OUTPUT_TENSOR,
                                                 true,
                                                 "g_0_output1_dynamic",
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 g_0_output1_dynamic_max_sizes,
                                                 1,
                                                 syn_type_single,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_output1_dynamic_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_output2_dynamic tensor
    unsigned      g_0_output2_dynamic_max_sizes[] = {5};
    unsigned      g_0_output2_dynamic_min_sizes[] = {5};
    unsigned      g_0_output2_dynamic             = createTensors(1,
                                                 OUTPUT_TENSOR,
                                                 true,
                                                 "g_0_output2_dynamic",
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 g_0_output2_dynamic_max_sizes,
                                                 1,
                                                 syn_type_int32,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_output2_dynamic_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_unique_fwd_f32_0_id;
    unsigned char g_0_unique_fwd_f32_0_params[] = {0, 0, 0, 0, 0, 0, 0, 0, 251, 255, 255, 255};
    addNodeToGraph("unique_fwd_f32",
                   {g_0_input_dynamic},
                   {g_0_output1_dynamic, g_0_output2_dynamic},
                   (void*)g_0_unique_fwd_f32_0_params,
                   12,
                   "g_0_unique_fwd_f32_0",
                   0 /*graphIndex*/,
                   &g_0_unique_fwd_f32_0_id);

    compileTopology("unique_zst_test_fwd_f32", 0);
    setActualSizes(g_0_input_dynamic, g_0_input_dynamic_max_sizes);
    setActualSizes(g_0_output1_dynamic, g_0_output1_dynamic_max_sizes);
    runTopology();
}
TEST_F_GC(SynTrainingZeroSizedTensorFlow, const_zst_concat_test_fwd_f32, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // create g_0_input0 tensor
    unsigned g_0_input0_max_sizes[] = {4, 0};
    unsigned g_0_input0_min_sizes[] = {4, 0};
    unsigned g_0_input0             = createTensors(1,
                                        INPUT_TENSOR,
                                        false,
                                        "g_0_input0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_input0_max_sizes,
                                        2,
                                        syn_type_float,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_input0_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    synNodeId     constant_f32_0_id;
    unsigned char constant_f32_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("constant_f32",
                   {},
                   {g_0_input0},
                   (void*)constant_f32_0_params,
                   4,
                   "constant_f32_0",
                   0 /*graphIndex*/,
                   &constant_f32_0_id);

    // create g_0_input1 tensor
    unsigned g_0_input1_max_sizes[] = {4, 1};
    unsigned g_0_input1_min_sizes[] = {4, 1};
    unsigned g_0_input1             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "g_0_input1",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_input1_max_sizes,
                                        2,
                                        syn_type_float,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_input1_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_output tensor
    unsigned      g_0_output_max_sizes[] = {4, 1};
    unsigned      g_0_output_min_sizes[] = {4, 1};
    unsigned      g_0_output             = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true,
                                        "g_0_output",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        g_0_output_max_sizes,
                                        2,
                                        syn_type_float,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_output_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_concat_0_id;
    unsigned char g_0_concat_0_params[] = {1, 0, 0, 0};
    addNodeToGraph("concat",
                   {g_0_input0, g_0_input1},
                   {g_0_output},
                   (void*)g_0_concat_0_params,
                   4,
                   "g_0_concat_0",
                   0 /*graphIndex*/,
                   &g_0_concat_0_id);

    compileTopology("const_zst_concat_test_fwd_f32", 0);
    setActualSizes(g_0_input0, g_0_input0_max_sizes);
    setActualSizes(g_0_input1, g_0_input1_max_sizes);
    setActualSizes(g_0_output, g_0_output_max_sizes);
    runTopology();
}

TEST_F_GC(SynTrainingZeroSizedTensorFlow, zero_sized_reshape_with_concat)
{
    unsigned zeroTensorSizes[] = {17, 0};
    unsigned reshapeIn         = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "reshapeIn",
                                       MEM_INIT_ALL_ONES,
                                       nullptr,
                                       zeroTensorSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       zeroTensorSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned reshapeOut = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "reshapeOut",
                                        MEM_INIT_ALL_ONES,
                                        nullptr,
                                        zeroTensorSizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        zeroTensorSizes,
                                        synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("reshape", {reshapeIn}, {reshapeOut}, nullptr, 0, "zero_reshape");

    unsigned                  concat1InSizes[] = {17, 4};
    std::array<float, 4 * 17> concat1InInit;
    std::iota(concat1InInit.begin(), concat1InInit.end(), 1);
    unsigned concat1In = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "concat1In",
                                       MEM_INIT_FROM_INITIALIZER,
                                       concat1InInit.data(),
                                       concat1InSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concat1InSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned concat1OutSizes[] = {17, 4};
    unsigned concat1Out        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "concat1Out",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        concat1OutSizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        concat1OutSizes,
                                        synTensorType::DATA_TENSOR)[0];

    synConcatenateParams concat1Params;
    concat1Params.axis = 1;
    addNodeToGraph("concat", {concat1In, reshapeOut}, {concat1Out}, &concat1Params, sizeof(concat1Params), "Concat1");

    unsigned                  concat2InSizes[] = {17, 1};
    std::array<float, 1 * 17> concat2InInit;
    std::iota(concat2InInit.begin(), concat2InInit.end(), concat1InInit.size() + 1);
    unsigned concat2In = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "concat2In",
                                       MEM_INIT_FROM_INITIALIZER,
                                       concat2InInit.data(),
                                       concat2InSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       concat2InSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned concat2OutSizes[] = {17, 5};
    unsigned concat2Out        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true,
                                        "concat2Out",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        concat2OutSizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        concat2OutSizes,
                                        synTensorType::DATA_TENSOR)[0];

    synConcatenateParams concat2Params;
    concat2Params.axis = 1;
    addNodeToGraph("concat", {concat1Out, concat2In}, {concat2Out}, &concat2Params, sizeof(concat2Params), "Concat2");

    compileAndRun();

    const auto  count   = (size_t)concat2OutSizes[0] * concat2OutSizes[1];
    const auto* pOutput = (float*)m_hostBuffers[concat2Out];
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(i + 1, pOutput[i]) << "Mismatch at index " << i << " Expected:" << (i + 1)
                                     << " pOutput: " << pOutput[i];
    }
}