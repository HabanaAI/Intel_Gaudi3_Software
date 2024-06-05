#include "syn_gaudi_two_run_compare_test.h"
TEST_F_GC(SynTrainingTwoRunCompareTest, fuse_cast_and_transpose_to_mme)
{
    // Graph #0

    /*************
     * g_0__reshape_54336_0 node
     * inputs:
     *     g_0_tensor_2755_125521_aten_mul[1024, 4, 33] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2769[1024, 132] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2755_125521_aten_mul tensor
    unsigned g_0_tensor_2755_125521_aten_mul_max_sizes[] = {1024,4,33};
    unsigned g_0_tensor_2755_125521_aten_mul_min_sizes[] = {1024,4,33};
    unsigned g_0_tensor_2755_125521_aten_mul = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_tensor_2755_125521_aten_mul",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_tensor_2755_125521_aten_mul_max_sizes,
                                                         3,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_tensor_2755_125521_aten_mul_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2769 tensor
    unsigned g_0_tensor_2769_max_sizes[] = {1024,132};
    unsigned g_0_tensor_2769_min_sizes[] = {1024,132};
    unsigned g_0_tensor_2769 = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "g_0_tensor_2769",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         g_0_tensor_2769_max_sizes,
                                         2,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         g_0_tensor_2769_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__reshape_54336_0_id;
    addNodeToGraph("reshape", {g_0_tensor_2755_125521_aten_mul}, {g_0_tensor_2769}, nullptr, 0, "g_0__reshape_54336_0", 0 /*graphIndex*/, &g_0__reshape_54336_0_id);

    /*************
     * g_0__transpose_54337_0 node
     * inputs:
     *     g_0_tensor_2768[4096, 132] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2770[132, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2768 tensor
    unsigned g_0_tensor_2768_max_sizes[] = {4096,132};
    unsigned g_0_tensor_2768_min_sizes[] = {4096,132};
    unsigned g_0_tensor_2768 = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "g_0_tensor_2768",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         g_0_tensor_2768_max_sizes,
                                         2,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         g_0_tensor_2768_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2770 tensor
    unsigned g_0_tensor_2770_max_sizes[] = {132,4096};
    unsigned g_0_tensor_2770_min_sizes[] = {132,4096};
    unsigned g_0_tensor_2770 = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "g_0_tensor_2770",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         g_0_tensor_2770_max_sizes,
                                         2,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         g_0_tensor_2770_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__transpose_54337_0_id;
    unsigned char g_0__transpose_54337_0_params[] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0};
    addNodeToGraph("transpose", {g_0_tensor_2768}, {g_0_tensor_2770}, (void*)g_0__transpose_54337_0_params, 24, "g_0__transpose_54337_0", 0 /*graphIndex*/, &g_0__transpose_54337_0_id);

    /*************
     * g_0__gemm_54338_0 node
     * inputs:
     *     g_0_tensor_2770[132, 4096] (dtype=bf16)
     *     g_0_tensor_2769[1024, 132] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2771[1024, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2771 tensor
    unsigned g_0_tensor_2771_max_sizes[] = {1024,4096};
    unsigned g_0_tensor_2771_min_sizes[] = {1024,4096};
    unsigned g_0_tensor_2771 = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "g_0_tensor_2771",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         g_0_tensor_2771_max_sizes,
                                         2,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         g_0_tensor_2771_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__gemm_54338_0_id;
    unsigned char g_0__gemm_54338_0_params[] = {0,0};
    addNodeToGraph("gemm", {g_0_tensor_2770, g_0_tensor_2769}, {g_0_tensor_2771}, (void*)g_0__gemm_54338_0_params, 2, "g_0__gemm_54338_0", 0 /*graphIndex*/, &g_0__gemm_54338_0_id);

    /*************
     * g_0__identity_54339_0 node
     * inputs:
     *     g_0_tensor_2771[1024, 4096] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2772_125536_hpu_matmul_backward[1024, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2772_125536_hpu_matmul_backward tensor
    unsigned g_0_tensor_2772_125536_hpu_matmul_backward_max_sizes[] = {1024,4096};
    unsigned g_0_tensor_2772_125536_hpu_matmul_backward_min_sizes[] = {1024,4096};
    unsigned g_0_tensor_2772_125536_hpu_matmul_backward = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "g_0_tensor_2772_125536_hpu_matmul_backward",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_tensor_2772_125536_hpu_matmul_backward_max_sizes,
                                                                    2,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_tensor_2772_125536_hpu_matmul_backward_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__identity_54339_0_id;
    addNodeToGraph("identity", {g_0_tensor_2771}, {g_0_tensor_2772_125536_hpu_matmul_backward}, nullptr, 0, "g_0__identity_54339_0", 0 /*graphIndex*/, &g_0__identity_54339_0_id);

    /*************
     * g_0__transpose_54340_0 node
     * inputs:
     *     g_0_tensor_2772_125536_hpu_matmul_backward[1024, 4096] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2773_125539_aten_t[4096, 1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2773_125539_aten_t tensor
    unsigned g_0_tensor_2773_125539_aten_t_max_sizes[] = {4096,1024};
    unsigned g_0_tensor_2773_125539_aten_t_min_sizes[] = {4096,1024};
    unsigned g_0_tensor_2773_125539_aten_t = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_tensor_2773_125539_aten_t",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_tensor_2773_125539_aten_t_max_sizes,
                                                       2,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_tensor_2773_125539_aten_t_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__transpose_54340_0_id;
    unsigned char g_0__transpose_54340_0_params[] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0};
    addNodeToGraph("transpose", {g_0_tensor_2772_125536_hpu_matmul_backward}, {g_0_tensor_2773_125539_aten_t}, (void*)g_0__transpose_54340_0_params, 24, "g_0__transpose_54340_0", 0 /*graphIndex*/, &g_0__transpose_54340_0_id);

    /*************
     * g_0__cast_bf16_to_f32_54341_0 node
     * inputs:
     *     g_0_tensor_2773_125539_aten_t[4096, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2774_125545_hpu_cast[4096, 1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2774_125545_hpu_cast tensor
    unsigned g_0_tensor_2774_125545_hpu_cast_max_sizes[] = {4096,1024};
    unsigned g_0_tensor_2774_125545_hpu_cast_min_sizes[] = {4096,1024};
    unsigned g_0_tensor_2774_125545_hpu_cast = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_tensor_2774_125545_hpu_cast",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_tensor_2774_125545_hpu_cast_max_sizes,
                                                         2,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_tensor_2774_125545_hpu_cast_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__cast_bf16_to_f32_54341_0_id;
    unsigned char g_0__cast_bf16_to_f32_54341_0_params[] = {4,0,0,0};
    addNodeToGraph("cast_bf16_to_f32", {g_0_tensor_2773_125539_aten_t}, {g_0_tensor_2774_125545_hpu_cast}, (void*)g_0__cast_bf16_to_f32_54341_0_params, 4, "g_0__cast_bf16_to_f32_54341_0", 0 /*graphIndex*/, &g_0__cast_bf16_to_f32_54341_0_id);

    /*************
     * g_0__memcpy_54342_0 node
     * inputs:
     *     g_0_tensor_2774_125545_hpu_cast[4096, 1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_2775[4096, 1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2775 tensor
    unsigned g_0_tensor_2775_max_sizes[] = {4096,1024};
    unsigned g_0_tensor_2775_min_sizes[] = {4096,1024};
    unsigned g_0_tensor_2775 = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "g_0_tensor_2775",
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         g_0_tensor_2775_max_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         g_0_tensor_2775_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__memcpy_54342_0_id;
    addNodeToGraph("memcpy", {g_0_tensor_2774_125545_hpu_cast}, {g_0_tensor_2775}, nullptr, 0, "g_0__memcpy_54342_0", 0 /*graphIndex*/, &g_0__memcpy_54342_0_id);

    addConfigurationToRun(FIRST_RUN, "FUSE_CAST_TO_MME", "true");
    addConfigurationToRun(SECOND_RUN, "FUSE_CAST_TO_MME", "false");
    compareRunsResults({g_0_tensor_2775});
}
