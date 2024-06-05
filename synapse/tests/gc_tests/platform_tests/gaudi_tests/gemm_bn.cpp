#include "gc_gaudi_test_infra.h"
// [SW-57121] Do not insert a user reshape on channels within a bundle
TEST_F_GC(SynTrainingTestInfra, gemm_bn)
{
    // Graph #0

    /*************
     * g_0_gradient_tape_model_ecr_pred_dense_MatMul_gemm_n4324_0 node
     * inputs:
     *     g_0_t8316_gradient_tape_model_ecr_mul_1_0[3, 8] (dtype=float32)
     *     g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0[3, 896] (dtype=float32)
     * outputs:
     *     g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0[896, 8] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t8316_gradient_tape_model_ecr_mul_1_0 tensor
    unsigned g_0_t8316_gradient_tape_model_ecr_mul_1_0_max_sizes[] = {3,8};
    unsigned g_0_t8316_gradient_tape_model_ecr_mul_1_0_min_sizes[] = {3,8};
    unsigned g_0_t8316_gradient_tape_model_ecr_mul_1_0 = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_t8316_gradient_tape_model_ecr_mul_1_0",
                                                                   MEM_INIT_ALL_ZERO,
                                                                   nullptr,
                                                                   g_0_t8316_gradient_tape_model_ecr_mul_1_0_max_sizes,
                                                                   2,
                                                                   syn_type_single,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_t8316_gradient_tape_model_ecr_mul_1_0_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0 tensor
    unsigned g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0_max_sizes[] = {3,896};
    unsigned g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0_min_sizes[] = {3,896};
    unsigned g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0 = createTensors(1,
                                                                                INPUT_TENSOR,
                                                                                true,
                                                                                "g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0",
                                                                                MEM_INIT_ALL_ZERO,
                                                                                nullptr,
                                                                                g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0_max_sizes,
                                                                                2,
                                                                                syn_type_single,
                                                                                nullptr,
                                                                                0,
                                                                                0,
                                                                                nullptr,
                                                                                false,
                                                                                g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0_min_sizes,
                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0 tensor
    unsigned g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0_max_sizes[] = {896,8};
    unsigned g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0_min_sizes[] = {896,8};
    unsigned g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0 = createTensors(1,
                                                                               OUTPUT_TENSOR,
                                                                               false,
                                                                               "g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0",
                                                                               MEM_INIT_ALL_ZERO,
                                                                               nullptr,
                                                                               g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0_max_sizes,
                                                                               2,
                                                                               syn_type_single,
                                                                               nullptr,
                                                                               0,
                                                                               0,
                                                                               nullptr,
                                                                               false,
                                                                               g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0_min_sizes,
                                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_tape_model_ecr_pred_dense_MatMul_gemm_n4324_0_id;
    unsigned char g_0_gradient_tape_model_ecr_pred_dense_MatMul_gemm_n4324_0_params[] = {0,1};
    addNodeToGraph("gemm", {g_0_t8316_gradient_tape_model_ecr_mul_1_0, g_0_t8237_model_ecr_pred_dense_matmul_readvariableop_0}, {g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0}, (void*)g_0_gradient_tape_model_ecr_pred_dense_MatMul_gemm_n4324_0_params, 2, "g_0_gradient_tape_model_ecr_pred_dense_MatMul_gemm_n4324_0", 0 /*graphIndex*/, &g_0_gradient_tape_model_ecr_pred_dense_MatMul_gemm_n4324_0_id);

    /*************
     * g_0_gradient_tape_model_ecr_pred_dense_flatten_Reshape_reshape_n4413_0 node
     * inputs:
     *     g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0[896, 8] (dtype=float32)
     *     g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape[64, 7, 2, 8] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0[64, 7, 2, 8] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape tensor
    unsigned g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape = createTensors(1,
                                                                                      INPUT_TENSOR,
                                                                                      false,
                                                                                      "g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape",
                                                                                      MEM_INIT_ALL_ZERO,
                                                                                      nullptr,
                                                                                      g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape_max_sizes,
                                                                                      4,
                                                                                      syn_type_uint32,
                                                                                      nullptr,
                                                                                      0,
                                                                                      0,
                                                                                      nullptr,
                                                                                      false,
                                                                                      g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape_min_sizes,
                                                                                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0 tensor
    unsigned g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0 = createTensors(1,
                                                                                        OUTPUT_TENSOR,
                                                                                        false,
                                                                                        "g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0",
                                                                                        MEM_INIT_ALL_ZERO,
                                                                                        nullptr,
                                                                                        g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0_max_sizes,
                                                                                        4,
                                                                                        syn_type_single,
                                                                                        nullptr,
                                                                                        0,
                                                                                        0,
                                                                                        nullptr,
                                                                                        false,
                                                                                        g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0_min_sizes,
                                                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_tape_model_ecr_pred_dense_flatten_Reshape_reshape_n4413_0_id;
    addNodeToGraph("reshape", {g_0_t8318_gradient_tape_model_ecr_pred_dense_MatMul_0, g_0_t8433_gradient_tape_model_ecr_pred_dense_flatten_Reshape}, {g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0}, nullptr, 0, "g_0_gradient_tape_model_ecr_pred_dense_flatten_Reshape_reshape_n4413_0", 0 /*graphIndex*/, &g_0_gradient_tape_model_ecr_pred_dense_flatten_Reshape_reshape_n4413_0_id);

    /*************
     * g_0_RectifiedAdam_gradients_AddN_23_add_fwd_f32_n4448_0 node
     * inputs:
     *     g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0[64, 7, 2, 8] (dtype=float32)
     *     g_0_t8480_RectifiedAdam_gradients_AddN_23[64, 7, 2, 8] (dtype=float32)
     * outputs:
     *     g_0_t8479_RectifiedAdam_gradients_AddN_23_0[64, 7, 2, 8] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t8480_RectifiedAdam_gradients_AddN_23 tensor
    unsigned g_0_t8480_RectifiedAdam_gradients_AddN_23_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8480_RectifiedAdam_gradients_AddN_23_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8480_RectifiedAdam_gradients_AddN_23 = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_t8480_RectifiedAdam_gradients_AddN_23",
                                                                   MEM_INIT_ALL_ZERO,
                                                                   nullptr,
                                                                   g_0_t8480_RectifiedAdam_gradients_AddN_23_max_sizes,
                                                                   4,
                                                                   syn_type_single,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_t8480_RectifiedAdam_gradients_AddN_23_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_t8479_RectifiedAdam_gradients_AddN_23_0 tensor
    unsigned g_0_t8479_RectifiedAdam_gradients_AddN_23_0_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8479_RectifiedAdam_gradients_AddN_23_0_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8479_RectifiedAdam_gradients_AddN_23_0 = createTensors(1,
                                                                     OUTPUT_TENSOR,
                                                                     false,
                                                                     "g_0_t8479_RectifiedAdam_gradients_AddN_23_0",
                                                                     MEM_INIT_ALL_ZERO,
                                                                     nullptr,
                                                                     g_0_t8479_RectifiedAdam_gradients_AddN_23_0_max_sizes,
                                                                     4,
                                                                     syn_type_single,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_t8479_RectifiedAdam_gradients_AddN_23_0_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_RectifiedAdam_gradients_AddN_23_add_fwd_f32_n4448_0_id;
    addNodeToGraph("add_fwd_f32", {g_0_t8432_gradient_tape_model_ecr_pred_dense_flatten_Reshape_0, g_0_t8480_RectifiedAdam_gradients_AddN_23}, {g_0_t8479_RectifiedAdam_gradients_AddN_23_0}, nullptr, 0, "g_0_RectifiedAdam_gradients_AddN_23_add_fwd_f32_n4448_0", 0 /*graphIndex*/, &g_0_RectifiedAdam_gradients_AddN_23_add_fwd_f32_n4448_0_id);

    /*************
     * g_0_gradient_tape_model_resUnit7_redu_activation_ReluGrad_relu_bwd_f32_n4449_0 node
     * inputs:
     *     g_0_t8479_RectifiedAdam_gradients_AddN_23_0[64, 7, 2, 8] (dtype=float32)
     *     g_0_t8292_model_resunit7_redu_activation_relu_0[64, 7, 2, 8] (dtype=float32)
     * outputs:
     *     g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0[64, 7, 2, 8] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t8292_model_resunit7_redu_activation_relu_0 tensor
    unsigned g_0_t8292_model_resunit7_redu_activation_relu_0_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8292_model_resunit7_redu_activation_relu_0_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8292_model_resunit7_redu_activation_relu_0 = createTensors(1,
                                                                         INPUT_TENSOR,
                                                                         true,
                                                                         "g_0_t8292_model_resunit7_redu_activation_relu_0",
                                                                         MEM_INIT_ALL_ZERO,
                                                                         nullptr,
                                                                         g_0_t8292_model_resunit7_redu_activation_relu_0_max_sizes,
                                                                         4,
                                                                         syn_type_single,
                                                                         nullptr,
                                                                         0,
                                                                         0,
                                                                         nullptr,
                                                                         false,
                                                                         g_0_t8292_model_resunit7_redu_activation_relu_0_min_sizes,
                                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0 tensor
    unsigned g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0 = createTensors(1,
                                                                                           OUTPUT_TENSOR,
                                                                                           false,
                                                                                           "g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0",
                                                                                           MEM_INIT_ALL_ZERO,
                                                                                           nullptr,
                                                                                           g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0_max_sizes,
                                                                                           4,
                                                                                           syn_type_single,
                                                                                           nullptr,
                                                                                           0,
                                                                                           0,
                                                                                           nullptr,
                                                                                           false,
                                                                                           g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0_min_sizes,
                                                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_tape_model_resUnit7_redu_activation_ReluGrad_relu_bwd_f32_n4449_0_id;
    addNodeToGraph("relu_bwd_f32", {g_0_t8479_RectifiedAdam_gradients_AddN_23_0, g_0_t8292_model_resunit7_redu_activation_relu_0}, {g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0}, nullptr, 0, "g_0_gradient_tape_model_resUnit7_redu_activation_ReluGrad_relu_bwd_f32_n4449_0", 0 /*graphIndex*/, &g_0_gradient_tape_model_resUnit7_redu_activation_ReluGrad_relu_bwd_f32_n4449_0_id);

    /*************
     * g_0_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n4454_0 node
     * inputs:
     *     g_0_t8302_model_resunit7_redu_biasadd_0[64, 7, 2, 8] (dtype=float32)
     *     g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0[64, 7, 2, 8] (dtype=float32)
     *     g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3[64] (dtype=float32)
     *     g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3[64] (dtype=float32)
     *     g_0_t8303_model_resunit7_redu_bn_readvariableop_0[64] (dtype=float32)
     * outputs:
     *     g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0[64, 7, 2, 8] (dtype=float32)
     *     g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2[64] (dtype=float32)
     *     g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t8302_model_resunit7_redu_biasadd_0 tensor
    unsigned g_0_t8302_model_resunit7_redu_biasadd_0_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8302_model_resunit7_redu_biasadd_0_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8302_model_resunit7_redu_biasadd_0 = createTensors(1,
                                                                 INPUT_TENSOR,
                                                                 true,
                                                                 "g_0_t8302_model_resunit7_redu_biasadd_0",
                                                                 MEM_INIT_ALL_ZERO,
                                                                 nullptr,
                                                                 g_0_t8302_model_resunit7_redu_biasadd_0_max_sizes,
                                                                 4,
                                                                 syn_type_single,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_t8302_model_resunit7_redu_biasadd_0_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3 tensor
    unsigned g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3_max_sizes[] = {64};
    unsigned g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3_min_sizes[] = {64};
    unsigned g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3 = createTensors(1,
                                                                             INPUT_TENSOR,
                                                                             true,
                                                                             "g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3",
                                                                             MEM_INIT_ALL_ZERO,
                                                                             nullptr,
                                                                             g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3_max_sizes,
                                                                             1,
                                                                             syn_type_single,
                                                                             nullptr,
                                                                             0,
                                                                             0,
                                                                             nullptr,
                                                                             false,
                                                                             g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3_min_sizes,
                                                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3 tensor
    unsigned g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_max_sizes[] = {64};
    unsigned g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_min_sizes[] = {64};
    unsigned g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3 = createTensors(1,
                                                                                             INPUT_TENSOR,
                                                                                             true,
                                                                                             "g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3",
                                                                                             MEM_INIT_ALL_ZERO,
                                                                                             nullptr,
                                                                                             g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_max_sizes,
                                                                                             1,
                                                                                             syn_type_single,
                                                                                             nullptr,
                                                                                             0,
                                                                                             0,
                                                                                             nullptr,
                                                                                             false,
                                                                                             g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_min_sizes,
                                                                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_t8303_model_resunit7_redu_bn_readvariableop_0 tensor
    unsigned g_0_t8303_model_resunit7_redu_bn_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t8303_model_resunit7_redu_bn_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t8303_model_resunit7_redu_bn_readvariableop_0 = createTensors(1,
                                                                           INPUT_TENSOR,
                                                                           true,
                                                                           "g_0_t8303_model_resunit7_redu_bn_readvariableop_0",
                                                                           MEM_INIT_ALL_ZERO,
                                                                           nullptr,
                                                                           g_0_t8303_model_resunit7_redu_bn_readvariableop_0_max_sizes,
                                                                           1,
                                                                           syn_type_single,
                                                                           nullptr,
                                                                           0,
                                                                           0,
                                                                           nullptr,
                                                                           false,
                                                                           g_0_t8303_model_resunit7_redu_bn_readvariableop_0_min_sizes,
                                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0 tensor
    unsigned g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0_max_sizes[] = {64,7,2,8};
    unsigned g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0_min_sizes[] = {64,7,2,8};
    unsigned g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0 = createTensors(1,
                                                                                               OUTPUT_TENSOR,
                                                                                               true,
                                                                                               "g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0",
                                                                                               MEM_INIT_ALL_ZERO,
                                                                                               nullptr,
                                                                                               g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0_max_sizes,
                                                                                               4,
                                                                                               syn_type_single,
                                                                                               nullptr,
                                                                                               0,
                                                                                               0,
                                                                                               nullptr,
                                                                                               false,
                                                                                               g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0_min_sizes,
                                                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2 tensor
    unsigned g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2_max_sizes[] = {64};
    unsigned g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2_min_sizes[] = {64};
    unsigned g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2 = createTensors(1,
                                                                                               OUTPUT_TENSOR,
                                                                                               true,
                                                                                               "g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2",
                                                                                               MEM_INIT_ALL_ZERO,
                                                                                               nullptr,
                                                                                               g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2_max_sizes,
                                                                                               1,
                                                                                               syn_type_single,
                                                                                               nullptr,
                                                                                               0,
                                                                                               0,
                                                                                               nullptr,
                                                                                               false,
                                                                                               g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2_min_sizes,
                                                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1 tensor
    unsigned g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1_max_sizes[] = {64};
    unsigned g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1_min_sizes[] = {64};
    unsigned g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1 = createTensors(1,
                                                                                               OUTPUT_TENSOR,
                                                                                               true,
                                                                                               "g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1",
                                                                                               MEM_INIT_ALL_ZERO,
                                                                                               nullptr,
                                                                                               g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1_max_sizes,
                                                                                               1,
                                                                                               syn_type_single,
                                                                                               nullptr,
                                                                                               0,
                                                                                               0,
                                                                                               nullptr,
                                                                                               false,
                                                                                               g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1_min_sizes,
                                                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n4454_0_id;
    unsigned char g_0_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n4454_0_params[] = {149,191,214,51,205,204,204,61,111,18,131,58,1,0,240,63};
    addNodeToGraph("batch_norm_bwd_f32", {g_0_t8302_model_resunit7_redu_biasadd_0, g_0_t8481_gradient_tape_model_resUnit7_redu_activation_ReluGrad_0, g_0_t8304_model_resunit7_redu_bn_fusedbatchnormv3_3, g_0_t8488_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3, g_0_t8303_model_resunit7_redu_bn_readvariableop_0}, {g_0_t8482_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_0, g_0_t8484_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_2, g_0_t8483_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_1}, (void*)g_0_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n4454_0_params, 16, "g_0_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n4454_0", 0 /*graphIndex*/, &g_0_gradient_tape_model_resUnit7_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n4454_0_id);


    compileTopology("gemm_bn", 0);
}
