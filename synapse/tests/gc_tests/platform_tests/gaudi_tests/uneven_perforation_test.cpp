#include "scoped_configuration_change.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"

class SynTrainingUnevenPerforationTest : public SynGaudiTwoRunCompareTest
{
public:
    SynTrainingUnevenPerforationTest() { setSupportedDevices({synDeviceGaudi3}); }

protected:
    void setConfigsForTest(bool layeredBrainEnabled)
    {
        if (layeredBrainEnabled)
        {
            addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "true");
            addConfigurationToRun(FIRST_RUN, "ENABLE_EVALUATE_PERFORATION_UTIL", "true");
        }
        else
        {
            addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
            addConfigurationToRun(FIRST_RUN, "ENABLE_UNEVEN_PERFORATION_IN_MME", "true");
        }

        // The reference is unsliced
        addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_EVALUATE_PERFORATION_UTIL", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_UNEVEN_PERFORATION_IN_MME", "false");
        addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
        addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    }
};

TEST_F_GC(SynTrainingUnevenPerforationTest, DISABLED_uneven_perforation_pt_transformer_1x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_gradient_decoder_5_dropout_module_mult_fwd_bf16_2177_0 node
     * inputs:
     *     g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul[1024, 40, 96] (dtype=bf16)
     *     g_0_tensor_2652[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul[1024, 40, 96] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul tensor
    unsigned g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul_max_sizes[] = {1024, 40, 96};
    unsigned g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul_min_sizes[] = {1024, 40, 96};
    unsigned g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2652 tensor
    unsigned g_0_tensor_2652_max_sizes[] = {1};
    unsigned g_0_tensor_2652_min_sizes[] = {1};
    unsigned g_0_tensor_2652             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_2652",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_2652_max_sizes,
                                             1,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2652_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul tensor
    unsigned g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul_max_sizes[] = {1024, 40, 96};
    unsigned g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul_min_sizes[] = {1024, 40, 96};
    unsigned g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_decoder_5_dropout_module_mult_fwd_bf16_2177_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_2648_id_8017_gradient_decoder_5_dropout_module_aten__mul, g_0_tensor_2652},
                   {g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul},
                   nullptr,
                   0,
                   "g_0_gradient_decoder_5_dropout_module_mult_fwd_bf16_2177_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_decoder_5_dropout_module_mult_fwd_bf16_2177_0_id);

    /*************
     * g_0_gradient_decoder_5_fc2_batch_gemm_2182_0 node
     * inputs:
     *     g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul[1024, 40, 96] (dtype=bf16)
     *     g_0_tensor_2658[4096, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2659[4096, 40, 96] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2658 tensor
    unsigned g_0_tensor_2658_max_sizes[] = {4096, 1024};
    unsigned g_0_tensor_2658_min_sizes[] = {4096, 1024};
    unsigned g_0_tensor_2658             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_2658",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_2658_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2658_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2659 tensor
    unsigned      g_0_tensor_2659_max_sizes[] = {4096, 40, 96};
    unsigned      g_0_tensor_2659_min_sizes[] = {4096, 40, 96};
    unsigned      g_0_tensor_2659             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_2659",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_2659_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2659_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_decoder_5_fc2_batch_gemm_2182_0_id;
    unsigned char g_0_gradient_decoder_5_fc2_batch_gemm_2182_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul, g_0_tensor_2658},
                   {g_0_tensor_2659},
                   (void*)g_0_gradient_decoder_5_fc2_batch_gemm_2182_0_params,
                   2,
                   "g_0_gradient_decoder_5_fc2_batch_gemm_2182_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_decoder_5_fc2_batch_gemm_2182_0_id);

    /*************
     * g_0_gradient_decoder_5_fc2_reshape_2187_0 node
     * inputs:
     *     g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul[1024, 40, 96] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2664[1024, 3840] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2664 tensor
    unsigned  g_0_tensor_2664_max_sizes[] = {1024, 3840};
    unsigned  g_0_tensor_2664_min_sizes[] = {1024, 3840};
    unsigned  g_0_tensor_2664             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_2664",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_2664_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2664_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_decoder_5_fc2_reshape_2187_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul},
                   {g_0_tensor_2664},
                   nullptr,
                   0,
                   "g_0_gradient_decoder_5_fc2_reshape_2187_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_decoder_5_fc2_reshape_2187_0_id);

    /*************
     * g_0_gradient_decoder_5_fc2_gemm_2189_0 node
     * inputs:
     *     g_0_tensor_2665[3840, 4096] (dtype=bf16)
     *     g_0_tensor_2664[1024, 3840] (dtype=bf16)
     * outputs:
     *     g_0_tensor_2666[1024, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_2665 tensor
    unsigned g_0_tensor_2665_max_sizes[] = {3840, 4096};
    unsigned g_0_tensor_2665_min_sizes[] = {3840, 4096};
    unsigned g_0_tensor_2665             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_2665",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_2665_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2665_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2666 tensor
    unsigned      g_0_tensor_2666_max_sizes[] = {1024, 4096};
    unsigned      g_0_tensor_2666_min_sizes[] = {1024, 4096};
    unsigned      g_0_tensor_2666             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_2666",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_2666_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_2666_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_decoder_5_fc2_gemm_2189_0_id;
    unsigned char g_0_gradient_decoder_5_fc2_gemm_2189_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_tensor_2665, g_0_tensor_2664},
                   {g_0_tensor_2666},
                   (void*)g_0_gradient_decoder_5_fc2_gemm_2189_0_params,
                   2,
                   "g_0_gradient_decoder_5_fc2_gemm_2189_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_decoder_5_fc2_gemm_2189_0_id);

    setConfigsForTest(false);
    compareRunsResults(
        {g_0_tensor_2650_id_8019_gradient_decoder_5_dropout_module_aten__mul, g_0_tensor_2666, g_0_tensor_2659});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, uneven_perforation_pt_mobilenet_1x_ASIC)
{
    /*************
     * g_0_features_14_conv_3_batch_norm_fwd_bf16_519_0 node
     * inputs:
     *     g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable[7, 7, 160, 256] (dtype=bf16)
     *     g_0_tensor_616[160] (dtype=float32)
     *     g_0_tensor_617[160] (dtype=float32)
     *     g_0_tensor_618[160] (dtype=float32)
     *     g_0_tensor_619[160] (dtype=float32)
     * outputs:
     *     g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training[7, 7, 160, 256] (dtype=bf16)
     *     g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training[160] (dtype=float32)
     *     g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training[160] (dtype=float32)
     *     g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training[160] (dtype=float32)
     *     g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training[160] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable tensor
    unsigned g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable_max_sizes[] = {7, 7, 160, 256};
    unsigned g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable_min_sizes[] = {7, 7, 160, 256};
    unsigned g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_616 tensor
    unsigned g_0_tensor_616_max_sizes[] = {160};
    unsigned g_0_tensor_616_min_sizes[] = {160};
    unsigned g_0_tensor_616             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_616",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_616_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_616_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_617 tensor
    unsigned g_0_tensor_617_max_sizes[] = {160};
    unsigned g_0_tensor_617_min_sizes[] = {160};
    unsigned g_0_tensor_617             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_617",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_617_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_617_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_618 tensor
    unsigned g_0_tensor_618_max_sizes[] = {160};
    unsigned g_0_tensor_618_min_sizes[] = {160};
    unsigned g_0_tensor_618             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_618",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_618_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_618_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_619 tensor
    unsigned g_0_tensor_619_max_sizes[] = {160};
    unsigned g_0_tensor_619_min_sizes[] = {160};
    unsigned g_0_tensor_619             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_619",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_619_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_619_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training tensor
    unsigned g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training_max_sizes[] = {7, 7, 160, 256};
    unsigned g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training_min_sizes[] = {7, 7, 160, 256};
    unsigned g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training tensor
    unsigned g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training_max_sizes[] = {160};
    unsigned g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training_min_sizes[] = {160};
    unsigned g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training tensor
    unsigned g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training_max_sizes[] = {160};
    unsigned g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training_min_sizes[] = {160};
    unsigned g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training tensor
    unsigned g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training_max_sizes[] = {160};
    unsigned g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training_min_sizes[] = {160};
    unsigned g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training tensor
    unsigned g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training_max_sizes[] = {160};
    unsigned g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training_min_sizes[] = {160};
    unsigned g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_features_14_conv_3_batch_norm_fwd_bf16_519_0_id;
    unsigned char g_0_features_14_conv_3_batch_norm_fwd_bf16_519_0_params[] =
        {0, 0, 0, 0, 205, 204, 204, 61, 172, 197, 39, 55, 1, 0, 0, 0};
    const char* bnInputLayouts[]  = {"WHCN", "", "", "", ""};
    const char* bnOutputLayouts[] = {"WHCN", "", "", "", ""};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_tensor_615_id_2883_features_14_conv_2_aten__convolution_overrideable,
                    g_0_tensor_616,
                    g_0_tensor_617,
                    g_0_tensor_618,
                    g_0_tensor_619},
                   {g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training,
                    g_0_tensor_621_id_2890_features_14_conv_3_hpu__native_batch_norm_training,
                    g_0_tensor_622_id_2892_features_14_conv_3_hpu__native_batch_norm_training,
                    g_0_tensor_623_id_750_features_14_conv_3_hpu__native_batch_norm_training,
                    g_0_tensor_624_id_753_features_14_conv_3_hpu__native_batch_norm_training},
                   (void*)g_0_features_14_conv_3_batch_norm_fwd_bf16_519_0_params,
                   16,
                   "g_0_features_14_conv_3_batch_norm_fwd_bf16_519_0",
                   0 /*graphIndex*/,
                   &g_0_features_14_conv_3_batch_norm_fwd_bf16_519_0_id,
                   bnInputLayouts,
                   bnOutputLayouts);

    /*************
     * g_0_features_15_conv_0_0_spatial_convolution_522_0 node
     * inputs:
     *     g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training[7, 7, 160, 256] (dtype=bf16)
     *     g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast[1, 1, 160, 960] (dtype=bf16)
     * outputs:
     *     g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable[7, 7, 960, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast tensor
    unsigned g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast_max_sizes[] = {1, 1, 160, 960};
    unsigned g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast_min_sizes[] = {1, 1, 160, 960};
    unsigned g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable tensor
    unsigned g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable_max_sizes[] = {7, 7, 960, 256};
    unsigned g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable_min_sizes[] = {7, 7, 960, 256};
    unsigned g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_features_15_conv_0_0_spatial_convolution_522_0_id;
    unsigned char g_0_features_15_conv_0_0_spatial_convolution_522_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const char* convInputLayouts[]  = {"WHCN", "SRCK"};
    const char* convOutputLayouts[] = {"WHCN"};
    addNodeToGraph("spatial_convolution",
                   {g_0_tensor_620_id_2888_features_14_conv_3_hpu__native_batch_norm_training,
                    g_0_tensor_628_id_2899_features_15_conv_0_0_hpu__cast},
                   {g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable},
                   (void*)g_0_features_15_conv_0_0_spatial_convolution_522_0_params,
                   112,
                   "g_0_features_15_conv_0_0_spatial_convolution_522_0",
                   0 /*graphIndex*/,
                   &g_0_features_15_conv_0_0_spatial_convolution_522_0_id,
                   convInputLayouts,
                   convOutputLayouts);

    setConfigsForTest(false);
    compareRunsResults({g_0_tensor_629_id_2902_features_15_conv_0_0_aten__convolution_overrideable});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, uneven_perforation_pt_clip_roberta_hf_8x_ASIC)
{
    /*************
     * g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_add_fwd_bf16_2730_0 node
     * inputs:
     *     g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward[768,
     *50, 64] (dtype=bf16)
     *     g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward[768,
     *50, 64] (dtype=bf16) outputs:
     *     g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add[768,
     *50, 64] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create
    // g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward_max_sizes
            [] = {768, 50, 64};
    unsigned
        g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward_min_sizes
            [] = {768, 50, 64};
    unsigned g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__"
            "linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward_max_sizes
            [] = {768, 50, 64};
    unsigned
        g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward_min_sizes
            [] = {768, 50, 64};
    unsigned g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__"
            "linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add
    // tensor
    unsigned
        g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add_max_sizes
            [] = {768, 50, 64};
    unsigned
        g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add_min_sizes
            [] = {768, 50, 64};
    unsigned g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_add_fwd_bf16_2730_0_id;
    addNodeToGraph(
        "add_fwd_bf16",
        {g_0_tensor_4673_id_16988_gradient_module_vision_model_vision_model_encoder_11_self_attn_v_proj_aten__linear_backward,
         g_0_tensor_4696_id_17020_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_aten__linear_backward},
        {g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add},
        nullptr,
        0,
        "g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_add_fwd_bf16_2730_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_add_fwd_bf16_2730_0_id);

    /*************
     * g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_add_fwd_bf16_2731_0 node
     * inputs:
     *     g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add[768,
     *50, 64] (dtype=bf16)
     *     g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward[768,
     *50, 64] (dtype=bf16) outputs:
     *     g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add[768,
     *50, 64] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create
    // g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward_max_sizes
            [] = {768, 50, 64};
    unsigned
        g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward_min_sizes
            [] = {768, 50, 64};
    unsigned g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__"
            "linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add
    // tensor
    unsigned
        g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add_max_sizes
            [] = {768, 50, 64};
    unsigned
        g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add_min_sizes
            [] = {768, 50, 64};
    unsigned g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_add_fwd_bf16_2731_0_id;
    addNodeToGraph(
        "add_fwd_bf16",
        {g_0_tensor_4730_id_17026_gradient_module_vision_model_vision_model_encoder_11_self_attn_k_proj_hpu__add,
         g_0_tensor_4717_id_17053_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_aten__linear_backward},
        {g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add},
        nullptr,
        0,
        "g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_add_fwd_bf16_2731_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_add_fwd_bf16_2731_0_id);

    /*************
     * g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2732_0 node
     * inputs:
     *     g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add[768,
     *50, 64] (dtype=bf16) outputs: g_0_tensor_4735[768, 3200, 1, 1] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_4735 tensor
    unsigned  g_0_tensor_4735_max_sizes[] = {768, 3200, 1, 1};
    unsigned  g_0_tensor_4735_min_sizes[] = {768, 3200, 1, 1};
    unsigned  g_0_tensor_4735             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_4735",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4735_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4735_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2732_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_tensor_4731_id_17059_gradient_module_vision_model_vision_model_encoder_11_self_attn_q_proj_hpu__add},
        {g_0_tensor_4735},
        nullptr,
        0,
        "g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2732_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2732_0_id);

    /*************
     * g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_layer_norm_bwd_bf16_2737_0 node
     * inputs:
     *     g_0_tensor_4736[768, 3200, 1, 1] (dtype=bf16)
     *     g_0_tensor_4735[768, 3200, 1, 1] (dtype=bf16)
     *     g_0_tensor_4738[1, 3200, 1, 1] (dtype=float32)
     *     g_0_tensor_4739[1, 3200, 1, 1] (dtype=float32)
     *     g_0_tensor_4737[768] (dtype=float32)
     * outputs:
     *     g_0_tensor_4740[768, 3200, 1, 1] (dtype=bf16)
     *     g_0_tensor_4741[768] (dtype=float32)
     *     g_0_tensor_4742[768] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_4736 tensor
    unsigned g_0_tensor_4736_max_sizes[] = {768, 3200, 1, 1};
    unsigned g_0_tensor_4736_min_sizes[] = {768, 3200, 1, 1};
    unsigned g_0_tensor_4736             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_4736",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4736_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4736_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4738 tensor
    unsigned g_0_tensor_4738_max_sizes[] = {1, 3200, 1, 1};
    unsigned g_0_tensor_4738_min_sizes[] = {1, 3200, 1, 1};
    unsigned g_0_tensor_4738             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_4738",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4738_max_sizes,
                                             4,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4738_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4739 tensor
    unsigned g_0_tensor_4739_max_sizes[] = {1, 3200, 1, 1};
    unsigned g_0_tensor_4739_min_sizes[] = {1, 3200, 1, 1};
    unsigned g_0_tensor_4739             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_4739",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4739_max_sizes,
                                             4,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4739_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4737 tensor
    unsigned g_0_tensor_4737_max_sizes[] = {768};
    unsigned g_0_tensor_4737_min_sizes[] = {768};
    unsigned g_0_tensor_4737             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_4737",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4737_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4737_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4740 tensor
    unsigned g_0_tensor_4740_max_sizes[] = {768, 3200, 1, 1};
    unsigned g_0_tensor_4740_min_sizes[] = {768, 3200, 1, 1};
    unsigned g_0_tensor_4740             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_4740",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4740_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4740_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4741 tensor
    unsigned g_0_tensor_4741_max_sizes[] = {768};
    unsigned g_0_tensor_4741_min_sizes[] = {768};
    unsigned g_0_tensor_4741             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_4741",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4741_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4741_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4742 tensor
    unsigned  g_0_tensor_4742_max_sizes[] = {768};
    unsigned  g_0_tensor_4742_min_sizes[] = {768};
    unsigned  g_0_tensor_4742             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_4742",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_4742_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_4742_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_layer_norm_bwd_bf16_2737_0_id;
    unsigned char
        g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_layer_norm_bwd_bf16_2737_0_params[] =
            {0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "layer_norm_bwd_bf16",
        {g_0_tensor_4736, g_0_tensor_4735, g_0_tensor_4738, g_0_tensor_4739, g_0_tensor_4737},
        {g_0_tensor_4740, g_0_tensor_4741, g_0_tensor_4742},
        (void*)g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_layer_norm_bwd_bf16_2737_0_params,
        8,
        "g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_layer_norm_bwd_bf16_2737_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_layer_norm_bwd_bf16_2737_0_id);

    /*************
     * g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2740_0 node
     * inputs:
     *     g_0_tensor_4740[768, 3200, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward[768,
     *50, 64] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create
    // g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward
    // tensor
    unsigned
        g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward_max_sizes
            [] = {768, 50, 64};
    unsigned
        g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward_min_sizes
            [] = {768, 50, 64};
    unsigned g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_"
            "layer_norm_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2740_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_tensor_4740},
        {g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward},
        nullptr,
        0,
        "g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2740_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_reshape_2740_0_id);

    /*************
     * g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_add_fwd_bf16_2747_0 node
     * inputs:
     *     g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add[768, 50,
     *64] (dtype=bf16)
     *     g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward[768,
     *50, 64] (dtype=bf16) outputs:
     *     g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add[768, 50,
     *64] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add tensor
    unsigned
        g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add_max_sizes[] =
            {768, 50, 64};
    unsigned
        g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add_min_sizes[] =
            {768, 50, 64};
    unsigned g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add tensor
    unsigned
        g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add_max_sizes[] =
            {768, 50, 64};
    unsigned
        g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add_min_sizes[] =
            {768, 50, 64};
    unsigned g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_add_fwd_bf16_2747_0_id;
    addNodeToGraph(
        "add_fwd_bf16",
        {g_0_tensor_4649_id_16889_gradient_module_vision_model_vision_model_encoder_11_layer_norm2_hpu__add,
         g_0_tensor_4732_id_17084_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_aten__native_layer_norm_backward},
        {g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add},
        nullptr,
        0,
        "g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_add_fwd_bf16_2747_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_add_fwd_bf16_2747_0_id);

    /*************
     * g_0_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_linear_temp_bwd_bf16_2748_0 node
     * inputs:
     *     g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul[3072, 50,
     *64] (dtype=bf16)
     *     g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add[768, 50,
     *64] (dtype=bf16) g_0_tensor_53[3072, 768] (dtype=bf16) outputs:
     *     g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward[3072,
     *50, 64] (dtype=bf16)
     *     g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward[3072,
     *768] (dtype=bf16)
     *     g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward[768]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul tensor
    unsigned
        g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul_max_sizes[] = {
            3072,
            50,
            64};
    unsigned
        g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul_min_sizes[] = {
            3072,
            50,
            64};
    unsigned g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_53 tensor
    unsigned g_0_tensor_53_max_sizes[] = {3072, 768};
    unsigned g_0_tensor_53_min_sizes[] = {3072, 768};
    unsigned g_0_tensor_53             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_53",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_53_max_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_53_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_max_sizes
            [] = {3072, 50, 64};
    unsigned
        g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_min_sizes
            [] = {3072, 50, 64};
    unsigned g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_max_sizes
            [] = {3072, 768};
    unsigned
        g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_min_sizes
            [] = {3072, 768};
    unsigned g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_max_sizes
            [] = {768};
    unsigned
        g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_min_sizes
            [] = {768};
    unsigned g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_max_sizes,
            1,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_linear_temp_bwd_bf16_2748_0_id;
    unsigned char
        g_0_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_linear_temp_bwd_bf16_2748_0_params[] = {1};
    addNodeToGraph(
        "linear_temp_bwd_bf16",
        {g_0_tensor_1099_id_10699_module_vision_model_vision_model_encoder_10_mlp_activation_fn_aten__mul,
         g_0_tensor_4755_id_17097_gradient_module_vision_model_vision_model_encoder_11_layer_norm1_hpu__add,
         g_0_tensor_53},
        {g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward,
         g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward,
         g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward},
        (void*)g_0_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_linear_temp_bwd_bf16_2748_0_params,
        1,
        "g_0_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_linear_temp_bwd_bf16_2748_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_linear_temp_bwd_bf16_2748_0_id);

    setConfigsForTest(false);
    compareRunsResults(
        {g_0_tensor_4756_id_17115_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward,
         g_0_tensor_4757_id_17117_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward,
         g_0_tensor_4758_id_17119_gradient_module_vision_model_vision_model_encoder_10_mlp_fc2_aten__linear_backward});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, DISABLED_uneven_perforation_pt_wav2vec2_8x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_layer_norm_bwd_bf16_1377_0 node
     * inputs:
     *     g_0_tensor_241[768, 4128, 1, 1] (dtype=bf16)
     *     g_0_tensor_240[768, 4128, 1, 1] (dtype=bf16)
     *     g_0_tensor_243[1, 4128, 1, 1] (dtype=float32)
     *     g_0_tensor_244[1, 4128, 1, 1] (dtype=float32)
     *     g_0_tensor_242[768] (dtype=float32)
     * outputs:
     *     g_0_tensor_245[768, 4128, 1, 1] (dtype=bf16)
     *     g_0_tensor_246[768] (dtype=float32)
     *     g_0_tensor_247[768] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_241 tensor
    unsigned g_0_tensor_241_max_sizes[] = {768, 4128, 1, 1};
    unsigned g_0_tensor_241_min_sizes[] = {768, 4128, 1, 1};
    unsigned g_0_tensor_241             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_241",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_241_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_241_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_240 tensor
    unsigned g_0_tensor_240_max_sizes[] = {768, 4128, 1, 1};
    unsigned g_0_tensor_240_min_sizes[] = {768, 4128, 1, 1};
    unsigned g_0_tensor_240             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_240",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_240_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_240_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_243 tensor
    unsigned g_0_tensor_243_max_sizes[] = {1, 4128, 1, 1};
    unsigned g_0_tensor_243_min_sizes[] = {1, 4128, 1, 1};
    unsigned g_0_tensor_243             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_243",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_243_max_sizes,
                                            4,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_243_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_244 tensor
    unsigned g_0_tensor_244_max_sizes[] = {1, 4128, 1, 1};
    unsigned g_0_tensor_244_min_sizes[] = {1, 4128, 1, 1};
    unsigned g_0_tensor_244             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_244",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_244_max_sizes,
                                            4,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_244_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_242 tensor
    unsigned g_0_tensor_242_max_sizes[] = {768};
    unsigned g_0_tensor_242_min_sizes[] = {768};
    unsigned g_0_tensor_242             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_242",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_242_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_242_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_245 tensor
    unsigned g_0_tensor_245_max_sizes[] = {768, 4128, 1, 1};
    unsigned g_0_tensor_245_min_sizes[] = {768, 4128, 1, 1};
    unsigned g_0_tensor_245             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_245",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_245_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_245_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_246 tensor
    unsigned g_0_tensor_246_max_sizes[] = {768};
    unsigned g_0_tensor_246_min_sizes[] = {768};
    unsigned g_0_tensor_246             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_246",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_246_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_246_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_247 tensor
    unsigned g_0_tensor_247_max_sizes[] = {768};
    unsigned g_0_tensor_247_min_sizes[] = {768};
    unsigned g_0_tensor_247             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_247",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_247_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_247_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_layer_norm_bwd_bf16_1377_0_id;
    unsigned char
        g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_layer_norm_bwd_bf16_1377_0_params
            [] = {0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "layer_norm_bwd_bf16",
        {g_0_tensor_241, g_0_tensor_240, g_0_tensor_243, g_0_tensor_244, g_0_tensor_242},
        {g_0_tensor_245, g_0_tensor_246, g_0_tensor_247},
        (void*)
            g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_layer_norm_bwd_bf16_1377_0_params,
        8,
        "g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_layer_norm_bwd_bf16_1377_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_layer_norm_bwd_bf16_1377_0_id);

    /*************
     * g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_reshape_1380_0 node
     * inputs:
     *     g_0_tensor_245[768, 4128, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward[768,
     *6, 688] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create
    // g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward
    // tensor
    unsigned
        g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward_max_sizes
            [] = {768, 6, 688};
    unsigned
        g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward_min_sizes
            [] = {768, 6, 688};
    unsigned g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten_"
            "_native_layer_norm_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_reshape_1380_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_tensor_245},
        {g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward},
        nullptr,
        0,
        "g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_reshape_1380_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_reshape_1380_0_id);

    /*************
     * g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1385_0 node
     * inputs:
     *     g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward[768,
     *6, 688] (dtype=bf16) g_0_tensor_254__placeholder_1[768, 6, 688] (dtype=int8) outputs: g_0_tensor_256[768, 6, 688]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_254__placeholder_1 tensor
    unsigned g_0_tensor_254__placeholder_1_max_sizes[] = {768, 6, 688};
    unsigned g_0_tensor_254__placeholder_1_min_sizes[] = {768, 6, 688};
    unsigned g_0_tensor_254__placeholder_1             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_tensor_254__placeholder_1",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_tensor_254__placeholder_1_max_sizes,
                                                           3,
                                                           syn_type_int8,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_tensor_254__placeholder_1_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_256 tensor
    unsigned  g_0_tensor_256_max_sizes[] = {768, 6, 688};
    unsigned  g_0_tensor_256_min_sizes[] = {768, 6, 688};
    unsigned  g_0_tensor_256             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_256",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_256_max_sizes,
                                            3,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_256_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1385_0_id;
    addNodeToGraph(
        "mult_fwd_bf16",
        {g_0_tensor_237_id_9086_gradient_module_model_model_orig_model_orig_model_encoder_11_final_layer_norm_aten__native_layer_norm_backward,
         g_0_tensor_254__placeholder_1},
        {g_0_tensor_256},
        nullptr,
        0,
        "g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1385_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1385_0_id);

    /*************
     * g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_constant_bf16_1386_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_257[768, 6, 688] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_257 tensor
    unsigned  g_0_tensor_257_max_sizes[] = {768, 6, 688};
    unsigned  g_0_tensor_257_min_sizes[] = {768, 6, 688};
    unsigned  g_0_tensor_257             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_257",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_257_max_sizes,
                                            3,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_257_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_constant_bf16_1386_0_id;
    unsigned char
        g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_constant_bf16_1386_0_params[] = {228,
                                                                                                                   56,
                                                                                                                   142,
                                                                                                                   63};
    addNodeToGraph(
        "constant_bf16",
        {},
        {g_0_tensor_257},
        (void*)g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_constant_bf16_1386_0_params,
        4,
        "g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_constant_bf16_1386_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_constant_bf16_1386_0_id);

    /*************
     * g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1387_0 node
     * inputs:
     *     g_0_tensor_256[768, 6, 688] (dtype=bf16)
     *     g_0_tensor_257[768, 6, 688] (dtype=bf16)
     * outputs:
     *     g_0_tensor_258[768, 6, 688] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_258 tensor
    unsigned  g_0_tensor_258_max_sizes[] = {768, 6, 688};
    unsigned  g_0_tensor_258_min_sizes[] = {768, 6, 688};
    unsigned  g_0_tensor_258             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_258",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_258_max_sizes,
                                            3,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_258_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1387_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_256, g_0_tensor_257},
                   {g_0_tensor_258},
                   nullptr,
                   0,
                   "g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1387_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_dropout3_mult_fwd_bf16_1387_0_id);

    /*************
     * g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_linear_temp_bwd_bf16_1389_0 node
     * inputs:
     *     g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view[3072, 6,
     *688] (dtype=bf16) g_0_tensor_258[768, 6, 688] (dtype=bf16) g_0_tensor_261__placeholder_2[3072, 768] (dtype=bf16)
     * outputs:
     *     g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward[3072,
     *6, 688] (dtype=bf16)
     *     g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward[3072,
     *768] (dtype=bf16)
     *     g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward[768]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view tensor
    unsigned
        g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view_max_sizes[] =
            {3072, 6, 688};
    unsigned
        g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view_min_sizes[] =
            {3072, 6, 688};
    unsigned g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_261__placeholder_2 tensor
    unsigned g_0_tensor_261__placeholder_2_max_sizes[] = {3072, 768};
    unsigned g_0_tensor_261__placeholder_2_min_sizes[] = {3072, 768};
    unsigned g_0_tensor_261__placeholder_2             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_tensor_261__placeholder_2",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_tensor_261__placeholder_2_max_sizes,
                                                           2,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_tensor_261__placeholder_2_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_max_sizes
            [] = {3072, 6, 688};
    unsigned
        g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_min_sizes
            [] = {3072, 6, 688};
    unsigned g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_max_sizes
            [] = {3072, 768};
    unsigned
        g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_min_sizes
            [] = {3072, 768};
    unsigned g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_max_sizes
            [] = {768};
    unsigned
        g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_min_sizes
            [] = {768};
    unsigned g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_max_sizes,
            1,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_linear_temp_bwd_bf16_1389_0_id;
    unsigned char
        g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_linear_temp_bwd_bf16_1389_0_params[] = {1};
    addNodeToGraph(
        "linear_temp_bwd_bf16",
        {g_0_tensor_260_id_8663_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__view,
         g_0_tensor_258,
         g_0_tensor_261__placeholder_2},
        {g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward,
         g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward,
         g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward},
        (void*)g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_linear_temp_bwd_bf16_1389_0_params,
        1,
        "g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_linear_temp_bwd_bf16_1389_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_linear_temp_bwd_bf16_1389_0_id);

    setConfigsForTest(false);
    compareRunsResults(
        {g_0_tensor_262_id_9103_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward,
         g_0_tensor_263_id_9105_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward,
         g_0_tensor_264_id_9107_gradient_module_model_model_orig_model_orig_model_encoder_11_fc2_aten__linear_backward});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, DISABLED_uneven_perforation_pt_swin_t_hf_8x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_gradient_module_swin_encoder_3_1_layernorm_before_layer_norm_bwd_bf16_2359_0 node
     * inputs:
     *     g_0_tensor_3721[1024, 3136, 1, 1] (dtype=bf16)
     *     g_0_tensor_3720[1024, 3136, 1, 1] (dtype=bf16)
     *     g_0_tensor_3723[1, 3136, 1, 1] (dtype=float32)
     *     g_0_tensor_3724[1, 3136, 1, 1] (dtype=float32)
     *     g_0_tensor_3722[1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_3725[1024, 3136, 1, 1] (dtype=bf16)
     *     g_0_tensor_3726[1024] (dtype=float32)
     *     g_0_tensor_3727[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_3721 tensor
    unsigned g_0_tensor_3721_max_sizes[] = {1024, 3136, 1, 1};
    unsigned g_0_tensor_3721_min_sizes[] = {1024, 3136, 1, 1};
    unsigned g_0_tensor_3721             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3721",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3721_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3721_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3720 tensor
    unsigned g_0_tensor_3720_max_sizes[] = {1024, 3136, 1, 1};
    unsigned g_0_tensor_3720_min_sizes[] = {1024, 3136, 1, 1};
    unsigned g_0_tensor_3720             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3720",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3720_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3720_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3723 tensor
    unsigned g_0_tensor_3723_max_sizes[] = {1, 3136, 1, 1};
    unsigned g_0_tensor_3723_min_sizes[] = {1, 3136, 1, 1};
    unsigned g_0_tensor_3723             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3723",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3723_max_sizes,
                                             4,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3723_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3724 tensor
    unsigned g_0_tensor_3724_max_sizes[] = {1, 3136, 1, 1};
    unsigned g_0_tensor_3724_min_sizes[] = {1, 3136, 1, 1};
    unsigned g_0_tensor_3724             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3724",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3724_max_sizes,
                                             4,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3724_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3722 tensor
    unsigned g_0_tensor_3722_max_sizes[] = {1024};
    unsigned g_0_tensor_3722_min_sizes[] = {1024};
    unsigned g_0_tensor_3722             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3722",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3722_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3722_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3725 tensor
    unsigned g_0_tensor_3725_max_sizes[] = {1024, 3136, 1, 1};
    unsigned g_0_tensor_3725_min_sizes[] = {1024, 3136, 1, 1};
    unsigned g_0_tensor_3725             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_3725",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3725_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3725_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3726 tensor
    unsigned g_0_tensor_3726_max_sizes[] = {1024};
    unsigned g_0_tensor_3726_min_sizes[] = {1024};
    unsigned g_0_tensor_3726             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3726",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3726_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3726_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3727 tensor
    unsigned      g_0_tensor_3727_max_sizes[] = {1024};
    unsigned      g_0_tensor_3727_min_sizes[] = {1024};
    unsigned      g_0_tensor_3727             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3727",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3727_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3727_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_swin_encoder_3_1_layernorm_before_layer_norm_bwd_bf16_2359_0_id;
    unsigned char g_0_gradient_module_swin_encoder_3_1_layernorm_before_layer_norm_bwd_bf16_2359_0_params[] =
        {0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("layer_norm_bwd_bf16",
                   {g_0_tensor_3721, g_0_tensor_3720, g_0_tensor_3723, g_0_tensor_3724, g_0_tensor_3722},
                   {g_0_tensor_3725, g_0_tensor_3726, g_0_tensor_3727},
                   (void*)g_0_gradient_module_swin_encoder_3_1_layernorm_before_layer_norm_bwd_bf16_2359_0_params,
                   8,
                   "g_0_gradient_module_swin_encoder_3_1_layernorm_before_layer_norm_bwd_bf16_2359_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_swin_encoder_3_1_layernorm_before_layer_norm_bwd_bf16_2359_0_id);

    /*************
     * g_0_gradient_module_swin_encoder_3_1_layernorm_before_reshape_2362_0 node
     * inputs:
     *     g_0_tensor_3725[1024, 3136, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward[1024,
     *49, 64] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create
    // g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward
    // tensor
    unsigned
        g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward_max_sizes
            [] = {1024, 49, 64};
    unsigned
        g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward_min_sizes
            [] = {1024, 49, 64};
    unsigned g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_swin_encoder_3_1_layernorm_before_reshape_2362_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_tensor_3725},
        {g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward},
        nullptr,
        0,
        "g_0_gradient_module_swin_encoder_3_1_layernorm_before_reshape_2362_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_swin_encoder_3_1_layernorm_before_reshape_2362_0_id);

    /*************
     * g_0_gradient_module_swin_encoder_3_1_layernorm_before_add_fwd_bf16_2369_0 node
     * inputs:
     *     g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add[1024, 49, 64] (dtype=bf16)
     *     g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward[1024,
     *49, 64] (dtype=bf16) outputs:
     *     g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add[1024, 49, 64]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add tensor
    unsigned g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add_max_sizes[] = {1024,
                                                                                                               49,
                                                                                                               64};
    unsigned g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add_min_sizes[] = {1024,
                                                                                                               49,
                                                                                                               64};
    unsigned g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add tensor
    unsigned g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add_max_sizes[] = {1024,
                                                                                                                49,
                                                                                                                64};
    unsigned g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add_min_sizes[] = {1024,
                                                                                                                49,
                                                                                                                64};
    unsigned g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_swin_encoder_3_1_layernorm_before_add_fwd_bf16_2369_0_id;
    addNodeToGraph(
        "add_fwd_bf16",
        {g_0_tensor_3595_id_15633_gradient_module_swin_encoder_3_1_layernorm_after_hpu__add,
         g_0_tensor_3717_id_15875_gradient_module_swin_encoder_3_1_layernorm_before_aten__native_layer_norm_backward},
        {g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add},
        nullptr,
        0,
        "g_0_gradient_module_swin_encoder_3_1_layernorm_before_add_fwd_bf16_2369_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_swin_encoder_3_1_layernorm_before_add_fwd_bf16_2369_0_id);

    /*************
     * g_0_gradient_module_swin_encoder_3_0_output_dense_linear_temp_bwd_bf16_2370_0 node
     * inputs:
     *     g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu[4096, 49, 64]
     *(dtype=bf16) g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add[1024, 49, 64]
     *(dtype=bf16) g_0_tensor_259[4096, 1024] (dtype=bf16) outputs:
     *     g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward[4096, 49, 64]
     *(dtype=bf16) g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward[4096,
     *1024] (dtype=bf16)
     *     g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward[1024]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu tensor
    unsigned g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu_max_sizes[] =
        {4096, 49, 64};
    unsigned g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu_min_sizes[] =
        {4096, 49, 64};
    unsigned g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_259 tensor
    unsigned g_0_tensor_259_max_sizes[] = {4096, 1024};
    unsigned g_0_tensor_259_min_sizes[] = {4096, 1024};
    unsigned g_0_tensor_259             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_259",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_259_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_259_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward tensor
    unsigned g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_max_sizes[] =
        {4096, 49, 64};
    unsigned g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_min_sizes[] =
        {4096, 49, 64};
    unsigned g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward tensor
    unsigned g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_max_sizes[] =
        {4096, 1024};
    unsigned g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_min_sizes[] =
        {4096, 1024};
    unsigned g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward tensor
    unsigned g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_max_sizes[] =
        {1024};
    unsigned g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_min_sizes[] =
        {1024};
    unsigned g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_max_sizes,
            1,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_swin_encoder_3_0_output_dense_linear_temp_bwd_bf16_2370_0_id;
    unsigned char g_0_gradient_module_swin_encoder_3_0_output_dense_linear_temp_bwd_bf16_2370_0_params[] = {1};
    addNodeToGraph("linear_temp_bwd_bf16",
                   {g_0_tensor_3361_id_15182_module_swin_encoder_3_0_intermediate_intermediate_act_fn_aten__gelu,
                    g_0_tensor_3740_id_15888_gradient_module_swin_encoder_3_1_layernorm_before_hpu__add,
                    g_0_tensor_259},
                   {g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward,
                    g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward,
                    g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward},
                   (void*)g_0_gradient_module_swin_encoder_3_0_output_dense_linear_temp_bwd_bf16_2370_0_params,
                   1,
                   "g_0_gradient_module_swin_encoder_3_0_output_dense_linear_temp_bwd_bf16_2370_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_swin_encoder_3_0_output_dense_linear_temp_bwd_bf16_2370_0_id);

    setConfigsForTest(false);
    compareRunsResults({g_0_tensor_3741_id_15906_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward,
                        g_0_tensor_3742_id_15908_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward,
                        g_0_tensor_3743_id_15910_gradient_module_swin_encoder_3_0_output_dense_aten__linear_backward});
}

TEST_F_GC(SynTrainingUnevenPerforationTest,
          DISABLED_uneven_perforation_pt_hubert_large_pre_training_ptl_8x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_gradient_module_model_transformer_layer_norm_layer_norm_bwd_bf16_21223_0 node
     * inputs:
     *     g_0_tensor_26[1024, 5428, 1, 1] (dtype=bf16)
     *     g_0_tensor_25[1024, 5428, 1, 1] (dtype=bf16)
     *     g_0_tensor_28[1, 5428, 1, 1] (dtype=float32)
     *     g_0_tensor_29[1, 5428, 1, 1] (dtype=float32)
     *     g_0_tensor_27[1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_30[1024, 5428, 1, 1] (dtype=bf16)
     *     g_0_tensor_31[1024] (dtype=float32)
     *     g_0_tensor_32[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_26 tensor
    unsigned g_0_tensor_26_max_sizes[] = {1024, 5428, 1, 1};
    unsigned g_0_tensor_26_min_sizes[] = {1024, 5428, 1, 1};
    unsigned g_0_tensor_26             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
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

    // create g_0_tensor_25 tensor
    unsigned g_0_tensor_25_max_sizes[] = {1024, 5428, 1, 1};
    unsigned g_0_tensor_25_min_sizes[] = {1024, 5428, 1, 1};
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

    // create g_0_tensor_28 tensor
    unsigned g_0_tensor_28_max_sizes[] = {1, 5428, 1, 1};
    unsigned g_0_tensor_28_min_sizes[] = {1, 5428, 1, 1};
    unsigned g_0_tensor_28             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
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

    // create g_0_tensor_29 tensor
    unsigned g_0_tensor_29_max_sizes[] = {1, 5428, 1, 1};
    unsigned g_0_tensor_29_min_sizes[] = {1, 5428, 1, 1};
    unsigned g_0_tensor_29             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_29",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_29_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_29_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_27 tensor
    unsigned g_0_tensor_27_max_sizes[] = {1024};
    unsigned g_0_tensor_27_min_sizes[] = {1024};
    unsigned g_0_tensor_27             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_27",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_27_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_27_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_30 tensor
    unsigned g_0_tensor_30_max_sizes[] = {1024, 5428, 1, 1};
    unsigned g_0_tensor_30_min_sizes[] = {1024, 5428, 1, 1};
    unsigned g_0_tensor_30             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_30",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_30_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_30_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_31 tensor
    unsigned g_0_tensor_31_max_sizes[] = {1024};
    unsigned g_0_tensor_31_min_sizes[] = {1024};
    unsigned g_0_tensor_31             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_31",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_31_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_31_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_32 tensor
    unsigned      g_0_tensor_32_max_sizes[] = {1024};
    unsigned      g_0_tensor_32_min_sizes[] = {1024};
    unsigned      g_0_tensor_32             = createTensors(1,
                                           OUTPUT_TENSOR,
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
    synNodeId     g_0_gradient_module_model_transformer_layer_norm_layer_norm_bwd_bf16_21223_0_id;
    unsigned char g_0_gradient_module_model_transformer_layer_norm_layer_norm_bwd_bf16_21223_0_params[] =
        {0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("layer_norm_bwd_bf16",
                   {g_0_tensor_26, g_0_tensor_25, g_0_tensor_28, g_0_tensor_29, g_0_tensor_27},
                   {g_0_tensor_30, g_0_tensor_31, g_0_tensor_32},
                   (void*)g_0_gradient_module_model_transformer_layer_norm_layer_norm_bwd_bf16_21223_0_params,
                   8,
                   "g_0_gradient_module_model_transformer_layer_norm_layer_norm_bwd_bf16_21223_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_layer_norm_layer_norm_bwd_bf16_21223_0_id);

    /*************
     * g_0_gradient_module_model_transformer_layer_norm_reshape_21226_0 node
     * inputs:
     *     g_0_tensor_30[1024, 5428, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward[1024,
     *236, 23] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward
    // tensor
    unsigned
        g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward_max_sizes
            [] = {1024, 236, 23};
    unsigned
        g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward_min_sizes
            [] = {1024, 236, 23};
    unsigned g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_transformer_layer_norm_reshape_21226_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_tensor_30},
        {g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward},
        nullptr,
        0,
        "g_0_gradient_module_model_transformer_layer_norm_reshape_21226_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_transformer_layer_norm_reshape_21226_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_feed_forward_output_dense_linear_temp_bwd_bf16_21233_0 node
     * inputs:
     *     g_0_tensor_47__placeholder_0[4096, 236, 23] (dtype=bf16)
     *     g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward[1024,
     *236, 23] (dtype=bf16) g_0_tensor_48__placeholder_2[4096, 1024] (dtype=bf16) outputs:
     *     g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward[4096,
     *236, 23] (dtype=bf16)
     *     g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward[4096,
     *1024] (dtype=bf16)
     *     g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward[1024]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_47__placeholder_0 tensor
    unsigned g_0_tensor_47__placeholder_0_max_sizes[] = {4096, 236, 23};
    unsigned g_0_tensor_47__placeholder_0_min_sizes[] = {4096, 236, 23};
    unsigned g_0_tensor_47__placeholder_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_47__placeholder_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_47__placeholder_0_max_sizes,
                                                          3,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_47__placeholder_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_48__placeholder_2 tensor
    unsigned g_0_tensor_48__placeholder_2_max_sizes[] = {4096, 1024};
    unsigned g_0_tensor_48__placeholder_2_min_sizes[] = {4096, 1024};
    unsigned g_0_tensor_48__placeholder_2             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_48__placeholder_2",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_48__placeholder_2_max_sizes,
                                                          2,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_48__placeholder_2_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_max_sizes
            [] = {4096, 236, 23};
    unsigned
        g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_min_sizes
            [] = {4096, 236, 23};
    unsigned g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_max_sizes
            [] = {4096, 1024};
    unsigned
        g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_min_sizes
            [] = {4096, 1024};
    unsigned g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create
    // g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward
    // tensor
    unsigned
        g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_max_sizes
            [] = {1024};
    unsigned
        g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_min_sizes
            [] = {1024};
    unsigned g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_max_sizes,
            1,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_transformer_23_feed_forward_output_dense_linear_temp_bwd_bf16_21233_0_id;
    unsigned char
        g_0_gradient_module_model_transformer_23_feed_forward_output_dense_linear_temp_bwd_bf16_21233_0_params[] = {1};
    addNodeToGraph(
        "linear_temp_bwd_bf16",
        {g_0_tensor_47__placeholder_0,
         g_0_tensor_22_id_272021_gradient_module_model_transformer_layer_norm_aten__native_layer_norm_backward,
         g_0_tensor_48__placeholder_2},
        {g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward,
         g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward,
         g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward},
        (void*)g_0_gradient_module_model_transformer_23_feed_forward_output_dense_linear_temp_bwd_bf16_21233_0_params,
        1,
        "g_0_gradient_module_model_transformer_23_feed_forward_output_dense_linear_temp_bwd_bf16_21233_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_transformer_23_feed_forward_output_dense_linear_temp_bwd_bf16_21233_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_feed_forward_output_dense_cast_bf16_to_f32_21234_0 node
     * inputs:
     *     g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward[4096,
     *1024] (dtype=bf16) outputs: g_0_tensor_53[4096, 1024] (dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_53 tensor
    unsigned      g_0_tensor_53_max_sizes[] = {4096, 1024};
    unsigned      g_0_tensor_53_min_sizes[] = {4096, 1024};
    unsigned      g_0_tensor_53             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_53",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_53_max_sizes,
                                           2,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_53_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_model_transformer_23_feed_forward_output_dense_cast_bf16_to_f32_21234_0_id;
    unsigned char g_0_gradient_module_model_transformer_23_feed_forward_output_dense_cast_bf16_to_f32_21234_0_params[] =
        {0, 0, 0, 0};
    addNodeToGraph(
        "cast_bf16_to_f32",
        {g_0_tensor_50_id_272053_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward},
        {g_0_tensor_53},
        (void*)g_0_gradient_module_model_transformer_23_feed_forward_output_dense_cast_bf16_to_f32_21234_0_params,
        4,
        "g_0_gradient_module_model_transformer_23_feed_forward_output_dense_cast_bf16_to_f32_21234_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_transformer_23_feed_forward_output_dense_cast_bf16_to_f32_21234_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_feed_forward_output_dense_memcpy_21235_0 node
     * inputs:
     *     g_0_tensor_53[4096, 1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_54[4096, 1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_54 tensor
    unsigned  g_0_tensor_54_max_sizes[] = {4096, 1024};
    unsigned  g_0_tensor_54_min_sizes[] = {4096, 1024};
    unsigned  g_0_tensor_54             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_54",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_54_max_sizes,
                                           2,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_54_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_transformer_23_feed_forward_output_dense_memcpy_21235_0_id;
    addNodeToGraph("memcpy",
                   {g_0_tensor_53},
                   {g_0_tensor_54},
                   nullptr,
                   0,
                   "g_0_gradient_module_model_transformer_23_feed_forward_output_dense_memcpy_21235_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_23_feed_forward_output_dense_memcpy_21235_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_feed_forward_output_dense_mult_fwd_f32_21244_0 node
     * inputs:
     *     g_0_tensor_54[4096, 1024] (dtype=float32)
     *     g_0_tensor_72__placeholder_1[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul[4096, 1024]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_72__placeholder_1 tensor
    unsigned g_0_tensor_72__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_72__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_72__placeholder_1             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_72__placeholder_1",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_72__placeholder_1_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_72__placeholder_1_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul tensor
    unsigned
        g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul_max_sizes[] = {
            4096,
            1024};
    unsigned
        g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul_min_sizes[] = {
            4096,
            1024};
    unsigned g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul_max_sizes,
            2,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_transformer_23_feed_forward_output_dense_mult_fwd_f32_21244_0_id;
    addNodeToGraph("mult_fwd_f32",
                   {g_0_tensor_54, g_0_tensor_72__placeholder_1},
                   {g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul},
                   nullptr,
                   0,
                   "g_0_gradient_module_model_transformer_23_feed_forward_output_dense_mult_fwd_f32_21244_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_23_feed_forward_output_dense_mult_fwd_f32_21244_0_id);

    setConfigsForTest(false);
    compareRunsResults(
        {g_0_tensor_49_id_272051_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward,
         g_0_tensor_75_id_272064_gradient_module_model_transformer_23_feed_forward_output_dense_aten__mul,
         g_0_tensor_51_id_272055_gradient_module_model_transformer_23_feed_forward_output_dense_aten__linear_backward});
}

TEST_F_GC(SynTrainingUnevenPerforationTest,
          DISABLED_uneven_perforation_pt_hubert_pre_training1_ptl_8x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_module_model_transformer_dropout_cast_f32_to_bf16_159_0 node
     * inputs:
     *     g_0_tensor_18__placeholder_0[768] (dtype=float32)
     * outputs:
     *     g_0_tensor_20[768] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_18__placeholder_0 tensor
    unsigned g_0_tensor_18__placeholder_0_max_sizes[] = {768};
    unsigned g_0_tensor_18__placeholder_0_min_sizes[] = {768};
    unsigned g_0_tensor_18__placeholder_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_18__placeholder_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_18__placeholder_0_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_18__placeholder_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_20 tensor
    unsigned      g_0_tensor_20_max_sizes[] = {768};
    unsigned      g_0_tensor_20_min_sizes[] = {768};
    unsigned      g_0_tensor_20             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "g_0_tensor_20",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_20_max_sizes,
                                           1,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_20_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_transformer_dropout_cast_f32_to_bf16_159_0_id;
    unsigned char g_0_module_model_transformer_dropout_cast_f32_to_bf16_159_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_18__placeholder_0},
                   {g_0_tensor_20},
                   (void*)g_0_module_model_transformer_dropout_cast_f32_to_bf16_159_0_params,
                   4,
                   "g_0_module_model_transformer_dropout_cast_f32_to_bf16_159_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_transformer_dropout_cast_f32_to_bf16_159_0_id);

    /*************
     * g_0_module_model_transformer_0_attention_q_proj_linear_fwd_bf16_160_0 node
     * inputs:
     *     g_0_tensor_9__placeholder_0[768, 712, 24] (dtype=bf16)
     *     g_0_tensor_2[768, 768] (dtype=bf16)
     *     g_0_tensor_20[768] (dtype=bf16)
     * outputs:
     *     g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear[768, 712, 24] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_9__placeholder_0 tensor
    unsigned g_0_tensor_9__placeholder_0_max_sizes[] = {768, 712, 24};
    unsigned g_0_tensor_9__placeholder_0_min_sizes[] = {768, 712, 24};
    unsigned g_0_tensor_9__placeholder_0             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_tensor_9__placeholder_0",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_tensor_9__placeholder_0_max_sizes,
                                                         3,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_tensor_9__placeholder_0_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2 tensor
    unsigned g_0_tensor_2_max_sizes[] = {768, 768};
    unsigned g_0_tensor_2_min_sizes[] = {768, 768};
    unsigned g_0_tensor_2             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_2",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_2_max_sizes,
                                          2,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_2_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear tensor
    unsigned g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear_max_sizes[] = {768,
                                                                                                           712,
                                                                                                           24};
    unsigned g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear_min_sizes[] = {768,
                                                                                                           712,
                                                                                                           24};
    unsigned g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_model_transformer_0_attention_q_proj_linear_fwd_bf16_160_0_id;
    addNodeToGraph("linear_fwd_bf16",
                   {g_0_tensor_9__placeholder_0, g_0_tensor_2, g_0_tensor_20},
                   {g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear},
                   nullptr,
                   0,
                   "g_0_module_model_transformer_0_attention_q_proj_linear_fwd_bf16_160_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_transformer_0_attention_q_proj_linear_fwd_bf16_160_0_id);

    /*************
     * g_0_module_model_transformer_0_attention_reshape_161_0 node
     * inputs:
     *     g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear[768, 712, 24] (dtype=bf16)
     * outputs:
     *     g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view[64, 12, 712, 24] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view tensor
    unsigned g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view_max_sizes[] = {64, 12, 712, 24};
    unsigned g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view_min_sizes[] = {64, 12, 712, 24};
    unsigned g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_model_transformer_0_attention_reshape_161_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_21_id_5705_module_model_transformer_0_attention_q_proj_aten__linear},
                   {g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view},
                   nullptr,
                   0,
                   "g_0_module_model_transformer_0_attention_reshape_161_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_transformer_0_attention_reshape_161_0_id);

    /*************
     * g_0_module_model_transformer_0_attention_transpose_162_0 node
     * inputs:
     *     g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view[64, 12, 712, 24] (dtype=bf16)
     * outputs:
     *     g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose[64, 712, 12, 24] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose tensor
    unsigned g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose_max_sizes[] = {64, 712, 12, 24};
    unsigned g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose_min_sizes[] = {64, 712, 12, 24};
    unsigned g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_transformer_0_attention_transpose_162_0_id;
    unsigned char g_0_module_model_transformer_0_attention_transpose_162_0_params[] = {
        0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_22_id_5707_module_model_transformer_0_attention_aten__view},
                   {g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose},
                   (void*)g_0_module_model_transformer_0_attention_transpose_162_0_params,
                   24,
                   "g_0_module_model_transformer_0_attention_transpose_162_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_transformer_0_attention_transpose_162_0_id);

    /*************
     * g_0_module_model_transformer_0_attention_mult_fwd_bf16_164_0 node
     * inputs:
     *     g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose[64, 712, 12, 24] (dtype=bf16)
     *     g_0_tensor_26[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul[64, 712, 12, 24] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_26 tensor
    unsigned g_0_tensor_26_max_sizes[] = {1};
    unsigned g_0_tensor_26_min_sizes[] = {1};
    unsigned g_0_tensor_26             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_26",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_26_max_sizes,
                                           1,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_26_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul tensor
    unsigned g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul_max_sizes[] = {64, 712, 12, 24};
    unsigned g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul_min_sizes[] = {64, 712, 12, 24};
    unsigned g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_model_transformer_0_attention_mult_fwd_bf16_164_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_23_id_5709_module_model_transformer_0_attention_aten__transpose, g_0_tensor_26},
                   {g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul},
                   nullptr,
                   0,
                   "g_0_module_model_transformer_0_attention_mult_fwd_bf16_164_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_transformer_0_attention_mult_fwd_bf16_164_0_id);

    /*************
     * g_0_module_model_transformer_0_attention_batch_gemm_168_0 node
     * inputs:
     *     g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul[64, 712, 12, 24] (dtype=bf16)
     *     g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute[712, 64, 12, 24] (dtype=bf16)
     * outputs:
     *     g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul[712, 712, 12, 24] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute tensor
    unsigned g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute_max_sizes[] = {712, 64, 12, 24};
    unsigned g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute_min_sizes[] = {712, 64, 12, 24};
    unsigned g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul tensor
    unsigned g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul_max_sizes[] = {712, 712, 12, 24};
    unsigned g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul_min_sizes[] = {712, 712, 12, 24};
    unsigned g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_transformer_0_attention_batch_gemm_168_0_id;
    unsigned char g_0_module_model_transformer_0_attention_batch_gemm_168_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_25_id_5742_module_model_transformer_0_attention_aten__mul,
                    g_0_tensor_30_id_5723_module_model_transformer_0_attention_aten__permute},
                   {g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul},
                   (void*)g_0_module_model_transformer_0_attention_batch_gemm_168_0_params,
                   2,
                   "g_0_module_model_transformer_0_attention_batch_gemm_168_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_transformer_0_attention_batch_gemm_168_0_id);

    setConfigsForTest(false);
    compareRunsResults({g_0_tensor_31_id_5744_module_model_transformer_0_attention_aten__matmul});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, DISABLED_uneven_perforation_pt_transformer8k_8x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_gradient_module_module_decoder_5_final_layer_norm_layer_norm_bwd_bf16_953_0 node
     * inputs:
     *     g_0_tensor_3722[1024, 8120, 1, 1] (dtype=bf16)
     *     g_0_tensor_3721[1024, 8120, 1, 1] (dtype=bf16)
     *     g_0_tensor_3725[1, 8120, 1, 1] (dtype=float32)
     *     g_0_tensor_3726[1, 8120, 1, 1] (dtype=float32)
     *     g_0_tensor_3724[1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_3727[1024, 8120, 1, 1] (dtype=bf16)
     *     g_0_tensor_3728[1024] (dtype=float32)
     *     g_0_tensor_3729[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_3722 tensor
    unsigned g_0_tensor_3722_max_sizes[] = {1024, 8120, 1, 1};
    unsigned g_0_tensor_3722_min_sizes[] = {1024, 8120, 1, 1};
    unsigned g_0_tensor_3722             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3722",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3722_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3722_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3721 tensor
    unsigned g_0_tensor_3721_max_sizes[] = {1024, 8120, 1, 1};
    unsigned g_0_tensor_3721_min_sizes[] = {1024, 8120, 1, 1};
    unsigned g_0_tensor_3721             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3721",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3721_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3721_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3725 tensor
    unsigned g_0_tensor_3725_max_sizes[] = {1, 8120, 1, 1};
    unsigned g_0_tensor_3725_min_sizes[] = {1, 8120, 1, 1};
    unsigned g_0_tensor_3725             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3725",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3725_max_sizes,
                                             4,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3725_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3726 tensor
    unsigned g_0_tensor_3726_max_sizes[] = {1, 8120, 1, 1};
    unsigned g_0_tensor_3726_min_sizes[] = {1, 8120, 1, 1};
    unsigned g_0_tensor_3726             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3726",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3726_max_sizes,
                                             4,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3726_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3724 tensor
    unsigned g_0_tensor_3724_max_sizes[] = {1024};
    unsigned g_0_tensor_3724_min_sizes[] = {1024};
    unsigned g_0_tensor_3724             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3724",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3724_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3724_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3727 tensor
    unsigned g_0_tensor_3727_max_sizes[] = {1024, 8120, 1, 1};
    unsigned g_0_tensor_3727_min_sizes[] = {1024, 8120, 1, 1};
    unsigned g_0_tensor_3727             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_3727",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3727_max_sizes,
                                             4,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3727_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3728 tensor
    unsigned g_0_tensor_3728_max_sizes[] = {1024};
    unsigned g_0_tensor_3728_min_sizes[] = {1024};
    unsigned g_0_tensor_3728             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3728",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3728_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3728_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3729 tensor
    unsigned      g_0_tensor_3729_max_sizes[] = {1024};
    unsigned      g_0_tensor_3729_min_sizes[] = {1024};
    unsigned      g_0_tensor_3729             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_3729",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3729_max_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3729_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_module_decoder_5_final_layer_norm_layer_norm_bwd_bf16_953_0_id;
    unsigned char g_0_gradient_module_module_decoder_5_final_layer_norm_layer_norm_bwd_bf16_953_0_params[] =
        {0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("layer_norm_bwd_bf16",
                   {g_0_tensor_3722, g_0_tensor_3721, g_0_tensor_3725, g_0_tensor_3726, g_0_tensor_3724},
                   {g_0_tensor_3727, g_0_tensor_3728, g_0_tensor_3729},
                   (void*)g_0_gradient_module_module_decoder_5_final_layer_norm_layer_norm_bwd_bf16_953_0_params,
                   8,
                   "g_0_gradient_module_module_decoder_5_final_layer_norm_layer_norm_bwd_bf16_953_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_module_decoder_5_final_layer_norm_layer_norm_bwd_bf16_953_0_id);

    /*************
     * g_0_gradient_module_module_decoder_5_final_layer_norm_reshape_956_0 node
     * inputs:
     *     g_0_tensor_3727[1024, 8120, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward[1024,
     *232, 35] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create
    // g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward
    // tensor
    unsigned
        g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward_max_sizes
            [] = {1024, 232, 35};
    unsigned
        g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward_min_sizes
            [] = {1024, 232, 35};
    unsigned g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_module_decoder_5_final_layer_norm_reshape_956_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_tensor_3727},
        {g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward},
        nullptr,
        0,
        "g_0_gradient_module_module_decoder_5_final_layer_norm_reshape_956_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_module_decoder_5_final_layer_norm_reshape_956_0_id);

    /*************
     * g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_961_0 node
     * inputs:
     *     g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward[1024,
     *232, 35] (dtype=bf16) g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout[1024,
     *232, 35] (dtype=int8) outputs: g_0_tensor_3739[1024, 232, 35] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout tensor
    unsigned g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout_max_sizes[] = {1024,
                                                                                                                232,
                                                                                                                35};
    unsigned g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout_min_sizes[] = {1024,
                                                                                                                232,
                                                                                                                35};
    unsigned g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout_max_sizes,
                      3,
                      syn_type_int8,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3739 tensor
    unsigned  g_0_tensor_3739_max_sizes[] = {1024, 232, 35};
    unsigned  g_0_tensor_3739_min_sizes[] = {1024, 232, 35};
    unsigned  g_0_tensor_3739             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_3739",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3739_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3739_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_961_0_id;
    addNodeToGraph(
        "mult_fwd_bf16",
        {g_0_tensor_3718_id_11771_gradient_module_module_decoder_5_final_layer_norm_aten__native_layer_norm_backward,
         g_0_tensor_3646_id_11630_module_module_decoder_5_dropout_module_hpu___fused_dropout},
        {g_0_tensor_3739},
        nullptr,
        0,
        "g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_961_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_961_0_id);

    /*************
     * g_0_gradient_module_module_decoder_5_dropout_module_constant_bf16_962_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_3740[1024, 232, 35] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_3740 tensor
    unsigned      g_0_tensor_3740_max_sizes[] = {1024, 232, 35};
    unsigned      g_0_tensor_3740_min_sizes[] = {1024, 232, 35};
    unsigned      g_0_tensor_3740             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_3740",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3740_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3740_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_module_decoder_5_dropout_module_constant_bf16_962_0_id;
    unsigned char g_0_gradient_module_module_decoder_5_dropout_module_constant_bf16_962_0_params[] = {110,
                                                                                                      219,
                                                                                                      182,
                                                                                                      63};
    addNodeToGraph("constant_bf16",
                   {},
                   {g_0_tensor_3740},
                   (void*)g_0_gradient_module_module_decoder_5_dropout_module_constant_bf16_962_0_params,
                   4,
                   "g_0_gradient_module_module_decoder_5_dropout_module_constant_bf16_962_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_module_decoder_5_dropout_module_constant_bf16_962_0_id);

    /*************
     * g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_963_0 node
     * inputs:
     *     g_0_tensor_3739[1024, 232, 35] (dtype=bf16)
     *     g_0_tensor_3740[1024, 232, 35] (dtype=bf16)
     * outputs:
     *     g_0_tensor_3741[1024, 232, 35] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_3741 tensor
    unsigned  g_0_tensor_3741_max_sizes[] = {1024, 232, 35};
    unsigned  g_0_tensor_3741_min_sizes[] = {1024, 232, 35};
    unsigned  g_0_tensor_3741             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_3741",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_3741_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_3741_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_963_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_3739, g_0_tensor_3740},
                   {g_0_tensor_3741},
                   nullptr,
                   0,
                   "g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_963_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_module_decoder_5_dropout_module_mult_fwd_bf16_963_0_id);

    /*************
     * g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_964_0 node
     * inputs:
     *     g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu[4096, 232, 35] (dtype=bf16)
     *     g_0_tensor_3741[1024, 232, 35] (dtype=bf16)
     *     g_0_tensor_1251[4096, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward[4096, 232, 35]
     *(dtype=bf16) g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward[4096, 1024]
     *(dtype=bf16) g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward[1024]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu tensor
    unsigned g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu_max_sizes[] = {4096, 232, 35};
    unsigned g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu_min_sizes[] = {4096, 232, 35};
    unsigned g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1251 tensor
    unsigned g_0_tensor_1251_max_sizes[] = {4096, 1024};
    unsigned g_0_tensor_1251_min_sizes[] = {4096, 1024};
    unsigned g_0_tensor_1251             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1251",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1251_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1251_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward tensor
    unsigned g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes[] = {4096,
                                                                                                                232,
                                                                                                                35};
    unsigned g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes[] = {4096,
                                                                                                                232,
                                                                                                                35};
    unsigned g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward tensor
    unsigned g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes[] = {4096,
                                                                                                                1024};
    unsigned g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes[] = {4096,
                                                                                                                1024};
    unsigned g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward tensor
    unsigned g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes[] = {1024};
    unsigned g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes[] = {1024};
    unsigned g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_964_0_id;
    unsigned char g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_964_0_params[] = {1};
    addNodeToGraph("linear_temp_bwd_bf16",
                   {g_0_tensor_3642_id_11619_module_module_decoder_5_aten__relu, g_0_tensor_3741, g_0_tensor_1251},
                   {g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward,
                    g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward,
                    g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward},
                   (void*)g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_964_0_params,
                   1,
                   "g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_964_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_964_0_id);

    setConfigsForTest(false);
    compareRunsResults({g_0_tensor_3742_id_11786_gradient_module_module_decoder_5_fc2_aten__linear_backward,
                        g_0_tensor_3743_id_11788_gradient_module_module_decoder_5_fc2_aten__linear_backward,
                        g_0_tensor_3744_id_11790_gradient_module_module_decoder_5_fc2_aten__linear_backward});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, uneven_perforation_tf_mobilenetv2_ASIC)
{
    /*************
     * g_0_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_batch_norm_fwd_bf16_n1108_0 node
     * inputs:
     *     g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0[160, 7, 7, 96] (dtype=bf16)
     *     g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0[160] (dtype=float32)
     *     g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0[160] (dtype=float32)
     *     g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3[160] (dtype=float32)
     *     g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3[160] (dtype=float32)
     * outputs:
     *     g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0[160, 7, 7, 96] (dtype=bf16)
     *     g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1[160] (dtype=float32)
     *     g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3[160] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *     g_0_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_batch_norm_fwd_bf16_n1108_control_edge_5792[]
     *(dtype=invalid)
     *************/

    // create g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0 tensor
    unsigned g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0_max_sizes[] = {160, 7, 7, 96};
    unsigned g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0_min_sizes[] = {160, 7, 7, 96};
    unsigned g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0 tensor
    unsigned g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0_max_sizes[] = {160};
    unsigned g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0_min_sizes[] = {160};
    unsigned g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0 tensor
    unsigned g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0_max_sizes[] = {160};
    unsigned g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0_min_sizes[] = {160};
    unsigned g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3 tensor
    unsigned g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_max_sizes[] = {160};
    unsigned g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_min_sizes[] = {160};
    unsigned g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3 tensor
    unsigned g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_max_sizes[] = {160};
    unsigned g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_min_sizes[] = {160};
    unsigned g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0 tensor
    unsigned g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0_max_sizes[] = {160, 7, 7, 96};
    unsigned g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0_min_sizes[] = {160, 7, 7, 96};
    unsigned g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1 tensor
    unsigned g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1_max_sizes[] = {160};
    unsigned g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1_min_sizes[] = {160};
    unsigned g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3 tensor
    unsigned g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_max_sizes[] = {160};
    unsigned g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_min_sizes[] = {160};
    unsigned g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_batch_norm_fwd_bf16_n1108_0_id;
    unsigned char
        g_0_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_batch_norm_fwd_bf16_n1108_0_params
            [] = {149, 191, 214, 51, 0, 0, 128, 63, 111, 18, 131, 58, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_fwd_bf16",
        {g_0_t2750_MobilenetV2_expanded_conv_13_project_Conv2D_0,
         g_0_t1106_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_1_0,
         g_0_t1023_mobilenetv2_expanded_conv_13_project_batchnorm_readvariableop_0,
         g_0_t2763_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3,
         g_0_t2764_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3},
        {g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0,
         g_0_t2752_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_1,
         g_0_t2760_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3},
        (void*)
            g_0_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_batch_norm_fwd_bf16_n1108_0_params,
        16,
        "g_0_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_batch_norm_fwd_bf16_n1108_0",
        0 /*graphIndex*/,
        &g_0_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_hfbn_v3_batch_norm_fwd_bf16_n1108_0_id);

    /*************
     * g_0_MobilenetV2_expanded_conv_14_expand_Conv2D_spatial_convolution_n1115_0 node
     * inputs:
     *     g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0[160, 7, 7, 96] (dtype=bf16)
     *     g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0[960, 160, 1, 1]
     *(dtype=bf16) outputs: g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0[960, 7, 7, 96] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0 tensor
    unsigned g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0_max_sizes[] =
        {960, 160, 1, 1};
    unsigned g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0_min_sizes[] =
        {960, 160, 1, 1};
    unsigned g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0 tensor
    unsigned g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0_max_sizes[] = {960, 7, 7, 96};
    unsigned g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0_min_sizes[] = {960, 7, 7, 96};
    unsigned g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_MobilenetV2_expanded_conv_14_expand_Conv2D_spatial_convolution_n1115_0_id;
    unsigned char g_0_MobilenetV2_expanded_conv_14_expand_Conv2D_spatial_convolution_n1115_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t2751_MobilenetV2_expanded_conv_13_project_BatchNorm_FusedBatchNormV3_0,
                    g_0_t1680_MobilenetV2_expanded_conv_14_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_469_0},
                   {g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0},
                   (void*)g_0_MobilenetV2_expanded_conv_14_expand_Conv2D_spatial_convolution_n1115_0_params,
                   112,
                   "g_0_MobilenetV2_expanded_conv_14_expand_Conv2D_spatial_convolution_n1115_0",
                   0 /*graphIndex*/,
                   &g_0_MobilenetV2_expanded_conv_14_expand_Conv2D_spatial_convolution_n1115_0_id);

    setConfigsForTest(false);
    compareRunsResults({g_0_t2774_MobilenetV2_expanded_conv_14_expand_Conv2D_0});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, DISABLED_uneven_perforation_pt_transformer16k_8x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_9232_0 node
     * inputs:
     *     g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu[4096, 168, 96] (dtype=bf16)
     *     g_0_tensor_1468[1024, 168, 96] (dtype=bf16)
     *     g_0_tensor_1366__placeholder_1[4096, 1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward[4096, 168, 96]
     *(dtype=bf16) g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward[4096, 1024]
     *(dtype=bf16) g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward[1024]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu tensor
    unsigned g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu_max_sizes[] = {4096, 168, 96};
    unsigned g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu_min_sizes[] = {4096, 168, 96};
    unsigned g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1468 tensor
    unsigned g_0_tensor_1468_max_sizes[] = {1024, 168, 96};
    unsigned g_0_tensor_1468_min_sizes[] = {1024, 168, 96};
    unsigned g_0_tensor_1468             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1468",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1468_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1468_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1366__placeholder_1 tensor
    unsigned g_0_tensor_1366__placeholder_1_max_sizes[] = {4096, 1024};
    unsigned g_0_tensor_1366__placeholder_1_min_sizes[] = {4096, 1024};
    unsigned g_0_tensor_1366__placeholder_1             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_1366__placeholder_1",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_tensor_1366__placeholder_1_max_sizes,
                                                            2,
                                                            syn_type_bf16,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_1366__placeholder_1_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward tensor
    unsigned g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes[] = {4096,
                                                                                                                168,
                                                                                                                96};
    unsigned g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes[] = {4096,
                                                                                                                168,
                                                                                                                96};
    unsigned g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward tensor
    unsigned g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes[] = {4096,
                                                                                                                1024};
    unsigned g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes[] = {4096,
                                                                                                                1024};
    unsigned g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward tensor
    unsigned g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes[] = {1024};
    unsigned g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes[] = {1024};
    unsigned g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_9232_0_id;
    unsigned char g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_9232_0_params[] = {1};
    addNodeToGraph(
        "linear_temp_bwd_bf16",
        {g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu, g_0_tensor_1468, g_0_tensor_1366__placeholder_1},
        {g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward,
         g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward,
         g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward},
        (void*)g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_9232_0_params,
        1,
        "g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_9232_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_module_decoder_5_fc2_linear_temp_bwd_bf16_9232_0_id);

    /*************
     * g_0_gradient_module_module_decoder_5_activation_dropout_module_relu_bwd_bf16_9235_0 node
     * inputs:
     *     g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward[4096, 168, 96]
     *(dtype=bf16) g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu[4096, 168, 96] (dtype=bf16) outputs:
     *     g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward[4096,
     *168, 96] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create
    // g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward
    // tensor
    unsigned
        g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward_max_sizes
            [] = {4096, 168, 96};
    unsigned
        g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward_min_sizes
            [] = {4096, 168, 96};
    unsigned g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_"
            "backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_module_decoder_5_activation_dropout_module_relu_bwd_bf16_9235_0_id;
    addNodeToGraph(
        "relu_bwd_bf16",
        {g_0_tensor_1469_id_41606_gradient_module_module_decoder_5_fc2_aten__linear_backward,
         g_0_tensor_1365_id_41445_module_module_decoder_5_aten__relu},
        {g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward},
        nullptr,
        0,
        "g_0_gradient_module_module_decoder_5_activation_dropout_module_relu_bwd_bf16_9235_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_module_decoder_5_activation_dropout_module_relu_bwd_bf16_9235_0_id);

    setConfigsForTest(true);
    compareRunsResults(
        {g_0_tensor_1478_id_41620_gradient_module_module_decoder_5_activation_dropout_module_aten__threshold_backward,
         g_0_tensor_1470_id_41608_gradient_module_module_decoder_5_fc2_aten__linear_backward,
         g_0_tensor_1471_id_41610_gradient_module_module_decoder_5_fc2_aten__linear_backward});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, uneven_perforation_pt_hubert_large_pre_training_ptl_8x_lb_ASIC)
{
    GlobalConfTestSetter conf("ENABLE_GRAD_A_RESHAPED_GRAD_B_PAIRING", "false");

    /*************
     * g_0_gradient_module_model_transformer_23_attention_q_proj_linear_temp_bwd_bf16_9061_0 node
     * inputs:
     *     g_0_tensor_49__placeholder_0[1024, 304, 18] (dtype=bf16)
     *     g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view[1024, 304, 18]
     *(dtype=bf16) g_0_tensor_50__placeholder_2[1024, 1024] (dtype=bf16) outputs:
     *     g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward[1024,
     *304, 18] (dtype=bf16)
     *     g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward[1024,
     *1024] (dtype=bf16)
     *     g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward[1024]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_49__placeholder_0 tensor
    unsigned g_0_tensor_49__placeholder_0_max_sizes[] = {1024, 304, 18};
    unsigned g_0_tensor_49__placeholder_0_min_sizes[] = {1024, 304, 18};
    unsigned g_0_tensor_49__placeholder_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_49__placeholder_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_49__placeholder_0_max_sizes,
                                                          3,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_49__placeholder_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view tensor
    unsigned g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view_max_sizes[] = {
        1024,
        304,
        18};
    unsigned g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view_min_sizes[] = {
        1024,
        304,
        18};
    unsigned g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view = createTensors(
        1,
        INPUT_TENSOR,
        true,
        "g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view_max_sizes,
        3,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_50__placeholder_2 tensor
    unsigned g_0_tensor_50__placeholder_2_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_50__placeholder_2_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_50__placeholder_2             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_50__placeholder_2",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_50__placeholder_2_max_sizes,
                                                          2,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_50__placeholder_2_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward tensor
    unsigned
        g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_max_sizes
            [] = {1024, 304, 18};
    unsigned
        g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_min_sizes
            [] = {1024, 304, 18};
    unsigned g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_max_sizes,
            3,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward tensor
    unsigned
        g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_max_sizes
            [] = {1024, 1024};
    unsigned
        g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_min_sizes
            [] = {1024, 1024};
    unsigned g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward tensor
    unsigned
        g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_max_sizes
            [] = {1024};
    unsigned
        g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_min_sizes
            [] = {1024};
    unsigned g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_max_sizes,
            1,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_model_transformer_23_attention_q_proj_linear_temp_bwd_bf16_9061_0_id;
    unsigned char g_0_gradient_module_model_transformer_23_attention_q_proj_linear_temp_bwd_bf16_9061_0_params[] = {1};
    addNodeToGraph(
        "linear_temp_bwd_bf16",
        {g_0_tensor_49__placeholder_0,
         g_0_tensor_48_id_169752_gradient_module_model_transformer_23_attention_q_proj_aten__view,
         g_0_tensor_50__placeholder_2},
        {g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward,
         g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward,
         g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward},
        (void*)g_0_gradient_module_model_transformer_23_attention_q_proj_linear_temp_bwd_bf16_9061_0_params,
        1,
        "g_0_gradient_module_model_transformer_23_attention_q_proj_linear_temp_bwd_bf16_9061_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_transformer_23_attention_q_proj_linear_temp_bwd_bf16_9061_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_attention_q_proj_cast_bf16_to_f32_9088_0 node
     * inputs:
     *     g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward[1024,
     *1024] (dtype=bf16) outputs: g_0_tensor_102[1024, 1024] (dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_102 tensor
    unsigned      g_0_tensor_102_max_sizes[] = {1024, 1024};
    unsigned      g_0_tensor_102_min_sizes[] = {1024, 1024};
    unsigned      g_0_tensor_102             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_102",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_102_max_sizes,
                                            2,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_102_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_model_transformer_23_attention_q_proj_cast_bf16_to_f32_9088_0_id;
    unsigned char g_0_gradient_module_model_transformer_23_attention_q_proj_cast_bf16_to_f32_9088_0_params[] = {0,
                                                                                                                0,
                                                                                                                0,
                                                                                                                0};
    addNodeToGraph(
        "cast_bf16_to_f32",
        {g_0_tensor_52_id_169762_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward},
        {g_0_tensor_102},
        (void*)g_0_gradient_module_model_transformer_23_attention_q_proj_cast_bf16_to_f32_9088_0_params,
        4,
        "g_0_gradient_module_model_transformer_23_attention_q_proj_cast_bf16_to_f32_9088_0",
        0 /*graphIndex*/,
        &g_0_gradient_module_model_transformer_23_attention_q_proj_cast_bf16_to_f32_9088_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_attention_q_proj_add_fwd_f32_9162_0 node
     * inputs:
     *     g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1[1024,
     *1024] (dtype=float32) g_0_tensor_102[1024, 1024] (dtype=float32) outputs:
     *     g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add_[1024, 1024]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1 tensor
    unsigned
        g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1_max_sizes[] =
            {1024, 1024};
    unsigned
        g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1_min_sizes[] =
            {1024, 1024};
    unsigned g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1_max_sizes,
            2,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add_ tensor
    unsigned g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add__max_sizes[] = {
        1024,
        1024};
    unsigned g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add__min_sizes[] = {
        1024,
        1024};
    unsigned g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add_ = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add_",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add__max_sizes,
        2,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add__min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_transformer_23_attention_q_proj_add_fwd_f32_9162_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_tensor_187_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_1,
                    g_0_tensor_102},
                   {g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add_},
                   nullptr,
                   0,
                   "g_0_gradient_module_model_transformer_23_attention_q_proj_add_fwd_f32_9162_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_23_attention_q_proj_add_fwd_f32_9162_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_attention_q_proj_strided_insert_9163_0 node
     * inputs:
     *     g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert[8400896]
     *(dtype=float32) g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add_[1024,
     *1024] (dtype=float32) outputs:
     *     g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert[8400896]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert tensor
    unsigned
        g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert_max_sizes[] =
            {8400896};
    unsigned
        g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert_min_sizes[] =
            {8400896};
    unsigned g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert tensor
    unsigned
        g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert_max_sizes[] =
            {8400896};
    unsigned
        g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert_min_sizes[] =
            {8400896};
    unsigned g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_model_transformer_23_attention_q_proj_strided_insert_9163_0_id;
    unsigned char g_0_gradient_module_model_transformer_23_attention_q_proj_strided_insert_9163_0_params[] = {
        0, 36, 48, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("strided_insert",
                   {g_0_tensor_186_id_169759_gradient_module_model_transformer_23_attention_k_proj_hpu__strided_insert,
                    g_0_tensor_188_id_169774_gradient_module_model_transformer_23_attention_q_proj_hpu__add_},
                   {g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert},
                   (void*)g_0_gradient_module_model_transformer_23_attention_q_proj_strided_insert_9163_0_params,
                   208,
                   "g_0_gradient_module_model_transformer_23_attention_q_proj_strided_insert_9163_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_23_attention_q_proj_strided_insert_9163_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_attention_q_proj_strided_view_9164_0 node
     * inputs:
     *     g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert[8400896]
     *(dtype=float32) outputs:
     *     g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view[1024, 1024]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view tensor
    unsigned
        g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_max_sizes[] = {
            1024,
            1024};
    unsigned
        g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_min_sizes[] = {
            1024,
            1024};
    unsigned g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_max_sizes,
            2,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_model_transformer_23_attention_q_proj_strided_view_9164_0_id;
    unsigned char g_0_gradient_module_model_transformer_23_attention_q_proj_strided_view_9164_0_params[] = {
        0, 36, 48, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("strided_view",
                   {g_0_tensor_189_id_169780_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_insert},
                   {g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view},
                   (void*)g_0_gradient_module_model_transformer_23_attention_q_proj_strided_view_9164_0_params,
                   208,
                   "g_0_gradient_module_model_transformer_23_attention_q_proj_strided_view_9164_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_23_attention_q_proj_strided_view_9164_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_attention_q_proj_cast_i32_to_f32_9165_0 node
     * inputs:
     *     g_0_tensor_127__placeholder_1[1] (dtype=int32)
     * outputs:
     *     g_0_tensor_192[1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_127__placeholder_1 tensor
    unsigned g_0_tensor_127__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_127__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_127__placeholder_1             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_tensor_127__placeholder_1",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_tensor_127__placeholder_1_max_sizes,
                                                           1,
                                                           syn_type_int32,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_tensor_127__placeholder_1_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_192 tensor
    unsigned      g_0_tensor_192_max_sizes[] = {1};
    unsigned      g_0_tensor_192_min_sizes[] = {1};
    unsigned      g_0_tensor_192             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_192",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_192_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_192_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_module_model_transformer_23_attention_q_proj_cast_i32_to_f32_9165_0_id;
    unsigned char g_0_gradient_module_model_transformer_23_attention_q_proj_cast_i32_to_f32_9165_0_params[] = {0,
                                                                                                               0,
                                                                                                               0,
                                                                                                               0};
    addNodeToGraph("cast_i32_to_f32",
                   {g_0_tensor_127__placeholder_1},
                   {g_0_tensor_192},
                   (void*)g_0_gradient_module_model_transformer_23_attention_q_proj_cast_i32_to_f32_9165_0_params,
                   4,
                   "g_0_gradient_module_model_transformer_23_attention_q_proj_cast_i32_to_f32_9165_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_23_attention_q_proj_cast_i32_to_f32_9165_0_id);

    /*************
     * g_0_gradient_module_model_transformer_23_attention_q_proj_div_f32_9166_0 node
     * inputs:
     *     g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view[1024, 1024]
     *(dtype=float32) g_0_tensor_192[1] (dtype=float32) outputs:
     *     g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div_[1024, 1024]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div_ tensor
    unsigned g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div__max_sizes[] = {
        1024,
        1024};
    unsigned g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div__min_sizes[] = {
        1024,
        1024};
    unsigned g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div_ = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div_",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div__max_sizes,
        2,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div__min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_module_model_transformer_23_attention_q_proj_div_f32_9166_0_id;
    addNodeToGraph("div_f32",
                   {g_0_tensor_190_id_154521_gradient_module_model_transformer_23_attention_q_proj_hpu__strided_view,
                    g_0_tensor_192},
                   {g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div_},
                   nullptr,
                   0,
                   "g_0_gradient_module_model_transformer_23_attention_q_proj_div_f32_9166_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_module_model_transformer_23_attention_q_proj_div_f32_9166_0_id);

    setConfigsForTest(true);
    compareRunsResults(
        {g_0_tensor_51_id_169758_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward,
         g_0_tensor_191_id_169782_gradient_module_model_transformer_23_attention_q_proj_aten__div_,
         g_0_tensor_53_id_169764_gradient_module_model_transformer_23_attention_q_proj_aten__linear_backward});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, uneven_perforation_syn_resnet50_bf16_ASIC)
{
    GlobalConfTestSetter conf1("ENABLE_EXPERIMENTAL_PATTERNS_FUSION", "1");
    GlobalConfTestSetter conf2("EW_RADIUS", "4");

    /*************
     * g_0_layer3_4_bn2_0 node
     * inputs:
     *     g_0_layer3_4_conv2_output[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_4_bn2_bias[256] (dtype=float32)
     *     g_0_layer3_4_bn2_weight[256] (dtype=float32)
     *     g_0_layer3_4_bn2_running_mean[256] (dtype=float32)
     *     g_0_layer3_4_bn2_running_var[256] (dtype=float32)
     * outputs:
     *     g_0_layer3_4_bn2_output[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_4_bn2_saved_mean[256, 1, 1, 1] (dtype=float32)
     *     g_0_layer3_4_bn2_saved_var[256, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_4_conv2_output tensor
    unsigned g_0_layer3_4_conv2_output_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_4_conv2_output_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_4_conv2_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_4_conv2_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_4_conv2_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_4_conv2_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn2_bias tensor
    unsigned g_0_layer3_4_bn2_bias_max_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_bias_min_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_bias             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_layer3_4_bn2_bias",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   g_0_layer3_4_bn2_bias_max_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_layer3_4_bn2_bias_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn2_weight tensor
    unsigned g_0_layer3_4_bn2_weight_max_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_weight_min_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer3_4_bn2_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer3_4_bn2_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer3_4_bn2_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn2_running_mean tensor
    unsigned g_0_layer3_4_bn2_running_mean_max_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_running_mean_min_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_running_mean             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer3_4_bn2_running_mean",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer3_4_bn2_running_mean_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer3_4_bn2_running_mean_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn2_running_var tensor
    unsigned g_0_layer3_4_bn2_running_var_max_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_running_var_min_sizes[] = {256};
    unsigned g_0_layer3_4_bn2_running_var             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_layer3_4_bn2_running_var",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer3_4_bn2_running_var_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer3_4_bn2_running_var_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn2_output tensor
    unsigned g_0_layer3_4_bn2_output_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_4_bn2_output_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_4_bn2_output             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     false,
                                                     "g_0_layer3_4_bn2_output",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer3_4_bn2_output_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer3_4_bn2_output_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn2_saved_mean tensor
    unsigned g_0_layer3_4_bn2_saved_mean_max_sizes[] = {256, 1, 1, 1};
    unsigned g_0_layer3_4_bn2_saved_mean_min_sizes[] = {256, 1, 1, 1};
    unsigned g_0_layer3_4_bn2_saved_mean             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer3_4_bn2_saved_mean",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_4_bn2_saved_mean_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_4_bn2_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn2_saved_var tensor
    unsigned      g_0_layer3_4_bn2_saved_var_max_sizes[] = {256, 1, 1, 1};
    unsigned      g_0_layer3_4_bn2_saved_var_min_sizes[] = {256, 1, 1, 1};
    unsigned      g_0_layer3_4_bn2_saved_var             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer3_4_bn2_saved_var",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer3_4_bn2_saved_var_max_sizes,
                                                        4,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer3_4_bn2_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer3_4_bn2_0_id;
    unsigned char g_0_layer3_4_bn2_0_params[] = {172, 197, 39, 55, 205, 204, 204, 61, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_layer3_4_conv2_output,
                    g_0_layer3_4_bn2_bias,
                    g_0_layer3_4_bn2_weight,
                    g_0_layer3_4_bn2_running_mean,
                    g_0_layer3_4_bn2_running_var},
                   {g_0_layer3_4_bn2_output, g_0_layer3_4_bn2_saved_mean, g_0_layer3_4_bn2_saved_var},
                   (void*)g_0_layer3_4_bn2_0_params,
                   12,
                   "g_0_layer3_4_bn2_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_4_bn2_0_id);

    /*************
     * g_0_layer3_4_relu2_0 node
     * inputs:
     *     g_0_layer3_4_bn2_output[256, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_4_relu2_output[256, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_4_relu2_output tensor
    unsigned  g_0_layer3_4_relu2_output_max_sizes[] = {256, 14, 14, 64};
    unsigned  g_0_layer3_4_relu2_output_min_sizes[] = {256, 14, 14, 64};
    unsigned  g_0_layer3_4_relu2_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_layer3_4_relu2_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_4_relu2_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_4_relu2_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_4_relu2_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_layer3_4_bn2_output},
                   {g_0_layer3_4_relu2_output},
                   nullptr,
                   0,
                   "g_0_layer3_4_relu2_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_4_relu2_0_id);

    /*************
     * g_0_layer3_4_conv3_0 node
     * inputs:
     *     g_0_layer3_4_relu2_output[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_4_conv3_weight[1024, 256, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_layer3_4_conv3_output[1024, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_4_conv3_weight tensor
    unsigned g_0_layer3_4_conv3_weight_max_sizes[] = {1024, 256, 1, 1};
    unsigned g_0_layer3_4_conv3_weight_min_sizes[] = {1024, 256, 1, 1};
    unsigned g_0_layer3_4_conv3_weight             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_4_conv3_weight",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_4_conv3_weight_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_4_conv3_weight_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_conv3_output tensor
    unsigned      g_0_layer3_4_conv3_output_max_sizes[] = {1024, 14, 14, 64};
    unsigned      g_0_layer3_4_conv3_output_min_sizes[] = {1024, 14, 14, 64};
    unsigned      g_0_layer3_4_conv3_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_layer3_4_conv3_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_4_conv3_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_4_conv3_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer3_4_conv3_0_id;
    unsigned char g_0_layer3_4_conv3_0_params[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_layer3_4_relu2_output, g_0_layer3_4_conv3_weight},
                   {g_0_layer3_4_conv3_output},
                   (void*)g_0_layer3_4_conv3_0_params,
                   112,
                   "g_0_layer3_4_conv3_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_4_conv3_0_id);

    /*************
     * g_0_layer3_4_bn3_0 node
     * inputs:
     *     g_0_layer3_4_conv3_output[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_4_bn3_bias[1024] (dtype=float32)
     *     g_0_layer3_4_bn3_weight[1024] (dtype=float32)
     *     g_0_layer3_4_bn3_running_mean[1024] (dtype=float32)
     *     g_0_layer3_4_bn3_running_var[1024] (dtype=float32)
     * outputs:
     *     g_0_layer3_4_bn3_output[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_4_bn3_saved_mean[1024, 1, 1, 1] (dtype=float32)
     *     g_0_layer3_4_bn3_saved_var[1024, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_4_bn3_bias tensor
    unsigned g_0_layer3_4_bn3_bias_max_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_bias_min_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_bias             = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_layer3_4_bn3_bias",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   g_0_layer3_4_bn3_bias_max_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_layer3_4_bn3_bias_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn3_weight tensor
    unsigned g_0_layer3_4_bn3_weight_max_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_weight_min_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer3_4_bn3_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer3_4_bn3_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer3_4_bn3_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn3_running_mean tensor
    unsigned g_0_layer3_4_bn3_running_mean_max_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_running_mean_min_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_running_mean             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer3_4_bn3_running_mean",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer3_4_bn3_running_mean_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer3_4_bn3_running_mean_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn3_running_var tensor
    unsigned g_0_layer3_4_bn3_running_var_max_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_running_var_min_sizes[] = {1024};
    unsigned g_0_layer3_4_bn3_running_var             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_layer3_4_bn3_running_var",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer3_4_bn3_running_var_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer3_4_bn3_running_var_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn3_output tensor
    unsigned g_0_layer3_4_bn3_output_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_4_bn3_output_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_4_bn3_output             = createTensors(1,
                                                     OUTPUT_TENSOR,
                                                     false,
                                                     "g_0_layer3_4_bn3_output",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer3_4_bn3_output_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer3_4_bn3_output_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn3_saved_mean tensor
    unsigned g_0_layer3_4_bn3_saved_mean_max_sizes[] = {1024, 1, 1, 1};
    unsigned g_0_layer3_4_bn3_saved_mean_min_sizes[] = {1024, 1, 1, 1};
    unsigned g_0_layer3_4_bn3_saved_mean             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer3_4_bn3_saved_mean",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_4_bn3_saved_mean_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_4_bn3_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_bn3_saved_var tensor
    unsigned      g_0_layer3_4_bn3_saved_var_max_sizes[] = {1024, 1, 1, 1};
    unsigned      g_0_layer3_4_bn3_saved_var_min_sizes[] = {1024, 1, 1, 1};
    unsigned      g_0_layer3_4_bn3_saved_var             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer3_4_bn3_saved_var",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer3_4_bn3_saved_var_max_sizes,
                                                        4,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer3_4_bn3_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer3_4_bn3_0_id;
    unsigned char g_0_layer3_4_bn3_0_params[] = {172, 197, 39, 55, 205, 204, 204, 61, 172, 197, 39, 55};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_layer3_4_conv3_output,
                    g_0_layer3_4_bn3_bias,
                    g_0_layer3_4_bn3_weight,
                    g_0_layer3_4_bn3_running_mean,
                    g_0_layer3_4_bn3_running_var},
                   {g_0_layer3_4_bn3_output, g_0_layer3_4_bn3_saved_mean, g_0_layer3_4_bn3_saved_var},
                   (void*)g_0_layer3_4_bn3_0_params,
                   12,
                   "g_0_layer3_4_bn3_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_4_bn3_0_id);

    /*************
     * g_0_layer3_4_add_residual_fwd0_0 node
     * inputs:
     *     g_0_layer3_4_bn3_output[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_3_relu3_output[1024, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_4_add_residual_fwd[1024, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_3_relu3_output tensor
    unsigned g_0_layer3_3_relu3_output_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_3_relu3_output_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_3_relu3_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_3_relu3_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_3_relu3_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_3_relu3_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_4_add_residual_fwd tensor
    unsigned  g_0_layer3_4_add_residual_fwd_max_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_4_add_residual_fwd_min_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_4_add_residual_fwd             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_layer3_4_add_residual_fwd",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer3_4_add_residual_fwd_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer3_4_add_residual_fwd_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_4_add_residual_fwd0_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_layer3_4_bn3_output, g_0_layer3_3_relu3_output},
                   {g_0_layer3_4_add_residual_fwd},
                   nullptr,
                   0,
                   "g_0_layer3_4_add_residual_fwd0_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_4_add_residual_fwd0_0_id);

    /*************
     * g_0_layer3_4_relu3_0 node
     * inputs:
     *     g_0_layer3_4_add_residual_fwd[1024, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_4_relu3_output[1024, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_4_relu3_output tensor
    unsigned  g_0_layer3_4_relu3_output_max_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_4_relu3_output_min_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_4_relu3_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_4_relu3_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_4_relu3_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_4_relu3_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_4_relu3_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_layer3_4_add_residual_fwd},
                   {g_0_layer3_4_relu3_output},
                   nullptr,
                   0,
                   "g_0_layer3_4_relu3_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_4_relu3_0_id);

    setConfigsForTest(true);
    compareRunsResults({g_0_layer3_4_relu3_output});
}

TEST_F_GC(SynTrainingUnevenPerforationTest, uneven_perforation_pt_wav2vec_ac_hf_8x_ASIC)
{
    /*************
     * g_0_module_wav2vec2_feature_extractor_1_conv_spatial_convolution_39_0 node
     * inputs:
     *     g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze[3199, 1, 512, 16] (dtype=bf16)
     *     g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze[3, 1, 512, 512] (dtype=bf16)
     * outputs:
     *     g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable[1599, 1, 512,
     *16] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze tensor
    unsigned g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_max_sizes[] = {3199,
                                                                                                          1,
                                                                                                          512,
                                                                                                          16};
    unsigned g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_min_sizes[] = {3199,
                                                                                                          1,
                                                                                                          512,
                                                                                                          16};
    unsigned g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze tensor
    unsigned g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_max_sizes[] = {3,
                                                                                                          1,
                                                                                                          512,
                                                                                                          512};
    unsigned g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_min_sizes[] = {3,
                                                                                                          1,
                                                                                                          512,
                                                                                                          512};
    unsigned g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable tensor
    unsigned g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable_max_sizes[] =
        {1599, 1, 512, 16};
    unsigned g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable_min_sizes[] =
        {1599, 1, 512, 16};
    unsigned g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_wav2vec2_feature_extractor_1_conv_spatial_convolution_39_0_id;
    unsigned char g_0_module_wav2vec2_feature_extractor_1_conv_spatial_convolution_39_0_params[] = {
        3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const char* convInputLayouts[]  = {"WHCN", "SRCK"};
    const char* convOutputLayouts[] = {"WHCN"};
    addNodeToGraph("spatial_convolution",
                   {g_0_tensor_3_id_4981_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze,
                    g_0_tensor_1_id_4993_module_wav2vec2_feature_extractor_1_conv_aten__unsqueeze},
                   {g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable},
                   (void*)g_0_module_wav2vec2_feature_extractor_1_conv_spatial_convolution_39_0_params,
                   112,
                   "g_0_module_wav2vec2_feature_extractor_1_conv_spatial_convolution_39_0",
                   0 /*graphIndex*/,
                   &g_0_module_wav2vec2_feature_extractor_1_conv_spatial_convolution_39_0_id,
                   convInputLayouts,
                   convOutputLayouts);

    setConfigsForTest(true);
    compareRunsResults({g_0_tensor_4_id_4994_module_wav2vec2_feature_extractor_1_conv_aten__convolution_overrideable});
}

TEST_F_GC(SynTrainingUnevenPerforationTest,
          DISABLED_uneven_perforation_transformer_zoom_en_de_28k_8x_ASIC)  // TODO: SW-169868
{
    /*************
     * g_0_module_module_encoder_0_self_attn_v_proj_linear_fwd_bf16_5758_0 node
     * inputs:
     *     g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose[1024, 568, 50] (dtype=bf16)
     *     g_0_tensor_47__placeholder_1[1024, 1024] (dtype=bf16)
     *     g_0_tensor_48__placeholder_2[1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear[1024, 568, 50] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose tensor
    unsigned g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose_max_sizes[] = {1024,
                                                                                                            568,
                                                                                                            50};
    unsigned g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose_min_sizes[] = {1024,
                                                                                                            568,
                                                                                                            50};
    unsigned g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_47__placeholder_1 tensor
    unsigned g_0_tensor_47__placeholder_1_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_47__placeholder_1_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_47__placeholder_1             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_47__placeholder_1",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_47__placeholder_1_max_sizes,
                                                          2,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_47__placeholder_1_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_48__placeholder_2 tensor
    unsigned g_0_tensor_48__placeholder_2_max_sizes[] = {1024};
    unsigned g_0_tensor_48__placeholder_2_min_sizes[] = {1024};
    unsigned g_0_tensor_48__placeholder_2             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_48__placeholder_2",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_48__placeholder_2_max_sizes,
                                                          1,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_48__placeholder_2_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear tensor
    unsigned g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear_max_sizes[] = {1024, 568, 50};
    unsigned g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear_min_sizes[] = {1024, 568, 50};
    unsigned g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_module_encoder_0_self_attn_v_proj_linear_fwd_bf16_5758_0_id;
    addNodeToGraph("linear_fwd_bf16",
                   {g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose,
                    g_0_tensor_47__placeholder_1,
                    g_0_tensor_48__placeholder_2},
                   {g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear},
                   nullptr,
                   0,
                   "g_0_module_module_encoder_0_self_attn_v_proj_linear_fwd_bf16_5758_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_0_self_attn_v_proj_linear_fwd_bf16_5758_0_id);

    /*************
     * g_0_module_module_encoder_0_self_attn_k_proj_linear_fwd_bf16_5764_0 node
     * inputs:
     *     g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose[1024, 568, 50] (dtype=bf16)
     *     g_0_tensor_55__placeholder_1[1024, 1024] (dtype=bf16)
     *     g_0_tensor_56__placeholder_2[1024] (dtype=bf16)
     * outputs:
     *     g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear[1024, 568, 50] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_55__placeholder_1 tensor
    unsigned g_0_tensor_55__placeholder_1_max_sizes[] = {1024, 1024};
    unsigned g_0_tensor_55__placeholder_1_min_sizes[] = {1024, 1024};
    unsigned g_0_tensor_55__placeholder_1             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_55__placeholder_1",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_55__placeholder_1_max_sizes,
                                                          2,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_55__placeholder_1_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_56__placeholder_2 tensor
    unsigned g_0_tensor_56__placeholder_2_max_sizes[] = {1024};
    unsigned g_0_tensor_56__placeholder_2_min_sizes[] = {1024};
    unsigned g_0_tensor_56__placeholder_2             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_tensor_56__placeholder_2",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_tensor_56__placeholder_2_max_sizes,
                                                          1,
                                                          syn_type_bf16,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_tensor_56__placeholder_2_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear tensor
    unsigned g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear_max_sizes[] = {1024, 568, 50};
    unsigned g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear_min_sizes[] = {1024, 568, 50};
    unsigned g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_module_encoder_0_self_attn_k_proj_linear_fwd_bf16_5764_0_id;
    addNodeToGraph("linear_fwd_bf16",
                   {g_0_tensor_46_id_34420_module_module_encoder_0_self_attn_v_proj_aten__transpose,
                    g_0_tensor_55__placeholder_1,
                    g_0_tensor_56__placeholder_2},
                   {g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear},
                   nullptr,
                   0,
                   "g_0_module_module_encoder_0_self_attn_k_proj_linear_fwd_bf16_5764_0",
                   0 /*graphIndex*/,
                   &g_0_module_module_encoder_0_self_attn_k_proj_linear_fwd_bf16_5764_0_id);

    setConfigsForTest(true);
    compareRunsResults({g_0_tensor_49_id_34426_module_module_encoder_0_self_attn_v_proj_aten__linear,
                        g_0_tensor_57_id_34424_module_module_encoder_0_self_attn_k_proj_aten__linear});
}