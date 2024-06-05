#include "log_manager.h"
#include "scoped_configuration_change.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"

class SynTrainingLayeredBrainAccuracyTest : public SynGaudiTwoRunCompareTest
{
protected:
    void setConfigsForTest()
    {
        addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "true");
        addConfigurationToRun(FIRST_RUN, "ENABLE_BUNDLE_MEMORY_MANAGEMENT", "true");

        // The reference is unsliced
        addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");
        addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
        addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");
    }
};

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, ms0_fused_softmax_dropout_bgemm_ASIC)
{
    // Graph #0

    /*************
     * bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0 node
     * inputs:
     *     t1360_bert_encoder_layer_1_attention_self_value_MatMul_0[1024, 14336] (dtype=bf16)
     *     t1362_bert_encoder_layer_1_attention_self_value_BiasAdd[1024, 1] (dtype=bf16)
     * outputs:
     *     t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1360_bert_encoder_layer_1_attention_self_value_MatMul_0 tensor
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_max_sizes[] = {1024, 14336};
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_min_sizes[] = {1024, 14336};
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "t1360_bert_encoder_layer_1_attention_self_value_MatMul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t1362_bert_encoder_layer_1_attention_self_value_BiasAdd tensor
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes[] = {1024, 1};
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_min_sizes[] = {1024, 1};
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "t1362_bert_encoder_layer_1_attention_self_value_BiasAdd",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0 tensor
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_max_sizes[] = {1024, 14336};
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_min_sizes[] = {1024, 14336};
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0",
                      MEM_INIT_NONE,
                      nullptr,
                      t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {t1360_bert_encoder_layer_1_attention_self_value_MatMul_0,
                    t1362_bert_encoder_layer_1_attention_self_value_BiasAdd},
                   {t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0},
                   nullptr,
                   0,
                   "bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0 node
     * inputs:
     *     t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0[1024, 14336] (dtype=bf16)
     *     t1365_bert_encoder_layer_1_attention_self_Reshape_2[64, 16, 512, 28] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1364_bert_encoder_layer_1_attention_self_Reshape_2_0[64, 16, 512, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1365_bert_encoder_layer_1_attention_self_Reshape_2 tensor
    unsigned t1365_bert_encoder_layer_1_attention_self_Reshape_2_max_sizes[] = {64, 16, 512, 28};
    unsigned t1365_bert_encoder_layer_1_attention_self_Reshape_2_min_sizes[] = {64, 16, 512, 28};
    unsigned t1365_bert_encoder_layer_1_attention_self_Reshape_2 =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "t1365_bert_encoder_layer_1_attention_self_Reshape_2",
                      MEM_INIT_NONE,
                      nullptr,
                      t1365_bert_encoder_layer_1_attention_self_Reshape_2_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1365_bert_encoder_layer_1_attention_self_Reshape_2_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create t1364_bert_encoder_layer_1_attention_self_Reshape_2_0 tensor
    unsigned t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_max_sizes[] = {64, 16, 512, 28};
    unsigned t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_min_sizes[] = {64, 16, 512, 28};
    unsigned t1364_bert_encoder_layer_1_attention_self_Reshape_2_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t1364_bert_encoder_layer_1_attention_self_Reshape_2_0",
                      MEM_INIT_NONE,
                      nullptr,
                      t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0_id;
    addNodeToGraph("reshape",
                   {t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0,
                    t1365_bert_encoder_layer_1_attention_self_Reshape_2},
                   {t1364_bert_encoder_layer_1_attention_self_Reshape_2_0},
                   nullptr,
                   0,
                   "bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0 node
     * inputs:
     *     t1364_bert_encoder_layer_1_attention_self_Reshape_2_0[64, 16, 512, 28] (dtype=bf16)
     * outputs:
     *     t1366_bert_encoder_layer_1_attention_self_transpose_2_0[64, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1366_bert_encoder_layer_1_attention_self_transpose_2_0 tensor
    unsigned t1366_bert_encoder_layer_1_attention_self_transpose_2_0_max_sizes[] = {64, 512, 16, 28};
    unsigned t1366_bert_encoder_layer_1_attention_self_transpose_2_0_min_sizes[] = {64, 512, 16, 28};
    unsigned t1366_bert_encoder_layer_1_attention_self_transpose_2_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t1366_bert_encoder_layer_1_attention_self_transpose_2_0",
                      MEM_INIT_NONE,
                      nullptr,
                      t1366_bert_encoder_layer_1_attention_self_transpose_2_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1366_bert_encoder_layer_1_attention_self_transpose_2_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_id;
    unsigned char bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_params[] = {
        0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0};
    addNodeToGraph("transpose",
                   {t1364_bert_encoder_layer_1_attention_self_Reshape_2_0},
                   {t1366_bert_encoder_layer_1_attention_self_transpose_2_0},
                   (void*)bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_params,
                   24,
                   "bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0 node
     * inputs:
     *     t1359_bert_encoder_layer_1_attention_self_MatMul_0[512, 512, 16, 28] (dtype=bf16)
     *     t1368_bert_encoder_layer_1_attention_self_Mul[1, 1, 1, 1] (dtype=bf16)
     * outputs:
     *     t1367_bert_encoder_layer_1_attention_self_Mul_0[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1359_bert_encoder_layer_1_attention_self_MatMul_0 tensor
    unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes[] = {512, 512, 16, 28};
    unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0_min_sizes[] = {512, 512, 16, 28};
    unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "t1359_bert_encoder_layer_1_attention_self_MatMul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1359_bert_encoder_layer_1_attention_self_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t1368_bert_encoder_layer_1_attention_self_Mul tensor
    unsigned t1368_bert_encoder_layer_1_attention_self_Mul_max_sizes[] = {1, 1, 1, 1};
    unsigned t1368_bert_encoder_layer_1_attention_self_Mul_min_sizes[] = {1, 1, 1, 1};
    unsigned t1368_bert_encoder_layer_1_attention_self_Mul =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "t1368_bert_encoder_layer_1_attention_self_Mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      t1368_bert_encoder_layer_1_attention_self_Mul_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1368_bert_encoder_layer_1_attention_self_Mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t1367_bert_encoder_layer_1_attention_self_Mul_0 tensor
    unsigned t1367_bert_encoder_layer_1_attention_self_Mul_0_max_sizes[] = {512, 512, 16, 28};
    unsigned t1367_bert_encoder_layer_1_attention_self_Mul_0_min_sizes[] = {512, 512, 16, 28};
    unsigned t1367_bert_encoder_layer_1_attention_self_Mul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t1367_bert_encoder_layer_1_attention_self_Mul_0",
                      MEM_INIT_NONE,
                      nullptr,
                      t1367_bert_encoder_layer_1_attention_self_Mul_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1367_bert_encoder_layer_1_attention_self_Mul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {t1359_bert_encoder_layer_1_attention_self_MatMul_0, t1368_bert_encoder_layer_1_attention_self_Mul},
                   {t1367_bert_encoder_layer_1_attention_self_Mul_0},
                   nullptr,
                   0,
                   "bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0 node
     * inputs:
     *     t1367_bert_encoder_layer_1_attention_self_Mul_0[512, 512, 16, 28] (dtype=bf16)
     *     t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0[512, 512, 1, 28] (dtype=bf16)
     * outputs:
     *     t1370_bert_encoder_layer_1_attention_self_add_0[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0 tensor
    unsigned t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_max_sizes[] = {512, 512, 1, 28};
    unsigned t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_min_sizes[] = {512, 512, 1, 28};
    unsigned t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t1370_bert_encoder_layer_1_attention_self_add_0 tensor
    unsigned t1370_bert_encoder_layer_1_attention_self_add_0_max_sizes[] = {512, 512, 16, 28};
    unsigned t1370_bert_encoder_layer_1_attention_self_add_0_min_sizes[] = {512, 512, 16, 28};
    unsigned t1370_bert_encoder_layer_1_attention_self_add_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t1370_bert_encoder_layer_1_attention_self_add_0",
                      MEM_INIT_NONE,
                      nullptr,
                      t1370_bert_encoder_layer_1_attention_self_add_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1370_bert_encoder_layer_1_attention_self_add_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {t1367_bert_encoder_layer_1_attention_self_Mul_0,
                    t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0},
                   {t1370_bert_encoder_layer_1_attention_self_add_0},
                   nullptr,
                   0,
                   "bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0 node
     * inputs:
     *     t1370_bert_encoder_layer_1_attention_self_add_0[512, 512, 16, 28] (dtype=bf16)
     * outputs:
     *     t1371_bert_encoder_layer_1_attention_self_Softmax_0[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1371_bert_encoder_layer_1_attention_self_Softmax_0 tensor
    unsigned t1371_bert_encoder_layer_1_attention_self_Softmax_0_max_sizes[] = {512, 512, 16, 28};
    unsigned t1371_bert_encoder_layer_1_attention_self_Softmax_0_min_sizes[] = {512, 512, 16, 28};
    unsigned t1371_bert_encoder_layer_1_attention_self_Softmax_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t1371_bert_encoder_layer_1_attention_self_Softmax_0",
                      MEM_INIT_NONE,
                      nullptr,
                      t1371_bert_encoder_layer_1_attention_self_Softmax_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1371_bert_encoder_layer_1_attention_self_Softmax_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_id;
    unsigned char bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("softmax_fwd_bf16",
                   {t1370_bert_encoder_layer_1_attention_self_add_0},
                   {t1371_bert_encoder_layer_1_attention_self_Softmax_0},
                   (void*)bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_params,
                   4,
                   "bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0 node
     * inputs:
     *     t1371_bert_encoder_layer_1_attention_self_Softmax_0[512, 512, 16, 28] (dtype=bf16)
     *     t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0[1]
     *(dtype=int32) outputs: t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28] (dtype=bf16)
     *     t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0
    // tensor
    unsigned
        t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_max_sizes
            [] = {1};
    unsigned
        t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_min_sizes
            [] = {1};
    unsigned t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_max_sizes,
            1,
            syn_type_int32,
            nullptr,
            0,
            0,
            nullptr,
            false,
            t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 tensor
    unsigned t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes[] = {512, 512, 16, 28};
    unsigned t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_min_sizes[] = {512, 512, 16, 28};
    unsigned t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 tensor
    unsigned t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes[] = {512, 512, 16, 28};
    unsigned t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_min_sizes[] = {512, 512, 16, 28};
    unsigned t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes,
                      4,
                      syn_type_int8,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId
        bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_id;
    unsigned char
        bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_params
            [] = {205, 204, 204, 61, 0, 0, 0, 0};
    addNodeToGraph(
        "dropout_fwd_bf16",
        {t1371_bert_encoder_layer_1_attention_self_Softmax_0,
         t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0},
        {t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0,
         t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0},
        (void*)
            bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_params,
        8,
        "bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0",
        0 /*graphIndex*/,
        &bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_id);

    setNodeDeterminstic(bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0 node
     * inputs:
     *     t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28] (dtype=bf16)
     *     t1366_bert_encoder_layer_1_attention_self_transpose_2_0[64, 512, 16, 28] (dtype=bf16)
     * outputs:
     *     t1374_bert_encoder_layer_1_attention_self_MatMul_1_0[64, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1374_bert_encoder_layer_1_attention_self_MatMul_1_0 tensor
    unsigned t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_max_sizes[] = {64, 512, 16, 28};
    unsigned t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_min_sizes[] = {64, 512, 16, 28};
    unsigned t1374_bert_encoder_layer_1_attention_self_MatMul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "t1374_bert_encoder_layer_1_attention_self_MatMul_1_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_id;
    unsigned char bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0,
                    t1366_bert_encoder_layer_1_attention_self_transpose_2_0},
                   {t1374_bert_encoder_layer_1_attention_self_MatMul_1_0},
                   (void*)bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_params,
                   2,
                   "bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_id);

    // TODO: SW-120885
    setConfigsForTest();
    compareRunsResults({t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0,
                        t1374_bert_encoder_layer_1_attention_self_MatMul_1_0});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_ASIC)
{
    const unsigned batchSize = 7, commonDim = 512, height = 512, width = 512;
    unsigned       aSizes[]   = {commonDim, height, batchSize};
    unsigned       bSizes[]   = {width, commonDim, batchSize};
    unsigned       outSizes[] = {width, height, batchSize};

    unsigned bgemmInA = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
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

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, partial_bgemm_ASIC)
{
    const unsigned commonDim = 4096, height = 512, width = 128;
    unsigned       aSizes[]   = {commonDim, height, 8, 4, 1};
    unsigned       bSizes[]   = {width, commonDim, 1, 4, 1};
    unsigned       outSizes[] = {width, height, 8, 4, 1};

    unsigned bgemmInA = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      5,
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
                                      5,
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
                                      5,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      outSizes,
                                      synTensorType::DATA_TENSOR)[0];

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_with_tpc_producers_and_consumer_ASIC)
{
    const unsigned batchSize = 13, commonDim = 256, height = 512, width = 768;
    unsigned       aSizes[]   = {commonDim, height, batchSize};
    unsigned       bSizes[]   = {width, commonDim, batchSize};
    unsigned       outSizes[] = {width, height, batchSize};

    unsigned reluAIn = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "reluAIn",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizes,
                                     3,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned bgemmInA = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSizes,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {reluAIn}, {bgemmInA}, nullptr, 0, "reluProducerA");

    unsigned reluBIn = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "reluBIn",
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

    unsigned bgemmInB = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
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

    addNodeToGraph("relu_fwd_bf16", {reluBIn}, {bgemmInB}, nullptr, 0, "reluProducerB");

    unsigned bgemmOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
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

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    unsigned reluOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "reluOut",
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

    addNodeToGraph("relu_fwd_bf16", {bgemmOut}, {reluOut}, nullptr, 0, "reluConsumer");

    setConfigsForTest();
    compareRunsResults({reluOut});
}

/*
 *   ┌───────┐        ┌────────┐      ┌───────────┐      ┌───────────┐         ┌─────┐
 *   │ add   ├───────►│reshape ├─────►│layer norm ├─────►│ reshape   ├────────►│     │
 *   └───────┘        └────────┘      └───────────┘      └───────────┘         │     │        ┌───────┐
 *                                                                             │gemm ├───────►│ add   │
 *                                                         ┌─────────┐         │     │        └───────┘
 *                                                         │ cast    ├────────►│     │
 *                                                         └─────────┘         └─────┘
 */
TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_tpc_producers_and_consumer_ASIC)
{
    /*************
     * g_0_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_cast_f32_to_bf16_n16_0
     *node inputs: g_0_t104_readvariableop_68_0[3072, 768] (dtype=float32) outputs:
     *     g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0[3072, 768]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t104_readvariableop_68_0 tensor
    unsigned g_0_t104_readvariableop_68_0_max_sizes[] = {3072, 768};
    unsigned g_0_t104_readvariableop_68_0_min_sizes[] = {3072, 768};
    unsigned g_0_t104_readvariableop_68_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t104_readvariableop_68_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t104_readvariableop_68_0_max_sizes,
                                                          2,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t104_readvariableop_68_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0 tensor
    unsigned g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_max_sizes[] =
        {3072, 768};
    unsigned g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_min_sizes[] =
        {3072, 768};
    unsigned g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_cast_f32_to_bf16_n16_0_id;
    addNodeToGraph(
        "cast_f32_to_bf16",
        {g_0_t104_readvariableop_68_0},
        {g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0},
        nullptr,
        0,
        "g_0_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_cast_f32_to_bf16_n16_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_cast_f32_to_bf16_n16_0_id);

    /*************
     * g_0_memcpy_1414_0 node
     * inputs:
     *     g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0[3072, 768]
     *(dtype=bf16) outputs:
     *     g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy[3072,
     *768] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy
    // tensor
    unsigned
        g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy_max_sizes
            [] = {3072, 768};
    unsigned
        g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy_min_sizes
            [] = {3072, 768};
    unsigned g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1414_0_id;
    addNodeToGraph(
        "memcpy",
        {g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0},
        {g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0_memcpy},
        nullptr,
        0,
        "g_0_memcpy_1414_0",
        0 /*graphIndex*/,
        &g_0_memcpy_1414_0_id);

    /*************
     * g_0_memcpy_1419_0 node
     * inputs:
     *     g_0_t248_bert_encoder_Reshape_1_0_before_memcpy[768, 4096] (dtype=bf16)
     * outputs:
     *     g_0_t248_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t248_bert_encoder_Reshape_1_0_before_memcpy tensor
    unsigned g_0_t248_bert_encoder_Reshape_1_0_before_memcpy_max_sizes[] = {768, 4096};
    unsigned g_0_t248_bert_encoder_Reshape_1_0_before_memcpy_min_sizes[] = {768, 4096};
    unsigned g_0_t248_bert_encoder_Reshape_1_0_before_memcpy =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t248_bert_encoder_Reshape_1_0_before_memcpy",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t248_bert_encoder_Reshape_1_0_before_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t248_bert_encoder_Reshape_1_0_before_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t248_bert_encoder_Reshape_1_0 tensor
    unsigned  g_0_t248_bert_encoder_Reshape_1_0_max_sizes[] = {768, 4096};
    unsigned  g_0_t248_bert_encoder_Reshape_1_0_min_sizes[] = {768, 4096};
    unsigned  g_0_t248_bert_encoder_Reshape_1_0             = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               false,
                                                               "g_0_t248_bert_encoder_Reshape_1_0",
                                                               MEM_INIT_ALL_ZERO,
                                                               nullptr,
                                                               g_0_t248_bert_encoder_Reshape_1_0_max_sizes,
                                                               2,
                                                               syn_type_bf16,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_t248_bert_encoder_Reshape_1_0_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1419_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t248_bert_encoder_Reshape_1_0_before_memcpy},
                   {g_0_t248_bert_encoder_Reshape_1_0},
                   nullptr,
                   0,
                   "g_0_memcpy_1419_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1419_0_id);

    /*************
     * g_0_memcpy_1420_0 node
     * inputs:
     *     g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy[768, 4096] (dtype=bf16)
     * outputs:
     *     g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy tensor
    unsigned g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy_max_sizes[] = {768, 4096};
    unsigned g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy_min_sizes[] = {768, 4096};
    unsigned g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0 tensor
    unsigned g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_max_sizes[] = {768, 4096};
    unsigned g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_min_sizes[] = {768, 4096};
    unsigned g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1420_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0_before_memcpy},
                   {g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0},
                   nullptr,
                   0,
                   "g_0_memcpy_1420_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1420_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_output_add_add_fwd_bf16_n256_0 node
     * inputs:
     *     g_0_t248_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     *     g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0[768, 4096] (dtype=bf16)
     * outputs:
     *     g_0_t544_bert_encoder_layer_0_attention_output_add_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t544_bert_encoder_layer_0_attention_output_add_0 tensor
    unsigned g_0_t544_bert_encoder_layer_0_attention_output_add_0_max_sizes[] = {768, 4096};
    unsigned g_0_t544_bert_encoder_layer_0_attention_output_add_0_min_sizes[] = {768, 4096};
    unsigned g_0_t544_bert_encoder_layer_0_attention_output_add_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t544_bert_encoder_layer_0_attention_output_add_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t544_bert_encoder_layer_0_attention_output_add_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t544_bert_encoder_layer_0_attention_output_add_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_attention_output_add_add_fwd_bf16_n256_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_t248_bert_encoder_Reshape_1_0, g_0_t542_bert_encoder_layer_0_attention_output_dropout_Mul_1_0},
                   {g_0_t544_bert_encoder_layer_0_attention_output_add_0},
                   nullptr,
                   0,
                   "g_0_bert_encoder_layer_0_attention_output_add_add_fwd_bf16_n256_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_attention_output_add_add_fwd_bf16_n256_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n257_0 node
     * inputs:
     *     g_0_t544_bert_encoder_layer_0_attention_output_add_0[768, 4096] (dtype=bf16)
     *     g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[768, 1, 1, 4096] (dtype=uint32)
     *(shape tensor) outputs: g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[768, 1, 1, 4096]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 1, 1, 4096};
    unsigned g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 1, 1, 4096};
    unsigned g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 1, 1, 4096};
    unsigned g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 1, 1, 4096};
    unsigned g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n257_0_id;
    addNodeToGraph("reshape",
                   {g_0_t544_bert_encoder_layer_0_attention_output_add_0,
                    g_0_t549_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm},
                   {g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm},
                   nullptr,
                   0,
                   "g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n257_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n257_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n261_0 node
     * inputs:
     *     g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[768, 1, 1, 4096] (dtype=bf16)
     *     g_0_t133_readvariableop_64_0[768] (dtype=float32)
     *     g_0_t132_readvariableop_60_0[768] (dtype=float32)
     * outputs:
     *     g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[768, 1, 1, 4096] (dtype=bf16)
     *     g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 4096] (dtype=float32)
     *     g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 4096] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t133_readvariableop_64_0 tensor
    unsigned g_0_t133_readvariableop_64_0_max_sizes[] = {768};
    unsigned g_0_t133_readvariableop_64_0_min_sizes[] = {768};
    unsigned g_0_t133_readvariableop_64_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t133_readvariableop_64_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t133_readvariableop_64_0_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t133_readvariableop_64_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t132_readvariableop_60_0 tensor
    unsigned g_0_t132_readvariableop_60_0_max_sizes[] = {768};
    unsigned g_0_t132_readvariableop_60_0_min_sizes[] = {768};
    unsigned g_0_t132_readvariableop_60_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t132_readvariableop_60_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t132_readvariableop_60_0_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t132_readvariableop_60_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 1, 1, 4096};
    unsigned g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 1, 1, 4096};
    unsigned g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 1, 1, 4096};
    unsigned g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 1, 1, 4096};
    unsigned g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 1, 1, 4096};
    unsigned g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 1, 1, 4096};
    unsigned g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n261_0_id;
    unsigned char
        g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n261_0_params[] =
            {1, 0, 0, 0, 111, 18, 131, 58};
    addNodeToGraph(
        "layer_norm_fwd_bf16",
        {g_0_t548_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm,
         g_0_t133_readvariableop_64_0,
         g_0_t132_readvariableop_60_0},
        {g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm,
         g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm,
         g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm},
        (void*)g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n261_0_params,
        8,
        "g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n261_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n261_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n258_0 node
     * inputs:
     *     g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[768, 1, 1, 4096] (dtype=bf16)
     *     g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[768, 4096] (dtype=uint32) (shape
     *tensor) outputs: g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0[768, 4096]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 4096};
    unsigned g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 4096};
    unsigned g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      2,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {768, 4096};
    unsigned g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {768, 4096};
    unsigned g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n258_0_id;
    addNodeToGraph("reshape",
                   {g_0_t550_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm,
                    g_0_t551_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm},
                   {g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0},
                   nullptr,
                   0,
                   "g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n258_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_reshape_n258_0_id);

    /*************
     * g_0_bert_encoder_layer_0_intermediate_dense_MatMul_gemm_n262_0 node
     * inputs:
     *     g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0[768, 4096] (dtype=bf16)
     *     g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0[3072, 768]
     *(dtype=bf16) outputs: g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0[3072, 4096] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0 tensor
    unsigned g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0_max_sizes[] = {3072, 4096};
    unsigned g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0_min_sizes[] = {3072, 4096};
    unsigned g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_layer_0_intermediate_dense_MatMul_gemm_n262_0_id;
    unsigned char g_0_bert_encoder_layer_0_intermediate_dense_MatMul_gemm_n262_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0,
                    g_0_t161_bert_encoder_layer_0_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_13_0},
                   {g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0},
                   (void*)g_0_bert_encoder_layer_0_intermediate_dense_MatMul_gemm_n262_0_params,
                   2,
                   "g_0_bert_encoder_layer_0_intermediate_dense_MatMul_gemm_n262_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_intermediate_dense_MatMul_gemm_n262_0_id);

    /*************
     * g_0_memcpy_1413_0 node
     * inputs:
     *     g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0[768, 4096] (dtype=bf16)
     * outputs:
     *     g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy tensor
    unsigned g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy_max_sizes[] = {768,
                                                                                                              4096};
    unsigned g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy_min_sizes[] = {768,
                                                                                                              4096};
    unsigned g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1413_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0},
                   {g_0_t545_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1413_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1413_0_id);

    /*************
     * g_0_memcpy_1415_0 node
     * inputs:
     *     g_0_t544_bert_encoder_layer_0_attention_output_add_0[768, 4096] (dtype=bf16)
     * outputs:
     *     g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy tensor
    unsigned g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy_max_sizes[] = {768, 4096};
    unsigned g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy_min_sizes[] = {768, 4096};
    unsigned g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1415_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t544_bert_encoder_layer_0_attention_output_add_0},
                   {g_0_t544_bert_encoder_layer_0_attention_output_add_0_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1415_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1415_0_id);

    /*************
     * g_0_memcpy_1416_0 node
     * inputs:
     *     g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 4096] (dtype=float32)
     * outputs:
     *     g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy[1, 1, 1, 4096]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy tensor
    unsigned g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_max_sizes[] = {1,
                                                                                                            1,
                                                                                                            1,
                                                                                                            4096};
    unsigned g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_min_sizes[] = {1,
                                                                                                            1,
                                                                                                            1,
                                                                                                            4096};
    unsigned g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1416_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm},
                   {g_0_t552_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1416_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1416_0_id);

    /*************
     * g_0_memcpy_1417_0 node
     * inputs:
     *     g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 4096] (dtype=float32)
     * outputs:
     *     g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy[1, 1, 1, 4096]
     *(dtype=float32) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy tensor
    unsigned g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_max_sizes[] = {1,
                                                                                                            1,
                                                                                                            1,
                                                                                                            4096};
    unsigned g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_min_sizes[] = {1,
                                                                                                            1,
                                                                                                            1,
                                                                                                            4096};
    unsigned g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1417_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm},
                   {g_0_t554_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1417_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1417_0_id);

    /*************
     * g_0_memcpy_1421_0 node
     * inputs:
     *     g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy[3072, 1] (dtype=bf16)
     * outputs:
     *     g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd[3072, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy tensor
    unsigned g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy_max_sizes[] = {3072, 1};
    unsigned g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy_min_sizes[] = {3072, 1};
    unsigned g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd tensor
    unsigned g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_max_sizes[] = {3072, 1};
    unsigned g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_min_sizes[] = {3072, 1};
    unsigned g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1421_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd_before_memcpy},
                   {g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd},
                   nullptr,
                   0,
                   "g_0_memcpy_1421_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1421_0_id);

    /*************
     * g_0_bert_encoder_layer_0_intermediate_dense_BiasAdd_add_fwd_bf16_n264_0 node
     * inputs:
     *     g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0[3072, 4096] (dtype=bf16)
     *     g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd[3072, 1] (dtype=bf16)
     * outputs:
     *     g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0[3072, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0 tensor
    unsigned g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_max_sizes[] = {3072, 4096};
    unsigned g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_min_sizes[] = {3072, 4096};
    unsigned g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_intermediate_dense_BiasAdd_add_fwd_bf16_n264_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_t556_bert_encoder_layer_0_intermediate_dense_MatMul_0,
                    g_0_t558_bert_encoder_layer_0_intermediate_dense_BiasAdd},
                   {g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0},
                   nullptr,
                   0,
                   "g_0_bert_encoder_layer_0_intermediate_dense_BiasAdd_add_fwd_bf16_n264_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_intermediate_dense_BiasAdd_add_fwd_bf16_n264_0_id);

    /*************
     * g_0_memcpy_1418_0 node
     * inputs:
     *     g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0[3072, 4096] (dtype=bf16)
     * outputs:
     *     g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy[3072, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy tensor
    unsigned g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy_max_sizes[] = {3072, 4096};
    unsigned g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy_min_sizes[] = {3072, 4096};
    unsigned g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1418_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0},
                   {g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1418_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1418_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_t557_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_memcpy});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, dedw_ASIC)
{
    std::vector<unsigned> xShape  = {4, 8, 8, 8, 8};
    std::vector<unsigned> dwShape = {4, 4, 3, 3, 3};

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

    const auto x  = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       xShape.data(),
                                       xShape.size(),
                                       syn_type_float,
                                       nullptr,
                                       "x");
    const auto dy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        xShape.data(),
                                        xShape.size(),
                                        syn_type_float,
                                        nullptr,
                                        "dy");
    const auto dw = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_NONE,
                                        nullptr,
                                        dwShape.data(),
                                        dwShape.size(),
                                        syn_type_float,
                                        nullptr,
                                        "dw");

    addNodeToGraph(NodeFactory::deDw3DNodeTypeName, {dy, x}, {dw}, &convParams, sizeof(convParams));

    setConfigsForTest();
    compareRunsResults({dw});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, dedw_with_cd_concurrency_ASIC)
{
    GlobalConfTestSetter enable_cdc("ENABLE_LB_MME_CONCURRENCY_OPT", "true");

    /*************
     * g_0_layer3_5_conv2_dedw_0 node
     * inputs:
     *     g_0_layer3_5_bn2_grad_input[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_5_relu1_output[256, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_5_conv2_weight_grad[256, 256, 3, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_5_bn2_grad_input tensor
    unsigned g_0_layer3_5_bn2_grad_input_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_bn2_grad_input_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_bn2_grad_input             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_layer3_5_bn2_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_5_bn2_grad_input_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_5_bn2_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_5_relu1_output tensor
    unsigned g_0_layer3_5_relu1_output_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_relu1_output_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_relu1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_5_relu1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_5_relu1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_5_relu1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_5_conv2_weight_grad tensor
    unsigned      g_0_layer3_5_conv2_weight_grad_max_sizes[] = {256, 256, 3, 3};
    unsigned      g_0_layer3_5_conv2_weight_grad_min_sizes[] = {256, 256, 3, 3};
    unsigned      g_0_layer3_5_conv2_weight_grad             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            true,
                                                            "g_0_layer3_5_conv2_weight_grad",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_layer3_5_conv2_weight_grad_max_sizes,
                                                            4,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_layer3_5_conv2_weight_grad_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer3_5_conv2_dedw_0_id;
    unsigned char g_0_layer3_5_conv2_dedw_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_layer3_5_bn2_grad_input, g_0_layer3_5_relu1_output},
                   {g_0_layer3_5_conv2_weight_grad},
                   (void*)g_0_layer3_5_conv2_dedw_0_params,
                   112,
                   "g_0_layer3_5_conv2_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_5_conv2_dedw_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_layer3_5_conv2_weight_grad});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, dedw_with_cd_concurrency_and_perforation_ASIC)
{
    // Last DEDW node from syn_resnet50_full_fwd_bwd_bf16 (worker_0_conv1_dedw).

    // create g_0_worker_0_bn1_grad_input tensor
    unsigned g_0_worker_0_bn1_grad_input_max_sizes[] = {64, 112, 112, 64};
    unsigned g_0_worker_0_bn1_grad_input_min_sizes[] = {64, 112, 112, 64};
    unsigned g_0_worker_0_bn1_grad_input             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_worker_0_bn1_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_worker_0_bn1_grad_input_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_worker_0_bn1_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_input tensor
    unsigned g_0_input_max_sizes[] = {3, 224, 224, 64};
    unsigned g_0_input_min_sizes[] = {3, 224, 224, 64};
    unsigned g_0_input             = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "g_0_input",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_input_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_input_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_worker_0_conv1_weight_grad tensor
    unsigned      g_0_worker_0_conv1_weight_grad_max_sizes[] = {64, 3, 7, 7};
    unsigned      g_0_worker_0_conv1_weight_grad_min_sizes[] = {64, 3, 7, 7};
    unsigned      g_0_worker_0_conv1_weight_grad             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            true,
                                                            "g_0_worker_0_conv1_weight_grad",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_worker_0_conv1_weight_grad_max_sizes,
                                                            4,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_worker_0_conv1_weight_grad_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_worker_0_conv1_dedw_0_id;
    unsigned char g_0_worker_0_conv1_dedw_0_params[] = {
        7, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0,
        3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_worker_0_bn1_grad_input, g_0_input},
                   {g_0_worker_0_conv1_weight_grad},
                   (void*)g_0_worker_0_conv1_dedw_0_params,
                   112,
                   "g_0_worker_0_conv1_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_worker_0_conv1_dedw_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_worker_0_conv1_weight_grad});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, dedw_with_cd_concurrency_cast_required_ASIC)
{
    GlobalConfTestSetter enable_cdc("ENABLE_LB_MME_CONCURRENCY_OPT", "true");

    /*************
     * g_0_layer3_5_conv2_dedw_0 node
     * inputs:
     *     g_0_layer3_5_bn2_grad_input[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_5_relu1_output[256, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_5_conv2_weight_grad[256, 256, 3, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_5_bn2_grad_input tensor
    unsigned g_0_layer3_5_bn2_grad_input_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_bn2_grad_input_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_bn2_grad_input             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_layer3_5_bn2_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_5_bn2_grad_input_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_5_bn2_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_5_relu1_output tensor
    unsigned g_0_layer3_5_relu1_output_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_relu1_output_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_5_relu1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_5_relu1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_5_relu1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_5_relu1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_5_conv2_weight_grad tensor
    unsigned      g_0_layer3_5_conv2_weight_grad_max_sizes[] = {256, 256, 3, 3};
    unsigned      g_0_layer3_5_conv2_weight_grad_min_sizes[] = {256, 256, 3, 3};
    unsigned      g_0_layer3_5_conv2_weight_grad             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            true,
                                                            "g_0_layer3_5_conv2_weight_grad",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_layer3_5_conv2_weight_grad_max_sizes,
                                                            4,
                                                            syn_type_bf16,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_layer3_5_conv2_weight_grad_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer3_5_conv2_dedw_0_id;
    unsigned char g_0_layer3_5_conv2_dedw_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_layer3_5_bn2_grad_input, g_0_layer3_5_relu1_output},
                   {g_0_layer3_5_conv2_weight_grad},
                   (void*)g_0_layer3_5_conv2_dedw_0_params,
                   112,
                   "g_0_layer3_5_conv2_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_5_conv2_dedw_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_layer3_5_conv2_weight_grad});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, dedx_ASIC)
{
    std::vector<unsigned> xShape = {4, 8, 8, 8, 8};
    std::vector<unsigned> wShape = {4, 4, 3, 3, 3};

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

    const auto dy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        xShape.data(),
                                        xShape.size(),
                                        syn_type_float,
                                        nullptr,
                                        "dy");

    const auto w  = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       wShape.data(),
                                       wShape.size(),
                                       syn_type_float,
                                       nullptr,
                                       "w");
    const auto dyShape = createShapeTensor(INPUT_TENSOR, xShape.data(), xShape.data(), xShape.size());
    const auto dx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_NONE,
                                        nullptr,
                                        xShape.data(),
                                        xShape.size(),
                                        syn_type_float,
                                        nullptr,
                                        "dx");

    addNodeToGraph(NodeFactory::deDx3DNodeTypeName, {dy, w, dyShape}, {dx}, &convParams, sizeof(convParams), "dedx");

    const auto reluIn   = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            xShape.data(),
                                            xShape.size(),
                                            syn_type_float,
                                            nullptr,
                                            "X");
    const auto reluGrad = createPersistTensor(OUTPUT_TENSOR,
                                              MEM_INIT_NONE,
                                              nullptr,
                                              xShape.data(),
                                              xShape.size(),
                                              syn_type_float,
                                              nullptr,
                                              "ReLU-DX");

    addNodeToGraph("relu_bwd_f32", {dx, reluIn}, {reluGrad}, nullptr, 0, "relu_bwd");
    setConfigsForTest();
    compareRunsResults({dx, reluGrad});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, resnet_conv_fwd_ASIC)
{
    // Graph #0

    /*************
     * g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_cast_f32_to_bf16_n233_0 node
     * inputs:
     *     g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0[1024, 512, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0[1024, 512, 1, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0 tensor
    unsigned g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0_max_sizes[] = {1024,
                                                                                                        512,
                                                                                                        1,
                                                                                                        1};
    unsigned g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0_min_sizes[] = {1024,
                                                                                                        512,
                                                                                                        1,
                                                                                                        1};
    unsigned g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0 tensor
    unsigned g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_max_sizes[] = {1024, 512, 1, 1};
    unsigned g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_min_sizes[] = {1024, 512, 1, 1};
    unsigned g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_cast_f32_to_bf16_n233_0_id;
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_t526_while_body__1_while_resnet50_res4a_branch1_conv2d_readvariableop_0},
                   {g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0},
                   nullptr,
                   0,
                   "g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_cast_f32_to_bf16_n233_0",
                   0 /*graphIndex*/,
                   &g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_cast_f32_to_bf16_n233_0_id);

    /*************
     * g_0_memcpy_1456_0 node
     * inputs:
     *     g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0[1024, 512, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy[1024, 512, 1, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy tensor
    unsigned g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy_max_sizes[] = {1024, 512, 1, 1};
    unsigned g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy_min_sizes[] = {1024, 512, 1, 1};
    unsigned g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1456_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0},
                   {g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1456_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1456_0_id);

    /*************
     * g_0_memcpy_1464_0 node
     * inputs:
     *     g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy[512, 28, 28, 256] (dtype=bf16)
     * outputs:
     *     g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0[512, 28, 28, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy tensor
    unsigned g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy_max_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy_min_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0 tensor
    unsigned g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_max_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_min_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1464_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0_before_memcpy},
                   {g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0},
                   nullptr,
                   0,
                   "g_0_memcpy_1464_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1464_0_id);

    /*************
     * g_0_memcpy_1465_0 node
     * inputs:
     *     g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy[512, 28, 28, 256] (dtype=bf16)
     * outputs:
     *     g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0[512, 28, 28, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy tensor
    unsigned g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy_max_sizes[] = {512,
                                                                                                         28,
                                                                                                         28,
                                                                                                         256};
    unsigned g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy_min_sizes[] = {512,
                                                                                                         28,
                                                                                                         28,
                                                                                                         256};
    unsigned g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0 tensor
    unsigned g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_max_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_min_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1465_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0_before_memcpy},
                   {g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0},
                   nullptr,
                   0,
                   "g_0_memcpy_1465_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1465_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n513_0 node
     * inputs:
     *     g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0[512, 28, 28, 256] (dtype=bf16)
     *     g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0[512] (dtype=float32)
     *     g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0[512] (dtype=float32)
     *     g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0[512] (dtype=float32)
     *     g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0[512] (dtype=float32)
     * outputs:
     *     g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0[512, 28, 28, 256] (dtype=bf16)
     *     g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3[512] (dtype=float32)
     *     g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3[512] (dtype=float32)
     *     g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1[512] (dtype=float32)
     *     g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3[512] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0 tensor
    unsigned g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0_max_sizes[] = {512};
    unsigned g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0_min_sizes[] = {512};
    unsigned g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0 tensor
    unsigned g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0_max_sizes[] = {512};
    unsigned g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0_min_sizes[] = {512};
    unsigned g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0_max_sizes[] = {512};
    unsigned g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0_min_sizes[] = {512};
    unsigned g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {
        512};
    unsigned g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {
        512};
    unsigned g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0 tensor
    unsigned g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0_max_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0_min_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3 tensor
    unsigned g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_max_sizes[] = {512};
    unsigned g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_min_sizes[] = {512};
    unsigned g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3 tensor
    unsigned g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_max_sizes[] = {512};
    unsigned g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_min_sizes[] = {512};
    unsigned g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1 tensor
    unsigned g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1_max_sizes[] = {512};
    unsigned g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1_min_sizes[] = {512};
    unsigned g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3 tensor
    unsigned g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_max_sizes[] = {512};
    unsigned g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_min_sizes[] = {512};
    unsigned g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n513_0_id;
    unsigned char g_0_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n513_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_fwd_bf16",
        {g_0_t1386_while_body__1_while_resnet50_res3d_branch2c_Conv2D_0,
         g_0_t662_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_1_0,
         g_0_t661_while_body__1_while_resnet50_bn3d_branch2c_readvariableop_0,
         g_0_t663_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_0,
         g_0_t664_while_body__1_while_resnet50_bn3d_branch2c_fusedbatchnormv3_readvariableop_1_0},
        {g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0,
         g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3,
         g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3,
         g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1,
         g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3},
        (void*)g_0_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n513_0_params,
        16,
        "g_0_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n513_0",
        0 /*graphIndex*/,
        &g_0_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_batch_norm_fwd_bf16_n513_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_add_6_add_add_fwd_bf16_n521_0 node
     * inputs:
     *     g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0[512, 28, 28, 256] (dtype=bf16)
     *     g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0[512, 28, 28, 256] (dtype=bf16)
     * outputs:
     *     g_0_t1406_while_body__1_while_resnet50_add_6_add_0[512, 28, 28, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1406_while_body__1_while_resnet50_add_6_add_0 tensor
    unsigned g_0_t1406_while_body__1_while_resnet50_add_6_add_0_max_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1406_while_body__1_while_resnet50_add_6_add_0_min_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1406_while_body__1_while_resnet50_add_6_add_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1406_while_body__1_while_resnet50_add_6_add_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1406_while_body__1_while_resnet50_add_6_add_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1406_while_body__1_while_resnet50_add_6_add_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_add_6_add_add_fwd_bf16_n521_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_t1343_while_body__1_while_resnet50_activation_18_Relu_0,
                    g_0_t1387_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_0},
                   {g_0_t1406_while_body__1_while_resnet50_add_6_add_0},
                   nullptr,
                   0,
                   "g_0_while_body__1_while_resnet50_add_6_add_add_fwd_bf16_n521_0",
                   0 /*graphIndex*/,
                   &g_0_while_body__1_while_resnet50_add_6_add_add_fwd_bf16_n521_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_activation_21_Relu_relu_fwd_bf16_n522_0 node
     * inputs:
     *     g_0_t1406_while_body__1_while_resnet50_add_6_add_0[512, 28, 28, 256] (dtype=bf16)
     * outputs:
     *     g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0[512, 28, 28, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0 tensor
    unsigned g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_max_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_min_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_activation_21_Relu_relu_fwd_bf16_n522_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_t1406_while_body__1_while_resnet50_add_6_add_0},
                   {g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0},
                   nullptr,
                   0,
                   "g_0_while_body__1_while_resnet50_activation_21_Relu_relu_fwd_bf16_n522_0",
                   0 /*graphIndex*/,
                   &g_0_while_body__1_while_resnet50_activation_21_Relu_relu_fwd_bf16_n522_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_spatial_convolution_n523_0 node
     * inputs:
     *     g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0[512, 28, 28, 256] (dtype=bf16)
     *     g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0[1024, 512, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0[1024, 14, 14, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0 tensor
    unsigned g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_max_sizes[] = {1024, 14, 14, 256};
    unsigned g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_min_sizes[] = {1024, 14, 14, 256};
    unsigned g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_spatial_convolution_n523_0_id;
    unsigned char g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_spatial_convolution_n523_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0,
                    g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0},
                   {g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0},
                   (void*)g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_spatial_convolution_n523_0_params,
                   112,
                   "g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_spatial_convolution_n523_0",
                   0 /*graphIndex*/,
                   &g_0_while_body__1_while_resnet50_res4a_branch1_Conv2D_spatial_convolution_n523_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_batch_norm_fwd_bf16_n525_0 node
     * inputs:
     *     g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0[1024, 14, 14, 256] (dtype=bf16)
     *     g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0[1024] (dtype=float32)
     *     g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0[1024] (dtype=float32)
     *     g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0[1024] (dtype=float32)
     *     g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0[1024] (dtype=float32)
     * outputs:
     *     g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0[1024, 14, 14, 256] (dtype=bf16)
     *     g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3[1024] (dtype=float32)
     *     g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3[1024] (dtype=float32)
     *     g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1[1024] (dtype=float32)
     *     g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0 tensor
    unsigned g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0_max_sizes[] = {1024};
    unsigned g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0_min_sizes[] = {1024};
    unsigned g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0 tensor
    unsigned g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0_max_sizes[] = {1024};
    unsigned g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0_min_sizes[] = {1024};
    unsigned g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0_max_sizes[] = {1024};
    unsigned g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0_min_sizes[] = {1024};
    unsigned g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {
        1024};
    unsigned g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {
        1024};
    unsigned g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0 tensor
    unsigned g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_max_sizes[] = {1024, 14, 14, 256};
    unsigned g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_min_sizes[] = {1024, 14, 14, 256};
    unsigned g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3 tensor
    unsigned g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_max_sizes[] = {1024};
    unsigned g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_min_sizes[] = {1024};
    unsigned g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3 tensor
    unsigned g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_max_sizes[] = {1024};
    unsigned g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_min_sizes[] = {1024};
    unsigned g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1 tensor
    unsigned g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1_max_sizes[] = {1024};
    unsigned g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1_min_sizes[] = {1024};
    unsigned g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3 tensor
    unsigned g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_max_sizes[] = {1024};
    unsigned g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_min_sizes[] = {1024};
    unsigned g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_batch_norm_fwd_bf16_n525_0_id;
    unsigned char g_0_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_batch_norm_fwd_bf16_n525_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_fwd_bf16",
        {g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0,
         g_0_t666_while_body__1_while_resnet50_bn4a_branch1_readvariableop_1_0,
         g_0_t665_while_body__1_while_resnet50_bn4a_branch1_readvariableop_0,
         g_0_t667_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_0,
         g_0_t668_while_body__1_while_resnet50_bn4a_branch1_fusedbatchnormv3_readvariableop_1_0},
        {g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0,
         g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3,
         g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3,
         g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1,
         g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3},
        (void*)g_0_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_batch_norm_fwd_bf16_n525_0_params,
        16,
        "g_0_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_batch_norm_fwd_bf16_n525_0",
        0 /*graphIndex*/,
        &g_0_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_batch_norm_fwd_bf16_n525_0_id);

    /*************
     * g_0_memcpy_1454_0 node
     * inputs:
     *     g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0[512, 28, 28, 256] (dtype=bf16)
     * outputs:
     *     g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy[512, 28, 28, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy tensor
    unsigned g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy_max_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy_min_sizes[] = {512, 28, 28, 256};
    unsigned g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1454_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0},
                   {g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1454_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1454_0_id);

    /*************
     * g_0_memcpy_1455_0 node
     * inputs:
     *     g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0[1024, 14, 14, 256] (dtype=bf16)
     * outputs:
     *     g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy[1024, 14, 14, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy tensor
    unsigned g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy_max_sizes[] = {1024, 14, 14, 256};
    unsigned g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy_min_sizes[] = {1024, 14, 14, 256};
    unsigned g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1455_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0},
                   {g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1455_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1455_0_id);

    /*************
     * g_0_memcpy_1457_0 node
     * inputs:
     *     g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3[512] (dtype=float32)
     * outputs:
     *     g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy[512] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy tensor
    unsigned g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_max_sizes[] = {512};
    unsigned g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_min_sizes[] = {512};
    unsigned g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1457_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3},
                   {g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1457_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1457_0_id);

    /*************
     * g_0_memcpy_1458_0 node
     * inputs:
     *     g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3[512] (dtype=float32)
     * outputs:
     *     g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy[512] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy tensor
    unsigned g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy_max_sizes[] = {512};
    unsigned g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy_min_sizes[] = {512};
    unsigned g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1458_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3},
                   {g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1458_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1458_0_id);

    /*************
     * g_0_memcpy_1459_0 node
     * inputs:
     *     g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3[512] (dtype=float32)
     * outputs:
     *     g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy[512] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy tensor
    unsigned g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_max_sizes[] = {512};
    unsigned g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_min_sizes[] = {512};
    unsigned g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1459_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3},
                   {g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1459_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1459_0_id);

    /*************
     * g_0_memcpy_1460_0 node
     * inputs:
     *     g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0[1024, 14, 14, 256] (dtype=bf16)
     * outputs:
     *     g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy[1024, 14, 14, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy tensor
    unsigned g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy_max_sizes[] = {1024,
                                                                                                          14,
                                                                                                          14,
                                                                                                          256};
    unsigned g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy_min_sizes[] = {1024,
                                                                                                          14,
                                                                                                          14,
                                                                                                          256};
    unsigned g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1460_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0},
                   {g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1460_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1460_0_id);

    /*************
     * g_0_memcpy_1461_0 node
     * inputs:
     *     g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3[1024] (dtype=float32)
     * outputs:
     *     g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy tensor
    unsigned g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_max_sizes[] = {1024};
    unsigned g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_min_sizes[] = {1024};
    unsigned g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1461_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3},
                   {g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1461_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1461_0_id);

    /*************
     * g_0_memcpy_1462_0 node
     * inputs:
     *     g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3[1024] (dtype=float32)
     * outputs:
     *     g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy tensor
    unsigned g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy_max_sizes[] = {1024};
    unsigned g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy_min_sizes[] = {1024};
    unsigned g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1462_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3},
                   {g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1462_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1462_0_id);

    /*************
     * g_0_memcpy_1463_0 node
     * inputs:
     *     g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3[1024] (dtype=float32)
     * outputs:
     *     g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy tensor
    unsigned g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_max_sizes[] = {1024};
    unsigned g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_min_sizes[] = {1024};
    unsigned g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_1463_0_id;
    addNodeToGraph("memcpy",
                   {g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3},
                   {g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_1463_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_1463_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_t1420_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy,
                        g_0_t1395_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_3_memcpy,
                        g_0_t1397_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy,
                        g_0_t1388_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_1,
                        g_0_t1394_while_body__1_while_resnet50_bn3d_branch2c_FusedBatchNormV3_memcpy,
                        g_0_t1410_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_0_memcpy,
                        g_0_t1411_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_1,
                        g_0_t1417_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_memcpy,
                        g_0_t1418_while_body__1_while_resnet50_bn4a_branch1_FusedBatchNormV3_3_memcpy,
                        g_0_t810_while_body__1_while_resnet50_res4a_branch1_Conv2D_Cast_0_memcpy,
                        g_0_t1407_while_body__1_while_resnet50_activation_21_Relu_0_memcpy,
                        g_0_t1408_while_body__1_while_resnet50_res4a_branch1_Conv2D_0_memcpy});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_exact_producer_ASIC)
{
    const unsigned commonDim = 1100, height = 2570, width = 9865;
    unsigned       aSizes[]      = {commonDim, height};
    unsigned       bSizes[]      = {width, commonDim};
    unsigned       outSizes[]    = {width, height};
    unsigned       onehotSizes[] = {commonDim};

    unsigned producerIn = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "producerIn",
                                        MEM_INIT_ALL_ONES,
                                        nullptr,
                                        onehotSizes,
                                        1,
                                        syn_type_int32,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        onehotSizes,
                                        synTensorType::DATA_TENSOR)[0];

    unsigned gemmInA = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInA",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizes,
                                     synTensorType::DATA_TENSOR)[0];

    ns_OneHotKernel::Params oneHotParams = {1, height, 1, 0};
    addNodeToGraph("one_hot_fwd_f32", {producerIn}, {gemmInA}, (void*)&oneHotParams, sizeof(oneHotParams));

    unsigned gemmInB = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemmInB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "gemmOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     outSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     outSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synGEMMParams gemmParams(false, false);
    addNodeToGraph("gemm", {gemmInA, gemmInB}, {gemmOut}, &gemmParams, sizeof(gemmParams), "GEMM");

    setConfigsForTest();
    compareRunsResults({gemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_tpc_broadcast1_ASIC)
{
    const unsigned heightA = 2500, commonDim = 1500, widthB = 3424;
    unsigned       aSizes[]            = {commonDim, heightA};
    unsigned       aBroadcastSizes[]   = {commonDim};
    unsigned       bSizes[]            = {widthB, commonDim};
    unsigned       bBroadcastSizes[]   = {widthB};
    unsigned       outSizes[]          = {widthB, heightA};
    unsigned       outBroadcastSizes[] = {widthB};

    unsigned addAIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn1",
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

    unsigned addAIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aBroadcastSizes,
                                     1,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInA = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInA",
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

    addNodeToGraph("add_fwd_bf16", {addAIn1, addAIn2}, {gemmInA}, nullptr, 0, "addA");

    unsigned addBIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned addBIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bBroadcastSizes,
                                     1,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInB = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_bf16", {addBIn1, addBIn2}, {gemmInB}, nullptr, 0, "addB");

    unsigned gemmOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmOut",
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

    synGEMMParams params(false, false);
    addNodeToGraph("gemm", {gemmInA, gemmInB}, {gemmOut}, &params, sizeof(params), "GEMM");

    unsigned addOutIn2 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "addOutIn2",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       outBroadcastSizes,
                                       1,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       outBroadcastSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned addConsumerOut = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "addConsumerOut",
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

    addNodeToGraph("add_fwd_bf16", {gemmOut, addOutIn2}, {addConsumerOut}, nullptr, 0, "addOut");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({addConsumerOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_tpc_broadcast2_ASIC)
{
    const unsigned heightA = 2500, commonDim = 1500, widthB = 3424;
    unsigned       aSizes[]            = {commonDim, heightA};
    unsigned       aBroadcastSizes[]   = {commonDim, 1};
    unsigned       bSizes[]            = {widthB, commonDim};
    unsigned       bBroadcastSizes[]   = {widthB, 1};
    unsigned       outSizes[]          = {widthB, heightA};
    unsigned       outBroadcastSizes[] = {widthB, 1};

    unsigned addAIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn1",
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

    unsigned addAIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aBroadcastSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInA = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInA",
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

    addNodeToGraph("add_fwd_bf16", {addAIn1, addAIn2}, {gemmInA}, nullptr, 0, "addA");

    unsigned addBIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned addBIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bBroadcastSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInB = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_bf16", {addBIn1, addBIn2}, {gemmInB}, nullptr, 0, "addB");

    unsigned gemmOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmOut",
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

    synGEMMParams params(false, false);
    addNodeToGraph("gemm", {gemmInA, gemmInB}, {gemmOut}, &params, sizeof(params), "GEMM");

    unsigned addOutIn2 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "addOutIn2",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       outBroadcastSizes,
                                       2,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       outBroadcastSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned addConsumerOut = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "addConsumerOut",
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

    addNodeToGraph("add_fwd_bf16", {gemmOut, addOutIn2}, {addConsumerOut}, nullptr, 0, "addOut");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({addConsumerOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_tpc_broadcast3_ASIC)
{
    const unsigned heightA = 2500, commonDim = 1500, widthB = 3424;
    unsigned       aSizes[]            = {commonDim, heightA};
    unsigned       aBroadcastSizes[]   = {1, heightA};
    unsigned       bSizes[]            = {widthB, commonDim};
    unsigned       bBroadcastSizes[]   = {1, commonDim};
    unsigned       outSizes[]          = {widthB, heightA};
    unsigned       outBroadcastSizes[] = {1, heightA};

    unsigned addAIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn1",
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

    unsigned addAIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aBroadcastSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInA = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInA",
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

    addNodeToGraph("add_fwd_bf16", {addAIn1, addAIn2}, {gemmInA}, nullptr, 0, "addA");

    unsigned addBIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned addBIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bBroadcastSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInB = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_bf16", {addBIn1, addBIn2}, {gemmInB}, nullptr, 0, "addB");

    unsigned gemmOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmOut",
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

    synGEMMParams params(false, false);
    addNodeToGraph("gemm", {gemmInA, gemmInB}, {gemmOut}, &params, sizeof(params), "GEMM");

    unsigned addOutIn2 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "addOutIn2",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       outBroadcastSizes,
                                       2,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       outBroadcastSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned addConsumerOut = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "addConsumerOut",
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

    addNodeToGraph("add_fwd_bf16", {gemmOut, addOutIn2}, {addConsumerOut}, nullptr, 0, "addOut");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({addConsumerOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_tpc_broadcast4_ASIC)
{
    const unsigned heightA = 2500, commonDim = 1500, widthB = 3424;
    unsigned       aSizes[]            = {commonDim, heightA};
    unsigned       aBroadcastSizes[]   = {1};
    unsigned       bSizes[]            = {widthB, commonDim};
    unsigned       bBroadcastSizes[]   = {1};
    unsigned       outSizes[]          = {widthB, heightA};
    unsigned       outBroadcastSizes[] = {1};

    unsigned addAIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn1",
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

    unsigned addAIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addAIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aBroadcastSizes,
                                     1,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInA = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInA",
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

    addNodeToGraph("add_fwd_bf16", {addAIn1, addAIn2}, {gemmInA}, nullptr, 0, "addA");

    unsigned addBIn1 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned addBIn2 = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "addBIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bBroadcastSizes,
                                     1,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bBroadcastSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInB = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_bf16", {addBIn1, addBIn2}, {gemmInB}, nullptr, 0, "addB");

    unsigned gemmOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmOut",
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

    synGEMMParams params(false, false);
    addNodeToGraph("gemm", {gemmInA, gemmInB}, {gemmOut}, &params, sizeof(params), "GEMM");

    unsigned addOutIn2 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "addOutIn2",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       outBroadcastSizes,
                                       1,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       outBroadcastSizes,
                                       synTensorType::DATA_TENSOR)[0];

    unsigned addConsumerOut = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "addConsumerOut",
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

    addNodeToGraph("add_fwd_bf16", {gemmOut, addOutIn2}, {addConsumerOut}, nullptr, 0, "addOut");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({addConsumerOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_with_broadcast_ASIC)
{
    const unsigned batchSize = 13, commonDim = 256, height = 512, width = 768;
    unsigned       aSizes[]   = {commonDim, height, batchSize};
    unsigned       bSizes[]   = {width, commonDim, 1};
    unsigned       outSizes[] = {width, height, batchSize};

    unsigned bgemmInA = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
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

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_with_broadcast_and_tpc_producer_ASIC)
{
    const unsigned batchSize = 13, commonDim = 256, height = 512, width = 768;
    unsigned       aSizes[]   = {commonDim, height, batchSize};
    unsigned       bSizes[]   = {width, commonDim};
    unsigned       outSizes[] = {width, height, batchSize};

    unsigned bgemmInA = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned reluBIn = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "reluBIn",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned bgemmInB = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "bgemmInB",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      bSizes,
                                      2,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      bSizes,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {reluBIn}, {bgemmInB}, nullptr, 0, "reluB");

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

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_dummy_dim_in_input_ASIC)
{
    const unsigned commonDim = 512, height = 3072, width = 1024;
    unsigned       aSizes[]   = {commonDim, height};
    unsigned       bSizes[]   = {width, commonDim, 1};
    unsigned       outSizes[] = {width, height};

    unsigned gemmInA = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemmInA",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInB = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemmInB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "gemmOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     outSizes,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     outSizes,
                                     synTensorType::DATA_TENSOR)[0];

    synGEMMParams params(false, false);
    addNodeToGraph("gemm", {gemmInA, gemmInB}, {gemmOut}, &params, sizeof(params), "GEMM");

    setConfigsForTest();
    compareRunsResults({gemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_and_reshape_producer_with_shape_tensor_ASIC)
{
    const unsigned batchSize = 7, commonDim = 256, height = 512, width = 2570;
    unsigned       aSize[]         = {commonDim, height, batchSize};
    unsigned       bSize[]         = {width, commonDim, batchSize};
    unsigned       oSize[]         = {width, height, batchSize};
    unsigned       aSizeReshaped[] = {commonDim, batchSize * height};

    unsigned reluIn = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "reluIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    aSizeReshaped,
                                    2,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    aSizeReshaped,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned reluOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "reluOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizeReshaped,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizeReshaped,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "reluA");

    unsigned reshapeShape = createTensors(1,
                                          INPUT_TENSOR,
                                          false,
                                          "reshape_shape",
                                          MEM_INIT_NONE,
                                          nullptr,
                                          aSize,
                                          3,
                                          syn_type_uint32,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          aSize,
                                          synTensorType::SHAPE_TENSOR)[0];

    unsigned bgemmInA = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSize,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSize,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("reshape", {reluOut, reshapeShape}, {bgemmInA}, nullptr, 0, "reshapeA");

    unsigned bgemmInB = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInB",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      bSize,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      bSize,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned bgemmOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "bgemmOut",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      oSize,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      oSize,
                                      synTensorType::DATA_TENSOR)[0];

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_and_expand_dims_producer_ASIC)
{
    const unsigned batchSize = 15, commonDim = 1, height = 2570, width = 1024;
    unsigned       aSizes[]             = {commonDim, height, batchSize};
    unsigned       bSizes[]             = {width, commonDim, batchSize};
    unsigned       outSizes[]           = {width, height, batchSize};
    unsigned       aSizesBeforeExpand[] = {height, batchSize};

    unsigned reluIn = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "reluIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    aSizesBeforeExpand,
                                    2,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    aSizesBeforeExpand,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned reluOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "reluOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizesBeforeExpand,
                                     2,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizesBeforeExpand,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "reluA");

    unsigned bgemmInA = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned expandDim = 0;
    addNodeToGraph("expand_dims", {reluOut}, {bgemmInA}, (void*)&expandDim, sizeof(expandDim), "expandDimsA");

    unsigned bgemmInB = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInB",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      bSizes,
                                      3,
                                      syn_type_single,
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
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      outSizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      outSizes,
                                      synTensorType::DATA_TENSOR)[0];

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_and_squeeze_producer_ASIC)
{
    const unsigned batchSize = 7, commonDim = 256, height = 2048, width = 512;
    unsigned       aSizes[]              = {commonDim, height, batchSize};
    unsigned       bSizes[]              = {width, commonDim, batchSize};
    unsigned       outSizes[]            = {width, height, batchSize};
    unsigned       aSizesBeforeSqueeze[] = {commonDim, 1, height, 1, batchSize};

    unsigned reluIn = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "reluIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    aSizesBeforeSqueeze,
                                    5,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    aSizesBeforeSqueeze,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned reluOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "reluOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizesBeforeSqueeze,
                                     5,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizesBeforeSqueeze,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "reluA");

    unsigned bgemmInA = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSizes,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("squeeze", {reluOut}, {bgemmInA}, nullptr, 0, "squeezeA");

    unsigned bgemmInB = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInB",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      bSizes,
                                      3,
                                      syn_type_single,
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
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      outSizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      outSizes,
                                      synTensorType::DATA_TENSOR)[0];

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_and_transpose_producer_ASIC)
{
    const unsigned batchSize = 17, commonDim = 256, height = 2048, width = 512;
    unsigned       aSizes[]           = {commonDim, height, batchSize};
    unsigned       bSizes[]           = {width, commonDim, batchSize};
    unsigned       outSizes[]         = {width, height, batchSize};
    unsigned       aSizesTransposed[] = {commonDim, batchSize, height};

    unsigned reluIn = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "reluIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    aSizesTransposed,
                                    3,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    aSizesTransposed,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned reluOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "reluOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizesTransposed,
                                     3,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizesTransposed,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "reluA");

    unsigned bgemmInA = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSizes,
                                      synTensorType::DATA_TENSOR)[0];

    synTransposeParams transposeParams = {{TPD_Channel, TPD_Height, TPD_Width}, 3};
    addNodeToGraph("transpose", {reluOut}, {bgemmInA}, &transposeParams, sizeof(transposeParams), "transposeA");

    unsigned bgemmInB = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "bgemmInB",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      bSizes,
                                      3,
                                      syn_type_single,
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
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      outSizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      outSizes,
                                      synTensorType::DATA_TENSOR)[0];

    synGEMMParams gemmParams(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &gemmParams, sizeof(gemmParams), "BGEMM");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({bgemmOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, bgemm_and_tpc_producers_with_persistent_outputs_ASIC)
{
    const unsigned batchSize = 13, commonDim = 256, height = 512, width = 768;
    unsigned       aSizes[]   = {commonDim, height, batchSize};
    unsigned       bSizes[]   = {width, commonDim, batchSize};
    unsigned       outSizes[] = {width, height, batchSize};

    unsigned reluAIn = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "reluAIn",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     aSizes,
                                     3,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     aSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned bgemmInA = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "bgemmInA",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      aSizes,
                                      3,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      aSizes,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {reluAIn}, {bgemmInA}, nullptr, 0, "reluA");

    unsigned reluBIn = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "reluBIn",
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

    unsigned bgemmInB = createTensors(1,
                                      OUTPUT_TENSOR,
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

    addNodeToGraph("relu_fwd_bf16", {reluBIn}, {bgemmInB}, nullptr, 0, "reluB");

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

    synGEMMParams params(false, false);
    addNodeToGraph("batch_gemm", {bgemmInA, bgemmInB}, {bgemmOut}, &params, sizeof(params), "BGEMM");

    unsigned outConsumer = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "outConsumer",
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

    addNodeToGraph("relu_fwd_bf16", {bgemmOut}, {outConsumer}, nullptr, 0, "reluPut");

    setConfigsForTest();
    compareRunsResults({bgemmInA, bgemmInB, bgemmOut, outConsumer});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, dedw_with_bf16_output_cast_required_ASIC)
{
    const unsigned batchSize  = 257;
    unsigned       in1Sizes[] = {64, 56, 56, batchSize};
    unsigned       in2Sizes[] = {64, 56, 56, batchSize};
    unsigned       outSizes[] = {64, 64, 1, 1};

    unsigned in1 = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "in1",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 in1Sizes,
                                 4,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in1Sizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned in2 = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "in2",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 in2Sizes,
                                 4,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in2Sizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned out = createTensors(1,
                                 OUTPUT_TENSOR,
                                 true,
                                 "out",
                                 MEM_INIT_ALL_ZERO,
                                 nullptr,
                                 outSizes,
                                 4,
                                 syn_type_bf16,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 outSizes,
                                 synTensorType::DATA_TENSOR)[0];

    synConvolutionParams dedwParams(1, 1, 1, 1, 0, 0, 0, 0, 1, 1);
    addNodeToGraph("dedw", {in1, in2}, {out}, &dedwParams, sizeof(dedwParams), "DEDW");

    unsigned reluOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "reluOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     outSizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     outSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {out}, {reluOut}, nullptr, 0, "reluConsumer");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    setConfigsForTest();
    compareRunsResults({out, reluOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest,
          masked_bgemm_ASIC,
          {synDeviceGaudi2})  // Masked BGEMM is currently not supported in Gaudi3
{
    const unsigned batchSize = 6, commonDim = 128, height = 512, width = 384, masksCommonDim = 13;
    unsigned       aSizes[]     = {commonDim, height, batchSize, batchSize};
    unsigned       bSizes[]     = {width, commonDim, batchSize, batchSize};
    unsigned       maskASizes[] = {masksCommonDim, height, 1, batchSize};
    unsigned       maskBSizes[] = {width, masksCommonDim, 1, batchSize};
    unsigned       cSizes[]     = {width, height, batchSize, batchSize};

    unsigned tensorAIdx = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              aSizes,
                                              4,
                                              syn_type_bf16,
                                              nullptr,
                                              "A",
                                              0,
                                              0,
                                              nullptr,
                                              aSizes);

    unsigned tensorBIdx = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              bSizes,
                                              4,
                                              syn_type_bf16,
                                              nullptr,
                                              "B",
                                              0,
                                              0,
                                              nullptr,
                                              bSizes);

    unsigned tensorMaskAIdx = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  maskASizes,
                                                  4,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  "MaskA",
                                                  0,
                                                  0,
                                                  nullptr,
                                                  maskASizes);

    unsigned tensorMaskBIdx = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  maskBSizes,
                                                  4,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  "MaskB",
                                                  0,
                                                  0,
                                                  nullptr,
                                                  maskBSizes);

    unsigned tensorCIdx = createPersistTensor(OUTPUT_TENSOR,
                                              MEM_INIT_ALL_ZERO,
                                              nullptr,
                                              cSizes,
                                              4,
                                              syn_type_bf16,
                                              nullptr,
                                              "C",
                                              0,
                                              0,
                                              nullptr,
                                              cSizes);

    synGEMMParams params(false, false);
    addNodeToGraph(NodeFactory::maskedBatchGemmNodeTypeName,
                   {tensorAIdx, tensorBIdx, tensorMaskAIdx, tensorMaskBIdx},
                   {tensorCIdx},
                   &params,
                   sizeof(params),
                   "MaskedBgemm",
                   0);

    setConfigsForTest();
    compareRunsResults({tensorCIdx});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, multi_gemms_bundle_from_bert_ASIC)
{
    // Graph #0

    /*************
     * g_0_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_cast_f32_to_bf16_n8_0
     *node inputs: g_0_t106_readvariableop_24_0[768, 768] (dtype=float32) outputs:
     *     g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0[768, 768]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t106_readvariableop_24_0 tensor
    unsigned g_0_t106_readvariableop_24_0_max_sizes[] = {768, 768};
    unsigned g_0_t106_readvariableop_24_0_min_sizes[] = {768, 768};
    unsigned g_0_t106_readvariableop_24_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t106_readvariableop_24_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t106_readvariableop_24_0_max_sizes,
                                                          2,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t106_readvariableop_24_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0 tensor
    unsigned
        g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0_max_sizes[] = {
            768,
            768};
    unsigned
        g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0_min_sizes[] = {
            768,
            768};
    unsigned g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_cast_f32_to_bf16_n8_0_id;
    addNodeToGraph(
        "cast_f32_to_bf16",
        {g_0_t106_readvariableop_24_0},
        {g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0},
        nullptr,
        0,
        "g_0_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_cast_f32_to_bf16_n8_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_cast_f32_to_bf16_n8_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_cast_f32_to_bf16_n9_0 node
     * inputs:
     *     g_0_t107_readvariableop_33_0[768, 768] (dtype=float32)
     * outputs:
     *     g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0[768, 768]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t107_readvariableop_33_0 tensor
    unsigned g_0_t107_readvariableop_33_0_max_sizes[] = {768, 768};
    unsigned g_0_t107_readvariableop_33_0_min_sizes[] = {768, 768};
    unsigned g_0_t107_readvariableop_33_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t107_readvariableop_33_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t107_readvariableop_33_0_max_sizes,
                                                          2,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t107_readvariableop_33_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0 tensor
    unsigned g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0_max_sizes[] =
        {768, 768};
    unsigned g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0_min_sizes[] =
        {768, 768};
    unsigned g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_cast_f32_to_bf16_n9_0_id;
    addNodeToGraph(
        "cast_f32_to_bf16",
        {g_0_t107_readvariableop_33_0},
        {g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0},
        nullptr,
        0,
        "g_0_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_cast_f32_to_bf16_n9_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_cast_f32_to_bf16_n9_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n10_0
     *node inputs: g_0_t108_readvariableop_42_0[768, 768] (dtype=float32) outputs:
     *     g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0[768, 768]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t108_readvariableop_42_0 tensor
    unsigned g_0_t108_readvariableop_42_0_max_sizes[] = {768, 768};
    unsigned g_0_t108_readvariableop_42_0_min_sizes[] = {768, 768};
    unsigned g_0_t108_readvariableop_42_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t108_readvariableop_42_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t108_readvariableop_42_0_max_sizes,
                                                          2,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t108_readvariableop_42_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0 tensor
    unsigned
        g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_max_sizes[] = {
            768,
            768};
    unsigned
        g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_min_sizes[] = {
            768,
            768};
    unsigned g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n10_0_id;
    addNodeToGraph(
        "cast_f32_to_bf16",
        {g_0_t108_readvariableop_42_0},
        {g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0},
        nullptr,
        0,
        "g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n10_"
        "0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_cast_f32_to_bf16_n10_0_id);

    /*************
     * g_0_bert_embeddings_Gather_gather_fwd_f32_n45_0 node
     * inputs:
     *     g_0_t10_bert_embeddings_word_embeddings_0[768, 30522] (dtype=float32)
     *     g_0_t213_bert_embeddings_Reshape_0[4096] (dtype=int32)
     * outputs:
     *     g_0_t216_bert_embeddings_Gather_0[768, 4096] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t10_bert_embeddings_word_embeddings_0 tensor
    unsigned g_0_t10_bert_embeddings_word_embeddings_0_max_sizes[] = {768, 30522};
    unsigned g_0_t10_bert_embeddings_word_embeddings_0_min_sizes[] = {768, 30522};
    unsigned g_0_t10_bert_embeddings_word_embeddings_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t10_bert_embeddings_word_embeddings_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t10_bert_embeddings_word_embeddings_0_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t10_bert_embeddings_word_embeddings_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t213_bert_embeddings_Reshape_0 tensor
    unsigned g_0_t213_bert_embeddings_Reshape_0_max_sizes[] = {4096};
    unsigned g_0_t213_bert_embeddings_Reshape_0_min_sizes[] = {4096};
    unsigned g_0_t213_bert_embeddings_Reshape_0             = createTensors(1,
                                                                INPUT_TENSOR,
                                                                true,
                                                                "g_0_t213_bert_embeddings_Reshape_0",
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                g_0_t213_bert_embeddings_Reshape_0_max_sizes,
                                                                1,
                                                                syn_type_int32,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_t213_bert_embeddings_Reshape_0_min_sizes,
                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t216_bert_embeddings_Gather_0 tensor
    unsigned      g_0_t216_bert_embeddings_Gather_0_max_sizes[] = {768, 4096};
    unsigned      g_0_t216_bert_embeddings_Gather_0_min_sizes[] = {768, 4096};
    unsigned      g_0_t216_bert_embeddings_Gather_0             = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               false,
                                                               "g_0_t216_bert_embeddings_Gather_0",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               g_0_t216_bert_embeddings_Gather_0_max_sizes,
                                                               2,
                                                               syn_type_single,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_t216_bert_embeddings_Gather_0_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_embeddings_Gather_gather_fwd_f32_n45_0_id;
    unsigned char g_0_bert_embeddings_Gather_gather_fwd_f32_n45_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("gather_fwd_f32",
                   {g_0_t10_bert_embeddings_word_embeddings_0, g_0_t213_bert_embeddings_Reshape_0},
                   {g_0_t216_bert_embeddings_Gather_0},
                   (void*)g_0_bert_embeddings_Gather_gather_fwd_f32_n45_0_params,
                   8,
                   "g_0_bert_embeddings_Gather_gather_fwd_f32_n45_0",
                   0 /*graphIndex*/,
                   &g_0_bert_embeddings_Gather_gather_fwd_f32_n45_0_id);

    /*************
     * g_0_bert_embeddings_Gather_fp32_to_bf16_cast_2_cast_f32_to_bf16_n46_0 node
     * inputs:
     *     g_0_t216_bert_embeddings_Gather_0[768, 4096] (dtype=float32)
     * outputs:
     *     g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0 tensor
    unsigned g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0_max_sizes[] = {768, 4096};
    unsigned g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0_min_sizes[] = {768, 4096};
    unsigned g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_Gather_fp32_to_bf16_cast_2_cast_f32_to_bf16_n46_0_id;
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_t216_bert_embeddings_Gather_0},
                   {g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0},
                   nullptr,
                   0,
                   "g_0_bert_embeddings_Gather_fp32_to_bf16_cast_2_cast_f32_to_bf16_n46_0",
                   0 /*graphIndex*/,
                   &g_0_bert_embeddings_Gather_fp32_to_bf16_cast_2_cast_f32_to_bf16_n46_0_id);

    /*************
     * g_0_bert_embeddings_Reshape_1_reshape_n48_0 node
     * inputs:
     *     g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0[768, 4096] (dtype=bf16)
     *     g_0_t222_bert_embeddings_Reshape_1[768, 128, 32] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t221_bert_embeddings_Reshape_1_0[768, 128, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t222_bert_embeddings_Reshape_1 tensor
    unsigned g_0_t222_bert_embeddings_Reshape_1_max_sizes[] = {768, 128, 32};
    unsigned g_0_t222_bert_embeddings_Reshape_1_min_sizes[] = {768, 128, 32};
    unsigned g_0_t222_bert_embeddings_Reshape_1             = createTensors(1,
                                                                INPUT_TENSOR,
                                                                false,
                                                                "g_0_t222_bert_embeddings_Reshape_1",
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                g_0_t222_bert_embeddings_Reshape_1_max_sizes,
                                                                3,
                                                                syn_type_uint32,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_t222_bert_embeddings_Reshape_1_min_sizes,
                                                                synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t221_bert_embeddings_Reshape_1_0 tensor
    unsigned  g_0_t221_bert_embeddings_Reshape_1_0_max_sizes[] = {768, 128, 32};
    unsigned  g_0_t221_bert_embeddings_Reshape_1_0_min_sizes[] = {768, 128, 32};
    unsigned  g_0_t221_bert_embeddings_Reshape_1_0             = createTensors(1,
                                                                  OUTPUT_TENSOR,
                                                                  false,
                                                                  "g_0_t221_bert_embeddings_Reshape_1_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t221_bert_embeddings_Reshape_1_0_max_sizes,
                                                                  3,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t221_bert_embeddings_Reshape_1_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_Reshape_1_reshape_n48_0_id;
    addNodeToGraph("reshape",
                   {g_0_t218_bert_embeddings_Gather_fp32_to_bf16_cast_2_0, g_0_t222_bert_embeddings_Reshape_1},
                   {g_0_t221_bert_embeddings_Reshape_1_0},
                   nullptr,
                   0,
                   "g_0_bert_embeddings_Reshape_1_reshape_n48_0",
                   0 /*graphIndex*/,
                   &g_0_bert_embeddings_Reshape_1_reshape_n48_0_id);

    /*************
     * g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_add_fwd_bf16_n49_0 node
     * inputs:
     *     g_0_t221_bert_embeddings_Reshape_1_0[768, 128, 32] (dtype=bf16)
     *     g_0_t219_bert_embeddings_Reshape_3_0[768, 128, 32] (dtype=bf16)
     * outputs:
     *     g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0[768, 128, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t219_bert_embeddings_Reshape_3_0 tensor
    unsigned g_0_t219_bert_embeddings_Reshape_3_0_max_sizes[] = {768, 128, 32};
    unsigned g_0_t219_bert_embeddings_Reshape_3_0_min_sizes[] = {768, 128, 32};
    unsigned g_0_t219_bert_embeddings_Reshape_3_0             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t219_bert_embeddings_Reshape_3_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t219_bert_embeddings_Reshape_3_0_max_sizes,
                                                                  3,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t219_bert_embeddings_Reshape_3_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    // create g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0 tensor
    unsigned g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_max_sizes[] = {768, 128, 32};
    unsigned g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_min_sizes[] = {768, 128, 32};
    unsigned g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_add_fwd_bf16_n49_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_t221_bert_embeddings_Reshape_1_0, g_0_t219_bert_embeddings_Reshape_3_0},
                   {g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0},
                   nullptr,
                   0,
                   "g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_add_fwd_bf16_n49_0",
                   0 /*graphIndex*/,
                   &g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_add_fwd_bf16_n49_0_id);

    /*************
     * g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n52_0 node
     * inputs:
     *     g_0_t228_bert_embeddings_Reshape_4_0[768, 128, 1] (dtype=bf16)
     *     g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0[768, 128, 32] (dtype=bf16)
     * outputs:
     *     g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0[768, 128, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t228_bert_embeddings_Reshape_4_0 tensor
    unsigned g_0_t228_bert_embeddings_Reshape_4_0_max_sizes[] = {768, 128, 1};
    unsigned g_0_t228_bert_embeddings_Reshape_4_0_min_sizes[] = {768, 128, 1};
    unsigned g_0_t228_bert_embeddings_Reshape_4_0             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t228_bert_embeddings_Reshape_4_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t228_bert_embeddings_Reshape_4_0_max_sizes,
                                                                  3,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t228_bert_embeddings_Reshape_4_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    // create g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0 tensor
    unsigned g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_max_sizes[] = {768, 128, 32};
    unsigned g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_min_sizes[] = {768, 128, 32};
    unsigned g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n52_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_t228_bert_embeddings_Reshape_4_0,
                    g_0_t223_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_Leaf_1_add_1_0},
                   {g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0},
                   nullptr,
                   0,
                   "g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n52_0",
                   0 /*graphIndex*/,
                   &g_0_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_add_fwd_bf16_n52_0_id);

    /*************
     * g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n53_0 node
     * inputs:
     *     g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0[768, 128, 32] (dtype=bf16)
     *     g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 128, 1, 32};
    unsigned g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 128, 1, 32};
    unsigned g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 128, 1, 32};
    unsigned g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 128, 1, 32};
    unsigned g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n53_0_id;
    addNodeToGraph("reshape",
                   {g_0_t230_bert_embeddings_ArithmeticOptimizer_AddOpsRewrite_add_1_0,
                    g_0_t235_bert_embeddings_LayerNorm_HabanaLayerNorm},
                   {g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm},
                   nullptr,
                   0,
                   "g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n53_0",
                   0 /*graphIndex*/,
                   &g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n53_0_id);

    /*************
     * g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n57_0 node
     * inputs:
     *     g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     *     g_0_t127_readvariableop_20_0[768] (dtype=float32)
     *     g_0_t126_readvariableop_16_0[768] (dtype=float32)
     * outputs:
     *     g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     *     g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm[1, 128, 1, 32] (dtype=float32)
     *     g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm[1, 128, 1, 32] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t127_readvariableop_20_0 tensor
    unsigned g_0_t127_readvariableop_20_0_max_sizes[] = {768};
    unsigned g_0_t127_readvariableop_20_0_min_sizes[] = {768};
    unsigned g_0_t127_readvariableop_20_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t127_readvariableop_20_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t127_readvariableop_20_0_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t127_readvariableop_20_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t126_readvariableop_16_0 tensor
    unsigned g_0_t126_readvariableop_16_0_max_sizes[] = {768};
    unsigned g_0_t126_readvariableop_16_0_min_sizes[] = {768};
    unsigned g_0_t126_readvariableop_16_0             = createTensors(1,
                                                          INPUT_TENSOR,
                                                          true,
                                                          "g_0_t126_readvariableop_16_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t126_readvariableop_16_0_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t126_readvariableop_16_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 128, 1, 32};
    unsigned g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 128, 1, 32};
    unsigned g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 128, 1, 32};
    unsigned g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 128, 1, 32};
    unsigned g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 128, 1, 32};
    unsigned g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 128, 1, 32};
    unsigned g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n57_0_id;
    unsigned char g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n57_0_params[] =
        {1, 0, 0, 0, 111, 18, 131, 58};
    addNodeToGraph("layer_norm_fwd_bf16",
                   {g_0_t234_bert_embeddings_LayerNorm_HabanaLayerNorm,
                    g_0_t127_readvariableop_20_0,
                    g_0_t126_readvariableop_16_0},
                   {g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm,
                    g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm,
                    g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm},
                   (void*)g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n57_0_params,
                   8,
                   "g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n57_0",
                   0 /*graphIndex*/,
                   &g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n57_0_id);

    /*************
     * g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n54_0 node
     * inputs:
     *     g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 1, 32] (dtype=bf16)
     *     g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm[768, 128, 32] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0[768, 128, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes[] = {768, 128, 32};
    unsigned g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes[] = {768, 128, 32};
    unsigned g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm_max_sizes,
                      3,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {768, 128, 32};
    unsigned g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {768, 128, 32};
    unsigned g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n54_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_t236_bert_embeddings_LayerNorm_HabanaLayerNorm, g_0_t237_bert_embeddings_LayerNorm_HabanaLayerNorm},
        {g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0},
        nullptr,
        0,
        "g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n54_0",
        0 /*graphIndex*/,
        &g_0_bert_embeddings_LayerNorm_HabanaLayerNorm_reshape_n54_0_id);

    /*************
     * g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n59_0 node
     * inputs:
     *     g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0[768, 128, 32] (dtype=bf16)
     *     g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0[1]
     *(dtype=int32) outputs: g_0_t246_bert_embeddings_dropout_Mul_1_0[768, 128, 32] (dtype=bf16)
     *     g_0_t247_bert_embeddings_dropout_Mul_1_0[768, 128, 32] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0
    // tensor
    unsigned
        g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_max_sizes
            [] = {1};
    unsigned
        g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_min_sizes
            [] = {1};
    unsigned g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_max_sizes,
            1,
            syn_type_int32,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_t246_bert_embeddings_dropout_Mul_1_0 tensor
    unsigned g_0_t246_bert_embeddings_dropout_Mul_1_0_max_sizes[] = {768, 128, 32};
    unsigned g_0_t246_bert_embeddings_dropout_Mul_1_0_min_sizes[] = {768, 128, 32};
    unsigned g_0_t246_bert_embeddings_dropout_Mul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t246_bert_embeddings_dropout_Mul_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t246_bert_embeddings_dropout_Mul_1_0_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t246_bert_embeddings_dropout_Mul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t247_bert_embeddings_dropout_Mul_1_0 tensor
    unsigned g_0_t247_bert_embeddings_dropout_Mul_1_0_max_sizes[] = {768, 128, 32};
    unsigned g_0_t247_bert_embeddings_dropout_Mul_1_0_min_sizes[] = {768, 128, 32};
    unsigned g_0_t247_bert_embeddings_dropout_Mul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t247_bert_embeddings_dropout_Mul_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t247_bert_embeddings_dropout_Mul_1_0_max_sizes,
                      3,
                      syn_type_int8,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t247_bert_embeddings_dropout_Mul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n59_0_id;
    unsigned char g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n59_0_params[] =
        {205, 204, 204, 61, 0, 0, 0, 0};
    addNodeToGraph(
        "dropout_fwd_bf16",
        {g_0_t231_bert_embeddings_LayerNorm_HabanaLayerNorm_0,
         g_0_t242_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_0_0},
        {g_0_t246_bert_embeddings_dropout_Mul_1_0, g_0_t247_bert_embeddings_dropout_Mul_1_0},
        (void*)g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n59_0_params,
        8,
        "g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n59_0",
        0 /*graphIndex*/,
        &g_0_bert_embeddings_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n59_0_id);

    /*************
     * g_0_bert_encoder_Reshape_1_reshape_n60_0 node
     * inputs:
     *     g_0_t246_bert_embeddings_dropout_Mul_1_0[768, 128, 32] (dtype=bf16)
     *     g_0_t249_bert_encoder_Reshape_1[768, 4096] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t248_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t249_bert_encoder_Reshape_1 tensor
    unsigned g_0_t249_bert_encoder_Reshape_1_max_sizes[] = {768, 4096};
    unsigned g_0_t249_bert_encoder_Reshape_1_min_sizes[] = {768, 4096};
    unsigned g_0_t249_bert_encoder_Reshape_1             = createTensors(1,
                                                             INPUT_TENSOR,
                                                             false,
                                                             "g_0_t249_bert_encoder_Reshape_1",
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             g_0_t249_bert_encoder_Reshape_1_max_sizes,
                                                             2,
                                                             syn_type_uint32,
                                                             nullptr,
                                                             0,
                                                             0,
                                                             nullptr,
                                                             false,
                                                             g_0_t249_bert_encoder_Reshape_1_min_sizes,
                                                             synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t248_bert_encoder_Reshape_1_0 tensor
    unsigned  g_0_t248_bert_encoder_Reshape_1_0_max_sizes[] = {768, 4096};
    unsigned  g_0_t248_bert_encoder_Reshape_1_0_min_sizes[] = {768, 4096};
    unsigned  g_0_t248_bert_encoder_Reshape_1_0             = createTensors(1,
                                                               OUTPUT_TENSOR,
                                                               false,
                                                               "g_0_t248_bert_encoder_Reshape_1_0",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               g_0_t248_bert_encoder_Reshape_1_0_max_sizes,
                                                               2,
                                                               syn_type_bf16,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_t248_bert_encoder_Reshape_1_0_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_Reshape_1_reshape_n60_0_id;
    addNodeToGraph("reshape",
                   {g_0_t246_bert_embeddings_dropout_Mul_1_0, g_0_t249_bert_encoder_Reshape_1},
                   {g_0_t248_bert_encoder_Reshape_1_0},
                   nullptr,
                   0,
                   "g_0_bert_encoder_Reshape_1_reshape_n60_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_Reshape_1_reshape_n60_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_query_MatMul_gemm_n61_0 node
     * inputs:
     *     g_0_t248_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     *     g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0[768, 768]
     *(dtype=bf16) outputs: g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0[768, 4096] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0 tensor
    unsigned g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0_max_sizes[] = {768, 4096};
    unsigned g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0_min_sizes[] = {768, 4096};
    unsigned g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_layer_0_attention_self_query_MatMul_gemm_n61_0_id;
    unsigned char g_0_bert_encoder_layer_0_attention_self_query_MatMul_gemm_n61_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t248_bert_encoder_Reshape_1_0,
                    g_0_t145_bert_encoder_layer_0_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_3_0},
                   {g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0},
                   (void*)g_0_bert_encoder_layer_0_attention_self_query_MatMul_gemm_n61_0_params,
                   2,
                   "g_0_bert_encoder_layer_0_attention_self_query_MatMul_gemm_n61_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_attention_self_query_MatMul_gemm_n61_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_key_MatMul_gemm_n66_0 node
     * inputs:
     *     g_0_t248_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     *     g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0[768, 768]
     *(dtype=bf16) outputs: g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0[768, 4096] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0 tensor
    unsigned g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0_max_sizes[] = {768, 4096};
    unsigned g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0_min_sizes[] = {768, 4096};
    unsigned g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_layer_0_attention_self_key_MatMul_gemm_n66_0_id;
    unsigned char g_0_bert_encoder_layer_0_attention_self_key_MatMul_gemm_n66_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t248_bert_encoder_Reshape_1_0,
                    g_0_t147_bert_encoder_layer_0_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_4_0},
                   {g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0},
                   (void*)g_0_bert_encoder_layer_0_attention_self_key_MatMul_gemm_n66_0_params,
                   2,
                   "g_0_bert_encoder_layer_0_attention_self_key_MatMul_gemm_n66_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_attention_self_key_MatMul_gemm_n66_0_id);

    /*************
     * g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n72_0 node
     * inputs:
     *     g_0_t248_bert_encoder_Reshape_1_0[768, 4096] (dtype=bf16)
     *     g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0[768, 768]
     *(dtype=bf16) outputs: g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0[768, 4096] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0 tensor
    unsigned g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0_max_sizes[] = {768, 4096};
    unsigned g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0_min_sizes[] = {768, 4096};
    unsigned g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n72_0_id;
    unsigned char g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n72_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t248_bert_encoder_Reshape_1_0,
                    g_0_t149_bert_encoder_layer_0_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_5_0},
                   {g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0},
                   (void*)g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n72_0_params,
                   2,
                   "g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n72_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_attention_self_value_MatMul_gemm_n72_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_t250_bert_encoder_layer_0_attention_self_query_MatMul_0,
                        g_0_t257_bert_encoder_layer_0_attention_self_key_MatMul_0,
                        g_0_t265_bert_encoder_layer_0_attention_self_value_MatMul_0,
                        g_0_t247_bert_embeddings_dropout_Mul_1_0,
                        g_0_t238_bert_embeddings_LayerNorm_HabanaLayerNorm,
                        g_0_t240_bert_embeddings_LayerNorm_HabanaLayerNorm});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, multi_bgemms_bundle_from_bert_ASIC)
{
    // Graph #0

    /*************
     * g_0_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_habana_dropout_grad_dropout_bwd_bf16_n618_0
     *node inputs: g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0[128, 128, 12, 32]
     *(dtype=bf16) g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0[128, 128, 12, 32] (dtype=int8) outputs:
     *     g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0[128, 128, 12, 32]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0 tensor
    unsigned g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0_max_sizes[] = {128,
                                                                                                             128,
                                                                                                             12,
                                                                                                             32};
    unsigned g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0_min_sizes[] = {128,
                                                                                                             128,
                                                                                                             12,
                                                                                                             32};
    unsigned g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0 tensor
    unsigned g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0_max_sizes[] = {128, 128, 12, 32};
    unsigned g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0_min_sizes[] = {128, 128, 12, 32};
    unsigned g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0_max_sizes,
                      4,
                      syn_type_int8,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0 tensor
    unsigned g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0_max_sizes[] = {128,
                                                                                                             128,
                                                                                                             12,
                                                                                                             32};
    unsigned g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0_min_sizes[] = {128,
                                                                                                             128,
                                                                                                             12,
                                                                                                             32};
    unsigned g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_habana_dropout_grad_dropout_bwd_bf16_n618_0_id;
    unsigned char
        g_0_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_habana_dropout_grad_dropout_bwd_bf16_n618_0_params
            [] = {205, 204, 204, 61, 0, 0, 0, 0};
    addNodeToGraph(
        "dropout_bwd_bf16",
        {g_0_t1106_gradients_1_bert_encoder_layer_0_attention_self_MatMul_1_grad_MatMul_0,
         g_0_t533_bert_encoder_layer_0_attention_self_dropout_Mul_1_0},
        {g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0},
        (void*)
            g_0_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_habana_dropout_grad_dropout_bwd_bf16_n618_0_params,
        8,
        "g_0_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_habana_dropout_grad_dropout_bwd_bf16_"
        "n618_0",
        0 /*graphIndex*/,
        &g_0_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_habana_dropout_grad_dropout_bwd_bf16_n618_0_id);

    /*************
     * g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_softmax_bwd_bf16_n619_0 node
     * inputs:
     *     g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0[128, 128, 12, 32] (dtype=bf16)
     *     g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0[128, 128, 12, 32]
     *(dtype=bf16) outputs:
     *     g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0[128, 128, 12,
     *32] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0 tensor
    unsigned g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0_max_sizes[] = {128, 128, 12, 32};
    unsigned g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0_min_sizes[] = {128, 128, 12, 32};
    unsigned g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0 tensor
    unsigned g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0_max_sizes[] =
        {128, 128, 12, 32};
    unsigned g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0_min_sizes[] =
        {128, 128, 12, 32};
    unsigned g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_softmax_bwd_bf16_n619_0_id;
    unsigned char
        g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_softmax_bwd_bf16_n619_0_params
            [] = {0, 0, 0, 0};
    addNodeToGraph(
        "softmax_bwd_bf16",
        {g_0_t531_bert_encoder_layer_0_attention_self_Softmax_0,
         g_0_t1107_gradients_1_bert_encoder_layer_0_attention_self_dropout_Mul_grad_Mul_0},
        {g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0},
        (void*)
            g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_softmax_bwd_bf16_n619_0_params,
        4,
        "g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_softmax_bwd_bf16_n619_0",
        0 /*graphIndex*/,
        &g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_softmax_bwd_bf16_n619_0_id);

    /*************
     * g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_mult_fwd_bf16_n673_0 node
     * inputs:
     *     g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A[1, 1, 1, 1]
     *(dtype=bf16) g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0[128,
     *128, 12, 32] (dtype=bf16) outputs: g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0[128,
     *128, 12, 32] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A tensor
    unsigned g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_max_sizes[] = {1,
                                                                                                                    1,
                                                                                                                    1,
                                                                                                                    1};
    unsigned g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_min_sizes[] = {1,
                                                                                                                    1,
                                                                                                                    1,
                                                                                                                    1};
    unsigned g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0 tensor
    unsigned g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0_max_sizes[] = {128, 128, 12, 32};
    unsigned g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0_min_sizes[] = {128, 128, 12, 32};
    unsigned g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_mult_fwd_bf16_n673_0_id;
    addNodeToGraph(
        "mult_fwd_bf16",
        {g_0_t1187_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A,
         g_0_t1108_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_habana_softmax_grad_0},
        {g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0},
        nullptr,
        0,
        "g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_mult_fwd_bf16_n673_0",
        0 /*graphIndex*/,
        &g_0_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_gradient_times_A_mult_fwd_bf16_n673_0_id);

    /*************
     * g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_batch_gemm_n674_0 node
     * inputs:
     *     g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0[128, 128, 12, 32] (dtype=bf16)
     *     g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0[64, 128, 12, 32] (dtype=bf16)
     * outputs:
     *     g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0[64, 128, 12, 32] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0 tensor
    unsigned g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0_max_sizes[] = {64, 128, 12, 32};
    unsigned g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0_min_sizes[] = {64, 128, 12, 32};
    unsigned g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0 tensor
    unsigned g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0_max_sizes[] = {64,
                                                                                                           128,
                                                                                                           12,
                                                                                                           32};
    unsigned g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0_min_sizes[] = {64,
                                                                                                           128,
                                                                                                           12,
                                                                                                           32};
    unsigned g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_batch_gemm_n674_0_id;
    unsigned char g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_batch_gemm_n674_0_params[] = {
        0,
        0};
    addNodeToGraph(
        "batch_gemm",
        {g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0,
         g_0_t263_bert_encoder_layer_0_attention_self_transpose_1_0},
        {g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0},
        (void*)g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_batch_gemm_n674_0_params,
        2,
        "g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_batch_gemm_n674_0",
        0 /*graphIndex*/,
        &g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_batch_gemm_n674_0_id);

    /*************
     * g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_batch_gemm_n694_0 node
     * inputs:
     *     g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0[128, 128, 12, 32] (dtype=bf16)
     *     g_0_t256_bert_encoder_layer_0_attention_self_transpose_0[64, 128, 12, 32] (dtype=bf16)
     * outputs:
     *     g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0[64, 128, 12, 32]
     *(dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t256_bert_encoder_layer_0_attention_self_transpose_0 tensor
    unsigned g_0_t256_bert_encoder_layer_0_attention_self_transpose_0_max_sizes[] = {64, 128, 12, 32};
    unsigned g_0_t256_bert_encoder_layer_0_attention_self_transpose_0_min_sizes[] = {64, 128, 12, 32};
    unsigned g_0_t256_bert_encoder_layer_0_attention_self_transpose_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t256_bert_encoder_layer_0_attention_self_transpose_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t256_bert_encoder_layer_0_attention_self_transpose_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t256_bert_encoder_layer_0_attention_self_transpose_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0 tensor
    unsigned g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0_max_sizes[] = {64,
                                                                                                             128,
                                                                                                             12,
                                                                                                             32};
    unsigned g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0_min_sizes[] = {64,
                                                                                                             128,
                                                                                                             12,
                                                                                                             32};
    unsigned g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_batch_gemm_n694_0_id;
    unsigned char g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_batch_gemm_n694_0_params[] =
        {1, 0};
    addNodeToGraph(
        "batch_gemm",
        {g_0_t1186_gradients_1_bert_encoder_layer_0_attention_self_Mul_grad_Mul_0,
         g_0_t256_bert_encoder_layer_0_attention_self_transpose_0},
        {g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0},
        (void*)g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_batch_gemm_n694_0_params,
        2,
        "g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_batch_gemm_n694_0",
        0 /*graphIndex*/,
        &g_0_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_batch_gemm_n694_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_t1189_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_0,
                        g_0_t1218_gradients_1_bert_encoder_layer_0_attention_self_MatMul_grad_MatMul_1_0});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_large_cd_ASIC)
{
    const unsigned commonDim = 9193, height = 512, width = 512;
    unsigned       aSizes[]   = {commonDim, height};
    unsigned       bSizes[]   = {width, commonDim};
    unsigned       outSizes[] = {width, height};

    unsigned a = createTensors(1,
                               INPUT_TENSOR,
                               true,
                               "A",
                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                               nullptr,
                               aSizes,
                               2,
                               syn_type_single,
                               nullptr,
                               0,
                               0,
                               nullptr,
                               false,
                               aSizes,
                               synTensorType::DATA_TENSOR)[0];

    unsigned b = createTensors(1,
                               INPUT_TENSOR,
                               true,
                               "B",
                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                               nullptr,
                               bSizes,
                               2,
                               syn_type_single,
                               nullptr,
                               0,
                               0,
                               nullptr,
                               false,
                               bSizes,
                               synTensorType::DATA_TENSOR)[0];

    unsigned out = createTensors(1,
                                 OUTPUT_TENSOR,
                                 true,
                                 "out",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 outSizes,
                                 2,
                                 syn_type_single,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 outSizes,
                                 synTensorType::DATA_TENSOR)[0];

    synGEMMParams gemmParams(false, false);
    addNodeToGraph("gemm", {a, b}, {out}, &gemmParams, sizeof(gemmParams), "GEMM");

    setConfigsForTest();
    compareRunsResults({out});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_large_cd_bf16_ASIC)
{
    const unsigned commonDim = 9193, height = 512, width = 512;
    unsigned       aSizes[]   = {commonDim, height};
    unsigned       bSizes[]   = {width, commonDim};
    unsigned       outSizes[] = {width, height};

    unsigned a = createTensors(1,
                               INPUT_TENSOR,
                               true,
                               "A",
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

    unsigned b = createTensors(1,
                               INPUT_TENSOR,
                               true,
                               "B",
                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                               nullptr,
                               bSizes,
                               2,
                               syn_type_bf16,
                               nullptr,
                               0,
                               0,
                               nullptr,
                               false,
                               bSizes,
                               synTensorType::DATA_TENSOR)[0];

    unsigned out = createTensors(1,
                                 OUTPUT_TENSOR,
                                 true,
                                 "out",
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
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

    synGEMMParams gemmParams(false, false);
    addNodeToGraph("gemm", {a, b}, {out}, &gemmParams, sizeof(gemmParams), "GEMM");

    setConfigsForTest();
    compareRunsResults({out});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, shared_input_gemms_bwd_from_bert_large_ASIC)
{
    // Graph #0

    /*************
     * g_0_gradients_1_AddN_gelu_bwd_bf16_n406_0 node
     * inputs:
     *     g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0[4096, 9216] (dtype=bf16)
     *     g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0[4096, 9216] (dtype=bf16)
     *     g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0[4096, 9216] (dtype=bf16)
     * outputs:
     *     g_0_t784_gradients_1_AddN_0[4096, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0 tensor
    unsigned g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0_max_sizes[] = {4096, 9216};
    unsigned g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0_min_sizes[] = {4096, 9216};
    unsigned g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "gelu_in0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0 tensor
    unsigned g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0_max_sizes[] = {4096, 9216};
    unsigned g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0_min_sizes[] = {4096, 9216};
    unsigned g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "gelu_in1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0 tensor
    unsigned g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0_max_sizes[] = {4096, 9216};
    unsigned g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0_min_sizes[] = {4096, 9216};
    unsigned g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "gelu_in2",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t784_gradients_1_AddN_0 tensor
    unsigned  g_0_t784_gradients_1_AddN_0_max_sizes[] = {4096, 9216};
    unsigned  g_0_t784_gradients_1_AddN_0_min_sizes[] = {4096, 9216};
    unsigned  g_0_t784_gradients_1_AddN_0             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "shared_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_t784_gradients_1_AddN_0_max_sizes,
                                                         2,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_t784_gradients_1_AddN_0_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_1_AddN_gelu_bwd_bf16_n406_0_id;
    addNodeToGraph("gelu_bwd_bf16",
                   {g_0_t783_gradients_1_bert_encoder_layer_1_output_dense_MatMul_grad_MatMul_0,
                    g_0_t629_bert_encoder_layer_1_intermediate_dense_BiasAdd_0,
                    g_0_t633_bert_encoder_layer_1_intermediate_dense_Tanh_0},
                   {g_0_t784_gradients_1_AddN_0},
                   nullptr,
                   0,
                   "gelu",
                   0 /*graphIndex*/,
                   &g_0_gradients_1_AddN_gelu_bwd_bf16_n406_0_id);

    /*************
     * g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_gemm_n407_0 node
     * inputs:
     *     g_0_t784_gradients_1_AddN_0[4096, 9216] (dtype=bf16)
     *     g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0[4096, 1024]
     *(dtype=bf16) outputs: g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0[1024,
     *9216] (dtype=bf16) ctrl inputs: ctrl outputs:
     *************/

    // create g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0 tensor
    unsigned g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0_max_sizes[] =
        {4096, 1024};
    unsigned g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0_min_sizes[] =
        {4096, 1024};
    unsigned g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "gemm1_in",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    // create g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0 tensor
    unsigned g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0_max_sizes[] = {1024,
                                                                                                              9216};
    unsigned g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0_min_sizes[] = {1024,
                                                                                                              9216};
    unsigned g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "gemm1_out",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_gemm_n407_0_id;
    unsigned char g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_gemm_n407_0_params[] = {0,
                                                                                                                     1};
    addNodeToGraph("gemm",
                   {g_0_t784_gradients_1_AddN_0,
                    g_0_t185_bert_encoder_layer_1_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_25_0},
                   {g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0},
                   (void*)g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_gemm_n407_0_params,
                   2,
                   "gemm1",
                   0 /*graphIndex*/,
                   &g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_gemm_n407_0_id);

    /*************
     * g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_gemm_n434_0 node
     * inputs:
     *     g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0[1024, 9216] (dtype=bf16)
     *     g_0_t784_gradients_1_AddN_0[4096, 9216] (dtype=bf16)
     * outputs:
     *     g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0[4096, 1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {1024, 9216};
    unsigned g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {1024, 9216};
    unsigned g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "gemm2_in",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0 tensor
    unsigned g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0_max_sizes[] = {4096,
                                                                                                                1024};
    unsigned g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0_min_sizes[] = {4096,
                                                                                                                1024};
    unsigned g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "gemm2_out",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_gemm_n434_0_id;
    unsigned char g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_gemm_n434_0_params[] = {
        1,
        0};
    addNodeToGraph(
        "gemm",
        {g_0_t617_bert_encoder_layer_1_attention_output_LayerNorm_HabanaLayerNorm_0, g_0_t784_gradients_1_AddN_0},
        {g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0},
        (void*)g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_gemm_n434_0_params,
        2,
        "gemm2",
        0 /*graphIndex*/,
        &g_0_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_gemm_n434_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_t785_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_0,
                        g_0_t825_gradients_1_bert_encoder_layer_1_intermediate_dense_MatMul_grad_MatMul_1_0});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, unaligned_dedw_output_bpt, {synDeviceGaudi3})
{
    // Graph #0

    /*************
     * g_0_memcpy_425_0 node
     * inputs:
     *     g_0_cross_entropy_loss0_logs_output_before_memcpy[1000, 64] (dtype=bf16)
     * outputs:
     *     g_0_cross_entropy_loss0_logs_output[1000, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_cross_entropy_loss0_logs_output_before_memcpy tensor
    unsigned g_0_cross_entropy_loss0_logs_output_before_memcpy_max_sizes[] = {1000, 64};
    unsigned g_0_cross_entropy_loss0_logs_output_before_memcpy_min_sizes[] = {1000, 64};
    unsigned g_0_cross_entropy_loss0_logs_output_before_memcpy =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_cross_entropy_loss0_logs_output_before_memcpy",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_cross_entropy_loss0_logs_output_before_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_cross_entropy_loss0_logs_output_before_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_cross_entropy_loss0_logs_output tensor
    unsigned  g_0_cross_entropy_loss0_logs_output_max_sizes[] = {1000, 64};
    unsigned  g_0_cross_entropy_loss0_logs_output_min_sizes[] = {1000, 64};
    unsigned  g_0_cross_entropy_loss0_logs_output             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_cross_entropy_loss0_logs_output",
                                                                 MEM_INIT_ALL_ZERO,
                                                                 nullptr,
                                                                 g_0_cross_entropy_loss0_logs_output_max_sizes,
                                                                 2,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_cross_entropy_loss0_logs_output_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_425_0_id;
    addNodeToGraph("memcpy",
                   {g_0_cross_entropy_loss0_logs_output_before_memcpy},
                   {g_0_cross_entropy_loss0_logs_output},
                   nullptr,
                   0,
                   "g_0_memcpy_425_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_425_0_id);

    /*************
     * g_0_cross_entropy_loss0_log_softmax_bwd_0 node
     * inputs:
     *     g_0_cross_entropy_loss0_logs_output[1000, 64] (dtype=bf16)
     *     g_0_target[64] (dtype=int32)
     * outputs:
     *     g_0_cross_entropy_loss0_grad_input[1000, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_target tensor
    unsigned g_0_target_max_sizes[] = {64};
    unsigned g_0_target_min_sizes[] = {64};
    unsigned g_0_target             = createTensors(1,
                                        INPUT_TENSOR,
                                        true,
                                        "g_0_target",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        g_0_target_max_sizes,
                                        1,
                                        syn_type_int32,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        g_0_target_min_sizes,
                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_cross_entropy_loss0_grad_input tensor
    unsigned      g_0_cross_entropy_loss0_grad_input_max_sizes[] = {1000, 64};
    unsigned      g_0_cross_entropy_loss0_grad_input_min_sizes[] = {1000, 64};
    unsigned      g_0_cross_entropy_loss0_grad_input             = createTensors(1,
                                                                OUTPUT_TENSOR,
                                                                false,
                                                                "g_0_cross_entropy_loss0_grad_input",
                                                                MEM_INIT_ALL_ZERO,
                                                                nullptr,
                                                                g_0_cross_entropy_loss0_grad_input_max_sizes,
                                                                2,
                                                                syn_type_bf16,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_cross_entropy_loss0_grad_input_min_sizes,
                                                                synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_cross_entropy_loss0_log_softmax_bwd_0_id;
    unsigned char g_0_cross_entropy_loss0_log_softmax_bwd_0_params[] = {1, 0, 0, 0, 64, 0, 0, 0};
    addNodeToGraph("softmax_cross_entropy_bwd_bf16",
                   {g_0_cross_entropy_loss0_logs_output, g_0_target},
                   {g_0_cross_entropy_loss0_grad_input},
                   (void*)g_0_cross_entropy_loss0_log_softmax_bwd_0_params,
                   8,
                   "g_0_cross_entropy_loss0_log_softmax_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_cross_entropy_loss0_log_softmax_bwd_0_id);

    /*************
     * g_0_worker_0_fc_dedw_reshape_0 node
     * inputs:
     *     g_0_cross_entropy_loss0_grad_input[1000, 64] (dtype=bf16)
     * outputs:
     *     g_0_worker_0_fc_dedw_tensor_reshape[1000, 1, 1, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_worker_0_fc_dedw_tensor_reshape tensor
    unsigned  g_0_worker_0_fc_dedw_tensor_reshape_max_sizes[] = {1000, 1, 1, 64};
    unsigned  g_0_worker_0_fc_dedw_tensor_reshape_min_sizes[] = {1000, 1, 1, 64};
    unsigned  g_0_worker_0_fc_dedw_tensor_reshape             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_worker_0_fc_dedw_tensor_reshape",
                                                                 MEM_INIT_ALL_ZERO,
                                                                 nullptr,
                                                                 g_0_worker_0_fc_dedw_tensor_reshape_max_sizes,
                                                                 4,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_worker_0_fc_dedw_tensor_reshape_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_worker_0_fc_dedw_reshape_0_id;
    addNodeToGraph("reshape",
                   {g_0_cross_entropy_loss0_grad_input},
                   {g_0_worker_0_fc_dedw_tensor_reshape},
                   nullptr,
                   0,
                   "g_0_worker_0_fc_dedw_reshape_0",
                   0 /*graphIndex*/,
                   &g_0_worker_0_fc_dedw_reshape_0_id);

    /*************
     * g_0_memcpy_424_0 node
     * inputs:
     *     g_0_cross_entropy_loss0_grad_input[1000, 64] (dtype=bf16)
     * outputs:
     *     g_0_cross_entropy_loss0_grad_input_memcpy[1000, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_cross_entropy_loss0_grad_input_memcpy tensor
    unsigned g_0_cross_entropy_loss0_grad_input_memcpy_max_sizes[] = {1000, 64};
    unsigned g_0_cross_entropy_loss0_grad_input_memcpy_min_sizes[] = {1000, 64};
    unsigned g_0_cross_entropy_loss0_grad_input_memcpy =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_cross_entropy_loss0_grad_input_memcpy",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_cross_entropy_loss0_grad_input_memcpy_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_cross_entropy_loss0_grad_input_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_424_0_id;
    addNodeToGraph("memcpy",
                   {g_0_cross_entropy_loss0_grad_input},
                   {g_0_cross_entropy_loss0_grad_input_memcpy},
                   nullptr,
                   0,
                   "g_0_memcpy_424_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_424_0_id);

    /*************
     * g_0_memcpy_426_0 node
     * inputs:
     *     g_0_worker_0_avgpool_output_before_memcpy[2048, 1, 1, 64] (dtype=bf16)
     * outputs:
     *     g_0_worker_0_avgpool_output[2048, 1, 1, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_worker_0_avgpool_output_before_memcpy tensor
    unsigned g_0_worker_0_avgpool_output_before_memcpy_max_sizes[] = {2048, 1, 1, 64};
    unsigned g_0_worker_0_avgpool_output_before_memcpy_min_sizes[] = {2048, 1, 1, 64};
    unsigned g_0_worker_0_avgpool_output_before_memcpy =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_worker_0_avgpool_output_before_memcpy",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_worker_0_avgpool_output_before_memcpy_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_worker_0_avgpool_output_before_memcpy_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_worker_0_avgpool_output tensor
    unsigned  g_0_worker_0_avgpool_output_max_sizes[] = {2048, 1, 1, 64};
    unsigned  g_0_worker_0_avgpool_output_min_sizes[] = {2048, 1, 1, 64};
    unsigned  g_0_worker_0_avgpool_output             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_worker_0_avgpool_output",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_worker_0_avgpool_output_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_worker_0_avgpool_output_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_426_0_id;
    addNodeToGraph("memcpy",
                   {g_0_worker_0_avgpool_output_before_memcpy},
                   {g_0_worker_0_avgpool_output},
                   nullptr,
                   0,
                   "g_0_memcpy_426_0",
                   0 /*graphIndex*/,
                   &g_0_memcpy_426_0_id);

    /*************
     * g_0_worker_0_fc_dedw_0 node
     * inputs:
     *     g_0_worker_0_fc_dedw_tensor_reshape[1000, 1, 1, 64] (dtype=bf16)
     *     g_0_worker_0_avgpool_output[2048, 1, 1, 64] (dtype=bf16)
     * outputs:
     *     g_0_worker_0_fc_weight_grad[1000, 2048, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_worker_0_fc_weight_grad tensor
    unsigned      g_0_worker_0_fc_weight_grad_max_sizes[] = {1000, 2048, 1, 1};
    unsigned      g_0_worker_0_fc_weight_grad_min_sizes[] = {1000, 2048, 1, 1};
    unsigned      g_0_worker_0_fc_weight_grad             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_worker_0_fc_weight_grad",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         g_0_worker_0_fc_weight_grad_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_worker_0_fc_weight_grad_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_worker_0_fc_dedw_0_id;
    unsigned char g_0_worker_0_fc_dedw_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_worker_0_fc_dedw_tensor_reshape, g_0_worker_0_avgpool_output},
                   {g_0_worker_0_fc_weight_grad},
                   (void*)g_0_worker_0_fc_dedw_0_params,
                   112,
                   "g_0_worker_0_fc_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_worker_0_fc_dedw_0_id);

    // create add0_in1 tensor
    unsigned add_in1_max_sizes[] = {1000, 2048, 1, 1};
    unsigned add_in1_min_sizes[] = {1000, 2048, 1, 1};
    unsigned add0_in1            = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "add0_in1",
                                      MEM_INIT_ALL_ZERO,  // Allows comparing inpersistent dedx output
                                      nullptr,
                                      add_in1_max_sizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      add_in1_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create add0_out tensor
    unsigned add_out_max_sizes[] = {1000, 2048, 1, 1};
    unsigned add_out_min_sizes[] = {1000, 2048, 1, 1};
    unsigned add0_out            = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "add0_output",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      add_out_max_sizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      add_out_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("add_fwd_f32",
                   {g_0_worker_0_fc_weight_grad, add0_in1},
                   {add0_out},
                   nullptr,
                   0,
                   "add0_fwd_f32",
                   0 /*graphIndex*/);

    // create add1_in1 tensor
    unsigned add1_in1 = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "add1_in1",
                                      MEM_INIT_ALL_ZERO,  // Allows comparing inpersistent dedx output
                                      nullptr,
                                      add_in1_max_sizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      add_in1_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create add1_out tensor
    unsigned add1_out = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "add1_output",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      add_out_max_sizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      add_out_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("add_fwd_f32",
                   {add1_in1, g_0_worker_0_fc_weight_grad},
                   {add1_out},
                   nullptr,
                   0,
                   "add1_fwd_f32",
                   0 /*graphIndex*/);

    addConfigurationToRun(FIRST_RUN, "ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE", "1");
    addConfigurationToRun(FIRST_RUN,
                          "MAX_RELATIVE_ALIGNMENT_INCREASE_RATIO",
                          std::to_string(std::numeric_limits<float>::max()));
    addConfigurationToRun(SECOND_RUN,
                          "ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE",
                          "0");  // fcd stride CL alignment disabled
    setConfigsForTest();

    // Dedw output g_0_worker_0_fc_weight_grad is a bpt and inpersistent and thus expected to be aligned
    // in the first run.
    // Two dummy add nodes consume the dedx output promising even if one of them will be bundled,
    // the other will not and dedw out will be a BPT.
    // Both adds add 0 to the dedx output writing the same data to a persistent (and thus comparable) tensor.
    compareRunsResults({g_0_cross_entropy_loss0_grad_input_memcpy, add0_out, add1_out});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, conv_with_inflate_for_utilization_ASIC)
{
    // Graph #0

    /*************
     * g_0_layer1_0_conv1_0 node
     * inputs:
     *     g_0_worker_0_maxpool_output[64, 56, 56, 64] (dtype=bf16)
     *     g_0_layer1_0_conv1_weight[64, 64, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_layer1_0_conv1_output[64, 56, 56, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_worker_0_maxpool_output tensor
    unsigned g_0_worker_0_maxpool_output_max_sizes[] = {64, 56, 56, 64};
    unsigned g_0_worker_0_maxpool_output_min_sizes[] = {64, 56, 56, 64};
    unsigned g_0_worker_0_maxpool_output             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_worker_0_maxpool_output",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
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

    // create g_0_layer1_0_conv1_weight tensor
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

    // create g_0_layer1_0_conv1_output tensor
    unsigned      g_0_layer1_0_conv1_output_max_sizes[] = {64, 56, 56, 64};
    unsigned      g_0_layer1_0_conv1_output_min_sizes[] = {64, 56, 56, 64};
    unsigned      g_0_layer1_0_conv1_output             = createTensors(1,
                                                       OUTPUT_TENSOR,
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
    synNodeId     g_0_layer1_0_conv1_0_id;
    unsigned char g_0_layer1_0_conv1_0_params[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_worker_0_maxpool_output, g_0_layer1_0_conv1_weight},
                   {g_0_layer1_0_conv1_output},
                   (void*)g_0_layer1_0_conv1_0_params,
                   112,
                   "g_0_layer1_0_conv1_0",
                   0 /*graphIndex*/,
                   &g_0_layer1_0_conv1_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_layer1_0_conv1_output});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, dedx_dedw_with_inflate_for_utilization_ASIC)
{
    // Graph #0

    /*************
     * g_0_layer4_2_conv2_dedx_0 node
     * inputs:
     *     g_0_layer4_2_bn2_grad_input[512, 7, 7, 64] (dtype=bf16)
     *     g_0_layer4_2_conv2_weight[512, 512, 3, 3] (dtype=bf16)
     * outputs:
     *     g_0_layer4_2_conv2_grad_input[512, 7, 7, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer4_2_bn2_grad_input tensor
    unsigned g_0_layer4_2_bn2_grad_input_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_bn2_grad_input_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_bn2_grad_input             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_layer4_2_bn2_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer4_2_bn2_grad_input_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer4_2_bn2_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer4_2_conv2_weight tensor
    unsigned g_0_layer4_2_conv2_weight_max_sizes[] = {512, 512, 3, 3};
    unsigned g_0_layer4_2_conv2_weight_min_sizes[] = {512, 512, 3, 3};
    unsigned g_0_layer4_2_conv2_weight             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer4_2_conv2_weight",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer4_2_conv2_weight_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer4_2_conv2_weight_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer4_2_conv2_grad_input tensor
    unsigned      g_0_layer4_2_conv2_grad_input_max_sizes[] = {512, 7, 7, 64};
    unsigned      g_0_layer4_2_conv2_grad_input_min_sizes[] = {512, 7, 7, 64};
    unsigned      g_0_layer4_2_conv2_grad_input             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           true,
                                                           "g_0_layer4_2_conv2_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer4_2_conv2_grad_input_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer4_2_conv2_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer4_2_conv2_dedx_0_id;
    unsigned char g_0_layer4_2_conv2_dedx_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_layer4_2_bn2_grad_input, g_0_layer4_2_conv2_weight},
                   {g_0_layer4_2_conv2_grad_input},
                   (void*)g_0_layer4_2_conv2_dedx_0_params,
                   112,
                   "g_0_layer4_2_conv2_dedx_0",
                   0 /*graphIndex*/,
                   &g_0_layer4_2_conv2_dedx_0_id);

    /*************
     * g_0_layer4_2_conv2_dedw_0 node
     * inputs:
     *     g_0_layer4_2_bn2_grad_input[512, 7, 7, 64] (dtype=bf16)
     *     g_0_layer4_2_relu1_output[512, 7, 7, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer4_2_conv2_weight_grad[512, 512, 3, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer4_2_relu1_output tensor
    unsigned g_0_layer4_2_relu1_output_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_relu1_output_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_relu1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer4_2_relu1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer4_2_relu1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer4_2_relu1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer4_2_conv2_weight_grad tensor
    unsigned      g_0_layer4_2_conv2_weight_grad_max_sizes[] = {512, 512, 3, 3};
    unsigned      g_0_layer4_2_conv2_weight_grad_min_sizes[] = {512, 512, 3, 3};
    unsigned      g_0_layer4_2_conv2_weight_grad             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            true,
                                                            "g_0_layer4_2_conv2_weight_grad",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_layer4_2_conv2_weight_grad_max_sizes,
                                                            4,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_layer4_2_conv2_weight_grad_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer4_2_conv2_dedw_0_id;
    unsigned char g_0_layer4_2_conv2_dedw_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_layer4_2_bn2_grad_input, g_0_layer4_2_relu1_output},
                   {g_0_layer4_2_conv2_weight_grad},
                   (void*)g_0_layer4_2_conv2_dedw_0_params,
                   112,
                   "g_0_layer4_2_conv2_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_layer4_2_conv2_dedw_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_layer4_2_conv2_grad_input, g_0_layer4_2_conv2_weight_grad});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, resnet_bwd_layer_ASIC)
{
    GlobalConfTestSetter conf("ENABLE_LB_DUPLICATE_SHARED_BUNDLE_INPUTS", "true");

    /*************
     * g_0_layer3_2_relu1_bwd_0 node
     * inputs:
     *     g_0_layer3_2_conv2_grad_input[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_2_relu1_output[256, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_2_relu1_grad_input[256, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_2_conv2_grad_input tensor
    unsigned g_0_layer3_2_conv2_grad_input_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_conv2_grad_input_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_conv2_grad_input             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer3_2_conv2_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer3_2_conv2_grad_input_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer3_2_conv2_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_relu1_output tensor
    unsigned g_0_layer3_2_relu1_output_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_relu1_output_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_relu1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_2_relu1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_2_relu1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_2_relu1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_relu1_grad_input tensor
    unsigned  g_0_layer3_2_relu1_grad_input_max_sizes[] = {256, 14, 14, 64};
    unsigned  g_0_layer3_2_relu1_grad_input_min_sizes[] = {256, 14, 14, 64};
    unsigned  g_0_layer3_2_relu1_grad_input             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_layer3_2_relu1_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer3_2_relu1_grad_input_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer3_2_relu1_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_2_relu1_bwd_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_layer3_2_conv2_grad_input, g_0_layer3_2_relu1_output},
                   {g_0_layer3_2_relu1_grad_input},
                   nullptr,
                   0,
                   "g_0_layer3_2_relu1_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_2_relu1_bwd_0_id);

    /*************
     * g_0_layer3_2_bn1_bwd_0 node
     * inputs:
     *     g_0_layer3_2_conv1_output[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_2_relu1_grad_input[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_2_bn1_saved_mean[256] (dtype=float32)
     *     g_0_layer3_2_bn1_saved_var[256] (dtype=float32)
     *     g_0_layer3_2_bn1_weight[256] (dtype=float32)
     * outputs:
     *     g_0_layer3_2_bn1_grad_input[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_2_bn1_bias_grad[256] (dtype=float32)
     *     g_0_layer3_2_bn1_weight_grad[256] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_2_conv1_output tensor
    unsigned g_0_layer3_2_conv1_output_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_conv1_output_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_conv1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_2_conv1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_2_conv1_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_2_conv1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_bn1_saved_mean tensor
    unsigned g_0_layer3_2_bn1_saved_mean_max_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_saved_mean_min_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_saved_mean             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_layer3_2_bn1_saved_mean",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_2_bn1_saved_mean_max_sizes,
                                                         1,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_2_bn1_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_bn1_saved_var tensor
    unsigned g_0_layer3_2_bn1_saved_var_max_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_saved_var_min_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_saved_var             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_layer3_2_bn1_saved_var",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer3_2_bn1_saved_var_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer3_2_bn1_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_bn1_weight tensor
    unsigned g_0_layer3_2_bn1_weight_max_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_weight_min_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer3_2_bn1_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer3_2_bn1_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer3_2_bn1_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_bn1_grad_input tensor
    unsigned g_0_layer3_2_bn1_grad_input_max_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_bn1_grad_input_min_sizes[] = {256, 14, 14, 64};
    unsigned g_0_layer3_2_bn1_grad_input             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_layer3_2_bn1_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_2_bn1_grad_input_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_2_bn1_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_bn1_bias_grad tensor
    unsigned g_0_layer3_2_bn1_bias_grad_max_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_bias_grad_min_sizes[] = {256};
    unsigned g_0_layer3_2_bn1_bias_grad             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer3_2_bn1_bias_grad",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer3_2_bn1_bias_grad_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer3_2_bn1_bias_grad_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_bn1_weight_grad tensor
    unsigned  g_0_layer3_2_bn1_weight_grad_max_sizes[] = {256};
    unsigned  g_0_layer3_2_bn1_weight_grad_min_sizes[] = {256};
    unsigned  g_0_layer3_2_bn1_weight_grad             = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          true,
                                                          "g_0_layer3_2_bn1_weight_grad",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer3_2_bn1_weight_grad_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer3_2_bn1_weight_grad_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_2_bn1_bwd_0_id;
    addNodeToGraph("batch_norm_bwd_bf16",
                   {g_0_layer3_2_conv1_output,
                    g_0_layer3_2_relu1_grad_input,
                    g_0_layer3_2_bn1_saved_mean,
                    g_0_layer3_2_bn1_saved_var,
                    g_0_layer3_2_bn1_weight},
                   {g_0_layer3_2_bn1_grad_input, g_0_layer3_2_bn1_bias_grad, g_0_layer3_2_bn1_weight_grad},
                   nullptr,
                   0,
                   "g_0_layer3_2_bn1_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_2_bn1_bwd_0_id);

    /*************
     * g_0_layer3_2_conv1_dedx_0 node
     * inputs:
     *     g_0_layer3_2_bn1_grad_input[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_2_conv1_weight[256, 1024, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_layer3_2_conv1_grad_input[1024, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_2_conv1_weight tensor
    unsigned g_0_layer3_2_conv1_weight_max_sizes[] = {256, 1024, 1, 1};
    unsigned g_0_layer3_2_conv1_weight_min_sizes[] = {256, 1024, 1, 1};
    unsigned g_0_layer3_2_conv1_weight             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_2_conv1_weight",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_2_conv1_weight_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_2_conv1_weight_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_conv1_grad_input tensor
    unsigned      g_0_layer3_2_conv1_grad_input_max_sizes[] = {1024, 14, 14, 64};
    unsigned      g_0_layer3_2_conv1_grad_input_min_sizes[] = {1024, 14, 14, 64};
    unsigned      g_0_layer3_2_conv1_grad_input             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_layer3_2_conv1_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer3_2_conv1_grad_input_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer3_2_conv1_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer3_2_conv1_dedx_0_id;
    unsigned char g_0_layer3_2_conv1_dedx_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_layer3_2_bn1_grad_input, g_0_layer3_2_conv1_weight},
                   {g_0_layer3_2_conv1_grad_input},
                   (void*)g_0_layer3_2_conv1_dedx_0_params,
                   112,
                   "g_0_layer3_2_conv1_dedx_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_2_conv1_dedx_0_id);

    /*************
     * g_0_layer3_2_conv1_dedw_0 node
     * inputs:
     *     g_0_layer3_2_bn1_grad_input[256, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_1_relu3_output[1024, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_2_conv1_weight_grad[256, 1024, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_1_relu3_output tensor
    unsigned g_0_layer3_1_relu3_output_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_relu3_output_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_relu3_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_1_relu3_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_1_relu3_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_1_relu3_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_conv1_weight_grad tensor
    unsigned      g_0_layer3_2_conv1_weight_grad_max_sizes[] = {256, 1024, 1, 1};
    unsigned      g_0_layer3_2_conv1_weight_grad_min_sizes[] = {256, 1024, 1, 1};
    unsigned      g_0_layer3_2_conv1_weight_grad             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            true,
                                                            "g_0_layer3_2_conv1_weight_grad",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_layer3_2_conv1_weight_grad_max_sizes,
                                                            4,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_layer3_2_conv1_weight_grad_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer3_2_conv1_dedw_0_id;
    unsigned char g_0_layer3_2_conv1_dedw_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_layer3_2_bn1_grad_input, g_0_layer3_1_relu3_output},
                   {g_0_layer3_2_conv1_weight_grad},
                   (void*)g_0_layer3_2_conv1_dedw_0_params,
                   112,
                   "g_0_layer3_2_conv1_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_2_conv1_dedw_0_id);

    /*************
     * g_0_layer3_2_add_residual_fwd1_0 node
     * inputs:
     *     g_0_layer3_2_conv1_grad_input[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_2_add_residual_grad_input1[1024, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_2_residual_upstream_grad_input[1024, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_2_add_residual_grad_input1 tensor
    unsigned g_0_layer3_2_add_residual_grad_input1_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_2_add_residual_grad_input1_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_2_add_residual_grad_input1             = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_layer3_2_add_residual_grad_input1",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_layer3_2_add_residual_grad_input1_max_sizes,
                                                                   4,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_layer3_2_add_residual_grad_input1_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_2_residual_upstream_grad_input tensor
    unsigned g_0_layer3_2_residual_upstream_grad_input_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_2_residual_upstream_grad_input_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_2_residual_upstream_grad_input =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_layer3_2_residual_upstream_grad_input",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_layer3_2_residual_upstream_grad_input_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_layer3_2_residual_upstream_grad_input_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_2_add_residual_fwd1_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_layer3_2_conv1_grad_input, g_0_layer3_2_add_residual_grad_input1},
                   {g_0_layer3_2_residual_upstream_grad_input},
                   nullptr,
                   0,
                   "g_0_layer3_2_add_residual_fwd1_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_2_add_residual_fwd1_0_id);

    /*************
     * g_0_layer3_1_relu3_bwd_0 node
     * inputs:
     *     g_0_layer3_2_residual_upstream_grad_input[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_1_relu3_output[1024, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_1_relu3_grad_input[1024, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_1_relu3_grad_input tensor
    unsigned  g_0_layer3_1_relu3_grad_input_max_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_1_relu3_grad_input_min_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_1_relu3_grad_input             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_layer3_1_relu3_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer3_1_relu3_grad_input_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer3_1_relu3_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_1_relu3_bwd_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_layer3_2_residual_upstream_grad_input, g_0_layer3_1_relu3_output},
                   {g_0_layer3_1_relu3_grad_input},
                   nullptr,
                   0,
                   "g_0_layer3_1_relu3_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_1_relu3_bwd_0_id);

    /*************
     * g_0_layer3_1_add_residual_bwd_0 node
     * inputs:
     *     g_0_layer3_1_relu3_grad_input[1024, 14, 14, 64] (dtype=bf16)
     * outputs:
     *     g_0_layer3_1_add_residual_grad_input0[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_1_add_residual_grad_input1[1024, 14, 14, 64] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_1_add_residual_grad_input0 tensor
    unsigned g_0_layer3_1_add_residual_grad_input0_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_add_residual_grad_input0_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_add_residual_grad_input0             = createTensors(1,
                                                                   OUTPUT_TENSOR,
                                                                   false,
                                                                   "g_0_layer3_1_add_residual_grad_input0",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_layer3_1_add_residual_grad_input0_max_sizes,
                                                                   4,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_layer3_1_add_residual_grad_input0_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_1_add_residual_grad_input1 tensor
    unsigned  g_0_layer3_1_add_residual_grad_input1_max_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_1_add_residual_grad_input1_min_sizes[] = {1024, 14, 14, 64};
    unsigned  g_0_layer3_1_add_residual_grad_input1             = createTensors(1,
                                                                   OUTPUT_TENSOR,
                                                                   true,
                                                                   "g_0_layer3_1_add_residual_grad_input1",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_layer3_1_add_residual_grad_input1_max_sizes,
                                                                   4,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_layer3_1_add_residual_grad_input1_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_1_add_residual_bwd_0_id;
    addNodeToGraph("add_bwd_bf16",
                   {g_0_layer3_1_relu3_grad_input},
                   {g_0_layer3_1_add_residual_grad_input0, g_0_layer3_1_add_residual_grad_input1},
                   nullptr,
                   0,
                   "g_0_layer3_1_add_residual_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_1_add_residual_bwd_0_id);

    /*************
     * g_0_layer3_1_bn3_bwd_0 node
     * inputs:
     *     g_0_layer3_1_conv3_output[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_1_add_residual_grad_input0[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_1_bn3_saved_mean[1024] (dtype=float32)
     *     g_0_layer3_1_bn3_saved_var[1024] (dtype=float32)
     *     g_0_layer3_1_bn3_weight[1024] (dtype=float32)
     * outputs:
     *     g_0_layer3_1_bn3_grad_input[1024, 14, 14, 64] (dtype=bf16)
     *     g_0_layer3_1_bn3_bias_grad[1024] (dtype=float32)
     *     g_0_layer3_1_bn3_weight_grad[1024] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_layer3_1_conv3_output tensor
    unsigned g_0_layer3_1_conv3_output_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_conv3_output_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_conv3_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer3_1_conv3_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer3_1_conv3_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer3_1_conv3_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_1_bn3_saved_mean tensor
    unsigned g_0_layer3_1_bn3_saved_mean_max_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_saved_mean_min_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_saved_mean             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_layer3_1_bn3_saved_mean",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_1_bn3_saved_mean_max_sizes,
                                                         1,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_1_bn3_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_1_bn3_saved_var tensor
    unsigned g_0_layer3_1_bn3_saved_var_max_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_saved_var_min_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_saved_var             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_layer3_1_bn3_saved_var",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer3_1_bn3_saved_var_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer3_1_bn3_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_1_bn3_weight tensor
    unsigned g_0_layer3_1_bn3_weight_max_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_weight_min_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer3_1_bn3_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer3_1_bn3_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer3_1_bn3_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_1_bn3_grad_input tensor
    unsigned g_0_layer3_1_bn3_grad_input_max_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_bn3_grad_input_min_sizes[] = {1024, 14, 14, 64};
    unsigned g_0_layer3_1_bn3_grad_input             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer3_1_bn3_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer3_1_bn3_grad_input_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer3_1_bn3_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_1_bn3_bias_grad tensor
    unsigned g_0_layer3_1_bn3_bias_grad_max_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_bias_grad_min_sizes[] = {1024};
    unsigned g_0_layer3_1_bn3_bias_grad             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer3_1_bn3_bias_grad",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer3_1_bn3_bias_grad_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer3_1_bn3_bias_grad_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_layer3_1_bn3_weight_grad tensor
    unsigned  g_0_layer3_1_bn3_weight_grad_max_sizes[] = {1024};
    unsigned  g_0_layer3_1_bn3_weight_grad_min_sizes[] = {1024};
    unsigned  g_0_layer3_1_bn3_weight_grad             = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          true,
                                                          "g_0_layer3_1_bn3_weight_grad",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer3_1_bn3_weight_grad_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer3_1_bn3_weight_grad_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer3_1_bn3_bwd_0_id;
    addNodeToGraph("batch_norm_bwd_bf16",
                   {g_0_layer3_1_conv3_output,
                    g_0_layer3_1_add_residual_grad_input0,
                    g_0_layer3_1_bn3_saved_mean,
                    g_0_layer3_1_bn3_saved_var,
                    g_0_layer3_1_bn3_weight},
                   {g_0_layer3_1_bn3_grad_input, g_0_layer3_1_bn3_bias_grad, g_0_layer3_1_bn3_weight_grad},
                   nullptr,
                   0,
                   "g_0_layer3_1_bn3_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer3_1_bn3_bwd_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_layer3_2_bn1_bias_grad,
                        g_0_layer3_2_bn1_weight_grad,
                        g_0_layer3_2_conv1_weight_grad,
                        g_0_layer3_1_bn3_grad_input,
                        g_0_layer3_1_bn3_bias_grad,
                        g_0_layer3_1_bn3_weight_grad});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, gemm_with_tpc_producers_and_consumer_uneven_dcore_split_ASIC)
{
    const unsigned commonDim = 256, height = 4597, width = 2143;
    unsigned       aSizes[]   = {commonDim, height};
    unsigned       bSizes[]   = {width, commonDim};
    unsigned       outSizes[] = {width, height};

    unsigned reluAIn = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "reluAIn",
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

    unsigned gemmInA = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInA",
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

    addNodeToGraph("relu_fwd_bf16", {reluAIn}, {gemmInA}, nullptr, 0, "reluProducerA");

    unsigned reluBIn = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "reluBIn",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned gemmInB = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmInB",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bSizes,
                                     2,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     bSizes,
                                     synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("relu_fwd_bf16", {reluBIn}, {gemmInB}, nullptr, 0, "reluProducerB");

    unsigned gemmOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmOut",
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

    synGEMMParams params(false, false);
    addNodeToGraph("gemm", {gemmInA, gemmInB}, {gemmOut}, &params, sizeof(params), "GEMM");

    unsigned reluOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "reluOut",
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

    addNodeToGraph("relu_fwd_bf16", {gemmOut}, {reluOut}, nullptr, 0, "reluConsumer");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");  // to avoid reshapes
    setConfigsForTest();
    compareRunsResults({reluOut});
}

TEST_F_GC(SynTrainingLayeredBrainAccuracyTest, attention_bundle_from_bert_ASIC)
{
    /*************
     * g_0_bert_encoder_0_attention_self_batch_gemm_674_0 node
     * inputs:
     *     g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute[64, 512, 16, 28] (dtype=bf16)
     *     g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute[512, 64, 16, 28] (dtype=bf16)
     * outputs:
     *     g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute tensor
    unsigned g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute_max_sizes[] = {64, 512, 16, 28};
    unsigned g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute_min_sizes[] = {64, 512, 16, 28};
    unsigned g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute tensor
    unsigned g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute_max_sizes[] = {512, 64, 16, 28};
    unsigned g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute_min_sizes[] = {512, 64, 16, 28};
    unsigned g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul tensor
    unsigned g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul_max_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul_min_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_0_attention_self_batch_gemm_674_0_id;
    unsigned char g_0_bert_encoder_0_attention_self_batch_gemm_674_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_146_id_989_bert_encoder_0_attention_self_aten__permute,
                    g_0_tensor_137_id_993_bert_encoder_0_attention_self_aten__permute},
                   {g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul},
                   (void*)g_0_bert_encoder_0_attention_self_batch_gemm_674_0_params,
                   2,
                   "g_0_bert_encoder_0_attention_self_batch_gemm_674_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_0_attention_self_batch_gemm_674_0_id);

    /*************
     * g_0_bert_encoder_0_attention_self_cast_f32_to_bf16_675_0 node
     * inputs:
     *     g_0_tensor_148__placeholder_1[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_150[1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_148__placeholder_1 tensor
    unsigned g_0_tensor_148__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_148__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_148__placeholder_1             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_tensor_148__placeholder_1",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_tensor_148__placeholder_1_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_tensor_148__placeholder_1_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_150 tensor
    unsigned      g_0_tensor_150_max_sizes[] = {1};
    unsigned      g_0_tensor_150_min_sizes[] = {1};
    unsigned      g_0_tensor_150             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_150",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_150_max_sizes,
                                            1,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_150_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_0_attention_self_cast_f32_to_bf16_675_0_id;
    unsigned char g_0_bert_encoder_0_attention_self_cast_f32_to_bf16_675_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_148__placeholder_1},
                   {g_0_tensor_150},
                   (void*)g_0_bert_encoder_0_attention_self_cast_f32_to_bf16_675_0_params,
                   4,
                   "g_0_bert_encoder_0_attention_self_cast_f32_to_bf16_675_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_0_attention_self_cast_f32_to_bf16_675_0_id);

    /*************
     * g_0_bert_encoder_0_attention_self_mult_fwd_bf16_676_0 node
     * inputs:
     *     g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul[512, 512, 16, 28] (dtype=bf16)
     *     g_0_tensor_150[1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul tensor
    unsigned g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul_max_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul_min_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_0_attention_self_mult_fwd_bf16_676_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_147_id_999_bert_encoder_0_attention_self_aten__matmul, g_0_tensor_150},
                   {g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul},
                   nullptr,
                   0,
                   "g_0_bert_encoder_0_attention_self_mult_fwd_bf16_676_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_0_attention_self_mult_fwd_bf16_676_0_id);

    /*************
     * g_0_bert_encoder_0_attention_self_add_fwd_bf16_677_0 node
     * inputs:
     *     g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul[512, 512, 16, 28] (dtype=bf16)
     *     g_0_tensor_127_id_907_bert_aten__mul[512, 512, 1, 28] (dtype=bf16)
     * outputs:
     *     g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_127_id_907_bert_aten__mul tensor
    unsigned g_0_tensor_127_id_907_bert_aten__mul_max_sizes[] = {512, 512, 1, 28};
    unsigned g_0_tensor_127_id_907_bert_aten__mul_min_sizes[] = {512, 512, 1, 28};
    unsigned g_0_tensor_127_id_907_bert_aten__mul             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_tensor_127_id_907_bert_aten__mul",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_tensor_127_id_907_bert_aten__mul_max_sizes,
                                                                  4,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_tensor_127_id_907_bert_aten__mul_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add tensor
    unsigned g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add_max_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add_min_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_0_attention_self_add_fwd_bf16_677_0_id;
    addNodeToGraph(
        "add_fwd_bf16",
        {g_0_tensor_149_id_1004_bert_encoder_0_attention_self_aten__mul, g_0_tensor_127_id_907_bert_aten__mul},
        {g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add},
        nullptr,
        0,
        "g_0_bert_encoder_0_attention_self_add_fwd_bf16_677_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_0_attention_self_add_fwd_bf16_677_0_id);

    /*************
     * g_0_bert_encoder_0_attention_self_softmax_fwd_bf16_678_0 node
     * inputs:
     *     g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add[512, 512, 16, 28] (dtype=bf16)
     * outputs:
     *     g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax[512, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax tensor
    unsigned g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax_max_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax_min_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_0_attention_self_softmax_fwd_bf16_678_0_id;
    unsigned char g_0_bert_encoder_0_attention_self_softmax_fwd_bf16_678_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("softmax_fwd_bf16",
                   {g_0_tensor_151_id_1006_bert_encoder_0_attention_self_aten__add},
                   {g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax},
                   (void*)g_0_bert_encoder_0_attention_self_softmax_fwd_bf16_678_0_params,
                   4,
                   "g_0_bert_encoder_0_attention_self_softmax_fwd_bf16_678_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_0_attention_self_softmax_fwd_bf16_678_0_id);

    /*************
     * g_0_bert_encoder_0_attention_self_dropout_dropout_fwd_bf16_679_0 node
     * inputs:
     *     g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax[512, 512, 16, 28] (dtype=bf16)
     *     g_0_tensor_153__placeholder_1[1] (dtype=int32)
     * outputs:
     *     g_0_tensor_156[512, 512, 16, 28] (dtype=bf16)
     *     g_0_tensor_157[512, 512, 16, 28] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_153__placeholder_1 tensor
    unsigned g_0_tensor_153__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_153__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_153__placeholder_1             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_tensor_153__placeholder_1",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_tensor_153__placeholder_1_max_sizes,
                                                           1,
                                                           syn_type_int32,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_tensor_153__placeholder_1_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_156 tensor
    unsigned g_0_tensor_156_max_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_156_min_sizes[] = {512, 512, 16, 28};
    unsigned g_0_tensor_156             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_156",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_156_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_156_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_157 tensor
    unsigned      g_0_tensor_157_max_sizes[] = {512, 512, 16, 28};
    unsigned      g_0_tensor_157_min_sizes[] = {512, 512, 16, 28};
    unsigned      g_0_tensor_157             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_157",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_157_max_sizes,
                                            4,
                                            syn_type_int8,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_157_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_0_attention_self_dropout_dropout_fwd_bf16_679_0_id;
    unsigned char g_0_bert_encoder_0_attention_self_dropout_dropout_fwd_bf16_679_0_params[] =
        {205, 204, 204, 61, 0, 0, 0, 0};
    addNodeToGraph("dropout_fwd_bf16",
                   {g_0_tensor_152_id_1009_bert_encoder_0_attention_self_aten___softmax, g_0_tensor_153__placeholder_1},
                   {g_0_tensor_156, g_0_tensor_157},
                   (void*)g_0_bert_encoder_0_attention_self_dropout_dropout_fwd_bf16_679_0_params,
                   8,
                   "g_0_bert_encoder_0_attention_self_dropout_dropout_fwd_bf16_679_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_0_attention_self_dropout_dropout_fwd_bf16_679_0_id);

    /*************
     * g_0_bert_encoder_0_attention_self_batch_gemm_680_0 node
     * inputs:
     *     g_0_tensor_156[512, 512, 16, 28] (dtype=bf16)
     *     g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute[64, 512, 16, 28] (dtype=bf16)
     * outputs:
     *     g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul[64, 512, 16, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute tensor
    unsigned g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute_max_sizes[] = {64, 512, 16, 28};
    unsigned g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute_min_sizes[] = {64, 512, 16, 28};
    unsigned g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul tensor
    unsigned g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul_max_sizes[] = {64, 512, 16, 28};
    unsigned g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul_min_sizes[] = {64, 512, 16, 28};
    unsigned g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_0_attention_self_batch_gemm_680_0_id;
    unsigned char g_0_bert_encoder_0_attention_self_batch_gemm_680_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_156, g_0_tensor_73_id_997_bert_encoder_0_attention_self_aten__permute},
                   {g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul},
                   (void*)g_0_bert_encoder_0_attention_self_batch_gemm_680_0_params,
                   2,
                   "g_0_bert_encoder_0_attention_self_batch_gemm_680_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_0_attention_self_batch_gemm_680_0_id);

    setConfigsForTest();
    compareRunsResults({g_0_tensor_158_id_1023_bert_encoder_0_attention_self_aten__matmul});
}