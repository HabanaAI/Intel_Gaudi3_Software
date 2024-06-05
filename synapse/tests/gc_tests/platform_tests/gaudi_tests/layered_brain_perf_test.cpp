////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                    //
// Tests in this file were auto generated from log snippets of topologies in order to reproduce compilation issues    //
// and check performance of specific bundles.                                                                         //
//                                                                                                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gc_gaudi_test_infra.h"
#include "scoped_configuration_change.h"

class SynGaudiLayeredBrainPerfTest_ASIC_CI : public SynTrainingTestInfra
{
protected:
    double getDeviceTimeMicro() const
    {
        static constexpr double NANO_TO_MICRO = 1e-3;
        return m_lastLaunchElapsedTime * NANO_TO_MICRO;
    }
};

// RAII style configuration to enable all the flags needed to run with layered brain
struct LayeredBrainEnableCfg
{
    ScopedConfigurationChange m_layeredBrainConf {"ENABLE_LAYERED_PIPELINE_BRAIN", "true"};
    ScopedConfigurationChange m_memoryManagementConf {"ENABLE_BUNDLE_MEMORY_MANAGEMENT", "true"};
    ScopedConfigurationChange m_forceConcatConf {"ENABLE_HBM_SLICES_ALLOCATION_OPTIMIZATION", "false"};

    virtual ~LayeredBrainEnableCfg() = default;
};

// clang-format off

TEST_F_GC(SynGaudiLayeredBrainPerfTest_ASIC_CI, dropout_layernorm_kqv)
{
    LayeredBrainEnableCfg lbCfg {}; // Enable layered brain GCFGs

    // Graph #0

    /*************
     * bert_encoder_layer_0_output_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n476_0 node
     * inputs:
     *     t1328_bert_encoder_layer_0_output_dense_BiasAdd_0[1024, 14336] (dtype=bf16)
     *     t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0[1] (dtype=int32)
     * outputs:
     *     t1331_bert_encoder_layer_0_output_dropout_Mul_1_0[1024, 14336] (dtype=bf16)
     *     t1332_bert_encoder_layer_0_output_dropout_Mul_1_0[1024, 14336] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1328_bert_encoder_layer_0_output_dense_BiasAdd_0 tensor
    unsigned t1328_bert_encoder_layer_0_output_dense_BiasAdd_0_max_sizes[] = {1024,14336};
    unsigned t1328_bert_encoder_layer_0_output_dense_BiasAdd_0_min_sizes[] = {1024,14336};
    unsigned t1328_bert_encoder_layer_0_output_dense_BiasAdd_0 = createTensors(1,
                                                                           INPUT_TENSOR,
                                                                           true,
                                                                           "t1328_bert_encoder_layer_0_output_dense_BiasAdd_0",
                                                                           MEM_INIT_ALL_ZERO,
                                                                           nullptr,
                                                                           t1328_bert_encoder_layer_0_output_dense_BiasAdd_0_max_sizes,
                                                                           2,
                                                                           syn_type_bf16,
                                                                           nullptr,
                                                                           0,
                                                                           0,
                                                                           nullptr,
                                                                           false,
                                                                           t1328_bert_encoder_layer_0_output_dense_BiasAdd_0_min_sizes,
                                                                           synTensorType::DATA_TENSOR)[0];

    // create t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0 tensor
    unsigned t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0_max_sizes[] = {1};
    unsigned t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0_min_sizes[] = {1};
    unsigned t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0 = createTensors(1,
                                                                                                                                INPUT_TENSOR,
                                                                                                                                true,
                                                                                                                                "t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0",
                                                                                                                                MEM_INIT_ALL_ZERO,
                                                                                                                                nullptr,
                                                                                                                                t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0_max_sizes,
                                                                                                                                1,
                                                                                                                                syn_type_int32,
                                                                                                                                nullptr,
                                                                                                                                0,
                                                                                                                                0,
                                                                                                                                nullptr,
                                                                                                                                false,
                                                                                                                                t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0_min_sizes,
                                                                                                                                synTensorType::DATA_TENSOR)[0];

    // create t1331_bert_encoder_layer_0_output_dropout_Mul_1_0 tensor
    unsigned t1331_bert_encoder_layer_0_output_dropout_Mul_1_0_max_sizes[] = {1024,14336};
    unsigned t1331_bert_encoder_layer_0_output_dropout_Mul_1_0_min_sizes[] = {1024,14336};
    unsigned t1331_bert_encoder_layer_0_output_dropout_Mul_1_0 = createTensors(1,
                                                                           OUTPUT_TENSOR,
                                                                           false,
                                                                           "t1331_bert_encoder_layer_0_output_dropout_Mul_1_0",
                                                                           MEM_INIT_ALL_ZERO,
                                                                           nullptr,
                                                                           t1331_bert_encoder_layer_0_output_dropout_Mul_1_0_max_sizes,
                                                                           2,
                                                                           syn_type_bf16,
                                                                           nullptr,
                                                                           0,
                                                                           0,
                                                                           nullptr,
                                                                           false,
                                                                           t1331_bert_encoder_layer_0_output_dropout_Mul_1_0_min_sizes,
                                                                           synTensorType::DATA_TENSOR)[0];

    // create t1332_bert_encoder_layer_0_output_dropout_Mul_1_0 tensor
    unsigned t1332_bert_encoder_layer_0_output_dropout_Mul_1_0_max_sizes[] = {1024,14336};
    unsigned t1332_bert_encoder_layer_0_output_dropout_Mul_1_0_min_sizes[] = {1024,14336};
    unsigned t1332_bert_encoder_layer_0_output_dropout_Mul_1_0 = createTensors(1,
                                                                           OUTPUT_TENSOR,
                                                                           true,
                                                                           "t1332_bert_encoder_layer_0_output_dropout_Mul_1_0",
                                                                           MEM_INIT_ALL_ZERO,
                                                                           nullptr,
                                                                           t1332_bert_encoder_layer_0_output_dropout_Mul_1_0_max_sizes,
                                                                           2,
                                                                           syn_type_int8,
                                                                           nullptr,
                                                                           0,
                                                                           0,
                                                                           nullptr,
                                                                           false,
                                                                           t1332_bert_encoder_layer_0_output_dropout_Mul_1_0_min_sizes,
                                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_0_output_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n476_0_id;
    unsigned char bert_encoder_layer_0_output_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n476_0_params[] = {205,204,204,61,0,0,0,0};
    addNodeToGraph("dropout_fwd_bf16", {t1328_bert_encoder_layer_0_output_dense_BiasAdd_0, t1107_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_3_0}, {t1331_bert_encoder_layer_0_output_dropout_Mul_1_0, t1332_bert_encoder_layer_0_output_dropout_Mul_1_0}, (void*)bert_encoder_layer_0_output_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n476_0_params, 8, "bert_encoder_layer_0_output_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n476_0", 0 /*graphIndex*/, &bert_encoder_layer_0_output_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n476_0_id);

    /*************
     * bert_encoder_layer_0_output_add_add_fwd_bf16_n477_0 node
     * inputs:
     *     t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0[1024, 14336] (dtype=bf16)
     *     t1331_bert_encoder_layer_0_output_dropout_Mul_1_0[1024, 14336] (dtype=bf16)
     * outputs:
     *     t1333_bert_encoder_layer_0_output_add_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {1024,14336};
    unsigned t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {1024,14336};
    unsigned t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0 = createTensors(1,
                                                                                                 INPUT_TENSOR,
                                                                                                 true,
                                                                                                 "t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0",
                                                                                                 MEM_INIT_ALL_ZERO,
                                                                                                 nullptr,
                                                                                                 t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes,
                                                                                                 2,
                                                                                                 syn_type_bf16,
                                                                                                 nullptr,
                                                                                                 0,
                                                                                                 0,
                                                                                                 nullptr,
                                                                                                 false,
                                                                                                 t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes,
                                                                                                 synTensorType::DATA_TENSOR)[0];

    // create t1333_bert_encoder_layer_0_output_add_0 tensor
    unsigned t1333_bert_encoder_layer_0_output_add_0_max_sizes[] = {1024,14336};
    unsigned t1333_bert_encoder_layer_0_output_add_0_min_sizes[] = {1024,14336};
    unsigned t1333_bert_encoder_layer_0_output_add_0 = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "t1333_bert_encoder_layer_0_output_add_0",
                                                                 MEM_INIT_ALL_ZERO,
                                                                 nullptr,
                                                                 t1333_bert_encoder_layer_0_output_add_0_max_sizes,
                                                                 2,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 t1333_bert_encoder_layer_0_output_add_0_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_0_output_add_add_fwd_bf16_n477_0_id;
    addNodeToGraph("add_fwd_bf16", {t1310_bert_encoder_layer_0_attention_output_LayerNorm_HabanaLayerNorm_0, t1331_bert_encoder_layer_0_output_dropout_Mul_1_0}, {t1333_bert_encoder_layer_0_output_add_0}, nullptr, 0, "bert_encoder_layer_0_output_add_add_fwd_bf16_n477_0", 0 /*graphIndex*/, &bert_encoder_layer_0_output_add_add_fwd_bf16_n477_0_id);

    /*************
     * bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n478_0 node
     * inputs:
     *     t1333_bert_encoder_layer_0_output_add_0[1024, 14336] (dtype=bf16)
     *     t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 14336] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024,1,1,14336};
    unsigned t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024,1,1,14336};
    unsigned t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     INPUT_TENSOR,
                                                                                     false,
                                                                                     "t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     4,
                                                                                     syn_type_uint32,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::SHAPE_TENSOR)[0];

    // create t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024,1,1,14336};
    unsigned t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024,1,1,14336};
    unsigned t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     OUTPUT_TENSOR,
                                                                                     false,
                                                                                     "t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     4,
                                                                                     syn_type_bf16,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n478_0_id;
    addNodeToGraph("reshape", {t1333_bert_encoder_layer_0_output_add_0, t1338_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm}, {t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm}, nullptr, 0, "bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n478_0", 0 /*graphIndex*/, &bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n478_0_id);

    /*************
     * bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n482_0 node
     * inputs:
     *     t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 14336] (dtype=bf16)
     *     t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0[1024] (dtype=float32)
     *     t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0[1024] (dtype=float32)
     * outputs:
     *     t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 14336] (dtype=bf16)
     *     t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 14336] (dtype=float32)
     *     t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 14336] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0 tensor
    unsigned t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0_max_sizes[] = {1024};
    unsigned t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0_min_sizes[] = {1024};
    unsigned t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0 = createTensors(1,
                                                                                                     INPUT_TENSOR,
                                                                                                     true,
                                                                                                     "t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0",
                                                                                                     MEM_INIT_ALL_ZERO,
                                                                                                     nullptr,
                                                                                                     t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0_max_sizes,
                                                                                                     1,
                                                                                                     syn_type_single,
                                                                                                     nullptr,
                                                                                                     0,
                                                                                                     0,
                                                                                                     nullptr,
                                                                                                     false,
                                                                                                     t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0_min_sizes,
                                                                                                     synTensorType::DATA_TENSOR)[0];

    // create t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0 tensor
    unsigned t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0_max_sizes[] = {1024};
    unsigned t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0_min_sizes[] = {1024};
    unsigned t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0 = createTensors(1,
                                                                                                       INPUT_TENSOR,
                                                                                                       true,
                                                                                                       "t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0",
                                                                                                       MEM_INIT_ALL_ZERO,
                                                                                                       nullptr,
                                                                                                       t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0_max_sizes,
                                                                                                       1,
                                                                                                       syn_type_single,
                                                                                                       nullptr,
                                                                                                       0,
                                                                                                       0,
                                                                                                       nullptr,
                                                                                                       false,
                                                                                                       t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0_min_sizes,
                                                                                                       synTensorType::DATA_TENSOR)[0];

    // create t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024,1,1,14336};
    unsigned t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024,1,1,14336};
    unsigned t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     OUTPUT_TENSOR,
                                                                                     false,
                                                                                     "t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     4,
                                                                                     syn_type_bf16,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::DATA_TENSOR)[0];

    // create t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1,1,1,14336};
    unsigned t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1,1,1,14336};
    unsigned t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     OUTPUT_TENSOR,
                                                                                     false,
                                                                                     "t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     4,
                                                                                     syn_type_single,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::DATA_TENSOR)[0];

    // create t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1,1,1,14336};
    unsigned t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1,1,1,14336};
    unsigned t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     OUTPUT_TENSOR,
                                                                                     false,
                                                                                     "t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     4,
                                                                                     syn_type_single,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n482_0_id;
    unsigned char bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n482_0_params[] = {1,0,0,0,204,188,140,43};
    addNodeToGraph("layer_norm_fwd_bf16", {t1337_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm, t328_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_0, t329_bert_encoder_layer_0_output_layernorm_habanalayernorm_readvariableop_1_0}, {t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm, t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm, t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm}, (void*)bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n482_0_params, 8, "bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n482_0", 0 /*graphIndex*/, &bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n482_0_id);

    /*************
     * bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n479_0 node
     * inputs:
     *     t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 14336] (dtype=bf16)
     *     t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 14336] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024,14336};
    unsigned t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024,14336};
    unsigned t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     INPUT_TENSOR,
                                                                                     false,
                                                                                     "t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     2,
                                                                                     syn_type_uint32,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::SHAPE_TENSOR)[0];

    // create t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {1024,14336};
    unsigned t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {1024,14336};
    unsigned t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0 = createTensors(1,
                                                                                       OUTPUT_TENSOR,
                                                                                       false,
                                                                                       "t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0",
                                                                                       MEM_INIT_ALL_ZERO,
                                                                                       nullptr,
                                                                                       t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_max_sizes,
                                                                                       2,
                                                                                       syn_type_bf16,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n479_0_id;
    addNodeToGraph("reshape", {t1339_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm, t1340_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm}, {t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0}, nullptr, 0, "bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n479_0", 0 /*graphIndex*/, &bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n479_0_id);

    /*************
     * bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n480_0 node
     * inputs:
     *     t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 14336] (dtype=float32)
     *     t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 14336] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1[1, 14336] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1,14336};
    unsigned t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1,14336};
    unsigned t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     INPUT_TENSOR,
                                                                                     false,
                                                                                     "t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     2,
                                                                                     syn_type_uint32,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::SHAPE_TENSOR)[0];

    // create t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1 tensor
    unsigned t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1_max_sizes[] = {1,14336};
    unsigned t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1_min_sizes[] = {1,14336};
    unsigned t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1 = createTensors(1,
                                                                                       OUTPUT_TENSOR,
                                                                                       true,
                                                                                       "t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1",
                                                                                       MEM_INIT_ALL_ZERO,
                                                                                       nullptr,
                                                                                       t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1_max_sizes,
                                                                                       2,
                                                                                       syn_type_single,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n480_0_id;
    addNodeToGraph("reshape", {t1341_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm, t1342_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm}, {t1335_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_1}, nullptr, 0, "bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n480_0", 0 /*graphIndex*/, &bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n480_0_id);

    /*************
     * bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n481_0 node
     * inputs:
     *     t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 14336] (dtype=float32)
     *     t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 14336] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2[1, 14336] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1,14336};
    unsigned t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1,14336};
    unsigned t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm = createTensors(1,
                                                                                     INPUT_TENSOR,
                                                                                     false,
                                                                                     "t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                                                                                     MEM_INIT_ALL_ZERO,
                                                                                     nullptr,
                                                                                     t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                                                                                     2,
                                                                                     syn_type_uint32,
                                                                                     nullptr,
                                                                                     0,
                                                                                     0,
                                                                                     nullptr,
                                                                                     false,
                                                                                     t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                                                                                     synTensorType::SHAPE_TENSOR)[0];

    // create t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2 tensor
    unsigned t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2_max_sizes[] = {1,14336};
    unsigned t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2_min_sizes[] = {1,14336};
    unsigned t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2 = createTensors(1,
                                                                                       OUTPUT_TENSOR,
                                                                                       true,
                                                                                       "t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2",
                                                                                       MEM_INIT_ALL_ZERO,
                                                                                       nullptr,
                                                                                       t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2_max_sizes,
                                                                                       2,
                                                                                       syn_type_single,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n481_0_id;
    addNodeToGraph("reshape", {t1343_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm, t1344_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm}, {t1336_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_2}, nullptr, 0, "bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n481_0", 0 /*graphIndex*/, &bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n481_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_query_MatMul_gemm_n483_0 node
     * inputs:
     *     t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0[1024, 14336] (dtype=bf16)
     *     t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0[1024, 1024] (dtype=bf16)
     * outputs:
     *     t1345_bert_encoder_layer_1_attention_self_query_MatMul_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0 tensor
    unsigned t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0_max_sizes[] = {1024,1024};
    unsigned t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0_min_sizes[] = {1024,1024};
    unsigned t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0 = createTensors(1,
                                                                                                                     INPUT_TENSOR,
                                                                                                                     true,
                                                                                                                     "t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0",
                                                                                                                     MEM_INIT_ALL_ZERO,
                                                                                                                     nullptr,
                                                                                                                     t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0_max_sizes,
                                                                                                                     2,
                                                                                                                     syn_type_bf16,
                                                                                                                     nullptr,
                                                                                                                     0,
                                                                                                                     0,
                                                                                                                     nullptr,
                                                                                                                     false,
                                                                                                                     t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0_min_sizes,
                                                                                                                     synTensorType::DATA_TENSOR)[0];

    // create t1345_bert_encoder_layer_1_attention_self_query_MatMul_0 tensor
    unsigned t1345_bert_encoder_layer_1_attention_self_query_MatMul_0_max_sizes[] = {1024,14336};
    unsigned t1345_bert_encoder_layer_1_attention_self_query_MatMul_0_min_sizes[] = {1024,14336};
    unsigned t1345_bert_encoder_layer_1_attention_self_query_MatMul_0 = createTensors(1,
                                                                                  OUTPUT_TENSOR,
                                                                                  false,
                                                                                  "t1345_bert_encoder_layer_1_attention_self_query_MatMul_0",
                                                                                  MEM_INIT_ALL_ZERO,
                                                                                  nullptr,
                                                                                  t1345_bert_encoder_layer_1_attention_self_query_MatMul_0_max_sizes,
                                                                                  2,
                                                                                  syn_type_bf16,
                                                                                  nullptr,
                                                                                  0,
                                                                                  0,
                                                                                  nullptr,
                                                                                  false,
                                                                                  t1345_bert_encoder_layer_1_attention_self_query_MatMul_0_min_sizes,
                                                                                  synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_query_MatMul_gemm_n483_0_id;
    unsigned char bert_encoder_layer_1_attention_self_query_MatMul_gemm_n483_0_params[] = {0,0};
    addNodeToGraph("gemm", {t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0, t455_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_21_0}, {t1345_bert_encoder_layer_1_attention_self_query_MatMul_0}, (void*)bert_encoder_layer_1_attention_self_query_MatMul_gemm_n483_0_params, 2, "bert_encoder_layer_1_attention_self_query_MatMul_gemm_n483_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_query_MatMul_gemm_n483_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_query_BiasAdd_reshape_n484_0 node
     * inputs:
     *     t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0[1024] (dtype=bf16)
     *     t1348_bert_encoder_layer_1_attention_self_query_BiasAdd[1024, 1] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1347_bert_encoder_layer_1_attention_self_query_BiasAdd[1024, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0 tensor
    unsigned t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0_max_sizes[] = {1024};
    unsigned t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0_min_sizes[] = {1024};
    unsigned t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0 = createTensors(1,
                                                                                                                      INPUT_TENSOR,
                                                                                                                      true,
                                                                                                                      "t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0",
                                                                                                                      MEM_INIT_ALL_ZERO,
                                                                                                                      nullptr,
                                                                                                                      t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0_max_sizes,
                                                                                                                      1,
                                                                                                                      syn_type_bf16,
                                                                                                                      nullptr,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      nullptr,
                                                                                                                      false,
                                                                                                                      t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0_min_sizes,
                                                                                                                      synTensorType::DATA_TENSOR)[0];

    // create t1348_bert_encoder_layer_1_attention_self_query_BiasAdd tensor
    unsigned t1348_bert_encoder_layer_1_attention_self_query_BiasAdd_max_sizes[] = {1024,1};
    unsigned t1348_bert_encoder_layer_1_attention_self_query_BiasAdd_min_sizes[] = {1024,1};
    unsigned t1348_bert_encoder_layer_1_attention_self_query_BiasAdd = createTensors(1,
                                                                                 INPUT_TENSOR,
                                                                                 false,
                                                                                 "t1348_bert_encoder_layer_1_attention_self_query_BiasAdd",
                                                                                 MEM_INIT_ALL_ZERO,
                                                                                 nullptr,
                                                                                 t1348_bert_encoder_layer_1_attention_self_query_BiasAdd_max_sizes,
                                                                                 2,
                                                                                 syn_type_uint32,
                                                                                 nullptr,
                                                                                 0,
                                                                                 0,
                                                                                 nullptr,
                                                                                 false,
                                                                                 t1348_bert_encoder_layer_1_attention_self_query_BiasAdd_min_sizes,
                                                                                 synTensorType::SHAPE_TENSOR)[0];

    // create t1347_bert_encoder_layer_1_attention_self_query_BiasAdd tensor
    unsigned t1347_bert_encoder_layer_1_attention_self_query_BiasAdd_max_sizes[] = {1024,1};
    unsigned t1347_bert_encoder_layer_1_attention_self_query_BiasAdd_min_sizes[] = {1024,1};
    unsigned t1347_bert_encoder_layer_1_attention_self_query_BiasAdd = createTensors(1,
                                                                                 OUTPUT_TENSOR,
                                                                                 false,
                                                                                 "t1347_bert_encoder_layer_1_attention_self_query_BiasAdd",
                                                                                 MEM_INIT_ALL_ZERO,
                                                                                 nullptr,
                                                                                 t1347_bert_encoder_layer_1_attention_self_query_BiasAdd_max_sizes,
                                                                                 2,
                                                                                 syn_type_bf16,
                                                                                 nullptr,
                                                                                 0,
                                                                                 0,
                                                                                 nullptr,
                                                                                 false,
                                                                                 t1347_bert_encoder_layer_1_attention_self_query_BiasAdd_min_sizes,
                                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_query_BiasAdd_reshape_n484_0_id;
    addNodeToGraph("reshape", {t461_bert_encoder_layer_1_attention_self_query_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_24_0, t1348_bert_encoder_layer_1_attention_self_query_BiasAdd}, {t1347_bert_encoder_layer_1_attention_self_query_BiasAdd}, nullptr, 0, "bert_encoder_layer_1_attention_self_query_BiasAdd_reshape_n484_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_query_BiasAdd_reshape_n484_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_query_BiasAdd_add_fwd_bf16_n485_0 node
     * inputs:
     *     t1345_bert_encoder_layer_1_attention_self_query_MatMul_0[1024, 14336] (dtype=bf16)
     *     t1347_bert_encoder_layer_1_attention_self_query_BiasAdd[1024, 1] (dtype=bf16)
     * outputs:
     *     t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0 tensor
    unsigned t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0_max_sizes[] = {1024,14336};
    unsigned t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0_min_sizes[] = {1024,14336};
    unsigned t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0 = createTensors(1,
                                                                                   OUTPUT_TENSOR,
                                                                                   false,
                                                                                   "t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0",
                                                                                   MEM_INIT_ALL_ZERO,
                                                                                   nullptr,
                                                                                   t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0_max_sizes,
                                                                                   2,
                                                                                   syn_type_bf16,
                                                                                   nullptr,
                                                                                   0,
                                                                                   0,
                                                                                   nullptr,
                                                                                   false,
                                                                                   t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0_min_sizes,
                                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_query_BiasAdd_add_fwd_bf16_n485_0_id;
    addNodeToGraph("add_fwd_bf16", {t1345_bert_encoder_layer_1_attention_self_query_MatMul_0, t1347_bert_encoder_layer_1_attention_self_query_BiasAdd}, {t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_query_BiasAdd_add_fwd_bf16_n485_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_query_BiasAdd_add_fwd_bf16_n485_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_Reshape_reshape_n486_0 node
     * inputs:
     *     t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0[1024, 14336] (dtype=bf16)
     *     t1350_bert_encoder_layer_1_attention_self_Reshape[64, 16, 512, 28] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1349_bert_encoder_layer_1_attention_self_Reshape_0[64, 16, 512, 28] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1350_bert_encoder_layer_1_attention_self_Reshape tensor
    unsigned t1350_bert_encoder_layer_1_attention_self_Reshape_max_sizes[] = {64,16,512,28};
    unsigned t1350_bert_encoder_layer_1_attention_self_Reshape_min_sizes[] = {64,16,512,28};
    unsigned t1350_bert_encoder_layer_1_attention_self_Reshape = createTensors(1,
                                                                           INPUT_TENSOR,
                                                                           false,
                                                                           "t1350_bert_encoder_layer_1_attention_self_Reshape",
                                                                           MEM_INIT_ALL_ZERO,
                                                                           nullptr,
                                                                           t1350_bert_encoder_layer_1_attention_self_Reshape_max_sizes,
                                                                           4,
                                                                           syn_type_uint32,
                                                                           nullptr,
                                                                           0,
                                                                           0,
                                                                           nullptr,
                                                                           false,
                                                                           t1350_bert_encoder_layer_1_attention_self_Reshape_min_sizes,
                                                                           synTensorType::SHAPE_TENSOR)[0];

    // create t1349_bert_encoder_layer_1_attention_self_Reshape_0 tensor
    unsigned t1349_bert_encoder_layer_1_attention_self_Reshape_0_max_sizes[] = {64,16,512,28};
    unsigned t1349_bert_encoder_layer_1_attention_self_Reshape_0_min_sizes[] = {64,16,512,28};
    unsigned t1349_bert_encoder_layer_1_attention_self_Reshape_0 = createTensors(1,
                                                                             OUTPUT_TENSOR,
                                                                             false,
                                                                             "t1349_bert_encoder_layer_1_attention_self_Reshape_0",
                                                                             MEM_INIT_ALL_ZERO,
                                                                             nullptr,
                                                                             t1349_bert_encoder_layer_1_attention_self_Reshape_0_max_sizes,
                                                                             4,
                                                                             syn_type_bf16,
                                                                             nullptr,
                                                                             0,
                                                                             0,
                                                                             nullptr,
                                                                             false,
                                                                             t1349_bert_encoder_layer_1_attention_self_Reshape_0_min_sizes,
                                                                             synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_Reshape_reshape_n486_0_id;
    addNodeToGraph("reshape", {t1346_bert_encoder_layer_1_attention_self_query_BiasAdd_0, t1350_bert_encoder_layer_1_attention_self_Reshape}, {t1349_bert_encoder_layer_1_attention_self_Reshape_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_Reshape_reshape_n486_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_Reshape_reshape_n486_0_id);

    // /*************
    //  * bert_encoder_layer_1_attention_self_transpose_transpose_n487_0 node
    //  * inputs:
    //  *     t1349_bert_encoder_layer_1_attention_self_Reshape_0[64, 16, 512, 28] (dtype=bf16)
    //  * outputs:
    //  *     t1351_bert_encoder_layer_1_attention_self_transpose_0[64, 512, 16, 28] (dtype=bf16)
    //  * ctrl inputs:
    //  * ctrl outputs:
    //  *************/

    // // create t1351_bert_encoder_layer_1_attention_self_transpose_0 tensor
    // unsigned t1351_bert_encoder_layer_1_attention_self_transpose_0_max_sizes[] = {64,512,16,28};
    // unsigned t1351_bert_encoder_layer_1_attention_self_transpose_0_min_sizes[] = {64,512,16,28};
    // unsigned t1351_bert_encoder_layer_1_attention_self_transpose_0 = createTensors(1,
    //                                                                            OUTPUT_TENSOR,
    //                                                                            false,
    //                                                                            "t1351_bert_encoder_layer_1_attention_self_transpose_0",
    //                                                                            MEM_INIT_ALL_ZERO,
    //                                                                            nullptr,
    //                                                                            t1351_bert_encoder_layer_1_attention_self_transpose_0_max_sizes,
    //                                                                            4,
    //                                                                            syn_type_bf16,
    //                                                                            nullptr,
    //                                                                            0,
    //                                                                            0,
    //                                                                            nullptr,
    //                                                                            false,
    //                                                                            t1351_bert_encoder_layer_1_attention_self_transpose_0_min_sizes,
    //                                                                            synTensorType::DATA_TENSOR)[0];
    // synNodeId bert_encoder_layer_1_attention_self_transpose_transpose_n487_0_id;
    // unsigned char bert_encoder_layer_1_attention_self_transpose_transpose_n487_0_params[] = {0,0,0,0,2,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0};
    // addNodeToGraph("transpose", {t1349_bert_encoder_layer_1_attention_self_Reshape_0}, {t1351_bert_encoder_layer_1_attention_self_transpose_0}, (void*)bert_encoder_layer_1_attention_self_transpose_transpose_n487_0_params, 24, "bert_encoder_layer_1_attention_self_transpose_transpose_n487_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_transpose_transpose_n487_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_key_MatMul_gemm_n488_0 node
     * inputs:
     *     t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0[1024, 14336] (dtype=bf16)
     *     t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0[1024, 1024] (dtype=bf16)
     * outputs:
     *     t1352_bert_encoder_layer_1_attention_self_key_MatMul_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0 tensor
    unsigned t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0_max_sizes[] = {1024,1024};
    unsigned t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0_min_sizes[] = {1024,1024};
    unsigned t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0 = createTensors(1,
                                                                                                                   INPUT_TENSOR,
                                                                                                                   true,
                                                                                                                   "t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0",
                                                                                                                   MEM_INIT_ALL_ZERO,
                                                                                                                   nullptr,
                                                                                                                   t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0_max_sizes,
                                                                                                                   2,
                                                                                                                   syn_type_bf16,
                                                                                                                   nullptr,
                                                                                                                   0,
                                                                                                                   0,
                                                                                                                   nullptr,
                                                                                                                   false,
                                                                                                                   t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0_min_sizes,
                                                                                                                   synTensorType::DATA_TENSOR)[0];

    // create t1352_bert_encoder_layer_1_attention_self_key_MatMul_0 tensor
    unsigned t1352_bert_encoder_layer_1_attention_self_key_MatMul_0_max_sizes[] = {1024,14336};
    unsigned t1352_bert_encoder_layer_1_attention_self_key_MatMul_0_min_sizes[] = {1024,14336};
    unsigned t1352_bert_encoder_layer_1_attention_self_key_MatMul_0 = createTensors(1,
                                                                                OUTPUT_TENSOR,
                                                                                false,
                                                                                "t1352_bert_encoder_layer_1_attention_self_key_MatMul_0",
                                                                                MEM_INIT_ALL_ZERO,
                                                                                nullptr,
                                                                                t1352_bert_encoder_layer_1_attention_self_key_MatMul_0_max_sizes,
                                                                                2,
                                                                                syn_type_bf16,
                                                                                nullptr,
                                                                                0,
                                                                                0,
                                                                                nullptr,
                                                                                false,
                                                                                t1352_bert_encoder_layer_1_attention_self_key_MatMul_0_min_sizes,
                                                                                synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_key_MatMul_gemm_n488_0_id;
    unsigned char bert_encoder_layer_1_attention_self_key_MatMul_gemm_n488_0_params[] = {0,0};
    addNodeToGraph("gemm", {t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0, t457_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_22_0}, {t1352_bert_encoder_layer_1_attention_self_key_MatMul_0}, (void*)bert_encoder_layer_1_attention_self_key_MatMul_gemm_n488_0_params, 2, "bert_encoder_layer_1_attention_self_key_MatMul_gemm_n488_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_key_MatMul_gemm_n488_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_key_BiasAdd_reshape_n489_0 node
     * inputs:
     *     t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0[1024] (dtype=bf16)
     *     t1355_bert_encoder_layer_1_attention_self_key_BiasAdd[1024, 1] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1354_bert_encoder_layer_1_attention_self_key_BiasAdd[1024, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0 tensor
    unsigned t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0_max_sizes[] = {1024};
    unsigned t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0_min_sizes[] = {1024};
    unsigned t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0 = createTensors(1,
                                                                                                                    INPUT_TENSOR,
                                                                                                                    true,
                                                                                                                    "t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0",
                                                                                                                    MEM_INIT_ALL_ZERO,
                                                                                                                    nullptr,
                                                                                                                    t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0_max_sizes,
                                                                                                                    1,
                                                                                                                    syn_type_bf16,
                                                                                                                    nullptr,
                                                                                                                    0,
                                                                                                                    0,
                                                                                                                    nullptr,
                                                                                                                    false,
                                                                                                                    t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0_min_sizes,
                                                                                                                    synTensorType::DATA_TENSOR)[0];

    // create t1355_bert_encoder_layer_1_attention_self_key_BiasAdd tensor
    unsigned t1355_bert_encoder_layer_1_attention_self_key_BiasAdd_max_sizes[] = {1024,1};
    unsigned t1355_bert_encoder_layer_1_attention_self_key_BiasAdd_min_sizes[] = {1024,1};
    unsigned t1355_bert_encoder_layer_1_attention_self_key_BiasAdd = createTensors(1,
                                                                               INPUT_TENSOR,
                                                                               false,
                                                                               "t1355_bert_encoder_layer_1_attention_self_key_BiasAdd",
                                                                               MEM_INIT_ALL_ZERO,
                                                                               nullptr,
                                                                               t1355_bert_encoder_layer_1_attention_self_key_BiasAdd_max_sizes,
                                                                               2,
                                                                               syn_type_uint32,
                                                                               nullptr,
                                                                               0,
                                                                               0,
                                                                               nullptr,
                                                                               false,
                                                                               t1355_bert_encoder_layer_1_attention_self_key_BiasAdd_min_sizes,
                                                                               synTensorType::SHAPE_TENSOR)[0];

    // create t1354_bert_encoder_layer_1_attention_self_key_BiasAdd tensor
    unsigned t1354_bert_encoder_layer_1_attention_self_key_BiasAdd_max_sizes[] = {1024,1};
    unsigned t1354_bert_encoder_layer_1_attention_self_key_BiasAdd_min_sizes[] = {1024,1};
    unsigned t1354_bert_encoder_layer_1_attention_self_key_BiasAdd = createTensors(1,
                                                                               OUTPUT_TENSOR,
                                                                               false,
                                                                               "t1354_bert_encoder_layer_1_attention_self_key_BiasAdd",
                                                                               MEM_INIT_ALL_ZERO,
                                                                               nullptr,
                                                                               t1354_bert_encoder_layer_1_attention_self_key_BiasAdd_max_sizes,
                                                                               2,
                                                                               syn_type_bf16,
                                                                               nullptr,
                                                                               0,
                                                                               0,
                                                                               nullptr,
                                                                               false,
                                                                               t1354_bert_encoder_layer_1_attention_self_key_BiasAdd_min_sizes,
                                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_key_BiasAdd_reshape_n489_0_id;
    addNodeToGraph("reshape", {t463_bert_encoder_layer_1_attention_self_key_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_25_0, t1355_bert_encoder_layer_1_attention_self_key_BiasAdd}, {t1354_bert_encoder_layer_1_attention_self_key_BiasAdd}, nullptr, 0, "bert_encoder_layer_1_attention_self_key_BiasAdd_reshape_n489_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_key_BiasAdd_reshape_n489_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_key_BiasAdd_add_fwd_bf16_n490_0 node
     * inputs:
     *     t1352_bert_encoder_layer_1_attention_self_key_MatMul_0[1024, 14336] (dtype=bf16)
     *     t1354_bert_encoder_layer_1_attention_self_key_BiasAdd[1024, 1] (dtype=bf16)
     * outputs:
     *     t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0 tensor
    unsigned t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0_max_sizes[] = {1024,14336};
    unsigned t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0_min_sizes[] = {1024,14336};
    unsigned t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0 = createTensors(1,
                                                                                 OUTPUT_TENSOR,
                                                                                 false,
                                                                                 "t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0",
                                                                                 MEM_INIT_ALL_ZERO,
                                                                                 nullptr,
                                                                                 t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0_max_sizes,
                                                                                 2,
                                                                                 syn_type_bf16,
                                                                                 nullptr,
                                                                                 0,
                                                                                 0,
                                                                                 nullptr,
                                                                                 false,
                                                                                 t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0_min_sizes,
                                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_1_attention_self_key_BiasAdd_add_fwd_bf16_n490_0_id;
    addNodeToGraph("add_fwd_bf16", {t1352_bert_encoder_layer_1_attention_self_key_MatMul_0, t1354_bert_encoder_layer_1_attention_self_key_BiasAdd}, {t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_key_BiasAdd_add_fwd_bf16_n490_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_key_BiasAdd_add_fwd_bf16_n490_0_id);

    // /*************
    //  * bert_encoder_layer_1_attention_self_Reshape_1_reshape_n491_0 node
    //  * inputs:
    //  *     t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0[1024, 14336] (dtype=bf16)
    //  *     t1357_bert_encoder_layer_1_attention_self_Reshape_1[64, 16, 512, 28] (dtype=uint32) (shape tensor)
    //  * outputs:
    //  *     t1356_bert_encoder_layer_1_attention_self_Reshape_1_0[64, 16, 512, 28] (dtype=bf16)
    //  * ctrl inputs:
    //  * ctrl outputs:
    //  *************/

    // // create t1357_bert_encoder_layer_1_attention_self_Reshape_1 tensor
    // unsigned t1357_bert_encoder_layer_1_attention_self_Reshape_1_max_sizes[] = {64,16,512,28};
    // unsigned t1357_bert_encoder_layer_1_attention_self_Reshape_1_min_sizes[] = {64,16,512,28};
    // unsigned t1357_bert_encoder_layer_1_attention_self_Reshape_1 = createTensors(1,
    //                                                                          INPUT_TENSOR,
    //                                                                          false,
    //                                                                          "t1357_bert_encoder_layer_1_attention_self_Reshape_1",
    //                                                                          MEM_INIT_ALL_ZERO,
    //                                                                          nullptr,
    //                                                                          t1357_bert_encoder_layer_1_attention_self_Reshape_1_max_sizes,
    //                                                                          4,
    //                                                                          syn_type_uint32,
    //                                                                          nullptr,
    //                                                                          0,
    //                                                                          0,
    //                                                                          nullptr,
    //                                                                          false,
    //                                                                          t1357_bert_encoder_layer_1_attention_self_Reshape_1_min_sizes,
    //                                                                          synTensorType::SHAPE_TENSOR)[0];

    // // create t1356_bert_encoder_layer_1_attention_self_Reshape_1_0 tensor
    // unsigned t1356_bert_encoder_layer_1_attention_self_Reshape_1_0_max_sizes[] = {64,16,512,28};
    // unsigned t1356_bert_encoder_layer_1_attention_self_Reshape_1_0_min_sizes[] = {64,16,512,28};
    // unsigned t1356_bert_encoder_layer_1_attention_self_Reshape_1_0 = createTensors(1,
    //                                                                            OUTPUT_TENSOR,
    //                                                                            false,
    //                                                                            "t1356_bert_encoder_layer_1_attention_self_Reshape_1_0",
    //                                                                            MEM_INIT_ALL_ZERO,
    //                                                                            nullptr,
    //                                                                            t1356_bert_encoder_layer_1_attention_self_Reshape_1_0_max_sizes,
    //                                                                            4,
    //                                                                            syn_type_bf16,
    //                                                                            nullptr,
    //                                                                            0,
    //                                                                            0,
    //                                                                            nullptr,
    //                                                                            false,
    //                                                                            t1356_bert_encoder_layer_1_attention_self_Reshape_1_0_min_sizes,
    //                                                                            synTensorType::DATA_TENSOR)[0];
    // synNodeId bert_encoder_layer_1_attention_self_Reshape_1_reshape_n491_0_id;
    // addNodeToGraph("reshape", {t1353_bert_encoder_layer_1_attention_self_key_BiasAdd_0, t1357_bert_encoder_layer_1_attention_self_Reshape_1}, {t1356_bert_encoder_layer_1_attention_self_Reshape_1_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_Reshape_1_reshape_n491_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_Reshape_1_reshape_n491_0_id);

    // /*************
    //  * bert_encoder_layer_1_attention_self_transpose_1_transpose_n492_0 node
    //  * inputs:
    //  *     t1356_bert_encoder_layer_1_attention_self_Reshape_1_0[64, 16, 512, 28] (dtype=bf16)
    //  * outputs:
    //  *     t1358_bert_encoder_layer_1_attention_self_transpose_1_0[64, 512, 16, 28] (dtype=bf16)
    //  * ctrl inputs:
    //  * ctrl outputs:
    //  *************/

    // // create t1358_bert_encoder_layer_1_attention_self_transpose_1_0 tensor
    // unsigned t1358_bert_encoder_layer_1_attention_self_transpose_1_0_max_sizes[] = {64,512,16,28};
    // unsigned t1358_bert_encoder_layer_1_attention_self_transpose_1_0_min_sizes[] = {64,512,16,28};
    // unsigned t1358_bert_encoder_layer_1_attention_self_transpose_1_0 = createTensors(1,
    //                                                                              OUTPUT_TENSOR,
    //                                                                              false,
    //                                                                              "t1358_bert_encoder_layer_1_attention_self_transpose_1_0",
    //                                                                              MEM_INIT_ALL_ZERO,
    //                                                                              nullptr,
    //                                                                              t1358_bert_encoder_layer_1_attention_self_transpose_1_0_max_sizes,
    //                                                                              4,
    //                                                                              syn_type_bf16,
    //                                                                              nullptr,
    //                                                                              0,
    //                                                                              0,
    //                                                                              nullptr,
    //                                                                              false,
    //                                                                              t1358_bert_encoder_layer_1_attention_self_transpose_1_0_min_sizes,
    //                                                                              synTensorType::DATA_TENSOR)[0];
    // synNodeId bert_encoder_layer_1_attention_self_transpose_1_transpose_n492_0_id;
    // unsigned char bert_encoder_layer_1_attention_self_transpose_1_transpose_n492_0_params[] = {0,0,0,0,2,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0};
    // addNodeToGraph("transpose", {t1356_bert_encoder_layer_1_attention_self_Reshape_1_0}, {t1358_bert_encoder_layer_1_attention_self_transpose_1_0}, (void*)bert_encoder_layer_1_attention_self_transpose_1_transpose_n492_0_params, 24, "bert_encoder_layer_1_attention_self_transpose_1_transpose_n492_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_transpose_1_transpose_n492_0_id);

    // /*************
    //  * bert_encoder_layer_1_attention_self_MatMul_batch_gemm_n493_0 node
    //  * inputs:
    //  *     t1351_bert_encoder_layer_1_attention_self_transpose_0[64, 512, 16, 28] (dtype=bf16)
    //  *     t1358_bert_encoder_layer_1_attention_self_transpose_1_0[64, 512, 16, 28] (dtype=bf16)
    //  * outputs:
    //  *     t1359_bert_encoder_layer_1_attention_self_MatMul_0[512, 512, 16, 28] (dtype=bf16)
    //  * ctrl inputs:
    //  * ctrl outputs:
    //  *************/

    // // create t1359_bert_encoder_layer_1_attention_self_MatMul_0 tensor
    // unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes[] = {512,512,16,28};
    // unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0_min_sizes[] = {512,512,16,28};
    // unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0 = createTensors(1,
    //                                                                         OUTPUT_TENSOR,
    //                                                                         true,
    //                                                                         "t1359_bert_encoder_layer_1_attention_self_MatMul_0",
    //                                                                         MEM_INIT_ALL_ZERO,
    //                                                                         nullptr,
    //                                                                         t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes,
    //                                                                         4,
    //                                                                         syn_type_bf16,
    //                                                                         nullptr,
    //                                                                         0,
    //                                                                         0,
    //                                                                         nullptr,
    //                                                                         false,
    //                                                                         t1359_bert_encoder_layer_1_attention_self_MatMul_0_min_sizes,
    //                                                                         synTensorType::DATA_TENSOR)[0];
    // synNodeId bert_encoder_layer_1_attention_self_MatMul_batch_gemm_n493_0_id;
    // unsigned char bert_encoder_layer_1_attention_self_MatMul_batch_gemm_n493_0_params[] = {0,1};
    // addNodeToGraph("batch_gemm", {t1351_bert_encoder_layer_1_attention_self_transpose_0, t1358_bert_encoder_layer_1_attention_self_transpose_1_0}, {t1359_bert_encoder_layer_1_attention_self_MatMul_0}, (void*)bert_encoder_layer_1_attention_self_MatMul_batch_gemm_n493_0_params, 2, "bert_encoder_layer_1_attention_self_MatMul_batch_gemm_n493_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_MatMul_batch_gemm_n493_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_value_MatMul_gemm_n494_0 node
     * inputs:
     *     t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0[1024, 14336] (dtype=bf16)
     *     t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0[1024, 1024] (dtype=bf16)
     * outputs:
     *     t1360_bert_encoder_layer_1_attention_self_value_MatMul_0[1024, 14336] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0 tensor
    unsigned t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0_max_sizes[] = {1024,1024};
    unsigned t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0_min_sizes[] = {1024,1024};
    unsigned t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0 = createTensors(1,
                                                                                                                     INPUT_TENSOR,
                                                                                                                     true,
                                                                                                                     "t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0",
                                                                                                                     MEM_INIT_ALL_ZERO,
                                                                                                                     nullptr,
                                                                                                                     t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0_max_sizes,
                                                                                                                     2,
                                                                                                                     syn_type_bf16,
                                                                                                                     nullptr,
                                                                                                                     0,
                                                                                                                     0,
                                                                                                                     nullptr,
                                                                                                                     false,
                                                                                                                     t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0_min_sizes,
                                                                                                                     synTensorType::DATA_TENSOR)[0];

    // create t1360_bert_encoder_layer_1_attention_self_value_MatMul_0 tensor
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_max_sizes[] = {1024,14336};
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_min_sizes[] = {1024,14336};
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0 = createTensors(1,
                                                                                  OUTPUT_TENSOR,
                                                                                  false,
                                                                                  "t1360_bert_encoder_layer_1_attention_self_value_MatMul_0",
                                                                                  MEM_INIT_ALL_ZERO,
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
    synNodeId bert_encoder_layer_1_attention_self_value_MatMul_gemm_n494_0_id;
    unsigned char bert_encoder_layer_1_attention_self_value_MatMul_gemm_n494_0_params[] = {0,0};
    addNodeToGraph("gemm", {t1334_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0, t459_bert_encoder_layer_1_attention_self_value_MatMul_ReadVariableOp_fp32_to_bf16_cast_23_0}, {t1360_bert_encoder_layer_1_attention_self_value_MatMul_0}, (void*)bert_encoder_layer_1_attention_self_value_MatMul_gemm_n494_0_params, 2, "bert_encoder_layer_1_attention_self_value_MatMul_gemm_n494_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_value_MatMul_gemm_n494_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_value_BiasAdd_reshape_n495_0 node
     * inputs:
     *     t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0[1024] (dtype=bf16)
     *     t1363_bert_encoder_layer_1_attention_self_value_BiasAdd[1024, 1] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1362_bert_encoder_layer_1_attention_self_value_BiasAdd[1024, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0 tensor
    unsigned t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0_max_sizes[] = {1024};
    unsigned t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0_min_sizes[] = {1024};
    unsigned t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0 = createTensors(1,
                                                                                                                      INPUT_TENSOR,
                                                                                                                      true,
                                                                                                                      "t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0",
                                                                                                                      MEM_INIT_ALL_ZERO,
                                                                                                                      nullptr,
                                                                                                                      t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0_max_sizes,
                                                                                                                      1,
                                                                                                                      syn_type_bf16,
                                                                                                                      nullptr,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      nullptr,
                                                                                                                      false,
                                                                                                                      t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0_min_sizes,
                                                                                                                      synTensorType::DATA_TENSOR)[0];

    // create t1363_bert_encoder_layer_1_attention_self_value_BiasAdd tensor
    unsigned t1363_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes[] = {1024,1};
    unsigned t1363_bert_encoder_layer_1_attention_self_value_BiasAdd_min_sizes[] = {1024,1};
    unsigned t1363_bert_encoder_layer_1_attention_self_value_BiasAdd = createTensors(1,
                                                                                 INPUT_TENSOR,
                                                                                 false,
                                                                                 "t1363_bert_encoder_layer_1_attention_self_value_BiasAdd",
                                                                                 MEM_INIT_ALL_ZERO,
                                                                                 nullptr,
                                                                                 t1363_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes,
                                                                                 2,
                                                                                 syn_type_uint32,
                                                                                 nullptr,
                                                                                 0,
                                                                                 0,
                                                                                 nullptr,
                                                                                 false,
                                                                                 t1363_bert_encoder_layer_1_attention_self_value_BiasAdd_min_sizes,
                                                                                 synTensorType::SHAPE_TENSOR)[0];

    // create t1362_bert_encoder_layer_1_attention_self_value_BiasAdd tensor
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes[] = {1024,1};
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_min_sizes[] = {1024,1};
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd = createTensors(1,
                                                                                 OUTPUT_TENSOR,
                                                                                 false,
                                                                                 "t1362_bert_encoder_layer_1_attention_self_value_BiasAdd",
                                                                                 MEM_INIT_ALL_ZERO,
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
    synNodeId bert_encoder_layer_1_attention_self_value_BiasAdd_reshape_n495_0_id;
    addNodeToGraph("reshape", {t465_bert_encoder_layer_1_attention_self_value_BiasAdd_ReadVariableOp_fp32_to_bf16_cast_26_0, t1363_bert_encoder_layer_1_attention_self_value_BiasAdd}, {t1362_bert_encoder_layer_1_attention_self_value_BiasAdd}, nullptr, 0, "bert_encoder_layer_1_attention_self_value_BiasAdd_reshape_n495_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_value_BiasAdd_reshape_n495_0_id);

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

    // create t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0 tensor
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_max_sizes[] = {1024,14336};
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_min_sizes[] = {1024,14336};
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0 = createTensors(1,
                                                                                   OUTPUT_TENSOR,
                                                                                   true,
                                                                                   "t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0",
                                                                                   MEM_INIT_ALL_ZERO,
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
    addNodeToGraph("add_fwd_bf16", {t1360_bert_encoder_layer_1_attention_self_value_MatMul_0, t1362_bert_encoder_layer_1_attention_self_value_BiasAdd}, {t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0_id);

    compileAndRun();
    LOG_INFO(SYN_TEST, "Device time {} usec", getDeviceTimeMicro());
}

TEST_F_GC(SynGaudiLayeredBrainPerfTest_ASIC_CI, fused_softmax_dropout_bgemm)
{
    LayeredBrainEnableCfg lbCfg {}; // Enable layered brain GCFGs

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
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_max_sizes[] = {1024,14336};
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0_min_sizes[] = {1024,14336};
    unsigned t1360_bert_encoder_layer_1_attention_self_value_MatMul_0 = createTensors(1,
                                                                                  INPUT_TENSOR,
                                                                                  true,
                                                                                  "t1360_bert_encoder_layer_1_attention_self_value_MatMul_0",
                                                                                  MEM_INIT_ALL_ZERO,
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
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_max_sizes[] = {1024,1};
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd_min_sizes[] = {1024,1};
    unsigned t1362_bert_encoder_layer_1_attention_self_value_BiasAdd = createTensors(1,
                                                                                 INPUT_TENSOR,
                                                                                 true,
                                                                                 "t1362_bert_encoder_layer_1_attention_self_value_BiasAdd",
                                                                                 MEM_INIT_ALL_ZERO,
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
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_max_sizes[] = {1024,14336};
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0_min_sizes[] = {1024,14336};
    unsigned t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0 = createTensors(1,
                                                                                   OUTPUT_TENSOR,
                                                                                   false,
                                                                                   "t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0",
                                                                                   MEM_INIT_ALL_ZERO,
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
    addNodeToGraph("add_fwd_bf16", {t1360_bert_encoder_layer_1_attention_self_value_MatMul_0, t1362_bert_encoder_layer_1_attention_self_value_BiasAdd}, {t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_value_BiasAdd_add_fwd_bf16_n496_0_id);

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
    unsigned t1365_bert_encoder_layer_1_attention_self_Reshape_2_max_sizes[] = {64,16,512,28};
    unsigned t1365_bert_encoder_layer_1_attention_self_Reshape_2_min_sizes[] = {64,16,512,28};
    unsigned t1365_bert_encoder_layer_1_attention_self_Reshape_2 = createTensors(1,
                                                                             INPUT_TENSOR,
                                                                             false,
                                                                             "t1365_bert_encoder_layer_1_attention_self_Reshape_2",
                                                                             MEM_INIT_ALL_ZERO,
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
    unsigned t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_max_sizes[] = {64,16,512,28};
    unsigned t1364_bert_encoder_layer_1_attention_self_Reshape_2_0_min_sizes[] = {64,16,512,28};
    unsigned t1364_bert_encoder_layer_1_attention_self_Reshape_2_0 = createTensors(1,
                                                                               OUTPUT_TENSOR,
                                                                               false,
                                                                               "t1364_bert_encoder_layer_1_attention_self_Reshape_2_0",
                                                                               MEM_INIT_ALL_ZERO,
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
    addNodeToGraph("reshape", {t1361_bert_encoder_layer_1_attention_self_value_BiasAdd_0, t1365_bert_encoder_layer_1_attention_self_Reshape_2}, {t1364_bert_encoder_layer_1_attention_self_Reshape_2_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_Reshape_2_reshape_n497_0_id);

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
    unsigned t1366_bert_encoder_layer_1_attention_self_transpose_2_0_max_sizes[] = {64,512,16,28};
    unsigned t1366_bert_encoder_layer_1_attention_self_transpose_2_0_min_sizes[] = {64,512,16,28};
    unsigned t1366_bert_encoder_layer_1_attention_self_transpose_2_0 = createTensors(1,
                                                                                 OUTPUT_TENSOR,
                                                                                 false,
                                                                                 "t1366_bert_encoder_layer_1_attention_self_transpose_2_0",
                                                                                 MEM_INIT_ALL_ZERO,
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
    synNodeId bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_id;
    unsigned char bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_params[] = {0,0,0,0,2,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0};
    addNodeToGraph("transpose", {t1364_bert_encoder_layer_1_attention_self_Reshape_2_0}, {t1366_bert_encoder_layer_1_attention_self_transpose_2_0}, (void*)bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_params, 24, "bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_transpose_2_transpose_n498_0_id);

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
    unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0_max_sizes[] = {512,512,16,28};
    unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0_min_sizes[] = {512,512,16,28};
    unsigned t1359_bert_encoder_layer_1_attention_self_MatMul_0 = createTensors(1,
                                                                            INPUT_TENSOR,
                                                                            true,
                                                                            "t1359_bert_encoder_layer_1_attention_self_MatMul_0",
                                                                            MEM_INIT_ALL_ZERO,
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
    unsigned t1368_bert_encoder_layer_1_attention_self_Mul_max_sizes[] = {1,1,1,1};
    unsigned t1368_bert_encoder_layer_1_attention_self_Mul_min_sizes[] = {1,1,1,1};
    unsigned t1368_bert_encoder_layer_1_attention_self_Mul = createTensors(1,
                                                                       INPUT_TENSOR,
                                                                       true,
                                                                       "t1368_bert_encoder_layer_1_attention_self_Mul",
                                                                       MEM_INIT_ALL_ZERO,
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
    unsigned t1367_bert_encoder_layer_1_attention_self_Mul_0_max_sizes[] = {512,512,16,28};
    unsigned t1367_bert_encoder_layer_1_attention_self_Mul_0_min_sizes[] = {512,512,16,28};
    unsigned t1367_bert_encoder_layer_1_attention_self_Mul_0 = createTensors(1,
                                                                         OUTPUT_TENSOR,
                                                                         false,
                                                                         "t1367_bert_encoder_layer_1_attention_self_Mul_0",
                                                                         MEM_INIT_ALL_ZERO,
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
    addNodeToGraph("mult_fwd_bf16", {t1359_bert_encoder_layer_1_attention_self_MatMul_0, t1368_bert_encoder_layer_1_attention_self_Mul}, {t1367_bert_encoder_layer_1_attention_self_Mul_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_Mul_mult_fwd_bf16_n500_0_id);

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
    unsigned t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_max_sizes[] = {512,512,1,28};
    unsigned t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0_min_sizes[] = {512,512,1,28};
    unsigned t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0 = createTensors(1,
                                                                                                 INPUT_TENSOR,
                                                                                                 true,
                                                                                                 "t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0",
                                                                                                 MEM_INIT_ALL_ZERO,
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
    unsigned t1370_bert_encoder_layer_1_attention_self_add_0_max_sizes[] = {512,512,16,28};
    unsigned t1370_bert_encoder_layer_1_attention_self_add_0_min_sizes[] = {512,512,16,28};
    unsigned t1370_bert_encoder_layer_1_attention_self_add_0 = createTensors(1,
                                                                         OUTPUT_TENSOR,
                                                                         false,
                                                                         "t1370_bert_encoder_layer_1_attention_self_add_0",
                                                                         MEM_INIT_ALL_ZERO,
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
    addNodeToGraph("add_fwd_bf16", {t1367_bert_encoder_layer_1_attention_self_Mul_0, t1273_bert_encoder_layer_23_attention_self_mul_1_fp32_to_bf16_cast_14_0}, {t1370_bert_encoder_layer_1_attention_self_add_0}, nullptr, 0, "bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_add_add_fwd_bf16_n501_0_id);

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
    unsigned t1371_bert_encoder_layer_1_attention_self_Softmax_0_max_sizes[] = {512,512,16,28};
    unsigned t1371_bert_encoder_layer_1_attention_self_Softmax_0_min_sizes[] = {512,512,16,28};
    unsigned t1371_bert_encoder_layer_1_attention_self_Softmax_0 = createTensors(1,
                                                                             OUTPUT_TENSOR,
                                                                             false,
                                                                             "t1371_bert_encoder_layer_1_attention_self_Softmax_0",
                                                                             MEM_INIT_ALL_ZERO,
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
    synNodeId bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_id;
    unsigned char bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_params[] = {0,0,0,0};
    addNodeToGraph("softmax_fwd_bf16", {t1370_bert_encoder_layer_1_attention_self_add_0}, {t1371_bert_encoder_layer_1_attention_self_Softmax_0}, (void*)bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_params, 4, "bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_Softmax_softmax_fwd_bf16_n502_0_id);

    /*************
     * bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0 node
     * inputs:
     *     t1371_bert_encoder_layer_1_attention_self_Softmax_0[512, 512, 16, 28] (dtype=bf16)
     *     t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0[1] (dtype=int32)
     * outputs:
     *     t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28] (dtype=bf16)
     *     t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0[512, 512, 16, 28] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0 tensor
    unsigned t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_max_sizes[] = {1};
    unsigned t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0_min_sizes[] = {1};
    unsigned t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0 = createTensors(1,
                                                                                                                                INPUT_TENSOR,
                                                                                                                                true,
                                                                                                                                "t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0",
                                                                                                                                MEM_INIT_ALL_ZERO,
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
    unsigned t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes[] = {512,512,16,28};
    unsigned t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_min_sizes[] = {512,512,16,28};
    unsigned t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 = createTensors(1,
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
    unsigned t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_max_sizes[] = {512,512,16,28};
    unsigned t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0_min_sizes[] = {512,512,16,28};
    unsigned t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0 = createTensors(1,
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
    synNodeId bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_id;
    unsigned char bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_params[] = {205,204,204,61,0,0,0,0};
    addNodeToGraph("dropout_fwd_bf16", {t1371_bert_encoder_layer_1_attention_self_Softmax_0, t1111_HabanaRandomSeedGroupedSlice_bert_embeddings_dropout_Mul_1_habana_dropout_habana_random_seed_5_0}, {t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0, t1373_bert_encoder_layer_1_attention_self_dropout_Mul_1_0}, (void*)bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_params, 8, "bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_dropout_Mul_1_habana_dropout_dropout_stateful_dropout_fwd_bf16_n503_0_id);

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
    unsigned t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_max_sizes[] = {64,512,16,28};
    unsigned t1374_bert_encoder_layer_1_attention_self_MatMul_1_0_min_sizes[] = {64,512,16,28};
    unsigned t1374_bert_encoder_layer_1_attention_self_MatMul_1_0 = createTensors(1,
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
    synNodeId bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_id;
    unsigned char bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_params[] = {0,0};
    addNodeToGraph("batch_gemm", {t1372_bert_encoder_layer_1_attention_self_dropout_Mul_1_0, t1366_bert_encoder_layer_1_attention_self_transpose_2_0}, {t1374_bert_encoder_layer_1_attention_self_MatMul_1_0}, (void*)bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_params, 2, "bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0", 0 /*graphIndex*/, &bert_encoder_layer_1_attention_self_MatMul_1_batch_gemm_n504_0_id);

    compileAndRun();
    LOG_INFO(SYN_TEST, "Device time {} usec", getDeviceTimeMicro());
}

TEST_F_GC(SynGaudiLayeredBrainPerfTest_ASIC_CI, bwd_gemms)
{
    LayeredBrainEnableCfg lbCfg {}; // Enable layered brain GCFGs

    // Graph #0

    /*************
     * gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_gemm_n3504_0 node
     * inputs:
     *     t6482_gradients_1_AddN_9_0[4096, 9216] (dtype=bf16)
     *     t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0[4096, 1024] (dtype=bf16)
     * outputs:
     *     t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0[1024, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t6482_gradients_1_AddN_9_0 tensor
    unsigned t6482_gradients_1_AddN_9_0_max_sizes[] = {4096,9216};
    unsigned t6482_gradients_1_AddN_9_0_min_sizes[] = {4096,9216};
    unsigned t6482_gradients_1_AddN_9_0 = createTensors(1,
                                                    INPUT_TENSOR,
                                                    true,
                                                    "t6482_gradients_1_AddN_9_0",
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    t6482_gradients_1_AddN_9_0_max_sizes,
                                                    2,
                                                    syn_type_bf16,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    t6482_gradients_1_AddN_9_0_min_sizes,
                                                    synTensorType::DATA_TENSOR)[0];

    // create t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0 tensor
    unsigned t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0_max_sizes[] = {4096,1024};
    unsigned t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0_min_sizes[] = {4096,1024};
    unsigned t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0 = createTensors(1,
                                                                                                                      INPUT_TENSOR,
                                                                                                                      true,
                                                                                                                      "t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0",
                                                                                                                      MEM_INIT_ALL_ZERO,
                                                                                                                      nullptr,
                                                                                                                      t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0_max_sizes,
                                                                                                                      2,
                                                                                                                      syn_type_bf16,
                                                                                                                      nullptr,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      nullptr,
                                                                                                                      false,
                                                                                                                      t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0_min_sizes,
                                                                                                                      synTensorType::DATA_TENSOR)[0];

    // create t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0 tensor
    unsigned t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0_max_sizes[] = {1024,9216};
    unsigned t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0_min_sizes[] = {1024,9216};
    unsigned t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0 = createTensors(1,
                                                                                                         OUTPUT_TENSOR,
                                                                                                         true,
                                                                                                         "t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0",
                                                                                                         MEM_INIT_ALL_ZERO,
                                                                                                         nullptr,
                                                                                                         t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0_max_sizes,
                                                                                                         2,
                                                                                                         syn_type_bf16,
                                                                                                         nullptr,
                                                                                                         0,
                                                                                                         0,
                                                                                                         nullptr,
                                                                                                         false,
                                                                                                         t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0_min_sizes,
                                                                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_gemm_n3504_0_id;
    unsigned char gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_gemm_n3504_0_params[] = {0,1};
    addNodeToGraph("gemm", {t6482_gradients_1_AddN_9_0, t1697_bert_encoder_layer_20_intermediate_dense_MatMul_ReadVariableOp_fp32_to_bf16_cast_253_0}, {t6483_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_0}, (void*)gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_gemm_n3504_0_params, 2, "gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_gemm_n3504_0", 0 /*graphIndex*/, &gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_gemm_n3504_0_id);

    /*************
     * gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_gemm_n3514_0 node
     * inputs:
     *     t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0[1024, 9216] (dtype=bf16)
     *     t6482_gradients_1_AddN_9_0[4096, 9216] (dtype=bf16)
     * outputs:
     *     t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0[4096, 1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {1024,9216};
    unsigned t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {1024,9216};
    unsigned t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0 = createTensors(1,
                                                                                                  INPUT_TENSOR,
                                                                                                  true,
                                                                                                  "t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0",
                                                                                                  MEM_INIT_ALL_ZERO,
                                                                                                  nullptr,
                                                                                                  t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0_max_sizes,
                                                                                                  2,
                                                                                                  syn_type_bf16,
                                                                                                  nullptr,
                                                                                                  0,
                                                                                                  0,
                                                                                                  nullptr,
                                                                                                  false,
                                                                                                  t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0_min_sizes,
                                                                                                  synTensorType::DATA_TENSOR)[0];

    // create t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0 tensor
    unsigned t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0_max_sizes[] = {4096,1024};
    unsigned t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0_min_sizes[] = {4096,1024};
    unsigned t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0 = createTensors(1,
                                                                                                           OUTPUT_TENSOR,
                                                                                                           true,
                                                                                                           "t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0",
                                                                                                           MEM_INIT_ALL_ZERO,
                                                                                                           nullptr,
                                                                                                           t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0_max_sizes,
                                                                                                           2,
                                                                                                           syn_type_bf16,
                                                                                                           nullptr,
                                                                                                           0,
                                                                                                           0,
                                                                                                           nullptr,
                                                                                                           false,
                                                                                                           t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0_min_sizes,
                                                                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_gemm_n3514_0_id;
    unsigned char gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_gemm_n3514_0_params[] = {1,0};
    addNodeToGraph("gemm", {t5370_bert_encoder_layer_20_attention_output_LayerNorm_HabanaLayerNorm_0, t6482_gradients_1_AddN_9_0}, {t6498_gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_0}, (void*)gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_gemm_n3514_0_params, 2, "gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_gemm_n3514_0", 0 /*graphIndex*/, &gradients_1_bert_encoder_layer_20_intermediate_dense_MatMul_grad_MatMul_1_gemm_n3514_0_id);


    compileAndRun();
    LOG_INFO(SYN_TEST, "Device time {} usec", getDeviceTimeMicro());
}
// clang-format on