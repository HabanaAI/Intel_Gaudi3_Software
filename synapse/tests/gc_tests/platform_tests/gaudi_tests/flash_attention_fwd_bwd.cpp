#include "syn_gaudi_two_run_compare_test.h"

TEST_F_GC(SynTrainingTwoRunCompareTest, flash_attention_fwd_bwd_dropout, {synDeviceGaudi2, synDeviceGaudi3})
{
    // Graph #0

    /*************
     * g_0_sdpa_recomp_fwd_bf16_1_0 node
     * inputs:
     *     Q[8, 16, 5, 3] (dtype=bf16)
     *     K[8, 16, 5, 3] (dtype=bf16)
     *     V[8, 16, 5, 3] (dtype=bf16)
     * outputs:
     *     O[8, 16, 5, 3] (dtype=bf16)
     *     stat1[1, 16, 5, 3] (dtype=bf16)
     *     stat2[1, 16, 5, 3] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/
    const unsigned H = 2;
    const unsigned B = 2;
    // create Q tensor
    unsigned Q_max_sizes[] = {8,16,H,B};
    unsigned Q_min_sizes[] = {8,16,H,B};
    unsigned Q = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "Q",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     Q_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     Q_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create K tensor
    unsigned K_max_sizes[] = {8,16,H,B};
    unsigned K_min_sizes[] = {8,16,H,B};
    unsigned K = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "K",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     K_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     K_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create V tensor
    unsigned V_max_sizes[] = {8,16,H,B};
    unsigned V_min_sizes[] = {8,16,H,B};
    unsigned V = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "V",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     V_max_sizes,
                                                     4,
                                                     syn_type_bf16,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     V_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    // create O tensor
    unsigned O_max_sizes[] = {8,16,H,B};
    unsigned O_min_sizes[] = {8,16,H,B};
    unsigned O = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "O",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      O_max_sizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      O_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create stat1 tensor
    unsigned stat1_max_sizes[] = {1,16,H,B};
    unsigned stat1_min_sizes[] = {1,16,H,B};
    unsigned stat1 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "stat1",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      stat1_max_sizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      stat1_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create stat2 tensor
    unsigned stat2_max_sizes[] = {1,16,H,B};
    unsigned stat2_min_sizes[] = {1,16,H,B};
    unsigned stat2 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "stat2",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      stat2_max_sizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      stat2_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
        unsigned dropoutMaskSizes[] = {1};
        unsigned seed_in = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "seed_in",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      dropoutMaskSizes,
                                      1,
                                      syn_type_int32,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      dropoutMaskSizes,
                                      synTensorType::DATA_TENSOR)[0];
            unsigned seed_out = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "seed_out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      dropoutMaskSizes,
                                      1,
                                      syn_type_int32,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      dropoutMaskSizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_sdpa_recomp_fwd_bf16_1_0_id;
    ns_Sdpa::Params g_0_sdpa_recomp_fwd_bf16_1_0_params = {0};
    g_0_sdpa_recomp_fwd_bf16_1_0_params.scale           = 1.0 / sqrt(8);
    g_0_sdpa_recomp_fwd_bf16_1_0_params.dropout.ratio   = 0.1;
    g_0_sdpa_recomp_fwd_bf16_1_0_params.is_causal       = true;
    addNodeToGraph("sdpa_recomp_fwd_bf16", {Q, K, V, INVALID_TENSOR_INDEX, seed_in}, {O, stat1, stat2, seed_out}, static_cast<void*>(&g_0_sdpa_recomp_fwd_bf16_1_0_params), sizeof(g_0_sdpa_recomp_fwd_bf16_1_0_params), "g_0_sdpa_recomp_fwd_bf16_1_0", 0 /*graphIndex*/, &g_0_sdpa_recomp_fwd_bf16_1_0_id);
    setNodeDeterminstic(g_0_sdpa_recomp_fwd_bf16_1_0_id);

    /*************
     * g_0_sdpa_recomp_bwd_bf16_2_0 node
     * inputs:
     *     dO[8, 16, 5, 3] (dtype=bf16)
     *     Q[8, 16, 5, 3] (dtype=bf16)
     *     K[8, 16, 5, 3] (dtype=bf16)
     *     V[8, 16, 5, 3] (dtype=bf16)
     *     stat1[1, 16, 5, 3] (dtype=bf16)
     *     stat2[1, 16, 5, 3] (dtype=float32)
     * outputs:
     *     dQ[8, 16, 5, 3] (dtype=bf16)
     *     dK[8, 16, 5, 3] (dtype=bf16)
     *     dV[8, 16, 5, 3] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create dO tensor
    unsigned dO_max_sizes[] = {8,16,H,B};
    unsigned dO_min_sizes[] = {8,16,H,B};
    unsigned dO = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "dO",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      dO_max_sizes,
                                                      4,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      dO_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create dQ tensor
    unsigned dQ_max_sizes[] = {8,16,H,B};
    unsigned dQ_min_sizes[] = {8,16,H,B};
    unsigned dQ = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "dQ",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       dQ_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       dQ_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create dK tensor
    unsigned dK_max_sizes[] = {8,16,H,B};
    unsigned dK_min_sizes[] = {8,16,H,B};
    unsigned dK = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "dK",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       dK_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       dK_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create dV tensor
    unsigned dV_max_sizes[] = {8,16,H,B};
    unsigned dV_min_sizes[] = {8,16,H,B};
    unsigned dV = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "dV",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       dV_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       dV_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_sdpa_recomp_bwd_bf16_2_0_id;
    ns_Sdpa::Params g_0_sdpa_recomp_bwd_bf16_2_0_params = {0};
    g_0_sdpa_recomp_bwd_bf16_2_0_params.scale           = 1.0 / sqrt(8);
    g_0_sdpa_recomp_bwd_bf16_2_0_params.dropout.ratio   = 0.1;
    g_0_sdpa_recomp_bwd_bf16_2_0_params.is_causal       = true;
    addNodeToGraph("sdpa_recomp_bwd_bf16", {dO, Q, K, V, INVALID_TENSOR_INDEX, stat1, stat2, seed_in}, {dQ, dK, dV}, static_cast<void*>(&g_0_sdpa_recomp_bwd_bf16_2_0_params), sizeof(g_0_sdpa_recomp_bwd_bf16_2_0_params), "g_0_sdpa_recomp_bwd_bf16_2_0", 0 /*graphIndex*/, &g_0_sdpa_recomp_bwd_bf16_2_0_id);
    setNodeDeterminstic(g_0_sdpa_recomp_bwd_bf16_2_0_id);

    /*************
     * g_0_memcpy_3_0 node
     * inputs:
     *     dQ[8, 16, 5, 3] (dtype=bf16)
     * outputs:
     *     g_0_tensor_19[8, 16, 5, 3] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_19 tensor
    unsigned g_0_tensor_19_max_sizes[] = {8,16,H,B};
    unsigned g_0_tensor_19_min_sizes[] = {8,16,H,B};
    unsigned g_0_tensor_19 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_19",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_19_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_19_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_3_0_id;
    addNodeToGraph("memcpy", {dQ}, {g_0_tensor_19}, nullptr, 0, "g_0_memcpy_3_0", 0 /*graphIndex*/, &g_0_memcpy_3_0_id);

    /*************
     * g_0_memcpy_4_0 node
     * inputs:
     *     dK[8, 16, 5, 3] (dtype=bf16)
     * outputs:
     *     g_0_tensor_22[8, 16, 5, 3] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_22 tensor
    unsigned g_0_tensor_22_max_sizes[] = {8,16,H,B};
    unsigned g_0_tensor_22_min_sizes[] = {8,16,H,B};
    unsigned g_0_tensor_22 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_22",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_22_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_22_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_memcpy_4_0_id;
    addNodeToGraph("memcpy", {dK}, {g_0_tensor_22}, nullptr, 0, "g_0_memcpy_4_0", 0 /*graphIndex*/, &g_0_memcpy_4_0_id);

    /*************
     * g_0_memcpy_5_0 node
     * inputs:
     *     dV[8, 16, 5, 3] (dtype=bf16)
     * outputs:
     *     g_0_tensor_25[8, 16, 5, 3] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_25 tensor
    unsigned g_0_tensor_25_max_sizes[] = {8,16,H,B};
    unsigned g_0_tensor_25_min_sizes[] = {8,16,H,B};
    unsigned g_0_tensor_25 = createTensors(1,
                                       OUTPUT_TENSOR,
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
    synNodeId g_0_memcpy_5_0_id;
    addNodeToGraph("memcpy", {dV}, {g_0_tensor_25}, nullptr, 0, "g_0_memcpy_5_0", 0 /*graphIndex*/, &g_0_memcpy_5_0_id);


    addConfigurationToRun(FIRST_RUN, "ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE", "false");
    compareRunsResults({O, g_0_tensor_19, g_0_tensor_22, g_0_tensor_25});

}

// This test checks the gc's tracking mechanism over the flash attention subgraphs throughout compilation.
// In the compilation of the following graph some of the flash attention subgraph nodes are fused with nodes that don't
// belong to it. The fused node should be considered as part of the flash attention subgraph, otherwise the flash
// attention scheduler will cause to compilation failure. (Scenario from SW-167636)
TEST_F_GC(SynTrainingTwoRunCompareTest, flash_attention_subgraph_tracking_test)
{
    // Graph #0

    /*************
     * g_0_transformer_0_self_attention_reshape_2757_0 node
     * inputs:
     *     g_0_tensor_51[64, 256, 232] (dtype=bf16)
     * outputs:
     *     g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view[64, 256, 232, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_51 tensor
    unsigned g_0_tensor_51_max_sizes[] = {64,256,232};
    unsigned g_0_tensor_51_min_sizes[] = {64,256,232};
    unsigned g_0_tensor_51 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "g_0_tensor_51",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_51_max_sizes,
                                       3,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_51_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view tensor
    unsigned g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view_max_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view_min_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view = createTensors(1,
                                                                                         OUTPUT_TENSOR,
                                                                                         false,
                                                                                         "g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view",
                                                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                         nullptr,
                                                                                         g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view_max_sizes,
                                                                                         4,
                                                                                         syn_type_bf16,
                                                                                         nullptr,
                                                                                         0,
                                                                                         0,
                                                                                         nullptr,
                                                                                         false,
                                                                                         g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view_min_sizes,
                                                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_transformer_0_self_attention_reshape_2757_0_id;
    addNodeToGraph("reshape", {g_0_tensor_51}, {g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view}, nullptr, 0, "g_0_transformer_0_self_attention_reshape_2757_0", 0 /*graphIndex*/, &g_0_transformer_0_self_attention_reshape_2757_0_id);

    /*************
     * g_0_transformer_0_self_attention_cast_i8_to_f32_2747_0 node
     * inputs:
     *     g_0_tensor_61__placeholder_0[256, 256, 1, 1] (dtype=int8)
     * outputs:
     *     g_0_tensor_64[256, 256, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_61__placeholder_0 tensor
    unsigned g_0_tensor_61__placeholder_0_max_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_61__placeholder_0_min_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_61__placeholder_0 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_tensor_61__placeholder_0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_tensor_61__placeholder_0_max_sizes,
                                                      4,
                                                      syn_type_int8,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_tensor_61__placeholder_0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_64 tensor
    unsigned g_0_tensor_64_max_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_64_min_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_64 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_64",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_64_max_sizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_64_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_transformer_0_self_attention_cast_i8_to_f32_2747_0_id;
    unsigned char g_0_transformer_0_self_attention_cast_i8_to_f32_2747_0_params[] = {0,0,0,0};
    addNodeToGraph("cast_i8_to_f32", {g_0_tensor_61__placeholder_0}, {g_0_tensor_64}, (void*)g_0_transformer_0_self_attention_cast_i8_to_f32_2747_0_params, 4, "g_0_transformer_0_self_attention_cast_i8_to_f32_2747_0", 0 /*graphIndex*/, &g_0_transformer_0_self_attention_cast_i8_to_f32_2747_0_id);

    /*************
     * g_0_transformer_0_self_attention_constant_f32_2749_0 node
     * inputs:
     * outputs:
     *     g_0_tensor_66[1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_66 tensor
    unsigned g_0_tensor_66_max_sizes[] = {1};
    unsigned g_0_tensor_66_min_sizes[] = {1};
    unsigned g_0_tensor_66 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_66",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_66_max_sizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_66_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_transformer_0_self_attention_constant_f32_2749_0_id;
    unsigned char g_0_transformer_0_self_attention_constant_f32_2749_0_params[] = {0,0,127,255};
    addNodeToGraph("constant_f32", {}, {g_0_tensor_66}, (void*)g_0_transformer_0_self_attention_constant_f32_2749_0_params, 4, "g_0_transformer_0_self_attention_constant_f32_2749_0", 0 /*graphIndex*/, &g_0_transformer_0_self_attention_constant_f32_2749_0_id);

    /*************
     * g_0_transformer_0_self_attention_mult_fwd_f32_2748_0 node
     * inputs:
     *     g_0_tensor_64[256, 256, 1, 1] (dtype=float32)
     *     g_0_tensor_62__placeholder_1[1] (dtype=float32)
     * outputs:
     *     g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul[256, 256, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_62__placeholder_1 tensor
    unsigned g_0_tensor_62__placeholder_1_max_sizes[] = {1};
    unsigned g_0_tensor_62__placeholder_1_min_sizes[] = {1};
    unsigned g_0_tensor_62__placeholder_1 = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_tensor_62__placeholder_1",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_tensor_62__placeholder_1_max_sizes,
                                                      1,
                                                      syn_type_single,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_tensor_62__placeholder_1_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul tensor
    unsigned g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul_max_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul_min_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul = createTensors(1,
                                                                                        OUTPUT_TENSOR,
                                                                                        false,
                                                                                        "g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul",
                                                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                        nullptr,
                                                                                        g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul_max_sizes,
                                                                                        4,
                                                                                        syn_type_single,
                                                                                        nullptr,
                                                                                        0,
                                                                                        0,
                                                                                        nullptr,
                                                                                        false,
                                                                                        g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul_min_sizes,
                                                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_transformer_0_self_attention_mult_fwd_f32_2748_0_id;
    addNodeToGraph("mult_fwd_f32", {g_0_tensor_64, g_0_tensor_62__placeholder_1}, {g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul}, nullptr, 0, "g_0_transformer_0_self_attention_mult_fwd_f32_2748_0", 0 /*graphIndex*/, &g_0_transformer_0_self_attention_mult_fwd_f32_2748_0_id);

    /*************
     * g_0_transformer_0_self_attention_where_f32_2750_0 node
     * inputs:
     *     g_0_tensor_61__placeholder_0[256, 256, 1, 1] (dtype=int8)
     *     g_0_tensor_66[1] (dtype=float32)
     *     g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul[256, 256, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_67[256, 256, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_67 tensor
    unsigned g_0_tensor_67_max_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_67_min_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_67 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       false,
                                       "g_0_tensor_67",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_67_max_sizes,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_67_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_transformer_0_self_attention_where_f32_2750_0_id;
    addNodeToGraph("where_f32", {g_0_tensor_61__placeholder_0, g_0_tensor_66, g_0_tensor_63_id_265272_transformer_0_self_attention_aten__mul}, {g_0_tensor_67}, nullptr, 0, "g_0_transformer_0_self_attention_where_f32_2750_0", 0 /*graphIndex*/, &g_0_transformer_0_self_attention_where_f32_2750_0_id);

    /*************
     * g_0_transformer_0_self_attention_cast_f32_to_bf16_2751_0 node
     * inputs:
     *     g_0_tensor_67[256, 256, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_69[256, 256, 1, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_69 tensor
    unsigned g_0_tensor_69_max_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_69_min_sizes[] = {256,256,1,1};
    unsigned g_0_tensor_69 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_69",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_69_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_69_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_transformer_0_self_attention_cast_f32_to_bf16_2751_0_id;
    unsigned char g_0_transformer_0_self_attention_cast_f32_to_bf16_2751_0_params[] = {0,0,0,0};
    addNodeToGraph("cast_f32_to_bf16", {g_0_tensor_67}, {g_0_tensor_69}, (void*)g_0_transformer_0_self_attention_cast_f32_to_bf16_2751_0_params, 4, "g_0_transformer_0_self_attention_cast_f32_to_bf16_2751_0", 0 /*graphIndex*/, &g_0_transformer_0_self_attention_cast_f32_to_bf16_2751_0_id);

    /*************
     * g_0_transformer_0_self_attention_sdpa_fwd_bf16_2758_0 node
     * inputs:
     *     g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view[64, 256, 232, 1] (dtype=bf16)
     *     g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view[64, 256, 232, 1] (dtype=bf16)
     *     g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view[64, 256, 232, 1] (dtype=bf16)
     *     g_0_tensor_69[256, 256, 1, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_79[64, 256, 232, 1] (dtype=bf16)
     *     g_0_tensor_80[256, 256, 232, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view tensor
    unsigned g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view_max_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view_min_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view = createTensors(1,
                                                                                         INPUT_TENSOR,
                                                                                         true,
                                                                                         "g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view",
                                                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                         nullptr,
                                                                                         g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view_max_sizes,
                                                                                         4,
                                                                                         syn_type_bf16,
                                                                                         nullptr,
                                                                                         0,
                                                                                         0,
                                                                                         nullptr,
                                                                                         false,
                                                                                         g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view_min_sizes,
                                                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view tensor
    unsigned g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view_max_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view_min_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view = createTensors(1,
                                                                                         INPUT_TENSOR,
                                                                                         true,
                                                                                         "g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view",
                                                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                         nullptr,
                                                                                         g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view_max_sizes,
                                                                                         4,
                                                                                         syn_type_bf16,
                                                                                         nullptr,
                                                                                         0,
                                                                                         0,
                                                                                         nullptr,
                                                                                         false,
                                                                                         g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view_min_sizes,
                                                                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_79 tensor
    unsigned g_0_tensor_79_max_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_79_min_sizes[] = {64,256,232,1};
    unsigned g_0_tensor_79 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_79",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_79_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_79_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_80 tensor
    unsigned g_0_tensor_80_max_sizes[] = {256,256,232,1};
    unsigned g_0_tensor_80_min_sizes[] = {256,256,232,1};
    unsigned g_0_tensor_80 = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "g_0_tensor_80",
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       g_0_tensor_80_max_sizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_80_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_transformer_0_self_attention_sdpa_fwd_bf16_2758_0_id;
    unsigned char g_0_transformer_0_self_attention_sdpa_fwd_bf16_2758_0_params[] = {0,0,0,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("sdpa_fwd_bf16", {g_0_tensor_75_id_265281_transformer_0_self_attention_aten__view, g_0_tensor_74_id_265283_transformer_0_self_attention_aten__view, g_0_tensor_73_id_265285_transformer_0_self_attention_aten__view, g_0_tensor_69}, {g_0_tensor_79, g_0_tensor_80}, (void*)g_0_transformer_0_self_attention_sdpa_fwd_bf16_2758_0_params, 24, "g_0_transformer_0_self_attention_sdpa_fwd_bf16_2758_0", 0 /*graphIndex*/, &g_0_transformer_0_self_attention_sdpa_fwd_bf16_2758_0_id);


    addConfigurationToRun(FIRST_RUN, "ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE", "true");

    compareRunsResults({g_0_tensor_79, g_0_tensor_80});
}

TEST_F_GC(SynTrainingTwoRunCompareTest, flash_attention_fwd_slicing_ASIC, {synDeviceGaudi2, synDeviceGaudi3})
{
    /*************
     * g_0_4_self_attention_core_attention_sdpa_fwd_bf16_6301_0 node
     * inputs:
     *     g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose[128, 256, 32, 4] (dtype=bf16)
     * outputs:
     *     g_0_tensor_96[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_97[256, 256, 32, 4] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose tensor
    unsigned g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose tensor
    unsigned g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose tensor
    unsigned g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_96 tensor
    unsigned g_0_tensor_96_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_96_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_96             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_96",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_96_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_96_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_97 tensor
    unsigned      g_0_tensor_97_max_sizes[] = {256, 256, 32, 4};
    unsigned      g_0_tensor_97_min_sizes[] = {256, 256, 32, 4};
    unsigned      g_0_tensor_97             = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "g_0_tensor_97",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_97_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_97_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_4_self_attention_core_attention_sdpa_fwd_bf16_6301_0_id;
    unsigned char g_0_4_self_attention_core_attention_sdpa_fwd_bf16_6301_0_params[] = {
        243, 4, 181, 61, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("sdpa_fwd_bf16",
                   {g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose,
                    g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose,
                    g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose},
                   {g_0_tensor_96, g_0_tensor_97},
                   (void*)g_0_4_self_attention_core_attention_sdpa_fwd_bf16_6301_0_params,
                   28,
                   "g_0_4_self_attention_core_attention_sdpa_fwd_bf16_6301_0",
                   0 /*graphIndex*/,
                   &g_0_4_self_attention_core_attention_sdpa_fwd_bf16_6301_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_FLASH_ATTENTION_SLICING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FLASH_ATTENTION_SLICING", "true");

    compareRunsResults({g_0_tensor_96, g_0_tensor_97});
}

TEST_F_GC(SynTrainingTwoRunCompareTest, flash_attention_bwd_slicing_ASIC, {synDeviceGaudi2, synDeviceGaudi3})
{
    /*************
     * g_0_gradient_4_self_attention_core_attention_sdpa_bwd_bf16_6410_0 node
     * inputs:
     *     g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_97[256, 256, 32, 4] (dtype=bf16)
     * outputs:
     *     g_0_tensor_251[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_252[128, 256, 32, 4] (dtype=bf16)
     *     g_0_tensor_253[128, 256, 32, 4] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute tensor
    unsigned g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute_max_sizes[] = {128,
                                                                                                           256,
                                                                                                           32,
                                                                                                           4};
    unsigned g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute_min_sizes[] = {128,
                                                                                                           256,
                                                                                                           32,
                                                                                                           4};
    unsigned g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose tensor
    unsigned g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose tensor
    unsigned g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose tensor
    unsigned g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_97 tensor
    unsigned g_0_tensor_97_max_sizes[] = {256, 256, 32, 4};
    unsigned g_0_tensor_97_min_sizes[] = {256, 256, 32, 4};
    unsigned g_0_tensor_97             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_97",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_97_max_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_97_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_251 tensor
    unsigned g_0_tensor_251_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_251_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_251             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_251",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_251_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_251_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_252 tensor
    unsigned g_0_tensor_252_max_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_252_min_sizes[] = {128, 256, 32, 4};
    unsigned g_0_tensor_252             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_252",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_252_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_252_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_253 tensor
    unsigned      g_0_tensor_253_max_sizes[] = {128, 256, 32, 4};
    unsigned      g_0_tensor_253_min_sizes[] = {128, 256, 32, 4};
    unsigned      g_0_tensor_253             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "g_0_tensor_253",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_253_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_253_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_4_self_attention_core_attention_sdpa_bwd_bf16_6410_0_id;
    unsigned char g_0_gradient_4_self_attention_core_attention_sdpa_bwd_bf16_6410_0_params[] = {
        243, 4, 181, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("sdpa_bwd_bf16",
                   {g_0_tensor_247_id_22856_gradient_4_self_attention_core_attention_aten__permute,
                    g_0_tensor_92_id_22391_4_self_attention_core_attention_aten__transpose,
                    g_0_tensor_86_id_22395_4_self_attention_core_attention_aten__transpose,
                    g_0_tensor_78_id_22399_4_self_attention_core_attention_aten__transpose,
                    g_0_tensor_97},
                   {g_0_tensor_251, g_0_tensor_252, g_0_tensor_253},
                   (void*)g_0_gradient_4_self_attention_core_attention_sdpa_bwd_bf16_6410_0_params,
                   28,
                   "g_0_gradient_4_self_attention_core_attention_sdpa_bwd_bf16_6410_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_4_self_attention_core_attention_sdpa_bwd_bf16_6410_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_FLASH_ATTENTION_SLICING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FLASH_ATTENTION_SLICING", "true");

    compareRunsResults({g_0_tensor_251, g_0_tensor_252, g_0_tensor_253});
}