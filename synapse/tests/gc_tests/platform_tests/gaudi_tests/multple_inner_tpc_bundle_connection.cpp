#include "gc_gaudi_test_infra.h"
TEST_F_GC(SynGaudiTestInfra, multple_inner_tpc_bundle_connection)
{
    /*************
     * n25753_bert_encoder_layer_0_intermediate_dense_MatMul node
     * inputs: [t30019_bert_encoder_layer_0_attention_output_LayerNorm_add_layer_norm_0[1024, 1024](dtype=bf16), t29902_F_TO_BF_Cast_13_0[4096, 1024](dtype=bf16)]
     * output: [t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30019_bert_encoder_layer_0_attention_output_LayerNorm_add_layer_norm_0 tensor
    unsigned t30019_bert_encoder_layer_0_attention_output_LayerNorm_add_layer_norm_0_sizes[] = {1024,1024};
    unsigned t30019_bert_encoder_layer_0_attention_output_LayerNorm_add_layer_norm_0 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t30019_bert_encoder_layer_0_attention_output_LayerNorm_add_layer_norm_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30019_bert_encoder_layer_0_attention_output_LayerNorm_add_layer_norm_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t29902_F_TO_BF_Cast_13_0 tensor
    unsigned t29902_F_TO_BF_Cast_13_0_sizes[] = {4096,1024};
    unsigned t29902_F_TO_BF_Cast_13_0 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t29902_F_TO_BF_Cast_13_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t29902_F_TO_BF_Cast_13_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0 tensor
    unsigned t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0_sizes[] = {4096,1024};
    unsigned t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char n25753_bert_encoder_layer_0_intermediate_dense_MatMul_params[] = {0,0};
    addNodeToGraph("gemm", {t30019_bert_encoder_layer_0_attention_output_LayerNorm_add_layer_norm_0, t29902_F_TO_BF_Cast_13_0}, {t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0}, (void*)n25753_bert_encoder_layer_0_intermediate_dense_MatMul_params, 2, "n25753_bert_encoder_layer_0_intermediate_dense_MatMul");

    /*************
     * n25755_bert_encoder_layer_0_intermediate_dense_BiasAdd node
     * inputs: [t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0[4096, 1024](dtype=bf16), t30028[4096, 1](dtype=bf16)]
     * output: [t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30028 tensor
    unsigned t30028_sizes[] = {4096,1};
    unsigned t30028 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t30028",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30028_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0 tensor
    unsigned t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_sizes[] = {4096,1024};
    unsigned t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_bf16", {t30026_bert_encoder_layer_0_intermediate_dense_MatMul_0, t30028}, {t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0}, nullptr, 0, "n25755_bert_encoder_layer_0_intermediate_dense_BiasAdd");

    /*************
     * n25756_BF_TO_F_Cast_375 node
     * inputs: [t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0[4096, 1024](dtype=bf16)]
     * output: [t30029_BF_TO_F_Cast_375_0[4096, 1024](dtype=float32)]
     *************/

    // create t30029_BF_TO_F_Cast_375_0 tensor
    unsigned t30029_BF_TO_F_Cast_375_0_sizes[] = {4096,1024};
    unsigned t30029_BF_TO_F_Cast_375_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30029_BF_TO_F_Cast_375_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30029_BF_TO_F_Cast_375_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_bf16_to_f32", {t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0}, {t30029_BF_TO_F_Cast_375_0}, nullptr, 0, "n25756_BF_TO_F_Cast_375");

    /*************
     * n25758_bert_encoder_layer_0_intermediate_dense_mul_2 node
     * inputs: [t30029_BF_TO_F_Cast_375_0[4096, 1024](dtype=float32), t30031[1, 1](dtype=float32)]
     * output: [t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0[4096, 1024](dtype=float32)]
     *************/

    // create t30031 tensor
    unsigned t30031_sizes[] = {1,1};
    unsigned t30031 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t30031",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30031_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0 tensor
    unsigned t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0_sizes[] = {4096,1024};
    unsigned t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {t30029_BF_TO_F_Cast_375_0, t30031}, {t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0}, nullptr, 0, "n25758_bert_encoder_layer_0_intermediate_dense_mul_2");

    /*************
     * n25759_bert_encoder_layer_0_intermediate_dense_Pow_mul_1 node
     * inputs: [t30029_BF_TO_F_Cast_375_0[4096, 1024](dtype=float32), t30029_BF_TO_F_Cast_375_0[4096, 1024](dtype=float32)]
     * output: [t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0[4096, 1024](dtype=float32)]
     *************/

    // create t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0 tensor
    unsigned t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0_sizes[] = {4096,1024};
    unsigned t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {t30029_BF_TO_F_Cast_375_0, t30029_BF_TO_F_Cast_375_0}, {t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0}, nullptr, 0, "n25759_bert_encoder_layer_0_intermediate_dense_Pow_mul_1");

    /*************
     * n25760_bert_encoder_layer_0_intermediate_dense_Pow_mul_2 node
     * inputs: [t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0[4096, 1024](dtype=float32), t30029_BF_TO_F_Cast_375_0[4096, 1024](dtype=float32)]
     * output: [t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0[4096, 1024](dtype=float32)]
     *************/

    // create t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0 tensor
    unsigned t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0_sizes[] = {4096,1024};
    unsigned t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {t30032_bert_encoder_layer_0_intermediate_dense_Pow_mul_1_0, t30029_BF_TO_F_Cast_375_0}, {t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0}, nullptr, 0, "n25760_bert_encoder_layer_0_intermediate_dense_Pow_mul_2");

    /*************
     * n25762_bert_encoder_layer_0_intermediate_dense_mul node
     * inputs: [t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0[4096, 1024](dtype=float32), t30035[1, 1](dtype=float32)]
     * output: [t30034_bert_encoder_layer_0_intermediate_dense_mul_0[4096, 1024](dtype=float32)]
     *************/

    // create t30035 tensor
    unsigned t30035_sizes[] = {1,1};
    unsigned t30035 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t30035",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30035_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30034_bert_encoder_layer_0_intermediate_dense_mul_0 tensor
    unsigned t30034_bert_encoder_layer_0_intermediate_dense_mul_0_sizes[] = {4096,1024};
    unsigned t30034_bert_encoder_layer_0_intermediate_dense_mul_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30034_bert_encoder_layer_0_intermediate_dense_mul_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30034_bert_encoder_layer_0_intermediate_dense_mul_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {t30033_bert_encoder_layer_0_intermediate_dense_Pow_mul_2_0, t30035}, {t30034_bert_encoder_layer_0_intermediate_dense_mul_0}, nullptr, 0, "n25762_bert_encoder_layer_0_intermediate_dense_mul");

    /*************
     * n25763_F_TO_BF_Cast_15 node
     * inputs: [t30034_bert_encoder_layer_0_intermediate_dense_mul_0[4096, 1024](dtype=float32)]
     * output: [t30036_F_TO_BF_Cast_15_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30036_F_TO_BF_Cast_15_0 tensor
    unsigned t30036_F_TO_BF_Cast_15_0_sizes[] = {4096,1024};
    unsigned t30036_F_TO_BF_Cast_15_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30036_F_TO_BF_Cast_15_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30036_F_TO_BF_Cast_15_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_f32_to_bf16", {t30034_bert_encoder_layer_0_intermediate_dense_mul_0}, {t30036_F_TO_BF_Cast_15_0}, nullptr, 0, "n25763_F_TO_BF_Cast_15");

    /*************
     * n25764_bert_encoder_layer_0_intermediate_dense_add node
     * inputs: [t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0[4096, 1024](dtype=bf16), t30036_F_TO_BF_Cast_15_0[4096, 1024](dtype=bf16)]
     * output: [t30037_bert_encoder_layer_0_intermediate_dense_add_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30037_bert_encoder_layer_0_intermediate_dense_add_0 tensor
    unsigned t30037_bert_encoder_layer_0_intermediate_dense_add_0_sizes[] = {4096,1024};
    unsigned t30037_bert_encoder_layer_0_intermediate_dense_add_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30037_bert_encoder_layer_0_intermediate_dense_add_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30037_bert_encoder_layer_0_intermediate_dense_add_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_bf16", {t30027_bert_encoder_layer_0_intermediate_dense_BiasAdd_0, t30036_F_TO_BF_Cast_15_0}, {t30037_bert_encoder_layer_0_intermediate_dense_add_0}, nullptr, 0, "n25764_bert_encoder_layer_0_intermediate_dense_add");

    /*************
     * n25765_BF_TO_F_Cast_376 node
     * inputs: [t30037_bert_encoder_layer_0_intermediate_dense_add_0[4096, 1024](dtype=bf16)]
     * output: [t30038_BF_TO_F_Cast_376_0[4096, 1024](dtype=float32)]
     *************/

    // create t30038_BF_TO_F_Cast_376_0 tensor
    unsigned t30038_BF_TO_F_Cast_376_0_sizes[] = {4096,1024};
    unsigned t30038_BF_TO_F_Cast_376_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30038_BF_TO_F_Cast_376_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30038_BF_TO_F_Cast_376_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_bf16_to_f32", {t30037_bert_encoder_layer_0_intermediate_dense_add_0}, {t30038_BF_TO_F_Cast_376_0}, nullptr, 0, "n25765_BF_TO_F_Cast_376");

    /*************
     * n25767_bert_encoder_layer_0_intermediate_dense_mul_1 node
     * inputs: [t30038_BF_TO_F_Cast_376_0[4096, 1024](dtype=float32), t30040[1, 1](dtype=float32)]
     * output: [t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0[4096, 1024](dtype=float32)]
     *************/

    // create t30040 tensor
    unsigned t30040_sizes[] = {1,1};
    unsigned t30040 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t30040",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30040_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0 tensor
    unsigned t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0_sizes[] = {4096,1024};
    unsigned t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {t30038_BF_TO_F_Cast_376_0, t30040}, {t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0}, nullptr, 0, "n25767_bert_encoder_layer_0_intermediate_dense_mul_1");

    /*************
     * n25768_F_TO_BF_Cast_16 node
     * inputs: [t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0[4096, 1024](dtype=float32)]
     * output: [t30041_F_TO_BF_Cast_16_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30041_F_TO_BF_Cast_16_0 tensor
    unsigned t30041_F_TO_BF_Cast_16_0_sizes[] = {4096,1024};
    unsigned t30041_F_TO_BF_Cast_16_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30041_F_TO_BF_Cast_16_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30041_F_TO_BF_Cast_16_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_f32_to_bf16", {t30039_bert_encoder_layer_0_intermediate_dense_mul_1_0}, {t30041_F_TO_BF_Cast_16_0}, nullptr, 0, "n25768_F_TO_BF_Cast_16");

    /*************
     * n25769_bert_encoder_layer_0_intermediate_dense_Tanh node
     * inputs: [t30041_F_TO_BF_Cast_16_0[4096, 1024](dtype=bf16)]
     * output: [t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0 tensor
    unsigned t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0_sizes[] = {4096,1024};
    unsigned t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("tanh_fwd_bf16", {t30041_F_TO_BF_Cast_16_0}, {t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0}, nullptr, 0, "n25769_bert_encoder_layer_0_intermediate_dense_Tanh");

    /*************
     * n25771_bert_encoder_layer_0_intermediate_dense_add_1 node
     * inputs: [t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0[4096, 1024](dtype=bf16), t30044[1, 1](dtype=bf16)]
     * output: [t30043_bert_encoder_layer_0_intermediate_dense_add_1_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30044 tensor
    unsigned t30044_sizes[] = {1,1};
    unsigned t30044 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t30044",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30044_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30043_bert_encoder_layer_0_intermediate_dense_add_1_0 tensor
    unsigned t30043_bert_encoder_layer_0_intermediate_dense_add_1_0_sizes[] = {4096,1024};
    unsigned t30043_bert_encoder_layer_0_intermediate_dense_add_1_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30043_bert_encoder_layer_0_intermediate_dense_add_1_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30043_bert_encoder_layer_0_intermediate_dense_add_1_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_bf16", {t30042_bert_encoder_layer_0_intermediate_dense_Tanh_0, t30044}, {t30043_bert_encoder_layer_0_intermediate_dense_add_1_0}, nullptr, 0, "n25771_bert_encoder_layer_0_intermediate_dense_add_1");

    /*************
     * n25772_BF_TO_F_Cast_377 node
     * inputs: [t30043_bert_encoder_layer_0_intermediate_dense_add_1_0[4096, 1024](dtype=bf16)]
     * output: [t30045_BF_TO_F_Cast_377_0[4096, 1024](dtype=float32)]
     *************/

    // create t30045_BF_TO_F_Cast_377_0 tensor
    unsigned t30045_BF_TO_F_Cast_377_0_sizes[] = {4096,1024};
    unsigned t30045_BF_TO_F_Cast_377_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30045_BF_TO_F_Cast_377_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30045_BF_TO_F_Cast_377_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_bf16_to_f32", {t30043_bert_encoder_layer_0_intermediate_dense_add_1_0}, {t30045_BF_TO_F_Cast_377_0}, nullptr, 0, "n25772_BF_TO_F_Cast_377");

    /*************
     * n25773_bert_encoder_layer_0_intermediate_dense_mul_3 node
     * inputs: [t30045_BF_TO_F_Cast_377_0[4096, 1024](dtype=float32), t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0[4096, 1024](dtype=float32)]
     * output: [t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0[4096, 1024](dtype=float32)]
     *************/

    // create t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0 tensor
    unsigned t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0_sizes[] = {4096,1024};
    unsigned t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {t30045_BF_TO_F_Cast_377_0, t30030_bert_encoder_layer_0_intermediate_dense_mul_2_0}, {t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0}, nullptr, 0, "n25773_bert_encoder_layer_0_intermediate_dense_mul_3");

    /*************
     * n25774_F_TO_BF_Cast_18 node
     * inputs: [t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0[4096, 1024](dtype=float32)]
     * output: [t30047_F_TO_BF_Cast_18_0[4096, 1024](dtype=bf16)]
     *************/

    // create t30047_F_TO_BF_Cast_18_0 tensor
    unsigned t30047_F_TO_BF_Cast_18_0_sizes[] = {4096,1024};
    unsigned t30047_F_TO_BF_Cast_18_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30047_F_TO_BF_Cast_18_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30047_F_TO_BF_Cast_18_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_f32_to_bf16", {t30046_bert_encoder_layer_0_intermediate_dense_mul_3_0}, {t30047_F_TO_BF_Cast_18_0}, nullptr, 0, "n25774_F_TO_BF_Cast_18");

    /*************
     * n25775_bert_encoder_layer_0_output_dense_MatMul node
     * inputs: [t30047_F_TO_BF_Cast_18_0[4096, 1024](dtype=bf16), t29931_F_TO_BF_Cast_19_0[1024, 4096](dtype=bf16)]
     * output: [t30048_bert_encoder_layer_0_output_dense_MatMul_0[1024, 1024](dtype=bf16)]
     *************/

    // create t29931_F_TO_BF_Cast_19_0 tensor
    unsigned t29931_F_TO_BF_Cast_19_0_sizes[] = {1024,4096};
    unsigned t29931_F_TO_BF_Cast_19_0 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t29931_F_TO_BF_Cast_19_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t29931_F_TO_BF_Cast_19_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30048_bert_encoder_layer_0_output_dense_MatMul_0 tensor
    unsigned t30048_bert_encoder_layer_0_output_dense_MatMul_0_sizes[] = {1024,1024};
    unsigned t30048_bert_encoder_layer_0_output_dense_MatMul_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "t30048_bert_encoder_layer_0_output_dense_MatMul_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30048_bert_encoder_layer_0_output_dense_MatMul_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char n25775_bert_encoder_layer_0_output_dense_MatMul_params[] = {0,0};
    addNodeToGraph("gemm", {t30047_F_TO_BF_Cast_18_0, t29931_F_TO_BF_Cast_19_0}, {t30048_bert_encoder_layer_0_output_dense_MatMul_0}, (void*)n25775_bert_encoder_layer_0_output_dense_MatMul_params, 2, "n25775_bert_encoder_layer_0_output_dense_MatMul");

    /*************
     * n25777_bert_encoder_layer_0_output_dense_BiasAdd node
     * inputs: [t30048_bert_encoder_layer_0_output_dense_MatMul_0[1024, 1024](dtype=bf16), t30050[1024, 1](dtype=bf16)]
     * output: [t30049_bert_encoder_layer_0_output_dense_BiasAdd_0[1024, 1024](dtype=bf16)]
     *************/

    // create t30050 tensor
    unsigned t30050_sizes[] = {1024,1};
    unsigned t30050 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "t30050",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30050_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create t30049_bert_encoder_layer_0_output_dense_BiasAdd_0 tensor
    unsigned t30049_bert_encoder_layer_0_output_dense_BiasAdd_0_sizes[] = {1024,1024};
    unsigned t30049_bert_encoder_layer_0_output_dense_BiasAdd_0 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "t30049_bert_encoder_layer_0_output_dense_BiasAdd_0",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        t30049_bert_encoder_layer_0_output_dense_BiasAdd_0_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_bf16", {t30048_bert_encoder_layer_0_output_dense_MatMul_0, t30050}, {t30049_bert_encoder_layer_0_output_dense_BiasAdd_0}, nullptr, 0, "n25777_bert_encoder_layer_0_output_dense_BiasAdd");


    compileTopology("multple_inner_tpc_bundle_connection");
}
