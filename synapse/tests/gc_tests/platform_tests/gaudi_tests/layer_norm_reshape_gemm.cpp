#include "gc_gaudi_test_infra.h"
TEST_F_GC(SynTrainingTestInfra, layer_norm_reshape_gemm)
{
    GlobalConfTestSetter pipelineMgmt("ENABLE_PIPELINE_MANAGEMENT", "true");

    // Graph #0

    /*************
     * bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n2931_0 node
     * inputs:
     *     t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=bf16)
     *     t1060_readvariableop_1700_0[1024] (dtype=float32)
     *     t1059_readvariableop_1696_0[1024] (dtype=float32)
     * outputs:
     *     t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=bf16)
     *     t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 9216] (dtype=float32)
     *     t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 9216] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024, 1, 1, 9216};
    unsigned t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024, 1, 1, 9216};
    unsigned t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t1060_readvariableop_1700_0 tensor
    unsigned t1060_readvariableop_1700_0_max_sizes[] = {1024};
    unsigned t1060_readvariableop_1700_0_min_sizes[] = {1024};
    unsigned t1060_readvariableop_1700_0             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "t1060_readvariableop_1700_0",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         t1060_readvariableop_1700_0_max_sizes,
                                                         1,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         t1060_readvariableop_1700_0_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create t1059_readvariableop_1696_0 tensor
    unsigned t1059_readvariableop_1696_0_max_sizes[] = {1024};
    unsigned t1059_readvariableop_1696_0_min_sizes[] = {1024};
    unsigned t1059_readvariableop_1696_0             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "t1059_readvariableop_1696_0",
                                                         MEM_INIT_ALL_ZERO,
                                                         nullptr,
                                                         t1059_readvariableop_1696_0_max_sizes,
                                                         1,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         t1059_readvariableop_1696_0_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    // create t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024, 1, 1, 9216};
    unsigned t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024, 1, 1, 9216};
    unsigned t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 1, 1, 9216};
    unsigned t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 1, 1, 9216};
    unsigned t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 1, 1, 9216};
    unsigned t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 1, 1, 9216};
    unsigned t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n2931_0_id;
    unsigned char bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n2931_0_params[] =
        {1, 0, 0, 0, 111, 18, 131, 58};
    addNodeToGraph("layer_norm_fwd_bf16",
                   {t5604_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm,
                    t1060_readvariableop_1700_0,
                    t1059_readvariableop_1696_0},
                   {t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm,
                    t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm,
                    t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm},
                   (void*)bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n2931_0_params,
                   8,
                   "bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n2931_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n2931_0_id);

    /*************
     * bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2928_0 node
     * inputs:
     *     t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=bf16)
     *     t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1024, 9216] (dtype=uint32) (shape tensor)
     * outputs:
     *     t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0[1024, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024, 9216};
    unsigned t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024, 9216};
    unsigned t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      2,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {1024, 9216};
    unsigned t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {1024, 9216};
    unsigned t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2928_0_id;
    addNodeToGraph("reshape",
                   {t5606_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm,
                    t5607_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm},
                   {t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0},
                   nullptr,
                   0,
                   "bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2928_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2928_0_id);

    /*************
     * bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2929_0 node
     * inputs:
     *     t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 9216] (dtype=float32)
     *     t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1, 9216] (dtype=uint32) (shape tensor)
     * outputs:
     *     t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1[1, 9216] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 9216};
    unsigned t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 9216};
    unsigned t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      2,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1 tensor
    unsigned t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1_max_sizes[] = {1, 9216};
    unsigned t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1_min_sizes[] = {1, 9216};
    unsigned t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2929_0_id;
    addNodeToGraph("reshape",
                   {t5608_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm,
                    t5609_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm},
                   {t5602_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_1},
                   nullptr,
                   0,
                   "bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2929_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2929_0_id);

    /*************
     * bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2930_0 node
     * inputs:
     *     t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 9216] (dtype=float32)
     *     t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm[1, 9216] (dtype=uint32) (shape tensor)
     * outputs:
     *     t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2[1, 9216] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm tensor
    unsigned t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 9216};
    unsigned t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 9216};
    unsigned t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      2,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2 tensor
    unsigned t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2_max_sizes[] = {1, 9216};
    unsigned t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2_min_sizes[] = {1, 9216};
    unsigned t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2930_0_id;
    addNodeToGraph("reshape",
                   {t5610_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm,
                    t5611_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm},
                   {t5603_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_2},
                   nullptr,
                   0,
                   "bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2930_0",
                   0 /*graphIndex*/,
                   &bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_reshape_n2930_0_id);

    /*************
     * MatMul_gemm_n2932_0 node
     * inputs:
     *     t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0[1024, 9216] (dtype=bf16)
     *     t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0[1024, 2] (dtype=bf16)
     * outputs:
     *     t5612_MatMul_0[2, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0 tensor
    unsigned t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0_max_sizes[] = {1024, 2};
    unsigned t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0_min_sizes[] = {1024, 2};
    unsigned t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create t5612_MatMul_0 tensor
    unsigned      t5612_MatMul_0_max_sizes[] = {2, 9216};
    unsigned      t5612_MatMul_0_min_sizes[] = {2, 9216};
    unsigned      t5612_MatMul_0             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            true,
                                            "t5612_MatMul_0",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            t5612_MatMul_0_max_sizes,
                                            2,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            t5612_MatMul_0_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId     MatMul_gemm_n2932_0_id;
    unsigned char MatMul_gemm_n2932_0_params[] = {0, 1};
    addNodeToGraph("gemm",
                   {t5601_bert_encoder_layer_23_output_LayerNorm_HabanaLayerNorm_0,
                    t1769_MatMul_ReadVariableOp_fp32_to_bf16_cast_293_0},
                   {t5612_MatMul_0},
                   (void*)MatMul_gemm_n2932_0_params,
                   2,
                   "MatMul_gemm_n2932_0",
                   0 /*graphIndex*/,
                   &MatMul_gemm_n2932_0_id);

    compileTopology("layer_norm_reshape_gemm", 0);
}
