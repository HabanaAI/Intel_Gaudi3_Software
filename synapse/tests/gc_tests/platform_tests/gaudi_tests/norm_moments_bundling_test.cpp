#include "syn_gaudi_two_run_compare_test.h"

class NormsSlicingTest : public SynGaudiTwoRunCompareTest
{
protected:
    void SetUpTest() override
    {
        // The flag COMPLEXGUID_USER_ALLOWLIST can't be set using the GlobalConfTestSetter because it is not synapse
        // configuration, so setting it directly using setenv.
        // COMPLEXGUID_USER_ALLOWLIST=<Ssace seperated guid string without the data type to enable ComplexGuid which are disabled by default>
        m_complexGuidPrevCfg = std::getenv("COMPLEXGUID_USER_ALLOWLIST");
        setenv("COMPLEXGUID_USER_ALLOWLIST", "batch_norm_fwd batch_norm_bwd layer_norm_fwd", true);
        SynGaudiTwoRunCompareTest::SetUpTest();
    }

    void TearDownTest() override
    {
        SynGaudiTwoRunCompareTest::TearDownTest();
        // Reset env var to its original value
        if (m_complexGuidPrevCfg) setenv("COMPLEXGUID_USER_ALLOWLIST", m_complexGuidPrevCfg, true);
    }

    void runTest(const std::vector<unsigned>& outputToCompareIdx)
    {
        GlobalConfTestSetter conf("NORM_MOMENTS_CLUSTERING", "true");

        addConfigurationToRun(FIRST_RUN, "ENABLE_SLICE_NORM_BUNDLING", "true");
        addConfigurationToRun(SECOND_RUN, "ENABLE_SLICE_NORM_BUNDLING", "false");

        compareRunsResults(outputToCompareIdx);
    }

    const char* m_complexGuidPrevCfg;
};

TEST_F_GC(NormsSlicingTest, layer_norm_as_norm_moments)
{
    // Graph #0

    /*************
     * g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1834_0 node
     * inputs:
     *     g_0_t3893_bert_encoder_layer_0_output_add_0[1024, 9216] (dtype=bf16)
     *     g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=uint32) (shape
     *tensor) outputs: g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/
    // create g_0_t3893_bert_encoder_layer_0_output_add_0 tensor
    unsigned g_0_t3893_bert_encoder_layer_0_output_add_0_max_sizes[] = {1024, 9216};
    unsigned g_0_t3893_bert_encoder_layer_0_output_add_0_min_sizes[] = {1024, 9216};
    unsigned g_0_t3893_bert_encoder_layer_0_output_add_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3893_bert_encoder_layer_0_output_add_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3893_bert_encoder_layer_0_output_add_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3893_bert_encoder_layer_0_output_add_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024, 1, 1, 9216};
    unsigned g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024, 1, 1, 9216};
    unsigned g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024, 1, 1, 9216};
    unsigned g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024, 1, 1, 9216};
    unsigned g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1834_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_t3893_bert_encoder_layer_0_output_add_0, g_0_t3898_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm},
        {g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm},
        nullptr,
        0,
        "g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1834_0",
        0 /*graphIndex*/,
        &g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1834_0_id);

    /*************
     * g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n1838_0 node
     * inputs:
     *     g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=bf16)
     *     g_0_t1075_readvariableop_90_0[1024] (dtype=float32)
     *     g_0_t1074_readvariableop_86_0[1024] (dtype=float32)
     * outputs:
     *     g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=bf16)
     *     g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 9216] (dtype=float32)
     *     g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1, 1, 1, 9216] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t1075_readvariableop_90_0 tensor
    unsigned g_0_t1075_readvariableop_90_0_max_sizes[] = {1024};
    unsigned g_0_t1075_readvariableop_90_0_min_sizes[] = {1024};
    unsigned g_0_t1075_readvariableop_90_0             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_t1075_readvariableop_90_0",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_t1075_readvariableop_90_0_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_t1075_readvariableop_90_0_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_t1074_readvariableop_86_0 tensor
    unsigned g_0_t1074_readvariableop_86_0_max_sizes[] = {1024};
    unsigned g_0_t1074_readvariableop_86_0_min_sizes[] = {1024};
    unsigned g_0_t1074_readvariableop_86_0             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_t1074_readvariableop_86_0",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_t1074_readvariableop_86_0_max_sizes,
                                                           1,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_t1074_readvariableop_86_0_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024, 1, 1, 9216};
    unsigned g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024, 1, 1, 9216};
    unsigned g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 1, 1, 9216};
    unsigned g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 1, 1, 9216};
    unsigned g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1, 1, 1, 9216};
    unsigned g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1, 1, 1, 9216};
    unsigned g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n1838_0_id;
    unsigned char g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n1838_0_params[] =
        {1, 0, 0, 0, 111, 18, 131, 58};
    addNodeToGraph("layer_norm_fwd_bf16",
                   {g_0_t3897_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm,
                    g_0_t1075_readvariableop_90_0,
                    g_0_t1074_readvariableop_86_0},
                   {g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm,
                    g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm,
                    g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm},
                   (void*)g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n1838_0_params,
                   8,
                   "g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n1838_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_layer_norm_fwd_bf16_n1838_0_id);

    /*************
     * g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1835_0 node
     * inputs:
     *     g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 1, 1, 9216] (dtype=bf16)
     *     g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm[1024, 9216] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0[1024, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm tensor
    unsigned g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes[] = {1024, 9216};
    unsigned g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes[] = {1024, 9216};
    unsigned g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_max_sizes,
                      2,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0 tensor
    unsigned g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_max_sizes[] = {1024, 9216};
    unsigned g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_min_sizes[] = {1024, 9216};
    unsigned g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1835_0_id;
    addNodeToGraph("reshape",
                   {g_0_t3899_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm,
                    g_0_t3900_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm},
                   {g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0},
                   nullptr,
                   0,
                   "g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1835_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_reshape_n1835_0_id);

    /*************
     * g_0_bert_encoder_layer_1_attention_self_query_MatMul_gemm_n1839_0 node
     * inputs:
     *     g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0[1024, 9216] (dtype=bf16)
     *     g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0[1024, 1024]
     *(dtype=bf16) outputs: g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0[1024, 9216] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0 tensor
    unsigned g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0_max_sizes[] = {1024, 9216};
    unsigned g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0_min_sizes[] = {1024, 9216};
    unsigned g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0 tensor
    unsigned
        g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0_max_sizes[] = {
            1024,
            1024};
    unsigned
        g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0_min_sizes[] = {
            1024,
            1024};
    unsigned g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    synNodeId     g_0_bert_encoder_layer_1_attention_self_query_MatMul_gemm_n1839_0_id;
    unsigned char g_0_bert_encoder_layer_1_attention_self_query_MatMul_gemm_n1839_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0,
                    g_0_t1225_bert_encoder_layer_1_attention_self_query_MatMul_ReadVariableOp_fp32_to_bf16_cast_17_0},
                   {g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0},
                   (void*)g_0_bert_encoder_layer_1_attention_self_query_MatMul_gemm_n1839_0_params,
                   2,
                   "g_0_bert_encoder_layer_1_attention_self_query_MatMul_gemm_n1839_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_1_attention_self_query_MatMul_gemm_n1839_0_id);

    /*************
     * g_0_bert_encoder_layer_1_attention_self_key_MatMul_gemm_n1844_0 node
     * inputs:
     *     g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0[1024, 9216] (dtype=bf16)
     *     g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0[1024, 1024]
     *(dtype=bf16) outputs: g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0[1024, 9216] (dtype=bf16) ctrl
     *inputs: ctrl outputs:
     *************/

    // create g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0 tensor
    unsigned g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0_max_sizes[] = {1024, 9216};
    unsigned g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0_min_sizes[] = {1024, 9216};
    unsigned g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0 tensor
    unsigned
        g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0_max_sizes[] = {
            1024,
            1024};
    unsigned
        g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0_min_sizes[] = {
            1024,
            1024};
    unsigned g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0",
            MEM_INIT_ALL_ZERO,
            nullptr,
            g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0_max_sizes,
            2,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    synNodeId     g_0_bert_encoder_layer_1_attention_self_key_MatMul_gemm_n1844_0_id;
    unsigned char g_0_bert_encoder_layer_1_attention_self_key_MatMul_gemm_n1844_0_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {g_0_t3894_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm_0,
                    g_0_t1227_bert_encoder_layer_1_attention_self_key_MatMul_ReadVariableOp_fp32_to_bf16_cast_18_0},
                   {g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0},
                   (void*)g_0_bert_encoder_layer_1_attention_self_key_MatMul_gemm_n1844_0_params,
                   2,
                   "g_0_bert_encoder_layer_1_attention_self_key_MatMul_gemm_n1844_0",
                   0 /*graphIndex*/,
                   &g_0_bert_encoder_layer_1_attention_self_key_MatMul_gemm_n1844_0_id);

    runTest({g_0_t3912_bert_encoder_layer_1_attention_self_key_MatMul_0,
             g_0_t3901_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm,
             g_0_t3903_bert_encoder_layer_0_output_LayerNorm_HabanaLayerNorm,
             g_0_t3905_bert_encoder_layer_1_attention_self_query_MatMul_0});
}

TEST_F_GC(NormsSlicingTest, batch_norm_as_norm_moments_ASIC_CI)
{
    // Graph #0

    /*************
     * g_0_model_conv2d_Conv2D_spatial_convolution_n16_0 node
     * inputs:
     *     g_0_t66_iteratorgetnext_0[64, 256, 256, 32] (dtype=float32)
     *     g_0_t67_model_conv2d_conv2d_readvariableop_0[64, 64, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_t76_model_conv2d_Conv2D_0[64, 256, 256, 32] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t66_iteratorgetnext_0 tensor
    unsigned g_0_t66_iteratorgetnext_0_max_sizes[] = {64,256,256,32};
    unsigned g_0_t66_iteratorgetnext_0_min_sizes[] = {64,256,256,32};
    unsigned g_0_t66_iteratorgetnext_0 = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "g_0_t66_iteratorgetnext_0",
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   g_0_t66_iteratorgetnext_0_max_sizes,
                                                   4,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_t66_iteratorgetnext_0_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_t67_model_conv2d_conv2d_readvariableop_0 tensor
    unsigned g_0_t67_model_conv2d_conv2d_readvariableop_0_max_sizes[] = {64,64,1,1};
    unsigned g_0_t67_model_conv2d_conv2d_readvariableop_0_min_sizes[] = {64,64,1,1};
    unsigned g_0_t67_model_conv2d_conv2d_readvariableop_0 = createTensors(1,
                                                                      INPUT_TENSOR,
                                                                      true,
                                                                      "g_0_t67_model_conv2d_conv2d_readvariableop_0",
                                                                      MEM_INIT_ALL_ZERO,
                                                                      nullptr,
                                                                      g_0_t67_model_conv2d_conv2d_readvariableop_0_max_sizes,
                                                                      4,
                                                                      syn_type_single,
                                                                      nullptr,
                                                                      0,
                                                                      0,
                                                                      nullptr,
                                                                      false,
                                                                      g_0_t67_model_conv2d_conv2d_readvariableop_0_min_sizes,
                                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t76_model_conv2d_Conv2D_0 tensor
    unsigned g_0_t76_model_conv2d_Conv2D_0_max_sizes[] = {64,256,256,32};
    unsigned g_0_t76_model_conv2d_Conv2D_0_min_sizes[] = {64,256,256,32};
    unsigned g_0_t76_model_conv2d_Conv2D_0 = createTensors(1,
                                                       OUTPUT_TENSOR,
                                                       false,
                                                       "g_0_t76_model_conv2d_Conv2D_0",
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       g_0_t76_model_conv2d_Conv2D_0_max_sizes,
                                                       4,
                                                       syn_type_single,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_t76_model_conv2d_Conv2D_0_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_id;
    unsigned char g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,161,117,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("spatial_convolution", {g_0_t66_iteratorgetnext_0, g_0_t67_model_conv2d_conv2d_readvariableop_0}, {g_0_t76_model_conv2d_Conv2D_0}, (void*)g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_params, 104, "g_0_model_conv2d_Conv2D_spatial_convolution_n16_0", 0 /*graphIndex*/, &g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_id);

    /*************
     * g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0 node
     * inputs:
     *     g_0_t76_model_conv2d_Conv2D_0[64, 256, 256, 32] (dtype=float32)
     * outputs:
     *     g_0_t77_model_batch_normalization_moments_mean_0[64, 1, 1, 1] (dtype=float32)
     *     g_0_t78_model_batch_normalization_moments_variance_0[64, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t77_model_batch_normalization_moments_mean_0 tensor
    unsigned g_0_t77_model_batch_normalization_moments_mean_0_max_sizes[] = {64,1,1,1};
    unsigned g_0_t77_model_batch_normalization_moments_mean_0_min_sizes[] = {64,1,1,1};
    unsigned g_0_t77_model_batch_normalization_moments_mean_0 = createTensors(1,
                                                                          OUTPUT_TENSOR,
                                                                          false,
                                                                          "g_0_t77_model_batch_normalization_moments_mean_0",
                                                                          MEM_INIT_ALL_ZERO,
                                                                          nullptr,
                                                                          g_0_t77_model_batch_normalization_moments_mean_0_max_sizes,
                                                                          4,
                                                                          syn_type_single,
                                                                          nullptr,
                                                                          0,
                                                                          0,
                                                                          nullptr,
                                                                          false,
                                                                          g_0_t77_model_batch_normalization_moments_mean_0_min_sizes,
                                                                          synTensorType::DATA_TENSOR)[0];

    // create g_0_t78_model_batch_normalization_moments_variance_0 tensor
    unsigned g_0_t78_model_batch_normalization_moments_variance_0_max_sizes[] = {64,1,1,1};
    unsigned g_0_t78_model_batch_normalization_moments_variance_0_min_sizes[] = {64,1,1,1};
    unsigned g_0_t78_model_batch_normalization_moments_variance_0 = createTensors(1,
                                                                              OUTPUT_TENSOR,
                                                                              false,
                                                                              "g_0_t78_model_batch_normalization_moments_variance_0",
                                                                              MEM_INIT_ALL_ZERO,
                                                                              nullptr,
                                                                              g_0_t78_model_batch_normalization_moments_variance_0_max_sizes,
                                                                              4,
                                                                              syn_type_single,
                                                                              nullptr,
                                                                              0,
                                                                              0,
                                                                              nullptr,
                                                                              false,
                                                                              g_0_t78_model_batch_normalization_moments_variance_0_min_sizes,
                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_id;
    unsigned char g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_params[] = {14,0,0,0};
    addNodeToGraph("norm_moments_fwd_f32", {g_0_t76_model_conv2d_Conv2D_0}, {g_0_t77_model_batch_normalization_moments_mean_0, g_0_t78_model_batch_normalization_moments_variance_0}, (void*)g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_params, 4, "g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_id);

    /*************
     * g_0_model_batch_normalization_moments_Squeeze_reshape_n18_0 node
     * inputs:
     *     g_0_t77_model_batch_normalization_moments_mean_0[64, 1, 1, 1] (dtype=float32)
     *     g_0_t80_model_batch_normalization_moments_Squeeze[64] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t79_model_batch_normalization_moments_Squeeze_0[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t80_model_batch_normalization_moments_Squeeze tensor
    unsigned g_0_t80_model_batch_normalization_moments_Squeeze_max_sizes[] = {64};
    unsigned g_0_t80_model_batch_normalization_moments_Squeeze_min_sizes[] = {64};
    unsigned g_0_t80_model_batch_normalization_moments_Squeeze = createTensors(1,
                                                                           INPUT_TENSOR,
                                                                           false,
                                                                           "g_0_t80_model_batch_normalization_moments_Squeeze",
                                                                           MEM_INIT_ALL_ZERO,
                                                                           nullptr,
                                                                           g_0_t80_model_batch_normalization_moments_Squeeze_max_sizes,
                                                                           1,
                                                                           syn_type_uint32,
                                                                           nullptr,
                                                                           0,
                                                                           0,
                                                                           nullptr,
                                                                           false,
                                                                           g_0_t80_model_batch_normalization_moments_Squeeze_min_sizes,
                                                                           synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t79_model_batch_normalization_moments_Squeeze_0 tensor
    unsigned g_0_t79_model_batch_normalization_moments_Squeeze_0_max_sizes[] = {64};
    unsigned g_0_t79_model_batch_normalization_moments_Squeeze_0_min_sizes[] = {64};
    unsigned g_0_t79_model_batch_normalization_moments_Squeeze_0 = createTensors(1,
                                                                             OUTPUT_TENSOR,
                                                                             true,
                                                                             "g_0_t79_model_batch_normalization_moments_Squeeze_0",
                                                                             MEM_INIT_ALL_ZERO,
                                                                             nullptr,
                                                                             g_0_t79_model_batch_normalization_moments_Squeeze_0_max_sizes,
                                                                             1,
                                                                             syn_type_single,
                                                                             nullptr,
                                                                             0,
                                                                             0,
                                                                             nullptr,
                                                                             false,
                                                                             g_0_t79_model_batch_normalization_moments_Squeeze_0_min_sizes,
                                                                             synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_moments_Squeeze_reshape_n18_0_id;
    addNodeToGraph("reshape", {g_0_t77_model_batch_normalization_moments_mean_0, g_0_t80_model_batch_normalization_moments_Squeeze}, {g_0_t79_model_batch_normalization_moments_Squeeze_0}, nullptr, 0, "g_0_model_batch_normalization_moments_Squeeze_reshape_n18_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_moments_Squeeze_reshape_n18_0_id);

    /*************
     * g_0_model_batch_normalization_moments_Squeeze_1_reshape_n19_0 node
     * inputs:
     *     g_0_t78_model_batch_normalization_moments_variance_0[64, 1, 1, 1] (dtype=float32)
     *     g_0_t82_model_batch_normalization_moments_Squeeze_1[64] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t81_model_batch_normalization_moments_Squeeze_1_0[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t82_model_batch_normalization_moments_Squeeze_1 tensor
    unsigned g_0_t82_model_batch_normalization_moments_Squeeze_1_max_sizes[] = {64};
    unsigned g_0_t82_model_batch_normalization_moments_Squeeze_1_min_sizes[] = {64};
    unsigned g_0_t82_model_batch_normalization_moments_Squeeze_1 = createTensors(1,
                                                                             INPUT_TENSOR,
                                                                             false,
                                                                             "g_0_t82_model_batch_normalization_moments_Squeeze_1",
                                                                             MEM_INIT_ALL_ZERO,
                                                                             nullptr,
                                                                             g_0_t82_model_batch_normalization_moments_Squeeze_1_max_sizes,
                                                                             1,
                                                                             syn_type_uint32,
                                                                             nullptr,
                                                                             0,
                                                                             0,
                                                                             nullptr,
                                                                             false,
                                                                             g_0_t82_model_batch_normalization_moments_Squeeze_1_min_sizes,
                                                                             synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t81_model_batch_normalization_moments_Squeeze_1_0 tensor
    unsigned g_0_t81_model_batch_normalization_moments_Squeeze_1_0_max_sizes[] = {64};
    unsigned g_0_t81_model_batch_normalization_moments_Squeeze_1_0_min_sizes[] = {64};
    unsigned g_0_t81_model_batch_normalization_moments_Squeeze_1_0 = createTensors(1,
                                                                               OUTPUT_TENSOR,
                                                                               true,
                                                                               "g_0_t81_model_batch_normalization_moments_Squeeze_1_0",
                                                                               MEM_INIT_ALL_ZERO,
                                                                               nullptr,
                                                                               g_0_t81_model_batch_normalization_moments_Squeeze_1_0_max_sizes,
                                                                               1,
                                                                               syn_type_single,
                                                                               nullptr,
                                                                               0,
                                                                               0,
                                                                               nullptr,
                                                                               false,
                                                                               g_0_t81_model_batch_normalization_moments_Squeeze_1_0_min_sizes,
                                                                               synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_moments_Squeeze_1_reshape_n19_0_id;
    addNodeToGraph("reshape", {g_0_t78_model_batch_normalization_moments_variance_0, g_0_t82_model_batch_normalization_moments_Squeeze_1}, {g_0_t81_model_batch_normalization_moments_Squeeze_1_0}, nullptr, 0, "g_0_model_batch_normalization_moments_Squeeze_1_reshape_n19_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_moments_Squeeze_1_reshape_n19_0_id);

    /*************
     * g_0_model_batch_normalization_batchnorm_mul_2_mult_fwd_f32_n40_0 node
     * inputs:
     *     g_0_t118_model_batch_normalization_batchnorm_mul_0[64] (dtype=float32)
     *     g_0_t79_model_batch_normalization_moments_Squeeze_0[64] (dtype=float32)
     * outputs:
     *     g_0_t119_model_batch_normalization_batchnorm_mul_2_0[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t118_model_batch_normalization_batchnorm_mul_0 tensor
    unsigned g_0_t118_model_batch_normalization_batchnorm_mul_0_max_sizes[] = {64};
    unsigned g_0_t118_model_batch_normalization_batchnorm_mul_0_min_sizes[] = {64};
    unsigned g_0_t118_model_batch_normalization_batchnorm_mul_0 = createTensors(1,
                                                                            INPUT_TENSOR,
                                                                            true,
                                                                            "g_0_t118_model_batch_normalization_batchnorm_mul_0",
                                                                            MEM_INIT_ALL_ZERO,
                                                                            nullptr,
                                                                            g_0_t118_model_batch_normalization_batchnorm_mul_0_max_sizes,
                                                                            1,
                                                                            syn_type_single,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t118_model_batch_normalization_batchnorm_mul_0_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_t119_model_batch_normalization_batchnorm_mul_2_0 tensor
    unsigned g_0_t119_model_batch_normalization_batchnorm_mul_2_0_max_sizes[] = {64};
    unsigned g_0_t119_model_batch_normalization_batchnorm_mul_2_0_min_sizes[] = {64};
    unsigned g_0_t119_model_batch_normalization_batchnorm_mul_2_0 = createTensors(1,
                                                                              OUTPUT_TENSOR,
                                                                              false,
                                                                              "g_0_t119_model_batch_normalization_batchnorm_mul_2_0",
                                                                              MEM_INIT_ALL_ZERO,
                                                                              nullptr,
                                                                              g_0_t119_model_batch_normalization_batchnorm_mul_2_0_max_sizes,
                                                                              1,
                                                                              syn_type_single,
                                                                              nullptr,
                                                                              0,
                                                                              0,
                                                                              nullptr,
                                                                              false,
                                                                              g_0_t119_model_batch_normalization_batchnorm_mul_2_0_min_sizes,
                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_batchnorm_mul_2_mult_fwd_f32_n40_0_id;
    addNodeToGraph("mult_fwd_f32", {g_0_t118_model_batch_normalization_batchnorm_mul_0, g_0_t79_model_batch_normalization_moments_Squeeze_0}, {g_0_t119_model_batch_normalization_batchnorm_mul_2_0}, nullptr, 0, "g_0_model_batch_normalization_batchnorm_mul_2_mult_fwd_f32_n40_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_batchnorm_mul_2_mult_fwd_f32_n40_0_id);

    /*************
     * g_0_model_batch_normalization_batchnorm_sub_sub_fwd_f32_n41_0 node
     * inputs:
     *     g_0_t71_model_batch_normalization_batchnorm_readvariableop_0[64] (dtype=float32)
     *     g_0_t119_model_batch_normalization_batchnorm_mul_2_0[64] (dtype=float32)
     * outputs:
     *     g_0_t120_model_batch_normalization_batchnorm_sub_0[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t71_model_batch_normalization_batchnorm_readvariableop_0 tensor
    unsigned g_0_t71_model_batch_normalization_batchnorm_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t71_model_batch_normalization_batchnorm_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t71_model_batch_normalization_batchnorm_readvariableop_0 = createTensors(1,
                                                                                      INPUT_TENSOR,
                                                                                      true,
                                                                                      "g_0_t71_model_batch_normalization_batchnorm_readvariableop_0",
                                                                                      MEM_INIT_ALL_ZERO,
                                                                                      nullptr,
                                                                                      g_0_t71_model_batch_normalization_batchnorm_readvariableop_0_max_sizes,
                                                                                      1,
                                                                                      syn_type_single,
                                                                                      nullptr,
                                                                                      0,
                                                                                      0,
                                                                                      nullptr,
                                                                                      false,
                                                                                      g_0_t71_model_batch_normalization_batchnorm_readvariableop_0_min_sizes,
                                                                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t120_model_batch_normalization_batchnorm_sub_0 tensor
    unsigned g_0_t120_model_batch_normalization_batchnorm_sub_0_max_sizes[] = {64};
    unsigned g_0_t120_model_batch_normalization_batchnorm_sub_0_min_sizes[] = {64};
    unsigned g_0_t120_model_batch_normalization_batchnorm_sub_0 = createTensors(1,
                                                                            OUTPUT_TENSOR,
                                                                            false,
                                                                            "g_0_t120_model_batch_normalization_batchnorm_sub_0",
                                                                            MEM_INIT_ALL_ZERO,
                                                                            nullptr,
                                                                            g_0_t120_model_batch_normalization_batchnorm_sub_0_max_sizes,
                                                                            1,
                                                                            syn_type_single,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t120_model_batch_normalization_batchnorm_sub_0_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_batchnorm_sub_sub_fwd_f32_n41_0_id;
    addNodeToGraph("sub_fwd_f32", {g_0_t71_model_batch_normalization_batchnorm_readvariableop_0, g_0_t119_model_batch_normalization_batchnorm_mul_2_0}, {g_0_t120_model_batch_normalization_batchnorm_sub_0}, nullptr, 0, "g_0_model_batch_normalization_batchnorm_sub_sub_fwd_f32_n41_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_batchnorm_sub_sub_fwd_f32_n41_0_id);

    /*************
     * g_0_model_batch_normalization_batchnorm_add_1_reshape_n44_0 node
     * inputs:
     *     g_0_t120_model_batch_normalization_batchnorm_sub_0[64] (dtype=float32)
     *     g_0_t126_model_batch_normalization_batchnorm_add_1[64, 1, 1, 1] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_t125_model_batch_normalization_batchnorm_add_1[64, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t126_model_batch_normalization_batchnorm_add_1 tensor
    unsigned g_0_t126_model_batch_normalization_batchnorm_add_1_max_sizes[] = {64,1,1,1};
    unsigned g_0_t126_model_batch_normalization_batchnorm_add_1_min_sizes[] = {64,1,1,1};
    unsigned g_0_t126_model_batch_normalization_batchnorm_add_1 = createTensors(1,
                                                                            INPUT_TENSOR,
                                                                            false,
                                                                            "g_0_t126_model_batch_normalization_batchnorm_add_1",
                                                                            MEM_INIT_ALL_ZERO,
                                                                            nullptr,
                                                                            g_0_t126_model_batch_normalization_batchnorm_add_1_max_sizes,
                                                                            4,
                                                                            syn_type_uint32,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t126_model_batch_normalization_batchnorm_add_1_min_sizes,
                                                                            synTensorType::SHAPE_TENSOR)[0];

    // create g_0_t125_model_batch_normalization_batchnorm_add_1 tensor
    unsigned g_0_t125_model_batch_normalization_batchnorm_add_1_max_sizes[] = {64,1,1,1};
    unsigned g_0_t125_model_batch_normalization_batchnorm_add_1_min_sizes[] = {64,1,1,1};
    unsigned g_0_t125_model_batch_normalization_batchnorm_add_1 = createTensors(1,
                                                                            OUTPUT_TENSOR,
                                                                            false,
                                                                            "g_0_t125_model_batch_normalization_batchnorm_add_1",
                                                                            MEM_INIT_ALL_ZERO,
                                                                            nullptr,
                                                                            g_0_t125_model_batch_normalization_batchnorm_add_1_max_sizes,
                                                                            4,
                                                                            syn_type_single,
                                                                            nullptr,
                                                                            0,
                                                                            0,
                                                                            nullptr,
                                                                            false,
                                                                            g_0_t125_model_batch_normalization_batchnorm_add_1_min_sizes,
                                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_batchnorm_add_1_reshape_n44_0_id;
    addNodeToGraph("reshape", {g_0_t120_model_batch_normalization_batchnorm_sub_0, g_0_t126_model_batch_normalization_batchnorm_add_1}, {g_0_t125_model_batch_normalization_batchnorm_add_1}, nullptr, 0, "g_0_model_batch_normalization_batchnorm_add_1_reshape_n44_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_batchnorm_add_1_reshape_n44_0_id);

    /*************
     * g_0_model_batch_normalization_batchnorm_add_1_add_fwd_f32_n45_0 node
     * inputs:
     *     g_0_t121_model_batch_normalization_batchnorm_mul_1_0[64, 256, 256, 32] (dtype=float32)
     *     g_0_t125_model_batch_normalization_batchnorm_add_1[64, 1, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_t124_model_batch_normalization_batchnorm_add_1_0[64, 256, 256, 32] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t121_model_batch_normalization_batchnorm_mul_1_0 tensor
    unsigned g_0_t121_model_batch_normalization_batchnorm_mul_1_0_max_sizes[] = {64,256,256,32};
    unsigned g_0_t121_model_batch_normalization_batchnorm_mul_1_0_min_sizes[] = {64,256,256,32};
    unsigned g_0_t121_model_batch_normalization_batchnorm_mul_1_0 = createTensors(1,
                                                                              INPUT_TENSOR,
                                                                              true,
                                                                              "g_0_t121_model_batch_normalization_batchnorm_mul_1_0",
                                                                              MEM_INIT_ALL_ZERO,
                                                                              nullptr,
                                                                              g_0_t121_model_batch_normalization_batchnorm_mul_1_0_max_sizes,
                                                                              4,
                                                                              syn_type_single,
                                                                              nullptr,
                                                                              0,
                                                                              0,
                                                                              nullptr,
                                                                              false,
                                                                              g_0_t121_model_batch_normalization_batchnorm_mul_1_0_min_sizes,
                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t124_model_batch_normalization_batchnorm_add_1_0 tensor
    unsigned g_0_t124_model_batch_normalization_batchnorm_add_1_0_max_sizes[] = {64,256,256,32};
    unsigned g_0_t124_model_batch_normalization_batchnorm_add_1_0_min_sizes[] = {64,256,256,32};
    unsigned g_0_t124_model_batch_normalization_batchnorm_add_1_0 = createTensors(1,
                                                                              OUTPUT_TENSOR,
                                                                              false,
                                                                              "g_0_t124_model_batch_normalization_batchnorm_add_1_0",
                                                                              MEM_INIT_ALL_ZERO,
                                                                              nullptr,
                                                                              g_0_t124_model_batch_normalization_batchnorm_add_1_0_max_sizes,
                                                                              4,
                                                                              syn_type_single,
                                                                              nullptr,
                                                                              0,
                                                                              0,
                                                                              nullptr,
                                                                              false,
                                                                              g_0_t124_model_batch_normalization_batchnorm_add_1_0_min_sizes,
                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_batchnorm_add_1_add_fwd_f32_n45_0_id;
    addNodeToGraph("add_fwd_f32", {g_0_t121_model_batch_normalization_batchnorm_mul_1_0, g_0_t125_model_batch_normalization_batchnorm_add_1}, {g_0_t124_model_batch_normalization_batchnorm_add_1_0}, nullptr, 0, "g_0_model_batch_normalization_batchnorm_add_1_add_fwd_f32_n45_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_batchnorm_add_1_add_fwd_f32_n45_0_id);

    /*************
     * g_0_model_activation_Relu_relu_fwd_f32_n46_0 node
     * inputs:
     *     g_0_t124_model_batch_normalization_batchnorm_add_1_0[64, 256, 256, 32] (dtype=float32)
     * outputs:
     *     g_0_t127_model_activation_Relu_0[64, 256, 256, 32] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *     g_0_model_activation_Relu_relu_fwd_f32_n46_control_edge_410[] (dtype=invalid)
     *************/

    // create g_0_t127_model_activation_Relu_0 tensor
    unsigned g_0_t127_model_activation_Relu_0_max_sizes[] = {64,256,256,32};
    unsigned g_0_t127_model_activation_Relu_0_min_sizes[] = {64,256,256,32};
    unsigned g_0_t127_model_activation_Relu_0 = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          false,
                                                          "g_0_t127_model_activation_Relu_0",
                                                          MEM_INIT_ALL_ZERO,
                                                          nullptr,
                                                          g_0_t127_model_activation_Relu_0_max_sizes,
                                                          4,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t127_model_activation_Relu_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_activation_Relu_relu_fwd_f32_n46_0_id;
    addNodeToGraph("relu_fwd_f32", {g_0_t124_model_batch_normalization_batchnorm_add_1_0}, {g_0_t127_model_activation_Relu_0}, nullptr, 0, "g_0_model_activation_Relu_relu_fwd_f32_n46_0", 0 /*graphIndex*/, &g_0_model_activation_Relu_relu_fwd_f32_n46_0_id);

    /*************
     * g_0_model_conv2d_1_Conv2D_spatial_convolution_n47_0 node
     * inputs:
     *     g_0_t127_model_activation_Relu_0[64, 256, 256, 32] (dtype=float32)
     *     g_0_t72_model_conv2d_1_conv2d_readvariableop_0[64, 64, 3, 3] (dtype=float32)
     * outputs:
     *     g_0_t128_model_conv2d_1_Conv2D_0[64, 256, 256, 32] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t72_model_conv2d_1_conv2d_readvariableop_0 tensor
    unsigned g_0_t72_model_conv2d_1_conv2d_readvariableop_0_max_sizes[] = {64,64,3,3};
    unsigned g_0_t72_model_conv2d_1_conv2d_readvariableop_0_min_sizes[] = {64,64,3,3};
    unsigned g_0_t72_model_conv2d_1_conv2d_readvariableop_0 = createTensors(1,
                                                                        INPUT_TENSOR,
                                                                        true,
                                                                        "g_0_t72_model_conv2d_1_conv2d_readvariableop_0",
                                                                        MEM_INIT_ALL_ZERO,
                                                                        nullptr,
                                                                        g_0_t72_model_conv2d_1_conv2d_readvariableop_0_max_sizes,
                                                                        4,
                                                                        syn_type_single,
                                                                        nullptr,
                                                                        0,
                                                                        0,
                                                                        nullptr,
                                                                        false,
                                                                        g_0_t72_model_conv2d_1_conv2d_readvariableop_0_min_sizes,
                                                                        synTensorType::DATA_TENSOR)[0];

    // create g_0_t128_model_conv2d_1_Conv2D_0 tensor
    unsigned g_0_t128_model_conv2d_1_Conv2D_0_max_sizes[] = {64,256,256,32};
    unsigned g_0_t128_model_conv2d_1_Conv2D_0_min_sizes[] = {64,256,256,32};
    unsigned g_0_t128_model_conv2d_1_Conv2D_0 = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          false,
                                                          "g_0_t128_model_conv2d_1_Conv2D_0",
                                                          MEM_INIT_ALL_ZERO,
                                                          nullptr,
                                                          g_0_t128_model_conv2d_1_Conv2D_0_max_sizes,
                                                          4,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t128_model_conv2d_1_Conv2D_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_conv2d_1_Conv2D_spatial_convolution_n47_0_id;
    unsigned char g_0_model_conv2d_1_Conv2D_spatial_convolution_n47_0_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,48,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("spatial_convolution", {g_0_t127_model_activation_Relu_0, g_0_t72_model_conv2d_1_conv2d_readvariableop_0}, {g_0_t128_model_conv2d_1_Conv2D_0}, (void*)g_0_model_conv2d_1_Conv2D_spatial_convolution_n47_0_params, 104, "g_0_model_conv2d_1_Conv2D_spatial_convolution_n47_0", 0 /*graphIndex*/, &g_0_model_conv2d_1_Conv2D_spatial_convolution_n47_0_id);

    /*************
     * g_0_model_batch_normalization_1_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n48_0 node
     * inputs:
     *     g_0_t128_model_conv2d_1_Conv2D_0[64, 256, 256, 32] (dtype=float32)
     * outputs:
     *     g_0_t129_model_batch_normalization_1_moments_mean_0[64, 1, 1, 1] (dtype=float32)
     *     g_0_t130_model_batch_normalization_1_moments_variance_0[64, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t129_model_batch_normalization_1_moments_mean_0 tensor
    unsigned g_0_t129_model_batch_normalization_1_moments_mean_0_max_sizes[] = {64,1,1,1};
    unsigned g_0_t129_model_batch_normalization_1_moments_mean_0_min_sizes[] = {64,1,1,1};
    unsigned g_0_t129_model_batch_normalization_1_moments_mean_0 = createTensors(1,
                                                                             OUTPUT_TENSOR,
                                                                             true,
                                                                             "g_0_t129_model_batch_normalization_1_moments_mean_0",
                                                                             MEM_INIT_ALL_ZERO,
                                                                             nullptr,
                                                                             g_0_t129_model_batch_normalization_1_moments_mean_0_max_sizes,
                                                                             4,
                                                                             syn_type_single,
                                                                             nullptr,
                                                                             0,
                                                                             0,
                                                                             nullptr,
                                                                             false,
                                                                             g_0_t129_model_batch_normalization_1_moments_mean_0_min_sizes,
                                                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_t130_model_batch_normalization_1_moments_variance_0 tensor
    unsigned g_0_t130_model_batch_normalization_1_moments_variance_0_max_sizes[] = {64,1,1,1};
    unsigned g_0_t130_model_batch_normalization_1_moments_variance_0_min_sizes[] = {64,1,1,1};
    unsigned g_0_t130_model_batch_normalization_1_moments_variance_0 = createTensors(1,
                                                                                 OUTPUT_TENSOR,
                                                                                 true,
                                                                                 "g_0_t130_model_batch_normalization_1_moments_variance_0",
                                                                                 MEM_INIT_ALL_ZERO,
                                                                                 nullptr,
                                                                                 g_0_t130_model_batch_normalization_1_moments_variance_0_max_sizes,
                                                                                 4,
                                                                                 syn_type_single,
                                                                                 nullptr,
                                                                                 0,
                                                                                 0,
                                                                                 nullptr,
                                                                                 false,
                                                                                 g_0_t130_model_batch_normalization_1_moments_variance_0_min_sizes,
                                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_1_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n48_0_id;
    unsigned char g_0_model_batch_normalization_1_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n48_0_params[] = {14,0,0,0};
    addNodeToGraph("norm_moments_fwd_f32", {g_0_t128_model_conv2d_1_Conv2D_0}, {g_0_t129_model_batch_normalization_1_moments_mean_0, g_0_t130_model_batch_normalization_1_moments_variance_0}, (void*)g_0_model_batch_normalization_1_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n48_0_params, 4, "g_0_model_batch_normalization_1_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n48_0", 0 /*graphIndex*/, &g_0_model_batch_normalization_1_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n48_0_id);

    runTest({g_0_t130_model_batch_normalization_1_moments_variance_0,
             g_0_t129_model_batch_normalization_1_moments_mean_0,
             g_0_t81_model_batch_normalization_moments_Squeeze_1_0,
             g_0_t79_model_batch_normalization_moments_Squeeze_0});
}

TEST_F_GC(NormsSlicingTest, layer_norm_as_norm_moments_unsliced)
{
    unsigned g_0_tensor_121_max_sizes[] = {768, 3072};
    unsigned g_0_tensor_121_min_sizes[] = {768, 3072};
    unsigned g_0_tensor_121             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_121",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_121_max_sizes,
                                            2,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_121_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast_max_sizes[] = {768, 3072};
    unsigned g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast_min_sizes[] = {768, 3072};
    unsigned g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_encoder_0_fc1_cast_f32_to_bf16_11525_0_id;
    unsigned char g_0_model_encoder_0_fc1_cast_f32_to_bf16_11525_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_121},
                   {g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast},
                   (void*)g_0_model_encoder_0_fc1_cast_f32_to_bf16_11525_0_params,
                   4,
                   "g_0_model_encoder_0_fc1_cast_f32_to_bf16_11525_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_fc1_cast_f32_to_bf16_11525_0_id);

    unsigned g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t_max_sizes[] = {3072, 768};
    unsigned g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t_min_sizes[] = {3072, 768};
    unsigned g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t_max_sizes,
                      2,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_encoder_0_fc1_transpose_11526_0_id;
    unsigned char g_0_model_encoder_0_fc1_transpose_11526_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {g_0_tensor_122_id_1669048_model_encoder_0_fc1_hpu__cast},
                   {g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t},
                   (void*)g_0_model_encoder_0_fc1_transpose_11526_0_params,
                   24,
                   "g_0_model_encoder_0_fc1_transpose_11526_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_fc1_transpose_11526_0_id);

    unsigned g_0_tensor_124_max_sizes[] = {768};
    unsigned g_0_tensor_124_min_sizes[] = {768};
    unsigned g_0_tensor_124             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_124",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_124_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_124_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast_max_sizes[] = {768};
    unsigned g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast_min_sizes[] = {768};
    unsigned g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_encoder_0_self_attn_out_proj_cast_f32_to_bf16_11527_0_id;
    unsigned char g_0_model_encoder_0_self_attn_out_proj_cast_f32_to_bf16_11527_0_params[] = {0, 0, 0, 0};
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_tensor_124},
                   {g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast},
                   (void*)g_0_model_encoder_0_self_attn_out_proj_cast_f32_to_bf16_11527_0_params,
                   4,
                   "g_0_model_encoder_0_self_attn_out_proj_cast_f32_to_bf16_11527_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_self_attn_out_proj_cast_f32_to_bf16_11527_0_id);

    unsigned g_0_tensor_130_id_1669032_model_encoder_0_self_attn_out_proj_aten__matmul_max_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_130_id_1669032_model_encoder_0_self_attn_out_proj_aten__matmul_min_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_130_id_1669032_model_encoder_0_self_attn_out_proj_aten__matmul =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_130_id_1669032_model_encoder_0_self_attn_out_proj_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_130_id_1669032_model_encoder_0_self_attn_out_proj_aten__matmul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_130_id_1669032_model_encoder_0_self_attn_out_proj_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add__max_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add__min_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add_ =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add_",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add__max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add__min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_add_fwd_bf16_88983_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_tensor_130_id_1669032_model_encoder_0_self_attn_out_proj_aten__matmul,
                    g_0_tensor_125_id_1669029_model_encoder_0_self_attn_out_proj_hpu__cast},
                   {g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add_},
                   nullptr,
                   0,
                   "g_0_add_fwd_bf16_88983_0",
                   0 /*graphIndex*/,
                   &g_0_add_fwd_bf16_88983_0_id);

    unsigned g_0_tensor_132_id_1669035_model_encoder_0_aten__view_max_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_132_id_1669035_model_encoder_0_aten__view_min_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_132_id_1669035_model_encoder_0_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_132_id_1669035_model_encoder_0_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_132_id_1669035_model_encoder_0_aten__view_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_132_id_1669035_model_encoder_0_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_encoder_0_reshape_11532_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_131_id_1669032_model_encoder_0_self_attn_out_proj_aten__add_},
                   {g_0_tensor_132_id_1669035_model_encoder_0_aten__view},
                   nullptr,
                   0,
                   "g_0_model_encoder_0_reshape_11532_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_reshape_11532_0_id);

    unsigned g_0_tensor_57_id_1668896_model_encoder_0_self_attn_k_proj_aten__view_max_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_57_id_1668896_model_encoder_0_self_attn_k_proj_aten__view_min_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_57_id_1668896_model_encoder_0_self_attn_k_proj_aten__view =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_57_id_1668896_model_encoder_0_self_attn_k_proj_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_57_id_1668896_model_encoder_0_self_attn_k_proj_aten__view_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_57_id_1668896_model_encoder_0_self_attn_k_proj_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_133_id_1669037_model_encoder_0_aten__add_max_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_133_id_1669037_model_encoder_0_aten__add_min_sizes[] = {768, 128, 4};
    unsigned g_0_tensor_133_id_1669037_model_encoder_0_aten__add =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_133_id_1669037_model_encoder_0_aten__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_133_id_1669037_model_encoder_0_aten__add_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_133_id_1669037_model_encoder_0_aten__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_encoder_0_add_fwd_bf16_11533_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_tensor_57_id_1668896_model_encoder_0_self_attn_k_proj_aten__view,
                    g_0_tensor_132_id_1669035_model_encoder_0_aten__view},
                   {g_0_tensor_133_id_1669037_model_encoder_0_aten__add},
                   nullptr,
                   0,
                   "g_0_model_encoder_0_add_fwd_bf16_11533_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_add_fwd_bf16_11533_0_id);

    unsigned  g_0_tensor_136_max_sizes[] = {768, 512, 1, 1};
    unsigned  g_0_tensor_136_min_sizes[] = {768, 512, 1, 1};
    unsigned  g_0_tensor_136             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_136",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_136_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_136_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_encoder_0_self_attn_layer_norm_reshape_11534_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_133_id_1669037_model_encoder_0_aten__add},
                   {g_0_tensor_136},
                   nullptr,
                   0,
                   "g_0_model_encoder_0_self_attn_layer_norm_reshape_11534_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_self_attn_layer_norm_reshape_11534_0_id);

    unsigned g_0_tensor_137_max_sizes[] = {768};
    unsigned g_0_tensor_137_min_sizes[] = {768};
    unsigned g_0_tensor_137             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_137",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_137_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_137_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_138_max_sizes[] = {768};
    unsigned g_0_tensor_138_min_sizes[] = {768};
    unsigned g_0_tensor_138             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_138",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_138_max_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_138_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_139_max_sizes[] = {768, 512, 1, 1};
    unsigned g_0_tensor_139_min_sizes[] = {768, 512, 1, 1};
    unsigned g_0_tensor_139             = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,
                                            "g_0_tensor_139",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_139_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_139_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes[] = {1,
                                                                                                                   512,
                                                                                                                   1,
                                                                                                                   1};
    unsigned g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes[] = {1,
                                                                                                                   512,
                                                                                                                   1,
                                                                                                                   1};
    unsigned g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes[] = {1,
                                                                                                                   512,
                                                                                                                   1,
                                                                                                                   1};
    unsigned g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes[] = {1,
                                                                                                                   512,
                                                                                                                   1,
                                                                                                                   1};
    unsigned g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_11537_0_id;
    unsigned char g_0_model_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_11537_0_params[] =
        {1, 114, 21, 141, 172, 197, 39, 55};
    addNodeToGraph("layer_norm_fwd_bf16",
                   {g_0_tensor_136, g_0_tensor_137, g_0_tensor_138},
                   {g_0_tensor_139,
                    g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm,
                    g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm},
                   (void*)g_0_model_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_11537_0_params,
                   8,
                   "g_0_model_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_11537_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_self_attn_layer_norm_layer_norm_fwd_bf16_11537_0_id);

    unsigned g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes[] = {768,
                                                                                                                   128,
                                                                                                                   4};
    unsigned g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes[] = {768,
                                                                                                                   128,
                                                                                                                   4};
    unsigned g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_encoder_0_self_attn_layer_norm_reshape_11538_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_139},
                   {g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm},
                   nullptr,
                   0,
                   "g_0_model_encoder_0_self_attn_layer_norm_reshape_11538_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_self_attn_layer_norm_reshape_11538_0_id);

    unsigned g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul_max_sizes[] = {3072, 128, 4};
    unsigned g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul_min_sizes[] = {3072, 128, 4};
    unsigned g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_encoder_0_fc1_batch_gemm_11539_0_id;
    unsigned char g_0_model_encoder_0_fc1_batch_gemm_11539_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_142_id_1669040_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm,
                    g_0_tensor_123_id_1669052_model_encoder_0_fc1_aten__t},
                   {g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul},
                   (void*)g_0_model_encoder_0_fc1_batch_gemm_11539_0_params,
                   2,
                   "g_0_model_encoder_0_fc1_batch_gemm_11539_0",
                   0 /*graphIndex*/,
                   &g_0_model_encoder_0_fc1_batch_gemm_11539_0_id);

    runTest({g_0_tesor_143_id_1669054_model_encoder_0_fc1_aten__matmul,
             g_0_tensor_140_id_1669042_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm,
             g_0_tensor_141_id_1669044_model_encoder_0_self_attn_layer_norm_aten__native_layer_norm});
}