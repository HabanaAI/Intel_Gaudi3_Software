#include "syn_gaudi_two_run_compare_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"

class SynTrainingFuseBroadcastAndBGEMMTest
: public SynTrainingTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<TestSizes /*m_broadcastIn0Size*/,
                                                TestSizes /*m_broadcastOutSize*/,
                                                unsigned /*m_numOfTotalCastsNodes*/,
                                                TestSizes /*m_reshapeSizes*/,
                                                bool /*m_reshape*/>>
{
public:
    SynTrainingFuseBroadcastAndBGEMMTest()
    : m_broadcast0InSize(std::get<0>(GetParam())),
      m_broadcastOutSize(std::get<1>(GetParam())),
      m_numOfTotalCastsNodes(std::get<2>(GetParam())),
      m_reshapeSizes(std::get<3>(GetParam())),
      m_reshape(std::get<4>(GetParam()))
    {
    }

    void runAndCheckResults(const std::vector<unsigned>& outputToCompareIdx)
    {
        addConfigurationToRun(FIRST_RUN, "ENABLE_FUSE_BROADCAST_BGEMM", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_FUSE_BROADCAST_BGEMM", "true");
        compareRunsResults(outputToCompareIdx);
    }

    unsigned createBroadcastNode(TestSizes& bcastSizes)
    {
        auto bcastIn  = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "broadcast_in",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bcastSizes.data(),
                                     bcastSizes.size(),
                                     m_numOfTotalCastsNodes > 0 ? syn_type_bf16 : syn_type_single)[0];
        auto bcastOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "broadcast_out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      m_broadcastOutSize.data(),
                                      m_broadcastOutSize.size(),
                                      m_numOfTotalCastsNodes > 0 ? syn_type_bf16 : syn_type_single)[0];
        addNodeToGraph(NodeFactory::broadcastNodeTypeName, {bcastIn}, {bcastOut}, nullptr, 0, "broadcast");
        return bcastOut;
    }

    unsigned createBGEMMOtherInput()
    {
        auto addIn = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "bgemm_other_input",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   m_reshape ? m_reshapeSizes.data() : m_broadcastOutSize.data(),
                                   m_reshape ? m_reshapeSizes.size() : m_broadcastOutSize.size())[0];
        return addIn;
    }

    unsigned createCastNode(unsigned inputTensorIndex)
    {
        auto castOut = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     fmt::format("cast{}_out", m_castCounter++).c_str(),
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     m_broadcastOutSize.data(),
                                     m_broadcastOutSize.size(),
                                     syn_type_single)[0];
        addNodeToGraph("cast_bf16_to_f32",
                       {inputTensorIndex},
                       {castOut},
                       nullptr,
                       0,
                       fmt::format("cast{}", m_castCounter).c_str());
        return castOut;
    }

    unsigned createReshapeNode(unsigned inputTensorIndex)
    {
        auto reshapeOut = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "reshape_out",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        m_reshapeSizes.data(),
                                        m_reshapeSizes.size())[0];
        addNodeToGraph(NodeFactory::reshapeNodeTypeName, {inputTensorIndex}, {reshapeOut}, nullptr, 0, "reshape");
        return reshapeOut;
    }

    void runBroadcastProducerCase()
    {
        TensorIndices inputs;
        inputs.push_back(createBroadcastNode(m_broadcast0InSize));
        inputs.push_back(createBGEMMOtherInput());

        auto bgemmFirstInput = inputs[0];
        if (m_numOfTotalCastsNodes > 0)
        {
            auto newCastInput = bgemmFirstInput;
            for (int i = 0; i < m_numOfTotalCastsNodes; ++i)
            {
                newCastInput = createCastNode(newCastInput);
            }
            bgemmFirstInput = newCastInput;
        }

        if (m_reshape)
        {
            bgemmFirstInput = createReshapeNode(bgemmFirstInput);
        }
        inputs[0] = bgemmFirstInput;

        auto bGEMMOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "batch_gemm_out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      m_reshape ? m_reshapeSizes.data() : m_broadcastOutSize.data(),
                                      m_reshape ? m_reshapeSizes.size() : m_broadcastOutSize.size())[0];
        addNodeToGraph("batch_gemm", inputs, {bGEMMOut}, nullptr, 0, "batch_gemm");
        runAndCheckResults({bGEMMOut});
    }

    void runSingleTest() { runBroadcastProducerCase(); }

protected:
    TestSizes m_broadcast0InSize;
    TestSizes m_broadcastOutSize;
    unsigned  m_numOfTotalCastsNodes;
    TestSizes m_reshapeSizes;
    bool      m_reshape;
    unsigned  m_castCounter = 0;
};

// Currently in gaudi3 broadcast turns to logical broadcast and
// the following pattern fails in the mme run: logical broadcast -> bgemm, because mme doesn't accept operand with
// stride 0.
// When SW-170087 is solved, change GCFG_ENABLE_FUSE_BROADCAST_BGEMM to true for gaudi3 and enable tests for it.
TEST_P_GC(SynTrainingFuseBroadcastAndBGEMMTest, fuse_broadcast_bgemm, {synDeviceGaudi2})
{
    runSingleTest();
}

/*
    The following parameters create these kind of patterns:
    broadcast one/multiple dims -> non broadcasted bgemm
    broadcast one/multiple dims -> cast -> non broadcasted bgemm
    broadcast one/multiple dims -> reshape -> non broadcasted bgemm
    broadcast one/multiple dims -> cast -> reshape -> non broadcasted bgemm
*/

INSTANTIATE_TEST_SUITE_P(broadcast_producer_without_reshape,
                         SynTrainingFuseBroadcastAndBGEMMTest,
                         testing::Combine(testing::Values(TestSizes({16, 16, 1, 1, 1}), TestSizes({16, 16, 1, 8, 12})),
                                          testing::Values(TestSizes({16, 16, 4, 8, 12})),
                                          testing::Values(0, 1),  // more than one cast is not supported
                                          testing::Values(TestSizes({0})), // reshape sizes (no use if reshape is false)
                                          testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(broadcast_producer_with_reshape,
                         SynTrainingFuseBroadcastAndBGEMMTest,
                         testing::Combine(testing::Values(TestSizes({16, 16, 1, 1, 1}), TestSizes({16, 16, 1, 8, 12})),
                                          testing::Values(TestSizes({16, 16, 4, 8, 12})),
                                          testing::Values(0, 1),  // more than one cast is not supported
                                          testing::Values(TestSizes({16, 16, 32, 1, 12}), TestSizes({16, 16, 8, 2, 24})),
                                          testing::Values(true)));

TEST_F_GC(SynTrainingTwoRunCompareTest, fuse_broadcast_with_bgemm_and_handle_connecting_reshape, {synDeviceGaudi2})
{
    // Graph #0

    /*************
     * g_0_module_model_1_self_attn_broadcast_4034_0 node
     * inputs:
     *     g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice[128, 2048, 1, 8, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand[128, 2048, 8, 8, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice tensor
    unsigned g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice_max_sizes[] = {128, 2048, 1, 8, 1};
    unsigned g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice_min_sizes[] = {128, 2048, 1, 8, 1};
    unsigned g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand tensor
    unsigned g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand_max_sizes[] = {128, 2048, 8, 8, 1};
    unsigned g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand_min_sizes[] = {128, 2048, 8, 8, 1};
    unsigned g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand_max_sizes,
                      5,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_model_1_self_attn_broadcast_4034_0_id;
    addNodeToGraph("broadcast",
                   {g_0_tensor_202_id_20951_module_model_1_self_attn_aten__slice},
                   {g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand},
                   nullptr,
                   0,
                   "g_0_module_model_1_self_attn_broadcast_4034_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_1_self_attn_broadcast_4034_0_id);

    /*************
     * g_0_module_model_1_self_attn_reshape_4036_0 node
     * inputs:
     *     g_0_tensor_204_id_20957_module_model_1_self_attn_aten__clone[128, 2048, 8, 8, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view[128, 2048, 64, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view tensor
    unsigned g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view_max_sizes[] = {128, 2048, 64, 1};
    unsigned g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view_min_sizes[] = {128, 2048, 64, 1};
    unsigned g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_module_model_1_self_attn_reshape_4036_0_id;
    addNodeToGraph("reshape",
                   {g_0_tensor_203_id_20953_module_model_1_self_attn_hpu__expand},
                   {g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view},
                   nullptr,
                   0,
                   "g_0_module_model_1_self_attn_reshape_4036_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_1_self_attn_reshape_4036_0_id);

    /*************
     * g_0_module_model_1_self_attn_batch_gemm_4044_0 node
     * inputs:
     *     g_0_tensor_216[128, 1, 64, 1] (dtype=bf16)
     *     g_0_tensor_206_id_20981_module_model_1_self_attn_aten__transpose[2048, 128, 64, 1] (dtype=bf16)
     * outputs:
     *     g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul[2048, 1, 64, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_216 tensor
    unsigned g_0_tensor_216_max_sizes[] = {128, 1, 64, 1};
    unsigned g_0_tensor_216_min_sizes[] = {128, 1, 64, 1};
    unsigned g_0_tensor_216             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_216",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_216_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_216_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul tensor
    unsigned g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul_max_sizes[] = {2048, 1, 64, 1};
    unsigned g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul_min_sizes[] = {2048, 1, 64, 1};
    unsigned g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_module_model_1_self_attn_batch_gemm_4044_0_id;
    unsigned char g_0_module_model_1_self_attn_batch_gemm_4044_0_params[] = {0, 1};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_216, g_0_tensor_205_id_20960_module_model_1_self_attn_aten__view},
                   {g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul},
                   (void*)g_0_module_model_1_self_attn_batch_gemm_4044_0_params,
                   2,
                   "g_0_module_model_1_self_attn_batch_gemm_4044_0",
                   0 /*graphIndex*/,
                   &g_0_module_model_1_self_attn_batch_gemm_4044_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_FUSE_BROADCAST_BGEMM", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FUSE_BROADCAST_BGEMM", "true");

    compareRunsResults({g_0_tensor_217_id_20983_module_model_1_self_attn_aten__matmul});
}
