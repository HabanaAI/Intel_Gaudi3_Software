#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include <string>

class FlashAttentionFwdBwdTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<unsigned,  // B
                                                unsigned,  // H
                                                unsigned,  // T
                                                unsigned,  // D
                                                bool,      // Dropout enabled
                                                bool,      // slice b and h in cguid enabled
                                                bool,      // reshape softmax enabled
                                                bool>>     // Run test

{
public:
    void runTest();
    struct PrintToStringParamName
    {
        template<class ParamType>
        std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
        {
            std::stringstream ss;
            const unsigned    B           = std::get<0>(info.param);
            const unsigned    H           = std::get<1>(info.param);
            const unsigned    T           = std::get<2>(info.param);
            const unsigned    D           = std::get<3>(info.param);
            bool              doEnabled   = std::get<4>(info.param);
            bool              bhSlice     = std::get<5>(info.param);
            bool              reshapeSmax = std::get<6>(info.param);
            bool              runTest     = std::get<7>(info.param);
            ss << std::to_string(B) << "x" << std::to_string(H) << "x" << std::to_string(T) << "x" << std::to_string(D)
               << "__Dropout_" << std::to_string(doEnabled) << "__bhSlice_" << std::to_string(bhSlice)
               << "__reshapeSmax_" << std::to_string(reshapeSmax) << "__runTest_" << std::to_string(runTest);
            return ss.str();
        }
    };

protected:
    void SetUpTest() override
    {
        SynTrainingTestInfra::SetUpTest();
        m_sdpaSlicingPrevCfg = std::getenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE");
        setenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE", std::get<5>(GetParam()) ? "1" : "0", true);
        m_sdpaReshapePrevCfg = std::getenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE");
        setenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE",
               std::get<6>(GetParam()) ? "1" : "0",
               true);  // Use reshape optimizations for softmax
    }

    void TearDownTest() override
    {
        // Reset env var to its original value, or unset it if it wasn't previously set
        if (m_sdpaSlicingPrevCfg)
        {
            setenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE", m_sdpaSlicingPrevCfg, true);
        }
        else
        {
            unsetenv("PT_HPU_SDPA_BATCH_NUMHEADS_SLICE");
        }
        if (m_sdpaReshapePrevCfg)
        {
            setenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE", m_sdpaReshapePrevCfg, true);
        }
        else
        {
            unsetenv("PT_HPU_SDPA_RESHAPED_SOFTMAX_MODE");
        }
        SynTrainingTestInfra::TearDownTest();
    }

    const char* m_sdpaSlicingPrevCfg = nullptr;
    const char* m_sdpaReshapePrevCfg = nullptr;
};

void FlashAttentionFwdBwdTest::runTest()
{
    const unsigned B             = std::get<0>(GetParam());
    const unsigned H             = std::get<1>(GetParam());
    const unsigned T             = std::get<2>(GetParam());
    const unsigned D             = std::get<3>(GetParam());
    auto           dropoutRatio  = std::get<4>(GetParam()) ? 0.1 : 0.0;
    auto           shouldRunTest = std::get<7>(GetParam());

    // create fwd tensors
    unsigned sizes[] = {D, T, H, B};
    unsigned Q =
        createTensors(1, INPUT_TENSOR, true, "Q", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];
    unsigned K =
        createTensors(1, INPUT_TENSOR, true, "K", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];
    unsigned V =
        createTensors(1, INPUT_TENSOR, true, "V", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];
    unsigned O =
        createTensors(1, OUTPUT_TENSOR, true, "O", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];

    // create stats tensors
    unsigned stats_sizes[] = {1, T, H, B};
    unsigned stat1         = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "stat1",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   stats_sizes,
                                   4,
                                   syn_type_bf16)[0];
    unsigned stat2         = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "stat2",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   stats_sizes,
                                   4,
                                   syn_type_single)[0];

    synNodeId       sdpa_recomp_fwd_id;
    ns_Sdpa::Params sdpa_recomp_fwd_params = {0};
    sdpa_recomp_fwd_params.scale           = 1.0 / sqrt(D);
    sdpa_recomp_fwd_params.dropout.ratio   = dropoutRatio;
    sdpa_recomp_fwd_params.is_causal       = true;

    // create gradient tensors
    unsigned dO =
        createTensors(1, INPUT_TENSOR, true, "dO", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];
    unsigned dQ =
        createTensors(1, INPUT_TENSOR, true, "dQ", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];
    unsigned dK =
        createTensors(1, INPUT_TENSOR, true, "dK", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];
    unsigned dV =
        createTensors(1, INPUT_TENSOR, true, "dV", MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 4, syn_type_bf16)[0];

    synNodeId       sdpa_recomp_bwd_id;
    ns_Sdpa::Params sdpa_recomp_bwd_params = {0};
    sdpa_recomp_bwd_params.scale           = 1.0 / sqrt(D);
    sdpa_recomp_bwd_params.dropout.ratio   = dropoutRatio;
    sdpa_recomp_bwd_params.is_causal       = true;
    TensorIndices fwdInputs                = {Q, K, V};
    TensorIndices fwdOutputs               = {O, stat1, stat2};
    TensorIndices bwdInputs                = {dO, Q, K, V, INVALID_TENSOR_INDEX, stat1, stat2};
    TensorIndices bwdOutputs               = {dQ, dK, dV};
    if (dropoutRatio > 0)
    {
        unsigned dropoutMaskSizes[] = {1};
        unsigned seed_in            = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "seed_in",
                                         MEM_INIT_RANDOM_POSITIVE,
                                         nullptr,
                                         dropoutMaskSizes,
                                         1,
                                         syn_type_int32)[0];
        unsigned seed_out           = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "seed_out",
                                          MEM_INIT_RANDOM_POSITIVE,
                                          nullptr,
                                          dropoutMaskSizes,
                                          1,
                                          syn_type_int32)[0];
        fwdInputs.push_back(INVALID_TENSOR_INDEX);
        fwdInputs.push_back(seed_in);
        fwdOutputs.push_back(seed_out);
        bwdInputs.push_back(seed_in);
    }
    addNodeToGraph("sdpa_recomp_fwd_bf16",
                   fwdInputs,
                   fwdOutputs,
                   static_cast<void*>(&sdpa_recomp_fwd_params),
                   sizeof(sdpa_recomp_fwd_params),
                   "sdpa_recomp_fwd",
                   0 /*graphIndex*/,
                   &sdpa_recomp_fwd_id);
    addNodeToGraph("sdpa_recomp_bwd_bf16",
                   bwdInputs,
                   bwdOutputs,
                   static_cast<void*>(&sdpa_recomp_bwd_params),
                   sizeof(sdpa_recomp_bwd_params),
                   "sdpa_recomp_bwd",
                   0 /*graphIndex*/,
                   &sdpa_recomp_bwd_id);
    synNodeSetDeterministic(getGraph(0).graphHandle, sdpa_recomp_fwd_id, true);
    synNodeSetDeterministic(getGraph(0).graphHandle, sdpa_recomp_bwd_id, true);

    compileTopology();
    if (shouldRunTest) runTopology();
}

TEST_P_GC(FlashAttentionFwdBwdTest, fwd_bwd)
{
    runTest();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_fwd_bwd,
                         FlashAttentionFwdBwdTest,
                         testing::Combine(testing::Values(1, 2),              // B
                                          testing::Values(2, 16),             // H
                                          testing::Values(1024, 2048, 4096),  // N
                                          testing::Values(128),               // D
                                          testing::Values(true, false),       // Dropout enabled
                                          testing::Values(true, false),       // Slice B&H in cguid
                                          testing::Values(true, false),       // Reshape softmax
                                          testing::Values(true, false)),      // Run test
                         FlashAttentionFwdBwdTest::PrintToStringParamName {});
