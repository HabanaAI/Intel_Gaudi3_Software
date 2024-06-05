#include "gc_gaudi_test_infra.h"
#include "gaudi_tests/auto_generated/graph_manager.h"
#include "node_factory.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "tensor.h"
#include "types.h"
#include "syn_singleton.hpp"
#include "gtest/gtest.h"

enum class FuseBroadcastsTpcTestMode
{
    BROADCAST_PRODUCER           = 0,  // Binary elementwise op as consumer
    BROADCAST_CONSUMER_BY_PARAMS = 1,  // Constant op as producer
    BROADCAST_CONSUMER_BY_DATA   = 2   // Constant op as producer
};

class SynTrainingFuseBroadcastAndTpcTest
: public SynTrainingTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<FuseBroadcastsTpcTestMode,
                                                TestSizes /*m_broadcastIn0Size*/,
                                                TestSizes /*m_broadcastIn1Size*/,
                                                TestSizes /*m_broadcastOutSize*/,
                                                bool /*broadcastIn0*/,
                                                bool /*broadcastIn1*/>>
{
public:
    SynTrainingFuseBroadcastAndTpcTest()
    : m_mode(std::get<0>(GetParam())),
      m_broadcast0InSize(std::get<1>(GetParam())),
      m_broadcast1InSize(std::get<2>(GetParam())),
      m_broadcastOutSize(std::get<3>(GetParam())),
      m_broadcastIn0(std::get<4>(GetParam())),
      m_broadcastIn1(std::get<5>(GetParam()))
    {
    }

    void runSingleTest()
    {
        switch (m_mode)
        {
            case FuseBroadcastsTpcTestMode::BROADCAST_PRODUCER:
                runBroadcastProducerCase();
                break;
            case FuseBroadcastsTpcTestMode::BROADCAST_CONSUMER_BY_DATA:
                runBroadcastConsumerByDataCase();
                break;
            case FuseBroadcastsTpcTestMode::BROADCAST_CONSUMER_BY_PARAMS:
                runBroadcastConsumerByParamsCase();
                break;
            default:
                ASSERT_TRUE(false);
        }
    }

    void runAndCheckResults(const std::vector<unsigned>& outputToCompareIdx)
    {
        addConfigurationToRun(FIRST_RUN, "ENABLE_BROADCAST_TPC_FUSION", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_BROADCAST_TPC_FUSION", "true");
        compareRunsResults(outputToCompareIdx);
    }

    unsigned createBroadcastNode(TestSizes& bcastSizes)
    {
        static unsigned counter  = 0;
        auto            bcastIn  = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     ("broadcast_in" + std::to_string(counter++)).c_str(),
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     bcastSizes.data())[0];
        auto            bcastOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      ("broadcast_in" + std::to_string(counter++)).c_str(),
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      m_broadcastOutSize.data())[0];
        addNodeToGraph(NodeFactory::broadcastNodeTypeName,
                       {bcastIn},
                       {bcastOut},
                       nullptr,
                       0,
                       ("broadcast_" + std::to_string(counter++)).c_str());
        return bcastOut;
    }

    unsigned createAddInput()
    {
        static unsigned counter = 0;
        auto            addIn   = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   ("add_in_" + std::to_string(counter++)).c_str(),
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   m_broadcastOutSize.data())[0];
        return addIn;
    }

    void runBroadcastProducerCase()
    {
        if (!m_broadcastIn0 && !m_broadcastIn1) GTEST_SKIP() << "Skipping no broadcast combination";

        TensorIndices inputs;
        inputs.push_back(m_broadcastIn0 ? createBroadcastNode(m_broadcast0InSize) : createAddInput());
        inputs.push_back(m_broadcastIn1 ? createBroadcastNode(m_broadcast1InSize) : createAddInput());

        auto addOut =
            createTensors(1, OUTPUT_TENSOR, true, "add_out", MEM_INIT_ALL_ZERO, nullptr, m_broadcastOutSize.data())[0];
        addNodeToGraph("add_fwd_f32", inputs, {addOut}, nullptr, 0, "add");
        runAndCheckResults({addOut});
    }

    void runBroadcastConsumerByDataCase()
    {
        unsigned constInTensor  = createTensors(1,
                                               INPUT_TENSOR,
                                               true,
                                               "const_in",
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               m_broadcast0InSize.data())[0];
        unsigned constOutTensor = createTensors(1,
                                                OUTPUT_TENSOR,
                                                true,
                                                "const_out",
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                m_broadcast0InSize.data())[0];
        addNodeToGraph("constant_f32", {constInTensor}, {constOutTensor}, nullptr, 0, "constant");
        auto bcastOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "broadcast_out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      m_broadcastOutSize.data())[0];
        addNodeToGraph(NodeFactory::broadcastNodeTypeName, {constOutTensor}, {bcastOut}, nullptr, 0, "broadcast");
        runAndCheckResults({bcastOut});
    }

    void runBroadcastConsumerByParamsCase()
    {
        ns_ConstantKernel::Params constParams;
        constParams.constant.f = 0.3;
        unsigned constTensor   = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "const_out",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             m_broadcast0InSize.data())[0];
        addNodeToGraph("constant_f32", {}, {constTensor}, &constParams, sizeof(ns_ConstantKernel::Params), "constant");
        auto bcastOut = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "broadcast_out",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      m_broadcastOutSize.data())[0];
        addNodeToGraph(NodeFactory::broadcastNodeTypeName, {constTensor}, {bcastOut}, nullptr, 0, "broadcast");
        runAndCheckResults({bcastOut});
    }

protected:
    const FuseBroadcastsTpcTestMode m_mode;
    TestSizes                       m_broadcast0InSize;
    TestSizes                       m_broadcast1InSize;
    TestSizes                       m_broadcastOutSize;
    bool                            m_broadcastIn0;
    bool                            m_broadcastIn1;
};

TEST_P_GC(SynTrainingFuseBroadcastAndTpcTest, fuse_broadcast_tpc)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(broadcast_producer,
                         SynTrainingFuseBroadcastAndTpcTest,
                         testing::Combine(testing::Values(FuseBroadcastsTpcTestMode::BROADCAST_PRODUCER),
                                          testing::Values(TestSizes({64, 16, 1, 1, 1}), TestSizes({1, 16, 1, 12, 4})),
                                          testing::Values(TestSizes({64, 1, 32, 12, 4}), TestSizes({1, 1, 1, 12, 4})),
                                          testing::Values(TestSizes({64, 16, 32, 12, 4})),
                                          testing::Values(true, false),
                                          testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(broadcast_consumer,
                         SynTrainingFuseBroadcastAndTpcTest,
                         testing::Combine(testing::Values(FuseBroadcastsTpcTestMode::BROADCAST_CONSUMER_BY_DATA,
                                                          FuseBroadcastsTpcTestMode::BROADCAST_CONSUMER_BY_PARAMS),
                                          testing::Values(TestSizes({1, 1, 1, 1, 1}), TestSizes({1, 16, 1, 12, 4})),
                                          testing::Values(TestSizes({})),
                                          testing::Values(TestSizes({64, 16, 32, 12, 4})),
                                          testing::Values(true),
                                          testing::Values(true)));

// Subgraph of prophetnet
TEST_F_GC(SynTrainingTwoRunCompareTest, fuse_bcast_cast_tpc)
{
    // Graph #0

    /*************
     * g_0__broadcast_52089_0 node
     * inputs:
     *     g_0_tensor_5124_id_117590_aten__neg[1, 3328] (dtype=bf16)
     *     g_0_tensor_5125[250012, 3328] (dtype=uint32) (shape tensor)
     * outputs:
     *     g_0_tensor_5126_id_117593_hpu__expand[250012, 3328] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/
    unsigned bcastDimSize = 1024;  // originally was 250012

    // create g_0_tensor_5124_id_117590_aten__neg tensor
    unsigned g_0_tensor_5124_id_117590_aten__neg_max_sizes[] = {1, 3328};
    unsigned g_0_tensor_5124_id_117590_aten__neg_min_sizes[] = {1, 3328};
    unsigned g_0_tensor_5124_id_117590_aten__neg             = createTensors(1,
                                                                 INPUT_TENSOR,
                                                                 true,
                                                                 "g_0_tensor_5124_id_117590_aten__neg",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_5124_id_117590_aten__neg_max_sizes,
                                                                 2,
                                                                 syn_type_bf16,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_5124_id_117590_aten__neg_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_5125 tensor
    unsigned g_0_tensor_5125_max_sizes[] = {bcastDimSize, 3328};
    unsigned g_0_tensor_5125_min_sizes[] = {bcastDimSize, 3328};
    unsigned g_0_tensor_5125             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_5125",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_5125_max_sizes,
                                             2,
                                             syn_type_uint32,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_5125_min_sizes,
                                             synTensorType::SHAPE_TENSOR)[0];

    // create g_0_tensor_5126_id_117593_hpu__expand tensor
    unsigned  g_0_tensor_5126_id_117593_hpu__expand_max_sizes[] = {bcastDimSize, 3328};
    unsigned  g_0_tensor_5126_id_117593_hpu__expand_min_sizes[] = {bcastDimSize, 3328};
    unsigned  g_0_tensor_5126_id_117593_hpu__expand             = createTensors(1,
                                                                   OUTPUT_TENSOR,
                                                                   false,
                                                                   "g_0_tensor_5126_id_117593_hpu__expand",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_tensor_5126_id_117593_hpu__expand_max_sizes,
                                                                   2,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_tensor_5126_id_117593_hpu__expand_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__broadcast_52089_0_id;
    addNodeToGraph("broadcast",
                   {g_0_tensor_5124_id_117590_aten__neg, g_0_tensor_5125},
                   {g_0_tensor_5126_id_117593_hpu__expand},
                   nullptr,
                   0,
                   "g_0__broadcast_52089_0",
                   0 /*graphIndex*/,
                   &g_0__broadcast_52089_0_id);

    /*************
     * g_0__cast_bf16_to_f32_52090_0 node
     * inputs:
     *     g_0_tensor_5126_id_117593_hpu__expand[250012, 3328] (dtype=bf16)
     * outputs:
     *     g_0_tensor_5127_id_117596_hpu__cast[250012, 3328] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5127_id_117596_hpu__cast tensor
    unsigned      g_0_tensor_5127_id_117596_hpu__cast_max_sizes[] = {bcastDimSize, 3328};
    unsigned      g_0_tensor_5127_id_117596_hpu__cast_min_sizes[] = {bcastDimSize, 3328};
    unsigned      g_0_tensor_5127_id_117596_hpu__cast             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_tensor_5127_id_117596_hpu__cast",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_5127_id_117596_hpu__cast_max_sizes,
                                                                 2,
                                                                 syn_type_single,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_5127_id_117596_hpu__cast_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__cast_bf16_to_f32_52090_0_id;
    unsigned char g_0__cast_bf16_to_f32_52090_0_params[] = {4, 0, 0, 0};
    addNodeToGraph("cast_bf16_to_f32",
                   {g_0_tensor_5126_id_117593_hpu__expand},
                   {g_0_tensor_5127_id_117596_hpu__cast},
                   (void*)g_0__cast_bf16_to_f32_52090_0_params,
                   4,
                   "g_0__cast_bf16_to_f32_52090_0",
                   0 /*graphIndex*/,
                   &g_0__cast_bf16_to_f32_52090_0_id);

    /*************
     * g_0__add_fwd_f32_52091_0 node
     * inputs:
     *     g_0_tensor_5127_id_117596_hpu__cast[250012, 3328] (dtype=float32)
     *     g_0_tensor_5110[250012, 3328] (dtype=float32)
     * outputs:
     *     g_0_tensor_5128_id_117608_aten__add[250012, 3328] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5110 tensor
    unsigned g_0_tensor_5110_max_sizes[] = {bcastDimSize, 3328};
    unsigned g_0_tensor_5110_min_sizes[] = {bcastDimSize, 3328};
    unsigned g_0_tensor_5110             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_5110",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_5110_max_sizes,
                                             2,
                                             syn_type_single,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_5110_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_5128_id_117608_aten__add tensor
    unsigned  g_0_tensor_5128_id_117608_aten__add_max_sizes[] = {bcastDimSize, 3328};
    unsigned  g_0_tensor_5128_id_117608_aten__add_min_sizes[] = {bcastDimSize, 3328};
    unsigned  g_0_tensor_5128_id_117608_aten__add             = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 true,
                                                                 "g_0_tensor_5128_id_117608_aten__add",
                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                 nullptr,
                                                                 g_0_tensor_5128_id_117608_aten__add_max_sizes,
                                                                 2,
                                                                 syn_type_single,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_5128_id_117608_aten__add_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__add_fwd_f32_52091_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_tensor_5127_id_117596_hpu__cast, g_0_tensor_5110},
                   {g_0_tensor_5128_id_117608_aten__add},
                   nullptr,
                   0,
                   "g_0__add_fwd_f32_52091_0",
                   0 /*graphIndex*/,
                   &g_0__add_fwd_f32_52091_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_BROADCAST_TPC_FUSION", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_BROADCAST_TPC_FUSION", "false");

    addConfigurationToRun(FIRST_RUN, "RUN_TPC_FUSER", "false");
    addConfigurationToRun(SECOND_RUN, "RUN_TPC_FUSER", "false");

    compareRunsResults({g_0_tensor_5128_id_117608_aten__add});
}

TEST_F_GC(SynTrainingTestInfra, double_input_broadcast_tpc)
{
    unsigned broadcastedSizes[] = {512, 32, 32, 8};
    unsigned inSizes[]          = {1, 1, 1, 8};
    unsigned dim                = 4;

    unsigned bcastIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSizes, dim);
    unsigned mulOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, broadcastedSizes, dim);
    unsigned mulIn   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, broadcastedSizes, dim);

    addNodeToGraph("broadcast", {bcastIn}, {mulIn}, nullptr, 0, "broadcast");
    addNodeToGraph("mult_fwd_f32", {mulIn, mulIn}, {mulOut}, nullptr, 0, "mul");

    compileAndRun();

    const float* inData     = castHostBuffer<float>(bcastIn);  // [1,1,1,8]
    const float* mulOutData = castHostBuffer<float>(mulOut);   // [512, 32, 32, 8]

    // validate correctness
    uint64_t elementIdx = 0;
    for (uint64_t nn = 0; nn < broadcastedSizes[3]; nn++)
    {
        for (uint64_t hh = 0; hh < broadcastedSizes[2]; hh++)
        {
            for (uint64_t ww = 0; ww < broadcastedSizes[1]; ww++)
            {
                for (uint64_t cc = 0; cc < broadcastedSizes[0]; cc++)
                {
                    float out      = mulOutData[elementIdx++];
                    float expected = inData[nn] * inData[nn];
                    ASSERT_FLOAT_EQ(out, expected) << "mismatch at offset " << elementIdx - 1;
                }
            }
        }
    }
}

// Subgraph of mid-journey
TEST_F_GC(SynTrainingTestInfra, swap_broadcast_tpc)
{
    unsigned broadcastedSizes[] = {512, 32, 32, 8};
    unsigned in0Sizes[]         = {1};
    unsigned in1Sizes[]         = {1, 1, 1, 8};
    unsigned dim                = 4;

    unsigned bcastIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, in1Sizes, dim);
    unsigned divIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, in0Sizes, 1);
    unsigned addIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, broadcastedSizes, dim);
    unsigned addOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, broadcastedSizes, dim);
    unsigned bcastOut   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, broadcastedSizes, dim);
    unsigned divOut     = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, broadcastedSizes, dim);
    unsigned bcastShape = createShapeTensor(INPUT_TENSOR, broadcastedSizes, broadcastedSizes, dim);

    addNodeToGraph("broadcast", {bcastIn, bcastShape}, {bcastOut}, nullptr, 0, "broadcast");
    addNodeToGraph("div_f32", {bcastOut, divIn}, {divOut}, nullptr, 0, "div");
    addNodeToGraph("add_fwd_f32", {divOut, addIn}, {addOut}, nullptr, 0, "add");

    compileAndRun();

    const float* divInData   = castHostBuffer<float>(divIn);    // [1]
    const float* bcastInData = castHostBuffer<float>(bcastIn);  // [1,1,1,8]
    const float* addInData   = castHostBuffer<float>(addIn);    // [512, 32, 32, 8]
    const float* addOutData  = castHostBuffer<float>(addOut);   // [512, 32, 32, 8]

    // validate correctness
    uint64_t strides[4] = {1};
    strides[1]          = strides[0] * broadcastedSizes[0];
    strides[2]          = strides[1] * broadcastedSizes[1];
    strides[3]          = strides[2] * broadcastedSizes[2];
    for (uint64_t nn = 0; nn < broadcastedSizes[3]; nn++)
    {
        for (uint64_t hh = 0; hh < broadcastedSizes[2]; hh++)
        {
            for (uint64_t ww = 0; ww < broadcastedSizes[1]; ww++)
            {
                for (uint64_t cc = 0; cc < broadcastedSizes[0]; cc++)
                {
                    uint64_t offset   = cc * strides[0] + ww * strides[1] + hh * strides[2] + nn * strides[3];
                    float    expected = (bcastInData[nn] / divInData[0]) + addInData[offset];
                    ASSERT_FLOAT_EQ(addOutData[offset], expected) << "mismatch at offset " << offset;
                }
            }
        }
    }

    // check broadcast was fused
    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(getGraph(0).graphHandle);
    for (const NodePtr& n : graph->getNodes())
    {
        ASSERT_TRUE(n->getNodeName().find("broadcast") == std::string::npos);
    }
}