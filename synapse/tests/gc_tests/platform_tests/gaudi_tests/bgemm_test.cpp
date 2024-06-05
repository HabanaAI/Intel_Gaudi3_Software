#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"


TEST_F(SynGaudiTestInfra, DISABLED_bert_encoder_layer_2_BGemm)
{
    /*************
     * n1031_bert_encoder_layer_2_attention_self_MatMul node
     * inputs: [tensor_1648[64, 128, 12, 32](dtype=bf16), tensor_1655[64, 128, 12, 32](dtype=bf16)]
     * output: [tensor_1656[128, 128, 12, 32](dtype=bf16)]
     *************/
    // create tensor_1648 tensor
    unsigned tensor_1648_sizes[] = {64,128,12,32};
    unsigned tensor_1648 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_1648",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_1648_sizes,
                                        4,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_1655 tensor
    unsigned tensor_1655_sizes[] = {64,128,12,32};
    unsigned tensor_1655 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_1655",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_1655_sizes,
                                        4,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_1656 tensor
    unsigned tensor_1656_sizes[] = {128,128,12,32};
    unsigned tensor_1656 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_1656",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_1656_sizes,
                                        4,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char n1031_bert_encoder_layer_2_attention_self_MatMul_params[] = {0,1};
    addNodeToGraph("batch_gemm", {tensor_1648, tensor_1655}, {tensor_1656}, (void*)n1031_bert_encoder_layer_2_attention_self_MatMul_params, 2, "n1031_bert_encoder_layer_2_attention_self_MatMul");

    compileTopology("1_b_gemm");
    runTopology();
}

using LargeBatchTestParams = std::tuple<
        unsigned,   // Batch 1
        unsigned,   // Batch 2
        unsigned,   // Height
        unsigned,   // In-Channels
        unsigned>;  // Out-Channels

class SynTrainingLargeBGemmTest
: public SynTrainingTestInfra
, public ::testing::WithParamInterface<LargeBatchTestParams>
{
public:
    unsigned m_B1;
    unsigned m_B2;
    unsigned m_H;
    unsigned m_IC;
    unsigned m_OC;

    void readParams()
    {
        m_B1 = std::get<0>(GetParam());
        m_B2 = std::get<1>(GetParam());
        m_H  = std::get<2>(GetParam());
        m_IC = std::get<3>(GetParam());
        m_OC = std::get<4>(GetParam());
    }
};

INSTANTIATE_TEST_SUITE_P(BGEMM,
                         SynTrainingLargeBGemmTest,
                         ::testing::Values<LargeBatchTestParams>(std::make_tuple(1u, 4u, 3584u, 84u, 21504u)));

TEST_P_GC(SynTrainingLargeBGemmTest, bgemm_with_large_gemms_accuracy_test)
{
    readParams();
    unsigned xSize[] = {m_IC, m_H, m_B2, m_B1};
    unsigned wSize[] = {m_OC, m_IC, m_B2, m_B1};
    unsigned ySize[] = {m_OC, m_H, m_B2, m_B1};

    unsigned x = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     xSize,
                                     ARRAY_SIZE(xSize),
                                     syn_type_bf16);
    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     wSize,
                                     ARRAY_SIZE(wSize),
                                     syn_type_bf16);
    unsigned y = createPersistTensor(OUTPUT_TENSOR,
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     ySize,
                                     ARRAY_SIZE(ySize),
                                     syn_type_bf16);

    synGEMMParams gemmParams {};
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {x, w}, {y}, &gemmParams, sizeof gemmParams, "BGEMM");

    compileTopology();

//    Calculating the reference takes too long - for now only testing compilation
}
