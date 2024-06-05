#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "gc_gaudi_test_infra.h"

TEST_F_GC(SynTrainingTestInfra, reduce_sum_square_with_tpc_memset)
{
    GlobalConfTestSetter conf1("MAX_RMW_TENSOR_BYTES", "20");
    GlobalConfTestSetter conf2("FORCE_TPC_MEMSET_FOR_RMW", "true");

    unsigned inputDims[] = {12345};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, inputDims, 1, syn_type_single, nullptr, "input");

    unsigned outputDims[] = {1};

    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputDims, 1, syn_type_single, nullptr, "output");

    unsigned char params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_square_fwd_f32", {in}, {out}, (void*)params, 4, "reduce_sum");

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[out];
    float outputResult = *pOutputBuffer;
    float outputExpected = inputDims[0]; // Input is initialized to ones so expected output should be the size of the input

    ASSERT_EQ(outputExpected, outputResult) << "Mismatch for reduce sum output:"
                                            << " Expected:"             << outputExpected
                                            << " Result: "              << outputResult;
}
