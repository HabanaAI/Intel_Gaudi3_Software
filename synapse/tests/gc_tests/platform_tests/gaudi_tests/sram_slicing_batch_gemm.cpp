#include "gc_gaudi_test_infra.h"
#include "log_manager.h"
#include "node_factory.h"
#include "global_conf_manager.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "gtest/gtest.h"
#include <cmath>

class SynGaudiBatchGemmSRAMSlicing : public SynGaudiTestInfra
{
};

TEST_F_GC(SynGaudiBatchGemmSRAMSlicing, batch_gemm_max_batch_size_L2)
{
    ScopedConfigurationChange("SRAM_SLICER_FORCE_BATCHGEMM_MAX_BATCH_SIZE", "true");
    synGEMMParams           params = synGEMMParams();
    synSplitParams          splitParams;
    synConcatenateParams    concatParams;
    unsigned                m = 24, k = 12, n = 16, batch1 = 16, batch2 = 16;
    std::array<unsigned, 4> input1Sizes  = {k, m, batch1, batch2};
    std::array<unsigned, 4> input2Sizes  = {n, k, batch1, batch2};
    std::array<unsigned, 4> outputSizes  = {n, m, batch1, batch2};
    std::array<unsigned, 4> input1Batch2 = {k, m, batch1, 1};
    std::array<unsigned, 4> input2Batch2 = {n, k, batch1, 1};
    std::array<unsigned, 4> outputBatch2 = {n, m, batch1, 1};
    std::array<unsigned, 4> input1Batch1 = {k, m, 1, 1};
    std::array<unsigned, 4> input2Batch1 = {n, k, 1, 1};
    std::array<unsigned, 4> outputBatch1 = {n, m, 1, 1};

    unsigned input1    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, input1Sizes.data());
    unsigned input2    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, input2Sizes.data());
    unsigned output    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes.data());
    unsigned outputRef = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes.data());

    addNodeToGraph("batch_gemm", {input1, input2}, {output}, (void*) &params, sizeof(params));
    std::vector<unsigned> split1B2;
    std::vector<unsigned> split2B2;
    std::vector<unsigned> concatB2;
    for (unsigned unused_i = 0; unused_i < batch2; ++unused_i)
    {
        unsigned input1B2    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, input1Batch2.data());
        unsigned input2B2    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, input2Batch2.data());
        unsigned outputRefB2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputBatch2.data());
        split1B2.push_back(input1B2);
        split2B2.push_back(input2B2);
        concatB2.push_back(outputRefB2);

        std::vector<unsigned> split1B1;
        std::vector<unsigned> split2B1;
        std::vector<unsigned> concatB1;
        for (unsigned unused_j = 0; unused_j < batch1; ++unused_j)
        {
            unsigned input1B1    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, input1Batch1.data());
            unsigned input2B1    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, input2Batch1.data());
            unsigned outputRefB1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputBatch1.data());
            split1B1.push_back(input1B1);
            split2B1.push_back(input2B1);
            concatB1.push_back(outputRefB1);

            addNodeToGraph("gemm", {input1B1, input2B1}, {outputRefB1}, (void*) &params, sizeof(params));
        }
        splitParams.axis  = 2;
        concatParams.axis = 2;
        addNodeToGraph("split", {input1B2}, split1B1, (void*) &splitParams, sizeof(splitParams));
        addNodeToGraph("split", {input2B2}, split2B1, (void*) &splitParams, sizeof(splitParams));
        addNodeToGraph("concat", concatB1, {outputRefB2}, (void*) &concatParams, sizeof(concatParams));
    }
    splitParams.axis  = 3;
    concatParams.axis = 3;
    addNodeToGraph("split", {input1}, split1B2, (void*) &splitParams, sizeof(splitParams));
    addNodeToGraph("split", {input2}, split2B2, (void*) &splitParams, sizeof(splitParams));
    addNodeToGraph("concat", concatB2, {outputRef}, (void*) &concatParams, sizeof(concatParams));

    compileAndRun();

    float* outputBuffer    = (float*) m_hostBuffers[output];
    float* outputRefBuffer = (float*) m_hostBuffers[outputRef];

    for (unsigned i = 0; i < m * n * batch1 * batch2; ++i)
    {
        ASSERT_EQ(*outputBuffer, *outputRefBuffer) << "mismatch in index: " << i;
        ++outputBuffer;
        ++outputRefBuffer;
    }
}
