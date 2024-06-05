#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

class SynGaudiTPCFuserTest : public SynTrainingTestInfra
{
};

TEST_F_GC(SynGaudiTPCFuserTest, tpc_fuser_one_cluster_one_persistent_intermediate_fwd)
{
    unsigned in1_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned in1_2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned out1  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

    unsigned in2_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned in2_2 = connectOutputTensorToInputTensor(out1);
    unsigned out2  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

    unsigned in3_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned in3_2 = connectOutputTensorToInputTensor(out2);
    unsigned out3  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

    unsigned in4_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned in4_2 = connectOutputTensorToInputTensor(out3);
    unsigned out4  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);

    addNodeToGraph("add_fwd_f32", {in1_1, in1_2}, {out1});
    addNodeToGraph("add_fwd_f32", {in2_1, in2_2}, {out2});
    addNodeToGraph("add_fwd_f32", {in3_1, in3_2}, {out3});
    addNodeToGraph("add_fwd_f32", {in4_1, in4_2}, {out4});

    compileAndRun();

    float* input1 = (float*)m_hostBuffers[in1_1];
    float* input2 = (float*)m_hostBuffers[in1_2];
    float* input3 = (float*)m_hostBuffers[in2_1];
    float* input4 = (float*)m_hostBuffers[in3_1];
    float* input5 = (float*)m_hostBuffers[in4_1];

    float* output3 = (float*)m_hostBuffers[out3];
    float* output  = (float*)m_hostBuffers[out4];

    for (uint64_t idx = 0; idx < getDefaultNumberOfElements(); idx++)
    {
        float expected_out3 = *input1 + *input2 + *input3 + *input4;

        ASSERT_EQ(expected_out3, *output3) << "OUTPUT3: Mismatch for at index " << idx
                                           << " |Expected:" << expected_out3
                                           << " |Result: " << *output3
                                           << " |Operands: "
                                           << *input1 << ", " << *input2 << ", " << *input3 << ", " << *input4;

        float expected = expected_out3 + *input5;

        ASSERT_EQ(expected, *output) << "OUTPUT: Mismatch for at index " << idx
                                     << " |Expected:" << expected
                                     << " |Result: " << *output
                                     << " |operand " << *output3 << ", " << *input5;

        input1++;
        input2++;
        input3++;
        input4++;
        input5++;

        output3++;
        output++;
    }
}

TEST_F_GC(SynGaudiTPCFuserTest, tpc_fuser_fuse_relu)
{
    // Graph have two TPC nodes:  [relu_fwd]->[relu_bwd]
    // that will be fused to one TPC node

    ns_ReluKernel::Params reluParams = {0};
    reluParams.threshold.i = 0;
    unsigned fwdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned fwdOut = createTensor(OUTPUT_TENSOR);
    addNodeToGraph("relu_fwd_f32", {fwdIn}, {fwdOut}, &reluParams, sizeof(reluParams));

    unsigned bwdIn1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE); // input data (grad) on which relu backward is computed
    unsigned bwdIn2 = connectOutputTensorToInputTensor(fwdOut);
    unsigned bwdOut = createPersistTensor(OUTPUT_TENSOR);

    reluParams.threshold.f = 0.;
    addNodeToGraph("relu_bwd_f32", {bwdIn1, bwdIn2}, {bwdOut}, &reluParams, sizeof(reluParams));

    compileTopology();
    runTopology();

    float* pFwdInput  = (float*)m_hostBuffers[fwdIn];
    float* pBwdInput  = (float*)m_hostBuffers[bwdIn1];
    float* pBwdOutput = (float*)m_hostBuffers[bwdOut];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        float expectedResult = (*pFwdInput > 0.) ? *pBwdInput : 0;
        ASSERT_EQ(expectedResult, *pBwdOutput) << "Mismatch for at index " << i
                                               << " Expected:"             << expectedResult
                                               << " BwdOutput: "           << *pBwdOutput
                                               << " FwdInput: "            << *pFwdInput
                                               << " BwdInput "             << *pBwdInput;
        pFwdInput++;
        pBwdInput++;
        pBwdOutput++;
    }
}
