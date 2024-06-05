#include "graph_compiler/habana_nodes/node_factory.h"
#include "gc_gaudi_test_infra.h"

/*
+
|[cd,w,h]
+------>------+ [cd,w,h] +-------+       +----+
        | ADD +--------->+Reshape+------>+GEMM|
+------>------+          +-------+       +----+
|[cd,1,1]
+
*/
class SynTrainingReshapeAlignTest : public SynTrainingTestInfra
{
protected:
    void runTest(unsigned inDims[3], unsigned gemmDims[2], unsigned wDims[2])
    {
    unsigned aSize[] = {inDims[0], inDims[1], inDims[2]};
    unsigned bSize[] = {inDims[0], 1, 1};
    unsigned gemmSize[] = {gemmDims[0], gemmDims[1]};
    unsigned wSize[] = {wDims[0], wDims[1]};
    unsigned oSize[] = {wDims[0], gemmDims[1]};

    unsigned aIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, aSize, ARRAY_SIZE(aSize), syn_type_float);
    unsigned bIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, bSize, ARRAY_SIZE(bSize), syn_type_float);
    unsigned addOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, aSize, ARRAY_SIZE(aSize), syn_type_float);
    addNodeToGraph("add_fwd_f32", {aIn, bIn}, {addOut});
    unsigned reshapeOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, gemmSize, ARRAY_SIZE(gemmSize), syn_type_float);
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {addOut}, {reshapeOut});
    synGEMMParams gemmParams{};
    unsigned out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oSize, ARRAY_SIZE(oSize), syn_type_float);
    unsigned weight = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSize, ARRAY_SIZE(wSize), syn_type_float);
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {reshapeOut, weight}, {out}, &gemmParams, sizeof(gemmParams), "GEMM");

    compileAndRun();

    synTensorDescriptor gemmWeightDesc = m_tensorDescs[weight];
    synTensorDescriptor gemmInDesc = m_tensorDescs[reshapeOut];
    synTensorDescriptor gemmResDesc = m_tensorDescs[out];
    char* operandA = (char*)m_hostBuffers[aIn];
    char* operandB = (char*)m_hostBuffers[bIn];
    char* weightIn = (char*)m_hostBuffers[weight];
    char* gemmRes = (char*)m_hostBuffers[out];
    char* gemmResRef = new char[oSize[0]*oSize[1]*4];
    char* gemmIn = new char[gemmSize[0]*gemmSize[1]*4];

    // execute broadcast add
    for (int i = 0; i < inDims[1]*inDims[2]; i++)
    {
        for (int cdIndex = 0; cdIndex < inDims[0]; cdIndex++)
        {
            ((float*)gemmIn)[i*inDims[0] + cdIndex] = ((float*)operandA)[i*inDims[0] + cdIndex] + ((float*)operandB)[cdIndex];
        }
    }
    calculateGemm(gemmInDesc,
                  gemmIn,
                  gemmWeightDesc,
                  weightIn,
                  gemmResDesc,
                  gemmResRef,
                  gemmParams,
                  REFERENCE_OP_FWD,
                  m_deviceType);
    validateResults(gemmResDesc, gemmResRef, gemmRes);
    delete[] gemmResRef;
    delete[] gemmIn;
    }
};

TEST_F_GC(SynTrainingReshapeAlignTest, gemm_reshape_add_broadcast_producer)
{
    // JIRA: SW-24592
    //   Pushing reshape in front of tpc producer, when 1 of the inputs is broadcasted.
    unsigned inSize[] = {32, 16, 8};
    unsigned gemmSize[] = {32, 128};
    unsigned wSize[] = {4, 32};
    runTest(inSize, gemmSize, wSize);
}

TEST_F_GC(SynTrainingReshapeAlignTest, gemm_reshape_add_broadcast_producer_negative1)
{
    //   check that we DON'T Push reshape in front of tpc producer.
    unsigned inSize[] = {32, 2, 64};
    unsigned gemmSize[] = {64, 64};
    unsigned wSize[] = {4, 64};
    runTest(inSize, gemmSize, wSize);
}

TEST_F_GC(SynTrainingReshapeAlignTest, gemm_reshape_add_broadcast_producer_negative2)
{
    //   check that we DON'T Push reshape in front of tpc producer.
    unsigned inSize[] = {32, 16, 8};
    unsigned gemmSize[] = {64, 64};
    unsigned wSize[] = {4, 64};
    runTest(inSize, gemmSize, wSize);
}

class SynTrainingMMEFlatenReshapeAlignTest : public SynTrainingTestInfra
{
protected:
    void runTest(bool negativeTest)
    {
        unsigned aSize[] = {64, 8, 8, 64};
        unsigned bSize[] = {1, 1, 1, 1};
        unsigned broadcastIndex = negativeTest? 1 : 0;
        bSize[broadcastIndex] = aSize[broadcastIndex];
        unsigned wSize[] = {64, 64, 1, 1};

        unsigned aIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, aSize, ARRAY_SIZE(aSize), syn_type_float);
        unsigned bIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, bSize, ARRAY_SIZE(bSize), syn_type_float);
        unsigned addOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, aSize, ARRAY_SIZE(aSize), syn_type_float);
        addNodeToGraph("add_fwd_f32", {aIn, bIn}, {addOut});

        synConvolutionParams params{};
        unsigned out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, aSize, ARRAY_SIZE(aSize), syn_type_float);
        unsigned weight = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSize, ARRAY_SIZE(wSize), syn_type_float);
        addNodeToGraph(NodeFactory::convolutionNodeTypeName, {addOut, weight}, {out}, &params, sizeof(params), "conv");

        compileAndRun();

        synTensorDescriptor weightDesc = m_tensorDescs[weight];
        synTensorDescriptor convInDesc = m_tensorDescs[addOut];
        synTensorDescriptor convOutDesc = m_tensorDescs[out];
        char* operandA = (char*)m_hostBuffers[aIn];
        char* operandB = (char*)m_hostBuffers[bIn];
        char* weightIn = (char*)m_hostBuffers[weight];
        char* convOut = (char*)m_hostBuffers[out];
        char* convOutRef = new char[aSize[0]*aSize[1]*aSize[2]*aSize[3] * 4];
        char* convIn = new char[aSize[0]*aSize[1]*aSize[2]*aSize[3] * 4];

        // execute broadcast add
        if (negativeTest)
        {
            for (int i = 0; i < aSize[2]*aSize[3]; i++)
            for (int j = 0; j < aSize[1]; j++)
            for (int cdIndex = 0; cdIndex < aSize[0]; cdIndex++)
                ((float*)convIn)[i*aSize[0]*aSize[1] + j*aSize[0] + cdIndex] =
                    ((float*)operandA)[i*aSize[0]*aSize[1] + j*aSize[0] + cdIndex] + ((float*)operandB)[j];
        }
        else
        {
            for (int i = 0; i < aSize[1]*aSize[2]*aSize[3]; i++)
            for (int cdIndex = 0; cdIndex < aSize[0]; cdIndex++)
                ((float*)convIn)[i*aSize[0] + cdIndex] =
                     ((float*)operandA)[i*aSize[0] + cdIndex] + ((float*)operandB)[cdIndex];
        }

        // execute conv
        calculateFwdConvolution(convInDesc,
                                convIn,
                                weightDesc,
                                weightIn,
                                convOutDesc,
                                convOutRef,
                                params,
                                m_deviceType);
        validateResults(convOutDesc, convOutRef, convOut);
        delete[] convOutRef;
        delete[] convIn;
    }
};

TEST_F_GC(SynTrainingMMEFlatenReshapeAlignTest, mme_flatten_with_broadcast)
{
    runTest(true);
}

TEST_F_GC(SynTrainingMMEFlatenReshapeAlignTest, mme_flatten_with_broadcast_negative)
{
    runTest(false);
}