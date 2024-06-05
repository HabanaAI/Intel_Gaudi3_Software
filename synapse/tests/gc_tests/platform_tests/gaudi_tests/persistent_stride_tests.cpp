#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, relu_forward_strided)
{
    unsigned inTensorSize[4] = {2, 2, 2, 2};
    unsigned strides[5]      = {4, 16, 32, 64, 128};

    float initializer[] = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                           1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_FROM_INITIALIZER,
                                      initializer,
                                      inTensorSize,
                                      4,
                                      syn_type_single,
                                      strides,
                                      "inputTensor");

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       inTensorSize,
                                       4,
                                       syn_type_single,
                                       nullptr,
                                       "outputTensor");
    addNodeToGraph("relu_fwd_f32");
    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[out];
    float* pInputBuffer  = (float*)m_hostBuffers[in];

    UNUSED(pInputBuffer);
    for (unsigned i = 0; i < 2 * 2 * 2 * 2; i++)
    {
        float expectedResult = 1.0;
        ASSERT_EQ(expectedResult, *pOutputBuffer)
            << "Mismatch for at index " << i << " Expected:" << expectedResult << " Result: " << *pOutputBuffer;
    }
}
TEST_F_GC(SynTrainingTestInfra, GEMM_forward_strided_ASIC_CI)
{
    unsigned inTensorSize[2] = {1024, 1024};
    unsigned inStrides[3]    = {4, 8192, 8388608};
    unsigned a               = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     inTensorSize,
                                     2,
                                     syn_type_single,
                                     inStrides,
                                     nullptr);

    unsigned b = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     inTensorSize,
                                     2,
                                     syn_type_single,
                                     nullptr);

    unsigned out =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inTensorSize, 2, syn_type_single, nullptr);
    synGEMMParams params;

    addNodeToGraph("gemm", {a, b}, {out}, &params, sizeof(params));

    compileAndRun();

    float* pA = castHostBuffer<float>(a);

    std::vector<float> contiguousA(1024 * 1024);
    for (unsigned j = 0, k = 0; j < 2048; j += 2)
    {
        for (unsigned i = 0; i < 1024; i++)
        {
            contiguousA[k * 1024 + i] = pA[j * 1024 + i];
        }
        k++;
    }
    float*             pB = castHostBuffer<float>(b);
    std::vector<float> outRef(1024 * 1024);

    synTensorDescriptor aDesc   = m_tensorDescs[a];
    synTensorDescriptor bDesc   = m_tensorDescs[b];
    synTensorDescriptor outDesc = m_tensorDescs[out];

    params.transpose_a = false;
    params.transpose_b = false;

    calculateGemm(aDesc,
                  (char*)contiguousA.data(),
                  bDesc,
                  (char*)pB,
                  outDesc,
                  (char*)(outRef.data()),
                  params,
                  REFERENCE_OP_AB,
                  m_deviceType);

    float* pOut = castHostBuffer<float>(out);
    validateResult(outRef.data(), pOut, 1024 * 1024);
}
