#include "gc_gaudi_test_infra.h"

class DuplicateMmeInputTest : public SynGaudiTestInfra
{
};

TEST_F_GC(DuplicateMmeInputTest, duplicate_input)
{
    unsigned inSizes[] = {1024, 320};
    unsigned ySizes[]  = {1024, 1024};

    unsigned inIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSizes, 2);
    unsigned yIndex  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 2);

    synGEMMParams gemmParams(true /* transpose_a */, false /* transpose_b */);
    addNodeToGraph("gemm", {inIndex, inIndex}, {yIndex}, &gemmParams, sizeof(gemmParams), "gemm");

    compileAndRun();

    synTensorDescriptor xDesc = m_tensorDescs[inIndex];
    synTensorDescriptor wDesc = m_tensorDescs[inIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yIndex];
    void*               xData = m_hostBuffers[inIndex];
    void*               wData = m_hostBuffers[inIndex];
    void*               yData = m_hostBuffers[yIndex];

    CoordArray wrongIdx       = {0};
    float      expectedResult = 0;
    bool       ret            = checkBatchGemmOp(xDesc,
                                (char*)xData,
                                wDesc,
                                (char*)wData,
                                yDesc,
                                (char*)yData,
                                REFERENCE_OP_ATB,
                                wrongIdx,
                                (float*)yData,
                                m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, yDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                         << " Got value: " << getIndexValue(sizes, wrongIdx, yDesc.m_dataType, yData)
                         << " Expected: " << expectedResult;
}

TEST_F_GC(DuplicateMmeInputTest, duplicate_input_cin)
{
    unsigned inSizes[] = {512, 512};
    unsigned ySizes[]  = {512, 512};

    unsigned inIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSizes, 2);
    unsigned yIndex  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 2);

    synGEMMParams gemmParams(false /* transpose_a */, false /* transpose_b */);
    addNodeToGraph("gemm", {inIndex, inIndex, inIndex}, {yIndex}, &gemmParams, sizeof(gemmParams), "gemm");

    compileAndRun();

    synTensorDescriptor xDesc = m_tensorDescs[inIndex];
    synTensorDescriptor wDesc = m_tensorDescs[inIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yIndex];
    float*              xData = (float*)m_hostBuffers[inIndex];
    float*              wData = xData;
    float*              yData = (float*)m_hostBuffers[yIndex];

    for (unsigned i = 0; i < inSizes[0] * inSizes[1]; i++)
    {
        yData[i] -= xData[i];
    }

    CoordArray wrongIdx       = {0};
    float      expectedResult = 0;
    bool       ret            = checkBatchGemmOp(xDesc,
                                (char*)xData,
                                wDesc,
                                (char*)wData,
                                yDesc,
                                (char*)yData,
                                REFERENCE_OP_AB,
                                wrongIdx,
                                yData,
                                m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, yDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                         << " Got value: " << getIndexValue(sizes, wrongIdx, yDesc.m_dataType, yData)
                         << " Expected: " << expectedResult;
}