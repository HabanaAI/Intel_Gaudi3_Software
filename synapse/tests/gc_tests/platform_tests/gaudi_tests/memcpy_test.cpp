#include "gc_gaudi_test_infra.h"

class SynTrainingMemcpyTest : public SynTrainingTestInfra
{
};

TEST_F_GC(SynTrainingMemcpyTest, memcpy_no_strides)
{
    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    auto out = createPersistTensor(OUTPUT_TENSOR);
    addNodeToGraph("memcpy");
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        ASSERT_EQ(*pInputBuffer, *pOutputBuffer) << "Mismatch for at index " << i
                                                  << " Expected:"             << *pInputBuffer
                                                  << " Result: "              << *pOutputBuffer;
        pInputBuffer++;
        pOutputBuffer++;
    }
}

template <typename ElemType>
void compareStridedTensorData(const ElemType serialData[], const ElemType stridedData[],
                              unsigned strides[4], unsigned dims[4])
{
    unsigned        elemStrides[]      = {strides[1] / sizeof(ElemType),
                              strides[2] / sizeof(ElemType),
                              strides[3] / sizeof(ElemType)};
    const ElemType *pNextSerialElement = &serialData[0];

    for (unsigned a = 0; a < dims[3]; ++a)
    {
        for (unsigned b = 0; b < dims[2]; ++b)
        {
            for (unsigned c = 0; c < dims[1]; ++c)
            {
                for (unsigned d = 0; d < dims[0]; ++d)
                {
                    float elem1 = *pNextSerialElement++;
                    float elem2 = *(stridedData + a * elemStrides[2] + b * elemStrides[1] + c * elemStrides[0] + d);
                    ASSERT_EQ(elem1, elem2)
                                        << "Mismatch at {" << a << ", " << b << ", " << c << ", " << d << "}"
                                        << " elem1: " << elem1
                                        << " elem2: " << elem2;

                }
            }
        }
    }
}

template <typename ElemType>
void compareStridedTensorDataRegularMemcpy(const ElemType serialData[], const ElemType stridedData[], unsigned dims[4])
{
    auto elements = dims[0] * dims[1] * dims[2] * dims[3];
    for (unsigned i = 0; i < elements; i++)
    {
        float serialElement = serialData[i];
        float stridedElement = stridedData[i];
        EXPECT_EQ(serialElement, stridedElement) << "Mismatch for at index " << i
                                                 << " Expected:"             << serialElement
                                                 << " Result: "              << stridedElement;
    }
}

TEST_F_GC(SynTrainingMemcpyTest, memcpy_strided_input_f32_regular_memcpy)
{
    const unsigned      elemSize  = 4;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dims, DEFAULT_SIZES, syn_type_float, strides);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_float);
    addNodeToGraph("memcpy");
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];
    compareStridedTensorData(pOutputBuffer, pInputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, memcpy_strided_output_f32_regular_memcpy)
{
    const unsigned      elemSize  = 4;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dims, DEFAULT_SIZES, syn_type_float);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_float, strides);
    addNodeToGraph("memcpy");
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];
    compareStridedTensorData(pInputBuffer, pOutputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, DISABLED_memcpy_strided_input_f32)
{
    const unsigned      elemSize  = 4;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dims, DEFAULT_SIZES ,syn_type_float, strides);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_float);
    addNodeToGraph("memcpy");
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];
    compareStridedTensorData(pOutputBuffer, pInputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, DISABLED_memcpy_strided_output_f32)
{
    const unsigned      elemSize  = 4;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dims, syn_type_float);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_float, strides);
    addNodeToGraph("memcpy");
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];
    compareStridedTensorData(pInputBuffer, pOutputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, memcpy_strided_input_output_f32)
{
    const unsigned      elemSize  = 4;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dims, DEFAULT_SIZES, syn_type_float, strides);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_float, strides);
    addNodeToGraph("memcpy");
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];
    for (uint64_t i = 0; i < getNumberOfElements(dims); i++)
    {
        ASSERT_EQ(*pInputBuffer, *pOutputBuffer) << "Mismatch for at index " << i
                                                 << " Expected:"             << *pInputBuffer
                                                 << " Result: "              << *pOutputBuffer;
        pInputBuffer++;
        pOutputBuffer++;
    }
}

TEST_F_GC(SynTrainingMemcpyTest, memcpy_strided_input_bf16_regular_memcpy)
{
    const unsigned      elemSize  = 2;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16, strides);
    fillWithRandom((bfloat16*)m_hostBuffers[in], dims[0] * dims[1] * dims[2] * dims[3], std::make_pair(-1.0f, 1.0f));
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16);
    addNodeToGraph("memcpy");
    compileAndRun();

    bfloat16* pInputBuffer  = (bfloat16*)m_hostBuffers[in];
    bfloat16* pOutputBuffer = (bfloat16*)m_hostBuffers[out];
    compareStridedTensorData(pOutputBuffer, pInputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, memcpy_strided_output_bf16_regular_memcpy)
{
    const unsigned      elemSize  = 2;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16);
    fillWithRandom((bfloat16*)m_hostBuffers[in], dims[0] * dims[1] * dims[2] * dims[3], std::make_pair(-1.0f, 1.0f));
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16, strides);
    addNodeToGraph("memcpy");
    compileAndRun();

    bfloat16* pInputBuffer  = (bfloat16*)m_hostBuffers[in];
    bfloat16* pOutputBuffer = (bfloat16*)m_hostBuffers[out];
    compareStridedTensorData(pInputBuffer, pOutputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, DISABLED_memcpy_strided_input_bf16)
{
    const unsigned      elemSize  = 2;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16, strides);
    fillWithRandom((bfloat16*)m_hostBuffers[in], dims[0] * dims[1] * dims[2] * dims[3], std::make_pair(-1.0f, 1.0f));
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, syn_type_bf16);
    addNodeToGraph("memcpy");
    compileAndRun();

    bfloat16* pInputBuffer  = (bfloat16*)m_hostBuffers[in];
    bfloat16* pOutputBuffer = (bfloat16*)m_hostBuffers[out];
    compareStridedTensorData(pOutputBuffer, pInputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, DISABLED_memcpy_strided_output_bf16)
{
    const unsigned      elemSize  = 2;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, syn_type_bf16);
    fillWithRandom((bfloat16*)m_hostBuffers[in], dims[0] * dims[1] * dims[2] * dims[3], std::make_pair(-1.0f, 1.0f));
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16, strides);
    addNodeToGraph("memcpy");
    compileAndRun();

    bfloat16* pInputBuffer  = (bfloat16*)m_hostBuffers[in];
    bfloat16* pOutputBuffer = (bfloat16*)m_hostBuffers[out];
    compareStridedTensorData(pInputBuffer, pOutputBuffer, strides, dims);
}

TEST_F_GC(SynTrainingMemcpyTest, memcpy_strided_input_output_bf16)
{
    const unsigned      elemSize  = 2;
    unsigned            dims[]    = {4, 16, 16, 1};
    unsigned            strides[] = {elemSize,
                          elemSize * dims[1] * dims[0],
                          elemSize * dims[0],
                          elemSize * dims[0] * dims[1] * dims[2],
                          elemSize * dims[0] * dims[1] * dims[2] * dims[3]};

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16, strides);
    fillWithRandom((bfloat16*)m_hostBuffers[in], dims[0] * dims[1] * dims[2] * dims[3], std::make_pair(-1.0f, 1.0f));
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_bf16, strides);
    addNodeToGraph("memcpy");
    compileAndRun();

    bfloat16* pInputBuffer  = (bfloat16*)m_hostBuffers[in];
    bfloat16* pOutputBuffer = (bfloat16*)m_hostBuffers[out];
    for (uint64_t i = 0; i < getNumberOfElements(dims); i++)
    {
        float inputElement = *pInputBuffer;
        float outputElement = *pOutputBuffer;
        ASSERT_EQ(inputElement, outputElement) << "Mismatch for at index " << i
                                               << " Expected:"             << inputElement
                                               << " Result: "              << outputElement;
        pInputBuffer++;
        pOutputBuffer++;
    }
}

TEST_F_GC(SynTrainingMemcpyTest, mem_copy)
{
    createGraph();

    auto in  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE);
    auto out = createPersistTensor(OUTPUT_TENSOR);
    addNodeToGraph("memcpy");
    compileTopology();
    runTopology();

    float* pInput  = (float*)m_hostBuffers[in];
    float* pOutput = (float*)m_hostBuffers[out];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        float expectedResult = *pInput;
        ASSERT_EQ(expectedResult, *pOutput) << "Mismatch for at index " << i << " Expected:" << expectedResult
                                            << " pOutput1: " << *pOutput << " pInput0: " << *pInput;
        pInput++;
        pOutput++;
    }
}

TEST_F_GC(SynTrainingMemcpyTest, zero_size_allocation)
{
    void*          hostBuffer;
    const uint64_t size  = 0;
    const uint32_t flags = 0;

    // check allocation with 0 size
    synStatus status = synHostMalloc(m_deviceId, size, flags, &hostBuffer);
    EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory for persistent tensor";

    status = synHostFree(m_deviceId, hostBuffer, flags);
    EXPECT_EQ(status, synSuccess) << "Failed to free host memory for persistent tensor";

    // check allocation with 0 size and nullptr host buffer
    hostBuffer = nullptr;

    status = synHostMalloc(m_deviceId, size, flags, &hostBuffer);
    EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory for persistent tensor";

    status = synHostFree(m_deviceId, hostBuffer, flags);
    EXPECT_EQ(status, synSuccess) << "Failed to free host memory for persistent tensor";

    uint64_t deviceAddr;
    uint64_t requestedAddress = 0;

    // check device allocation with 0 size
    status = synDeviceMalloc(m_deviceId, size, flags, requestedAddress, &deviceAddr);
    EXPECT_EQ(status, synSuccess) << "Failed to allocate device memory for persistent tensor";

    status = synDeviceFree(m_deviceId, deviceAddr, flags);
    EXPECT_EQ(status, synSuccess) << "Failed to free device memory for persistent tensor";
}
