#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include <gtest/gtest.h>

static std::ostream& operator<<(std::ostream& os, const bfloat16& rhs)
{
    os << rhs.value();
    return os;
}

template<typename T>
static void print_array(T* a, uint64_t size1, uint64_t size2)
{
    std::cout << std::endl;
    for (size_t j = 0; j < size2; j++)
    {
        for (size_t i = 0; i < size1; i++)
        {
            std::cout << a[i + j * size1] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

class SynTrainingTransposeTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<int, int>>
{
public:
    SynTrainingTransposeTest() { setTestPackage(TEST_PACKAGE_TRANSPOSE); }

    void SetUpTest() override
    {
        SynTrainingTestInfra::SetUpTest();
        GCFG_ENABLE_INTERNAL_NODES.setValue(true);
    }
};

TEST_P_GC(SynTrainingTransposeTest, transpose_bf16)
{
    createGraph();

    // CNHW
    int C;
    int W;
    std::tie(C, W) = GetParam();

    using DataType = bfloat16;
    // constexpr auto ELEMENT_SIZE = sizeof(DataType);
    const auto            SYN_DATA_TYPE = dataTypeToSynType<DataType>();
    std::vector<uint16_t> arr(C * W, 0);

    for (size_t i = 0; i < C * W; i++)
    {
        arr[i] = 10000 + (i % 4048);
    }

    unsigned src_size_[] = {C, W, 1, 1, 1};
    auto     in          = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                  (float*)arr.data(),
                                  src_size_,
                                  4,
                                  SYN_DATA_TYPE);
    unsigned dst_size_[] = {W, C, 1, 1, 1};

    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dst_size_, 4, SYN_DATA_TYPE);
    synTransposeParams params = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    addNodeToGraph("transpose", {in}, {out}, &params, sizeof(params));
    compileAndRun();

    bfloat16* pInput  = castHostInBuffer<bfloat16>(in);
    bfloat16* pOutput = castHostOutBuffer<bfloat16>(out);
    for (uint32_t j = 0; j < W; j++)
    {
        for (uint32_t i = 0; i < C; i++)
        {
            if ((float)pInput[i + j * C] != (float)pOutput[j + i * W])
            {
                print_array(pInput, C, W);
                print_array(pOutput, W, C);
            }
            ASSERT_EQ(pInput[i + j * C].value(), pOutput[j + i * W].value())
                << "Mismatch for at index " << i << ", " << j;
        }
    }
}

TEST_P_GC(SynTrainingTransposeTest, transpose_float)
{
    createGraph();

    // CNHW
    int C;
    int W;
    std::tie(C, W) = GetParam();

    using DataType                      = float;
    const auto            SYN_DATA_TYPE = dataTypeToSynType<DataType>();
    std::vector<uint32_t> arr(C * W, 0);

    for (size_t i = 0; i < C * W; i++)
    {
        arr[i] = 10000 + (i % 4048);
    }

    unsigned src_size_[] = {C, W, 1, 1, 1};
    auto     in          = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                  (float*)arr.data(),
                                  src_size_,
                                  4,
                                  SYN_DATA_TYPE);
    unsigned dst_size_[] = {W, C, 1, 1, 1};

    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dst_size_, 4, SYN_DATA_TYPE);
    synTransposeParams params = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    addNodeToGraph("transpose", {in}, {out}, &params, sizeof(params));

    compileAndRun();

    auto* pInput  = castHostInBuffer<float>(in);
    auto* pOutput = castHostOutBuffer<float>(out);
    for (uint32_t j = 0; j < W; j++)
    {
        for (uint32_t i = 0; i < C; i++)
        {
            if (pInput[i + j * C] != pOutput[j + i * W])
            {
                print_array(pInput, C, W);
                print_array(pOutput, W, C);
            }
            ASSERT_FLOAT_EQ(pInput[i + j * C], pOutput[j + i * W]) << "Mismatch for at index " << i << ", " << j;
        }
    }
}

TEST_P_GC(SynTrainingTransposeTest, no_glitches_with_memcpy)
{
    createGraph();

    // CNHW
    int C;
    int W;
    std::tie(C, W) = GetParam();

    using DataType                      = bfloat16;
    const auto            SYN_DATA_TYPE = dataTypeToSynType<DataType>();
    std::vector<uint16_t> arr(C * W, 0);

    for (size_t i = 0; i < C * W; i++)
    {
        arr[i] = 10000 + (i % 4048);
    }

    unsigned src_size_[] = {C, W, 1, 1, 1};
    auto     in          = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                  (float*)arr.data(),
                                  src_size_,
                                  4,
                                  SYN_DATA_TYPE);
    unsigned dst_size_[] = {W, C, 1, 1, 1};

    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dst_size_, 4, SYN_DATA_TYPE);

    synTransposeParams params = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    addNodeToGraph("transpose", {in}, {out}, &params, sizeof(params));
    auto afterTranspose = connectOutputTensorToInputTensor(out);
    auto outputTensor2  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dst_size_, 4, SYN_DATA_TYPE);
    addNodeToGraph("memcpy", {afterTranspose}, {outputTensor2});

    auto afterTranspose2 = connectOutputTensorToInputTensor(outputTensor2);
    auto outputTensor3   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, src_size_, 4, SYN_DATA_TYPE);
    addNodeToGraph("transpose", {afterTranspose2}, {outputTensor3}, &params, sizeof(params));

    auto afterTranspose3 = connectOutputTensorToInputTensor(outputTensor3);
    auto outputTensor4   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, src_size_, 4, SYN_DATA_TYPE);
    addNodeToGraph("memcpy", {afterTranspose3}, {outputTensor4});

    auto afterTranspose4 = connectOutputTensorToInputTensor(afterTranspose3);
    auto outputTensor5   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dst_size_, 4, SYN_DATA_TYPE);
    addNodeToGraph("transpose", {afterTranspose4}, {outputTensor5}, &params, sizeof(params));

    auto lastOutput = outputTensor5;
    compileAndRun();

    bfloat16* pInput  = castHostInBuffer<bfloat16>(in);
    bfloat16* pOutput = castHostOutBuffer<bfloat16>(lastOutput);
    for (uint32_t j = 0; j < W; j++)
    {
        for (uint32_t i = 0; i < C; i++)
        {
            if ((float)pInput[i + j * C] != (float)pOutput[j + i * W])
            {
                print_array(pInput, C, W);
                print_array(pOutput, W, C);
            }
            ASSERT_EQ(pInput[i + j * C].value(), pOutput[j + i * W].value())
                << "Mismatch for at index " << i << ", " << j;
        }
    }
}

template<typename DataType>
std::array<unsigned, 6> stridesOf(unsigned dim0, unsigned dim1)
{
    std::array<unsigned, 6> ret = {sizeof(DataType),
                                   sizeof(DataType) * dim0,
                                   sizeof(DataType) * dim0 * dim1,
                                   sizeof(DataType) * dim0 * dim1,
                                   sizeof(DataType) * dim0 * dim1,
                                   sizeof(DataType) * dim0 * dim1};
    return ret;
}

TEST_P_GC(SynTrainingTransposeTest, transpose_strided_input)
{
    createGraph();

    // CNHW
    auto multiplier = 2;
    int  C;
    int  W;
    std::tie(C, W) = GetParam();

    using DataType                      = bfloat16;
    const auto            SYN_DATA_TYPE = dataTypeToSynType<DataType>();
    std::vector<uint16_t> arr(C * W * multiplier, 0);

    for (size_t i = 0; i < C * W * multiplier; i++)
    {
        arr[i] = 10000 + (i % 4048);
    }

    unsigned src_size[]  = {C * multiplier, W, 1, 1, 1};
    unsigned src2_size[] = {C, W, 1, 1, 1};
    auto     in          = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                  (float*)arr.data(),
                                  src_size,
                                  4,
                                  SYN_DATA_TYPE);
    auto     afterSplitDst =
        createTensors(2, TensorUsage::OUTPUT_TENSOR, false, nullptr, MEM_INIT_ALL_ZERO, 0, src2_size, 4, SYN_DATA_TYPE);
    TensorIndices afterSplitSrc {connectOutputTensorToInputTensor(afterSplitDst[0]),
                                 connectOutputTensorToInputTensor(afterSplitDst[1])};
    unsigned      dst_size[]  = {W, C, 1, 1, 1};
    auto          dst_strides = stridesOf<DataType>(W, C);
    auto          outTensor =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dst_size, 4, SYN_DATA_TYPE, dst_strides.data());
    unsigned dim = 0;
    addNodeToGraph("split", {0}, afterSplitDst, &dim, sizeof(unsigned));

    synTransposeParams params = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    addNodeToGraph("transpose", {afterSplitSrc[0]}, {outTensor}, &params, sizeof(params));
    compileAndRun();

    auto* pInput  = castHostInBuffer<DataType>(in);
    auto* pOutput = castHostOutBuffer<DataType>(outTensor);
    for (uint32_t j = 0; j < W; j++)
    {
        for (uint32_t i = 0; i < C; i++)
        {
            if (pInput[i + j * C * multiplier].value() != pOutput[j + i * W].value())
            {
                print_array(pInput, C * multiplier, W);
                print_array(pOutput, W, C);
            }
            ASSERT_EQ(pInput[i + j * C * multiplier].value(), pOutput[j + i * W].value())
                << "Mismatch for at index " << i << ", " << j;
        }
    }
    // TODO: Maybe add strided output test too
}

INSTANTIATE_TEST_SUITE_P(
    Sanity,
    SynTrainingTransposeTest,
    testing::Values(std::make_tuple(5, 7), std::make_tuple(64, 33), std::make_tuple(64, 32), std::make_tuple(64, 31)));

INSTANTIATE_TEST_SUITE_P(Big,
                         SynTrainingTransposeTest,
                         testing::Values(std::make_tuple(32, 48),
                                         std::make_tuple(1, 512),
                                         std::make_tuple(64, 256),
                                         std::make_tuple(256, 64),
                                         std::make_tuple(256, 256),
                                         std::make_tuple(512, 512),
                                         std::make_tuple(1024, 1024),
                                         std::make_tuple(1920, 1080),
                                         std::make_tuple(1023, 957)));
