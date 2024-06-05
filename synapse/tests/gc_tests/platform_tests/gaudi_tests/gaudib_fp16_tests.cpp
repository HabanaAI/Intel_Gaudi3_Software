#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "../utils/supported_devices_macros.h"
#include <functional>
#include "data_type_utils.h"

class SynGaudiFP16Test : public SynGaudiTestInfra
{
public:
    SynGaudiFP16Test() { setSupportedDevices({synDeviceGaudi2}); }

    template<class BinaryOperation>
    void checkFp16BinaryOp(const char* guid, BinaryOperation binaryOp);
};

template<class BinaryOperation>
void SynGaudiFP16Test::checkFp16BinaryOp(const char* guid, BinaryOperation binaryOp)
{
    unsigned sizes[] = {4, 4, 1, 1};
    unsigned dims = 2;
    uint64_t numElems = getNumberOfElements(sizes, dims);

    unsigned in1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, dims, syn_type_fp16, nullptr, "in1");
    unsigned in2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, dims, syn_type_fp16, nullptr, "in2");
    unsigned out  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_fp16, nullptr, "out");

    addNodeToGraph(guid, {in1, in2}, {out});

    compileAndRun();

    auto* input1 = castHostBuffer<fp16_t>(in1);
    auto* input2 = castHostBuffer<fp16_t>(in2);
    auto* output = castHostBuffer<fp16_t>(out);

    float* in1float = float16BufferToFloatBuffer(input1, numElems);
    float* in2float = float16BufferToFloatBuffer(input2, numElems);
    float* outfloat = float16BufferToFloatBuffer(output, numElems);

    for (unsigned idx = 0; idx < numElems; idx++)
    {
        float expected = binaryOp(in1float[idx], in2float[idx]);
        float actual = outfloat[idx];

        ASSERT_TRUE(float_eq(expected, actual)) << "OUTPUT: Mismatch for at index " << idx
                                           << " |Expected:" << expected
                                           << " |Result: " << actual
                                           << " |Operands: "
                                           << in1float[idx] << ", " << in2float[idx];
    }

    delete[] in1float;
    delete[] in2float;
    delete[] outfloat;
}

TEST_F_GC(SynGaudiFP16Test, add_fwd_f16)
{
    checkFp16BinaryOp("add_fwd_f16", std::plus<float>());
}

TEST_F_GC(SynGaudiFP16Test, sub_fwd_f16)
{
    checkFp16BinaryOp("sub_fwd_f16", std::minus<float>());
}

TEST_F_GC(SynGaudiFP16Test, mult_fwd_f16)
{
    checkFp16BinaryOp("mult_fwd_f16", std::multiplies<float>());
}

TEST_F_GC(SynGaudiFP16Test, cast_f16_to_f32)
{
    unsigned sizes[] = {4, 4, 1, 1};
    unsigned dims = 2;
    uint64_t numElems = getNumberOfElements(sizes, dims);

    unsigned castIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, dims, syn_type_fp16, nullptr, "castIn");
    unsigned castOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_single, nullptr, "castOut");

    addNodeToGraph("cast_f16_to_f32", {castIn}, {castOut});

    compileAndRun();

    auto* input  = castHostBuffer<fp16_t>(castIn);
    auto* output = castHostBuffer<float>(castOut);

    float* expectedOutput = float16BufferToFloatBuffer(input, numElems);

    for (uint64_t idx = 0; idx < numElems; idx++)
    {
        ASSERT_FLOAT_EQ(expectedOutput[idx], output[idx]) << "OUTPUT: Mismatch for at index " << idx
                                           << " |Expected:" << expectedOutput[idx]
                                           << " |Result: " << output[idx];
    }

    delete[] expectedOutput;
}

TEST_F_GC(SynGaudiFP16Test, cast_f32_to_f16)
{
    unsigned sizes[] = {4, 4, 1, 1};
    unsigned dims = 2;
    uint64_t numElems = getNumberOfElements(sizes, dims);

    unsigned castIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, dims, syn_type_single, nullptr, "castIn");
    unsigned castOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_fp16, nullptr, "castOut");

    addNodeToGraph("cast_f32_to_f16", {castIn}, {castOut});

    compileAndRun();

    auto* input = castHostBuffer<float>(castIn);
    auto* output = castHostBuffer<fp16_t>(castOut);

    fp16_t* expectedOutput = convertBuffer<fp16_t>(input, numElems);

    for (unsigned idx = 0; idx < numElems; idx ++)
    {
        ASSERT_EQ(expectedOutput[idx].value(), output[idx].value());
    }

    delete[] expectedOutput;
}