#include "gc_gaudi_test_infra.h"

template <typename CastFromType, typename CastToType>
static void castAnyFloatToAnyFloat(CastFromType* fromBuffer,
                            CastToType*   toBuffer,
                            unsigned      elementsNum)
{
    // don't allow compilation if types don't match function
    static_assert(!std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum; i++)
    {
        // cast first to fp32 then to target type
        toBuffer[i] = CastToType(float(fromBuffer[i]));
    }
}

class RemoveConvertsTest : public SynTrainingTestInfra
{
};

TEST_F_GC(RemoveConvertsTest, remove_converts_one_remove_bf16_to_fp8)
{
    // --t1-->(add)--t3-->(convert_to_fp8_bf16)--t4-->(convert_from_fp8_bf16)--t5-->(add)--t7-->
    //          ^                        ^                       ^                    ^
    //          |                        |                       |                    |
    //          t2                     scale1                  scale2                 t6

    std::vector<float>      scaleBuffer = {1};
    std::vector<unsigned>     scaleSize = {1};
    const unsigned           dims        = 1;
    unsigned                 sizes[]     = {4};

    unsigned t1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr, sizes, dims, syn_type_bf16, nullptr);
    unsigned t2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr, sizes, dims, syn_type_bf16, nullptr);
    unsigned t6 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                nullptr, sizes, dims, syn_type_bf16, nullptr);
    unsigned t7 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_bf16, nullptr);

    unsigned t3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_bf16, nullptr);
    unsigned t4 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_fp8_143, nullptr);
    unsigned t5 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_bf16, nullptr);

    unsigned scale1 = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER,
                                    scaleBuffer.data(), scaleSize.data(), scaleSize.size(),
                                    syn_type_float, nullptr, "scale1");
    unsigned scale2 = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER,
                                    scaleBuffer.data(), scaleSize.data(), scaleSize.size(),
                                    syn_type_float, nullptr, "scale2");

    addNodeToGraph("add_fwd_bf16", {t1, t2}, {t3}, nullptr, 0, "add1");
    addNodeToGraph("convert_to_fp8_bf16", {t3, scale1}, {t4}, nullptr, 0, "convert_from_bf16_to_fp8");
    addNodeToGraph("convert_from_fp8_bf16", {t4, scale2}, {t5}, nullptr, 0, "convert_from_fp8_to_bf16");
    addNodeToGraph("add_fwd_bf16", {t5, t6}, {t7}, nullptr, 0, "add2");

    compileAndRun();

    std::vector<float> in1 = {0, 0, 0, 0};
    std::vector<float> in2 = {0, 0, 0, 0};
    std::vector<float> in3 = {0, 0, 0, 0};
    std::vector<float> out = {0, 0, 0, 0};

    castAnyFloatToAnyFloat<bfloat16, float>(castHostBuffer<bfloat16>(t1), in1.data(), 4);
    castAnyFloatToAnyFloat<bfloat16, float>(castHostBuffer<bfloat16>(t2), in2.data(), 4);
    castAnyFloatToAnyFloat<bfloat16, float>(castHostBuffer<bfloat16>(t6), in3.data(), 4);
    castAnyFloatToAnyFloat<bfloat16, float>(castHostBuffer<bfloat16>(t7), out.data(), 4);

    for (int i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(in1[i] + in2[i] + in3[i], out[i], 0.1);
    }
}

TEST_F_GC(RemoveConvertsTest, remove_converts_one_remove_fp8_to_bf16)
{
    // --t1-->(add)--t3-->(convert_from_fp8_bf16)--t4-->(convert_to_fp8_bf16)--t5-->(add)--t7-->
    //          ^                        ^                       ^                    ^
    //          |                        |                       |                    |
    //          t2                     scale1                  scale2                 t6

    std::vector<float>      scaleBuffer = {1};
    std::vector<unsigned>     scaleSize = {1};
    const unsigned           dims        = 1;
    unsigned                 sizes[]     = {4};

    unsigned t1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr, sizes, dims, syn_type_fp8_143, nullptr);
    unsigned t2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr, sizes, dims, syn_type_fp8_143, nullptr);
    unsigned t6 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                nullptr, sizes, dims, syn_type_fp8_143, nullptr);
    unsigned t7 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_fp8_143, nullptr);

    unsigned t3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_fp8_143, nullptr);
    unsigned t4 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_bf16, nullptr);
    unsigned t5 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
                                    nullptr, sizes, dims, syn_type_fp8_143, nullptr);

    unsigned scale1 = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER,
                                    scaleBuffer.data(), scaleSize.data(), scaleSize.size(),
                                    syn_type_float, nullptr, "scale1");
    unsigned scale2 = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER,
                                    scaleBuffer.data(), scaleSize.data(), scaleSize.size(),
                                    syn_type_float, nullptr, "scale2");

    addNodeToGraph("add_fwd_bf16", {t1, t2}, {t3}, nullptr, 0, "add1");
    addNodeToGraph("convert_to_fp8_bf16", {t3, scale1}, {t4}, nullptr, 0, "convert_from_bf16_to_fp8");
    addNodeToGraph("convert_from_fp8_bf16", {t4, scale2}, {t5}, nullptr, 0, "convert_from_fp8_to_bf16");
    addNodeToGraph("add_fwd_bf16", {t5, t6}, {t7}, nullptr, 0, "add2");

    compileAndRun();

    std::vector<float> in1 = {0, 0, 0, 0};
    std::vector<float> in2 = {0, 0, 0, 0};
    std::vector<float> in3 = {0, 0, 0, 0};
    std::vector<float> out = {0, 0, 0, 0};

    castAnyFloatToAnyFloat<fp8_143_t, float>(castHostBuffer<fp8_143_t>(t1), in1.data(), 4);
    castAnyFloatToAnyFloat<fp8_143_t, float>(castHostBuffer<fp8_143_t>(t2), in2.data(), 4);
    castAnyFloatToAnyFloat<fp8_143_t, float>(castHostBuffer<fp8_143_t>(t6), in3.data(), 4);
    castAnyFloatToAnyFloat<fp8_143_t, float>(castHostBuffer<fp8_143_t>(t7), out.data(), 4);

    for (int i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(in1[i] + in2[i] + in3[i], out[i], 0.1);
    }
}