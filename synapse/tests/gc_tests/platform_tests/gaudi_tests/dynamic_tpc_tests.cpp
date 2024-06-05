#include "gc_dynamic_shapes_infra.h"

// This class handles tests for dynamic TPC, with both fully disabled and partial ROIs.
//

class SynGaudiDynamicTPCTestRelu : public SynGaudiDynamicShapesTestsInfra
{
  public:

    void compileTest(unsigned tensorDim, unsigned minSizes[SYN_MAX_TENSOR_DIM],
                     unsigned maxSizes[SYN_MAX_TENSOR_DIM], unsigned& inTensor, unsigned& outTensor)
    {
        inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                       maxSizes, tensorDim, syn_type_single, nullptr, nullptr,
                                       0, 0, nullptr, minSizes);

        outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        maxSizes, tensorDim, syn_type_single, nullptr, nullptr,
                                        0, 0, nullptr, minSizes);

        addNodeToGraph("relu_fwd_f32", {inTensor}, {outTensor});

        compileTopology();

        ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
        shape_plane_graph_t *recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
        ASSERT_NE(recipe, nullptr);
    }

    void runTest(unsigned dynamicDim, unsigned actualSize,
                 unsigned maxSizes[SYN_MAX_TENSOR_DIM], unsigned& inTensor, unsigned& outTensor)
    {
        unsigned actualSizes[SYN_MAX_TENSOR_DIM];
        memcpy(actualSizes, maxSizes, sizeof(actualSizes));

        actualSizes[dynamicDim] = actualSize;
        setActualSizes(inTensor, actualSizes);
        setActualSizes(outTensor, actualSizes);

        runTopology(0, true);
    }
};

class SynGaudiDynamicTPCTestReluSingle : public SynGaudiDynamicTPCTestRelu,
                                         public testing::WithParamInterface<unsigned> {};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicTPCTestReluSingle, ::testing::Values(2, 3, 63, 64, 65, 127, 128, 129, 255, 256));

class SynGaudiDynamicTPCTestReluMulti : public SynGaudiDynamicTPCTestRelu,
                                        public testing::WithParamInterface<std::vector<unsigned>> {};

const std::vector<std::vector<unsigned>> basic_dynamic_batchsizeParams{{2},
                                                                       {3},
                                                                       {63},
                                                                       {64},
                                                                       {65},
                                                                       {127},
                                                                       {128},
                                                                       {129},
                                                                       {255},
                                                                       {256},
                                                                       {256, 129, 127, 64, 3, 2, 63, 65, 128, 255}};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicTPCTestReluMulti, ::testing::ValuesIn(basic_dynamic_batchsizeParams));

TEST_P_GC(SynGaudiDynamicTPCTestReluMulti, basic_dynamic_batchsize, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{

    unsigned inTensor, outTensor;

    const unsigned tensorDim = 4;
    const unsigned lastDim = tensorDim - 1;
    const unsigned minBatch = 2;
    const unsigned maxBatch = 256;
    unsigned H = 2;
    unsigned W = 2;
    unsigned C = 2;

    unsigned maxSizes[SYN_MAX_TENSOR_DIM] = {C, W, H, maxBatch};
    unsigned minSizes[SYN_MAX_TENSOR_DIM] = {C, W, H, minBatch};

    compileTest(tensorDim, minSizes, maxSizes, inTensor, outTensor);

    std::vector<unsigned> batches = GetParam();

    for (auto actualBatch : batches)
    {
        runTest(lastDim, actualBatch, maxSizes, inTensor, outTensor);

        float *inBuffer = castHostInBuffer<float>(inTensor);
        float *outBuffer = castHostOutBuffer<float>(outTensor);
        const uint64_t tensorBatchSizeElements = getNumberOfElements(minSizes, lastDim);
        const uint64_t tensorSizeElements      = tensorBatchSizeElements * actualBatch;
        const uint64_t garbageElements         = tensorBatchSizeElements * (maxBatch - actualBatch);

        // Test by the actual batch size.
        for (uint64_t i = 0; i < tensorSizeElements; i++)
        {
            ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, std::max(0.0f, inBuffer[i]))) << i;
            outBuffer[i] = 0;
        }
        for (uint64_t i = tensorSizeElements; i < tensorSizeElements + garbageElements; i++)
        {
            ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, 0.0f)) << i;
        }
    }
}

TEST_P_GC(SynGaudiDynamicTPCTestReluSingle, basic_dynamic_dim0, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{

    unsigned inTensor, outTensor;

    const unsigned tensorDim = 2;
    const unsigned dynamicDim = 0;
    const unsigned actualW = GetParam();
    const unsigned minW = 2;
    const unsigned maxW = 256;
    unsigned H = 256;

    unsigned maxSizes[SYN_MAX_TENSOR_DIM] = {maxW, H};
    unsigned minSizes[SYN_MAX_TENSOR_DIM] = {minW, H};

    unsigned actualSize = GetParam();
    unsigned actualSizes[] = {actualW, H};

    compileTest(tensorDim, minSizes, maxSizes, inTensor, outTensor);
    runTest(dynamicDim, actualSize, maxSizes, inTensor, outTensor);

    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    for (int i = 0 ; i < actualSizes[0] * actualSizes[1]; i++)
    {
        float expected = inBuffer[i] > 0 ? inBuffer[i] : 0;
        ASSERT_EQ(expected, outBuffer[i]) << i;
    }
}

TEST_F_GC(SynGaudiDynamicTPCTestRelu, DISABLED_big_tensor_but_under_4G_ASIC, {synDeviceGaudi2})
{

    unsigned inTensor, outTensor;

    unsigned tensorDim  = 2;
    unsigned dynamicDim = 0;
    unsigned actualW    = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    unsigned minW       = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    unsigned maxW       = 3ULL * 1024ULL * 1024ULL * 1024ULL;
    unsigned H          = 1;

    unsigned maxSizes[SYN_MAX_TENSOR_DIM] = {maxW, H};
    unsigned minSizes[SYN_MAX_TENSOR_DIM] = {minW, H};

    unsigned actualSize = actualW;
    unsigned actualSizes[] = {actualW, H};

    compileTest(tensorDim, minSizes, maxSizes, inTensor, outTensor);
    runTest(dynamicDim, actualSize, maxSizes, inTensor, outTensor);

    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    for (int i = 0 ; i < actualSizes[0] * actualSizes[1]; i++)
    {
        float expected = inBuffer[i] > 0 ? inBuffer[i] : 0;
        ASSERT_EQ(expected, outBuffer[i]) << i;
    }
}

class SynGaudiDynamicScatterTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicScatterTest, scatter, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned t61MinSizes[] = {1,1};
    unsigned t61MaxSizes[] = {1,4};
    unsigned t61Dim = 2;

    auto t61 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, t61MaxSizes, t61Dim,
                                   syn_type_int32, nullptr, "t61", 0, 0, nullptr, t61MinSizes);

    unsigned t62MinSizes[] = {1,1};
    unsigned t62MaxSizes[] = {1,4};
    unsigned t62Dim = 2;
    auto t62 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, t62MaxSizes, t62Dim,
                                   syn_type_int32, nullptr, "t62", 0, 0, nullptr, t62MinSizes);

    unsigned t63MinSizes[] = {4,4,1};
    unsigned t63MaxSizes[] = {4,4,4};
    unsigned t63Dim = 3;
    auto t63 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, t63MaxSizes, t63Dim,
                                   syn_type_float, nullptr, "t63", 0, 0, nullptr, t63MinSizes);

    unsigned t65MaxSizes[] = {4,4,4};
    unsigned t65Dim = 3;
    auto t65 = createShapeTensor(INPUT_TENSOR, t65MaxSizes, t65MaxSizes, t65Dim,
                                 syn_type_uint32, "t65");

    unsigned t64MaxSizes[] = {4,4,4};
    unsigned t64Dim = 3;
    auto t64 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, t64MaxSizes, t64Dim,
                                   syn_type_float, nullptr, "t64");

    ns_ScatterNDKernel::Params params;

    params.origIndicesDims = 2;
    memset(params.origIndicesShape, 0, sizeof(params.origIndicesShape));
    params.origIndicesShape[0] = 1;
    params.origIndicesShape[1] = 4;

    addNodeToGraph("scatter_nd_fwd_f32", {t61, t62, t63, t65}, {t64}, &params, sizeof(params));

    compileTopology();
    setActualSizes(t61, {1, 2});
    setActualSizes(t62, {1, 2});
    setActualSizes(t63, {4, 4, 2});
    runTopology();
}

class SynGaudiDynamicEwaddTest : public SynGaudiDynamicShapesTestsInfra,
                                        public testing::WithParamInterface<std::tuple<unsigned, unsigned, unsigned, unsigned>>
{
};

const std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>> ewaddParams =
  {
      { 10240, 10240, 6000, 6000 },
      { 10240, 1024, 6000, 600 },
      { 1024, 1024, 600, 600 },
      { 32, 32, 10, 10 },
      { 320, 320, 10, 10 },
      { 8, 8, 3, 3 },
      { 8, 13, 5, 11 }
  };

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicEwaddTest, ::testing::ValuesIn(ewaddParams));

TEST_P_GC(SynGaudiDynamicEwaddTest, ewadd, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    auto  param = GetParam();
    auto  maxW = std::get<0>(param);
    auto  maxH = std::get<1>(param);
    auto  actW = std::get<2>(param);
    auto  actH = std::get<3>(param);

    unsigned tMinSizes[] = {1, 1, 1, 1};
    unsigned tMaxSizes[] = {maxH, maxW, 1, 1};
    unsigned tActSizes[] = {actH, actW, 1, 1};
    unsigned tDim        = 4;

    unsigned tOutMinSizes[] = {1, 1, 1, 1};
    unsigned tOutMaxSizes[] = {1, maxW, 1, 1};
    unsigned tOutActSizes[] = {1, actW, 1, 1};
    unsigned tOutDim        = 4;

    auto t1 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_float,
                                  nullptr,
                                  "t1",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);
    auto t2 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_float,
                                  nullptr,
                                  "t2",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);

    auto t3 = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ZERO,
                                  nullptr,
                                  tOutMaxSizes,
                                  tOutDim,
                                  syn_type_float,
                                  nullptr,
                                  "t3",
                                  0,
                                  0,
                                  nullptr,
                                  tOutMinSizes);

    // Sanity testing for DSD + SFG
    //    Tensor* tpcTensor = reinterpret_cast<Tensor*>(getTensorByIndex(t3));
    //    tpcTensor->setTensorAsExternal(true);

    auto interm = createTensors(1,
                                OUTPUT_TENSOR,
                                false,  // isPersistent
                                "interm",
                                MEM_INIT_ALL_ZERO,
                                nullptr,  // initializer
                                tMaxSizes,
                                tDim,
                                syn_type_float,
                                nullptr,
                                0,
                                0,
                                nullptr,
                                false,
                                tMinSizes)[0];

    addNodeToGraph("add_fwd_f32", {t1, t2}, {interm}, nullptr, 0);

    ns_Reduction::Params params;
    params.reductionDimension = 0;  // sum and reduce first dimension size to 1
    addNodeToGraph("reduce_sum_fwd_f32", {interm}, {t3}, &params, sizeof(params));

    compileTopology();
    setActualSizes(t1, tActSizes);
    setActualSizes(t2, tActSizes);
    setActualSizes(t3, tOutActSizes);

    float* input1 = (float*)m_hostBuffers[t1];
    float* input2 = (float*)m_hostBuffers[t2];
    float* output = (float*)m_hostBuffers[t3];

    memset(output, 0, tOutActSizes[1] * sizeof(float));

    runTopology();

    // validate the output
    //

    for (int i = 0; i < tOutActSizes[1]; ++i)
    {
        float sum = 0;
        for (int j = 0; j < tActSizes[0]; ++j)
        {
            sum += input1[i * tActSizes[0] + j] + input2[i * tActSizes[0] + j];
        }
        ASSERT_NEAR(sum, output[i], sum * 1e-7 * tActSizes[0])
            << "Wrong result, got " << output[i] << " expected " << sum << " at index " << i;
    }

}

TEST_F_GC(SynGaudiDynamicEwaddTest, ewadd5, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned tMinSizes[] = {1, 1, 1, 1, 1};
    unsigned tMaxSizes[] = {5, 11, 17, 23, 103};
    unsigned tActSizes[] = {3, 7, 13, 19, 29};
    unsigned tDim        = 5;

    unsigned tOutMinSizes[] = {1, 1, 1, 1, 1};
    unsigned tOutMaxSizes[] = {1, 11, 17, 23, 103};
    unsigned tOutActSizes[] = {1, 7, 13, 19, 29};
    unsigned tOutDim        = 5;

    auto t1 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_float,
                                  nullptr,
                                  "t1",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);
    auto t2 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_float,
                                  nullptr,
                                  "t2",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);

    auto t3 = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ZERO,
                                  nullptr,
                                  tOutMaxSizes,
                                  tOutDim,
                                  syn_type_float,
                                  nullptr,
                                  "t3",
                                  0,
                                  0,
                                  nullptr,
                                  tOutMinSizes);

    // Sanity testing for DSD + SFG
    // Tensor* tpcTensor = reinterpret_cast<Tensor*>(getTensorByIndex(t3));
    // tpcTensor->setTensorAsExternal(true);

    auto interm = createTensors(1,
                                OUTPUT_TENSOR,
                                false,  // isPersistent
                                "interm",
                                MEM_INIT_ALL_ZERO,
                                nullptr,  // initializer
                                tMaxSizes,
                                tDim,
                                syn_type_float,
                                nullptr,
                                0,
                                0,
                                nullptr,
                                false,
                                tMinSizes)[0];

    addNodeToGraph("add_fwd_f32", {t1, t2}, {interm}, nullptr, 0);

    ns_Reduction::Params params;
    params.reductionDimension = 0;  // sum and reduce first dimension size to 1
    addNodeToGraph("reduce_sum_fwd_f32", {interm}, {t3}, &params, sizeof(params));

    compileTopology();
    setActualSizes(t1, tActSizes);
    setActualSizes(t2, tActSizes);
    setActualSizes(t3, tOutActSizes);

    float* input1 = (float*)m_hostBuffers[t1];
    float* input2 = (float*)m_hostBuffers[t2];
    float* output = (float*)m_hostBuffers[t3];

    memset(output, 0, tOutActSizes[1] * sizeof(float));

    runTopology();

    // validate the output
    //
    //
    auto outTotalSize = tOutActSizes[1] *  tOutActSizes[2] * tOutActSizes[3] * tOutActSizes[4];

    for (int i = 0; i < outTotalSize; ++i)
    {
        float sum = 0;
        for (int j = 0; j < tActSizes[0]; ++j)
        {
            sum += input1[i * tActSizes[0] + j] + input2[i * tActSizes[0] + j];
        }
        ASSERT_NEAR(sum, output[i], sum * 1e-7 * tActSizes[0])
            << "Wrong result, got " << output[i] << " expected " << sum << " at index " << i;
    }

}

class SynGaudiDynamicNotTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicNotTest, suggestedManipulationTest)
{
    unsigned dim        = 2;
    unsigned minSizes[] = {32, 1};
    unsigned maxSizes[] = {32, 2};

    int8_t inData[maxSizes[0] * maxSizes[1]];
    for (unsigned i = 0; i < maxSizes[0] * maxSizes[1]; i++)
    {
        inData[i] = i % 2;
    }

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                      (float*)inData,
                                      maxSizes,
                                      dim,
                                      syn_type_fixed,
                                      nullptr,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      minSizes);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       maxSizes,
                                       dim,
                                       syn_type_fixed,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       minSizes);

    addNodeToGraph("not_fwd_i8", {in}, {out});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(in, minSizes);
    setActualSizes(out, minSizes);
    runTopology(0, true);

    int8_t* inBuffer  = castHostInBuffer<int8_t>(in);
    int8_t* outBuffer = castHostOutBuffer<int8_t>(out);

    for (int i = 0; i < minSizes[0] * minSizes[1]; i++)
    {
        int8_t expected = !inBuffer[i];
        ASSERT_EQ(expected, outBuffer[i]) << i;
    }
}

class SynGaudiDynamicConstTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicConstTest, suggestedManipulationTest)
{
    unsigned dim        = 4;
    unsigned minSizes[] = {14, 14, 256, 12};
    unsigned maxSizes[] = {14, 14, 256, 24};

    unsigned in = createShapeTensor(INPUT_TENSOR, maxSizes, minSizes, dim);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       maxSizes,
                                       dim,
                                       syn_type_bf16,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       minSizes);

    ns_ConstantKernel::Params constantParam = {};
    constantParam.constant.f                = 0.2f;
    addNodeToGraph("constant_bf16", {in}, {out}, &constantParam, sizeof(constantParam));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(in, minSizes);
    setActualSizes(out, minSizes);
    runTopology(0, true);

    bfloat16* outBuffer = castHostOutBuffer<bfloat16>(out);
    bfloat16  expected(constantParam.constant.f);
    for (int i = 0; i < multiplyElements(minSizes, minSizes + dim); i++)
    {
        ASSERT_EQ(expected, outBuffer[i])
            << "mismatch in index " << i << "expected: " << expected << "result: " << outBuffer[i];
    }
}

class SynGaudiCatchNanTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiCatchNanTest, catch_nan)
{
    unsigned dim = 2;
    unsigned sizes[] = {2,2};
    float inData1[] = { 1, 1, 0, 0 };
    float inData2[] = { 1, 0, 1, 0 };
    unsigned in1 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                       (float*)inData1,
                                       sizes,
                                       dim,
                                       syn_type_float);
    unsigned in2 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                       (float*)inData2,
                                       sizes,
                                       dim,
                                       syn_type_float);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       sizes,
                                       dim,
                                       syn_type_float);
    addNodeToGraph("div_f32", {in1, in2}, {out}, nullptr, 0);

    compileTopology();
    runTopology();

    float* outBuffer = castHostOutBuffer<float>(out);
    ASSERT_TRUE(std::isinf(outBuffer[1]));
    ASSERT_TRUE(std::isnan(outBuffer[3]));
    ASSERT_FALSE(std::isinf(outBuffer[0]));
    ASSERT_FALSE(std::isinf(outBuffer[2]));
    ASSERT_FALSE(std::isnan(outBuffer[0]));
    ASSERT_FALSE(std::isnan(outBuffer[2]));
}

