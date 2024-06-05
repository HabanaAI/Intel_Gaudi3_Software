#include "eager_tests_defs.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "transpose_utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <vector>

using namespace eager_mode;

// Simple GEMM test for experimental purposes
TEST_F_GC(SynTrainingEagerTests, DISABLED_simple_gemm)
{
    TestSizeVec opASizes({4352, 4096});
    TestSizeVec opBSizes({2048, 4352});
    TestSizeVec outSizes({2048, 4096});
    const bool  runAndValidate = true;

    unsigned opA = createPersistTensor(INPUT_TENSOR,
                                       runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                       nullptr,
                                       opASizes.data(),
                                       opASizes.size());
    unsigned opB = createPersistTensor(INPUT_TENSOR,
                                       runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                       nullptr,
                                       opBSizes.data(),
                                       opBSizes.size());
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       runAndValidate ? MEM_INIT_ALL_ZERO : MEM_INIT_COMPILATION_ONLY,
                                       nullptr,
                                       outSizes.data(),
                                       outSizes.size());

    synGEMMParams params = {false, false};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {opA, opB}, {out}, (void*)&params, sizeof(params));

    compileTopology();

    if (runAndValidate)
    {
        runTopology();
        float* outPtr   = castHostBuffer<float>(out);
        TSize  elements = std::accumulate(outSizes.begin(), outSizes.end(), TSize(1), std::multiplies<TSize>());
        for (TSize i = 0; i < elements; ++i)
        {
            ASSERT_FLOAT_EQ(opASizes[0], outPtr[i])
                << "mismatch at index " << i << "\nexpected: " << opASizes[0] << "\nresult: " << outPtr[i];
        }
    }
}

TEST_F_GC(SynTrainingEagerTests, DISABLED_softmax_fwd)
{
    std::array<unsigned, 4> sizes = {128, 128, 16, 128};
    auto in  = createPersistTensor(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   sizes.data(),
                                   sizes.size(),
                                   syn_type_float);
    auto out = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   sizes.data(),
                                   sizes.size(),
                                   syn_type_float);

    unsigned char params[] = {0, 0, 0, 0};
    addNodeToGraph("softmax_fwd_f32", {in}, {out}, &params, sizeof(params), "softmax");
    compileAndRun();

    const float*   outBuf    = reinterpret_cast<float*>(m_hostBuffers[out]);
    const unsigned totalSize = multiplyElements(sizes.data(), sizes.data() + sizes.size());
    float          sum       = 0;
    for (unsigned i = 0; i < totalSize; ++i)
    {
        sum += outBuf[i];
    }
    ASSERT_EQ(sum, 1);
}

TEST_F_GC(SynTrainingEagerTests, DISABLED_softmax_bwd)
{
    std::array<unsigned, 4> sizes = {10, 128, 256, 4};
    unsigned                in1   = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       sizes.data(),
                                       sizes.size(),
                                       syn_type_single);

    unsigned in2 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       sizes.data(),
                                       sizes.size(),
                                       syn_type_single);

    unsigned out =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), syn_type_single);

    unsigned char softmaxParams[] = {0, 0, 0, 0};
    addNodeToGraph("softmax_bwd_f32", {in1, in2}, {out}, (void*)softmaxParams, 4, "softmax");

    compileAndRun();
}

TEST_F_GC(SynTrainingEagerTests, broadcast)
{
    std::array<unsigned, 1> inSizes  = {1};
    auto                    in       = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  inSizes.data(),
                                  inSizes.size(),
                                  syn_type_float);
    std::array<unsigned, 5> outSizes = {127, 192, 4, 4, 11};
    auto                    out      = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   outSizes.data(),
                                   outSizes.size(),
                                   syn_type_float);

    addNodeToGraph("broadcast", {in}, {out}, nullptr, 0, "broadcast");
    compileAndRun();

    const float    inputVal  = *reinterpret_cast<float*>(m_hostBuffers[in]);
    const float*   outBuf    = reinterpret_cast<float*>(m_hostBuffers[out]);
    const unsigned totalSize = multiplyElements(outSizes.data(), outSizes.data() + outSizes.size());
    for (unsigned i = 0; i < totalSize; ++i)
    {
        ASSERT_EQ(outBuf[i], inputVal);
    }
}

TEST_F_GC(SynTrainingEagerTests, broadcast_b2b)
{
    std::array<unsigned, 1> inSizes   = {1};
    auto                    in        = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  inSizes.data(),
                                  inSizes.size(),
                                  syn_type_float);
    std::array<unsigned, 1> outSizes1 = {32};
    auto                    out1      = createPersistTensor(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    outSizes1.data(),
                                    outSizes1.size(),
                                    syn_type_float);
    std::array<unsigned, 3> outSizes2 = {32, 10, 10};
    auto                    out2      = createPersistTensor(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    outSizes2.data(),
                                    outSizes2.size(),
                                    syn_type_float);

    addNodeToGraph("broadcast", {in}, {out1}, nullptr, 0, "broadcast1");
    addNodeToGraph("broadcast", {out1}, {out2}, nullptr, 0, "broadcast2");
    compileAndRun();

    const float    inputVal  = *reinterpret_cast<float*>(m_hostBuffers[in]);
    const float*   outBuf    = reinterpret_cast<float*>(m_hostBuffers[out2]);
    const unsigned totalSize = multiplyElements(outSizes2.data(), outSizes2.data() + outSizes2.size());
    for (unsigned i = 0; i < totalSize; ++i)
    {
        ASSERT_EQ(outBuf[i], inputVal);
    }
}

TEST_F_GC(SynTrainingEagerTests, broadcast_tpc_broadcast)
{
    std::array<unsigned, 1> inSizes         = {1};
    auto                    in              = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  inSizes.data(),
                                  inSizes.size(),
                                  syn_type_float);
    std::array<unsigned, 1> broadcastSizes1 = {32};
    auto                    broadcastOut1   = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             broadcastSizes1.data(),
                                             broadcastSizes1.size(),
                                             syn_type_float);

    auto addIn  = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_ALL_ONES,
                                     nullptr,
                                     broadcastSizes1.data(),
                                     broadcastSizes1.size(),
                                     syn_type_float);
    auto addOut = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      broadcastSizes1.data(),
                                      broadcastSizes1.size(),
                                      syn_type_float);

    std::array<unsigned, 3> broadcastSizes2 = {32, 10, 10};
    auto                    broadcastOut2   = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             broadcastSizes2.data(),
                                             broadcastSizes2.size(),
                                             syn_type_float);

    addNodeToGraph("broadcast", {in}, {broadcastOut1}, nullptr, 0, "broadcast1");
    addNodeToGraph("add_fwd_f32", {addIn, broadcastOut1}, {addOut}, nullptr, 0, "add_fwd_f32");
    addNodeToGraph("broadcast", {addOut}, {broadcastOut2}, nullptr, 0, "broadcast2");

    compileAndRun();

    const float    inputVal = *reinterpret_cast<float*>(m_hostBuffers[in]) + 1; // addIn tensor is all ones
    const float*   outBuf   = reinterpret_cast<float*>(m_hostBuffers[broadcastOut2]);
    const unsigned totalSize =
        multiplyElements(broadcastSizes2.data(), broadcastSizes2.data() + broadcastSizes2.size());
    for (unsigned i = 0; i < totalSize; ++i)
    {
        ASSERT_EQ(outBuf[i], inputVal);
    }
}

TEST_F_GC(SynTrainingEagerTests, tpc_broadcast_tpc)
{
    std::array<unsigned, 1> reluSize = {1};
    auto                    reluIn   = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      reluSize.data(),
                                      reluSize.size(),
                                      syn_type_float);
    auto                    reluOut  = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       reluSize.data(),
                                       reluSize.size(),
                                       syn_type_float);
    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "relu_fwd_f32");

    std::array<unsigned, 5> broadcastSizes = {127, 192, 4, 4, 11};
    auto                    broadcastOut   = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            broadcastSizes.data(),
                                            broadcastSizes.size(),
                                            syn_type_float);
    addNodeToGraph("broadcast", {reluOut}, {broadcastOut}, nullptr, 0, "broadcast");

    auto addIn  = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     broadcastSizes.data(),
                                     broadcastSizes.size(),
                                     syn_type_float);
    auto addOut = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      broadcastSizes.data(),
                                      broadcastSizes.size(),
                                      syn_type_float);
    addNodeToGraph("add_fwd_f32", {addIn, broadcastOut}, {addOut}, nullptr, 0, "add_fwd_f32");

    compileAndRun();

    const float    reluOutVal = *reinterpret_cast<float*>(m_hostBuffers[reluOut]);
    const float*   addInBuf   = reinterpret_cast<float*>(m_hostBuffers[addIn]);
    const float*   addOutBuf  = reinterpret_cast<float*>(m_hostBuffers[addOut]);
    const unsigned totalSize  = multiplyElements(broadcastSizes.data(), broadcastSizes.data() + broadcastSizes.size());
    for (unsigned i = 0; i < totalSize; ++i)
    {
        ASSERT_EQ(addOutBuf[i], reluOutVal + addInBuf[i]);
    }
}

TEST_F_GC(SynTrainingEagerTests, high_mme_activations)
{
    pushGlobalConf("ENABLE_EAGER_SB_REUSE", "1");  // This test focus on multi MME activation
    pushGlobalConf("ENABLE_EAGER_MME_CONCURRENCY", "0");  // CDC causes memset to be added, this is not what the test intends to do

    // Vars for debug:
    unsigned           level    = 2;  // Level determines number of activations: 18, 26, 36, 262
    constexpr unsigned hwSize[] = {8, 9, 10, 22};

    synConvolutionParams convParams;
    convParams.kW   = 3;
    convParams.kH   = 5;
    std::array<unsigned, SYN_MAX_TENSOR_DIM> xSize = {1, hwSize[level], hwSize[level], 500, 1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> wSize = {64, xSize[0], convParams.kW, convParams.kH, 1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> ySize = {
        64,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3],
        1};

    unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        ySize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_bf16);
    unsigned x    = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     xSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_bf16);
    unsigned dedw = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,  // initializer
                                        wSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_float);
    addNodeToGraph("dedw", {dedy, x}, {dedw}, &convParams, sizeof(convParams));
    compileAndRun();

    CoordArray wrongIdx;
    float      expectedResult = 0;
    bool       ret            = checkMmeOp(m_tensorDescs[x],
                          (char*)m_hostBuffers[x],
                          m_tensorDescs[dedw],
                          (char*)m_hostBuffers[dedw],
                          m_tensorDescs[dedy],
                          (char*)m_hostBuffers[dedy],
                          convParams,
                          REFERENCE_OP_DEDW,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[dedw].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ((ret), true)
        << "Wrong value for DEDW op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
        << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedw].m_dataType, m_hostBuffers[dedw])
        << " Expected: " << expectedResult;
}

struct SynGaudi2EagerDmaCopyTest
: SynTrainingEagerTests
, testing::WithParamInterface<std::tuple<synDataType, SizeVector>>
{
};

TEST_P_GC(SynGaudi2EagerDmaCopyTest, t, {synDeviceGaudi2})
{
    ScopedConfigurationChange scc("ENABLE_INTERNAL_NODES", "true");

    const auto& [dataType, sizes] = GetParam();

    // TODO: createPersistTensor taking TSize...
    std::vector<unsigned> tmp_sizes(sizes.data(), sizes.data() + sizes.size());

    auto in = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tmp_sizes.data(),
                                  tmp_sizes.size(),
                                  dataType);
    auto out =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tmp_sizes.data(), tmp_sizes.size(), dataType);
    addNodeToGraph(NodeFactory::dmaMemcpyNodeTypeName, {in}, {out});
    compileAndRun();

    const auto* pInput  = reinterpret_cast<char*>(m_hostBuffers[in]);   // castHostInBuffer<uint8_t>(in);
    const auto* pOutput = reinterpret_cast<char*>(m_hostBuffers[out]);  // castHostOutBuffer<uint8_t>(out);

    const size_t numBytes = prod(sizes) * getElementSizeInBytes(dataType);
    for (size_t i = 0; i < numBytes; ++i)
    {
        ASSERT_EQ(pInput[i], pOutput[i]) << "Mismatch for output at byte index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(dma_copy_test,
                         SynGaudi2EagerDmaCopyTest,
                         ::testing::Combine(::testing::ValuesIn({syn_type_uint8, syn_type_bf16, syn_type_float}),
                                            ::testing::ValuesIn({SizeVector {1},
                                                                 SizeVector {64},
                                                                 SizeVector {1024},
                                                                 SizeVector {1024ul * 1024, 1024},
                                                                 SizeVector {63, 1, 1, 63},
                                                                 SizeVector {65, 1, 1, 65}})),
                         [](const testing::TestParamInfo<SynGaudi2EagerDmaCopyTest::ParamType>& info) {
                             const auto&            dataType = std::get<0>(info.param);
                             const auto&            sizes    = std::get<1>(info.param);
                             const std::string_view dt       = [](synDataType dataType) {
                                 switch (dataType)
                                 {
                                     case syn_type_uint8:
                                         return "u8";
                                     case syn_type_bf16:
                                         return "bf16";
                                     case syn_type_float:
                                         return "f32";
                                     default:
                                         assert(false);
                                         return "unknown";
                                 }
                             }(dataType);
                             return fmt::format("dma_copy_{}_{}{}",
                                                dt,
                                                fmt::join(sizes.begin(), sizes.end(), "_"),
                                                prod(sizes) > (1ul << 20) ? "_ASIC" : "");
                         });

struct SynGaudi2EagerDmaTranposeTest
: SynTrainingEagerTests
, testing::WithParamInterface<std::tuple<synDataType, std::tuple<std::string_view, SizeVector, SizeVector>>>
{
};

TEST_P_GC(SynGaudi2EagerDmaTranposeTest, t, {synDeviceGaudi2})
{
    ScopedConfigurationChange scc("ENABLE_INTERNAL_NODES", "true");

    const auto& [dataType, rest]       = GetParam();
    const auto& [guid, sizes, indices] = rest;

    assert(guid == NodeFactory::transposeNodeTypeName || guid == NodeFactory::transposeDmaNodeTypeName);

    {
        assert(sizes.size() == indices.size());
        SizeVector tmp(indices);
        SizeVector tmp2(indices.size());
        std::sort(tmp.begin(), tmp.end());
        std::iota(tmp2.begin(), tmp2.end(), 0);
        assert(tmp == tmp2);
    }

    SizeVector outDims(sizes.size());
    for (size_t i = 0; i < outDims.size(); ++i)
    {
        outDims[i] = sizes[indices[i]];
    }

    // TODO: createPersistTensor taking TSize...
    std::vector<unsigned> tmp_sizes(sizes.data(), sizes.data() + sizes.size());
    std::vector<unsigned> tmp_outDims(outDims.data(), outDims.data() + outDims.size());

    auto in  = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tmp_sizes.data(),
                                  tmp_sizes.size(),
                                  dataType);
    auto out = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   tmp_outDims.data(),
                                   tmp_outDims.size(),
                                   dataType);

    if (guid == NodeFactory::transposeNodeTypeName)
    {
        synTransposeParams params;
        params.tensorDim = indices.size();
        for (size_t i = 0; i < std::size(params.permutation); ++i)
        {
            params.permutation[i] = TransposePermutationDim(i < indices.size() ? indices[i] : i);
        }
        addNodeToGraph(std::string(guid).c_str(), {in}, {out}, &params, sizeof(params));
    }
    else
    {
        assert(guid == NodeFactory::transposeDmaNodeTypeName);
        addNodeToGraph(std::string(guid).c_str(), {in}, {out});  //, &params, sizeof(params));
    }

    compileAndRun();

    const auto numBytes = getElementSizeInBytes(dataType);

    assert(sizes.size() <= 5);
    const std::array<size_t, 5> inSizes = {sizes.size() > 0 ? sizes[0] : 0,
                                           sizes.size() > 1 ? sizes[1] : 1,
                                           sizes.size() > 2 ? sizes[2] : 1,
                                           sizes.size() > 3 ? sizes[3] : 1,
                                           sizes.size() > 4 ? sizes[4] : 1};
    std::array<size_t, 5> inStrides;
    for (size_t i = 0; i < inSizes.size(); ++i)
    {
        inStrides[i] = i > 0 ? inStrides[i - 1] * inSizes[i - 1] : numBytes;
    }

    const std::array<size_t, 5> outSizes = {indices.size() > 0 ? sizes[indices[0]] : 0,
                                            indices.size() > 1 ? sizes[indices[1]] : 1,
                                            indices.size() > 2 ? sizes[indices[2]] : 1,
                                            indices.size() > 3 ? sizes[indices[3]] : 1,
                                            indices.size() > 4 ? sizes[indices[4]] : 1};
    std::array<size_t, 5> outStrides;
    for (size_t i = 0; i < inSizes.size(); ++i)
    {
        outStrides[i] = i > 0 ? outStrides[i - 1] * outSizes[i - 1] : numBytes;
    }

    const auto* pInput  = reinterpret_cast<char*>(m_hostBuffers[in]);   // castHostInBuffer<uint8_t>(in);
    const auto* pOutput = reinterpret_cast<char*>(m_hostBuffers[out]);  // castHostOutBuffer<uint8_t>(out);

    // clang-format off
    for (size_t i4 = 0; i4 < inSizes[4]; ++i4)
    for (size_t i3 = 0; i3 < inSizes[3]; ++i3)
    for (size_t i2 = 0; i2 < inSizes[2]; ++i2)
    for (size_t i1 = 0; i1 < inSizes[1]; ++i1)
    for (size_t i0 = 0; i0 < inSizes[0]; ++i0)
    {
        const size_t is[] = {i0, i1, i2, i3, i4};
        const size_t inIdx  = (i0 * inStrides[0]) +
                              (i1 * inStrides[1]) +
                              (i2 * inStrides[2]) +
                              (i3 * inStrides[3]) +
                              (i4 * inStrides[4]);
        const size_t outIdx = (indices.size() > 0 ? (is[indices[0]] * outStrides[0]) : 0) +
                              (indices.size() > 1 ? (is[indices[1]] * outStrides[1]) : 0) +
                              (indices.size() > 2 ? (is[indices[2]] * outStrides[2]) : 0) +
                              (indices.size() > 3 ? (is[indices[3]] * outStrides[3]) : 0) +
                              (indices.size() > 4 ? (is[indices[4]] * outStrides[4]) : 0);
        for (size_t i = 0; i < numBytes; ++i)
        {
            ASSERT_EQ(pInput[inIdx + i], pOutput[outIdx + i]) << "Mismatch for output at { inIdx: " <<  inIdx << ", outIdx: " << outIdx << ", byte: " << i << "}";
        }
    }
    // clang-format on
}

INSTANTIATE_TEST_SUITE_P(
    dma_tranpose_test,
    SynGaudi2EagerDmaTranposeTest,
    ::testing::Combine(
        ::testing::Values(syn_type_uint8, syn_type_bf16, syn_type_float),
        ::testing::Values(
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {1, 128}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {2, 128}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {1024, 1}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {64, 64}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {63, 65}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {2, 3, 4, 5}, SizeVector {2, 3, 0, 1}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {2, 3, 4, 5}, SizeVector {3, 0, 1, 2}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {1024ul * 1024, 1024}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {32, 2, 1, 64}, SizeVector {1, 0, 2, 3}),
            std::make_tuple(NodeFactory::transposeNodeTypeName, SizeVector {32, 2, 1, 64}, SizeVector {3, 1, 2, 0}),

            std::make_tuple(NodeFactory::transposeDmaNodeTypeName, SizeVector {1, 128}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeDmaNodeTypeName, SizeVector {2, 128}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeDmaNodeTypeName, SizeVector {1024, 1}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeDmaNodeTypeName, SizeVector {64, 64}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeDmaNodeTypeName, SizeVector {63, 65}, SizeVector {1, 0}),
            std::make_tuple(NodeFactory::transposeDmaNodeTypeName,
                            SizeVector {1024ul * 1024, 1024},
                            SizeVector {1, 0}))),
    [](const testing::TestParamInfo<SynGaudi2EagerDmaTranposeTest::ParamType>& info) {
        const auto&            dataType = std::get<0>(info.param);
        const auto&            guid     = std::get<0>(std::get<1>(info.param));
        const auto&            sizes    = std::get<1>(std::get<1>(info.param));
        const auto&            indices  = std::get<2>(std::get<1>(info.param));
        const std::string_view dt       = [](synDataType dataType) {
            switch (dataType)
            {
                case syn_type_uint8:
                    return "u8";
                case syn_type_bf16:
                    return "bf16";
                case syn_type_float:
                    return "f32";
                default:
                    assert(false);
                    return "unknown";
            }
        }(dataType);
        return fmt::format("guid_{}__dt_{}__dims_{}__indices_{}{}",
                           guid,
                           dt,
                           fmt::join(sizes.begin(), sizes.end(), "_"),
                           fmt::join(indices.begin(), indices.end(), "_"),
                           prod(sizes) > (1ul << 20) ? "_ASIC" : "");
    });

TEST_F_GC(SynTrainingEagerTests, transpose_bf16, {synDeviceGaudi2})
{
    ScopedConfigurationChange scc("ENABLE_INTERNAL_NODES", "true");

    // CNHW
    int C;
    int W;
    std::tie(C, W) = std::make_tuple(5, 7);

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
    addNodeToGraph(NodeFactory::transposeDmaNodeTypeName, {in}, {out});
    compileAndRun();

    bfloat16* pInput  = castHostInBuffer<bfloat16>(in);
    bfloat16* pOutput = castHostOutBuffer<bfloat16>(out);
    for (uint32_t j = 0; j < W; j++)
    {
        for (uint32_t i = 0; i < C; i++)
        {
            ASSERT_EQ(pInput[i + j * C].value(), pOutput[j + i * W].value())
                << "Mismatch for at index " << i << ", " << j;
        }
    }
}

TEST_F_GC(SynTrainingEagerTests, nDims_neg)
{
    std::array<unsigned, 8> sizes = {2, 2, 2, 2, 2, 2, 2, 2};
    auto in  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), sizes.size());
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, sizes.data(), sizes.size());

    addNodeToGraph("neg_fwd_f32", {in}, {out}, nullptr, 0, "neg");
    compileAndRun();

    const auto*  pInBuf     = castHostInBuffer<float>(in);
    const auto*  pOutBuf    = castHostOutBuffer<float>(out);
    const size_t totalSizes = prod(sizes);
    for (size_t i = 0; i < totalSizes; ++i)
    {
        ASSERT_EQ(pOutBuf[i], -pInBuf[i]) << "Mismatch for at index " << i;
    }
}

// testing memcpy kernel with dims = 6
TEST_F_GC(SynTrainingEagerTests, nDims_memcpy_test)
{
    std::array<unsigned, 6> sizes = {4, 5, 8, 2, 3, 6};
    auto in  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), sizes.size());
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, sizes.data(), sizes.size());
    addNodeToGraph("memcpy", {in}, {out}, nullptr, 0, "memcpy");
    compileAndRun();
    const auto* pInBuf  = castHostInBuffer<float>(in);
    const auto* pOutBuf = castHostOutBuffer<float>(out);
    ASSERT_TRUE(std::memcmp(pInBuf, pOutBuf, sizeof(float) * prod(sizes)) == 0);
}

// test fallback to DMA memset in case of a type not supported for TPC memset
TEST_F_GC(SynTrainingEagerTests, DMA_memset, {synDeviceGaudi2})
{
    std::array<unsigned, 5> sizes = {4, 5, 8, 2, 3};
    auto all_zeros = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), syn_type_uint32);
    auto out       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes.data(), sizes.size(), syn_type_uint32);
    addNodeToGraph("memset", {}, {out}, nullptr, 0, "memset");
    compileAndRun();
    const auto* pZeroBuf = castHostInBuffer<unsigned>(all_zeros);
    const auto* pOutBuf  = castHostOutBuffer<unsigned>(out);
    ASSERT_TRUE(std::memcmp(pZeroBuf, pOutBuf, sizeof(unsigned) * prod(sizes)) == 0);
}

TEST_F_GC(SynTrainingEagerTests, fp8_memcpy, {synDeviceGaudi2})
{
    std::array<unsigned, 4> sizes = {2, 5, 10, 100};
    auto                    in    = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  sizes.data(),
                                  sizes.size(),
                                  syn_type_fp8_152);
    auto                    out =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), syn_type_fp8_152);
    addNodeToGraph("memcpy", {in}, {out}, nullptr, 0, "memcpy");
    compileAndRun();
    const auto* pInBuf  = reinterpret_cast<std::byte*>(m_hostBuffers[in]);
    const auto* pOutBuf = reinterpret_cast<std::byte*>(m_hostBuffers[out]);
    ASSERT_TRUE(std::memcmp(pInBuf, pOutBuf, sizeof(std::byte) * prod(sizes)) == 0);
}

TEST_F_GC(SynTrainingEagerTests, reshape_test)
{
    const unsigned FCD    = 4;
    const unsigned WIDTH  = 4;
    const unsigned HEIGHT = 4;
    const unsigned BATCH  = 1;

    unsigned int flat_input_dimensions[]  = {FCD, WIDTH, HEIGHT, BATCH};
    unsigned int flat_output_dimensions[] = {FCD * WIDTH, HEIGHT};

    const unsigned int flat_input_dim_num = 4;
    const unsigned int flat_out_dim_num   = 2;

    const unsigned inputSize  = FCD * WIDTH * HEIGHT * BATCH;
    float*         inputArray = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        inputArray[i] = (float)i;
    }

    unsigned inputTensor      = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               inputArray,
                                               flat_input_dimensions,
                                               flat_input_dim_num,
                                               syn_type_single,
                                               nullptr,
                                               "inputTensor");
    unsigned flatOutputTensor = createPersistTensor(OUTPUT_TENSOR,
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    flat_output_dimensions,
                                                    flat_out_dim_num,
                                                    syn_type_single,
                                                    nullptr,
                                                    "flatOutputTensor");

    // Flatten setting
    synFlattenParams flattenAttr;
    flattenAttr.axis = 1;

    addNodeToGraph(NodeFactory::flattenNodeTypeName,
                   {inputTensor},
                   {flatOutputTensor},
                   (void*)&flattenAttr,
                   sizeof(synFlattenParams));

    compileAndRun();

    float* pFlatOutputBuffer = (float*)m_hostBuffers[flatOutputTensor];

    // validate the output
    for (unsigned i = 0; i < inputSize; i++)
    {
        ASSERT_EQ(inputArray[i], pFlatOutputBuffer[i])
            << "Mismatch for at index " << i << " Expected:" << inputArray[i] << " Result: " << pFlatOutputBuffer[i]
            << " operand " << inputArray[i];
    }
    delete[] inputArray;
}

TEST_F_GC(SynTrainingEagerTests, consecutive_add_test)
{
    const uint32_t dims   = 2;
    const unsigned WIDTH  = 4;
    const unsigned HEIGHT = 4;

    unsigned int add_dimensions[] = {WIDTH, HEIGHT};

    std::array<unsigned, 3> inputs;
    std::array<unsigned, 2> outputs;
    for (unsigned i = 0; i < inputs.size(); ++i)
    {
        inputs[i] = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        add_dimensions,
                                        dims,
                                        syn_type_single,
                                        nullptr,
                                        ("in" + std::to_string(i)).c_str());
    }

    for (unsigned i = 0; i < outputs.size(); ++i)
    {
        outputs[i] = createPersistTensor(OUTPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,  // initializer
                                         add_dimensions,
                                         dims,
                                         syn_type_single,
                                         nullptr,
                                         ("out" + std::to_string(i)).c_str());
    }

    addNodeToGraph("add_fwd_f32", {inputs[0], inputs[1]}, {outputs[0]});
    addNodeToGraph("relu_fwd_f32", {outputs[0]}, {outputs[1]});
    compileAndRun();
    float* in1        = (float*)m_hostBuffers[inputs[0]];
    float* in2        = (float*)m_hostBuffers[inputs[1]];
    float* outputData = (float*)m_hostBuffers[outputs[1]];
    for (uint32_t i = 0; i < multiplyElements(add_dimensions, add_dimensions + dims); ++i)
    {
        float res = in1[i] + in2[i];
        ASSERT_EQ(outputData[i], res > 0 ? res : 0);
    }
}

// this test also makes sure these nodes work with 16 tensors
TEST_F_GC(SynTrainingEagerTests, multiple_logical_ops)
{
    const unsigned WIDTH  = 4;
    const unsigned HEIGHT = 4;
    const unsigned BATCH  = 15;

    uint32_t     dims           = 3;
    unsigned int splitInDims[]  = {WIDTH, HEIGHT, BATCH};
    unsigned int splitOutDims[] = {WIDTH, HEIGHT, 1};
    unsigned int reshapedDims[] = {WIDTH * HEIGHT};
    unsigned int concatOutDims[] = {WIDTH * HEIGHT, BATCH};

    auto                        splitIn = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,  // initializer
                                       splitInDims,
                                       dims,
                                       syn_type_single,
                                       nullptr,
                                       "splitIn");
    std::array<unsigned, BATCH> splitOut;
    std::array<unsigned, BATCH> reshapeOut;
    for (unsigned i = 0; i < splitOut.size(); ++i)
    {
        splitOut[i] = createTensor(OUTPUT_TENSOR,
                                   MEM_INIT_NONE,
                                   nullptr,  // initializer
                                   splitOutDims,
                                   dims,
                                   syn_type_single);
    }

    for (unsigned i = 0; i < reshapeOut.size(); ++i)
    {
        reshapeOut[i] = createTensor(OUTPUT_TENSOR,
                                     MEM_INIT_NONE,
                                     nullptr,  // initializer
                                     reshapedDims,
                                     1,
                                     syn_type_single);
    }

    auto concatOut = createPersistTensor(OUTPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,  // initializer
                                         concatOutDims,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         "concatOut");

    synAxisParams splitParams;
    splitParams.axis = 2;
    addNodeToGraph("split",
                   {splitIn},
                   std::vector<unsigned>(splitOut.begin(), splitOut.end()),
                   &splitParams,
                   sizeof(splitParams));
    for (auto i = 0; i < BATCH; ++i)
    {
        addNodeToGraph("reshape", {splitOut[i]}, {reshapeOut[i]});
    }
    synAxisParams concatParams;
    concatParams.axis = 1;
    addNodeToGraph("concat",
                   std::vector<unsigned>(reshapeOut.begin(), reshapeOut.end()),
                   {concatOut},
                   &concatParams,
                   sizeof(concatParams));
    compileAndRun();
    float* in1        = (float*)m_hostBuffers[splitIn];
    float* outputData = (float*)m_hostBuffers[concatOut];
    for (uint32_t i = 0; i < multiplyElements(splitInDims, splitInDims + dims); ++i)
    {
        ASSERT_EQ(outputData[i], in1[i]);
    }
}

TEST_F_GC(SynTrainingEagerTests, cast_i32_to_i64, {synDeviceGaudi2})
{
    const unsigned FCD    = 4;
    const unsigned WIDTH  = 4;
    const unsigned HEIGHT = 4;
    const unsigned BATCH  = 1;

    unsigned int dims[] = {FCD, WIDTH, HEIGHT, BATCH};

    const unsigned int dimNum = 4;

    unsigned inputTensor  = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               dims,
                                               dimNum,
                                               syn_type_int32,
                                               nullptr,
                                               "inputTensor");
    unsigned outputTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                dims,
                                                dimNum,
                                                syn_type_int64,
                                                nullptr,
                                                "outputTensor");

    addNodeToGraph("cast_i32_to_i64", {inputTensor}, {outputTensor});

    compileAndRun();

    int32_t* inputData  = (int32_t*)m_hostBuffers[inputTensor];
    int64_t* outputData = (int64_t*)m_hostBuffers[outputTensor];
    for (uint32_t index = 0; index < multiplyElements(dims, dims + dimNum); ++index)
    {
        ASSERT_EQ(outputData[index], inputData[index]);
    }
}

TEST_F_GC(SynTrainingEagerTests, TpcRangeTemplate)
{
    constexpr auto val = 5;

    // TODO: why doesn't createPersistTensor take a ptr to const...
    std::array<unsigned, 1> dims {val};
    auto                    out = createPersistTensor(OUTPUT_TENSOR,
                                   /*MEM_INIT_ALL_ZERO*/ MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   dims.data(),
                                   dims.size(),
                                   syn_type_int32);
    (void)out;

    auto rangeParams    = ns_RangeKernel::Params {};
    rangeParams.start.i = 0;
    rangeParams.limit.i = val;
    rangeParams.delta.i = 1;

    std::array<int32_t, val> outputRef;
    std::iota(std::begin(outputRef), std::end(outputRef), 0);

    addNodeToGraph("range_i32", &rangeParams, sizeof(rangeParams));
    compileAndRun();

    // validate results
    const auto count = getTensorElementCount(out);
    ASSERT_EQ(count, val);
    const int32_t* pOutputBuffer = castHostOutBuffer<int32_t>(out);
    const int32_t* pOutputRef    = outputRef.data();
    for (uint64_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(pOutputBuffer[i], pOutputRef[i]) << "Mismatch at index " << i << " Expected:" << pOutputRef[i]
                                                   << " Result: " << pOutputBuffer[i] << std::endl;
    }
}

TEST_F_GC(SynDualExecutionGaudiTestInfra, nDims_add_test)
{
    std::array<unsigned, 6> sizes = {4, 5, 8, 2, 3, 6};
    createPersistTensors(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), sizes.size());
    createPersistTensors(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), sizes.size());
    auto ouputTensorIndexPair =
        createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size());

    addNodesToGraphs("add_f32", nullptr, 0, "add_node");

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[ouputTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[ouputTensorIndexPair.eager]);
    for (uint64_t i = 0; i < prod(sizes); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingEagerTests, scatter_fwd_test)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4]  = {1, 1, 2, 1};

    unsigned inputData =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4, syn_type_single);
    unsigned inputIndices =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4, syn_type_int32);
    unsigned inputUpdates =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4, syn_type_single);

    unsigned outputTensor =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4, syn_type_single);

    addNodeToGraph("scatter_fwd_f32",
                   {inputData, inputIndices, inputUpdates},
                   {outputTensor},
                   &params,
                   sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology(0, true);

    float* pFmOutput = (float*)m_hostBuffers[outputTensor];

    float outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynTrainingEagerTests, b2b_tpc)
{
    bool addAddNode = true;  // Add "ADD" node after the relu - a flag for debug

    unsigned   tensorsizes[] = {5, 5, 5, 5};
    const auto memorySize    = 3000;

    // fills vector with floats.
    std::vector<float> floats;
    float              x = 0.;
    std::generate_n(std::back_inserter(floats), getNumberOfElements(tensorsizes), [&]() mutable {
        if (x > 1) x = 0;
        return x += 0.1;
    });
    unsigned sectionIndex1 = createSection(memorySize);
    unsigned sectionIndex2 = createSection(memorySize);
    unsigned sectionIndex3 = createSection(memorySize);
    unsigned sectionIndex4 = createSection(memorySize);
    unsigned sectionIndex5 = createSection(memorySize);

    unsigned inputIndex1 = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               floats.data(),
                                               tensorsizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               nullptr,
                                               0,
                                               0,
                                               &sectionIndex1);

    unsigned inputIndex2 = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               floats.data(),
                                               tensorsizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               nullptr,
                                               0,
                                               0,
                                               &sectionIndex2);

    unsigned outputIndex1 = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                tensorsizes,
                                                4,
                                                syn_type_single,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                &sectionIndex3);

    unsigned outputIndex2 = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                tensorsizes,
                                                4,
                                                syn_type_single,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                &sectionIndex4);

    unsigned outputIndex3 = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                tensorsizes,
                                                4,
                                                syn_type_single,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                &sectionIndex5);

    // using relu and not memcpy since we can optimize memcpy out.
    synNodeId node1Id;
    synNodeId node2Id;
    addNodeToGraph("relu_fwd_f32",
                   TensorIndices {inputIndex1},
                   TensorIndices {outputIndex1},
                   nullptr,
                   0,
                   nullptr,
                   0,
                   &node1Id);
    addNodeToGraph("relu_fwd_f32",
                   TensorIndices {inputIndex2},
                   TensorIndices {outputIndex2},
                   nullptr,
                   0,
                   nullptr,
                   0,
                   &node2Id);
    if (addAddNode)
    {
        addNodeToGraph("add_f32", {outputIndex1, outputIndex2}, {outputIndex3}, nullptr, 0, "add");
    }

    compileAndRun();

    const float* out1 = reinterpret_cast<float*>(m_hostBuffers[outputIndex1]);
    const float* out2 = reinterpret_cast<float*>(m_hostBuffers[outputIndex2]);
    const float* out3 = reinterpret_cast<float*>(m_hostBuffers[outputIndex3]);
    for (size_t i = 0; i < floats.size(); ++i)
    {
        ASSERT_FLOAT_EQ(out1[i], out2[i]);
        if (addAddNode)
        {
            ASSERT_FLOAT_EQ(out1[i] + out2[i], out3[i]);
        }
    }
}

TEST_F_GC(SynTrainingEagerTests, b2b_mme)
{
    pushGlobalConf("ENABLE_EAGER_SB_REUSE", "1");  // This test focus on multi node with multi MME activation
    pushGlobalConf("ENABLE_EAGER_MME_CONCURRENCY", "0");  // CDC causes memset to be added, this is not what the test intends to do
    // Flags for debug:
    bool addDedx             = true;  // Add DEDX node
    bool addDedw             = true;  // Add DEDW node
    bool use3DedwActivations = true;  // Make DEDW 3 activations

    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 3;
    constexpr uint32_t yChannels  = 64;
    std::array<unsigned, SYN_MAX_TENSOR_DIM> xSize = {64, 7, 7, (use3DedwActivations ? 128 : 64), 1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> wSize = {yChannels, xSize[0], convParams.kW, convParams.kH, 1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> ySize = {
        yChannels,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3],
        1};

    unsigned dedy = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        ySize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_bf16);

    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     wSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_bf16);

    unsigned dedw = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,  // initializer
                                        wSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_float);

    unsigned x = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     xSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_bf16);

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,  // initializer
                                        xSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_bf16);

    if (addDedx)
    {
        addNodeToGraph("dedx", {dedy, w}, {dedx}, &convParams, sizeof(convParams));
    }
    if (addDedw)
    {
        addNodeToGraph("dedw", {dedy, x}, {dedw}, &convParams, sizeof(convParams));
    }

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[dedy];
    synTensorDescriptor xDesc    = m_tensorDescs[x];
    synTensorDescriptor dedxDesc = m_tensorDescs[dedx];
    synTensorDescriptor wDesc    = m_tensorDescs[w];
    synTensorDescriptor dedwDesc = m_tensorDescs[dedw];

    char* dedyData = (char*)m_hostBuffers[dedy];
    char* xData    = (char*)m_hostBuffers[x];
    char* dedxData = (char*)m_hostBuffers[dedx];
    char* wData    = (char*)m_hostBuffers[w];
    char* dedwData = (char*)m_hostBuffers[dedw];

    CoordArray wrongIdx;
    float      expectedResult = 0;

    bool ret = !addDedx || checkMmeOp(dedxDesc,
                                      dedxData,
                                      wDesc,
                                      wData,
                                      dedyDesc,
                                      dedyData,
                                      convParams,
                                      REFERENCE_OP_DEDX,
                                      wrongIdx,
                                      m_deviceType,
                                      &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[dedx].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true)
        << "Wrong value for DEDX op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
        << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedx].m_dataType, m_hostBuffers[dedx])
        << " Expected: " << expectedResult;

    ret = !addDedw || checkMmeOp(xDesc,
                                 xData,
                                 dedwDesc,
                                 dedwData,
                                 dedyDesc,
                                 dedyData,
                                 convParams,
                                 REFERENCE_OP_DEDW,
                                 wrongIdx,
                                 m_deviceType,
                                 &expectedResult);

    castNcopy(sizes, m_tensorDescs[dedw].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ((ret), true)
        << "Wrong value for DEDW op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
        << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedw].m_dataType, m_hostBuffers[dedw])
        << " Expected: " << expectedResult;
}

TEST_F_GC(SynTrainingEagerTests, tpc_mme_tpc)
{
    pushGlobalConf("ENABLE_EAGER_SB_REUSE", "1");  // This test focus on multi node with multi MME activation
    // Flags for debug:
    bool addAddNode = true;  // Add "ADD" node before DEDX
    bool addDedx    = true;  // Add DEDX node after ADD
    bool addNeg     = true;  // Add NEG node after DEDX

    //
    // Define tensors
    //

    synConvolutionParams convParams;
    convParams.kH = convParams.kW = 3;
    constexpr uint32_t yChannels  = 64;
    std::array<unsigned, SYN_MAX_TENSOR_DIM> xSize = {64, 7, 7, 64, 1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> wSize = {yChannels, xSize[0], convParams.kW, convParams.kH, 1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> ySize = {
        yChannels,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        xSize[3],
        1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> addIn1 = ySize;
    std::array<unsigned, SYN_MAX_TENSOR_DIM> addIn2 = ySize;

    unsigned add1 = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        addIn1.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    unsigned add2 = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        addIn2.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    unsigned dedy = createPersistTensor(addAddNode ? OUTPUT_TENSOR : INPUT_TENSOR,
                                        addAddNode ? MEM_INIT_ALL_ZERO : MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,  // initializer
                                        ySize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    unsigned w = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     wSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_single);

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ONES,
                                        nullptr,  // initializer
                                        xSize.data(),
                                        DEFAULT_SIZES,
                                        syn_type_single);

    unsigned negOut = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,  // initializer
                                          xSize.data(),
                                          DEFAULT_SIZES,
                                          syn_type_single);

    //
    // Define the graph and launch it
    //

    if (addAddNode)
    {
        addNodeToGraph("add_f32", {add1, add2}, {dedy}, nullptr, 0, "add");
    }
    if (addDedx)
    {
        addNodeToGraph("dedx", {dedy, w}, {dedx}, &convParams, sizeof(convParams));
    }
    if (addNeg)
    {
        addNodeToGraph("neg_f32", {dedx}, {negOut}, nullptr, 0, "neg");
    }

    compileAndRun();

    //
    // Checking results
    //

    if (addAddNode)
    {
        const size_t size = getNumberOfElements(ySize.data(), ySize.size());
        const float* in1  = reinterpret_cast<float*>(m_hostBuffers[add1]);
        const float* in2  = reinterpret_cast<float*>(m_hostBuffers[add2]);
        const float* out  = reinterpret_cast<float*>(m_hostBuffers[dedy]);
        for (size_t i = 0; i < size; ++i)
        {
            ASSERT_FLOAT_EQ(in1[i] + in2[i], out[i]);
        }
    }

    if (addDedx)
    {
        synTensorDescriptor dedyDesc = m_tensorDescs[dedy];
        synTensorDescriptor dedxDesc = m_tensorDescs[dedx];
        synTensorDescriptor wDesc    = m_tensorDescs[w];

        char* dedyData = (char*)m_hostBuffers[dedy];
        char* dedxData = (char*)m_hostBuffers[dedx];
        char* wData    = (char*)m_hostBuffers[w];

        CoordArray wrongIdx;
        float      expectedResult = 0;

        bool ret = checkMmeOp(dedxDesc,
                              dedxData,
                              wDesc,
                              wData,
                              dedyDesc,
                              dedyData,
                              convParams,
                              REFERENCE_OP_DEDX,
                              wrongIdx,
                              m_deviceType,
                              &expectedResult);

        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[dedx].m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true)
            << "Wrong value for DEDX op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
            << getIndexValue(sizes, wrongIdx, m_tensorDescs[dedx].m_dataType, m_hostBuffers[dedx])
            << " Expected: " << expectedResult;
    }

    if (addNeg)
    {
        const size_t size = getNumberOfElements(xSize.data(), xSize.size());
        const float* in   = reinterpret_cast<float*>(m_hostBuffers[dedx]);
        const float* out  = reinterpret_cast<float*>(m_hostBuffers[negOut]);
        for (size_t i = 0; i < size; ++i)
        {
            ASSERT_FLOAT_EQ(-in[i], out[i]);
        }
    }
}

TEST_F_GC(SynTrainingEagerTests, tpc_with_const_tensor)
{
    std::array<unsigned, 4>               ifmDimSizes  = {2, 3, 1, 1};
    std::array<unsigned, 4>               ofmDimSizes  = {1, 3, 1, 1};
    static constexpr std::array<float, 6> ifmBuffer    = {-1.0, -1.0, 1.0, 3.0, 4.0, 5.0};
    static constexpr std::array<float, 3> ofmRefBuffer = {-2.0, 4.0, 9.0};

    ns_Reduction::Params params;
    params.reductionDimension = 0;  // sum and reduce first dimension size to 1

    unsigned xTensorIndex = createConstTensor(MEM_INIT_FROM_INITIALIZER,
                                              ifmBuffer.data(),
                                              ifmDimSizes.data(),
                                              ifmDimSizes.size(),
                                              syn_type_single);

    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                ofmDimSizes.data(),
                                                ofmDimSizes.size(),
                                                syn_type_single);

    addNodeToGraph("reduce_sum_fwd_f32", {xTensorIndex}, {yTensorIndex}, (void*)&params, sizeof(ns_Reduction::Params));
    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes.data(), ofmDimSizes.size()); i++)
    {
        ASSERT_EQ(*pOutputBuffer, ofmRefBuffer[i])
            << "Mismatch at index " << i << " Result:" << *pOutputBuffer << " Ref: " << ofmRefBuffer[i];
        pOutputBuffer++;
    }
}

template<size_t N>
constexpr std::array<size_t, N> calcElementStrides(const std::array<unsigned, N>& dims)
{
    std::array<size_t, N> res {};
    for (size_t i = 0; i < N; ++i)
    {
        res[i] = i != 0 ? res[i - 1] * dims[i - 1] : 1;
    }
    return res;
}

namespace
{
struct MatrixView;
struct Matrix;

// A 2D Matrix for gemm with extra 2 batch dims that owns the data
struct Matrix
{
    static constexpr auto DIM = 4;

    explicit Matrix(const std::array<unsigned, DIM>& dims)
    : m_data(prod(dims)), m_dims(dims), m_strides(calcElementStrides(dims))
    {
    }

    operator MatrixView();

    std::vector<float>        m_data;
    std::array<unsigned, DIM> m_dims;
    std::array<size_t, DIM>   m_strides;
};

// A 2D Matrix for gemm with extra 2 batch dims that DOESN'T own the data
struct MatrixView
{
    static constexpr auto DIM = 4;

    MatrixView(float* data, const std::array<unsigned, DIM>& dims)
    : m_data(data), m_dims(dims), m_strides(calcElementStrides(dims))
    {
    }

    MatrixView(float* data, const std::array<unsigned, DIM>& dims, const std::array<size_t, DIM>& strides)
    : m_data(data), m_dims(dims), m_strides(strides)
    {
    }

    float*                    m_data;
    std::array<unsigned, DIM> m_dims;
    std::array<size_t, DIM>   m_strides;
};

Matrix::operator MatrixView()
{
    return MatrixView(m_data.data(), m_dims, m_strides);
}
}  // namespace

template<size_t N>
constexpr size_t dotProduct(const std::array<unsigned, N>& dims, const std::array<size_t, N>& strides)
{
    size_t res = 0;
    for (size_t i = 0; i < N; ++i)
    {
        res += dims[i] * strides[i];
    }
    return res;
}

static Matrix matrixBatchMul(const MatrixView& in0, const MatrixView& in1)
{
    assert(in0.m_dims[3] == in1.m_dims[3] && "mismatched batch dim 2");
    assert(in0.m_dims[2] == in1.m_dims[2] && "mismatched batch dim 1");
    assert(in0.m_dims[0] == in1.m_dims[1] && "mismatch common_dim");

    Matrix res({in1.m_dims[0], in0.m_dims[1], in0.m_dims[2], in0.m_dims[3]});

    // clang-format off
    for (unsigned b2 = 0; b2 < res.m_dims[3]; ++b2)
    for (unsigned b1 = 0; b1 < res.m_dims[2]; ++b1)
    for (unsigned h  = 0; h  < res.m_dims[1]; ++h)
    for (unsigned w  = 0; w  < res.m_dims[0]; ++w)
    {
        float dotProd = 0;
        for (size_t i = 0; i < in0.m_dims[0]; ++i)
        {
            const size_t in0_index = dotProduct({i, h, b1, b2}, in0.m_strides);
            const size_t in1_index = dotProduct({w, i, b1, b2}, in1.m_strides);
            dotProd += in0.m_data[in0_index] * in1.m_data[in1_index];
        }

        const size_t out_index = dotProduct({w, h, b1, b2}, res.m_strides);
        res.m_data[out_index] = dotProd;
    }
    // clang-format on

    return res;
}

static Matrix matrixSumBroadCastBatch(const MatrixView& in0, const MatrixView& in1)
{
    assert(in0.m_dims[0] == in1.m_dims[0]);
    assert(in0.m_dims[1] == in1.m_dims[1]);
    assert(in0.m_dims[2] == in1.m_dims[2] || in0.m_dims[2] == 1 || in1.m_dims[2] == 1);
    assert(in0.m_dims[3] == in1.m_dims[3] || in0.m_dims[3] == 1 || in1.m_dims[3] == 1);

    Matrix res(
        {in0.m_dims[0], in0.m_dims[1], std::max(in0.m_dims[2], in1.m_dims[2]), std::max(in0.m_dims[3], in1.m_dims[3])});

    const std::array<size_t, 4> in0_strides = {res.m_dims[0] == in0.m_dims[0] ? in0.m_strides[0] : 0,
                                               res.m_dims[1] == in0.m_dims[1] ? in0.m_strides[1] : 0,
                                               res.m_dims[2] == in0.m_dims[2] ? in0.m_strides[2] : 0,
                                               res.m_dims[3] == in0.m_dims[3] ? in0.m_strides[3] : 0};

    const std::array<size_t, 4> in1_strides = {res.m_dims[0] == in1.m_dims[0] ? in1.m_strides[0] : 0,
                                               res.m_dims[1] == in1.m_dims[1] ? in1.m_strides[1] : 0,
                                               res.m_dims[2] == in1.m_dims[2] ? in1.m_strides[2] : 0,
                                               res.m_dims[3] == in1.m_dims[3] ? in1.m_strides[3] : 0};

    // clang-format off
    std::array<unsigned, 4> idx;
    for (idx[3] = 0; idx[3] < res.m_dims[3]; ++idx[3])
    for (idx[2] = 0; idx[2] < res.m_dims[2]; ++idx[2])
    for (idx[1] = 0; idx[1] < res.m_dims[1]; ++idx[1])
    for (idx[0] = 0; idx[0] < res.m_dims[0]; ++idx[0])
    {
        const size_t in0_index = dotProduct(idx, in0_strides);
        const size_t in1_index = dotProduct(idx, in1_strides);
        const size_t res_index = dotProduct(idx, res.m_strides);
        res.m_data[res_index] = in0.m_data[in0_index] + in1.m_data[in1_index];
    }
    // clang-format on

    return res;
}

// TODO [SW-152705]: drop FALLBACK from the name
// TODO [SW-83012] - for now only gaudi2 supports this node type
TEST_F_GC(SynTrainingEagerTests, FALLBACK_masked_batch_gemm, {synDeviceGaudi2})
{
    std::array<std::array<unsigned, 4>, 4> inputs_dims {
        {{15, 11, 9, 7}, {13, 15, 9, 7}, {17, 11, 1, 7}, {13, 17, 1, 7}}};
    std::array<unsigned, 4> output_dims {{13, 11, 9, 7}};

    std::array<int, 4>                inputs;
    std::array<std::vector<float>, 4> inputs_data;
    {
        std::mt19937                       gen {};
        std::uniform_int_distribution<int> dist {0, 1};
        for (size_t i = 0; i < 4; ++i)
        {
            // init input data with 0.f or 1.f to minimize chance of accuracy issues
            inputs_data[i].resize(prod(inputs_dims[i]));
            std::generate(inputs_data[i].begin(), inputs_data[i].end(), [&] { return static_cast<float>(dist(gen)); });

            inputs[i] = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_FROM_INITIALIZER,
                                            inputs_data[i].data(),
                                            inputs_dims[i].data(),
                                            inputs_dims[i].size(),
                                            syn_type_float);
        }
    }
    auto output = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      output_dims.data(),
                                      output_dims.size(),
                                      syn_type_float);

    unsigned char params[] = {0, 0};  // no transposes
    addNodeToGraph("masked_batch_gemm",
                   {inputs.begin(), inputs.end()},
                   {output},
                   &params,
                   sizeof(params),
                   "masked_batch_gemm");
    compileAndRun();

    const Matrix reference = [&] {
        auto A     = MatrixView(inputs_data[0].data(), inputs_dims[0]);
        auto B     = MatrixView(inputs_data[1].data(), inputs_dims[1]);
        auto maskA = MatrixView(inputs_data[2].data(), inputs_dims[2]);
        auto maskB = MatrixView(inputs_data[3].data(), inputs_dims[3]);

        return matrixSumBroadCastBatch(matrixBatchMul(A, B), matrixBatchMul(maskA, maskB));
    }();

    const float* res = reinterpret_cast<float*>(m_hostBuffers[output]);
    const float* ref = reference.m_data.data();

    const size_t elementCount = prod(output_dims);
    for (size_t i = 0; i < elementCount; ++i)
    {
        ASSERT_EQ(res[i], ref[i]) << "found diff at element" << i;
    }
}

// CGUID will take care of huge tensors
// This test does not know how to handle the huge tensor. Leaving it disabled.
TEST_F_GC(SynTrainingEagerTests, DISABLED_tpc_node_with_huge_tensor_ASIC_CI)
{
    ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
    ScopedConfigurationChange sramSliceConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    constexpr bool          enableHugeTensor = true;  // Sanity check flag for the test
    std::array<unsigned, 3> defaultSizes({enableHugeTensor ? (1 << 10) : 2, enableHugeTensor ? (1 << 20) : 100, 2});

    unsigned in  = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      defaultSizes.data(),
                                      defaultSizes.size(),
                                      syn_type_float);
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       defaultSizes.data(),
                                       defaultSizes.size(),
                                       syn_type_float);

    addNodeToGraph("neg_f32", {in}, {out}, nullptr, 0, "tpc_node_with_huge_tensor");

    compileAndRun();
    const auto*  inBuf  = reinterpret_cast<const float*>(m_hostBuffers[in]);
    const auto*  outBuf = reinterpret_cast<const float*>(m_hostBuffers[out]);
    const size_t defaultFlatSize(prod(defaultSizes));
    for (size_t i = 0; i < defaultFlatSize; ++i)
    {
        ASSERT_FLOAT_EQ(-inBuf[i], outBuf[i]);
    }
}

// This test suppose to generate 21 nodes as a result of split on common dim
// The test got killed on device. Leaving it for compilation only.
TEST_F_GC(SynTrainingEagerTests, mme_node_with_huge_tensor_split_on_common_dim_ASIC_CI)
{
    ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
    ScopedConfigurationChange sramSliceConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    constexpr bool          enableHugeTensor = true;  // Sanity check flag for the test
    constexpr unsigned      bigDim           = enableHugeTensor ? (1 << 28) : 64;
    std::array<unsigned, 3> in1Sizes({bigDim, 8, 2});
    std::array<unsigned, 3> in2Sizes({4, bigDim, 2});
    std::array<unsigned, 3> outSizes({4, 8, 2});

    unsigned in1 = createPersistTensor(INPUT_TENSOR,
                                       enableHugeTensor ? MEM_INIT_COMPILATION_ONLY : MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       in1Sizes.data(),
                                       in1Sizes.size(),
                                       syn_type_float);
    unsigned in2 = createPersistTensor(INPUT_TENSOR,
                                       enableHugeTensor ? MEM_INIT_COMPILATION_ONLY : MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       in2Sizes.data(),
                                       in2Sizes.size(),
                                       syn_type_float);
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       enableHugeTensor ? MEM_INIT_COMPILATION_ONLY : MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizes.data(),
                                       outSizes.size(),
                                       syn_type_float);

    synGEMMParams bgemmParams(false, false);
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {in1, in2}, {out}, &bgemmParams, sizeof(bgemmParams));

    if constexpr (enableHugeTensor)
    {
        compileTopology("", 0);
    }
    else
    {
        compileAndRun();

        synTensorDescriptor in1Desc = m_tensorDescs[in1];
        synTensorDescriptor in2Desc = m_tensorDescs[in2];
        synTensorDescriptor outDesc = m_tensorDescs[out];

        char* in1Data = (char*)m_hostBuffers[in1];
        char* in2Data = (char*)m_hostBuffers[in2];
        char* outData = (char*)m_hostBuffers[out];

        CoordArray wrongIdx;
        float      expectedResult = 0;
        bool       ret            = checkBatchGemmOp(in1Desc,
                                    in1Data,
                                    in2Desc,
                                    in2Data,
                                    outDesc,
                                    outData,
                                    REFERENCE_OP_AB,
                                    wrongIdx,
                                    &expectedResult,
                                    m_deviceType);

        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, outDesc.m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true) << "Wrong value for BGEMM op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                             << " Got value: " << getIndexValue(sizes, wrongIdx, outDesc.m_dataType, outData)
                             << " Expected: " << expectedResult;
    }
}

// This test suppose to generate 7 nodes as a result of split on spatial dim
TEST_F_GC(SynTrainingEagerTests, mme_node_with_huge_tensor_split_on_spatial_ASIC_CI)
{
    ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
    ScopedConfigurationChange sramSliceConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    constexpr bool          enableHugeTensor = true;  // Sanity check flag for the test
    constexpr unsigned      bigDim           = enableHugeTensor ? (1 << 26) : 8;
    std::array<unsigned, 3> in1Sizes({16, bigDim, 2});
    std::array<unsigned, 3> in2Sizes({4, 16, 2});
    std::array<unsigned, 3> outSizes({4, bigDim, 2});

    unsigned in1 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       in1Sizes.data(),
                                       in1Sizes.size(),
                                       syn_type_float);
    unsigned in2 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       in2Sizes.data(),
                                       in2Sizes.size(),
                                       syn_type_float);
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizes.data(),
                                       outSizes.size(),
                                       syn_type_float);

    synGEMMParams bgemmParams(false, false);
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {in1, in2}, {out}, &bgemmParams, sizeof(bgemmParams));

    compileAndRun();

    synTensorDescriptor in1Desc = m_tensorDescs[in1];
    synTensorDescriptor in2Desc = m_tensorDescs[in2];
    synTensorDescriptor outDesc = m_tensorDescs[out];

    char* in1Data = (char*)m_hostBuffers[in1];
    char* in2Data = (char*)m_hostBuffers[in2];
    char* outData = (char*)m_hostBuffers[out];

    CoordArray wrongIdx;
    float      expectedResult = 0;
    bool       ret            = checkBatchGemmOp(in1Desc,
                                in1Data,
                                in2Desc,
                                in2Data,
                                outDesc,
                                outData,
                                REFERENCE_OP_AB,
                                wrongIdx,
                                &expectedResult,
                                m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, outDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_EQ(ret, true) << "Wrong value for BGEMM op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                         << " Got value: " << getIndexValue(sizes, wrongIdx, outDesc.m_dataType, outData)
                         << " Expected: " << expectedResult;
}

// Transpose a huge tensor
TEST_F_GC(SynTrainingEagerTests, dma_node_with_huge_tensor_ASIC_CI)
{
    ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
    ScopedConfigurationChange sramSliceConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    using D = TransposePermutationDim;
    TestSizeVec               inputSizes({9216, 9216, 15, 2});
    TransposePermutationArray permutation({(D)1, (D)2, (D)3, (D)0});
    const bool                runAndValidate = true;

    HB_ASSERT(permutation.size() == inputSizes.size(), "size mismatch");
    TestSizeVec outputSize(permutation.size());
    applyPermutation(inputSizes.data(), permutation, outputSize.data());

    unsigned in  = createPersistTensor(INPUT_TENSOR,
                                      runAndValidate ? MEM_INIT_RANDOM_WITH_NEGATIVE : MEM_INIT_COMPILATION_ONLY,
                                      nullptr,
                                      inputSizes.data(),
                                      inputSizes.size());
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       runAndValidate ? MEM_INIT_ALL_ZERO : MEM_INIT_COMPILATION_ONLY,
                                       nullptr,
                                       outputSize.data(),
                                       outputSize.size());

    synTransposeParamsNDims params = permutationToParams(permutation);

    addNodeToGraph("transpose", {in}, {out}, (void*)&params, sizeof(params));

    compileTopology();
    if (runAndValidate)
    {
        HB_ASSERT(permutation.size() == 4, "compare results is implemented only for 4 dim tensor");
        runTopology();

        float* iPtr = castHostBuffer<float>(in);
        float* oPtr = castHostBuffer<float>(out);

        uint64_t iStrides[4] = {1, 0, 0, 0}, oStrides[4] = {1, 0, 0, 0};
        for (int i = 1; i < 4; ++i)
        {
            iStrides[i] = iStrides[i - 1] * inputSizes[i - 1];
            oStrides[i] = oStrides[i - 1] * outputSize[i - 1];
        }

        const auto& p = permutation;
        for (unsigned i = 0; i < outputSize[3]; ++i)
        {
            for (unsigned j = 0; j < outputSize[2]; ++j)
            {
                for (unsigned k = 0; k < outputSize[1]; ++k)
                {
                    for (unsigned l = 0; l < outputSize[0]; ++l)
                    {
                        ASSERT_FLOAT_EQ(iPtr[i * iStrides[p[3]] + j * iStrides[p[2]] + k * iStrides[p[1]] + l * iStrides[p[0]]],
                                        oPtr[i * oStrides[3] + j * oStrides[2] + k * oStrides[1] + l * oStrides[0]]);
                    }
                }
            }
        }
    }
}

// Biased gemm to cause multiple reductions
TEST_F_GC(SynTrainingEagerTests, biased_gemm_with_huge_tensor_ASIC_CI)
{
    ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
    ScopedConfigurationChange sramSliceConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    TestSizeVec opASizes({9216, 9216});
    TestSizeVec opBSizes({9216 * 30, 9216});
    TestSizeVec outSizes({9216 * 30, 9216});
    const bool  runAndValidate = true;

    unsigned     opA = createPersistTensor(INPUT_TENSOR,
                                       runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                       nullptr,
                                       opASizes.data(),
                                       opASizes.size());
    unsigned     opB = createPersistTensor(INPUT_TENSOR,
                                       runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                       nullptr,
                                       opBSizes.data(),
                                       opBSizes.size());
    unsigned int biasSizes[1];
    biasSizes[0]  = outSizes.data()[0];
    unsigned bias = createPersistTensor(INPUT_TENSOR,
                                        runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                        nullptr,
                                        biasSizes,
                                        1);
    unsigned out  = createPersistTensor(OUTPUT_TENSOR,
                                       runAndValidate ? MEM_INIT_ALL_ZERO : MEM_INIT_COMPILATION_ONLY,
                                       nullptr,
                                       outSizes.data(),
                                       outSizes.size());

    synGEMMParams params = {false, false};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {opA, opB, bias}, {out}, (void*)&params, sizeof(params));

    compileTopology();

    if (runAndValidate)
    {
        runTopology();
        float* outPtr = castHostBuffer<float>(out);
        float  resValue =
            opASizes[0] +
            1;  // Since all inputs are 1's the values of the output is the common dim (+1 because of the bias)
        TSize elements = std::accumulate(outSizes.begin(), outSizes.end(), TSize(1), std::multiplies<TSize>());
        for (TSize i = 0; i < elements; ++i)
        {
            ASSERT_FLOAT_EQ(resValue, outPtr[i])
                << "mismatch at index " << i << "\nexpected: " << resValue << "\nresult: " << outPtr[i];
        }
    }
}

TEST_F_GC(SynTrainingEagerTests, strided_insert_view_input_alias)
{
    unsigned                outputNumElements = 1200;
    std::array<unsigned, 4> insertInputSizes  = {3, 2, 5, 4};

    unsigned viewTensorIdx = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 &outputNumElements,
                                                 1,
                                                 syn_type_single);

    SynTrainingTestInfra::GraphData& graphData = getGraph(0);
    unsigned inputViewSectionIndex             = graphData.tensorCreationParams.begin()->second.concreteSectionIndex;

    unsigned insertTensorIdx = createPersistTensor(INPUT_TENSOR,
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   insertInputSizes.data(),
                                                   insertInputSizes.size(),
                                                   syn_type_single);

    // We need to create at least 3 nodes to enter the flow in toploigcal re-ordering to avoid
    // optimized flow for single\two nodes.
    // We achieve it by adding non related negation nodes.
    for (int i = 0; i < 4; i++)
    {
        unsigned negOutTensorIdx = createPersistTensor(OUTPUT_TENSOR,
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       insertInputSizes.data(),
                                                       insertInputSizes.size(),
                                                       syn_type_single);
        addNodeToGraph("neg_fwd_f32", {insertTensorIdx}, {negOutTensorIdx});
    }

    unsigned outTensorIdx = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                &outputNumElements,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                &inputViewSectionIndex);

    synStridedOpParams params = {};
    params.strides[0]         = 1;
    for (int i = 0; i < insertInputSizes.size() - 1; i++)
    {
        params.strides[i + 1] = params.strides[i] * insertInputSizes[i];
    }

    addNodeToGraph("strided_insert", {viewTensorIdx, insertTensorIdx}, {outTensorIdx}, &params, sizeof(params));

    compileAndRun();

    auto     viewTensorBuffer        = static_cast<float*>(m_hostBuffers[viewTensorIdx]);
    auto     insertTensorBuffer      = static_cast<float*>(m_hostBuffers[insertTensorIdx]);
    auto     outTensorBuffer         = static_cast<float*>(m_hostBuffers[outTensorIdx]);
    unsigned insertTensorNumElements = getNumberOfElements(insertInputSizes.data(), insertInputSizes.size());
    for (uint64_t i = 0; i < outputNumElements; i++)
    {
        float expected = (i >= insertTensorNumElements) ? viewTensorBuffer[i] : insertTensorBuffer[i];
        ASSERT_EQ(outTensorBuffer[i], expected)
            << "mismatch at index " << i << " expected " << expected << " actual " << outTensorBuffer[i];
    }
}

void SynTrainingEagerTests::runNdimMemcpy(synDataType dataType, std::string_view guid)
{
    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    int graphIndex = createGraph();

    std::array<unsigned, 6> sizes = {2, 3, 4, 5, 6, 7};

    auto memcpyIn = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        sizes.data(),
                                        sizes.size(),
                                        dataType,
                                        nullptr,
                                        nullptr,
                                        graphIndex);

    auto memcpyOut = createPersistTensor(OUTPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sizes.data(),
                                         sizes.size(),
                                         dataType,
                                         nullptr,
                                         nullptr,
                                         graphIndex);

    addNodeToGraph(guid.data(), {memcpyIn}, {memcpyOut}, nullptr, 0, nullptr, graphIndex);

    compileTopology("topology", graphIndex);
    runTopology(graphIndex);

    ASSERT_EQ(std::memcmp(m_hostBuffers[memcpyIn],
                          m_hostBuffers[memcpyOut],
                          dataTypeSizeInBytes(dataType) * getNumberOfElements(sizes.data(), sizes.size())),
              0);
}

TEST_F_GC(SynTrainingEagerTests, basic_ndim_memcpy_supported_types_partial_coverage)
{
    for (synDataType dataType :
         {syn_type_int8, syn_type_uint16, syn_type_int32, syn_type_bf16, syn_type_fp16, syn_type_float})
    {
        runNdimMemcpy(dataType, "memcpy");
        runNdimMemcpy(dataType, fmt::format("memcpy_{}", getDtypeSuffixFromSynDataType(dataType)));
        runNdimMemcpy(dataType, fmt::format("memcpy_nd_{}", getDtypeSuffixFromSynDataType(dataType)));
    }
}

TEST_F_GC(SynTrainingEagerTests, basic_ndim_memcpy_64bit)
{
    for (synDataType dataType : {syn_type_int64, syn_type_uint64})
    {
        runNdimMemcpy(dataType, "memcpy");
        runNdimMemcpy(dataType, fmt::format("memcpy_{}", getDtypeSuffixFromSynDataType(dataType)));
        runNdimMemcpy(dataType, fmt::format("memcpy_nd_{}", getDtypeSuffixFromSynDataType(dataType)));
    }
    runNdimMemcpy(syn_type_int64, "memcpy_i32");
    runNdimMemcpy(syn_type_uint64, "memcpy_u32");
    runNdimMemcpy(syn_type_int64, "memcpy_nd_i32");
    runNdimMemcpy(syn_type_uint64, "memcpy_nd_u32");
}

// Following kernel is special since it has no outputs and the side effect
// is reflected in LFSR vector register.
TEST_F_GC(SynTrainingEagerTests, random_seed_u32)
{
    std::array<unsigned, 1> sizes = {1};

    unsigned input = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sizes.data(),
                                         sizes.size(),
                                         syn_type_uint32);

    addNodeToGraph("random_seed_u32", {input}, {});
    compileAndRun();
}

void SynTrainingEagerTests::broadcastOnFcdTest(synDataType dataType, std::string_view guid)
{
    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    int graphIndex = createGraph();

    TestSizeVec inSizes  = {1};
    TestSizeVec outSizes = {7, 3, 4, 5, 6};

    auto in  = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  inSizes.data(),
                                  inSizes.size(),
                                  dataType,
                                  nullptr,
                                  nullptr,
                                  graphIndex);
    auto out = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   outSizes.data(),
                                   outSizes.size(),
                                   dataType,
                                   nullptr,
                                   nullptr,
                                   graphIndex);

    addNodeToGraph(guid.data(), {in}, {out}, nullptr, 0, nullptr, graphIndex);
    compileTopology("topology", graphIndex);
    runTopology(graphIndex);

    const size_t totalSizes  = eager_mode::prod(outSizes);
    const size_t elementSize = dataTypeSizeInBytes(dataType);
    auto         pInBuf      = reinterpret_cast<const char*>(m_hostBuffers[in]);
    auto         pOutBuf     = reinterpret_cast<const char*>(m_hostBuffers[out]);
    for (int i = 0; i < totalSizes; i++)
    {
        ASSERT_TRUE(!std::memcmp(pInBuf, pOutBuf + i * elementSize, elementSize));
    }
}

TEST_F_GC(SynTrainingEagerTests, basic_fcd_broadcast_64bit)
{
    for (synDataType dataType : {syn_type_int64, syn_type_uint64})
    {
        broadcastOnFcdTest(dataType, "broadcast");
    }
}

void SynTrainingEagerTests::broadcastNonFcdTest(synDataType dataType, std::string_view guid)
{
    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    int graphIndex = createGraph();

    TestSizeVec inSizes  = {2, 1, 1, 1, 1, 1};
    TestSizeVec outSizes = {2, 3, 4, 5, 6, 7};

    auto in  = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                  nullptr,
                                  inSizes.data(),
                                  inSizes.size(),
                                  dataType,
                                  nullptr,
                                  nullptr,
                                  graphIndex);
    auto out = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   outSizes.data(),
                                   outSizes.size(),
                                   dataType,
                                   nullptr,
                                   nullptr,
                                   graphIndex);

    addNodeToGraph(guid.data(), {in}, {out}, nullptr, 0, nullptr, graphIndex);
    compileTopology("topology", graphIndex);
    runTopology(graphIndex);

    const size_t fcdSizeInBytes = static_cast<size_t>(dataTypeSizeInBytes(dataType)) * inSizes[0];
    const size_t iters_count    = eager_mode::prod(outSizes) / fcdSizeInBytes;
    const auto*  pInBuf         = reinterpret_cast<std::byte*>(m_hostBuffers[in]);
    const auto*  pOutBuf        = reinterpret_cast<std::byte*>(m_hostBuffers[out]);
    for (int i = 0; i < iters_count; i++)
    {
        ASSERT_TRUE(!std::memcmp(pInBuf, pOutBuf + i * fcdSizeInBytes, fcdSizeInBytes));
    }
}

TEST_F_GC(SynTrainingEagerTests, basic_non_fcd_broadcast_64bit)
{
    for (synDataType dataType : {syn_type_int64, syn_type_uint64})
    {
        broadcastNonFcdTest(dataType, "broadcast");
        broadcastNonFcdTest(dataType, fmt::format("broadcast_nd_fwd_{}", getDtypeSuffixFromSynDataType(dataType)));
    }
    broadcastNonFcdTest(syn_type_int64, "broadcast_nd_fwd_i32");
    broadcastNonFcdTest(syn_type_uint64, "broadcast_nd_fwd_u32");
}