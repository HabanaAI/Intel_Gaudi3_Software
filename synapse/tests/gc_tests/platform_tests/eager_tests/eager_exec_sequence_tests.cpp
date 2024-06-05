#include "eager_tests_defs.h"
#include "eager/eager_interface.h"

#include "habana_graph.h"
#include "spdlog/fmt/bundled/format.h"
#include "utils.inl"
#include "gtest/gtest.h"
#include "scoped_configuration_change.h"
#include "transpose_utils.h"

using namespace eager_mode;

///////////////////////////////////////////////////////////////////////////////////////////////////
// SynTrainingEagerExecSequenceTests
///////////////////////////////////////////////////////////////////////////////////////////////////

struct SynTrainingEagerExecSequenceTests : public SynTrainingEagerTests
{
    std::array<unsigned, 3> defaultSizes;
    const size_t            defaultFlatSize;

    SynTrainingEagerExecSequenceTests()
    : SynTrainingEagerTests(), defaultSizes({5, 11, 17}), defaultFlatSize(prod(defaultSizes))
    {
    }

    unsigned newInputTensor()
    {
        return createPersistTensor(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   defaultSizes.data(),
                                   defaultSizes.size(),
                                   syn_type_float);
    }

    unsigned newOutputTensor()
    {
        return createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   defaultSizes.data(),
                                   defaultSizes.size(),
                                   syn_type_float);
    }
};

TEST_F_GC(SynTrainingEagerExecSequenceTests, 1_nodes)
{
    unsigned in1  = newInputTensor();
    unsigned in2  = newInputTensor();
    unsigned out1 = newOutputTensor();

    addNodeToGraph("add_fwd_f32", {in1, in2}, {out1}, nullptr, 0, "node-0");

    compileAndRun();

    const float* in1Buf  = reinterpret_cast<float*>(m_hostBuffers[in1]);
    const float* in2Buf  = reinterpret_cast<float*>(m_hostBuffers[in2]);
    const float* out1Buf = reinterpret_cast<float*>(m_hostBuffers[out1]);
    for (size_t i = 0; i < defaultFlatSize; ++i)
    {
        ASSERT_FLOAT_EQ(in1Buf[i] + in2Buf[i], out1Buf[i]);
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, 2_nodes)
{
    unsigned in1  = newInputTensor();
    unsigned out1 = newOutputTensor();
    unsigned out2 = newOutputTensor();

    addNodeToGraph("relu_fwd_f32", {out1}, {out2}, nullptr, 0, "node-1");
    addNodeToGraph("neg_f32", {in1}, {out1}, nullptr, 0, "node-0");

    compileAndRun();

    const auto* in1Buf  = reinterpret_cast<float*>(m_hostBuffers[in1]);
    const auto* out2Buf = reinterpret_cast<float*>(m_hostBuffers[out2]);
    for (size_t i = 0; i < defaultFlatSize; ++i)
    {
        const float ref = (in1Buf[i] < 0) ? std::abs(in1Buf[i]) : 0;
        ASSERT_FLOAT_EQ(ref, out2Buf[i]);
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, 3_nodes)
{
    unsigned in1  = newInputTensor();
    unsigned out1 = newOutputTensor();
    unsigned out2 = newOutputTensor();
    unsigned out3 = newOutputTensor();

    addNodeToGraph("neg_f32", {out2}, {out3}, nullptr, 0, "node-2");
    addNodeToGraph("relu_fwd_f32", {out1}, {out2}, nullptr, 0, "node-1");
    addNodeToGraph("neg_f32", {in1}, {out1}, nullptr, 0, "node-0");

    compileAndRun();

    const auto* in1Buf  = reinterpret_cast<float*>(m_hostBuffers[in1]);
    const auto* out3Buf = reinterpret_cast<float*>(m_hostBuffers[out3]);
    for (size_t i = 0; i < defaultFlatSize; ++i)
    {
        const float ref = (in1Buf[i] < 0) ? in1Buf[i] : 0;
        ASSERT_FLOAT_EQ(ref, out3Buf[i]);
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, 10_nodes)
{
    std::array<unsigned, 6>  inT;
    std::array<unsigned, 10> outT;
    for (auto& in : inT)
    {
        in = newInputTensor();
    }
    for (auto& out : outT)
    {
        out = newOutputTensor();
    }

    addNodeToGraph("add_fwd_f32", {outT[8], outT[6]}, {outT[9]}, nullptr, 0, "node-9");
    addNodeToGraph("add_fwd_f32", {outT[7], inT[5]}, {outT[8]}, nullptr, 0, "node-8");
    addNodeToGraph("add_fwd_f32", {outT[4], outT[5]}, {outT[7]}, nullptr, 0, "node-7");
    addNodeToGraph("add_fwd_f32", {outT[4], inT[4]}, {outT[6]}, nullptr, 0, "node-6");
    addNodeToGraph("add_fwd_f32", {outT[2], outT[3]}, {outT[5]}, nullptr, 0, "node-5");
    addNodeToGraph("add_fwd_f32", {outT[0], outT[1]}, {outT[4]}, nullptr, 0, "node-4");
    addNodeToGraph("neg_f32", {inT[3]}, {outT[3]}, nullptr, 0, "node-3");
    addNodeToGraph("neg_f32", {inT[2]}, {outT[2]}, nullptr, 0, "node-2");
    addNodeToGraph("neg_f32", {inT[1]}, {outT[1]}, nullptr, 0, "node-1");
    addNodeToGraph("neg_f32", {inT[0]}, {outT[0]}, nullptr, 0, "node-0");

    compileAndRun();

    std::array<float*, inT.size()>  inBufs;
    std::array<float*, outT.size()> outBufs;
    for (size_t i = 0; i < inBufs.size(); ++i)
    {
        inBufs[i] = reinterpret_cast<float*>(m_hostBuffers[inT[i]]);
    }
    for (size_t i = 0; i < outBufs.size(); ++i)
    {
        outBufs[i] = reinterpret_cast<float*>(m_hostBuffers[outT[i]]);
    }

    std::array<float, outT.size()> outRef;
    for (size_t i = 0; i < defaultFlatSize; ++i)
    {
        outRef[0] = -inBufs[0][i];
        outRef[1] = -inBufs[1][i];
        outRef[2] = -inBufs[2][i];
        outRef[3] = -inBufs[3][i];
        outRef[4] = outRef[0] + outRef[1];
        outRef[5] = outRef[2] + outRef[3];
        outRef[6] = outRef[4] + inBufs[4][i];
        outRef[7] = outRef[4] + outRef[5];
        outRef[8] = outRef[7] + inBufs[5][i];
        outRef[9] = outRef[8] + outRef[6];

        for (size_t j = 0; j < outBufs.size(); ++j)
        {
            ASSERT_FLOAT_EQ(outRef[j], outBufs[j][i]);
        }
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, many_nodes_parallel)
{
    constexpr unsigned addNodesNr = 100;
    pushGlobalConf("MAX_NODES_IN_EAGER_GRAPH", std::to_string(addNodesNr + 10));
    std::array<unsigned, addNodesNr> outT;
    for (auto& out : outT)
    {
        out = newOutputTensor();
    }
    const unsigned inT = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_ALL_ONES,
                                             nullptr,
                                             defaultSizes.data(),
                                             defaultSizes.size(),
                                             syn_type_float);
    for (unsigned i = 0; i < addNodesNr; ++i)
    {
        addNodeToGraph("add_fwd_f32", {inT, inT}, {outT[i]}, nullptr, 0, fmt::format("add-node-{}", i).c_str());
    }
    compileAndRun();

    float* inBuf = reinterpret_cast<float*>(m_hostBuffers[inT]);
    for (unsigned i = 0; i < addNodesNr; ++i)
    {
        float* outResBuf = reinterpret_cast<float*>(m_hostBuffers[outT[i]]);
        for (unsigned j = 0; j < defaultFlatSize; ++j)
        {
            ASSERT_FLOAT_EQ(outResBuf[j], inBuf[j] * 2) << " i=" << i << " j=" << j;
        }
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, many_parallel_nodes_inputs_from_single_const)
{
    constexpr unsigned addNodesNr = 100;
    pushGlobalConf("MAX_NODES_IN_EAGER_GRAPH", std::to_string(addNodesNr + 10));
    std::array<unsigned, addNodesNr> outT;
    for (auto& out : outT)
    {
        out = newOutputTensor();
    }
    const unsigned constTensor = newOutputTensor();
    for (unsigned i = 0; i < addNodesNr; ++i)
    {
        addNodeToGraph("add_fwd_f32",
                       {constTensor, constTensor},
                       {outT[i]},
                       nullptr,
                       0,
                       fmt::format("add-node-{}", i).c_str());
    }
    constexpr float           theConst    = 2.0;
    ns_ConstantKernel::Params constParams = {theConst};
    addNodeToGraph("constant_f32", {}, {constTensor}, &constParams, sizeof(constParams), "constant");
    compileAndRun();

    for (unsigned i = 0; i < addNodesNr; ++i)
    {
        float* outResBuf = reinterpret_cast<float*>(m_hostBuffers[outT[i]]);
        for (unsigned j = 0; j < defaultFlatSize; ++j)
        {
            ASSERT_FLOAT_EQ(outResBuf[j], theConst * 2) << " i=" << i << " j=" << j;
        }
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, many_parallel_nodes_inputs_from_multiple_consts)
{
    constexpr unsigned addNodesNr = 50;
    pushGlobalConf("MAX_NODES_IN_EAGER_GRAPH", std::to_string(2 * addNodesNr + 10));
    std::array<unsigned, addNodesNr> outT;
    for (auto& out : outT)
    {
        out = newOutputTensor();
    }
    std::array<unsigned, addNodesNr> constTensors;
    for (auto& constTensor : constTensors)
    {
        constTensor = newOutputTensor();
    }
    const unsigned oneTensor = newOutputTensor();

    for (unsigned i = 0; i < addNodesNr; ++i)
    {
        addNodeToGraph("add_fwd_f32",
                       {oneTensor, constTensors[i]},
                       {outT[i]},
                       nullptr,
                       0,
                       fmt::format("add-node-{}", i).c_str());
        // 'Constant' after 'Add' order to validate execution schedule
        const float               theConst    = 1.0 + i;
        ns_ConstantKernel::Params constParams = {theConst};
        addNodeToGraph("constant_f32",
                       {},
                       {constTensors[i]},
                       &constParams,
                       sizeof(constParams),
                       fmt::format("const-node-{}", i).c_str());
    }
    constexpr float           oneConst    = 1.0;
    ns_ConstantKernel::Params constParams = {oneConst};
    addNodeToGraph("constant_f32", {}, {oneTensor}, &constParams, sizeof(constParams), "constant-one");
    compileAndRun();

    for (unsigned i = 0; i < addNodesNr; ++i)
    {
        float*      outResBuf   = reinterpret_cast<float*>(m_hostBuffers[outT[i]]);
        const float expectedRes = 2.0 + i;
        for (unsigned j = 0; j < defaultFlatSize; ++j)
        {
            ASSERT_FLOAT_EQ(outResBuf[j], expectedRes) << " i=" << i << " j=" << j;
        }
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, many_nodes_sequential)
{
    constexpr unsigned addNodesNr = 100;
    pushGlobalConf("MAX_NODES_IN_EAGER_GRAPH", std::to_string(addNodesNr + 10));
    std::array<unsigned, addNodesNr> outT;
    for (auto& out : outT)
    {
        out = newOutputTensor();
    }
    const unsigned constTensor = newOutputTensor();
    const unsigned inT         = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_ALL_ONES,
                                             nullptr,
                                             defaultSizes.data(),
                                             defaultSizes.size(),
                                             syn_type_float);
    for (unsigned i = addNodesNr - 1; i >= 1; --i)  // Intentially reversed to validate execution schedule
    {
        addNodeToGraph("add_fwd_f32",
                       {outT[i - 1], constTensor},
                       {outT[i]},
                       nullptr,
                       0,
                       fmt::format("add-node-{}", i).c_str());
    }
    addNodeToGraph("add_fwd_f32", {inT, constTensor}, {outT[0]}, nullptr, 0, "add-node-0");
    constexpr float           theConst    = 1.0;
    ns_ConstantKernel::Params constParams = {theConst};
    addNodeToGraph("constant_f32", {}, {constTensor}, &constParams, sizeof(constParams), "constant-one");

    compileAndRun();

    float* inBuf = reinterpret_cast<float*>(m_hostBuffers[inT]);
    for (unsigned i = 0; i < addNodesNr; ++i)
    {
        const float s         = (i + 1) * theConst;
        float*      outResBuf = reinterpret_cast<float*>(m_hostBuffers[outT[i]]);
        for (unsigned j = 0; j < defaultFlatSize; ++j)
        {
            ASSERT_FLOAT_EQ(outResBuf[j], inBuf[j] + s) << " i=" << i << " j=" << j;
        }
    }
}

TEST_F_GC(SynTrainingEagerExecSequenceTests, input_duplication)
{
    std::array<unsigned, 3> inT;
    std::array<unsigned, 8> outT;
    for (auto& in : inT)
    {
        in = newInputTensor();
    }
    for (auto& out : outT)
    {
        out = newOutputTensor();
    }

    addNodeToGraph("add_fwd_f32", {outT[2], outT[6]}, {outT[7]}, nullptr, 0, "node-7");
    addNodeToGraph("add_fwd_f32", {outT[4], outT[5]}, {outT[6]}, nullptr, 0, "node-6");
    addNodeToGraph("neg_f32", {outT[3]}, {outT[5]}, nullptr, 0, "node-5");
    addNodeToGraph("add_fwd_f32", {outT[3], outT[3]}, {outT[4]}, nullptr, 0, "node-4");  // Input duplication
    addNodeToGraph("add_fwd_f32", {outT[1], outT[0]}, {outT[3]}, nullptr, 0, "node-3");
    addNodeToGraph("add_fwd_f32", {inT[2], inT[2]}, {outT[2]}, nullptr, 0, "node-2");  // Graph input duplication
    addNodeToGraph("neg_f32", {inT[1]}, {outT[1]}, nullptr, 0, "node-1");
    addNodeToGraph("neg_f32", {inT[0]}, {outT[0]}, nullptr, 0, "node-0");

    compileAndRun();

    std::array<float*, inT.size()>  inBufs;
    std::array<float*, outT.size()> outBufs;
    for (size_t i = 0; i < inBufs.size(); ++i)
    {
        inBufs[i] = reinterpret_cast<float*>(m_hostBuffers[inT[i]]);
    }
    for (size_t i = 0; i < outBufs.size(); ++i)
    {
        outBufs[i] = reinterpret_cast<float*>(m_hostBuffers[outT[i]]);
    }

    std::array<float, outT.size()> outRef;
    for (size_t i = 0; i < defaultFlatSize; ++i)
    {
        outRef[0] = -inBufs[0][i];
        outRef[1] = -inBufs[1][i];
        outRef[2] = inBufs[2][i] + inBufs[2][i];
        outRef[3] = outRef[0] + outRef[1];
        outRef[4] = outRef[3] + outRef[3];
        outRef[5] = -outRef[3];
        outRef[6] = outRef[4] + outRef[5];
        outRef[7] = outRef[6] + outRef[2];

        for (size_t j = 0; j < outBufs.size(); ++j)
        {
            ASSERT_FLOAT_EQ(outRef[j], outBufs[j][i]);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// SynTrainingEagerParallelExecTests
///////////////////////////////////////////////////////////////////////////////////////////////////

struct SynTrainingEagerParallelExecTests : public SynTrainingEagerTests
{
};

TEST_F_GC(SynTrainingEagerParallelExecTests, all_engines_parallel_exec)
{
    // Use FP8 to make memcpy be executed on DMA
    const auto dataType = syn_type_fp8_152;

    // Create BGEMM
    std::array<unsigned, 3> in1Sizes({16, 8, 2048});
    std::array<unsigned, 3> in2Sizes({4, 16, 2048});
    std::array<unsigned, 3> outSizes({4, 8, 2048});
    auto                    inAllOnes =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, in1Sizes.data(), in1Sizes.size(), dataType);
    auto inRnd = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     in2Sizes.data(),
                                     in2Sizes.size(),
                                     dataType);
    auto outBGemm =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), dataType);
    synGEMMParams bgemmParams(false, false);
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName,
                   {inAllOnes, inRnd},
                   {outBGemm},
                   &bgemmParams,
                   sizeof(bgemmParams),
                   "BGEMM");

    // Create MEMCPY
    auto inMemcpy =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, outSizes.data(), outSizes.size(), dataType);
    auto outMemcpy =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), dataType);
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inMemcpy}, {outMemcpy}, nullptr, 0, "MEMCPY");

    // Create NEG
    auto inNeg =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, outSizes.data(), outSizes.size(), dataType);
    auto outNeg =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), dataType);
    addNodeToGraph("neg_f32", {inNeg}, {outNeg}, nullptr, 0, "NEG");

    // Create CONCAT
    std::array<unsigned, 3> concatOutSizes(outSizes);
    concatOutSizes[2] *= 3;
    auto     outConcat    = createPersistTensor(OUTPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         concatOutSizes.data(),
                                         concatOutSizes.size(),
                                         dataType);
    unsigned concatcatDim = 2;
    addNodeToGraph("concat", {outBGemm, outMemcpy, outNeg}, {outConcat}, &concatcatDim, sizeof(concatcatDim), "CONCAT");

    ////////////////
    compileAndRun();
    ////////////////

    const unsigned totalOutputSize = prod(outSizes);
    auto*          pOutBuf         = reinterpret_cast<char*>(m_hostBuffers[outConcat]);

    // Check BGEMM
    {
        synTensorDescriptor in1Desc = m_tensorDescs[inAllOnes];
        synTensorDescriptor in2Desc = m_tensorDescs[inRnd];
        synTensorDescriptor outDesc = m_tensorDescs[outBGemm];

        char* in1Data = (char*)m_hostBuffers[inAllOnes];
        char* in2Data = (char*)m_hostBuffers[inRnd];
        char* outData = (char*)pOutBuf;

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

    // Check MEMCPY
    {
        pOutBuf += totalOutputSize;
        ASSERT_TRUE(std::all_of(pOutBuf, pOutBuf + totalOutputSize, [](char element) { return element == 1; }));
    }

    // Check NEG
    {
        pOutBuf += totalOutputSize;
        ASSERT_TRUE(std::all_of(pOutBuf, pOutBuf + totalOutputSize, [](char element) { return element == -127; }));
    }
}

// The following test exams the emitting signals from one engine (TPC) but different nodes to different ones (MME and
// DMA). It's expected from TPC and DMA to manage the sync correctly and listen to the correct TPC node. This scenario
// is so prevalent so I keep it disabled, just for development debug.
TEST_F_GC(SynTrainingEagerParallelExecTests, DISABLED_two_tpc_nodes_used_by_mme_and_dma)
{
    const auto dataType = syn_type_float;

    std::array<unsigned, 2> in1Sizes({32, 4});
    std::array<unsigned, 2> in2Sizes({2, 32});
    std::array<unsigned, 2> outSizes({2, 4});

    // Create 1st NEG
    auto inNeg1 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, in1Sizes.data(), in1Sizes.size(), dataType);
    auto outNeg1 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, in1Sizes.data(), in1Sizes.size(), dataType);
    addNodeToGraph("neg_f32", {inNeg1}, {outNeg1}, nullptr, 0, "NEG1");

    // Create BGEMM
    auto inRnd = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     in2Sizes.data(),
                                     in2Sizes.size(),
                                     dataType);
    auto outBGemm =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), dataType);
    synGEMMParams bgemmParams(false, false);
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName,
                   {outNeg1, inRnd},
                   {outBGemm},
                   &bgemmParams,
                   sizeof(bgemmParams),
                   "BGEMM");

    // Create 2nd NEG
    auto outNeg2 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), dataType);
    addNodeToGraph("neg_f32", {outBGemm}, {outNeg2}, nullptr, 0, "NEG2");

    ScopedConfigurationChange scc("ENABLE_INTERNAL_NODES", "true");

    // TRANSPOSE
    std::array<unsigned, 2> transposeOutSizes({outSizes[1], outSizes[0]});
    auto                    outTranspose = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            transposeOutSizes.data(),
                                            transposeOutSizes.size(),
                                            dataType);
    addNodeToGraph(NodeFactory::transposeDmaNodeTypeName, {outNeg2}, {outTranspose});
    compileAndRun();
}
