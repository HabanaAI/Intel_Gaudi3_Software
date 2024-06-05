#include "gaudi_dual_execution_test_infra.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "eager_tests_defs.h"
#include "global_conf_test_setter.h"
#include "synapse_common_types.h"
#include "type_utils.h"

class SynTrainingDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    SynTrainingDualExecutionTest() { ReleaseDevice(); }
    void broadcastTest(TestSizeVec& inSizes, TestSizeVec& outSizes, synDataType dataType = synDataType::syn_type_float);
    template<int N>
    void sliceInsertTest(const std::array<unsigned, N>& realSizes,
                         const std::array<unsigned, N>& insertSizes,
                         const std::array<unsigned, N>& starts,
                         const std::array<unsigned, N>& ends,
                         const std::array<unsigned, N>& steps);
};

void SynTrainingDualExecutionTest::broadcastTest(TestSizeVec& inSizes, TestSizeVec& outSizes, synDataType dataType)
{
    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    GraphIndexPair graphIndexPair = createNewGraphPair();

    auto inPair  = createPersistTensors(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       inSizes.data(),
                                       inSizes.size(),
                                       dataType,
                                       false,
                                       nullptr,
                                       nullptr,
                                       graphIndexPair);
    auto outPair = createPersistTensors(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        outSizes.data(),
                                        outSizes.size(),
                                        dataType,
                                        false,
                                        nullptr,
                                        nullptr,
                                        graphIndexPair);

    TensorIndicesPair broadcastInIndices  = {{inPair.graph}, {inPair.eager}};
    TensorIndicesPair broadcastOutIndices = {{outPair.graph}, {outPair.eager}};

    addNodesToGraphs("broadcast", broadcastInIndices, broadcastOutIndices, nullptr, 0, nullptr, graphIndexPair);
    compileTopology("topology", graphIndexPair);
    runTopology(graphIndexPair);

    const size_t totalSizes             = eager_mode::prod(outSizes);
    auto         pGraphBuf              = reinterpret_cast<const char*>(m_hostBuffers[outPair.graph]);
    auto         pEagerBuf              = reinterpret_cast<const char*>(m_hostBuffers[outPair.eager]);
    ASSERT_EQ(std::memcmp(pGraphBuf, pEagerBuf, dataTypeSizeInBytes(dataType) * totalSizes), 0);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_all_types_coverage)
{
    TestSizeVec inSizes  = {1};
    TestSizeVec outSizes = {2, 3, 4, 5, 6};
    // supported data types
    for (synDataType dataType : {syn_type_float,
                                 syn_type_fp16,
                                 syn_type_bf16,
                                 syn_type_fp8_152,
                                 syn_type_int8,
                                 syn_type_uint8,
                                 syn_type_int16,
                                 syn_type_uint16,
                                 syn_type_int32,
                                 syn_type_uint32})
    {
        broadcastTest(inSizes, outSizes, dataType);
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_1_to_5)
{
    TestSizeVec inSizes  = {1};
    TestSizeVec outSizes = {2, 3, 4, 5, 6};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_1_to_4)
{
    TestSizeVec inSizes  = {1};
    TestSizeVec outSizes = {2, 4, 5, 6};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_5_to_5_1)
{
    TestSizeVec inSizes  = {10, 2, 1, 5, 6};
    TestSizeVec outSizes = {10, 2, 4, 5, 6};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_5_to_5_2)
{
    TestSizeVec inSizes  = {1, 2, 1, 5, 6};
    TestSizeVec outSizes = {10, 2, 4, 5, 6};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_5_to_5_3)
{
    TestSizeVec inSizes  = {1, 2, 1, 5, 1};
    TestSizeVec outSizes = {10, 2, 4, 5, 6};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_5_to_6)
{
    TestSizeVec inSizes  = {10, 2, 4, 5, 6};
    TestSizeVec outSizes = {10, 2, 4, 5, 6, 7};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_6_to_6)
{
    TestSizeVec inSizes  = {10, 2, 4, 5, 6, 1};
    TestSizeVec outSizes = {10, 2, 4, 5, 6, 7};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_6_to_8)
{
    TestSizeVec inSizes  = {10, 2, 4, 5, 6, 1};
    TestSizeVec outSizes = {10, 2, 4, 5, 6, 7, 9, 3};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, broadcast_1_to_8)
{
    TestSizeVec inSizes  = {1};
    TestSizeVec outSizes = {2, 3, 5, 4, 6, 7, 8, 9};
    broadcastTest(inSizes, outSizes);
}

TEST_F_GC(SynTrainingDualExecutionTest, auxiliary_tensors)
{
    std::array<unsigned, SYN_MAX_TENSOR_DIM> sizeInA  = {40, 30, 20, 2};
    const unsigned                           dimsInA  = 4;
    std::array<unsigned, SYN_MAX_TENSOR_DIM> sizeInB  = {8, 2};
    const unsigned                           dimsInB  = 2;
    std::array<unsigned, SYN_MAX_TENSOR_DIM> sizeOutA = {8, 30, 20, 2};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> sizeOutB = {320, 30, 20, 2};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> sizeOutC = {16, 30, 20, 2};
    const unsigned                           dimsOut  = 4;
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         sizeInA.data(),
                         dimsInA,
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         sizeInA.data(),
                         dimsInA,
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         sizeInB.data(),
                         dimsInB,
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         sizeInB.data(),
                         dimsInB,
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         sizeInB.data(),
                         dimsInB,
                         syn_type_single);
    const auto output0 =
        createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizeOutA.data(), dimsOut, syn_type_single);
    const auto output1 =
        createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizeOutB.data(), dimsOut, syn_type_single);
    const auto output2 =
        createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizeOutC.data(), dimsOut, syn_type_single);

    ns_SpatialCorrelationKernel::Params nodeParams {};
    static const char*                  guid = "spatial_correlation_fwd_f32";
    addNodesToGraphs(guid, (void*)&nodeParams, sizeof(nodeParams), guid);

    compileAndRun();

    const TensorIndexPair outputPairs[] = {output0, output1, output2};
    const uint64_t        sizes[]       = {getNumberOfElements(sizeOutA.data(), dimsOut),
                              getNumberOfElements(sizeOutB.data(), dimsOut),
                              getNumberOfElements(sizeOutC.data(), dimsOut)};
    for (size_t outIdx = 0; outIdx < 3; ++outIdx)
    {
        const TensorIndexPair& output   = outputPairs[outIdx];
        auto                   outGraph = static_cast<float*>(m_hostBuffers[output.graph]);
        auto                   outEager = static_cast<float*>(m_hostBuffers[output.eager]);
        for (uint64_t i = 0; i < sizes[outIdx]; ++i)
        {
            ASSERT_EQ(outGraph[i], outEager[i]) << "Mismatch at output[" << outIdx << "]. Data mismatch at index " << i
                                                << " Graph mode:" << outGraph[i] << " Eager mode: " << outEager[i];
        }
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, nDims_strided_view)
{
    std::array<unsigned, 5> sizes = {2, 12, 24, 64, 2};
    createPersistTensors(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), sizes.size());
    auto outPair = createPersistTensors(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, sizes.data(), sizes.size());

    char params[] = {
        0,  0,    0,    0,   0,    0,   0,  0,   24,  0,   0,   0,   0,    0,   0,    0,    2,    0,   0,   0,   0,
        0,  0,    0,    1,   0,    0,   0,  0,   0,   0,   0,   64,  2,    0,   0,    0,    0,    0,   0,   0,   -112,
        0,  0,    0,    0,   0,    0,   0,  -63, 19,  9,   0,   0,   0,    0,   -128, 107,  -107, 106, 59,  127, 0,
        0,  0,    0,    0,   0,    0,   0,  0,   0,   -32, 90,  -77, 7,    0,   0,    0,    0,    0,   0,   0,   0,
        0,  0,    0,    0,   16,   89,  29, 9,   0,   0,   0,   0,   0,    0,   0,    0,    0,    0,   0,   0,   0,
        53, -90,  -115, -25, -10,  -8,  67, 0,   0,   0,   0,   0,   0,    0,   0,    -40,  90,   -77, 7,   0,   0,
        0,  0,    -128, 113, -107, 106, 59, 127, 0,   0,   -96, 110, -107, 106, 59,   127,  0,    0,   112, 88,  29,
        9,  0,    0,    0,   0,    0,   0,  0,   0,   0,   0,   0,   0,    48,  110,  -107, 106,  59,  127, 0,   0,
        98, -107, -11,  -75, 59,   127, 0,  0,   -64, 121, -12, 0,   0,    0,   0,    0,    32,   0,   0,   0,   0,
        0,  0,    0,    0,   125,  102, 23, 0,   0,   0,   0,   69,  63,   -23, -24,  59,   127,  0,   0};
    addNodesToGraphs("strided_view", params, sizeof(params), "strided_view");
    compileAndRun();

    const auto*  graphBuf   = castHostInBuffer<float>(outPair.graph);
    const auto*  eagerBuf   = castHostOutBuffer<float>(outPair.eager);
    const size_t totalSizes = eager_mode::prod(sizes);
    for (size_t i = 0; i < totalSizes; ++i)
    {
        ASSERT_EQ(graphBuf[i], eagerBuf[i]) << "Mismatch for at index " << i;
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, strided_slice_grad)
{
    std::array<unsigned, 4> sizes  = {2, 4, 1, 1};
    std::array<unsigned, 4> starts = {0, 0, 0, 0};
    std::array<unsigned, 4> steps  = {1, 1, 1, 1};

    TensorIndices inputs;
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         sizes.data(),
                         sizes.size(),
                         syn_type_single);

    createShapeTensors(INPUT_TENSOR, sizes.data(), sizes.size());
    createShapeTensors(INPUT_TENSOR, steps.data(), steps.size());
    createShapeTensors(INPUT_TENSOR, starts.data(), starts.size());

    auto outPair =
        createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), syn_type_single);

    addNodesToGraphs("strided_slice_grad");
    compileAndRun();

    const auto*  graphBuf   = castHostInBuffer<float>(outPair.graph);
    const auto*  eagerBuf   = castHostOutBuffer<float>(outPair.eager);
    const size_t totalSizes = eager_mode::prod(sizes);
    for (size_t i = 0; i < totalSizes; ++i)
    {
        ASSERT_EQ(graphBuf[i], eagerBuf[i]) << "Mismatch for at index " << i;
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, tpc_multi_add)
{
    std::array<unsigned, 2>      sizes  = {4, 8};
    constexpr int                numOps = 200;
    GlobalConfTestSetter         gConvVar("MAX_NODES_IN_EAGER_GRAPH", std::to_string(numOps));
    std::vector<TensorIndexPair> inputs;
    inputs.reserve(numOps + 1);
    for (int i = 0; i < numOps + 1; i++)
    {
        inputs.push_back(createPersistTensors(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              sizes.data(),
                                              sizes.size(),
                                              syn_type_float));
    }
    std::vector<TensorIndexPair> outputs;
    outputs.reserve(numOps);
    for (int i = 0; i < numOps; i++)
    {
        outputs.push_back(createPersistTensors(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               sizes.size(),
                                               syn_type_float));

        TensorIndexPair   secondInput = (i == 0) ? inputs[1] : outputs[i - 1];
        TensorIndicesPair inIndices   = {{inputs[i].graph, secondInput.graph}, {inputs[i].eager, secondInput.eager}};
        TensorIndicesPair outIndices  = {{outputs[i].graph}, {outputs[i].eager}};
        addNodesToGraphs("add_fwd_f32", inIndices, outIndices);
    }

    compileAndRun();

    for (int i = 0; i < numOps; i++)
    {
        const auto*  graphBuf   = castHostInBuffer<float>(outputs[i].graph);
        const auto*  eagerBuf   = castHostOutBuffer<float>(outputs[i].eager);
        const size_t totalSizes = eager_mode::prod(sizes);
        for (size_t i = 0; i < totalSizes; ++i)
        {
            ASSERT_EQ(graphBuf[i], eagerBuf[i]) << "Mismatch for at index " << i;
        }
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, mme_multi_gemm)
{
    constexpr unsigned           dimSize = 7;
    std::array<unsigned, 2>      sizes   = {dimSize, dimSize};
    constexpr int                numOps  = 200;
    GlobalConfTestSetter         gConvVar("MAX_NODES_IN_EAGER_GRAPH", std::to_string(numOps));
    synGEMMParams                gemmParams = {};
    std::vector<TensorIndexPair> inputs;
    inputs.reserve(numOps + 1);
    for (int i = 0; i < numOps + 1; i++)
    {
        inputs.push_back(
            createPersistTensors(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes.data(), sizes.size(), syn_type_float));
    }
    std::vector<TensorIndexPair> outputs;
    outputs.reserve(numOps);
    for (int i = 0; i < numOps; i++)
    {
        outputs.push_back(createPersistTensors(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               sizes.size(),
                                               syn_type_float));

        TensorIndexPair   secondInput = (i == 0) ? inputs[1] : outputs[i - 1];
        TensorIndicesPair inIndices   = {{inputs[i].graph, secondInput.graph}, {inputs[i].eager, secondInput.eager}};
        TensorIndicesPair outIndices  = {{outputs[i].graph}, {outputs[i].eager}};
        addNodesToGraphs(NodeFactory::gemmNodeTypeName, inIndices, outIndices, &gemmParams, sizeof(gemmParams));
    }

    compileAndRun();

    for (int i = 0; i < numOps; i++)
    {
        const auto*  graphBuf   = castHostInBuffer<float>(outputs[i].graph);
        const auto*  eagerBuf   = castHostOutBuffer<float>(outputs[i].eager);
        const size_t totalSizes = eager_mode::prod(sizes);
        for (size_t i = 0; i < totalSizes; ++i)
        {
            ASSERT_EQ(graphBuf[i], eagerBuf[i]) << "Mismatch for at index " << i;
        }
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, multi_transpose)
{
    std::array<unsigned, 2> sizes1 = {4, 8};
    std::array<unsigned, 2> sizes2 = {8, 4};
    synTransposeParams      params;
    params.tensorDim                    = 2;
    params.permutation[0]               = TransposePermutationDim::TPD_Width;
    params.permutation[1]               = TransposePermutationDim::TPD_Channel;
    constexpr int                numOps = 200;
    GlobalConfTestSetter         gConvVar("MAX_NODES_IN_EAGER_GRAPH", std::to_string(numOps));
    std::vector<TensorIndexPair> inputs;
    inputs.reserve(numOps);
    for (int i = 0; i < numOps; i++)
    {
        std::array<unsigned, 2>& sizes = (i % 2) ? sizes1 : sizes2;
        inputs.push_back(createPersistTensors(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              sizes.data(),
                                              sizes.size(),
                                              syn_type_float));
    }
    std::vector<TensorIndexPair> outputs;
    outputs.reserve(numOps);
    for (int i = 0; i < numOps; i++)
    {
        std::array<unsigned, 2>& sizes = (i % 2) ? sizes2 : sizes1;
        outputs.push_back(createPersistTensors(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               sizes.size(),
                                               syn_type_float));

        TensorIndicesPair inIndices  = {{inputs[i].graph}, {inputs[i].eager}};
        TensorIndicesPair outIndices = {{outputs[i].graph}, {outputs[i].eager}};
        addNodesToGraphs(NodeFactory::transposeNodeTypeName, inIndices, outIndices, &params, sizeof(params));
    }

    compileAndRun();

    for (int i = 0; i < numOps; i++)
    {
        const auto*  graphBuf   = castHostInBuffer<float>(outputs[i].graph);
        const auto*  eagerBuf   = castHostOutBuffer<float>(outputs[i].eager);
        const size_t totalSizes = eager_mode::prod(sizes1);
        for (size_t i = 0; i < totalSizes; ++i)
        {
            ASSERT_EQ(graphBuf[i], eagerBuf[i]) << "Mismatch for at index " << i;
        }
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, multi_nodes_all_engines)
{
    std::array<unsigned, 2> sizes = {4, 4};
    synTransposeParams      transposeParams;
    transposeParams.tensorDim             = 2;
    transposeParams.permutation[0]        = TransposePermutationDim::TPD_Width;
    transposeParams.permutation[1]        = TransposePermutationDim::TPD_Channel;
    constexpr int                numIters = 70;
    constexpr int                numOps   = 3 * numIters;
    GlobalConfTestSetter         gConvVar("MAX_NODES_IN_EAGER_GRAPH", std::to_string(numOps));
    std::vector<TensorIndexPair> outputs;
    outputs.reserve(numOps);
    for (int i = 0; i < numIters; i++)
    {
        TensorIndexPair addIn1;
        if (i == 0)
        {
            addIn1 = createPersistTensors(INPUT_TENSOR,
                                          MEM_INIT_ALL_ONES,
                                          nullptr,
                                          sizes.data(),
                                          sizes.size(),
                                          syn_type_float);
        }
        else
        {
            addIn1 = outputs.back();
        }

        auto addIn2 =
            createPersistTensors(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes.data(), sizes.size(), syn_type_float);

        outputs.push_back(createPersistTensors(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               sizes.size(),
                                               syn_type_float));

        TensorIndicesPair inIndices  = {{addIn1.graph, addIn2.graph}, {addIn1.eager, addIn2.eager}};
        TensorIndicesPair outIndices = {{outputs.back().graph}, {outputs.back().eager}};
        addNodesToGraphs("add_fwd_f32", inIndices, outIndices);

        auto negIn = outputs.back();

        outputs.push_back(createPersistTensors(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               sizes.size(),
                                               syn_type_float));

        inIndices  = {{negIn.graph}, {negIn.eager}};
        outIndices = {{outputs.back().graph}, {outputs.back().eager}};
        addNodesToGraphs("neg_f32", inIndices, outIndices);

        auto transposeIn = outputs.back();

        outputs.push_back(createPersistTensors(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               sizes.size(),
                                               syn_type_float));

        inIndices  = {{transposeIn.graph}, {transposeIn.eager}};
        outIndices = {{outputs.back().graph}, {outputs.back().eager}};
        addNodesToGraphs(NodeFactory::transposeNodeTypeName,
                         inIndices,
                         outIndices,
                         &transposeParams,
                         sizeof(transposeParams));
    }

    compileAndRun();

    for (int i = 0; i < numOps; i++)
    {
        const auto*  graphBuf   = castHostInBuffer<float>(outputs[i].graph);
        const auto*  eagerBuf   = castHostOutBuffer<float>(outputs[i].eager);
        const size_t totalSizes = eager_mode::prod(sizes);
        for (size_t i = 0; i < totalSizes; ++i)
        {
            ASSERT_EQ(graphBuf[i], eagerBuf[i]) << "Mismatch for at index " << i;
        }
    }
}

template<int N>
void SynTrainingDualExecutionTest::sliceInsertTest(const std::array<unsigned, N>& realSizes,
                                                   const std::array<unsigned, N>& insertSizes,
                                                   const std::array<unsigned, N>& starts,
                                                   const std::array<unsigned, N>& ends,
                                                   const std::array<unsigned, N>& steps)
{
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         const_cast<unsigned*>(realSizes.data()),
                         realSizes.size(),
                         syn_type_single);

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         const_cast<unsigned*>(insertSizes.data()),
                         insertSizes.size(),
                         syn_type_single);

    auto outPair = createPersistTensors(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        const_cast<unsigned*>(realSizes.data()),
                                        realSizes.size(),
                                        syn_type_single);

    synSliceParamsNDims sliceParams = {};
    for (unsigned i = 0; i < N; i++)
    {
        sliceParams.axes[i]   = i;
        sliceParams.starts[i] = starts[i];
        sliceParams.ends[i]   = ends[i];
        sliceParams.steps[i]  = steps[i];
    }

    addNodesToGraphs("slice_insert", &sliceParams, sizeof(sliceParams));
    compileAndRun();

    const auto*  graphBuf   = castHostInBuffer<float>(outPair.graph);
    const auto*  eagerBuf   = castHostOutBuffer<float>(outPair.eager);
    const size_t totalSizes = eager_mode::prod(realSizes);
    for (size_t i = 0; i < totalSizes; ++i)
    {
        ASSERT_EQ(graphBuf[i], eagerBuf[i]) << "Mismatch for at index " << i;
    }
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert1)
{
    SynTrainingDualExecutionTest::sliceInsertTest<3>({8, 3, 2}, {4, 1, 1}, {0, 0, 0}, {8, 3, 2}, {2, 3, 2});
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert2)
{
    SynTrainingDualExecutionTest::sliceInsertTest<5>({14, 12, 10, 16, 16},
                                                     {6, 6, 4, 8, 5},
                                                     {2, 3, 4, 5, 2},
                                                     {8, 9, 8, 13, 11},
                                                     {1, 1, 1, 1, 2});
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert3)
{
    SynTrainingDualExecutionTest::sliceInsertTest<5>({14, 12, 10, 32, 32},
                                                     {6, 6, 4, 16, 20},
                                                     {6, 4, 2, 0, 11},
                                                     {12, 10, 9, 32, 31},
                                                     {1, 1, 2, 2, 1});
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert4)
{
    SynTrainingDualExecutionTest::sliceInsertTest<4>({14, 12, 10, 32},
                                                     {6, 6, 4, 16},
                                                     {6, 4, 2},
                                                     {12, 10, 9, 32},
                                                     {1, 1, 2, 2});
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert5)
{
    SynTrainingDualExecutionTest::sliceInsertTest<4>({14, 12, 10, 16},
                                                     {6, 6, 4, 8},
                                                     {2, 3, 4, 5},
                                                     {8, 9, 8, 13},
                                                     {1, 1, 1, 1});
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert6)
{
    SynTrainingDualExecutionTest::sliceInsertTest<2>({4, 4}, {2, 2}, {2, 1}, {4, 4}, {1, 2});
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert7)
{
    SynTrainingDualExecutionTest::sliceInsertTest<3>({2, 4, 2}, {2, 1, 2}, {0, 1}, {2, 2, 2}, {1, 1, 1});
}

TEST_F_GC(SynTrainingDualExecutionTest, slice_insert8)
{
    SynTrainingDualExecutionTest::sliceInsertTest<2>({4, 4}, {1, 2}, {1, 1}, {2, 3}, {1, 1});
}
