#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
#include "habana_global_conf.h"
#include "types.h"

class SynTrainingEagerGraphDuplicateTest : public SynDualExecutionGaudiTestInfra
{
public:
    using SizeArray2D = std::array<unsigned, 2>;
    void addTest(SizeArray2D initialDimSizes, SizeArray2D newDimSizes);
};

TEST_F_GC(SynTrainingEagerGraphDuplicateTest, eager_norm_moments_clustering)
{
    const unsigned c    = 128;
    const unsigned w    = 8;
    const unsigned h    = 4;
    const unsigned b    = 2;
    const unsigned wOut = 1;
    const unsigned hOut = 1;
    // create_tensor's layout
    TestSizes inSizes  = {c, w, h, b, 1};
    TestSizes outSizes = {c, wOut, hOut, b, 1};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         inSizes.data(),
                         inSizes.size(),
                         syn_type_single);
    auto meanTensorIndexPair     = createPersistTensors(OUTPUT_TENSOR,
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    outSizes.data(),
                                                    outSizes.size(),
                                                    syn_type_single);
    auto varianceTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        outSizes.data(),
                                                        outSizes.size(),
                                                        syn_type_single);

    ns_NormMomentsKernel::Params params;
    params.NormAxisBmp = 6;

    addNodesToGraphs("norm_moments_fwd_f32", (void*)&params, sizeof(params));

    auto newEagerGraphIndex = duplicateGraph(DEFAULT_EAGER_MODE_INDEX);

    compileAndRun();

    SynGaudiTestInfra::compileAndRun(newEagerGraphIndex);

    std::array<std::pair<TensorIndexPair, std::string>, 2> resultPairs = {
        make_pair(meanTensorIndexPair, std::string("mean tensor")),
        make_pair(varianceTensorIndexPair, std::string("variance tensor"))};

    for (const auto& resultPair : resultPairs)
    {
        auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[resultPair.first.graph]);
        auto dupTensorIndex         = getDuplicateTensorIndex(newEagerGraphIndex, resultPair.first.eager);
        auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[dupTensorIndex]);
        for (uint64_t i = 0; i < getNumberOfElements(outSizes.data(), outSizes.size()); i++)
        {
            ASSERT_LE(abs(pOutputBufferGraphMode[i] - pOutputBufferEagerMode[i]), 0.00001)
                << resultPair.second << " Graph mode mismatch at index " << i
                << " Graph mode:" << pOutputBufferGraphMode[i] << " Eager mode: " << pOutputBufferEagerMode[i];
        }
    }
}

void SynTrainingEagerGraphDuplicateTest::addTest(SizeArray2D initialDimSizes, SizeArray2D newDimSizes)
{
    auto operand1TensorIndexPair = createPersistTensors(INPUT_TENSOR,
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        initialDimSizes.data(),
                                                        initialDimSizes.size(),
                                                        syn_type_single);
    auto operand2TensorIndexPair = createPersistTensors(INPUT_TENSOR,
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        initialDimSizes.data(),
                                                        initialDimSizes.size(),
                                                        syn_type_single);
    auto outputTensorIndexPair   = createPersistTensors(OUTPUT_TENSOR,
                                                      MEM_INIT_ALL_ZERO,
                                                      nullptr,
                                                      initialDimSizes.data(),
                                                      initialDimSizes.size(),
                                                      syn_type_single);
    addNodesToGraphs("add_fwd_f32");

    auto newEagerGraphIndex = duplicateGraph(DEFAULT_EAGER_MODE_INDEX);

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[outputTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[outputTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(initialDimSizes.data(), initialDimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }

    for (const auto tensorIndex : getGraphTensorIndices(newEagerGraphIndex))
    {
        auto newDimSizesVecExt = getVector(newDimSizes.data(), newDimSizes.size());
        changeTensorGeometry(newEagerGraphIndex,
                             tensorIndex,
                             getVectorRawPtrOrNull(newDimSizesVecExt),
                             newDimSizes.size(),
                             MEM_INIT_RANDOM_WITH_NEGATIVE);
    }
    SynGaudiTestInfra::compileAndRun(newEagerGraphIndex);

    auto getTensorHostBuffer = [this](unsigned newEagerGraphIndex, unsigned origTensorIndex) {
        auto dupOutputTensorIndex = getDuplicateTensorIndex(newEagerGraphIndex, origTensorIndex);
        return static_cast<float*>(m_hostBuffers[dupOutputTensorIndex]);
    };

    auto pDupOutputBufferEagerMode   = getTensorHostBuffer(newEagerGraphIndex, outputTensorIndexPair.eager);
    auto pDupOperand1BufferEagerMode = getTensorHostBuffer(newEagerGraphIndex, operand1TensorIndexPair.eager);
    auto pDupOperand2BufferEagerMode = getTensorHostBuffer(newEagerGraphIndex, operand2TensorIndexPair.eager);

    for (uint64_t i = 0; i < getNumberOfElements(newDimSizes.data(), newDimSizes.size()); i++)
    {
        ASSERT_EQ(pDupOperand1BufferEagerMode[i] + pDupOperand2BufferEagerMode[i], pDupOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i
            << " expected: " << pDupOperand1BufferEagerMode[i] + pDupOperand2BufferEagerMode[i]
            << " actual: " << pDupOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingEagerGraphDuplicateTest, eager_add_with_graph_duplicate_shrink)
{
    addTest({5, 10}, {3, 3});
}

TEST_F_GC(SynTrainingEagerGraphDuplicateTest, eager_add_with_graph_duplicate_expand)
{
    addTest({4, 7}, {10, 10});
}

TEST_F_GC(SynTrainingEagerGraphDuplicateTest, eager_add_with_graph_duplicate_unchanged_size)
{
    addTest({6, 9}, {6, 9});
}