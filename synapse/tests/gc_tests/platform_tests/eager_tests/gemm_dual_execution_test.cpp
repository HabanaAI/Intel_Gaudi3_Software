#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
class SynTrainingGemmDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    SynTrainingGemmDualExecutionTest() { ReleaseDevice(); }
};

TEST_F_GC(SynTrainingGemmDualExecutionTest, basic_gemm)
{
    static constexpr unsigned NUM_DIMS = 3;

    std::array<unsigned, NUM_DIMS> in1DimSizes = {90, 45, 2};
    std::array<unsigned, NUM_DIMS> in2DimSizes = {35, 90, 2};
    std::array<unsigned, NUM_DIMS> outDimSizes = {35, 45, 2};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in1DimSizes.data(),
                         in1DimSizes.size(),
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in2DimSizes.data(),
                         in2DimSizes.size(),
                         syn_type_single);
    auto yTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outDimSizes.data(),
                                                 outDimSizes.size(),
                                                 syn_type_single);

    addNodesToGraphs("batch_gemm");

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outDimSizes.data(), outDimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingGemmDualExecutionTest, basic_gemm2)
{
    std::array<unsigned, 3> in1DimSizes = {9, 8, 79};
    std::array<unsigned, 4> in2DimSizes = {8, 9, 79, 30};
    std::array<unsigned, 4> outDimSizes = {8, 8, 79, 30};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in1DimSizes.data(),
                         in1DimSizes.size(),
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in2DimSizes.data(),
                         in2DimSizes.size(),
                         syn_type_single);
    auto yTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outDimSizes.data(),
                                                 outDimSizes.size(),
                                                 syn_type_single);

    addNodesToGraphs("batch_gemm");

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outDimSizes.data(), outDimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}