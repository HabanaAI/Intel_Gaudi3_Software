#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
#include "data_type_utils.h"
#include "test_types.hpp"
#include <numeric>

using namespace gc;

class SynTrainingTensorPermutationDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    void addTest(const Permutation& in1Permutation,
                 const Permutation& in2Permutation = Permutation(),
                 const Permutation& outPermutation = Permutation());

    void addSameInputTest(const Permutation& inPermutation, const Permutation& outPermutation = Permutation());
};

void SynTrainingTensorPermutationDualExecutionTest::addTest(const Permutation& in1Permutation,
                                                         const Permutation& in2Permutation,
                                                         const Permutation& outPermutation)
{
    std::array<unsigned, 4> dimSizes = {8, 16, 20, 25};

    auto in1 = createPersistTensors(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dimSizes.data(),
                                    dimSizes.size(),
                                    syn_type_single);
    auto in2 = createPersistTensors(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    dimSizes.data(),
                                    dimSizes.size(),
                                    syn_type_single);
    auto out = createPersistTensors(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    dimSizes.data(),
                                    dimSizes.size(),
                                    syn_type_single);

    setTensorPermutation(in1, in1Permutation);
    setTensorPermutation(in2, in2Permutation);
    setTensorPermutation(out, outPermutation);

    addNodesToGraphs("add_fwd_f32");

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[out.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[out.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(dimSizes.data(), dimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_input_permuted)
{
    addTest(Permutation({2, 3, 0, 1}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_different_input_permutations)
{
    addTest(Permutation({2, 3, 0, 1}), Permutation({1, 3, 0, 2}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_different_permutations_all)
{
    addTest(Permutation({2, 3, 0, 1}), Permutation({1, 3, 0, 2}), Permutation({0, 3, 2, 1}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_different_permutations_input_output)
{
    addTest(Permutation({1, 2, 0, 3}), Permutation(), Permutation({0, 3, 2, 1}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_different_permutations_input2_output)
{
    addTest(Permutation(), Permutation({3, 2, 0, 1}), Permutation({0, 3, 2, 1}));
}

void SynTrainingTensorPermutationDualExecutionTest::addSameInputTest(const Permutation& inPermutation,
                                                                  const Permutation& outPermutation)
{
    std::array<unsigned, 4> dimSizes = {8, 16, 20, 25};

    auto in = createPersistTensors(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   dimSizes.data(),
                                   dimSizes.size(),
                                   syn_type_single);

    auto out = createPersistTensors(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    dimSizes.data(),
                                    dimSizes.size(),
                                    syn_type_single);

    setTensorPermutation(in, inPermutation);
    setTensorPermutation(out, outPermutation);

    TensorIndicesPair addInIndices  = {{in.graph, in.graph}, {in.eager, in.eager}};
    TensorIndicesPair addOutIndices = {{out.graph}, {out.eager}};

    addNodesToGraphs("add_fwd_f32", addInIndices, addOutIndices);

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[out.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[out.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(dimSizes.data(), dimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_same_input_output_permuted)
{
    addSameInputTest(Permutation(), Permutation({3, 2, 0, 1}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_same_input_permuted)
{
    addSameInputTest(Permutation({0, 3, 2, 1}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_same_input_all_permuted)
{
    addSameInputTest(Permutation({3, 2, 0, 1}), Permutation({0, 3, 2, 1}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, add_same_input_all_permuted2)
{
    addSameInputTest(Permutation({3, 1, 2, 0}), Permutation({3, 1, 2, 0}));
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, strided_view_with_gemm)
{
    std::array<unsigned, 4> inStridedViewDimSizes  = {1, 1, 512, 256};
    std::array<unsigned, 2> outStridedViewDimSizes = {512, 256};

    auto stridedViewIn = createPersistTensors(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              inStridedViewDimSizes.data(),
                                              inStridedViewDimSizes.size(),
                                              syn_type_single);

    setTensorPermutation(stridedViewIn, Permutation({2, 0, 1, 3}));

    auto stridedViewOut = createPersistTensors(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               outStridedViewDimSizes.data(),
                                               outStridedViewDimSizes.size(),
                                               syn_type_single);

    synStridedOpParams params = {};
    params.strides[0]         = 1;
    params.strides[1]         = 512;

    addNodesToGraphs(NodeFactory::stridedViewNodeTypeName, &params, sizeof(params));

    std::array<unsigned, 2> in2DimSizes = {80, 512};
    std::array<unsigned, 2> outDimSizes = {80, 256};

    auto gemmIn2 = createPersistTensors(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        in2DimSizes.data(),
                                        in2DimSizes.size(),
                                        syn_type_single);

    auto gemmOut = createPersistTensors(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        outDimSizes.data(),
                                        outDimSizes.size(),
                                        syn_type_single);

    TensorIndicesPair gemmInIndices  = {{stridedViewOut.graph, gemmIn2.graph}, {stridedViewOut.eager, gemmIn2.eager}};
    TensorIndicesPair gemmOutIndices = {{gemmOut.graph}, {gemmOut.eager}};

    addNodesToGraphs(NodeFactory::gemmNodeTypeName, gemmInIndices, gemmOutIndices);

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[gemmOut.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[gemmOut.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outDimSizes.data(), outDimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTensorPermutationDualExecutionTest, const_mult_add)
{
    std::array<unsigned, 4> dimSizes  = {1, 1, 512, 2048};
    std::array<unsigned, 1> dimSizes2 = {1};

    auto constantOutPair =
        createTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dimSizes2.data(), dimSizes2.size(), syn_type_single);

    ns_ConstantKernel::Params constParams        = {1.0};
    TensorIndicesPair         constantOutIndices = {{constantOutPair.graph}, {constantOutPair.eager}};
    addNodesToGraphs("constant_f32", {}, {constantOutIndices}, &constParams, sizeof(constParams));

    auto multIn = createPersistTensors(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       dimSizes.data(),
                                       dimSizes.size(),
                                       syn_type_single);
    auto multOut =
        createTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dimSizes.data(), dimSizes.size(), syn_type_single);

    setTensorPermutation(multIn, Permutation({3, 2, 0, 1}));

    TensorIndicesPair multInIndices  = {{multIn.graph, constantOutPair.graph}, {multIn.eager, constantOutPair.eager}};
    TensorIndicesPair multOutIndices = {{multOut.graph}, {multOut.eager}};

    addNodesToGraphs("mult_fwd_f32", multInIndices, multOutIndices);

    auto addIn = createPersistTensors(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      dimSizes.data(),
                                      dimSizes.size(),
                                      syn_type_single);

    auto addOut = createPersistTensors(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       dimSizes.data(),
                                       dimSizes.size(),
                                       syn_type_single,
                                       true);

    TensorIndicesPair addInIndices  = {{addIn.graph, multOut.graph}, {addIn.eager, multOut.eager}};
    TensorIndicesPair addOutIndices = {{addOut.graph}, {addOut.eager}};

    addNodesToGraphs("add_fwd_f32", addInIndices, addOutIndices);

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[addOut.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[addOut.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(dimSizes.data(), dimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}