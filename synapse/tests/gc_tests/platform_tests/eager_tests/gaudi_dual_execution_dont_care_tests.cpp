#include "gaudi_dual_execution_test_infra.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "data_type_utils.h"
#include <numeric>

using namespace gc;

// TODO : Enable tests for Gaudi3 (SW-160880)

class SynTrainingDontCareDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    SynTrainingDontCareDualExecutionTest () { ReleaseDevice(); }

    static constexpr unsigned NUM_DIMS = 4;
    void                      reluTest(const Permutation& inPermutation,
                                       const Permutation& outPermutation      = Permutation(),
                                       bool               inAllowPermutation  = false,
                                       bool               outAllowPermutation = false);

    void sliceTest(const Permutation& inPermutation, const Permutation& outPermutation = Permutation());
    void concatTest(const Permutation& in1Permutation,
                    const Permutation& in2Permutation = Permutation(),
                    const Permutation& outPermutation = Permutation());
};

void SynTrainingDontCareDualExecutionTest::reluTest(const Permutation& inPermutation,
                                                 const Permutation& outPermutation,
                                                 bool               inAllowPermutation,
                                                 bool               outAllowPermutation)
{
    std::array<unsigned, NUM_DIMS> ifmDimSizes = {3, 32, 64, 5};

    auto inTensorIndexPair  = createPersistTensors(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_POSITIVE,
                                                  nullptr,
                                                  ifmDimSizes.data(),
                                                  ifmDimSizes.size(),
                                                  syn_type_single);
    auto outTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   ifmDimSizes.data(),
                                                   ifmDimSizes.size(),
                                                   syn_type_single);

    setTensorPermutation(inTensorIndexPair, inPermutation);
    if (inAllowPermutation) setTensorAllowPermutations(inTensorIndexPair);
    setTensorPermutation(outTensorIndexPair, outPermutation);
    if (outAllowPermutation) setTensorAllowPermutations(outTensorIndexPair);

    addNodesToGraphs("relu_fwd_f32");

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[outTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[outTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(ifmDimSizes.data()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_input_permuted)
{
    reluTest(Permutation({2, 1, 0, 3}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_input_permuted2)
{
    reluTest(Permutation({2, 0, 1, 3}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_input_permuted3)
{
    reluTest(Permutation({1, 2, 0, 3}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_input_permuted4)
{
    reluTest(Permutation({1, 0, 3, 2}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_input_permuted_allow_permutation)
{
    reluTest(Permutation({2, 1, 0, 3}), Permutation(), true);
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_output_permuted)
{
    reluTest(Permutation(), Permutation({2, 1, 0, 3}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_output_permuted_allow_permutation)
{
    reluTest(Permutation(), Permutation({2, 1, 0, 3}), false, true);
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_both_permuted)
{
    reluTest(Permutation({3, 0, 1, 2}), Permutation({2, 1, 0, 3}), false, false);
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_both_permuted_allow_permutation)
{
    reluTest(Permutation({3, 0, 1, 2}), Permutation({2, 1, 0, 3}), true, true);
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_both_permuted_same_permutation)
{
    reluTest(Permutation({3, 0, 1, 2}), Permutation({3, 0, 1, 2}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, relu_fwd_f32_identity_permutation)
{
    reluTest(Permutation({0, 1, 2, 3}), Permutation({0, 1, 2, 3}));
}

void SynTrainingDontCareDualExecutionTest::sliceTest(const Permutation& inPermutation, const Permutation& outPermutation)
{
    std::array<unsigned, NUM_DIMS> initialDimSizes = {256, 5, 2, 3};
    std::array<unsigned, NUM_DIMS> dimSizes        = {128, 5, 2, 3};
    auto                           in              = createPersistTensors(INPUT_TENSOR,
                                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                          nullptr,
                                                                          initialDimSizes.data(),
                                                                          initialDimSizes.size(),
                                                                          syn_type_single);
    auto                           out             = createPersistTensors(OUTPUT_TENSOR,
                                                                          MEM_INIT_ALL_ZERO,
                                                                          nullptr,
                                                                          dimSizes.data(),
                                                                          dimSizes.size(),
                                                                          syn_type_single);

    setTensorPermutation(in, inPermutation);
    setTensorPermutation(out, outPermutation);

    synSliceParams params = {};
    std::iota(std::begin(params.axes), std::end(params.axes), 0);
    std::fill(std::begin(params.steps), std::end(params.steps), 1);
    memcpy(params.ends, dimSizes.data(), dimSizes.size() * sizeof(params.ends[0]));
    addNodesToGraphs(NodeFactory::sliceNodeTypeName, &params, sizeof(params));

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

TEST_F_GC(SynTrainingDontCareDualExecutionTest, slice_permuted_input)
{
    sliceTest(Permutation({2, 3, 1, 0}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, slice_permuted_output)
{
    sliceTest(Permutation(), Permutation({2, 3, 0, 1}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, slice_permuted_both)
{
    sliceTest(Permutation({3, 2, 1, 0}), Permutation({2, 3, 0, 1}));
}

void SynTrainingDontCareDualExecutionTest::concatTest(const Permutation& in1Permutation,
                                                   const Permutation& in2Permutation,
                                                   const Permutation& outPermutation)
{
    constexpr TSize                c                    = 3;
    constexpr TSize                w                    = 20;
    constexpr TSize                h                    = 10;
    constexpr TSize                batch                = 1;
    std::array<unsigned, NUM_DIMS> concatInputDimSizes  = {w, h, c, batch};
    std::array<unsigned, NUM_DIMS> concatOutputDimSizes = {2 * w, h, c, batch};

    auto in1 = createPersistTensors(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    concatInputDimSizes.data(),
                                    concatInputDimSizes.size(),
                                    syn_type_single);

    auto in2 = createPersistTensors(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    concatInputDimSizes.data(),
                                    concatInputDimSizes.size(),
                                    syn_type_single);

    auto out = createPersistTensors(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    concatOutputDimSizes.data(),
                                    concatOutputDimSizes.size(),
                                    syn_type_single);

    setTensorPermutation(in1, in1Permutation);
    setTensorPermutation(in2, in2Permutation);
    setTensorPermutation(out, outPermutation);

    synConcatenateParams concatParams = {};
    addNodesToGraphs("concat", &concatParams, sizeof(concatParams));

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[out.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[out.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(concatOutputDimSizes.data(), concatOutputDimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_input1)
{
    concatTest(Permutation({2, 3, 1, 0}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_input2)
{
    concatTest(Permutation(), Permutation({2, 3, 0, 1}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_both_inputs_same_permutation)
{
    concatTest(Permutation({2, 3, 0, 1}), Permutation({2, 3, 0, 1}), Permutation());
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_both_inputs, {synDeviceGaudi2})
{
    concatTest(Permutation({2, 3, 0, 1}), Permutation({3, 2, 1, 0}), Permutation());
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_output)
{
    concatTest(Permutation(), Permutation(), Permutation({2, 3, 0, 1}));
}

TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_input1_and_output, {synDeviceGaudi2})
{
    concatTest(Permutation({1, 3, 0, 2}), Permutation(), Permutation({2, 3, 0, 1}));
}

// TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_input2_and_output)
// {
//     concatTest(Permutation(), Permutation({0, 3, 2, 1}), Permutation({2, 3, 0, 1}));
// }

TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_all, {synDeviceGaudi2})
{
    concatTest(Permutation({0, 3, 2, 1}), Permutation({2, 3, 0, 1}), Permutation({1, 3, 0, 2}));
}

// TEST_F_GC(SynTrainingDontCareDualExecutionTest, concat_permuted_all_same_permutation)
// {
//     concatTest(Permutation({0, 3, 2, 1}), Permutation({0, 3, 2, 1}), Permutation({0, 3, 2, 1}));
// }

TEST_F_GC(SynTrainingDontCareDualExecutionTest, unet_slice_of_permuted_tensor, {synDeviceGaudi2})
{
    std::array<unsigned, 4> sliceInputSizes  = {10, 12, 1024, 64};
    std::array<unsigned, 4> sliceOutputSizes = {10, 12, 512, 64};

    auto sliceInput = createPersistTensors(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           sliceInputSizes.data(),
                                           sliceInputSizes.size(),
                                           syn_type_single);

    setTensorPermutation(sliceInput, Permutation({2, 0, 1, 3}));

    auto sliceOutput = createPersistTensors(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            sliceOutputSizes.data(),
                                            sliceOutputSizes.size(),
                                            syn_type_single);

    synSliceParams params = {};
    params.axes[0]        = 2;
    params.ends[0]        = 512;
    std::fill(std::begin(params.steps), std::end(params.steps), 1);
    addNodesToGraphs(NodeFactory::sliceNodeTypeName, &params, sizeof(params));

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[sliceOutput.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[sliceOutput.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(sliceOutputSizes.data(), sliceOutputSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}