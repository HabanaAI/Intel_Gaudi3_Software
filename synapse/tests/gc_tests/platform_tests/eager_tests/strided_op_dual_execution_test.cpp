#include "defs.h"
#include "fmt-9.1.0/include/fmt/core.h"
#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "synapse_common_types.h"
#include "types.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <data_types/bfloat16.h>
#include <numeric>
#include "graph_compiler/habana_nodes/transpose_utils.h"
#include "graph_compiler/habana_nodes/strided_insert_node.h"

using DimSizesVector = llvm_vecsmall::SmallVector<unsigned, tpc_lib_api::MAX_TENSOR_DIM>;
using StridedInsertInputSizesArr = std::array<DimSizesVector, StridedInsertNode::MIN_NUM_INPUTS>;

class SynTrainingStridedOpExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    TensorStridesVector getStridesInElements(const DimSizesVector& sizes);

    void stridedViewTest(const DimSizesVector&      inputSizes,
                         const DimSizesVector&      outputSizes,
                         const TensorStridesVector& strides,
                         uint64_t                   baseOffset);

    void stridedInsertTest(const StridedInsertInputSizesArr& inputSizes,
                           const DimSizesVector&             outputSizes,
                           const TensorStridesVector&        strides,
                           uint64_t                          baseOffset);
};

TensorStridesVector SynTrainingStridedOpExecutionTest::getStridesInElements(const DimSizesVector& sizes)
{
    TensorStridesVector result;
    result.reserve(sizes.size());
    result.push_back(1);
    for (int i = 0; i < sizes.size() - 1; i++)
    {
        result.push_back(result.back() * sizes[i]);
    }
    return result;
}

void SynTrainingStridedOpExecutionTest::stridedViewTest(const DimSizesVector&      inputSizes,
                                                        const DimSizesVector&      outputSizes,
                                                        const TensorStridesVector& strides,
                                                        uint64_t                   baseOffset)
{
    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    GraphIndexPair graphIndexPair = createNewGraphPair();

    auto stridedViewInPair  = createPersistTensors(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  const_cast<unsigned*>(inputSizes.data()),
                                                  inputSizes.size(),
                                                  syn_type_single,
                                                  false,
                                                  nullptr,
                                                  nullptr,
                                                  graphIndexPair);
    auto stridedViewOutPair = createPersistTensors(OUTPUT_TENSOR,
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   const_cast<unsigned*>(outputSizes.data()),
                                                   outputSizes.size(),
                                                   syn_type_single,
                                                   false,
                                                   nullptr,
                                                   nullptr,
                                                   graphIndexPair);

    synStridedOpParams params = {};
    params.baseOffset         = baseOffset;
    std::copy(strides.begin(), strides.end(), std::begin(params.strides));

    TensorIndicesPair inIndices  = {{stridedViewInPair.graph}, {stridedViewInPair.eager}};
    TensorIndicesPair outIndices = {{stridedViewOutPair.graph}, {stridedViewOutPair.eager}};

    addNodesToGraphs("strided_view", inIndices, outIndices, &params, sizeof(params), "strided_view", graphIndexPair);

    // reference is without the optimization
    GCFG_ENABLE_STRIDED_OP_DECODING.setValue(false);
    SynGaudiTestInfra::compileTopology("topology_graph", graphIndexPair.graph);
    GCFG_ENABLE_STRIDED_OP_DECODING.setValue(true);
    SynGaudiTestInfra::compileTopology("topology_eager", graphIndexPair.eager);

    runTopology(graphIndexPair);

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[stridedViewOutPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[stridedViewOutPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outputSizes.data(), outputSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, full_view_with_fallback, {synDeviceGaudi2})
{
    DimSizesVector      inputSizes  = {8, 8, 8};
    DimSizesVector      outputSizes = {4, 4, 4, 4};
    TensorStridesVector strides     = {1, 8, 0, 0};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         const_cast<unsigned*>(inputSizes.data()),
                         inputSizes.size(),
                         syn_type_single);
    auto stridedViewOutPair = createPersistTensors(OUTPUT_TENSOR,
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   const_cast<unsigned*>(outputSizes.data()),
                                                   outputSizes.size(),
                                                   syn_type_single);

    synStridedOpParams stridedOpParams = {};
    std::copy(strides.begin(), strides.end(), std::begin(stridedOpParams.strides));

    addNodesToGraphs("strided_view", &stridedOpParams, sizeof(stridedOpParams));

    TensorIndicesPair inputs;
    inputs.reserve(4);
    inputs.push_back(stridedViewOutPair);
    inputs.push_back(stridedViewOutPair);

    DimSizesVector maskSizes = {4, 4, 1, 4};
    for (int i = 0; i < 2; i++)
    {
        auto inputPair = createPersistTensors(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              const_cast<unsigned*>(maskSizes.data()),
                                              maskSizes.size(),
                                              syn_type_single);
        inputs.push_back(inputPair);
    }

    auto maskedBatchGemmOutPair = createPersistTensors(OUTPUT_TENSOR,
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       const_cast<unsigned*>(outputSizes.data()),
                                                       outputSizes.size(),
                                                       syn_type_single);

    TensorIndicesPair outputs;
    outputs.push_back(maskedBatchGemmOutPair);

    synGEMMParams gemmParams = {};
    addNodesToGraphs("masked_batch_gemm", inputs, outputs, &gemmParams, sizeof(gemmParams));

    long origConfigVal = GCFG_FORCE_EAGER.value();
    GCFG_FORCE_EAGER.setValue(2);
    compileAndRun();
    GCFG_FORCE_EAGER.setValue(origConfigVal);

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[stridedViewOutPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[stridedViewOutPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outputSizes.data(), outputSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, full_view_with_broadcast)
{
    stridedViewTest({3, 4, 5}, {12, 5, 6}, {1, 12, 0}, 0);
    stridedViewTest({3, 4, 5, 1}, {12, 5, 6}, {1, 12, 0}, 0);
    stridedViewTest({3, 4, 5}, {3, 4, 5, 6}, {1, 3, 12, 0}, 0);
    stridedViewTest({3, 4, 5}, {2, 3, 4, 5, 6}, {0, 1, 3, 12, 0}, 0);
    stridedViewTest({3, 4, 5}, {7, 3, 4, 2, 5}, {0, 1, 3, 0, 12}, 0);
    stridedViewTest({3, 4, 5}, {3, 4, 7, 5, 6}, {1, 3, 0, 12, 0}, 0);
    stridedViewTest({3, 4, 1, 5}, {3, 4, 7, 5, 6}, {1, 3, 0, 12, 0}, 0);
    stridedViewTest({1, 3, 4, 5, 1}, {2, 3, 4, 5, 6}, {0, 1, 3, 12, 0}, 0);
    stridedViewTest({1, 1, 1}, {3, 4, 5}, {0, 0, 0}, 0);
    stridedViewTest({1}, {3, 4, 5}, {0, 0, 0}, 0);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, partial_view_with_broadcast)
{
    stridedViewTest({8}, {4, 4, 4}, {1, 0, 0}, 0);
    stridedViewTest({8}, {4, 4, 4}, {2, 0, 0}, 0);
    stridedViewTest({8}, {4, 4, 4}, {1, 0, 0}, 4);
    stridedViewTest({16}, {4, 4, 4}, {2, 0, 0}, 4);
    stridedViewTest({16}, {4, 4, 4}, {0, 2, 0}, 4);
    stridedViewTest({16}, {4, 4, 4}, {0, 0, 2}, 4);
    stridedViewTest({16}, {4, 4, 4}, {0, 0, 2}, 0);
    stridedViewTest({1, 1, 16}, {4, 4, 4}, {0, 0, 2}, 0);
    stridedViewTest({1, 1, 16}, {4, 4, 4}, {0, 1, 4}, 0);
    stridedViewTest({1, 1, 32}, {4, 4, 4}, {0, 1, 4}, 16);
    stridedViewTest({8, 8}, {2, 4, 4}, {0, 2, 16}, 0);
    stridedViewTest({8, 8}, {2, 4, 4}, {0, 2, 16}, 4);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, full_view_with_permute, {synDeviceGaudi2})
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation  = identity;
    DimSizesVector            inputSizes   = {2, 3, 4, 5};
    TensorStridesVector       inputStrides = getStridesInElements(inputSizes);
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(inputSizes.data(), permutation, outputSizes.data());
        applyPermutation(inputStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, full_view_with_permute_and_trivial_dims, {synDeviceGaudi2})
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation  = identity;
    DimSizesVector            inputSizes   = {1, 3, 4, 1};
    TensorStridesVector       inputStrides = getStridesInElements(inputSizes);
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(inputSizes.data(), permutation, outputSizes.data());
        applyPermutation(inputStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, partial_view_with_permute)
{
    constexpr unsigned        dims = 3;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation         = identity;
    DimSizesVector            inputSizes          = {2, 4, 8};
    DimSizesVector            partialInputSizes   = {1, 2, 4};
    TensorStridesVector       partialInputStrides = getStridesInElements(partialInputSizes);
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(partialInputSizes.data(), permutation, outputSizes.data());
        applyPermutation(partialInputStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, full_view_with_permute_and_broadcast_no_expand)
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            inputSizes         = {1, 3, 4, 1};
    DimSizesVector            broadcastedSizes   = {5, 3, 4, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 0};
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
        applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, full_view_with_permute_and_broadcast)
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            inputSizes         = {3, 4};
    DimSizesVector            broadcastedSizes   = {5, 3, 4, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 0};
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
        applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, full_view_with_permute_and_broadcast2)
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            inputSizes         = {3, 4, 7};
    DimSizesVector            broadcastedSizes   = {5, 3, 4, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 12};
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
        applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, partial_view_with_permute_and_broadcast, {synDeviceGaudi2})
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            inputSizes         = {3, 4, 7};
    DimSizesVector            broadcastedSizes   = {5, 3, 2, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 12};
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
        applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, partial_view_with_permute_and_broadcast2, {synDeviceGaudi2})
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            inputSizes         = {3, 4, 7};
    DimSizesVector            broadcastedSizes   = {5, 3, 2, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 6, 12};
    do
    {
        DimSizesVector      outputSizes(dims);
        TensorStridesVector outputStrides(dims);
        applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
        applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
        stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

void SynTrainingStridedOpExecutionTest::stridedInsertTest(const StridedInsertInputSizesArr& inputSizes,
                                                          const DimSizesVector&             outputSizes,
                                                          const TensorStridesVector&        strides,
                                                          uint64_t                          baseOffset)
{
    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    GraphIndexPair graphIndexPair = createNewGraphPair();

    auto stridedViewInPair =
        createPersistTensors(INPUT_TENSOR,
                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                             nullptr,
                             const_cast<unsigned*>(inputSizes[StridedInsertNode::ORIGINAL_TENSOR].data()),
                             inputSizes[StridedInsertNode::ORIGINAL_TENSOR].size(),
                             syn_type_single,
                             false,
                             nullptr,
                             nullptr,
                             graphIndexPair);

    auto stridedInsertInPair =
        createPersistTensors(INPUT_TENSOR,
                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                             nullptr,
                             const_cast<unsigned*>(inputSizes[StridedInsertNode::INSERT_TENSOR].data()),
                             inputSizes[StridedInsertNode::INSERT_TENSOR].size(),
                             syn_type_single,
                             false,
                             nullptr,
                             nullptr,
                             graphIndexPair);

    auto stridedInsertOutPair = createPersistTensors(OUTPUT_TENSOR,
                                                     MEM_INIT_ALL_ZERO,
                                                     nullptr,
                                                     const_cast<unsigned*>(outputSizes.data()),
                                                     outputSizes.size(),
                                                     syn_type_single,
                                                     false,
                                                     nullptr,
                                                     nullptr,
                                                     graphIndexPair);

    synStridedOpParams params = {};
    params.baseOffset         = baseOffset;
    std::copy(strides.begin(), strides.end(), std::begin(params.strides));

    TensorIndicesPair inIndices  = {{stridedViewInPair.graph, stridedInsertInPair.graph},
                                    {stridedViewInPair.eager, stridedInsertInPair.eager}};
    TensorIndicesPair outIndices = {{stridedInsertOutPair.graph}, {stridedInsertOutPair.eager}};

    addNodesToGraphs("strided_insert",
                     inIndices,
                     outIndices,
                     &params,
                     sizeof(params),
                     "strided_insert",
                     graphIndexPair);

    // reference is without the optimization
    GCFG_ENABLE_STRIDED_OP_DECODING.setValue(false);
    SynGaudiTestInfra::compileTopology("topology_graph", graphIndexPair.graph);
    GCFG_ENABLE_STRIDED_OP_DECODING.setValue(true);
    SynGaudiTestInfra::compileTopology("topology_eager", graphIndexPair.eager);

    runTopology(graphIndexPair);

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[stridedInsertOutPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[stridedInsertOutPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outputSizes.data(), outputSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, strided_insert_with_permute)
{
    constexpr unsigned        dims = 3;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            outputSizes        = {6, 8, 10};
    DimSizesVector            insertInputSizes   = {3, 2, 5};
    TensorStridesVector       insertInputStrides = getStridesInElements(insertInputSizes);
    unsigned                  iteration          = 0;
    do
    {
        TensorStridesVector insertStrides(dims);
        DimSizesVector      insertSizes(dims);
        applyPermutation(insertInputStrides.data(), permutation, insertStrides.data());
        applyPermutation(insertInputSizes.data(), permutation, insertSizes.data());
        stridedInsertTest({outputSizes, insertSizes}, outputSizes, insertStrides, iteration % dims);
        std::next_permutation(permutation.begin(), permutation.end());
        ++iteration;
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, strided_insert_with_permute2)
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            outputSizes        = {1200};
    DimSizesVector            insertInputSizes   = {3, 2, 5, 4};
    TensorStridesVector       insertInputStrides = getStridesInElements(insertInputSizes);
    unsigned                  iteration          = 0;
    do
    {
        TensorStridesVector insertStrides(dims);
        DimSizesVector      insertSizes(dims);
        applyPermutation(insertInputStrides.data(), permutation, insertStrides.data());
        applyPermutation(insertInputSizes.data(), permutation, insertSizes.data());
        stridedInsertTest({outputSizes, insertSizes}, outputSizes, insertStrides, iteration % dims);
        std::next_permutation(permutation.begin(), permutation.end());
        ++iteration;
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, strided_insert_with_permute_and_reshape_full_overwrite)
{
    constexpr unsigned        dims = 4;
    TransposePermutationArray identity;
    identity.reserve(dims);
    for (int i = 0; i < dims; i++)
    {
        identity.push_back(static_cast<TransposePermutationDim>(i));
    }
    TransposePermutationArray permutation        = identity;
    DimSizesVector            outputSizes        = {120};
    DimSizesVector            insertInputSizes   = {3, 2, 5, 4};
    TensorStridesVector       insertInputStrides = getStridesInElements(insertInputSizes);
    do
    {
        TensorStridesVector insertStrides(dims);
        DimSizesVector      insertSizes(dims);
        applyPermutation(insertInputStrides.data(), permutation, insertStrides.data());
        applyPermutation(insertInputSizes.data(), permutation, insertSizes.data());
        stridedInsertTest({outputSizes, insertSizes}, outputSizes, insertStrides, 0);
        std::next_permutation(permutation.begin(), permutation.end());
    } while (identity != permutation);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, strided_insert_no_optimization)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {4, 5}}}, {2, 3, 4, 5, 6}, {1, 20}, 0);
    stridedInsertTest({{{2, 3, 4, 5, 6}, {4, 5}}}, {2, 3, 4, 5, 6}, {1, 4}, 0);
    stridedInsertTest({{{2, 3, 4, 5, 6}, {4, 5}}}, {2, 3, 4, 5, 6}, {1, 10}, 5);
    stridedInsertTest({{{30, 20}, {4, 5}}}, {30, 20}, {1, 20}, 0);
    stridedInsertTest({{{30, 20}, {4, 5}}}, {30, 20}, {1, 4}, 0);
    stridedInsertTest({{{30, 20}, {4, 5}}}, {30, 20}, {1, 10}, 5);
}

TEST_F_GC(SynTrainingStridedOpExecutionTest, strided_insert_with_reshape_full_overwrite)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}, {2, 3, 4, 5, 6}, {1, 2, 6, 24, 120}, 0);
    stridedInsertTest({{{2, 3, 4, 5, 6}, {720}}}, {2, 3, 4, 5, 6}, {1}, 0);
    stridedInsertTest({{{2, 3, 4, 5, 6}, {72, 10}}}, {2, 3, 4, 5, 6}, {1, 72}, 0);
    stridedInsertTest({{{720}, {72, 10}}}, {720}, {1, 72}, 0);
}