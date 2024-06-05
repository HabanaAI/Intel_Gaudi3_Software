#include "syn_gaudi_two_run_compare_test.h"
#include "graph_compiler/habana_nodes/transpose_utils.h"
#include "graph_compiler/habana_nodes/strided_insert_node.h"

using DimSizesVector             = llvm_vecsmall::SmallVector<unsigned, tpc_lib_api::MAX_TENSOR_DIM>;
using StridedInsertInputSizesArr = std::array<DimSizesVector, StridedInsertNode::MIN_NUM_INPUTS>;

class SynTrainingStridedOpDecoderTest : public SynTrainingTwoRunCompareTest
{
public:
    void stridedViewTest(const DimSizesVector&      inputSizes,
                         const DimSizesVector&      outputSizes,
                         const TensorStridesVector& strides,
                         uint64_t                   baseOffset);

    void stridedInsertTest(const StridedInsertInputSizesArr& inputSizes,
                           const DimSizesVector&             outputSizes,
                           const TensorStridesVector&        strides,
                           uint64_t                          baseOffset);
};

static TensorStridesVector getStridesInElements(const DimSizesVector& sizes)
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

void SynTrainingStridedOpDecoderTest::stridedViewTest(const DimSizesVector&      inputSizes,
                                                      const DimSizesVector&      outputSizes,
                                                      const TensorStridesVector& strides,
                                                      uint64_t                   baseOffset)
{
    auto negIn = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     const_cast<unsigned*>(inputSizes.data()),
                                     inputSizes.size(),
                                     syn_type_single);

    auto negOut = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      const_cast<unsigned*>(inputSizes.data()),
                                      inputSizes.size(),
                                      syn_type_single);

    addNodeToGraph("neg_fwd_f32", {negIn}, {negOut});

    auto stridedViewOut = createPersistTensor(OUTPUT_TENSOR,
                                              MEM_INIT_ALL_ZERO,
                                              nullptr,
                                              const_cast<unsigned*>(outputSizes.data()),
                                              outputSizes.size(),
                                              syn_type_single);

    synStridedOpParams params = {};
    params.baseOffset         = baseOffset;
    std::copy(strides.begin(), strides.end(), std::begin(params.strides));
    addNodeToGraph("strided_view", {negOut}, {stridedViewOut}, &params, sizeof(params));

    auto addOut = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      const_cast<unsigned*>(outputSizes.data()),
                                      outputSizes.size(),
                                      syn_type_single);

    addNodeToGraph("add_fwd_f32", {stridedViewOut, stridedViewOut}, {addOut});

    addConfigurationToRun(FIRST_RUN, "ENABLE_STRIDED_OP_DECODING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_STRIDED_OP_DECODING", "true");
    compareRunsResults({addOut});
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_broadcast1)
{
    stridedViewTest({3, 4, 5}, {12, 5, 6}, {1, 12, 0}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_broadcast2)
{
    stridedViewTest({3, 4, 5, 1}, {12, 5, 6}, {1, 12, 0}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_broadcast3)
{
    stridedViewTest({3, 4, 5}, {7, 3, 4, 2, 5}, {0, 1, 3, 0, 12}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_broadcast4)
{
    stridedViewTest({1}, {3, 4, 5}, {0, 0, 0}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, partial_view_with_broadcast1)
{
    stridedViewTest({8}, {4, 4, 4}, {1, 0, 0}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, partial_view_with_broadcast2)
{
    stridedViewTest({8}, {4, 4, 4}, {1, 0, 0}, 4);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, partial_view_with_broadcast3)
{
    stridedViewTest({8, 8}, {2, 4, 4}, {0, 2, 16}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, partial_view_with_broadcast4)
{
    stridedViewTest({8, 8}, {2, 4, 4}, {0, 2, 16}, 4);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_permute_WHCN)
{
    constexpr unsigned        dims         = 4;
    TransposePermutationArray permutation  = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            inputSizes   = {2, 3, 4, 5};
    TensorStridesVector       inputStrides = getStridesInElements(inputSizes);
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(inputSizes.data(), permutation, outputSizes.data());
    applyPermutation(inputStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_permute_and_trivial_dims_WHCN)
{
    constexpr unsigned        dims         = 4;
    TransposePermutationArray permutation  = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            inputSizes   = {1, 3, 4, 1};
    TensorStridesVector       inputStrides = getStridesInElements(inputSizes);
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(inputSizes.data(), permutation, outputSizes.data());
    applyPermutation(inputStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, partial_view_with_permute_WHC)
{
    constexpr unsigned        dims                = 3;
    TransposePermutationArray permutation         = {TPD_Width, TPD_Height, TPD_Channel};
    DimSizesVector            inputSizes          = {2, 4, 8};
    DimSizesVector            partialInputSizes   = {1, 2, 4};
    TensorStridesVector       partialInputStrides = getStridesInElements(partialInputSizes);
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(partialInputSizes.data(), permutation, outputSizes.data());
    applyPermutation(partialInputStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_permute_and_broadcast_no_expand_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            inputSizes         = {1, 3, 4, 1};
    DimSizesVector            broadcastedSizes   = {5, 3, 4, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 0};
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
    applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_permute_and_broadcast_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            inputSizes         = {3, 4};
    DimSizesVector            broadcastedSizes   = {5, 3, 4, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 0};
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
    applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, full_view_with_permute_and_broadcast2_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            inputSizes         = {3, 4, 7};
    DimSizesVector            broadcastedSizes   = {5, 3, 4, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 12};
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
    applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, partial_view_with_permute_and_broadcast_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            inputSizes         = {3, 4, 7};
    DimSizesVector            broadcastedSizes   = {5, 3, 2, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 3, 12};
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
    applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, partial_view_with_permute_and_broadcast2_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            inputSizes         = {3, 4, 7};
    DimSizesVector            broadcastedSizes   = {5, 3, 2, 7};
    TensorStridesVector       broadcastedStrides = {0, 1, 6, 12};
    DimSizesVector            outputSizes(dims);
    TensorStridesVector       outputStrides(dims);
    applyPermutation(broadcastedSizes.data(), permutation, outputSizes.data());
    applyPermutation(broadcastedStrides.data(), permutation, outputStrides.data());
    stridedViewTest(inputSizes, outputSizes, outputStrides, 0);
}

void SynTrainingStridedOpDecoderTest::stridedInsertTest(const StridedInsertInputSizesArr& inputSizes,
                                                        const DimSizesVector&             outputSizes,
                                                        const TensorStridesVector&        strides,
                                                        uint64_t                          baseOffset)
{
    unsigned origTensor =
        createPersistTensor(INPUT_TENSOR,
                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                            nullptr,
                            const_cast<unsigned*>(inputSizes[StridedInsertNode::ORIGINAL_TENSOR].data()),
                            inputSizes[StridedInsertNode::ORIGINAL_TENSOR].size(),
                            syn_type_single);

    unsigned insertTensor =
        createPersistTensor(INPUT_TENSOR,
                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                            nullptr,
                            const_cast<unsigned*>(inputSizes[StridedInsertNode::INSERT_TENSOR].data()),
                            inputSizes[StridedInsertNode::INSERT_TENSOR].size(),
                            syn_type_single);

    unsigned stridedInsertOut = createPersistTensor(OUTPUT_TENSOR,
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    const_cast<unsigned*>(outputSizes.data()),
                                                    outputSizes.size(),
                                                    syn_type_single);

    synStridedOpParams params = {};
    params.baseOffset         = baseOffset;
    std::copy(strides.begin(), strides.end(), std::begin(params.strides));

    addNodeToGraph("strided_insert", {origTensor, insertTensor}, {stridedInsertOut}, &params, sizeof(params));

    auto negOut = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      const_cast<unsigned*>(outputSizes.data()),
                                      outputSizes.size(),
                                      syn_type_single);

    addNodeToGraph("neg_fwd_f32", {stridedInsertOut}, {negOut});

    addConfigurationToRun(FIRST_RUN, "ENABLE_STRIDED_OP_DECODING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_STRIDED_OP_DECODING", "true");
    compareRunsResults({negOut});
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_permute_WHC)
{
    constexpr unsigned        dims               = 3;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel};
    DimSizesVector            outputSizes        = {6, 8, 10};
    DimSizesVector            insertInputSizes   = {3, 2, 5};
    TensorStridesVector       insertInputStrides = getStridesInElements(insertInputSizes);
    TensorStridesVector       insertStrides(dims);
    DimSizesVector            insertSizes(dims);
    applyPermutation(insertInputStrides.data(), permutation, insertStrides.data());
    applyPermutation(insertInputSizes.data(), permutation, insertSizes.data());
    stridedInsertTest({outputSizes, insertSizes}, outputSizes, insertStrides, 0);
    std::next_permutation(permutation.begin(), permutation.end());
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_permute2_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            outputSizes        = {1200};
    DimSizesVector            insertInputSizes   = {3, 2, 5, 4};
    TensorStridesVector       insertInputStrides = getStridesInElements(insertInputSizes);
    TensorStridesVector       insertStrides(dims);
    DimSizesVector            insertSizes(dims);
    applyPermutation(insertInputStrides.data(), permutation, insertStrides.data());
    applyPermutation(insertInputSizes.data(), permutation, insertSizes.data());
    stridedInsertTest({outputSizes, insertSizes}, outputSizes, insertStrides, 0);
    std::next_permutation(permutation.begin(), permutation.end());
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_permute3_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            outputSizes        = {1200};
    DimSizesVector            insertInputSizes   = {3, 2, 5, 4};
    TensorStridesVector       insertInputStrides = getStridesInElements(insertInputSizes);
    TensorStridesVector       insertStrides(dims);
    DimSizesVector            insertSizes(dims);
    applyPermutation(insertInputStrides.data(), permutation, insertStrides.data());
    applyPermutation(insertInputSizes.data(), permutation, insertSizes.data());
    stridedInsertTest({outputSizes, insertSizes}, outputSizes, insertStrides, 1);
    std::next_permutation(permutation.begin(), permutation.end());
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_permute_and_reshape_full_overwrite_WHCN)
{
    constexpr unsigned        dims               = 4;
    TransposePermutationArray permutation        = {TPD_Width, TPD_Height, TPD_Channel, TPD_Depth};
    DimSizesVector            outputSizes        = {120};
    DimSizesVector            insertInputSizes   = {3, 2, 5, 4};
    TensorStridesVector       insertInputStrides = getStridesInElements(insertInputSizes);
    TensorStridesVector       insertStrides(dims);
    DimSizesVector            insertSizes(dims);
    applyPermutation(insertInputStrides.data(), permutation, insertStrides.data());
    applyPermutation(insertInputSizes.data(), permutation, insertSizes.data());
    stridedInsertTest({outputSizes, insertSizes}, outputSizes, insertStrides, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_no_optimization1)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {4, 5}}}, {2, 3, 4, 5, 6}, {1, 20}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_no_optimization2)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {4, 5}}}, {2, 3, 4, 5, 6}, {1, 4}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_no_optimization3)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {4, 5}}}, {2, 3, 4, 5, 6}, {1, 10}, 5);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_no_optimization4)
{
    stridedInsertTest({{{30, 20}, {4, 5}}}, {30, 20}, {1, 20}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_no_optimization5)
{
    stridedInsertTest({{{30, 20}, {4, 5}}}, {30, 20}, {1, 4}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_no_optimization6)
{
    stridedInsertTest({{{30, 20}, {4, 5}}}, {30, 20}, {1, 10}, 5);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_reshape_full_overwrite1)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}, {2, 3, 4, 5, 6}, {1, 2, 6, 24, 120}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_reshape_full_overwrite2)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {720}}}, {2, 3, 4, 5, 6}, {1}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_reshape_full_overwrite3)
{
    stridedInsertTest({{{2, 3, 4, 5, 6}, {72, 10}}}, {2, 3, 4, 5, 6}, {1, 72}, 0);
}

TEST_F_GC(SynTrainingStridedOpDecoderTest, strided_insert_with_reshape_full_overwrite4)
{
    stridedInsertTest({{{720}, {72, 10}}}, {720}, {1, 72}, 0);
}