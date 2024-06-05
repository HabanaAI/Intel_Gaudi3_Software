#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "timer.h"
#include <cstdint>
#include <data_types/bfloat16.h>
#include <iostream>
#include <tuple>
#include <deque>
#include "tensor_io_indicies.h"
#include "synapse_api_types.h"
#include "utils.h"

class SynTrainingTestInfraConstFolding : public SynTrainingTestInfra
{
protected:
    virtual void SetUpTest() override
    {
        SynTrainingTestInfra::SetUpTest();
        pushGlobalConf("ENABLE_CONSTANT_FOLDING", "true");
    }
};

TEST_F_GC(SynTrainingTestInfraConstFolding, add_forward_with_single_const_section)
{
    unsigned dims    = 4;
    unsigned sizes[] = {5, 2, 1, 3};

    unsigned totalSizeInBytes = dataTypeSizeInBytes(syn_type_bf16);
    for (int i = 0; i < dims; i++)
    {
        totalSizeInBytes *= sizes[i];
    }
    // create graph
    unsigned addIn = createPersistTensor(TensorUsage::INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sizes,
                                         dims,
                                         syn_type_single,
                                         nullptr,
                                         "addInTensor");

    unsigned castIn  = createConstPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               sizes,
                                               dims,
                                               syn_type_bf16,
                                               nullptr,
                                               "castInTensor");
    unsigned castOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_single);

    auto outputIndex =
        createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_single);

    setGraphInferenceMode();

    addNodeToGraph("cast_bf16_to_f32", {castIn}, {castOut}, nullptr, 0, "castNode");
    addNodeToGraph("add_fwd_f32", {castOut, addIn}, {outputIndex});

    compileTopology();

    synSectionId    castInputSectionId;
    uint64_t        section_size    = 0;
    synRecipeHandle recipeHandle    = getGraph(0).recipeHandle;
    synTensor       castInputTensor = getTensorByIndex(castIn);

    getTensorSectionId(recipeHandle, castInputTensor, castInputSectionId);

    ASSERT_EQ(synSuccess, synRecipeSectionGetProp(recipeHandle, castInputSectionId, SECTION_SIZE, &section_size));
    ASSERT_GT(section_size, 0) << "const section size is zero!";
    // in the const section there is the output of the cast
    ASSERT_EQ(totalSizeInBytes * 2, section_size);

    runTopology();

    // validate results
    float*    InputBuffer1 = castHostInBuffer<float>(addIn);
    bfloat16* InputBuffer2 = castHostInBuffer<bfloat16>(castIn);
    float*    outputBuffer = castHostOutBuffer<float>(outputIndex);

    for (uint64_t i = 0; i < getTensorElementCount(castIn); ++i)
    {
        ASSERT_EQ(InputBuffer1[i] + float(InputBuffer2[i]), outputBuffer[i]);
    }
}

TEST_F_GC(SynTrainingTestInfraConstFolding, reshape_with_single_const_section)
{
    unsigned dimsIn     = 3;
    unsigned dimsOut    = 4;
    unsigned sizesIn[]  = {8, 4, 4};
    unsigned sizesOut[] = {16, 4, 1, 2};

    unsigned reshapeIn = createConstPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  sizesIn,
                                                  dimsIn,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  "reshapeIn");

    unsigned reshapeOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesOut, dimsOut, syn_type_bf16);
    unsigned castOut =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesOut, dimsOut, syn_type_single);

    setGraphInferenceMode();

    addNodeToGraph("reshape", {reshapeIn}, {reshapeOut}, nullptr, 0, "reshapeNode");
    addNodeToGraph("cast_bf16_to_f32", {reshapeOut}, {castOut}, nullptr, 0, "castNode");

    compileTopology();

    runTopology();

    // validate results
    bfloat16* InputBuffer  = castHostInBuffer<bfloat16>(reshapeIn);
    float*    outputBuffer = castHostOutBuffer<float>(castOut);

    for (uint64_t i = 0; i < getTensorElementCount(reshapeIn); ++i)
    {
        ASSERT_EQ(float(InputBuffer[i]), outputBuffer[i]);
    }
}

TEST_F_GC(SynTrainingTestInfraConstFolding, transpose_with_single_const_section)
{
    unsigned dims       = 2;
    unsigned sizesIn[]  = {4224, 128};
    unsigned sizesOut[] = {128, 4224};

    unsigned transposeIn = createConstPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                    nullptr,
                                                    sizesIn,
                                                    dims,
                                                    syn_type_bf16,
                                                    nullptr,
                                                    "transposeIn");

    unsigned transposeOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesOut, dims, syn_type_bf16);
    unsigned castOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesOut, dims, syn_type_single);

    synTransposeParams transposNodeParams = {{TPD_Width, TPD_Channel}, 2};

    setGraphInferenceMode();

    addNodeToGraph("transpose",
                   {transposeIn},
                   {transposeOut},
                   &transposNodeParams,
                   sizeof(transposNodeParams),
                   "transposeNode");
    addNodeToGraph("cast_bf16_to_f32", {transposeOut}, {castOut}, nullptr, 0, "castNode");

    compileTopology();

    runTopology();

    // validate results
    bfloat16*      InputBuffer  = castHostInBuffer<bfloat16>(transposeIn);
    float*         outputBuffer = castHostOutBuffer<float>(castOut);
    const unsigned lastIndex    = getTensorElementCount(castOut) - 1;
    for (uint64_t i = 0; i < lastIndex; ++i)
    {
        int j = (i * sizesIn[1]) % (lastIndex);
        ASSERT_EQ(float(InputBuffer[i]), outputBuffer[j]);
    }
    ASSERT_EQ(float(InputBuffer[lastIndex]), outputBuffer[lastIndex]);
}