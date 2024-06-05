#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "timer.h"
#include <tuple>
#include <deque>
#include "tensor_io_indicies.h"

class SynTrainingTestInfraTpcAdd : public SynTrainingTestInfra
{
public:
    SynTrainingTestInfraTpcAdd() { ReleaseDevice(); }

    template<typename T, size_t dimSize>
    TensorsIOIndicies addNodeAddFwd(const float* initInput1, const float* initInput2, unsigned (&sizes)[dimSize], const char* nodeGuid)
    {
        auto inputIndex1 = createPersistTensor(
            TensorUsage::INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, initInput1, sizes, dimSize, asSynType<T>());

        auto inputIndex2 = createPersistTensor(
            TensorUsage::INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, initInput2, sizes, dimSize, asSynType<T>());

        auto outputsIndex =
            createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dimSize, asSynType<T>());

        addNodeToGraph(nodeGuid, TensorIndices{inputIndex1, inputIndex2}, TensorIndices{outputsIndex});
        return TensorsIOIndicies{std::vector<int>{inputIndex1, inputIndex2}, std::vector<int>{outputsIndex}};
    }
};

TEST_F_GC(SynTrainingTestInfraTpcAdd, add_bf16_forward)
{
    // create data
    unsigned sizes[] = {1, 1, 1, 1};

    float firstValue  = 1;
    float secondValue = 0;

    // create graph
    TensorsIOIndicies inputOutputTensors = addNodeAddFwd<bfloat16>(&firstValue, &secondValue, sizes, "add_fwd_bf16");
    compileAndRun();

    // Validating output results
    ASSERT_EQ(static_cast<float>(castHostInBuffer<bfloat16>(inputOutputTensors.outputs[0])[0]), 1);
}

TEST_F_GC(SynTrainingTestInfraTpcAdd, add_forward)
{
    // create data
    unsigned sizes[] = {1000, 1, 1, 64};
    float inputBuffer1[1000* 1* 1* 64];
    fillWithRandom(inputBuffer1, 1000* 1* 1* 64);
    float inputBuffer2[1000* 1* 1* 64];
    fillWithRandom(inputBuffer2, 1000* 1* 1* 64);

    // create graph
    auto tensorsIndexes = addNodeAddFwd<float>(inputBuffer1, inputBuffer2, sizes, "add_fwd_f32");
    compileAndRun();

    // validate results
    float* myInputBuffer1 = castHostInBuffer<float>(tensorsIndexes.inputs[0]);
    float* myInputBuffer2 = castHostInBuffer<float>(tensorsIndexes.inputs[1]);
    float* outputBuffer   = castHostOutBuffer<float>(tensorsIndexes.outputs[0]);

    for (uint64_t i = 0; i < getTensorElementCount(tensorsIndexes.inputs[0]); ++i)
    {
        ASSERT_EQ(myInputBuffer1[i] + myInputBuffer2[i], outputBuffer[i]);
    }
}

TEST_F_GC(SynTrainingTestInfraTpcAdd, add_forward_with_single_const_section)
{
    // create data
    unsigned dims    = 4;
    unsigned sizes[] = {1000, 1, 1, 64};
    float    inputBuffer1[1000 * 1 * 1 * 64];
    fillWithRandom(inputBuffer1, 1000 * 1 * 1 * 64);
    float inputBuffer2[1000 * 1 * 1 * 64];
    fillWithRandom(inputBuffer2, 1000 * 1 * 1 * 64);

    unsigned totalSizeInBytes = sizeof(syn_type_single);
    for (int i = 0; i < dims; i++)
    {
        totalSizeInBytes *= sizes[i];
    }
    // create graph
    auto inputIndex1 = createPersistTensor(TensorUsage::INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           inputBuffer1,
                                           sizes,
                                           dims,
                                           syn_type_single);

    unsigned constSectionIndex = createConstSection();
    unsigned wTensorIndex      = createConstPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                inputBuffer2,
                                                sizes,
                                                dims,
                                                syn_type_single,
                                                nullptr,
                                                "weights",
                                                0,
                                                0,
                                                &constSectionIndex);

    auto outputsIndex =
        createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_single);

    synTensor weightTensor    = getTensorByIndex(wTensorIndex);

    addNodeToGraph("add_fwd_f32", TensorIndices {inputIndex1, wTensorIndex}, TensorIndices {outputsIndex});

    compileTopology();

    synRecipeHandle recipeHandle = getGraph(0).recipeHandle;

    synSectionId tensorSectionId;
    getTensorSectionId(recipeHandle, weightTensor, tensorSectionId);

    uint64_t section_size = 0;
    ASSERT_EQ(synSuccess, synRecipeSectionGetProp(recipeHandle, tensorSectionId, SECTION_SIZE, &section_size));

    ASSERT_EQ(totalSizeInBytes, section_size);

    runTopology();

    // validate results
    float* myInputBuffer1 = castHostInBuffer<float>(inputIndex1);
    float* myInputBuffer2 = castHostInBuffer<float>(wTensorIndex);
    float* outputBuffer   = castHostOutBuffer<float>(outputsIndex);

    for (uint64_t i = 0; i < getTensorElementCount(inputIndex1); ++i)
    {
        ASSERT_EQ(myInputBuffer1[i] + myInputBuffer2[i], outputBuffer[i]);
    }
}

TEST_F_GC(SynTrainingTestInfraTpcAdd, add_forward_with_both_tensors_in_const_sections)
{
    // create data
    unsigned dims    = 4;
    unsigned sizes[] = {1000, 1, 1, 64};
    float    inputBuffer1[1000 * 1 * 1 * 64];
    fillWithRandom(inputBuffer1, 1000 * 1 * 1 * 64);
    float inputBuffer2[1000 * 1 * 1 * 64];
    fillWithRandom(inputBuffer2, 1000 * 1 * 1 * 64);

    unsigned totalSizeInBytes = sizeof(syn_type_single);
    for (int i = 0; i < dims; i++)
    {
        totalSizeInBytes *= sizes[i];
    }
    // create graph

    unsigned constSectionIndex1 = createConstSection();
    unsigned inputIndex1        = createConstPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               inputBuffer1,
                                               sizes,
                                               dims,
                                               syn_type_single,
                                               nullptr,
                                               "input1",
                                               0,
                                               0,
                                               &constSectionIndex1);

    unsigned constSectionIndex2 = createConstSection();
    unsigned inputIndex2        = createConstPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               inputBuffer2,
                                               sizes,
                                               dims,
                                               syn_type_single,
                                               nullptr,
                                               "input2",
                                               0,
                                               0,
                                               &constSectionIndex2);

    auto outputsIndex =
        createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_single);

    synTensor inputTensor1    = getTensorByIndex(inputIndex1);
    synTensor inputTensor2    = getTensorByIndex(inputIndex2);

    addNodeToGraph("add_fwd_f32", TensorIndices {inputIndex1, inputIndex2}, TensorIndices {outputsIndex});

    compileTopology();

    synRecipeHandle recipeHandle = getGraph(0).recipeHandle;

    synSectionId tensor1SectionId;
    getTensorSectionId(recipeHandle, inputTensor1, tensor1SectionId);

    synSectionId tensor2SectionId;
    getTensorSectionId(recipeHandle, inputTensor2, tensor2SectionId);

    uint64_t section_size1 = 0, section_size2 = 0;
    ASSERT_EQ(synSuccess, synRecipeSectionGetProp(recipeHandle, tensor1SectionId, SECTION_SIZE, &section_size1));
    ASSERT_GT(section_size1, 0) << "const section size is zero!";
    ASSERT_EQ(totalSizeInBytes, section_size1);

    ASSERT_EQ(synSuccess, synRecipeSectionGetProp(recipeHandle, tensor2SectionId, SECTION_SIZE, &section_size2));
    ASSERT_GT(section_size2, 0) << "const section size is zero!";
    ASSERT_EQ(totalSizeInBytes, section_size2);

    runTopology();

    // validate results
    float* myInputBuffer1 = castHostInBuffer<float>(inputIndex1);
    float* myInputBuffer2 = castHostInBuffer<float>(inputIndex2);
    float* outputBuffer   = castHostOutBuffer<float>(outputsIndex);

    for (uint64_t i = 0; i < getTensorElementCount(inputIndex1); ++i)
    {
        ASSERT_EQ(myInputBuffer1[i] + myInputBuffer2[i], outputBuffer[i]);
    }
}

TEST_F_GC(SynTrainingTestInfraTpcAdd, add_forward_with_both_tensors_in_same_const_section)
{
    // create data
    unsigned dims    = 4;
    unsigned sizes[] = {1000, 1, 1, 64};
    float    inputBuffer1[1000 * 1 * 1 * 64];
    fillWithRandom(inputBuffer1, 1000 * 1 * 1 * 64);
    float inputBuffer2[1000 * 1 * 1 * 64];
    fillWithRandom(inputBuffer2, 1000 * 1 * 1 * 64);

    unsigned totalSizeInBytes = sizeof(syn_type_single);
    for (int i = 0; i < dims; i++)
    {
        totalSizeInBytes *= sizes[i];
    }
    // create graph

    unsigned constSectionIndex = createConstSection();
    unsigned inputIndex1       = createConstPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               inputBuffer1,
                                               sizes,
                                               dims,
                                               syn_type_single,
                                               nullptr,
                                               "input1",
                                               0,
                                               0,
                                               &constSectionIndex);

    unsigned inputIndex2 = createConstPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               inputBuffer2,
                                               sizes,
                                               dims,
                                               syn_type_single,
                                               nullptr,
                                               "input2",
                                               0,
                                               totalSizeInBytes,
                                               &constSectionIndex);

    auto outputsIndex =
        createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, syn_type_single);

    synTensor inputTensor1    = getTensorByIndex(inputIndex1);
    synTensor inputTensor2    = getTensorByIndex(inputIndex2);

    addNodeToGraph("add_fwd_f32", TensorIndices {inputIndex1, inputIndex2}, TensorIndices {outputsIndex});

    compileTopology();


    synRecipeHandle recipeHandle = getGraph(0).recipeHandle;

    synSectionId tensor1SectionId;
    getTensorSectionId(recipeHandle, inputTensor1, tensor1SectionId);

    synSectionId tensor2SectionId;
    getTensorSectionId(recipeHandle, inputTensor2, tensor2SectionId);

    ASSERT_EQ(tensor1SectionId, tensor2SectionId);

    uint64_t section_size = 0;
    ASSERT_EQ(synSuccess, synRecipeSectionGetProp(recipeHandle, tensor1SectionId, SECTION_SIZE, &section_size));
    ASSERT_GT(section_size, 0) << "const section size is zero!";
    ASSERT_EQ(totalSizeInBytes * 2, section_size);

    runTopology();

    // validate results
    float* myInputBuffer1 = castHostInBuffer<float>(inputIndex1);
    float* myInputBuffer2 = castHostInBuffer<float>(inputIndex2);
    float* outputBuffer   = castHostOutBuffer<float>(outputsIndex);

    for (uint64_t i = 0; i < getTensorElementCount(inputIndex1); ++i)
    {
        ASSERT_EQ(myInputBuffer1[i] + myInputBuffer2[i], outputBuffer[i]);
    }
}
