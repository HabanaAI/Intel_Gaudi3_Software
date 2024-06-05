#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

class SynGaudiTPCFuserOverlapTest : public SynTrainingTestInfra
{
};

TEST_F_GC(SynGaudiTPCFuserOverlapTest, PartialOverlapInputOutputTensors)
{
    // 3 TPC nodes having 2 persistent tensors that partial overlap
    // (same section ID overlap address range with some offset)
    // one tesnor is input and the other output
    // Those nodes should not be fused, check correctness

    unsigned tensorsizes[] = {10, 10, 10, 10};

    const auto offset     = 50 * sizeof(float);
    const auto memorySize = 10000 * sizeof(float) + offset * 2;

    // fills vector with floats.
    std::vector<float> floats;

    float x = 0.;
    std::generate_n(std::back_inserter(floats), getNumberOfElements(tensorsizes), [&]() mutable { return x += 0.1; });

    unsigned sectionIndex = createSection(memorySize);

    unsigned inputIndex = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_FROM_INITIALIZER,
                                              floats.data(),
                                              tensorsizes,
                                              4,
                                              syn_type_single,
                                              nullptr,
                                              nullptr,
                                              0,
                                              0,
                                              &sectionIndex);

    unsigned middleIndexAsOut = createPersistTensor(TensorUsage::OUTPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    floats.data(),
                                                    tensorsizes,
                                                    4,
                                                    syn_type_single);

    unsigned middleIndexAsIn = connectOutputTensorToInputTensor(middleIndexAsOut);

    unsigned middleIndexAsOut2 = createPersistTensor(TensorUsage::OUTPUT_TENSOR,
                                                     MEM_INIT_FROM_INITIALIZER,
                                                     floats.data(),
                                                     tensorsizes,
                                                     4,
                                                     syn_type_single);

    unsigned middleIndexAsIn2 = connectOutputTensorToInputTensor(middleIndexAsOut2);

    unsigned outputIndex = createPersistTensor(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               tensorsizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               nullptr,
                                               0,
                                               offset,
                                               &sectionIndex);

    addNodeToGraph("relu_fwd_f32", TensorIndices {inputIndex}, TensorIndices {middleIndexAsOut});
    addNodeToGraph("relu_fwd_f32", TensorIndices {middleIndexAsIn}, TensorIndices {middleIndexAsOut2});
    addNodeToGraph("relu_fwd_f32", TensorIndices {middleIndexAsIn2}, TensorIndices {outputIndex});

    compileAndRun();
    for (uint64_t i = 0; i < getTensorElementCount(outputIndex); ++i)
    {
        ASSERT_FLOAT_EQ(castHostOutBuffer<float>(outputIndex)[i], floats[i]);
    }
}

TEST_F_GC(SynGaudiTPCFuserOverlapTest, PartialOverlapTwoInputsTensors)
{
    // 3 TPC nodes having 2 persistent tensors that partial overlap
    // (same section ID overlap address range with some offset)
    // both tensors are inputs of different nodes
    // Those nodes should not be fused, check correctness

    unsigned tensorsizes[] = {10, 10, 10, 10};

    const auto offset     = 50 * sizeof(float);
    const auto memorySize = 10000 * sizeof(float) + offset * 2;

    // fills vector with floats.
    std::vector<float> floats;

    float x = 0.;
    std::generate_n(std::back_inserter(floats), getNumberOfElements(tensorsizes), [&]() mutable { return x += 0.1; });

    unsigned sectionIndex = createSection(memorySize);

    unsigned inputIndex = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_FROM_INITIALIZER,
                                              floats.data(),
                                              tensorsizes,
                                              4,
                                              syn_type_single,
                                              nullptr,
                                              nullptr,
                                              0,
                                              0,
                                              &sectionIndex);

    unsigned middleIndexAsOut = createPersistTensor(TensorUsage::OUTPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    floats.data(),
                                                    tensorsizes,
                                                    4,
                                                    syn_type_single);

    unsigned middleIndexAsIn = connectOutputTensorToInputTensor(middleIndexAsOut);

    unsigned middleIndexAsOut2 = createPersistTensor(TensorUsage::OUTPUT_TENSOR,
                                                     MEM_INIT_FROM_INITIALIZER,
                                                     floats.data(),
                                                     tensorsizes,
                                                     4,
                                                     syn_type_single,
                                                     nullptr,
                                                     nullptr,
                                                     0,
                                                     offset,
                                                     &sectionIndex);

    unsigned middleIndexAsIn2 = connectOutputTensorToInputTensor(middleIndexAsOut2);

    unsigned outputIndex = createPersistTensor(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               tensorsizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               nullptr,
                                               0,
                                               0);

    addNodeToGraph("relu_fwd_f32", TensorIndices {inputIndex}, TensorIndices {middleIndexAsOut});
    addNodeToGraph("relu_fwd_f32", TensorIndices {middleIndexAsIn}, TensorIndices {middleIndexAsOut2});
    addNodeToGraph("relu_fwd_f32", TensorIndices {middleIndexAsIn2}, TensorIndices {outputIndex});

    compileAndRun();
    for (uint64_t i = 0; i < getTensorElementCount(outputIndex); ++i)
    {
        ASSERT_FLOAT_EQ(castHostOutBuffer<float>(outputIndex)[i], floats[i]);
    }
}

TEST_F_GC(SynGaudiTPCFuserOverlapTest, PartialOverlapTwoOutputTensors)
{
    // 3 TPC nodes having 2 persistent tensors that partial overlap
    // (same section ID overlap address range with some offset)
    // both tensors are outputs of different nodes
    // Those nodes should not be fused, check correctness

    unsigned tensorsizes[] = {10, 10, 10, 10};

    const auto offset     = 50 * sizeof(float);
    const auto memorySize = 10000 * sizeof(float) + offset * 2;

    // fills vector with floats.
    std::vector<float> floats;

    float x = 0.;
    std::generate_n(std::back_inserter(floats), getNumberOfElements(tensorsizes), [&]() mutable { return x += 0.1; });

    unsigned sectionIndex  = createSection(memorySize);
    unsigned sectionIndex2 = createSection(memorySize);

    unsigned inputIndex =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, floats.data(), tensorsizes, 4, syn_type_single);

    unsigned middleIndexAsOut = createPersistTensor(TensorUsage::OUTPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    floats.data(),
                                                    tensorsizes,
                                                    4,
                                                    syn_type_single,
                                                    nullptr,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    &sectionIndex2);

    unsigned middleIndexAsIn = connectOutputTensorToInputTensor(middleIndexAsOut);

    unsigned middleIndexAsOut2 = createPersistTensor(TensorUsage::OUTPUT_TENSOR,
                                                     MEM_INIT_FROM_INITIALIZER,
                                                     floats.data(),
                                                     tensorsizes,
                                                     4,
                                                     syn_type_single,
                                                     nullptr,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     &sectionIndex);

    unsigned middleIndexAsIn2 = connectOutputTensorToInputTensor(middleIndexAsOut2);

    unsigned outputIndex = createPersistTensor(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               tensorsizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               nullptr,
                                               0,
                                               offset,
                                               &sectionIndex2);

    addNodeToGraph("relu_fwd_f32", TensorIndices {inputIndex}, TensorIndices {middleIndexAsOut});
    addNodeToGraph("relu_fwd_f32", TensorIndices {middleIndexAsIn}, TensorIndices {middleIndexAsOut2});
    addNodeToGraph("relu_fwd_f32", TensorIndices {middleIndexAsIn2}, TensorIndices {outputIndex});

    compileAndRun();
    for (uint64_t i = 0; i < getTensorElementCount(outputIndex); ++i)
    {
        ASSERT_FLOAT_EQ(castHostOutBuffer<float>(outputIndex)[i], floats[i]);
    }
}
