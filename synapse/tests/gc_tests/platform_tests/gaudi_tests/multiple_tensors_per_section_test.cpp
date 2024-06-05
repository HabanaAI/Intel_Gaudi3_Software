#include <graph_compiler/habana_nodes/node_factory.h>
#include "infra/gc_synapse_test.h"
#include "synapse_api.h"
#include "gc_gaudi_test_infra.h"

using namespace std;

// tests memcpy with overapping tensors
TEST_F_GC(SynTrainingTestInfra, TensorOffset)
{
    unsigned tensorsizes[] = {1, 1, 1, 1};


    const auto  memorySize = 100;
    const auto  offset     = 50;

    float data = 15;
    unsigned sectionIndex = createSection(memorySize);
    unsigned in = createTensors(1, INPUT_TENSOR, true, nullptr, MEM_INIT_FROM_INITIALIZER, &data, tensorsizes, 4,syn_type_single, nullptr, 0, 0, &sectionIndex)[0];
    unsigned out = createTensors(1, OUTPUT_TENSOR, true, nullptr, MEM_INIT_ALL_ZERO, nullptr, tensorsizes, 4,syn_type_single, nullptr, 0, offset, &sectionIndex)[0];
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, TensorIndices{in}, TensorIndices{out});
    compileAndRun();
    EXPECT_EQ(m_persistentSections.size(), 1);
    ASSERT_EQ(*reinterpret_cast<float*>(m_hostBuffers[out]), data);
}

static float refRelu(float i)
{
    if (i > 0) return i;
    return 0;
}

// tests relu+overlap
// In0 -> relu -> Out0
// Out0 Out1 overlap
TEST_F_GC(SynTrainingTestInfra, OverlappingTensors_L2)
{
    unsigned tensorsizes[] = {5, 5, 5, 5};

    const auto memorySize = 3000;
    const auto elementsOffset= 50;
    const auto offset     = elementsOffset * sizeof(float);

    // fills vector with floats.
    std::vector<float> floats(getNumberOfElements(tensorsizes), 1.1);

    unsigned in1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, tensorsizes, 4, syn_type_single);
    unsigned sectionIndex = createSection(memorySize);

    unsigned out1 = createPersistTensor(OUTPUT_TENSOR,
                        MEM_INIT_ALL_ZERO,
                        nullptr,
                        tensorsizes,
                        4,
                        syn_type_single,
                        nullptr,
                        nullptr,
                        0,
                        0,
                        &sectionIndex);
    unsigned out2 = createPersistTensor(OUTPUT_TENSOR,
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

    addNodeToGraph("relu_fwd_f32", TensorIndices{in1}, TensorIndices{out1});
    compileAndRun();
    EXPECT_EQ(m_persistentSections.size(), 2);
    for (uint64_t i = 0; i < getNumberOfElements(tensorsizes) - elementsOffset; ++i)
    {
        EXPECT_FLOAT_EQ(castHostOutBuffer<float>(out2)[i], refRelu(castHostInBuffer<float>(in1)[i + elementsOffset])) << i;
    }
}

// tests relu
// In0 -> relu -> Out0
// In0 Out0 overlap
TEST_F_GC(SynTrainingTestInfra, OverlappingInputOutputTensors_L2)
{
    unsigned tensorsizes[] = {5, 5, 5, 5};

    const auto memorySize = 3000;
    const auto offset     = 50 * sizeof(float);

    // fills vector with floats.
    std::vector<float> floats(getNumberOfElements(tensorsizes), 1.1);

    unsigned sectionIndex = createSection(memorySize);
    // cannot change to random init since output overrides input.
    unsigned inputIndex   = createPersistTensor(INPUT_TENSOR,
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
    unsigned outputIndex  = createPersistTensor(OUTPUT_TENSOR,
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
    // using relu and not memcpy since we can optimize memcpy out.
    addNodeToGraph("relu_fwd_f32", TensorIndices{inputIndex}, TensorIndices{outputIndex});

    compileAndRun();

    for (uint64_t i = 0; i < getNumberOfElements(tensorsizes); ++i)
    {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(m_hostBuffers[outputIndex])[i], 1.1);
    }
}


// tests relu+overlap
// In0 -> relu -> Mid0 -> Relu -> Out0
// Out0 In0 overlap
TEST_F_GC(SynTrainingTestInfra, OverlappingInputOutputTensors2_L2)
{
    unsigned tensorsizes[] = {10, 10, 10, 10};

    const auto offset     = 50 * sizeof(float);
    const auto memorySize = 10000 * sizeof(float) + offset * 2;

    // fills vector with floats.
    std::vector<float> floats;

    float x = 0.;
    std::generate_n(std::back_inserter(floats), getNumberOfElements(tensorsizes), [&]() mutable { return x+= 0.1; });

    unsigned sectionIndex     = createSection(memorySize);
    unsigned inputIndex       = createPersistTensor(INPUT_TENSOR,
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
    unsigned middleIndexAsOut = createPersistTensor(
        TensorUsage::OUTPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, floats.data(), tensorsizes, 4, syn_type_single);
    unsigned middleIndexAsIn = connectOutputTensorToInputTensor(middleIndexAsOut);
    unsigned outputIndex     = createPersistTensor(OUTPUT_TENSOR,
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
    // using relu and not memcpy since we can optimize memcpy out.
    addNodeToGraph("relu_fwd_f32", TensorIndices{inputIndex}, TensorIndices{middleIndexAsOut});
    addNodeToGraph("relu_fwd_f32", TensorIndices{middleIndexAsIn}, TensorIndices{outputIndex});
    compileAndRun();
    EXPECT_EQ(m_persistentSections.size(), 2);
    for (uint64_t i = 0; i < getTensorElementCount(outputIndex); ++i)
    {
        ASSERT_FLOAT_EQ(castHostOutBuffer<float>(outputIndex)[i], floats[i]);
    }
}

// tests relu+overlap
// In0 -> relu -> Out1
// In1 -> relu -> Out0
// In0 In1 overlap
// Out0 Out1 overlap
TEST_F_GC(SynTrainingTestInfra, OverlappingInputTensors2_L2)
{
    unsigned tensorsizes[] = {5, 5, 5, 5};

    const auto memorySize = 3000;
    const auto offset     = 50 * sizeof(float);

    // fills vector with floats.
    std::vector<float> floats;
    float x = 0.;
    std::generate_n(std::back_inserter(floats), getNumberOfElements(tensorsizes), [&]() mutable {
        if(x> 1) x = 0;
        return x += 0.1;
    });
    unsigned sectionIndex     = createSection(memorySize);
    unsigned sectionIndex2    = createSection(memorySize);
    unsigned inputIndex       = createPersistTensor(INPUT_TENSOR,
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

    unsigned inputIndex2      = createPersistTensor(INPUT_TENSOR,
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
    unsigned outputIndex     = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           tensorsizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           nullptr,
                                           0,
                                           0,
                                           &sectionIndex2);
    unsigned outputIndex2     = createPersistTensor(OUTPUT_TENSOR,
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
    // using relu and not memcpy since we can optimize memcpy out.
    synNodeId node1Id;
    synNodeId node2Id;
    addNodeToGraph("relu_fwd_f32", TensorIndices{inputIndex}, TensorIndices{outputIndex2}, nullptr, 0, nullptr, 0, &node1Id);
    addNodeToGraph("relu_fwd_f32", TensorIndices{inputIndex2}, TensorIndices{outputIndex}, nullptr, 0, nullptr, 0, &node2Id);

    uint64_t blockingNodesId[] = {node1Id};
    uint64_t blockedNodesId[]  = {node2Id};
    setNodeDependency(blockingNodesId, blockedNodesId, 1, 1);

    compileAndRun();
    EXPECT_EQ(m_persistentSections.size(), 2);

    ASSERT_FLOAT_EQ(*(reinterpret_cast<float*>(m_hostBuffers[outputIndex])),*(reinterpret_cast<float*>(m_hostBuffers[outputIndex2])));
}


// tests relu+overlap
// fwdIn ->  relu_fwd -> fwdOut -> bwdIn -> relu_bwd -> bwdOut
// fwdIn and bwdOut overlap
TEST_F_GC(SynTrainingTestInfra, relu_forward_and_backward_overlapping)
{
    // Graph will have two nodes:  [relu_fwd]->[relu_bwd]

    unsigned sectionIndex   = createSection(getMemorySize(getDefaultSizes(), asSynType<float>()));

    unsigned fwdIn  = createPersistTensor(
        INPUT_TENSOR,
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        nullptr,
        4,
        syn_type_single,
        nullptr,
        nullptr,
        0,
        0,
        &sectionIndex);
    unsigned fwdOut = createTensor(OUTPUT_TENSOR);
    addNodeToGraph("relu_fwd_f32", {fwdIn}, {fwdOut});

    unsigned bwdIn1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE); // input data (grad) on which relu backward is computed
    unsigned bwdIn2 = connectOutputTensorToInputTensor(fwdOut);
    unsigned bwdOut = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          nullptr,
                                          4,
                                          syn_type_single,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0,
                                          &sectionIndex
        );
    addNodeToGraph("relu_bwd_f32", {bwdIn1, bwdIn2}, {bwdOut});

    compileTopology();
    runTopology();

    float* pFwdInput  = (float*)m_hostBuffers[fwdIn];
    float* pBwdInput  = (float*)m_hostBuffers[bwdIn1];
    float* pBwdOutput = (float*)m_hostBuffers[bwdOut];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        float expectedResult = (*pFwdInput > 0) ? *pBwdInput : 0;
        ASSERT_EQ(expectedResult, *pBwdOutput) << "Mismatch for at index " << i
                                               << " Expected:"             << expectedResult
                                               << " BwdOutput: "           << *pBwdOutput
                                               << " FwdInput: "            << *pFwdInput
                                               << " BwdInput "             << *pBwdInput;
        pFwdInput++;
        pBwdInput++;
        pBwdOutput++;
    }
}

TEST_F_GC(SynGaudiTestInfra, transpose_with_persitent_overlap_2dim)  // [SW-65351]
{
    unsigned       inSize[]      = {128, 256};
    unsigned       outSize[]     = {256, 128};
    const unsigned numOfElements = inSize[0] * inSize[1];

    unsigned sectionIndex  = createSection(numOfElements * sizeof(float));
    unsigned sectionIndex2 = sectionIndex;  // createSection(numOfElements * sizeof(float));

    unsigned tensorIn  = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inSize,
                                            2,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            &sectionIndex);
    unsigned tensorOut = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             outSize,
                                             2,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             &sectionIndex2);

    synTransposeParams params;
    params.permutation[0] = (TransposePermutationDim)1;
    params.permutation[1] = (TransposePermutationDim)0;
    params.tensorDim      = 2;
    addNodeToGraph(NodeFactory::transposeNodeTypeName, {tensorIn}, {tensorOut}, (void*)&params, sizeof(params));
    compileAndRun();

    // validate
    float* pOutputBuffer = (float*)m_hostBuffers[tensorOut];
    float* pInputBuffer  = (float*)m_hostBuffers[tensorIn];

    for (int i = 0; i < outSize[1]; ++i)
    {
        for (int j = 0; j < outSize[0]; ++j)
        {
            ASSERT_FLOAT_EQ(pOutputBuffer[i * outSize[0] + j], pInputBuffer[j * inSize[0] + i]);
        }
    }
}
