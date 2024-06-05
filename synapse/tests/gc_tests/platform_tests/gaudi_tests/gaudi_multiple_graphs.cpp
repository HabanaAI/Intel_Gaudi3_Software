#include <thread>
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"

class SynTrainingMultipleGraphsTests : public SynTrainingTestInfra
{
public:
    SynTrainingMultipleGraphsTests() = default;

    void testCompileTopology(unsigned graphIndex);

    void compileParallel(unsigned numOfGraphs);
};

void SynTrainingMultipleGraphsTests::testCompileTopology(unsigned graphIndex)
{
    std::stringstream ss;
    ss << GetTestFileName() << "_" << graphIndex;
    std::string topologyName = ss.str();
    compileTopology(topologyName, graphIndex);
}

void SynTrainingMultipleGraphsTests::compileParallel(const unsigned numOfGraphs)
{
    std::vector<std::thread> threadVector;
    std::vector<std::string> graphNames;
    for (unsigned graphIndex = 0; graphIndex < numOfGraphs; graphIndex++)
    {
        //graphNames.push_back("gaudi_relu_forward" + std::to_string(graphIndex));

        // Create a new thread to trigger compilation
        LOG_TRACE(SYN_TEST, "Compilation thread started (graph {})", graphIndex);
        std::thread th(&SynTrainingMultipleGraphsTests::testCompileTopology, this, graphIndex);
        threadVector.push_back(std::move(th));
    }

    for (auto& th : threadVector)
    {
        th.join();
    }
}

TEST_F_GC(SynTrainingMultipleGraphsTests, relu_compile_parallel)
{
    const unsigned numOfGraphs = 5;

    for (unsigned graphIndex = 1; graphIndex < numOfGraphs; graphIndex++)
    {
        // The first graph already exists
        createGraph();
    }

    for (unsigned graphIndex = 0; graphIndex < numOfGraphs; graphIndex++)
    {
        auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                            nullptr, DEFAULT_SIZES, syn_type_single, nullptr, nullptr, graphIndex);
        auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, nullptr, DEFAULT_SIZES,
                            syn_type_single, nullptr, nullptr, graphIndex);

        // We pass the indices of the input and of the output tensors, to associate them with the node (in the creation
        // of the tensors we only associate persist tensors with the graph).
        addNodeToGraph("relu_fwd_f32", {in}, {out}, nullptr, 0, nullptr, graphIndex);
    }

    compileParallel(numOfGraphs);
}

TEST_F_GC(SynTrainingMultipleGraphsTests, relu_compile_parallel_and_run)
{
    const unsigned numOfGraphs = 5;

    for (unsigned graphIndex = 1; graphIndex < numOfGraphs; graphIndex++)
    {
        // The first graph already exists
        createGraph();
    }

    using GraphIndexToInOut = std::map<unsigned, std::pair<unsigned, unsigned>>;

    GraphIndexToInOut graphIndicesMap;

    for (unsigned graphIndex = 0; graphIndex < numOfGraphs; graphIndex++)
    {
        auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                            nullptr, DEFAULT_SIZES, syn_type_single, nullptr, nullptr, graphIndex);
        auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, nullptr, DEFAULT_SIZES,
                            syn_type_single, nullptr, nullptr, graphIndex);

        graphIndicesMap[graphIndex] = std::make_pair(in, out);
        // We pass the indices of the input and of the output tensors, to associate them with the node (in the creation
        // of the tensors we only associate persist tensors with the graph).
        addNodeToGraph("relu_fwd_f32", {in}, {out}, nullptr, 0, nullptr, graphIndex);
    }

    compileParallel(numOfGraphs);

    for (unsigned graphIndex = 0; graphIndex < numOfGraphs; graphIndex++)
    {
        runTopology(graphIndex);

        float *pInputBuffer = (float *) m_hostBuffers[graphIndicesMap[graphIndex].first];
        float *pOutputBuffer = (float *) m_hostBuffers[graphIndicesMap[graphIndex].second];

        for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
        {
            float expectedResult = std::max((float) 0.0, *pInputBuffer);
            ASSERT_EQ(expectedResult, *pOutputBuffer) << "Mismatch for at index " << i
                                                      << " Expected:" << expectedResult
                                                      << " Result: " << *pOutputBuffer
                                                      << " operand " << *pInputBuffer;
            pInputBuffer++;
            pOutputBuffer++;
        }
    }
}

TEST_F_GC(SynTrainingMultipleGraphsTests, relu_compile_and_run_L2)
{
    const unsigned numOfGraphs = 3;

    for (unsigned graphIndex = 1; graphIndex < numOfGraphs; graphIndex++)
    {
        // The first graph already exists
        createGraph();
    }

    unsigned inputTensorSizes[]  = {16, 32, 32, 16};
    unsigned outputTensorSizes[] = {16, 32, 32, 16};

    for (unsigned graphIndex = 0; graphIndex < numOfGraphs; graphIndex++)
    {
        auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                            inputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr, nullptr, graphIndex);
        auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputTensorSizes,
                            DEFAULT_SIZES, syn_type_single, nullptr, nullptr, graphIndex);

        // We pass the indices of the input and of the output tensors, to associate them with the node (in the creation
        // of the tensors we only associate persist tensors with the graph).
        addNodeToGraph("relu_fwd_f32", {in}, {out}, nullptr, 0, nullptr, graphIndex);

        compileTopology("gaudi_multiple_graphs_relu_g"+std::to_string(graphIndex), graphIndex);

        runTopology(graphIndex);

        float *pInputBuffer = (float *) m_hostBuffers[in];
        float *pOutputBuffer = (float *) m_hostBuffers[out];

        for (uint64_t i = 0; i < getNumberOfElements(outputTensorSizes); i++)
        {
            float expectedResult = std::max((float) 0.0, *pInputBuffer);
            ASSERT_EQ(expectedResult, *pOutputBuffer) << "Mismatch for at index " << i
                                                      << " Expected:" << expectedResult
                                                      << " Result: " << *pOutputBuffer
                                                      << " operand " << *pInputBuffer << " in graph:" << graphIndex;
            pInputBuffer++;
            pOutputBuffer++;
        }
    }
}

TEST_F_GC(SynTrainingMultipleGraphsTests, add_input_tensor_to_invalid_graph)
{
    // The first graph is auto-generated
    createGraph();

    unsigned validGraphIndex = 0;
    unsigned invalidGraphIndex = 1;

    unsigned inputTensorSizes[]  = {16, 32, 32, 16};
    unsigned outputTensorSizes[] = {16, 32, 32, 16};

    unsigned firstTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                    inputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                    nullptr, validGraphIndex);

    unsigned secondTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                     inputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                     nullptr, invalidGraphIndex);

    unsigned thirdTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                    outputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                     nullptr, validGraphIndex);


    std::vector<synTensor> inTensors;
    std::vector<synTensor> outTensors;

    inTensors.push_back(m_tensors[firstTensorIndex]);
    inTensors.push_back(m_tensors[secondTensorIndex]);
    outTensors.push_back(m_tensors[thirdTensorIndex]);


    synGraphHandle graphHandle = m_graphs[validGraphIndex].graphHandle;

    // Tensor2 is attached to the second graph
    ASSERT_EQ(synInvalidArgument, synNodeCreate(graphHandle, inTensors.data(), outTensors.data(), inTensors.size(),
                                                outTensors.size(), nullptr, 0, "relu_fwd_f32",
                                                nullptr, nullptr, nullptr));
}

TEST_F_GC(SynTrainingMultipleGraphsTests, add_output_tensor_to_invalid_graph)
{
    // The first graph is auto-generated
    createGraph();

    unsigned validGraphIndex = 0;
    unsigned invalidGraphIndex = 1;

    unsigned inputTensorSizes[]  = {16, 32, 32, 16};
    unsigned outputTensorSizes[] = {16, 32, 32, 16};

    unsigned firstTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                    inputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                    nullptr, validGraphIndex);

    unsigned secondTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                     inputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                     nullptr, validGraphIndex);

    unsigned thirdTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                    outputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                    nullptr, invalidGraphIndex);


    std::vector<synTensor> inTensors;
    std::vector<synTensor> outTensors;

    inTensors.push_back(m_tensors[firstTensorIndex]);
    inTensors.push_back(m_tensors[secondTensorIndex]);
    outTensors.push_back(m_tensors[thirdTensorIndex]);


    synGraphHandle graphHandle = m_graphs[validGraphIndex].graphHandle;

    // Tensor2 is attached to the second graph
    ASSERT_EQ(synInvalidArgument, synNodeCreate(graphHandle, inTensors.data(), outTensors.data(), inTensors.size(),
                                                outTensors.size(), nullptr, 0, "relu_fwd_f32",
                                                nullptr, nullptr, nullptr));
}

TEST_F_GC(SynTrainingMultipleGraphsTests, add_const_input_tensor_to_invalid_graph)
{
    // The first graph is auto-generated
    createGraph();

    unsigned validGraphIndex = 0;
    unsigned invalidGraphIndex = 1;

    unsigned inputTensorSizes[]  = {16, 32, 32, 16};
    unsigned outputTensorSizes[] = {16, 32, 32, 16};

    unsigned firstTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                    inputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                    nullptr, validGraphIndex);

    unsigned secondTensorIndex = createConstTensor(MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                     inputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                     nullptr, invalidGraphIndex);

    unsigned thirdTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                    outputTensorSizes, DEFAULT_SIZES, syn_type_single, nullptr,
                                                    nullptr, validGraphIndex);


    std::vector<synTensor> inTensors;
    std::vector<synTensor> outTensors;

    inTensors.push_back(m_tensors[firstTensorIndex]);
    inTensors.push_back(m_tensors[secondTensorIndex]);
    outTensors.push_back(m_tensors[thirdTensorIndex]);


    synGraphHandle graphHandle = m_graphs[validGraphIndex].graphHandle;

    // Const tensors aren't allocated on a section attached to a specific graph
    // The fact that input2 initialized with the invalid graph matters because it is attached.
    ASSERT_NE(synSuccess,
              synNodeCreate(graphHandle,
                            inTensors.data(),
                            outTensors.data(),
                            inTensors.size(),
                            outTensors.size(),
                            nullptr,
                            0,
                            "relu_fwd_f32",
                            nullptr,
                            nullptr,
                            nullptr));
}
