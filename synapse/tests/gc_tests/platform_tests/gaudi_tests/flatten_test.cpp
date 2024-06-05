#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, test_flatten)
{
    const unsigned FCD = 4;
    const unsigned WIDTH = 4;
    const unsigned HEIGHT = 4;
    const unsigned BATCH = 1;

    unsigned int flat_input_dimensions[] = {FCD, WIDTH, HEIGHT, BATCH};
    unsigned int flat_output_dimensions[] = {FCD * WIDTH, HEIGHT};

    const unsigned int flat_input_dim_num = 4;
    const unsigned int flat_out_dim_num = 2;

    const unsigned inputSize = FCD * WIDTH * HEIGHT * BATCH;
    float *inputArray = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        inputArray[i] = (float) i;
    }

    unsigned inputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputArray, flat_input_dimensions,
                                                flat_input_dim_num, syn_type_single, nullptr, "inputTensor");
    unsigned flatOutputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, flat_output_dimensions,
                                                     flat_out_dim_num, syn_type_single, nullptr, "flatOutputTensor");

    // Flatten setting
    synFlattenParams    flattenAttr;
    flattenAttr.axis    = 1;

    addNodeToGraph(NodeFactory::flattenNodeTypeName, {inputTensor}, {flatOutputTensor}, (void *) &flattenAttr, sizeof(synFlattenParams));

    compileAndRun();

    float* pFlatOutputBuffer = (float*)m_hostBuffers[flatOutputTensor];

    // validate the output
    for (unsigned i = 0; i < inputSize; i++)
    {
        ASSERT_EQ(inputArray[i], pFlatOutputBuffer[i]) << "Mismatch for at index " << i
                                                       << " Expected:"             << inputArray[i]
                                                       << " Result: "              << pFlatOutputBuffer[i]
                                                       << " operand "              << inputArray[i];
    }
    delete[] inputArray;
}

TEST_F_GC(SynTrainingTestInfra, relu_forward_with_flatten)
{
    const unsigned FCD = 4;
    const unsigned WIDTH = 4;
    const unsigned HEIGHT = 4;
    const unsigned BATCH = 1;

    unsigned int relu_input_dimensions[] = {FCD, WIDTH, HEIGHT, BATCH};
    unsigned int flat_output_dimensions[] = {FCD * WIDTH, HEIGHT};

    const unsigned int relu_dim_num = 4;
    const unsigned int flat_dim_num = 2;

    const unsigned inputSize = FCD * WIDTH * HEIGHT * BATCH;
    float *inputArray = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        inputArray[i] = (float) i;
    }

    unsigned inputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputArray, relu_input_dimensions,
                                                relu_dim_num, syn_type_single, nullptr, "inputTensor");
    unsigned reluOutputTensor  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, relu_input_dimensions,
                                              relu_dim_num, syn_type_single, nullptr);
    unsigned flattenIn = connectOutputTensorToInputTensor(reluOutputTensor);
    unsigned flatOutputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, flat_output_dimensions,
                                                     flat_dim_num, syn_type_single, nullptr, "flattenOutputTensor");

    addNodeToGraph("relu_fwd_f32", {inputTensor}, {reluOutputTensor});

    // Flatten setting
    synFlattenParams    flattenAttr;
    flattenAttr.axis    = 1;

    addNodeToGraph(NodeFactory::flattenNodeTypeName, {flattenIn}, {flatOutputTensor}, (void *) &flattenAttr, sizeof(synFlattenParams));

    compileAndRun();

    float* pFlatOutputBuffer = (float*)m_hostBuffers[flatOutputTensor];

    // validate flatten output
    for (unsigned i = 0; i < inputSize; i++)
    {
        float expectedResult = std::max((float)0.0, inputArray[i]);
        ASSERT_EQ(expectedResult, pFlatOutputBuffer[i]) << "Mismatch for at index " << i
                                                        << " Expected:"             << expectedResult
                                                        << " Result: "              << pFlatOutputBuffer[i]
                                                        << " operand "              << inputArray[i];
    }

    delete[] inputArray;
}
