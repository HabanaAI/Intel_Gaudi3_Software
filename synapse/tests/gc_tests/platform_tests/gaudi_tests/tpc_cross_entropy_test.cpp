#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTpcTestInfra, cross_entropy_forward_L2)
{
    typedef float       kernelFlavor;

    const synDataType   synDataTypeFlavor = syn_type_single;

    const unsigned widthSize    = 3;
    const unsigned heightSize   = 4;

    const unsigned crossEntropySize = 1;

    const unsigned dataDims         = 2;
    const unsigned modelParamsDims  = 1;
    const unsigned crossEntropyDims = 1;

    unsigned dataDimSizes[dataDims]                 = { widthSize, heightSize };
    unsigned modelParamsDimSizes[modelParamsDims]   = { heightSize };
    unsigned crossEntropyDimSizes[crossEntropyDims] = { crossEntropySize };

    const unsigned numOfDataElements       = widthSize * heightSize;
    const unsigned numModelParamElements   = heightSize;
    const unsigned numCrossEntropyElements = crossEntropySize;
    const unsigned modelParamsBufferSize  = numModelParamElements * sizeof(int32_t);

    kernelFlavor inputBuffer[numOfDataElements] = { 1,   2,   3,
                                                    4,   5,   6,
                                                    7,   8,   9,
                                                    10,  11,  12 };

    int32_t labelsBuffer[numModelParamElements] = { 1,   2,   0,   2 };

    float labelsBufFloat[numModelParamElements];
    memcpy(labelsBufFloat, labelsBuffer, modelParamsBufferSize);

    // Input
    unsigned inputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputBuffer, dataDimSizes, dataDims,
                                                synDataTypeFlavor, nullptr, "inputTensor");
    unsigned labelsTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, labelsBufFloat, modelParamsDimSizes, modelParamsDims,
                                                syn_type_int32, nullptr, "labelsTensor");
    // Output
    unsigned crossEntropyTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, crossEntropyDimSizes, crossEntropyDims,
                                                synDataTypeFlavor, nullptr, "crossEntropyTensor");
    unsigned logSoftMaxTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDimSizes, dataDims,
                                                synDataTypeFlavor, nullptr, "logSoftMaxTensor");

    ns_SoftmaxCrossEntropy::Params userParams;
    userParams.mode = CROSS_ENTROPY_MODE_MEAN;
    userParams.batchSize = 4;

    char nodeGuid[] = "softmax_cross_entropy_fwd_f32";
    addNodeToGraph(nodeGuid,
                   {inputTensor, labelsTensor},
                   {crossEntropyTensor, logSoftMaxTensor},
                   &userParams, sizeof(ns_SoftmaxCrossEntropy::Params));

    compileAndRun();

    kernelFlavor crossEntropyRef = 1.157605964;
    float* pCrossEntropyVal = (float*)m_hostBuffers[crossEntropyTensor];
    validateResult(&crossEntropyRef, pCrossEntropyVal, numCrossEntropyElements);

    kernelFlavor logSoftMaxRef[numOfDataElements] =
    {
        -2.407605964, -1.407605964, -0.407605964,
        -2.407605964, -1.407605964, -0.407605964,
        -2.407605964, -1.407605964, -0.407605964,
        -2.407605964, -1.407605964, -0.407605964
    };

    float* pOutputBuffer = (float*)m_hostBuffers[logSoftMaxTensor];
    validateResult(logSoftMaxRef, pOutputBuffer, numOfDataElements);
}

TEST_F_GC(SynTrainingTpcTestInfra, cross_entropy_backward_L2)
{
    typedef float       kernelFlavor;

    const synDataType   synDataTypeFlavor = syn_type_single;

    const unsigned batchSize    = 5;
    const unsigned classSize    = 3;

    const unsigned dataDims        = 2;
    const unsigned modelParamsDims = 1;

    unsigned dataDimSizes[dataDims]               = {classSize, batchSize};
    unsigned modelParamsDimSizes[modelParamsDims] = {batchSize};

    const unsigned numOfDataElements     = classSize * batchSize;
    const unsigned numModelParamElements = batchSize;
    const unsigned modelParamsBufferSize = numModelParamElements * sizeof(int32_t);

    kernelFlavor inputBuffer[numOfDataElements] = {1, 3, 4,
                                                   2, 4, 4,
                                                   3, 5, 5,
                                                   3, 2, 3,
                                                   5, 1, 1};

    int32_t labelsBuffer[numModelParamElements] = {2,   1,   1,   0,   2};

    float labelsBufferFloat[numModelParamElements];
    memcpy(labelsBufferFloat, labelsBuffer, modelParamsBufferSize);

    // Input
    unsigned inputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputBuffer, dataDimSizes, dataDims,
                                                synDataTypeFlavor, nullptr, "inputTensor");
    unsigned labelsTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, labelsBufferFloat, modelParamsDimSizes, modelParamsDims,
                                                syn_type_int32, nullptr, "labelsTensor");
    // Output
    unsigned outputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDimSizes, dataDims,
                                                synDataTypeFlavor, nullptr, "outputTensor");
    ns_SoftmaxCrossEntropy::Params userParams;
    userParams.mode = CROSS_ENTROPY_MODE_MEAN;
    userParams.batchSize = batchSize;

    char nodeGuid[] = "softmax_cross_entropy_bwd_f32";
    addNodeToGraph(nodeGuid,
                   {inputTensor, labelsTensor},
                   {outputTensor},
                   &userParams, sizeof(ns_SoftmaxCrossEntropy::Params));

    compileAndRun();

    kernelFlavor outputRef [numOfDataElements] = {0.543656366,  4.017107385, 10.71963001,
                                                  1.47781122,  10.71963001,  10.91963001,
                                                  4.017107385, 29.48263182,  29.68263182,
                                                  3.817107385,  1.47781122,   4.017107385,
                                                 29.68263182,   0.543656366,  0.343656366};

    float* pOutputBuffer = (float*)m_hostBuffers[outputTensor];
    validateResult(outputRef, pOutputBuffer, numOfDataElements);
}
