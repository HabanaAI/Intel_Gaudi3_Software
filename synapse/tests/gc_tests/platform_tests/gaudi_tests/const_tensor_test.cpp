#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "scoped_configuration_change.h"

TEST_F_GC(SynGaudiTestInfra, const_tensor_single_input)
{
    const unsigned dims = 4;
    unsigned ifmDimSizes[] = {2, 3, 1, 1};
    unsigned ofmDimSizes[] = {1, 3, 1, 1};

    const float ifmBuffer[] = {-1.0, -1.0,
                               1.0,  3.0,
                               4.0,  5.0};

    const float ofmRefBuffer[] = {-2.0, 4.0, 9.0};

    ns_Reduction::Params params;
    params.reductionDimension = 0; // sum and reduce first dimension size to 1

    unsigned xTensorIndex = createConstTensor(MEM_INIT_FROM_INITIALIZER, ifmBuffer, ifmDimSizes,
                                              dims, syn_type_single);

    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims,
                                                syn_type_single);

    TensorIndices inputIndices  = {xTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph("reduce_sum_fwd_f32", inputIndices, outputIndices, (void*)&params, sizeof(ns_Reduction::Params));
    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes); i++)
    {
        ASSERT_EQ(*pOutputBuffer, ofmRefBuffer[i]) << "Mismatch at index " << i
                                                   << " Result:" << *pOutputBuffer
                                                   << " Ref: " << ofmRefBuffer[i];
        pOutputBuffer++;
    }
}

TEST_F_GC(SynGaudiTestInfra, const_tensor_all_const_float)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    unsigned inputData = createConstTensor(MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4 ,syn_type_single);
    unsigned inputIndices = createConstTensor(MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createConstTensor(MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_single);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_single);

    addNodeToGraph("scatter_fwd_f32", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    float* pFmOutput = (float*)m_hostBuffers[outputTensor];

    float outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynGaudiTestInfra, const_tensor_partial_const_float_L2)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    unsigned inputData = createConstTensor(MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4 ,syn_type_single);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_single);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_single);

    addNodeToGraph("scatter_fwd_f32", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    float* pFmOutput = (float*)m_hostBuffers[outputTensor];

    float outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynGaudiTestInfra, const_tensor_all_const_bf16)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    unsigned inputData = createConstTensor(MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4 ,syn_type_bf16);
    unsigned inputIndices = createConstTensor(MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createConstTensor(MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_bf16);

    unsigned outputTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_bf16);

    addNodeToGraph("scatter_fwd_bf16", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[outputTensor];

    bfloat16 outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynGaudiTestInfra, const_tensor_partial_const_bf16_L2)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    unsigned inputData = createConstTensor(MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4 ,syn_type_bf16);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_bf16);

    unsigned outputTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_bf16);

    addNodeToGraph("scatter_fwd_bf16", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[outputTensor];

    bfloat16 outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynGaudiTestInfra, const_tensor_large_memcpy)
{
    ScopedConfigurationChange hbmSizeCfg("HBM_GLOBAL_MEM_SIZE_MEGAS", "256");
    ScopedConfigurationChange constTensorSizeCfg("MAX_CONST_TENSOR_SIZE_BYTES", "0x6400000");

    const unsigned dims = 4;
    unsigned ifmDimSizes[] = {589824, 1, 1, 1};
    unsigned ofmDimSizes[] = {589824, 1, 1, 1};

    auto in = createConstTensor(MEM_INIT_RANDOM_POSITIVE, nullptr, ifmDimSizes, dims);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims);
    addNodeToGraph("memcpy");
    compileTopology();
    runTopology();

    float* pInput  = (float*)m_hostBuffers[in];
    float* pOutput = (float*)m_hostBuffers[out];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        float expectedResult = *pInput;
        ASSERT_EQ(expectedResult, *pOutput) << "Mismatch for at index " << i
                                            << " Expected:"             << expectedResult
                                            << " pOutput1: "           << *pOutput
                                            << " pInput0: "            << *pInput;
        pInput++;
        pOutput++;
    }
}
