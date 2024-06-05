#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTpcTestInfra, relu_forward)
{
    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    auto out = createPersistTensor(OUTPUT_TENSOR);
    addNodeToGraph("relu_fwd_f32");
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        float expectedResult = std::max((float)0.0, *pInputBuffer);
        ASSERT_EQ(expectedResult, *pOutputBuffer) << "Mismatch for at index " << i
                                                  << " Expected:"             << expectedResult
                                                  << " Result: "              << *pOutputBuffer
                                                  << " operand "              << *pInputBuffer;
        pInputBuffer++;
        pOutputBuffer++;
    }
}

TEST_F_GC(SynTrainingTpcTestInfra, relu_backward_L2)
{
    auto in1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE); // input data (grad) on which relu backward is computed
    auto in2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE);      // artificial relu forward results
    auto out = createPersistTensor(OUTPUT_TENSOR);
    addNodeToGraph("relu_bwd_f32");
    compileAndRun();

    float* pFwdResult    = (float*)m_hostBuffers[in2];
    float* pInputBuffer  = (float*)m_hostBuffers[in1];
    float* pOutputBuffer = (float*)m_hostBuffers[out];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        float expectedResult = (*pFwdResult > 0) ? *pInputBuffer : 0;
        ASSERT_EQ(expectedResult, *pOutputBuffer) << "Mismatch for at index " << i
                                                  << " Expected:"             << expectedResult
                                                  << " Result: "              << *pOutputBuffer
                                                  << " FwdResult: "           << *pFwdResult
                                                  << " operand "              << *pInputBuffer;
        pInputBuffer++;
        pOutputBuffer++;
        pFwdResult++;
    }
}

TEST_F_GC(SynTrainingTpcTestInfra, relu_forward_and_backward)
{
    // Graph will have two nodes:  [relu_fwd]->[relu_bwd]

    unsigned fwdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned fwdOut = createTensor(OUTPUT_TENSOR);
    addNodeToGraph("relu_fwd_f32", {fwdIn}, {fwdOut});

    unsigned bwdIn1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE); // input data (grad) on which relu backward is computed
    unsigned bwdIn2 = connectOutputTensorToInputTensor(fwdOut);
    unsigned bwdOut = createPersistTensor(OUTPUT_TENSOR);
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

TEST_F_GC(SynTrainingTpcTestInfra, relu_slice_L2)
{
    std::array<unsigned, 2> aSize({761,256});

    unsigned sliceAmount = 200;

    std::array<unsigned, 2> sliceSize({aSize[0] - sliceAmount, aSize[1]});

    const unsigned dims = 2;
    unsigned aDimSizes[]   = {aSize[0], aSize[1], 1, 1};

    unsigned sliceSizes[] = {sliceSize[0], sliceSize[1], 1, 1};

    unsigned aTensorIndex   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, aDimSizes, dims,
                                                  syn_type_single);


    unsigned outTensorIndex = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, aDimSizes, dims,
                                           syn_type_single);

    unsigned sliceInTensor = connectOutputTensorToInputTensor(outTensorIndex);

    unsigned sliceOutTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sliceSizes, dims,
                                                    syn_type_single);


    addNodeToGraph("relu_fwd_f32", {aTensorIndex}, {outTensorIndex});

    synSliceParams sliceParams;

    sliceParams.axes[0] = 0;
    sliceParams.axes[1] = 1;
    sliceParams.axes[2] = 0;
    sliceParams.axes[3] = 0;
    sliceParams.axes[4] = 0;

    sliceParams.starts[0] = sliceAmount;
    sliceParams.starts[1] = 0;
    sliceParams.starts[2] = 0;
    sliceParams.starts[3] = 0;
    sliceParams.starts[4] = 0;

    sliceParams.ends[0] = aSize[0];
    sliceParams.ends[1] = aSize[1];
    sliceParams.ends[2] = 0;
    sliceParams.ends[3] = 0;
    sliceParams.ends[4] = 0;

    sliceParams.steps[0] = 1;
    sliceParams.steps[1] = 1;
    sliceParams.steps[2] = 0;
    sliceParams.steps[3] = 0;
    sliceParams.steps[4] = 0;

    addNodeToGraph("slice", {sliceInTensor}, {sliceOutTensor}, &sliceParams, sizeof(sliceParams));

    compileAndRun();


    float* outRef = new float[aSize[0]*aSize[1]];

    float* pInputBuffer = (float*)m_hostBuffers[aTensorIndex];

    for (unsigned i =0 ; i < aSize[0]*aSize[1]; i++)
    {
        outRef[i] = pInputBuffer[i] < 0 ? 0 : pInputBuffer[i];
    }


    float* pOutputBuffer = (float*)m_hostBuffers[sliceOutTensor];


    float* afterSlice = new float[sliceSize[0]*sliceSize[1]];

    for (unsigned j=0; j < aSize[1]; j++)
    {
        for (unsigned i = 0; i< aSize[0] - sliceAmount ; i++)
        {
            afterSlice[(j*(aSize[0]-sliceAmount)) +i] = outRef[i + (j*aSize[0]) + sliceAmount] ;
        }
    }

    validateResult(afterSlice, pOutputBuffer, sliceSize[0]*sliceSize[1]);


    delete[] outRef;
    delete[] afterSlice;
}
