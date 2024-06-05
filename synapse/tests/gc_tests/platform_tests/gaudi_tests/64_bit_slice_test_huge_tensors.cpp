#include "64_bit_slice_test_huge_tensors.h"

TEST_F_GC(SynGaudiHugeTensors, huge_tensors_test, {synDeviceGaudi2})
{
    TSize sizesIn[]       = {TSize(1 << 5), TSize(1 << 20), TSize(1 << 7)};
    TSize sizesSliceOut[] = {TSize(1 << 5), TSize(1 << 7), TSize(1 << 7)};
    TSize sizesOut[]      = {TSize(1 << 7), TSize(1 << 7), TSize(1 << 5)};

    unsigned inputTensor    = createHugeTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "input_tensor",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             sizesIn,
                                             3)[0];
    unsigned sliceOutTensor = createHugeTensors(1,
                                                OUTPUT_TENSOR,
                                                false,
                                                "slice_out",
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                sizesSliceOut,
                                                3)[0];
    unsigned outputTensor   = createHugeTensors(1,
                                              OUTPUT_TENSOR,
                                              true,
                                              "output_tensor",
                                              MEM_INIT_ALL_ZERO,
                                              nullptr,
                                              sizesOut,
                                              3)[0];

    synSliceParamsNDims sliceParams = {{0}};
    sliceParams.axes[0]             = 0;
    sliceParams.starts[0]           = 0;
    sliceParams.ends[0]             = sizesSliceOut[0];
    sliceParams.steps[0]            = 1;

    sliceParams.axes[1]   = 1;
    sliceParams.starts[1] = 1 << 2;
    sliceParams.ends[1]   = sliceParams.starts[1] + sizesSliceOut[1];
    sliceParams.steps[1]  = 1;

    sliceParams.axes[2]   = 2;
    sliceParams.starts[2] = 0;
    sliceParams.ends[2]   = sizesSliceOut[2];
    sliceParams.steps[2]  = 1;

    addNodeToGraph(NodeFactory::sliceNodeTypeName,
                   {inputTensor},
                   {sliceOutTensor},
                   &sliceParams,
                   sizeof(sliceParams),
                   "slice");

    synTransposeParamsNDims transposeParams;
    transposeParams.tensorDim      = 3;
    transposeParams.permutation[0] = 1;
    transposeParams.permutation[1] = 2;
    transposeParams.permutation[2] = 0;

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {sliceOutTensor},
                   {outputTensor},
                   &transposeParams,
                   sizeof(transposeParams),
                   "transpose");

    compileTopology("huge_tensor_logical_op_recipe", 0);

    runTopology(0);

    float* pinputBuffer  = castHostOutBuffer<float>(inputTensor);
    float* pOutputBuffer = castHostOutBuffer<float>(outputTensor);

    TStride inputStride[3] = {1};
    inputStride[0]         = sliceParams.steps[0];
    inputStride[1]         = sliceParams.steps[1] * sizesIn[0];
    inputStride[2]         = sliceParams.steps[2] * sizesIn[0] * sizesIn[1];

    TStride outputStride[3] = {1};
    outputStride[0]         = sizesOut[1] * sizesOut[0];
    outputStride[1]         = 1;
    outputStride[2]         = sizesOut[0];

    for (TSize b = 0; b < sizesSliceOut[2]; b++)
    {
        for (TSize r = 0; r < sizesSliceOut[1]; r++)
        {
            for (TSize c = 0; c < sizesSliceOut[0]; c++)
            {
                TSize inputOffset = (b + sliceParams.starts[2]) * inputStride[2] +
                                    (r + sliceParams.starts[1]) * inputStride[1] +
                                    (c + sliceParams.starts[0]) * inputStride[0];
                TSize outputOffset = b * outputStride[2] + r * outputStride[1] + c * outputStride[0];
                float expected     = pinputBuffer[inputOffset];
                float actual       = pOutputBuffer[outputOffset];
                ASSERT_EQ(expected, actual) << "result mismatch on index: (" << c << ',' << r << "," << b
                                            << "), expected: " << expected << ", actual: " << actual;
            }
        }
    }
}