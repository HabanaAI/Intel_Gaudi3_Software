#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "gc_gaudi_test_infra.h"

TEST_F_GC(SynTrainingTpcTestInfra, maxpool_forward_L2)
{

    /* Initialize params */
    SpatialReduction2DDef kernel_params;
    kernel_params.pad_w_begin = 0;
    kernel_params.pad_h_end   = 0;
    kernel_params.pad_w_end   = 0;
    kernel_params.pad_h_begin = 0;
    kernel_params.kernel_w    = 2;
    kernel_params.kernel_h    = 2;
    kernel_params.stride_w    = 1;
    kernel_params.stride_h    = 1;
    kernel_params.dilation_w  = 1;
    kernel_params.dilation_h  = 1;

    /* const params */
    const unsigned inZ = 2;
    const unsigned inW = 4;
    const unsigned inH = 4;
    const unsigned batch = 1;

    float inputData [batch][inH][inW][inZ] =
            {{
                     {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
                     {{7, 6}, {5, 4}, {3, 2}, {1, 0}},
                     {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
                     {{7, 6}, {5, 4}, {3, 2}, {1, 0}}
             }};

    float outputMaxRef [batch][3][3][2] =
            {{
                     {{7, 6}, {5, 5}, {6, 7}},
                     {{7, 6}, {5, 5}, {6, 7}},
                     {{7, 6}, {5, 5}, {6, 7}},
             }};

    unsigned outW = (inW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1)/ kernel_params.stride_w + 1;
    unsigned outH = (inH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1)/ kernel_params.stride_h + 1;

    // Tensor size [NHWC]
    unsigned inTensorSize[4]  = { inZ, inW, inH, 1 };
    unsigned outTensorSize[2][4] = {
            { inZ, outW, outH, 1 },
            { inZ, outW, outH, 1 }
    };

    unsigned inputBufferSize = inTensorSize[0] * inTensorSize[1] * inTensorSize[2] * inTensorSize[3];
    unsigned outputBufferSize[2] = {
            outTensorSize[0][0] * outTensorSize[0][1] * outTensorSize[0][2] * outTensorSize[0][3],
            outTensorSize[1][0] * outTensorSize[1][1] * outTensorSize[1][2] * outTensorSize[1][3]
    };

    const unsigned inputTotalSize = inputBufferSize * sizeof(float);
    float inputBuffer[inputBufferSize];
    memcpy(inputBuffer, inputData, inputTotalSize);

    unsigned inputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputBuffer, inTensorSize, 4 ,
                                                syn_type_single, nullptr, "inputTensor");
    unsigned firstOutputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outTensorSize[0], 4,
                                                      syn_type_uint8, nullptr, "firstOutputTensor");
    unsigned secondOutputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outTensorSize[1], 4,
                                                       syn_type_single, nullptr, "secondOutputTensor");

    addNodeToGraph("maxpool_2d_fwd_f32", {inputTensor}, {firstOutputTensor, secondOutputTensor}, &kernel_params,
                   sizeof(SpatialReduction2DDef));

    compileAndRun();

    // validate
    float* pOutputBuffer = (float*)m_hostBuffers[secondOutputTensor];
    validateResult((float*)outputMaxRef, pOutputBuffer, outputBufferSize[1]);

}

TEST_F_GC(SynTrainingTpcTestInfra, maxpool_backward_L2)
{

    /* Initialize params */
    SpatialReduction2DDef kernel_params;
    kernel_params.pad_w_begin = 0;
    kernel_params.pad_h_end   = 0;
    kernel_params.pad_w_end   = 0;
    kernel_params.pad_h_begin = 0;
    kernel_params.kernel_w    = 3;
    kernel_params.kernel_h    = 3;
    kernel_params.stride_w    = 2;
    kernel_params.stride_h    = 2;
    kernel_params.dilation_w  = 1;
    kernel_params.dilation_h  = 1;

    /* const params */
    const unsigned inZ = 64;
    const unsigned inW = 2;
    const unsigned inH = 2;

    float firstInputData [1][inH][inW][inZ];
    //{{
    //    {{1,1,1,1,1,...},
    //     {9,9,9,9,9,...}},
    //    {{8,8,8,8,8,...},
    //     {2,2,2,2,2,...}}
    //}};
    for (unsigned i = 0; i < inH; ++i)
    {
        for (unsigned j = 0; j < inW; ++j)
        {
            float initializer = (i == 0 && j == 0) ? 1 :
                                (i == 0 && j == 1) ? 9 :
                                (i == 1 && j == 0) ? 8 : 2;
            for (unsigned k = 0; k < inZ; ++k)
            {
                firstInputData[0][i][j][k] = initializer;
            }
        }
    }

    uint8_t secondInputData [1][inH][inW][inZ];
    //{{
    //    {{0,0,0,0,0,...},
    //     {5,5,5,5,5,...}},
    //    {{3,3,3,3,3,...},
    //     {5,5,5,5,5,...}}
    //}};
    for (unsigned i = 0; i < inH; ++i)
    {
        for (unsigned j = 0; j < inW; ++j)
        {
            uint8_t initializer = (i == 0 && j == 0) ? 0 :
                                  (i == 1 && j == 0) ? 3 : 5;
            for (unsigned k = 0; k < inZ; ++k)
            {
                secondInputData[0][i][j][k] = initializer;
            }
        }
    }

    float outputRef [1][5][5][inZ];// = {0};
    //{{
    //    {{1,1,1,1,1,1,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...}},
    //    {{0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {9,9,9,9,9,9,...}},
    //    {{0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...}},
    //    {{8,8,8,8,8,8,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {2,2,2,2,2,2,...}},
    //    {{0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...},
    //     {0,0,0,0,0,0,...}}
    //}};
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 0; j < 5; ++j)
        {
            float initializer = (i == 0 && j == 0) ? 1 :
                                (i == 1 && j == 4) ? 9 :
                                (i == 3 && j == 0) ? 8 :
                                (i == 3 && j == 4) ? 2 : 0;
            for (unsigned k = 0; k < inZ; ++k)
            {
                outputRef[0][i][j][k] = initializer;
            }
        }
    }

    unsigned outW = ((inW - 1) * kernel_params.stride_w) - (kernel_params.pad_w_begin + kernel_params.pad_w_end) + (kernel_params.dilation_w * (kernel_params.kernel_w-1)) + 1;
    unsigned outH = ((inH - 1) * kernel_params.stride_h) - (kernel_params.pad_h_begin + kernel_params.pad_h_end) + (kernel_params.dilation_h * (kernel_params.kernel_h-1)) + 1;

    // Tensor size [NHWC]
    unsigned inTensorSize[2][4]  = {
            { inZ, inW, inH, 1 },
            { inZ, inW, inH, 1 }
    };
    unsigned outTensorSize[4] = { inZ, outW, outH, 1 };

    unsigned inputBufferSize[2] = {
            inTensorSize[0][0] * inTensorSize[0][1] * inTensorSize[0][2] * inTensorSize[0][3],
            inTensorSize[1][0] * inTensorSize[1][1] * inTensorSize[1][2] * inTensorSize[1][3]
    };
    unsigned outputBufferSize = outTensorSize[0] * outTensorSize[1] * outTensorSize[2] * outTensorSize[3];

    unsigned firstInputTotalSize   = inputBufferSize[0] * sizeof(float);
    unsigned secondInputTotalSize  = inputBufferSize[1] * sizeof(uint8_t);

    float firstInputBuffer[inputBufferSize[0]];
    memcpy(firstInputBuffer, firstInputData, firstInputTotalSize);

    float secondInputBuffer[inputBufferSize[1]];
    memcpy(secondInputBuffer, secondInputData, secondInputTotalSize);

    unsigned firstInputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, firstInputBuffer,
                                                     inTensorSize[0], 4 ,syn_type_single, nullptr, "firstInputTensor");
    unsigned secondInputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, secondInputBuffer,
                                                      inTensorSize[1], 4 ,syn_type_uint8, nullptr, "secondInputTensor");
    unsigned outputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                 outTensorSize, 4, syn_type_single, nullptr, "outputTensor");

    addNodeToGraph("maxpool_2d_bwd_f32", {firstInputTensor, secondInputTensor}, {outputTensor}, &kernel_params, sizeof(SpatialReduction2DDef));

    compileAndRun();

    // validate
    float* pOutputBuffer = (float*)m_hostBuffers[outputTensor];
    validateResult((float*)outputRef, pOutputBuffer, outputBufferSize);

}
